"""Collect paired one-shot continuations for outcome-aware stopping labels.

The action planner is deliberately out of scope for this program.  Its input is
an immutable JSON/JSONL plan naming a cached teacher-forced position and two
already-computed action token IDs.  This collector reconstructs that position
from the pinned cache evidence, forces each action into the assistant prefix,
and asks the target model for the *remaining continuation in one request*.

Safety/resource properties are fail-closed:

* the proxy manifest, source manifest, source record, dataset row, tokenizer
  mapping, labels, and Qwen/DeepInfra request protocol are cross-checked;
* target calls are elided only when the fully serialized branch requests are
  identical (token-ID equality is neither checked early nor used as a proxy);
* every non-terminal mismatch has exactly one temperature-zero request per arm;
* intents are persisted before requests, so resume never silently re-sends an
  arm whose billing outcome is uncertain;
* the global request-attempt cap includes completed and pending intents; and
* judge-ready JSONL is rebuilt by atomic replacement and contains complete
  two-arm groups only.

``--dry_run`` performs the complete local preflight but neither resolves an API
key nor creates output/state files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference_openrouter import validate_handoff_response  # noqa: E402
from pre_logits_sampled_openrouter import (  # noqa: E402
    OpenRouterClient,
    messages_for_prefix,
    resolve_api_key,
)
from materialize_proxy_mc_cache import (  # noqa: E402
    canonical_json_bytes,
    source_positions,
    tokenizer_mapping_sha256,
)


OUTPUT_SCHEMA_VERSION = 1
PLAN_SCHEMA = "outcome_regret_action_plan_v1"
COLLECTOR_PROTOCOL = "paired_base_continuation_t0_qwen_v1"
TARGET_MODEL = "qwen/qwen3-32b"
TARGET_PROVIDER = "DeepInfra"
TARGET_CHAT_PROTOCOL = "messages_for_prefix_qwen_v1"
EXPECTED_DATASET = "LLM-LAT/harmful-dataset"
EXPECTED_DATASET_SPLIT = "train"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LOCAL_OVERFLOW_ERROR_RE = re.compile(
    r"Gate handoff exceeded its remaining local token budget: "
    r"(?P<observed>[1-9][0-9]*) > (?P<remaining>[1-9][0-9]*)\."
)


def generate_collector_continuation(
    client: OpenRouterClient,
    tokenizer,
    prompt: str,
    prefix_text: str,
    remaining_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, dict[str, Any]]:
    """Collector-only handoff accepting audited local retokenization overflow.

    The provider still receives exactly ``remaining_tokens``.  Qwen's returned
    text can tokenize to one extra local token at the prefix boundary; that is
    a measurement artifact, not extra provider generation.  The shared legacy
    online helper keeps its existing fail-on-overflow default.
    """

    if remaining_tokens <= 0:
        raise ValueError("remaining_tokens must be positive for collection.")
    response = client.generate(
        messages=messages_for_prefix(
            prompt,
            prefix_text,
            qwen_hard_no_think_prefill=True,
        ),
        temperature=temperature,
        top_p=top_p,
        max_tokens=remaining_tokens,
    )
    return validate_handoff_response(
        response,
        tokenizer,
        prefix_text,
        remaining_tokens,
        reject_reasoning_tokens=True,
        allow_local_token_overflow=True,
        requested_model=TARGET_MODEL,
    )


@dataclass(frozen=True)
class PlanPair:
    plan_index: int
    pair_id: str
    candidate_id: Optional[str]
    record_name: str
    dataset_idx: int
    position: int
    partial_action: int
    reference_action: int
    plan_row_sha256: str
    plan_row: dict[str, Any]


@dataclass(frozen=True)
class PreparedArm:
    role: str
    action_token_id: int
    completion_order: int
    raw_forced_prefix: str
    visible_forced_prefix: str
    remaining_tokens: int
    local_terminal: bool
    teacher_prefix_roundtrip: bool
    teacher_prefix_retokenized_ids: tuple[int, ...]
    teacher_prefix_retokenized_ids_sha256: str
    forced_prefix_roundtrip: bool
    forced_prefix_retokenized_ids: tuple[int, ...]
    forced_prefix_retokenized_ids_sha256: str
    forced_extends_teacher_text: bool


@dataclass(frozen=True)
class PreparedPair:
    plan: PlanPair
    prompt: str
    api_prompt: str
    data_sha256: str
    answer_token_count: int
    source_record_sha256: str
    partial: PreparedArm
    reference: PreparedArm
    partial_branch_serialization: dict[str, Any]
    reference_branch_serialization: dict[str, Any]
    partial_branch_serialization_sha256: str
    reference_branch_serialization_sha256: str
    structural_zero: bool
    structural_zero_reason: Optional[str]

    @property
    def arms(self) -> tuple[PreparedArm, PreparedArm]:
        return (self.partial, self.reference)


@dataclass(frozen=True)
class Preflight:
    pairs: tuple[PreparedPair, ...]
    token_equal_rows: int
    plan_row_count: int
    selected_source_records: tuple[str, ...]
    planned_api_calls: int
    planned_requested_output_tokens: int
    proxy_manifest: dict[str, Any]
    proxy_manifest_sha256: str
    source_manifest: dict[str, Any]
    source_manifest_sha256: str
    dataset_fingerprint: Optional[str]
    dataset_revision: Optional[str]
    tokenizer_sha256: str
    tokenizer_name_or_path: str
    tokenizer_revision: Optional[str]
    tokenizer: Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_manifest_fingerprint(manifest: Mapping[str, Any], *, label: str) -> dict:
    if manifest.get("manifest_schema_version") != 1:
        raise ValueError(f"Unsupported {label} manifest schema version.")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"{label} manifest has no configuration object.")
    expected = manifest.get("configuration_fingerprint")
    actual = sha256_payload(configuration)
    if expected != actual:
        raise ValueError(f"{label} manifest configuration fingerprint is invalid.")
    return configuration


def _strict_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Plan field {field!r} must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"Plan field {field!r} must be >= {minimum}.")
    return result


def _plan_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing action plan: {path}")
    if path.suffix.casefold() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Plan line {line_number} must be a JSON object.")
            if row.get("type") in {"manifest", "header"}:
                continue
            rows.append(row)
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("pairs"), list):
        schema = payload.get("schema")
        if schema is not None and schema != PLAN_SCHEMA:
            raise ValueError(f"Unsupported plan schema: {schema!r}.")
        rows = payload["pairs"]
    else:
        raise ValueError("JSON action plan must be a list or an object with a pairs list.")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Every action-plan row must be a JSON object.")
    return list(rows)


def load_plan(path: Path, *, max_pairs: Optional[int] = None) -> tuple[list[PlanPair], int, int]:
    """Load plan rows without inferring request equality from action IDs."""

    rows = _plan_rows(path)
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("--max_pairs must be positive.")
    pairs: list[PlanPair] = []
    token_equal_rows = 0
    candidate_presence = ["candidate_id" in row for row in rows]
    candidate_mode = any(candidate_presence)
    if candidate_mode and not all(candidate_presence):
        raise ValueError(
            "When any plan row declares candidate_id, every row must declare it."
        )
    seen_branch_identities: set[tuple[str, int, int, int, int]] = set()
    seen_candidate_ids: set[str] = set()
    seen_pair_ids: set[str] = set()
    for plan_index, row in enumerate(rows):
        missing = sorted(
            {"record_name", "dataset_idx", "position", "partial_action", "reference_action"}
            - set(row)
        )
        if missing:
            raise ValueError(
                f"Plan row {plan_index} is missing required fields: {missing}."
            )
        record_name = str(row["record_name"])
        if Path(record_name).name != record_name or not record_name.endswith(".pt"):
            raise ValueError(
                f"Plan row {plan_index} record_name must be one .pt basename."
            )
        dataset_idx = _strict_integer(row["dataset_idx"], field="dataset_idx")
        position = _strict_integer(row["position"], field="position")
        partial = _strict_integer(row["partial_action"], field="partial_action")
        reference = _strict_integer(row["reference_action"], field="reference_action")
        candidate_id: Optional[str] = None
        if candidate_mode:
            value = row["candidate_id"]
            if not isinstance(value, str) or not re.fullmatch(
                r"[A-Za-z0-9_.:-]{1,256}", value
            ):
                raise ValueError(
                    f"Plan row {plan_index} candidate_id must be a non-empty "
                    "portable identifier of at most 256 characters."
                )
            candidate_id = value
            if candidate_id in seen_candidate_ids:
                raise ValueError(
                    f"Duplicate candidate_id at plan row {plan_index}: "
                    f"{candidate_id!r}."
                )
            seen_candidate_ids.add(candidate_id)
        branch_identity = (record_name, dataset_idx, position, partial, reference)
        if not candidate_mode:
            if branch_identity in seen_branch_identities:
                raise ValueError(
                    f"Duplicate action-plan row identity at index {plan_index}."
                )
            seen_branch_identities.add(branch_identity)
        if partial == reference:
            token_equal_rows += 1
        canonical_identity = {
            "record_name": record_name,
            "dataset_idx": dataset_idx,
            "position": position,
            "partial_action": partial,
            "reference_action": reference,
        }
        if candidate_id is not None:
            canonical_identity["candidate_id"] = candidate_id
        pair_id = f"orp-{sha256_payload(canonical_identity)[:24]}"
        if pair_id in seen_pair_ids:
            raise ValueError(f"Derived pair_id collision at plan row {plan_index}.")
        seen_pair_ids.add(pair_id)
        pairs.append(
            PlanPair(
                plan_index=plan_index,
                pair_id=pair_id,
                candidate_id=candidate_id,
                record_name=record_name,
                dataset_idx=dataset_idx,
                position=position,
                partial_action=partial,
                reference_action=reference,
                plan_row_sha256=sha256_payload(row),
                plan_row=dict(row),
            )
        )
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    if not pairs:
        raise ValueError("The selected action plan contains no pairs.")
    return pairs, token_equal_rows, len(rows)


def _validate_protocol(
    proxy_configuration: Mapping[str, Any],
    source_configuration: Mapping[str, Any],
) -> None:
    required_source = {
        "source": "sampled_openrouter",
        "model": TARGET_MODEL,
        "samples_per_token": 50,
        "sample_completion_policy": "exact",
        "append_no_think": True,
        "qwen_hard_no_think_prefill": True,
        "reasoning_mode": "enabled_false",
        "reject_reasoning_tokens": True,
        "provider_order": [TARGET_PROVIDER],
        "provider_allow_fallbacks": False,
        "disable_openrouter_response_cache": True,
        "dataset_name": EXPECTED_DATASET,
        "dataset_split": EXPECTED_DATASET_SPLIT,
    }
    for field, expected in required_source.items():
        if source_configuration.get(field) != expected:
            raise ValueError(
                f"Source protocol {field!r} must be {expected!r}, got "
                f"{source_configuration.get(field)!r}."
            )
    required_proxy = {
        "mc_fusion_mode": "proxy_dirichlet_v1",
        "target_model": TARGET_MODEL,
        "target_provider": TARGET_PROVIDER,
        "proxy_chat_template_protocol": TARGET_CHAT_PROTOCOL,
        "dataset_name": EXPECTED_DATASET,
        "dataset_split": EXPECTED_DATASET_SPLIT,
    }
    for field, expected in required_proxy.items():
        if proxy_configuration.get(field) != expected:
            raise ValueError(
                f"Proxy protocol {field!r} must be {expected!r}, got "
                f"{proxy_configuration.get(field)!r}."
            )
    if proxy_configuration.get("source_configuration_fingerprint") != sha256_payload(
        dict(source_configuration)
    ):
        raise ValueError("Proxy/source configuration fingerprints disagree.")
    if source_configuration.get("max_answer_tokens") is None:
        raise ValueError("Source manifest must declare max_answer_tokens.")
    max_answer_tokens = int(source_configuration["max_answer_tokens"])
    if max_answer_tokens <= 0:
        raise ValueError("Source max_answer_tokens must be positive.")


def validate_action_prefix(
    tokenizer,
    *,
    prefix_token_ids: Sequence[int],
    action_token_id: int,
    remaining_tokens: int,
    role: str,
    completion_order: int,
) -> PreparedArm:
    """Construct the online decode-to-API prefix and audit retokenization.

    Prefix instability is expected for byte/BPE tokenizers.  Online inference
    decodes the controlled token sequence to text and the provider tokenizes
    that serialized text again.  A changed ID sequence is therefore evidence
    to retain, not a reason to reject an otherwise representable action.
    """

    action_token_id = int(action_token_id)
    if action_token_id < 0 or action_token_id >= len(tokenizer):
        raise ValueError(f"Action token ID {action_token_id} is outside the tokenizer.")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    special_ids = {int(value) for value in (getattr(tokenizer, "all_special_ids", []) or [])}
    if action_token_id in special_ids and action_token_id != eos_token_id:
        raise ValueError(
            f"Action token ID {action_token_id} is a non-EOS special token."
        )

    prefix_ids = [int(value) for value in prefix_token_ids]
    raw_prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    teacher_retokenized = tuple(
        int(value)
        for value in tokenizer.encode(raw_prefix, add_special_tokens=False)
    )
    forced_ids = prefix_ids + [action_token_id]
    raw_forced = tokenizer.decode(forced_ids, skip_special_tokens=False)
    forced_retokenized = tuple(
        int(value)
        for value in tokenizer.encode(raw_forced, add_special_tokens=False)
    )
    visible_forced = tokenizer.decode(forced_ids, skip_special_tokens=True)
    local_terminal = bool(action_token_id == eos_token_id or remaining_tokens == 0)
    return PreparedArm(
        role=role,
        action_token_id=action_token_id,
        completion_order=completion_order,
        raw_forced_prefix=raw_forced,
        visible_forced_prefix=visible_forced,
        remaining_tokens=int(remaining_tokens),
        local_terminal=local_terminal,
        teacher_prefix_roundtrip=teacher_retokenized == tuple(prefix_ids),
        teacher_prefix_retokenized_ids=teacher_retokenized,
        teacher_prefix_retokenized_ids_sha256=sha256_payload(
            list(teacher_retokenized)
        ),
        forced_prefix_roundtrip=forced_retokenized == tuple(forced_ids),
        forced_prefix_retokenized_ids=forced_retokenized,
        forced_prefix_retokenized_ids_sha256=sha256_payload(
            list(forced_retokenized)
        ),
        forced_extends_teacher_text=raw_forced.startswith(raw_prefix),
    )


def branch_serialization(api_prompt: str, arm: PreparedArm) -> dict[str, Any]:
    """Canonical representation of everything that determines a branch request.

    This object deliberately contains the decoded assistant text, not the
    planner's action token ID.  It is the decode-to-API serialization that the
    target actually observes.  ``local_terminal`` is included so two locally
    terminated branches can be compared under the same rule.
    """

    return {
        "api_prompt": api_prompt,
        "raw_forced_prefix": arm.raw_forced_prefix,
        "remaining_tokens": int(arm.remaining_tokens),
        "local_terminal": bool(arm.local_terminal),
        "model": TARGET_MODEL,
        "provider_order": [TARGET_PROVIDER],
        "provider_allow_fallbacks": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning_mode": "enabled_false",
        "reject_reasoning_tokens": True,
        "qwen_hard_no_think_prefill": True,
        "append_no_think": True,
        "disable_openrouter_response_cache": True,
        "request_protocol": COLLECTOR_PROTOCOL,
    }


def _record_summaries(proxy_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = proxy_manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Proxy manifest contains no record summaries.")
    summaries: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("file"), str):
            raise ValueError("Proxy manifest has an invalid record summary.")
        name = record["file"]
        if name in summaries:
            raise ValueError(f"Duplicate proxy manifest record summary: {name}.")
        summaries[name] = record
    return summaries


def _validate_source_record(
    *,
    source_path: Path,
    source_summary: Mapping[str, Any],
    source_configuration: Mapping[str, Any],
    source_fingerprint: str,
    dataset,
    tokenizer,
    planned_pairs: Sequence[PlanPair],
) -> list[PreparedPair]:
    expected_source_sha = source_summary.get("source_sha256")
    actual_source_sha = sha256_file(source_path)
    if actual_source_sha != expected_source_sha:
        raise ValueError(f"Source cache SHA256 mismatch for {source_path.name}.")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Source cache {source_path.name} is not a dictionary.")
    labels = payload.get("labels")
    valid_counts = payload.get("valid_sample_counts")
    metadata = payload.get("metadata")
    if (
        not isinstance(labels, torch.Tensor)
        or labels.dim() != 2
        or labels.shape[0] != 1
        or not isinstance(valid_counts, torch.Tensor)
        or valid_counts.shape != labels.shape
        or not isinstance(metadata, dict)
    ):
        raise ValueError(f"Source cache {source_path.name} has invalid evidence tensors.")
    if metadata.get("cache_configuration_fingerprint") != source_fingerprint:
        raise ValueError(f"Source metadata/manifest fingerprint mismatch in {source_path.name}.")
    expected_samples = int(source_configuration["samples_per_token"])
    if not torch.equal(
        valid_counts.detach().cpu().long(),
        torch.full_like(valid_counts.detach().cpu().long(), expected_samples),
    ):
        raise ValueError(f"Source record {source_path.name} is not exact MC{expected_samples}.")

    dataset_indices = {pair.dataset_idx for pair in planned_pairs}
    if len(dataset_indices) != 1:
        raise ValueError(f"Plan record {source_path.name} maps to multiple dataset indices.")
    dataset_idx = next(iter(dataset_indices))
    if int(source_summary.get("dataset_idx", -1)) != dataset_idx:
        raise ValueError(f"Plan/proxy dataset_idx mismatch for {source_path.name}.")
    if int(metadata.get("dataset_idx", -1)) != dataset_idx:
        raise ValueError(f"Plan/source dataset_idx mismatch for {source_path.name}.")
    if dataset_idx < 0 or dataset_idx >= len(dataset):
        raise ValueError(f"dataset_idx={dataset_idx} is outside the reconstructed dataset.")
    item = dataset[dataset_idx]
    prompt = str(item["prompt"])
    answer = str(item["rejected"])
    data_sha256 = hashlib.sha256((prompt + answer).encode("utf-8")).hexdigest()
    expected_name = hashlib.md5((prompt + answer).encode("utf-8")).hexdigest() + ".pt"
    if expected_name != source_path.name:
        raise ValueError(f"Dataset-derived record filename mismatch for {source_path.name}.")
    if metadata.get("data_sha256") != data_sha256:
        raise ValueError(f"Dataset/source data SHA256 mismatch for {source_path.name}.")
    if source_summary.get("data_sha256") != data_sha256:
        raise ValueError(f"Dataset/proxy data SHA256 mismatch for {source_path.name}.")

    for field in (
        "source",
        "model",
        "samples_per_token",
        "sample_completion_policy",
        "sample_temperature",
        "top_p",
        "qwen_hard_no_think_prefill",
        "reasoning_mode",
        "reject_reasoning_tokens",
        "provider_order",
        "provider_allow_fallbacks",
        "disable_openrouter_response_cache",
    ):
        if metadata.get(field) != source_configuration.get(field):
            raise ValueError(
                f"Source metadata {field!r} disagrees with its manifest in "
                f"{source_path.name}."
            )

    answer_ids = [
        int(value) for value in tokenizer.encode(answer, add_special_tokens=False)
    ][: int(source_configuration["max_answer_tokens"])]
    positions = source_positions(payload, int(labels.shape[1]))
    if positions != tuple(range(len(positions))):
        raise ValueError(
            f"Source record {source_path.name} is not an every-prefix cache."
        )
    if not positions or positions[-1] >= len(answer_ids):
        raise ValueError(f"Source cached positions are invalid in {source_path.name}.")
    expected_labels = torch.tensor(
        [answer_ids[position] for position in positions], dtype=torch.long
    )
    if not torch.equal(labels[0].detach().cpu().long(), expected_labels):
        raise ValueError(f"Source labels do not match dataset tokens in {source_path.name}.")
    metadata_eos = metadata.get("canonical_eos_token_id")
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if metadata_eos is None or tokenizer_eos is None or int(metadata_eos) != int(tokenizer_eos):
        raise ValueError(f"Canonical EOS identity mismatch in {source_path.name}.")

    api_prompt = prompt + "\n/no_think"
    prepared: list[PreparedPair] = []
    for pair in planned_pairs:
        if pair.position >= len(positions):
            raise ValueError(
                f"Plan position {pair.position} is outside {source_path.name}."
            )
        remaining = int(source_configuration["max_answer_tokens"]) - (pair.position + 1)
        if remaining < 0:
            raise ValueError("A planned position exceeds the target answer budget.")
        prefix_ids = answer_ids[: pair.position]
        partial = validate_action_prefix(
            tokenizer,
            prefix_token_ids=prefix_ids,
            action_token_id=pair.partial_action,
            remaining_tokens=remaining,
            role="partial",
            completion_order=0,
        )
        reference = validate_action_prefix(
            tokenizer,
            prefix_token_ids=prefix_ids,
            action_token_id=pair.reference_action,
            remaining_tokens=remaining,
            role="reference",
            completion_order=1,
        )
        partial_serialization = branch_serialization(api_prompt, partial)
        reference_serialization = branch_serialization(api_prompt, reference)
        partial_serialization_sha = sha256_payload(partial_serialization)
        reference_serialization_sha = sha256_payload(reference_serialization)
        structural_zero = partial_serialization == reference_serialization
        prepared.append(
            PreparedPair(
                plan=pair,
                prompt=prompt,
                api_prompt=api_prompt,
                data_sha256=data_sha256,
                answer_token_count=len(answer_ids),
                source_record_sha256=actual_source_sha,
                partial=partial,
                reference=reference,
                partial_branch_serialization=partial_serialization,
                reference_branch_serialization=reference_serialization,
                partial_branch_serialization_sha256=partial_serialization_sha,
                reference_branch_serialization_sha256=reference_serialization_sha,
                structural_zero=structural_zero,
                structural_zero_reason=(
                    "identical_serialized_branch_requests"
                    if structural_zero
                    else None
                ),
            )
        )
    return prepared


def run_preflight(args: argparse.Namespace) -> Preflight:
    proxy_dir = Path(args.proxy_cache_dir).expanduser().resolve()
    proxy_manifest_path = proxy_dir / "proxy_mc_manifest.json"
    if not proxy_manifest_path.is_file():
        raise FileNotFoundError(f"Missing proxy manifest: {proxy_manifest_path}")
    proxy_manifest = json.loads(proxy_manifest_path.read_text(encoding="utf-8"))
    proxy_configuration = validate_manifest_fingerprint(
        proxy_manifest, label="proxy MC"
    )
    source_dir = Path(str(proxy_configuration.get("source_cache_dir", ""))).expanduser().resolve()
    source_manifest_path = source_dir / "cache_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest_path}")
    source_manifest_sha = sha256_file(source_manifest_path)
    if source_manifest_sha != proxy_configuration.get("source_manifest_sha256"):
        raise ValueError("Proxy manifest points to a source manifest with the wrong SHA256.")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_configuration = validate_manifest_fingerprint(
        source_manifest, label="source cache"
    )
    _validate_protocol(proxy_configuration, source_configuration)

    plan_path = Path(args.plan).expanduser().resolve()
    plan_pairs, token_equal_rows, plan_row_count = load_plan(
        plan_path, max_pairs=args.max_pairs
    )
    summaries = _record_summaries(proxy_manifest)
    selected_names = sorted({pair.record_name for pair in plan_pairs})
    missing = sorted(set(selected_names) - set(summaries))
    if missing:
        raise ValueError(f"Planned records are absent from the proxy manifest: {missing}.")

    dataset_name = str(proxy_configuration.get("dataset_name"))
    dataset_split = str(proxy_configuration.get("dataset_split"))
    manifest_dataset_revision = proxy_configuration.get("dataset_revision")
    if (
        args.dataset_revision is not None
        and manifest_dataset_revision is not None
        and args.dataset_revision != manifest_dataset_revision
    ):
        raise ValueError("--dataset_revision disagrees with the proxy manifest.")
    dataset_revision = (
        args.dataset_revision
        if args.dataset_revision is not None
        else manifest_dataset_revision
    )
    dataset = load_dataset(
        dataset_name,
        split=dataset_split,
        revision=dataset_revision,
    )
    dataset_fingerprint = getattr(dataset, "_fingerprint", None)
    expected_dataset_fingerprint = proxy_configuration.get("dataset_fingerprint")
    if (
        expected_dataset_fingerprint is not None
        and dataset_fingerprint != expected_dataset_fingerprint
    ):
        raise ValueError(
            "Reconstructed dataset fingerprint disagrees with the proxy manifest."
        )

    tokenizer_name = str(proxy_configuration.get("target_tokenizer_name_or_path", ""))
    tokenizer_revision = proxy_configuration.get("target_tokenizer_revision")
    if not tokenizer_name:
        raise ValueError("Proxy manifest does not pin the target tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision,
        trust_remote_code=False,
        local_files_only=bool(args.local_files_only),
    )
    tokenizer_sha = tokenizer_mapping_sha256(tokenizer)
    expected_tokenizer_sha = proxy_configuration.get("target_tokenizer_sha256")
    if tokenizer_sha != expected_tokenizer_sha:
        raise ValueError("Target tokenizer mapping SHA256 mismatch.")
    if len(tokenizer) != int(proxy_configuration.get("shared_vocab_size", -1)):
        raise ValueError("Target tokenizer vocabulary size mismatch.")

    grouped: dict[str, list[PlanPair]] = {name: [] for name in selected_names}
    for pair in plan_pairs:
        grouped[pair.record_name].append(pair)
    prepared: list[PreparedPair] = []
    for name in selected_names:
        source_path = source_dir / name
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing linked source cache record: {source_path}")
        prepared.extend(
            _validate_source_record(
                source_path=source_path,
                source_summary=summaries[name],
                source_configuration=source_configuration,
                source_fingerprint=str(source_manifest["configuration_fingerprint"]),
                dataset=dataset,
                tokenizer=tokenizer,
                planned_pairs=grouped[name],
            )
        )
    by_id = {pair.plan.pair_id: pair for pair in prepared}
    prepared = [by_id[pair.pair_id] for pair in plan_pairs]
    planned_calls = sum(
        int(not pair.structural_zero and not arm.local_terminal)
        for pair in prepared
        for arm in pair.arms
    )
    planned_tokens = sum(
        arm.remaining_tokens
        for pair in prepared
        for arm in pair.arms
        if not pair.structural_zero and not arm.local_terminal
    )
    return Preflight(
        pairs=tuple(prepared),
        token_equal_rows=token_equal_rows,
        plan_row_count=plan_row_count,
        selected_source_records=tuple(selected_names),
        planned_api_calls=planned_calls,
        planned_requested_output_tokens=planned_tokens,
        proxy_manifest=proxy_manifest,
        proxy_manifest_sha256=sha256_file(proxy_manifest_path),
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha,
        dataset_fingerprint=dataset_fingerprint,
        dataset_revision=dataset_revision,
        tokenizer_sha256=tokenizer_sha,
        tokenizer_name_or_path=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        tokenizer=tokenizer,
    )


def preflight_summary(preflight: Preflight) -> dict[str, Any]:
    structural_zero_pairs = sum(pair.structural_zero for pair in preflight.pairs)
    return {
        "dry_run": True,
        "collector_protocol": COLLECTOR_PROTOCOL,
        "plan_rows": preflight.plan_row_count,
        "selected_pairs": len(preflight.pairs),
        "selected_mismatch_pairs": sum(
            pair.plan.partial_action != pair.plan.reference_action
            for pair in preflight.pairs
        ),
        "token_equal_rows": preflight.token_equal_rows,
        "structural_zero_pairs": structural_zero_pairs,
        "source_records": len(preflight.selected_source_records),
        "planned_api_calls": preflight.planned_api_calls,
        "planned_requested_output_tokens": preflight.planned_requested_output_tokens,
        "arms_without_api": 2 * len(preflight.pairs) - preflight.planned_api_calls,
        "terminal_arms": sum(
            arm.local_terminal for pair in preflight.pairs for arm in pair.arms
        ),
        "proxy_manifest_sha256": preflight.proxy_manifest_sha256,
        "source_manifest_sha256": preflight.source_manifest_sha256,
        "dataset_fingerprint": preflight.dataset_fingerprint,
        "dataset_revision": preflight.dataset_revision,
        "tokenizer_sha256": preflight.tokenizer_sha256,
        "target_model": TARGET_MODEL,
        "target_provider": TARGET_PROVIDER,
        "temperature": 0.0,
        "top_p": 1.0,
        "requests_per_nonterminal_arm": 1,
    }


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    data = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(path, data)


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    chunks = [
        json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
        for row in rows
    ]
    atomic_write_bytes(path, b"".join(chunks))


def state_directory(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.state")


def manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def audit_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".audit.json")


def reconciliation_export_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".reconciliation.jsonl")


def lock_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".lock")


def arm_id(pair_id: str, role: str) -> str:
    return f"{pair_id}:{role}"


def state_path(root: Path, identifier: str) -> Path:
    return root / f"{hashlib.sha256(identifier.encode('utf-8')).hexdigest()}.json"


def reconciliation_directory(state_root: Path) -> Path:
    return state_root / "reconciliation"


def load_states(root: Path, *, configuration_fingerprint: str) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return states
    for path in sorted(root.glob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get("arm_id"), str):
            raise ValueError(f"Invalid collector state file: {path}")
        identifier = state["arm_id"]
        if path != state_path(root, identifier):
            raise ValueError(f"Collector state filename/content mismatch: {path}")
        if state.get("configuration_fingerprint") != configuration_fingerprint:
            raise ValueError(f"Collector state configuration mismatch: {path}")
        if state.get("status") not in {"intent", "uncertain", "result"}:
            raise ValueError(f"Collector state has invalid status: {path}")
        if identifier in states:
            raise ValueError(f"Duplicate collector state for {identifier}.")
        states[identifier] = state
    return states


def load_reconciliation_entries(
    root: Path, *, configuration_fingerprint: str
) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return entries
    if not root.is_dir():
        raise ValueError(f"Reconciliation path is not a directory: {root}")
    for path in sorted(root.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(entry, dict)
            or entry.get("ledger_schema_version") != 1
            or entry.get("ledger_type")
            != "verified_local_token_overflow_retry_v1"
            or not isinstance(entry.get("arm_id"), str)
        ):
            raise ValueError(f"Invalid reconciliation ledger entry: {path}")
        identifier = entry["arm_id"]
        if path != state_path(root, identifier):
            raise ValueError(f"Reconciliation filename/content mismatch: {path}")
        if entry.get("configuration_fingerprint") != configuration_fingerprint:
            raise ValueError(f"Reconciliation configuration mismatch: {path}")
        if identifier in entries:
            raise ValueError(f"Duplicate reconciliation ledger for {identifier}.")
        original = entry.get("original_uncertain_state")
        if not isinstance(original, dict) or sha256_payload(original) != entry.get(
            "original_uncertain_state_sha256"
        ):
            raise ValueError(f"Reconciliation original-state hash mismatch: {path}")
        entries[identifier] = entry
    return entries


def materialize_reconciliation_ledger(
    path: Path, entries: Mapping[str, Mapping[str, Any]]
) -> None:
    atomic_write_jsonl(path, [entries[key] for key in sorted(entries)])


def _prepared_arms(preflight: Preflight) -> dict[str, tuple[PreparedPair, PreparedArm]]:
    return {
        arm_id(pair.plan.pair_id, arm.role): (pair, arm)
        for pair in preflight.pairs
        for arm in pair.arms
    }


def verify_local_overflow_uncertain(
    state: Mapping[str, Any],
    *,
    pair: PreparedPair,
    arm: PreparedArm,
) -> dict[str, Any]:
    """Validate the narrow historical failure eligible for one explicit retry."""

    if state.get("status") != "uncertain" or state.get("error_type") != "RuntimeError":
        raise ValueError("Only a RuntimeError uncertain state can be overflow-reconciled.")
    match = LOCAL_OVERFLOW_ERROR_RE.fullmatch(str(state.get("error", "")))
    if match is None:
        raise ValueError("Uncertain state is not the verified local-token-overflow error.")
    observed = int(match.group("observed"))
    remaining = int(match.group("remaining"))
    identifier = arm_id(pair.plan.pair_id, arm.role)
    expected_fields = {
        "arm_id": identifier,
        "pair_id": pair.plan.pair_id,
        "record_name": pair.plan.record_name,
        "dataset_idx": pair.plan.dataset_idx,
        "position": pair.plan.position,
        "role": arm.role,
        "action_token_id": arm.action_token_id,
        "remaining_tokens": arm.remaining_tokens,
    }
    differing = {
        key: (state.get(key), expected)
        for key, expected in expected_fields.items()
        if state.get(key) != expected
    }
    if differing:
        raise ValueError(f"Overflow uncertain state/plan mismatch: {differing!r}")
    if pair.structural_zero or arm.local_terminal:
        raise ValueError("A zero-call arm cannot have a valid overflow uncertain state.")
    if remaining != arm.remaining_tokens or observed <= remaining:
        raise ValueError("Overflow error counts do not match the planned arm budget.")
    delta = state.get("observed_client_delta")
    if not isinstance(delta, dict):
        raise ValueError("Overflow uncertain state has no client accounting delta.")
    calls = int(delta.get("calls", -1))
    attempts = int(delta.get("request_attempts", -1))
    tokens = int(delta.get("total_tokens", -1))
    cost = float(delta.get("total_cost", float("nan")))
    if calls != 1 or attempts != 1 or tokens < 0 or not math.isfinite(cost) or cost < 0:
        raise ValueError("Overflow uncertain state has invalid historical request accounting.")
    return {
        "verified_error": str(state["error"]),
        "observed_local_tokens": observed,
        "remaining_local_tokens": remaining,
        "historical_request_audit": {
            "successful_http_responses": calls,
            "api_request_attempts": attempts,
            "api_cost": cost,
            "total_tokens": tokens,
        },
    }


def prepare_reconciliation(
    *,
    preflight: Preflight,
    states: Mapping[str, Mapping[str, Any]],
    reconciliation_root: Path,
    configuration_fingerprint: str,
    recover_verified_local_overflow: bool,
    recovery_collector_source_sha256: str,
    reconciliation_output_path: Optional[Path] = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    """Archive eligible uncertain states and authorize one retry in memory."""

    mutable_states = {key: dict(value) for key, value in states.items()}
    entries = load_reconciliation_entries(
        reconciliation_root,
        configuration_fingerprint=configuration_fingerprint,
    )
    valid_arms = _prepared_arms(preflight)
    unexpected_entries = sorted(set(entries) - set(valid_arms))
    if unexpected_entries:
        raise ValueError(
            "Reconciliation ledger contains arms absent from this plan: "
            f"{unexpected_entries}."
        )
    unresolved = {
        identifier: state
        for identifier, state in mutable_states.items()
        if state.get("status") in {"intent", "uncertain"}
    }
    if not unresolved:
        if reconciliation_output_path is not None and entries:
            materialize_reconciliation_ledger(reconciliation_output_path, entries)
        return mutable_states, entries, set()
    if not recover_verified_local_overflow:
        raise RuntimeError(
            "Refusing to re-send arms with unresolved persisted intents: "
            f"{sorted(unresolved)}. Reconcile their provider billing/results first."
        )

    retry_ids: set[str] = set()
    for identifier, state in unresolved.items():
        if state.get("status") == "intent":
            raise RuntimeError(
                "Recovery never retries a bare intent with unknown outbound status: "
                f"{identifier}."
            )
        prepared = valid_arms.get(identifier)
        if prepared is None:
            raise ValueError(f"Unresolved arm is absent from this plan: {identifier}.")
        pair, arm = prepared
        verification = verify_local_overflow_uncertain(state, pair=pair, arm=arm)
        if identifier in entries:
            entry = entries[identifier]
            if sha256_payload(state) != entry.get("original_uncertain_state_sha256"):
                raise RuntimeError(
                    "An already reconciled arm failed or remained uncertain again; "
                    f"refusing a second retry: {identifier}."
                )
            if entry.get("verification") != verification:
                raise ValueError(
                    f"Persisted reconciliation verification mismatch: {identifier}."
                )
            # Idempotent restart after the ledger was persisted but before the
            # retry intent replaced the original uncertain state.
            del mutable_states[identifier]
            retry_ids.add(identifier)
            continue
        entry = {
            "ledger_schema_version": 1,
            "ledger_type": "verified_local_token_overflow_retry_v1",
            "arm_id": identifier,
            "pair_id": pair.plan.pair_id,
            "candidate_id": pair.plan.candidate_id,
            "configuration_fingerprint": configuration_fingerprint,
            "authorized_at_utc": utc_now(),
            "recovery_collector_source_sha256": recovery_collector_source_sha256,
            "original_uncertain_state_sha256": sha256_payload(state),
            "original_uncertain_state": dict(state),
            "verification": verification,
            "retry_limit": 1,
        }
        reconciliation_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        atomic_write_json(state_path(reconciliation_root, identifier), entry)
        entries[identifier] = entry
        retry_ids.add(identifier)
        # The state file remains untouched until collect_prepared_pairs writes a
        # new atomic intent.  Removing it only from this in-memory view prevents
        # any unlogged gap between archival and retry intent persistence.
        del mutable_states[identifier]

    if reconciliation_output_path is not None:
        materialize_reconciliation_ledger(reconciliation_output_path, entries)
    return mutable_states, entries, retry_ids


def _counter_delta(after: Mapping[Any, int], before: Mapping[Any, int]) -> dict[str, int]:
    keys = set(after) | set(before)
    result = {
        str(key): int(after.get(key, 0)) - int(before.get(key, 0))
        for key in keys
        if int(after.get(key, 0)) - int(before.get(key, 0))
    }
    if any(value < 0 for value in result.values()):
        raise ValueError("OpenRouter audit counters moved backwards.")
    return dict(sorted(result.items()))


def client_snapshot(client) -> dict[str, Any]:
    return {
        "calls": int(getattr(client, "calls", 0)),
        "request_attempts": int(getattr(client, "request_attempts", 0)),
        "total_cost": float(getattr(client, "total_cost", 0.0)),
        "total_tokens": int(getattr(client, "total_tokens", 0)),
        "provider_call_counts": dict(getattr(client, "provider_call_counts", {})),
        "response_model_call_counts": dict(
            getattr(client, "response_model_call_counts", {})
        ),
        "response_cache_status_counts": dict(
            getattr(client, "response_cache_status_counts", {})
        ),
        "api_max_tokens_call_counts": dict(
            getattr(client, "api_max_tokens_call_counts", {})
        ),
        "missing_router_metadata_calls": int(
            getattr(client, "missing_router_metadata_calls", 0)
        ),
        "reasoning_response_calls": int(
            getattr(client, "reasoning_response_calls", 0)
        ),
        "reasoning_tokens": int(getattr(client, "reasoning_tokens", 0)),
    }


def client_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "successful_http_responses": int(after["calls"]) - int(before["calls"]),
        "api_request_attempts": int(after["request_attempts"])
        - int(before["request_attempts"]),
        "api_cost": float(after["total_cost"]) - float(before["total_cost"]),
        "total_tokens": int(after["total_tokens"]) - int(before["total_tokens"]),
        "missing_router_metadata_calls": int(after["missing_router_metadata_calls"])
        - int(before["missing_router_metadata_calls"]),
        "reasoning_response_calls": int(after["reasoning_response_calls"])
        - int(before["reasoning_response_calls"]),
        "reasoning_tokens": int(after["reasoning_tokens"])
        - int(before["reasoning_tokens"]),
    }
    for field in (
        "provider_call_counts",
        "response_model_call_counts",
        "response_cache_status_counts",
        "api_max_tokens_call_counts",
    ):
        result[field] = _counter_delta(after[field], before[field])
    if (
        result["successful_http_responses"] != 1
        or result["api_request_attempts"] != 1
        or result["missing_router_metadata_calls"] != 0
        or result["reasoning_response_calls"] != 0
        or result["reasoning_tokens"] != 0
        or result["provider_call_counts"] != {TARGET_PROVIDER: 1}
        or result["response_model_call_counts"] != {TARGET_MODEL: 1}
    ):
        raise ValueError(f"Target routing/protocol audit failed: {result!r}")
    if not math.isfinite(result["api_cost"]) or result["api_cost"] < 0:
        raise ValueError("OpenRouter returned an invalid cost delta.")
    return result


def _result_row(
    pair: PreparedPair,
    arm: PreparedArm,
    *,
    continuation: str,
    handoff_audit: Mapping[str, Any],
    request_audit: Mapping[str, Any],
    configuration_fingerprint: str,
) -> dict[str, Any]:
    completion = f"{arm.visible_forced_prefix}{continuation}"
    branch_serialized = (
        pair.partial_branch_serialization
        if arm.role == "partial"
        else pair.reference_branch_serialization
    )
    branch_serialized_sha = (
        pair.partial_branch_serialization_sha256
        if arm.role == "partial"
        else pair.reference_branch_serialization_sha256
    )
    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "collector_protocol": COLLECTOR_PROTOCOL,
        "configuration_fingerprint": configuration_fingerprint,
        "prompt": pair.prompt,
        "completion": completion,
        "logibreak_group_id": pair.plan.pair_id,
        "completion_order": arm.completion_order,
        "target_completions_in_group": 2,
        "pair_id": pair.plan.pair_id,
        "candidate_id": pair.plan.candidate_id,
        "arm_id": arm_id(pair.plan.pair_id, arm.role),
        "arm_role": arm.role,
        "record_name": pair.plan.record_name,
        "dataset_idx": pair.plan.dataset_idx,
        "position": pair.plan.position,
        "action_token_id": arm.action_token_id,
        "partial_action": pair.plan.partial_action,
        "reference_action": pair.plan.reference_action,
        "forced_prefix_text": arm.visible_forced_prefix,
        "continuation": continuation,
        "remaining_tokens": arm.remaining_tokens,
        "local_terminal": arm.local_terminal,
        "structural_zero": pair.structural_zero,
        "structural_zero_reason": pair.structural_zero_reason,
        "branch_serialization": branch_serialized,
        "branch_serialization_sha256": branch_serialized_sha,
        "prefix_tokenization_audit": {
            "teacher_prefix_roundtrip": arm.teacher_prefix_roundtrip,
            "teacher_prefix_retokenized_ids": list(
                arm.teacher_prefix_retokenized_ids
            ),
            "teacher_prefix_retokenized_ids_sha256": (
                arm.teacher_prefix_retokenized_ids_sha256
            ),
            "forced_prefix_roundtrip": arm.forced_prefix_roundtrip,
            "forced_prefix_retokenized_ids": list(
                arm.forced_prefix_retokenized_ids
            ),
            "forced_prefix_retokenized_ids_sha256": (
                arm.forced_prefix_retokenized_ids_sha256
            ),
            "forced_extends_teacher_text": arm.forced_extends_teacher_text,
        },
        "data_sha256": pair.data_sha256,
        "source_record_sha256": pair.source_record_sha256,
        "plan_row_sha256": pair.plan.plan_row_sha256,
        "plan_row": pair.plan.plan_row,
        "target_protocol": {
            "model": TARGET_MODEL,
            "provider_order": [TARGET_PROVIDER],
            "provider_allow_fallbacks": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "reasoning_mode": "enabled_false",
            "reject_reasoning_tokens": True,
            "qwen_hard_no_think_prefill": True,
            "append_no_think": True,
            "disable_openrouter_response_cache": True,
            "one_request_continuation": True,
        },
        "handoff_audit": dict(handoff_audit),
        "request_audit": dict(request_audit),
    }


def _local_result(
    pair: PreparedPair,
    arm: PreparedArm,
    *,
    configuration_fingerprint: str,
) -> dict[str, Any]:
    return _result_row(
        pair,
        arm,
        continuation="",
        handoff_audit={
            "occurred": False,
            "reason": "eos_action" if arm.remaining_tokens > 0 else "answer_budget_exhausted",
            "remaining_tokens_requested": 0,
        },
        request_audit={
            "successful_http_responses": 0,
            "api_request_attempts": 0,
            "api_cost": 0.0,
            "total_tokens": 0,
            "provider_call_counts": {},
            "response_model_call_counts": {},
            "response_cache_status_counts": {},
            "api_max_tokens_call_counts": {},
        },
        configuration_fingerprint=configuration_fingerprint,
    )


def _structural_zero_result(
    pair: PreparedPair,
    arm: PreparedArm,
    *,
    configuration_fingerprint: str,
) -> dict[str, Any]:
    if not pair.structural_zero:
        raise ValueError("Structural-zero result requested for unequal branches.")
    if pair.partial_branch_serialization != pair.reference_branch_serialization:
        raise ValueError("Structural-zero branch serializations are not identical.")
    if pair.partial.visible_forced_prefix != pair.reference.visible_forced_prefix:
        raise ValueError("Identical serialized branches produced different visible prefixes.")
    return _result_row(
        pair,
        arm,
        continuation="",
        handoff_audit={
            "occurred": False,
            "reason": "structural_zero",
            "structural_zero_reason": pair.structural_zero_reason,
            "remaining_tokens_requested": 0,
        },
        request_audit={
            "successful_http_responses": 0,
            "api_request_attempts": 0,
            "api_cost": 0.0,
            "total_tokens": 0,
            "provider_call_counts": {},
            "response_model_call_counts": {},
            "response_cache_status_counts": {},
            "api_max_tokens_call_counts": {},
        },
        configuration_fingerprint=configuration_fingerprint,
    )


def _state_result(
    row: Mapping[str, Any], *, configuration_fingerprint: str
) -> dict[str, Any]:
    return {
        "state_schema_version": 1,
        "status": "result",
        "arm_id": row["arm_id"],
        "pair_id": row["pair_id"],
        "configuration_fingerprint": configuration_fingerprint,
        "updated_at_utc": utc_now(),
        "row": dict(row),
    }


def materialize_completed_output(
    output_path: Path,
    pairs: Sequence[PreparedPair],
    states: Mapping[str, Mapping[str, Any]],
) -> int:
    rows: list[dict[str, Any]] = []
    complete_pairs = 0
    for pair in pairs:
        pair_states = [states.get(arm_id(pair.plan.pair_id, arm.role)) for arm in pair.arms]
        if all(state is not None and state.get("status") == "result" for state in pair_states):
            complete_pairs += 1
            rows.extend(dict(state["row"]) for state in pair_states if state is not None)
    atomic_write_jsonl(output_path, rows)
    return complete_pairs


def reconciliation_totals(
    entries: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    audits = [
        entry.get("verification", {}).get("historical_request_audit", {})
        for entry in entries.values()
    ]
    return {
        "successful_http_responses": sum(
            int(audit.get("successful_http_responses", 0)) for audit in audits
        ),
        "api_request_attempts": sum(
            int(audit.get("api_request_attempts", 0)) for audit in audits
        ),
        "api_cost": sum(float(audit.get("api_cost", 0.0)) for audit in audits),
        "total_tokens": sum(int(audit.get("total_tokens", 0)) for audit in audits),
    }


def collection_summary(
    *,
    preflight: Preflight,
    states: Mapping[str, Mapping[str, Any]],
    hard_cap: int,
    configuration_fingerprint: str,
    reconciliation_entries: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> dict[str, Any]:
    reconciliation_entries = reconciliation_entries or {}
    results = [state for state in states.values() if state.get("status") == "result"]
    unresolved = [
        identifier
        for identifier, state in states.items()
        if state.get("status") in {"intent", "uncertain"}
    ]
    rows = [state["row"] for state in results]
    request_audits = [row.get("request_audit") or {} for row in rows]
    current_totals = {
        "successful_http_responses": sum(
            int(audit.get("successful_http_responses", 0)) for audit in request_audits
        ),
        "api_request_attempts": sum(
            int(audit.get("api_request_attempts", 0)) for audit in request_audits
        ),
        "api_cost": sum(float(audit.get("api_cost", 0.0)) for audit in request_audits),
        "total_tokens": sum(int(audit.get("total_tokens", 0)) for audit in request_audits),
    }
    historical_totals = reconciliation_totals(reconciliation_entries)
    complete_pair_ids = Counter(row.get("pair_id") for row in rows)
    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "collector_protocol": COLLECTOR_PROTOCOL,
        "configuration_fingerprint": configuration_fingerprint,
        "updated_at_utc": utc_now(),
        "planned_pairs": len(preflight.pairs),
        "planned_arms": 2 * len(preflight.pairs),
        "token_equal_rows": preflight.token_equal_rows,
        "structural_zero_pairs": sum(
            pair.structural_zero for pair in preflight.pairs
        ),
        "reconciled_local_overflow_arms": len(reconciliation_entries),
        "completed_pairs": sum(count == 2 for count in complete_pair_ids.values()),
        "completed_arms": len(results),
        "unresolved_arm_ids": sorted(unresolved),
        "reserved_request_attempts": sum(
            int(state.get("status") in {"intent", "uncertain"})
            + int((state.get("row") or {}).get("request_audit", {}).get("api_request_attempts", 0))
            for state in states.values()
        )
        + historical_totals["api_request_attempts"],
        "successful_result_request_attempts": current_totals["api_request_attempts"],
        "historical_reconciled_failed_request_attempts": historical_totals[
            "api_request_attempts"
        ],
        "confirmed_request_attempts": current_totals["api_request_attempts"]
        + historical_totals["api_request_attempts"],
        "successful_result_http_responses": current_totals[
            "successful_http_responses"
        ],
        "historical_reconciled_http_responses": historical_totals[
            "successful_http_responses"
        ],
        "successful_http_responses": current_totals["successful_http_responses"]
        + historical_totals["successful_http_responses"],
        "successful_result_api_cost": current_totals["api_cost"],
        "historical_reconciled_failed_api_cost": historical_totals["api_cost"],
        "api_cost": current_totals["api_cost"] + historical_totals["api_cost"],
        "successful_result_api_total_tokens": current_totals["total_tokens"],
        "historical_reconciled_api_total_tokens": historical_totals["total_tokens"],
        "api_total_tokens": current_totals["total_tokens"]
        + historical_totals["total_tokens"],
        "max_api_request_attempts": hard_cap,
    }


def collect_prepared_pairs(
    *,
    preflight: Preflight,
    output_path: Path,
    state_root: Path,
    configuration_fingerprint: str,
    hard_cap: int,
    client,
    generator: Callable[..., tuple[str, dict[str, Any]]] = generate_collector_continuation,
    recover_verified_local_overflow: bool = False,
    reconciliation_root: Optional[Path] = None,
    reconciliation_output_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Collect missing arms sequentially; every request has a durable intent."""

    states = load_states(state_root, configuration_fingerprint=configuration_fingerprint)
    valid_arm_ids = {
        arm_id(pair.plan.pair_id, arm.role)
        for pair in preflight.pairs
        for arm in pair.arms
    }
    unexpected = sorted(set(states) - valid_arm_ids)
    if unexpected:
        raise ValueError(f"State directory contains arms absent from this plan: {unexpected}.")
    effective_reconciliation_root = (
        reconciliation_root
        if reconciliation_root is not None
        else reconciliation_directory(state_root)
    )
    states, reconciliation_entries, retry_ids = prepare_reconciliation(
        preflight=preflight,
        states=states,
        reconciliation_root=effective_reconciliation_root,
        configuration_fingerprint=configuration_fingerprint,
        recover_verified_local_overflow=recover_verified_local_overflow,
        recovery_collector_source_sha256=sha256_file(Path(__file__).resolve()),
        reconciliation_output_path=reconciliation_output_path,
    )
    historical_totals = reconciliation_totals(reconciliation_entries)
    reserved = sum(
        int((state.get("row") or {}).get("request_audit", {}).get("api_request_attempts", 0))
        for state in states.values()
    ) + int(historical_totals["api_request_attempts"])
    missing_calls = sum(
        int(
            not pair.structural_zero
            and not arm.local_terminal
            and arm_id(pair.plan.pair_id, arm.role) not in states
        )
        for pair in preflight.pairs
        for arm in pair.arms
    )
    if reserved + missing_calls > hard_cap:
        raise ValueError(
            "The hard global request cap cannot complete the selected plan: "
            f"{reserved}+{missing_calls}>{hard_cap}."
        )
    client.args.max_api_request_attempts = hard_cap - reserved

    for pair in preflight.pairs:
        for arm in pair.arms:
            identifier = arm_id(pair.plan.pair_id, arm.role)
            if identifier in states:
                continue
            destination = state_path(state_root, identifier)
            if pair.structural_zero:
                row = _structural_zero_result(
                    pair, arm, configuration_fingerprint=configuration_fingerprint
                )
                state = _state_result(
                    row, configuration_fingerprint=configuration_fingerprint
                )
                atomic_write_json(destination, state)
                states[identifier] = state
                continue
            if arm.local_terminal:
                row = _local_result(
                    pair, arm, configuration_fingerprint=configuration_fingerprint
                )
                state = _state_result(
                    row, configuration_fingerprint=configuration_fingerprint
                )
                atomic_write_json(destination, state)
                states[identifier] = state
                continue

            intent = {
                "state_schema_version": 1,
                "status": "intent",
                "arm_id": identifier,
                "pair_id": pair.plan.pair_id,
                "record_name": pair.plan.record_name,
                "dataset_idx": pair.plan.dataset_idx,
                "position": pair.plan.position,
                "role": arm.role,
                "action_token_id": arm.action_token_id,
                "remaining_tokens": arm.remaining_tokens,
                "configuration_fingerprint": configuration_fingerprint,
                "created_at_utc": utc_now(),
            }
            if identifier in retry_ids:
                ledger = reconciliation_entries[identifier]
                intent["reconciliation"] = {
                    "ledger_type": ledger["ledger_type"],
                    "original_uncertain_state_sha256": ledger[
                        "original_uncertain_state_sha256"
                    ],
                    "retry_ordinal": 1,
                }
            atomic_write_json(destination, intent)
            states[identifier] = intent
            before = client_snapshot(client)
            try:
                continuation, handoff_audit = generator(
                    client=client,
                    tokenizer=preflight.tokenizer,
                    prompt=pair.api_prompt,
                    prefix_text=arm.raw_forced_prefix,
                    remaining_tokens=arm.remaining_tokens,
                    temperature=0.0,
                    top_p=1.0,
                )
                after = client_snapshot(client)
                request_audit = client_delta(before, after)
                if handoff_audit.get("reasoning_detected"):
                    raise ValueError("One-shot continuation exposed reasoning.")
                if handoff_audit.get("response_model") != TARGET_MODEL:
                    raise ValueError("One-shot continuation response model mismatch.")
                row = _result_row(
                    pair,
                    arm,
                    continuation=continuation,
                    handoff_audit=handoff_audit,
                    request_audit=request_audit,
                    configuration_fingerprint=configuration_fingerprint,
                )
                if identifier in reconciliation_entries:
                    ledger = reconciliation_entries[identifier]
                    row["reconciliation"] = {
                        "ledger_type": ledger["ledger_type"],
                        "original_uncertain_state_sha256": ledger[
                            "original_uncertain_state_sha256"
                        ],
                        "ledger_record_sha256": sha256_payload(ledger),
                        "historical_request_audit": ledger["verification"][
                            "historical_request_audit"
                        ],
                        "verified_error": ledger["verification"]["verified_error"],
                        "retry_ordinal": 1,
                    }
                state = _state_result(
                    row, configuration_fingerprint=configuration_fingerprint
                )
                atomic_write_json(destination, state)
                states[identifier] = state
            except BaseException as exc:
                after = client_snapshot(client)
                uncertain = {
                    **intent,
                    "status": "uncertain",
                    "updated_at_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                    "observed_client_delta": {
                        key: (
                            float(after[key]) - float(before[key])
                            if key in {"total_cost"}
                            else int(after[key]) - int(before[key])
                        )
                        for key in ("calls", "request_attempts", "total_cost", "total_tokens")
                    },
                }
                if identifier in reconciliation_entries:
                    uncertain["reconciliation"] = {
                        "ledger_record_sha256": sha256_payload(
                            reconciliation_entries[identifier]
                        ),
                        "retry_ordinal": 1,
                    }
                atomic_write_json(destination, uncertain)
                states[identifier] = uncertain
                materialize_completed_output(output_path, preflight.pairs, states)
                raise

        materialize_completed_output(output_path, preflight.pairs, states)
    return collection_summary(
        preflight=preflight,
        states=states,
        hard_cap=hard_cap,
        configuration_fingerprint=configuration_fingerprint,
        reconciliation_entries=reconciliation_entries,
    )


def collector_configuration(
    *, args: argparse.Namespace, preflight: Preflight, plan_sha256: str
) -> dict[str, Any]:
    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "collector_protocol": COLLECTOR_PROTOCOL,
        "collector_source_sha256": sha256_file(Path(__file__).resolve()),
        "plan_path": str(Path(args.plan).expanduser().resolve()),
        "plan_sha256": plan_sha256,
        "proxy_cache_dir": str(Path(args.proxy_cache_dir).expanduser().resolve()),
        "proxy_manifest_sha256": preflight.proxy_manifest_sha256,
        "source_manifest_sha256": preflight.source_manifest_sha256,
        "dataset_revision": preflight.dataset_revision,
        "dataset_fingerprint": preflight.dataset_fingerprint,
        "tokenizer_name_or_path": preflight.tokenizer_name_or_path,
        "tokenizer_revision": preflight.tokenizer_revision,
        "tokenizer_sha256": preflight.tokenizer_sha256,
        "selected_pair_ids": [pair.plan.pair_id for pair in preflight.pairs],
        "max_pairs": args.max_pairs,
        "target_model": TARGET_MODEL,
        "target_provider_order": [TARGET_PROVIDER],
        "target_provider_allow_fallbacks": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning_mode": "enabled_false",
        "reject_reasoning_tokens": True,
        "qwen_hard_no_think_prefill": True,
        "append_no_think": True,
        "disable_openrouter_response_cache": True,
        "max_api_request_attempts": int(args.max_api_request_attempts),
        "network_retries": 0,
        "reasoning_retries": 0,
    }


def build_client_args(args: argparse.Namespace, *, remaining_cap: int) -> argparse.Namespace:
    return argparse.Namespace(
        model=TARGET_MODEL,
        api_url=OPENROUTER_URL,
        site_url=args.site_url,
        app_name=args.app_name,
        request_timeout=args.request_timeout,
        max_retries=0,
        retry_sleep=0.0,
        max_api_request_attempts=remaining_cap,
        max_mc_requested_samples=None,
        reasoning_mode="enabled_false",
        reject_reasoning_tokens=True,
        max_reasoning_retries=0,
        reasoning_retry_sleep=0.0,
        reasoning_fallback_temperature=None,
        reasoning_fallback_mode=None,
        max_reasoning_fallback_retries=0,
        provider_order=[TARGET_PROVIDER],
        provider_allow_fallbacks=False,
        provider_quantizations=None,
        router_metadata=True,
        disable_openrouter_response_cache=True,
        qwen_hard_no_think_prefill=True,
        api_key=None,
        api_key_env=args.api_key_env,
        api_key_file=args.api_key_file,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one-shot paired continuations for outcome-aware gate labels."
    )
    parser.add_argument("--plan", required=True, help="Precomputed JSON/JSONL action plan.")
    parser.add_argument("--proxy_cache_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--max_api_request_attempts", type=int, default=None)
    parser.add_argument("--api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument("--api_key_file", default=None)
    parser.add_argument("--site_url", default="https://anonymous.invalid")
    parser.add_argument("--app_name", default="Anonymous outcome-regret collector")
    parser.add_argument("--request_timeout", type=float, default=90.0)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--recover_verified_local_overflow",
        action="store_true",
        help=(
            "With --resume only, archive and retry an uncertain arm whose sole "
            "persisted failure is the verified post-response local-token-overflow "
            "validation. Bare intents and every other uncertain failure remain blocked."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run complete local preflight without reading an API key or writing output.",
    )
    return parser.parse_args(argv)


def _validate_cli(args: argparse.Namespace) -> None:
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise ValueError("--max_pairs must be positive.")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        raise ValueError("--request_timeout must be finite and positive.")
    if not args.dry_run and (
        args.max_api_request_attempts is None or args.max_api_request_attempts <= 0
    ):
        raise ValueError(
            "A positive --max_api_request_attempts is required for collection."
        )
    if args.recover_verified_local_overflow and not args.resume:
        raise ValueError(
            "--recover_verified_local_overflow requires --resume."
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _validate_cli(args)
    preflight = run_preflight(args)
    summary = preflight_summary(preflight)
    if args.dry_run:
        print(json.dumps(summary, sort_keys=True, ensure_ascii=False), flush=True)
        return
    assert args.max_api_request_attempts is not None
    if preflight.planned_api_calls > args.max_api_request_attempts:
        raise ValueError(
            "The selected plan requires more calls than --max_api_request_attempts: "
            f"{preflight.planned_api_calls}>{args.max_api_request_attempts}."
        )

    output_path = Path(args.output_jsonl).expanduser().resolve()
    state_root = state_directory(output_path)
    run_manifest_path = manifest_path(output_path)
    run_audit_path = audit_path(output_path)
    run_reconciliation_path = reconciliation_export_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = [
        path
        for path in (
            output_path,
            state_root,
            run_manifest_path,
            run_audit_path,
            run_reconciliation_path,
        )
        if path.exists()
    ]
    if existing and not args.resume:
        raise FileExistsError(
            "Refusing to overwrite existing collector artifacts: "
            f"{[str(path) for path in existing]}. Pass --resume."
        )
    if args.resume and (not state_root.is_dir() or not run_manifest_path.is_file()):
        raise FileNotFoundError("--resume requires both the state directory and run manifest.")

    lock_file = lock_path(output_path)
    lock_handle = lock_file.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"Another collector holds the run lock: {lock_file}") from exc
    try:
        configuration = collector_configuration(
            args=args,
            preflight=preflight,
            plan_sha256=sha256_file(Path(args.plan).expanduser().resolve()),
        )
        configuration_fingerprint = sha256_payload(configuration)
        if args.resume:
            existing_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            existing_configuration = existing_manifest.get("configuration")
            existing_fingerprint = existing_manifest.get("configuration_fingerprint")
            if (
                not isinstance(existing_configuration, dict)
                or sha256_payload(existing_configuration) != existing_fingerprint
            ):
                raise ValueError("Existing run manifest fingerprint is invalid.")
            if existing_fingerprint != configuration_fingerprint:
                differing = sorted(
                    key
                    for key in set(existing_configuration) | set(configuration)
                    if existing_configuration.get(key) != configuration.get(key)
                )
                if not (
                    args.recover_verified_local_overflow
                    and differing == ["collector_source_sha256"]
                ):
                    raise ValueError(
                        "Resume configuration does not match the run manifest; "
                        f"differing fields: {differing}."
                    )
                # Preserve the original immutable run identity.  The recovery
                # ledger records the new implementation hash that performed the
                # narrowly authorized retry.
                configuration_fingerprint = str(existing_fingerprint)
        else:
            state_root.mkdir(mode=0o700, exist_ok=False)
            atomic_write_json(
                run_manifest_path,
                {
                    "manifest_schema_version": 1,
                    "created_at_utc": utc_now(),
                    "configuration": configuration,
                    "configuration_fingerprint": configuration_fingerprint,
                    "local_preflight": summary,
                },
            )

        states = load_states(
            state_root, configuration_fingerprint=configuration_fingerprint
        )
        states, reconciliation_entries, _ = prepare_reconciliation(
            preflight=preflight,
            states=states,
            reconciliation_root=reconciliation_directory(state_root),
            configuration_fingerprint=configuration_fingerprint,
            recover_verified_local_overflow=bool(
                args.recover_verified_local_overflow
            ),
            recovery_collector_source_sha256=sha256_file(Path(__file__).resolve()),
            reconciliation_output_path=run_reconciliation_path,
        )
        historical_totals = reconciliation_totals(reconciliation_entries)
        reserved = sum(
            int((state.get("row") or {}).get("request_audit", {}).get("api_request_attempts", 0))
            for state in states.values()
        ) + int(historical_totals["api_request_attempts"])
        if reserved > int(args.max_api_request_attempts):
            raise ValueError(
                "Historical/current request attempts already exceed the hard cap: "
                f"{reserved}>{int(args.max_api_request_attempts)}."
            )
        client_args = build_client_args(
            args, remaining_cap=int(args.max_api_request_attempts) - reserved
        )
        # Deliberately after all local validation, state/config checks, and pending
        # intent checks.  In particular, --dry_run never reaches this line.
        api_key = resolve_api_key(client_args)
        client = OpenRouterClient(client_args, api_key)
        try:
            final_summary = collect_prepared_pairs(
                preflight=preflight,
                output_path=output_path,
                state_root=state_root,
                configuration_fingerprint=configuration_fingerprint,
                hard_cap=int(args.max_api_request_attempts),
                client=client,
                recover_verified_local_overflow=bool(
                    args.recover_verified_local_overflow
                ),
                reconciliation_root=reconciliation_directory(state_root),
                reconciliation_output_path=run_reconciliation_path,
            )
        except BaseException as exc:
            # The per-arm state is authoritative.  This sidecar makes partial
            # spend visible without weakening the no-resend rule on resume.
            failed_states = load_states(
                state_root, configuration_fingerprint=configuration_fingerprint
            )
            failed_summary = collection_summary(
                preflight=preflight,
                states=failed_states,
                hard_cap=int(args.max_api_request_attempts),
                configuration_fingerprint=configuration_fingerprint,
                reconciliation_entries=load_reconciliation_entries(
                    reconciliation_directory(state_root),
                    configuration_fingerprint=configuration_fingerprint,
                ),
            )
            failed_summary.update(
                {
                    "run_status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                }
            )
            atomic_write_json(run_audit_path, failed_summary)
            raise
        final_summary["run_status"] = "complete"
        atomic_write_json(run_audit_path, final_summary)
        print(json.dumps(final_summary, sort_keys=True, ensure_ascii=False), flush=True)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    main()
