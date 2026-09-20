"""Utilities for exact partial-K replay of proxy + MC caches.

The source cache stores aggregate MC50 counts rather than sampling order.  MC
draws are exchangeable, so a seeded permutation of the count-expanded event
multiset gives a conditionally exact nested replay path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


TARGET_PROTOCOL_SCHEMA_VERSION = 1
TARGET_PROTOCOL_SOURCE_FIELDS = {
    "api_url": "target_api_url",
    "model": "target_model",
    "provider_order": "target_provider_order",
    "provider_allow_fallbacks": "target_provider_allow_fallbacks",
    "provider_quantizations": "target_provider_quantizations",
    "append_no_think": "target_append_no_think",
    "qwen_hard_no_think_prefill": "target_qwen_hard_no_think_prefill",
    "api_max_tokens": "target_api_max_tokens",
    "empty_response_token": "target_empty_response_token",
    "disable_openrouter_response_cache": (
        "target_disable_openrouter_response_cache"
    ),
    "sample_choices_per_request": "target_sample_choices_per_request",
    "max_sample_refill_rounds": "target_max_sample_refill_rounds",
    "empty_length_retry_max_tokens": "target_empty_length_retry_max_tokens",
    "max_empty_length_retry_rounds": "target_max_empty_length_retry_rounds",
    "reasoning_mode": "target_reasoning_mode",
    "reject_reasoning_tokens": "target_reject_reasoning_tokens",
    "max_reasoning_retries": "target_max_reasoning_retries",
    "reasoning_fallback_temperature": "target_reasoning_fallback_temperature",
    "reasoning_fallback_mode": "target_reasoning_fallback_mode",
    "max_reasoning_fallback_retries": (
        "target_max_reasoning_fallback_retries"
    ),
}
TARGET_PROTOCOL_METADATA_FIELDS = (
    "target_protocol_schema_version",
    *TARGET_PROTOCOL_SOURCE_FIELDS.values(),
)
_TARGET_PROTOCOL_REQUIRED_SOURCE_FIELDS = (
    "api_url",
    "model",
    "provider_order",
    "provider_allow_fallbacks",
    "append_no_think",
    "qwen_hard_no_think_prefill",
    "api_max_tokens",
    "empty_response_token",
    "disable_openrouter_response_cache",
    "sample_choices_per_request",
    "max_sample_refill_rounds",
    "empty_length_retry_max_tokens",
    "max_empty_length_retry_rounds",
    "reasoning_mode",
    "reject_reasoning_tokens",
)
_TARGET_PROTOCOL_LEGACY_DEFAULTS = {
    # These controls were added after the exact-MC50 source cache was built.
    # Their absence in a schema-v1 source manifest means the old zero/disabled
    # behavior, not an unconstrained runtime value.
    "provider_quantizations": None,
    "max_reasoning_retries": 0,
    "reasoning_fallback_temperature": None,
    "reasoning_fallback_mode": None,
    "max_reasoning_fallback_retries": 0,
}
POLICY_SPEC_HASH_FIELD = "policy_spec_payload_sha256"
PORTABLE_STOPPER_PROTOCOL = "python_float64_scalar_v1"
BIASNET_PARAMETER_DTYPE = "float32"
BIASNET_AUTOCAST_ENABLED = True
BIASNET_AUTOCAST_DTYPE = "float16"


def target_protocol_metadata_from_manifest(manifest: dict) -> dict[str, Any]:
    """Extract the black-box sampling contract frozen by the source cache."""

    source_manifest = manifest.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise ValueError("Proxy MC manifest has no embedded source_manifest.")
    source_configuration = source_manifest.get("configuration")
    if not isinstance(source_configuration, dict):
        raise ValueError("Source manifest has no configuration mapping.")
    missing = [
        field
        for field in _TARGET_PROTOCOL_REQUIRED_SOURCE_FIELDS
        if field not in source_configuration
    ]
    if missing:
        raise ValueError(
            "Source manifest is missing target-protocol fields: "
            f"{sorted(missing)}."
        )
    protocol: dict[str, Any] = {
        "target_protocol_schema_version": TARGET_PROTOCOL_SCHEMA_VERSION
    }
    for source_field, metadata_field in TARGET_PROTOCOL_SOURCE_FIELDS.items():
        if source_field in source_configuration:
            value = source_configuration[source_field]
        else:
            value = _TARGET_PROTOCOL_LEGACY_DEFAULTS[source_field]
        if isinstance(value, tuple):
            value = list(value)
        protocol[metadata_field] = value
    return protocol


def policy_spec_payload_sha256(payload: dict) -> str:
    """Hash a policy payload canonically, excluding its self-hash field."""

    canonical_payload = {
        key: value for key, value in payload.items() if key != POLICY_SPEC_HASH_FIELD
    }
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def portable_linear_stopper_confidence(
    features: Iterable[float],
    *,
    scaler_mean: Iterable[float],
    scaler_scale: Iterable[float],
    coefficient: Iterable[float],
    intercept: float,
) -> float:
    """Evaluate the exact scalar rule used by calibration and online stopping.

    Both callers first materialize features as float32.  Converting those values
    to Python floats here fixes one arithmetic order for standardization, the
    linear rule, and the stable sigmoid, avoiding sklearn/runtime boundary drift.
    """

    values = tuple(float(value) for value in features)
    means = tuple(float(value) for value in scaler_mean)
    scales = tuple(float(value) for value in scaler_scale)
    coefficients = tuple(float(value) for value in coefficient)
    width = len(values)
    if not (len(means) == len(scales) == len(coefficients) == width):
        raise ValueError("Portable stopper vectors must have one common width.")
    if width == 0:
        raise ValueError("Portable stopper vectors cannot be empty.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Portable stopper features must be finite.")
    if any(
        not math.isfinite(value)
        for value in (*means, *scales, *coefficients, float(intercept))
    ):
        raise ValueError("Portable stopper parameters must be finite.")
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("Portable stopper scales must be positive.")
    logit = float(intercept) + sum(
        weight * ((value - mean) / scale)
        for value, mean, scale, weight in zip(
            values, means, scales, coefficients
        )
    )
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


@dataclass
class AnytimeProxyMCCache:
    proxy_logits: torch.Tensor
    events: torch.Tensor
    labels: torch.Tensor
    base_token_ids: torch.Tensor
    position_ids: torch.Tensor
    record_ids: torch.Tensor
    row_keys: list[str]
    record_names: list[str]
    metadata: dict

    @property
    def num_rows(self) -> int:
        return int(self.labels.numel())

    @property
    def vocab_size(self) -> int:
        return int(self.proxy_logits.shape[-1])

    @property
    def max_samples(self) -> int:
        return int(self.events.shape[-1])


def parse_budgets(value: str | Iterable[int], *, max_samples: int) -> tuple[int, ...]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
        budgets = tuple(int(part) for part in raw_values)
    else:
        budgets = tuple(int(part) for part in value)
    if not budgets:
        raise ValueError("At least one MC budget is required.")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("MC budgets must be strictly increasing and unique.")
    if budgets[0] != 0 or budgets[-1] != int(max_samples):
        raise ValueError(
            f"MC budgets must start at 0 and end at {max_samples}."
        )
    if any(budget > max_samples for budget in budgets):
        raise ValueError(f"MC budgets cannot exceed {max_samples}.")
    return budgets


def deterministic_record_order(
    record_names: Sequence[str], *, split_seed: str
) -> list[str]:
    if len(record_names) != len(set(record_names)):
        raise ValueError("record_names must be unique.")
    return sorted(
        record_names,
        key=lambda name: hashlib.sha256(
            f"{split_seed}:{name}".encode("utf-8")
        ).hexdigest(),
    )


def split_records(
    record_names: Sequence[str],
    *,
    test_count: int,
    calibration_count: int,
    split_seed: str,
) -> dict[str, list[str]]:
    ordered = deterministic_record_order(record_names, split_seed=split_seed)
    if test_count < 0 or calibration_count < 0:
        raise ValueError("Split counts must be non-negative.")
    if test_count + calibration_count >= len(ordered):
        raise ValueError("The split must leave at least one training record.")
    test = sorted(ordered[:test_count])
    calibration = sorted(ordered[test_count : test_count + calibration_count])
    held_out = set(test) | set(calibration)
    train = sorted(name for name in ordered if name not in held_out)
    return {"train": train, "calibration": calibration, "test": test}


def counts_to_events(mc_counts: torch.Tensor) -> torch.Tensor:
    """Expand dense integer count rows into equal-length event multisets."""

    if mc_counts.dim() != 2:
        raise ValueError("mc_counts must have shape [rows, vocab].")
    if mc_counts.dtype == torch.bool or mc_counts.is_floating_point():
        raise TypeError("mc_counts must use an integer dtype.")
    if (mc_counts < 0).any():
        raise ValueError("mc_counts must be non-negative.")
    totals = mc_counts.sum(dim=-1).long()
    if totals.numel() == 0:
        raise ValueError("mc_counts must contain at least one row.")
    unique_totals = torch.unique(totals)
    if unique_totals.numel() != 1 or int(unique_totals.item()) <= 0:
        raise ValueError("Every count row must have the same positive total.")
    sample_count = int(unique_totals.item())
    events = torch.empty(
        (mc_counts.shape[0], sample_count), dtype=torch.long
    )
    for row_index, row in enumerate(mc_counts):
        token_ids = torch.nonzero(row, as_tuple=False).flatten()
        repeated = torch.repeat_interleave(token_ids, row[token_ids].long())
        if repeated.numel() != sample_count:
            raise RuntimeError("Count expansion produced an invalid event total.")
        events[row_index] = repeated
    return events


def _row_seed(row_key: str, *, replay_seed: int, replicate: int) -> int:
    digest = hashlib.sha256(
        f"{replay_seed}:{replicate}:{row_key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def permute_events(
    events: torch.Tensor,
    row_keys: Sequence[str],
    *,
    replay_seed: int,
    replicate: int,
) -> torch.Tensor:
    if events.dim() != 2:
        raise ValueError("events must have shape [rows, samples].")
    if len(row_keys) != events.shape[0]:
        raise ValueError("row_keys must match the number of event rows.")
    permuted = torch.empty_like(events)
    for row_index, row_key in enumerate(row_keys):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _row_seed(row_key, replay_seed=replay_seed, replicate=replicate)
        )
        order = torch.randperm(events.shape[1], generator=generator)
        permuted[row_index] = events[row_index, order]
    return permuted


def prefix_counts(
    permuted_events: torch.Tensor,
    *,
    budget: int,
    vocab_size: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if permuted_events.dim() != 2:
        raise ValueError("permuted_events must have shape [rows, samples].")
    budget = int(budget)
    if budget < 0 or budget > permuted_events.shape[1]:
        raise ValueError("budget must lie within the available replay path.")
    counts = torch.zeros(
        (permuted_events.shape[0], int(vocab_size)),
        device=permuted_events.device,
        dtype=dtype,
    )
    if budget:
        selected = permuted_events[:, :budget]
        increments = torch.ones_like(selected, dtype=dtype)
        counts.scatter_add_(1, selected, increments)
    return counts


def _load_manifest(cache_dir: Path) -> dict:
    manifest_path = cache_dir / "proxy_mc_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing proxy MC manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema_version") != 1:
        raise ValueError("Unsupported proxy MC manifest schema version.")
    return manifest


def load_anytime_cache(
    cache_dir: str | Path,
    *,
    record_names: Sequence[str] | None = None,
) -> AnytimeProxyMCCache:
    cache_path = Path(cache_dir).expanduser().resolve()
    manifest = _load_manifest(cache_path)
    target_protocol = target_protocol_metadata_from_manifest(manifest)
    available = sorted(path.name for path in cache_path.glob("*.pt"))
    selected = available if record_names is None else list(record_names)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise FileNotFoundError(f"Missing cache records: {missing}")
    if not selected:
        raise ValueError("No cache records selected.")

    proxy_chunks = []
    event_chunks = []
    label_chunks = []
    base_chunks = []
    position_chunks = []
    record_id_chunks = []
    row_keys: list[str] = []
    vocab_size = None
    max_samples = None
    metadata_reference = None

    for record_id, name in enumerate(selected):
        payload = torch.load(cache_path / name, map_location="cpu", weights_only=False)
        proxy_logits = payload.get("proxy_logits")
        mc_counts = payload.get("mc_counts")
        labels = payload.get("labels")
        base_token_ids = payload.get("risk_gate_token_ids")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (proxy_logits, mc_counts, labels, base_token_ids)
        ):
            raise ValueError(f"{name} is missing required anytime replay tensors.")
        if proxy_logits.dim() != 3 or proxy_logits.shape[0] != 1:
            raise ValueError(f"{name} proxy_logits must have shape [1, rows, vocab].")
        if mc_counts.shape != proxy_logits.shape:
            raise ValueError(f"{name} mc_counts must match proxy_logits.")
        if labels.shape != proxy_logits.shape[:2]:
            raise ValueError(f"{name} labels must match proxy rows.")
        if base_token_ids.shape != labels.shape:
            raise ValueError(f"{name} risk_gate_token_ids must match labels.")
        current_vocab = int(proxy_logits.shape[-1])
        if vocab_size is None:
            vocab_size = current_vocab
        elif current_vocab != vocab_size:
            raise ValueError("All cache records must share one vocabulary size.")
        events = counts_to_events(mc_counts[0])
        current_max = int(events.shape[-1])
        if max_samples is None:
            max_samples = current_max
        elif current_max != max_samples:
            raise ValueError("All cache records must share one exact MC count.")
        metadata = payload.get("metadata") or {}
        for source_field, target_field in TARGET_PROTOCOL_SOURCE_FIELDS.items():
            if (
                source_field in metadata
                and metadata[source_field] != target_protocol[target_field]
            ):
                raise ValueError(
                    f"{name} target protocol {source_field} disagrees with "
                    "the embedded source manifest."
                )
        relevant_metadata = {
            key: metadata.get(key)
            for key in (
                "mc_fusion_mode",
                "proxy_temperature",
                "proxy_prior_strength",
                "proxy_model_name_or_path",
                "proxy_model_revision",
                "proxy_tokenizer_sha256",
                "proxy_chat_template_protocol",
                "shared_vocab_size",
                "proxy_vocab_size",
                "proxy_vocab_tail_policy",
                "proxy_dtype",
                "proxy_quantization",
                "sample_temperature",
                "top_p",
                "sample_completion_policy",
                "observed_alpha",
                "floor_mass",
            )
        }
        if metadata_reference is None:
            metadata_reference = relevant_metadata
        elif relevant_metadata != metadata_reference:
            raise ValueError("Cache records have inconsistent fusion metadata.")

        rows = int(labels.shape[1])
        proxy_chunks.append(proxy_logits[0].contiguous())
        event_chunks.append(events)
        label_chunks.append(labels[0].long())
        base_chunks.append(base_token_ids[0].long())
        position_chunks.append(torch.arange(rows, dtype=torch.long))
        record_id_chunks.append(torch.full((rows,), record_id, dtype=torch.long))
        row_keys.extend(f"{name}:{row_index}" for row_index in range(rows))

    assert vocab_size is not None and max_samples is not None
    assert metadata_reference is not None
    metadata_reference["manifest_configuration_fingerprint"] = manifest.get(
        "configuration_fingerprint"
    )
    metadata_reference.update(target_protocol)
    return AnytimeProxyMCCache(
        proxy_logits=torch.cat(proxy_chunks, dim=0).contiguous(),
        events=torch.cat(event_chunks, dim=0).contiguous(),
        labels=torch.cat(label_chunks, dim=0).contiguous(),
        base_token_ids=torch.cat(base_chunks, dim=0).contiguous(),
        position_ids=torch.cat(position_chunks, dim=0).contiguous(),
        record_ids=torch.cat(record_id_chunks, dim=0).contiguous(),
        row_keys=row_keys,
        record_names=selected,
        metadata=metadata_reference,
    )
