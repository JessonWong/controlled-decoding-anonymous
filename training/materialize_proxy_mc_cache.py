"""Fuse a local Qwen3 proxy prior into a legacy exact-MC target cache.

This materializer is intentionally fail-closed.  It reconstructs the original
dataset row and the exact OpenRouter Qwen no-think conversation for every
cached answer prefix, recovers integer MC counts from the legacy floor
distribution, obtains all proxy rows in one teacher-forced forward pass, and
writes the Dirichlet posterior-predictive log probabilities to a new cache.

The supported first experiment is Qwen3-32B (OpenRouter/DeepInfra) with a
Qwen3-1.7B local proxy.  Qwen3's model head is padded to 151936 rows while its
tokenizer has 151669 tokens.  The fusion core crops that padded tail *before*
normalizing the proxy logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mc_reconstruction import (  # noqa: E402
    chat_template_sha256,
    floor_log_probs_to_mc_counts,
    fuse_proxy_logits_with_mc_counts,
)


QWEN_HARD_NO_THINK_PREFILL = "<think>\n\n</think>\n\n"
SHARED_QWEN3_VOCAB_SIZE = 151669
PADDED_QWEN3_OUTPUT_VOCAB_SIZE = 151936
FUSION_MODE = "proxy_dirichlet_v1"
CHAT_TEMPLATE_PROTOCOL = "messages_for_prefix_qwen_v1"
# Same message builder without the Qwen empty-think prefill. Stamped distinctly so a
# non-prefill cache can never satisfy a runtime contract that expects the Qwen protocol.
CHAT_TEMPLATE_PROTOCOL_PLAIN = "messages_for_prefix_plain_v1"
PROXY_VOCAB_TAIL_POLICY = "crop_to_shared_vocab_before_softmax"
TOKENIZER_HASH_SCHEMA = "token_to_id_v1"


@dataclass(frozen=True)
class RecordData:
    """Dataset-derived values needed to materialize one source record."""

    dataset_idx: int
    prompt: str
    answer: str
    api_prompt: str
    answer_token_ids: tuple[int, ...]
    cached_positions: tuple[int, ...]


@dataclass(frozen=True)
class TeacherForcedBatch:
    """Independently rendered cache prefixes, left-padded for one LM call."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor

    @property
    def row_count(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def unpadded_token_count(self) -> int:
        return int(self.attention_mask.sum().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse exact target-model MC counts with a local Qwen3 proxy prior. "
            "The output directory must not already exist."
        )
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--proxy_model_name_or_path", default="Qwen/Qwen3-1.7B"
    )
    parser.add_argument(
        "--proxy_model_revision",
        default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )
    parser.add_argument(
        "--proxy_tokenizer_name_or_path",
        default=None,
        help="Defaults to --proxy_model_name_or_path.",
    )
    parser.add_argument(
        "--proxy_tokenizer_revision",
        default=None,
        help="Defaults to --proxy_model_revision.",
    )
    parser.add_argument(
        "--proxy_tokenizer_sha256",
        default=None,
        help="Optional expected canonical token-to-ID mapping hash.",
    )
    parser.add_argument(
        "--target_tokenizer_name_or_path",
        default=None,
        help="Defaults to tokenizer_name in the source cache manifest.",
    )
    parser.add_argument(
        "--target_tokenizer_revision",
        default="9216db5781bf21249d130ec9da846c4624c16137",
    )
    parser.add_argument(
        "--target_tokenizer_sha256",
        default=None,
        help="Optional expected target tokenizer mapping hash.",
    )
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument(
        "--expect_append_no_think",
        type=lambda v: str(v).lower() not in {"false", "0", "no"},
        default=True,
        help="Source manifest value required for append_no_think. Qwen caches use True; "
             "targets without a /no_think switch (e.g. Claude) need False.",
    )
    parser.add_argument(
        "--expect_qwen_hard_no_think_prefill",
        type=lambda v: str(v).lower() not in {"false", "0", "no"},
        default=True,
        help="Source manifest value required for qwen_hard_no_think_prefill, and the "
             "prefill actually rendered into the proxy prompt.",
    )
    parser.add_argument("--expected_target_model", default="qwen/qwen3-32b")
    parser.add_argument("--expected_provider", default="DeepInfra")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help="Proxy model loading dtype.",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help="Transformers device_map string, or 'none' to omit it.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--prior_strength",
        "--kappa",
        dest="prior_strength",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--proxy_batch_size",
        type=int,
        default=0,
        help=(
            "Number of independently rendered cached prefixes per proxy "
            "forward. Zero batches every prefix in a record into one call."
        ),
    )
    parser.add_argument(
        "--store_dtype", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument(
        "--proxy_logits_store_dtype",
        choices=["float16", "float32"],
        default="float16",
        help=(
            "Storage dtype for uncalibrated proxy logits cropped to the shared "
            "vocabulary. These scores permit offline temperature/kappa sweeps."
        ),
    )
    parser.add_argument(
        "--shared_vocab_size", type=int, default=SHARED_QWEN3_VOCAB_SIZE
    )
    parser.add_argument(
        "--expected_proxy_vocab_size",
        type=int,
        default=PADDED_QWEN3_OUTPUT_VOCAB_SIZE,
    )
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Validate the source, dataset, tokenizer identity, labels, count "
            "recovery, and chat-prefix nesting without loading the proxy model "
            "or creating the output directory."
        ),
    )
    return parser.parse_args()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_mapping_payload(tokenizer) -> dict[str, Any]:
    """Return the canonical token-to-ID object shared with online inference."""

    vocabulary = tokenizer.get_vocab()
    tokens = sorted(
        ([int(token_id), str(token)] for token, token_id in vocabulary.items()),
        key=lambda item: (item[0], item[1]),
    )
    token_ids = [item[0] for item in tokens]
    if len(token_ids) != len(set(token_ids)):
        raise ValueError("Tokenizer get_vocab() maps multiple tokens to one token ID.")
    if len(vocabulary) != len(tokenizer):
        raise ValueError(
            "Tokenizer get_vocab() size differs from len(tokenizer); cannot form "
            "an unambiguous shared-vocabulary fingerprint."
        )
    if set(token_ids) != set(range(len(tokenizer))):
        raise ValueError(
            "Tokenizer token IDs must be the contiguous range [0, vocab_size)."
        )
    return {
        "schema": TOKENIZER_HASH_SCHEMA,
        "vocab_size": len(tokenizer),
        "tokens": tokens,
    }


def tokenizer_mapping_sha256(tokenizer) -> str:
    return hashlib.sha256(canonical_json_bytes(tokenizer_mapping_payload(tokenizer))).hexdigest()


def validate_same_tokenizer_mapping(target_tokenizer, proxy_tokenizer) -> None:
    target_vocab = target_tokenizer.get_vocab()
    proxy_vocab = proxy_tokenizer.get_vocab()
    if target_vocab == proxy_vocab and len(target_tokenizer) == len(proxy_tokenizer):
        return
    target_items = {(str(token), int(token_id)) for token, token_id in target_vocab.items()}
    proxy_items = {(str(token), int(token_id)) for token, token_id in proxy_vocab.items()}
    target_only = sorted(target_items - proxy_items)[:3]
    proxy_only = sorted(proxy_items - target_items)[:3]
    raise ValueError(
        "Target and proxy token-to-ID mappings are not identical: "
        f"target_size={len(target_tokenizer)}, proxy_size={len(proxy_tokenizer)}, "
        f"target_only={target_only}, proxy_only={proxy_only}."
    )


def messages_for_prefix(
    question: str,
    prefix_text: str,
    *,
    qwen_hard_no_think_prefill: bool,
) -> list[dict[str, str]]:
    """Mirror pre_logits_sampled_openrouter.messages_for_prefix exactly."""

    messages = [{"role": "user", "content": question}]
    if qwen_hard_no_think_prefill:
        messages.append(
            {
                "role": "assistant",
                "content": QWEN_HARD_NO_THINK_PREFILL + prefix_text,
            }
        )
    elif prefix_text:
        messages.append({"role": "assistant", "content": prefix_text})
    return messages


def render_prefix_ids(
    tokenizer,
    question: str,
    prefix_text: str,
    *,
    qwen_hard_no_think_prefill: bool,
) -> torch.Tensor:
    """Render the exact locally conditioned token sequence for one API prefix."""

    messages = messages_for_prefix(
        question,
        prefix_text,
        qwen_hard_no_think_prefill=qwen_hard_no_think_prefill,
    )
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "return_tensors": "pt",
    }
    if messages[-1]["role"] == "assistant":
        kwargs["continue_final_message"] = True
    else:
        kwargs["add_generation_prompt"] = True
    # Deliberately do not pass enable_thinking.  The hard empty-think block is
    # literal assistant content, matching the OpenRouter cache producer.
    rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    elif hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    rendered = torch.as_tensor(rendered, dtype=torch.long)
    if rendered.dim() == 2:
        if rendered.shape[0] != 1:
            raise ValueError("Chat template unexpectedly returned a multi-sample batch.")
        rendered = rendered[0]
    if rendered.dim() != 1 or rendered.numel() == 0:
        raise ValueError("Chat template must return one non-empty token sequence.")
    return rendered.cpu()


def build_teacher_forced_input(
    proxy_tokenizer,
    target_tokenizer,
    *,
    question: str,
    answer_token_ids: Iterable[int],
    cached_positions: Iterable[int],
    qwen_hard_no_think_prefill: bool,
) -> TeacherForcedBatch:
    """Render every cached API prefix independently and left-pad the rows.

    Byte-level BPE tokenization is not prefix-stable.  For example, two spaces
    at the end of a message can form one token while the same two spaces before
    a following character form two tokens.  Consequently, logits selected
    from one tokenization of the longest answer are not always the logits for
    the shorter API prefixes.  Rendering each prefix independently exactly
    matches online ``messages_for_prefix_qwen_v1`` semantics; left padding
    lets all rows share one model call without changing their position IDs.
    """

    answer_ids = [int(token_id) for token_id in answer_token_ids]
    positions = [int(position) for position in cached_positions]
    if not positions:
        raise ValueError("A cache record must contain at least one label position.")
    if positions != sorted(positions) or len(positions) != len(set(positions)):
        raise ValueError("Cached positions must be unique and increasing.")
    if positions[0] < 0 or positions[-1] >= len(answer_ids):
        raise ValueError("Cached positions fall outside the tokenized answer.")

    rendered_rows: list[torch.Tensor] = []
    for position in positions:
        prefix_text = target_tokenizer.decode(
            answer_ids[:position], skip_special_tokens=False
        )
        prefix_ids = render_prefix_ids(
            proxy_tokenizer,
            question,
            prefix_text,
            qwen_hard_no_think_prefill=qwen_hard_no_think_prefill,
        )
        rendered_rows.append(prefix_ids)

    pad_token_id = getattr(proxy_tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(proxy_tokenizer, "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("Proxy tokenizer must define a pad or EOS token ID.")
    maximum_length = max(int(row.numel()) for row in rendered_rows)
    input_ids = torch.full(
        (len(rendered_rows), maximum_length),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, row in enumerate(rendered_rows):
        row_length = int(row.numel())
        input_ids[row_index, -row_length:] = row
        attention_mask[row_index, -row_length:] = 1
    position_ids = attention_mask.cumsum(dim=-1).sub(1).clamp_min(0)
    if not attention_mask[:, -1].all().item():
        raise AssertionError("Every independently rendered prefix must be non-empty.")
    return TeacherForcedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )


def model_input_device(model) -> torch.device:
    embedding = model.get_input_embeddings()
    if embedding is not None and hasattr(embedding, "weight"):
        device = embedding.weight.device
        if device.type != "meta":
            return device
    try:
        device = next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("Proxy model has no parameters.") from exc
    if device.type == "meta":
        raise ValueError("Cannot place proxy input on a meta device.")
    return device


def forward_proxy_rows(
    model,
    batch: TeacherForcedBatch,
    *,
    batch_size: int = 0,
) -> tuple[torch.Tensor, int]:
    """Forward independently rendered prefixes, normally in one batched call."""

    if batch.input_ids.dim() != 2 or batch.input_ids.shape != batch.attention_mask.shape:
        raise ValueError("Teacher-forced input_ids/attention_mask shapes are invalid.")
    if batch.position_ids.shape != batch.input_ids.shape:
        raise ValueError("Teacher-forced position_ids shape is invalid.")
    if batch.row_count <= 0:
        raise ValueError("Teacher-forced batch cannot be empty.")
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative.")
    effective_batch_size = batch_size or batch.row_count
    device = model_input_device(model)
    row_logits: list[torch.Tensor] = []
    forward_count = 0
    with torch.inference_mode():
        for start in range(0, batch.row_count, effective_batch_size):
            stop = min(start + effective_batch_size, batch.row_count)
            outputs = model(
                input_ids=batch.input_ids[start:stop].to(device),
                attention_mask=batch.attention_mask[start:stop].to(device),
                position_ids=batch.position_ids[start:stop].to(device),
                use_cache=False,
                logits_to_keep=1,
            )
            logits = outputs.logits
            if (
                logits.dim() != 3
                or logits.shape[0] != stop - start
                or logits.shape[1] != 1
            ):
                raise ValueError(
                    "Proxy model must honor logits_to_keep=1 and return logits "
                    "shaped [cached_rows, 1, vocab]."
                )
            row_logits.append(logits[:, 0, :])
            forward_count += 1
    return torch.cat(row_logits, dim=0), forward_count


def resolve_model_dtype(name: str):
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_store_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def effective_model_dtype(model) -> str:
    floating_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point() and parameter.device.type != "meta"
    }
    if len(floating_dtypes) != 1:
        raise ValueError(
            "Proxy model must have exactly one effective floating parameter dtype; "
            f"got {sorted(map(str, floating_dtypes))}."
        )
    return dtype_name(next(iter(floating_dtypes)))


def compact_count_dtype(max_count: int) -> torch.dtype:
    if max_count <= torch.iinfo(torch.uint8).max:
        return torch.uint8
    if max_count <= torch.iinfo(torch.int16).max:
        return torch.int16
    if max_count <= torch.iinfo(torch.int32).max:
        return torch.int32
    return torch.int64


def source_positions(payload: dict[str, Any], row_count: int) -> tuple[int, ...]:
    active = payload.get("risk_gate_active_positions")
    if isinstance(active, torch.Tensor):
        values = tuple(int(value) for value in active.flatten().long().tolist())
        if len(values) != row_count:
            raise ValueError("risk_gate_active_positions does not match cache rows.")
        return values
    return tuple(range(row_count))


def validate_source_manifest(
    manifest: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Source cache_manifest.json has no configuration object.")
    expected_fingerprint = manifest.get("configuration_fingerprint")
    actual_fingerprint = hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()
    if expected_fingerprint != actual_fingerprint:
        raise ValueError("Source cache manifest configuration fingerprint is invalid.")

    required = {
        "source": "sampled_openrouter",
        "model": args.expected_target_model,
        "samples_per_token": 50,
        "sample_completion_policy": "exact",
        "append_no_think": bool(getattr(args, "expect_append_no_think", True)),
        "qwen_hard_no_think_prefill": bool(
            getattr(args, "expect_qwen_hard_no_think_prefill", True)
        ),
        "provider_allow_fallbacks": False,
    }
    for key, expected in required.items():
        if configuration.get(key) != expected:
            raise ValueError(
                f"Source cache configuration {key!r} must be {expected!r}, "
                f"got {configuration.get(key)!r}."
            )
    if configuration.get("provider_order") != [args.expected_provider]:
        raise ValueError(
            "Source cache must be provider-locked to "
            f"[{args.expected_provider!r}], got {configuration.get('provider_order')!r}."
        )
    observed_alpha = float(configuration.get("observed_alpha", -1.0))
    floor_mass = float(configuration.get("floor_mass", -1.0))
    if not math.isfinite(observed_alpha) or observed_alpha < 0:
        raise ValueError("Source observed_alpha must be finite and non-negative.")
    if not math.isfinite(floor_mass) or not 0 <= floor_mass < 1:
        raise ValueError("Source floor_mass must be in [0, 1).")
    return configuration


def validate_record(
    path: Path,
    payload: dict[str, Any],
    dataset,
    target_tokenizer,
    source_configuration: dict[str, Any],
    *,
    shared_vocab_size: int,
) -> RecordData:
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a dictionary payload.")
    log_probs = payload.get("log_probs")
    labels = payload.get("labels")
    if not isinstance(log_probs, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError(f"{path} is missing log_probs or labels tensors.")
    if log_probs.dim() != 3 or labels.dim() != 2 or log_probs.shape[:2] != labels.shape:
        raise ValueError(f"Invalid legacy tensor shapes in {path}.")
    if log_probs.shape[0] != 1 or log_probs.shape[-1] != shared_vocab_size:
        raise ValueError(
            f"{path} must have shape [1, rows, {shared_vocab_size}], got "
            f"{tuple(log_probs.shape)}."
        )
    row_count = int(labels.shape[1])
    if row_count <= 0:
        raise ValueError(f"{path} contains no cached rows.")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} has no metadata dictionary.")
    dataset_idx = metadata.get("dataset_idx")
    if dataset_idx is None:
        raise ValueError(f"{path} has no metadata.dataset_idx.")
    dataset_idx = int(dataset_idx)
    if dataset_idx < 0 or dataset_idx >= len(dataset):
        raise ValueError(f"{path} dataset_idx={dataset_idx} is outside the dataset.")
    item = dataset[dataset_idx]
    prompt = str(item["prompt"])
    answer = str(item["rejected"])
    expected_data_sha256 = hashlib.sha256((prompt + answer).encode("utf-8")).hexdigest()
    if metadata.get("data_sha256") != expected_data_sha256:
        raise ValueError(f"Dataset content SHA256 mismatch for {path}.")
    expected_filename = hashlib.md5((prompt + answer).encode("utf-8")).hexdigest() + ".pt"
    if path.name != expected_filename:
        raise ValueError(
            f"Source filename mismatch for dataset_idx={dataset_idx}: "
            f"expected {expected_filename}, got {path.name}."
        )
    manifest_fingerprint = source_configuration.get("_manifest_fingerprint")
    if manifest_fingerprint is not None and metadata.get(
        "cache_configuration_fingerprint"
    ) != manifest_fingerprint:
        raise ValueError(f"Source manifest fingerprint mismatch in {path}.")

    for key in (
        "source",
        "model",
        "samples_per_token",
        "sample_completion_policy",
        "sample_temperature",
        "top_p",
        "observed_alpha",
        "floor_mass",
        "qwen_hard_no_think_prefill",
        "provider_order",
        "provider_allow_fallbacks",
    ):
        if key in source_configuration and metadata.get(key) != source_configuration[key]:
            raise ValueError(f"{path} metadata.{key} disagrees with the source manifest.")

    max_answer_tokens = source_configuration.get("max_answer_tokens")
    answer_ids = [
        int(token_id)
        for token_id in target_tokenizer.encode(answer, add_special_tokens=False)
    ]
    if max_answer_tokens is not None:
        answer_ids = answer_ids[: int(max_answer_tokens)]
    positions = source_positions(payload, row_count)
    if not positions or min(positions) < 0 or max(positions) >= len(answer_ids):
        raise ValueError(f"Cached label positions are invalid in {path}.")
    expected_labels = torch.tensor(
        [answer_ids[position] for position in positions], dtype=torch.long
    )
    if not torch.equal(labels[0].detach().cpu().long(), expected_labels):
        raise ValueError(f"Legacy labels do not match the reconstructed dataset row in {path}.")

    valid_counts = payload.get("valid_sample_counts")
    if not isinstance(valid_counts, torch.Tensor) or valid_counts.shape != labels.shape:
        raise ValueError(f"{path} has invalid valid_sample_counts.")
    expected_samples = int(source_configuration["samples_per_token"])
    if not torch.equal(
        valid_counts.detach().cpu().long(),
        torch.full_like(valid_counts.detach().cpu().long(), expected_samples),
    ):
        raise ValueError(f"{path} is not an exact-MC{expected_samples} record.")

    api_prompt = prompt + "\n/no_think" if source_configuration["append_no_think"] else prompt
    return RecordData(
        dataset_idx=dataset_idx,
        prompt=prompt,
        answer=answer,
        api_prompt=api_prompt,
        answer_token_ids=tuple(answer_ids),
        cached_positions=positions,
    )


def recover_record_counts(
    payload: dict[str, Any], source_configuration: dict[str, Any]
) -> torch.Tensor:
    counts = floor_log_probs_to_mc_counts(
        log_probs=payload["log_probs"],
        sample_counts=payload["valid_sample_counts"],
        observed_alpha=float(source_configuration["observed_alpha"]),
        floor_mass=float(source_configuration["floor_mass"]),
        dtype=torch.long,
        validate_recovered_counts=True,
    )
    recovered_totals = counts.sum(dim=-1)
    expected_totals = payload["valid_sample_counts"].detach().cpu().long()
    if not torch.equal(recovered_totals.cpu(), expected_totals):
        raise ValueError("Recovered MC counts do not equal valid_sample_counts.")
    return counts


def load_tokenizer(name_or_path: str, revision: Optional[str], args: argparse.Namespace):
    return AutoTokenizer.from_pretrained(
        name_or_path,
        revision=revision,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )


def validate_positive_fusion_parameters(args: argparse.Namespace) -> None:
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("--temperature must be finite and positive.")
    if not math.isfinite(args.prior_strength) or args.prior_strength <= 0:
        raise ValueError("--prior_strength/--kappa must be finite and positive.")
    if args.shared_vocab_size <= 0 or args.expected_proxy_vocab_size < args.shared_vocab_size:
        raise ValueError("Proxy vocabulary must be at least as large as shared vocabulary.")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max_records must be positive.")
    if args.proxy_batch_size < 0:
        raise ValueError("--proxy_batch_size must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_positive_fusion_parameters(args)
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    input_files = sorted(input_dir.glob("*.pt"))
    if not input_files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}.")
    if args.max_records is not None:
        input_files = input_files[: args.max_records]
    if not args.dry_run and output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output path: {output_dir}"
        )

    source_manifest_path = input_dir / "cache_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_configuration = validate_source_manifest(source_manifest, args)
    chat_template_protocol = (
        CHAT_TEMPLATE_PROTOCOL
        if source_configuration["qwen_hard_no_think_prefill"]
        else CHAT_TEMPLATE_PROTOCOL_PLAIN
    )
    source_configuration = dict(source_configuration)
    source_configuration["_manifest_fingerprint"] = source_manifest[
        "configuration_fingerprint"
    ]

    dataset_name = args.dataset_name or source_configuration.get("dataset_name")
    dataset_split = args.dataset_split or source_configuration.get("dataset_split")
    if not dataset_name or not dataset_split:
        raise ValueError("Dataset name/split are absent from both CLI and source manifest.")
    if args.dataset_name and args.dataset_name != source_configuration.get("dataset_name"):
        raise ValueError("--dataset_name must match the source cache manifest.")
    if args.dataset_split and args.dataset_split != source_configuration.get("dataset_split"):
        raise ValueError("--dataset_split must match the source cache manifest.")
    dataset = load_dataset(
        dataset_name,
        split=dataset_split,
        revision=args.dataset_revision,
    )

    proxy_tokenizer_name = (
        args.proxy_tokenizer_name_or_path or args.proxy_model_name_or_path
    )
    proxy_tokenizer_revision = (
        args.proxy_tokenizer_revision
        if args.proxy_tokenizer_revision is not None
        else args.proxy_model_revision
    )
    target_tokenizer_name = (
        args.target_tokenizer_name_or_path
        or str(source_configuration["tokenizer_name"])
    )
    target_tokenizer = load_tokenizer(
        target_tokenizer_name, args.target_tokenizer_revision, args
    )
    proxy_tokenizer = load_tokenizer(
        proxy_tokenizer_name, proxy_tokenizer_revision, args
    )
    if len(target_tokenizer) != args.shared_vocab_size:
        raise ValueError(
            f"Target tokenizer length must be {args.shared_vocab_size}, got "
            f"{len(target_tokenizer)}."
        )
    if len(proxy_tokenizer) != args.shared_vocab_size:
        raise ValueError(
            f"Proxy tokenizer length must be {args.shared_vocab_size}, got "
            f"{len(proxy_tokenizer)}."
        )
    validate_same_tokenizer_mapping(target_tokenizer, proxy_tokenizer)
    target_tokenizer_hash = tokenizer_mapping_sha256(target_tokenizer)
    proxy_tokenizer_hash = tokenizer_mapping_sha256(proxy_tokenizer)
    if args.target_tokenizer_sha256 and args.target_tokenizer_sha256 != target_tokenizer_hash:
        raise ValueError("Target tokenizer SHA256 does not match --target_tokenizer_sha256.")
    if args.proxy_tokenizer_sha256 and args.proxy_tokenizer_sha256 != proxy_tokenizer_hash:
        raise ValueError("Proxy tokenizer SHA256 does not match --proxy_tokenizer_sha256.")

    # First pass validates all immutable source evidence before model loading or
    # output creation.  It also makes --dry_run useful on CPU-only nodes.
    seen_dataset_indices: set[int] = set()
    total_rows = 0
    for path in input_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        record = validate_record(
            path,
            payload,
            dataset,
            target_tokenizer,
            source_configuration,
            shared_vocab_size=args.shared_vocab_size,
        )
        if record.dataset_idx in seen_dataset_indices:
            raise ValueError(f"Duplicate dataset_idx={record.dataset_idx} in source cache.")
        seen_dataset_indices.add(record.dataset_idx)
        # Count recovery is validated immediately before each record is fused
        # and written below.  Do not materialize the same dense count tensor in
        # this source-integrity pass; the immutable manifest, dataset, labels,
        # tokenizer rendering, and exact-MC row metadata are fully checked here.
        total_rows += len(record.cached_positions)
        try:
            build_teacher_forced_input(
                proxy_tokenizer,
                target_tokenizer,
                question=record.api_prompt,
                answer_token_ids=record.answer_token_ids,
                cached_positions=record.cached_positions,
                qwen_hard_no_think_prefill=bool(
                    source_configuration["qwen_hard_no_think_prefill"]
                ),
            )
        except Exception as exc:
            raise ValueError(
                f"Failed to render {path} (dataset_idx={record.dataset_idx})."
            ) from exc
        del payload

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "records": len(input_files),
                    "rows": total_rows,
                    "shared_vocab_size": args.shared_vocab_size,
                    "proxy_tokenizer_sha256": proxy_tokenizer_hash,
                    "target_tokenizer_sha256": target_tokenizer_hash,
                    "chat_template_protocol": chat_template_protocol,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return

    model_kwargs: dict[str, Any] = {
        "revision": args.proxy_model_revision,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "torch_dtype": resolve_model_dtype(args.dtype),
    }
    if args.device_map.casefold() != "none":
        model_kwargs["device_map"] = args.device_map
    proxy_model = AutoModelForCausalLM.from_pretrained(
        args.proxy_model_name_or_path, **model_kwargs
    )
    proxy_model.eval()
    config_proxy_vocab = int(proxy_model.config.vocab_size)
    if config_proxy_vocab != args.expected_proxy_vocab_size:
        raise ValueError(
            f"Proxy config vocab_size must be {args.expected_proxy_vocab_size}, got "
            f"{config_proxy_vocab}."
        )
    proxy_dtype = effective_model_dtype(proxy_model)
    proxy_quantization = "none"

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    store_dtype = resolve_store_dtype(args.store_dtype)
    proxy_logits_store_dtype = resolve_store_dtype(args.proxy_logits_store_dtype)
    materializer_sha = sha256_file(Path(__file__).resolve())
    fusion_core_path = PROJECT_ROOT / "mc_reconstruction.py"
    fusion_core_sha = sha256_file(fusion_core_path)
    record_summaries: list[dict[str, Any]] = []
    count_dtypes: set[str] = set()
    total_input_tokens = 0

    for record_number, path in enumerate(input_files, start=1):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        record = validate_record(
            path,
            payload,
            dataset,
            target_tokenizer,
            source_configuration,
            shared_vocab_size=args.shared_vocab_size,
        )
        mc_counts = recover_record_counts(payload, source_configuration)
        teacher_batch = build_teacher_forced_input(
            proxy_tokenizer,
            target_tokenizer,
            question=record.api_prompt,
            answer_token_ids=record.answer_token_ids,
            cached_positions=record.cached_positions,
            qwen_hard_no_think_prefill=bool(
                source_configuration["qwen_hard_no_think_prefill"]
            ),
        )
        proxy_logits, proxy_forward_count = forward_proxy_rows(
            proxy_model,
            teacher_batch,
            batch_size=args.proxy_batch_size,
        )
        if proxy_logits.shape != (
            len(record.cached_positions),
            args.expected_proxy_vocab_size,
        ):
            raise ValueError(
                f"Unexpected proxy logits shape for {path}: {tuple(proxy_logits.shape)}."
            )
        fused_rows = fuse_proxy_logits_with_mc_counts(
            proxy_logits,
            mc_counts[0].to(proxy_logits.device),
            temperature=args.temperature,
            prior_strength=args.prior_strength,
            dtype=store_dtype,
        ).cpu()
        if fused_rows.shape != payload["log_probs"][0].shape:
            raise ValueError("Fused rows do not match the source cache shape.")
        # Persist uncalibrated scores so temperature and kappa can be selected
        # offline without another proxy forward.  The padded model-head tail is
        # intentionally absent: it is outside the shared tokenizer vocabulary
        # and must never participate in normalization.
        stored_proxy_logits = proxy_logits[
            ..., : args.shared_vocab_size
        ].detach().to(device="cpu", dtype=proxy_logits_store_dtype)

        max_count = int(mc_counts.max().item())
        count_dtype = compact_count_dtype(max_count)
        compact_counts = mc_counts.to(count_dtype).cpu()
        count_dtype_name = str(count_dtype).removeprefix("torch.")
        count_dtypes.add(count_dtype_name)
        source_metadata = dict(payload["metadata"])
        output_metadata = {
            **source_metadata,
            "payload_schema_version": 3,
            "mc_fusion_mode": FUSION_MODE,
            "log_probs_semantics": "proxy_dirichlet_posterior_predictive_log_probs",
            "mc_counts_key": "mc_counts",
            "mc_counts_dtype": count_dtype_name,
            "mc_count_reconstruction": "floor_log_probs_to_mc_counts_v1",
            "proxy_model_name_or_path": args.proxy_model_name_or_path,
            "proxy_model_revision": args.proxy_model_revision,
            "proxy_model_commit": getattr(proxy_model.config, "_commit_hash", None),
            "proxy_tokenizer_name_or_path": proxy_tokenizer_name,
            "proxy_tokenizer_revision": proxy_tokenizer_revision,
            "proxy_tokenizer_sha256": proxy_tokenizer_hash,
            "target_tokenizer_name_or_path": target_tokenizer_name,
            "target_tokenizer_revision": args.target_tokenizer_revision,
            "target_tokenizer_sha256": target_tokenizer_hash,
            "proxy_temperature": float(args.temperature),
            "proxy_prior_strength": float(args.prior_strength),
            "proxy_chat_template_protocol": chat_template_protocol,
            "proxy_chat_template_sha256": chat_template_sha256(proxy_tokenizer),
            "shared_vocab_size": int(args.shared_vocab_size),
            "proxy_vocab_size": int(proxy_logits.shape[-1]),
            "proxy_vocab_tail_policy": PROXY_VOCAB_TAIL_POLICY,
            "proxy_logits_key": "proxy_logits",
            "proxy_logits_semantics": "uncalibrated_raw_logits_shared_vocab",
            "proxy_logits_vocab_size": int(stored_proxy_logits.shape[-1]),
            "proxy_logits_dtype": args.proxy_logits_store_dtype,
            "proxy_model_dtype": args.dtype,
            "proxy_dtype": proxy_dtype,
            "proxy_quantization": proxy_quantization,
            "proxy_device_map": args.device_map,
            "proxy_batch_size": int(args.proxy_batch_size),
            "fused_log_probs_dtype": args.store_dtype,
            "proxy_teacher_forced_forward_count": int(proxy_forward_count),
            "proxy_teacher_forced_input_tokens": int(
                teacher_batch.unpadded_token_count
            ),
            "proxy_teacher_forced_padded_tokens": int(
                teacher_batch.input_ids.numel()
            ),
            "source_cache_file": str(path),
            "source_cache_sha256": sha256_file(path),
            "source_log_probs_dtype": str(payload["log_probs"].dtype).removeprefix("torch."),
            "materializer_source_sha256": materializer_sha,
            "fusion_core_source_sha256": fusion_core_sha,
        }
        output_payload = dict(payload)
        output_payload["log_probs"] = fused_rows.unsqueeze(0)
        output_payload["mc_counts"] = compact_counts
        output_payload["proxy_logits"] = stored_proxy_logits.unsqueeze(0)
        output_payload["metadata"] = output_metadata

        destination = output_dir / path.name
        temporary = output_dir / f".{path.name}.tmp"
        torch.save(output_payload, temporary)
        os.replace(temporary, destination)
        output_sha = sha256_file(destination)
        record_summaries.append(
            {
                "file": path.name,
                "dataset_idx": record.dataset_idx,
                "rows": len(record.cached_positions),
                "source_sha256": output_metadata["source_cache_sha256"],
                "output_sha256": output_sha,
                "data_sha256": source_metadata["data_sha256"],
                "teacher_forced_input_tokens": int(
                    teacher_batch.unpadded_token_count
                ),
                "teacher_forced_padded_tokens": int(
                    teacher_batch.input_ids.numel()
                ),
                "teacher_forced_forward_count": int(proxy_forward_count),
                "max_mc_count": max_count,
                "mc_counts_dtype": count_dtype_name,
            }
        )
        total_input_tokens += int(teacher_batch.unpadded_token_count)
        print(
            f"materialized={record_number}/{len(input_files)} "
            f"dataset_idx={record.dataset_idx} rows={len(record.cached_positions)}",
            flush=True,
        )
        del (
            payload,
            output_payload,
            mc_counts,
            compact_counts,
            proxy_logits,
            stored_proxy_logits,
            fused_rows,
            teacher_batch,
        )

    manifest_configuration = {
        "mc_fusion_mode": FUSION_MODE,
        "source_cache_dir": str(input_dir),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_configuration_fingerprint": source_manifest[
            "configuration_fingerprint"
        ],
        "target_model": args.expected_target_model,
        "target_provider": args.expected_provider,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "dataset_revision": args.dataset_revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "proxy_model_name_or_path": args.proxy_model_name_or_path,
        "proxy_model_revision": args.proxy_model_revision,
        "proxy_model_commit": getattr(proxy_model.config, "_commit_hash", None),
        "proxy_tokenizer_name_or_path": proxy_tokenizer_name,
        "proxy_tokenizer_revision": proxy_tokenizer_revision,
        "proxy_tokenizer_sha256": proxy_tokenizer_hash,
        "target_tokenizer_name_or_path": target_tokenizer_name,
        "target_tokenizer_revision": args.target_tokenizer_revision,
        "target_tokenizer_sha256": target_tokenizer_hash,
        "proxy_temperature": float(args.temperature),
        "proxy_prior_strength": float(args.prior_strength),
        "proxy_chat_template_protocol": chat_template_protocol,
        "proxy_chat_template_sha256": chat_template_sha256(proxy_tokenizer),
        "shared_vocab_size": int(args.shared_vocab_size),
        "proxy_vocab_size": int(config_proxy_vocab),
        "proxy_vocab_tail_policy": PROXY_VOCAB_TAIL_POLICY,
        "proxy_logits_key": "proxy_logits",
        "proxy_logits_semantics": "uncalibrated_raw_logits_shared_vocab",
        "proxy_logits_vocab_size": int(args.shared_vocab_size),
        "proxy_logits_dtype": args.proxy_logits_store_dtype,
        "proxy_model_dtype": args.dtype,
        "proxy_dtype": proxy_dtype,
        "proxy_quantization": proxy_quantization,
        "proxy_device_map": args.device_map,
        "proxy_batch_size": int(args.proxy_batch_size),
        "fused_log_probs_dtype": args.store_dtype,
        "mc_counts_dtypes": sorted(count_dtypes),
        "materializer_source_sha256": materializer_sha,
        "fusion_core_source_sha256": fusion_core_sha,
    }
    manifest = {
        "manifest_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": manifest_configuration,
        "configuration_fingerprint": hashlib.sha256(
            canonical_json_bytes(manifest_configuration)
        ).hexdigest(),
        "source_manifest": source_manifest,
        "records": record_summaries,
        "summary": {
            "record_count": len(record_summaries),
            "row_count": sum(item["rows"] for item in record_summaries),
            "teacher_forced_forward_count": sum(
                item["teacher_forced_forward_count"] for item in record_summaries
            ),
            "teacher_forced_input_tokens": total_input_tokens,
        },
    }
    manifest_path = output_dir / "proxy_mc_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                **manifest["summary"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
