"""Materialize tokenizer-specific BiasNet tensors from completion-text counters.

The API sampler stores one ``Counter[str, int]`` per sampled prefix.  Those
continuation strings are the source of truth; dense log-probabilities are only
a convenience materialization for a particular proxy tokenizer.

Changing tokenizers can also change the teacher-forced prefix grid.  This tool
therefore supports either exact prefix matching (``strict``) or retaining the
intersection of source and destination text prefixes (``intersection``).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoTokenizer

from pre_logits_sampled_openweight import sampled_ids_to_log_probs


RAW_COUNTER_KEY = "sampled_completion_text_counts"
PREFIX_TEXT_KEY = "sampled_prefix_texts"
RAW_SCHEMA_NAME = "completion_text_counter_v1"


def resolve_store_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported store dtype: {name}")


def encode_completion_counter(
    counter: dict[str, int], tokenizer
) -> tuple[list[int], int, int]:
    """Expand a raw-text counter into first-token IDs.

    Returns ``(sample_ids, multi_token_samples, unmappable_samples)``.  Empty
    strings represent a canonical EOS event, as recorded by the sampler.
    """

    sample_ids: list[int] = []
    multi_token_samples = 0
    unmappable_samples = 0
    for completion_text, raw_count in counter.items():
        count = int(raw_count)
        if count <= 0:
            raise ValueError(
                f"Completion counts must be positive, got {count} for {completion_text!r}."
            )
        if completion_text == "":
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is None:
                raise ValueError(
                    "The raw cache contains canonical EOS samples, but the destination "
                    "tokenizer has no eos_token_id."
                )
            token_ids = [int(eos_token_id)]
        else:
            token_ids = tokenizer.encode(completion_text, add_special_tokens=False)
        if not token_ids:
            unmappable_samples += count
            continue
        if len(token_ids) > 1:
            multi_token_samples += count
        sample_ids.extend([int(token_ids[0])] * count)
    return sample_ids, multi_token_samples, unmappable_samples


def _select_source_rows(value: torch.Tensor, source_indices: list[int]) -> torch.Tensor:
    index = torch.tensor(source_indices, dtype=torch.long)
    return value.index_select(1, index)


def _row_tensor(value: list[int], dtype: torch.dtype = torch.long) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype).unsqueeze(0)


def materialize_payload(
    payload: dict[str, Any],
    tokenizer,
    *,
    tokenizer_name: str,
    alignment_policy: str = "strict",
    max_answer_tokens: Optional[int] = None,
    observed_alpha: Optional[float] = None,
    floor_mass: Optional[float] = None,
    store_dtype: torch.dtype = torch.float16,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if alignment_policy not in {"strict", "intersection"}:
        raise ValueError("alignment_policy must be 'strict' or 'intersection'.")
    if RAW_COUNTER_KEY not in payload or PREFIX_TEXT_KEY not in payload:
        raise ValueError(
            f"Payload must contain {RAW_COUNTER_KEY!r} and {PREFIX_TEXT_KEY!r}."
        )
    answer_text = payload.get("answer_text")
    if not isinstance(answer_text, str):
        raise ValueError("Payload is missing string answer_text.")

    counters = payload[RAW_COUNTER_KEY]
    source_prefixes = payload[PREFIX_TEXT_KEY]
    if not isinstance(counters, list) or not isinstance(source_prefixes, list):
        raise ValueError("Completion counters and prefix texts must be lists.")
    if len(counters) != len(source_prefixes):
        raise ValueError(
            f"Counter/prefix length mismatch: {len(counters)} != {len(source_prefixes)}."
        )
    source_row_count = len(source_prefixes)
    if source_row_count == 0:
        raise ValueError("Cannot materialize an empty raw cache record.")
    if len(set(source_prefixes)) != source_row_count:
        raise ValueError("Source prefix texts are not unique within the cache record.")

    metadata = dict(payload.get("metadata") or {})
    if observed_alpha is None:
        observed_alpha = float(metadata.get("observed_alpha", 0.1))
    if floor_mass is None:
        floor_mass = float(metadata.get("floor_mass", 1e-4))

    destination_ids = [
        int(token_id)
        for token_id in tokenizer.encode(answer_text, add_special_tokens=False)
    ]
    destination_limit = (
        int(max_answer_tokens)
        if max_answer_tokens is not None
        else source_row_count
    )
    if destination_limit <= 0:
        raise ValueError("max_answer_tokens must be positive.")
    destination_ids = destination_ids[:destination_limit]
    source_by_prefix = {
        prefix_text: source_index
        for source_index, prefix_text in enumerate(source_prefixes)
    }

    source_indices: list[int] = []
    destination_positions: list[int] = []
    destination_prefixes: list[str] = []
    destination_labels: list[int] = []
    missing_prefixes: list[tuple[int, str]] = []
    for position, label in enumerate(destination_ids):
        prefix_text = tokenizer.decode(
            destination_ids[:position], skip_special_tokens=False
        )
        source_index = source_by_prefix.get(prefix_text)
        if source_index is None:
            missing_prefixes.append((position, prefix_text))
            continue
        source_indices.append(source_index)
        destination_positions.append(position)
        destination_prefixes.append(prefix_text)
        destination_labels.append(label)

    if missing_prefixes and alignment_policy == "strict":
        preview = ", ".join(
            f"position={position} prefix={prefix!r}"
            for position, prefix in missing_prefixes[:3]
        )
        raise ValueError(
            f"Destination tokenizer introduced {len(missing_prefixes)} unmatched "
            f"prefixes ({preview}). Use --alignment_policy intersection to retain "
            "only exactly shared text prefixes."
        )
    if not source_indices:
        raise ValueError("No destination prefix has an exact source-text match.")

    selected_counters = [dict(counters[index]) for index in source_indices]
    sample_id_rows: list[list[int]] = []
    valid_counts: list[int] = []
    multi_token_counts: list[int] = []
    unmappable_counts: list[int] = []
    eos_counts: list[int] = []
    for counter in selected_counters:
        sample_ids, multi_token_count, unmappable_count = encode_completion_counter(
            counter, tokenizer
        )
        if unmappable_count:
            raise ValueError(
                f"Destination tokenizer could not encode {unmappable_count} accepted samples."
            )
        if not sample_ids:
            raise ValueError("A selected prefix has no materializable samples.")
        sample_id_rows.append(sample_ids)
        valid_counts.append(len(sample_ids))
        multi_token_counts.append(multi_token_count)
        unmappable_counts.append(unmappable_count)
        eos_counts.append(int(counter.get("", 0)))

    sample_lengths = {len(row) for row in sample_id_rows}
    if len(sample_lengths) == 1:
        sampled_tensor = torch.tensor(sample_id_rows, dtype=torch.long)
        materialized_log_probs = sampled_ids_to_log_probs(
            sampled_token_ids=sampled_tensor,
            vocab_size=len(tokenizer),
            observed_alpha=float(observed_alpha),
            floor_mass=float(floor_mass),
            dtype=store_dtype,
        )
    else:
        materialized_log_probs = torch.stack(
            [
                sampled_ids_to_log_probs(
                    sampled_token_ids=torch.tensor([sample_ids], dtype=torch.long),
                    vocab_size=len(tokenizer),
                    observed_alpha=float(observed_alpha),
                    floor_mass=float(floor_mass),
                    dtype=store_dtype,
                )[0]
                for sample_ids in sample_id_rows
            ],
            dim=0,
        )

    output: dict[str, Any] = {}
    ignored_row_keys = {
        "log_probs",
        "labels",
        "risk_gate_token_ids",
        "risk_gate_active_positions",
        "valid_sample_counts",
        "multi_token_sample_counts",
        "unmappable_nonempty_sample_counts",
        "canonical_eos_projection_counts",
    }
    for key, value in payload.items():
        if key in ignored_row_keys:
            continue
        if (
            isinstance(value, torch.Tensor)
            and value.dim() >= 2
            and value.shape[0] == 1
            and value.shape[1] == source_row_count
        ):
            output[key] = _select_source_rows(value, source_indices)
        else:
            output[key] = value

    output.update(
        {
            "log_probs": materialized_log_probs.unsqueeze(0).cpu(),
            "labels": _row_tensor(destination_labels),
            RAW_COUNTER_KEY: selected_counters,
            PREFIX_TEXT_KEY: destination_prefixes,
            "source_row_indices": _row_tensor(source_indices),
            "destination_token_positions": _row_tensor(destination_positions),
            "valid_sample_counts": _row_tensor(valid_counts),
            "multi_token_sample_counts": _row_tensor(multi_token_counts),
            "unmappable_nonempty_sample_counts": _row_tensor(unmappable_counts),
            "canonical_eos_projection_counts": _row_tensor(eos_counts),
        }
    )

    if "risk_gate_active_positions" in payload:
        output["risk_gate_active_positions"] = _row_tensor(destination_positions)
    deterministic_texts = payload.get("deterministic_completion_texts")
    if isinstance(deterministic_texts, list) and len(deterministic_texts) == source_row_count:
        deterministic_ids: list[int] = []
        for source_index in source_indices:
            ids, _, unmappable = encode_completion_counter(
                {str(deterministic_texts[source_index]): 1}, tokenizer
            )
            if unmappable or len(ids) != 1:
                raise ValueError("Unable to rematerialize a deterministic risk-gate token.")
            deterministic_ids.append(ids[0])
        output["risk_gate_token_ids"] = _row_tensor(deterministic_ids)

    source_tokenizer = metadata.get("materialized_log_probs_tokenizer") or metadata.get(
        "tokenizer_name"
    )
    output["metadata"] = {
        **metadata,
        "payload_schema_version": 2,
        "sample_representation": RAW_SCHEMA_NAME,
        "materialized_log_probs_tokenizer": tokenizer_name,
        "materialized_vocab_size": len(tokenizer),
        "materialized_observed_alpha": float(observed_alpha),
        "materialized_floor_mass": float(floor_mass),
        "materialized_store_dtype": str(store_dtype).removeprefix("torch."),
        "materialization_alignment_policy": alignment_policy,
        "materialization_source_tokenizer": source_tokenizer,
        "materialization_source_rows": source_row_count,
        "materialization_destination_rows": len(destination_ids),
        "materialization_retained_rows": len(source_indices),
        "materialization_dropped_destination_rows": len(missing_prefixes),
        "materialization_exact_prefix_rate": len(source_indices)
        / max(len(destination_ids), 1),
    }
    summary = {
        "source_rows": source_row_count,
        "destination_rows": len(destination_ids),
        "retained_rows": len(source_indices),
        "dropped_destination_rows": len(missing_prefixes),
        "exact_prefix_rate": len(source_indices) / max(len(destination_ids), 1),
        "samples": sum(valid_counts),
    }
    return output, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize BiasNet log-prob tensors from raw completion-text counters."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--tokenizer_use_fast", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--alignment_policy", choices=["strict", "intersection"], default="strict"
    )
    parser.add_argument("--max_answer_tokens", type=int, default=None)
    parser.add_argument("--observed_alpha", type=float, default=None)
    parser.add_argument("--floor_mass", type=float, default=None)
    parser.add_argument("--store_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    input_files = sorted(input_dir.glob("*.pt"))
    if not input_files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}.")
    existing = sorted(output_dir.glob("*.pt")) if output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} .pt files; pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        revision=args.tokenizer_revision,
        use_fast=args.tokenizer_use_fast,
        trust_remote_code=args.trust_remote_code,
    )
    summaries = []
    for input_path in input_files:
        payload = torch.load(input_path, map_location="cpu", weights_only=False)
        output, summary = materialize_payload(
            payload,
            tokenizer,
            tokenizer_name=args.tokenizer_name,
            alignment_policy=args.alignment_policy,
            max_answer_tokens=args.max_answer_tokens,
            observed_alpha=args.observed_alpha,
            floor_mass=args.floor_mass,
            store_dtype=resolve_store_dtype(args.store_dtype),
        )
        destination = output_dir / input_path.name
        temporary = output_dir / f".{input_path.name}.tmp"
        torch.save(output, temporary)
        os.replace(temporary, destination)
        summaries.append(summary)

    total_destination = sum(item["destination_rows"] for item in summaries)
    total_retained = sum(item["retained_rows"] for item in summaries)
    manifest = {
        "schema_version": 1,
        "source_cache_dir": str(input_dir.resolve()),
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_revision": args.tokenizer_revision,
        "vocab_size": len(tokenizer),
        "alignment_policy": args.alignment_policy,
        "records": len(summaries),
        "destination_rows": total_destination,
        "retained_rows": total_retained,
        "exact_prefix_rate": total_retained / max(total_destination, 1),
        "store_dtype": args.store_dtype,
        "observed_alpha_override": args.observed_alpha,
        "floor_mass_override": args.floor_mass,
    }
    (output_dir / "materialization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
