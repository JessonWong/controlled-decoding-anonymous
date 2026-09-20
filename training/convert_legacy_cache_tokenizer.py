"""Approximately convert legacy dense sampled caches to a new tokenizer.

Legacy caches discarded the raw API completion strings.  This converter
recovers MC counts from the smoothed log-probabilities, decodes each observed
source token, and treats that decoded token text as the sampled continuation.
Only exact shared *text* prefixes are retained.  The result is explicitly
marked approximate and must not be confused with a freshly sampled raw-text
cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from materialize_text_counter_cache import (
    PREFIX_TEXT_KEY,
    RAW_COUNTER_KEY,
    materialize_payload,
    resolve_store_dtype,
)


def recover_mc_token_counts(
    log_prob_row: torch.Tensor,
    *,
    sample_count: int,
    observed_alpha: float,
    floor_mass: float,
) -> dict[int, int]:
    """Invert sampled_ids_to_log_probs for the observed vocabulary entries."""

    row = log_prob_row.detach().cpu().float()
    if row.dim() != 1:
        raise ValueError("log_prob_row must be one-dimensional.")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if floor_mass == 0:
        observed_mask = torch.isfinite(row)
    else:
        floor_value = row.min()
        # Observed MC50 probabilities are orders of magnitude above the
        # artificial per-vocabulary floor.  A one-nat margin is conservative
        # and robust to float16 storage.
        observed_mask = row > floor_value + 1.0
    observed_ids = torch.nonzero(observed_mask, as_tuple=False).flatten()
    observed_count = int(observed_ids.numel())
    if observed_count == 0:
        raise ValueError("No observed vocabulary entries found in legacy row.")

    denominator = float(sample_count) + observed_alpha * float(observed_count)
    estimates = (
        row.index_select(0, observed_ids).exp()
        * denominator
        / (1.0 - floor_mass)
        - observed_alpha
    )
    rounded = estimates.round().long().clamp_min(1)
    difference = int(sample_count - rounded.sum().item())
    residuals = estimates - rounded.float()
    while difference > 0:
        index = int(torch.argmax(residuals).item())
        rounded[index] += 1
        residuals[index] -= 1.0
        difference -= 1
    while difference < 0:
        candidates = torch.nonzero(rounded > 1, as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError("Unable to reconcile recovered MC counts with sample_count.")
        candidate_residuals = residuals.index_select(0, candidates)
        index = int(candidates[torch.argmin(candidate_residuals)].item())
        rounded[index] -= 1
        residuals[index] += 1.0
        difference += 1
    return {
        int(token_id): int(count)
        for token_id, count in zip(observed_ids.tolist(), rounded.tolist())
    }


def decoded_completion_counter(
    token_counts: dict[int, int],
    source_tokenizer,
    *,
    canonical_eos_count: int,
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    eos_token_id = getattr(source_tokenizer, "eos_token_id", None)
    remaining_eos = int(canonical_eos_count)
    for token_id, count in token_counts.items():
        remaining = int(count)
        if eos_token_id is not None and token_id == int(eos_token_id) and remaining_eos:
            projected = min(remaining, remaining_eos)
            counter[""] += projected
            remaining -= projected
            remaining_eos -= projected
        if remaining:
            text = source_tokenizer.decode(
                [int(token_id)], skip_special_tokens=False
            )
            counter[str(text)] += remaining
    if remaining_eos:
        raise ValueError(
            f"Cache reports {canonical_eos_count} canonical EOS samples, but only "
            f"{canonical_eos_count - remaining_eos} were recoverable from EOS mass."
        )
    return dict(sorted(counter.items()))


def source_positions(payload: dict[str, Any], row_count: int) -> list[int]:
    active = payload.get("risk_gate_active_positions")
    if isinstance(active, torch.Tensor):
        values = active.flatten().long().tolist()
        if len(values) != row_count:
            raise ValueError("risk_gate_active_positions does not match cache rows.")
        return [int(value) for value in values]
    return list(range(row_count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approximately convert a legacy dense sampled cache to a new tokenizer."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_tokenizer_name", required=True)
    parser.add_argument("--source_tokenizer_revision", default=None)
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--dataset_name", default="LLM-LAT/harmful-dataset")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--alignment_policy", choices=["strict", "intersection"], default="intersection")
    parser.add_argument("--max_answer_tokens", type=int, default=None)
    parser.add_argument("--store_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--trust_remote_code", action="store_true")
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

    source_tokenizer = AutoTokenizer.from_pretrained(
        args.source_tokenizer_name,
        revision=args.source_tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )
    destination_tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        revision=args.tokenizer_revision,
        trust_remote_code=args.trust_remote_code,
    )
    dataset = load_dataset(args.dataset_name)[args.dataset_split]
    source_manifest_path = input_dir / "cache_manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )
    source_configuration = source_manifest.get("configuration") or {}
    observed_alpha = float(source_configuration.get("observed_alpha", 0.1))
    floor_mass = float(source_configuration.get("floor_mass", 1e-4))
    configured_max_tokens = source_configuration.get("max_answer_tokens")
    max_answer_tokens = (
        args.max_answer_tokens
        if args.max_answer_tokens is not None
        else configured_max_tokens
    )
    append_no_think = bool(source_configuration.get("append_no_think", False))

    summaries = []
    total_legacy_multi_token_samples = 0
    total_samples = 0
    for input_path in input_files:
        payload = torch.load(input_path, map_location="cpu", weights_only=False)
        log_probs = payload["log_probs"]
        labels = payload["labels"]
        if log_probs.dim() != 3 or labels.dim() != 2 or log_probs.shape[:2] != labels.shape:
            raise ValueError(f"Invalid legacy tensor shapes in {input_path}.")
        row_count = int(labels.shape[1])
        metadata = dict(payload.get("metadata") or {})
        dataset_index = metadata.get("dataset_idx")
        if dataset_index is None:
            raise ValueError(f"{input_path} has no metadata.dataset_idx.")
        item = dataset[int(dataset_index)]
        prompt = str(item["prompt"])
        answer = str(item["rejected"])
        api_prompt = prompt + "\n/no_think" if append_no_think else prompt
        expected_sha256 = metadata.get("data_sha256")
        actual_sha256 = hashlib.sha256((prompt + answer).encode("utf-8")).hexdigest()
        if expected_sha256 is not None and expected_sha256 != actual_sha256:
            raise ValueError(f"Dataset content hash mismatch for {input_path}.")

        answer_ids = [
            int(token_id)
            for token_id in source_tokenizer.encode(answer, add_special_tokens=False)
        ]
        positions = source_positions(payload, row_count)
        expected_labels = torch.tensor(
            [answer_ids[position] for position in positions], dtype=torch.long
        )
        if not torch.equal(labels[0].long(), expected_labels):
            raise ValueError(f"Legacy labels do not match source tokenizer in {input_path}.")
        prefixes = [
            source_tokenizer.decode(
                answer_ids[:position], skip_special_tokens=False
            )
            for position in positions
        ]

        valid_tensor = payload.get("valid_sample_counts")
        canonical_tensor = payload.get("canonical_eos_projection_counts")
        counters: list[dict[str, int]] = []
        for row_index in range(row_count):
            sample_count = (
                int(valid_tensor[0, row_index].item())
                if isinstance(valid_tensor, torch.Tensor)
                else int(metadata.get("samples_per_token", source_configuration.get("samples_per_token", 50)))
            )
            canonical_eos_count = (
                int(canonical_tensor[0, row_index].item())
                if isinstance(canonical_tensor, torch.Tensor)
                else 0
            )
            token_counts = recover_mc_token_counts(
                log_probs[0, row_index],
                sample_count=sample_count,
                observed_alpha=observed_alpha,
                floor_mass=floor_mass,
            )
            counter = decoded_completion_counter(
                token_counts,
                source_tokenizer,
                canonical_eos_count=canonical_eos_count,
            )
            if sum(counter.values()) != sample_count:
                raise ValueError(f"Recovered sample count mismatch in {input_path} row {row_index}.")
            counters.append(counter)
            total_samples += sample_count

        legacy_multi = payload.get("multi_token_sample_counts")
        if isinstance(legacy_multi, torch.Tensor):
            total_legacy_multi_token_samples += int(legacy_multi.sum().item())

        approximate_raw = {
            **payload,
            RAW_COUNTER_KEY: counters,
            PREFIX_TEXT_KEY: prefixes,
            "prompt_text": api_prompt,
            "answer_text": answer,
            "metadata": {
                **metadata,
                "payload_schema_version": 2,
                "sample_representation": "legacy_decoded_token_counter_v1",
                "completion_text_semantics": "decoded_first_source_token_approximation",
                "empty_completion_text_semantics": "canonical_eos",
                "materialized_log_probs_tokenizer": args.source_tokenizer_name,
                "legacy_conversion_approximate": True,
                "legacy_conversion_information_loss": (
                    "raw API completion text and non-first source tokens were not stored"
                ),
            },
        }
        output, summary = materialize_payload(
            approximate_raw,
            destination_tokenizer,
            tokenizer_name=args.tokenizer_name,
            alignment_policy=args.alignment_policy,
            max_answer_tokens=max_answer_tokens,
            observed_alpha=observed_alpha,
            floor_mass=floor_mass,
            store_dtype=resolve_store_dtype(args.store_dtype),
        )
        output["metadata"]["legacy_conversion_approximate"] = True
        output["metadata"]["source_cache_file"] = str(input_path.resolve())
        destination = output_dir / input_path.name
        temporary = output_dir / f".{input_path.name}.tmp"
        torch.save(output, temporary)
        os.replace(temporary, destination)
        summaries.append(summary)

    destination_rows = sum(item["destination_rows"] for item in summaries)
    retained_rows = sum(item["retained_rows"] for item in summaries)
    manifest = {
        "schema_version": 1,
        "conversion": "legacy_decoded_token_projection_v1",
        "approximate": True,
        "source_cache_dir": str(input_dir.resolve()),
        "source_tokenizer_name": args.source_tokenizer_name,
        "source_tokenizer_revision": args.source_tokenizer_revision,
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_revision": args.tokenizer_revision,
        "vocab_size": len(destination_tokenizer),
        "alignment_policy": args.alignment_policy,
        "records": len(summaries),
        "destination_rows": destination_rows,
        "retained_rows": retained_rows,
        "exact_prefix_rate": retained_rows / max(destination_rows, 1),
        "legacy_multi_token_samples": total_legacy_multi_token_samples,
        "total_samples": total_samples,
        "legacy_multi_token_sample_rate": total_legacy_multi_token_samples
        / max(total_samples, 1),
        "observed_alpha": observed_alpha,
        "floor_mass": floor_mass,
        "store_dtype": args.store_dtype,
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
