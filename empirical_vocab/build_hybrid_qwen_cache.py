"""Materialize exact-MC caches in Qwen + fixed empirical-extension coordinates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (ROOT, os.path.join(ROOT, "training")):
    if path not in sys.path:
        sys.path.insert(0, path)

from empirical_vocab.hybrid_qwen import (  # noqa: E402
    SCHEMA,
    build_hybrid_qwen_vocab,
    counter_to_hybrid_counts,
)
from training.pre_logits_sampled_openweight import (  # noqa: E402
    resolve_torch_dtype,
    sampled_ids_to_log_probs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vocab_out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--extension_capacity", type=int, default=5000)
    parser.add_argument(
        "--store_dtype", choices=["float16", "float32"], default="float16"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uniform_sample_count(valid: torch.Tensor, path: Path) -> int:
    values = torch.unique(valid.detach().cpu().long())
    if values.numel() != 1 or int(values[0]) <= 0:
        raise ValueError(
            f"{path} must have one positive exact-MC sample count at every position; "
            f"found {values.tolist()}."
        )
    return int(values[0])


def main() -> None:
    args = parse_args()
    if args.extension_capacity <= 0:
        raise ValueError("--extension_capacity must be positive.")

    from transformers import AutoTokenizer

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    vocab_out = Path(args.vocab_out)
    files = sorted(input_dir.glob("*.pt"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No input .pt files found in {input_dir}.")
    existing = sorted(output_dir.glob("*.pt")) if output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} cache files; "
            "pass --overwrite to replace them."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=False,
        local_files_only=args.local_files_only,
    )
    payloads = [
        torch.load(path, map_location="cpu", weights_only=False) for path in files
    ]
    all_counters: list[dict[str, int]] = []
    for path, payload in zip(files, payloads):
        counters = payload.get("sampled_completion_text_counts")
        if not isinstance(counters, list) or not counters:
            raise ValueError(f"{path} has no sampled_completion_text_counts list.")
        labels = payload.get("labels")
        if not isinstance(labels, torch.Tensor) or labels.shape != (1, len(counters)):
            raise ValueError(f"{path} labels do not match its sampled positions.")
        all_counters.extend(dict(counter) for counter in counters)

    vocab, extension_frequencies = build_hybrid_qwen_vocab(
        all_counters,
        tokenizer,
        extension_capacity=args.extension_capacity,
    )
    vocab.save(
        vocab_out,
        tokenizer_name_or_path=str(args.tokenizer),
        extension_frequencies=dict(extension_frequencies),
    )
    vocab_sha256 = _sha256(vocab_out)
    store_dtype = resolve_torch_dtype(args.store_dtype)

    total_mass = 0
    extension_mass = 0
    oov_mass = 0
    summaries: list[dict[str, Any]] = []
    for index, (path, payload) in enumerate(zip(files, payloads), start=1):
        counters = [dict(counter) for counter in payload["sampled_completion_text_counts"]]
        labels = payload["labels"].detach().cpu().long()
        if labels.numel() and (
            int(labels.min()) < 0 or int(labels.max()) >= vocab.base_vocab_size
        ):
            raise ValueError(
                f"{path} labels are not in the Qwen base vocabulary; source cache "
                "must be collected at Qwen token boundaries."
            )
        valid = payload.get("valid_sample_counts")
        if not isinstance(valid, torch.Tensor) or valid.shape != labels.shape:
            raise ValueError(f"{path} has invalid valid_sample_counts.")
        samples_per_position = _uniform_sample_count(valid, path)

        sampled_action_rows: list[list[int]] = []
        support_ids: list[torch.Tensor] = []
        support_counts: list[torch.Tensor] = []
        per_position_extension: list[int] = []
        per_position_oov: list[int] = []
        for position, counter in enumerate(counters):
            sparse = counter_to_hybrid_counts(counter, vocab, tokenizer)
            if sum(sparse.values()) != samples_per_position:
                raise ValueError(
                    f"{path} position {position} counter total {sum(sparse.values())} "
                    f"does not match exact-MC count {samples_per_position}."
                )
            ordered = sorted(sparse.items())
            ids = [action_id for action_id, _ in ordered]
            values = [count for _, count in ordered]
            row = [
                action_id
                for action_id, count in ordered
                for _ in range(int(count))
            ]
            sampled_action_rows.append(row)
            support_ids.append(torch.tensor(ids, dtype=torch.long))
            support_counts.append(torch.tensor(values, dtype=torch.long))
            ext_count = sum(
                count
                for action_id, count in ordered
                if vocab.extension_start_id
                <= action_id
                < vocab.extension_start_id + vocab.extension_capacity
            )
            current_oov = int(sparse.get(vocab.oov_id, 0))
            per_position_extension.append(int(ext_count))
            per_position_oov.append(current_oov)
            total_mass += sum(values)
            extension_mass += int(ext_count)
            oov_mass += current_oov

        sampled_actions = torch.tensor(sampled_action_rows, dtype=torch.long)
        metadata = dict(payload.get("metadata") or {})
        observed_alpha = float(metadata.get("observed_alpha", 0.1))
        floor_mass = float(metadata.get("floor_mass", 1e-4))
        log_probs = sampled_ids_to_log_probs(
            sampled_token_ids=sampled_actions,
            vocab_size=vocab.size,
            observed_alpha=observed_alpha,
            floor_mass=floor_mass,
            dtype=store_dtype,
        ).unsqueeze(0)

        source_vocab_size = int(payload["log_probs"].shape[-1])
        metadata.update(
            {
                "source_materialized_vocab_size": source_vocab_size,
                "materialized_log_probs_tokenizer": str(args.tokenizer),
                "hybrid_action_schema": SCHEMA,
                "hybrid_base_vocab_size": vocab.base_vocab_size,
                "hybrid_extension_capacity": vocab.extension_capacity,
                "hybrid_extension_count": vocab.extension_count,
                "hybrid_extension_start_id": vocab.extension_start_id,
                "hybrid_oov_id": vocab.oov_id,
                "hybrid_total_vocab_size": vocab.size,
                "hybrid_vocab_path": str(vocab_out.resolve()),
                "hybrid_vocab_sha256": vocab_sha256,
                "hybrid_input_tokenization": "unchanged_qwen",
                "hybrid_extension_semantics": "output_string_macro_action",
            }
        )
        output = dict(payload)
        output.update(
            {
                "log_probs": log_probs.cpu(),
                "labels": labels,
                "hybrid_mc_support_ids": support_ids,
                "hybrid_mc_support_counts": support_counts,
                "hybrid_extension_sample_counts": torch.tensor(
                    per_position_extension, dtype=torch.long
                ).unsqueeze(0),
                "hybrid_oov_sample_counts": torch.tensor(
                    per_position_oov, dtype=torch.long
                ).unsqueeze(0),
                "metadata": metadata,
            }
        )
        destination = output_dir / path.name
        temporary = output_dir / f".{path.name}.tmp"
        torch.save(output, temporary)
        os.replace(temporary, destination)
        summary = {
            "record": path.name,
            "positions": labels.shape[1],
            "valid_min": int(valid.min()),
            "valid_max": int(valid.max()),
            "extension_samples": sum(per_position_extension),
            "oov_samples": sum(per_position_oov),
            "labels_in_qwen_base": int(labels.numel()),
            "finite_log_probs": bool(torch.isfinite(log_probs).all()),
            "normalized_max_abs_error": float(
                (torch.logsumexp(log_probs.float(), dim=-1)).abs().max()
            ),
        }
        summaries.append(summary)
        print(
            f"[{index}/{len(files)}] {path.name} positions={labels.shape[1]} "
            f"shape={tuple(log_probs.shape)} extension_samples={summary['extension_samples']} "
            f"oov_samples={summary['oov_samples']} finite={summary['finite_log_probs']}",
            flush=True,
        )

    manifest = {
        "schema": SCHEMA,
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "vocab_path": str(vocab_out.resolve()),
        "vocab_sha256": vocab_sha256,
        "records": len(summaries),
        "base_vocab_size": vocab.base_vocab_size,
        "extension_capacity": vocab.extension_capacity,
        "extension_count": vocab.extension_count,
        "unused_extension_slots": vocab.extension_capacity - vocab.extension_count,
        "extension_start_id": vocab.extension_start_id,
        "oov_id": vocab.oov_id,
        "total_vocab_size": vocab.size,
        "sampled_mass": total_mass,
        "extension_sampled_mass": extension_mass,
        "extension_sampled_mass_fraction": extension_mass / max(total_mass, 1),
        "oov_sampled_mass": oov_mass,
        "sampled_mass_coverage": (total_mass - oov_mass) / max(total_mass, 1),
        "summaries": summaries,
    }
    manifest_path = output_dir / "hybrid_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "MANIFEST:",
        json.dumps({key: value for key, value in manifest.items() if key != "summaries"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
