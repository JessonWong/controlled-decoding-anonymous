"""Recover pre-action text for legacy Qwen fused caches using verified sampler evidence.

Writes a small JSON ledger, never changes the original MC or fused cache tensors.
The ledger can be passed to cache_context_features.py --state_manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_encoder import CONTEXT_PROTOCOL, file_sha256
from training.cache_context_features import atomic_json
from training.materialize_proxy_mc_cache import (
    source_positions, tokenizer_mapping_sha256, validate_record, validate_source_manifest,
)


def recover_record(source, payload, raw_path, raw_payload, dataset, tokenizer,
                   source_configuration, fused_record):
    if any(key in payload for key in ("source_row_indices", "destination_token_positions")):
        raise ValueError("Legacy recovery does not support caches remapped to a different tokenizer.")
    source_hash = file_sha256(source)
    if fused_record.get("output_sha256") != source_hash:
        raise ValueError(f"Fused manifest checksum mismatch for {source.name}.")
    raw_hash = file_sha256(raw_path)
    if fused_record.get("source_sha256") != raw_hash:
        raise ValueError(f"Original sampler checksum mismatch for {source.name}.")
    record = validate_record(raw_path, raw_payload, dataset, tokenizer, source_configuration,
                             shared_vocab_size=payload["log_probs"].shape[-1])
    if (not torch.equal(payload["labels"], raw_payload["labels"])
            or source_positions(payload, payload["labels"].shape[1]) != record.cached_positions
            or fused_record.get("dataset_idx") != record.dataset_idx
            or payload.get("metadata", {}).get("data_sha256") != fused_record.get("data_sha256")
            or fused_record.get("data_sha256") != raw_payload["metadata"]["data_sha256"]):
        raise ValueError(f"Fused cache rows differ from original sampler rows for {source.name}.")
    # Position t uses ONLY answer_token_ids[:t], never the label at t.
    prefixes = [tokenizer.decode(record.answer_token_ids[:t], skip_special_tokens=False)
                for t in record.cached_positions]
    if source_hash != file_sha256(source) or raw_hash != file_sha256(raw_path):
        raise ValueError("Cache changed while recovering context text.")
    return {"source_cache_sha256": source_hash, "original_sampler_sha256": raw_hash,
            "prompt_text": record.api_prompt, "sampled_prefix_texts": prefixes,
            "dataset_idx": record.dataset_idx, "cached_positions": list(record.cached_positions)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--source_dir", type=Path, default=None)
    parser.add_argument("--target_tokenizer", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    if args.output_file.exists():
        raise ValueError("Recovered state output already exists; choose a new output file.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive.")
    fused_manifest_path = args.input_dir / "proxy_mc_manifest.json"
    fused_manifest = json.loads(fused_manifest_path.read_text())
    configuration = fused_manifest["configuration"]
    raw_dir = args.source_dir or Path(configuration["source_cache_dir"])
    raw_manifest_path = raw_dir / "cache_manifest.json"
    raw_manifest = json.loads(raw_manifest_path.read_text())
    if configuration.get("source_manifest_sha256") != file_sha256(raw_manifest_path):
        raise ValueError("Original sampler manifest differs from fused cache provenance.")
    source_configuration = dict(validate_source_manifest(raw_manifest, SimpleNamespace(
        expected_target_model="qwen/qwen3-32b", expected_provider="DeepInfra",
        expect_append_no_think=True, expect_qwen_hard_no_think_prefill=True,
    )))
    source_configuration["_manifest_fingerprint"] = raw_manifest["configuration_fingerprint"]
    from datasets import DownloadConfig, load_dataset
    from transformers import AutoTokenizer

    dataset = load_dataset(
        configuration["dataset_name"], revision=configuration.get("dataset_revision"),
        download_config=DownloadConfig(local_files_only=args.local_files_only),
    )[configuration["dataset_split"]]
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_tokenizer or configuration["target_tokenizer_name_or_path"],
        revision=configuration.get("target_tokenizer_revision"),
        local_files_only=args.local_files_only,
    )
    if tokenizer_mapping_sha256(tokenizer) != configuration["target_tokenizer_sha256"]:
        raise ValueError("Target tokenizer differs from fused cache provenance.")
    records = {r["file"]: r for r in fused_manifest["records"]}
    if len(records) != len(fused_manifest["records"]):
        raise ValueError("Duplicate record names in fused manifest.")
    files = sorted(args.input_dir.glob("*.pt"))
    files = files[:args.limit] if args.limit is not None else files
    if not files:
        raise ValueError("No legacy fused cache files found.")
    result = {"schema_version": 1, "protocol": CONTEXT_PROTOCOL,
              "recovery": "verified_qwen_exact_mc50_sampler_v1",
              "fused_manifest_sha256": file_sha256(fused_manifest_path),
              "sampler_manifest_sha256": file_sha256(raw_manifest_path), "files": {}}
    for i, source in enumerate(files, 1):
        raw_path = raw_dir / source.name
        payload = torch.load(source, map_location="cpu", weights_only=True)
        raw_payload = torch.load(raw_path, map_location="cpu", weights_only=True)
        result["files"][source.name] = recover_record(
            source, payload, raw_path, raw_payload, dataset, tokenizer,
            source_configuration, records.get(source.name, {}),
        )
        print(f"recovered_context_records={i}/{len(files)} file={source.name}", flush=True)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_file, result)


if __name__ == "__main__":
    main()
