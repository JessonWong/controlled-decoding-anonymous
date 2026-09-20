"""Materialize frozen pre-action context features alongside existing MC caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from context_encoder import (
    CONTEXT_MANIFEST, FrozenContextEncoder, cache_states, file_sha256,
    json_sha256, load_context_manifest, load_context_sidecar, state_identity, states_from_manifest,
)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def materialize_record(source: Path, destination: Path, encoder: FrozenContextEncoder,
                       batch_size: int, state_manifest=None, state_manifest_sha256=None) -> dict:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    source_hash = file_sha256(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    recovered = "prompt_text" not in payload and "sampled_prefix_texts" not in payload
    if recovered and state_manifest is not None:
        prompt, prefixes = states_from_manifest(source, payload, state_manifest)
    else:
        prompt, prefixes = cache_states(payload)
    features = []
    for start in range(0, len(prefixes), batch_size):
        part = prefixes[start:start + batch_size]
        features.append(encoder.encode([prompt] * len(part), part).cpu())
    if source_hash != file_sha256(source):
        raise ValueError(f"Source changed during context extraction: {source.name}.")
    sidecar = {
        "schema_version": 1, "context_features": torch.cat(features),
        "source_row_indices": list(range(len(prefixes))),
        **state_identity(prompt, prefixes),
        "source_cache_sha256": source_hash,
        "context_contract_sha256": json_sha256(encoder.contract),
    }
    if recovered:
        if not state_manifest_sha256:
            raise ValueError("Recovered state manifest requires its SHA256.")
        sidecar["state_manifest_sha256"] = state_manifest_sha256
        sidecar["recovered_state"] = {"prompt_text": prompt, "sampled_prefix_texts": prefixes}
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(sidecar, temporary)
    os.replace(temporary, destination)
    return {"source_cache_sha256": source_hash, "sidecar_sha256": file_sha256(destination),
            "rows": len(prefixes)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--encoder_model", required=True)
    parser.add_argument("--state_manifest", type=Path, default=None,
                        help="Explicit verified text recovery for legacy MC caches without row prefixes.")
    parser.add_argument("--encoder_tokenizer", default=None)
    parser.add_argument("--encoder_revision", default=None)
    parser.add_argument("--encoder_tokenizer_revision", default=None)
    parser.add_argument("--encoder_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("Context sidecars must use a separate output directory.")
    if args.batch_size <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("batch_size and limit must be positive.")
    files = sorted(p for p in args.input_dir.glob("*.pt") if p.name != "global_unigram_prior.pt")
    files = files[:args.limit] if args.limit is not None else files
    if not files:
        raise ValueError("No MC source files found.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / CONTEXT_MANIFEST
    manifest = load_context_manifest(args.output_dir) if manifest_path.exists() else None
    if any(args.output_dir.iterdir()) and not args.resume:
        raise ValueError("Output directory is not empty; use --resume to validate and continue.")
    state_manifest = json.loads(args.state_manifest.read_text()) if args.state_manifest else None
    state_manifest_hash = file_sha256(args.state_manifest) if args.state_manifest else None
    if manifest is not None and manifest.get("state_manifest_sha256") != state_manifest_hash:
        raise ValueError("Resume requires the same recovered state manifest.")
    encoder = FrozenContextEncoder.load(
        args.encoder_model, tokenizer_name_or_path=args.encoder_tokenizer,
        revision=args.encoder_revision, tokenizer_revision=args.encoder_tokenizer_revision,
        dtype=args.encoder_dtype, max_length=args.max_length, device=args.device,
        local_files_only=args.local_files_only,
        expected_contract=manifest["contract"] if manifest else None,
    )
    if manifest is None:
        manifest = {"schema_version": 1, "contract": encoder.contract,
                    "contract_sha256": json_sha256(encoder.contract), "files": {},
                    "state_manifest_sha256": state_manifest_hash}
    manifest["status"] = "in_progress"
    atomic_json(manifest_path, manifest)
    for index, source in enumerate(files, 1):
        destination = args.output_dir / source.name
        if destination.exists():
            payload = torch.load(source, map_location="cpu", weights_only=True)
            if source.name not in manifest["files"]:
                # Recover a completed sidecar after interruption before manifest commit.
                manifest["files"][source.name] = {
                    "source_cache_sha256": file_sha256(source),
                    "sidecar_sha256": file_sha256(destination),
                    "rows": int(payload["labels"].shape[1]),
                }
            load_context_sidecar(source, payload, args.output_dir, manifest)
        else:
            manifest["files"][source.name] = materialize_record(
                source, destination, encoder, args.batch_size, state_manifest, state_manifest_hash,
            )
        atomic_json(manifest_path, manifest)
        print(f"context_records={index}/{len(files)} file={source.name}", flush=True)
    manifest["status"] = "complete"
    atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
