"""Replace cached risk-gate scores without repeating target-model sampling.

The sampled/fused cache already contains every target token, deterministic
base token, and dataset row needed to reconstruct the post-action prefixes
that were scored during cache construction.  This utility preserves all MC
and proxy tensors and only replaces ``risk_gate_scores`` and
``risk_gate_mask`` in a fresh output directory.

The output is written through a sibling partial directory and atomically
renamed only after every payload, source hash, dataset hash, and manifest
record has been validated.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from risk_gate import PrefixRiskGate


SCHEMA_VERSION = 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locally rescore a sampled/fused cache with a new PrefixRiskGate."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gate-checkpoint", required=True)
    parser.add_argument("--gate-model-name", required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument(
        "--target-tokenizer-trust-remote-code",
        action="store_true",
        help="Allow custom code when loading the local target tokenizer.",
    )
    parser.add_argument(
        "--target-tokenizer-fix-mistral-regex",
        action="store_true",
        help=(
            "Pass fix_mistral_regex=True when loading the target tokenizer. "
            "Required by Mistral-Small-24B-Instruct-2501 with current Transformers."
        ),
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--dataset-name", default="LLM-LAT/harmful-dataset")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--expected-records", type=int, default=41)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--gate-prompt-source",
        choices=("dataset", "target"),
        default="dataset",
        help=(
            "dataset uses the raw prompt seen while fitting the handoff gate; "
            "target appends the target model's /no_think switch."
        ),
    )
    return parser.parse_args(argv)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_mapping_sha256(tokenizer) -> str:
    vocabulary = tokenizer.get_vocab()
    tokens = sorted(
        ([int(token_id), str(token)] for token, token_id in vocabulary.items()),
        key=lambda item: (item[0], item[1]),
    )
    token_ids = [item[0] for item in tokens]
    if len(vocabulary) != len(tokenizer) or set(token_ids) != set(range(len(tokenizer))):
        raise ValueError("Target tokenizer does not define one contiguous token-to-ID map.")
    return sha256_payload(
        {
            "schema": "token_to_id_v1",
            "vocab_size": len(tokenizer),
            "tokens": tokens,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def resolve_threshold(config: Mapping[str, Any], requested: float | None) -> float:
    configured = config.get("recommended_handoff_threshold")
    if not isinstance(configured, (int, float)) or isinstance(configured, bool):
        raise ValueError("Gate config has no finite recommended_handoff_threshold.")
    configured = float(configured)
    if not math.isfinite(configured):
        raise ValueError("Gate recommended_handoff_threshold is not finite.")
    threshold = configured if requested is None else float(requested)
    if not math.isfinite(threshold):
        raise ValueError("Requested risk-gate threshold is not finite.")
    if not math.isclose(threshold, configured, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"Requested threshold {threshold} differs from the gate's recommended "
            f"handoff threshold {configured}."
        )
    return threshold


def validate_cache_tensors(payload: Mapping[str, Any], path: Path) -> tuple[int, list[int], list[int]]:
    if "labels" not in payload or not torch.is_tensor(payload["labels"]):
        raise ValueError(f"Missing tensor 'labels' in {path}.")
    labels = payload["labels"]
    base_ids = payload["risk_gate_token_ids"]
    old_scores = payload.get("risk_gate_scores")
    old_mask = payload.get("risk_gate_mask")
    if labels.ndim != 2 or labels.shape[0] != 1 or labels.numel() == 0:
        raise ValueError(f"labels must have shape [1, nonzero_seq_len] in {path}.")
    if base_ids.shape != labels.shape:
        raise ValueError(f"Base token IDs do not align with labels in {path}.")
    if labels.dtype != torch.long or base_ids.dtype != torch.long:
        raise ValueError(f"labels/base token IDs must be int64 in {path}.")
    if old_scores is not None or old_mask is not None:
        if (
            not torch.is_tensor(old_scores)
            or not torch.is_tensor(old_mask)
            or old_scores.shape != labels.shape
            or old_mask.shape != labels.shape
            or not old_scores.dtype.is_floating_point
            or old_mask.dtype != torch.bool
        ):
            raise ValueError(
                f"Old risk score/mask dtypes or shapes do not align with labels in {path}."
            )
    return (
        int(labels.shape[1]),
        [int(value) for value in labels[0].tolist()],
        [int(value) for value in base_ids[0].tolist()],
    )


def build_post_action_prefixes(
    tokenizer,
    target_ids: Sequence[int],
    base_token_ids: Sequence[int],
) -> list[str]:
    if not target_ids or len(target_ids) != len(base_token_ids):
        raise ValueError("Target and base token sequences must have the same nonzero length.")
    return [
        tokenizer.decode(
            list(target_ids[:position]) + [int(base_token_id)],
            skip_special_tokens=False,
        )
        for position, base_token_id in enumerate(base_token_ids)
    ]


def quantiles(values: torch.Tensor) -> dict[str, float]:
    levels = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    result = torch.quantile(values.double(), torch.tensor(levels, dtype=torch.double))
    return {f"p{int(level * 100):02d}": float(value) for level, value in zip(levels, result)}


def resolve_source_manifest(source_dir: Path) -> tuple[Path, dict[str, Any], bool]:
    """Load either a proxy-fused or a legacy/raw MC cache manifest.

    Proxy materialization manifests contain a record-level hash ledger.  The
    original exact-MC50 cache predates that ledger and only has
    ``cache_manifest.json``; its per-file hashes are established while the
    output ledger is built below.
    """

    proxy_path = source_dir / "proxy_mc_manifest.json"
    raw_path = source_dir / "cache_manifest.json"
    if proxy_path.is_file():
        return proxy_path, load_json(proxy_path), True
    if raw_path.is_file():
        manifest = load_json(raw_path)
        configuration = manifest.get("configuration")
        if not isinstance(configuration, dict):
            raise ValueError("Raw cache_manifest.json has no configuration object.")
        fingerprint = manifest.get("configuration_fingerprint")
        if fingerprint != sha256_payload(configuration):
            raise ValueError("Raw cache manifest configuration fingerprint is invalid.")
        return raw_path, manifest, False
    raise FileNotFoundError(
        f"Expected proxy_mc_manifest.json or cache_manifest.json in {source_dir}."
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source_dir = Path(args.source_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint = Path(args.gate_checkpoint).expanduser().resolve()
    gate_model_name = Path(args.gate_model_name).expanduser().resolve()
    target_tokenizer_path = Path(args.target_tokenizer).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    if not checkpoint.is_dir() or not gate_model_name.is_dir() or not target_tokenizer_path.is_dir():
        raise FileNotFoundError("Gate checkpoint, backbone, and target tokenizer must be local directories.")
    if args.expected_records <= 0 or args.batch_size <= 0:
        raise ValueError("expected-records and batch-size must be positive.")

    source_files = sorted(source_dir.glob("*.pt"))
    if len(source_files) != args.expected_records:
        raise ValueError(
            f"Expected {args.expected_records} source records, found {len(source_files)}."
        )
    source_manifest_path, source_manifest, has_record_ledger = resolve_source_manifest(
        source_dir
    )
    records_by_file: dict[str, dict[str, Any]] = {}
    if has_record_ledger:
        source_records = source_manifest.get("records")
        if not isinstance(source_records, list) or len(source_records) != len(source_files):
            raise ValueError("Source proxy_mc_manifest.json does not cover every cache file.")
        records_by_file = {
            record.get("file"): record
            for record in source_records
            if isinstance(record, dict) and isinstance(record.get("file"), str)
        }
        if set(records_by_file) != {path.name for path in source_files}:
            raise ValueError("Source manifest filenames do not exactly match source cache files.")
    else:
        source_configuration = source_manifest["configuration"]
        if not source_configuration.get("tokenizer_name"):
            raise ValueError("The legacy/raw cache does not declare a tokenizer name.")

    gate_config_path = checkpoint / "risk_head_config.json"
    gate_head_path = checkpoint / "risk_head.pt"
    gate_config = load_json(gate_config_path)
    threshold = resolve_threshold(gate_config, args.threshold)
    if gate_config.get("apply_biasnet_when") != "score < recommended_handoff_threshold":
        raise ValueError("Gate config has an incompatible BiasNet decision direction.")

    target_tokenizer = AutoTokenizer.from_pretrained(
        target_tokenizer_path,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=bool(args.target_tokenizer_trust_remote_code),
        fix_mistral_regex=bool(args.target_tokenizer_fix_mistral_regex),
    )
    target_tokenizer_sha256 = tokenizer_mapping_sha256(target_tokenizer)

    print(
        json.dumps(
            {
                "stage": "load_gate",
                "checkpoint": str(checkpoint),
                "threshold": threshold,
                "prompt_source": args.gate_prompt_source,
                "records": len(source_files),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    gate = PrefixRiskGate(
        checkpoint=checkpoint,
        device=torch.device(args.device),
        threshold=threshold,
        batch_size=args.batch_size,
        dtype=args.dtype,
        model_name=str(gate_model_name),
        load_in_4bit=bool(args.load_in_4bit),
        local_files_only=bool(args.local_files_only),
    )

    configuration = deepcopy(source_manifest.get("configuration", {}))
    if not has_record_ledger and isinstance(configuration.get("risk_gate"), dict):
        configuration["source_risk_gate"] = deepcopy(configuration["risk_gate"])
        configuration["risk_gate"] = {
            "path": str(checkpoint),
            "risk_head_config_sha256": sha256_file(gate_config_path),
            "risk_head_sha256": sha256_file(gate_head_path),
        }
    configuration.update(
        {
            "risk_gate_checkpoint": str(checkpoint),
            "risk_gate_config_sha256": sha256_file(gate_config_path),
            "risk_gate_head_sha256": sha256_file(gate_head_path),
            "risk_gate_model_name": str(gate_model_name),
            "risk_gate_threshold": threshold,
            "risk_gate_score_semantics": gate_config.get("estimand"),
            "risk_gate_apply_biasnet_when": gate_config.get("apply_biasnet_when"),
            "risk_gate_prompt_source": args.gate_prompt_source,
            "risk_gate_prompt_protocol": (
                "raw_dataset_prompt"
                if args.gate_prompt_source == "dataset"
                else "dataset_prompt_plus_newline_no_think"
            ),
            "risk_gate_rescore_schema_version": SCHEMA_VERSION,
            "risk_gate_rescorer_source_sha256": sha256_file(Path(__file__).resolve()),
            "target_tokenizer_sha256": target_tokenizer_sha256,
            "target_tokenizer_fix_mistral_regex": bool(
                args.target_tokenizer_fix_mistral_regex
            ),
        }
    )
    output_configuration_fingerprint = sha256_payload(configuration)

    from datasets import load_dataset

    dataset = load_dataset(args.dataset_name, revision=args.dataset_revision)[args.dataset_split]
    partial_dir = output_dir.with_name(
        f".{output_dir.name}.partial-{os.environ.get('SLURM_JOB_ID', os.getpid())}"
    )
    if partial_dir.exists():
        raise FileExistsError(f"Refusing to reuse partial directory: {partial_dir}")
    partial_dir.mkdir(parents=True)

    new_records: list[dict[str, Any]] = []
    all_scores: list[torch.Tensor] = []
    all_positions: list[torch.Tensor] = []
    total_masked = 0
    total_tokens = 0
    seen_dataset_indices: set[int] = set()
    for file_index, source_path in enumerate(source_files, 1):
        source_sha256 = sha256_file(source_path)
        parent_record = records_by_file.get(source_path.name, {})
        if has_record_ledger and parent_record.get("output_sha256") != source_sha256:
            raise ValueError(f"Source payload hash disagrees with manifest: {source_path}")
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
            raise ValueError(f"Malformed cache payload: {source_path}")
        metadata = dict(payload["metadata"])
        dataset_idx = metadata.get("dataset_idx")
        if isinstance(dataset_idx, bool) or not isinstance(dataset_idx, int):
            raise ValueError(f"Invalid dataset_idx in {source_path}.")
        if dataset_idx in seen_dataset_indices:
            raise ValueError(f"Duplicate dataset_idx={dataset_idx}.")
        seen_dataset_indices.add(dataset_idx)
        if has_record_ledger and parent_record.get("dataset_idx") != dataset_idx:
            raise ValueError(f"Manifest dataset_idx mismatch in {source_path}.")
        if not has_record_ledger:
            fingerprint = metadata.get("cache_configuration_fingerprint")
            if fingerprint != source_manifest.get("configuration_fingerprint"):
                raise ValueError(
                    f"Raw payload configuration fingerprint mismatch in {source_path}."
                )

        row = dataset[dataset_idx]
        question = row.get("prompt")
        answer = row.get("rejected")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(f"Dataset row {dataset_idx} has no prompt/rejected text.")
        data_sha256 = hashlib.sha256((question + answer).encode("utf-8")).hexdigest()
        data_md5 = hashlib.md5((question + answer).encode("utf-8")).hexdigest()
        if metadata.get("data_sha256") != data_sha256 or (
            has_record_ledger and parent_record.get("data_sha256") != data_sha256
        ):
            raise ValueError(f"Dataset content hash mismatch at dataset_idx={dataset_idx}.")
        if source_path.stem != data_md5:
            raise ValueError(f"Dataset content MD5 does not match filename {source_path.name}.")
        recorded_tokenizer_sha256 = metadata.get("target_tokenizer_sha256")
        if recorded_tokenizer_sha256 is not None and recorded_tokenizer_sha256 != target_tokenizer_sha256:
            raise ValueError(f"Target tokenizer hash mismatch in {source_path}.")

        rows, target_ids, base_token_ids = validate_cache_tensors(payload, source_path)
        prefixes = build_post_action_prefixes(target_tokenizer, target_ids, base_token_ids)
        score_prompt = question if args.gate_prompt_source == "dataset" else question + "\n/no_think"
        scores = gate.score_prefixes([score_prompt] * rows, prefixes).reshape(-1).float()
        if scores.shape != (rows,) or not torch.isfinite(scores).all():
            raise ValueError(f"Gate returned invalid scores for {source_path}.")
        mask = scores < threshold

        payload["risk_gate_scores"] = scores.unsqueeze(0).cpu()
        payload["risk_gate_mask"] = mask.unsqueeze(0).cpu()
        metadata.update(
            {
                "risk_gate_checkpoint": str(checkpoint),
                "risk_gate_threshold": threshold,
                "risk_gate_model_name": str(gate_model_name),
                "risk_gate_local_files_only": bool(args.local_files_only),
                "risk_gate_score_semantics": gate_config.get("estimand"),
                "risk_gate_prompt_source": args.gate_prompt_source,
                "risk_gate_prompt_protocol": (
                    "raw_dataset_prompt" if args.gate_prompt_source == "dataset" else "dataset_prompt_plus_newline_no_think"
                ),
                "risk_gate_rescore_schema_version": SCHEMA_VERSION,
                "risk_gate_rescored_from_cache_file": str(source_path),
                "risk_gate_rescored_from_cache_sha256": source_sha256,
                "target_tokenizer_sha256": target_tokenizer_sha256,
                "source_cache_configuration_fingerprint": metadata.get(
                    "cache_configuration_fingerprint"
                ),
                "cache_configuration_fingerprint": output_configuration_fingerprint,
            }
        )
        payload["metadata"] = metadata
        temporary_path = partial_dir / f".{source_path.name}.tmp"
        output_path = partial_dir / source_path.name
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
        output_sha256 = sha256_file(output_path)

        record = deepcopy(parent_record)
        record.update(
            {
                "file": source_path.name,
                "dataset_idx": dataset_idx,
                "data_sha256": data_sha256,
                "rows": rows,
                "source_sha256": source_sha256,
                "output_sha256": output_sha256,
                "risk_gate_rescore_source_sha256": source_sha256,
                "risk_gate_threshold": threshold,
                "risk_gate_score_min": float(scores.min()),
                "risk_gate_score_max": float(scores.max()),
                "risk_gate_score_mean": float(scores.mean()),
                "risk_gate_bias_mask_count": int(mask.sum()),
            }
        )
        new_records.append(record)
        all_scores.append(scores.cpu())
        all_positions.append(torch.arange(rows, dtype=torch.long))
        total_masked += int(mask.sum())
        total_tokens += rows
        print(
            f"rescored={file_index}/{len(source_files)} file={source_path.name} "
            f"dataset_idx={dataset_idx} rows={rows} bias_mask_rate={float(mask.float().mean()):.6f}",
            flush=True,
        )

    combined_scores = torch.cat(all_scores)
    combined_positions = torch.cat(all_positions)
    summary = deepcopy(source_manifest.get("summary", {}))
    summary.update(
        {
            "records": len(new_records),
            "tokens": total_tokens,
            "risk_gate_bias_mask_count": total_masked,
            "risk_gate_bias_mask_rate": total_masked / total_tokens,
            "risk_gate_score_mean": float(combined_scores.mean()),
            "risk_gate_score_std": float(combined_scores.std(unbiased=False)),
            "risk_gate_score_quantiles": quantiles(combined_scores),
            "risk_gate_position_score_correlation": float(
                torch.corrcoef(
                    torch.stack((combined_positions.double(), combined_scores.double()))
                )[0, 1]
            ),
        }
    )
    manifest = deepcopy(source_manifest)
    manifest.update(
        {
            "manifest_schema_version": int(source_manifest.get("manifest_schema_version", 1)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration,
            "configuration_fingerprint": output_configuration_fingerprint,
            "records": new_records,
            "summary": summary,
            "risk_gate_rescore": {
                "schema_version": SCHEMA_VERSION,
                "source_cache_dir": str(source_dir),
                "source_manifest_sha256": sha256_file(source_manifest_path),
                "source_manifest_name": source_manifest_path.name,
                "gate_prompt_source": args.gate_prompt_source,
                "target_prompt_uses_no_think": True,
                "gate_prompt_uses_no_think": args.gate_prompt_source == "target",
                "api_or_target_model_calls": 0,
            },
        }
    )
    manifest_path = partial_dir / source_manifest_path.name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial_dir, output_dir)
    print(
        json.dumps(
            {
                "stage": "complete",
                "output_dir": str(output_dir),
                "records": len(new_records),
                "tokens": total_tokens,
                "bias_mask_rate": total_masked / total_tokens,
                "score_quantiles": quantiles(combined_scores),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
