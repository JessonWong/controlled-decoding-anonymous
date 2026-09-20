"""Apply calibrated proxy/MC fusion parameters to an existing raw artifact.

This is the cheap second stage after ``eval_proxy_mc_fusion.py``.  It reuses
the persisted uncalibrated ``proxy_logits`` and exact ``mc_counts``; no model
or API is loaded.  Files are processed one at a time into a staging directory,
written through per-file atomic replacements, and the completed directory is
renamed into place only after its manifest has been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mc_reconstruction import fuse_proxy_logits_with_mc_counts  # noqa: E402


MANIFEST_NAME = "proxy_mc_manifest.json"
FUSION_MODE = "proxy_dirichlet_v1"
CALIBRATION_TOOL = "training/eval_proxy_mc_fusion.py"
HYPERPARAMETER_SOURCE = "offline_held_out_mc_event_nll"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rematerialize fused log_probs at the label-free T/kappa selected by "
            "eval_proxy_mc_fusion.py. No proxy forward or API call is performed."
        )
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--calibration_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--store_dtype",
        choices=["inherit", "float16", "float32"],
        default="inherit",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for fusion arithmetic; input/output tensors remain on CPU.",
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
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read {description} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _validate_manifest(manifest: Mapping[str, Any], path: Path) -> dict[str, Any]:
    if manifest.get("manifest_schema_version") != 1:
        raise ValueError(f"Unsupported proxy manifest schema in {path}.")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"Proxy manifest has no configuration object: {path}")
    fingerprint = hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()
    if manifest.get("configuration_fingerprint") != fingerprint:
        raise ValueError(f"Proxy manifest configuration fingerprint mismatch: {path}")
    if configuration.get("mc_fusion_mode") != FUSION_MODE:
        raise ValueError(f"Unsupported mc_fusion_mode in {path}.")
    for key, expected in {
        "proxy_logits_key": "proxy_logits",
        "proxy_logits_semantics": "uncalibrated_raw_logits_shared_vocab",
    }.items():
        if configuration.get(key) != expected:
            raise ValueError(
                f"Proxy manifest configuration {key} must be {expected!r}."
            )
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Proxy manifest has no records: {path}")
    names = [record.get("file") for record in records if isinstance(record, dict)]
    if len(names) != len(records) or any(not isinstance(name, str) for name in names):
        raise ValueError(f"Proxy manifest contains an invalid record entry: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Proxy manifest contains duplicate filenames: {path}")
    return configuration


def _selected_fusion(
    calibration: Mapping[str, Any], calibration_path: Path, input_dir: Path
) -> tuple[float, float, dict[str, Any]]:
    if calibration.get("tool") != CALIBRATION_TOOL:
        raise ValueError(
            f"Calibration JSON was not produced by {CALIBRATION_TOOL}: {calibration_path}"
        )
    selection = calibration.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Calibration JSON has no selection object.")
    if selection.get("uses_answer_labels") is not False:
        raise ValueError("Calibration must explicitly declare uses_answer_labels=false.")
    if selection.get("uses_risk_gate_token_ids") is not False:
        raise ValueError(
            "Calibration must explicitly declare uses_risk_gate_token_ids=false."
        )
    if selection.get("objective") != "aggregate_held_out_mc_event_mean_nll":
        raise ValueError("Calibration used an unsupported selection objective.")
    selected = selection.get("selected_fusion")
    if not isinstance(selected, dict):
        raise ValueError("Calibration JSON has no selected_fusion object.")
    try:
        temperature = float(selected["proxy_temperature"])
        prior_strength = float(selected["prior_strength"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Calibration selected_fusion is incomplete.") from error
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Selected proxy temperature must be finite and positive.")
    if not math.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("Selected prior strength must be finite and positive.")
    data = calibration.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("input_dir"), str):
        raise ValueError("Calibration JSON does not identify its input_dir.")
    calibrated_input = Path(data["input_dir"]).expanduser().resolve()
    if calibrated_input != input_dir:
        raise ValueError(
            "Calibration input_dir does not match the rematerialization input: "
            f"{calibrated_input} != {input_dir}."
        )
    calibration_summary = {
        "schema_version": calibration.get("schema_version"),
        "tool": calibration.get("tool"),
        "objective": selection["objective"],
        "split_method": selection.get("split_method"),
        "calibration_fraction": selection.get("calibration_fraction"),
        "split_seeds": selection.get("split_seeds"),
        "uses_answer_labels": False,
        "uses_risk_gate_token_ids": False,
        "selected_fusion": {
            "proxy_temperature": temperature,
            "prior_strength": prior_strength,
        },
    }
    return temperature, prior_strength, calibration_summary


def _resolve_store_dtype(name: str, configuration: Mapping[str, Any]) -> tuple[str, torch.dtype]:
    if name == "inherit":
        name = str(configuration.get("fused_log_probs_dtype", ""))
    if name == "float16":
        return name, torch.float16
    if name == "float32":
        return name, torch.float32
    raise ValueError(
        "--store_dtype inherit requires manifest fused_log_probs_dtype to be "
        "float16 or float32."
    )


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def _validate_payload(
    payload: Mapping[str, Any],
    path: Path,
    configuration: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    proxy_logits = payload.get("proxy_logits")
    mc_counts = payload.get("mc_counts")
    log_probs = payload.get("log_probs")
    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (proxy_logits, mc_counts, log_probs)
    ):
        raise ValueError(f"{path} is missing proxy_logits, mc_counts, or log_probs.")
    assert isinstance(proxy_logits, torch.Tensor)
    assert isinstance(mc_counts, torch.Tensor)
    assert isinstance(log_probs, torch.Tensor)
    if proxy_logits.dim() != 3 or proxy_logits.shape[0] != 1:
        raise ValueError(f"{path}: proxy_logits must have shape [1, rows, vocab].")
    if mc_counts.shape != proxy_logits.shape or log_probs.shape != proxy_logits.shape:
        raise ValueError(f"{path}: proxy/count/fused tensor shapes do not match.")
    if not proxy_logits.is_floating_point() or not torch.isfinite(proxy_logits).all().item():
        raise ValueError(f"{path}: proxy_logits must be finite floating-point values.")
    if mc_counts.dtype == torch.bool or mc_counts.is_floating_point() or mc_counts.is_complex():
        raise ValueError(f"{path}: mc_counts must use a real integer dtype.")
    if (mc_counts < 0).any().item():
        raise ValueError(f"{path}: mc_counts must be non-negative.")
    shared_vocab_size = int(configuration.get("shared_vocab_size", -1))
    proxy_logits_vocab_size = int(configuration.get("proxy_logits_vocab_size", -1))
    if proxy_logits.shape[-1] != shared_vocab_size or shared_vocab_size != proxy_logits_vocab_size:
        raise ValueError(f"{path}: tensor vocabulary disagrees with the manifest.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: missing metadata object.")
    for key in (
        "mc_fusion_mode",
        "proxy_temperature",
        "proxy_prior_strength",
        "shared_vocab_size",
        "proxy_logits_key",
        "proxy_logits_semantics",
        "proxy_logits_vocab_size",
    ):
        if metadata.get(key) != configuration.get(key):
            raise ValueError(f"{path}: record metadata disagrees on {key}.")
    return proxy_logits, mc_counts, metadata


def rematerialize_cache(
    input_dir: Path | str,
    calibration_json: Path | str,
    output_dir: Path | str,
    *,
    store_dtype: str = "inherit",
    device: str = "auto",
) -> dict[str, Any]:
    input_dir = Path(input_dir).expanduser().resolve()
    calibration_path = Path(calibration_json).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    manifest_path = input_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing input manifest: {manifest_path}")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Missing calibration JSON: {calibration_path}")

    source_manifest = _read_json(manifest_path, "proxy manifest")
    source_configuration = _validate_manifest(source_manifest, manifest_path)
    calibration = _read_json(calibration_path, "calibration JSON")
    temperature, prior_strength, calibration_summary = _selected_fusion(
        calibration, calibration_path, input_dir
    )
    store_dtype_name, output_dtype = _resolve_store_dtype(
        store_dtype, source_configuration
    )
    compute_device = _resolve_device(device)
    source_manifest_sha256 = sha256_file(manifest_path)
    calibration_sha256 = sha256_file(calibration_path)
    rematerializer_sha256 = sha256_file(Path(__file__).resolve())
    fusion_core_sha256 = sha256_file(PROJECT_ROOT / "mc_reconstruction.py")

    manifest_configuration = {
        **source_configuration,
        "proxy_temperature": temperature,
        "proxy_prior_strength": prior_strength,
        "fused_log_probs_dtype": store_dtype_name,
        "fusion_hyperparameter_source": HYPERPARAMETER_SOURCE,
        "fusion_calibration_json": str(calibration_path),
        "fusion_calibration_json_sha256": calibration_sha256,
        "fusion_calibration_schema_version": calibration.get("schema_version"),
        "fusion_calibration_objective": calibration_summary["objective"],
        "fusion_calibration_split_method": calibration_summary["split_method"],
        "fusion_calibration_fraction": calibration_summary["calibration_fraction"],
        "fusion_calibration_split_seeds": calibration_summary["split_seeds"],
        "rematerialized_from_cache_dir": str(input_dir),
        "rematerialized_from_manifest_sha256": source_manifest_sha256,
        "rematerialized_from_configuration_fingerprint": source_manifest[
            "configuration_fingerprint"
        ],
        "rematerializer_source_sha256": rematerializer_sha256,
        "fusion_core_source_sha256": fusion_core_sha256,
    }
    configuration_fingerprint = hashlib.sha256(
        canonical_json_bytes(manifest_configuration)
    ).hexdigest()

    records = source_manifest["records"]
    record_by_name = {record["file"]: record for record in records}
    input_files = sorted(input_dir.glob("*.pt"))
    if {path.name for path in input_files} != set(record_by_name):
        raise ValueError("Input .pt filenames disagree with proxy_mc_manifest.json.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    output_records: list[dict[str, Any]] = []
    total_rows = 0
    for record_number, path in enumerate(input_files, start=1):
        source_record = record_by_name[path.name]
        source_file_sha256 = sha256_file(path)
        if source_file_sha256 != source_record.get("output_sha256"):
            raise ValueError(f"Input artifact hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} does not contain a dictionary payload.")
        proxy_logits, mc_counts, source_metadata = _validate_payload(
            payload, path, source_configuration
        )
        fused_float32 = fuse_proxy_logits_with_mc_counts(
            proxy_logits.to(compute_device),
            mc_counts.to(compute_device),
            temperature=temperature,
            prior_strength=prior_strength,
            dtype=torch.float32,
        ).cpu()
        normalizers = torch.logsumexp(fused_float32, dim=-1)
        if not torch.allclose(
            normalizers, torch.zeros_like(normalizers), atol=2e-5, rtol=0.0
        ):
            raise ValueError(f"Fusion normalization check failed for {path}.")
        fused_log_probs = fused_float32.to(output_dtype)

        output_metadata = {
            **source_metadata,
            "proxy_temperature": temperature,
            "proxy_prior_strength": prior_strength,
            "fused_log_probs_dtype": store_dtype_name,
            "fusion_hyperparameter_source": HYPERPARAMETER_SOURCE,
            "fusion_calibration_json": str(calibration_path),
            "fusion_calibration_json_sha256": calibration_sha256,
            "fusion_calibration_schema_version": calibration.get("schema_version"),
            "fusion_calibration_objective": calibration_summary["objective"],
            "fusion_calibration_split_method": calibration_summary["split_method"],
            "fusion_calibration_fraction": calibration_summary["calibration_fraction"],
            "fusion_calibration_split_seeds": calibration_summary["split_seeds"],
            "proxy_mc_configuration_fingerprint": configuration_fingerprint,
            "rematerialized_from_cache_file": str(path),
            "rematerialized_from_cache_sha256": source_file_sha256,
            "rematerialized_from_manifest_sha256": source_manifest_sha256,
            "rematerialized_from_configuration_fingerprint": source_manifest[
                "configuration_fingerprint"
            ],
            "rematerializer_source_sha256": rematerializer_sha256,
            "fusion_core_source_sha256": fusion_core_sha256,
        }
        output_payload = dict(payload)
        output_payload["log_probs"] = fused_log_probs
        output_payload["metadata"] = output_metadata

        destination = staging_dir / path.name
        temporary = staging_dir / f".{path.name}.tmp"
        torch.save(output_payload, temporary)
        os.replace(temporary, destination)
        output_sha256 = sha256_file(destination)
        rows = int(proxy_logits.shape[1])
        output_record = {
            **source_record,
            "parent_output_sha256": source_file_sha256,
            "output_sha256": output_sha256,
            "rows": rows,
            "proxy_temperature": temperature,
            "proxy_prior_strength": prior_strength,
            "fused_log_probs_dtype": store_dtype_name,
        }
        output_records.append(output_record)
        total_rows += rows
        print(
            f"rematerialized={record_number}/{len(input_files)} "
            f"file={path.name} rows={rows}",
            flush=True,
        )
        del (
            payload,
            output_payload,
            proxy_logits,
            mc_counts,
            fused_float32,
            fused_log_probs,
        )

    summary = dict(source_manifest.get("summary") or {})
    summary.update(
        {
            "record_count": len(output_records),
            "row_count": total_rows,
            "rematerialization_forward_count": 0,
            "rematerialization_api_call_count": 0,
        }
    )
    output_manifest = {
        "manifest_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": manifest_configuration,
        "configuration_fingerprint": configuration_fingerprint,
        # Keep the original exact-MC source manifest at the legacy key used by
        # the training preflight, and separately embed this stage's parent.
        "source_manifest": source_manifest.get("source_manifest"),
        "parent_proxy_mc_manifest": source_manifest,
        "calibration": calibration_summary,
        "records": output_records,
        "summary": summary,
    }
    temporary_manifest = staging_dir / f".{MANIFEST_NAME}.tmp"
    temporary_manifest.write_text(
        json.dumps(
            output_manifest,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, staging_dir / MANIFEST_NAME)
    # output_dir was checked before all work and is checked again immediately
    # before the atomic directory rename to fail closed under concurrent runs.
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite concurrently-created output: {output_dir}")
    os.replace(staging_dir, output_dir)
    return {
        "output_dir": str(output_dir),
        "manifest": str(output_dir / MANIFEST_NAME),
        "configuration_fingerprint": configuration_fingerprint,
        "proxy_temperature": temperature,
        "proxy_prior_strength": prior_strength,
        "fused_log_probs_dtype": store_dtype_name,
        "device": str(compute_device),
        "records": len(output_records),
        "rows": total_rows,
    }


def main() -> None:
    args = parse_args()
    result = rematerialize_cache(
        args.input_dir,
        args.calibration_json,
        args.output_dir,
        store_dtype=args.store_dtype,
        device=args.device,
    )
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
