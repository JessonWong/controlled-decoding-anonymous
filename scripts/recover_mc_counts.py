"""Portable extraction of the existing floor-cache count-recovery step.

This operation makes no API calls and refuses to overwrite an output directory.
It applies the same reconstruction validation as the original job script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mc_reconstruction import floor_log_probs_to_mc_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    sources = sorted(args.raw_dir.glob("*.pt"))
    if not sources:
        raise ValueError("No .pt records found in raw_dir.")
    if args.output_dir.exists():
        raise FileExistsError("Output directory already exists.")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=".counts-partial-", dir=args.output_dir.parent))
    try:
        for source in sources:
            payload = torch.load(source, map_location="cpu", weights_only=False)
            metadata = payload["metadata"]
            counts = floor_log_probs_to_mc_counts(
                payload["log_probs"], sample_counts=payload["valid_sample_counts"],
                observed_alpha=float(metadata["observed_alpha"]),
                floor_mass=float(metadata["floor_mass"]), dtype=torch.int32,
                validate_recovered_counts=True,
            )
            if counts.shape != payload["log_probs"].shape:
                raise ValueError(f"Recovered count shape mismatch: {source.name}")
            if not torch.equal(counts.sum(dim=-1).long(), payload["valid_sample_counts"].long()):
                raise ValueError(f"Recovered count total mismatch: {source.name}")
            torch.save({"mc_counts": counts, "valid_sample_counts": payload["valid_sample_counts"]},
                       partial / source.name)
        manifest = {"records": len(sources), "reconstruction": "floor_log_probs_to_mc_counts_v1", "api_calls": 0}
        (partial / "counts_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        partial.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(partial)
        raise
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()

