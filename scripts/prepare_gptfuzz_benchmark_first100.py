#!/usr/bin/env python3
"""Materialize the first 100 normalized benchmark prompts for GPTFuzz."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_data import load_benchmark_records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=("advbench", "harmbench", "sorrybench"),
        required=True,
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")

    records = load_benchmark_records(
        args.benchmark, path=args.source, limit=args.limit
    )
    if len(records) != args.limit:
        raise RuntimeError(
            f"{args.benchmark}: expected {args.limit} records, found {len(records)}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("index", "text"))
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow({"index": index, "text": record.prompt})

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "benchmark": args.benchmark,
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "output": str(args.output.resolve()),
        "question_count": len(records),
        "records": [
            {"benchmark_index": index, **record.metadata}
            for index, record in enumerate(records)
        ],
    }
    args.metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "benchmark": args.benchmark,
                "question_count": len(records),
                "source_sha256": metadata["source_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
