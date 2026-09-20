#!/usr/bin/env python3
"""Snapshot and flatten the current GPTFuzz Top-k templates for judging."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(path: Path, limit: int) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "goal" not in rows[0]:
        raise ValueError(f"Expected AdvBench goal/target CSV: {path}")
    questions = [row["goal"] for row in rows[:limit]]
    if len(questions) != limit:
        raise ValueError(f"Expected at least {limit} AdvBench rows, found {len(questions)}")
    return questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--question-limit", type=int, default=100)
    args = parser.parse_args()
    if args.top_k <= 0 or args.question_limit <= 0:
        parser.error("top-k and question-limit must be positive")

    csv.field_size_limit(sys.maxsize)
    questions = load_questions(args.questions, args.question_limit)
    with args.results.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    candidates = []
    for position, row in enumerate(rows):
        labels = [int(value) for value in ast.literal_eval(row["results"])]
        responses = ast.literal_eval(row["response"])
        if len(labels) != args.question_limit or len(responses) != args.question_limit:
            continue
        candidates.append(
            {
                "row_position": position,
                "source_index": row.get("index"),
                "template": row["prompt"],
                "labels": labels,
                "responses": [str(response) for response in responses],
                "jailbreaks": sum(labels),
            }
        )
    if not candidates:
        raise RuntimeError("No complete 100-question GPTFuzz records found")

    candidates.sort(key=lambda item: (-item["jailbreaks"], item["row_position"]))
    selected = candidates[: min(args.top_k, len(candidates))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    metadata_records = []
    record_index = 0
    with args.output.open("w", encoding="utf-8") as output:
        for rank, candidate in enumerate(selected, start=1):
            template_asr = candidate["jailbreaks"] / args.question_limit
            for question_index, (question, response) in enumerate(
                zip(questions, candidate["responses"])
            ):
                record = {
                    "prompt": question,
                    "completion": response,
                    "benchmark": "advbench",
                    "benchmark_index": question_index,
                    "template_rank": rank,
                    "template_source_index": candidate["source_index"],
                    "template_asr": template_asr,
                    "gptfuzz_predictor_label": candidate["labels"][question_index],
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                metadata_records.append(
                    {
                        "source_record_index": record_index,
                        "template_rank": rank,
                        "template_source_index": candidate["source_index"],
                        "benchmark_index": question_index,
                    }
                )
                record_index += 1

    metadata = {
        "source_results": str(args.results.resolve()),
        "source_results_sha256": sha256_file(args.results),
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256_file(args.questions),
        "question_count": args.question_limit,
        "top_k": len(selected),
        "candidate_count": len(candidates),
        "selected_templates": [
            {
                "rank": rank,
                "source_index": item["source_index"],
                "jailbreaks": item["jailbreaks"],
                "asr": item["jailbreaks"] / args.question_limit,
            }
            for rank, item in enumerate(selected, start=1)
        ],
        "records": metadata_records,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata["selected_templates"]))


if __name__ == "__main__":
    main()
