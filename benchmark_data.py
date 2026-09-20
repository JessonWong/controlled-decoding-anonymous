"""Load evaluation prompts from AdvBench, HarmBench, and SORRY-Bench.

The benchmark corpora are intentionally not vendored in this repository.  In
particular, SORRY-Bench requires accepting the dataset's access agreement.
This module only normalizes a user-provided local file into prompt records
that the inference scripts can consume.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_FILES = {
    "advbench": REPO_ROOT / "data" / "advbench.txt",
    "harmbench": REPO_ROOT / "data" / "harmbench_behaviors_text_test.csv",
    "sorrybench": REPO_ROOT / "data" / "sorry_bench" / "question.jsonl",
}

BENCHMARK_ALIASES = {
    "advbench": "advbench",
    "adv": "advbench",
    "harmbench": "harmbench",
    "harm": "harmbench",
    "sorrybench": "sorrybench",
    "sorry": "sorrybench",
}


@dataclass(frozen=True)
class PromptRecord:
    prompt: str
    metadata: dict[str, Any]


def normalize_benchmark_name(name: str) -> str:
    try:
        return BENCHMARK_ALIASES[name.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(sorted({"advbench", "harmbench", "sorrybench"}))
        raise ValueError(f"Unsupported benchmark {name!r}; choose from {choices}.") from exc


def _default_path(benchmark: str, mutation: Optional[str]) -> Path:
    path = DEFAULT_BENCHMARK_FILES[benchmark]
    if benchmark == "sorrybench" and mutation:
        safe_mutation = mutation.strip()
        if not safe_mutation or Path(safe_mutation).name != safe_mutation:
            raise ValueError("SORRY-Bench mutation must be a simple file suffix.")
        path = path.with_name(f"question_{safe_mutation}.jsonl")
    return path


def resolve_benchmark_path(
    benchmark: str,
    path: Optional[str | Path] = None,
    mutation: Optional[str] = None,
) -> Path:
    benchmark = normalize_benchmark_name(benchmark)
    resolved = Path(path).expanduser() if path else _default_path(benchmark, mutation)
    if resolved.exists():
        return resolved

    if benchmark == "harmbench":
        hint = (
            "Download harmbench_behaviors_text_test.csv or "
            "harmbench_behaviors_text_all.csv from "
            "https://github.com/centerforaisafety/HarmBench/tree/main/data/behavior_datasets."
        )
    elif benchmark == "sorrybench":
        hint = (
            "After accepting the SORRY-Bench dataset terms, download question.jsonl "
            "into a local directory from https://huggingface.co/datasets/sorry-bench/sorry-bench-202503."
        )
    else:
        hint = "The repository's data/advbench.txt file is missing."
    raise FileNotFoundError(f"{benchmark} source file not found: {resolved}. {hint}")


def _metadata(benchmark: str, index: int, **values: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "benchmark": benchmark,
        "benchmark_index": index,
    }
    result.update({key: value for key, value in values.items() if value not in (None, "")})
    return result


def _load_text(path: Path, benchmark: str) -> list[PromptRecord]:
    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        prompt = line.strip()
        if prompt:
            records.append(PromptRecord(prompt, _metadata(benchmark, index)))
    return records


def _first_csv_value(row: dict[str, Any], *names: str) -> Optional[str]:
    """Return the first non-empty value among case-sensitive CSV aliases."""

    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _load_advbench(path: Path) -> list[PromptRecord]:
    if path.suffix.lower() != ".csv":
        return _load_text(path, "advbench")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "goal" not in (rows[0].keys() if rows else ()):
        raise ValueError(f"AdvBench CSV must contain a goal column: {path}")

    records = []
    for index, row in enumerate(rows):
        prompt = _first_csv_value(row, "goal")
        if prompt is None:
            raise ValueError(f"AdvBench row {index} has an empty goal field: {path}")
        records.append(
            PromptRecord(
                prompt,
                _metadata(
                    "advbench",
                    index,
                    benchmark_id=_first_csv_value(row, "id"),
                    benchmark_target=_first_csv_value(row, "target"),
                ),
            )
        )
    return records


def _load_harmbench(path: Path) -> list[PromptRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not any(
        name in rows[0] for name in ("Behavior", "behavior", "goal")
    ):
        raise ValueError(
            f"HarmBench CSV must contain Behavior, behavior, or goal: {path}"
        )

    records = []
    for index, row in enumerate(rows):
        behavior = _first_csv_value(row, "Behavior", "behavior")
        goal = _first_csv_value(row, "goal")
        if behavior is None and goal is None:
            raise ValueError(f"HarmBench row {index} has an empty behavior field: {path}")
        context = _first_csv_value(row, "ContextString", "context_string") or ""
        # This is the same direct-request composition used by HarmBench for
        # contextual behaviors: context first, then the requested behavior.
        prompt = goal or (f"{context}\n\n---\n\n{behavior}" if context else behavior)
        tags_text = _first_csv_value(row, "Tags", "tags") or ""
        tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
        records.append(
            PromptRecord(
                prompt,
                _metadata(
                    "harmbench",
                    index,
                    benchmark_id=_first_csv_value(
                        row, "BehaviorID", "behavior_id", "id"
                    ),
                    benchmark_behavior=behavior,
                    benchmark_category=_first_csv_value(
                        row, "SemanticCategory", "semantic_category"
                    ),
                    benchmark_functional_category=_first_csv_value(
                        row, "FunctionalCategory", "functional_category"
                    ),
                    benchmark_tags=tags,
                    benchmark_contextual=bool(context),
                    benchmark_context=context,
                    benchmark_split=_first_csv_value(row, "source_split"),
                ),
            )
        )
    return records


def _first_string(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _sorrybench_record(row: dict[str, Any], index: int, path: Path) -> PromptRecord:
    prompt = (
        _first_string(row.get("goal"))
        or _first_string(row.get("prompt"))
        or _first_string(row.get("question"))
    )
    turns = row.get("turns")
    if prompt is None and isinstance(turns, list) and turns:
        prompt = _first_string(turns[0])
    if prompt is None:
        raise ValueError(
            f"SORRY-Bench row {index} has no goal/prompt/question/turns[0] text: {path}"
        )

    metadata_values = {
        "benchmark_id": row.get("question_id") or row.get("id"),
        "benchmark_category": row.get("category") or row.get("safety_category"),
        "benchmark_category_name": row.get("category_name"),
        "benchmark_subcategory": row.get("subcategory"),
        "benchmark_prompt_style": row.get("prompt_style"),
    }
    return PromptRecord(prompt, _metadata("sorrybench", index, **metadata_values))


def _load_sorrybench(path: Path) -> list[PromptRecord]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"SORRY-Bench CSV contains no rows: {path}")
        return [_sorrybench_record(row, index, path) for index, row in enumerate(rows)]

    records = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"SORRY-Bench row {index} must be a JSON object: {path}")

        records.append(_sorrybench_record(row, index, path))
    return records


def load_benchmark_records(
    benchmark: str,
    path: Optional[str | Path] = None,
    mutation: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[PromptRecord]:
    benchmark = normalize_benchmark_name(benchmark)
    if limit is not None and limit <= 0:
        raise ValueError("Benchmark limit must be positive when provided.")
    source = resolve_benchmark_path(benchmark, path, mutation)
    if benchmark == "harmbench":
        records = _load_harmbench(source)
    elif benchmark == "sorrybench":
        records = _load_sorrybench(source)
    else:
        records = _load_advbench(source)
    return records[:limit] if limit is not None else records


def collect_prompt_records(
    *,
    prompt: Optional[str] = None,
    prompts: Optional[Iterable[str]] = None,
    prompt_file: Optional[str | Path] = None,
    benchmark: Optional[str] = None,
    benchmark_file: Optional[str | Path] = None,
    benchmark_mutation: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[PromptRecord]:
    explicit_sources = bool(prompt or prompts or prompt_file)
    if benchmark:
        if explicit_sources:
            raise ValueError("Use --benchmark or prompt arguments, not both.")
        if benchmark_file and benchmark_mutation:
            raise ValueError("Use --benchmark_file or --benchmark_mutation, not both.")
        return load_benchmark_records(benchmark, benchmark_file, benchmark_mutation, limit)
    if benchmark_file or benchmark_mutation:
        raise ValueError("--benchmark_file and --benchmark_mutation require --benchmark.")

    values: list[str] = []
    if prompt:
        values.append(prompt)
    if prompts:
        values.extend(value for value in prompts if value)
    if prompt_file:
        values.extend(
            line.strip()
            for line in Path(prompt_file).expanduser().read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not values:
        raise ValueError(
            "No prompts were provided. Use --prompt, --prompts, --prompt_file, or --benchmark."
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("Prompt limit must be positive when provided.")
        values = values[:limit]
    return [PromptRecord(value, {}) for value in values]
