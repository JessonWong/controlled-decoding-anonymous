"""Route FlipAttack JSONL outputs to this repository's benchmark evaluators."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate FlipAttack generations with benchmark-appropriate metrics."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=("advbench", "harmbench", "sorrybench"),
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--metric",
        choices=("auto", "harmful-score"),
        default="auto",
        help=(
            "auto uses harmful-score for AdvBench, the HarmBench classifier for "
            "HarmBench, and binary compliance for SORRY-Bench."
        ),
    )

    parser.add_argument("--judge-provider", choices=("openai", "gemini"), default="openai")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-max-output-tokens", type=int, default=4096)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-seed", type=int, default=1234)
    parser.add_argument("--gemini-safety-threshold", default="BLOCK_NONE")
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--max-judge-attempts", type=int, default=5)

    parser.add_argument(
        "--harmbench-behaviors-file",
        default=str(REPO_ROOT / "flip_data" / "harmbench.csv"),
    )
    parser.add_argument("--harmbench-classifier", default="cais/HarmBench-Llama-2-13b-cls")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )

    parser.add_argument(
        "--sorrybench-judge-file",
        default=None,
        help=(
            "Optional judge_prompts.jsonl override. The official base-scoreonly "
            "and base-#thescore prompts are built in."
        ),
    )
    parser.add_argument("--sorrybench-judge-prompt", default="base-scoreonly")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate input and print the delegated command."
    )
    return parser


def validate_generation_file(path: Path, benchmark: str, limit: Optional[int]) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Generation file not found: {path}")
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object.")
            if row.get("benchmark") not in (None, benchmark):
                raise ValueError(
                    f"{path}:{line_number} has benchmark={row.get('benchmark')!r}, "
                    f"expected {benchmark!r}."
                )
            if not isinstance(row.get("prompt"), str):
                raise ValueError(f"{path}:{line_number} has no string prompt.")
            if not isinstance(row.get("completion"), str):
                raise ValueError(
                    f"{path}:{line_number} has no string completion; materialized-only "
                    "attack files cannot be evaluated."
                )
            count += 1
            if limit is not None and count >= limit:
                break
    if count == 0:
        raise ValueError(f"No generation rows found in {path}.")
    return count


def _append_optional(command: list[str], flag: str, value) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def _judge_arguments(args: argparse.Namespace) -> list[str]:
    command = [
        "--judge_provider",
        args.judge_provider,
        "--judge_max_output_tokens",
        str(args.judge_max_output_tokens),
        "--judge_temperature",
        str(args.judge_temperature),
        "--judge_seed",
        str(args.judge_seed),
        "--gemini_safety_threshold",
        args.gemini_safety_threshold,
    ]
    _append_optional(command, "--judge_model", args.judge_model)
    _append_optional(command, "--judge_api_key", args.judge_api_key)
    return command


def build_evaluation_command(args: argparse.Namespace) -> list[str]:
    input_path = str(Path(args.input_file).expanduser())
    metric = "harmful-score" if args.metric == "harmful-score" else args.benchmark

    if metric in {"harmful-score", "advbench"}:
        command = [
            sys.executable,
            str(REPO_ROOT / "test" / "eval_harmful_score.py"),
            "--input_file",
            input_path,
            "--retry_sleep",
            str(args.retry_sleep),
            "--max_judge_attempts",
            str(args.max_judge_attempts),
        ]
        command.extend(_judge_arguments(args))
    elif metric == "harmbench":
        command = [
            sys.executable,
            str(REPO_ROOT / "test" / "eval_harmbench.py"),
            "--input_file",
            input_path,
            "--behaviors_file",
            str(Path(args.harmbench_behaviors_file).expanduser()),
            "--classifier",
            args.harmbench_classifier,
            "--batch_size",
            str(args.batch_size),
            "--dtype",
            args.dtype,
        ]
        _append_optional(command, "--device", args.device)
    else:
        command = [
            sys.executable,
            str(REPO_ROOT / "test" / "eval_sorrybench.py"),
            "--input_file",
            input_path,
            "--judge_prompt",
            args.sorrybench_judge_prompt,
            "--retry_sleep",
            str(args.retry_sleep),
            "--max_judge_attempts",
            str(args.max_judge_attempts),
        ]
        if args.sorrybench_judge_file:
            judge_file = Path(args.sorrybench_judge_file).expanduser()
            if not judge_file.is_file():
                raise FileNotFoundError(
                    f"SORRY-Bench judge prompt file not found: {judge_file}"
                )
            command.extend(("--judge_file", str(judge_file)))
        command.extend(_judge_arguments(args))

    _append_optional(command, "--output_file", args.output_file)
    _append_optional(command, "--limit", args.limit)
    return command


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    validate_generation_file(Path(args.input_file).expanduser(), args.benchmark, args.limit)
    command = build_evaluation_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    completed = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
