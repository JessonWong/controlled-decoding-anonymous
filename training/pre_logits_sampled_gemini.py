"""Cache Gemini MC next-token distributions in a Gemma tokenizer space."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference_gemini_sampled import GeminiMCSampler  # noqa: E402
from pre_logits_sampled_openrouter import load_tokenizer  # noqa: E402
from pre_logits_sampled_openweight import sampled_ids_to_log_probs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache exact-MC Gemini distributions projected with a Gemma tokenizer."
    )
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--tokenizer_name", default="google/gemma-3-1b-pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default="LLM-LAT/harmful-dataset")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--start_index", type=int, default=100)
    parser.add_argument("--end_index", type=int, default=141)
    parser.add_argument("--max_samples", type=int, default=41)
    parser.add_argument("--max_answer_tokens", type=int, default=80)
    parser.add_argument("--mc_samples_per_token", type=int, default=50)
    parser.add_argument("--mc_sample_temperature", type=float, default=1.0)
    parser.add_argument("--mc_top_p", type=float, default=1.0)
    parser.add_argument("--mc_observed_alpha", type=float, default=0.1)
    parser.add_argument("--mc_floor_mass", type=float, default=1e-4)
    parser.add_argument("--candidate_count", type=int, default=1)
    parser.add_argument("--parallel_requests", type=int, default=8)
    parser.add_argument("--parallel_positions", type=int, default=1)
    parser.add_argument("--api_max_output_tokens", type=int, default=8)
    parser.add_argument(
        "--empty_response_token",
        choices=["skip", "stop_eos"],
        default="stop_eos",
    )
    parser.add_argument("--thinking_budget", type=int, default=0)
    parser.add_argument("--max_batch_calls", type=int, default=70)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY")
    parser.add_argument("--api_key_file", default=None)
    parser.add_argument("--store_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress_steps", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0 or args.end_index <= args.start_index:
        raise ValueError("Require 0 <= start_index < end_index.")
    for name in ("max_samples", "max_answer_tokens", "mc_samples_per_token"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive.")
    if args.candidate_count != 1 and args.model == "gemini-3.5-flash":
        raise ValueError("gemini-3.5-flash requires --candidate_count 1.")
    if args.max_batch_calls < args.mc_samples_per_token:
        raise ValueError("--max_batch_calls must permit exact MC completion.")
    if args.parallel_positions <= 0:
        raise ValueError("--parallel_positions must be positive.")


def cache_configuration(args: argparse.Namespace, tokenizer) -> dict:
    return {
        "schema_version": 1,
        "source": "sampled_gemini_gemma_projection",
        "model": args.model,
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "tokenizer_length": len(tokenizer),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "max_samples": args.max_samples,
        "max_answer_tokens": args.max_answer_tokens,
        "mc_samples_per_token": args.mc_samples_per_token,
        "mc_sample_temperature": args.mc_sample_temperature,
        "mc_top_p": args.mc_top_p,
        "mc_observed_alpha": args.mc_observed_alpha,
        "mc_floor_mass": args.mc_floor_mass,
        "candidate_count": args.candidate_count,
        "parallel_requests": args.parallel_requests,
        "parallel_positions": args.parallel_positions,
        "api_max_output_tokens": args.api_max_output_tokens,
        "empty_response_token": args.empty_response_token,
        "thinking_budget": args.thinking_budget,
        "store_dtype": args.store_dtype,
        "seed": args.seed,
        "projection_rule": "first Gemma text token from each Gemini continuation",
    }


def record_name(dataset_index: int, prompt: str, answer: str) -> str:
    digest = hashlib.sha256(
        f"{dataset_index}\0{prompt}\0{answer}".encode("utf-8")
    ).hexdigest()
    return f"{dataset_index:06d}_{digest[:24]}.pt"


def validate_existing(path: Path, configuration: dict, dataset_index: int) -> bool:
    if not path.is_file():
        return False
    payload = torch.load(path, map_location="cpu")
    log_probs = payload.get("log_probs")
    labels = payload.get("labels")
    metadata = payload.get("metadata") or {}
    if (
        not isinstance(log_probs, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
        or log_probs.dim() != 3
        or labels.dim() != 2
        or log_probs.shape[:2] != labels.shape
        or log_probs.shape[2] != configuration["tokenizer_vocab_size"]
        or metadata.get("configuration") != configuration
        or metadata.get("dataset_index") != dataset_index
        or int(metadata.get("valid_sample_min", -1))
        != configuration["mc_samples_per_token"]
    ):
        raise RuntimeError(f"Existing cache record is incompatible: {path}")
    return True


def save_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.tokenizer_name, use_fast=True)
    vocab_size = int(tokenizer.vocab_size)
    if len(tokenizer) < vocab_size:
        raise RuntimeError("Tokenizer length is smaller than its base vocabulary.")
    configuration = cache_configuration(args, tokenizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "cache_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != configuration:
            raise RuntimeError(f"Cache manifest mismatch: {manifest_path}")
    else:
        manifest_path.write_text(
            json.dumps(configuration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    dataset = load_dataset(args.dataset_name)[args.dataset_split]
    stop = min(args.end_index, len(dataset))
    indices = list(range(args.start_index, stop))[: args.max_samples]
    sampler = GeminiMCSampler(args, tokenizer)
    dtype = torch.float16 if args.store_dtype == "float16" else torch.float32
    completed = 0

    for ordinal, dataset_index in enumerate(indices, start=1):
        item = dataset[dataset_index]
        prompt = str(item["prompt"])
        answer = str(item["rejected"])
        path = output_dir / record_name(dataset_index, prompt, answer)
        if validate_existing(path, configuration, dataset_index):
            completed += 1
            print(f"reused={ordinal}/{len(indices)} dataset_index={dataset_index}", flush=True)
            continue

        answer_ids = tokenizer.encode(answer, add_special_tokens=False)[
            : args.max_answer_tokens
        ]
        if not answer_ids:
            raise RuntimeError(f"Empty Gemma target at dataset index {dataset_index}.")
        rows: list[torch.Tensor | None] = [None] * len(answer_ids)
        position_stats: list[dict | None] = [None] * len(answer_ids)
        started = time.time()

        def collect_position(position: int) -> tuple[int, torch.Tensor, dict]:
            prefix = tokenizer.decode(
                answer_ids[:position], skip_special_tokens=False
            )
            sampled_ids, stats = sampler.sample_ids(prompt, prefix)
            if len(sampled_ids) != args.mc_samples_per_token:
                raise RuntimeError(
                    f"dataset_index={dataset_index} position={position}: "
                    f"only {len(sampled_ids)}/{args.mc_samples_per_token} samples: {stats}"
                )
            sampled = torch.tensor([sampled_ids], dtype=torch.long)
            row = sampled_ids_to_log_probs(
                sampled_token_ids=sampled,
                vocab_size=vocab_size,
                observed_alpha=args.mc_observed_alpha,
                floor_mass=args.mc_floor_mass,
                dtype=dtype,
            )[0]
            return position, row, stats

        with ThreadPoolExecutor(
            max_workers=min(args.parallel_positions, len(answer_ids))
        ) as executor:
            futures = [
                executor.submit(collect_position, position)
                for position in range(len(answer_ids))
            ]
            completed_positions = 0
            for future in as_completed(futures):
                position, row, stats = future.result()
                rows[position] = row
                position_stats[position] = stats
                completed_positions += 1
                if args.progress_steps:
                    print(
                        f"dataset_index={dataset_index} completed_positions="
                        f"{completed_positions}/{len(answer_ids)} "
                        f"position={position + 1} valid={stats['valid']}",
                        flush=True,
                    )

        if any(row is None for row in rows) or any(
            stats is None for stats in position_stats
        ):
            raise RuntimeError(f"Incomplete position collection at {dataset_index}.")
        complete_rows = [row for row in rows if row is not None]
        complete_stats = [stats for stats in position_stats if stats is not None]
        valid_counts = [int(x["valid"]) for x in complete_stats]
        payload = {
            "log_probs": torch.stack(complete_rows, dim=0).unsqueeze(0).cpu(),
            "labels": torch.tensor(answer_ids, dtype=torch.long).unsqueeze(0),
            "metadata": {
                "configuration": configuration,
                "dataset_index": dataset_index,
                "prompt": prompt,
                "target_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
                "valid_sample_min": min(valid_counts),
                "valid_sample_max": max(valid_counts),
                "position_stats": complete_stats,
                "elapsed_sec": time.time() - started,
            },
        }
        save_atomic(payload, path)
        completed += 1
        print(
            f"saved={completed}/{len(indices)} dataset_index={dataset_index} "
            f"tokens={len(answer_ids)} elapsed_sec={payload['metadata']['elapsed_sec']:.1f} "
            f"path={path}",
            flush=True,
        )

    print(
        f"Finished Gemini cache. records={completed} calls={sampler.total_calls} "
        f"valid={sampler.total_valid} empty={sampler.total_empty} "
        f"multi_token_first_used={sampler.total_multi} "
        f"failed_calls={sampler.total_failed_calls}",
        flush=True,
    )


if __name__ == "__main__":
    main()
