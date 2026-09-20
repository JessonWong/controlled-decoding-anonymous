"""Build a development-only base-vs-controller action plan from generation traces.

The source traces must come from a run without proxy logits.  This planner only
uses early, forced-warmup BiasNet steps, so the old learned handoff gate cannot
have selected the action.  Every planned controller action must also be known
to lie on the target's sampled support according to the source trace audit.

No target or judge requests are made by this module.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from transformers import AutoTokenizer


SCHEMA_VERSION = 1
PLAN_PROTOCOL = "target_only_base_vs_controller_action_v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}.")
        rows.append(row)
    if not rows:
        raise ValueError(f"Generation trace is empty: {path}")
    return rows


def _contains_proxy_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            "proxy" in str(key).casefold() or _contains_proxy_field(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_proxy_field(child) for child in value)
    return False


def _priority(seed: str, *parts: Any) -> str:
    material = ":".join([seed, *(str(part) for part in parts)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _validate_source_record(row: dict[str, Any], *, source: Path) -> dict[str, Any]:
    runtime = row.get("risk_gate_runtime")
    if not isinstance(runtime, dict):
        raise ValueError(f"{source} lacks a risk_gate_runtime trace.")
    configuration = runtime.get("configuration")
    summary = runtime.get("summary")
    steps = runtime.get("steps")
    if not isinstance(configuration, dict) or not isinstance(summary, dict):
        raise ValueError(f"{source} has an incomplete runtime audit.")
    if not isinstance(steps, list):
        raise ValueError(f"{source} has no per-step trace.")
    if _contains_proxy_field(row):
        raise ValueError(f"Proxy-related fields are present in source trace {source}.")
    prior = configuration.get("mc_static_prior")
    if not isinstance(prior, dict) or prior.get("mode") != "uniform_dirichlet_v1":
        raise ValueError("Target-only pilot requires the uniform Dirichlet source arm.")
    if int(summary.get("off_support_steps", -1)) != 0:
        raise ValueError("Source record has off-support controller actions.")
    if int(configuration.get("warmup_tokens", -1)) <= 0:
        raise ValueError("Source run did not audit a positive forced-warmup window.")
    return runtime


def build_plan(
    rows: Sequence[dict[str, Any]],
    tokenizer,
    *,
    source_path: Path,
    max_step: int,
    total_answer_tokens: int,
    max_pairs_per_prompt: int,
    validation_prompts: int,
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if max_step <= 0 or total_answer_tokens <= max_step:
        raise ValueError("max_step must be positive and below total_answer_tokens.")
    if max_pairs_per_prompt <= 0:
        raise ValueError("max_pairs_per_prompt must be positive.")

    candidates: list[dict[str, Any]] = []
    prompt_keys: set[str] = set()
    skip_counts: Counter[str] = Counter()
    special_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())
    for row in rows:
        runtime = _validate_source_record(row, source=source_path)
        prompt = row.get("prompt")
        benchmark_index = row.get("benchmark_index")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Every source record requires a non-empty prompt.")
        prompt_key = str(row.get("benchmark_id", benchmark_index))
        prompt_keys.add(prompt_key)
        configuration = runtime["configuration"]
        initial_prefix = str(configuration.get("initial_prefix") or "")
        generated_ids = [
            int(token_id)
            for token_id in tokenizer.encode(initial_prefix, add_special_tokens=False)
        ]
        expected_step = 1
        for step in runtime["steps"]:
            step_number = int(step.get("step", -1))
            if step_number != expected_step:
                raise ValueError(
                    f"Non-contiguous trace for prompt {prompt_key}: "
                    f"expected step {expected_step}, found {step_number}."
                )
            expected_step += 1
            final_token_id = int(step["final_token_id"])
            base_token_id = step.get("base_token_id")
            if (
                step_number <= max_step
                and bool(step.get("used_biasnet"))
                and bool(step.get("forced_warmup"))
                and base_token_id is not None
                and int(base_token_id) != final_token_id
            ):
                base_token_id = int(base_token_id)
                if base_token_id in special_ids or final_token_id in special_ids:
                    skip_counts["special_action"] += 1
                else:
                    prefix_ids = list(generated_ids)
                    prefix_text = tokenizer.decode(
                        prefix_ids, clean_up_tokenization_spaces=False
                    )
                    base_prefix = tokenizer.decode(
                        prefix_ids + [base_token_id],
                        clean_up_tokenization_spaces=False,
                    )
                    controller_prefix = tokenizer.decode(
                        prefix_ids + [final_token_id],
                        clean_up_tokenization_spaces=False,
                    )
                    base_roundtrip = tokenizer.encode(
                        base_prefix, add_special_tokens=False
                    ) == prefix_ids + [base_token_id]
                    controller_roundtrip = tokenizer.encode(
                        controller_prefix, add_special_tokens=False
                    ) == prefix_ids + [final_token_id]
                    if not base_roundtrip or not controller_roundtrip:
                        skip_counts["forced_prefix_roundtrip"] += 1
                    else:
                        identity = {
                            "benchmark": row.get("benchmark"),
                            "benchmark_index": benchmark_index,
                            "benchmark_id": row.get("benchmark_id"),
                            "step": step_number,
                            "base_action_token_id": base_token_id,
                            "controller_action_token_id": final_token_id,
                            "source_sha256": sha256_payload(row),
                        }
                        pair_id = "toa-" + sha256_payload(identity)[:24]
                        candidates.append(
                            {
                                "plan_schema_version": SCHEMA_VERSION,
                                "plan_protocol": PLAN_PROTOCOL,
                                "pair_id": pair_id,
                                "record_name": f"{row.get('benchmark', 'unknown')}:{prompt_key}",
                                "prompt_group": prompt_key,
                                "prompt": prompt,
                                "api_prompt": prompt + "\n/no_think",
                                "benchmark": row.get("benchmark"),
                                "benchmark_index": benchmark_index,
                                "benchmark_id": row.get("benchmark_id"),
                                "position": step_number - 1,
                                "step": step_number,
                                "position_fraction": float(step_number - 1)
                                / float(total_answer_tokens - 1),
                                "prefix_token_ids": prefix_ids,
                                "prefix_text": prefix_text,
                                "base_action_token_id": base_token_id,
                                "controller_action_token_id": final_token_id,
                                "base_forced_prefix": base_prefix,
                                "controller_forced_prefix": controller_prefix,
                                "remaining_tokens": total_answer_tokens - step_number,
                                "source_generation_path": str(source_path.resolve()),
                                "source_record_sha256": sha256_payload(row),
                                "source_runtime_configuration": {
                                    "mc_samples_per_token": configuration.get(
                                        "mc_samples_per_token"
                                    ),
                                    "mc_sample_temperature": configuration.get(
                                        "mc_sample_temperature"
                                    ),
                                    "mc_static_prior": configuration.get(
                                        "mc_static_prior"
                                    ),
                                    "warmup_tokens": configuration.get("warmup_tokens"),
                                    "qwen_hard_no_think_prefill": True,
                                    "append_no_think": True,
                                },
                                "controller_action_provenance": (
                                    "uniform_dirichlet_biasnet_forced_warmup_"
                                    "observed_support"
                                ),
                            }
                        )
            generated_ids.append(final_token_id)

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_prompt[candidate["prompt_group"]].append(candidate)
    if validation_prompts <= 0 or validation_prompts >= len(by_prompt):
        raise ValueError(
            "validation_prompts must leave at least one prompt in each split; "
            f"found {len(by_prompt)} eligible prompts."
        )
    ordered_groups = sorted(by_prompt, key=lambda key: _priority(seed, "split", key))
    validation_groups = set(ordered_groups[:validation_prompts])

    selected: list[dict[str, Any]] = []
    for prompt_group, prompt_rows in by_prompt.items():
        ordered = sorted(
            prompt_rows,
            key=lambda row: _priority(seed, prompt_group, row["step"], row["pair_id"]),
        )
        for row in ordered[:max_pairs_per_prompt]:
            row = dict(row)
            row["split"] = (
                "legacy_validation" if prompt_group in validation_groups else "train"
            )
            row["plan_row_sha256"] = sha256_payload(row)
            selected.append(row)
    selected.sort(key=lambda row: (row["split"], row["record_name"], row["step"]))
    split_counts = Counter(row["split"] for row in selected)
    summary = {
        "plan_schema_version": SCHEMA_VERSION,
        "plan_protocol": PLAN_PROTOCOL,
        "analysis": "development_only_target_only_action_pair_plan_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_generation_path": str(source_path.resolve()),
        "source_generation_sha256": sha256_file(source_path),
        "source_records": len(rows),
        "source_proxy_field_audit": "no_proxy_named_fields",
        "eligible_prompts": len(by_prompt),
        "eligible_pairs": len(candidates),
        "selected_pairs": len(selected),
        "selected_split_counts": dict(sorted(split_counts.items())),
        "validation_prompt_groups": sorted(validation_groups),
        "max_step": max_step,
        "max_pairs_per_prompt": max_pairs_per_prompt,
        "total_answer_tokens": total_answer_tokens,
        "selection_seed": seed,
        "skipped": dict(sorted(skip_counts.items())),
        "limitations": [
            "The source AdvBench prompts are a repeatedly used development set.",
            "Only forced-warmup states are used so the source Llama handoff gate cannot select actions.",
            "The controller is uniform-Dirichlet BiasNet, not the not-yet-trained candidate reranker.",
            "A zero off-support count is available per source trajectory, not per individual step.",
        ],
    }
    return selected, summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--max-step", type=int, default=3)
    parser.add_argument("--total-answer-tokens", type=int, default=80)
    parser.add_argument("--max-pairs-per-prompt", type=int, default=2)
    parser.add_argument("--validation-prompts", type=int, default=10)
    parser.add_argument("--seed", default="target-only-action-pairs-v1")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.generation_jsonl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    plan, summary = build_plan(
        read_jsonl(source),
        tokenizer,
        source_path=source,
        max_step=args.max_step,
        total_answer_tokens=args.total_answer_tokens,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        validation_prompts=args.validation_prompts,
        seed=args.seed,
    )
    output_dir.mkdir(parents=True)
    _atomic_write_jsonl(output_dir / "action_pair_plan.jsonl", plan)
    summary["plan_sha256"] = sha256_file(output_dir / "action_pair_plan.jsonl")
    _atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
