"""Strictly audit a handoff-gate generation artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence


EXPECTED_THRESHOLD = 0.9641336778984433
EXPECTED_GATE_CONFIG_SHA256 = "50dba7a1057edc429d5a6216e7d15c04aee505101f6ee8496af54bc77e4307b3"
EXPECTED_GATE_HEAD_SHA256 = "f8a78263c8ce452ea804fe2e11e7f4a17698f97a8166764d6e1ec0b6088ae463"
EXPECTED_TOKENIZER_SHA256 = "d1e5f232b1bbb7cb8b86443cd6ec1a0a60343f10d44121752a75a398a5b669f5"
FUSION_MODES = ("proxy", "raw")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--proxy-snapshot", required=True)
    parser.add_argument("--proxy-revision", required=True)
    parser.add_argument("--fusion-mode", choices=FUSION_MODES, default="proxy")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Generation JSONL contains a non-object row.")
    return rows


def require_equal(actual: Any, expected: Any, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: {actual!r} != {expected!r}")


def require_same_path(actual: Any, expected: Path, context: str) -> None:
    if not isinstance(actual, str):
        raise ValueError(f"{context}: expected a path string, got {actual!r}")
    actual_path = Path(actual).resolve()
    if actual_path != expected:
        raise ValueError(f"{context}: {actual_path!s} != {expected!s}")


def require_checkpoint_fusion_contract(
    checkpoint_config: dict[str, Any], fusion_mode: str
) -> None:
    """Validate the checkpoint fields that distinguish proxy fusion from raw MC."""
    if fusion_mode == "raw":
        require_equal(
            checkpoint_config.get("mc_fusion_mode"),
            None,
            "checkpoint mc_fusion_mode",
        )
        non_null_proxy_fields = {
            key: value
            for key, value in checkpoint_config.items()
            if key.startswith("proxy_") and value is not None
        }
        if non_null_proxy_fields:
            raise ValueError(
                "Raw-MC checkpoint has non-null proxy fields: "
                f"{non_null_proxy_fields!r}"
            )
        return
    if fusion_mode != "proxy":
        raise ValueError(f"Unsupported fusion mode: {fusion_mode!r}")
    for key, expected in {
        "mc_fusion_mode": "proxy_dirichlet_v1",
        "proxy_temperature": 2.5,
        "proxy_prior_strength": 8.0,
        "proxy_tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
    }.items():
        require_equal(checkpoint_config.get(key), expected, f"checkpoint {key}")


def require_runtime_fusion_contract(
    configuration: dict[str, Any],
    fusion_mode: str,
    proxy_snapshot: Path,
    proxy_revision: str,
    context: str,
) -> None:
    """Validate the per-generation fusion configuration."""
    proxy = configuration.get("proxy_fusion")
    if fusion_mode == "raw":
        require_equal(proxy, None, f"{context} proxy_fusion")
        return
    if fusion_mode != "proxy":
        raise ValueError(f"Unsupported fusion mode: {fusion_mode!r}")
    if not isinstance(proxy, dict):
        raise ValueError(f"{context} has no proxy fusion audit.")
    for key, expected in {
        "mc_fusion_mode": "proxy_dirichlet_v1",
        "proxy_model_name_or_path": str(proxy_snapshot),
        "proxy_model_revision": proxy_revision,
        "proxy_tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
        "proxy_temperature": 2.5,
        "proxy_prior_strength": 8.0,
        "proxy_dtype": "float16",
        "proxy_quantization": "none",
    }.items():
        require_equal(proxy.get(key), expected, f"{context} proxy {key}")


def audited_proxy_calls(
    summary: dict[str, Any], fusion_mode: str, mc_steps: int, context: str
) -> int:
    """Validate proxy-call accounting and normalize raw-MC traces to zero calls."""
    if fusion_mode == "raw":
        proxy_calls = summary.get("proxy_calls", 0)
        if proxy_calls != 0:
            raise ValueError(f"{context} raw MC trace has non-zero proxy calls.")
        return 0
    if fusion_mode != "proxy":
        raise ValueError(f"Unsupported fusion mode: {fusion_mode!r}")
    proxy_calls = int(summary.get("proxy_calls", -1))
    if proxy_calls != mc_steps:
        raise ValueError(f"{context} does not have one proxy call per MC step.")
    return proxy_calls


def audit_fusion_metadata(fusion_mode: str) -> dict[str, Any]:
    """Return the top-level audit fields without changing the proxy schema."""
    if fusion_mode == "raw":
        return {"fusion_mode": "raw_mc50", "proxy": None}
    if fusion_mode == "proxy":
        return {"proxy": {"temperature": 2.5, "prior_strength": 8.0}}
    raise ValueError(f"Unsupported fusion mode: {fusion_mode!r}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.begin < 0 or args.end <= args.begin:
        raise ValueError("--begin/--end must be nonnegative with end > begin")
    generation = Path(args.generation).resolve()
    audit_path = Path(args.audit).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    gate = Path(args.gate).resolve()
    benchmark = Path(args.benchmark).resolve()
    proxy_snapshot = Path(args.proxy_snapshot).resolve()
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {audit_path}")
    for path in (
        generation,
        checkpoint / "config.json",
        checkpoint / "pytorch_model.bin",
        checkpoint / "training_metrics.json",
        checkpoint / "training_provenance.json",
        gate / "risk_head_config.json",
        gate / "risk_head.pt",
        benchmark,
        proxy_snapshot / "config.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    require_equal(sha256_file(gate / "risk_head_config.json"), EXPECTED_GATE_CONFIG_SHA256, "gate config hash")
    require_equal(sha256_file(gate / "risk_head.pt"), EXPECTED_GATE_HEAD_SHA256, "gate head hash")

    checkpoint_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    common_checkpoint_contract = {
        "hidden_size": 1024,
        "vocab_size": 151669,
        "mc_input_representation": "floor_logprob",
        "mc_base_score_representation": "floor_logprob",
        "input_projection_mode": "count_sketch",
        "input_hidden_normalization": "layernorm",
        "count_sketch_hashes": 4,
        "count_sketch_seed": 42,
        "risk_gate_training": "hard",
        "always_train_first_n": 3,
    }
    for key, expected in common_checkpoint_contract.items():
        require_equal(checkpoint_config.get(key), expected, f"checkpoint {key}")
    require_checkpoint_fusion_contract(checkpoint_config, args.fusion_mode)
    held_out = checkpoint_config.get("held_out_files")
    if not isinstance(held_out, list) or len(held_out) != 8:
        raise ValueError("Checkpoint must record the controlled eight-file held-out split.")

    rows = load_jsonl(generation)
    with benchmark.open(encoding="utf-8-sig", newline="") as handle:
        benchmark_rows = list(csv.DictReader(handle))
    expected_indices = list(range(args.begin, args.end))
    expected_prompts = [row["goal"] for row in benchmark_rows[args.begin : args.end]]
    if len(expected_prompts) != len(expected_indices):
        raise ValueError("Requested AdvBench range extends past the benchmark.")
    if len(rows) != len(expected_indices) or [row.get("prompt") for row in rows] != expected_prompts:
        raise ValueError(
            f"Output is not exactly AdvBench rows {args.begin}..{args.end - 1}."
        )

    total_api_calls = 0
    total_api_cost = 0.0
    per_prompt: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        benchmark_index = args.begin + index
        require_equal(row.get("benchmark"), "advbench", f"row {index} benchmark")
        require_equal(
            row.get("benchmark_index"), benchmark_index, f"row {index} benchmark index"
        )
        if not isinstance(row.get("completion"), str):
            raise ValueError(f"Row {index} completion is not text.")
        runtime = row.get("risk_gate_runtime")
        routing = row.get("openrouter_routing")
        if not isinstance(runtime, dict) or not isinstance(routing, dict):
            raise ValueError(f"Row {index} lacks runtime/routing audit.")
        configuration = runtime.get("configuration")
        summary = runtime.get("summary")
        steps = runtime.get("steps")
        handoff = runtime.get("handoff")
        if not all(isinstance(value, dict) for value in (configuration, summary, handoff)):
            raise ValueError(f"Row {index} runtime audit is incomplete.")
        if not isinstance(steps, list):
            raise ValueError(f"Row {index} has no token trace.")
        expected_runtime = {
            "requested_model": "qwen/qwen3-32b",
            "tokenizer_name": str(proxy_snapshot),
            "mc_input_representation": "floor_logprob",
            "mc_base_score_representation": "floor_logprob",
            "mc_score_interface": "shared_v1",
            "mc_log_count_alpha": None,
            "mc_samples_per_token": 50,
            "mc_independent_views": 1,
            "mc_sample_temperature": 1.0,
            "mc_top_p": 1.0,
            "mc_observed_alpha": 0.1,
            "mc_floor_mass": 1e-4,
            "mc_store_dtype": "float16",
            "mc_completion_policy": "exact",
            "risk_gate_prompt_source": "dataset",
            "mode": "hard",
            "threshold": EXPECTED_THRESHOLD,
            "warmup_tokens": 3,
            "latch_off": True,
            "latch_patience": 1,
            "latch_threshold": EXPECTED_THRESHOLD,
            "handoff_on_latch": True,
            "speculative_draft": False,
            "reasoning_mode": "enabled_false",
        }
        for key, expected in expected_runtime.items():
            require_equal(configuration.get(key), expected, f"row {index} runtime {key}")
        require_same_path(
            configuration.get("biasnet_checkpoint"),
            checkpoint,
            f"row {index} runtime biasnet_checkpoint",
        )
        require_same_path(
            configuration.get("risk_gate_checkpoint"),
            gate,
            f"row {index} runtime risk_gate_checkpoint",
        )
        require_runtime_fusion_contract(
            configuration,
            args.fusion_mode,
            proxy_snapshot,
            args.proxy_revision,
            f"Row {index}",
        )
        for key, expected in {
            "requested_model": "qwen/qwen3-32b",
            "requested_provider_order": ["DeepInfra"],
            "provider_allow_fallbacks": False,
            "router_metadata_requested": True,
            "openrouter_response_cache_disabled": True,
            "qwen_hard_no_think_prefill": True,
            "mc_samples_per_token": 50,
            "mc_independent_views": 1,
            "mc_input_representation": "floor_logprob",
            "mc_base_score_representation": "floor_logprob",
            "mc_completion_policy": "exact",
            "reasoning_mode": "enabled_false",
            "reject_reasoning_tokens": True,
        }.items():
            require_equal(routing.get(key), expected, f"row {index} routing {key}")
        providers = routing.get("actual_provider_call_counts")
        models = routing.get("actual_response_model_call_counts")
        if not isinstance(providers, dict) or set(providers) != {"DeepInfra"}:
            raise ValueError(f"Row {index} was not served exclusively by DeepInfra.")
        if not isinstance(models, dict) or set(models) != {"qwen/qwen3-32b"}:
            raise ValueError(f"Row {index} response model changed.")
        api_calls = routing.get("api_calls")
        api_cost = routing.get("api_cost")
        if not isinstance(api_calls, int) or api_calls <= 0:
            raise ValueError(f"Row {index} has invalid API call count.")
        if not isinstance(api_cost, (int, float)) or not math.isfinite(api_cost) or api_cost < 0:
            raise ValueError(f"Row {index} has invalid API cost.")
        if summary.get("controlled_token_steps", 0) > args.max_new_tokens:
            raise ValueError(f"Row {index} exceeded the token budget.")
        controlled_token_steps = int(summary.get("controlled_token_steps", -1))
        mc_steps = int(summary.get("mc_steps", -1))
        base_only_steps = int(summary.get("base_only_steps", -1))
        if controlled_token_steps != len(steps):
            raise ValueError(f"Row {index} token trace length disagrees with its summary.")
        if mc_steps < 0 or int(summary.get("mc_requested_samples", -1)) != 50 * mc_steps:
            raise ValueError(f"Row {index} does not have an exact MC50 trace.")
        proxy_calls = audited_proxy_calls(
            summary, args.fusion_mode, mc_steps, f"Row {index}"
        )
        if base_only_steps != controlled_token_steps - mc_steps:
            raise ValueError(f"Row {index} base/MC step counts do not partition the trace.")
        handoff_occurred = bool(summary.get("handoff_occurred"))
        if handoff_occurred != bool(handoff.get("occurred")):
            raise ValueError(f"Row {index} handoff summary disagrees with its audit.")
        total_api_calls += api_calls
        total_api_cost += float(api_cost)
        per_prompt.append(
            {
                "benchmark_index": benchmark_index,
                "completion_sha256": hashlib.sha256(row["completion"].encode("utf-8")).hexdigest(),
                "controlled_token_steps": controlled_token_steps,
                "mc_steps": mc_steps,
                "proxy_calls": proxy_calls,
                "base_only_steps": base_only_steps,
                "latch_step": summary.get("latch_step"),
                "handoff_occurred": handoff_occurred,
                "api_calls": api_calls,
                "api_cost": float(api_cost),
            }
        )

    audit = {
        "schema_version": 1,
        "status": "passed",
        "output_jsonl": str(generation),
        "output_sha256": sha256_file(generation),
        "checkpoint": str(checkpoint),
        "checkpoint_weights_sha256": sha256_file(checkpoint / "pytorch_model.bin"),
        "gate_checkpoint": str(gate),
        "gate_config_sha256": EXPECTED_GATE_CONFIG_SHA256,
        "gate_head_sha256": EXPECTED_GATE_HEAD_SHA256,
        "benchmark_file": str(benchmark),
        "benchmark_sha256": sha256_file(benchmark),
        "benchmark_indices": expected_indices,
        "target_model": "qwen/qwen3-32b",
        "provider": "DeepInfra",
        "max_new_tokens": args.max_new_tokens,
        "speculative_decoding": False,
        "risk_gate": {
            "mode": "hard",
            "threshold": EXPECTED_THRESHOLD,
            "warmup_tokens": 3,
            "latch_patience": 1,
            "handoff_on_latch": True,
            "prompt_source": "dataset",
        },
        "total_api_calls": total_api_calls,
        "total_api_cost": total_api_cost,
        "totals": {
            "controlled_token_steps": sum(row["controlled_token_steps"] for row in per_prompt),
            "mc_steps": sum(row["mc_steps"] for row in per_prompt),
            "proxy_calls": sum(row["proxy_calls"] for row in per_prompt),
            "base_only_steps": sum(row["base_only_steps"] for row in per_prompt),
            "handoff_count": sum(bool(row["handoff_occurred"]) for row in per_prompt),
            "api_calls": total_api_calls,
            "api_cost": total_api_cost,
        },
        "per_prompt": per_prompt,
    }
    audit.update(audit_fusion_metadata(args.fusion_mode))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_path.with_name(f".{audit_path.name}.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, audit_path)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
