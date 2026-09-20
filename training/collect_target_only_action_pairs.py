"""Collect target-only base-vs-controller continuation pairs.

The input is produced by ``plan_target_only_action_pairs.py``.  Each pair has
two distinct, locally audited target-token prefixes.  The collector performs
one deterministic Qwen/DeepInfra continuation request per arm and emits
complete two-arm groups for the harmfulness judge.

``--dry-run`` loads the pinned tokenizer and performs the full local preflight,
but does not read an API key or create any artifact.  A real run persists an
intent before every request and fails closed after an uncertain request; it
never silently retries potentially billed work.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from transformers import AutoTokenizer

from collect_outcome_regret_pairs import (
    OPENROUTER_URL,
    TARGET_MODEL,
    TARGET_PROVIDER,
    atomic_write_json,
    atomic_write_jsonl,
    build_client_args,
    client_delta,
    client_snapshot,
    generate_collector_continuation,
    sha256_file,
    sha256_payload,
)
from pre_logits_sampled_openrouter import OpenRouterClient, resolve_api_key


OUTPUT_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
PLAN_PROTOCOL = "target_only_base_vs_controller_action_v1"
COLLECTOR_PROTOCOL = "target_only_paired_continuation_t0_qwen_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}.")
        rows.append(value)
    if not rows:
        raise ValueError(f"Empty action-pair plan: {path}")
    return rows


def state_directory(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.name}.state")


def state_path(root: Path, arm_id: str) -> Path:
    return root / f"{hashlib.sha256(arm_id.encode('utf-8')).hexdigest()}.json"


def manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".manifest.json")


def audit_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".audit.json")


def lock_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".lock")


def _without_row_hash(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "plan_row_sha256"}


def _validate_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def prepare_plan(
    path: Path,
    tokenizer,
    *,
    max_pairs: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = read_jsonl(path)
    if max_pairs is not None:
        if max_pairs <= 0:
            raise ValueError("--max-pairs must be positive.")
        raw_rows = raw_rows[:max_pairs]

    prepared: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    seen_arm_serializations: set[str] = set()
    split_counts: Counter[str] = Counter()
    for index, row in enumerate(raw_rows):
        if row.get("plan_schema_version") != PLAN_SCHEMA_VERSION:
            raise ValueError(f"Plan row {index} has an unsupported schema version.")
        if row.get("plan_protocol") != PLAN_PROTOCOL:
            raise ValueError(f"Plan row {index} has an unsupported protocol.")
        expected_row_hash = row.get("plan_row_sha256")
        if expected_row_hash != sha256_payload(_without_row_hash(row)):
            raise ValueError(f"Plan row {index} has an invalid content hash.")

        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.startswith("toa-"):
            raise ValueError(f"Plan row {index} has an invalid pair_id.")
        if pair_id in seen_pairs:
            raise ValueError(f"Duplicate pair_id: {pair_id}")
        seen_pairs.add(pair_id)

        prompt = row.get("prompt")
        api_prompt = row.get("api_prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Plan row {index} has no prompt.")
        if api_prompt != prompt + "\n/no_think":
            raise ValueError(f"Plan row {index} violates the hard no-think prompt contract.")
        prefix_ids = row.get("prefix_token_ids")
        if not isinstance(prefix_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in prefix_ids
        ):
            raise ValueError(f"Plan row {index} has invalid prefix_token_ids.")
        if tokenizer.decode(
            prefix_ids, clean_up_tokenization_spaces=False
        ) != row.get("prefix_text"):
            raise ValueError(f"Plan row {index} prefix text/token IDs disagree.")

        step = _validate_integer(row.get("step"), name="step", minimum=1)
        position = _validate_integer(row.get("position"), name="position")
        remaining = _validate_integer(
            row.get("remaining_tokens"), name="remaining_tokens", minimum=1
        )
        if position != step - 1 or remaining != 80 - step:
            raise ValueError(f"Plan row {index} violates the fixed 80-token budget.")
        if row.get("controller_action_provenance") != (
            "uniform_dirichlet_biasnet_forced_warmup_observed_support"
        ):
            raise ValueError(f"Plan row {index} has unaudited controller provenance.")
        runtime = row.get("source_runtime_configuration")
        if not isinstance(runtime, dict) or (runtime.get("mc_static_prior") or {}).get(
            "mode"
        ) != "uniform_dirichlet_v1":
            raise ValueError(f"Plan row {index} is not from the target-only prior arm.")

        arms: list[dict[str, Any]] = []
        for order, role in enumerate(("base", "controller")):
            action = _validate_integer(
                row.get(f"{role}_action_token_id"),
                name=f"{role}_action_token_id",
            )
            if action >= len(tokenizer):
                raise ValueError(f"Plan row {index} {role} action is outside tokenizer.")
            if action in set(getattr(tokenizer, "all_special_ids", ()) or ()):
                raise ValueError(f"Plan row {index} {role} action is special.")
            forced_ids = [*prefix_ids, action]
            raw_prefix = tokenizer.decode(
                forced_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            visible_prefix = tokenizer.decode(
                forced_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if raw_prefix != row.get(f"{role}_forced_prefix"):
                raise ValueError(f"Plan row {index} {role} forced prefix disagrees.")
            retokenized = tokenizer.encode(raw_prefix, add_special_tokens=False)
            if retokenized != forced_ids:
                raise ValueError(f"Plan row {index} {role} forced prefix is unstable.")
            branch = {
                "api_prompt": api_prompt,
                "raw_forced_prefix": raw_prefix,
                "remaining_tokens": remaining,
                "model": TARGET_MODEL,
                "provider_order": [TARGET_PROVIDER],
                "provider_allow_fallbacks": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "reasoning_mode": "enabled_false",
                "qwen_hard_no_think_prefill": True,
                "disable_openrouter_response_cache": True,
            }
            branch_sha = sha256_payload(branch)
            arm_id = f"{pair_id}:{role}"
            if arm_id in seen_arm_serializations:
                raise ValueError(f"Duplicate arm identity: {arm_id}")
            seen_arm_serializations.add(arm_id)
            arms.append(
                {
                    "arm_id": arm_id,
                    "arm_role": role,
                    "completion_order": order,
                    "action_token_id": action,
                    "raw_forced_prefix": raw_prefix,
                    "visible_forced_prefix": visible_prefix,
                    "remaining_tokens": remaining,
                    "forced_token_ids": forced_ids,
                    "branch_serialization": branch,
                    "branch_serialization_sha256": branch_sha,
                }
            )
        if arms[0]["action_token_id"] == arms[1]["action_token_id"]:
            raise ValueError(f"Plan row {index} has identical action tokens.")
        if arms[0]["branch_serialization_sha256"] == arms[1]["branch_serialization_sha256"]:
            raise ValueError(f"Plan row {index} serializes to identical branch requests.")

        split = row.get("split")
        if split not in {"train", "legacy_validation"}:
            raise ValueError(f"Plan row {index} has invalid split {split!r}.")
        split_counts[str(split)] += 1
        prepared.append(
            {
                "plan_index": index,
                "pair_id": pair_id,
                "prompt": prompt,
                "api_prompt": api_prompt,
                "split": split,
                "plan_row": dict(row),
                "arms": arms,
            }
        )

    if not prepared:
        raise ValueError("No plan pairs selected.")
    return prepared, {
        "plan_path": str(path),
        "plan_sha256": sha256_file(path),
        "selected_pairs": len(prepared),
        "selected_arms": 2 * len(prepared),
        "planned_api_requests": 2 * len(prepared),
        "planned_requested_output_tokens": sum(
            arm["remaining_tokens"] for pair in prepared for arm in pair["arms"]
        ),
        "split_counts": dict(sorted(split_counts.items())),
        "all_prefix_roundtrips_exact": True,
        "all_branch_requests_distinct_within_pair": True,
    }


def materialize_output(
    output_path: Path,
    pairs: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
) -> int:
    rows: list[dict[str, Any]] = []
    complete = 0
    for pair in pairs:
        pair_states = [states.get(arm["arm_id"]) for arm in pair["arms"]]
        if all(state is not None and state.get("status") == "result" for state in pair_states):
            complete += 1
            rows.extend(dict(state["row"]) for state in pair_states if state is not None)
    atomic_write_jsonl(output_path, rows)
    return complete


def result_row(
    pair: Mapping[str, Any],
    arm: Mapping[str, Any],
    *,
    continuation: str,
    handoff_audit: Mapping[str, Any],
    request_audit: Mapping[str, Any],
    configuration_fingerprint: str,
) -> dict[str, Any]:
    plan_row = pair["plan_row"]
    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "collector_protocol": COLLECTOR_PROTOCOL,
        "configuration_fingerprint": configuration_fingerprint,
        "prompt": pair["prompt"],
        "completion": arm["visible_forced_prefix"] + continuation,
        "logibreak_group_id": pair["pair_id"],
        "completion_order": arm["completion_order"],
        "target_completions_in_group": 2,
        "pair_id": pair["pair_id"],
        "arm_id": arm["arm_id"],
        "arm_role": arm["arm_role"],
        "split": pair["split"],
        "record_name": plan_row["record_name"],
        "position": plan_row["position"],
        "position_fraction": plan_row["position_fraction"],
        "benchmark": plan_row.get("benchmark"),
        "benchmark_index": plan_row.get("benchmark_index"),
        "benchmark_id": plan_row.get("benchmark_id"),
        "action_token_id": arm["action_token_id"],
        "base_action_token_id": plan_row["base_action_token_id"],
        "controller_action_token_id": plan_row["controller_action_token_id"],
        "forced_prefix_text": arm["visible_forced_prefix"],
        "continuation": continuation,
        "remaining_tokens": arm["remaining_tokens"],
        "branch_serialization": arm["branch_serialization"],
        "branch_serialization_sha256": arm["branch_serialization_sha256"],
        "plan_row_sha256": plan_row["plan_row_sha256"],
        "target_only_features": {
            "position": plan_row["position"],
            "position_fraction": plan_row["position_fraction"],
            "prefix_text": plan_row["prefix_text"],
            "base_action_token_id": plan_row["base_action_token_id"],
            "controller_action_token_id": plan_row["controller_action_token_id"],
        },
        "target_protocol": {
            "model": TARGET_MODEL,
            "provider_order": [TARGET_PROVIDER],
            "provider_allow_fallbacks": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "reasoning_mode": "enabled_false",
            "reject_reasoning_tokens": True,
            "qwen_hard_no_think_prefill": True,
            "append_no_think": True,
            "disable_openrouter_response_cache": True,
            "one_request_continuation": True,
        },
        "handoff_audit": dict(handoff_audit),
        "request_audit": dict(request_audit),
    }


def summarize(
    pairs: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
    *,
    hard_cap: int,
) -> dict[str, Any]:
    result_states = [state for state in states.values() if state.get("status") == "result"]
    unresolved = [
        arm_id
        for arm_id, state in states.items()
        if state.get("status") in {"intent", "uncertain"}
    ]
    rows = [state["row"] for state in result_states]
    pair_counts = Counter(row["pair_id"] for row in rows)
    return {
        "collector_protocol": COLLECTOR_PROTOCOL,
        "updated_at_utc": utc_now(),
        "planned_pairs": len(pairs),
        "planned_arms": 2 * len(pairs),
        "completed_pairs": sum(count == 2 for count in pair_counts.values()),
        "completed_arms": len(result_states),
        "unresolved_arm_ids": sorted(unresolved),
        "api_request_attempts": sum(
            int((row.get("request_audit") or {}).get("api_request_attempts", 0))
            for row in rows
        ),
        "successful_http_responses": sum(
            int((row.get("request_audit") or {}).get("successful_http_responses", 0))
            for row in rows
        ),
        "api_cost": sum(
            float((row.get("request_audit") or {}).get("api_cost", 0.0)) for row in rows
        ),
        "api_total_tokens": sum(
            int((row.get("request_audit") or {}).get("total_tokens", 0)) for row in rows
        ),
        "max_api_request_attempts": hard_cap,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--tokenizer-name", required=True)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-api-request-attempts", type=int, default=None)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--api-key-file", default=None)
    parser.add_argument("--site-url", default="https://anonymous.invalid")
    parser.add_argument("--app-name", default="JULI target-only action-pair collector")
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        raise ValueError("--request-timeout must be finite and positive.")
    if not args.dry_run and (
        args.max_api_request_attempts is None or args.max_api_request_attempts <= 0
    ):
        raise ValueError("A positive hard request cap is required.")

    plan_path = Path(args.plan).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        local_files_only=args.local_files_only,
        trust_remote_code=False,
    )
    pairs, preflight = prepare_plan(plan_path, tokenizer, max_pairs=args.max_pairs)
    if args.dry_run:
        print(json.dumps(preflight, sort_keys=True, ensure_ascii=False), flush=True)
        return

    hard_cap = int(args.max_api_request_attempts)
    if hard_cap != preflight["planned_api_requests"]:
        raise ValueError(
            "For a fresh target-only collection the hard request cap must exactly "
            f"match planned requests: {hard_cap}!={preflight['planned_api_requests']}."
        )
    output_path = Path(args.output_jsonl).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_root = state_directory(output_path)
    run_manifest = manifest_path(output_path)
    run_audit = audit_path(output_path)
    existing = [
        path
        for path in (output_path, state_root, run_manifest, run_audit)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"Refusing to overwrite collector artifacts: {existing}")

    lock = lock_path(output_path)
    lock_handle = lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError(f"Another collector holds {lock}.") from exc

    configuration = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "collector_protocol": COLLECTOR_PROTOCOL,
        "collector_source_sha256": sha256_file(Path(__file__).resolve()),
        "plan_path": str(plan_path),
        "plan_sha256": preflight["plan_sha256"],
        "selected_pair_ids": [pair["pair_id"] for pair in pairs],
        "max_pairs": args.max_pairs,
        "target_model": TARGET_MODEL,
        "target_provider_order": [TARGET_PROVIDER],
        "target_provider_allow_fallbacks": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "reasoning_mode": "enabled_false",
        "reject_reasoning_tokens": True,
        "qwen_hard_no_think_prefill": True,
        "append_no_think": True,
        "disable_openrouter_response_cache": True,
        "max_api_request_attempts": hard_cap,
        "network_retries": 0,
        "reasoning_retries": 0,
    }
    fingerprint = sha256_payload(configuration)
    states: dict[str, dict[str, Any]] = {}
    try:
        state_root.mkdir(mode=0o700, exist_ok=False)
        atomic_write_json(
            run_manifest,
            {
                "manifest_schema_version": 1,
                "created_at_utc": utc_now(),
                "configuration": configuration,
                "configuration_fingerprint": fingerprint,
                "local_preflight": preflight,
            },
        )
        client_args = build_client_args(args, remaining_cap=hard_cap)
        client_args.api_url = OPENROUTER_URL
        api_key = resolve_api_key(client_args)
        client = OpenRouterClient(client_args, api_key)

        for pair in pairs:
            for arm in pair["arms"]:
                arm_id = arm["arm_id"]
                destination = state_path(state_root, arm_id)
                intent = {
                    "state_schema_version": 1,
                    "status": "intent",
                    "arm_id": arm_id,
                    "pair_id": pair["pair_id"],
                    "arm_role": arm["arm_role"],
                    "configuration_fingerprint": fingerprint,
                    "created_at_utc": utc_now(),
                }
                atomic_write_json(destination, intent)
                states[arm_id] = intent
                before = client_snapshot(client)
                try:
                    continuation, handoff_audit = generate_collector_continuation(
                        client=client,
                        tokenizer=tokenizer,
                        prompt=pair["api_prompt"],
                        prefix_text=arm["raw_forced_prefix"],
                        remaining_tokens=arm["remaining_tokens"],
                        temperature=0.0,
                        top_p=1.0,
                    )
                    request_audit = client_delta(before, client_snapshot(client))
                    if handoff_audit.get("reasoning_detected"):
                        raise ValueError("Continuation exposed reasoning tokens.")
                    if handoff_audit.get("response_model") != TARGET_MODEL:
                        raise ValueError("Continuation response-model mismatch.")
                    row = result_row(
                        pair,
                        arm,
                        continuation=continuation,
                        handoff_audit=handoff_audit,
                        request_audit=request_audit,
                        configuration_fingerprint=fingerprint,
                    )
                    result = {
                        "state_schema_version": 1,
                        "status": "result",
                        "arm_id": arm_id,
                        "pair_id": pair["pair_id"],
                        "configuration_fingerprint": fingerprint,
                        "updated_at_utc": utc_now(),
                        "row": row,
                    }
                    atomic_write_json(destination, result)
                    states[arm_id] = result
                except BaseException as exc:
                    after = client_snapshot(client)
                    uncertain = {
                        **intent,
                        "status": "uncertain",
                        "updated_at_utc": utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:2000],
                        "observed_client_delta": {
                            "calls": int(after["calls"]) - int(before["calls"]),
                            "request_attempts": int(after["request_attempts"])
                            - int(before["request_attempts"]),
                            "total_cost": float(after["total_cost"])
                            - float(before["total_cost"]),
                            "total_tokens": int(after["total_tokens"])
                            - int(before["total_tokens"]),
                        },
                    }
                    atomic_write_json(destination, uncertain)
                    states[arm_id] = uncertain
                    materialize_output(output_path, pairs, states)
                    failed = summarize(pairs, states, hard_cap=hard_cap)
                    failed.update(
                        {
                            "run_status": "failed_closed",
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:2000],
                        }
                    )
                    atomic_write_json(run_audit, failed)
                    raise
            materialize_output(output_path, pairs, states)

        summary = summarize(pairs, states, hard_cap=hard_cap)
        summary["run_status"] = "complete"
        atomic_write_json(run_audit, summary)
        print(json.dumps(summary, sort_keys=True, ensure_ascii=False), flush=True)
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    main()
