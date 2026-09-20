"""Plan a stratified one-shot outcome-regret pilot from exact MC50 caches.

This module does not contact the target or a judge.  It reconstructs one
exchangeable nested partial-K replay, evaluates the K-conditioned controller
and the original fixed-MC50 controller, and freezes both the paid mismatch
sample and the full state universe needed for later weighting.

The cached MC draws have no historical order.  A seeded permutation of each
aggregate MC50 multiset is therefore a *conditional replay*, not a recovery of
the requests' original order.  The output manifest states that limitation
explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeling_biasnet import BiasNet  # noqa: E402
from training.anytime_proxy_mc import (  # noqa: E402
    BIASNET_AUTOCAST_ENABLED,
    BIASNET_PARAMETER_DTYPE,
    load_anytime_cache,
    parse_budgets,
)
from training.anytime_stopping_features import FEATURE_NAMES  # noqa: E402
from training.eval_anytime_stopping import (  # noqa: E402
    _row_indices_for_records,
    evaluate_reference_actions,
    evaluate_replay_paths,
)


PLAN_SCHEMA_VERSION = 1
DEFAULT_REPLAY_SEED = 20260831
DEFAULT_SELECTION_SEED = "outcome-regret-pilot-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze mismatch states for one-shot outcome-regret collection. "
            "This planning stage makes no API calls."
        )
    )
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference_checkpoint", required=True)
    parser.add_argument("--policy_spec", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--budgets", default="0,4,8,16,32,50")
    parser.add_argument("--paid_budgets", default="0,4,8,16,32")
    parser.add_argument("--train_pairs", type=int, default=240)
    parser.add_argument("--validation_pairs", type=int, default=60)
    parser.add_argument("--replay_seed", type=int, default=DEFAULT_REPLAY_SEED)
    parser.add_argument("--selection_seed", default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Debug-only cap applied independently to train and validation.",
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


def _parse_paid_budgets(value: str, all_budgets: tuple[int, ...]) -> tuple[int, ...]:
    paid = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not paid or tuple(sorted(set(paid))) != paid:
        raise ValueError("--paid_budgets must be increasing and unique.")
    if any(budget not in all_budgets[:-1] for budget in paid):
        raise ValueError("Paid budgets must be non-terminal members of --budgets.")
    return paid


def _position_bin(position: int, normalizer: int) -> str:
    if position < 0 or normalizer <= 0 or position > normalizer:
        raise ValueError("Position lies outside the frozen normalizer.")
    fraction = float(position) / float(normalizer)
    if fraction < 1.0 / 3.0:
        return "early"
    if fraction < 2.0 / 3.0:
        return "middle"
    return "late"


def _priority(candidate_id: str, selection_seed: str) -> str:
    return hashlib.sha256(
        f"{selection_seed}:{candidate_id}".encode("utf-8")
    ).hexdigest()


def _allocate_integer(total: int, keys: Iterable[Any]) -> dict[Any, int]:
    keys = list(keys)
    if total < 0 or not keys:
        raise ValueError("Allocation requires a non-negative total and keys.")
    quotient, remainder = divmod(total, len(keys))
    return {
        key: quotient + (1 if index < remainder else 0)
        for index, key in enumerate(keys)
    }


def select_stratified_mismatches(
    universe: list[dict[str, Any]],
    *,
    total: int,
    paid_budgets: tuple[int, ...],
    selection_seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select paid mismatches with fixed K and position-bin quotas.

    Sampling is deterministic given ``selection_seed``.  It is uniform by a
    pseudorandom hash within each stratum.  Prefix reuse across K is allowed so
    the per-stratum inclusion fraction remains explicit; the collector may
    still reuse identical serialized branch requests.
    """

    position_bins = ("early", "middle", "late")
    budget_quota = _allocate_integer(total, paid_budgets)
    requested: dict[tuple[int, str], int] = {}
    for budget in paid_budgets:
        sub = _allocate_integer(budget_quota[budget], position_bins)
        requested.update({(budget, name): count for name, count in sub.items()})

    by_stratum: dict[tuple[int, str], list[dict[str, Any]]] = {
        key: [] for key in requested
    }
    for row in universe:
        key = (int(row["budget"]), str(row["position_bin"]))
        if row["action_mismatch"] and key in by_stratum:
            by_stratum[key].append(row)

    selected: list[dict[str, Any]] = []
    deficit = 0
    strata: dict[str, Any] = {}
    selected_ids: set[str] = set()
    for key in sorted(requested):
        eligible = sorted(
            by_stratum[key],
            key=lambda row: _priority(row["candidate_id"], selection_seed),
        )
        take = min(requested[key], len(eligible))
        chosen = eligible[:take]
        for row in chosen:
            copy = dict(row)
            copy["selection_stratum"] = f"k{key[0]}:{key[1]}"
            copy["stratum_eligible_mismatches"] = len(eligible)
            copy["stratum_selected_mismatches"] = take
            copy["inclusion_fraction_within_stratum"] = (
                float(take) / float(len(eligible)) if eligible else 0.0
            )
            selected.append(copy)
            selected_ids.add(copy["candidate_id"])
        deficit += requested[key] - take
        strata[f"k{key[0]}:{key[1]}"] = {
            "requested": requested[key],
            "eligible": len(eligible),
            "selected_initial": take,
        }

    if deficit:
        # Pre-registered deterministic fallback: fill from any remaining paid
        # mismatch, preserving the original per-stratum inclusion metadata.
        remaining = sorted(
            (
                row
                for row in universe
                if row["action_mismatch"]
                and int(row["budget"]) in paid_budgets
                and row["candidate_id"] not in selected_ids
            ),
            key=lambda row: _priority(row["candidate_id"], selection_seed + ":fallback"),
        )
        if len(remaining) < deficit:
            raise ValueError(
                f"Only {len(selected) + len(remaining)} paid mismatches exist; "
                f"cannot select requested total {total}."
            )
        for row in remaining[:deficit]:
            copy = dict(row)
            key = (int(copy["budget"]), str(copy["position_bin"]))
            copy["selection_stratum"] = f"k{key[0]}:{key[1]}"
            copy["selection_fallback"] = True
            copy["stratum_eligible_mismatches"] = len(by_stratum[key])
            copy["stratum_selected_mismatches"] = None
            copy["inclusion_fraction_within_stratum"] = None
            selected.append(copy)

    selected.sort(key=lambda row: row["candidate_id"])
    if len(selected) != total:
        raise RuntimeError("Stratified mismatch selection returned the wrong total.")
    return selected, {
        "requested_total": total,
        "selected_total": len(selected),
        "fallback_count": sum(bool(row.get("selection_fallback")) for row in selected),
        "strata": strata,
    }


def _dataset_index_by_record(cache_dir: Path, record_names: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in record_names:
        payload = torch.load(cache_dir / name, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("dataset_idx") is None:
            raise ValueError(f"{name} has no metadata.dataset_idx.")
        result[name] = int(metadata["dataset_idx"])
    return result


def _make_universe(
    *,
    paths: dict[str, Any],
    cache,
    row_indices: torch.Tensor,
    dataset_indices: dict[str, int],
    split: str,
    budgets: tuple[int, ...],
    paid_budgets: tuple[int, ...],
    replay_seed: int,
    position_normalizer: int,
) -> list[dict[str, Any]]:
    if paths["actions"].shape[0] != 1:
        raise ValueError("The pilot planner requires exactly one replay.")
    actions = paths["actions"][0]
    features = paths["features"][0]
    reference = np.asarray(paths["reference_actions"], dtype=np.int64)
    if not np.array_equal(actions[:, -1], reference):
        drift = float(np.mean(actions[:, -1] != reference))
        raise ValueError(
            "Anytime Kmax action drifted from original MC50 reference: "
            f"rate={drift:.8f}."
        )
    universe: list[dict[str, Any]] = []
    for local_row, global_index in enumerate(row_indices.tolist()):
        record_name = str(paths["record_names"][local_row])
        row_key = str(paths["row_keys"][local_row])
        position = int(cache.position_ids[global_index].item())
        for stage, budget in enumerate(budgets[:-1]):
            if budget not in paid_budgets:
                continue
            partial_action = int(actions[local_row, stage])
            reference_action = int(reference[local_row])
            identity = {
                "row_key": row_key,
                "budget": int(budget),
                "replay_seed": int(replay_seed),
                "replay_replicate": 0,
                "partial_action_token_id": partial_action,
                "reference_action_token_id": reference_action,
            }
            candidate_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
            state_features = [float(value) for value in features[local_row, stage]]
            if len(state_features) != len(FEATURE_NAMES) or any(
                not math.isfinite(value) for value in state_features
            ):
                raise ValueError("Planner produced invalid stopping features.")
            universe.append(
                {
                    "candidate_id": candidate_id,
                    "split": split,
                    "record_name": record_name,
                    "dataset_idx": dataset_indices[record_name],
                    "row_key": row_key,
                    "row_index_within_record": position,
                    "position": position,
                    "position_bin": _position_bin(position, position_normalizer),
                    "budget": int(budget),
                    "replay_seed": int(replay_seed),
                    "replay_replicate": 0,
                    "partial_action_token_id": partial_action,
                    "reference_action_token_id": reference_action,
                    # Stable collector aliases.  The explicit *_token_id names
                    # remain in the universe so downstream analyses cannot
                    # mistake these integers for decoded strings.
                    "partial_action": partial_action,
                    "reference_action": reference_action,
                    "action_mismatch": partial_action != reference_action,
                    "base_token_id": int(paths["base_token_ids"][local_row]),
                    "teacher_token_id": int(paths["labels"][local_row]),
                    "feature_names": list(FEATURE_NAMES),
                    "features": state_features,
                }
            )
    return universe


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )


def main() -> None:
    args = parse_args()
    if args.train_pairs <= 0 or args.validation_pairs <= 0:
        raise ValueError("Train and validation paid-pair counts must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max_rows must be positive when provided.")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    reference_checkpoint = Path(args.reference_checkpoint).expanduser().resolve()
    policy_path = Path(args.policy_spec).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite plan output: {output_dir}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if Path(policy["development_cache_dir"]).resolve() != cache_dir:
        raise ValueError("Policy development cache differs from --cache_dir.")
    if Path(policy["checkpoint"]).resolve() != checkpoint:
        raise ValueError("Policy checkpoint differs from --checkpoint.")
    if Path(policy["reference_checkpoint"]).resolve() != reference_checkpoint:
        raise ValueError("Policy reference checkpoint differs from CLI.")
    expected_hashes = {
        "checkpoint_weights_sha256": checkpoint / "pytorch_model.bin",
        "checkpoint_config_sha256": checkpoint / "config.json",
        "reference_checkpoint_weights_sha256": reference_checkpoint / "pytorch_model.bin",
        "reference_checkpoint_config_sha256": reference_checkpoint / "config.json",
        "development_cache_manifest_sha256": cache_dir / "proxy_mc_manifest.json",
    }
    for field, path in expected_hashes.items():
        if policy.get(field) != sha256_file(path):
            raise ValueError(f"Frozen policy hash mismatch for {field}.")

    cache = load_anytime_cache(cache_dir)
    budgets = parse_budgets(args.budgets, max_samples=cache.max_samples)
    if list(budgets) != policy.get("budgets"):
        raise ValueError("Planner budget schedule differs from frozen policy.")
    paid_budgets = _parse_paid_budgets(args.paid_budgets, budgets)
    position_normalizer = int(policy["position_normalizer"])
    train_files = list(policy.get("stopper_fit_files") or ())
    validation_files = list(policy.get("legacy_validation_files") or ())
    if not train_files or not validation_files or set(train_files) & set(validation_files):
        raise ValueError("Frozen policy has invalid train/legacy-validation files.")

    model = BiasNet.from_pretrained(checkpoint, map_location="cpu")
    reference_model = BiasNet.from_pretrained(reference_checkpoint, map_location="cpu")
    parameter_dtypes = {
        str(parameter.dtype).replace("torch.", "") for parameter in model.parameters()
    }
    if parameter_dtypes != {BIASNET_PARAMETER_DTYPE}:
        raise ValueError(f"Anytime model parameter dtypes are {sorted(parameter_dtypes)}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if BIASNET_AUTOCAST_ENABLED and device.type != "cuda":
        raise ValueError("Planning requires CUDA to match the frozen autocast contract.")
    model = model.to(device)
    model.set_up_proj()
    model.eval()
    reference_model = reference_model.to(device)
    reference_model.set_up_proj()
    reference_model.eval()

    dataset_indices = _dataset_index_by_record(
        cache_dir, [*train_files, *validation_files]
    )
    all_universe: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}
    for split, record_names, target in (
        ("train", train_files, args.train_pairs),
        ("legacy_validation", validation_files, args.validation_pairs),
    ):
        row_indices = _row_indices_for_records(cache, record_names)
        if args.max_rows is not None:
            row_indices = row_indices[: args.max_rows]
        paths = evaluate_replay_paths(
            cache=cache,
            row_indices=row_indices,
            model=model,
            budgets=budgets,
            replays=1,
            replay_seed=args.replay_seed,
            batch_size=args.batch_size,
            device=device,
            position_normalizer=position_normalizer,
        )
        paths["reference_actions"] = evaluate_reference_actions(
            cache=cache,
            row_indices=row_indices,
            model=reference_model,
            batch_size=args.batch_size,
            device=device,
        )
        universe = _make_universe(
            paths=paths,
            cache=cache,
            row_indices=row_indices,
            dataset_indices=dataset_indices,
            split=split,
            budgets=budgets,
            paid_budgets=paid_budgets,
            replay_seed=args.replay_seed,
            position_normalizer=position_normalizer,
        )
        chosen, report = select_stratified_mismatches(
            universe,
            total=target,
            paid_budgets=paid_budgets,
            selection_seed=f"{args.selection_seed}:{split}",
        )
        all_universe.extend(universe)
        selected.extend(chosen)
        split_reports[split] = {
            "records": len(record_names),
            "rows": int(row_indices.numel()),
            "states": len(universe),
            "mismatch_states": sum(row["action_mismatch"] for row in universe),
            **report,
        }

    output_dir.mkdir(parents=True)
    try:
        universe_path = output_dir / "state_universe.jsonl"
        plan_path = output_dir / "paid_pair_plan.jsonl"
        _write_jsonl(universe_path, all_universe)
        _write_jsonl(plan_path, selected)
        manifest = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "planner": "stratified_exchangeable_nested_replay_v1",
            "offline_estimand": "base_policy_one_step_counterfactual_harmful_regret",
            "partial_k_semantics": (
                "conditional_exchangeable_replay_not_historical_request_order"
            ),
            "cache_dir": str(cache_dir),
            "cache_manifest_sha256": sha256_file(cache_dir / "proxy_mc_manifest.json"),
            "checkpoint": str(checkpoint),
            "checkpoint_weights_sha256": sha256_file(checkpoint / "pytorch_model.bin"),
            "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
            "reference_checkpoint": str(reference_checkpoint),
            "reference_checkpoint_weights_sha256": sha256_file(
                reference_checkpoint / "pytorch_model.bin"
            ),
            "reference_checkpoint_config_sha256": sha256_file(
                reference_checkpoint / "config.json"
            ),
            "policy_spec": str(policy_path),
            "policy_spec_sha256": sha256_file(policy_path),
            "budgets": list(budgets),
            "paid_budgets": list(paid_budgets),
            "feature_names": list(FEATURE_NAMES),
            "position_normalizer": position_normalizer,
            "replay_seed": args.replay_seed,
            "replay_replicates": 1,
            "selection_seed": args.selection_seed,
            "train_files": train_files,
            "legacy_validation_files": validation_files,
            "fresh_test_used": False,
            "split_reports": split_reports,
            "state_universe_file": universe_path.name,
            "state_universe_sha256": sha256_file(universe_path),
            "paid_pair_plan_file": plan_path.name,
            "paid_pair_plan_sha256": sha256_file(plan_path),
            "paid_pair_count": len(selected),
            "action_kmax_reference_drift": 0.0,
        }
        manifest["manifest_payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(manifest)
        ).hexdigest()
        (output_dir / "plan_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
