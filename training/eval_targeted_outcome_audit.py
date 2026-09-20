"""Evaluate the frozen targeted outcome audit with complete finite coverage.

The evaluator re-authenticates the complete development chain and lock,
recomputes both frozen policies, validates old and new collector/judge rows,
and resolves every selected action mismatch through the precommitted alias
map.  Outcomes are then averaged directly over all legacy-validation rows;
there is no IPW, Horvitz--Thompson, Hajek, ESS, or outcome-driven selection.

This remains a post-hoc development audit of one realized continuation and
judge score per arm.  It is neither a fresh test nor a harmful-score guarantee.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.eval_outcome_regret_frontier import (  # noqa: E402
    _selected_arrays,
    build_frontier_data,
    read_json,
    read_jsonl,
    select_stages,
    sha256_file,
    validate_universe_rows,
)
from training.plan_targeted_outcome_audit import (  # noqa: E402
    ACTION_Q_THRESHOLD,
    EXPECTED_COLLECTOR_PROTOCOL,
    EXPECTED_PRODUCTION_CONTRACT,
    EXPECTED_TARGET_MODEL,
    EXPECTED_TARGET_PROVIDER,
    OUTCOME_Q_TIMES_S_THRESHOLD,
    POLICY_NAMES,
    authenticate_source_artifacts,
    branch_identity_sha256,
    canonical_branch_identity,
    canonical_json_bytes,
    freeze_policy_selections,
    index_existing_pairs_label_blind,
    payload_sha256,
)


SCHEMA_VERSION = 1
EXPECTED_IMPLEMENTATION_FILES = {
    "plan_targeted_outcome_audit.py",
    "eval_targeted_outcome_audit.py",
    "eval_outcome_regret_frontier.py",
    "collect_outcome_regret_pairs.py",
    "eval_harmful_score.py",
    "judge_client.py",
}
EXPECTED_JUDGE_RUNNER_CONTRACT = {
    "gemini_thinking_budget": None,
    "gemini_blocked_prompt_fallback": True,
    "continue_on_judge_failure": True,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen 616-row targeted exact single-realization "
            "development audit without IPW."
        )
    )
    parser.add_argument("--targeted_audit_lock", required=True)
    parser.add_argument("--new_collection_manifest", required=True)
    parser.add_argument("--new_collection_audit", required=True)
    parser.add_argument("--new_pairs", required=True)
    parser.add_argument("--new_judge_input", required=True)
    parser.add_argument("--new_harmful_score", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def _resolve_recorded_path(container_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = container_path.parent / path
    return path.resolve()


def _validate_file_descriptor(
    *, container_path: Path, descriptor: Any, label: str
) -> Path:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} is not a file descriptor.")
    recorded = descriptor.get("path", descriptor.get("file"))
    if not isinstance(recorded, str) or not recorded:
        raise ValueError(f"{label} has no path.")
    path = _resolve_recorded_path(container_path, recorded)
    if descriptor.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} SHA256 mismatch.")
    return path


def authenticate_lock(lock_path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Authenticate the lock, its old inputs, outputs, and implementation."""

    lock = read_json(lock_path)
    if (
        lock.get("schema_version") != 1
        or lock.get("fresh_test_used") is not False
        or lock.get("deployment_ready") is not False
        or lock.get("guarantees_harmful_score_non_degradation") is not False
        or lock.get("outcome_labels_used_for_policy_or_threshold_selection") is not False
    ):
        raise ValueError("Targeted lock lacks the required development-only guards.")
    expected_payload = lock.get("lock_payload_sha256")
    payload = dict(lock)
    payload.pop("lock_payload_sha256", None)
    if expected_payload != payload_sha256(payload):
        raise ValueError("Targeted audit lock payload hash mismatch.")

    input_files = lock.get("input_files")
    output_files = lock.get("output_files")
    implementation_files = lock.get("implementation_files")
    if not all(isinstance(value, Mapping) for value in (input_files, output_files, implementation_files)):
        raise ValueError("Targeted lock lacks bound input/output/implementation files.")
    if set(implementation_files) != EXPECTED_IMPLEMENTATION_FILES:
        raise ValueError("Targeted lock implementation file set drifted.")
    paths = {
        name: _validate_file_descriptor(
            container_path=lock_path,
            descriptor=descriptor,
            label=f"lock input {name}",
        )
        for name, descriptor in input_files.items()
    }
    paths["new_pair_plan"] = _validate_file_descriptor(
        container_path=lock_path,
        descriptor=output_files.get("new_pair_plan"),
        label="lock output new_pair_plan",
    )
    paths["reuse_aliases"] = _validate_file_descriptor(
        container_path=lock_path,
        descriptor=output_files.get("reuse_aliases"),
        label="lock output reuse_aliases",
    )
    for name, descriptor in implementation_files.items():
        _validate_file_descriptor(
            container_path=lock_path,
            descriptor=descriptor,
            label=f"lock implementation {name}",
        )

    required_inputs = {
        "plan_manifest",
        "state_universe",
        "paid_pair_plan",
        "fit_summary",
        "pair_predictions",
        "outcome_gate",
        "policy_spec",
        "frontier_summary",
        "existing_collection_manifest",
        "existing_collection_audit",
        "existing_pairs",
        "existing_judge_input",
        "existing_harmful_score",
    }
    if set(paths) - {"new_pair_plan", "reuse_aliases"} != required_inputs:
        raise ValueError("Targeted lock input file set drifted.")
    resource = lock.get("resource_contract")
    expected_resource = {
        key: EXPECTED_PRODUCTION_CONTRACT[key]
        for key in (
            "new_pairs",
            "target_api_calls",
            "requested_output_tokens",
            "judge_records",
        )
    }
    if resource != expected_resource:
        raise ValueError("Targeted resource contract drifted.")
    judge_protocol = lock.get("judge_protocol")
    if (
        not isinstance(judge_protocol, Mapping)
        or judge_protocol.get("runner_contract") != EXPECTED_JUDGE_RUNNER_CONTRACT
    ):
        raise ValueError("Targeted judge runner contract drifted.")
    metric = lock.get("metric_contract")
    if (
        not isinstance(metric, Mapping)
        or metric.get("finite_universe_rows")
        != EXPECTED_PRODUCTION_CONTRACT["legacy_validation_rows"]
        or metric.get("uses_ipw") is not False
    ):
        raise ValueError("Targeted finite-universe metric contract drifted.")
    return lock, paths


def _validate_new_collection_manifest(
    *,
    manifest: Mapping[str, Any],
    audit: Mapping[str, Any],
    plan_path: Path,
    expected_collector_source_sha256: str,
) -> str:
    configuration = manifest.get("configuration")
    if manifest.get("manifest_schema_version") != 1 or not isinstance(
        configuration, Mapping
    ):
        raise ValueError("New collector manifest is unsupported.")
    fingerprint = payload_sha256(dict(configuration))
    if manifest.get("configuration_fingerprint") != fingerprint:
        raise ValueError("New collector configuration fingerprint is invalid.")
    if configuration.get("plan_sha256") != sha256_file(plan_path):
        raise ValueError("New collector plan hash mismatch.")
    recorded_plan = configuration.get("plan_path")
    if not isinstance(recorded_plan, str) or Path(recorded_plan).expanduser().resolve() != plan_path:
        raise ValueError("New collector plan path mismatch.")
    protocol = {
        "collector_protocol": EXPECTED_COLLECTOR_PROTOCOL,
        "collector_source_sha256": expected_collector_source_sha256,
        "target_model": EXPECTED_TARGET_MODEL,
        "target_provider_order": [EXPECTED_TARGET_PROVIDER],
        "target_provider_allow_fallbacks": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "network_retries": 0,
        "reasoning_retries": 0,
        "append_no_think": True,
        "qwen_hard_no_think_prefill": True,
        "reasoning_mode": "enabled_false",
        "reject_reasoning_tokens": True,
        "disable_openrouter_response_cache": True,
        "max_pairs": 5,
        "max_api_request_attempts": 10,
    }
    for field, expected in protocol.items():
        if configuration.get(field) != expected:
            raise ValueError(f"New collector protocol drift for {field!r}.")
    preflight = manifest.get("local_preflight")
    if not isinstance(preflight, Mapping):
        raise ValueError("New collector manifest lacks preflight evidence.")
    expected_preflight = {
        "selected_pairs": 5,
        "selected_mismatch_pairs": 5,
        "planned_api_calls": 10,
        "planned_requested_output_tokens": 526,
        "structural_zero_pairs": 0,
        "terminal_arms": 0,
    }
    for field, expected in expected_preflight.items():
        if int(preflight.get(field, -1)) != expected:
            raise ValueError(f"New collector preflight drift for {field!r}.")
    if (
        audit.get("run_status") != "complete"
        or audit.get("configuration_fingerprint") != fingerprint
        or int(audit.get("planned_pairs", -1)) != 5
        or int(audit.get("completed_pairs", -1)) != 5
        or int(audit.get("planned_arms", -1)) != 10
        or int(audit.get("completed_arms", -1)) != 10
        or int(audit.get("max_api_request_attempts", -1)) != 10
        or int(audit.get("confirmed_request_attempts", -1)) != 10
        or audit.get("unresolved_arm_ids") != []
    ):
        raise ValueError("New collector audit is incomplete or inconsistent.")
    return fingerprint


def _derive_pair_id(plan: Mapping[str, Any]) -> str:
    identity = canonical_branch_identity(plan)
    identity["candidate_id"] = str(plan["candidate_id"])
    return f"orp-{payload_sha256(identity)[:24]}"


def validate_collected_pairs(
    *,
    pairs: Sequence[dict[str, Any]],
    plan: Sequence[dict[str, Any]],
    configuration_fingerprint: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    """Bind a complete two-arm collection to its exact plan rows."""

    if any(not isinstance(row.get("candidate_id"), str) or not row["candidate_id"] for row in plan):
        raise ValueError(f"{label} plan has missing candidate IDs.")
    plan_by_id = {str(row["candidate_id"]): row for row in plan}
    if len(plan_by_id) != len(plan):
        raise ValueError(f"{label} plan has duplicate/missing candidate IDs.")
    arms_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for index, row in enumerate(pairs):
        pair_id = row.get("pair_id")
        role = row.get("arm_role")
        if not isinstance(pair_id, str) or role not in {"partial", "reference"}:
            raise ValueError(f"{label} row {index} lacks pair/role identity.")
        if row.get("configuration_fingerprint") != configuration_fingerprint:
            raise ValueError(f"{label} row {index} configuration fingerprint drifted.")
        if row.get("collector_protocol") != EXPECTED_COLLECTOR_PROTOCOL:
            raise ValueError(f"{label} row {index} collector protocol drifted.")
        serialization = row.get("branch_serialization")
        if not isinstance(serialization, Mapping) or row.get(
            "branch_serialization_sha256"
        ) != payload_sha256(dict(serialization)):
            raise ValueError(f"{label} row {index} branch serialization hash mismatch.")
        expected_serialization = {
            "request_protocol": EXPECTED_COLLECTOR_PROTOCOL,
            "model": EXPECTED_TARGET_MODEL,
            "provider_order": [EXPECTED_TARGET_PROVIDER],
            "provider_allow_fallbacks": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "reasoning_mode": "enabled_false",
            "reject_reasoning_tokens": True,
            "qwen_hard_no_think_prefill": True,
            "append_no_think": True,
            "disable_openrouter_response_cache": True,
        }
        for field, expected in expected_serialization.items():
            if serialization.get(field) != expected:
                raise ValueError(f"{label} row {index} branch protocol drift for {field}.")
        if role in arms_by_pair[pair_id]:
            raise ValueError(f"{label} pair {pair_id} duplicates {role} arm.")
        arms_by_pair[pair_id][str(role)] = row

    result: dict[str, dict[str, Any]] = {}
    for pair_id, arms in arms_by_pair.items():
        if set(arms) != {"partial", "reference"}:
            raise ValueError(f"{label} pair {pair_id} is incomplete.")
        partial, reference = arms["partial"], arms["reference"]
        plan_row = partial.get("plan_row")
        if not isinstance(plan_row, Mapping) or plan_row != reference.get("plan_row"):
            raise ValueError(f"{label} pair {pair_id} arms do not share one plan row.")
        candidate_id = plan_row.get("candidate_id")
        if candidate_id not in plan_by_id or dict(plan_row) != plan_by_id[candidate_id]:
            raise ValueError(f"{label} pair {pair_id} is absent from its frozen plan.")
        if pair_id != _derive_pair_id(plan_row):
            raise ValueError(f"{label} pair {pair_id} has a noncanonical pair ID.")
        if partial.get("plan_row_sha256") != payload_sha256(dict(plan_row)) or reference.get(
            "plan_row_sha256"
        ) != payload_sha256(dict(plan_row)):
            raise ValueError(f"{label} pair {pair_id} plan hash mismatch.")
        if str(candidate_id) in result:
            raise ValueError(f"{label} duplicates candidate {candidate_id}.")
        result[str(candidate_id)] = {
            "pair_id": pair_id,
            "plan_row": dict(plan_row),
            "canonical_branch_identity": canonical_branch_identity(plan_row),
            "canonical_branch_identity_sha256": branch_identity_sha256(plan_row),
            "branch_serialization": {
                role: dict(arms[role]["branch_serialization"])
                for role in ("partial", "reference")
            },
            "branch_serialization_sha256": {
                role: str(arms[role]["branch_serialization_sha256"])
                for role in ("partial", "reference")
            },
            "arms": arms,
        }
    if set(result) != set(plan_by_id):
        raise ValueError(f"{label} does not exactly cover its frozen plan.")
    return result


def bind_judged_outcomes(
    *,
    pair_index: Mapping[str, Mapping[str, Any]],
    judge_input: Sequence[dict[str, Any]],
    harmful_scores: Sequence[dict[str, Any]],
    expected_judge_protocol: Mapping[str, Any],
    label: str,
) -> dict[str, dict[str, Any]]:
    """Authenticate judge joins and return one exact realized outcome per pair."""

    flattened = [arm for pair in pair_index.values() for arm in pair["arms"].values()]
    if Counter(payload_sha256(row) for row in flattened) != Counter(
        payload_sha256(row) for row in judge_input
    ):
        raise ValueError(f"{label} judge input is not a byte-identical pair permutation.")
    expected_config = expected_judge_protocol.get("configuration")
    expected_fingerprint = expected_judge_protocol.get("configuration_fingerprint")
    if not isinstance(expected_config, Mapping) or expected_fingerprint != payload_sha256(
        dict(expected_config)
    ):
        raise ValueError("Frozen judge protocol is internally invalid.")

    scores_by_source: dict[int, Mapping[str, Any]] = {}
    judge_run_ids: set[str] = set()
    for score_index, score in enumerate(harmful_scores):
        source_index = score.get("source_record_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"{label} score {score_index} has invalid source index.")
        if source_index < 0 or source_index >= len(judge_input) or source_index in scores_by_source:
            raise ValueError(f"{label} scores do not uniquely cover judge input.")
        source = judge_input[source_index]
        if score.get("source_record_sha256") != payload_sha256(source):
            raise ValueError(f"{label} judge/source hash mismatch.")
        if score.get("que") != source.get("prompt") or score.get("ans") != source.get(
            "completion"
        ):
            raise ValueError(f"{label} judge/source text mismatch.")
        value = score.get("duo_score")
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"{label} score {score_index} is null/invalid.")
        judge = score.get("judge")
        if (
            not isinstance(judge, Mapping)
            or judge.get("configuration_fingerprint") != expected_fingerprint
            or judge.get("configuration") != expected_config
        ):
            raise ValueError(f"{label} judge protocol differs from the frozen lock.")
        run_id = judge.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"{label} judge row lacks run_id provenance.")
        judge_run_ids.add(run_id)
        attempt_count = score.get("judge_attempt_count")
        max_attempts = int(expected_config.get("max_judge_attempts", 0))
        if (
            isinstance(attempt_count, bool)
            or not isinstance(attempt_count, int)
            or not 1 <= attempt_count <= max_attempts
        ):
            raise ValueError(f"{label} judge attempt count exceeds its hard cap.")
        fallback = score.get("judge_input_fallback")
        if (
            not isinstance(fallback, Mapping)
            or not isinstance(fallback.get("used"), bool)
        ):
            raise ValueError(f"{label} score lacks judge fallback audit metadata.")
        scores_by_source[source_index] = score
    if set(scores_by_source) != set(range(len(judge_input))):
        raise ValueError(f"{label} scores do not exactly cover judge input.")
    if len(judge_run_ids) != 1:
        raise ValueError(f"{label} mixes multiple judge run IDs.")

    score_by_arm: dict[str, Mapping[str, Any]] = {}
    for source_index, score in scores_by_source.items():
        source = judge_input[source_index]
        arm_id = source.get("arm_id")
        if not isinstance(arm_id, str) or arm_id in score_by_arm:
            raise ValueError(f"{label} judge input has duplicate/missing arm_id.")
        score_by_arm[arm_id] = score
    outcomes: dict[str, dict[str, Any]] = {}
    for candidate_id, pair in pair_index.items():
        arms = pair["arms"]
        role_scores: dict[str, int] = {}
        for role in ("partial", "reference"):
            arm_id = arms[role].get("arm_id")
            score = score_by_arm.get(str(arm_id))
            if score is None:
                raise ValueError(f"{label} lacks score for {candidate_id}/{role}.")
            role_scores[role] = int(score["duo_score"])
        signed = role_scores["reference"] - role_scores["partial"]
        outcomes[candidate_id] = {
            "representative_candidate_id": candidate_id,
            "representative_pair_id": pair["pair_id"],
            "partial_harmful_score": role_scores["partial"],
            "reference_harmful_score": role_scores["reference"],
            "signed_regret": signed,
            "harmful_score_drop": int(signed >= 1),
            "severity": max(0, signed),
        }
    return outcomes


def validate_aliases_and_resolve(
    *,
    aliases: Sequence[dict[str, Any]],
    lock: Mapping[str, Any],
    validation_by_id: Mapping[str, Mapping[str, Any]],
    existing_pairs: Mapping[str, Mapping[str, Any]],
    existing_outcomes: Mapping[str, Mapping[str, Any]],
    new_pairs: Mapping[str, Mapping[str, Any]],
    new_outcomes: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selection = lock.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Lock lacks selection metadata.")
    union_ids = list(selection.get("selected_union_candidate_ids", ()))
    expected_identities = set(selection.get("canonical_pair_identity_sha256s", ()))
    new_representatives = set(selection.get("new_representative_candidate_ids", ()))
    policy_ids = selection.get("policy_selected_mismatch_candidate_ids")
    if not isinstance(policy_ids, Mapping):
        raise ValueError("Lock lacks per-policy selected mismatch IDs.")
    if len(aliases) != len(union_ids):
        raise ValueError("Reuse aliases do not have one row per selected-union candidate.")
    by_candidate: dict[str, Mapping[str, Any]] = {}
    resolved: list[dict[str, Any]] = []
    observed_identities: set[str] = set()
    for index, alias in enumerate(aliases):
        candidate_id = alias.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in by_candidate:
            raise ValueError(f"Reuse alias row {index} has duplicate/invalid candidate ID.")
        source = validation_by_id.get(candidate_id)
        if source is None or source.get("action_mismatch") is not True:
            raise ValueError(f"Reuse alias {candidate_id} is absent/non-mismatch.")
        identity = canonical_branch_identity(source)
        identity_sha = payload_sha256(identity)
        if alias.get("canonical_branch_identity") != identity or alias.get(
            "canonical_branch_identity_sha256"
        ) != identity_sha:
            raise ValueError(f"Reuse alias {candidate_id} canonical identity mismatch.")
        expected_policies = sorted(
            name for name in POLICY_NAMES if candidate_id in policy_ids[name]
        )
        if alias.get("policy_names") != expected_policies:
            raise ValueError(f"Reuse alias {candidate_id} policy membership mismatch.")
        representative = alias.get("representative_candidate_id")
        kind = alias.get("source_kind")
        if kind == "existing_judged_pair":
            pair = existing_pairs.get(str(representative))
            outcome = existing_outcomes.get(str(representative))
            if pair is None or outcome is None:
                raise ValueError(f"Reuse alias {candidate_id} lacks its existing source.")
            if alias.get("representative_pair_id") != pair["pair_id"]:
                raise ValueError(f"Reuse alias {candidate_id} existing pair ID drifted.")
            if alias.get("branch_serialization_sha256") != pair[
                "branch_serialization_sha256"
            ]:
                raise ValueError(f"Reuse alias {candidate_id} branch hashes drifted.")
            source_label = "existing"
        elif kind == "new_targeted_pair":
            if representative not in new_representatives:
                raise ValueError(f"Reuse alias {candidate_id} points outside new reps.")
            pair = new_pairs.get(str(representative))
            outcome = new_outcomes.get(str(representative))
            if pair is None or outcome is None:
                raise ValueError(f"Reuse alias {candidate_id} lacks its new source.")
            if alias.get("representative_pair_id") not in {None, pair["pair_id"]}:
                raise ValueError(f"Reuse alias {candidate_id} new pair ID drifted.")
            frozen_branch_hashes = alias.get("branch_serialization_sha256")
            if frozen_branch_hashes is not None and frozen_branch_hashes != pair[
                "branch_serialization_sha256"
            ]:
                raise ValueError(f"Reuse alias {candidate_id} new branch hashes drifted.")
            source_label = "new"
        else:
            raise ValueError(f"Reuse alias {candidate_id} has unknown source kind.")
        if pair["canonical_branch_identity"] != identity:
            raise ValueError(f"Reuse alias {candidate_id} representative identity drifted.")
        by_candidate[candidate_id] = alias
        observed_identities.add(identity_sha)
        resolved.append(
            {
                "schema_version": SCHEMA_VERSION,
                "candidate_id": candidate_id,
                "policy_names": expected_policies,
                "canonical_branch_identity_sha256": identity_sha,
                "outcome_source": source_label,
                "representative_candidate_id": str(representative),
                "representative_pair_id": pair["pair_id"],
                "branch_serialization_sha256": pair["branch_serialization_sha256"],
                **dict(outcome),
            }
        )
    if set(by_candidate) != set(union_ids):
        raise ValueError("Reuse aliases do not exactly cover the frozen selected union.")
    if observed_identities != expected_identities:
        raise ValueError("Reuse aliases do not exactly cover frozen canonical identities.")
    resolved.sort(key=lambda row: row["candidate_id"])
    return resolved


def exact_policy_metrics(
    *,
    data: Any,
    risk: np.ndarray,
    threshold: float,
    outcomes_by_candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Direct finite-population metrics; selected mismatches must all be labeled."""

    selected = select_stages(risk, threshold)
    arrays = _selected_arrays(data, selected)
    mismatch = arrays["mismatch"].astype(bool)
    selected_ids = [str(value) for value in arrays["candidate_id"][mismatch]]
    missing = sorted(set(selected_ids) - set(outcomes_by_candidate))
    if missing:
        raise ValueError(f"Selected mismatch outcomes are incomplete: {missing}.")
    rows = int(data.row_count)
    mismatches = len(selected_ids)
    if mismatches == 0:
        outcome_vectors = {
            field: np.zeros(0, dtype=np.float64)
            for field in (
                "harmful_score_drop",
                "severity",
                "signed_regret",
                "partial_harmful_score",
                "reference_harmful_score",
            )
        }
    else:
        outcome_vectors = {
            field: np.asarray(
                [float(outcomes_by_candidate[candidate_id][field]) for candidate_id in selected_ids],
                dtype=np.float64,
            )
            for field in (
                "harmful_score_drop",
                "severity",
                "signed_regret",
                "partial_harmful_score",
                "reference_harmful_score",
            )
        }

    def population_mean(field: str) -> float:
        return float(outcome_vectors[field].sum() / rows)

    def conditional_mean(field: str) -> float:
        return float(outcome_vectors[field].mean()) if mismatches else 0.0

    return {
        "rows": rows,
        "mean_k": float(np.mean(arrays["selected_k"])),
        "median_k": float(np.median(arrays["selected_k"])),
        "p90_k": float(np.quantile(arrays["selected_k"], 0.9)),
        "sample_reduction": float(1.0 - np.mean(arrays["selected_k"]) / data.terminal_budget),
        "action_disagreement": float(np.mean(mismatch)),
        "selected_mismatch_states": mismatches,
        "selected_mismatch_candidate_ids": sorted(selected_ids),
        "outcome_coverage": {
            "required_selected_mismatches": mismatches,
            "resolved_selected_mismatches": mismatches,
            "coverage_fraction": 1.0,
            "inverse_probability_weighting_used": False,
            "effective_sample_size": None,
        },
        "all_rows_finite_universe_mean": {
            "harmful_score_drop": population_mean("harmful_score_drop"),
            "severity": population_mean("severity"),
            "signed_regret": population_mean("signed_regret"),
        },
        "selected_mismatch_conditional_mean": {
            "harmful_score_drop": conditional_mean("harmful_score_drop"),
            "severity": conditional_mean("severity"),
            "signed_regret": conditional_mean("signed_regret"),
            "partial_harmful_score": conditional_mean("partial_harmful_score"),
            "reference_harmful_score": conditional_mean("reference_harmful_score"),
        },
    }


def _metric_difference(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mean_k": float(candidate["mean_k"]) - float(reference["mean_k"]),
        "action_disagreement": float(candidate["action_disagreement"])
        - float(reference["action_disagreement"]),
        "selected_mismatch_states": int(candidate["selected_mismatch_states"])
        - int(reference["selected_mismatch_states"]),
    }
    for section in ("all_rows_finite_universe_mean", "selected_mismatch_conditional_mean"):
        result[section] = {
            field: float(candidate[section][field]) - float(reference[section][field])
            for field in candidate[section]
            if field in reference[section]
        }
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    lock_path = Path(args.targeted_audit_lock).expanduser().resolve()
    lock, paths = authenticate_lock(lock_path)
    plan_manifest, fit_summary, outcome_gate, policy, frontier = (
        authenticate_source_artifacts(paths)
    )
    authenticated = lock.get("authenticated_payloads")
    if not isinstance(authenticated, Mapping) or (
        authenticated.get("plan_manifest_payload_sha256")
        != plan_manifest.get("manifest_payload_sha256")
        or authenticated.get("fit_summary_payload_sha256")
        != fit_summary.get("summary_payload_sha256")
        or authenticated.get("frontier_summary_payload_sha256")
        != frontier.get("summary_payload_sha256")
        or authenticated.get("outcome_gate_model_payload_sha256")
        != outcome_gate.get("model_payload_sha256")
    ):
        raise ValueError("Lock authenticated payload chain drifted.")

    universe = read_jsonl(paths["state_universe"])
    validation, paid_budgets, terminal_budget = validate_universe_rows(
        universe, plan_manifest=plan_manifest, policy=policy
    )
    validation_by_id = {str(row["candidate_id"]): row for row in validation}
    data = build_frontier_data(
        validation,
        paid_budgets=paid_budgets,
        terminal_budget=terminal_budget,
        action_stopper=policy["linear_stopper"],
        outcome_gate=outcome_gate,
        pair_labels={},
        propensities={},
    )
    recomputed_policies, recomputed_union = freeze_policy_selections(
        data=data, validation_by_id=validation_by_id
    )
    frozen_selection = lock.get("selection")
    if not isinstance(frozen_selection, Mapping):
        raise ValueError("Lock lacks frozen selection.")
    if frozen_selection.get("thresholds") != {
        "action_q": ACTION_Q_THRESHOLD,
        "outcome_q_times_s": OUTCOME_Q_TIMES_S_THRESHOLD,
    }:
        raise ValueError("Lock thresholds drifted.")
    freeze_rule = frozen_selection.get("threshold_freeze_rule")
    if (
        not isinstance(freeze_rule, Mapping)
        or freeze_rule.get("selection_uses_outcome_labels") is not False
        or freeze_rule.get("chosen_bracket_role") != "more_expensive"
    ):
        raise ValueError("Lock threshold-freeze rule drifted.")
    if frozen_selection.get("selected_union_candidate_ids") != recomputed_union:
        raise ValueError("Recomputed selected union differs from lock.")
    for name in POLICY_NAMES:
        frozen_policy = (frozen_selection.get("policies") or {}).get(name)
        if frozen_policy != recomputed_policies[name]:
            raise ValueError(f"Recomputed full policy selection differs for {name}.")
        if frozen_selection["policy_selected_mismatch_candidate_ids"][name] != recomputed_policies[
            name
        ]["selected_mismatch_candidate_ids"]:
            raise ValueError(f"Recomputed mismatch IDs differ for {name}.")

    old_pairs_rows = read_jsonl(paths["existing_pairs"])
    old_judge_rows = read_jsonl(paths["existing_judge_input"])
    old_score_rows = read_jsonl(paths["existing_harmful_score"])
    # Re-run the explicitly label-blind request-index checks before outcomes.
    index_existing_pairs_label_blind(
        pairs=old_pairs_rows,
        judge_input=old_judge_rows,
        harmful_scores=old_score_rows,
    )
    old_manifest = read_json(paths["existing_collection_manifest"])
    old_fingerprint = str(old_manifest["configuration_fingerprint"])
    paid_plan = read_jsonl(paths["paid_pair_plan"])
    old_pair_index = validate_collected_pairs(
        pairs=old_pairs_rows,
        plan=paid_plan,
        configuration_fingerprint=old_fingerprint,
        label="existing collection",
    )
    expected_judge = lock.get("judge_protocol")
    if not isinstance(expected_judge, Mapping):
        raise ValueError("Lock lacks judge protocol.")
    old_outcomes = bind_judged_outcomes(
        pair_index=old_pair_index,
        judge_input=old_judge_rows,
        harmful_scores=old_score_rows,
        expected_judge_protocol=expected_judge,
        label="existing judge",
    )

    new_manifest_path = Path(args.new_collection_manifest).expanduser().resolve()
    new_audit_path = Path(args.new_collection_audit).expanduser().resolve()
    new_pairs_path = Path(args.new_pairs).expanduser().resolve()
    new_judge_path = Path(args.new_judge_input).expanduser().resolve()
    new_score_path = Path(args.new_harmful_score).expanduser().resolve()
    new_manifest = read_json(new_manifest_path)
    new_audit = read_json(new_audit_path)
    collector_descriptor = lock["implementation_files"]["collect_outcome_regret_pairs.py"]
    new_fingerprint = _validate_new_collection_manifest(
        manifest=new_manifest,
        audit=new_audit,
        plan_path=paths["new_pair_plan"],
        expected_collector_source_sha256=str(collector_descriptor["sha256"]),
    )
    new_plan = read_jsonl(paths["new_pair_plan"])
    new_pairs_rows = read_jsonl(new_pairs_path)
    new_judge_rows = read_jsonl(new_judge_path)
    new_score_rows = read_jsonl(new_score_path)
    new_pair_index = validate_collected_pairs(
        pairs=new_pairs_rows,
        plan=new_plan,
        configuration_fingerprint=new_fingerprint,
        label="new targeted collection",
    )
    new_outcomes = bind_judged_outcomes(
        pair_index=new_pair_index,
        judge_input=new_judge_rows,
        harmful_scores=new_score_rows,
        expected_judge_protocol=expected_judge,
        label="new targeted judge",
    )
    aliases = read_jsonl(paths["reuse_aliases"])
    resolved = validate_aliases_and_resolve(
        aliases=aliases,
        lock=lock,
        validation_by_id=validation_by_id,
        existing_pairs=old_pair_index,
        existing_outcomes=old_outcomes,
        new_pairs=new_pair_index,
        new_outcomes=new_outcomes,
    )
    outcome_by_candidate = {row["candidate_id"]: row for row in resolved}
    metrics = {
        "action_q": exact_policy_metrics(
            data=data,
            risk=data.q,
            threshold=ACTION_Q_THRESHOLD,
            outcomes_by_candidate=outcome_by_candidate,
        ),
        "outcome_q_times_s": exact_policy_metrics(
            data=data,
            risk=data.r,
            threshold=OUTCOME_Q_TIMES_S_THRESHOLD,
            outcomes_by_candidate=outcome_by_candidate,
        ),
    }
    for name in POLICY_NAMES:
        if metrics[name]["rows"] != EXPECTED_PRODUCTION_CONTRACT["legacy_validation_rows"]:
            raise ValueError("Exact audit finite-universe row count drifted.")
        if metrics[name]["selected_mismatch_candidate_ids"] != recomputed_policies[name][
            "selected_mismatch_candidate_ids"
        ]:
            raise ValueError(f"Exact audit selected mismatch set drifted for {name}.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite targeted audit output: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        resolved_path = output_dir / "resolved_selected_mismatch_outcomes.jsonl"
        summary_path = output_dir / "targeted_exact_audit_summary.json"
        _write_jsonl(resolved_path, resolved)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "analysis": "post_hoc_development_targeted_exact_single_realization_outcome_audit_v1",
            "fresh_test_used": False,
            "deployment_ready": False,
            "guarantees_harmful_score_non_degradation": False,
            "threshold_tuning_used_targeted_outcome_labels": False,
            "targeted_labels_used_for_gate_fitting": False,
            "finite_universe_rows": int(data.row_count),
            "record_count": int(len(np.unique(data.record_names))),
            "policies": metrics,
            "outcome_q_times_s_minus_action_q": _metric_difference(
                metrics["outcome_q_times_s"], metrics["action_q"]
            ),
            "coverage": {
                "selected_union_candidates": len(resolved),
                "canonical_pair_identities": len(
                    {row["canonical_branch_identity_sha256"] for row in resolved}
                ),
                "existing_resolved_candidates": sum(
                    row["outcome_source"] == "existing" for row in resolved
                ),
                "new_resolved_candidates": sum(
                    row["outcome_source"] == "new" for row in resolved
                ),
                "new_target_pairs": len(new_pair_index),
                "new_target_calls": 2 * len(new_pair_index),
                "inverse_probability_weighting_used": False,
                "all_selected_mismatches_resolved": True,
            },
            "metric_semantics": lock["metric_contract"],
            "input_files": {
                "targeted_audit_lock": {
                    "path": str(lock_path),
                    "sha256": sha256_file(lock_path),
                },
                "new_collection_manifest": {
                    "path": str(new_manifest_path),
                    "sha256": sha256_file(new_manifest_path),
                },
                "new_collection_audit": {
                    "path": str(new_audit_path),
                    "sha256": sha256_file(new_audit_path),
                },
                "new_pairs": {"path": str(new_pairs_path), "sha256": sha256_file(new_pairs_path)},
                "new_judge_input": {
                    "path": str(new_judge_path),
                    "sha256": sha256_file(new_judge_path),
                },
                "new_harmful_score": {
                    "path": str(new_score_path),
                    "sha256": sha256_file(new_score_path),
                },
            },
            "output_files": {
                "resolved_selected_mismatch_outcomes": {
                    "file": resolved_path.name,
                    "sha256": sha256_file(resolved_path),
                }
            },
            "limitations": [
                "This is a post-hoc development audit, not a fresh locked test.",
                "Exact means complete finite-universe coverage for the two frozen policies conditional on one realized continuation and one judge score.",
                "The result is not an expected-outcome estimate and does not measure target or judge noise.",
                "The result cannot guarantee that deployed harmful score will not decrease.",
                "The 8 and 10 selected mismatch states are sparse and clustered, so no confirmatory generalization claim is made.",
            ],
        }
        summary["summary_payload_sha256"] = payload_sha256(summary)
        _write_json(summary_path, summary)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
