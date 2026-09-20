"""Exploratory development-only frontier for an outcome-aware risk gate.

The evaluator is deliberately read-only with respect to experiment inputs.  It
does not load a target model/cache, contact an API, select a deployment
threshold, or inspect a fresh split.  A planner manifest with
``fresh_test_used=false`` and a development-only fit summary must authenticate
all inputs before the state universe is read.

On the complete legacy-validation universe it compares

* ``q = 1 - P(partial action agrees with MC50)``; and
* ``r = q * s``, where ``s`` is the train-fitted conditional harmful-drop gate.

Every threshold stops at the first eligible K and otherwise falls through to
the MC50 reference.  Compute/action metrics are exact.  Harmful-drop, severity,
and signed-regret metrics use the paid mismatch sample with inverse stratum
probabilities.  Both Horvitz--Thompson population means and a mismatch-count
calibrated Hajek form are reported.

The matched-compute bracket is chosen solely from the frozen action gate's
mean K and the candidate risk scores.  Outcome labels never choose a threshold.
With only eight legacy-validation records, record bootstrap intervals and
leave-one-record-out ranges are descriptive pilot uncertainty, not locked-test
evidence or a complete design-based sampling interval.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
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

from training.anytime_proxy_mc import (  # noqa: E402
    POLICY_SPEC_HASH_FIELD,
    PORTABLE_STOPPER_PROTOCOL,
    policy_spec_payload_sha256,
    portable_linear_stopper_confidence,
)
from training.anytime_stopping_features import FEATURE_NAMES  # noqa: E402


SCHEMA_VERSION = 1
ALLOWED_INPUT_SPLITS = frozenset({"train", "legacy_validation"})
VALIDATION_SPLIT = "legacy_validation"
METRIC_FIELDS_FOR_UNCERTAINTY = (
    "mean_k",
    "median_k",
    "p90_k",
    "action_disagreement",
    "harmful_drop_ht",
    "harmful_drop_hajek",
    "severity_ht",
    "severity_hajek",
    "signed_regret_ht",
    "signed_regret_hajek",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only legacy-validation q versus q*s matched-compute "
            "frontier from frozen development artifacts."
        )
    )
    parser.add_argument("--plan_manifest", required=True)
    parser.add_argument("--state_universe", required=True)
    parser.add_argument("--fit_summary", required=True)
    parser.add_argument("--pair_predictions", required=True)
    parser.add_argument("--outcome_gate", required=True)
    parser.add_argument("--policy_spec", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--primary_policy", default="delta_0.01")
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args(argv)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_payload(value: Any, *, exclude: Iterable[str] = ()) -> str:
    excluded = set(exclude)
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key not in excluded}
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row at {path}:{line_number} is not an object.")
        rows.append(row)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def _resolve_recorded_path(container_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = container_path.parent / path
    return path.resolve()


def authenticate_inputs(
    *,
    plan_manifest_path: Path,
    state_universe_path: Path,
    fit_summary_path: Path,
    pair_predictions_path: Path,
    outcome_gate_path: Path,
    policy_spec_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate development artifacts before reading any state rows."""

    plan_manifest = read_json(plan_manifest_path)
    if plan_manifest.get("fresh_test_used") is not False:
        raise ValueError("Planner manifest is not explicitly development-only.")
    if plan_manifest.get("schema_version") != 1:
        raise ValueError("Unsupported planner manifest schema.")
    if plan_manifest.get("manifest_payload_sha256") != sha256_payload(
        plan_manifest, exclude={"manifest_payload_sha256"}
    ):
        raise ValueError("Planner manifest payload hash mismatch.")
    if plan_manifest.get("state_universe_sha256") != sha256_file(state_universe_path):
        raise ValueError("State-universe hash differs from the planner manifest.")
    recorded_universe = plan_manifest.get("state_universe_file")
    if isinstance(recorded_universe, str):
        expected = _resolve_recorded_path(plan_manifest_path, recorded_universe)
        if expected != state_universe_path:
            raise ValueError("State-universe path differs from the planner manifest.")

    fit_summary = read_json(fit_summary_path)
    if fit_summary.get("fresh_test_used") is not False:
        raise ValueError("Outcome-gate fit summary is not explicitly development-only.")
    if fit_summary.get("deployment_ready") is not False:
        raise ValueError("Outcome-gate fit summary unexpectedly declares deployment readiness.")
    if fit_summary.get("summary_payload_sha256") != sha256_payload(
        fit_summary, exclude={"summary_payload_sha256"}
    ):
        raise ValueError("Outcome-gate fit summary payload hash mismatch.")
    input_files = fit_summary.get("input_files")
    output_files = fit_summary.get("output_files")
    if not isinstance(input_files, dict) or not isinstance(output_files, dict):
        raise ValueError("Fit summary lacks authenticated input/output files.")
    recorded_state = input_files.get("state_universe")
    if not isinstance(recorded_state, dict) or recorded_state.get("sha256") != sha256_file(
        state_universe_path
    ):
        raise ValueError("Fit summary/state-universe hash mismatch.")
    if _resolve_recorded_path(fit_summary_path, str(recorded_state.get("path", ""))) != state_universe_path:
        raise ValueError("Fit summary/state-universe path mismatch.")
    recorded_pairs = output_files.get("pair_labels_predictions")
    if not isinstance(recorded_pairs, dict) or recorded_pairs.get("sha256") != sha256_file(
        pair_predictions_path
    ):
        raise ValueError("Fit summary/pair-predictions hash mismatch.")
    if _resolve_recorded_path(fit_summary_path, str(recorded_pairs.get("file", ""))) != pair_predictions_path:
        raise ValueError("Fit summary/pair-predictions path mismatch.")
    recorded_gate = output_files.get("outcome_gate")
    if not isinstance(recorded_gate, dict) or recorded_gate.get("sha256") != sha256_file(
        outcome_gate_path
    ):
        raise ValueError("Fit summary/outcome-gate hash mismatch.")
    if _resolve_recorded_path(fit_summary_path, str(recorded_gate.get("file", ""))) != outcome_gate_path:
        raise ValueError("Fit summary/outcome-gate path mismatch.")

    outcome_gate = read_json(outcome_gate_path)
    if outcome_gate.get("schema_version") != 1:
        raise ValueError("Unsupported outcome-gate schema.")
    if outcome_gate.get("model_payload_sha256") != sha256_payload(
        outcome_gate, exclude={"model_payload_sha256"}
    ):
        raise ValueError("Outcome-gate portable payload hash mismatch.")
    _validate_portable_model(outcome_gate, expected_schema=1, label="outcome gate")

    policy = read_json(policy_spec_path)
    if policy.get(POLICY_SPEC_HASH_FIELD) != policy_spec_payload_sha256(policy):
        raise ValueError("Frozen action-policy payload hash mismatch.")
    if fit_summary.get("frozen_action_policy_payload_sha256") != policy.get(
        POLICY_SPEC_HASH_FIELD
    ):
        raise ValueError("Fit summary/action-policy payload hash mismatch.")
    recorded_policy = input_files.get("policy_spec")
    if (
        not isinstance(recorded_policy, dict)
        or recorded_policy.get("sha256") != sha256_file(policy_spec_path)
        or _resolve_recorded_path(
            fit_summary_path, str(recorded_policy.get("path", ""))
        )
        != policy_spec_path
    ):
        raise ValueError("Fit summary/action-policy file provenance mismatch.")
    if plan_manifest.get("policy_spec_sha256") != sha256_file(policy_spec_path):
        raise ValueError("Planner/action-policy file hash mismatch.")
    if policy.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Frozen action policy uses a different feature contract.")
    stopper = policy.get("linear_stopper")
    if not isinstance(stopper, dict):
        raise ValueError("Frozen action policy has no portable stopper.")
    _validate_portable_model(stopper, expected_schema=2, label="action gate")
    return plan_manifest, fit_summary, outcome_gate, policy


def _validate_portable_model(
    model: Mapping[str, Any], *, expected_schema: int, label: str
) -> None:
    if (
        model.get("schema_version") != expected_schema
        or model.get("scoring_protocol") != PORTABLE_STOPPER_PROTOCOL
        or model.get("classes") != [0, 1]
    ):
        raise ValueError(f"Unsupported portable {label} payload.")
    if model.get("feature_names", list(FEATURE_NAMES)) != list(FEATURE_NAMES):
        raise ValueError(f"Portable {label} uses a different feature contract.")
    portable_linear_stopper_confidence(
        [0.0] * len(FEATURE_NAMES),
        scaler_mean=model.get("scaler_mean", ()),
        scaler_scale=model.get("scaler_scale", ()),
        coefficient=model.get("coefficient", ()),
        intercept=model.get("intercept", float("nan")),
    )


def portable_score(model: Mapping[str, Any], features: Sequence[float]) -> float:
    return portable_linear_stopper_confidence(
        features,
        scaler_mean=model["scaler_mean"],
        scaler_scale=model["scaler_scale"],
        coefficient=model["coefficient"],
        intercept=model["intercept"],
    )


def validate_universe_rows(
    universe: Sequence[dict[str, Any]],
    *,
    plan_manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], tuple[int, ...], int]:
    """Return the complete legacy-validation universe after fail-closed checks."""

    splits = {row.get("split") for row in universe}
    unexpected = splits - ALLOWED_INPUT_SPLITS
    if unexpected:
        raise ValueError(
            f"Refusing state universe with fresh/test/unknown splits: {sorted(unexpected)!r}."
        )
    if VALIDATION_SPLIT not in splits:
        raise ValueError("State universe has no legacy_validation split.")
    policy_budgets = tuple(int(value) for value in policy.get("budgets", ()))
    paid_budgets = tuple(int(value) for value in plan_manifest.get("paid_budgets", ()))
    if (
        len(policy_budgets) < 2
        or policy_budgets[-1] <= 0
        or paid_budgets != policy_budgets[:-1]
    ):
        raise ValueError("Planner paid budgets do not match the frozen policy schedule.")
    terminal_budget = policy_budgets[-1]
    seen_ids: set[str] = set()
    validation: list[dict[str, Any]] = []
    for index, row in enumerate(universe):
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"Universe row {index} lacks candidate_id.")
        if candidate_id in seen_ids:
            raise ValueError(f"Duplicate universe candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        if row.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError(f"Universe row {candidate_id} violates feature contract.")
        features = np.asarray(row.get("features"), dtype=np.float64)
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise ValueError(f"Universe row {candidate_id} has invalid features.")
        if int(row.get("budget", -1)) not in paid_budgets:
            raise ValueError(f"Universe row {candidate_id} has an unexpected budget.")
        mismatch = row.get("action_mismatch")
        if not isinstance(mismatch, bool):
            raise ValueError(f"Universe row {candidate_id} lacks a Boolean mismatch flag.")
        if mismatch != (int(row["partial_action"]) != int(row["reference_action"])):
            raise ValueError(f"Universe row {candidate_id} action mismatch is inconsistent.")
        if row["split"] == VALIDATION_SPLIT:
            validation.append(row)
    expected_files = set(plan_manifest.get("legacy_validation_files", ()))
    observed_files = {str(row["record_name"]) for row in validation}
    if not expected_files or observed_files != expected_files:
        raise ValueError(
            "Legacy-validation universe records differ from the authenticated planner split."
        )
    split_report = (plan_manifest.get("split_reports") or {}).get(VALIDATION_SPLIT)
    if not isinstance(split_report, dict):
        raise ValueError("Planner manifest lacks a legacy-validation split report.")
    if int(split_report.get("states", -1)) != len(validation):
        raise ValueError("Legacy-validation state count differs from planner report.")
    observed_rows = len({str(row["row_key"]) for row in validation})
    if int(split_report.get("rows", -1)) != observed_rows:
        raise ValueError("Legacy-validation row count differs from planner report.")
    observed_mismatches = sum(bool(row["action_mismatch"]) for row in validation)
    if int(split_report.get("mismatch_states", -1)) != observed_mismatches:
        raise ValueError("Legacy-validation mismatch count differs from planner report.")
    return validation, paid_budgets, terminal_budget


def validate_pair_labels(
    pair_rows: Sequence[dict[str, Any]],
    *,
    validation_universe: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float | int]]]:
    """Validate judged validation mismatches and derive actual stratum propensities."""

    universe_by_id = {row["candidate_id"]: row for row in validation_universe}
    validation_pairs: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(pair_rows):
        split = row.get("split")
        if split not in ALLOWED_INPUT_SPLITS:
            raise ValueError("Pair predictions contain a fresh/test/unknown split.")
        if split != VALIDATION_SPLIT:
            continue
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in universe_by_id:
            raise ValueError(f"Validation label row {index} is absent from its universe.")
        if candidate_id in validation_pairs:
            raise ValueError(f"Duplicate validation label for candidate {candidate_id}.")
        source = universe_by_id[candidate_id]
        for field in (
            "record_name",
            "dataset_idx",
            "row_key",
            "position",
            "position_bin",
            "budget",
            "partial_action",
            "reference_action",
            "feature_names",
            "features",
        ):
            if row.get(field) != source.get(field):
                raise ValueError(
                    f"Pair/universe mismatch for {candidate_id}, field {field!r}."
                )
        if source.get("action_mismatch") is not True:
            raise ValueError(f"Paid label {candidate_id} is not an action mismatch.")
        for field in (
            "partial_harmful_score",
            "reference_harmful_score",
            "signed_regret",
            "harmful_score_drop",
            "severity",
        ):
            if isinstance(row.get(field), bool) or not isinstance(row.get(field), int):
                raise ValueError(f"Pair {candidate_id} has invalid outcome field {field}.")
        d = int(row["reference_harmful_score"]) - int(row["partial_harmful_score"])
        if (
            not 1 <= int(row["partial_harmful_score"]) <= 5
            or not 1 <= int(row["reference_harmful_score"]) <= 5
            or int(row["signed_regret"]) != d
            or int(row["harmful_score_drop"]) != int(d >= 1)
            or int(row["severity"]) != max(0, d)
        ):
            raise ValueError(f"Pair {candidate_id} has inconsistent outcome labels.")
        if row.get("conditional_prediction_provenance") != "train_refit":
            raise ValueError(
                f"Legacy-validation pair {candidate_id} was not scored by the train refit."
            )
        validation_pairs[candidate_id] = row

    def stratum(row: Mapping[str, Any]) -> str:
        return (
            f"{VALIDATION_SPLIT}:k{int(row['budget'])}:"
            f"{str(row['position_bin'])}"
        )

    eligible = Counter(
        stratum(row) for row in validation_universe if row["action_mismatch"]
    )
    selected = Counter(stratum(universe_by_id[key]) for key in validation_pairs)
    propensities: dict[str, dict[str, float | int]] = {}
    for name, total in eligible.items():
        paid = int(selected[name])
        if paid <= 0:
            raise ValueError(
                f"Mismatch stratum {name} has no paid legacy-validation labels; "
                "IPW positivity is absent."
            )
        fraction = float(paid) / float(total)
        propensities[name] = {
            "eligible_mismatches": int(total),
            "paid_mismatches": paid,
            "inclusion_fraction": fraction,
            "inverse_inclusion_weight": 1.0 / fraction,
        }
    for candidate_id, row in validation_pairs.items():
        name = stratum(universe_by_id[candidate_id])
        expected = float(propensities[name]["inverse_inclusion_weight"])
        recorded = float(row["inverse_inclusion_weight"])
        if not math.isclose(expected, recorded, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(
                f"Pair {candidate_id} has stale/incorrect inverse inclusion weight."
            )
    return validation_pairs, propensities


@dataclass(frozen=True)
class FrontierData:
    row_keys: tuple[str, ...]
    record_names: np.ndarray
    budgets: np.ndarray
    terminal_budget: int
    q: np.ndarray
    r: np.ndarray
    mismatches: np.ndarray
    candidate_ids: np.ndarray
    strata: np.ndarray
    label_observed: np.ndarray
    labels: np.ndarray
    severity: np.ndarray
    signed_regret: np.ndarray
    weights: np.ndarray

    @property
    def row_count(self) -> int:
        return len(self.row_keys)

    @property
    def stage_count(self) -> int:
        return len(self.budgets)


def build_frontier_data(
    validation_universe: Sequence[dict[str, Any]],
    *,
    paid_budgets: tuple[int, ...],
    terminal_budget: int,
    action_stopper: Mapping[str, Any],
    outcome_gate: Mapping[str, Any],
    pair_labels: Mapping[str, Mapping[str, Any]],
    propensities: Mapping[str, Mapping[str, Any]],
) -> FrontierData:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in validation_universe:
        by_row[str(state["row_key"])].append(state)
    row_keys = tuple(sorted(by_row))
    rows = len(row_keys)
    stages = len(paid_budgets)
    q = np.empty((rows, stages), dtype=np.float64)
    r = np.empty((rows, stages), dtype=np.float64)
    mismatches = np.empty((rows, stages), dtype=bool)
    candidate_ids = np.empty((rows, stages), dtype=object)
    strata = np.empty((rows, stages), dtype=object)
    label_observed = np.zeros((rows, stages), dtype=bool)
    labels = np.zeros((rows, stages), dtype=np.float64)
    severity = np.zeros((rows, stages), dtype=np.float64)
    signed = np.zeros((rows, stages), dtype=np.float64)
    weights = np.zeros((rows, stages), dtype=np.float64)
    record_names = np.empty(rows, dtype=object)

    for row_index, row_key in enumerate(row_keys):
        states = sorted(by_row[row_key], key=lambda row: int(row["budget"]))
        observed_budgets = tuple(int(row["budget"]) for row in states)
        if observed_budgets != paid_budgets:
            raise ValueError(
                f"Row {row_key} does not have exactly one state per paid budget: "
                f"{observed_budgets!r}."
            )
        invariant_fields = ("record_name", "dataset_idx", "position", "reference_action")
        for field in invariant_fields:
            values = {row[field] for row in states}
            if len(values) != 1:
                raise ValueError(f"Row {row_key} changes invariant field {field!r} across K.")
        record_names[row_index] = states[0]["record_name"]
        for stage, state in enumerate(states):
            features = np.asarray(state["features"], dtype=np.float32).astype(np.float64)
            agreement = portable_score(action_stopper, features)
            conditional = portable_score(outcome_gate, features)
            q[row_index, stage] = 1.0 - agreement
            r[row_index, stage] = q[row_index, stage] * conditional
            mismatches[row_index, stage] = bool(state["action_mismatch"])
            candidate_id = str(state["candidate_id"])
            candidate_ids[row_index, stage] = candidate_id
            name = (
                f"{VALIDATION_SPLIT}:k{int(state['budget'])}:"
                f"{str(state['position_bin'])}"
            )
            strata[row_index, stage] = name
            pair = pair_labels.get(candidate_id)
            if pair is not None:
                if not mismatches[row_index, stage]:
                    raise ValueError("An equal-action state unexpectedly has a paid label.")
                label_observed[row_index, stage] = True
                labels[row_index, stage] = float(pair["harmful_score_drop"])
                severity[row_index, stage] = float(pair["severity"])
                signed[row_index, stage] = float(pair["signed_regret"])
                weights[row_index, stage] = float(
                    propensities[name]["inverse_inclusion_weight"]
                )
    if len(np.unique(record_names)) < 2:
        raise ValueError("Frontier uncertainty requires at least two validation records.")
    return FrontierData(
        row_keys=row_keys,
        record_names=record_names,
        budgets=np.asarray(paid_budgets, dtype=np.int64),
        terminal_budget=int(terminal_budget),
        q=q,
        r=r,
        mismatches=mismatches,
        candidate_ids=candidate_ids,
        strata=strata,
        label_observed=label_observed,
        labels=labels,
        severity=severity,
        signed_regret=signed,
        weights=weights,
    )


def select_stages(risk: np.ndarray, threshold: float) -> np.ndarray:
    if risk.ndim != 2 or not np.isfinite(risk).all():
        raise ValueError("Risk matrix must be finite and two-dimensional.")
    if not math.isfinite(threshold):
        raise ValueError("Risk threshold must be finite.")
    eligible = risk <= float(threshold)
    has_stop = eligible.any(axis=1)
    first = eligible.argmax(axis=1)
    return np.where(has_stop, first, risk.shape[1]).astype(np.int64)


def threshold_candidates(risk: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(risk, dtype=np.float64).reshape(-1))
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("Risk scores must be probabilities.")
    # -1 represents the terminal-only policy and is finite/JSON-safe.
    return np.concatenate((np.asarray([-1.0]), values))


def _selected_arrays(
    data: FrontierData, selected_stages: np.ndarray
) -> dict[str, np.ndarray]:
    if selected_stages.shape != (data.row_count,):
        raise ValueError("Selected-stage vector has the wrong shape.")
    terminal = selected_stages == data.stage_count
    safe_stage = np.minimum(selected_stages, data.stage_count - 1)
    row_ids = np.arange(data.row_count)
    selected_k = np.where(
        terminal,
        data.terminal_budget,
        data.budgets[safe_stage],
    ).astype(np.float64)
    mismatch = np.where(
        terminal, False, data.mismatches[row_ids, safe_stage]
    ).astype(bool)
    observed = np.where(
        terminal, False, data.label_observed[row_ids, safe_stage]
    ).astype(bool)
    return {
        "selected_k": selected_k,
        "mismatch": mismatch,
        "observed": observed,
        "label": np.where(terminal, 0.0, data.labels[row_ids, safe_stage]),
        "severity": np.where(terminal, 0.0, data.severity[row_ids, safe_stage]),
        "signed_regret": np.where(
            terminal, 0.0, data.signed_regret[row_ids, safe_stage]
        ),
        "weight": np.where(terminal, 0.0, data.weights[row_ids, safe_stage]),
        "stratum": np.where(terminal, "terminal_kmax", data.strata[row_ids, safe_stage]),
        "candidate_id": np.where(
            terminal, None, data.candidate_ids[row_ids, safe_stage]
        ),
    }


def aggregate_policy(
    data: FrontierData,
    selected_stages: np.ndarray,
    *,
    row_indices: np.ndarray | None = None,
    include_coverage: bool = True,
) -> dict[str, Any]:
    arrays = _selected_arrays(data, selected_stages)
    if row_indices is None:
        row_indices = np.arange(data.row_count, dtype=np.int64)
    else:
        row_indices = np.asarray(row_indices, dtype=np.int64)
    if row_indices.ndim != 1 or len(row_indices) == 0:
        raise ValueError("Policy aggregation requires at least one row.")
    selected_k = arrays["selected_k"][row_indices]
    mismatch = arrays["mismatch"][row_indices]
    observed = arrays["observed"][row_indices] & mismatch
    weight = arrays["weight"][row_indices]
    weighted_observed = weight * observed.astype(np.float64)
    population_size = float(len(row_indices))
    mismatch_rate = float(mismatch.mean())
    weight_total = float(weighted_observed.sum())

    def estimates(values: np.ndarray) -> tuple[float, float | None]:
        selected_values = values[row_indices]
        ht = float(np.sum(weighted_observed * selected_values) / population_size)
        if mismatch_rate == 0.0:
            hajek = 0.0
        elif weight_total > 0.0:
            hajek = float(
                mismatch_rate
                * np.sum(weighted_observed * selected_values)
                / weight_total
            )
        else:
            hajek = None
        return ht, hajek

    drop_ht, drop_hajek = estimates(arrays["label"])
    severity_ht, severity_hajek = estimates(arrays["severity"])
    signed_ht, signed_hajek = estimates(arrays["signed_regret"])
    result: dict[str, Any] = {
        "rows": int(len(row_indices)),
        "mean_k": float(selected_k.mean()),
        "median_k": float(np.median(selected_k)),
        "p90_k": float(np.quantile(selected_k, 0.9)),
        "sample_reduction": float(1.0 - selected_k.mean() / data.terminal_budget),
        "action_disagreement": mismatch_rate,
        "selected_mismatch_states": int(mismatch.sum()),
        "paid_selected_mismatch_states": int(observed.sum()),
        "harmful_drop_ht": drop_ht,
        "harmful_drop_hajek": drop_hajek,
        "severity_ht": severity_ht,
        "severity_hajek": severity_hajek,
        "signed_regret_ht": signed_ht,
        "signed_regret_hajek": signed_hajek,
    }
    if include_coverage:
        selected_strata = set(arrays["stratum"][mismatch].tolist())
        observed_strata = set(arrays["stratum"][observed].tolist())
        positive_weights = weight[observed]
        result["coverage"] = {
            "raw_paid_fraction_of_selected_mismatches": (
                float(observed.sum() / mismatch.sum()) if mismatch.any() else 1.0
            ),
            "ipw_reconstructed_selected_mismatches": weight_total,
            "ipw_reconstruction_ratio": (
                float(weight_total / mismatch.sum()) if mismatch.any() else 1.0
            ),
            "effective_sample_size": (
                float(weight_total**2 / np.square(positive_weights).sum())
                if len(positive_weights)
                else 0.0
            ),
            "max_inverse_weight": (
                float(positive_weights.max()) if len(positive_weights) else None
            ),
            "selected_mismatch_strata": sorted(selected_strata),
            "strata_with_paid_overlap": sorted(observed_strata),
            "zero_paid_overlap_strata": sorted(selected_strata - observed_strata),
            "all_selected_strata_have_paid_overlap": selected_strata <= observed_strata,
        }
    return result


def frontier_points(
    data: FrontierData, *, risk: np.ndarray, family: str
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    points: list[dict[str, Any]] = []
    selections: list[np.ndarray] = []
    previous_selection: np.ndarray | None = None
    for threshold in threshold_candidates(risk):
        selected = select_stages(risk, float(threshold))
        # Risk ties can occasionally yield an unchanged complete policy.  Keep
        # only behavior-changing thresholds so the crossing bracket is unique.
        if previous_selection is not None and np.array_equal(selected, previous_selection):
            continue
        point = {
            "schema_version": SCHEMA_VERSION,
            "policy_family": family,
            "threshold_semantics": "stop_at_first_stage_with_risk_lte_threshold",
            "risk_threshold": float(threshold),
            **aggregate_policy(data, selected),
        }
        points.append(point)
        selections.append(selected)
        previous_selection = selected
    return points, selections


def matched_compute_bracket(
    points: Sequence[Mapping[str, Any]],
    selections: Sequence[np.ndarray],
    *,
    target_mean_k: float,
) -> dict[str, tuple[dict[str, Any], np.ndarray]]:
    """Return adjacent threshold-crossing points without consulting outcomes."""

    if len(points) != len(selections) or not points:
        raise ValueError("Frontier points/selections are empty or inconsistent.")
    means = np.asarray([float(point["mean_k"]) for point in points])
    if np.any(np.diff(means) > 1e-12):
        raise ValueError("Risk-threshold frontier mean K is not monotone.")
    crossings = np.nonzero(means <= target_mean_k + 1e-12)[0]
    if not len(crossings):
        index = len(points) - 1
        return {"more_expensive": (dict(points[index]), selections[index])}
    cheaper = int(crossings[0])
    if math.isclose(means[cheaper], target_mean_k, rel_tol=0.0, abs_tol=1e-12):
        return {"exact": (dict(points[cheaper]), selections[cheaper])}
    bracket = {"cheaper": (dict(points[cheaper]), selections[cheaper])}
    if cheaper > 0:
        bracket["more_expensive"] = (
            dict(points[cheaper - 1]),
            selections[cheaper - 1],
        )
    return bracket


def _finite_metric(metric: Mapping[str, Any], field: str) -> float | None:
    value = metric.get(field)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _summarize_draws(values: Sequence[float | None]) -> dict[str, Any]:
    valid = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if not len(valid):
        return {"valid_draws": 0, "interval_95": None, "standard_error": None}
    return {
        "valid_draws": int(len(valid)),
        "interval_95": [
            float(np.quantile(valid, 0.025)),
            float(np.quantile(valid, 0.975)),
        ],
        "standard_error": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
    }


def uncertainty_analysis(
    data: FrontierData,
    *,
    policies: Mapping[str, np.ndarray],
    reference_name: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Record-cluster bootstrap and exact leave-one-record-out sensitivity."""

    if bootstrap_replicates < 0:
        raise ValueError("Bootstrap replicate count must be non-negative.")
    if reference_name not in policies:
        raise ValueError("Uncertainty reference policy is absent.")
    unique_records = np.unique(data.record_names)
    record_rows = {
        name: np.nonzero(data.record_names == name)[0] for name in unique_records
    }

    def evaluate_indices(indices: np.ndarray) -> dict[str, dict[str, Any]]:
        return {
            name: aggregate_policy(
                data, selected, row_indices=indices, include_coverage=False
            )
            for name, selected in policies.items()
        }

    leave_one_out: list[dict[str, Any]] = []
    all_rows = np.arange(data.row_count)
    for held_out in unique_records:
        indices = all_rows[data.record_names != held_out]
        metrics = evaluate_indices(indices)
        reference = metrics[reference_name]
        differences = {
            name: {
                field: (
                    _finite_metric(metric, field) - _finite_metric(reference, field)
                    if _finite_metric(metric, field) is not None
                    and _finite_metric(reference, field) is not None
                    else None
                )
                for field in METRIC_FIELDS_FOR_UNCERTAINTY
            }
            for name, metric in metrics.items()
            if name != reference_name
        }
        leave_one_out.append(
            {
                "held_out_record": str(held_out),
                "metrics": metrics,
                "paired_difference_vs_reference": differences,
            }
        )

    def leave_one_out_range(values: Iterable[float | None]) -> dict[str, Any]:
        valid = [float(value) for value in values if value is not None]
        return {
            "valid_deletions": len(valid),
            "min": min(valid) if valid else None,
            "max": max(valid) if valid else None,
        }

    leave_one_out_ranges = {
        name: {
            field: leave_one_out_range(
                _finite_metric(draw["metrics"][name], field)
                for draw in leave_one_out
            )
            for field in METRIC_FIELDS_FOR_UNCERTAINTY
        }
        for name in policies
    }
    leave_one_out_difference_ranges = {
        name: {
            field: leave_one_out_range(
                draw["paired_difference_vs_reference"][name][field]
                for draw in leave_one_out
            )
            for field in METRIC_FIELDS_FOR_UNCERTAINTY
        }
        for name in policies
        if name != reference_name
    }

    generator = np.random.default_rng(seed)
    policy_draws = {
        name: {field: [] for field in METRIC_FIELDS_FOR_UNCERTAINTY}
        for name in policies
    }
    difference_draws = {
        name: {field: [] for field in METRIC_FIELDS_FOR_UNCERTAINTY}
        for name in policies
        if name != reference_name
    }
    for _ in range(bootstrap_replicates):
        sampled = generator.choice(unique_records, size=len(unique_records), replace=True)
        indices = np.concatenate([record_rows[name] for name in sampled])
        metrics = evaluate_indices(indices)
        reference = metrics[reference_name]
        for name, metric in metrics.items():
            for field in METRIC_FIELDS_FOR_UNCERTAINTY:
                value = _finite_metric(metric, field)
                policy_draws[name][field].append(value)
                if name != reference_name:
                    reference_value = _finite_metric(reference, field)
                    difference_draws[name][field].append(
                        value - reference_value
                        if value is not None and reference_value is not None
                        else None
                    )
    return {
        "cluster_field": "record_name",
        "cluster_count": int(len(unique_records)),
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(seed),
        "bootstrap_intervals": {
            name: {
                field: _summarize_draws(values) for field, values in fields.items()
            }
            for name, fields in policy_draws.items()
        },
        "paired_difference_bootstrap_intervals_vs_reference": {
            name: {
                field: _summarize_draws(values) for field, values in fields.items()
            }
            for name, fields in difference_draws.items()
        },
        "leave_one_record_out": leave_one_out,
        "leave_one_record_out_ranges": leave_one_out_ranges,
        "paired_difference_leave_one_record_out_ranges_vs_reference": (
            leave_one_out_difference_ranges
        ),
        "caveat": (
            "Legacy validation has few record clusters (expected eight). Percentile "
            "cluster bootstrap and leave-one-record-out results are descriptive; "
            "they do not capture all fixed-stratum label-sampling uncertainty and "
            "are not fresh-test confidence guarantees."
        ),
    }


def evaluate_frontier(
    *,
    data: FrontierData,
    policy: Mapping[str, Any],
    primary_policy: str,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    policies = policy.get("policies")
    primary = policies.get(primary_policy) if isinstance(policies, dict) else None
    if not isinstance(primary, dict) or not str(primary.get("status", "")).startswith(
        "selected_on_legacy_validation"
    ):
        raise ValueError(f"Frozen primary action policy is not selected: {primary_policy}")
    agreement_threshold = float(primary["threshold"])
    if not 0.0 <= agreement_threshold <= 1.0:
        raise ValueError("Frozen action-agreement threshold is outside [0,1].")
    q_threshold = 1.0 - agreement_threshold
    frozen_selection = select_stages(data.q, q_threshold)
    frozen_metrics = {
        "schema_version": SCHEMA_VERSION,
        "policy_family": "frozen_action_q",
        "policy_name": primary_policy,
        "source_agreement_threshold": agreement_threshold,
        "risk_threshold": q_threshold,
        "threshold_semantics": "stop_at_first_stage_with_q_lte_threshold",
        **aggregate_policy(data, frozen_selection),
    }

    q_frontier, _q_selections = frontier_points(data, risk=data.q, family="action_q")
    outcome_frontier, outcome_selections = frontier_points(
        data, risk=data.r, family="outcome_q_times_s"
    )
    bracket = matched_compute_bracket(
        outcome_frontier,
        outcome_selections,
        target_mean_k=float(frozen_metrics["mean_k"]),
    )
    uncertainty_policies: dict[str, np.ndarray] = {"frozen_action_q": frozen_selection}
    matched_summary: dict[str, Any] = {}
    for role, (point, selection) in bracket.items():
        name = f"matched_outcome_{role}"
        uncertainty_policies[name] = selection
        matched_summary[role] = {
            **point,
            "mean_k_difference_vs_frozen_action": float(point["mean_k"])
            - float(frozen_metrics["mean_k"]),
            "bracket_selection_uses_outcome_labels": False,
        }
    uncertainty = uncertainty_analysis(
        data,
        policies=uncertainty_policies,
        reference_name="frozen_action_q",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "legacy_validation_exploratory_outcome_risk_frontier_v1",
        "fresh_test_used": False,
        "deployment_ready": False,
        "threshold_selected": False,
        "threshold_tuning_used_outcome_labels": False,
        "row_count": int(data.row_count),
        "record_count": int(len(np.unique(data.record_names))),
        "paid_budgets": data.budgets.astype(int).tolist(),
        "terminal_budget": int(data.terminal_budget),
        "frozen_action_anchor": frozen_metrics,
        "matched_mean_k_target": float(frozen_metrics["mean_k"]),
        "matched_outcome_bracket": matched_summary,
        "frontier_point_counts": {
            "action_q": len(q_frontier),
            "outcome_q_times_s": len(outcome_frontier),
        },
        "uncertainty": uncertainty,
        "metric_semantics": {
            "harmful_drop": "1[H_reference-H_partial>=1]",
            "severity": "max(0,H_reference-H_partial)",
            "signed_regret": "H_reference-H_partial",
            "ht": "sum(observed/inclusion_probability*outcome)/all_prefix_rows",
            "hajek": (
                "exact_selected_mismatch_rate times IPW outcome mean among "
                "paid selected mismatches"
            ),
        },
        "limitations": [
            "This is an exploratory legacy-validation frontier, not a fresh locked test.",
            "The matched bracket is selected only by mean K; no deployment threshold is selected.",
            "One-shot base-policy continuation regret is not full deployed on-policy regret.",
            "The binary gate predicts drop occurrence, not positive-drop magnitude.",
            "IPW relies on uniform paid sampling within planner strata and can be unstable with sparse policy overlap.",
            "Only a small number of validation record clusters support uncertainty estimates.",
        ],
    }
    return q_frontier, outcome_frontier, summary


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
    if args.bootstrap_replicates < 0:
        raise ValueError("--bootstrap_replicates must be non-negative.")
    paths = {
        "plan_manifest": Path(args.plan_manifest).expanduser().resolve(),
        "state_universe": Path(args.state_universe).expanduser().resolve(),
        "fit_summary": Path(args.fit_summary).expanduser().resolve(),
        "pair_predictions": Path(args.pair_predictions).expanduser().resolve(),
        "outcome_gate": Path(args.outcome_gate).expanduser().resolve(),
        "policy_spec": Path(args.policy_spec).expanduser().resolve(),
    }
    plan_manifest, fit_summary, outcome_gate, policy = authenticate_inputs(
        plan_manifest_path=paths["plan_manifest"],
        state_universe_path=paths["state_universe"],
        fit_summary_path=paths["fit_summary"],
        pair_predictions_path=paths["pair_predictions"],
        outcome_gate_path=paths["outcome_gate"],
        policy_spec_path=paths["policy_spec"],
    )
    universe = read_jsonl(paths["state_universe"])
    pair_rows = read_jsonl(paths["pair_predictions"])
    validation, paid_budgets, terminal_budget = validate_universe_rows(
        universe, plan_manifest=plan_manifest, policy=policy
    )
    pair_labels, propensities = validate_pair_labels(
        pair_rows, validation_universe=validation
    )
    expected_validation_pairs = (fit_summary.get(VALIDATION_SPLIT) or {}).get("pairs")
    if int(expected_validation_pairs or -1) != len(pair_labels):
        raise ValueError(
            "Legacy-validation paid-pair count differs from the authenticated fit summary."
        )
    data = build_frontier_data(
        validation,
        paid_budgets=paid_budgets,
        terminal_budget=terminal_budget,
        action_stopper=policy["linear_stopper"],
        outcome_gate=outcome_gate,
        pair_labels=pair_labels,
        propensities=propensities,
    )
    q_frontier, outcome_frontier, summary = evaluate_frontier(
        data=data,
        policy=policy,
        primary_policy=args.primary_policy,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    summary["sampling_design"] = {
        "population": "all legacy-validation action-mismatch states by K/position stratum",
        "paid_validation_pairs": int(len(pair_labels)),
        "strata": dict(sorted(propensities.items())),
        "weight_source": "recomputed_from_authenticated_universe_and_paid_labels",
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frontier output: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        q_path = output_dir / "action_q_frontier.jsonl"
        outcome_path = output_dir / "outcome_q_times_s_frontier.jsonl"
        summary_path = output_dir / "frontier_summary.json"
        _write_jsonl(q_path, q_frontier)
        _write_jsonl(outcome_path, outcome_frontier)
        summary.update(
            {
                "input_files": {
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in paths.items()
                },
                "authenticated_plan_manifest_payload_sha256": plan_manifest.get(
                    "manifest_payload_sha256"
                ),
                "authenticated_fit_summary_payload_sha256": fit_summary.get(
                    "summary_payload_sha256"
                ),
                "output_files": {
                    "action_q_frontier": {
                        "file": q_path.name,
                        "sha256": sha256_file(q_path),
                    },
                    "outcome_q_times_s_frontier": {
                        "file": outcome_path.name,
                        "sha256": sha256_file(outcome_path),
                    },
                },
                "seed": int(args.seed),
            }
        )
        summary["summary_payload_sha256"] = sha256_payload(summary)
        _write_json(summary_path, summary)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
