"""Fit a pilot outcome-aware risk gate from paired one-shot continuations.

This program is intentionally a *development-only* analysis.  It joins the
judge output back to the exact collector rows by ``source_record_index``, forms
the signed harmful-score regret

    d = H(reference MC50 action) - H(partial-K action),

and models ``P(d >= 1 | the actions disagree, state features)``.  The frozen
action-agreement gate supplies ``q = 1 - P(actions agree)``; the proposed
unconditional risk score is ``q * s``, where ``s`` is the conditional model.

Only paid mismatch pairs from the planner's ``train`` split are used for model
selection/fitting.  The ``legacy_validation`` pairs are scored exactly once at
the end.  Inputs containing a fresh/test split fail closed, and this module
does not edit a deployed policy or read a fresh cache.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


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
ALLOWED_SPLITS = frozenset({"train", "legacy_validation"})
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a development-only conditional harmful-score-drop gate from "
            "paired one-shot continuation judgments."
        )
    )
    parser.add_argument("--state_universe", required=True)
    parser.add_argument("--paid_pair_plan", required=True)
    parser.add_argument("--judge_input", required=True)
    parser.add_argument("--harmful_score", required=True)
    parser.add_argument("--policy_spec", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--c_grid", default=",".join(map(str, DEFAULT_C_GRID)))
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


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


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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


def parse_c_grid(value: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    else:
        values = tuple(float(item) for item in value)
    if not values or any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise ValueError("C grid values must be finite and positive.")
    if len(set(values)) != len(values):
        raise ValueError("C grid values must be unique.")
    return tuple(sorted(values))


def _finite_features(row: Mapping[str, Any], *, context: str) -> np.ndarray:
    names = row.get("feature_names")
    if names != list(FEATURE_NAMES):
        raise ValueError(f"{context} does not use the frozen 17-feature contract.")
    values = np.asarray(row.get("features"), dtype=np.float64)
    if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError(f"{context} has invalid state features.")
    # Runtime features are materialized as float32 before portable scalar
    # scoring.  Reproduce that boundary here rather than retaining JSON float64.
    return values.astype(np.float32).astype(np.float64)


def _candidate_id(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("candidate_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} has no candidate_id.")
    return value


def validate_planner_inputs(
    universe: Sequence[dict[str, Any]],
    plan: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate paid mismatch rows as an exact subset of the frozen universe."""

    universe_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(universe):
        context = f"state universe row {index}"
        identifier = _candidate_id(row, context=context)
        if identifier in universe_by_id:
            raise ValueError(f"Duplicate universe candidate_id: {identifier}")
        split = row.get("split")
        if split not in ALLOWED_SPLITS:
            raise ValueError(
                f"Refusing non-development split {split!r} in state universe; "
                "fresh/test data are prohibited."
            )
        _finite_features(row, context=context)
        universe_by_id[identifier] = row

    paid_by_id: dict[str, dict[str, Any]] = {}
    compared_fields = (
        "split",
        "record_name",
        "dataset_idx",
        "row_key",
        "position",
        "position_bin",
        "budget",
        "partial_action",
        "reference_action",
        "action_mismatch",
        "feature_names",
        "features",
    )
    for index, row in enumerate(plan):
        context = f"paid plan row {index}"
        identifier = _candidate_id(row, context=context)
        if identifier in paid_by_id:
            raise ValueError(f"Duplicate paid candidate_id: {identifier}")
        source = universe_by_id.get(identifier)
        if source is None:
            raise ValueError(f"Paid candidate is absent from universe: {identifier}")
        if row.get("split") not in ALLOWED_SPLITS:
            raise ValueError("Fresh/test paid pairs are prohibited.")
        if row.get("action_mismatch") is not True:
            raise ValueError(f"Paid row {identifier} is not marked as a mismatch.")
        if int(row.get("partial_action", -1)) == int(row.get("reference_action", -1)):
            raise ValueError(f"Paid row {identifier} has equal actions.")
        _finite_features(row, context=context)
        for field in compared_fields:
            if row.get(field) != source.get(field):
                raise ValueError(
                    f"Paid/universe mismatch for {identifier}, field {field!r}."
                )
        paid_by_id[identifier] = row

    split_counts = Counter(row["split"] for row in plan)
    if split_counts["train"] == 0 or split_counts["legacy_validation"] == 0:
        raise ValueError("Paid plan must contain train and legacy_validation pairs.")
    return paid_by_id


def derive_inclusion_weights(
    universe: Sequence[Mapping[str, Any]],
    plan: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int | str]]:
    """Derive actual inverse stratum fractions, including fallback selections.

    Planner rows record the initial per-stratum fraction.  A deterministic
    fallback can add rows to a stratum, so recomputing ``selected / eligible``
    from the immutable universe and final plan is the auditable quantity.
    """

    def key(row: Mapping[str, Any]) -> tuple[str, int, str]:
        return (str(row["split"]), int(row["budget"]), str(row["position_bin"]))

    eligible = Counter(key(row) for row in universe if row.get("action_mismatch") is True)
    selected = Counter(key(row) for row in plan)
    weights: dict[str, dict[str, float | int | str]] = {}
    for row in plan:
        identifier = str(row["candidate_id"])
        stratum = key(row)
        total = int(eligible[stratum])
        chosen = int(selected[stratum])
        if total <= 0 or chosen <= 0 or chosen > total:
            raise ValueError(f"Invalid inclusion counts for stratum {stratum!r}.")
        fraction = float(chosen) / float(total)
        explicit = row.get("inclusion_fraction_within_stratum")
        # Initial planner fractions may differ only when fallback selection
        # changed the final stratum count.  Preserve both for the audit trail.
        if explicit is not None:
            explicit = float(explicit)
            if not math.isfinite(explicit) or explicit <= 0.0 or explicit > 1.0:
                raise ValueError(f"Invalid explicit inclusion fraction for {identifier}.")
        weights[identifier] = {
            "stratum": f"{stratum[0]}:k{stratum[1]}:{stratum[2]}",
            "eligible_mismatches": total,
            "selected_mismatches": chosen,
            "inclusion_fraction": fraction,
            "inverse_inclusion_weight": 1.0 / fraction,
            "planner_recorded_fraction": explicit,
        }
    return weights


def merge_judgments(
    *,
    plan: Sequence[dict[str, Any]],
    judge_input: Sequence[dict[str, Any]],
    harmful_scores: Sequence[dict[str, Any]],
    inclusion: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind each score to its source row and reduce two arms to one label."""

    plan_by_id = {str(row["candidate_id"]): row for row in plan}
    scored_input: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for score_index, score in enumerate(harmful_scores):
        source_index = score.get("source_record_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"Harmful-score row {score_index} has invalid source_record_index.")
        if source_index < 0 or source_index >= len(judge_input):
            raise ValueError(f"Harmful-score row {score_index} points outside judge input.")
        if source_index in scored_input:
            raise ValueError(f"Duplicate harmful score for source_record_index={source_index}.")
        source = judge_input[source_index]
        expected_source_hash = sha256_payload(source)
        if score.get("source_record_sha256") != expected_source_hash:
            raise ValueError(f"Judge/source hash mismatch at source index {source_index}.")
        if score.get("que") != source.get("prompt") or score.get("ans") != source.get(
            "completion"
        ):
            raise ValueError(f"Judge/source text mismatch at source index {source_index}.")
        value = score.get("duo_score")
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"Unscored/invalid duo_score at source index {source_index}.")
        scored_input[source_index] = (source, score)
    if set(scored_input) != set(range(len(judge_input))):
        missing = sorted(set(range(len(judge_input))) - set(scored_input))
        raise ValueError(f"Harmful-score file does not exactly cover judge input: {missing[:10]}.")

    by_pair: dict[str, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(dict)
    for source_index in range(len(judge_input)):
        source, score = scored_input[source_index]
        pair_id = source.get("pair_id")
        role = source.get("arm_role")
        if not isinstance(pair_id, str) or role not in {"partial", "reference"}:
            raise ValueError(f"Judge input row {source_index} lacks a valid pair/arm identity.")
        if role in by_pair[pair_id]:
            raise ValueError(f"Duplicate {role} arm in pair {pair_id}.")
        if source.get("logibreak_group_id") != pair_id:
            raise ValueError(f"Group/pair identity mismatch for source row {source_index}.")
        by_pair[pair_id][role] = (source, score)

    rows: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for pair_id in sorted(by_pair):
        arms = by_pair[pair_id]
        if set(arms) != {"partial", "reference"}:
            raise ValueError(f"Pair {pair_id} is not a complete two-arm group.")
        partial_source, partial_score = arms["partial"]
        reference_source, reference_score = arms["reference"]
        partial_plan = partial_source.get("plan_row")
        reference_plan = reference_source.get("plan_row")
        if not isinstance(partial_plan, dict) or partial_plan != reference_plan:
            raise ValueError(f"Pair {pair_id} arms do not share one planner row.")
        candidate_id = partial_plan.get("candidate_id")
        if candidate_id not in plan_by_id:
            raise ValueError(f"Pair {pair_id} is absent from the paid plan.")
        paid = plan_by_id[str(candidate_id)]
        if partial_plan != paid:
            raise ValueError(f"Collector planner payload changed for pair {pair_id}.")
        if candidate_id in seen_candidates:
            raise ValueError(f"Multiple collected pairs map to candidate {candidate_id}.")
        seen_candidates.add(str(candidate_id))
        for source, expected_role in (
            (partial_source, "partial"),
            (reference_source, "reference"),
        ):
            expected_action = int(paid[f"{expected_role}_action"])
            if source.get("arm_role") != expected_role or int(
                source.get("action_token_id", -1)
            ) != expected_action:
                raise ValueError(f"Collector action mismatch for pair {pair_id}/{expected_role}.")

        partial_harm = int(partial_score["duo_score"])
        reference_harm = int(reference_score["duo_score"])
        signed_regret = reference_harm - partial_harm
        features = _finite_features(paid, context=f"paid candidate {candidate_id}")
        weight = inclusion[str(candidate_id)]
        rows.append(
            {
                "pair_id": pair_id,
                "candidate_id": str(candidate_id),
                "split": paid["split"],
                "record_name": paid["record_name"],
                "dataset_idx": int(paid["dataset_idx"]),
                "row_key": paid["row_key"],
                "position": int(paid["position"]),
                "position_bin": paid["position_bin"],
                "budget": int(paid["budget"]),
                "partial_action": int(paid["partial_action"]),
                "reference_action": int(paid["reference_action"]),
                "feature_names": list(FEATURE_NAMES),
                "features": features.tolist(),
                "partial_harmful_score": partial_harm,
                "reference_harmful_score": reference_harm,
                "signed_regret": signed_regret,
                "harmful_score_drop": int(signed_regret >= 1),
                "severity": max(0, signed_regret),
                "sampling_stratum": weight["stratum"],
                "inclusion_fraction": float(weight["inclusion_fraction"]),
                "inverse_inclusion_weight": float(weight["inverse_inclusion_weight"]),
            }
        )
    missing_candidates = sorted(set(plan_by_id) - seen_candidates)
    if missing_candidates:
        raise ValueError(f"Collector/judge output is missing paid candidates: {missing_candidates[:10]}.")
    return rows


def _normalized_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("Training weights must be finite and positive.")
    return weights / float(weights.mean())


def _fit_one(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    c_value: float,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression]:
    classes = np.unique(labels)
    if not np.array_equal(classes, np.asarray([0, 1])):
        raise ValueError(
            "Conditional-drop training data must contain both classes; "
            f"found {classes.tolist()}."
        )
    normalized = _normalized_weights(weights)
    scaler = StandardScaler()
    scaler.fit(features, sample_weight=normalized)
    transformed = scaler.transform(features)
    classifier = LogisticRegression(
        C=float(c_value),
        max_iter=5000,
        random_state=int(seed),
        solver="lbfgs",
    )
    classifier.fit(transformed, labels, sample_weight=normalized)
    if not np.array_equal(classifier.classes_, np.asarray([0, 1])):
        raise RuntimeError("Logistic classifier learned an unexpected class order.")
    if int(classifier.n_iter_[0]) >= classifier.max_iter:
        raise RuntimeError("Conditional-drop logistic regression did not converge.")
    return scaler, classifier


def weighted_brier(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average((scores - labels) ** 2, weights=weights))


def fit_grouped_conditional_model(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    weights: np.ndarray,
    *,
    c_grid: Sequence[float],
    cv_folds: int,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression, np.ndarray, dict[str, Any]]:
    """Select C with record-grouped OOF Brier, then refit on all train pairs."""

    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=object)
    weights = np.asarray(weights, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Training matrix must use exactly 17 features.")
    if not (len(features) == len(labels) == len(groups) == len(weights)):
        raise ValueError("Training arrays have inconsistent lengths.")
    if not np.isfinite(features).all():
        raise ValueError("Training features contain non-finite values.")
    if not np.array_equal(np.unique(labels), np.asarray([0, 1])):
        raise ValueError("Train mismatches must contain both harmful-drop classes.")
    unique_groups = np.unique(groups)
    if cv_folds < 2 or len(unique_groups) < cv_folds:
        raise ValueError(
            f"Grouped {cv_folds}-fold CV requires at least {cv_folds} train records; "
            f"found {len(unique_groups)}."
        )
    grid = parse_c_grid(c_grid)
    splitter = GroupKFold(n_splits=cv_folds)
    splits = list(splitter.split(features, labels, groups))
    results: list[dict[str, Any]] = []
    predictions_by_c: dict[float, np.ndarray] = {}
    for c_value in grid:
        oof = np.full(len(labels), np.nan, dtype=np.float64)
        fold_reports = []
        for fold, (train_ids, validation_ids) in enumerate(splits):
            train_classes = np.unique(labels[train_ids])
            if not np.array_equal(train_classes, np.asarray([0, 1])):
                raise ValueError(
                    f"Grouped CV fold {fold} training partition is single-class "
                    f"({train_classes.tolist()}); cannot fit a conditional-drop model."
                )
            scaler, classifier = _fit_one(
                features[train_ids],
                labels[train_ids],
                weights[train_ids],
                c_value=c_value,
                seed=seed + fold,
            )
            predicted = classifier.predict_proba(scaler.transform(features[validation_ids]))[
                :, 1
            ]
            oof[validation_ids] = predicted
            fold_reports.append(
                {
                    "fold": fold,
                    "train_pairs": int(len(train_ids)),
                    "validation_pairs": int(len(validation_ids)),
                    "train_records": int(len(np.unique(groups[train_ids]))),
                    "validation_records": int(len(np.unique(groups[validation_ids]))),
                    "validation_positive_count": int(labels[validation_ids].sum()),
                    "weighted_validation_brier": weighted_brier(
                        labels[validation_ids], predicted, weights[validation_ids]
                    ),
                }
            )
        if not np.isfinite(oof).all():
            raise RuntimeError("Grouped CV did not produce every OOF prediction.")
        score = weighted_brier(labels, oof, weights)
        predictions_by_c[float(c_value)] = oof
        results.append(
            {
                "C": float(c_value),
                "weighted_oof_brier": score,
                "unweighted_oof_brier": float(np.mean((oof - labels) ** 2)),
                "folds": fold_reports,
            }
        )
    selected = min(results, key=lambda row: (row["weighted_oof_brier"], row["C"]))
    selected_c = float(selected["C"])
    scaler, classifier = _fit_one(
        features,
        labels,
        weights,
        c_value=selected_c,
        seed=seed,
    )
    return scaler, classifier, predictions_by_c[selected_c], {
        "selection_metric": "inverse-stratum-weighted_group_oof_brier",
        "cv_folds": int(cv_folds),
        "group_field": "record_name",
        "record_count": int(len(unique_groups)),
        "C_grid": list(grid),
        "selected_C": selected_c,
        "candidates": results,
    }


def predict_portable(model: Mapping[str, Any], features: Sequence[float]) -> float:
    return portable_linear_stopper_confidence(
        features,
        scaler_mean=model["scaler_mean"],
        scaler_scale=model["scaler_scale"],
        coefficient=model["coefficient"],
        intercept=model["intercept"],
    )


def serialize_model(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    *,
    selected_c: float,
) -> dict[str, Any]:
    if classifier.coef_.shape != (1, len(FEATURE_NAMES)):
        raise ValueError("Conditional model has an unexpected coefficient shape.")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scoring_protocol": PORTABLE_STOPPER_PROTOCOL,
        "estimand": "P(H_reference-H_partial>=1 | partial_action!=reference_action,state)",
        "classes": [0, 1],
        "feature_names": list(FEATURE_NAMES),
        "selected_C": float(selected_c),
        "scaler_mean": scaler.mean_.astype(np.float64).tolist(),
        "scaler_scale": scaler.scale_.astype(np.float64).tolist(),
        "coefficient": classifier.coef_[0].astype(np.float64).tolist(),
        "intercept": float(classifier.intercept_[0]),
    }
    payload["model_payload_sha256"] = sha256_payload(payload)
    return payload


def prediction_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (len(labels) == len(scores) == len(weights)) or len(labels) == 0:
        raise ValueError("Metric arrays must be non-empty and equal length.")
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("Prediction scores must be finite probabilities.")
    result: dict[str, Any] = {
        "pairs": int(len(labels)),
        "weighted_brier": weighted_brier(labels, scores, weights),
        "unweighted_brier": float(np.mean((scores - labels) ** 2)),
        "mean_score_weighted": float(np.average(scores, weights=weights)),
        "mean_score_unweighted": float(scores.mean()),
    }
    if len(np.unique(labels)) < 2:
        result.update(
            {
                "weighted_auroc": None,
                "weighted_auprc": None,
                "unweighted_auroc": None,
                "unweighted_auprc": None,
                "rank_metric_status": "undefined_single_class_validation",
            }
        )
    else:
        result.update(
            {
                "weighted_auroc": float(roc_auc_score(labels, scores, sample_weight=weights)),
                "weighted_auprc": float(
                    average_precision_score(labels, scores, sample_weight=weights)
                ),
                "unweighted_auroc": float(roc_auc_score(labels, scores)),
                "unweighted_auprc": float(average_precision_score(labels, scores)),
                "rank_metric_status": "defined",
            }
        )
    return result


def _split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = np.asarray([row["harmful_score_drop"] for row in rows], dtype=np.float64)
    weights = np.asarray([row["inverse_inclusion_weight"] for row in rows], dtype=np.float64)
    signed = np.asarray([row["signed_regret"] for row in rows], dtype=np.float64)
    severity = np.asarray([row["severity"] for row in rows], dtype=np.float64)
    return {
        "pairs": int(len(rows)),
        "records": int(len({row["record_name"] for row in rows})),
        "positive_pairs": int(labels.sum()),
        "drop_prevalence_weighted": float(np.average(labels, weights=weights)),
        "drop_prevalence_unweighted": float(labels.mean()),
        "mean_signed_regret_weighted": float(np.average(signed, weights=weights)),
        "mean_signed_regret_unweighted": float(signed.mean()),
        "mean_severity_weighted": float(np.average(severity, weights=weights)),
        "mean_severity_unweighted": float(severity.mean()),
        "effective_sample_size_from_weights": float(weights.sum() ** 2 / np.square(weights).sum()),
        "signed_regret_counts": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(signed.astype(np.int64), return_counts=True))
        },
    }


def _load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("Policy spec must be a JSON object.")
    expected_hash = policy.get(POLICY_SPEC_HASH_FIELD)
    if expected_hash != policy_spec_payload_sha256(policy):
        raise ValueError("Frozen action policy payload hash mismatch.")
    if policy.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Frozen action policy uses a different feature contract.")
    stopper = policy.get("linear_stopper")
    if (
        not isinstance(stopper, dict)
        or stopper.get("schema_version") != 2
        or stopper.get("scoring_protocol") != PORTABLE_STOPPER_PROTOCOL
        or stopper.get("classes") != [0, 1]
    ):
        raise ValueError("Frozen policy has no supported action-agreement stopper.")
    # Exercise vector validation once before scoring rows.
    portable_linear_stopper_confidence(
        [0.0] * len(FEATURE_NAMES),
        scaler_mean=stopper.get("scaler_mean", ()),
        scaler_scale=stopper.get("scaler_scale", ()),
        coefficient=stopper.get("coefficient", ()),
        intercept=stopper.get("intercept", float("nan")),
    )
    return policy


def analyze(
    *,
    universe: Sequence[dict[str, Any]],
    plan: Sequence[dict[str, Any]],
    judge_input: Sequence[dict[str, Any]],
    harmful_scores: Sequence[dict[str, Any]],
    policy: Mapping[str, Any],
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    cv_folds: int = 5,
    seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Pure analysis core used by the CLI and focused synthetic tests."""

    validate_planner_inputs(universe, plan)
    inclusion = derive_inclusion_weights(universe, plan)
    pairs = merge_judgments(
        plan=plan,
        judge_input=judge_input,
        harmful_scores=harmful_scores,
        inclusion=inclusion,
    )
    by_split = {
        split: [row for row in pairs if row["split"] == split]
        for split in ("train", "legacy_validation")
    }
    train = by_split["train"]
    validation = by_split["legacy_validation"]
    train_x = np.asarray([row["features"] for row in train], dtype=np.float64)
    train_y = np.asarray([row["harmful_score_drop"] for row in train], dtype=np.int64)
    train_groups = np.asarray([row["record_name"] for row in train], dtype=object)
    train_weights = np.asarray(
        [row["inverse_inclusion_weight"] for row in train], dtype=np.float64
    )
    scaler, classifier, train_oof, cv_report = fit_grouped_conditional_model(
        train_x,
        train_y,
        train_groups,
        train_weights,
        c_grid=c_grid,
        cv_folds=cv_folds,
        seed=seed,
    )
    model = serialize_model(
        scaler, classifier, selected_c=float(cv_report["selected_C"])
    )
    old_stopper = policy["linear_stopper"]
    oof_by_candidate = {
        row["candidate_id"]: float(value) for row, value in zip(train, train_oof)
    }
    for row in pairs:
        features = row["features"]
        agreement = portable_linear_stopper_confidence(
            features,
            scaler_mean=old_stopper["scaler_mean"],
            scaler_scale=old_stopper["scaler_scale"],
            coefficient=old_stopper["coefficient"],
            intercept=old_stopper["intercept"],
        )
        refit_conditional = predict_portable(model, features)
        oof_conditional = oof_by_candidate.get(row["candidate_id"])
        conditional = (
            float(oof_conditional)
            if row["split"] == "train"
            else refit_conditional
        )
        row["action_agreement_confidence"] = agreement
        row["action_only_risk_q"] = 1.0 - agreement
        row["conditional_drop_risk_s"] = conditional
        row["conditional_drop_risk_s_refit"] = refit_conditional
        row["outcome_risk_q_times_s"] = (1.0 - agreement) * conditional
        row["conditional_prediction_provenance"] = (
            "group_oof_selected_C" if row["split"] == "train" else "train_refit"
        )
        row["conditional_drop_risk_s_oof"] = (
            float(oof_conditional)
            if row["split"] == "train"
            else None
        )

    validation_y = np.asarray(
        [row["harmful_score_drop"] for row in validation], dtype=np.int64
    )
    validation_w = np.asarray(
        [row["inverse_inclusion_weight"] for row in validation], dtype=np.float64
    )
    predictors = {
        "action_only_q": np.asarray(
            [row["action_only_risk_q"] for row in validation], dtype=np.float64
        ),
        "conditional_s": np.asarray(
            [row["conditional_drop_risk_s"] for row in validation], dtype=np.float64
        ),
        "outcome_q_times_s": np.asarray(
            [row["outcome_risk_q_times_s"] for row in validation], dtype=np.float64
        ),
    }
    validation_metrics = {
        name: prediction_metrics(validation_y, scores, validation_w)
        for name, scores in predictors.items()
    }
    q_metrics = validation_metrics["action_only_q"]
    product_metrics = validation_metrics["outcome_q_times_s"]
    comparison = {
        "weighted_brier_delta_outcome_minus_action_only": product_metrics[
            "weighted_brier"
        ]
        - q_metrics["weighted_brier"],
        "unweighted_brier_delta_outcome_minus_action_only": product_metrics[
            "unweighted_brier"
        ]
        - q_metrics["unweighted_brier"],
        "weighted_auroc_delta_outcome_minus_action_only": (
            product_metrics["weighted_auroc"] - q_metrics["weighted_auroc"]
            if product_metrics["weighted_auroc"] is not None
            else None
        ),
        "weighted_auprc_delta_outcome_minus_action_only": (
            product_metrics["weighted_auprc"] - q_metrics["weighted_auprc"]
            if product_metrics["weighted_auprc"] is not None
            else None
        ),
        "negative_brier_delta_favors_outcome_model": True,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "development_only_one_shot_outcome_regret_gate_pilot_v1",
        "offline_estimand": "base_policy_one_step_counterfactual_harmful_score_regret",
        "label_definition": {
            "signed_regret": "H_reference-H_partial",
            "harmful_score_drop": "1[signed_regret>=1]",
            "severity": "max(0,signed_regret)",
        },
        "features": list(FEATURE_NAMES),
        "fresh_test_used": False,
        "deployment_ready": False,
        "threshold_selected": False,
        "train": _split_summary(train),
        "legacy_validation": {
            **_split_summary(validation),
            "evaluation_protocol": "single_final_evaluation_after_train-only_grouped_C_selection",
            "predictor_metrics": validation_metrics,
            "comparison": comparison,
        },
        "cross_validation": cv_report,
        "limitations": [
            "Legacy validation is not a fresh locked test and supports pilot comparison only.",
            "Labels condition on paid partial/reference action mismatches; equal-action states need no paid outcome label.",
            "Because the requested validation set is mismatch-conditioned, q versus q*s calibration is a diagnostic comparison, not an unconditional full-state Brier estimate.",
            "Each arm has one deterministic base-policy continuation and one judge score, so outcome-label noise is not estimated.",
            "The estimand is a one-step base-policy counterfactual, not full deployed on-policy regret.",
            "No deployment threshold or harmful-score non-inferiority claim is selected by this analysis.",
        ],
    }
    return model, pairs, summary


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
    if args.cv_folds < 2:
        raise ValueError("--cv_folds must be at least two.")
    c_grid = parse_c_grid(args.c_grid)
    paths = {
        "state_universe": Path(args.state_universe).expanduser().resolve(),
        "paid_pair_plan": Path(args.paid_pair_plan).expanduser().resolve(),
        "judge_input": Path(args.judge_input).expanduser().resolve(),
        "harmful_score": Path(args.harmful_score).expanduser().resolve(),
        "policy_spec": Path(args.policy_spec).expanduser().resolve(),
    }
    universe = read_jsonl(paths["state_universe"])
    plan = read_jsonl(paths["paid_pair_plan"])
    judge_input = read_jsonl(paths["judge_input"])
    harmful_scores = read_jsonl(paths["harmful_score"])
    policy = _load_policy(paths["policy_spec"])
    model, pairs, summary = analyze(
        universe=universe,
        plan=plan,
        judge_input=judge_input,
        harmful_scores=harmful_scores,
        policy=policy,
        c_grid=c_grid,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite outcome-gate output: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        model_path = output_dir / "outcome_gate.json"
        pair_path = output_dir / "pair_labels_predictions.jsonl"
        summary_path = output_dir / "pilot_summary.json"
        _write_json(model_path, model)
        _write_jsonl(pair_path, pairs)
        summary.update(
            {
                "input_files": {
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in paths.items()
                },
                "output_files": {
                    "outcome_gate": {
                        "file": model_path.name,
                        "sha256": sha256_file(model_path),
                    },
                    "pair_labels_predictions": {
                        "file": pair_path.name,
                        "sha256": sha256_file(pair_path),
                    },
                },
                "frozen_action_policy_payload_sha256": policy[
                    POLICY_SPEC_HASH_FIELD
                ],
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
