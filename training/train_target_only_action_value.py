"""Fit and evaluate target-only action-value and handoff pilots.

The paired continuations and harmfulness labels are joined by authenticated
source indices and pair IDs.  All model selection uses prompt-grouped OOF
predictions from the frozen train split.  ``legacy_validation`` is evaluated
once after choosing the value-model regularization and conservative action
margin.  Features contain only target-visible prompt/prefix/action text and
position; proxy-model values and Llama hidden states are not accepted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from scipy.stats import binomtest


ALLOWED_SPLITS = frozenset({"train", "legacy_validation"})
PLAN_PROTOCOL = "target_only_base_vs_controller_action_v1"
COLLECTOR_PROTOCOL = "target_only_paired_continuation_t0_qwen_v1"
DEFAULT_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)
DEFAULT_MARGIN_GRID = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        raise ValueError(f"Empty JSONL: {path}")
    return rows


def parse_grid(value: str, *, name: str, positive: bool) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(
        not math.isfinite(item) or (positive and item <= 0) for item in values
    ):
        raise ValueError(f"{name} contains invalid values.")
    return tuple(sorted(set(values)))


def make_features(plan: Mapping[str, Any], arm: Mapping[str, Any]) -> str:
    role = arm["arm_role"]
    forced = plan[f"{role}_forced_prefix"]
    prefix = plan["prefix_text"]
    return (
        f"__POSITION_{int(plan['position'])}__\n"
        f"__ROLE_{role.upper()}__\n"
        f"TARGET_PREFIX:\n{prefix}\n"
        f"TARGET_POST_ACTION_PREFIX:\n{forced}"
    )


def build_samples(
    plan_rows: Sequence[dict[str, Any]],
    continuations: Sequence[dict[str, Any]],
    scores: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    plans: dict[str, dict[str, Any]] = {}
    split_groups: dict[str, set[str]] = {split: set() for split in ALLOWED_SPLITS}
    for index, plan in enumerate(plan_rows):
        if plan.get("plan_protocol") != PLAN_PROTOCOL:
            raise ValueError(f"Plan row {index} has an invalid protocol.")
        pair_id = plan.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in plans:
            raise ValueError(f"Invalid/duplicate plan pair_id at row {index}.")
        split = plan.get("split")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Invalid split at plan row {index}.")
        prompt_group = str(plan.get("prompt_group"))
        split_groups[str(split)].add(prompt_group)
        plans[pair_id] = plan
    overlap = split_groups["train"] & split_groups["legacy_validation"]
    if overlap:
        raise ValueError(f"Prompt groups cross splits: {sorted(overlap)[:5]}")

    by_source: dict[int, dict[str, Any]] = {}
    for index, score in enumerate(scores):
        source_index = score.get("source_record_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"Score row {index} has an invalid source index.")
        if source_index < 0 or source_index >= len(continuations):
            raise ValueError(f"Score row {index} points outside continuation input.")
        if source_index in by_source:
            raise ValueError(f"Duplicate score source index {source_index}.")
        source = continuations[source_index]
        if score.get("source_record_sha256") != sha256_payload(source):
            raise ValueError(f"Judge/source hash mismatch at source index {source_index}.")
        if score.get("que") != source.get("prompt") or score.get("ans") != source.get(
            "completion"
        ):
            raise ValueError(f"Judge/source text mismatch at source index {source_index}.")
        harmful = score.get("duo_score")
        if isinstance(harmful, bool) or not isinstance(harmful, int) or not 1 <= harmful <= 5:
            raise ValueError(f"Unscored/invalid harmful score at source index {source_index}.")
        by_source[source_index] = score
    if set(by_source) != set(range(len(continuations))):
        raise ValueError("Judge output does not exactly cover the continuation input.")

    arms: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for source_index, source in enumerate(continuations):
        if source.get("collector_protocol") != COLLECTOR_PROTOCOL:
            raise ValueError(f"Continuation row {source_index} has an invalid protocol.")
        pair_id = source.get("pair_id")
        if pair_id not in plans:
            raise ValueError(f"Continuation row {source_index} is absent from the plan.")
        plan = plans[str(pair_id)]
        role = source.get("arm_role")
        if role not in {"base", "controller"}:
            raise ValueError(f"Invalid arm role at source index {source_index}.")
        if source.get("split") != plan.get("split"):
            raise ValueError(f"Split mismatch at source index {source_index}.")
        if source.get("prompt") != plan.get("prompt"):
            raise ValueError(f"Prompt mismatch at source index {source_index}.")
        if source.get("plan_row_sha256") != plan.get("plan_row_sha256"):
            raise ValueError(f"Plan hash mismatch at source index {source_index}.")
        score = int(by_source[source_index]["duo_score"])
        sample = {
            "source_index": source_index,
            "pair_id": pair_id,
            "prompt_group": str(plan["prompt_group"]),
            "split": plan["split"],
            "role": role,
            "text": make_features(plan, source),
            "harmful_score": score,
            "score5": int(score == 5),
            "position": int(plan["position"]),
            "plan": plan,
        }
        if role in grouped.setdefault(str(pair_id), {}):
            raise ValueError(f"Duplicate {role} arm for pair {pair_id}.")
        grouped[str(pair_id)][str(role)] = sample
        arms.append(sample)

    pairs: list[dict[str, Any]] = []
    for pair_id, group in grouped.items():
        if set(group) != {"base", "controller"}:
            raise ValueError(f"Incomplete pair {pair_id}.")
        base, controller = group["base"], group["controller"]
        if base["split"] != controller["split"] or base["prompt_group"] != controller[
            "prompt_group"
        ]:
            raise ValueError(f"Within-pair metadata mismatch for {pair_id}.")
        pairs.append(
            {
                "pair_id": pair_id,
                "prompt_group": base["prompt_group"],
                "split": base["split"],
                "base": base,
                "controller": controller,
            }
        )
    pairs.sort(key=lambda row: row["pair_id"])
    return arms, pairs


def make_text_pipeline(estimator) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=4096,
                    sublinear_tf=True,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=2048,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    return Pipeline([("features", features), ("model", estimator)])


def value_model(alpha: float) -> Pipeline:
    return make_text_pipeline(Ridge(alpha=float(alpha), solver="lsqr"))


def score5_model(c_value: float, seed: int) -> Pipeline:
    return make_text_pipeline(
        LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=seed,
        )
    )


def pair_metrics(
    pairs: Sequence[Mapping[str, Any]],
    base_values: Sequence[float] | None = None,
    controller_values: Sequence[float] | None = None,
    *,
    margin: float = 0.0,
) -> dict[str, Any]:
    if (base_values is None) != (controller_values is None):
        raise ValueError("Both predicted-value vectors must be supplied together.")
    decisions: list[str] = []
    observed: list[int] = []
    for index, pair in enumerate(pairs):
        choose_controller = (
            base_values is not None
            and float(controller_values[index]) - float(base_values[index]) > margin
        )
        role = "controller" if choose_controller else "base"
        decisions.append(role)
        observed.append(int(pair[role]["harmful_score"]))
    return {
        "pairs": len(pairs),
        "mean_harmful_score": float(np.mean(observed)),
        "score5_count": int(sum(value == 5 for value in observed)),
        "score5_rate": float(np.mean(np.asarray(observed) == 5)),
        "controller_selected": int(sum(role == "controller" for role in decisions)),
        "base_selected": int(sum(role == "base" for role in decisions)),
        "score_distribution": dict(sorted(Counter(observed).items())),
    }


def fixed_arm_metrics(pairs: Sequence[Mapping[str, Any]], role: str) -> dict[str, Any]:
    values = [int(pair[role]["harmful_score"]) for pair in pairs]
    return {
        "pairs": len(pairs),
        "mean_harmful_score": float(np.mean(values)),
        "score5_count": int(sum(value == 5 for value in values)),
        "score5_rate": float(np.mean(np.asarray(values) == 5)),
        "score_distribution": dict(sorted(Counter(values).items())),
    }


def oracle_metrics(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        max(int(pair["base"]["harmful_score"]), int(pair["controller"]["harmful_score"]))
        for pair in pairs
    ]
    return {
        "pairs": len(pairs),
        "mean_harmful_score": float(np.mean(values)),
        "score5_count": int(sum(value == 5 for value in values)),
        "score5_rate": float(np.mean(np.asarray(values) == 5)),
        "score_distribution": dict(sorted(Counter(values).items())),
    }


def paired_comparison(
    pairs: Sequence[Mapping[str, Any]], *, seed: int, bootstrap_samples: int = 10000
) -> dict[str, Any]:
    deltas = np.asarray(
        [
            int(pair["controller"]["harmful_score"])
            - int(pair["base"]["harmful_score"])
            for pair in pairs
        ],
        dtype=float,
    )
    score5_deltas = np.asarray(
        [
            int(pair["controller"]["harmful_score"] == 5)
            - int(pair["base"]["harmful_score"] == 5)
            for pair in pairs
        ],
        dtype=float,
    )
    wins = int(np.sum(deltas > 0))
    losses = int(np.sum(deltas < 0))
    groups: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        groups.setdefault(str(pair["prompt_group"]), []).append(index)
    group_names = sorted(groups)
    rng = np.random.default_rng(seed)
    boot_mean: list[float] = []
    boot_score5: list[float] = []
    for _ in range(bootstrap_samples):
        selected_groups = rng.choice(group_names, size=len(group_names), replace=True)
        selected_indices = [index for group in selected_groups for index in groups[str(group)]]
        boot_mean.append(float(np.mean(deltas[selected_indices])))
        boot_score5.append(float(np.mean(score5_deltas[selected_indices])))
    non_ties = wins + losses
    return {
        "pairs": len(pairs),
        "prompt_groups": len(group_names),
        "controller_better_pairs": wins,
        "ties": int(np.sum(deltas == 0)),
        "base_better_pairs": losses,
        "controller_minus_base_mean": float(np.mean(deltas)),
        "controller_minus_base_mean_cluster_bootstrap_95ci": [
            float(np.quantile(boot_mean, 0.025)),
            float(np.quantile(boot_mean, 0.975)),
        ],
        "controller_minus_base_score5_rate": float(np.mean(score5_deltas)),
        "controller_minus_base_score5_rate_cluster_bootstrap_95ci": [
            float(np.quantile(boot_score5, 0.025)),
            float(np.quantile(boot_score5, 0.975)),
        ],
        "two_sided_sign_test_p_excluding_ties": (
            float(binomtest(wins, non_ties, 0.5).pvalue) if non_ties else None
        ),
        "bootstrap_unit": "prompt_group",
        "bootstrap_samples": bootstrap_samples,
    }


def grouped_oof_value(
    train_arms: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    folds: int,
) -> np.ndarray:
    groups = np.asarray([sample["prompt_group"] for sample in train_arms])
    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("At least two train prompt groups are required.")
    texts = np.asarray([sample["text"] for sample in train_arms], dtype=object)
    labels = np.asarray([sample["harmful_score"] for sample in train_arms], dtype=float)
    oof = np.full(len(train_arms), np.nan, dtype=float)
    for fit_indices, held_indices in GroupKFold(n_splits=n_splits).split(texts, labels, groups):
        model = value_model(alpha)
        model.fit(texts[fit_indices].tolist(), labels[fit_indices])
        oof[held_indices] = model.predict(texts[held_indices].tolist())
    if np.isnan(oof).any():
        raise RuntimeError("Incomplete value OOF predictions.")
    return oof


def pair_predictions(
    pairs: Sequence[Mapping[str, Any]],
    predictions_by_source: Mapping[int, float],
) -> tuple[list[float], list[float]]:
    return (
        [float(predictions_by_source[pair["base"]["source_index"]]) for pair in pairs],
        [
            float(predictions_by_source[pair["controller"]["source_index"]])
            for pair in pairs
        ],
    )


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    result = {
        "samples": int(len(labels)),
        "positive": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "mean_probability": float(probabilities.mean()),
    }
    if len(np.unique(labels)) == 2:
        result["auroc"] = float(roc_auc_score(labels, probabilities))
        result["auprc"] = float(average_precision_score(labels, probabilities))
    else:
        result["auroc"] = None
        result["auprc"] = None
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--judge-input", required=True)
    parser.add_argument("--harmful-score", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha-grid", default=",".join(map(str, DEFAULT_ALPHA_GRID)))
    parser.add_argument("--c-grid", default=",".join(map(str, DEFAULT_C_GRID)))
    parser.add_argument("--margin-grid", default=",".join(map(str, DEFAULT_MARGIN_GRID)))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    plan_path = Path(args.plan).expanduser().resolve()
    judge_input_path = Path(args.judge_input).expanduser().resolve()
    harmful_path = Path(args.harmful_score).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    alphas = parse_grid(args.alpha_grid, name="alpha grid", positive=True)
    c_values = parse_grid(args.c_grid, name="C grid", positive=True)
    margins = parse_grid(args.margin_grid, name="margin grid", positive=False)
    if any(margin < 0 for margin in margins):
        raise ValueError("Conservative action margins must be non-negative.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least two.")

    arms, pairs = build_samples(
        read_jsonl(plan_path), read_jsonl(judge_input_path), read_jsonl(harmful_path)
    )
    train_arms = [sample for sample in arms if sample["split"] == "train"]
    validation_arms = [
        sample for sample in arms if sample["split"] == "legacy_validation"
    ]
    train_pairs = [pair for pair in pairs if pair["split"] == "train"]
    validation_pairs = [pair for pair in pairs if pair["split"] == "legacy_validation"]
    if not train_pairs or not validation_pairs:
        raise ValueError("Both frozen splits must contain complete pairs.")

    value_search: list[dict[str, Any]] = []
    value_oof_by_alpha: dict[float, np.ndarray] = {}
    for alpha in alphas:
        oof = grouped_oof_value(train_arms, alpha=alpha, folds=args.cv_folds)
        value_oof_by_alpha[alpha] = oof
        by_source = {
            int(sample["source_index"]): float(prediction)
            for sample, prediction in zip(train_arms, oof)
        }
        base_pred, controller_pred = pair_predictions(train_pairs, by_source)
        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        oof
                        - np.asarray(
                            [sample["harmful_score"] for sample in train_arms], dtype=float
                        )
                    )
                    ** 2
                )
            )
        )
        for margin in margins:
            metrics = pair_metrics(
                train_pairs, base_pred, controller_pred, margin=float(margin)
            )
            value_search.append(
                {"alpha": alpha, "margin": margin, "arm_value_rmse": rmse, **metrics}
            )
    selected = max(
        value_search,
        key=lambda row: (
            row["mean_harmful_score"],
            row["score5_rate"],
            -row["controller_selected"],
            row["margin"],
            row["alpha"],
        ),
    )
    selected_alpha = float(selected["alpha"])
    selected_margin = float(selected["margin"])
    fitted_value = value_model(selected_alpha)
    fitted_value.fit(
        [sample["text"] for sample in train_arms],
        [sample["harmful_score"] for sample in train_arms],
    )
    validation_predictions = fitted_value.predict(
        [sample["text"] for sample in validation_arms]
    )
    validation_by_source = {
        int(sample["source_index"]): float(prediction)
        for sample, prediction in zip(validation_arms, validation_predictions)
    }
    val_base_pred, val_controller_pred = pair_predictions(
        validation_pairs, validation_by_source
    )

    # Absolute score-5 handoff/value pilot.  C is selected solely by grouped
    # train OOF Brier score and is not allowed to affect the held-out selector.
    train_score5 = np.asarray([sample["score5"] for sample in train_arms], dtype=int)
    handoff_report: dict[str, Any]
    fitted_handoff = None
    if len(np.unique(train_score5)) == 2:
        groups = np.asarray([sample["prompt_group"] for sample in train_arms])
        texts = np.asarray([sample["text"] for sample in train_arms], dtype=object)
        handoff_search: list[dict[str, Any]] = []
        handoff_oof: dict[float, np.ndarray] = {}
        splitter = GroupKFold(n_splits=min(args.cv_folds, len(np.unique(groups))))
        for c_value in c_values:
            probabilities = np.full(len(train_arms), np.nan, dtype=float)
            valid = True
            for fit_indices, held_indices in splitter.split(texts, train_score5, groups):
                if len(np.unique(train_score5[fit_indices])) < 2:
                    valid = False
                    break
                model = score5_model(c_value, args.seed)
                model.fit(texts[fit_indices].tolist(), train_score5[fit_indices])
                probabilities[held_indices] = model.predict_proba(
                    texts[held_indices].tolist()
                )[:, 1]
            if valid and not np.isnan(probabilities).any():
                handoff_oof[c_value] = probabilities
                handoff_search.append(
                    {"C": c_value, **binary_metrics(train_score5, probabilities)}
                )
        if handoff_search:
            handoff_selected = min(handoff_search, key=lambda row: (row["brier"], row["C"]))
            fitted_handoff = score5_model(float(handoff_selected["C"]), args.seed)
            fitted_handoff.fit(texts.tolist(), train_score5)
            validation_labels = np.asarray(
                [sample["score5"] for sample in validation_arms], dtype=int
            )
            validation_probabilities = fitted_handoff.predict_proba(
                [sample["text"] for sample in validation_arms]
            )[:, 1]
            handoff_report = {
                "available": True,
                "selection_metric": "prompt-grouped-train-OOF Brier",
                "search": handoff_search,
                "selected_C": handoff_selected["C"],
                "train_oof": binary_metrics(
                    train_score5, handoff_oof[float(handoff_selected["C"])]
                ),
                "legacy_validation": binary_metrics(
                    validation_labels, validation_probabilities
                ),
            }
        else:
            handoff_report = {
                "available": False,
                "reason": "At least one grouped training fold had only one class.",
            }
    else:
        handoff_report = {
            "available": False,
            "reason": "The train split has only one score-5 class.",
        }

    baselines = {}
    for split_name, split_pairs in (
        ("train", train_pairs),
        ("legacy_validation", validation_pairs),
        ("all_development", pairs),
    ):
        baselines[split_name] = {
            "always_base": fixed_arm_metrics(split_pairs, "base"),
            "always_controller": fixed_arm_metrics(split_pairs, "controller"),
            "pair_oracle": oracle_metrics(split_pairs),
            "paired_controller_vs_base": paired_comparison(
                split_pairs,
                seed=args.seed
                + {"train": 0, "legacy_validation": 1, "all_development": 2}[
                    split_name
                ],
            ),
        }
    by_step: dict[str, Any] = {}
    for step in sorted({int(pair["base"]["plan"]["step"]) for pair in pairs}):
        step_pairs = [
            pair for pair in pairs if int(pair["base"]["plan"]["step"]) == step
        ]
        by_step[str(step)] = {
            "always_base": fixed_arm_metrics(step_pairs, "base"),
            "always_controller": fixed_arm_metrics(step_pairs, "controller"),
            "pair_oracle": oracle_metrics(step_pairs),
            "paired_controller_vs_base": paired_comparison(
                step_pairs, seed=args.seed + 100 + step
            ),
        }
    report = {
        "analysis": "development_only_target_only_action_value_v1",
        "feature_contract": (
            "target-visible position + generated prefix + base/controller action token text; "
            "no proxy values and no Llama hidden states"
        ),
        "counts": {
            "pairs": len(pairs),
            "arms": len(arms),
            "train_pairs": len(train_pairs),
            "legacy_validation_pairs": len(validation_pairs),
            "train_prompt_groups": len({pair["prompt_group"] for pair in train_pairs}),
            "legacy_validation_prompt_groups": len(
                {pair["prompt_group"] for pair in validation_pairs}
            ),
        },
        "input_sha256": {
            "plan": sha256_file(plan_path),
            "judge_input": sha256_file(judge_input_path),
            "harmful_score": sha256_file(harmful_path),
        },
        "baselines": baselines,
        "exploratory_by_step": by_step,
        "action_value": {
            "selection_metric": (
                "max train prompt-grouped OOF harmful mean, then score5 rate, then fewer "
                "controller actions and larger conservative margin"
            ),
            "search": value_search,
            "selected_alpha": selected_alpha,
            "selected_margin": selected_margin,
            "train_oof_at_selected": selected,
            "legacy_validation_at_selected": pair_metrics(
                validation_pairs,
                val_base_pred,
                val_controller_pred,
                margin=selected_margin,
            ),
        },
        "absolute_score5_handoff": handoff_report,
        "limitations": [
            "AdvBench-50 is a repeatedly used development set, so this is not a final estimate.",
            "Only early forced-warmup mismatches from one uniform-prior BiasNet run are covered.",
            "The frozen legacy-validation split contains few prompt groups and has wide uncertainty.",
            "The learned artifacts are pilots and are not wired into online decoding by this script.",
        ],
    }

    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    joblib.dump(
        {
            "model": fitted_value,
            "margin": selected_margin,
            "feature_contract": report["feature_contract"],
        },
        output_dir / "target_only_action_value.joblib",
    )
    if fitted_handoff is not None:
        joblib.dump(
            {"model": fitted_handoff, "feature_contract": report["feature_contract"]},
            output_dir / "target_only_score5_handoff.joblib",
        )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
