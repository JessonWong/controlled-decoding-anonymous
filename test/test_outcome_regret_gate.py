import hashlib
import json
import unittest

import numpy as np

from training.anytime_stopping_features import FEATURE_NAMES
from training.train_outcome_regret_gate import (
    analyze,
    derive_inclusion_weights,
    fit_grouped_conditional_model,
    merge_judgments,
    predict_portable,
    serialize_model,
    validate_planner_inputs,
)


def payload_hash(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def planner_row(index, *, split="train", record_name=None, label=0, budget=4):
    if record_name is None:
        record_name = f"record-{index // 2}.pt"
    features = [0.0] * len(FEATURE_NAMES)
    features[0] = index / 20.0
    features[3] = 0.1 + 0.7 * label
    return {
        "candidate_id": f"candidate-{index}",
        "split": split,
        "record_name": record_name,
        "dataset_idx": index,
        "row_key": f"{record_name}:{index}",
        "position": index,
        "position_bin": ("early", "middle", "late")[index % 3],
        "budget": budget,
        "partial_action": 10 + index,
        "reference_action": 100 + index,
        "action_mismatch": True,
        "feature_names": list(FEATURE_NAMES),
        "features": features,
        "synthetic_label": label,
        "inclusion_fraction_within_stratum": 1.0,
    }


def judge_rows(plan):
    inputs = []
    scores = []
    for pair_number, row in enumerate(plan):
        pair_id = f"pair-{pair_number}"
        label = int(row["synthetic_label"])
        # d = H_reference - H_partial: positive label has d=+1, otherwise 0.
        harmful = {"partial": 2, "reference": 3 if label else 2}
        for role in ("partial", "reference"):
            source = {
                "prompt": f"prompt-{pair_number}",
                "completion": f"completion-{pair_number}-{role}",
                "pair_id": pair_id,
                "logibreak_group_id": pair_id,
                "arm_role": role,
                "action_token_id": row[f"{role}_action"],
                "plan_row": row,
            }
            source_index = len(inputs)
            inputs.append(source)
            scores.append(
                {
                    "source_record_index": source_index,
                    "source_record_sha256": payload_hash(source),
                    "que": source["prompt"],
                    "ans": source["completion"],
                    "duo_score": harmful[role],
                }
            )
    return inputs, scores


def action_policy():
    return {
        "linear_stopper": {
            "schema_version": 2,
            "scoring_protocol": "python_float64_scalar_v1",
            "classes": [0, 1],
            "scaler_mean": [0.0] * len(FEATURE_NAMES),
            "scaler_scale": [1.0] * len(FEATURE_NAMES),
            "coefficient": [0.0] * len(FEATURE_NAMES),
            "intercept": 0.0,
        }
    }


class OutcomeLabelTests(unittest.TestCase):
    def test_judge_indices_are_authoritative_and_regret_direction_is_reference_minus_partial(self):
        plan = [planner_row(0, label=1), planner_row(1, label=0)]
        universe = list(plan)
        inclusion = derive_inclusion_weights(universe, plan)
        inputs, scores = judge_rows(plan)
        scores.reverse()  # Join by source_record_index, never file order.

        pairs = merge_judgments(
            plan=plan,
            judge_input=inputs,
            harmful_scores=scores,
            inclusion=inclusion,
        )

        by_candidate = {row["candidate_id"]: row for row in pairs}
        positive = by_candidate["candidate-0"]
        self.assertEqual(positive["signed_regret"], 1)
        self.assertEqual(positive["harmful_score_drop"], 1)
        self.assertEqual(positive["severity"], 1)
        neutral = by_candidate["candidate-1"]
        self.assertEqual(neutral["signed_regret"], 0)
        self.assertEqual(neutral["harmful_score_drop"], 0)
        self.assertEqual(neutral["severity"], 0)

    def test_source_hash_mismatch_fails_closed(self):
        plan = [planner_row(0, label=1)]
        inputs, scores = judge_rows(plan)
        scores[0]["source_record_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            merge_judgments(
                plan=plan,
                judge_input=inputs,
                harmful_scores=scores,
                inclusion=derive_inclusion_weights(plan, plan),
            )

    def test_inclusion_fraction_is_recomputed_after_fallback(self):
        universe = [planner_row(i, label=i % 2) for i in range(4)]
        # Put all candidates in one actual stratum.
        for row in universe:
            row["position_bin"] = "early"
            row["budget"] = 4
            row["inclusion_fraction_within_stratum"] = 0.5
        plan = universe[:3]
        weights = derive_inclusion_weights(universe, plan)
        self.assertTrue(
            all(item["inclusion_fraction"] == 0.75 for item in weights.values())
        )
        self.assertTrue(
            all(item["inverse_inclusion_weight"] == 4.0 / 3.0 for item in weights.values())
        )

    def test_fresh_split_is_rejected_before_training(self):
        row = planner_row(0, split="fresh_test", label=1)
        with self.assertRaisesRegex(ValueError, "fresh/test data are prohibited"):
            validate_planner_inputs([row], [row])


class ConditionalModelTests(unittest.TestCase):
    def test_grouped_c_selection_and_portable_export(self):
        features = []
        labels = []
        groups = []
        for group in range(6):
            for label in (0, 1, 0, 1):
                row = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
                row[0] = group / 10.0
                row[3] = -1.0 if label == 0 else 1.0
                features.append(row)
                labels.append(label)
                groups.append(f"record-{group}")
        features = np.asarray(features)
        labels = np.asarray(labels)
        weights = np.linspace(1.0, 2.0, len(labels))

        scaler, classifier, oof, report = fit_grouped_conditional_model(
            features,
            labels,
            np.asarray(groups, dtype=object),
            weights,
            c_grid=(0.1, 1.0),
            cv_folds=3,
            seed=7,
        )
        model = serialize_model(
            scaler, classifier, selected_c=report["selected_C"]
        )
        portable = np.asarray([predict_portable(model, row) for row in features])
        sklearn_scores = classifier.predict_proba(scaler.transform(features))[:, 1]

        self.assertEqual(len(oof), len(labels))
        self.assertTrue(np.isfinite(oof).all())
        np.testing.assert_allclose(portable, sklearn_scores, rtol=1e-12, atol=1e-12)
        self.assertIn(report["selected_C"], (0.1, 1.0))
        self.assertEqual(len(model["coefficient"]), len(FEATURE_NAMES))

    def test_single_class_training_fold_fails_clearly(self):
        features = np.zeros((4, len(FEATURE_NAMES)), dtype=np.float64)
        labels = np.asarray([1, 1, 0, 0])
        groups = np.asarray(["positive", "positive", "negative", "negative"])
        with self.assertRaisesRegex(ValueError, "single-class"):
            fit_grouped_conditional_model(
                features,
                labels,
                groups,
                np.ones(4),
                c_grid=(1.0,),
                cv_folds=2,
                seed=1,
            )

    def test_end_to_end_core_never_uses_equal_or_fresh_rows(self):
        plan = []
        # Six train groups, each with both classes, plus four legacy pairs.
        index = 0
        for group in range(6):
            for label in (0, 1):
                plan.append(
                    planner_row(
                        index,
                        split="train",
                        record_name=f"train-{group}.pt",
                        label=label,
                    )
                )
                index += 1
        for group, label in enumerate((0, 1, 0, 1)):
            plan.append(
                planner_row(
                    index,
                    split="legacy_validation",
                    record_name=f"validation-{group}.pt",
                    label=label,
                )
            )
            index += 1
        inputs, scores = judge_rows(plan)

        model, pairs, summary = analyze(
            universe=list(plan),
            plan=list(plan),
            judge_input=inputs,
            harmful_scores=scores,
            policy=action_policy(),
            c_grid=(0.1, 1.0),
            cv_folds=3,
            seed=3,
        )

        self.assertEqual(len(pairs), len(plan))
        self.assertFalse(summary["fresh_test_used"])
        self.assertFalse(summary["deployment_ready"])
        self.assertEqual(summary["train"]["pairs"], 12)
        self.assertEqual(summary["legacy_validation"]["pairs"], 4)
        self.assertEqual(
            set(summary["legacy_validation"]["predictor_metrics"]),
            {"action_only_q", "conditional_s", "outcome_q_times_s"},
        )
        self.assertEqual(len(model["feature_names"]), 17)
        validation_rows = [row for row in pairs if row["split"] == "legacy_validation"]
        self.assertTrue(
            all(row["conditional_prediction_provenance"] == "train_refit" for row in validation_rows)
        )
        self.assertTrue(all(row["conditional_drop_risk_s_oof"] is None for row in validation_rows))


if __name__ == "__main__":
    unittest.main()
