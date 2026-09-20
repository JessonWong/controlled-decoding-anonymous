import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.eval_anytime_stopping import (
    PORTABLE_STOPPER_PROTOCOL,
    _aggregate_metrics,
    _policy_arrays,
    _select_stage,
    _serialize_linear_stopper,
    calibrate_thresholds,
)
from training.anytime_proxy_mc import portable_linear_stopper_confidence


class AnytimeStoppingTest(unittest.TestCase):
    def make_paths(self):
        # [replay, row, stage], with the final stage serving as the MC50 action.
        actions = np.asarray(
            [
                [
                    [1, 2, 2],
                    [3, 3, 3],
                    [4, 4, 5],
                ]
            ],
            dtype=np.int64,
        )
        return {
            "actions": actions,
            "labels": np.asarray([2, 3, 5]),
            "base_token_ids": np.asarray([0, 3, 0]),
            "record_names": np.asarray(["a", "b", "c"], dtype=object),
        }

    def test_stage_selection_uses_earliest_confident_state_or_full(self):
        confidence = np.asarray([[[0.2, 0.9], [0.8, 0.9], [0.1, 0.2]]])
        selected = _select_stage(confidence, 0.75)
        np.testing.assert_array_equal(selected, np.asarray([[1, 0, 2]]))

    def test_policy_metrics_compare_against_full_action(self):
        paths = self.make_paths()
        selected = np.asarray([[1, 0, 2]])
        arrays = _policy_arrays(paths, (0, 4, 50), selected)
        metrics = _aggregate_metrics(arrays, 50)

        self.assertEqual(metrics["action_disagreement"], 0.0)
        self.assertEqual(metrics["label_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["mean_samples"], 18.0)
        self.assertEqual(metrics["beneficial_flip_recall"], 1.0)

    def test_threshold_calibration_picks_cheapest_feasible_policy(self):
        paths = self.make_paths()
        confidence = np.asarray([[[0.1, 0.9], [0.9, 0.9], [0.1, 0.1]]])
        policies = calibrate_thresholds(
            paths=paths,
            confidence=confidence,
            budgets=(0, 4, 50),
            error_budgets=(0.0,),
        )

        policy = policies["delta_0"]
        self.assertEqual(policy["action_disagreement"], 0.0)
        self.assertLess(policy["mean_samples"], 50.0)

    def test_serialized_linear_stopper_matches_pipeline(self):
        features = np.zeros((6, 17), dtype=np.float32)
        features[:, 0] = np.arange(6)
        features[:, 1] = [0, 1, 0, 1, 0, 1]
        labels = np.asarray([0, 0, 0, 1, 1, 1])
        model = make_pipeline(StandardScaler(), LogisticRegression()).fit(
            features, labels
        )
        serialized = _serialize_linear_stopper(model)
        actual = np.asarray(
            [
                portable_linear_stopper_confidence(
                    row,
                    scaler_mean=serialized["scaler_mean"],
                    scaler_scale=serialized["scaler_scale"],
                    coefficient=serialized["coefficient"],
                    intercept=serialized["intercept"],
                )
                for row in features
            ]
        )

        np.testing.assert_allclose(
            actual, model.predict_proba(features)[:, 1], rtol=1e-6, atol=1e-7
        )
        self.assertEqual(serialized["schema_version"], 2)
        self.assertEqual(
            serialized["scoring_protocol"], PORTABLE_STOPPER_PROTOCOL
        )


if __name__ == "__main__":
    unittest.main()
