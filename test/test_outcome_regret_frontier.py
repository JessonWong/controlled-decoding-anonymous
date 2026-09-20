import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.eval_outcome_regret_frontier import (
    FrontierData,
    aggregate_policy,
    authenticate_inputs,
    evaluate_frontier,
    frontier_points,
    matched_compute_bracket,
    select_stages,
    sha256_file,
    sha256_payload,
    uncertainty_analysis,
    validate_universe_rows,
)
from training.anytime_proxy_mc import policy_spec_payload_sha256
from training.anytime_stopping_features import FEATURE_NAMES


def fixture_data():
    rows = 4
    stages = 2
    labels = np.zeros((rows, stages), dtype=np.float64)
    severity = np.zeros((rows, stages), dtype=np.float64)
    signed = np.zeros((rows, stages), dtype=np.float64)
    weights = np.zeros((rows, stages), dtype=np.float64)
    observed = np.zeros((rows, stages), dtype=bool)
    # At K0 all four actions mismatch. Two uniformly paid states represent the
    # four-state stratum with inverse inclusion weight two.
    observed[:2, 0] = True
    labels[:2, 0] = [1.0, 0.0]
    severity[:2, 0] = [2.0, 0.0]
    signed[:2, 0] = [2.0, -1.0]
    weights[:2, 0] = 2.0
    return FrontierData(
        row_keys=tuple(f"row-{index}" for index in range(rows)),
        record_names=np.asarray(["a.pt", "b.pt", "c.pt", "d.pt"], dtype=object),
        budgets=np.asarray([0, 4], dtype=np.int64),
        terminal_budget=10,
        q=np.asarray(
            [
                [0.6, 0.2],
                [0.7, 0.3],
                [0.8, 0.4],
                [0.9, 0.45],
            ],
            dtype=np.float64,
        ),
        r=np.asarray(
            [
                [0.3, 0.1],
                [0.35, 0.12],
                [0.4, 0.2],
                [0.45, 0.22],
            ],
            dtype=np.float64,
        ),
        mismatches=np.asarray(
            [[True, False], [True, False], [True, False], [True, False]],
            dtype=bool,
        ),
        candidate_ids=np.asarray(
            [[f"c-{row}-0", f"c-{row}-4"] for row in range(rows)], dtype=object
        ),
        strata=np.asarray(
            [["legacy_validation:k0:early", "legacy_validation:k4:early"]]
            * rows,
            dtype=object,
        ),
        label_observed=observed,
        labels=labels,
        severity=severity,
        signed_regret=signed,
        weights=weights,
    )


class IPWPolicyTests(unittest.TestCase):
    def test_ht_hajek_coverage_and_exact_compute(self):
        data = fixture_data()
        selected = np.zeros(data.row_count, dtype=np.int64)

        metrics = aggregate_policy(data, selected)

        self.assertEqual(metrics["mean_k"], 0.0)
        self.assertEqual(metrics["action_disagreement"], 1.0)
        self.assertEqual(metrics["harmful_drop_ht"], 0.5)
        self.assertEqual(metrics["harmful_drop_hajek"], 0.5)
        self.assertEqual(metrics["severity_ht"], 1.0)
        self.assertEqual(metrics["severity_hajek"], 1.0)
        self.assertEqual(metrics["signed_regret_ht"], 0.5)
        self.assertEqual(metrics["signed_regret_hajek"], 0.5)
        self.assertEqual(metrics["coverage"]["effective_sample_size"], 2.0)
        self.assertEqual(metrics["coverage"]["max_inverse_weight"], 2.0)
        self.assertEqual(
            metrics["coverage"]["raw_paid_fraction_of_selected_mismatches"], 0.5
        )

    def test_terminal_mc_reference_has_known_zero_regret(self):
        data = fixture_data()
        terminal = np.full(data.row_count, data.stage_count, dtype=np.int64)

        metrics = aggregate_policy(data, terminal)

        self.assertEqual(metrics["mean_k"], 10.0)
        self.assertEqual(metrics["action_disagreement"], 0.0)
        self.assertEqual(metrics["harmful_drop_ht"], 0.0)
        self.assertEqual(metrics["harmful_drop_hajek"], 0.0)
        self.assertEqual(metrics["severity_hajek"], 0.0)
        self.assertTrue(metrics["coverage"]["all_selected_strata_have_paid_overlap"])


class FrontierSelectionTests(unittest.TestCase):
    def test_first_low_risk_stage_and_fallthrough(self):
        risk = np.asarray([[0.8, 0.2], [0.9, 0.7], [0.1, 0.05]])
        np.testing.assert_array_equal(select_stages(risk, 0.25), [1, 2, 0])

    def test_matched_bracket_uses_adjacent_compute_crossing(self):
        points = [
            {"mean_k": 10.0},
            {"mean_k": 8.0},
            {"mean_k": 5.0},
            {"mean_k": 2.0},
        ]
        selections = [np.asarray([index]) for index in range(len(points))]

        bracket = matched_compute_bracket(
            points, selections, target_mean_k=6.0
        )

        self.assertEqual(bracket["cheaper"][0]["mean_k"], 5.0)
        self.assertEqual(bracket["more_expensive"][0]["mean_k"], 8.0)

    def test_frontier_is_monotone_in_compute(self):
        data = fixture_data()
        points, _ = frontier_points(data, risk=data.r, family="outcome_q_times_s")
        means = [point["mean_k"] for point in points]
        self.assertTrue(all(left >= right for left, right in zip(means, means[1:])))
        self.assertEqual(means[0], data.terminal_budget)

    def test_uncertainty_is_record_clustered_and_paired(self):
        data = fixture_data()
        reference = select_stages(data.q, 0.5)
        candidate = select_stages(data.r, 0.2)

        result = uncertainty_analysis(
            data,
            policies={"reference": reference, "candidate": candidate},
            reference_name="reference",
            bootstrap_replicates=20,
            seed=9,
        )

        self.assertEqual(result["cluster_count"], 4)
        self.assertEqual(len(result["leave_one_record_out"]), 4)
        self.assertIn(
            "candidate", result["paired_difference_bootstrap_intervals_vs_reference"]
        )
        self.assertEqual(
            result["bootstrap_intervals"]["reference"]["mean_k"]["valid_draws"],
            20,
        )

    def test_end_to_end_summary_never_selects_a_threshold(self):
        data = fixture_data()
        policy = {
            "policies": {
                "delta_0.01": {
                    "status": "selected_on_legacy_validation",
                    "threshold": 0.5,
                }
            }
        }

        q_points, outcome_points, summary = evaluate_frontier(
            data=data,
            policy=policy,
            primary_policy="delta_0.01",
            bootstrap_replicates=10,
            seed=3,
        )

        self.assertTrue(q_points)
        self.assertTrue(outcome_points)
        self.assertFalse(summary["fresh_test_used"])
        self.assertFalse(summary["threshold_selected"])
        self.assertFalse(summary["threshold_tuning_used_outcome_labels"])
        self.assertIn("matched_outcome_bracket", summary)


class FreshSplitGuardTests(unittest.TestCase):
    def test_state_universe_rejects_fresh_rows(self):
        row = {
            "candidate_id": "fresh",
            "split": "fresh_test",
            "feature_names": list(FEATURE_NAMES),
            "features": [0.0] * len(FEATURE_NAMES),
            "budget": 0,
            "action_mismatch": False,
            "partial_action": 1,
            "reference_action": 1,
        }
        with self.assertRaisesRegex(ValueError, "fresh/test/unknown"):
            validate_universe_rows(
                [row],
                plan_manifest={"paid_budgets": [0]},
                policy={"budgets": [0, 10]},
            )


class ArtifactAuthenticationTests(unittest.TestCase):
    def test_development_manifests_authenticate_before_state_read(self):
        portable = {
            "schema_version": 1,
            "scoring_protocol": "python_float64_scalar_v1",
            "classes": [0, 1],
            "feature_names": list(FEATURE_NAMES),
            "scaler_mean": [0.0] * len(FEATURE_NAMES),
            "scaler_scale": [1.0] * len(FEATURE_NAMES),
            "coefficient": [0.0] * len(FEATURE_NAMES),
            "intercept": 0.0,
        }
        portable["model_payload_sha256"] = sha256_payload(portable)
        action_stopper = dict(portable)
        action_stopper["schema_version"] = 2
        action_stopper.pop("feature_names")
        action_stopper.pop("model_payload_sha256")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state_universe.jsonl"
            pairs = root / "pair_labels_predictions.jsonl"
            gate = root / "outcome_gate.json"
            policy_path = root / "policy_spec.json"
            plan_path = root / "plan_manifest.json"
            fit_path = root / "pilot_summary.json"
            state.write_text("{}\n", encoding="utf-8")
            pairs.write_text("{}\n", encoding="utf-8")
            gate.write_text(json.dumps(portable) + "\n", encoding="utf-8")
            policy = {
                "schema_version": 2,
                "feature_names": list(FEATURE_NAMES),
                "linear_stopper": action_stopper,
                "budgets": [0, 10],
            }
            policy["policy_spec_payload_sha256"] = policy_spec_payload_sha256(policy)
            policy_path.write_text(json.dumps(policy) + "\n", encoding="utf-8")
            plan = {
                "schema_version": 1,
                "fresh_test_used": False,
                "state_universe_file": state.name,
                "state_universe_sha256": sha256_file(state),
                "policy_spec_sha256": sha256_file(policy_path),
            }
            plan["manifest_payload_sha256"] = sha256_payload(plan)
            plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
            fit = {
                "fresh_test_used": False,
                "deployment_ready": False,
                "frozen_action_policy_payload_sha256": policy[
                    "policy_spec_payload_sha256"
                ],
                "input_files": {
                    "state_universe": {
                        "path": str(state),
                        "sha256": sha256_file(state),
                    },
                    "policy_spec": {
                        "path": str(policy_path),
                        "sha256": sha256_file(policy_path),
                    },
                },
                "output_files": {
                    "pair_labels_predictions": {
                        "file": pairs.name,
                        "sha256": sha256_file(pairs),
                    },
                    "outcome_gate": {
                        "file": gate.name,
                        "sha256": sha256_file(gate),
                    },
                },
            }
            fit["summary_payload_sha256"] = sha256_payload(fit)
            fit_path.write_text(json.dumps(fit) + "\n", encoding="utf-8")

            authenticated = authenticate_inputs(
                plan_manifest_path=plan_path,
                state_universe_path=state,
                fit_summary_path=fit_path,
                pair_predictions_path=pairs,
                outcome_gate_path=gate,
                policy_spec_path=policy_path,
            )

            self.assertEqual(authenticated[0]["fresh_test_used"], False)
            self.assertEqual(authenticated[1]["deployment_ready"], False)


if __name__ == "__main__":
    unittest.main()
