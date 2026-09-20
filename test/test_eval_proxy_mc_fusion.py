import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from eval_proxy_mc_fusion import (
    METHOD_FUSED,
    METHOD_MC,
    deterministic_binomial_split,
    evaluate_artifacts,
)


class ProxyMCFusionEvaluationTest(unittest.TestCase):
    def test_binomial_split_is_row_keyed_and_reproducible(self):
        counts = torch.tensor([9, 4, 0, 7])
        first = deterministic_binomial_split(
            counts, 0.5, seed=17, row_key="record.pt:0"
        )
        second = deterministic_binomial_split(
            counts, 0.5, seed=17, row_key="record.pt:0"
        )

        torch.testing.assert_close(first[0], second[0])
        torch.testing.assert_close(first[1], second[1])
        torch.testing.assert_close(first[0] + first[1], counts)

    def test_selection_ignores_answer_labels_and_reports_strata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_dir = Path(temporary_directory)
            payload = {
                "proxy_logits": torch.tensor(
                    [
                        [
                            [3.0, 1.0, -1.0, -2.0],
                            [-1.0, 1.0, 3.0, -2.0],
                        ]
                    ],
                    dtype=torch.float32,
                ),
                "mc_counts": torch.tensor(
                    [[[8, 2, 0, 0], [0, 2, 8, 0]]], dtype=torch.uint8
                ),
                "labels": torch.tensor([[0, 2]]),
                "risk_gate_token_ids": torch.tensor([[0, 2]]),
                "metadata": {
                    "observed_alpha": 0.1,
                    "floor_mass": 1e-4,
                    "proxy_logits_semantics": "uncalibrated_raw_logits_shared_vocab",
                },
            }
            artifact = cache_dir / "000.pt"
            torch.save(payload, artifact)
            common = dict(
                temperatures=[0.5, 1.0, 2.0],
                prior_strengths=[0.5, 2.0, 8.0],
                split_seeds=[3, 5, 7],
                calibration_fraction=0.5,
                top_k=2,
            )
            first = evaluate_artifacts(cache_dir, **common)
            json.dumps(first, allow_nan=False)

            payload["labels"] = torch.tensor([[3, 3]])
            torch.save(payload, artifact)
            second = evaluate_artifacts(cache_dir, **common)

            self.assertEqual(first["selection"], second["selection"])
            self.assertEqual(first["calibration_grid"], second["calibration_grid"])
            self.assertFalse(first["label_diagnostics"]["used_for_selection"])
            self.assertNotEqual(
                first["label_diagnostics"][METHOD_FUSED],
                second["label_diagnostics"][METHOD_FUSED],
            )
            predictive = first["evaluation"][METHOD_FUSED][
                "aggregate_over_splits"
            ]["predictive_count_likelihood"]
            self.assertGreater(predictive["all"]["events"], 0)
            self.assertEqual(
                predictive["all"]["events"],
                predictive["observed_in_calibration"]["events"]
                + predictive["unobserved_in_calibration"]["events"],
            )
            gate = first["evaluation"][METHOD_MC]["aggregate_over_splits"][
                "deterministic_risk_gate_agreement"
            ]["all"]
            self.assertEqual(gate["examples"], 6)
            self.assertTrue(math.isfinite(predictive["all"]["mean_nll"]))


if __name__ == "__main__":
    unittest.main()
