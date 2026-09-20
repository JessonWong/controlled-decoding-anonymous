import unittest

from training.plan_outcome_regret_pairs import (
    _position_bin,
    select_stratified_mismatches,
)


class OutcomeRegretPlanTests(unittest.TestCase):
    def test_position_bins_use_frozen_normalizer(self):
        self.assertEqual(_position_bin(0, 79), "early")
        self.assertEqual(_position_bin(26, 79), "early")
        self.assertEqual(_position_bin(27, 79), "middle")
        self.assertEqual(_position_bin(52, 79), "middle")
        self.assertEqual(_position_bin(53, 79), "late")
        self.assertEqual(_position_bin(79, 79), "late")

    def test_stratified_selection_is_deterministic_and_mismatch_only(self):
        rows = []
        for budget in (0, 4):
            for position_bin in ("early", "middle", "late"):
                for index in range(5):
                    rows.append(
                        {
                            "candidate_id": f"{budget}:{position_bin}:{index}",
                            "budget": budget,
                            "position_bin": position_bin,
                            "action_mismatch": index != 4,
                        }
                    )
        first, report = select_stratified_mismatches(
            rows,
            total=12,
            paid_budgets=(0, 4),
            selection_seed="fixed",
        )
        second, _ = select_stratified_mismatches(
            rows,
            total=12,
            paid_budgets=(0, 4),
            selection_seed="fixed",
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertTrue(all(row["action_mismatch"] for row in first))
        self.assertEqual(report["fallback_count"], 0)
        self.assertEqual(
            {row["selection_stratum"] for row in first},
            {
                "k0:early",
                "k0:middle",
                "k0:late",
                "k4:early",
                "k4:middle",
                "k4:late",
            },
        )

    def test_short_stratum_uses_documented_fallback(self):
        rows = []
        for budget in (0, 4):
            for position_bin in ("early", "middle", "late"):
                count = 1 if (budget, position_bin) == (0, "early") else 5
                for index in range(count):
                    rows.append(
                        {
                            "candidate_id": f"{budget}:{position_bin}:{index}",
                            "budget": budget,
                            "position_bin": position_bin,
                            "action_mismatch": True,
                        }
                    )
        selected, report = select_stratified_mismatches(
            rows,
            total=12,
            paid_budgets=(0, 4),
            selection_seed="fixed",
        )
        self.assertEqual(len(selected), 12)
        self.assertEqual(report["fallback_count"], 1)


if __name__ == "__main__":
    unittest.main()
