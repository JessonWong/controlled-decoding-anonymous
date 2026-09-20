import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from pre_logits_sampled_openweight import sampled_ids_to_log_probs
from mc_reconstruction import (
    floor_log_probs_to_log_counts,
    sampled_ids_to_log_counts,
)


class SampledLogProbSmoothingTest(unittest.TestCase):
    def test_floor_mass_distribution_is_normalized_and_ordered(self):
        sampled_ids = torch.tensor([[2, 2, 2, 3, 4]], dtype=torch.long)
        log_probs = sampled_ids_to_log_probs(
            sampled_token_ids=sampled_ids,
            vocab_size=8,
            observed_alpha=0.1,
            floor_mass=1e-3,
            dtype=torch.float32,
        )

        probs = log_probs.exp()
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-6))
        self.assertGreater(probs[0, 2].item(), probs[0, 3].item())
        self.assertGreater(probs[0, 3].item(), probs[0, 0].item())

        unseen_probs = probs[0, [0, 1, 5, 6, 7]]
        self.assertTrue(torch.allclose(unseen_probs, unseen_probs[0].expand_as(unseen_probs)))
        self.assertAlmostEqual(unseen_probs.sum().item(), 1e-3, places=7)

    def test_rejects_bad_shapes(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            sampled_ids_to_log_probs(torch.tensor([1, 2, 3]), vocab_size=8)

    def test_rejects_invalid_floor_mass(self):
        with self.assertRaisesRegex(ValueError, "floor_mass"):
            sampled_ids_to_log_probs(torch.tensor([[1, 2, 3]]), vocab_size=8, floor_mass=1.0)

    def test_centered_log_counts_bound_the_support_jump(self):
        sampled_ids = torch.tensor([[2, 2, 2, 3, 4]], dtype=torch.long)
        scores = sampled_ids_to_log_counts(
            sampled_ids,
            vocab_size=8,
            alpha=1.0,
            dtype=torch.float32,
        )

        expected = torch.zeros(1, 8)
        expected[0, 2] = torch.log(torch.tensor(4.0))
        expected[0, 3] = torch.log(torch.tensor(2.0))
        expected[0, 4] = torch.log(torch.tensor(2.0))
        torch.testing.assert_close(scores, expected)
        self.assertAlmostEqual(scores[0, 3].item(), torch.log(torch.tensor(2.0)).item())

    def test_legacy_float16_cache_recovers_exact_log_counts(self):
        sampled_ids = torch.tensor(
            [[2, 2, 2, 3, 4], [1, 1, 1, 1, 1]], dtype=torch.long
        )
        legacy = sampled_ids_to_log_probs(
            sampled_token_ids=sampled_ids,
            vocab_size=8,
            observed_alpha=0.1,
            floor_mass=1e-4,
            dtype=torch.float16,
        )
        recovered = floor_log_probs_to_log_counts(
            log_probs=legacy,
            sample_counts=torch.tensor([5, 5]),
            observed_alpha=0.1,
            floor_mass=1e-4,
            log_count_alpha=1.0,
            dtype=torch.float32,
        )
        direct = sampled_ids_to_log_counts(
            sampled_ids,
            vocab_size=8,
            alpha=1.0,
            dtype=torch.float32,
        )

        torch.testing.assert_close(recovered, direct)

    def test_log_count_rejects_nonpositive_alpha(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            sampled_ids_to_log_counts(
                torch.tensor([[1, 2, 3]]), vocab_size=8, alpha=0
            )


if __name__ == "__main__":
    unittest.main()
