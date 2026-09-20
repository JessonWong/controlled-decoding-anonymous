import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mc_reconstruction import (
    floor_log_probs_to_mc_counts,
    fuse_proxy_logits_with_mc_counts,
)


class ProxyMCFusionTest(unittest.TestCase):
    def test_recovers_raw_integer_counts_from_legacy_float16_cache(self):
        observed_alpha = 0.1
        floor_mass = 1e-4
        sample_count = 4
        observed_count = 2
        denominator = sample_count + observed_alpha * observed_count
        probabilities = torch.full((1, 4), floor_mass / 2.0)
        probabilities[0, 1] = (
            (1.0 - floor_mass) * (3.0 + observed_alpha) / denominator
        )
        probabilities[0, 3] = (
            (1.0 - floor_mass) * (1.0 + observed_alpha) / denominator
        )

        actual = floor_log_probs_to_mc_counts(
            probabilities.log().half(),
            sample_counts=sample_count,
            observed_alpha=observed_alpha,
            floor_mass=floor_mass,
        )

        self.assertEqual(actual.dtype, torch.long)
        torch.testing.assert_close(actual, torch.tensor([[0, 3, 0, 1]]))

    def test_recovers_multiple_rows_with_distinct_sparse_supports(self):
        observed_alpha = 0.1
        floor_mass = 1e-4
        expected = torch.tensor(
            [
                [[0, 3, 0, 1, 0, 0, 0, 0], [2, 0, 1, 0, 2, 0, 0, 0]],
                [[0, 0, 0, 6, 0, 0, 0, 0], [1, 1, 1, 1, 1, 2, 0, 0]],
            ],
            dtype=torch.long,
        )
        sample_counts = expected.sum(dim=-1)
        probabilities = torch.empty_like(expected, dtype=torch.float32)
        for batch_index in range(expected.shape[0]):
            for row_index in range(expected.shape[1]):
                counts = expected[batch_index, row_index]
                observed = counts > 0
                observed_count = int(observed.sum())
                denominator = float(sample_counts[batch_index, row_index]) + (
                    observed_alpha * observed_count
                )
                probabilities[batch_index, row_index].fill_(
                    floor_mass / (~observed).sum().item()
                )
                probabilities[batch_index, row_index, observed] = (
                    (1.0 - floor_mass)
                    * (counts[observed].float() + observed_alpha)
                    / denominator
                )

        actual = floor_log_probs_to_mc_counts(
            probabilities.log().half(),
            sample_counts=sample_counts,
            observed_alpha=observed_alpha,
            floor_mass=floor_mass,
            dtype=torch.int16,
        )

        self.assertEqual(actual.dtype, torch.int16)
        torch.testing.assert_close(actual.long(), expected)

    def test_count_recovery_still_fails_closed_per_row(self):
        log_probs = torch.tensor(
            [[[-9.0, -0.4, -9.0], [-9.0, -0.4, -9.0]]],
            dtype=torch.float16,
        )

        with self.assertRaisesRegex(
            ValueError, r"\(row, recovered, expected\): \[\(1,"
        ):
            floor_log_probs_to_mc_counts(
                log_probs,
                sample_counts=torch.tensor([[1, 2]]),
                observed_alpha=0.0,
                floor_mass=1e-4,
            )

    def test_matches_dirichlet_posterior_predictive(self):
        proxy_logits = torch.log(torch.tensor([[3.0, 1.0]], dtype=torch.float32))
        mc_counts = torch.tensor([[0, 2]], dtype=torch.long)

        actual = fuse_proxy_logits_with_mc_counts(
            proxy_logits,
            mc_counts,
            prior_strength=2.0,
        )

        # q = [3/4, 1/4], so (counts + 2q) / (2 + 2) = [3/8, 5/8].
        expected = torch.tensor([[3.0 / 8.0, 5.0 / 8.0]]).log()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual.exp().sum(dim=-1), torch.ones(1), atol=1e-6, rtol=1e-6
        )

    def test_crops_padded_proxy_tail_before_softmax(self):
        proxy_logits = torch.tensor([[0.0, 0.0, 1000.0]])
        mc_counts = torch.zeros((1, 2), dtype=torch.long)

        actual = fuse_proxy_logits_with_mc_counts(
            proxy_logits,
            mc_counts,
            temperature=0.5,
            prior_strength=7.0,
        )

        expected = torch.full((1, 2), math.log(0.5))
        torch.testing.assert_close(actual, expected)

    def test_supports_arbitrary_leading_dimensions_and_zero_sample_rows(self):
        proxy_logits = torch.tensor(
            [
                [[0.0, math.log(4.0)], [math.log(9.0), 0.0]],
                [[math.log(16.0), 0.0], [0.0, math.log(25.0)]],
            ],
            dtype=torch.float16,
        )
        mc_counts = torch.zeros((2, 2, 2), dtype=torch.int32)

        actual = fuse_proxy_logits_with_mc_counts(
            proxy_logits,
            mc_counts,
            temperature=2.0,
            dtype=torch.float64,
        )
        expected = torch.log_softmax(proxy_logits.float() / 2.0, dim=-1)

        self.assertEqual(actual.shape, mc_counts.shape)
        self.assertEqual(actual.dtype, torch.float64)
        torch.testing.assert_close(actual.float(), expected)

    def test_rejects_incompatible_shapes(self):
        cases = (
            (torch.zeros(2), torch.tensor(0), "at least one dimension"),
            (torch.zeros(2, 3), torch.zeros(1, 3), "leading dimensions"),
            (torch.zeros(2, 2), torch.zeros(2, 3), "at least as large"),
            (torch.zeros(2, 2), torch.zeros(2, 0), "non-empty"),
        )

        for proxy_logits, mc_counts, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    fuse_proxy_logits_with_mc_counts(proxy_logits, mc_counts)

    def test_rejects_invalid_counts(self):
        proxy_logits = torch.zeros(1, 2)
        cases = (
            (torch.tensor([[0, -1]]), "non-negative"),
            (torch.tensor([[0.0, 1.5]]), "integer-valued"),
            (torch.tensor([[0.0, float("nan")]]), "finite"),
            (torch.tensor([[False, True]]), "bool/complex"),
        )

        for mc_counts, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((ValueError, TypeError), message):
                    fuse_proxy_logits_with_mc_counts(proxy_logits, mc_counts)

    def test_rejects_nonfinite_logits_parameters_and_nonfloating_output(self):
        counts = torch.zeros(1, 2, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "finite values"):
            fuse_proxy_logits_with_mc_counts(
                torch.tensor([[0.0, float("inf")]]), counts
            )

        for name, value in (("temperature", 0.0), ("temperature", float("nan")),
                            ("prior_strength", 0.0),
                            ("prior_strength", float("inf"))):
            with self.subTest(name=name, value=value):
                kwargs = {name: value}
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    fuse_proxy_logits_with_mc_counts(
                        torch.zeros(1, 2), counts, **kwargs
                    )

        with self.assertRaisesRegex(TypeError, "floating-point dtype"):
            fuse_proxy_logits_with_mc_counts(
                torch.zeros(1, 2), counts, dtype=torch.long
            )


if __name__ == "__main__":
    unittest.main()
