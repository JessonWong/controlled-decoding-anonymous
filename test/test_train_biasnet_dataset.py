import tempfile
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from train_biasnet import (
    CachedLogitsDataset,
    intervention_objective_masks,
    top1_margin_loss,
)
from pre_logits_sampled_openweight import sampled_ids_to_log_probs


class CachedLogitsDatasetRiskGateTest(unittest.TestCase):
    def test_intervention_objective_masks_opening_and_boundary(self):
        positions = torch.tensor([0, 1, 2, 3, 7], dtype=torch.long)

        ce_mask, margin_mask = intervention_objective_masks(
            positions,
            ce_first_n_tokens=3,
            base_margin_start_position=3,
        )

        self.assertEqual(ce_mask.tolist(), [True, True, True, False, False])
        self.assertEqual(margin_mask.tolist(), [False, False, False, True, True])

    def test_intervention_objective_mask_defaults_preserve_all_positions(self):
        positions = torch.tensor([0, 3, 9], dtype=torch.long)

        ce_mask, margin_mask = intervention_objective_masks(positions)

        self.assertTrue(ce_mask.all().item())
        self.assertTrue(margin_mask.all().item())

    def test_top1_margin_targets_highest_non_target(self):
        outputs = torch.tensor(
            [[5.0, 2.0, 1.0], [3.0, 4.0, 2.0]], dtype=torch.float32
        )
        labels = torch.tensor([0, 2], dtype=torch.long)

        losses, best_other = top1_margin_loss(outputs, labels, margin_value=1.0)

        torch.testing.assert_close(best_other, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(losses, torch.tensor([0.0, 3.0]))

    def write_cache(
        self,
        directory: str,
        include_mask: bool = True,
        include_base_tokens: bool = False,
        include_scores: bool = True,
    ) -> None:
        payload = {
            "log_probs": torch.randn(1, 4, 8, dtype=torch.float16),
            "labels": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        }
        if include_mask:
            payload["risk_gate_mask"] = torch.tensor([[True, False, True, False]])
        if include_base_tokens:
            payload["risk_gate_token_ids"] = torch.tensor([[1, 7, 0, 4]])
        if include_scores:
            payload["risk_gate_scores"] = torch.tensor(
                [[0.05, 0.10, 0.15, 0.50]], dtype=torch.float32
            )
        torch.save(payload, Path(directory) / "sample.pt")

    def test_hard_gate_uses_mask_as_binary_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir)
            dataset = CachedLogitsDataset(tmpdir, risk_gate_training="hard")

            self.assertEqual(dataset.num_tokens, 4)
            self.assertEqual(dataset.risk_active_tokens, 2)
            self.assertAlmostEqual(dataset.risk_active_rate, 0.5)
            self.assertEqual([dataset[i][2].item() for i in range(4)], [1.0, 0.0, 1.0, 0.0])

    def test_hard_gate_can_be_recomputed_from_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir)
            dataset = CachedLogitsDataset(
                tmpdir,
                risk_gate_training="hard",
                risk_gate_hard_from_scores=True,
                risk_gate_threshold=0.1,
            )

            self.assertEqual(
                [dataset[i][2].item() for i in range(4)],
                [1.0, 0.0, 0.0, 0.0],
            )
            self.assertEqual(dataset.risk_active_tokens, 1)

    def test_hard_gate_from_scores_requires_cached_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir, include_scores=False)

            with self.assertRaisesRegex(ValueError, "risk_gate_scores"):
                CachedLogitsDataset(
                    tmpdir,
                    risk_gate_training="hard",
                    risk_gate_hard_from_scores=True,
                )

    def test_soft_gate_keeps_inactive_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir)
            dataset = CachedLogitsDataset(
                tmpdir,
                risk_gate_training="soft",
                risk_gate_inactive_weight=0.25,
                max_tokens_per_sample=3,
            )

            self.assertEqual(dataset.num_tokens, 3)
            self.assertEqual([dataset[i][2].item() for i in range(3)], [1.0, 0.25, 1.0])
            self.assertEqual(
                [dataset[i][5].item() for i in range(3)],
                [1.0, 1.0, 1.0],
            )

    def test_runtime_soft_gate_matches_inference_scaling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir)
            dataset = CachedLogitsDataset(
                tmpdir,
                risk_gate_training="runtime_soft",
                risk_gate_threshold=0.1,
                risk_gate_soft_temperature=0.05,
                risk_gate_min_scale=0.01,
                risk_gate_warmup_tokens=1,
            )

            scales = torch.tensor([dataset[i][5].item() for i in range(4)])
            expected = torch.tensor(
                [1.0, 0.5, torch.sigmoid(torch.tensor(-1.0)).item(), 0.0]
            )
            torch.testing.assert_close(scales, expected)
            self.assertEqual(
                [dataset[i][2].item() for i in range(4)],
                [1.0, 1.0, 1.0, 0.0],
            )
            self.assertEqual(dataset.runtime_controlled_tokens, 3)
            self.assertEqual(dataset.risk_active_tokens, 1)

    def test_runtime_soft_gate_requires_cached_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir, include_scores=False)

            with self.assertRaisesRegex(ValueError, "risk_gate_scores"):
                CachedLogitsDataset(
                    tmpdir,
                    risk_gate_training="runtime_soft",
                )

    def test_gated_training_requires_cached_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir, include_mask=False)

            with self.assertRaisesRegex(ValueError, "risk_gate_mask"):
                CachedLogitsDataset(tmpdir, risk_gate_training="hard")

    def test_first_n_is_unioned_with_gate_mask(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir)
            dataset = CachedLogitsDataset(
                tmpdir,
                risk_gate_training="hard",
                always_train_first_n=2,
            )

            self.assertEqual(
                [dataset[i][2].item() for i in range(4)],
                [1.0, 1.0, 1.0, 0.0],
            )
            self.assertEqual(dataset.primary_tokens, 3)

    def test_base_wrong_positions_receive_hard_weight_and_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir, include_base_tokens=True)
            dataset = CachedLogitsDataset(
                tmpdir,
                risk_gate_training="hard",
                always_train_first_n=2,
                base_token_hard_weight=4.0,
            )

            self.assertEqual(
                [dataset[i][2].item() for i in range(4)],
                [1.0, 4.0, 4.0, 0.0],
            )
            self.assertEqual(
                [dataset[i][3].item() for i in range(4)],
                [1, 7, 0, 4],
            )
            self.assertEqual(dataset.primary_base_wrong_tokens, 2)

    def test_sparse_boundary_keeps_first_eligible_divergences_per_sample(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir, include_base_tokens=True)
            dataset = CachedLogitsDataset(
                tmpdir,
                risk_gate_training="hard",
                always_train_first_n=2,
                sparse_boundary_start_position=1,
                sparse_boundary_max_per_sample=1,
            )

            # Eligible divergences are positions 1 and 2. Position 1 belongs to
            # the forced opening, position 2 to the cached gate; only the first
            # one is retained by the per-sample budget.
            self.assertEqual(
                dataset.sparse_boundary_mask.tolist(),
                [False, True, False, False],
            )
            self.assertEqual(dataset.sparse_boundary_tokens, 1)
            self.assertTrue(dataset[1][8].item())

    def test_sparse_boundary_respects_start_and_survives_zero_weight_drop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir, include_base_tokens=True)
            dataset = CachedLogitsDataset(
                tmpdir,
                risk_gate_training="hard",
                always_train_first_n=2,
                sparse_boundary_start_position=2,
                sparse_boundary_max_per_sample=2,
                drop_zero_weight_tokens=True,
            )

            self.assertEqual(dataset.position_ids.tolist(), [0, 1, 2])
            self.assertEqual(
                dataset.sparse_boundary_mask.tolist(),
                [False, False, True],
            )
            self.assertEqual(dataset.sparse_boundary_tokens, 1)

    def test_sparse_boundary_expands_to_contiguous_controlled_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir, include_base_tokens=True)
            dataset = CachedLogitsDataset(
                tmpdir,
                risk_gate_training="hard",
                always_train_first_n=2,
                sparse_boundary_start_position=1,
                sparse_boundary_max_per_sample=1,
                sparse_boundary_window_size=3,
            )

            # The window begins at position 1 and spans through 3, but only
            # positions controlled by the opening/gate contract are retained.
            self.assertEqual(
                dataset.sparse_boundary_mask.tolist(),
                [False, True, True, False],
            )
            self.assertEqual(dataset.sparse_boundary_tokens, 2)

    def test_zero_weight_positions_can_be_dropped_after_union(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write_cache(tmpdir, include_base_tokens=True)
            dataset = CachedLogitsDataset(
                tmpdir,
                risk_gate_training="hard",
                always_train_first_n=2,
                drop_zero_weight_tokens=True,
            )

            self.assertEqual(dataset.num_source_tokens, 4)
            self.assertEqual(dataset.num_tokens, 3)
            self.assertEqual(dataset.labels.tolist(), [1, 2, 3])
            self.assertEqual(dataset.position_ids.tolist(), [0, 1, 2])

    def test_log_count_representation_recovers_legacy_mc_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampled_ids = torch.tensor(
                [
                    [2, 2, 2, 3, 4],
                    [1, 1, 1, 1, 1],
                    [5, 6, 6, 6, 6],
                    [7, 7, 7, 7, 7],
                ],
                dtype=torch.long,
            )
            payload = {
                "log_probs": sampled_ids_to_log_probs(
                    sampled_ids,
                    vocab_size=8,
                    observed_alpha=0.1,
                    floor_mass=1e-4,
                    dtype=torch.float16,
                ).unsqueeze(0),
                "labels": torch.tensor([[2, 1, 6, 7]], dtype=torch.long),
                "valid_sample_counts": torch.full((1, 4), 5, dtype=torch.long),
                "metadata": {
                    "observed_alpha": 0.1,
                    "floor_mass": 1e-4,
                    "sample_temperature": 1.0,
                    "top_p": 1.0,
                    "sample_completion_policy": "exact",
                },
            }
            torch.save(payload, Path(tmpdir) / "sample.pt")

            dataset = CachedLogitsDataset(
                tmpdir,
                mc_input_representation="log_count",
                mc_log_count_alpha=1.0,
            )

            expected_first = torch.zeros(8)
            expected_first[2] = torch.log(torch.tensor(4.0))
            expected_first[3] = torch.log(torch.tensor(2.0))
            expected_first[4] = torch.log(torch.tensor(2.0))
            torch.testing.assert_close(dataset.logits[0], expected_first)
            self.assertEqual(dataset.mc_input_representation, "log_count")
            self.assertEqual(dataset.mc_log_count_alpha, 1.0)
            self.assertEqual(dataset.mc_samples_per_token, 5)

    def test_dual_representation_preserves_floor_base_and_log_count_feature(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sampled_ids = torch.tensor(
                [[2, 2, 2, 3, 4]] * 4, dtype=torch.long
            )
            legacy = sampled_ids_to_log_probs(
                sampled_ids,
                vocab_size=8,
                observed_alpha=0.1,
                floor_mass=1e-4,
                dtype=torch.float16,
            ).unsqueeze(0)
            torch.save(
                {
                    "log_probs": legacy,
                    "labels": torch.tensor([[2, 2, 2, 2]], dtype=torch.long),
                    "valid_sample_counts": torch.full((1, 4), 5, dtype=torch.long),
                    "metadata": {
                        "observed_alpha": 0.1,
                        "floor_mass": 1e-4,
                        "sample_temperature": 1.0,
                        "top_p": 1.0,
                        "sample_completion_policy": "exact",
                    },
                },
                Path(tmpdir) / "sample.pt",
            )

            dataset = CachedLogitsDataset(
                tmpdir,
                mc_input_representation="log_count",
                mc_base_score_representation="floor_logprob",
                mc_log_count_alpha=1.0,
            )

            torch.testing.assert_close(dataset.logits, legacy[0].float())
            self.assertEqual(dataset.feature_logits[0, 0].item(), 0.0)
            self.assertAlmostEqual(
                dataset.feature_logits[0, 2].item(),
                torch.log(torch.tensor(4.0)).item(),
                places=5,
            )
            item = dataset[0]
            torch.testing.assert_close(item[0], legacy[0, 0].float())
            torch.testing.assert_close(item[6], dataset.feature_logits[0])
            self.assertEqual(dataset.mc_base_score_representation, "floor_logprob")

    def test_proxy_fusion_uses_native_counts_and_preserves_fused_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fused = torch.log_softmax(
                torch.tensor(
                    [[[2.0, 0.0, -1.0, -2.0], [0.0, 3.0, -1.0, -2.0]]]
                ),
                dim=-1,
            ).to(torch.float16)
            counts = torch.tensor(
                [[[0, 2, 1, 0], [3, 0, 0, 0]]], dtype=torch.uint8
            )
            fusion_metadata = {
                "mc_fusion_mode": "proxy_dirichlet_v1",
                "proxy_model_name_or_path": "Qwen/Qwen3-1.7B",
                "proxy_model_revision": "proxy-revision",
                "proxy_tokenizer_sha256": "tokenizer-sha",
                "proxy_temperature": 1.25,
                "proxy_prior_strength": 2.0,
                "proxy_chat_template_protocol": "messages_for_prefix_qwen_v1",
                "shared_vocab_size": 4,
                "proxy_vocab_size": 6,
                "proxy_vocab_tail_policy": "crop_to_shared_vocab_before_softmax",
                "proxy_dtype": "float16",
                "proxy_quantization": "none",
            }
            torch.save(
                {
                    "log_probs": fused,
                    "mc_counts": counts,
                    "labels": torch.tensor([[1, 0]], dtype=torch.long),
                    "valid_sample_counts": torch.tensor([[3, 3]], dtype=torch.long),
                    "metadata": {
                        "observed_alpha": 0.1,
                        "floor_mass": 1e-4,
                        "sample_temperature": 1.0,
                        "top_p": 1.0,
                        "sample_completion_policy": "exact",
                        **fusion_metadata,
                    },
                },
                Path(tmpdir) / "sample.pt",
            )

            dataset = CachedLogitsDataset(
                tmpdir,
                mc_input_representation="log_count",
                mc_base_score_representation="floor_logprob",
                mc_log_count_alpha=1.0,
            )

            torch.testing.assert_close(dataset.logits, fused[0].float())
            torch.testing.assert_close(
                dataset.feature_logits,
                torch.log1p(counts[0].float()),
            )
            self.assertEqual(dataset.mc_fusion_mode, "proxy_dirichlet_v1")
            self.assertEqual(dataset.proxy_model_name_or_path, "Qwen/Qwen3-1.7B")
            self.assertEqual(dataset.proxy_temperature, 1.25)
            self.assertEqual(dataset.proxy_prior_strength, 2.0)
            self.assertEqual(dataset.proxy_dtype, "float16")
            self.assertEqual(dataset.proxy_quantization, "none")

            dense_dataset = CachedLogitsDataset(
                tmpdir,
                mc_input_representation="floor_logprob",
                mc_base_score_representation="floor_logprob",
            )
            torch.testing.assert_close(dense_dataset.logits, fused[0].float())
            torch.testing.assert_close(
                dense_dataset.feature_logits,
                fused[0].float(),
            )
            self.assertEqual(dense_dataset.mc_samples_per_token, 3)
            self.assertEqual(dense_dataset.mc_observed_alpha, 0.1)
            self.assertEqual(dense_dataset.mc_floor_mass, 1e-4)
            self.assertEqual(dense_dataset.mc_sample_temperature, 1.0)
            self.assertEqual(dense_dataset.mc_top_p, 1.0)
            self.assertEqual(dense_dataset.mc_completion_policy, "exact")

    def test_proxy_fusion_requires_mc_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = {
                "observed_alpha": 0.1,
                "floor_mass": 1e-4,
                "sample_temperature": 1.0,
                "top_p": 1.0,
                "sample_completion_policy": "exact",
                "mc_fusion_mode": "proxy_dirichlet_v1",
            }
            torch.save(
                {
                    "log_probs": torch.randn(1, 1, 4),
                    "labels": torch.tensor([[1]]),
                    "valid_sample_counts": torch.tensor([[3]]),
                    "metadata": metadata,
                },
                Path(tmpdir) / "sample.pt",
            )

            with self.assertRaisesRegex(ValueError, "no mc_counts tensor"):
                CachedLogitsDataset(
                    tmpdir,
                    mc_input_representation="log_count",
                    mc_base_score_representation="floor_logprob",
                )


if __name__ == "__main__":
    unittest.main()
