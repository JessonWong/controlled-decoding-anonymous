import json
from pathlib import Path
import tempfile
import unittest

import torch

from modeling_biasnet import BiasConfig, BiasNet
from modeling_biasnet import COUNT_SKETCH_INPUT_CENTERING_ROW_MIN
from modeling_biasnet import MC_SAMPLE_COUNT_CONDITIONING_EXACT_ANCHOR
from training.train_anytime_biasnet import _configure_trainable_parameters


class PositionAwareBiasNetTest(unittest.TestCase):
    def test_config_supports_transformers_default_instantiation(self):
        config = BiasConfig()

        self.assertIsNone(config.hidden_size)
        self.assertIsNone(config.output_head_hidden_size)

    def make_model(self) -> BiasNet:
        torch.manual_seed(7)
        model = BiasNet(BiasConfig(hidden_size=4, vocab_size=8))
        model.set_up_proj()
        model.eval()
        return model

    def test_zero_initialized_position_bias_preserves_existing_output(self):
        model = self.make_model()
        logits = torch.randn(2, 8)
        expected = model(logits)

        model.enable_position_bias(3)
        actual = model(logits, position_ids=torch.tensor([0, 2]))

        torch.testing.assert_close(actual, expected)

    def test_count_sketch_layernorm_is_deterministic_and_scale_stable(self):
        config = BiasConfig(
            hidden_size=8,
            vocab_size=32,
            input_projection_mode="count_sketch",
            input_hidden_normalization="layernorm",
            count_sketch_hashes=4,
            count_sketch_seed=17,
        )
        model = BiasNet(config)
        model.set_up_proj()
        features = torch.zeros(2, 32)
        features[0, [2, 7, 19]] = torch.tensor([1.0, 2.0, 3.0])
        features[1] = features[0] * 100

        hidden = model.inverse_mapping(features)

        self.assertIsNone(model.up_proj)
        torch.testing.assert_close(hidden[0], hidden[1], atol=1e-5, rtol=1e-5)
        self.assertAlmostEqual(hidden[0].mean().item(), 0.0, places=5)
        self.assertAlmostEqual(
            hidden[0].pow(2).mean().sqrt().item(), 1.0, places=4
        )

    def test_count_sketch_configuration_round_trips(self):
        model = BiasNet(
            BiasConfig(
                hidden_size=8,
                vocab_size=32,
                input_projection_mode="count_sketch",
                input_hidden_normalization="layernorm",
                count_sketch_hashes=3,
                count_sketch_seed=99,
                count_sketch_input_centering=COUNT_SKETCH_INPUT_CENTERING_ROW_MIN,
            )
        )
        model.set_up_proj()
        model.eval()
        features = torch.zeros(1, 32)
        features[0, [1, 5]] = torch.tensor([1.0, 2.0])
        expected = model(features)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            restored = BiasNet.from_pretrained(tmpdir)
            restored.set_up_proj()
            restored.eval()
            actual = restored(features)

        self.assertEqual(restored.input_projection_mode, "count_sketch")
        self.assertEqual(restored.count_sketch_hashes, 3)
        self.assertEqual(
            restored.count_sketch_input_centering,
            COUNT_SKETCH_INPUT_CENTERING_ROW_MIN,
        )
        torch.testing.assert_close(actual, expected)

    def test_count_sketch_row_min_centering_removes_dense_floor(self):
        centered = BiasNet(
            BiasConfig(
                hidden_size=8,
                vocab_size=32,
                input_projection_mode="count_sketch",
                input_hidden_normalization="none",
                count_sketch_hashes=4,
                count_sketch_seed=17,
                count_sketch_input_centering=COUNT_SKETCH_INPUT_CENTERING_ROW_MIN,
            )
        )
        legacy = BiasNet(
            BiasConfig(
                hidden_size=8,
                vocab_size=32,
                input_projection_mode="count_sketch",
                input_hidden_normalization="none",
                count_sketch_hashes=4,
                count_sketch_seed=17,
            )
        )
        dense_floor = torch.full((2, 32), -9.0)
        dense_floor[0, [2, 7, 19]] = torch.tensor([-1.0, -3.0, -2.0])
        dense_floor[1, [1, 5]] = torch.tensor([-4.0, -2.0])
        explicit_centered = dense_floor - dense_floor.amin(dim=-1, keepdim=True)

        actual = centered.inverse_mapping(dense_floor)
        expected = legacy.inverse_mapping(explicit_centered)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_count_sketch_row_min_centering_is_shift_invariant(self):
        model = BiasNet(
            BiasConfig(
                hidden_size=8,
                vocab_size=32,
                input_projection_mode="count_sketch",
                input_hidden_normalization="layernorm",
                count_sketch_hashes=4,
                count_sketch_seed=17,
                count_sketch_input_centering=COUNT_SKETCH_INPUT_CENTERING_ROW_MIN,
            )
        )
        features = torch.full((2, 32), -8.0)
        features[0, [2, 7, 19]] = torch.tensor([-1.0, -3.0, -2.0])
        features[1] = features[0] + 123.0

        hidden = model.inverse_mapping(features)

        torch.testing.assert_close(hidden[0], hidden[1], atol=1e-5, rtol=1e-5)

    def test_position_aware_checkpoint_requires_and_uses_position_ids(self):
        model = self.make_model()
        model.enable_position_bias(3)
        logits = torch.randn(1, 8)

        with self.assertRaisesRegex(ValueError, "position_ids"):
            model(logits)

        with torch.no_grad():
            model.position_embedding.weight[1].fill_(1.0)
        position_zero = model(logits, position_ids=0)
        position_one = model(logits, position_ids=1)
        self.assertFalse(torch.equal(position_zero, position_one))

    def test_position_embedding_round_trips_through_checkpoint(self):
        model = self.make_model()
        model.enable_position_bias(3)
        with torch.no_grad():
            model.position_embedding.weight[2].fill_(0.25)
        logits = torch.randn(1, 8)
        expected = model(logits, position_ids=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            restored = BiasNet.from_pretrained(tmpdir)
            restored.set_up_proj()
            restored.eval()
            actual = restored(logits, position_ids=2)

        self.assertEqual(restored.num_position_buckets, 3)
        torch.testing.assert_close(actual, expected)

    def test_zero_initialized_mc_count_conditioning_preserves_output(self):
        model = self.make_model()
        logits = torch.randn(2, 8)
        expected = model(logits)

        model.enable_mc_sample_count_conditioning(50)
        actual = model(logits, mc_sample_counts=torch.tensor([0, 50]))

        torch.testing.assert_close(actual, expected)

    def test_mc_count_conditioning_requires_valid_counts(self):
        model = self.make_model()
        model.enable_mc_sample_count_conditioning(50)
        logits = torch.randn(1, 8)

        with self.assertRaisesRegex(ValueError, "mc_sample_counts"):
            model(logits)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            model(logits, mc_sample_counts=-1)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            model(logits, mc_sample_counts=51)

    def test_mc_count_conditioning_changes_and_round_trips(self):
        model = self.make_model()
        model.enable_mc_sample_count_conditioning(50)
        with torch.no_grad():
            model.mc_sample_count_projection.weight.copy_(
                torch.arange(1, 5, dtype=torch.float32).unsqueeze(1)
            )
        logits = torch.randn(1, 8)
        hidden = model.inverse_mapping(logits)
        at_zero_bias = model.mc_sample_count_hidden_bias(0, hidden)
        at_fifty_bias = model.mc_sample_count_hidden_bias(50, hidden)
        at_fifty = model(logits, mc_sample_counts=50)
        self.assertFalse(torch.equal(at_zero_bias, at_fifty_bias))

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            # Simulate an anytime checkpoint written before the mode field was
            # introduced. Such checkpoints must retain their affine module and
            # state-dict interpretation.
            config_path = Path(tmpdir) / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.pop("mc_sample_count_conditioning_mode")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            restored = BiasNet.from_pretrained(tmpdir)
            restored.set_up_proj()
            restored.eval()
            actual = restored(logits, mc_sample_counts=50)

        self.assertTrue(restored.mc_sample_count_conditioning)
        self.assertEqual(restored.mc_max_samples_per_token, 50)
        torch.testing.assert_close(actual, at_fifty)

    def test_exact_anchor_count_vector_preserves_original_kmax_logits(self):
        reference = self.make_model()
        conditioned = self.make_model()
        conditioned.load_state_dict(reference.state_dict())
        conditioned.enable_mc_sample_count_conditioning(
            50, mode=MC_SAMPLE_COUNT_CONDITIONING_EXACT_ANCHOR
        )
        with torch.no_grad():
            conditioned.mc_sample_count_vector.copy_(
                torch.tensor([3.0, -2.0, 1.0, 4.0])
            )
        logits = torch.randn(2, 8)

        expected = reference(logits)
        actual = conditioned(logits, mc_sample_counts=torch.tensor([50, 50]))
        hidden = conditioned.inverse_mapping(logits)
        anchor_bias = conditioned.mc_sample_count_hidden_bias(50, hidden)

        self.assertTrue(torch.equal(anchor_bias, torch.zeros_like(anchor_bias)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_exact_anchor_count_only_freezes_everything_but_vector(self):
        model = self.make_model()
        model.enable_mc_sample_count_conditioning(
            50, mode=MC_SAMPLE_COUNT_CONDITIONING_EXACT_ANCHOR
        )

        trainable = _configure_trainable_parameters(
            model, freeze_except_mc_conditioning=True
        )

        self.assertEqual([name for name, _ in trainable], ["mc_sample_count_vector"])
        self.assertEqual(sum(parameter.numel() for _, parameter in trainable), 4)

    def test_exact_anchor_count_vector_round_trips(self):
        model = self.make_model()
        model.enable_mc_sample_count_conditioning(
            50, mode=MC_SAMPLE_COUNT_CONDITIONING_EXACT_ANCHOR
        )
        with torch.no_grad():
            model.mc_sample_count_vector.fill_(0.5)
        logits = torch.randn(1, 8)
        expected = model(logits, mc_sample_counts=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)
            restored = BiasNet.from_pretrained(tmpdir)
            restored.set_up_proj()
            restored.eval()
            actual = restored(logits, mc_sample_counts=4)

        self.assertEqual(
            restored.mc_sample_count_conditioning_mode,
            MC_SAMPLE_COUNT_CONDITIONING_EXACT_ANCHOR,
        )
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
