import tempfile
import unittest
from pathlib import Path

import torch

from modeling_biasnet import BiasConfig, BiasNet


class BiasNetLmHeadLoraTest(unittest.TestCase):
    def make_model(self, rank=0, alpha=None):
        torch.manual_seed(7)
        return BiasNet(
            BiasConfig(
                hidden_size=8,
                vocab_size=19,
                input_projection_mode="count_sketch",
                count_sketch_hashes=2,
                lm_head_lora_rank=rank,
                lm_head_lora_alpha=alpha,
            )
        )

    def test_zero_initialized_lora_starts_identical_to_baseline(self):
        baseline = self.make_model()
        adapted = self.make_model(rank=3, alpha=6)
        baseline.eval()
        adapted.eval()

        logits = torch.randn(4, 19)
        torch.testing.assert_close(adapted(logits), baseline(logits))
        self.assertEqual(adapted.lm_head_lora_scale, 2.0)
        self.assertEqual(torch.count_nonzero(adapted.lm_head_lora_b.weight), 0)

    def test_lora_construction_preserves_control_rng_stream(self):
        self.make_model()
        baseline_rng_state = torch.random.get_rng_state()
        self.make_model(rank=3, alpha=6)
        adapted_rng_state = torch.random.get_rng_state()

        torch.testing.assert_close(adapted_rng_state, baseline_rng_state)

    def test_lora_output_factor_receives_gradient_while_base_head_stays_frozen(self):
        model = self.make_model(rank=3)
        for parameter in model.lm_head.parameters():
            parameter.requires_grad = False

        loss = model(torch.randn(2, 19)).square().mean()
        loss.backward()

        self.assertIsNone(model.lm_head.weight.grad)
        self.assertIsNotNone(model.lm_head_lora_b.weight.grad)
        self.assertGreater(
            torch.count_nonzero(model.lm_head_lora_b.weight.grad), 0
        )

    def test_lora_checkpoint_round_trip_preserves_outputs(self):
        model = self.make_model(rank=3, alpha=6)
        with torch.no_grad():
            model.lm_head_lora_b.weight.normal_(mean=0, std=0.01)
        model.eval()
        logits = torch.randn(2, 19)
        expected = model(logits)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = BiasNet.from_pretrained(Path(directory))
            loaded.eval()
            torch.testing.assert_close(loaded(logits), expected)
            self.assertEqual(loaded.lm_head_lora_rank, 3)
            self.assertEqual(loaded.lm_head_lora_alpha, 6)


if __name__ == "__main__":
    unittest.main()
