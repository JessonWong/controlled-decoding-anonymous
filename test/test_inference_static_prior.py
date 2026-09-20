import argparse
import math
import unittest

import torch

from inference_openrouter import (
    apply_static_prior_to_mc_samples,
    load_static_prior_log_probs,
    resolve_static_prior_configuration,
)
from modeling_biasnet import BiasConfig, BiasNet


class StaticPriorRuntimeTest(unittest.TestCase):
    def make_args(self, mode=None, strength=None, path=None):
        return argparse.Namespace(
            mc_static_prior_mode=mode,
            mc_static_prior_strength=strength,
            mc_static_prior_path=path,
        )

    def make_model(self):
        config = BiasConfig(hidden_size=4, vocab_size=5)
        config.mc_static_prior_mode = "uniform_dirichlet_v1"
        config.mc_static_prior_strength = 2.0
        config.mc_static_prior_path = None
        return BiasNet(config)

    def test_checkpoint_defaults_resolve_and_construct_uniform_prior(self):
        args = self.make_args()
        resolve_static_prior_configuration(args, [self.make_model()])
        load_static_prior_log_probs(args, vocab_size=5, device=torch.device("cpu"))

        self.assertEqual(args.mc_static_prior_mode, "uniform_dirichlet_v1")
        self.assertEqual(args.mc_static_prior_strength, 2.0)
        torch.testing.assert_close(
            args._mc_static_prior_log_probs,
            torch.full((5,), -math.log(5)),
        )

    def test_uniform_prior_posterior_uses_recorded_strength(self):
        args = self.make_args("uniform_dirichlet_v1", 2.0)
        resolve_static_prior_configuration(args, [self.make_model()])
        load_static_prior_log_probs(args, vocab_size=5, device=torch.device("cpu"))

        actual = apply_static_prior_to_mc_samples(
            [1, 1, 3],
            args,
            vocab_size=5,
            dtype=torch.float32,
            device=torch.device("cpu"),
        ).exp()
        expected = torch.tensor([0.4, 2.4, 0.4, 1.4, 0.4]) / 5.0

        torch.testing.assert_close(actual, expected)

    def test_runtime_prior_must_match_checkpoint(self):
        args = self.make_args("none")
        with self.assertRaisesRegex(ValueError, "does not match"):
            resolve_static_prior_configuration(args, [self.make_model()])


if __name__ == "__main__":
    unittest.main()
