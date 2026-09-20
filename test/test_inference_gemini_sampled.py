import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_gemini_sampled import generate_one


class FakeTokenizer:
    eos_token_id = 0
    pieces = {0: "", 1: "a", 2: "b", 3: "c", 4: "d", 5: "x"}

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.pieces[int(token_id)] for token_id in token_ids)


class FakeSampler:
    vocab_size = 8

    def __init__(self):
        self.base_ids = iter((1, 2, 0))
        self.sample_calls = 0

    def deterministic_token_id(self, question, prefix):
        return next(self.base_ids), {}

    def draft_token_ids(self, question, prefix, remaining_tokens, draft_tokens):
        return [3, 4, 3], {
            "requested_tokens": min(remaining_tokens, draft_tokens),
            "finish_reason": "MAX_TOKENS",
        }

    def sample_ids(self, question, prefix):
        self.sample_calls += 1
        return [1] * 50, {"valid": 50, "calls": 50}


class FakeRiskGate:
    threshold = 0.1

    def score_prefixes(self, prompts, prefixes):
        if len(prefixes) > 1:
            return torch.tensor([1.0, 0.0, 0.0])
        return torch.tensor([1.0])


class FakeBiasNet:
    def __call__(self, log_probs):
        residual = torch.zeros_like(log_probs)
        residual[:, 5] = 100.0
        return residual


class GeminiSpeculativeGenerationTest(unittest.TestCase):
    def test_draft_rolls_back_at_first_biasnet_token(self):
        args = SimpleNamespace(
            biasnet_dtype="float32",
            max_new_tokens=5,
            risk_gate_speculative_draft=True,
            risk_gate_speculative_min_base_streak=2,
            risk_gate_speculative_draft_tokens=80,
            risk_gate_warmup_tokens=0,
            risk_gate_mode="soft",
            risk_gate_soft_temperature=0.05,
            risk_gate_min_scale=0.01,
            mc_samples_per_token=50,
            mc_observed_alpha=0.1,
            mc_floor_mass=1e-4,
            temperature=0.0,
            progress_steps=False,
            stop_on_mc_failure=False,
            risk_gate_trace=True,
            model="gemini-3.5-flash",
            tokenizer_name="fixture",
            biasnet_ckpt="fixture-biasnet",
            risk_gate_checkpoint="fixture-risk-gate",
            risk_gate_threshold=0.1,
            parallel_requests=50,
            candidate_count=1,
            api_max_output_tokens=8,
        )
        sampler = FakeSampler()
        completion, stats, runtime = generate_one(
            "fixture prompt",
            sampler,
            FakeTokenizer(),
            FakeBiasNet(),
            FakeRiskGate(),
            torch.device("cpu"),
            args,
        )

        self.assertEqual(completion, "abcx")
        self.assertEqual(sampler.sample_calls, 1)
        self.assertEqual(stats["mc_steps"], 1)
        self.assertEqual(runtime["summary"]["speculative_draft_calls"], 1)
        self.assertEqual(runtime["summary"]["speculative_verified_tokens"], 3)
        self.assertEqual(runtime["summary"]["speculative_accepted_tokens"], 1)
        self.assertEqual(runtime["summary"]["speculative_rejected_tokens"], 2)
        self.assertEqual(runtime["summary"]["speculative_rollbacks"], 1)
        self.assertEqual(
            runtime["steps"][3]["decision"],
            "speculative_rollback_soft_bias",
        )


if __name__ == "__main__":
    unittest.main()
