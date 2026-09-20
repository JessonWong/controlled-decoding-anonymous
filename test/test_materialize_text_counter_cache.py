import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from convert_legacy_cache_tokenizer import recover_mc_token_counts
from materialize_text_counter_cache import materialize_payload
from pre_logits_sampled_openweight import sampled_ids_to_log_probs


class FakeTokenizer:
    eos_token_id = 9

    def __len__(self):
        return 16

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [{"a": 1, "b": 2, "x": 3}[character] for character in text]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        inverse = {1: "a", 2: "b", 3: "x", 9: ""}
        return "".join(inverse[token_id] for token_id in token_ids)


class TextCounterMaterializationTest(unittest.TestCase):
    def test_materializes_raw_counters_and_labels(self):
        payload = {
            "sampled_completion_text_counts": [
                {"a": 2, "b": 1, "": 1},
                {"b": 4},
            ],
            "sampled_prefix_texts": ["", "a"],
            "prompt_text": "prompt",
            "answer_text": "ab",
            "labels": torch.tensor([[7, 8]]),
            "risk_gate_scores": torch.tensor([[0.1, 0.2]]),
            "metadata": {"observed_alpha": 0.1, "floor_mass": 1e-4},
        }
        output, summary = materialize_payload(
            payload,
            FakeTokenizer(),
            tokenizer_name="fake",
            alignment_policy="strict",
            store_dtype=torch.float32,
        )

        self.assertEqual(output["labels"].tolist(), [[1, 2]])
        self.assertEqual(tuple(output["log_probs"].shape), (1, 2, 16))
        self.assertTrue(
            torch.allclose(
                output["log_probs"].exp().sum(dim=-1),
                torch.ones(1, 2),
                atol=1e-6,
            )
        )
        self.assertEqual(output["valid_sample_counts"].tolist(), [[4, 4]])
        self.assertEqual(output["canonical_eos_projection_counts"].tolist(), [[1, 0]])
        self.assertTrue(
            torch.allclose(
                output["risk_gate_scores"], torch.tensor([[0.1, 0.2]])
            )
        )
        self.assertEqual(summary["retained_rows"], 2)

    def test_recovers_integer_counts_from_float16_legacy_row(self):
        sampled = torch.tensor([[2, 2, 2, 4, 4, 7]], dtype=torch.long)
        row = sampled_ids_to_log_probs(
            sampled,
            vocab_size=16,
            observed_alpha=0.1,
            floor_mass=1e-4,
            dtype=torch.float16,
        )[0]
        counts = recover_mc_token_counts(
            row,
            sample_count=6,
            observed_alpha=0.1,
            floor_mass=1e-4,
        )
        self.assertEqual(counts, {2: 3, 4: 2, 7: 1})


if __name__ == "__main__":
    unittest.main()
