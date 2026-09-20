import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from materialize_proxy_mc_cache import build_teacher_forced_input


class BoundaryMergingTokenizer:
    """Toy tokenizer whose two trailing spaces merge only at end-of-text."""

    pad_token_id = 0
    eos_token_id = 0

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join({1: "A", 2: " ", 3: " "}[int(value)] for value in token_ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        return_tensors,
        continue_final_message=False,
        add_generation_prompt=False,
    ):
        del tokenize, return_tensors, continue_final_message, add_generation_prompt
        text = "H" + messages[-1]["content"]
        values = [9 if character == "H" else 8 for character in text]
        if text.endswith("  "):
            values = values[:-2] + [7]
        return torch.tensor([values], dtype=torch.long)


class ProxyMaterializerPrefixTest(unittest.TestCase):
    def test_independent_rendering_handles_non_nested_bpe_boundary(self):
        tokenizer = BoundaryMergingTokenizer()

        batch = build_teacher_forced_input(
            tokenizer,
            tokenizer,
            question="question",
            answer_token_ids=[1, 2, 3, 1],
            cached_positions=[0, 1, 2, 3],
            qwen_hard_no_think_prefill=True,
        )

        self.assertEqual(batch.row_count, 4)
        # Position 3 ends in two spaces, represented by the toy merged token 7.
        self.assertEqual(batch.input_ids[3, -1].item(), 7)
        self.assertTrue(batch.attention_mask[:, -1].all().item())
        # Left padding keeps each real sequence's position IDs at 0..length-1.
        for row, mask in zip(batch.position_ids, batch.attention_mask):
            length = int(mask.sum().item())
            self.assertEqual(row[mask.bool()].tolist(), list(range(length)))


if __name__ == "__main__":
    unittest.main()
