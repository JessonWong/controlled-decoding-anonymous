import unittest

import torch

from training.anytime_proxy_mc import (
    POLICY_SPEC_HASH_FIELD,
    counts_to_events,
    parse_budgets,
    permute_events,
    policy_spec_payload_sha256,
    prefix_counts,
    split_records,
    target_protocol_metadata_from_manifest,
)
from training.train_anytime_biasnet import _mixed_prefix_counts


class AnytimeProxyMCTest(unittest.TestCase):
    def test_count_expansion_and_nested_prefixes_are_exact(self):
        counts = torch.tensor([[2, 0, 1, 2], [0, 3, 1, 1]], dtype=torch.uint8)
        events = counts_to_events(counts)
        permuted = permute_events(
            events,
            ["a:0", "b:0"],
            replay_seed=7,
            replicate=0,
        )
        at_two = prefix_counts(permuted, budget=2, vocab_size=4)
        at_five = prefix_counts(permuted, budget=5, vocab_size=4)

        torch.testing.assert_close(at_five.long(), counts.long())
        torch.testing.assert_close(at_two.sum(dim=-1), torch.tensor([2.0, 2.0]))
        self.assertTrue((at_two <= at_five).all())

    def test_replay_is_deterministic_and_replicate_specific(self):
        events = torch.arange(20).reshape(2, 10)
        keys = ["a:0", "b:0"]
        first = permute_events(events, keys, replay_seed=11, replicate=3)
        second = permute_events(events, keys, replay_seed=11, replicate=3)
        other = permute_events(events, keys, replay_seed=11, replicate=4)

        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, other))

    def test_record_split_is_deterministic_and_disjoint(self):
        names = [f"record-{index}.pt" for index in range(10)]
        first = split_records(
            names,
            test_count=2,
            calibration_count=2,
            split_seed="paper-v1",
        )
        second = split_records(
            list(reversed(names)),
            test_count=2,
            calibration_count=2,
            split_seed="paper-v1",
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 6)
        self.assertFalse(set(first["train"]) & set(first["test"]))
        self.assertFalse(set(first["calibration"]) & set(first["test"]))

    def test_budget_parser_requires_sorted_schedule_ending_at_max(self):
        self.assertEqual(parse_budgets("0,4,8,50", max_samples=50), (0, 4, 8, 50))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_budgets("0,8,4,50", max_samples=50)
        with self.assertRaisesRegex(ValueError, "end at 50"):
            parse_budgets("0,4,8", max_samples=50)
        with self.assertRaisesRegex(ValueError, "start at 0"):
            parse_budgets("4,8,50", max_samples=50)

    def test_mixed_budget_batch_has_exact_per_row_totals(self):
        events = torch.tensor([[1, 2, 3, 4], [4, 4, 2, 1], [0, 1, 2, 3]])
        budgets = torch.tensor([0, 2, 4])

        counts = _mixed_prefix_counts(
            events, budgets, vocab_size=5, dtype=torch.float32
        )

        torch.testing.assert_close(counts.sum(dim=-1), budgets.float())
        torch.testing.assert_close(counts[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0]))

    def test_target_protocol_is_frozen_from_source_manifest(self):
        configuration = {
            "api_url": "https://openrouter.ai/api/v1/chat/completions",
            "model": "qwen/qwen3-32b",
            "provider_order": ["DeepInfra"],
            "provider_allow_fallbacks": False,
            "append_no_think": True,
            "qwen_hard_no_think_prefill": True,
            "api_max_tokens": 1,
            "empty_response_token": "stop_eos",
            "disable_openrouter_response_cache": True,
            "sample_choices_per_request": 1,
            "max_sample_refill_rounds": 5,
            "empty_length_retry_max_tokens": None,
            "max_empty_length_retry_rounds": 0,
            "reasoning_mode": "enabled_false",
            "reject_reasoning_tokens": True,
        }

        protocol = target_protocol_metadata_from_manifest(
            {"source_manifest": {"configuration": configuration}}
        )

        self.assertEqual(protocol["target_model"], "qwen/qwen3-32b")
        self.assertEqual(protocol["target_provider_order"], ["DeepInfra"])
        self.assertEqual(protocol["target_api_max_tokens"], 1)
        self.assertEqual(protocol["target_max_reasoning_retries"], 0)
        self.assertIsNone(protocol["target_provider_quantizations"])

    def test_target_protocol_missing_core_field_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "target-protocol fields"):
            target_protocol_metadata_from_manifest(
                {"source_manifest": {"configuration": {}}}
            )

    def test_policy_payload_self_hash_detects_mutation(self):
        payload = {"schema_version": 2, "threshold": 0.75}
        payload[POLICY_SPEC_HASH_FIELD] = policy_spec_payload_sha256(payload)
        self.assertEqual(
            payload[POLICY_SPEC_HASH_FIELD], policy_spec_payload_sha256(payload)
        )
        payload["threshold"] = 0.5
        self.assertNotEqual(
            payload[POLICY_SPEC_HASH_FIELD], policy_spec_payload_sha256(payload)
        )


if __name__ == "__main__":
    unittest.main()
