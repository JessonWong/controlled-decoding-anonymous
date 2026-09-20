import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

import collect_outcome_regret_pairs as collector
from pre_logits_sampled_openrouter import OpenRouterResponse


class CharacterTokenizer:
    eos_token_id = 255
    all_special_ids = [255]

    def __len__(self):
        return 256

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        result = []
        index = 0
        while index < len(text):
            if text.startswith("<eos>", index):
                result.append(self.eos_token_id)
                index += len("<eos>")
            else:
                result.append(ord(text[index]))
                index += 1
        return result

    def decode(self, token_ids, skip_special_tokens=False):
        values = []
        for token_id in token_ids:
            if int(token_id) == self.eos_token_id:
                if not skip_special_tokens:
                    values.append("<eos>")
            else:
                values.append(chr(int(token_id)))
        return "".join(values)


class PrefixUnstableTokenizer:
    """Toy decode-to-API mapping with expected non-round-tripping prefixes."""

    eos_token_id = 9
    all_special_ids = [9]

    def __len__(self):
        return 10

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        mapping = {
            (1,): "teacher",
            (1, 2): "forced",
            (1, 3): "forced",
            (9,): "<eos>",
        }
        return mapping.get(tuple(int(value) for value in token_ids), "")

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {
            "teacher": [4],
            "forced": [5],
            "<eos>": [9],
            "": [],
        }[text]


class FakeClient:
    def __init__(self):
        self.args = SimpleNamespace(max_api_request_attempts=None)
        self.calls = 0
        self.request_attempts = 0
        self.total_cost = 0.0
        self.total_tokens = 0
        self.provider_call_counts = Counter()
        self.response_model_call_counts = Counter()
        self.response_cache_status_counts = Counter()
        self.api_max_tokens_call_counts = Counter()
        self.missing_router_metadata_calls = 0
        self.reasoning_response_calls = 0
        self.reasoning_tokens = 0


def prepared_fixture():
    tokenizer = CharacterTokenizer()
    plan = collector.PlanPair(
        plan_index=0,
        pair_id="orp-fixture",
        candidate_id="fixture-candidate",
        record_name="fixture.pt",
        dataset_idx=3,
        position=1,
        partial_action=66,
        reference_action=67,
        plan_row_sha256="plan-sha",
        plan_row={
            "record_name": "fixture.pt",
            "dataset_idx": 3,
            "position": 1,
            "partial_action": 66,
            "reference_action": 67,
            "feature": 1.25,
        },
    )
    partial = collector.validate_action_prefix(
        tokenizer,
        prefix_token_ids=[65],
        action_token_id=66,
        remaining_tokens=3,
        role="partial",
        completion_order=0,
    )
    reference = collector.validate_action_prefix(
        tokenizer,
        prefix_token_ids=[65],
        action_token_id=67,
        remaining_tokens=3,
        role="reference",
        completion_order=1,
    )
    pair = collector.PreparedPair(
        plan=plan,
        prompt="question",
        api_prompt="question\n/no_think",
        data_sha256="data-sha",
        answer_token_count=5,
        source_record_sha256="source-sha",
        partial=partial,
        reference=reference,
        partial_branch_serialization=collector.branch_serialization(
            "question\n/no_think", partial
        ),
        reference_branch_serialization=collector.branch_serialization(
            "question\n/no_think", reference
        ),
        partial_branch_serialization_sha256=collector.sha256_payload(
            collector.branch_serialization("question\n/no_think", partial)
        ),
        reference_branch_serialization_sha256=collector.sha256_payload(
            collector.branch_serialization("question\n/no_think", reference)
        ),
        structural_zero=False,
        structural_zero_reason=None,
    )
    preflight = collector.Preflight(
        pairs=(pair,),
        token_equal_rows=0,
        plan_row_count=3,
        selected_source_records=("fixture.pt",),
        planned_api_calls=2,
        planned_requested_output_tokens=6,
        proxy_manifest={},
        proxy_manifest_sha256="proxy-manifest-sha",
        source_manifest={},
        source_manifest_sha256="source-manifest-sha",
        dataset_fingerprint="dataset-fingerprint",
        dataset_revision="dataset-revision",
        tokenizer_sha256="tokenizer-sha",
        tokenizer_name_or_path="tokenizer",
        tokenizer_revision="tokenizer-revision",
        tokenizer=tokenizer,
    )
    return tokenizer, preflight


def structural_zero_fixture():
    tokenizer = PrefixUnstableTokenizer()
    plan = collector.PlanPair(
        plan_index=0,
        pair_id="orp-structural-zero",
        candidate_id="structural-zero-candidate",
        record_name="fixture.pt",
        dataset_idx=3,
        position=1,
        partial_action=2,
        reference_action=3,
        plan_row_sha256="plan-sha",
        plan_row={
            "record_name": "fixture.pt",
            "dataset_idx": 3,
            "position": 1,
            "partial_action": 2,
            "reference_action": 3,
        },
    )
    partial = collector.validate_action_prefix(
        tokenizer,
        prefix_token_ids=[1],
        action_token_id=2,
        remaining_tokens=4,
        role="partial",
        completion_order=0,
    )
    reference = collector.validate_action_prefix(
        tokenizer,
        prefix_token_ids=[1],
        action_token_id=3,
        remaining_tokens=4,
        role="reference",
        completion_order=1,
    )
    api_prompt = "question\n/no_think"
    partial_serialization = collector.branch_serialization(api_prompt, partial)
    reference_serialization = collector.branch_serialization(api_prompt, reference)
    assert partial_serialization == reference_serialization
    pair = collector.PreparedPair(
        plan=plan,
        prompt="question",
        api_prompt=api_prompt,
        data_sha256="data-sha",
        answer_token_count=5,
        source_record_sha256="source-sha",
        partial=partial,
        reference=reference,
        partial_branch_serialization=partial_serialization,
        reference_branch_serialization=reference_serialization,
        partial_branch_serialization_sha256=collector.sha256_payload(
            partial_serialization
        ),
        reference_branch_serialization_sha256=collector.sha256_payload(
            reference_serialization
        ),
        structural_zero=True,
        structural_zero_reason="identical_serialized_branch_requests",
    )
    return collector.Preflight(
        pairs=(pair,),
        token_equal_rows=0,
        plan_row_count=1,
        selected_source_records=("fixture.pt",),
        planned_api_calls=0,
        planned_requested_output_tokens=0,
        proxy_manifest={},
        proxy_manifest_sha256="proxy-manifest-sha",
        source_manifest={},
        source_manifest_sha256="source-manifest-sha",
        dataset_fingerprint="dataset-fingerprint",
        dataset_revision="dataset-revision",
        tokenizer_sha256="tokenizer-sha",
        tokenizer_name_or_path="tokenizer",
        tokenizer_revision="tokenizer-revision",
        tokenizer=tokenizer,
    )


def successful_generator(call_log):
    def generate(**kwargs):
        client = kwargs["client"]
        call_log.append(dict(kwargs))
        client.calls += 1
        client.request_attempts += 1
        client.total_cost += 0.001
        client.total_tokens += 7
        client.provider_call_counts[collector.TARGET_PROVIDER] += 1
        client.response_model_call_counts[collector.TARGET_MODEL] += 1
        client.response_cache_status_counts["MISS"] += 1
        client.api_max_tokens_call_counts[kwargs["remaining_tokens"]] += 1
        return "!", {
            "reasoning_detected": False,
            "response_model": collector.TARGET_MODEL,
            "remaining_tokens_requested": kwargs["remaining_tokens"],
            "generation_id": f"gen-{len(call_log)}",
        }

    return generate


class PlanAndPrefixTest(unittest.TestCase):
    def test_plan_keeps_equal_actions_until_serialization_is_reconstructed(self):
        rows = [
            {
                "record_name": "a.pt",
                "dataset_idx": 1,
                "position": 0,
                "partial_action": 4,
                "reference_action": 4,
            },
            {
                "record_name": "a.pt",
                "dataset_idx": 1,
                "position": 1,
                "partial_action": 4,
                "reference_action": 5,
            },
            {
                "record_name": "a.pt",
                "dataset_idx": 1,
                "position": 2,
                "partial_action": 6,
                "reference_action": 7,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            pairs, skipped, total = collector.load_plan(path, max_pairs=1)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].position, 0)
        self.assertEqual(skipped, 1)
        self.assertEqual(total, 3)

    def test_unique_candidate_ids_allow_replicated_branch_identities(self):
        common = {
            "record_name": "a.pt",
            "dataset_idx": 1,
            "position": 7,
            "partial_action": 4,
            "reference_action": 5,
        }
        rows = [
            {**common, "candidate_id": "candidate-k4", "budget": 4},
            {**common, "candidate_id": "candidate-k8", "budget": 8},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            pairs, _, _ = collector.load_plan(path)

            self.assertEqual(len(pairs), 2)
            self.assertEqual(
                [pair.candidate_id for pair in pairs],
                ["candidate-k4", "candidate-k8"],
            )
            self.assertEqual(len({pair.pair_id for pair in pairs}), 2)

            duplicate = rows + [dict(rows[0])]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in duplicate),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate candidate_id"):
                collector.load_plan(path)

    def test_eos_action_is_local_terminal_and_round_trips(self):
        tokenizer = CharacterTokenizer()
        arm = collector.validate_action_prefix(
            tokenizer,
            prefix_token_ids=[65],
            action_token_id=tokenizer.eos_token_id,
            remaining_tokens=10,
            role="partial",
            completion_order=0,
        )

        self.assertTrue(arm.local_terminal)
        self.assertEqual(arm.visible_forced_prefix, "A")
        self.assertEqual(arm.raw_forced_prefix, "A<eos>")

    def test_non_eos_special_action_fails_closed(self):
        tokenizer = CharacterTokenizer()
        tokenizer.all_special_ids = [254, 255]
        with self.assertRaisesRegex(ValueError, "non-EOS special"):
            collector.validate_action_prefix(
                tokenizer,
                prefix_token_ids=[65],
                action_token_id=254,
                remaining_tokens=2,
                role="partial",
                completion_order=0,
            )

    def test_prefix_instability_is_audited_instead_of_rejected(self):
        tokenizer = PrefixUnstableTokenizer()
        arm = collector.validate_action_prefix(
            tokenizer,
            prefix_token_ids=[1],
            action_token_id=2,
            remaining_tokens=4,
            role="partial",
            completion_order=0,
        )

        self.assertFalse(arm.teacher_prefix_roundtrip)
        self.assertEqual(arm.teacher_prefix_retokenized_ids, (4,))
        self.assertFalse(arm.forced_prefix_roundtrip)
        self.assertEqual(arm.forced_prefix_retokenized_ids, (5,))
        self.assertFalse(arm.forced_extends_teacher_text)
        self.assertEqual(arm.raw_forced_prefix, "forced")


class CollectionStateTest(unittest.TestCase):
    def test_collector_wrapper_allows_audited_local_boundary_overflow(self):
        tokenizer = CharacterTokenizer()

        class ResponseClient:
            args = SimpleNamespace()

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return OpenRouterResponse(
                    content="!!",
                    finish_reason="length",
                    native_finish_reason="length",
                    usage={},
                    model=collector.TARGET_MODEL,
                    request_max_tokens=kwargs["max_tokens"],
                    request_temperature=kwargs["temperature"],
                    request_reasoning_mode="enabled_false",
                )

        client = ResponseClient()
        continuation, audit = collector.generate_collector_continuation(
            client=client,
            tokenizer=tokenizer,
            prompt="question\n/no_think",
            prefix_text="AB",
            remaining_tokens=1,
            temperature=0.0,
            top_p=1.0,
        )

        self.assertEqual(continuation, "!!")
        self.assertEqual(audit["continuation_local_token_overflow"], 1)
        self.assertEqual(client.kwargs["max_tokens"], 1)

    def test_mismatch_uses_exactly_two_t0_calls_and_resume_sends_none(self):
        _, preflight = prepared_fixture()
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pairs.jsonl"
            state = collector.state_directory(output)
            state.mkdir()
            client = FakeClient()
            summary = collector.collect_prepared_pairs(
                preflight=preflight,
                output_path=output,
                state_root=state,
                configuration_fingerprint="configuration",
                hard_cap=2,
                client=client,
                generator=successful_generator(calls),
            )

            self.assertEqual(len(calls), 2)
            self.assertTrue(all(call["temperature"] == 0.0 for call in calls))
            self.assertTrue(all(call["top_p"] == 1.0 for call in calls))
            self.assertTrue(all(call["prompt"].endswith("\n/no_think") for call in calls))
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["arm_role"] for row in rows}, {"partial", "reference"})
            self.assertEqual({row["logibreak_group_id"] for row in rows}, {"orp-fixture"})
            self.assertEqual(summary["confirmed_request_attempts"], 2)
            self.assertAlmostEqual(summary["api_cost"], 0.002)

            resumed_calls = []
            resumed_client = FakeClient()
            resumed = collector.collect_prepared_pairs(
                preflight=preflight,
                output_path=output,
                state_root=state,
                configuration_fingerprint="configuration",
                hard_cap=2,
                client=resumed_client,
                generator=successful_generator(resumed_calls),
            )
            self.assertEqual(resumed_calls, [])
            self.assertEqual(resumed["completed_pairs"], 1)

    def test_cap_fails_before_intent_or_request(self):
        _, preflight = prepared_fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pairs.jsonl"
            state = collector.state_directory(output)
            state.mkdir()
            with self.assertRaisesRegex(ValueError, "hard global request cap"):
                collector.collect_prepared_pairs(
                    preflight=preflight,
                    output_path=output,
                    state_root=state,
                    configuration_fingerprint="configuration",
                    hard_cap=1,
                    client=FakeClient(),
                    generator=successful_generator([]),
                )
            self.assertEqual(list(state.iterdir()), [])

    def test_serialized_request_equality_is_zero_cost_even_for_different_ids(self):
        preflight = structural_zero_fixture()
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pairs.jsonl"
            state = collector.state_directory(output)
            state.mkdir()
            summary = collector.collect_prepared_pairs(
                preflight=preflight,
                output_path=output,
                state_root=state,
                configuration_fingerprint="configuration",
                hard_cap=1,
                client=FakeClient(),
                generator=successful_generator(calls),
            )

            self.assertEqual(calls, [])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["completion"], rows[1]["completion"])
            self.assertEqual(
                rows[0]["branch_serialization_sha256"],
                rows[1]["branch_serialization_sha256"],
            )
            self.assertTrue(all(row["structural_zero"] for row in rows))
            self.assertTrue(
                all(
                    row["structural_zero_reason"]
                    == "identical_serialized_branch_requests"
                    for row in rows
                )
            )
            self.assertEqual(summary["confirmed_request_attempts"], 0)
            self.assertEqual(summary["structural_zero_pairs"], 1)

    def test_failed_request_becomes_unresolved_and_is_never_resent(self):
        _, preflight = prepared_fixture()

        def fail_after_attempt(**kwargs):
            client = kwargs["client"]
            client.request_attempts += 1
            raise RuntimeError("network outcome unknown")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pairs.jsonl"
            state = collector.state_directory(output)
            state.mkdir()
            with self.assertRaisesRegex(RuntimeError, "outcome unknown"):
                collector.collect_prepared_pairs(
                    preflight=preflight,
                    output_path=output,
                    state_root=state,
                    configuration_fingerprint="configuration",
                    hard_cap=2,
                    client=FakeClient(),
                    generator=fail_after_attempt,
                )
            saved = [json.loads(path.read_text()) for path in state.iterdir()]
            self.assertEqual([item["status"] for item in saved], ["uncertain"])
            self.assertEqual(output.read_text(), "")

            calls = []
            with self.assertRaisesRegex(RuntimeError, "unresolved persisted intents"):
                collector.collect_prepared_pairs(
                    preflight=preflight,
                    output_path=output,
                    state_root=state,
                    configuration_fingerprint="configuration",
                    hard_cap=2,
                    client=FakeClient(),
                    generator=successful_generator(calls),
                )
            self.assertEqual(calls, [])

    def test_verified_overflow_recovery_archives_and_accounts_historical_spend(self):
        _, preflight = prepared_fixture()
        pair = preflight.pairs[0]
        arm = pair.partial
        identifier = collector.arm_id(pair.plan.pair_id, arm.role)
        historical_cost = 0.00002324
        uncertain = {
            "state_schema_version": 1,
            "status": "uncertain",
            "arm_id": identifier,
            "pair_id": pair.plan.pair_id,
            "record_name": pair.plan.record_name,
            "dataset_idx": pair.plan.dataset_idx,
            "position": pair.plan.position,
            "role": arm.role,
            "action_token_id": arm.action_token_id,
            "remaining_tokens": arm.remaining_tokens,
            "configuration_fingerprint": "configuration",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "updated_at_utc": "2026-01-01T00:00:01+00:00",
            "error_type": "RuntimeError",
            "error": (
                "Gate handoff exceeded its remaining local token budget: "
                f"{arm.remaining_tokens + 1} > {arm.remaining_tokens}."
            ),
            "observed_client_delta": {
                "calls": 1,
                "request_attempts": 1,
                "total_cost": historical_cost,
                "total_tokens": 108,
            },
        }
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pairs.jsonl"
            state = collector.state_directory(output)
            state.mkdir()
            collector.atomic_write_json(
                collector.state_path(state, identifier), uncertain
            )
            ledger_root = collector.reconciliation_directory(state)
            ledger_export = collector.reconciliation_export_path(output)

            summary = collector.collect_prepared_pairs(
                preflight=preflight,
                output_path=output,
                state_root=state,
                configuration_fingerprint="configuration",
                hard_cap=3,
                client=FakeClient(),
                generator=successful_generator(calls),
                recover_verified_local_overflow=True,
                reconciliation_root=ledger_root,
                reconciliation_output_path=ledger_export,
            )

            self.assertEqual(len(calls), 2)
            self.assertEqual(summary["successful_result_request_attempts"], 2)
            self.assertEqual(
                summary["historical_reconciled_failed_request_attempts"], 1
            )
            self.assertEqual(summary["confirmed_request_attempts"], 3)
            self.assertAlmostEqual(
                summary["historical_reconciled_failed_api_cost"], historical_cost
            )
            self.assertAlmostEqual(summary["api_cost"], 0.002 + historical_cost)
            ledgers = [
                json.loads(line) for line in ledger_export.read_text().splitlines()
            ]
            self.assertEqual(len(ledgers), 1)
            self.assertEqual(ledgers[0]["original_uncertain_state"], uncertain)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            recovered = next(row for row in rows if row["arm_id"] == identifier)
            self.assertEqual(recovered["reconciliation"]["retry_ordinal"], 1)
            self.assertEqual(
                recovered["reconciliation"]["historical_request_audit"][
                    "api_request_attempts"
                ],
                1,
            )

    def test_recovery_flag_does_not_allow_other_uncertain_errors(self):
        _, preflight = prepared_fixture()
        pair = preflight.pairs[0]
        arm = pair.partial
        identifier = collector.arm_id(pair.plan.pair_id, arm.role)
        uncertain = {
            "state_schema_version": 1,
            "status": "uncertain",
            "arm_id": identifier,
            "pair_id": pair.plan.pair_id,
            "record_name": pair.plan.record_name,
            "dataset_idx": pair.plan.dataset_idx,
            "position": pair.plan.position,
            "role": arm.role,
            "action_token_id": arm.action_token_id,
            "remaining_tokens": arm.remaining_tokens,
            "configuration_fingerprint": "configuration",
            "error_type": "RuntimeError",
            "error": "network outcome unknown",
            "observed_client_delta": {
                "calls": 0,
                "request_attempts": 1,
                "total_cost": 0.0,
                "total_tokens": 0,
            },
        }
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pairs.jsonl"
            state = collector.state_directory(output)
            state.mkdir()
            collector.atomic_write_json(
                collector.state_path(state, identifier), uncertain
            )
            with self.assertRaisesRegex(ValueError, "not the verified local-token-overflow"):
                collector.collect_prepared_pairs(
                    preflight=preflight,
                    output_path=output,
                    state_root=state,
                    configuration_fingerprint="configuration",
                    hard_cap=3,
                    client=FakeClient(),
                    generator=successful_generator(calls),
                    recover_verified_local_overflow=True,
                )
            self.assertEqual(calls, [])

    def test_dry_run_never_resolves_api_key(self):
        _, preflight = prepared_fixture()
        argv = [
            "--plan",
            "unused-plan.json",
            "--proxy_cache_dir",
            "unused-cache",
            "--output_jsonl",
            "unused-output.jsonl",
            "--dry_run",
        ]
        with mock.patch.object(collector, "run_preflight", return_value=preflight), mock.patch.object(
            collector, "resolve_api_key", side_effect=AssertionError("key read")
        ) as resolve_key, mock.patch("builtins.print"):
            collector.main(argv)
        resolve_key.assert_not_called()


if __name__ == "__main__":
    unittest.main()
