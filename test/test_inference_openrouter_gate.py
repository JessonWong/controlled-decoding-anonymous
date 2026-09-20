import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_openrouter import (
    BiasMCInputs,
    FatalOpenRouterResponseError,
    apply_bias_model,
    build_openrouter_routing_metadata,
    estimate_log_probs,
    generate_one,
    generate_speculative_draft,
    mc_ensemble_audit,
    parse_args,
    resolve_mc_input_configuration,
    risk_gate_bias_scale,
    snapshot_openrouter_audit,
    validate_handoff_response,
    validate_empty_length_retry_args,
    validate_mc_ensemble_args,
    validate_reasoning_retry_args,
    validate_risk_gate_runtime_args,
)
from pre_logits_sampled_openrouter import OpenRouterResponse


class FakeRiskGate:
    def __init__(self, score: float, threshold: float = 0.1):
        self.score = score
        self.threshold = threshold

    def score_prefixes(self, prompts, answer_prefixes):
        return torch.tensor([self.score], dtype=torch.float32)


class SequenceRiskGate:
    def __init__(self, scores, threshold: float = 0.1):
        self.scores = list(scores)
        self.threshold = threshold
        self.calls = []

    def score_prefixes(self, prompts, answer_prefixes):
        self.calls.append((list(prompts), list(answer_prefixes)))
        if not self.scores:
            raise AssertionError("Risk gate was scored more times than expected.")
        return torch.tensor([self.scores.pop(0)], dtype=torch.float32)


class BatchedSequenceRiskGate:
    def __init__(self, score_batches, threshold: float = 0.1):
        self.score_batches = [list(batch) for batch in score_batches]
        self.threshold = threshold
        self.calls = []

    def score_prefixes(self, prompts, answer_prefixes):
        self.calls.append((list(prompts), list(answer_prefixes)))
        if not self.score_batches:
            raise AssertionError("Risk gate was scored more times than expected.")
        scores = self.score_batches.pop(0)
        if len(scores) != len(prompts):
            raise AssertionError(f"Expected {len(prompts)} scores, found {len(scores)}.")
        return torch.tensor(scores, dtype=torch.float32)


class ConstantResidual(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.value = value

    def forward(self, logits):
        return torch.full_like(logits, self.value)


class IdentityResidual(torch.nn.Module):
    num_position_buckets = 0

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, logits):
        return logits


class PositionResidual(torch.nn.Module):
    num_position_buckets = 3

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.seen_position_ids = []

    def forward(self, logits, position_ids=None):
        self.seen_position_ids.append(position_ids)
        return torch.full_like(logits, float(position_ids))


class CharacterTokenizer:
    eos_token_id = None
    _pieces = {1: "A", 2: "B", 3: "C"}

    def __len__(self):
        return 8

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self._pieces[token_id] for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        return list(text)


def generation_args(**overrides):
    values = {
        "max_new_tokens": 4,
        "api_max_tokens": 1,
        "sample_completion_policy": "exact",
        "mc_independent_views": 1,
        "mc_samples_per_token": 50,
        "stop_on_mc_failure": False,
        "temperature": 0.0,
        "top_p": 1.0,
        "progress_steps": False,
        "risk_gate_mode": "soft",
        "risk_gate_threshold": 0.1,
        "risk_gate_soft_temperature": 1.0,
        "risk_gate_warmup_tokens": 0,
        "risk_gate_min_scale": 0.0,
        "risk_gate_latch_off": False,
        "risk_gate_latch_patience": 2,
        "risk_gate_latch_threshold": None,
        "risk_gate_handoff_on_latch": False,
        "risk_gate_trace": False,
        "generation_trace": False,
        "model": "z-ai/glm-5",
        "tokenizer_name": "zai-org/GLM-5",
        "biasnet_ckpt": "dense-checkpoint",
        "risk_gate_checkpoint": "risk-checkpoint",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def openrouter_response(
    content,
    *,
    model="z-ai/glm-5",
    finish_reason="stop",
    native_finish_reason="stop",
    choice_error=None,
    request_max_tokens=None,
    request_reasoning_mode=None,
    reasoning=None,
    reasoning_details=None,
    reasoning_tokens=0,
):
    return OpenRouterResponse(
        content=content,
        finish_reason=finish_reason,
        native_finish_reason=native_finish_reason,
        usage={},
        model=model,
        generation_id="generation-test",
        choice_error=choice_error,
        response_cache_status="miss",
        request_max_tokens=request_max_tokens,
        request_reasoning_mode=request_reasoning_mode,
        reasoning=reasoning,
        reasoning_details=reasoning_details,
        reasoning_tokens=reasoning_tokens,
    )


class OpenRouterRiskGateTest(unittest.TestCase):
    def test_hard_gate_preserves_binary_threshold_behavior(self):
        self.assertEqual(risk_gate_bias_scale(FakeRiskGate(0.05), "p", "a"), 1.0)
        self.assertEqual(risk_gate_bias_scale(FakeRiskGate(0.1), "p", "a"), 0.0)
        self.assertEqual(risk_gate_bias_scale(FakeRiskGate(0.8), "p", "a"), 0.0)

    def test_soft_gate_uses_sigmoid_distance_from_threshold(self):
        at_threshold = risk_gate_bias_scale(
            FakeRiskGate(0.1),
            "p",
            "a",
            mode="soft",
            soft_temperature=0.05,
        )
        below = risk_gate_bias_scale(
            FakeRiskGate(0.05),
            "p",
            "a",
            mode="soft",
            soft_temperature=0.05,
        )
        above = risk_gate_bias_scale(
            FakeRiskGate(0.15),
            "p",
            "a",
            mode="soft",
            soft_temperature=0.05,
        )

        self.assertAlmostEqual(at_threshold, 0.5, places=6)
        self.assertAlmostEqual(below, 1.0 / (1.0 + math.exp(-1.0)), places=6)
        self.assertAlmostEqual(above, 1.0 / (1.0 + math.exp(1.0)), places=6)

    def test_apply_bias_model_scales_only_the_residual(self):
        model = ConstantResidual(2.0)
        base = torch.tensor([[-3.0, -1.0]])

        result = apply_bias_model(model, base, residual_scale=0.25)

        torch.testing.assert_close(result, torch.tensor([[-2.5, -0.5]]))

    def test_apply_bias_model_forwards_position_to_position_aware_checkpoint(self):
        model = PositionResidual()
        base = torch.tensor([[1.0, 2.0]])

        result = apply_bias_model(
            model,
            base,
            residual_scale=0.5,
            position_id=2,
        )

        torch.testing.assert_close(result, torch.tensor([[2.0, 3.0]]))
        self.assertEqual(model.seen_position_ids, [2])

    def test_soft_gate_rejects_non_positive_temperature(self):
        with self.assertRaisesRegex(ValueError, "temperature must be positive"):
            risk_gate_bias_scale(
                FakeRiskGate(0.2),
                "p",
                "a",
                mode="soft",
                soft_temperature=0.0,
            )


class StatefulRiskGateInferenceTest(unittest.TestCase):
    def test_cli_defaults_preserve_opt_in_runtime_controls(self):
        with mock.patch.object(
            sys,
            "argv",
            ["inference_openrouter.py", "--output_json", "out.jsonl"],
        ):
            args = parse_args()

        self.assertEqual(args.risk_gate_min_scale, 0.0)
        self.assertFalse(args.risk_gate_latch_off)
        self.assertEqual(args.risk_gate_latch_patience, 2)
        self.assertIsNone(args.risk_gate_latch_threshold)
        self.assertFalse(args.risk_gate_handoff_on_latch)
        self.assertFalse(args.risk_gate_trace)
        self.assertEqual(args.mc_independent_views, 1)
        self.assertIsNone(args.reasoning_fallback_temperature)
        self.assertIsNone(args.reasoning_fallback_mode)
        self.assertEqual(args.max_reasoning_fallback_retries, 0)

    def test_cli_accepts_explicit_stateful_gate_controls(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "inference_openrouter.py",
                "--output_json",
                "out.jsonl",
                "--risk_gate_min_scale",
                "0.02",
                "--risk_gate_latch_off",
                "--risk_gate_latch_patience",
                "3",
                "--risk_gate_latch_threshold",
                "0.9",
                "--risk_gate_handoff_on_latch",
                "--risk_gate_trace",
            ],
        ):
            args = parse_args()

        self.assertAlmostEqual(args.risk_gate_min_scale, 0.02)
        self.assertTrue(args.risk_gate_latch_off)
        self.assertEqual(args.risk_gate_latch_patience, 3)
        self.assertAlmostEqual(args.risk_gate_latch_threshold, 0.9)
        self.assertTrue(args.risk_gate_handoff_on_latch)
        self.assertTrue(args.risk_gate_trace)

    def test_stateful_gate_configuration_validation_fails_closed(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "inference_openrouter.py",
                "--output_json",
                "out.jsonl",
                "--biasnet_ckpt",
                "biasnet",
                "--risk_gate_checkpoint",
                "risk-gate",
                "--risk_gate_latch_off",
                "--risk_gate_handoff_on_latch",
                "--risk_gate_trace",
            ],
        ):
            valid = parse_args()
        validate_risk_gate_runtime_args(valid)

        invalid_cases = [
            ({"risk_gate_min_scale": float("nan")}, "min_scale"),
            ({"risk_gate_latch_patience": 0}, "patience"),
            (
                {
                    "risk_gate_latch_off": False,
                    "risk_gate_handoff_on_latch": True,
                },
                "requires --risk_gate_latch_off",
            ),
            ({"risk_gate_checkpoint": None}, "risk_gate_checkpoint"),
            ({"biasnet_ckpt": None}, "biasnet_ckpt"),
        ]
        for updates, message in invalid_cases:
            with self.subTest(updates=updates):
                candidate = SimpleNamespace(**vars(valid))
                for name, value in updates.items():
                    setattr(candidate, name, value)
                with self.assertRaisesRegex(ValueError, message):
                    validate_risk_gate_runtime_args(candidate)

    def test_min_scale_uses_base_token_without_mc_sampling(self):
        tokenizer = CharacterTokenizer()
        gate = SequenceRiskGate([0.2])
        audit = {}
        args = generation_args(
            max_new_tokens=1,
            risk_gate_min_scale=0.5,
        )

        with (
            mock.patch(
                "inference_openrouter.query_next_token_id",
                return_value=(
                    1,
                    {
                        "empty_length_retry_attempts": 0,
                        "request_temperature": 0.0,
                        "request_reasoning_mode": "effort_none",
                        "reasoning_fallback_used": True,
                        "reasoning_fallback_activations": 1,
                        "reasoning_fallback_result_used": True,
                    },
                ),
            ),
            mock.patch("inference_openrouter.estimate_log_probs") as estimator,
        ):
            result = generate_one(
                client=object(),
                tokenizer=tokenizer,
                prompt="prompt",
                risk_gate_prompt="raw dataset prompt",
                args=args,
                device=torch.device("cpu"),
                bias_model=ConstantResidual(2.0),
                risk_gate=gate,
                generation_audit=audit,
            )

        self.assertEqual(result, "A")
        self.assertEqual(gate.calls[0][0], ["raw dataset prompt"])
        estimator.assert_not_called()
        self.assertEqual(audit["steps"][0]["decision"], "below_min_scale_base")
        self.assertFalse(audit["steps"][0]["used_biasnet"])
        self.assertEqual(audit["summary"]["mc_steps"], 0)
        self.assertEqual(audit["summary"]["base_only_steps"], 1)
        self.assertEqual(audit["summary"]["below_min_scale_steps"], 1)
        self.assertEqual(audit["configuration"]["risk_gate_prompt_source"], "target")
        self.assertEqual(
            audit["steps"][0]["base_request_reasoning_mode"],
            "effort_none",
        )
        self.assertTrue(audit["steps"][0]["base_reasoning_fallback_used"])
        self.assertTrue(
            audit["steps"][0]["base_reasoning_fallback_result_used"]
        )

    def test_biasnet_max_tokens_forces_dense_prefix_then_base_without_gate(self):
        tokenizer = CharacterTokenizer()
        audit = {}
        args = generation_args(
            max_new_tokens=4,
            biasnet_max_tokens=2,
        )
        log_probs = torch.tensor([[0.0, 0.0, 10.0, 0.0]])

        with (
            mock.patch(
                "inference_openrouter.query_next_token_id",
                return_value=(1, {"empty_length_retry_attempts": 0}),
            ) as base_query,
            mock.patch(
                "inference_openrouter.estimate_log_probs",
                return_value=log_probs,
            ) as estimator,
        ):
            result = generate_one(
                client=object(),
                tokenizer=tokenizer,
                prompt="prompt",
                args=args,
                device=torch.device("cpu"),
                bias_model=ConstantResidual(0.0),
                risk_gate=None,
                generation_audit=audit,
            )

        self.assertEqual(result, "BBAA")
        self.assertEqual(estimator.call_count, 2)
        self.assertEqual(base_query.call_count, 2)
        self.assertEqual(
            [step["decision"] for step in audit["steps"]],
            [
                "dense_bias_no_gate",
                "dense_bias_no_gate",
                "biasnet_budget_base",
                "biasnet_budget_base",
            ],
        )
        self.assertEqual(audit["summary"]["mc_steps"], 2)
        self.assertEqual(audit["summary"]["base_only_steps"], 2)
        self.assertEqual(audit["configuration"]["biasnet_max_tokens"], 2)

    def test_biasnet_max_tokens_validation_fails_closed(self):
        valid = generation_args(
            biasnet_max_tokens=2,
            risk_gate_checkpoint=None,
        )
        validate_risk_gate_runtime_args(valid)

        for value, checkpoint, message in (
            (0, "dense-checkpoint", "must be positive"),
            (2, None, "requires --biasnet_ckpt"),
        ):
            with self.subTest(value=value, checkpoint=checkpoint):
                candidate = SimpleNamespace(**vars(valid))
                candidate.biasnet_max_tokens = value
                candidate.biasnet_ckpt = checkpoint
                with self.assertRaisesRegex(ValueError, message):
                    validate_risk_gate_runtime_args(candidate)

        candidate = SimpleNamespace(**vars(valid))
        candidate.risk_gate_checkpoint = "risk-gate"
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            validate_risk_gate_runtime_args(candidate)

    def test_speculative_draft_verifies_all_tokens_and_rolls_back_at_mc_token(self):
        tokenizer = CharacterTokenizer()
        gate = BatchedSequenceRiskGate(
            [[0.9], [0.9], [0.9, 0.9, 0.0, 0.9], [0.9]]
        )
        audit = {}
        args = generation_args(
            max_new_tokens=6,
            risk_gate_min_scale=0.5,
            risk_gate_speculative_draft=True,
            risk_gate_speculative_min_base_streak=2,
            risk_gate_speculative_draft_tokens=80,
        )
        log_probs = torch.tensor([[0.0, 0.0, 0.0, 10.0]])

        with (
            mock.patch(
                "inference_openrouter.query_next_token_id",
                return_value=(1, {"empty_length_retry_attempts": 0}),
            ) as base_query,
            mock.patch(
                "inference_openrouter.generate_speculative_draft",
                return_value=(
                    [2, 2, 2, 2],
                    {
                        "requested_tokens": 4,
                        "finish_reason": "length",
                        "native_finish_reason": "length",
                    },
                ),
            ) as drafter,
            mock.patch(
                "inference_openrouter.estimate_log_probs",
                return_value=log_probs,
            ) as estimator,
        ):
            result = generate_one(
                client=object(),
                tokenizer=tokenizer,
                prompt="prompt",
                args=args,
                device=torch.device("cpu"),
                bias_model=ConstantResidual(0.0),
                risk_gate=gate,
                generation_audit=audit,
            )

        self.assertEqual(result, "AABBCA")
        self.assertEqual(base_query.call_count, 3)
        drafter.assert_called_once()
        estimator.assert_called_once()
        self.assertEqual(audit["summary"]["controlled_token_steps"], 6)
        self.assertEqual(audit["summary"]["speculative_verified_tokens"], 4)
        self.assertEqual(audit["summary"]["speculative_accepted_tokens"], 2)
        self.assertEqual(audit["summary"]["speculative_rejected_tokens"], 2)
        self.assertEqual(audit["summary"]["speculative_rollbacks"], 1)
        self.assertEqual(
            audit["steps"][4]["decision"],
            "speculative_rollback_soft_bias",
        )

    def test_opt_in_defaults_preserve_current_soft_gate_sampling(self):
        tokenizer = CharacterTokenizer()
        gate = SequenceRiskGate([0.9, 0.8])
        audit = {}
        args = generation_args(
            max_new_tokens=2,
            risk_gate_min_scale=0.0,
            risk_gate_latch_off=False,
        )
        log_probs = torch.tensor([[0.0, 10.0, 0.0, 0.0]])

        with (
            mock.patch(
                "inference_openrouter.query_next_token_id",
                return_value=(1, {"empty_length_retry_attempts": 0}),
            ),
            mock.patch(
                "inference_openrouter.estimate_log_probs",
                return_value=log_probs,
            ) as estimator,
        ):
            result = generate_one(
                client=object(),
                tokenizer=tokenizer,
                prompt="prompt",
                args=args,
                device=torch.device("cpu"),
                bias_model=ConstantResidual(0.0),
                risk_gate=gate,
                generation_audit=audit,
            )

        self.assertEqual(result, "AA")
        self.assertEqual(estimator.call_count, 2)
        self.assertEqual(
            [step["decision"] for step in audit["steps"]],
            ["soft_bias", "soft_bias"],
        )
        self.assertEqual(audit["summary"]["mc_steps"], 2)
        self.assertIsNone(audit["summary"]["latch_step"])
        self.assertFalse(audit["summary"]["final_latched_off"])
        self.assertEqual(
            audit["configuration"]["biasnet_checkpoint"],
            "dense-checkpoint",
        )
        self.assertEqual(
            audit["configuration"]["risk_gate_checkpoint"],
            "risk-checkpoint",
        )

    def test_warmup_does_not_score_or_advance_latch(self):
        tokenizer = CharacterTokenizer()
        gate = SequenceRiskGate([0.9, 0.8])
        audit = {}
        args = generation_args(
            max_new_tokens=3,
            risk_gate_warmup_tokens=1,
            risk_gate_latch_off=True,
            risk_gate_latch_threshold=0.5,
            risk_gate_latch_patience=2,
        )
        log_probs = torch.tensor([[0.0, 10.0, 0.0, 0.0]])

        with (
            mock.patch(
                "inference_openrouter.query_next_token_id",
                side_effect=[
                    (1, {"empty_length_retry_attempts": 0}),
                    (1, {"empty_length_retry_attempts": 0}),
                    (1, {"empty_length_retry_attempts": 0}),
                ],
            ),
            mock.patch(
                "inference_openrouter.estimate_log_probs",
                return_value=log_probs,
            ) as estimator,
        ):
            result = generate_one(
                client=object(),
                tokenizer=tokenizer,
                prompt="prompt",
                args=args,
                device=torch.device("cpu"),
                bias_model=ConstantResidual(0.0),
                risk_gate=gate,
                generation_audit=audit,
            )

        self.assertEqual(result, "AAA")
        self.assertEqual(estimator.call_count, 2)
        self.assertEqual(len(gate.calls), 2)
        self.assertEqual(gate.calls[0][1], ["AA"])
        self.assertEqual(gate.calls[1][1], ["AAA"])
        self.assertEqual(audit["steps"][0]["decision"], "warmup_bias")
        self.assertIsNone(audit["steps"][0]["risk_score"])
        self.assertEqual(audit["steps"][0]["consecutive_high_risk"], 0)
        self.assertEqual(audit["steps"][2]["decision"], "latch_base")
        self.assertEqual(audit["summary"]["latch_step"], 3)

    def test_low_risk_step_resets_consecutive_latch_streak(self):
        tokenizer = CharacterTokenizer()
        gate = SequenceRiskGate([0.9, 0.1, 0.8, 0.7])
        audit = {}
        args = generation_args(
            risk_gate_latch_off=True,
            risk_gate_latch_threshold=0.5,
            risk_gate_latch_patience=2,
        )
        log_probs = torch.tensor([[0.0, 10.0, 0.0, 0.0]])

        with (
            mock.patch(
                "inference_openrouter.query_next_token_id",
                return_value=(1, {"empty_length_retry_attempts": 0}),
            ),
            mock.patch(
                "inference_openrouter.estimate_log_probs",
                return_value=log_probs,
            ) as estimator,
        ):
            generate_one(
                client=object(),
                tokenizer=tokenizer,
                prompt="prompt",
                args=args,
                device=torch.device("cpu"),
                bias_model=ConstantResidual(0.0),
                risk_gate=gate,
                generation_audit=audit,
            )

        self.assertEqual(
            [step["consecutive_high_risk"] for step in audit["steps"]],
            [1, 0, 1, 2],
        )
        self.assertEqual(audit["summary"]["latch_step"], 4)
        self.assertEqual(estimator.call_count, 3)

    def test_latch_handoff_is_one_bounded_prefix_preserving_request(self):
        tokenizer = CharacterTokenizer()
        gate = SequenceRiskGate([0.9, 0.8])
        audit = {}
        args = generation_args(
            risk_gate_latch_off=True,
            risk_gate_latch_threshold=0.5,
            risk_gate_latch_patience=2,
            risk_gate_handoff_on_latch=True,
        )
        log_probs = torch.tensor([[0.0, 10.0, 0.0, 0.0]])
        client = SimpleNamespace(
            generate=mock.Mock(
                return_value=openrouter_response(
                    "AAxy",
                    request_max_tokens=2,
                )
            )
        )

        with (
            mock.patch(
                "inference_openrouter.query_next_token_id",
                return_value=(1, {"empty_length_retry_attempts": 0}),
            ) as base_query,
            mock.patch(
                "inference_openrouter.estimate_log_probs",
                return_value=log_probs,
            ) as estimator,
        ):
            result = generate_one(
                client=client,
                tokenizer=tokenizer,
                prompt="prompt",
                args=args,
                device=torch.device("cpu"),
                bias_model=ConstantResidual(0.0),
                risk_gate=gate,
                generation_audit=audit,
            )

        self.assertEqual(result, "AAxy")
        self.assertEqual(base_query.call_count, 2)
        self.assertEqual(estimator.call_count, 1)
        client.generate.assert_called_once_with(
            messages=[
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "AA"},
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=2,
        )
        self.assertEqual(audit["summary"]["controlled_token_steps"], 2)
        self.assertEqual(audit["summary"]["latch_step"], 2)
        self.assertTrue(audit["summary"]["handoff_occurred"])
        self.assertEqual(audit["handoff"]["trigger_step"], 2)
        self.assertEqual(audit["handoff"]["continuation_local_tokens"], 2)
        self.assertEqual(audit["steps"][1]["decision"], "latch_base")


class GateHandoffValidationTest(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CharacterTokenizer()

    def test_exact_prefill_echo_is_removed_and_audited(self):
        continuation, audit = validate_handoff_response(
            openrouter_response(
                "AAxy",
                request_max_tokens=2,
                request_reasoning_mode="effort_none",
            ),
            self.tokenizer,
            prefix_text="AA",
            remaining_tokens=2,
        )

        self.assertEqual(continuation, "xy")
        self.assertEqual(audit["continuation_local_tokens"], 2)
        self.assertEqual(audit["remaining_tokens_requested"], 2)
        self.assertFalse(audit["reasoning_detected"])
        self.assertEqual(audit["request_reasoning_mode"], "effort_none")

    def test_claude_45_handoff_strips_trailing_whitespace_replay(self):
        continuation, audit = validate_handoff_response(
            openrouter_response(
                "\n    return",
                model="anthropic/claude-haiku-4.5",
                request_max_tokens=8,
            ),
            self.tokenizer,
            prefix_text="def square(x):\n    ",
            remaining_tokens=8,
        )

        self.assertEqual(continuation, "return")
        self.assertEqual(audit["claude_45_whitespace_overlap_chars"], 5)
        self.assertTrue(audit["claude_45_whitespace_overlap_applied"])

    def test_claude_45_handoff_uses_requested_model_when_response_omits_it(self):
        continuation, audit = validate_handoff_response(
            openrouter_response("\n    return", request_max_tokens=8),
            self.tokenizer,
            prefix_text="def square(x):\n    ",
            remaining_tokens=8,
            requested_model="anthropic/claude-haiku-4.5",
        )

        self.assertEqual(continuation, "return")
        self.assertEqual(audit["claude_45_whitespace_overlap_chars"], 5)

    def test_reasoning_in_handoff_can_fail_closed(self):
        item = openrouter_response(
            "xy",
            reasoning="hidden",
            reasoning_details=[{"type": "reasoning.text"}],
            reasoning_tokens=1,
        )
        continuation, audit = validate_handoff_response(
            item,
            self.tokenizer,
            prefix_text="AA",
            remaining_tokens=2,
        )
        self.assertEqual(continuation, "xy")
        self.assertTrue(audit["reasoning_detected"])

        with self.assertRaisesRegex(
            FatalOpenRouterResponseError,
            "visible multi-token handoff",
        ):
            validate_handoff_response(
                item,
                self.tokenizer,
                prefix_text="AA",
                remaining_tokens=2,
                reject_reasoning_tokens=True,
            )

    def test_empty_stop_is_valid_but_empty_length_is_fatal(self):
        continuation, audit = validate_handoff_response(
            openrouter_response(""),
            self.tokenizer,
            prefix_text="AA",
            remaining_tokens=2,
        )
        self.assertEqual(continuation, "")
        self.assertEqual(audit["continuation_local_tokens"], 0)

        with self.assertRaisesRegex(RuntimeError, "empty length-truncated"):
            validate_handoff_response(
                openrouter_response(
                    "",
                    finish_reason="length",
                    native_finish_reason="length",
                ),
                self.tokenizer,
                prefix_text="AA",
                remaining_tokens=2,
            )

        with self.assertRaisesRegex(RuntimeError, "without a stop finish reason"):
            validate_handoff_response(
                openrouter_response(
                    "",
                    finish_reason=None,
                    native_finish_reason=None,
                ),
                self.tokenizer,
                prefix_text="AA",
                remaining_tokens=2,
            )

    def test_budget_validation_uses_the_prefix_token_boundary(self):
        class BoundaryMergingTokenizer:
            @staticmethod
            def encode(text, add_special_tokens=False):
                if text == "Axyz":
                    return ["Ax", "y", "z"]
                return list(text)

        continuation, audit = validate_handoff_response(
            openrouter_response("Axyz"),
            BoundaryMergingTokenizer(),
            prefix_text="A",
            remaining_tokens=2,
        )

        self.assertEqual(continuation, "xyz")
        self.assertEqual(audit["continuation_local_tokens"], 2)
        self.assertEqual(audit["continuation_isolated_local_tokens"], 3)

    def test_choice_error_and_over_budget_continuation_fail_closed(self):
        with self.assertRaises(FatalOpenRouterResponseError):
            validate_handoff_response(
                openrouter_response(
                    "",
                    finish_reason="error",
                    choice_error={"message": "provider rejected request"},
                ),
                self.tokenizer,
                prefix_text="AA",
                remaining_tokens=2,
            )

        with self.assertRaisesRegex(RuntimeError, "exceeded.*token budget"):
            validate_handoff_response(
                openrouter_response("AAxyz"),
                self.tokenizer,
                prefix_text="AA",
                remaining_tokens=2,
            )

    def test_speculative_draft_truncates_proxy_tokenizer_overflow(self):
        class OrdTokenizer(CharacterTokenizer):
            def encode(self, text, add_special_tokens=False):
                return [ord(character) for character in text]

        client = SimpleNamespace(
            args=SimpleNamespace(
                qwen_hard_no_think_prefill=False,
                reject_reasoning_tokens=False,
            ),
            generate=mock.Mock(return_value=openrouter_response("AAxyz")),
        )

        token_ids, audit = generate_speculative_draft(
            client,
            OrdTokenizer(),
            prompt="prompt",
            prefix_text="AA",
            remaining_tokens=2,
            draft_tokens=2,
            temperature=0.0,
            top_p=1.0,
        )

        self.assertEqual(token_ids, [ord("x"), ord("y")])
        self.assertEqual(audit["continuation_local_token_overflow"], 1)
        self.assertEqual(audit["draft_isolated_local_token_count"], 3)
        self.assertEqual(audit["draft_local_token_count"], 2)
        self.assertEqual(audit["draft_local_tokens_truncated"], 1)


class AdaptiveEmptyLengthInferenceTest(unittest.TestCase):
    def test_log_count_checkpoint_configuration_is_fail_closed(self):
        model = SimpleNamespace(
            config=SimpleNamespace(
                mc_input_representation="log_count",
                mc_log_count_alpha=1.0,
                mc_samples_per_token=50,
            )
        )
        args = SimpleNamespace(
            mc_input_representation=None,
            mc_log_count_alpha=None,
            mc_samples_per_token=50,
            sample_completion_policy="exact",
            temperature=0.0,
        )
        resolve_mc_input_configuration(args, [model])
        self.assertEqual(args.mc_input_representation, "log_count")
        self.assertEqual(args.mc_log_count_alpha, 1.0)

        bad = SimpleNamespace(**vars(args))
        bad.mc_samples_per_token = 51
        with self.assertRaisesRegex(ValueError, "sample count"):
            resolve_mc_input_configuration(bad, [model])

        bad = SimpleNamespace(**vars(args))
        bad.temperature = 0.7
        with self.assertRaisesRegex(ValueError, "greedy"):
            resolve_mc_input_configuration(bad, [model])

    def test_exact_generation_fails_closed_on_deterministic_exhaustion(self):
        class Tokenizer:
            eos_token_id = 2

            @staticmethod
            def decode(_token_ids, skip_special_tokens=False):
                return ""

        args = SimpleNamespace(
            max_new_tokens=1,
            api_max_tokens=1,
            sample_completion_policy="exact",
            stop_on_mc_failure=False,
        )
        retry_info = {
            "empty_length_retry_attempts": 4,
            "finish_reason": "length",
            "native_finish_reason": "length",
        }
        with mock.patch(
            "inference_openrouter.query_next_token_id",
            return_value=(None, retry_info),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "generation step 1 after 4 adaptive empty-length retries",
            ):
                generate_one(
                    client=object(),
                    tokenizer=Tokenizer(),
                    prompt="prompt",
                    args=args,
                    device=torch.device("cpu"),
                    bias_model=None,
                    risk_gate=None,
                )

    def test_cli_defaults_disabled_and_accepts_explicit_retry_policy(self):
        with mock.patch.object(
            sys,
            "argv",
            ["inference_openrouter.py", "--output_json", "out.jsonl"],
        ):
            args = parse_args()
        self.assertIsNone(args.empty_length_retry_max_tokens)
        self.assertEqual(args.max_empty_length_retry_rounds, 0)

        with mock.patch.object(
            sys,
            "argv",
            [
                "inference_openrouter.py",
                "--output_json",
                "out.jsonl",
                "--empty_length_retry_max_tokens",
                "8",
                "--max_empty_length_retry_rounds",
                "4",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.empty_length_retry_max_tokens, 8)
        self.assertEqual(args.max_empty_length_retry_rounds, 4)

    def test_retry_policy_validation_fails_closed(self):
        valid = SimpleNamespace(
            api_max_tokens=1,
            empty_length_retry_max_tokens=8,
            max_empty_length_retry_rounds=4,
        )
        validate_empty_length_retry_args(valid)

        cases = [
            (
                SimpleNamespace(
                    api_max_tokens=1,
                    empty_length_retry_max_tokens=None,
                    max_empty_length_retry_rounds=1,
                ),
                "is required",
            ),
            (
                SimpleNamespace(
                    api_max_tokens=1,
                    empty_length_retry_max_tokens=1,
                    max_empty_length_retry_rounds=1,
                ),
                "must exceed",
            ),
            (
                SimpleNamespace(
                    api_max_tokens=1,
                    empty_length_retry_max_tokens=8,
                    max_empty_length_retry_rounds=-1,
                ),
                "must be non-negative",
            ),
        ]
        for args, message in cases:
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, message):
                    validate_empty_length_retry_args(args)

    def test_reasoning_mode_fallback_cli_and_validation(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "inference_openrouter.py",
                "--output_json",
                "out.jsonl",
                "--reasoning_mode",
                "enabled_false",
                "--reject_reasoning_tokens",
                "--max_reasoning_retries",
                "8",
                "--reasoning_fallback_mode",
                "effort_none",
                "--max_reasoning_fallback_retries",
                "64",
            ],
        ):
            args = parse_args()

        validate_reasoning_retry_args(args)
        self.assertEqual(args.reasoning_mode, "enabled_false")
        self.assertEqual(args.reasoning_fallback_mode, "effort_none")
        self.assertIsNone(args.reasoning_fallback_temperature)
        self.assertEqual(args.max_reasoning_retries, 8)
        self.assertEqual(args.max_reasoning_fallback_retries, 64)

    def test_reasoning_mode_fallback_validation_fails_closed(self):
        base = {
            "max_reasoning_retries": 8,
            "reasoning_retry_sleep": 0.0,
            "max_reasoning_fallback_retries": 64,
            "reasoning_fallback_temperature": None,
            "reasoning_fallback_mode": "effort_none",
            "reject_reasoning_tokens": True,
            "reasoning_mode": "enabled_false",
        }
        cases = [
            (
                {"reasoning_fallback_temperature": 0.1},
                "mutually exclusive",
            ),
            (
                {"reasoning_mode": "effort_none"},
                "requires --reasoning_mode=enabled_false",
            ),
            (
                {"reject_reasoning_tokens": False},
                "requires --reject_reasoning_tokens",
            ),
            (
                {"max_reasoning_retries": 0},
                "requires positive --max_reasoning_retries",
            ),
            (
                {"max_reasoning_fallback_retries": 65},
                "cannot exceed 64",
            ),
        ]
        for overrides, message in cases:
            values = dict(base)
            values.update(overrides)
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    validate_reasoning_retry_args(SimpleNamespace(**values))

    def test_estimate_log_probs_forwards_retry_policy_to_sampler(self):
        class Tokenizer:
            def __len__(self):
                return 16

        args = SimpleNamespace(
            mc_samples_per_token=1,
            mc_sample_temperature=1.0,
            mc_top_p=1.0,
            api_max_tokens=1,
            parallel_requests=1,
            sample_choices_per_request=1,
            sample_completion_policy="exact",
            max_sample_refill_rounds=2,
            empty_length_retry_max_tokens=8,
            max_empty_length_retry_rounds=4,
            empty_response_token="stop_eos",
            delay_seconds=0.0,
            store_dtype="float32",
            mc_observed_alpha=0.1,
            mc_floor_mass=1e-4,
            mc_input_representation="floor_logprob",
            mc_log_count_alpha=None,
            qwen_hard_no_think_prefill=True,
            reject_reasoning_tokens=True,
        )
        with mock.patch(
            "inference_openrouter.sample_position_token_ids",
            return_value=([3], {"valid_samples": 1}),
        ) as sampler:
            result = estimate_log_probs(
                client=object(),
                tokenizer=Tokenizer(),
                prompt="prompt",
                prefix_text="prefix",
                args=args,
                device=torch.device("cpu"),
            )

        forwarded = sampler.call_args.kwargs["args"]
        self.assertEqual(forwarded.empty_length_retry_max_tokens, 8)
        self.assertEqual(forwarded.max_empty_length_retry_rounds, 4)
        self.assertTrue(forwarded.qwen_hard_no_think_prefill)
        self.assertTrue(forwarded.reject_reasoning_tokens)
        self.assertEqual(tuple(result.shape), (1, 16))

    def test_estimate_log_probs_can_materialize_centered_log_counts(self):
        class Tokenizer:
            def __len__(self):
                return 8

        args = SimpleNamespace(
            mc_samples_per_token=5,
            mc_sample_temperature=1.0,
            mc_top_p=1.0,
            api_max_tokens=1,
            parallel_requests=1,
            sample_choices_per_request=1,
            sample_completion_policy="exact",
            max_sample_refill_rounds=2,
            empty_length_retry_max_tokens=8,
            max_empty_length_retry_rounds=4,
            empty_response_token="stop_eos",
            delay_seconds=0.0,
            store_dtype="float32",
            mc_observed_alpha=0.1,
            mc_floor_mass=1e-4,
            mc_input_representation="log_count",
            mc_log_count_alpha=1.0,
            qwen_hard_no_think_prefill=True,
            reject_reasoning_tokens=True,
        )
        with mock.patch(
            "inference_openrouter.sample_position_token_ids",
            return_value=([2, 2, 2, 3, 4], {"valid_samples": 5}),
        ):
            result = estimate_log_probs(
                client=object(),
                tokenizer=Tokenizer(),
                prompt="prompt",
                prefix_text="prefix",
                args=args,
                device=torch.device("cpu"),
            )

        expected = torch.zeros(1, 8)
        expected[0, 2] = torch.log(torch.tensor(4.0))
        expected[0, 3] = torch.log(torch.tensor(2.0))
        expected[0, 4] = torch.log(torch.tensor(2.0))
        torch.testing.assert_close(result, expected)

    def test_estimate_log_probs_builds_dual_views_from_one_sample_batch(self):
        class Tokenizer:
            def __len__(self):
                return 8

        args = SimpleNamespace(
            mc_samples_per_token=5,
            mc_sample_temperature=1.0,
            mc_top_p=1.0,
            api_max_tokens=1,
            parallel_requests=1,
            sample_choices_per_request=1,
            sample_completion_policy="exact",
            max_sample_refill_rounds=2,
            empty_length_retry_max_tokens=8,
            max_empty_length_retry_rounds=4,
            empty_response_token="stop_eos",
            delay_seconds=0.0,
            store_dtype="float32",
            mc_observed_alpha=0.1,
            mc_floor_mass=1e-4,
            mc_input_representation="log_count",
            mc_base_score_representation="floor_logprob",
            mc_log_count_alpha=1.0,
            qwen_hard_no_think_prefill=True,
            reject_reasoning_tokens=True,
        )
        with mock.patch(
            "inference_openrouter.sample_position_token_ids",
            return_value=([2, 2, 2, 3, 4], {"valid_samples": 5}),
        ) as sampler:
            result = estimate_log_probs(
                client=object(),
                tokenizer=Tokenizer(),
                prompt="prompt",
                prefix_text="prefix",
                args=args,
                device=torch.device("cpu"),
            )

        self.assertIsInstance(result, BiasMCInputs)
        self.assertEqual(sampler.call_count, 1)
        self.assertEqual(result.features[0, 0].item(), 0.0)
        self.assertLess(result.base_scores[0, 0].item(), -5.0)
        self.assertAlmostEqual(
            result.features[0, 2].item(),
            torch.log(torch.tensor(4.0)).item(),
            places=5,
        )

    def test_estimate_log_probs_builds_independent_checkpoint_native_views(self):
        class Tokenizer:
            def __len__(self):
                return 8

        args = SimpleNamespace(
            mc_samples_per_token=5,
            mc_independent_views=2,
            mc_sample_temperature=1.0,
            mc_top_p=1.0,
            api_max_tokens=1,
            parallel_requests=1,
            sample_choices_per_request=1,
            sample_completion_policy="exact",
            max_sample_refill_rounds=2,
            empty_length_retry_max_tokens=8,
            max_empty_length_retry_rounds=4,
            empty_response_token="stop_eos",
            delay_seconds=0.0,
            store_dtype="float32",
            mc_observed_alpha=0.1,
            mc_floor_mass=1e-4,
            mc_input_representation="log_count",
            mc_base_score_representation="floor_logprob",
            mc_log_count_alpha=1.0,
            qwen_hard_no_think_prefill=True,
            reject_reasoning_tokens=True,
        )
        sample_views = [
            ([2, 2, 2, 3, 4], {"valid_samples": 5}),
            ([2, 3, 3, 3, 5], {"valid_samples": 5}),
        ]
        with mock.patch(
            "inference_openrouter.sample_position_token_ids",
            side_effect=sample_views,
        ) as sampler:
            result = estimate_log_probs(
                client=object(),
                tokenizer=Tokenizer(),
                prompt="prompt",
                prefix_text="prefix",
                args=args,
                device=torch.device("cpu"),
            )

        self.assertIsInstance(result, BiasMCInputs)
        self.assertEqual(result.independent_views, 2)
        self.assertEqual(tuple(result.features.shape), (2, 8))
        self.assertEqual(tuple(result.base_scores.shape), (2, 8))
        self.assertEqual(sampler.call_count, 2)
        self.assertAlmostEqual(result.features[0, 2].item(), math.log(4.0))
        self.assertAlmostEqual(result.features[1, 3].item(), math.log(4.0))

    def test_apply_bias_model_ensembles_per_view_probabilities(self):
        views = BiasMCInputs(
            features=torch.tensor([[2.0, 0.0], [0.0, 0.0]]),
            base_scores=torch.zeros(2, 2),
            independent_views=2,
        )

        actual = apply_bias_model(IdentityResidual(), views)
        view_probabilities = torch.stack(
            [
                torch.softmax(torch.tensor([2.0, 0.0]), dim=-1),
                torch.softmax(torch.tensor([0.0, 0.0]), dim=-1),
            ]
        )
        expected = view_probabilities.mean(dim=0, keepdim=True).log()

        torch.testing.assert_close(actual, expected)
        self.assertFalse(
            torch.allclose(
                actual,
                torch.log_softmax(torch.tensor([[1.0, 0.0]]), dim=-1),
            )
        )

    def test_mc_ensemble_audit_is_explicit_about_view_semantics(self):
        audit = mc_ensemble_audit(SimpleNamespace(
            mc_independent_views=4,
            mc_samples_per_token=50,
            disable_openrouter_response_cache=True,
        ))

        self.assertEqual(audit["schema_version"], 1)
        self.assertEqual(audit["independent_views"], 4)
        self.assertEqual(audit["samples_per_view"], 50)
        self.assertEqual(audit["total_requested_samples_per_token"], 200)
        self.assertEqual(audit["aggregation"], "mean_probability_v1")
        self.assertTrue(audit["openrouter_response_cache_disabled"])

    def test_mc_ensemble_validation_requires_exact_uncached_sampling(self):
        valid = SimpleNamespace(
            mc_independent_views=4,
            sample_completion_policy="exact",
            disable_openrouter_response_cache=True,
        )
        validate_mc_ensemble_args(valid)
        for updates, message in (
            ({"mc_independent_views": 0}, "must be positive"),
            ({"sample_completion_policy": "partial"}, "policy exact"),
            ({"disable_openrouter_response_cache": False}, "response_cache"),
        ):
            with self.subTest(updates=updates):
                candidate = SimpleNamespace(**vars(valid))
                for name, value in updates.items():
                    setattr(candidate, name, value)
                with self.assertRaisesRegex(ValueError, message):
                    validate_mc_ensemble_args(candidate)

        # The default single view remains compatible with legacy partial/cache use.
        validate_mc_ensemble_args(SimpleNamespace(
            mc_independent_views=1,
            sample_completion_policy="partial",
            disable_openrouter_response_cache=False,
        ))

    def test_apply_bias_model_uses_feature_view_and_separate_base_scores(self):
        class Model:
            num_position_buckets = 0

            def parameters(self):
                yield torch.nn.Parameter(torch.zeros(1))

            def __call__(self, features):
                return features * 2

        views = BiasMCInputs(
            features=torch.tensor([[1.0, 2.0, 3.0]]),
            base_scores=torch.tensor([[-10.0, -20.0, -30.0]]),
        )
        actual = apply_bias_model(Model(), views, residual_scale=0.5)
        expected = torch.tensor([[-9.0, -18.0, -27.0]])
        torch.testing.assert_close(actual, expected)

    def test_per_prompt_routing_metadata_reports_retry_counter_deltas(self):
        client = SimpleNamespace(
            calls=10,
            total_cost=1.0,
            provider_call_counts={"DeepInfra": 10},
            response_model_call_counts={"z-ai/glm-5": 10},
            response_cache_status_counts={"<missing>": 10},
            missing_router_metadata_calls=0,
            api_max_tokens_call_counts={1: 10},
            empty_length_retry_attempts=2,
            empty_length_retry_recoveries=2,
            empty_length_retry_exhaustions=0,
            max_routing_attempt=1,
        )
        before = snapshot_openrouter_audit(client)

        client.calls = 13
        client.total_cost = 1.25
        client.provider_call_counts["DeepInfra"] = 13
        client.response_model_call_counts["z-ai/glm-5"] = 13
        client.response_cache_status_counts["<missing>"] = 13
        client.api_max_tokens_call_counts = {1: 11, 2: 1, 8: 1}
        client.empty_length_retry_attempts = 4
        client.empty_length_retry_recoveries = 3
        client.empty_length_retry_exhaustions = 1
        client.reasoning_fallback_activations = 1
        client.reasoning_fallback_response_calls = 2
        client.reasoning_fallback_recoveries = 1
        client.reasoning_fallback_exhaustions = 0
        client.reasoning_fallback_provider_call_counts = {"DeepInfra": 2}
        client.reasoning_fallback_response_model_call_counts = {
            "z-ai/glm-5": 2
        }

        args = SimpleNamespace(
            model="z-ai/glm-5",
            provider_order=["DeepInfra"],
            provider_allow_fallbacks=False,
            router_metadata=True,
            disable_openrouter_response_cache=True,
            empty_response_token="stop_eos",
            api_max_tokens=1,
            empty_length_retry_max_tokens=8,
            max_empty_length_retry_rounds=4,
            reasoning_mode="enabled_false",
            reject_reasoning_tokens=True,
            max_reasoning_retries=8,
            reasoning_retry_sleep=1.0,
            reasoning_fallback_temperature=1.0,
            reasoning_fallback_mode=None,
            max_reasoning_fallback_retries=8,
        )
        routing = build_openrouter_routing_metadata(client, args, before)

        self.assertEqual(routing["api_calls"], 3)
        self.assertAlmostEqual(routing["api_cost"], 0.25)
        self.assertEqual(
            routing["api_max_tokens_call_counts"],
            {"1": 1, "2": 1, "8": 1},
        )
        self.assertEqual(routing["empty_length_retry_attempts"], 2)
        self.assertEqual(routing["empty_length_retry_recoveries"], 1)
        self.assertEqual(routing["empty_length_retry_exhaustions"], 1)
        self.assertEqual(routing["empty_length_retry_max_tokens"], 8)
        self.assertEqual(routing["max_empty_length_retry_rounds"], 4)
        self.assertEqual(routing["reasoning_fallback_temperature"], 1.0)
        self.assertIsNone(routing["reasoning_fallback_mode"])
        self.assertEqual(routing["max_reasoning_retries"], 8)
        self.assertEqual(routing["reasoning_retry_sleep"], 1.0)
        self.assertEqual(routing["max_reasoning_fallback_retries"], 8)
        self.assertEqual(routing["reasoning_fallback_activations"], 1)
        self.assertEqual(routing["reasoning_fallback_response_calls"], 2)
        self.assertEqual(routing["reasoning_fallback_recoveries"], 1)
        self.assertEqual(routing["reasoning_fallback_exhaustions"], 0)
        self.assertEqual(
            routing["reasoning_fallback_provider_call_counts"],
            {"DeepInfra": 2},
        )
        self.assertEqual(
            routing["reasoning_fallback_response_model_call_counts"],
            {"z-ai/glm-5": 2},
        )


if __name__ == "__main__":
    unittest.main()
