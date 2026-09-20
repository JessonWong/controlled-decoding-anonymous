import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import numpy as np

from inference_openrouter import (
    ANYTIME_FEATURE_NAMES,
    AnytimeEstimate,
    AnytimePolicy,
    estimate_anytime_scores,
    generate_one,
    load_anytime_policy,
    validate_anytime_runtime,
)
from training.anytime_proxy_mc import (
    POLICY_SPEC_HASH_FIELD,
    policy_spec_payload_sha256,
)
from training.eval_anytime_stopping import _predict_confidence, _select_stage


class TinyTokenizer:
    eos_token_id = None

    def __len__(self):
        return 5

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(str(int(token_id)) for token_id in token_ids)


class FakeProxy:
    def __init__(self):
        self.calls = 0
        self.total_latency_seconds = 0.0
        self.total_input_tokens = 0

    def next_token_logits(self, *args, **kwargs):
        self.calls += 1
        return torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0])

    def configuration(self, args):
        return {"mode": "fake"}


class ZeroResidualAnytimeModel(torch.nn.Module):
    num_position_buckets = 0
    mc_sample_count_conditioning = True

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.counts = []
        self.config = SimpleNamespace(
            anytime_schema_version=1,
            mc_sample_temperature=1.0,
            mc_top_p=1.0,
            mc_completion_policy="exact",
        )

    def forward(self, scores, position_ids=None, mc_sample_counts=None):
        self.counts.extend(int(value) for value in mc_sample_counts.cpu())
        return torch.zeros_like(scores) + self.anchor * 0


def make_target_protocol():
    return {
        "target_protocol_schema_version": 1,
        "target_api_url": "https://openrouter.ai/api/v1/chat/completions",
        "target_model": "qwen/qwen3-32b",
        "target_provider_order": ["DeepInfra"],
        "target_provider_allow_fallbacks": False,
        "target_provider_quantizations": None,
        "target_append_no_think": True,
        "target_qwen_hard_no_think_prefill": True,
        "target_api_max_tokens": 1,
        "target_empty_response_token": "stop_eos",
        "target_disable_openrouter_response_cache": True,
        "target_sample_choices_per_request": 1,
        "target_max_sample_refill_rounds": 5,
        "target_empty_length_retry_max_tokens": None,
        "target_max_empty_length_retry_rounds": 0,
        "target_reasoning_mode": "enabled_false",
        "target_reject_reasoning_tokens": True,
        "target_max_reasoning_retries": 0,
        "target_reasoning_fallback_temperature": None,
        "target_reasoning_fallback_mode": None,
        "target_max_reasoning_fallback_retries": 0,
    }


def make_policy(
    *,
    coefficient=0.0,
    intercept=0.0,
    threshold=0.5,
    autocast_enabled=True,
):
    coefficients = [0.0] * len(ANYTIME_FEATURE_NAMES)
    coefficients[0] = coefficient
    return AnytimePolicy(
        source_path="policy.json",
        source_sha256="source-sha",
        payload_sha256="payload-sha",
        checkpoint_config_sha256="config-sha",
        name="test",
        budgets=(0, 4, 8, 16, 32, 50),
        threshold=threshold,
        allowed_action_disagreement=0.1,
        position_normalizer=79,
        biasnet_parameter_dtype="float32",
        biasnet_autocast_enabled=autocast_enabled,
        biasnet_autocast_dtype="float16",
        target_protocol=make_target_protocol(),
        scaler_mean=(0.0,) * len(ANYTIME_FEATURE_NAMES),
        scaler_scale=(1.0,) * len(ANYTIME_FEATURE_NAMES),
        coefficient=tuple(coefficients),
        intercept=intercept,
    )


def make_args(**overrides):
    values = {
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "qwen/qwen3-32b",
        "provider_order": ["DeepInfra"],
        "provider_allow_fallbacks": False,
        "provider_quantizations": None,
        "append_no_think": True,
        "mc_sample_temperature": 1.0,
        "mc_top_p": 1.0,
        "api_max_tokens": 1,
        "parallel_requests": 1,
        "sample_choices_per_request": 1,
        "sample_completion_policy": "exact",
        "max_sample_refill_rounds": 5,
        "empty_length_retry_max_tokens": None,
        "max_empty_length_retry_rounds": 0,
        "empty_response_token": "stop_eos",
        "delay_seconds": 0.0,
        "qwen_hard_no_think_prefill": True,
        "reject_reasoning_tokens": True,
        "reasoning_mode": "enabled_false",
        "max_reasoning_retries": 0,
        "reasoning_fallback_temperature": None,
        "reasoning_fallback_mode": None,
        "max_reasoning_fallback_retries": 0,
        "disable_openrouter_response_cache": True,
        "biasnet_dtype": "float32",
        "initial_prefix": "",
        "proxy_temperature": 2.5,
        "proxy_prior_strength": 8.0,
        "require_faithful_support": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class AnytimeInferenceTest(unittest.TestCase):
    def generation_args(self, **overrides):
        return make_args(
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
            initial_prefix="",
            bootstrap_biasnet_tokens=0,
            biasnet_max_tokens=None,
            mask_non_eos_special_tokens=False,
            risk_gate_min_scale=0.0,
            risk_gate_latch_off=False,
            risk_gate_latch_patience=2,
            risk_gate_latch_threshold=None,
            risk_gate_handoff_on_latch=False,
            risk_gate_speculative_draft=False,
            risk_gate_speculative_min_base_streak=2,
            risk_gate_speculative_draft_tokens=8,
            risk_gate_warmup_tokens=0,
            risk_gate_mode="hard",
            risk_gate_soft_temperature=0.05,
            stop_on_mc_failure=False,
            progress_steps=False,
            mc_independent_views=1,
            store_dtype="float32",
            mc_input_representation="floor_logprob",
            mc_base_score_representation="floor_logprob",
            mc_observed_alpha=0.1,
            mc_floor_mass=1e-4,
            mc_samples_per_token=50,
            **overrides,
        )

    @mock.patch("inference_openrouter.estimate_anytime_scores")
    @mock.patch("inference_openrouter.query_next_token_id")
    def test_generate_k0_keeps_mc_steps_zero_and_splits_audit(
        self, query, estimate
    ):
        query.return_value = (1, {})
        estimate.return_value = AnytimeEstimate(
            scores=torch.tensor([[0.0, 0.0, 3.0, 0.0, 0.0]]),
            samples_used=0,
            requested_choices=0,
            stage_trace=(
                {"budget": 0, "stop_reason": "confidence", "stopped": True},
            ),
        )
        audit = {}
        client = SimpleNamespace(
            calls=1,
            total_cost=0.0,
            mc_requested_samples=0,
        )

        completion = generate_one(
            client=client,
            tokenizer=TinyTokenizer(),
            prompt="p",
            args=self.generation_args(),
            device=torch.device("cpu"),
            bias_model=ZeroResidualAnytimeModel(),
            risk_gate=None,
            generation_audit=audit,
            proxy=FakeProxy(),
            anytime_policy=make_policy(intercept=10.0),
        )

        self.assertEqual(completion, "2")
        self.assertEqual(audit["summary"]["mc_steps"], 0)
        self.assertEqual(audit["summary"]["mc_valid_samples"], 0)
        self.assertEqual(audit["summary"]["mc_requested_choices"], 0)
        self.assertEqual(audit["summary"]["zero_sample_biasnet_steps"], 1)
        self.assertEqual(audit["summary"]["anytime_budget_histogram"], {"0": 1})
        self.assertIsNone(
            audit["configuration"]["mc_ensemble"][
                "fixed_requested_samples_per_token"
            ]
        )

    @mock.patch("inference_openrouter.estimate_anytime_scores")
    @mock.patch("inference_openrouter.query_next_token_id")
    def test_generate_distinguishes_valid_k_from_refill_choices(
        self, query, estimate
    ):
        query.return_value = (1, {})
        client = SimpleNamespace(
            calls=1,
            total_cost=0.0,
            mc_requested_samples=0,
        )

        def sampled(**kwargs):
            del kwargs
            client.mc_requested_samples += 7
            return AnytimeEstimate(
                scores=torch.tensor([[0.0, 3.0, 0.0, 0.0, 0.0]]),
                samples_used=4,
                requested_choices=7,
                stage_trace=(
                    {"budget": 4, "stop_reason": "confidence", "stopped": True},
                ),
            )

        estimate.side_effect = sampled
        audit = {}
        generate_one(
            client=client,
            tokenizer=TinyTokenizer(),
            prompt="p",
            args=self.generation_args(),
            device=torch.device("cpu"),
            bias_model=ZeroResidualAnytimeModel(),
            risk_gate=None,
            generation_audit=audit,
            proxy=FakeProxy(),
            anytime_policy=make_policy(),
        )

        self.assertEqual(audit["summary"]["mc_steps"], 1)
        self.assertEqual(audit["summary"]["mc_valid_samples"], 4)
        self.assertEqual(audit["summary"]["mc_requested_choices"], 7)
        self.assertEqual(audit["summary"]["mc_requested_samples"], 7)
        self.assertEqual(
            audit["summary"]["mean_valid_mc_samples_per_anytime_step"], 4
        )
        self.assertEqual(
            audit["summary"]["mean_requested_mc_choices_per_anytime_step"], 7
        )

    def test_policy_loader_verifies_payload_and_checkpoint_config_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            weights_path = checkpoint / "pytorch_model.bin"
            config_path = checkpoint / "config.json"
            weights_path.write_bytes(b"weights")
            config_path.write_text("{}\n", encoding="utf-8")

            def sha256(path):
                return hashlib.sha256(path.read_bytes()).hexdigest()

            policy_payload = {
                "schema_version": 2,
                "feature_names": list(ANYTIME_FEATURE_NAMES),
                "checkpoint_weights_sha256": sha256(weights_path),
                "checkpoint_config_sha256": sha256(config_path),
                "cache_manifest_configuration_fingerprint": "fingerprint",
                "estimator_metadata": make_target_protocol(),
                "biasnet_parameter_dtype": "float32",
                "biasnet_autocast_enabled": True,
                "biasnet_autocast_dtype": "float16",
                "biasnet_runtime_dtype": "float32",
                "budgets": [0, 4, 8, 16, 32, 50],
                "position_normalizer": 79,
                "policies": {
                    "delta_0.05": {
                        "status": "selected_on_legacy_validation",
                        "threshold": 0.75,
                        "allowed_action_disagreement": 0.05,
                    }
                },
                "linear_stopper": {
                    "schema_version": 2,
                    "scoring_protocol": "python_float64_scalar_v1",
                    "classes": [0, 1],
                    "scaler_mean": [0.0] * len(ANYTIME_FEATURE_NAMES),
                    "scaler_scale": [1.0] * len(ANYTIME_FEATURE_NAMES),
                    "coefficient": [0.0] * len(ANYTIME_FEATURE_NAMES),
                    "intercept": 0.0,
                },
            }
            policy_payload[POLICY_SPEC_HASH_FIELD] = policy_spec_payload_sha256(
                policy_payload
            )
            policy_path = root / "policy.json"
            policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")
            model = ZeroResidualAnytimeModel()
            model.config.mc_sample_budgets = [0, 4, 8, 16, 32, 50]
            model.config.mc_max_samples_per_token = 50
            model.config.mc_sample_count_conditioning = True
            model.config.proxy_mc_manifest_configuration_fingerprint = "fingerprint"
            args = SimpleNamespace(
                anytime_policy_spec=str(policy_path),
                anytime_policy_name="delta_0.05",
                biasnet_ckpt=str(checkpoint),
                mc_samples_per_token=50,
            )

            loaded = load_anytime_policy(args, model, None)
            self.assertEqual(loaded.payload_sha256, policy_payload[POLICY_SPEC_HASH_FIELD])
            self.assertEqual(loaded.biasnet_parameter_dtype, "float32")
            self.assertTrue(loaded.biasnet_autocast_enabled)
            self.assertEqual(loaded.biasnet_autocast_dtype, "float16")

            config_path.write_text('{"mutated": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different BiasNet config"):
                load_anytime_policy(args, model, None)
            config_path.write_text("{}\n", encoding="utf-8")

            policy_payload["biasnet_autocast_dtype"] = "bfloat16"
            policy_payload[POLICY_SPEC_HASH_FIELD] = policy_spec_payload_sha256(
                policy_payload
            )
            policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "autocast protocol"):
                load_anytime_policy(args, model, None)
            policy_payload["biasnet_autocast_dtype"] = "float16"

            policy_payload["position_normalizer"] = 78
            policy_path.write_text(json.dumps(policy_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                load_anytime_policy(args, model, None)

    def test_portable_calibration_and_online_stop_stages_are_identical(self):
        policy = make_policy(coefficient=3.0, intercept=-0.4, threshold=0.55)
        features = np.linspace(
            -1.0, 1.0, num=2 * 4 * len(ANYTIME_FEATURE_NAMES), dtype=np.float32
        ).reshape(1, 2, 4, len(ANYTIME_FEATURE_NAMES))
        linear = {
            "schema_version": 2,
            "scoring_protocol": "python_float64_scalar_v1",
            "classes": [0, 1],
            "scaler_mean": list(policy.scaler_mean),
            "scaler_scale": list(policy.scaler_scale),
            "coefficient": list(policy.coefficient),
            "intercept": policy.intercept,
        }
        calibration_confidence = _predict_confidence(
            linear, {"features": features}
        )
        online_confidence = np.asarray(
            [
                [policy.confidence(features[0, row, stage]) for stage in range(3)]
                for row in range(2)
            ]
        )[None, ...]

        np.testing.assert_array_equal(calibration_confidence, online_confidence)
        np.testing.assert_array_equal(
            _select_stage(calibration_confidence, policy.threshold),
            _select_stage(online_confidence, policy.threshold),
        )

    @mock.patch("inference_openrouter.sample_position_token_ids")
    def test_incremental_sampling_stops_at_k8_and_reuses_proxy(self, sample):
        requested = []

        def draw(**kwargs):
            count = kwargs["args"].samples_per_token
            requested.append(count)
            return [1] * count, {
                "valid_samples": count,
                "requested_samples": count,
                "sampled_completion_text_counts": {"1": count},
            }

        sample.side_effect = draw
        model = ZeroResidualAnytimeModel()
        proxy = FakeProxy()
        result = estimate_anytime_scores(
            client=object(),
            tokenizer=TinyTokenizer(),
            prompt="p",
            prefix_text="",
            args=make_args(),
            device=torch.device("cpu"),
            proxy=proxy,
            bias_model=model,
            base_token_id=0,
            position_id=0,
            policy=make_policy(
                coefficient=20.0,
                intercept=-2.0,
                threshold=0.6,
                autocast_enabled=False,
            ),
        )

        self.assertEqual(requested, [4, 4])
        self.assertEqual(proxy.calls, 1)
        self.assertEqual(result.samples_used, 8)
        self.assertEqual(result.requested_choices, 8)
        self.assertEqual(model.counts, [0, 4, 8])
        self.assertEqual(
            [stage["budget"] for stage in result.stage_trace], [0, 4, 8]
        )

    @mock.patch("inference_openrouter.sample_position_token_ids")
    def test_k0_stop_makes_no_mc_request(self, sample):
        result = estimate_anytime_scores(
            client=object(),
            tokenizer=TinyTokenizer(),
            prompt="p",
            prefix_text="",
            args=make_args(),
            device=torch.device("cpu"),
            proxy=FakeProxy(),
            bias_model=ZeroResidualAnytimeModel(),
            base_token_id=0,
            position_id=2,
            policy=make_policy(
                intercept=10.0,
                threshold=0.5,
                autocast_enabled=False,
            ),
        )

        sample.assert_not_called()
        self.assertEqual(result.samples_used, 0)
        self.assertEqual(result.requested_choices, 0)
        self.assertEqual(result.stage_trace[-1]["stop_reason"], "confidence")

    @mock.patch("inference_openrouter.sample_position_token_ids")
    def test_full_path_uses_deltas_not_cumulative_budgets(self, sample):
        requested = []

        def draw(**kwargs):
            count = kwargs["args"].samples_per_token
            requested.append(count)
            return [1] * count, {
                "valid_samples": count,
                "requested_samples": count,
            }

        sample.side_effect = draw
        result = estimate_anytime_scores(
            client=object(),
            tokenizer=TinyTokenizer(),
            prompt="p",
            prefix_text="",
            args=make_args(),
            device=torch.device("cpu"),
            proxy=FakeProxy(),
            bias_model=ZeroResidualAnytimeModel(),
            base_token_id=0,
            position_id=2,
            policy=make_policy(
                intercept=-10.0,
                threshold=1.0,
                autocast_enabled=False,
            ),
        )

        self.assertEqual(requested, [4, 4, 8, 16, 18])
        self.assertEqual(result.samples_used, 50)
        self.assertEqual(result.requested_choices, 50)
        self.assertEqual(result.stage_trace[-1]["confidence"], None)
        self.assertEqual(result.stage_trace[-1]["stop_reason"], "full_budget")

    def test_runtime_validation_rejects_unfrozen_variants(self):
        model = ZeroResidualAnytimeModel()
        args = make_args(
            risk_gate_checkpoint=None,
            biasnet_max_tokens=None,
            temperature=0.0,
            sample_completion_policy="exact",
            mc_independent_views=1,
            disable_openrouter_response_cache=True,
            sample_choices_per_request=1,
            restrict_to_observed_support=False,
            require_faithful_support=False,
            mask_non_eos_special_tokens=False,
            proxy_model_name_or_path="proxy",
            max_new_tokens=80,
        )

        validate_anytime_runtime(
            args, make_policy(), [model], runtime_device=torch.device("cuda")
        )
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            validate_anytime_runtime(
                args, make_policy(), [model], runtime_device=torch.device("cpu")
            )
        args.risk_gate_checkpoint = "risk"
        with self.assertRaisesRegex(ValueError, "replaces"):
            validate_anytime_runtime(
                args, make_policy(), [model], runtime_device=torch.device("cuda")
            )

    def test_runtime_validation_rejects_target_protocol_and_initial_prefix(self):
        model = ZeroResidualAnytimeModel()
        args = make_args(
            risk_gate_checkpoint=None,
            biasnet_max_tokens=None,
            temperature=0.0,
            mc_independent_views=1,
            restrict_to_observed_support=False,
            require_faithful_support=False,
            mask_non_eos_special_tokens=False,
            proxy_model_name_or_path="proxy",
            max_new_tokens=80,
        )

        args.api_max_tokens = 2
        with self.assertRaisesRegex(ValueError, "target sampling protocol"):
            validate_anytime_runtime(
                args, make_policy(), [model], runtime_device=torch.device("cuda")
            )
        args.api_max_tokens = 1
        args.initial_prefix = "prefilled"
        with self.assertRaisesRegex(ValueError, "initial_prefix"):
            validate_anytime_runtime(
                args, make_policy(), [model], runtime_device=torch.device("cuda")
            )


if __name__ == "__main__":
    unittest.main()
