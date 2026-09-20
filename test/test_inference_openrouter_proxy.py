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
    LocalProxyRuntime,
    PROXY_CHAT_TEMPLATE_PROTOCOL,
    PROXY_FUSION_MODE,
    PROXY_VOCAB_TAIL_POLICY,
    estimate_log_probs,
    generate_one,
    resolve_proxy_fusion_configuration,
    tokenizer_mapping_sha256,
    validate_shared_tokenizer_mapping,
)
from mc_reconstruction import FLOOR_LOGPROB, LOG_COUNT


class TinyTokenizer:
    eos_token_id = None
    all_special_ids = []

    def __init__(self, vocabulary=None):
        self.vocabulary = vocabulary or {"a": 0, "b": 1, "c": 2}
        self.template_calls = []

    def __len__(self):
        return len(self.vocabulary)

    def get_vocab(self):
        return dict(self.vocabulary)

    def encode(self, text, add_special_tokens=False):
        return []

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.convert_ids_to_tokens(token_id) for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id):
        return next(token for token, value in self.vocabulary.items() if value == token_id)

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return torch.tensor([[4, 5, 6]])


class FakeProxy:
    def __init__(self, logits):
        self.logits = logits
        self.calls = 0

    def next_token_logits(self, prompt, prefix_text, **kwargs):
        self.calls += 1
        return self.logits


class NeverCalledProxy:
    calls = 0
    total_latency_seconds = 0.0
    total_input_tokens = 0

    def next_token_logits(self, *args, **kwargs):
        raise AssertionError("gate bypass must not call the local proxy")

    def configuration(self, args):
        return {}


class HighRiskGate:
    threshold = 0.1

    def score_prefixes(self, prompts, answer_prefixes):
        return torch.tensor([0.9])


class AnchorModel(torch.nn.Module):
    num_position_buckets = 0

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, values):
        return torch.zeros_like(values)


def sampling_args(**overrides):
    values = {
        "mc_samples_per_token": 2,
        "mc_sample_temperature": 1.0,
        "mc_top_p": 1.0,
        "api_max_tokens": 2,
        "parallel_requests": 1,
        "sample_choices_per_request": 1,
        "sample_completion_policy": "exact",
        "max_sample_refill_rounds": 0,
        "empty_length_retry_max_tokens": None,
        "max_empty_length_retry_rounds": 0,
        "empty_response_token": "skip",
        "delay_seconds": 0.0,
        "qwen_hard_no_think_prefill": True,
        "reject_reasoning_tokens": True,
        "mc_independent_views": 1,
        "store_dtype": "float32",
        "mc_input_representation": LOG_COUNT,
        "mc_base_score_representation": FLOOR_LOGPROB,
        "mc_log_count_alpha": 1.0,
        "mc_observed_alpha": 0.1,
        "mc_floor_mass": 1e-4,
        "proxy_temperature": 1.0,
        "proxy_prior_strength": 2.0,
        "proxy_dtype": "float16",
        "proxy_quantization": "none",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ProxyFusionInferenceTest(unittest.TestCase):
    def test_tokenizer_hash_and_mapping_are_exact(self):
        shared = TinyTokenizer({"z": 2, "a": 0, "b": 1})
        identical = TinyTokenizer({"b": 1, "z": 2, "a": 0})
        remapped = TinyTokenizer({"a": 1, "b": 0, "z": 2})

        fingerprint = validate_shared_tokenizer_mapping(shared, identical)

        self.assertEqual(fingerprint, tokenizer_mapping_sha256(shared))
        self.assertEqual(fingerprint, tokenizer_mapping_sha256(identical))
        with self.assertRaisesRegex(ValueError, "does not exactly match"):
            validate_shared_tokenizer_mapping(shared, remapped)

    @mock.patch("inference_openrouter.sample_position_token_ids")
    def test_estimator_uses_native_log_counts_and_fused_dense_base(self, sample):
        sample.return_value = ([0, 0], {"valid": 2})
        tokenizer = TinyTokenizer()
        proxy = FakeProxy(torch.log(torch.tensor([0.2, 0.3, 0.5])))

        result = estimate_log_probs(
            client=object(),
            tokenizer=tokenizer,
            prompt="question/no_think",
            prefix_text="answer",
            args=sampling_args(),
            device=torch.device("cpu"),
            proxy=proxy,
        )

        self.assertIsInstance(result, BiasMCInputs)
        torch.testing.assert_close(
            result.features,
            torch.tensor([[math.log(3.0), 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            result.base_scores.exp(),
            torch.tensor([[0.6, 0.15, 0.25]]),
        )
        self.assertEqual(proxy.calls, 1)

    def test_high_risk_bypass_does_not_call_proxy(self):
        args = sampling_args(
            max_new_tokens=1,
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
            temperature=0.0,
            top_p=1.0,
            stop_on_mc_failure=False,
            progress_steps=False,
        )
        with mock.patch(
            "inference_openrouter.query_next_token_id",
            return_value=(1, {}),
        ), mock.patch("inference_openrouter.estimate_log_probs") as estimate:
            audit = {}
            generate_one(
                client=SimpleNamespace(calls=1, total_cost=0.0),
                tokenizer=TinyTokenizer(),
                prompt="p",
                args=args,
                device=torch.device("cpu"),
                bias_model=AnchorModel(),
                risk_gate=HighRiskGate(),
                proxy=NeverCalledProxy(),
                generation_audit=audit,
            )

        estimate.assert_not_called()
        self.assertEqual(audit["summary"]["proxy_calls"], 0)
        self.assertEqual(audit["summary"]["proxy_latency_seconds"], 0.0)
        self.assertEqual(audit["configuration"]["proxy_fusion"], {})

    def test_fused_checkpoint_requires_exact_runtime_metadata(self):
        fields = {
            "mc_fusion_mode": PROXY_FUSION_MODE,
            "proxy_model_name_or_path": "Qwen/Qwen3-1.7B",
            "proxy_model_revision": "revision",
            "proxy_tokenizer_sha256": "abc",
            "proxy_temperature": 1.0,
            "proxy_prior_strength": 2.0,
            "proxy_chat_template_protocol": PROXY_CHAT_TEMPLATE_PROTOCOL,
            "shared_vocab_size": 151669,
            "proxy_vocab_size": 151936,
            "proxy_vocab_tail_policy": PROXY_VOCAB_TAIL_POLICY,
            "proxy_dtype": "float16",
            "proxy_quantization": "none",
        }
        model = SimpleNamespace(config=SimpleNamespace(**fields))
        args = sampling_args(
            proxy_model_name_or_path="Qwen/Qwen3-1.7B",
            proxy_model_revision="revision",
        )

        resolve_proxy_fusion_configuration(args, [model])
        args.proxy_prior_strength = 3.0
        with self.assertRaisesRegex(ValueError, "proxy_prior_strength"):
            resolve_proxy_fusion_configuration(args, [model])

    def test_legacy_checkpoint_rejects_opt_in_proxy(self):
        args = sampling_args(
            proxy_model_name_or_path="Qwen/Qwen3-1.7B",
            proxy_model_revision=None,
        )
        legacy = SimpleNamespace(config=SimpleNamespace())

        with self.assertRaisesRegex(ValueError, "legacy BiasNet checkpoint"):
            resolve_proxy_fusion_configuration(args, [legacy])

    def test_local_proxy_uses_continue_final_message_protocol(self):
        class Model:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=torch.zeros(1, 3, 5))

        tokenizer = TinyTokenizer()
        runtime = LocalProxyRuntime(
            model=Model(),
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            model_name_or_path="Qwen/Qwen3-1.7B",
            model_revision=None,
            tokenizer_sha256="hash",
            chat_template_sha256="template-hash",
            shared_chat_template_sha256="template-hash",
            shared_vocab_size=3,
            proxy_vocab_size=5,
            proxy_dtype="float16",
        )

        logits = runtime.next_token_logits(
            "question",
            "answer",
            qwen_hard_no_think_prefill=True,
        )

        self.assertEqual(tuple(logits.shape), (5,))
        messages, kwargs = tokenizer.template_calls[0]
        self.assertEqual(messages[0], {"role": "user", "content": "question"})
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertTrue(messages[1]["content"].endswith("answer"))
        self.assertTrue(kwargs["continue_final_message"])
        self.assertNotIn("add_generation_prompt", kwargs)

    def test_local_proxy_accepts_attribute_style_batch_encoding(self):
        class AttributeEncodingTokenizer(TinyTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                self.template_calls.append((messages, kwargs))
                return SimpleNamespace(input_ids=torch.tensor([[4, 5, 6]]))

        class Model:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=torch.zeros(1, 3, 5))

        runtime = LocalProxyRuntime(
            model=Model(),
            tokenizer=AttributeEncodingTokenizer(),
            device=torch.device("cpu"),
            model_name_or_path="Qwen/Qwen3-1.7B",
            model_revision=None,
            tokenizer_sha256="hash",
            chat_template_sha256="template-hash",
            shared_chat_template_sha256="template-hash",
            shared_vocab_size=3,
            proxy_vocab_size=5,
            proxy_dtype="float16",
        )

        logits = runtime.next_token_logits(
            "question",
            "answer",
            qwen_hard_no_think_prefill=True,
        )

        self.assertEqual(tuple(logits.shape), (5,))


if __name__ == "__main__":
    unittest.main()
