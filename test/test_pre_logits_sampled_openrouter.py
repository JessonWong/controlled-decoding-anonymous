import json
import io
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from pre_logits_sampled_openrouter import (
    FatalOpenRouterResponseError,
    OpenRouterResponse,
    OpenRouterClient,
    QWEN_HARD_NO_THINK_PREFILL,
    cache_configuration_fingerprint,
    empty_length_retry_budgets,
    ensure_cache_manifest,
    get_sampled_logprobs_openrouter,
    is_claude_45_model,
    is_provider_moderation_rejection,
    main,
    messages_for_prefix,
    parse_args,
    query_next_token_id,
    read_api_key_from_file,
    resolve_api_key,
    response_to_token_id,
    sample_position_token_ids,
    selected_provider_from_metadata,
)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, content, add_special_tokens=False):
        del add_special_tokens
        if not content or content == "unmappable":
            return []
        return [ord(content[0])]

    def decode(self, token_ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    def __len__(self):
        return 128


def response(
    content,
    provider=None,
    model="z-ai/glm-5",
    finish_reason="stop",
    native_finish_reason="stop",
    choice_error=None,
    response_cache_status=None,
    reasoning=None,
    reasoning_details=None,
    reasoning_tokens=0,
):
    routing_metadata = None
    if provider is not None:
        routing_metadata = {
            "attempt": 1,
            "endpoints": {
                "available": [{"provider": provider, "model": model, "selected": True}]
            },
        }
    return OpenRouterResponse(
        content=content,
        finish_reason=finish_reason,
        native_finish_reason=native_finish_reason,
        usage={},
        reasoning=reasoning,
        reasoning_details=reasoning_details,
        reasoning_tokens=reasoning_tokens,
        model=model,
        routing_metadata=routing_metadata,
        choice_error=choice_error,
        response_cache_status=response_cache_status,
    )


class FakeJSONHTTPResponse:
    headers = {"X-OpenRouter-Cache-Status": "MISS"}

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def qwen_router_payload(*, visible_content=None, reasoning=False):
    message = {"content": visible_content}
    usage = {"total_tokens": 1}
    if reasoning:
        message.update(
            {
                "reasoning": "hidden",
                "reasoning_details": [{"type": "reasoning.text"}],
            }
        )
        usage["completion_tokens_details"] = {"reasoning_tokens": 1}
    return {
        "id": "gen-reasoning" if reasoning else "gen-visible",
        "model": "qwen/qwen3-32b",
        "choices": [{"message": message, "finish_reason": "length"}],
        "usage": usage,
        "openrouter_metadata": {
            "attempt": 1,
            "endpoints": {
                "available": [
                    {
                        "provider": "DeepInfra",
                        "model": "qwen/qwen3-32b",
                        "selected": True,
                    }
                ]
            },
        },
    }


def exact_args(**overrides):
    values = {
        "samples_per_token": 4,
        "sample_choices_per_request": 4,
        "sample_temperature": 1.0,
        "top_p": 1.0,
        "api_max_tokens": 1,
        "empty_response_token": "skip",
        "parallel_requests": 1,
        "delay_seconds": 0.0,
        "sample_completion_policy": "exact",
        "max_sample_refill_rounds": 2,
        "empty_length_retry_max_tokens": None,
        "max_empty_length_retry_rounds": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class OneChoiceOnlyClient:
    """Simulates a provider that ignores n and always returns one choice."""

    def __init__(self):
        self.args = SimpleNamespace(empty_response_token="skip")
        self.requested_choice_counts = []
        self.next_token = ord("a")

    def generate_many(self, **kwargs):
        self.requested_choice_counts.append(kwargs["n"])
        content = chr(self.next_token)
        self.next_token += 1
        return [response(content)]


class ExactOpenRouterSamplingTest(unittest.TestCase):
    def test_named_api_key_can_be_selected_from_python_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keys.py"
            path.write_text(
                "OPENROUTER_API_KEY = 'default-key'\n"
                "OPENROUTER_API_KEY_ALTERNATE = 'alternate-test-key'\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_api_key_from_file(
                    str(path), preferred_name="OPENROUTER_API_KEY_ALTERNATE"
                ),
                "alternate-test-key",
            )
            with mock.patch.dict("os.environ", {}, clear=True):
                self.assertEqual(
                    resolve_api_key(
                        SimpleNamespace(
                            api_key=None,
                            api_key_env="OPENROUTER_API_KEY_ALTERNATE",
                            api_key_file=str(path),
                        )
                    ),
                    "alternate-test-key",
                )

    def test_hard_cap_cli_arguments_are_exposed(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "pre_logits_sampled_openrouter.py",
                "--max_api_request_attempts",
                "100000",
                "--max_mc_requested_samples",
                "85000",
                "--dataset_revision",
                "dataset-commit",
                "--fix_mistral_regex",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.max_api_request_attempts, 100000)
        self.assertEqual(args.max_mc_requested_samples, 85000)
        self.assertEqual(args.dataset_revision, "dataset-commit")
        self.assertTrue(args.fix_mistral_regex)

    def test_hard_caps_reserve_atomically_and_fail_before_overshoot(self):
        client = OpenRouterClient(
            SimpleNamespace(
                max_api_request_attempts=2,
                max_mc_requested_samples=3,
            ),
            api_key="unused",
        )

        client.reserve_request_attempt()
        client.reserve_request_attempt()
        with self.assertRaisesRegex(FatalOpenRouterResponseError, "outbound-request cap"):
            client.reserve_request_attempt()
        self.assertEqual(client.request_attempts, 2)

        client.reserve_mc_samples(2)
        with self.assertRaisesRegex(FatalOpenRouterResponseError, "sample cap"):
            client.reserve_mc_samples(2)
        self.assertEqual(client.mc_requested_samples, 2)

    def test_nonpositive_hard_caps_fail_before_loading_assets(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "pre_logits_sampled_openrouter.py",
                "--max_api_request_attempts",
                "0",
            ],
        ):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                main()

    def test_provider_moderation_rejection_detection_is_narrow(self):
        self.assertTrue(
            is_provider_moderation_rejection(
                FatalOpenRouterResponseError(
                    "HTTP 403: model requires moderation; input was flagged"
                )
            )
        )
        self.assertFalse(
            is_provider_moderation_rejection(
                FatalOpenRouterResponseError("HTTP 403: invalid API key")
            )
        )
        self.assertFalse(
            is_provider_moderation_rejection(
                FatalOpenRouterResponseError("HTTP 429: input was flagged")
            )
        )

    def test_claude_45_model_detection_accepts_openrouter_and_native_ids(self):
        self.assertTrue(is_claude_45_model("anthropic/claude-haiku-4.5"))
        self.assertTrue(is_claude_45_model("claude-sonnet-4-5-20250929"))
        self.assertTrue(is_claude_45_model("anthropic/claude-4.5-haiku-20251001"))
        self.assertFalse(is_claude_45_model("anthropic/claude-sonnet-4.6"))
        self.assertFalse(is_claude_45_model("z-ai/glm-5"))

    def test_claude_45_strips_replayed_trailing_prefill_whitespace(self):
        token_id, info = response_to_token_id(
            FakeTokenizer(),
            response(
                "\n    return x",
                model="anthropic/claude-haiku-4.5",
                finish_reason="length",
                native_finish_reason="max_tokens",
            ),
            prefix_text="def square(x):\n    ",
        )

        self.assertEqual(token_id, ord("r"))
        self.assertEqual(info["completion_text"], "return x")
        self.assertEqual(info["raw_completion_text"], "\n    return x")
        self.assertEqual(info["claude_45_whitespace_overlap_chars"], 5)
        self.assertTrue(info["claude_45_whitespace_overlap_applied"])

    def test_non_claude_preserves_matching_leading_whitespace(self):
        token_id, info = response_to_token_id(
            FakeTokenizer(),
            response(" next", model="z-ai/glm-5", finish_reason="length"),
            prefix_text="value ",
        )

        self.assertEqual(token_id, ord(" "))
        self.assertEqual(info["completion_text"], " next")
        self.assertEqual(info["claude_45_whitespace_overlap_chars"], 0)
        self.assertFalse(info["claude_45_whitespace_overlap_applied"])

    def test_claude_45_overlap_only_response_uses_empty_length_retry(self):
        class Client:
            args = SimpleNamespace(
                model="anthropic/claude-haiku-4.5",
                empty_response_token="stop_eos",
                empty_length_retry_max_tokens=4,
                max_empty_length_retry_rounds=2,
            )

            def __init__(self):
                self.max_tokens = []

            def generate(self, **kwargs):
                budget = kwargs["max_tokens"]
                self.max_tokens.append(budget)
                if budget == 1:
                    return response(
                        "\n    ",
                        model="anthropic/claude-haiku-4.5",
                        finish_reason="length",
                        native_finish_reason="max_tokens",
                    )
                return response(
                    "\n    return",
                    model="anthropic/claude-haiku-4.5",
                    finish_reason="length",
                    native_finish_reason="max_tokens",
                )

        client = Client()
        token_id, info = query_next_token_id(
            client=client,
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="def square(x):\n    ",
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
        )

        self.assertEqual(token_id, ord("r"))
        self.assertEqual(client.max_tokens, [1, 2])
        self.assertEqual(info["completion_text"], "return")
        self.assertEqual(info["max_tokens_attempted"], [1, 2])
        self.assertTrue(info["empty_length_retry_recovered"])

    def test_claude_45_overlap_is_counted_in_mc_sampling_audit(self):
        class Client:
            args = SimpleNamespace(
                model="anthropic/claude-haiku-4.5",
                empty_response_token="skip",
            )

            @staticmethod
            def generate_many(**kwargs):
                return [
                    response(
                        "\nX",
                        model="anthropic/claude-haiku-4.5",
                        finish_reason="length",
                    )
                    for _ in range(kwargs["n"])
                ]

        sampled_ids, stats = sample_position_token_ids(
            client=Client(),
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="line 1\n",
            args=exact_args(
                samples_per_token=2,
                sample_choices_per_request=1,
            ),
        )

        self.assertEqual(sampled_ids, [ord("X"), ord("X")])
        self.assertEqual(stats["sampled_completion_text_counts"], {"X": 2})
        self.assertEqual(
            stats["sampled_raw_completion_text_counts"],
            {"\nX": 2},
        )
        self.assertEqual(stats["claude_45_whitespace_overlap_choices"], 2)
        self.assertEqual(stats["claude_45_whitespace_overlap_chars"], 2)

    def test_qwen_hard_no_think_prefill_precedes_every_visible_prefix(self):
        self.assertEqual(
            messages_for_prefix(
                "question",
                "",
                qwen_hard_no_think_prefill=True,
            ),
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": QWEN_HARD_NO_THINK_PREFILL,
                },
            ],
        )
        self.assertEqual(
            messages_for_prefix(
                "question",
                "Sure",
                qwen_hard_no_think_prefill=True,
            )[-1],
            {
                "role": "assistant",
                "content": QWEN_HARD_NO_THINK_PREFILL + "Sure",
            },
        )

    def test_query_uses_qwen_hard_prefill_without_expanding_token_budget(self):
        class Client:
            args = SimpleNamespace(
                empty_response_token="stop_eos",
                empty_length_retry_max_tokens=8,
                max_empty_length_retry_rounds=4,
                qwen_hard_no_think_prefill=True,
                reject_reasoning_tokens=True,
            )

            def __init__(self):
                self.requests = []

            def generate(self, **kwargs):
                self.requests.append(kwargs)
                return response("x", finish_reason="length")

        client = Client()
        token_id, info = query_next_token_id(
            client=client,
            tokenizer=FakeTokenizer(),
            question="question/no_think",
            prefix_text="",
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
        )

        self.assertEqual(token_id, ord("x"))
        self.assertEqual(info["empty_length_retry_attempts"], 0)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["max_tokens"], 1)
        self.assertEqual(
            client.requests[0]["messages"][-1],
            {"role": "assistant", "content": QWEN_HARD_NO_THINK_PREFILL},
        )

    def test_reasoning_signals_are_audited_and_can_fail_closed(self):
        item = response(
            "x",
            reasoning="\n",
            reasoning_details=[{"type": "reasoning.text", "text": "\n"}],
            reasoning_tokens=1,
        )
        token_id, info = response_to_token_id(
            FakeTokenizer(),
            item,
            prefix_text="",
        )
        self.assertEqual(token_id, ord("x"))
        self.assertTrue(info["reasoning_detected"])
        self.assertEqual(info["reasoning_tokens"], 1)
        self.assertEqual(info["reasoning_chars"], 1)
        self.assertEqual(info["reasoning_detail_count"], 1)

        with self.assertRaisesRegex(
            FatalOpenRouterResponseError,
            "returned reasoning",
        ):
            response_to_token_id(
                FakeTokenizer(),
                item,
                prefix_text="",
                reject_reasoning_tokens=True,
            )

    def test_empty_length_retry_budget_schedule_doubles_to_cap(self):
        self.assertEqual(empty_length_retry_budgets(1, 8, 4), [2, 4, 8, 8])
        self.assertEqual(empty_length_retry_budgets(1, None, 4), [])
        self.assertEqual(empty_length_retry_budgets(1, 8, 0), [])
        with self.assertRaisesRegex(ValueError, "must exceed"):
            empty_length_retry_budgets(4, 4, 1)

    def test_stop_eos_projects_only_empty_stop_to_canonical_eos(self):
        token_id, info = response_to_token_id(
            FakeTokenizer(),
            response("", finish_reason="stop"),
            prefix_text="",
            empty_response_token="stop_eos",
        )
        self.assertEqual(token_id, FakeTokenizer.eos_token_id)
        self.assertTrue(info["canonical_eos_projected"])
        self.assertEqual(info["canonical_eos_projection_reason"], "stop")

        token_id, info = response_to_token_id(
            FakeTokenizer(),
            response("", finish_reason="length", native_finish_reason="length"),
            prefix_text="",
            empty_response_token="stop_eos",
        )
        self.assertIsNone(token_id)
        self.assertFalse(info["canonical_eos_projected"])
        self.assertTrue(info["retryable_empty_length"])

    def test_deterministic_empty_length_retries_immediately_and_recovers(self):
        class Client:
            def __init__(self):
                self.args = SimpleNamespace(
                    empty_response_token="stop_eos",
                    empty_length_retry_max_tokens=4,
                    max_empty_length_retry_rounds=2,
                )
                self.max_tokens = []
                self.empty_length_retry_attempts = 0
                self.empty_length_retry_recoveries = 0
                self.empty_length_retry_exhaustions = 0

            def generate(self, **kwargs):
                budget = kwargs["max_tokens"]
                self.max_tokens.append(budget)
                if budget == 1:
                    return response(
                        "",
                        finish_reason="length",
                        native_finish_reason="length",
                    )
                return response(
                    "x",
                    finish_reason="length",
                    native_finish_reason="length",
                )

            def record_empty_length_retry(
                self, *, attempts=0, recoveries=0, exhaustions=0
            ):
                self.empty_length_retry_attempts += attempts
                self.empty_length_retry_recoveries += recoveries
                self.empty_length_retry_exhaustions += exhaustions

        client = Client()
        token_id, info = query_next_token_id(
            client=client,
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="",
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
        )

        self.assertEqual(token_id, ord("x"))
        self.assertEqual(client.max_tokens, [1, 2])
        self.assertEqual(info["max_tokens_attempted"], [1, 2])
        self.assertEqual(info["empty_length_samples"], 1)
        self.assertEqual(info["empty_length_retry_attempts"], 1)
        self.assertTrue(info["empty_length_retry_recovered"])
        self.assertFalse(info["empty_length_retry_exhausted"])
        self.assertEqual(client.empty_length_retry_attempts, 1)
        self.assertEqual(client.empty_length_retry_recoveries, 1)
        self.assertEqual(client.empty_length_retry_exhaustions, 0)

    def test_deterministic_empty_length_retry_exhaustion_is_audited(self):
        class Client:
            def __init__(self):
                self.args = SimpleNamespace(
                    empty_response_token="stop_eos",
                    empty_length_retry_max_tokens=4,
                    max_empty_length_retry_rounds=2,
                )
                self.max_tokens = []
                self.empty_length_retry_attempts = 0
                self.empty_length_retry_recoveries = 0
                self.empty_length_retry_exhaustions = 0

            def generate(self, **kwargs):
                self.max_tokens.append(kwargs["max_tokens"])
                return response(
                    "",
                    finish_reason="length",
                    native_finish_reason="length",
                )

            def record_empty_length_retry(
                self, *, attempts=0, recoveries=0, exhaustions=0
            ):
                self.empty_length_retry_attempts += attempts
                self.empty_length_retry_recoveries += recoveries
                self.empty_length_retry_exhaustions += exhaustions

        client = Client()
        token_id, info = query_next_token_id(
            client=client,
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="",
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
        )

        self.assertIsNone(token_id)
        self.assertEqual(client.max_tokens, [1, 2, 4])
        self.assertEqual(info["empty_length_samples"], 3)
        self.assertEqual(info["empty_length_retry_attempts"], 2)
        self.assertFalse(info["empty_length_retry_recovered"])
        self.assertTrue(info["empty_length_retry_exhausted"])
        self.assertEqual(client.empty_length_retry_attempts, 2)
        self.assertEqual(client.empty_length_retry_recoveries, 0)
        self.assertEqual(client.empty_length_retry_exhaustions, 1)

    def test_nonempty_length_response_does_not_retry(self):
        class Client:
            args = SimpleNamespace(
                empty_response_token="stop_eos",
                empty_length_retry_max_tokens=8,
                max_empty_length_retry_rounds=4,
            )

            def __init__(self):
                self.max_tokens = []

            def generate(self, **kwargs):
                self.max_tokens.append(kwargs["max_tokens"])
                return response(
                    "x",
                    finish_reason="length",
                    native_finish_reason="length",
                )

        client = Client()
        token_id, info = query_next_token_id(
            client=client,
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="",
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
        )

        self.assertEqual(token_id, ord("x"))
        self.assertEqual(client.max_tokens, [1])
        self.assertEqual(info["empty_length_retry_attempts"], 0)

    def test_stop_eos_stats_count_terminal_event_as_valid_sample(self):
        class Client:
            args = SimpleNamespace(empty_response_token="stop_eos")

            @staticmethod
            def generate_many(**kwargs):
                return [
                    response(
                        "",
                        provider="DeepInfra",
                        response_cache_status="MISS",
                    )
                    for _ in range(kwargs["n"])
                ]

        sampled_ids, stats = sample_position_token_ids(
            client=Client(),
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="",
            args=exact_args(
                empty_response_token="stop_eos",
                samples_per_token=3,
                sample_choices_per_request=1,
            ),
        )

        self.assertEqual(sampled_ids, [99, 99, 99])
        self.assertEqual(stats["canonical_eos_projections"], 3)
        self.assertEqual(stats["raw_empty_samples"], 3)
        self.assertEqual(stats["invalid_empty_samples"], 0)
        self.assertEqual(stats["empty_samples"], 0)
        self.assertEqual(stats["finish_reason_counts"], {"stop": 3})
        self.assertEqual(stats["response_cache_status_counts"], {"MISS": 3})

    def test_filtered_and_choice_error_responses_fail_closed(self):
        cases = [
            response(
                "",
                finish_reason="content_filter",
                native_finish_reason="content_filter",
            ),
            response("", choice_error={"code": 400, "message": "choice failed"}),
        ]
        for item in cases:
            with self.subTest(response=item):
                with self.assertRaises(FatalOpenRouterResponseError):
                    response_to_token_id(
                        FakeTokenizer(),
                        item,
                        prefix_text="",
                        empty_response_token="stop_eos",
                    )

    def test_selected_provider_is_extracted_and_counted(self):
        class RoutedClient:
            args = SimpleNamespace(empty_response_token="skip")

            @staticmethod
            def generate_many(**kwargs):
                return [response("a", provider="DeepInfra") for _ in range(kwargs["n"])]

        sampled_ids, stats = sample_position_token_ids(
            client=RoutedClient(),
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="",
            args=exact_args(),
        )

        self.assertEqual(len(sampled_ids), 4)
        self.assertEqual(stats["actual_provider_counts"], {"DeepInfra": 4})
        self.assertEqual(stats["actual_response_model_counts"], {"z-ai/glm-5": 4})
        self.assertEqual(stats["missing_router_metadata_choices"], 0)
        self.assertEqual(stats["max_routing_attempt"], 1)
        self.assertEqual(
            selected_provider_from_metadata(
                {"attempts": [{"provider": "DeepInfra", "status": 200}]}
            ),
            "DeepInfra",
        )

    def test_partial_policy_preserves_legacy_short_response_behavior(self):
        client = OneChoiceOnlyClient()

        sampled_ids, stats = sample_position_token_ids(
            client=client,
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="",
            args=exact_args(sample_completion_policy="partial"),
        )

        self.assertEqual(sampled_ids, [ord("a")])
        self.assertEqual(client.requested_choice_counts, [4])
        self.assertEqual(stats["refill_rounds"], 0)
        self.assertFalse(stats["exact_samples"])

    def test_short_multi_choice_response_refills_to_exact_n_with_single_choice_calls(self):
        client = OneChoiceOnlyClient()

        sampled_ids, stats = sample_position_token_ids(
            client=client,
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="",
            args=exact_args(),
        )

        self.assertEqual(sampled_ids, [ord("a"), ord("b"), ord("c"), ord("d")])
        self.assertEqual(client.requested_choice_counts[0], 4)
        self.assertTrue(all(count == 1 for count in client.requested_choice_counts[1:]))
        self.assertEqual(stats["requested_samples"], 7)
        self.assertEqual(stats["valid_samples"], 4)
        self.assertEqual(stats["returned_choices"], 4)
        self.assertGreaterEqual(stats["refill_rounds"], 1)
        self.assertTrue(stats["used_single_choice_fallback"])
        self.assertTrue(stats["exact_samples"])
        self.assertEqual(
            stats["sampled_completion_text_counts"],
            {"a": 1, "b": 1, "c": 1, "d": 1},
        )

    def test_empty_and_unmappable_choices_are_replenished(self):
        class Client:
            def __init__(self):
                self.args = SimpleNamespace(empty_response_token="skip")
                self.calls = 0
                self.next_token = ord("c")

            def generate_many(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    self.assert_requested(kwargs["n"], 4)
                    return [
                        response("a"),
                        response(""),
                        response("unmappable"),
                        response("b"),
                    ]
                values = []
                for _ in range(kwargs["n"]):
                    values.append(response(chr(self.next_token)))
                    self.next_token += 1
                return values

            @staticmethod
            def assert_requested(actual, expected):
                if actual != expected:
                    raise AssertionError(f"expected n={expected}, got n={actual}")

        client = Client()
        sampled_ids, stats = sample_position_token_ids(
            client=client,
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="",
            args=exact_args(),
        )

        self.assertEqual(sampled_ids, [ord("a"), ord("b"), ord("c"), ord("d")])
        self.assertEqual(stats["valid_samples"], 4)
        self.assertEqual(stats["empty_samples"], 2)
        self.assertGreaterEqual(stats["refill_rounds"], 1)
        self.assertTrue(stats["used_single_choice_fallback"])
        self.assertTrue(stats["exact_samples"])

    def test_exact_policy_fails_closed_when_refills_never_produce_valid_samples(self):
        class AlwaysEmptyClient:
            def __init__(self):
                self.args = SimpleNamespace(empty_response_token="skip")
                self.calls = 0

            def generate_many(self, **kwargs):
                self.calls += 1
                return [response("") for _ in range(kwargs["n"])]

        client = AlwaysEmptyClient()
        args = exact_args(samples_per_token=3, sample_choices_per_request=3, max_sample_refill_rounds=1)

        with self.assertRaisesRegex(RuntimeError, r"exactly 3 valid"):
            sample_position_token_ids(
                client=client,
                tokenizer=FakeTokenizer(),
                question="question",
                prefix_text="",
                args=args,
            )

        self.assertEqual(client.calls, 4)

    def test_adaptive_sampling_starts_only_after_ordinary_refills_exhaust(self):
        class Client:
            def __init__(self):
                self.args = SimpleNamespace(empty_response_token="stop_eos")
                self.max_tokens = []
                self.empty_length_retry_attempts = 0
                self.empty_length_retry_recoveries = 0
                self.empty_length_retry_exhaustions = 0

            def generate_many(self, **kwargs):
                budget = kwargs["max_tokens"]
                self.max_tokens.append(budget)
                if budget == 1:
                    return [
                        response(
                            "",
                            finish_reason="length",
                            native_finish_reason="length",
                        )
                    ]
                return [response("x", finish_reason="length")]

            def record_empty_length_retry(
                self, *, attempts=0, recoveries=0, exhaustions=0
            ):
                self.empty_length_retry_attempts += attempts
                self.empty_length_retry_recoveries += recoveries
                self.empty_length_retry_exhaustions += exhaustions

        client = Client()
        sampled_ids, stats = sample_position_token_ids(
            client=client,
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="",
            args=exact_args(
                samples_per_token=2,
                sample_choices_per_request=1,
                max_sample_refill_rounds=1,
                empty_response_token="stop_eos",
                empty_length_retry_max_tokens=4,
                max_empty_length_retry_rounds=2,
            ),
        )

        self.assertEqual(sampled_ids, [ord("x"), ord("x")])
        self.assertEqual(client.max_tokens, [1, 1, 1, 1, 2, 2])
        self.assertEqual(stats["refill_rounds"], 1)
        self.assertEqual(stats["empty_length_retry_rounds"], 1)
        self.assertEqual(stats["empty_length_samples"], 4)
        self.assertEqual(stats["empty_length_retry_requests"], 2)
        self.assertEqual(stats["empty_length_retry_recoveries"], 2)
        self.assertFalse(stats["empty_length_retry_exhausted"])
        self.assertEqual(stats["sample_max_tokens_choice_counts"], {"1": 4, "2": 2})
        self.assertEqual(stats["requested_samples"], 6)
        self.assertEqual(stats["returned_choices"], 6)
        self.assertEqual(
            stats["returned_choices"],
            stats["valid_samples"] + stats["empty_samples"],
        )
        self.assertEqual(client.empty_length_retry_attempts, 2)
        self.assertEqual(client.empty_length_retry_recoveries, 2)
        self.assertEqual(client.empty_length_retry_exhaustions, 0)

    def test_successful_ordinary_refill_does_not_use_adaptive_budget(self):
        class Client:
            args = SimpleNamespace(empty_response_token="stop_eos")

            def __init__(self):
                self.max_tokens = []

            def generate_many(self, **kwargs):
                self.max_tokens.append(kwargs["max_tokens"])
                if len(self.max_tokens) == 1:
                    return [
                        response(
                            "",
                            finish_reason="length",
                            native_finish_reason="length",
                        )
                    ]
                return [response("x", finish_reason="length")]

        client = Client()
        sampled_ids, stats = sample_position_token_ids(
            client=client,
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="",
            args=exact_args(
                samples_per_token=1,
                sample_choices_per_request=1,
                max_sample_refill_rounds=1,
                empty_response_token="stop_eos",
                empty_length_retry_max_tokens=8,
                max_empty_length_retry_rounds=4,
            ),
        )

        self.assertEqual(sampled_ids, [ord("x")])
        self.assertEqual(client.max_tokens, [1, 1])
        self.assertEqual(stats["empty_length_samples"], 1)
        self.assertEqual(stats["empty_length_retry_requests"], 0)
        self.assertEqual(stats["empty_length_retry_recoveries"], 0)
        self.assertEqual(stats["empty_length_retry_rounds"], 0)

    def test_claude_45_empty_length_bypasses_same_budget_refill_loop(self):
        class Client:
            args = SimpleNamespace(
                model="anthropic/claude-haiku-4.5",
                empty_response_token="stop_eos",
            )

            def __init__(self):
                self.max_tokens = []

            def generate_many(self, **kwargs):
                budget = kwargs["max_tokens"]
                self.max_tokens.append(budget)
                content = " " if budget == 1 else " next"
                return [
                    response(
                        content,
                        model="anthropic/claude-haiku-4.5",
                        finish_reason="length",
                        native_finish_reason="max_tokens",
                    )
                ]

        client = Client()
        sampled_ids, stats = sample_position_token_ids(
            client=client,
            tokenizer=FakeTokenizer(),
            question="question",
            prefix_text="value ",
            args=exact_args(
                samples_per_token=1,
                sample_choices_per_request=1,
                max_sample_refill_rounds=5,
                empty_response_token="stop_eos",
                empty_length_retry_max_tokens=8,
                max_empty_length_retry_rounds=4,
            ),
        )

        self.assertEqual(sampled_ids, [ord("n")])
        self.assertEqual(client.max_tokens, [1, 2])
        self.assertEqual(stats["refill_rounds"], 0)
        self.assertEqual(stats["empty_length_retry_rounds"], 1)
        self.assertEqual(stats["empty_length_retry_requests"], 1)
        self.assertTrue(stats["claude_45_empty_refill_bypass"])

    def test_adaptive_sampling_exhaustion_remains_fail_closed(self):
        class Client:
            def __init__(self):
                self.args = SimpleNamespace(empty_response_token="stop_eos")
                self.max_tokens = []
                self.empty_length_retry_attempts = 0
                self.empty_length_retry_recoveries = 0
                self.empty_length_retry_exhaustions = 0

            def generate_many(self, **kwargs):
                self.max_tokens.append(kwargs["max_tokens"])
                return [
                    response(
                        "",
                        finish_reason="length",
                        native_finish_reason="length",
                    )
                ]

            def record_empty_length_retry(
                self, *, attempts=0, recoveries=0, exhaustions=0
            ):
                self.empty_length_retry_attempts += attempts
                self.empty_length_retry_recoveries += recoveries
                self.empty_length_retry_exhaustions += exhaustions

        client = Client()
        with self.assertRaisesRegex(
            RuntimeError,
            r"3 empty-length choices, 2 adaptive retry requests",
        ):
            sample_position_token_ids(
                client=client,
                tokenizer=FakeTokenizer(),
                question="question",
                prefix_text="",
                args=exact_args(
                    samples_per_token=1,
                    sample_choices_per_request=1,
                    max_sample_refill_rounds=0,
                    empty_response_token="stop_eos",
                    empty_length_retry_max_tokens=4,
                    max_empty_length_retry_rounds=2,
                ),
            )

        self.assertEqual(client.max_tokens, [1, 2, 4])
        self.assertEqual(client.empty_length_retry_attempts, 2)
        self.assertEqual(client.empty_length_retry_recoveries, 0)
        self.assertEqual(client.empty_length_retry_exhaustions, 1)

    def test_cache_payload_contains_retry_tensors_and_budget_audit(self):
        class Client:
            def __init__(self):
                self.args = SimpleNamespace(empty_response_token="stop_eos")
                self.calls = 0
                self.total_cost = 0.0
                self.provider_call_counts = {}
                self.response_model_call_counts = {}
                self.response_cache_status_counts = {}
                self.api_max_tokens_call_counts = {}
                self.empty_length_retry_attempts = 0
                self.empty_length_retry_recoveries = 0
                self.empty_length_retry_exhaustions = 0
                self.missing_router_metadata_calls = 0

            def generate_many(self, **kwargs):
                budget = kwargs["max_tokens"]
                self.calls += 1
                self.provider_call_counts["DeepInfra"] = (
                    self.provider_call_counts.get("DeepInfra", 0) + 1
                )
                self.response_model_call_counts["z-ai/glm-5"] = (
                    self.response_model_call_counts.get("z-ai/glm-5", 0) + 1
                )
                self.response_cache_status_counts["MISS"] = (
                    self.response_cache_status_counts.get("MISS", 0) + 1
                )
                self.api_max_tokens_call_counts[budget] = (
                    self.api_max_tokens_call_counts.get(budget, 0) + 1
                )
                return [
                    response(
                        "x",
                        provider="DeepInfra",
                        finish_reason="length",
                        response_cache_status="MISS",
                    )
                    for _ in range(kwargs["n"])
                ]

        args = exact_args(
            samples_per_token=2,
            sample_choices_per_request=1,
            model="z-ai/glm-5",
            tokenizer_name="zai-org/GLM-5",
            max_answer_tokens=None,
            store_dtype="float32",
            sample_only_risk_active=False,
            max_api_calls=None,
            parallel_positions=1,
            observed_alpha=0.1,
            floor_mass=1e-4,
            provider_order=["DeepInfra"],
            provider_allow_fallbacks=False,
            router_metadata=True,
            disable_openrouter_response_cache=True,
            reasoning_mode="enabled_false",
            cache_configuration_fingerprint="fingerprint",
        )
        payload = get_sampled_logprobs_openrouter(
            client=Client(),
            tokenizer=FakeTokenizer(),
            question="question",
            answer="a",
            args=args,
            risk_gate=None,
        )

        self.assertIsNotNone(payload)
        for key in (
            "empty_length_sample_counts",
            "empty_length_retry_request_counts",
            "empty_length_retry_recovery_counts",
            "empty_length_retry_exhausted_flags",
        ):
            self.assertEqual(tuple(payload[key].shape), (1, 1))
        self.assertEqual(payload["empty_length_sample_counts"].item(), 0)
        self.assertEqual(payload["empty_length_retry_request_counts"].item(), 0)
        self.assertFalse(payload["empty_length_retry_exhausted_flags"].item())
        metadata = payload["metadata"]
        self.assertEqual(payload["sampled_completion_text_counts"], [{"x": 2}])
        self.assertEqual(
            payload["sampled_raw_completion_text_counts"],
            [{"x": 2}],
        )
        self.assertEqual(payload["sampled_prefix_texts"], [""])
        self.assertEqual(payload["prompt_text"], "question")
        self.assertEqual(payload["answer_text"], "a")
        self.assertEqual(metadata["sample_representation"], "completion_text_counter_v1")
        self.assertEqual(
            metadata["raw_completion_text_semantics"],
            "openrouter_message_content_before_prefill_strip",
        )
        self.assertEqual(metadata["empty_length_retry_max_tokens"], None)
        self.assertEqual(metadata["max_empty_length_retry_rounds"], 0)
        self.assertEqual(metadata["sample_max_tokens_choice_counts"], {"1": 2})
        self.assertEqual(metadata["api_max_tokens_call_counts"], {"1": 2})
        self.assertEqual(metadata["api_request_attempts"], 0)
        self.assertEqual(metadata["mc_requested_samples"], 0)
        self.assertEqual(metadata["empty_length_retry_attempts"], 0)
        self.assertEqual(
            metadata["deterministic_empty_length_retry_attempts"], 0
        )

    def test_exact_cache_fails_immediately_on_exhausted_deterministic_query(self):
        class Client:
            calls = 0
            total_cost = 0.0
            provider_call_counts = {}
            response_model_call_counts = {}
            response_cache_status_counts = {}
            api_max_tokens_call_counts = {}
            empty_length_retry_attempts = 0
            empty_length_retry_recoveries = 0
            empty_length_retry_exhaustions = 0
            missing_router_metadata_calls = 0

        args = exact_args(
            samples_per_token=1,
            sample_choices_per_request=1,
            model="z-ai/glm-5",
            tokenizer_name="zai-org/GLM-5",
            max_answer_tokens=None,
            store_dtype="float32",
            max_api_calls=None,
            parallel_positions=1,
            observed_alpha=0.1,
            floor_mass=1e-4,
        )
        exhausted_info = {
            "empty_length_retry_attempts": 2,
            "finish_reason": "length",
            "native_finish_reason": "length",
        }
        sample_stats = {
            "valid_samples": 1,
        }

        for risk_active in (False, True):
            args.sample_only_risk_active = risk_active
            with (
                self.subTest(sample_only_risk_active=risk_active),
                mock.patch(
                    "pre_logits_sampled_openrouter.query_next_token_id",
                    return_value=(None, exhausted_info),
                ),
                mock.patch(
                    "pre_logits_sampled_openrouter.sample_position_token_ids",
                    return_value=([ord("x")], sample_stats),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"answer position 0 after 2 adaptive empty-length retries",
                ):
                    get_sampled_logprobs_openrouter(
                        client=Client(),
                        tokenizer=FakeTokenizer(),
                        question="question",
                        answer="a",
                        args=args,
                        risk_gate=object(),
                    )

    def test_over_returning_provider_fails_closed(self):
        class Client:
            args = SimpleNamespace(empty_response_token="skip")

            @staticmethod
            def generate_many(**kwargs):
                return [response("a"), response("b")]

        with self.assertRaisesRegex(
            FatalOpenRouterResponseError,
            "returned 2 choices for a request of n=1",
        ):
            sample_position_token_ids(
                client=Client(),
                tokenizer=FakeTokenizer(),
                question="question",
                prefix_text="",
                args=exact_args(
                    samples_per_token=1,
                    sample_choices_per_request=1,
                ),
            )


class CacheManifestTest(unittest.TestCase):
    def test_manifest_only_mode_exits_before_api_or_dataset_access(self):
        with tempfile.TemporaryDirectory() as output_dir:
            argv = [
                "pre_logits_sampled_openrouter.py",
                "--output_dir",
                output_dir,
                "--cache_manifest_policy",
                "require",
                "--write_cache_manifest_only",
                "--empty_length_retry_max_tokens",
                "8",
                "--max_empty_length_retry_rounds",
                "4",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch(
                    "pre_logits_sampled_openrouter.load_tokenizer",
                    return_value=FakeTokenizer(),
                ),
                mock.patch(
                    "pre_logits_sampled_openrouter.resolve_api_key"
                ) as resolve_api_key,
                mock.patch(
                    "pre_logits_sampled_openrouter.load_dataset"
                ) as load_dataset,
            ):
                main()

            resolve_api_key.assert_not_called()
            load_dataset.assert_not_called()
            manifest = json.loads(
                (Path(output_dir) / "cache_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            configuration = manifest["configuration"]
            self.assertEqual(configuration["empty_length_retry_max_tokens"], 8)
            self.assertEqual(configuration["max_empty_length_retry_rounds"], 4)

    def test_require_creates_and_reuses_matching_manifest(self):
        configuration = {"model": "z-ai/glm-5", "samples_per_token": 50}
        with tempfile.TemporaryDirectory() as output_dir:
            fingerprint = ensure_cache_manifest(output_dir, configuration, "require")
            manifest_path = Path(output_dir) / "cache_manifest.json"

            self.assertTrue(manifest_path.is_file())
            self.assertEqual(fingerprint, cache_configuration_fingerprint(configuration))
            self.assertEqual(
                ensure_cache_manifest(output_dir, configuration, "require"),
                fingerprint,
            )

    def test_require_rejects_configuration_mismatch(self):
        with tempfile.TemporaryDirectory() as output_dir:
            ensure_cache_manifest(output_dir, {"samples_per_token": 50}, "require")

            with self.assertRaisesRegex(ValueError, "samples_per_token"):
                ensure_cache_manifest(output_dir, {"samples_per_token": 5}, "require")

    def test_require_rejects_legacy_cache_without_manifest(self):
        with tempfile.TemporaryDirectory() as output_dir:
            (Path(output_dir) / "legacy.pt").touch()

            with self.assertRaisesRegex(ValueError, "no cache_manifest"):
                ensure_cache_manifest(output_dir, {"samples_per_token": 50}, "require")


class RouterMetadataClientTest(unittest.TestCase):
    @staticmethod
    def client_args(**overrides):
        values = {
            "model": "z-ai/glm-5",
            "reasoning_mode": "enabled_false",
            "provider_order": ["DeepInfra"],
            "provider_allow_fallbacks": False,
            "router_metadata": True,
            "disable_openrouter_response_cache": True,
            "site_url": "https://example.invalid",
            "app_name": "test",
            "api_url": "https://openrouter.ai/api/v1/chat/completions",
            "request_timeout": 1.0,
            "max_retries": 0,
            "retry_sleep": 0.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_client_requests_and_aggregates_actual_routing_metadata(self):
        args = self.client_args()
        response_payload = {
            "id": "gen-test",
            "model": "z-ai/glm-5",
            "choices": [
                {"message": {"content": "a"}, "finish_reason": "stop"},
                {"message": {"content": "b"}, "finish_reason": "stop"},
            ],
            "usage": {"total_tokens": 3, "cost": 0.01},
            "openrouter_metadata": {
                "attempt": 1,
                "endpoints": {
                    "available": [
                        {
                            "provider": "DeepInfra",
                            "model": "z-ai/glm-5",
                            "selected": True,
                        }
                    ]
                },
            },
        }

        class FakeHTTPResponse:
            headers = {"X-OpenRouter-Cache-Status": "MISS"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            @staticmethod
            def read():
                return json.dumps(response_payload).encode("utf-8")

        captured_requests = []

        def fake_urlopen(request, timeout):
            del timeout
            captured_requests.append(request)
            return FakeHTTPResponse()

        client = OpenRouterClient(args, "secret")
        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            responses = client.generate_many(
                messages=[{"role": "user", "content": "test"}],
                temperature=1.0,
                top_p=1.0,
                max_tokens=1,
                n=2,
            )

        self.assertEqual([item.content for item in responses], ["a", "b"])
        self.assertEqual(client.provider_call_counts, {"DeepInfra": 1})
        self.assertEqual(client.response_model_call_counts, {"z-ai/glm-5": 1})
        self.assertEqual(client.missing_router_metadata_calls, 0)
        self.assertEqual(client.max_routing_attempt, 1)
        self.assertEqual(
            captured_requests[0].get_header("X-openrouter-metadata"),
            "enabled",
        )
        self.assertEqual(
            captured_requests[0].get_header("X-openrouter-cache"),
            "false",
        )
        self.assertEqual(responses[0].response_cache_status, "MISS")
        self.assertEqual(client.response_cache_status_counts, {"MISS": 1})
        self.assertEqual(client.api_max_tokens_call_counts, {1: 1})
        request_payload = json.loads(captured_requests[0].data.decode("utf-8"))
        self.assertEqual(request_payload["n"], 2)
        self.assertEqual(
            request_payload["provider"],
            {"order": ["DeepInfra"], "allow_fallbacks": False},
        )

    def test_client_can_send_gemini_thinking_budget_without_top_p(self):
        response_payload = {
            "id": "gemini-visible",
            "model": "gemini-3.1-pro",
            "choices": [
                {"message": {"content": " token"}, "finish_reason": "length"}
            ],
            "usage": {
                "completion_tokens": 1,
                "completion_tokens_details": {"text_tokens": 1},
            },
        }
        captured_requests = []

        def fake_urlopen(request, timeout):
            del timeout
            captured_requests.append(request)
            return FakeJSONHTTPResponse(response_payload)

        args = self.client_args(
            model="gemini-3.1-pro",
            reasoning_mode="omit",
            thinking_budget=0,
            omit_top_p=True,
        )
        client = OpenRouterClient(args, "secret")
        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            item = client.generate_many(
                messages=[{"role": "user", "content": "test"}],
                temperature=1.0,
                top_p=1.0,
                max_tokens=1,
                n=1,
            )[0]

        request_payload = json.loads(captured_requests[0].data.decode("utf-8"))
        self.assertEqual(
            request_payload["thinking_config"], {"thinking_budget": 0}
        )
        self.assertNotIn("top_p", request_payload)
        self.assertNotIn("reasoning", request_payload)
        self.assertEqual(item.content, " token")

    def test_client_treats_nonempty_thinking_blocks_as_reasoning(self):
        response_payload = {
            "id": "gemini-thinking",
            "model": "gemini-3.1-pro",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "thinking_blocks": [{"type": "thinking", "text": "hidden"}],
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {"completion_tokens": 1},
        }
        args = self.client_args(
            model="gemini-3.1-pro",
            reasoning_mode="omit",
            thinking_budget=0,
        )
        client = OpenRouterClient(args, "secret")
        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            return_value=FakeJSONHTTPResponse(response_payload),
        ):
            item = client.generate_many(
                messages=[{"role": "user", "content": "test"}],
                temperature=1.0,
                top_p=1.0,
                max_tokens=1,
                n=1,
            )[0]

        self.assertEqual(len(item.reasoning_details), 1)
        self.assertEqual(client.reasoning_response_calls, 1)

    def test_client_preserves_and_counts_reasoning_response_fields(self):
        response_payload = {
            "id": "gen-reasoning",
            "model": "qwen/qwen3-32b",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning": "\n",
                        "reasoning_details": [
                            {"type": "reasoning.text", "text": "\n"}
                        ],
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "total_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        }

        class FakeHTTPResponse:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            @staticmethod
            def read():
                return json.dumps(response_payload).encode("utf-8")

        client = OpenRouterClient(self.client_args(), "secret")
        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            return_value=FakeHTTPResponse(),
        ):
            item = client.generate_many(
                messages=[{"role": "user", "content": "test"}],
                temperature=1.0,
                top_p=1.0,
                max_tokens=1,
                n=1,
            )[0]

        self.assertEqual(item.reasoning, "\n")
        self.assertEqual(len(item.reasoning_details), 1)
        self.assertEqual(item.reasoning_tokens, 1)
        self.assertEqual(client.reasoning_response_calls, 1)
        self.assertEqual(client.reasoning_message_choices, 1)
        self.assertEqual(client.reasoning_tokens, 1)

    def test_client_discards_and_retries_reasoning_response(self):
        reasoning_payload = {
            "id": "gen-reasoning",
            "model": "qwen/qwen3-32b",
            "choices": [
                {
                    "message": {
                        "content": None,
                        "reasoning": "hidden",
                        "reasoning_details": [{"type": "reasoning.text"}],
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "total_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
        }
        visible_payload = {
            "id": "gen-visible",
            "model": "qwen/qwen3-32b",
            "choices": [
                {"message": {"content": "a"}, "finish_reason": "length"}
            ],
            "usage": {"total_tokens": 1},
        }

        class FakeHTTPResponse:
            headers = {}

            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        args = self.client_args(
            reject_reasoning_tokens=True,
            max_reasoning_retries=2,
            reasoning_retry_sleep=0.0,
        )
        client = OpenRouterClient(args, "secret")
        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            side_effect=[
                FakeHTTPResponse(reasoning_payload),
                FakeHTTPResponse(visible_payload),
            ],
        ) as urlopen:
            item = client.generate_many(
                messages=[{"role": "user", "content": "test"}],
                temperature=1.0,
                top_p=1.0,
                max_tokens=1,
                n=1,
            )[0]

        self.assertEqual(item.content, "a")
        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(client.calls, 2)
        self.assertEqual(client.reasoning_response_calls, 1)
        self.assertEqual(client.reasoning_retry_attempts, 1)
        self.assertEqual(client.reasoning_retry_recoveries, 1)
        self.assertEqual(client.reasoning_retry_exhaustions, 0)

    def test_deterministic_reasoning_exhaustion_uses_opt_in_sampled_fallback(self):
        args = self.client_args(
            model="qwen/qwen3-32b",
            reject_reasoning_tokens=True,
            max_reasoning_retries=1,
            reasoning_retry_sleep=0.0,
            reasoning_fallback_temperature=1.0,
            max_reasoning_fallback_retries=1,
            empty_response_token="stop_eos",
            empty_length_retry_max_tokens=None,
            max_empty_length_retry_rounds=0,
        )
        client = OpenRouterClient(args, "secret")
        received_requests = []
        responses = [
            qwen_router_payload(reasoning=True),
            qwen_router_payload(reasoning=True),
            qwen_router_payload(reasoning=True),
            qwen_router_payload(visible_content="a"),
        ]

        def fake_urlopen(request, timeout):
            del timeout
            received_requests.append(request)
            return FakeJSONHTTPResponse(responses.pop(0))

        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            token_id, info = query_next_token_id(
                client=client,
                tokenizer=FakeTokenizer(),
                question="test",
                prefix_text="prefix",
                temperature=0.0,
                top_p=1.0,
                max_tokens=1,
            )

        temperatures = [
            json.loads(request.data.decode("utf-8"))["temperature"]
            for request in received_requests
        ]
        self.assertEqual(token_id, ord("a"))
        self.assertEqual(temperatures, [0.0, 0.0, 1.0, 1.0])
        self.assertTrue(info["reasoning_fallback_used"])
        self.assertTrue(info["reasoning_fallback_result_used"])
        self.assertEqual(info["reasoning_fallback_activations"], 1)
        self.assertEqual(info["request_temperature"], 1.0)
        self.assertEqual(info["request_reasoning_mode"], "enabled_false")
        self.assertFalse(info["reasoning_detected"])
        self.assertEqual(client.calls, 4)
        self.assertEqual(client.reasoning_response_calls, 3)
        self.assertEqual(client.reasoning_retry_attempts, 3)
        self.assertEqual(client.reasoning_retry_recoveries, 1)
        self.assertEqual(client.reasoning_retry_exhaustions, 0)
        self.assertEqual(client.reasoning_fallback_activations, 1)
        self.assertEqual(client.reasoning_fallback_response_calls, 2)
        self.assertEqual(client.reasoning_fallback_recoveries, 1)
        self.assertEqual(client.reasoning_fallback_exhaustions, 0)
        self.assertEqual(
            client.reasoning_fallback_provider_call_counts,
            {"DeepInfra": 2},
        )
        self.assertEqual(
            client.reasoning_fallback_response_model_call_counts,
            {"qwen/qwen3-32b": 2},
        )

    def test_deterministic_reasoning_exhaustion_uses_effort_none_fallback(self):
        args = self.client_args(
            model="qwen/qwen3-32b",
            reasoning_mode="enabled_false",
            reject_reasoning_tokens=True,
            max_reasoning_retries=1,
            reasoning_retry_sleep=0.0,
            reasoning_fallback_mode="effort_none",
            max_reasoning_fallback_retries=1,
            empty_response_token="stop_eos",
            empty_length_retry_max_tokens=None,
            max_empty_length_retry_rounds=0,
        )
        client = OpenRouterClient(args, "secret")
        received_requests = []
        responses = [
            qwen_router_payload(reasoning=True),
            qwen_router_payload(reasoning=True),
            qwen_router_payload(reasoning=True),
            qwen_router_payload(visible_content="a"),
        ]

        def fake_urlopen(request, timeout):
            del timeout
            received_requests.append(request)
            return FakeJSONHTTPResponse(responses.pop(0))

        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            token_id, info = query_next_token_id(
                client=client,
                tokenizer=FakeTokenizer(),
                question="test",
                prefix_text="prefix",
                temperature=0.0,
                top_p=1.0,
                max_tokens=1,
            )

        payloads = [
            json.loads(request.data.decode("utf-8"))
            for request in received_requests
        ]
        self.assertEqual([item["temperature"] for item in payloads], [0.0] * 4)
        self.assertEqual(
            [item["reasoning"] for item in payloads],
            [
                {"enabled": False},
                {"enabled": False},
                {"effort": "none"},
                {"effort": "none"},
            ],
        )
        self.assertEqual(token_id, ord("a"))
        self.assertEqual(info["request_temperature"], 0.0)
        self.assertEqual(info["request_reasoning_mode"], "effort_none")
        self.assertTrue(info["reasoning_fallback_used"])
        self.assertTrue(info["reasoning_fallback_result_used"])
        self.assertEqual(info["reasoning_fallback_activations"], 1)
        self.assertFalse(info["reasoning_detected"])
        self.assertEqual(client.calls, 4)
        self.assertEqual(client.reasoning_response_calls, 3)
        self.assertEqual(client.reasoning_retry_attempts, 3)
        self.assertEqual(client.reasoning_retry_recoveries, 1)
        self.assertEqual(client.reasoning_retry_exhaustions, 0)
        self.assertEqual(client.reasoning_fallback_activations, 1)
        self.assertEqual(client.reasoning_fallback_response_calls, 2)
        self.assertEqual(client.reasoning_fallback_recoveries, 1)
        self.assertEqual(client.reasoning_fallback_exhaustions, 0)
        self.assertEqual(
            client.reasoning_fallback_provider_call_counts,
            {"DeepInfra": 2},
        )
        self.assertEqual(
            client.reasoning_fallback_response_model_call_counts,
            {"qwen/qwen3-32b": 2},
        )

    def test_effort_none_fallback_exhaustion_still_fails_closed(self):
        args = self.client_args(
            model="qwen/qwen3-32b",
            reasoning_mode="enabled_false",
            reject_reasoning_tokens=True,
            max_reasoning_retries=1,
            reasoning_retry_sleep=0.0,
            reasoning_fallback_mode="effort_none",
            max_reasoning_fallback_retries=1,
            empty_response_token="stop_eos",
            empty_length_retry_max_tokens=None,
            max_empty_length_retry_rounds=0,
        )
        client = OpenRouterClient(args, "secret")
        received_requests = []
        responses = [qwen_router_payload(reasoning=True) for _ in range(4)]

        def fake_urlopen(request, timeout):
            del timeout
            received_requests.append(request)
            return FakeJSONHTTPResponse(responses.pop(0))

        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with self.assertRaisesRegex(
                FatalOpenRouterResponseError,
                "returned reasoning",
            ):
                query_next_token_id(
                    client=client,
                    tokenizer=FakeTokenizer(),
                    question="test",
                    prefix_text="prefix",
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1,
                )

        payloads = [
            json.loads(request.data.decode("utf-8"))
            for request in received_requests
        ]
        self.assertEqual([item["temperature"] for item in payloads], [0.0] * 4)
        self.assertEqual(
            [item["reasoning"] for item in payloads],
            [
                {"enabled": False},
                {"enabled": False},
                {"effort": "none"},
                {"effort": "none"},
            ],
        )
        self.assertEqual(client.reasoning_response_calls, 4)
        self.assertEqual(client.reasoning_retry_attempts, 3)
        self.assertEqual(client.reasoning_retry_recoveries, 0)
        self.assertEqual(client.reasoning_retry_exhaustions, 1)
        self.assertEqual(client.reasoning_fallback_activations, 1)
        self.assertEqual(client.reasoning_fallback_response_calls, 2)
        self.assertEqual(client.reasoning_fallback_recoveries, 0)
        self.assertEqual(client.reasoning_fallback_exhaustions, 1)

    def test_sampled_reasoning_fallback_exhaustion_still_fails_closed(self):
        args = self.client_args(
            model="qwen/qwen3-32b",
            reject_reasoning_tokens=True,
            max_reasoning_retries=1,
            reasoning_retry_sleep=0.0,
            reasoning_fallback_temperature=1.0,
            max_reasoning_fallback_retries=1,
            empty_response_token="stop_eos",
            empty_length_retry_max_tokens=None,
            max_empty_length_retry_rounds=0,
        )
        client = OpenRouterClient(args, "secret")
        received_requests = []
        responses = [qwen_router_payload(reasoning=True) for _ in range(4)]

        def fake_urlopen(request, timeout):
            del timeout
            received_requests.append(request)
            return FakeJSONHTTPResponse(responses.pop(0))

        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with self.assertRaisesRegex(
                FatalOpenRouterResponseError,
                "returned reasoning",
            ):
                query_next_token_id(
                    client=client,
                    tokenizer=FakeTokenizer(),
                    question="test",
                    prefix_text="prefix",
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1,
                )

        temperatures = [
            json.loads(request.data.decode("utf-8"))["temperature"]
            for request in received_requests
        ]
        self.assertEqual(temperatures, [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(client.reasoning_response_calls, 4)
        self.assertEqual(client.reasoning_retry_attempts, 3)
        self.assertEqual(client.reasoning_retry_recoveries, 0)
        self.assertEqual(client.reasoning_retry_exhaustions, 1)
        self.assertEqual(client.reasoning_fallback_activations, 1)
        self.assertEqual(client.reasoning_fallback_response_calls, 2)
        self.assertEqual(client.reasoning_fallback_recoveries, 0)
        self.assertEqual(client.reasoning_fallback_exhaustions, 1)

    def test_sampled_fallback_does_not_apply_to_non_deterministic_request(self):
        args = self.client_args(
            model="qwen/qwen3-32b",
            reject_reasoning_tokens=True,
            max_reasoning_retries=1,
            reasoning_retry_sleep=0.0,
            reasoning_fallback_temperature=1.0,
            max_reasoning_fallback_retries=8,
            empty_response_token="stop_eos",
            empty_length_retry_max_tokens=None,
            max_empty_length_retry_rounds=0,
        )
        client = OpenRouterClient(args, "secret")
        received_requests = []
        responses = [qwen_router_payload(reasoning=True) for _ in range(2)]

        def fake_urlopen(request, timeout):
            del timeout
            received_requests.append(request)
            return FakeJSONHTTPResponse(responses.pop(0))

        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with self.assertRaisesRegex(
                FatalOpenRouterResponseError,
                "returned reasoning",
            ):
                query_next_token_id(
                    client=client,
                    tokenizer=FakeTokenizer(),
                    question="test",
                    prefix_text="prefix",
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=1,
                )

        temperatures = [
            json.loads(request.data.decode("utf-8"))["temperature"]
            for request in received_requests
        ]
        self.assertEqual(temperatures, [1.0, 1.0])
        self.assertEqual(client.reasoning_retry_exhaustions, 1)
        self.assertEqual(client.reasoning_fallback_activations, 0)
        self.assertEqual(client.reasoning_fallback_response_calls, 0)

    def test_reasoning_mode_fallback_does_not_apply_to_mc_request(self):
        args = self.client_args(
            model="qwen/qwen3-32b",
            reasoning_mode="enabled_false",
            reject_reasoning_tokens=True,
            max_reasoning_retries=1,
            reasoning_retry_sleep=0.0,
            reasoning_fallback_mode="effort_none",
            max_reasoning_fallback_retries=64,
            empty_response_token="stop_eos",
            empty_length_retry_max_tokens=None,
            max_empty_length_retry_rounds=0,
        )
        client = OpenRouterClient(args, "secret")
        received_requests = []
        responses = [qwen_router_payload(reasoning=True) for _ in range(2)]

        def fake_urlopen(request, timeout):
            del timeout
            received_requests.append(request)
            return FakeJSONHTTPResponse(responses.pop(0))

        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with self.assertRaisesRegex(
                FatalOpenRouterResponseError,
                "returned reasoning",
            ):
                query_next_token_id(
                    client=client,
                    tokenizer=FakeTokenizer(),
                    question="test",
                    prefix_text="prefix",
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=1,
                )

        payloads = [
            json.loads(request.data.decode("utf-8"))
            for request in received_requests
        ]
        self.assertEqual([item["temperature"] for item in payloads], [1.0, 1.0])
        self.assertEqual(
            [item["reasoning"] for item in payloads],
            [{"enabled": False}, {"enabled": False}],
        )
        self.assertEqual(client.reasoning_retry_exhaustions, 1)
        self.assertEqual(client.reasoning_fallback_activations, 0)
        self.assertEqual(client.reasoning_fallback_response_calls, 0)

    def test_client_fails_closed_if_disabled_response_cache_reports_hit(self):
        args = self.client_args()
        response_payload = {
            "id": "gen-cache-hit",
            "model": "z-ai/glm-5",
            "choices": [{"message": {"content": "a"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 1, "cost": 0.0},
            "openrouter_metadata": {
                "attempt": 1,
                "endpoints": {
                    "available": [
                        {
                            "provider": "DeepInfra",
                            "model": "z-ai/glm-5",
                            "selected": True,
                        }
                    ]
                },
            },
        }

        class FakeHTTPResponse:
            headers = {"X-OpenRouter-Cache-Status": "HIT"}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            @staticmethod
            def read():
                return json.dumps(response_payload).encode("utf-8")

        client = OpenRouterClient(args, "secret")
        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            return_value=FakeHTTPResponse(),
        ):
            with self.assertRaisesRegex(
                FatalOpenRouterResponseError,
                "response-cache HIT",
            ):
                client.generate_many(
                    messages=[{"role": "user", "content": "test"}],
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=1,
                    n=1,
                )

    def test_client_retries_top_level_500_in_http_success_response(self):
        retry_payload = {
            "error": {
                "code": 500,
                "message": "Provider returned error",
            }
        }
        success_payload = {
            "id": "gen-after-retry",
            "model": "z-ai/glm-5",
            "choices": [{"message": {"content": "a"}, "finish_reason": "length"}],
            "usage": {"total_tokens": 1, "cost": 0.01},
            "openrouter_metadata": {
                "attempt": 1,
                "endpoints": {
                    "available": [
                        {
                            "provider": "DeepInfra",
                            "model": "z-ai/glm-5",
                            "selected": True,
                        }
                    ]
                },
            },
        }

        class FakeHTTPResponse:
            headers = {"X-OpenRouter-Cache-Status": "MISS"}

            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        client = OpenRouterClient(self.client_args(max_retries=1), "secret")
        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            side_effect=[
                FakeHTTPResponse(retry_payload),
                FakeHTTPResponse(success_payload),
            ],
        ) as urlopen:
            responses = client.generate_many(
                messages=[{"role": "user", "content": "test"}],
                temperature=1.0,
                top_p=1.0,
                max_tokens=1,
                n=1,
            )

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual([item.content for item in responses], ["a"])

    def test_client_fails_closed_on_top_level_non_retryable_error(self):
        response_payload = {
            "error": {
                "code": 403,
                "message": "Provider guardrail blocked the request",
            }
        }

        class FakeHTTPResponse:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            @staticmethod
            def read():
                return json.dumps(response_payload).encode("utf-8")

        client = OpenRouterClient(self.client_args(max_retries=4), "secret")
        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            return_value=FakeHTTPResponse(),
        ) as urlopen:
            with self.assertRaisesRegex(
                FatalOpenRouterResponseError,
                "top-level error in an HTTP success response",
            ):
                client.generate_many(
                    messages=[{"role": "user", "content": "test"}],
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=1,
                    n=1,
                )

        urlopen.assert_called_once()

    def test_client_fails_closed_on_non_retryable_http_error(self):
        body = io.BytesIO(
            json.dumps(
                {
                    "error": {
                        "code": 403,
                        "message": "Provider guardrail blocked the request",
                    }
                }
            ).encode("utf-8")
        )
        http_error = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=body,
        )
        client = OpenRouterClient(self.client_args(max_retries=4), "secret")
        with mock.patch(
            "pre_logits_sampled_openrouter.urllib.request.urlopen",
            side_effect=http_error,
        ) as urlopen:
            with self.assertRaisesRegex(
                FatalOpenRouterResponseError,
                "non-retryable HTTP response: HTTP 403",
            ):
                client.generate_many(
                    messages=[{"role": "user", "content": "test"}],
                    temperature=1.0,
                    top_p=1.0,
                    max_tokens=1,
                    n=1,
                )
        urlopen.assert_called_once()


if __name__ == "__main__":
    unittest.main()
