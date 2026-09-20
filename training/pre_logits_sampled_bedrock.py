#!/usr/bin/env python3
"""Run the sampled next-token cache pipeline through Bedrock Converse.

This is a deliberately thin backend adapter around
``pre_logits_sampled_openrouter.py``.  It reuses the existing dataset,
tokenization, prefix, MC reconstruction, manifest, and fail-closed reasoning
logic while replacing only the remote client and backend provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

SCRIPT_PATH = Path(__file__).resolve()
TRAINING_DIR = SCRIPT_PATH.parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import pre_logits_sampled_openrouter as pipeline  # noqa: E402


REASONING_MARKERS = ("<think>", "</think>", "<reasoning>", "</reasoning>")
DEEPSEEK_V32_TOKENIZER_REPO = "deepseek-ai/DeepSeek-V3.2"
NONRETRYABLE_BEDROCK_ERRORS = frozenset(
    {
        "AccessDeniedException",
        "ResourceNotFoundException",
        "ValidationException",
    }
)
BEDROCK_FINISH_REASON_MAP = {
    "max_tokens": "length",
    "end_turn": "stop",
    "stop_sequence": "stop",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
    "malformed_model_output": "error",
    "malformed_tool_use": "error",
    "model_context_window_exceeded": "error",
}


def parse_backend_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bedrock_region", default="us-east-1")
    parser.add_argument("--bedrock_credential_file", type=Path, default=None)
    parser.add_argument("--bedrock_connect_timeout", type=float, default=15.0)
    parser.add_argument("--bedrock_read_timeout", type=float, default=180.0)
    parser.add_argument("--bedrock_sdk_attempts", type=int, default=4)
    return parser.parse_known_args(argv)


def read_bedrock_token(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    bedrock_section = text.partition("## 2. AWS Bedrock")[2]
    match = re.search(r"Your key:\s*(ABSK[^\s`]+)", bedrock_section)
    if not match:
        raise RuntimeError(f"Could not find a Bedrock bearer token in {path}")
    return match.group(1)


def resolve_bedrock_token(
    args: argparse.Namespace,
    backend_args: argparse.Namespace,
) -> str:
    if args.api_key:
        return str(args.api_key)
    env_value = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
    if env_value:
        return env_value
    if backend_args.bedrock_credential_file is not None:
        return read_bedrock_token(backend_args.bedrock_credential_file)
    raise ValueError(
        "Bedrock bearer token not found. Set AWS_BEARER_TOKEN_BEDROCK, pass "
        "--api_key, or pass --bedrock_credential_file."
    )


def bedrock_reasoning_mode(model_id: str) -> str:
    if model_id == "us.amazon.nova-2-lite-v1:0":
        return "nova_reasoning_disabled"
    return "bedrock_default_fail_closed_audit"


def bedrock_additional_fields(model_id: str) -> Optional[dict[str, Any]]:
    if model_id == "us.amazon.nova-2-lite-v1:0":
        return {"reasoningConfig": {"type": "disabled"}}
    return None


def load_deepseek_v32_native_tokenizer(
    tokenizer_name: str,
    revision: Optional[str] = None,
) -> Optional[Any]:
    """Load the official tokenizer without parsing the 685B model config.

    Some Transformers versions try to instantiate the unknown `deepseek_v32`
    model configuration before loading its otherwise standard tokenizer.json.
    Constructing the generic fast tokenizer directly also preserves the
    tokenizer's ByteLevel decoder; forcing the Llama slow class loses spaces.
    """

    if tokenizer_name != DEEPSEEK_V32_TOKENIZER_REPO:
        return None

    from huggingface_hub import hf_hub_download
    from transformers import AddedToken, PreTrainedTokenizerFast

    tokenizer_path = hf_hub_download(
        tokenizer_name,
        "tokenizer.json",
        revision=revision,
    )
    config_path = hf_hub_download(
        tokenizer_name,
        "tokenizer_config.json",
        revision=revision,
    )
    with open(config_path, "r", encoding="utf-8") as handle:
        tokenizer_config = json.load(handle)

    kwargs: dict[str, Any] = {}
    for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
        value = tokenizer_config.get(key)
        if isinstance(value, dict) and value.get("__type") == "AddedToken":
            added_token_kwargs = {
                option: value[option]
                for option in (
                    "lstrip",
                    "rstrip",
                    "single_word",
                    "normalized",
                    "special",
                )
                if option in value
            }
            value = AddedToken(value["content"], **added_token_kwargs)
        if value is not None:
            kwargs[key] = value

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path, **kwargs)
    if tokenizer_config.get("model_max_length") is not None:
        tokenizer.model_max_length = int(tokenizer_config["model_max_length"])
    if tokenizer_config.get("padding_side") is not None:
        tokenizer.padding_side = tokenizer_config["padding_side"]
    if revision is not None:
        tokenizer.init_kwargs["_commit_hash"] = revision
    return tokenizer


def reasoning_observed(blocks: list[Any], visible_text: str) -> bool:
    for block in blocks:
        if isinstance(block, dict) and block.get("reasoningContent") is not None:
            return True
    folded = visible_text.casefold()
    return any(marker in folded for marker in REASONING_MARKERS)


def normalize_bedrock_response(
    response: dict[str, Any],
    *,
    model_id: str,
    max_tokens: int,
    temperature: float,
) -> pipeline.OpenRouterResponse:
    blocks = response.get("output", {}).get("message", {}).get("content", [])
    if not isinstance(blocks, list):
        blocks = []
    content = "".join(
        block.get("text", "")
        for block in blocks
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    )
    has_reasoning = reasoning_observed(blocks, content)
    bedrock_usage = response.get("usage") or {}
    bedrock_stop_reason = str(response.get("stopReason") or "")
    normalized_stop_reason = BEDROCK_FINISH_REASON_MAP.get(
        bedrock_stop_reason,
        bedrock_stop_reason or None,
    )
    usage = {
        "prompt_tokens": int(bedrock_usage.get("inputTokens") or 0),
        "completion_tokens": int(bedrock_usage.get("outputTokens") or 0),
        "total_tokens": int(bedrock_usage.get("totalTokens") or 0),
        "bedrock": dict(bedrock_usage),
        "bedrock_stop_reason": bedrock_stop_reason or None,
    }
    response_metadata = response.get("ResponseMetadata") or {}
    return pipeline.OpenRouterResponse(
        content=content,
        finish_reason=normalized_stop_reason,
        native_finish_reason=bedrock_stop_reason or None,
        usage=usage,
        # Do not retain chain-of-thought text.  A sentinel is sufficient for
        # the shared --reject_reasoning_tokens path to fail closed.
        reasoning="<bedrock_reasoning_observed>" if has_reasoning else None,
        reasoning_tokens=0,
        model=model_id,
        generation_id=response_metadata.get("RequestId"),
        routing_metadata={
            "attempts": [{"status": 200, "provider": "Amazon Bedrock"}]
        },
        request_max_tokens=int(max_tokens),
        request_temperature=float(temperature),
        request_reasoning_mode=bedrock_reasoning_mode(model_id),
    )


class BedrockConverseClient(pipeline.OpenRouterClient):
    def __init__(
        self,
        args: argparse.Namespace,
        api_key: str,
        backend_args: argparse.Namespace,
    ) -> None:
        super().__init__(args, api_key)
        if backend_args.bedrock_connect_timeout <= 0:
            raise ValueError("--bedrock_connect_timeout must be positive")
        if backend_args.bedrock_read_timeout <= 0:
            raise ValueError("--bedrock_read_timeout must be positive")
        if backend_args.bedrock_sdk_attempts <= 0:
            raise ValueError("--bedrock_sdk_attempts must be positive")

        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError("Install boto3 to use the Bedrock backend") from exc

        self.backend_args = backend_args
        self.bedrock_output_token_counts: Counter[tuple[int, int]] = Counter()
        self.bedrock_stop_reason_counts: Counter[str] = Counter()
        self.runtime = boto3.client(
            "bedrock-runtime",
            region_name=backend_args.bedrock_region,
            config=Config(
                connect_timeout=backend_args.bedrock_connect_timeout,
                read_timeout=backend_args.bedrock_read_timeout,
                retries={
                    "mode": "standard",
                    "total_max_attempts": backend_args.bedrock_sdk_attempts,
                },
            ),
        )

    def generate_many(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int = 1,
    ) -> list[pipeline.OpenRouterResponse]:
        if n != 1:
            raise pipeline.FatalOpenRouterResponseError(
                "Bedrock Converse does not support multiple choices per request; "
                "set --sample_choices_per_request=1."
            )
        self.reserve_request_attempt()
        request: dict[str, Any] = {
            "modelId": self.args.model,
            "messages": [
                {
                    "role": message["role"],
                    "content": [{"text": message["content"]}],
                }
                for message in messages
            ],
            "inferenceConfig": {
                "maxTokens": int(max_tokens),
                "temperature": float(temperature),
                "topP": float(top_p),
            },
        }
        additional_fields = bedrock_additional_fields(self.args.model)
        if additional_fields is not None:
            request["additionalModelRequestFields"] = additional_fields

        try:
            raw_response = self.runtime.converse(**request)
        except Exception as exc:
            error = getattr(exc, "response", {}).get("Error", {})
            code = str(error.get("Code") or "")
            message = str(error.get("Message") or exc)
            if code in NONRETRYABLE_BEDROCK_ERRORS:
                raise pipeline.FatalOpenRouterResponseError(
                    f"Amazon Bedrock {code}: {message[:700]}"
                ) from exc
            raise

        response = normalize_bedrock_response(
            raw_response,
            model_id=self.args.model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        selected_provider = "Amazon Bedrock"
        with self._lock:
            self.calls += 1
            self.total_tokens += int(response.usage.get("total_tokens") or 0)
            self.provider_call_counts[selected_provider] += 1
            self.response_model_call_counts[self.args.model] += 1
            self.response_cache_status_counts["<missing>"] += 1
            self.api_max_tokens_call_counts[int(max_tokens)] += 1
            self.bedrock_output_token_counts[
                (
                    int(max_tokens),
                    int(response.usage.get("completion_tokens") or 0),
                )
            ] += 1
            self.bedrock_stop_reason_counts[
                str(response.usage.get("bedrock_stop_reason") or "<missing>")
            ] += 1
            if response.reasoning is not None:
                self.reasoning_response_calls += 1
                self.reasoning_message_choices += 1
        return [response]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_backend(backend_args: argparse.Namespace) -> None:
    original_parse_args = pipeline.parse_args
    original_build_configuration = pipeline.build_cache_configuration
    original_get_sampled = pipeline.get_sampled_logprobs_openrouter
    original_load_tokenizer = pipeline.load_tokenizer

    def parse_args() -> argparse.Namespace:
        args = original_parse_args()
        args.bedrock_region = backend_args.bedrock_region
        return args

    def resolve_api_key(args: argparse.Namespace) -> str:
        return resolve_bedrock_token(args, backend_args)

    def load_tokenizer(
        tokenizer_name: str,
        use_fast: bool = False,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        fix_mistral_regex: bool = False,
    ) -> Any:
        deepseek_tokenizer = load_deepseek_v32_native_tokenizer(
            tokenizer_name,
            revision=revision,
        )
        if deepseek_tokenizer is not None:
            return deepseek_tokenizer
        return original_load_tokenizer(
            tokenizer_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
            revision=revision,
            fix_mistral_regex=fix_mistral_regex,
        )

    def build_cache_configuration(
        args: argparse.Namespace,
        tokenizer: Any,
    ) -> dict[str, Any]:
        configuration = original_build_configuration(args, tokenizer)
        configuration.update(
            {
                "source": "sampled_bedrock_converse",
                "sampler_source_sha256": sha256_file(SCRIPT_PATH),
                "api_url": (
                    f"bedrock-runtime:{backend_args.bedrock_region}:converse"
                ),
                "bedrock_region": backend_args.bedrock_region,
                "reasoning_mode": bedrock_reasoning_mode(args.model),
                "bedrock_additional_model_request_fields": (
                    bedrock_additional_fields(args.model)
                ),
                "bedrock_reported_output_token_policy": (
                    "audit_only_project_first_proxy_token"
                ),
            }
        )
        return configuration

    def get_sampled_logprobs_bedrock(*args: Any, **kwargs: Any) -> Optional[dict]:
        client = kwargs.get("client")
        if client is None and args:
            client = args[0]
        output_token_counts_before = Counter(
            getattr(client, "bedrock_output_token_counts", {})
        )
        stop_reason_counts_before = Counter(
            getattr(client, "bedrock_stop_reason_counts", {})
        )
        result = original_get_sampled(*args, **kwargs)
        if result is None:
            return None
        metadata = result.setdefault("metadata", {})
        metadata.update(
            {
                "source": "sampled_bedrock_converse",
                "completion_text_semantics": (
                    "continuation_after_assistant_prefill"
                ),
                "raw_completion_text_semantics": (
                    "bedrock_converse_text_content_before_prefill_strip"
                ),
                "reasoning_mode": bedrock_reasoning_mode(metadata["model"]),
                "bedrock_region": backend_args.bedrock_region,
                "bedrock_additional_model_request_fields": (
                    bedrock_additional_fields(metadata["model"])
                ),
                "bedrock_reported_output_token_counts": {
                    f"request_max={request_max},reported_output={reported_output}": count
                    for (request_max, reported_output), count in sorted(
                        (
                            Counter(
                                getattr(client, "bedrock_output_token_counts", {})
                            )
                            - output_token_counts_before
                        ).items()
                    )
                },
                "bedrock_stop_reason_counts": dict(
                    sorted(
                        (
                            Counter(
                                getattr(client, "bedrock_stop_reason_counts", {})
                            )
                            - stop_reason_counts_before
                        ).items()
                    )
                ),
            }
        )
        return result

    class ConfiguredBedrockClient(BedrockConverseClient):
        def __init__(self, args: argparse.Namespace, api_key: str) -> None:
            super().__init__(args, api_key, backend_args)

    pipeline.parse_args = parse_args
    pipeline.resolve_api_key = resolve_api_key
    pipeline.build_cache_configuration = build_cache_configuration
    pipeline.get_sampled_logprobs_openrouter = get_sampled_logprobs_bedrock
    pipeline.load_tokenizer = load_tokenizer
    pipeline.OpenRouterClient = ConfiguredBedrockClient


def main() -> None:
    backend_args, remaining = parse_backend_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]
    install_backend(backend_args)
    pipeline.main()


if __name__ == "__main__":
    main()
