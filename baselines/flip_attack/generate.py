"""Generate victim responses for FlipAttack on the three local benchmarks."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_data import PromptRecord, load_benchmark_records  # noqa: E402

from baselines.flip_attack.attack import FLIP_MODES, FlipAttack, FlipPrompt


DEFAULT_DATA_FILES = {
    benchmark: REPO_ROOT / "flip_data" / f"{benchmark}.csv"
    for benchmark in ("advbench", "harmbench", "sorrybench")
}
DEFAULT_OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_AZURE_URL = (
    "https://example.openai.azure.com/openai/v1/chat/completions"
)
QWEN_HARD_NO_THINK_PREFILL = "<think>\n\n</think>\n\n"


def _parse_bool(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one-query FlipAttack generation and write evaluator-ready JSONL."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=("advbench", "harmbench", "sorrybench"),
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Normalized CSV override. Defaults to flip_data/<benchmark>.csv.",
    )
    parser.add_argument("--begin", type=int, default=0, help="Inclusive row index.")
    parser.add_argument("--end", type=int, default=None, help="Exclusive row index.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Maximum rows after applying begin/end."
    )

    parser.add_argument(
        "--provider",
        choices=(
            "openai",
            "openai-compatible",
            "openrouter",
            "azure",
            "gemini",
            "transformers",
        ),
        default="openrouter",
    )
    parser.add_argument("--model", default=None, help="Victim model ID or local model path.")
    parser.add_argument("--tokenizer-name", default=None)
    parser.add_argument(
        "--provider-order",
        nargs="+",
        default=None,
        help="OpenRouter provider order, e.g. --provider-order DeepInfra.",
    )
    parser.add_argument(
        "--provider-allow-fallbacks",
        type=_parse_bool,
        default=None,
        help="OpenRouter provider allow_fallbacks setting.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Full chat-completions URL or an OpenAI-compatible /v1 base URL.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--api-key-file",
        default=None,
        help=(
            "Python file containing the provider API key (for example api_key.py). "
            "The key value is never written to output metadata."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="API-key environment variable (provider default when omitted).",
    )
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--parallel-requests", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--omit-top-p",
        action="store_true",
        help=(
            "Keep the nominal --top-p value in run metadata but omit top_p from "
            "the provider request. Some Claude endpoints reject requests that "
            "specify both temperature and top_p."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reasoning-mode",
        choices=("omit", "disabled", "enabled"),
        default="omit",
        help=(
            "OpenRouter reasoning control. 'omit' leaves provider defaults unchanged; "
            "'disabled' avoids reasoning-token latency/cost."
        ),
    )
    parser.add_argument(
        "--append-no-think",
        action="store_true",
        help="Append Qwen's /no_think switch to the API user message.",
    )
    parser.add_argument(
        "--qwen-hard-no-think-prefill",
        action="store_true",
        help="Prefill Qwen's assistant continuation with an empty think block.",
    )

    parser.add_argument("--flip-mode", choices=FLIP_MODES, default="FCS")
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--lang-gpt", action="store_true")
    parser.add_argument("--few-shot", action="store_true")

    parser.add_argument("--device", default=None, help="Local Transformers device.")
    parser.add_argument(
        "--device-map",
        default=None,
        help="Local Transformers device map, for example auto.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tokenizer-use-fast", action="store_true")

    parser.add_argument("--output-file", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and append to an interrupted run, skipping completed row indices.",
    )
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Write transformed attack messages without calling a victim model.",
    )
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.begin < 0:
        parser.error("--begin must be non-negative.")
    if args.end is not None and args.end < args.begin:
        parser.error("--end must be greater than or equal to --begin.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if not args.materialize_only and not args.model:
        parser.error("--model is required unless --materialize-only is used.")
    if args.parallel_requests <= 0:
        parser.error("--parallel-requests must be positive.")
    if args.provider == "transformers" and args.parallel_requests != 1:
        parser.error("Local Transformers generation requires --parallel-requests 1.")
    if args.provider != "openrouter" and (
        args.provider_order is not None or args.provider_allow_fallbacks is not None
    ):
        parser.error("--provider-order/--provider-allow-fallbacks require --provider openrouter.")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive.")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and non-negative.")
    if not math.isfinite(args.top_p) or not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1].")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative.")
    if not math.isfinite(args.retry_sleep) or args.retry_sleep < 0:
        parser.error("--retry-sleep must be finite and non-negative.")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be finite and positive.")


def _chat_completions_url(provider: str, base_url: Optional[str]) -> str:
    if base_url:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            return normalized
        return normalized + "/chat/completions"
    if provider == "openrouter":
        return DEFAULT_OPENROUTER_URL
    if provider == "azure":
        return DEFAULT_AZURE_URL
    return DEFAULT_OPENAI_URL


def _is_local_url(url: str) -> bool:
    host = (urllib.parse.urlparse(url).hostname or "").casefold()
    return host in {"localhost", "127.0.0.1", "::1"}


def _api_key(args: argparse.Namespace, url: str) -> Optional[str]:
    if args.api_key:
        return args.api_key
    environment_name = args.api_key_env
    if not environment_name:
        if args.provider == "openrouter":
            environment_name = "OPENROUTER_API_KEY"
        elif args.provider == "azure":
            environment_name = "AZURE_OPENAI_API_KEY"
        elif args.provider == "gemini":
            environment_name = "GOOGLE_API_KEY"
        else:
            environment_name = "OPENAI_API_KEY"
    value = os.environ.get(environment_name)
    if value:
        return value
    if args.api_key_file:
        key_file = Path(args.api_key_file).expanduser().resolve()
        if not key_file.is_file():
            raise ValueError(f"API key file does not exist: {key_file}")
        spec = importlib.util.spec_from_file_location("_flipattack_api_keys", key_file)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load API key file: {key_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidate_names = {
            "openrouter": ("OPENROUTER_API_KEY",),
            "azure": (
                "AZURE_OPENAI_API_KEY",
                "AZURE_API_KEY",
                "AZURE_API_KEY_SECONDARY",
            ),
            "gemini": (
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
                "GEMINI_API_KEY_ALTERNATE",
                "GEMINI_API_KEY_SECONDARY",
            ),
        }.get(args.provider, ("OPENAI_API_KEY",))
        for candidate_name in candidate_names:
            candidate = getattr(module, candidate_name, None)
            if candidate:
                return str(candidate)
        raise ValueError(
            f"No supported {args.provider} API key variable found in {key_file}."
        )
    if args.provider in {"openai", "openai-compatible", "openrouter"} and _is_local_url(url):
        return None
    raise ValueError(
        f"No API key found. Set {environment_name} or pass --api-key explicitly."
    )


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


class OpenAICompatibleVictim:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.url = _chat_completions_url(args.provider, args.base_url)
        self.key = _api_key(args, self.url)

    def generate(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.args.model,
            "messages": messages,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_new_tokens,
            "seed": self.args.seed,
        }
        if not self.args.omit_top_p:
            payload["top_p"] = self.args.top_p
        if self.args.provider == "openrouter":
            provider = {}
            if self.args.provider_order is not None:
                provider["order"] = list(self.args.provider_order)
            if self.args.provider_allow_fallbacks is not None:
                provider["allow_fallbacks"] = bool(self.args.provider_allow_fallbacks)
            if provider:
                payload["provider"] = provider
        if self.args.provider == "openrouter" and self.args.reasoning_mode != "omit":
            payload["reasoning"] = {"enabled": self.args.reasoning_mode == "enabled"}
        headers = {"Content-Type": "application/json"}
        if self.key:
            if self.args.provider == "azure":
                headers["api-key"] = self.key
            else:
                headers["Authorization"] = f"Bearer {self.key}"
        if self.args.provider == "openrouter":
            headers["HTTP-Referer"] = "https://anonymous.invalid"
            headers["X-Title"] = "Anonymous FlipAttack baseline"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.args.request_timeout
            ) as response:
                body = response.read().decode("utf-8")
                response_headers = dict(response.headers.items())
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Chat API returned HTTP {exc.code}: {error_body[:1000]}"
            ) from exc
        parsed = json.loads(body)
        choices = parsed.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError(f"Chat API response has no choices: {body[:1000]}")
        choice = choices[0]
        message = choice.get("message") or {}
        completion = _message_text(message.get("content"))
        if not completion and isinstance(choice.get("text"), str):
            completion = choice["text"]
        metadata = {
            "response_id": parsed.get("id"),
            "response_model": parsed.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "usage": parsed.get("usage"),
        }
        if self.args.provider == "openrouter":
            metadata["openrouter_generation_id"] = parsed.get("id")
            metadata["openrouter_cache_status"] = response_headers.get(
                "x-openrouter-cache"
            )
            metadata["openrouter_provider"] = response_headers.get(
                "x-openrouter-provider"
            )
            metadata["provider_order"] = self.args.provider_order
            metadata["provider_allow_fallbacks"] = self.args.provider_allow_fallbacks
        return completion, metadata


class GeminiVictim:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini generation requires `pip install google-genai`."
            ) from exc
        self.args = args
        self.types = types
        self.client = genai.Client(
            api_key=_api_key(args, "https://generativelanguage.googleapis.com")
        )

    def generate(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        config_kwargs = {
            "system_instruction": messages[0]["content"],
            "max_output_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature,
            "seed": self.args.seed,
        }
        if not self.args.omit_top_p:
            config_kwargs["top_p"] = self.args.top_p
        config = self.types.GenerateContentConfig(**config_kwargs)
        response = self.client.models.generate_content(
            model=self.args.model,
            contents=messages[1]["content"],
            config=config,
        )
        candidates = getattr(response, "candidates", None) or []
        finish_reason = (
            str(getattr(candidates[0], "finish_reason", "")) if candidates else None
        )
        usage = getattr(response, "usage_metadata", None)
        usage_fields = {}
        if usage is not None:
            for name in (
                "prompt_token_count",
                "candidates_token_count",
                "total_token_count",
                "thoughts_token_count",
                "cached_content_token_count",
            ):
                value = getattr(usage, name, None)
                if value is not None:
                    usage_fields[name] = value
        return getattr(response, "text", None) or "", {
            "response_id": getattr(response, "response_id", None),
            "response_model": getattr(response, "model_version", None),
            "finish_reason": finish_reason,
            "usage": usage_fields,
        }


class TransformersVictim:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:
            raise RuntimeError(
                "Local generation requires torch and transformers from the project environment."
            ) from exc

        self.args = args
        self.torch = torch
        set_seed(args.seed)
        tokenizer_name = args.tokenizer_name or args.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=args.tokenizer_use_fast,
            trust_remote_code=args.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": args.trust_remote_code,
        }
        if args.device_map:
            model_kwargs["device_map"] = args.device_map
        self.model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if not args.device_map:
            device = torch.device(
                args.device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.model = self.model.to(device)
        self.model.eval()

    def _input_device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration):
            return next(self.model.parameters()).device

    def generate(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        except (AttributeError, ValueError, TypeError):
            fallback = (
                f"System: {messages[0]['content']}\n\n"
                f"User: {messages[1]['content']}\n\nAssistant:"
            )
            encoded = self.tokenizer(fallback, return_tensors="pt")
        encoded = {key: value.to(self._input_device()) for key, value in encoded.items()}
        generation_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.args.temperature > 0:
            generation_kwargs["temperature"] = self.args.temperature
            if not self.args.omit_top_p:
                generation_kwargs["top_p"] = self.args.top_p
        with self.torch.inference_mode():
            generated = self.model.generate(**encoded, **generation_kwargs)
        input_length = encoded["input_ids"].shape[1]
        new_ids = generated[0, input_length:]
        completion = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return completion, {
            "response_model": self.args.model,
            "generated_token_count": int(new_ids.shape[0]),
        }


def create_victim(args: argparse.Namespace):
    if args.provider in {"openai", "openai-compatible", "openrouter", "azure"}:
        return OpenAICompatibleVictim(args)
    if args.provider == "gemini":
        return GeminiVictim(args)
    if args.provider == "transformers":
        return TransformersVictim(args)
    raise ValueError(f"Unsupported provider: {args.provider}")


def selected_records(args: argparse.Namespace) -> list[PromptRecord]:
    data_path = (
        Path(args.data_file).expanduser()
        if args.data_file
        else DEFAULT_DATA_FILES[args.benchmark]
    )
    records = load_benchmark_records(args.benchmark, data_path)
    if args.begin > len(records):
        raise ValueError(
            f"--begin {args.begin} exceeds {args.benchmark} size {len(records)}."
        )
    end = min(args.end if args.end is not None else len(records), len(records))
    records = records[args.begin:end]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(
            f"No {args.benchmark} records were selected by begin/end/limit."
        )
    return records


def output_path(args: argparse.Namespace) -> Path:
    if args.output_file:
        return Path(args.output_file).expanduser()
    model = args.model or "materialized"
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_")
    filename = f"flipattack_{args.benchmark}_{args.flip_mode.lower()}_{safe_model}.jsonl"
    return REPO_ROOT / "outputs" / filename


def _run_identity(args: argparse.Namespace) -> dict[str, Any]:
    data_file = (
        Path(args.data_file).expanduser()
        if args.data_file
        else DEFAULT_DATA_FILES[args.benchmark]
    )
    return {
        "schema_version": 1,
        "attack_method": "flipattack",
        "benchmark": args.benchmark,
        "data_file": str(data_file.resolve()),
        "flip_mode": args.flip_mode,
        "cot": bool(args.cot),
        "lang_gpt": bool(args.lang_gpt),
        "few_shot": bool(args.few_shot),
        "provider": "materialize-only" if args.materialize_only else args.provider,
        "requested_model": args.model,
        "tokenizer_name": args.tokenizer_name,
        "base_url": args.base_url,
        "provider_order": list(args.provider_order) if args.provider_order is not None else None,
        "provider_allow_fallbacks": args.provider_allow_fallbacks,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "omit_top_p": bool(args.omit_top_p),
        "seed": args.seed,
        "reasoning_mode": args.reasoning_mode,
        "append_no_think": bool(args.append_no_think),
        "qwen_hard_no_think_prefill": bool(args.qwen_hard_no_think_prefill),
    }


def completed_indices(path: Path, args: argparse.Namespace) -> set[int]:
    if not path.exists():
        return set()
    if not args.resume:
        raise FileExistsError(
            f"Output already exists: {path}. Use --resume or choose another path."
        )

    expected = _run_identity(args)
    completed: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("run_identity")
            if identity != expected:
                raise ValueError(
                    f"Resume configuration mismatch at {path}:{line_number}."
                )
            index = row.get("benchmark_index")
            if not isinstance(index, int):
                raise ValueError(
                    f"Resume row {path}:{line_number} has no integer benchmark_index."
                )
            if index in completed:
                raise ValueError(f"Duplicate benchmark_index {index} in {path}.")
            completed.add(index)
    return completed


def _build_attack(args: argparse.Namespace, record: PromptRecord) -> FlipPrompt:
    return FlipAttack(
        flip_mode=args.flip_mode,
        cot=args.cot,
        lang_gpt=args.lang_gpt,
        few_shot=args.few_shot,
        victim_model=args.model or "",
    ).build(record.prompt)


def _request_messages(args: argparse.Namespace, attack: FlipPrompt) -> list[dict[str, str]]:
    """Return API-only message additions without mutating the attack record."""
    messages = [dict(message) for message in attack.messages]
    if args.append_no_think:
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("--append-no-think requires a final user message.")
        messages[-1]["content"] = f'{messages[-1].get("content", "")}\n/no_think'
    if args.qwen_hard_no_think_prefill:
        messages.append({"role": "assistant", "content": QWEN_HARD_NO_THINK_PREFILL})
    return messages


def _result_row(
    args: argparse.Namespace,
    record: PromptRecord,
    attack: FlipPrompt,
    completion: Optional[str],
    response_metadata: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    single_input_prompt = f"{attack.system_prompt}\n\n{attack.user_prompt}"
    row: dict[str, Any] = {
        # Materialized files are attack-input datasets, so their prompt field
        # is directly consumable by a one-string inference harness. Normal
        # generation files retain the original goal for evaluator alignment.
        "prompt": single_input_prompt if args.materialize_only else record.prompt,
        "original_prompt": record.prompt,
        "evaluation_prompt": record.prompt,
        "goal": record.prompt,
        "completion": completion,
    }
    row.update(record.metadata)
    row.update(
        {
            "attack_method": "flipattack",
            "flip_mode": args.flip_mode,
            "flip_attack": attack.log,
            "attack_prompt": attack.user_prompt,
            "input_prompt": single_input_prompt,
            "attack_messages": attack.messages,
            "messages": attack.messages,
            "disguised_prompt": attack.disguised_prompt,
            "flipped_prompt": attack.disguised_prompt,
            "run_identity": _run_identity(args),
            "generation": {
                "provider": "materialize-only" if args.materialize_only else args.provider,
                "requested_model": args.model,
                "provider_order": list(args.provider_order) if args.provider_order is not None else None,
                "provider_allow_fallbacks": args.provider_allow_fallbacks,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "omit_top_p": bool(args.omit_top_p),
                "seed": args.seed,
                "reasoning_mode": args.reasoning_mode,
                "append_no_think": bool(args.append_no_think),
                "qwen_hard_no_think_prefill": bool(args.qwen_hard_no_think_prefill),
                "request_messages": _request_messages(args, attack),
                "elapsed_seconds": elapsed_seconds,
                "response": response_metadata,
            },
        }
    )
    return row


def generate_record(
    args: argparse.Namespace,
    record: PromptRecord,
    victim,
) -> dict[str, Any]:
    attack = _build_attack(args, record)
    if args.materialize_only:
        return _result_row(args, record, attack, None, {}, 0.0)

    last_error: Optional[BaseException] = None
    for attempt in range(1, args.max_retries + 2):
        started = time.monotonic()
        try:
            completion, response_metadata = victim.generate(_request_messages(args, attack))
            if not isinstance(completion, str):
                raise TypeError("Victim client returned a non-string completion.")
            response_metadata = dict(response_metadata or {})
            response_metadata["attempt_count"] = attempt
            return _result_row(
                args,
                record,
                attack,
                completion,
                response_metadata,
                time.monotonic() - started,
            )
        except Exception as exc:
            last_error = exc
            if attempt > args.max_retries:
                break
            delay = args.retry_sleep * (2 ** (attempt - 1)) + random.random() * 0.1
            print(
                f"benchmark_index={record.metadata.get('benchmark_index')} attempt="
                f"{attempt}/{args.max_retries + 1} failed: {exc}; retrying in "
                f"{delay:.2f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(
        f"Victim generation failed for benchmark_index="
        f"{record.metadata.get('benchmark_index')} after {args.max_retries + 1} attempts."
    ) from last_error


def run(args: argparse.Namespace, victim=None) -> Path:
    records = selected_records(args)
    path = output_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_indices(path, args)
    pending = [
        record
        for record in records
        if record.metadata.get("benchmark_index") not in completed
    ]
    if not pending:
        print(f"No pending records; {path} already covers the selected range.")
        return path

    if victim is None and not args.materialize_only:
        victim = create_victim(args)

    mode = "a" if path.exists() else "w"
    written = len(completed)
    with path.open(mode, encoding="utf-8") as handle:
        for start in range(0, len(pending), args.parallel_requests):
            batch = pending[start : start + args.parallel_requests]
            if args.parallel_requests == 1:
                rows = [generate_record(args, batch[0], victim)]
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.parallel_requests
                ) as executor:
                    rows = list(
                        executor.map(
                            lambda record: generate_record(args, record, victim),
                            batch,
                        )
                    )
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()
            processed_pending = start + len(batch)
            if (
                not args.materialize_only
                or processed_pending == len(pending)
                or processed_pending % 100 == 0
            ):
                print(
                    f"written={written} selected={len(records)} output={path}",
                    flush=True,
                )
    return path


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    random.seed(args.seed)
    path = run(args)
    print(f"Finished FlipAttack generation: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
