import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AddedToken, AutoTokenizer, PreTrainedTokenizerFast

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pre_logits_sampled_openweight import resolve_torch_dtype, sampled_ids_to_log_probs


QWEN_HARD_NO_THINK_PREFILL = "<think>\n\n</think>\n\n"
CLAUDE_45_WHITESPACE_PREFILL_FIX = "trailing_whitespace_overlap_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Monte Carlo next-token log-prob estimates from OpenRouter chat models."
    )
    parser.add_argument("--model", type=str, default="deepseek/deepseek-v4-flash")
    parser.add_argument("--tokenizer_name", type=str, default="deepseek-ai/DeepSeek-V4-Flash")
    parser.add_argument("--tokenizer_revision", type=str, default=None)
    parser.add_argument(
        "--fix_mistral_regex",
        action="store_true",
        help=(
            "Pass fix_mistral_regex=True when loading the tokenizer. Required by "
            "affected Mistral tokenizers to avoid silently incorrect pre-tokenization."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./cached_logits/deepseek_v4_flash_mc500_floor_riskgate")
    parser.add_argument(
        "--cache_manifest_policy",
        choices=["ignore", "require"],
        default="ignore",
        help=(
            "With 'require', create and validate cache_manifest.json and refuse to mix "
            "cache files produced by a different model/provider/sampling configuration."
        ),
    )
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--samples_per_token", type=int, default=500)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--omit_top_p",
        action="store_true",
        help=(
            "Do not send top_p to the chat-completions endpoint. Claude 4+ rejects "
            "requests that specify both temperature and top_p."
        ),
    )
    parser.add_argument("--observed_alpha", type=float, default=0.1)
    parser.add_argument("--floor_mass", type=float, default=1e-4)
    parser.add_argument("--max_answer_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="LLM-LAT/harmful-dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional immutable Hugging Face dataset revision used for cache provenance.",
    )
    parser.add_argument(
        "--dataset_jsonl",
        type=str,
        default=None,
        help=(
            "Optional local JSONL dataset. When set, this replaces the Hugging "
            "Face dataset and preserves the same start/end row semantics."
        ),
    )
    parser.add_argument("--dataset_prompt_field", type=str, default="prompt")
    parser.add_argument("--dataset_answer_field", type=str, default="rejected")
    parser.add_argument("--start_index", type=int, default=100)
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Optional exclusive dataset index bound used to lock the training row set.",
    )
    parser.add_argument("--store_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--api_key_file", type=str, default=None)
    parser.add_argument("--api_url", type=str, default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--site_url", type=str, default="https://anonymous.invalid")
    parser.add_argument("--app_name", type=str, default="Anonymous MC cache")
    parser.add_argument("--request_timeout", type=float, default=90.0)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--parallel_requests", type=int, default=1)
    parser.add_argument(
        "--parallel_positions",
        type=int,
        default=1,
        help="Sample multiple teacher-forced answer positions concurrently.",
    )
    parser.add_argument(
        "--sample_choices_per_request",
        type=int,
        default=1,
        help="Request this many sampled choices per OpenRouter call when estimating MC counts.",
    )
    parser.add_argument(
        "--sample_completion_policy",
        choices=["partial", "exact"],
        default="partial",
        help=(
            "Whether to accept fewer valid samples than --samples_per_token or refill "
            "with single-choice requests and fail closed. Use 'exact' for reproducible MCN caches."
        ),
    )
    parser.add_argument(
        "--max_sample_refill_rounds",
        type=int,
        default=2,
        help="Maximum single-choice refill rounds when --sample_completion_policy=exact.",
    )
    parser.add_argument("--api_max_tokens", type=int, default=1)
    parser.add_argument(
        "--empty_length_retry_max_tokens",
        type=int,
        default=None,
        help=(
            "Optional maximum token budget for adaptive retries of empty choices whose "
            "normalized finish_reason is 'length'. The retry budget doubles from "
            "--api_max_tokens up to this cap."
        ),
    )
    parser.add_argument(
        "--max_empty_length_retry_rounds",
        type=int,
        default=0,
        help=(
            "Maximum adaptive empty-length retry rounds. Deterministic queries retry "
            "immediately; exact MC sampling retries only after ordinary refill rounds "
            "are exhausted. Zero preserves the legacy behavior."
        ),
    )
    parser.add_argument(
        "--empty_response_token",
        choices=["skip", "eos", "stop_eos"],
        default="skip",
        help=(
            "How to map empty OpenRouter completions during token reconstruction. "
            "'skip' preserves legacy rejection/refill behavior, 'eos' maps every empty "
            "choice to the tokenizer EOS token, and 'stop_eos' only maps an empty choice "
            "whose normalized finish_reason is 'stop' to canonical EOS."
        ),
    )
    parser.add_argument("--max_api_calls", type=int, default=None)
    parser.add_argument(
        "--max_api_request_attempts",
        type=int,
        default=None,
        help=(
            "Hard process-wide cap on outbound HTTP attempts, including retries. "
            "Work is reserved atomically before concurrent requests are issued."
        ),
    )
    parser.add_argument(
        "--max_mc_requested_samples",
        type=int,
        default=None,
        help=(
            "Hard process-wide cap on requested stochastic MC choices, including "
            "refills. Work is reserved atomically before requests are issued."
        ),
    )
    parser.add_argument("--delay_seconds", type=float, default=0.0)
    parser.add_argument(
        "--sample_only_risk_active",
        action="store_true",
        help=(
            "When a risk gate is configured, first scan deterministic API tokens and "
            "only run MC sampling for positions whose risk mask is active. This is "
            "equivalent for hard risk-gated training, where inactive positions have zero loss."
        ),
    )
    parser.add_argument(
        "--reasoning_mode",
        choices=["enabled_false", "effort_none", "omit"],
        default="enabled_false",
        help="OpenRouter reasoning control. Use enabled_false to avoid invisible reasoning tokens.",
    )
    parser.add_argument(
        "--thinking_budget",
        type=int,
        default=None,
        help=(
            "Optional provider-native thinking budget sent as "
            "thinking_config.thinking_budget. NAIRR Gemini 3.1 Pro requires 0 "
            "for visible-only cache sampling."
        ),
    )
    parser.add_argument(
        "--append_no_think",
        action="store_true",
        help=(
            "Append Qwen's /no_think soft switch to each user prompt. This is useful "
            "for providers that ignore the OpenRouter reasoning=false field."
        ),
    )
    parser.add_argument(
        "--qwen_hard_no_think_prefill",
        action="store_true",
        help=(
            "Prefix every Qwen assistant continuation with its canonical empty think "
            "block, followed by the teacher-forced visible answer prefix, so every "
            "API output token belongs to the visible answer channel."
        ),
    )
    parser.add_argument(
        "--reject_reasoning_tokens",
        action="store_true",
        help=(
            "Fail closed when OpenRouter returns message reasoning/reasoning_details or "
            "reports any completion reasoning tokens."
        ),
    )
    parser.add_argument(
        "--provider_order",
        nargs="+",
        default=None,
        help="Optional OpenRouter provider order, e.g. --provider_order Together.",
    )
    parser.add_argument(
        "--provider_allow_fallbacks",
        type=parse_bool,
        default=None,
        help="Optional OpenRouter provider allow_fallbacks setting.",
    )
    parser.add_argument(
        "--provider_quantizations",
        nargs="+",
        default=None,
        help="Optional OpenRouter provider quantization filter, e.g. --provider_quantizations fp8.",
    )
    parser.add_argument(
        "--router_metadata",
        action="store_true",
        help=(
            "Request OpenRouter routing metadata and record the provider/model that "
            "actually served sampled choices."
        ),
    )
    parser.add_argument(
        "--disable_openrouter_response_cache",
        action="store_true",
        help=(
            "Send X-OpenRouter-Cache: false so repeated stochastic requests cannot be "
            "served by OpenRouter's response cache."
        ),
    )
    parser.add_argument(
        "--skip_provider_moderation_rejections",
        action="store_true",
        help=(
            "Skip an entire dataset example only when Amazon Bedrock/OpenRouter "
            "returns an explicit HTTP 403 moderation rejection. Other fatal "
            "responses remain fail-closed. Skips are recorded without prompt text."
        ),
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--risk_gate_checkpoint",
        type=str,
        default=None,
        help="Optional prefix-risk checkpoint. When set, cache a per-token mask for risk-gated training.",
    )
    parser.add_argument("--risk_gate_threshold", type=float, default=0.1)
    parser.add_argument("--risk_gate_batch_size", type=int, default=16)
    parser.add_argument("--risk_gate_max_length", type=int, default=None)
    parser.add_argument(
        "--risk_gate_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--risk_gate_device", type=str, default=None)
    parser.add_argument("--risk_gate_model_name", type=str, default=None)
    parser.add_argument("--risk_gate_load_in_4bit", action="store_true")
    parser.add_argument("--risk_gate_trust_remote_code", action="store_true")
    parser.add_argument(
        "--risk_gate_local_files_only",
        action="store_true",
        help="Require the risk-gate backbone to already exist in the local HF cache.",
    )
    parser.add_argument(
        "--store_deterministic_base_tokens",
        action="store_true",
        help=(
            "Also query and store the temperature-0 base token for every cached "
            "answer position, without loading a local risk gate. This allows a "
            "later gate-rescore pass to run independently of target-model MC sampling."
        ),
    )
    parser.add_argument(
        "--write_cache_manifest_only",
        action="store_true",
        help=(
            "Create or validate cache_manifest.json, then exit before resolving an API "
            "key, loading the risk gate, loading the dataset, or making API calls."
        ),
    )
    return parser.parse_args()


def read_api_key_from_file(
    path: str,
    preferred_name: Optional[str] = None,
) -> Optional[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("juli_api_key", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = (
        *((preferred_name,) if preferred_name else ()),
        "OPENROUTER_API_KEY",
        "openrouter_api_key",
        "OPEN_ROUTER_API_KEY",
        "api_key",
    )
    for name in dict.fromkeys(names):
        value = getattr(module, name, None)
        if value:
            return str(value)
    return None


def load_tokenizer(
    tokenizer_name: str,
    use_fast: bool = False,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    fix_mistral_regex: bool = False,
):
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
            revision=revision,
            fix_mistral_regex=fix_mistral_regex,
        )
    except Exception as exc:
        # Some public tokenizer repositories (notably DeepSeek-V4-Flash) ship
        # a model config that the installed Transformers release tries to
        # instantiate before reading tokenizer.json.  The tokenizer itself is
        # still usable, so fall back to constructing the fast tokenizer
        # directly from the repository's tokenizer files.
        direct_json_fallback = (
            "Tokenizer class TokenizersBackend" in str(exc)
            or "max_position_embeddings" in str(exc)
            # Newer model configs can fail validation in an older installed
            # Transformers/Hugging Face stack before tokenizer.json is read.
            # The tokenizer files remain independently usable in this case.
            or "layer_types" in str(exc)
            or "StrictDataclassClassValidationError" in str(exc)
        )
        if not direct_json_fallback:
            raise
        from huggingface_hub import hf_hub_download

        tokenizer_path = hf_hub_download(
            tokenizer_name, "tokenizer.json", revision=revision
        )
        config_path = hf_hub_download(
            tokenizer_name, "tokenizer_config.json", revision=revision
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            tokenizer_config = json.load(handle)
        kwargs = {}
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
        return tokenizer


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    env_value = os.environ.get(args.api_key_env)
    if env_value:
        return env_value
    if args.api_key_file:
        file_value = read_api_key_from_file(
            args.api_key_file,
            preferred_name=args.api_key_env,
        )
        if file_value:
            return file_value
    raise ValueError(
        "OpenRouter API key not found. Pass --api_key, set the environment "
        f"variable named by --api_key_env ({args.api_key_env}), or define that "
        "name in --api_key_file."
    )


def is_provider_moderation_rejection(exc: BaseException) -> bool:
    """Recognize the explicit, non-billable provider moderation refusal."""

    message = str(exc).casefold()
    return (
        "http 403" in message
        and "requires moderation" in message
        and "flagged" in message
    )


def record_provider_moderation_skip(
    output_dir: str,
    *,
    dataset_idx: int,
    question: str,
    answer: str,
) -> None:
    """Persist a deduplicated skip audit without retaining sensitive prompt text."""

    path = Path(output_dir) / "skipped_provider_moderation.jsonl"
    existing_indices: set[int] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                existing_indices.add(int(json.loads(line)["dataset_idx"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    if int(dataset_idx) in existing_indices:
        return
    record = {
        "dataset_idx": int(dataset_idx),
        "reason": "provider_moderation_http_403",
        "prompt_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "data_sha256": hashlib.sha256(
            (question + answer).encode("utf-8")
        ).hexdigest(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def risk_gate_identity(checkpoint: Optional[str]) -> Optional[dict]:
    if not checkpoint:
        return None
    root = Path(checkpoint).expanduser().resolve()
    return {
        "path": str(root),
        "risk_head_config_sha256": sha256_file(root / "risk_head_config.json"),
        "risk_head_sha256": sha256_file(root / "risk_head.pt"),
    }


def build_cache_configuration(args: argparse.Namespace, tokenizer) -> dict:
    tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    dataset_jsonl = (
        Path(args.dataset_jsonl).expanduser().resolve()
        if args.dataset_jsonl
        else None
    )
    return {
        "schema_version": 2,
        "source": "sampled_openrouter",
        "sample_representation": "completion_text_counter_v1",
        "materialized_log_probs_included": True,
        "sampler_source_sha256": sha256_file(Path(__file__).resolve()),
        "logprob_source_sha256": sha256_file(
            Path(PROJECT_ROOT) / "training" / "pre_logits_sampled_openweight.py"
        ),
        "model": args.model,
        "tokenizer_name": args.tokenizer_name,
        "tokenizer_commit": tokenizer_commit,
        "tokenizer_revision": args.tokenizer_revision,
        "fix_mistral_regex": bool(getattr(args, "fix_mistral_regex", False)),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "dataset_revision": args.dataset_revision,
        "dataset_jsonl": str(dataset_jsonl) if dataset_jsonl else None,
        "dataset_jsonl_sha256": sha256_file(dataset_jsonl) if dataset_jsonl else None,
        "dataset_prompt_field": args.dataset_prompt_field,
        "dataset_answer_field": args.dataset_answer_field,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "max_samples": args.max_samples,
        "max_answer_tokens": args.max_answer_tokens,
        "samples_per_token": args.samples_per_token,
        "sample_choices_per_request": args.sample_choices_per_request,
        "sample_completion_policy": args.sample_completion_policy,
        "max_sample_refill_rounds": args.max_sample_refill_rounds,
        "sample_temperature": args.sample_temperature,
        "top_p": args.top_p,
        "omit_top_p": bool(getattr(args, "omit_top_p", False)),
        "observed_alpha": args.observed_alpha,
        "floor_mass": args.floor_mass,
        "store_dtype": args.store_dtype,
        "api_max_tokens": args.api_max_tokens,
        "empty_length_retry_max_tokens": getattr(
            args, "empty_length_retry_max_tokens", None
        ),
        "max_empty_length_retry_rounds": int(
            getattr(args, "max_empty_length_retry_rounds", 0)
        ),
        "empty_response_token": args.empty_response_token,
        "reasoning_mode": args.reasoning_mode,
        "thinking_budget": getattr(args, "thinking_budget", None),
        "append_no_think": bool(getattr(args, "append_no_think", False)),
        "qwen_hard_no_think_prefill": bool(
            getattr(args, "qwen_hard_no_think_prefill", False)
        ),
        "claude_45_whitespace_prefill_fix": (
            CLAUDE_45_WHITESPACE_PREFILL_FIX
            if is_claude_45_model(args.model)
            else None
        ),
        "reject_reasoning_tokens": bool(
            getattr(args, "reject_reasoning_tokens", False)
        ),
        "provider_order": list(args.provider_order) if args.provider_order else None,
        "provider_allow_fallbacks": args.provider_allow_fallbacks,
        "router_metadata": args.router_metadata,
        "disable_openrouter_response_cache": bool(
            getattr(args, "disable_openrouter_response_cache", False)
        ),
        "skip_provider_moderation_rejections": bool(
            getattr(args, "skip_provider_moderation_rejections", False)
        ),
        "api_url": args.api_url,
        "sample_only_risk_active": args.sample_only_risk_active,
        "risk_gate": risk_gate_identity(args.risk_gate_checkpoint),
        "risk_gate_threshold": args.risk_gate_threshold if args.risk_gate_checkpoint else None,
        "risk_gate_max_length": args.risk_gate_max_length if args.risk_gate_checkpoint else None,
        "risk_gate_dtype": args.risk_gate_dtype if args.risk_gate_checkpoint else None,
        "risk_gate_model_name": args.risk_gate_model_name if args.risk_gate_checkpoint else None,
        "risk_gate_load_in_4bit": args.risk_gate_load_in_4bit if args.risk_gate_checkpoint else None,
        "risk_gate_local_files_only": (
            args.risk_gate_local_files_only if args.risk_gate_checkpoint else None
        ),
        "store_deterministic_base_tokens": bool(
            getattr(args, "store_deterministic_base_tokens", False)
        ),
    }


def load_training_data(args: argparse.Namespace):
    if not args.dataset_jsonl:
        dataset = load_dataset(args.dataset_name, revision=args.dataset_revision)
        return dataset[args.dataset_split]

    path = Path(args.dataset_jsonl).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Local JSONL dataset does not exist: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {path} at line {line_number}."
                )
            for field in (args.dataset_prompt_field, args.dataset_answer_field):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(
                        f"{path} line {line_number} requires a non-empty string "
                        f"field {field!r}."
                    )
            rows.append(row)
    if not rows:
        raise ValueError(f"Local JSONL dataset is empty: {path}")
    return rows


def cache_configuration_fingerprint(configuration: dict) -> str:
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_cache_manifest(
    output_dir: str,
    configuration: dict,
    policy: str,
) -> Optional[str]:
    if policy == "ignore":
        return None
    if policy != "require":
        raise ValueError("cache manifest policy must be 'ignore' or 'require'.")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "cache_manifest.json"
    fingerprint = cache_configuration_fingerprint(configuration)
    existing_cache_files = list(root.glob("*.pt"))

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("configuration_fingerprint") != fingerprint:
            old_config = manifest.get("configuration") or {}
            differing = sorted(
                key
                for key in set(old_config) | set(configuration)
                if old_config.get(key) != configuration.get(key)
            )
            raise ValueError(
                "Cache manifest does not match this run. Use a fresh --output_dir. "
                f"Differing fields: {', '.join(differing) or 'unknown'}."
            )
        return fingerprint

    if existing_cache_files:
        raise ValueError(
            f"{root} contains {len(existing_cache_files)} .pt files but no cache_manifest.json. "
            "Use a fresh directory for --cache_manifest_policy=require."
        )

    manifest = {
        "manifest_schema_version": 1,
        "configuration_fingerprint": fingerprint,
        "configuration": configuration,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary_path = manifest_path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    temporary_path.replace(manifest_path)
    return fingerprint


def apply_reasoning_mode(payload: dict, mode: str) -> None:
    if mode == "enabled_false":
        payload["reasoning"] = {"enabled": False}
    elif mode == "effort_none":
        payload["reasoning"] = {"effort": "none"}
    elif mode == "omit":
        payload.pop("reasoning", None)
    else:
        raise ValueError(f"Unsupported reasoning mode: {mode}")


def parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def apply_provider_routing(payload: dict, args: argparse.Namespace) -> None:
    provider_order = getattr(args, "provider_order", None)
    provider_allow_fallbacks = getattr(args, "provider_allow_fallbacks", None)
    provider_quantizations = getattr(args, "provider_quantizations", None)
    if (
        not provider_order
        and provider_allow_fallbacks is None
        and not provider_quantizations
    ):
        return
    provider = {}
    if provider_order:
        provider["order"] = list(provider_order)
    if provider_allow_fallbacks is not None:
        provider["allow_fallbacks"] = bool(provider_allow_fallbacks)
    if provider_quantizations:
        provider["quantizations"] = list(provider_quantizations)
    payload["provider"] = provider


class FatalOpenRouterResponseError(RuntimeError):
    """A semantic response failure that must not be hidden by retries or refills."""


RETRYABLE_OPENROUTER_STATUS_CODES = frozenset(
    {408, 409, 425, 429, 500, 502, 503, 504}
)


@dataclass
class OpenRouterResponse:
    content: str
    finish_reason: Optional[str]
    native_finish_reason: Optional[str]
    usage: dict
    reasoning: Any = None
    reasoning_details: Any = None
    reasoning_tokens: int = 0
    model: Optional[str] = None
    generation_id: Optional[str] = None
    routing_metadata: Optional[dict] = None
    choice_error: Optional[dict] = None
    response_cache_status: Optional[str] = None
    request_max_tokens: Optional[int] = None
    request_temperature: Optional[float] = None
    request_reasoning_mode: Optional[str] = None
    reasoning_fallback_used: bool = False


def selected_provider_from_metadata(metadata: Optional[dict]) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None
    endpoints = metadata.get("endpoints")
    available = endpoints.get("available") if isinstance(endpoints, dict) else None
    if isinstance(available, list):
        for endpoint in available:
            if isinstance(endpoint, dict) and endpoint.get("selected"):
                provider = endpoint.get("provider")
                if provider:
                    return str(provider)
    attempts = metadata.get("attempts")
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if not isinstance(attempt, dict):
                continue
            status = attempt.get("status")
            if status == 200 or str(status) == "200":
                provider = attempt.get("provider")
                if provider:
                    return str(provider)
    return None


class OpenRouterClient:
    def __init__(self, args: argparse.Namespace, api_key: str) -> None:
        self.args = args
        self.api_key = api_key
        self.calls = 0
        # ``calls`` counts successfully parsed HTTP responses.  These counters
        # instead reserve work before it is issued, so optional experiment caps
        # remain hard under parallel requests and retry loops.
        self.request_attempts = 0
        self.mc_requested_samples = 0
        self.total_cost = 0.0
        self.total_tokens = 0
        self.provider_call_counts: Counter[str] = Counter()
        self.response_model_call_counts: Counter[str] = Counter()
        self.response_cache_status_counts: Counter[str] = Counter()
        self.api_max_tokens_call_counts: Counter[int] = Counter()
        self.empty_length_retry_attempts = 0
        self.empty_length_retry_recoveries = 0
        self.empty_length_retry_exhaustions = 0
        self.reasoning_response_calls = 0
        self.reasoning_message_choices = 0
        self.reasoning_tokens = 0
        self.reasoning_retry_attempts = 0
        self.reasoning_retry_recoveries = 0
        self.reasoning_retry_exhaustions = 0
        self.reasoning_fallback_activations = 0
        self.reasoning_fallback_response_calls = 0
        self.reasoning_fallback_recoveries = 0
        self.reasoning_fallback_exhaustions = 0
        self.reasoning_fallback_provider_call_counts: Counter[str] = Counter()
        self.reasoning_fallback_response_model_call_counts: Counter[str] = Counter()
        self.missing_router_metadata_calls = 0
        self.max_routing_attempt = 0
        self._lock = threading.Lock()

    def reserve_request_attempt(self) -> None:
        with self._lock:
            limit = getattr(self.args, "max_api_request_attempts", None)
            if limit is not None and self.request_attempts >= int(limit):
                raise FatalOpenRouterResponseError(
                    "Reached the hard outbound-request cap "
                    f"--max_api_request_attempts={int(limit)}."
                )
            self.request_attempts += 1

    def reserve_mc_samples(self, count: int) -> None:
        count = int(count)
        if count < 0:
            raise ValueError("MC sample reservations must be non-negative.")
        with self._lock:
            limit = getattr(self.args, "max_mc_requested_samples", None)
            if (
                limit is not None
                and self.mc_requested_samples + count > int(limit)
            ):
                raise FatalOpenRouterResponseError(
                    "The next MC request would exceed the hard sample cap "
                    f"--max_mc_requested_samples={int(limit)} "
                    f"({self.mc_requested_samples}+{count})."
                )
            self.mc_requested_samples += count

    def record_empty_length_retry(
        self,
        *,
        attempts: int = 0,
        recoveries: int = 0,
        exhaustions: int = 0,
    ) -> None:
        with self._lock:
            self.empty_length_retry_attempts += int(attempts)
            self.empty_length_retry_recoveries += int(recoveries)
            self.empty_length_retry_exhaustions += int(exhaustions)

    def record_reasoning_retry(
        self,
        *,
        attempts: int = 0,
        recoveries: int = 0,
        exhaustions: int = 0,
    ) -> None:
        with self._lock:
            self.reasoning_retry_attempts += int(attempts)
            self.reasoning_retry_recoveries += int(recoveries)
            self.reasoning_retry_exhaustions += int(exhaustions)

    def record_reasoning_fallback(
        self,
        *,
        activations: int = 0,
        recoveries: int = 0,
        exhaustions: int = 0,
    ) -> None:
        with self._lock:
            self.reasoning_fallback_activations += int(activations)
            self.reasoning_fallback_recoveries += int(recoveries)
            self.reasoning_fallback_exhaustions += int(exhaustions)

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> OpenRouterResponse:
        return self.generate_many(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
        )[0]

    def generate_many(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int = 1,
    ) -> list[OpenRouterResponse]:
        if n <= 0:
            raise ValueError("n must be positive.")
        initial_temperature = float(temperature)
        payload = {
            "model": self.args.model,
            "messages": messages,
            "temperature": initial_temperature,
            "max_tokens": max_tokens,
        }
        if not bool(getattr(self.args, "omit_top_p", False)):
            payload["top_p"] = top_p
        thinking_budget = getattr(self.args, "thinking_budget", None)
        if thinking_budget is not None:
            payload["thinking_config"] = {
                "thinking_budget": int(thinking_budget),
            }
        if n > 1:
            payload["n"] = n
        primary_reasoning_mode = str(
            getattr(self.args, "reasoning_mode", "enabled_false")
        )
        active_reasoning_mode = primary_reasoning_mode
        apply_reasoning_mode(payload, primary_reasoning_mode)
        apply_provider_routing(payload, self.args)
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.args.site_url,
            "X-Title": self.args.app_name,
        }
        if getattr(self.args, "router_metadata", False):
            headers["X-OpenRouter-Metadata"] = "enabled"
        if getattr(self.args, "disable_openrouter_response_cache", False):
            headers["X-OpenRouter-Cache"] = "false"

        last_error = None
        network_retry_attempts = 0
        reasoning_retry_attempts = 0
        reasoning_fallback_retry_attempts = 0
        using_reasoning_fallback = False
        max_reasoning_retries = int(
            getattr(self.args, "max_reasoning_retries", 0)
        )
        reasoning_retry_sleep = float(
            getattr(self.args, "reasoning_retry_sleep", 0.0)
        )
        configured_fallback_temperature = getattr(
            self.args,
            "reasoning_fallback_temperature",
            None,
        )
        fallback_temperature = (
            float(configured_fallback_temperature)
            if configured_fallback_temperature is not None
            else None
        )
        max_reasoning_fallback_retries = int(
            getattr(self.args, "max_reasoning_fallback_retries", 0)
        )
        configured_fallback_mode = getattr(
            self.args,
            "reasoning_fallback_mode",
            None,
        )
        fallback_mode = (
            str(configured_fallback_mode)
            if configured_fallback_mode is not None
            else None
        )
        if fallback_temperature is not None and fallback_mode is not None:
            raise ValueError(
                "reasoning_fallback_temperature and reasoning_fallback_mode "
                "are mutually exclusive."
            )
        if fallback_mode is not None:
            if fallback_mode != "effort_none":
                raise ValueError(
                    "reasoning_fallback_mode currently supports only effort_none."
                )
            if primary_reasoning_mode != "enabled_false":
                raise ValueError(
                    "reasoning_fallback_mode=effort_none requires the primary "
                    "reasoning_mode=enabled_false."
                )
        reasoning_fallback_enabled = bool(
            getattr(self.args, "reject_reasoning_tokens", False)
            and max_reasoning_retries > 0
            and initial_temperature == 0.0
            and (
                (fallback_temperature is not None and fallback_temperature > 0.0)
                or fallback_mode is not None
            )
        )
        while True:
            self.reserve_request_attempt()
            request = urllib.request.Request(
                self.args.api_url,
                data=data,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.args.request_timeout) as response:
                    response_headers = getattr(response, "headers", None)
                    cache_status = (
                        response_headers.get("X-OpenRouter-Cache-Status")
                        if response_headers is not None
                        and hasattr(response_headers, "get")
                        else None
                    )
                    parsed = json.loads(response.read().decode("utf-8"))
                normalized_cache_status = (
                    str(cache_status).strip().upper() if cache_status else None
                )
                top_level_error = parsed.get("error")
                if top_level_error is not None:
                    encoded_error = json.dumps(
                        top_level_error,
                        sort_keys=True,
                        ensure_ascii=False,
                        default=str,
                    )
                    error_code = (
                        top_level_error.get("code")
                        if isinstance(top_level_error, dict)
                        else None
                    )
                    try:
                        error_code = int(error_code)
                    except (TypeError, ValueError):
                        error_code = None
                    if error_code in RETRYABLE_OPENROUTER_STATUS_CODES:
                        raise RuntimeError(
                            "OpenRouter returned a retryable top-level error in an "
                            f"HTTP success response: {encoded_error[:500]}"
                        )
                    raise FatalOpenRouterResponseError(
                        "OpenRouter returned a top-level error in an HTTP success response: "
                        f"{encoded_error[:500]}"
                    )
                usage = parsed.get("usage") or {}
                completion_token_details = (
                    usage.get("completion_tokens_details")
                    if isinstance(usage, dict)
                    else None
                )
                if not isinstance(completion_token_details, dict):
                    completion_token_details = {}
                try:
                    response_reasoning_tokens = int(
                        completion_token_details.get("reasoning_tokens") or 0
                    )
                except (TypeError, ValueError):
                    response_reasoning_tokens = 0
                response_model = parsed.get("model")
                generation_id = parsed.get("id")
                routing_metadata = parsed.get("openrouter_metadata")
                choices = parsed.get("choices", [{}])
                reasoning_message_choices = 0
                for choice in choices:
                    message = choice.get("message") or {}
                    if any(
                        message.get(field) is not None
                        for field in (
                            "reasoning",
                            "reasoning_content",
                            "reasoning_details",
                        )
                    ) or bool(message.get("thinking_blocks")):
                        reasoning_message_choices += 1
                selected_provider = (
                    selected_provider_from_metadata(routing_metadata)
                    if getattr(self.args, "router_metadata", False)
                    else None
                )
                with self._lock:
                    self.calls += 1
                    self.total_cost += float(usage.get("cost") or 0.0)
                    self.total_tokens += int(usage.get("total_tokens") or 0)
                    if response_model:
                        self.response_model_call_counts[str(response_model)] += 1
                    self.response_cache_status_counts[
                        normalized_cache_status or "<missing>"
                    ] += 1
                    self.api_max_tokens_call_counts[int(max_tokens)] += 1
                    self.reasoning_tokens += response_reasoning_tokens
                    self.reasoning_message_choices += reasoning_message_choices
                    self.reasoning_response_calls += int(
                        response_reasoning_tokens > 0
                        or reasoning_message_choices > 0
                    )
                    if using_reasoning_fallback:
                        self.reasoning_fallback_response_calls += 1
                        if response_model:
                            self.reasoning_fallback_response_model_call_counts[
                                str(response_model)
                            ] += 1
                        if selected_provider:
                            self.reasoning_fallback_provider_call_counts[
                                selected_provider
                            ] += 1
                    if getattr(self.args, "router_metadata", False):
                        if selected_provider:
                            self.provider_call_counts[selected_provider] += 1
                        else:
                            self.missing_router_metadata_calls += 1
                        routing_attempt = (
                            routing_metadata.get("attempt")
                            if isinstance(routing_metadata, dict)
                            else None
                        )
                        if isinstance(routing_attempt, int):
                            self.max_routing_attempt = max(
                                self.max_routing_attempt, routing_attempt
                            )
                if (
                    getattr(self.args, "disable_openrouter_response_cache", False)
                    and normalized_cache_status == "HIT"
                ):
                    raise FatalOpenRouterResponseError(
                        "OpenRouter reported a response-cache HIT even though "
                        "X-OpenRouter-Cache: false was requested."
                    )
                responses: list[OpenRouterResponse] = []
                for choice in choices:
                    message = choice.get("message") or {}
                    message_reasoning = message.get("reasoning")
                    if message_reasoning is None:
                        message_reasoning = message.get("reasoning_content")
                    message_reasoning_details = message.get("reasoning_details")
                    if message_reasoning_details is None and message.get(
                        "thinking_blocks"
                    ):
                        message_reasoning_details = message["thinking_blocks"]
                    responses.append(
                        OpenRouterResponse(
                            content=message.get("content") or "",
                            finish_reason=choice.get("finish_reason"),
                            native_finish_reason=choice.get("native_finish_reason"),
                            usage=usage,
                            reasoning=message_reasoning,
                            reasoning_details=message_reasoning_details,
                            reasoning_tokens=response_reasoning_tokens,
                            model=response_model,
                            generation_id=generation_id,
                            routing_metadata=routing_metadata,
                            choice_error=choice.get("error"),
                            response_cache_status=normalized_cache_status,
                            request_max_tokens=int(max_tokens),
                            request_temperature=float(payload["temperature"]),
                            request_reasoning_mode=active_reasoning_mode,
                            reasoning_fallback_used=using_reasoning_fallback,
                        )
                    )
                responses = responses or [
                    OpenRouterResponse(
                        content="",
                        finish_reason=None,
                        native_finish_reason=None,
                        usage=usage,
                        reasoning_tokens=response_reasoning_tokens,
                        model=response_model,
                        generation_id=generation_id,
                        routing_metadata=routing_metadata,
                        response_cache_status=normalized_cache_status,
                        request_max_tokens=int(max_tokens),
                        request_temperature=float(payload["temperature"]),
                        request_reasoning_mode=active_reasoning_mode,
                        reasoning_fallback_used=using_reasoning_fallback,
                    )
                ]
                reasoning_detected = any(
                    response.reasoning is not None
                    or response.reasoning_details is not None
                    or int(response.reasoning_tokens or 0) > 0
                    for response in responses
                )
                if (
                    bool(getattr(self.args, "reject_reasoning_tokens", False))
                    and reasoning_detected
                ):
                    if not using_reasoning_fallback and max_reasoning_retries > 0:
                        if reasoning_retry_attempts < max_reasoning_retries:
                            reasoning_retry_attempts += 1
                            self.record_reasoning_retry(attempts=1)
                            if reasoning_retry_sleep > 0:
                                time.sleep(reasoning_retry_sleep)
                            continue
                        if reasoning_fallback_enabled:
                            # The exhausted response is still discarded. Change only
                            # the configured request-level fallback field, preserving
                            # messages, prefix, routing, cache, token budget, and strict
                            # downstream reasoning validation.
                            using_reasoning_fallback = True
                            if fallback_temperature is not None:
                                payload["temperature"] = fallback_temperature
                            else:
                                assert fallback_mode is not None
                                active_reasoning_mode = fallback_mode
                                apply_reasoning_mode(payload, fallback_mode)
                            data = json.dumps(payload).encode("utf-8")
                            self.record_reasoning_retry(attempts=1)
                            self.record_reasoning_fallback(activations=1)
                            if reasoning_retry_sleep > 0:
                                time.sleep(reasoning_retry_sleep)
                            continue
                        self.record_reasoning_retry(exhaustions=1)
                    elif using_reasoning_fallback:
                        if (
                            reasoning_fallback_retry_attempts
                            < max_reasoning_fallback_retries
                        ):
                            reasoning_fallback_retry_attempts += 1
                            self.record_reasoning_retry(attempts=1)
                            if reasoning_retry_sleep > 0:
                                time.sleep(reasoning_retry_sleep)
                            continue
                        self.record_reasoning_retry(exhaustions=1)
                        self.record_reasoning_fallback(exhaustions=1)
                else:
                    if reasoning_retry_attempts > 0 or using_reasoning_fallback:
                        self.record_reasoning_retry(recoveries=1)
                    if using_reasoning_fallback:
                        self.record_reasoning_fallback(recoveries=1)
                return responses
            except FatalOpenRouterResponseError:
                raise
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTP {exc.code}: {body}"
                if exc.code not in RETRYABLE_OPENROUTER_STATUS_CODES:
                    raise FatalOpenRouterResponseError(
                        "OpenRouter returned a non-retryable HTTP response: "
                        f"{last_error}"
                    ) from exc
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if network_retry_attempts < self.args.max_retries:
                network_retry_attempts += 1
                time.sleep(self.args.retry_sleep * network_retry_attempts)
                continue
            break

        raise RuntimeError(f"OpenRouter request failed after retries: {last_error}")


def load_risk_gate(args: argparse.Namespace):
    if args.risk_gate_checkpoint is None:
        return None
    if args.risk_gate_batch_size <= 0:
        raise ValueError("--risk_gate_batch_size must be positive.")

    from risk_gate import PrefixRiskGate

    device = torch.device(
        args.risk_gate_device
        if args.risk_gate_device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return PrefixRiskGate(
        checkpoint=args.risk_gate_checkpoint,
        device=device,
        threshold=args.risk_gate_threshold,
        top_k=1,
        batch_size=args.risk_gate_batch_size,
        max_length=args.risk_gate_max_length,
        dtype=args.risk_gate_dtype,
        model_name=args.risk_gate_model_name,
        load_in_4bit=args.risk_gate_load_in_4bit,
        trust_remote_code=args.risk_gate_trust_remote_code,
        local_files_only=args.risk_gate_local_files_only,
    )


def messages_for_prefix(
    question: str,
    prefix_text: str,
    *,
    qwen_hard_no_think_prefill: bool = False,
) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": question}]
    if qwen_hard_no_think_prefill:
        messages.append(
            {
                "role": "assistant",
                "content": QWEN_HARD_NO_THINK_PREFILL + prefix_text,
            }
        )
    elif prefix_text:
        messages.append({"role": "assistant", "content": prefix_text})
    return messages


def sampling_messages_for_prefix(
    args: argparse.Namespace,
    question: str,
    prefix_text: str,
) -> list[dict[str, str]]:
    return messages_for_prefix(
        question,
        prefix_text,
        qwen_hard_no_think_prefill=bool(
            getattr(args, "qwen_hard_no_think_prefill", False)
        ),
    )


def is_claude_45_model(model_name: Optional[str]) -> bool:
    """Return whether a routed/requested model identifier names Claude 4.5."""

    if not model_name:
        return False
    normalized = str(model_name).strip().casefold().replace("_", "-")
    normalized = normalized.replace(".", "-")
    return "claude" in normalized and "4-5" in normalized


def _trailing_whitespace_overlap_chars(content: str, prefix_text: str) -> int:
    """Find replayed whitespace shared by the prefix tail and response head."""

    if not content or not prefix_text or not prefix_text[-1].isspace():
        return 0
    whitespace_start = len(prefix_text)
    while whitespace_start > 0 and prefix_text[whitespace_start - 1].isspace():
        whitespace_start -= 1
    whitespace_tail = prefix_text[whitespace_start:]
    maximum = min(len(whitespace_tail), len(content))
    for overlap in range(maximum, 0, -1):
        if content.startswith(whitespace_tail[-overlap:]):
            return overlap
    return 0


def strip_prefill_with_audit(
    content: str,
    prefix_text: str,
    *,
    model_name: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Remove provider-prefill replay while preserving an auditable decision."""

    if prefix_text and content.startswith(prefix_text):
        return content[len(prefix_text) :], {
            "prefill_exact_echo_chars": len(prefix_text),
            "claude_45_whitespace_overlap_chars": 0,
            "claude_45_whitespace_overlap_applied": False,
        }

    overlap = 0
    if is_claude_45_model(model_name):
        # Claude 4.5 can normalize trailing assistant-prefill whitespace and
        # then emit that whitespace again.  Treat only the longest matching
        # suffix/head overlap as replay; non-Claude models retain legacy
        # behavior, and non-whitespace continuation text is never removed.
        overlap = _trailing_whitespace_overlap_chars(content, prefix_text)
    return content[overlap:], {
        "prefill_exact_echo_chars": 0,
        "claude_45_whitespace_overlap_chars": overlap,
        "claude_45_whitespace_overlap_applied": overlap > 0,
    }


def strip_prefill(
    content: str,
    prefix_text: str,
    *,
    model_name: Optional[str] = None,
) -> str:
    stripped, _audit = strip_prefill_with_audit(
        content,
        prefix_text,
        model_name=model_name,
    )
    return stripped


def content_to_token_id(
    tokenizer,
    content: str,
    empty_response_token: str = "skip",
) -> tuple[Optional[int], int]:
    if not content:
        if empty_response_token == "eos" and tokenizer.eos_token_id is not None:
            return int(tokenizer.eos_token_id), 1
        return None, 0
    token_ids = tokenizer.encode(content, add_special_tokens=False)
    if not token_ids:
        return None, 0
    return int(token_ids[0]), len(token_ids)


def empty_length_retry_budgets(
    initial_max_tokens: int,
    retry_max_tokens: Optional[int],
    max_retry_rounds: int,
) -> list[int]:
    """Return monotonically non-decreasing retry budgets up to the configured cap."""

    initial_max_tokens = int(initial_max_tokens)
    max_retry_rounds = int(max_retry_rounds)
    if initial_max_tokens <= 0:
        raise ValueError("initial_max_tokens must be positive.")
    if max_retry_rounds < 0:
        raise ValueError("max_retry_rounds must be non-negative.")
    if max_retry_rounds == 0 or retry_max_tokens is None:
        return []

    retry_max_tokens = int(retry_max_tokens)
    if retry_max_tokens <= initial_max_tokens:
        raise ValueError(
            "empty_length_retry_max_tokens must exceed the initial max_tokens "
            "when adaptive retries are enabled."
        )

    budgets: list[int] = []
    current = initial_max_tokens
    for _ in range(max_retry_rounds):
        current = min(retry_max_tokens, max(current + 1, current * 2))
        budgets.append(current)
    return budgets


def record_client_empty_length_retry(
    client,
    *,
    attempts: int = 0,
    recoveries: int = 0,
    exhaustions: int = 0,
) -> None:
    recorder = getattr(client, "record_empty_length_retry", None)
    if callable(recorder):
        recorder(
            attempts=attempts,
            recoveries=recoveries,
            exhaustions=exhaustions,
        )


def query_next_token_id(
    client: OpenRouterClient,
    tokenizer,
    question: str,
    prefix_text: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> tuple[Optional[int], dict]:
    fallback_activations_before = int(
        getattr(client, "reasoning_fallback_activations", 0)
    )
    messages = sampling_messages_for_prefix(client.args, question, prefix_text)
    empty_response_token = getattr(client.args, "empty_response_token", "skip")
    retry_budgets = empty_length_retry_budgets(
        initial_max_tokens=max_tokens,
        retry_max_tokens=getattr(
            client.args, "empty_length_retry_max_tokens", None
        ),
        max_retry_rounds=int(
            getattr(client.args, "max_empty_length_retry_rounds", 0)
        ),
    )
    attempted_budgets = [int(max_tokens)]
    response = client.generate(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    token_id, info = response_to_token_id(
        tokenizer,
        response,
        prefix_text,
        empty_response_token,
        reject_reasoning_tokens=bool(
            getattr(client.args, "reject_reasoning_tokens", False)
        ),
        requested_model=getattr(client.args, "model", None),
    )
    info["request_max_tokens"] = int(max_tokens)
    empty_length_samples = int(bool(info["retryable_empty_length"]))
    retry_attempts = 0

    for retry_max_tokens in retry_budgets:
        if not info["retryable_empty_length"]:
            break
        retry_attempts += 1
        attempted_budgets.append(int(retry_max_tokens))
        record_client_empty_length_retry(client, attempts=1)
        response = client.generate(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=retry_max_tokens,
        )
        token_id, info = response_to_token_id(
            tokenizer,
            response,
            prefix_text,
            empty_response_token,
            reject_reasoning_tokens=bool(
                getattr(client.args, "reject_reasoning_tokens", False)
            ),
            requested_model=getattr(client.args, "model", None),
        )
        info["request_max_tokens"] = int(retry_max_tokens)
        empty_length_samples += int(bool(info["retryable_empty_length"]))

    recovered = retry_attempts > 0 and token_id is not None
    exhausted = retry_attempts > 0 and token_id is None
    if recovered:
        record_client_empty_length_retry(client, recoveries=1)
    if exhausted:
        record_client_empty_length_retry(client, exhaustions=1)
    reasoning_fallback_result_used = bool(
        info.get("reasoning_fallback_used", False)
    )
    info.update(
        {
            "empty_length_samples": empty_length_samples,
            "empty_length_retry_attempts": retry_attempts,
            "empty_length_retry_recovered": recovered,
            "empty_length_retry_exhausted": exhausted,
            "max_tokens_attempted": attempted_budgets,
            "reasoning_fallback_activations": max(
                0,
                int(getattr(client, "reasoning_fallback_activations", 0))
                - fallback_activations_before,
            ),
            "reasoning_fallback_result_used": reasoning_fallback_result_used,
        }
    )
    info["reasoning_fallback_used"] = bool(
        info.get("reasoning_fallback_used")
        or info["reasoning_fallback_activations"]
    )
    return token_id, info


def response_to_token_id(
    tokenizer,
    response: OpenRouterResponse,
    prefix_text: str,
    empty_response_token: str = "skip",
    *,
    reject_reasoning_tokens: bool = False,
    requested_model: Optional[str] = None,
) -> tuple[Optional[int], dict]:
    reasoning_detected = bool(
        response.reasoning is not None
        or response.reasoning_details is not None
        or int(response.reasoning_tokens or 0) > 0
    )
    if reject_reasoning_tokens and reasoning_detected:
        raise FatalOpenRouterResponseError(
            "OpenRouter returned reasoning during a visible next-token sample "
            f"(generation_id={response.generation_id!r}, "
            f"reasoning_tokens={int(response.reasoning_tokens or 0)}, "
            f"reasoning_field_present={response.reasoning is not None}, "
            "reasoning_details_present="
            f"{response.reasoning_details is not None})."
        )
    routed_or_requested_model = (
        response.model
        if is_claude_45_model(response.model)
        else requested_model
    )
    new_text, prefill_audit = strip_prefill_with_audit(
        response.content,
        prefix_text,
        model_name=routed_or_requested_model,
    )
    normalized_finish_reason = (
        str(response.finish_reason).strip().casefold()
        if response.finish_reason is not None
        else None
    )
    normalized_native_finish_reason = (
        str(response.native_finish_reason).strip().casefold()
        if response.native_finish_reason is not None
        else None
    )
    if response.choice_error is not None:
        encoded_error = json.dumps(
            response.choice_error,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        raise FatalOpenRouterResponseError(
            "OpenRouter returned a choice-level error"
            f" (generation_id={response.generation_id!r}, "
            f"finish_reason={response.finish_reason!r}): {encoded_error[:500]}"
        )
    if (
        normalized_finish_reason in {"content_filter", "error"}
        or normalized_native_finish_reason in {"content_filter", "error"}
    ):
        raise FatalOpenRouterResponseError(
            "OpenRouter returned a fatal sampled choice"
            f" (generation_id={response.generation_id!r}, "
            f"finish_reason={response.finish_reason!r}, "
            f"native_finish_reason={response.native_finish_reason!r})."
        )

    canonical_eos_projected = False
    canonical_eos_projection_reason = None
    if (
        not new_text
        and empty_response_token == "stop_eos"
        and normalized_finish_reason == "stop"
    ):
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            raise FatalOpenRouterResponseError(
                "--empty_response_token=stop_eos requires tokenizer.eos_token_id "
                "to project an empty stop choice to canonical EOS."
            )
        token_id, token_count = int(eos_token_id), 1
        canonical_eos_projected = True
        canonical_eos_projection_reason = "stop"
    else:
        token_id, token_count = content_to_token_id(
            tokenizer,
            new_text,
            empty_response_token,
        )
        if not new_text and token_id is not None and empty_response_token == "eos":
            canonical_eos_projected = True
            canonical_eos_projection_reason = "legacy_eos"

    retryable_empty_length = (
        token_id is None
        and not new_text
        and (
            normalized_finish_reason == "length"
            or (
                normalized_finish_reason is None
                and normalized_native_finish_reason == "length"
            )
        )
    )
    return token_id, {
        # Store the continuation after removing any echoed assistant prefill.
        # This is the tokenizer-independent source of truth for rematerializing
        # the sampled distribution under a different proxy tokenizer.  An empty
        # string is retained only when it was accepted as canonical EOS.
        "completion_text": new_text,
        # Preserve the provider's exact message content as well.  For the usual
        # max_tokens=1 request this is the original returned native-token text,
        # before Claude-specific whitespace de-echoing or proxy tokenization.
        "raw_completion_text": response.content,
        **prefill_audit,
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "content_chars": len(response.content),
        "new_text_chars": len(new_text),
        "mapped_token_count": token_count,
        "canonical_eos_projected": canonical_eos_projected,
        "canonical_eos_projection_reason": canonical_eos_projection_reason,
        "retryable_empty_length": retryable_empty_length,
        "choice_error": response.choice_error,
        "reasoning_detected": reasoning_detected,
        "reasoning_tokens": int(response.reasoning_tokens or 0),
        "reasoning_chars": (
            len(response.reasoning)
            if isinstance(response.reasoning, str)
            else 0
        ),
        "reasoning_detail_count": (
            len(response.reasoning_details)
            if isinstance(response.reasoning_details, list)
            else int(response.reasoning_details is not None)
        ),
        "usage": response.usage,
        "response_model": response.model,
        "generation_id": response.generation_id,
        "response_cache_status": response.response_cache_status,
        "request_max_tokens": response.request_max_tokens,
        "request_temperature": response.request_temperature,
        "request_reasoning_mode": response.request_reasoning_mode,
        "reasoning_fallback_used": bool(response.reasoning_fallback_used),
        "selected_provider": selected_provider_from_metadata(response.routing_metadata),
        "routing_attempt": (
            response.routing_metadata.get("attempt")
            if isinstance(response.routing_metadata, dict)
            else None
        ),
    }


def sample_position_token_ids(
    client: OpenRouterClient,
    tokenizer,
    question: str,
    prefix_text: str,
    args: argparse.Namespace,
) -> tuple[list[int], dict]:
    sample_ids: list[int] = []
    sample_texts: list[str] = []
    sample_raw_texts: list[str] = []
    empty = 0
    raw_empty = 0
    invalid_empty = 0
    unmappable_nonempty = 0
    canonical_eos_projections = 0
    claude_45_whitespace_overlap_choices = 0
    claude_45_whitespace_overlap_chars = 0
    multi_token = 0
    failures = 0
    missing_choices = 0
    request_failures = 0
    requested_samples = 0
    returned_choices = 0
    refill_rounds = 0
    empty_length_samples = 0
    empty_length_retry_requests = 0
    empty_length_retry_recoveries = 0
    empty_length_retry_rounds = 0
    empty_length_retry_exhausted = False
    claude_45_empty_refill_bypass = False
    reasoning_choices = 0
    reasoning_tokens = 0
    used_single_choice_fallback = False
    provider_counts: Counter[str] = Counter()
    response_model_counts: Counter[str] = Counter()
    finish_reason_counts: Counter[str] = Counter()
    native_finish_reason_counts: Counter[str] = Counter()
    canonical_eos_projection_reason_counts: Counter[str] = Counter()
    response_cache_status_counts: Counter[str] = Counter()
    sample_max_tokens_choice_counts: Counter[int] = Counter()
    request_error_counts: Counter[str] = Counter()
    missing_router_metadata_choices = 0
    max_routing_attempt = 0
    choices_per_request = max(1, int(getattr(args, "sample_choices_per_request", 1)))
    completion_policy = getattr(args, "sample_completion_policy", "partial")
    max_refill_rounds = int(getattr(args, "max_sample_refill_rounds", 2))
    if completion_policy not in {"partial", "exact"}:
        raise ValueError("sample_completion_policy must be 'partial' or 'exact'.")
    if max_refill_rounds < 0:
        raise ValueError("max_sample_refill_rounds must be non-negative.")
    retry_budgets = empty_length_retry_budgets(
        initial_max_tokens=args.api_max_tokens,
        retry_max_tokens=getattr(args, "empty_length_retry_max_tokens", None),
        max_retry_rounds=int(
            getattr(args, "max_empty_length_retry_rounds", 0)
        ),
    )

    def one_request(count: int, request_max_tokens: int, is_empty_length_retry: bool):
        if is_empty_length_retry:
            record_client_empty_length_retry(client, attempts=1)
        responses = client.generate_many(
            messages=sampling_messages_for_prefix(args, question, prefix_text),
            temperature=args.sample_temperature,
            top_p=args.top_p,
            max_tokens=request_max_tokens,
            n=count,
        )
        empty_response_token = getattr(args, "empty_response_token", "skip")
        results = []
        for response in responses:
            token_id, info = response_to_token_id(
                tokenizer,
                response,
                prefix_text,
                empty_response_token,
                reject_reasoning_tokens=bool(
                    getattr(args, "reject_reasoning_tokens", False)
                ),
                requested_model=getattr(
                    getattr(client, "args", None),
                    "model",
                    None,
                ),
            )
            info["request_max_tokens"] = int(request_max_tokens)
            results.append((token_id, info))
        return results

    def consume_results(
        size: int,
        results: list[tuple[Optional[int], dict]],
        *,
        is_empty_length_retry: bool,
    ) -> None:
        nonlocal empty, failures, max_routing_attempt, missing_choices
        nonlocal raw_empty, invalid_empty, unmappable_nonempty
        nonlocal canonical_eos_projections
        nonlocal claude_45_whitespace_overlap_choices
        nonlocal claude_45_whitespace_overlap_chars
        nonlocal missing_router_metadata_choices, multi_token, returned_choices
        nonlocal empty_length_samples, empty_length_retry_recoveries
        nonlocal reasoning_choices, reasoning_tokens
        if len(results) > size:
            raise FatalOpenRouterResponseError(
                f"OpenRouter returned {len(results)} choices for a request of n={size}."
            )
        returned_choices += len(results)
        if len(results) < size:
            missing = size - len(results)
            failures += missing
            missing_choices += missing
        for token_id, info in results[:size]:
            overlap_chars = int(
                info.get("claude_45_whitespace_overlap_chars") or 0
            )
            if overlap_chars > 0:
                claude_45_whitespace_overlap_choices += 1
                claude_45_whitespace_overlap_chars += overlap_chars
            if info.get("reasoning_detected"):
                reasoning_choices += 1
            reasoning_tokens += int(info.get("reasoning_tokens") or 0)
            request_max_tokens = info.get("request_max_tokens")
            if isinstance(request_max_tokens, int):
                sample_max_tokens_choice_counts[request_max_tokens] += 1
            finish_reason_counts[
                str(info["finish_reason"])
                if info.get("finish_reason") is not None
                else "<missing>"
            ] += 1
            native_finish_reason_counts[
                str(info["native_finish_reason"])
                if info.get("native_finish_reason") is not None
                else "<missing>"
            ] += 1
            response_cache_status_counts[
                str(info["response_cache_status"])
                if info.get("response_cache_status") is not None
                else "<missing>"
            ] += 1
            is_raw_empty = info.get("new_text_chars") == 0
            if is_raw_empty:
                raw_empty += 1
            if info.get("retryable_empty_length"):
                empty_length_samples += 1
            if info.get("canonical_eos_projected"):
                canonical_eos_projections += 1
                canonical_eos_projection_reason_counts[
                    str(info.get("canonical_eos_projection_reason") or "<missing>")
                ] += 1
            provider = info.get("selected_provider")
            if provider:
                provider_counts[str(provider)] += 1
            else:
                missing_router_metadata_choices += 1
            response_model = info.get("response_model")
            if response_model:
                response_model_counts[str(response_model)] += 1
            routing_attempt = info.get("routing_attempt")
            if isinstance(routing_attempt, int):
                max_routing_attempt = max(max_routing_attempt, routing_attempt)
            if token_id is None:
                empty += 1
                if is_raw_empty:
                    invalid_empty += 1
                else:
                    unmappable_nonempty += 1
            else:
                sample_ids.append(token_id)
                sample_texts.append(str(info.get("completion_text", "")))
                sample_raw_texts.append(str(info.get("raw_completion_text", "")))
                if is_empty_length_retry:
                    empty_length_retry_recoveries += 1
                    record_client_empty_length_retry(client, recoveries=1)
                if info["mapped_token_count"] > 1:
                    multi_token += 1

    def run_requests(
        request_sizes: list[int],
        *,
        request_max_tokens: int,
        is_empty_length_retry: bool = False,
    ) -> None:
        nonlocal failures, request_failures, requested_samples
        nonlocal empty_length_retry_requests
        batch_requested_samples = sum(request_sizes)
        if hasattr(client, "reserve_mc_samples"):
            client.reserve_mc_samples(batch_requested_samples)
        requested_samples += batch_requested_samples
        if is_empty_length_retry:
            empty_length_retry_requests += sum(request_sizes)
        if args.parallel_requests <= 1:
            for size in request_sizes:
                try:
                    results = one_request(
                        size, request_max_tokens, is_empty_length_retry
                    )
                except FatalOpenRouterResponseError:
                    raise
                except Exception as exc:
                    failures += size
                    request_failures += size
                    request_error_counts[
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    ] += size
                    continue
                consume_results(
                    size,
                    results,
                    is_empty_length_retry=is_empty_length_retry,
                )
                if args.delay_seconds > 0:
                    time.sleep(args.delay_seconds)
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_requests) as executor:
            futures = {
                executor.submit(
                    one_request,
                    size,
                    request_max_tokens,
                    is_empty_length_retry,
                ): size
                for size in request_sizes
            }
            for future in concurrent.futures.as_completed(futures):
                size = futures[future]
                try:
                    results = future.result()
                except FatalOpenRouterResponseError:
                    for pending in futures:
                        pending.cancel()
                    raise
                except Exception as exc:
                    failures += size
                    request_failures += size
                    request_error_counts[
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    ] += size
                    continue
                consume_results(
                    size,
                    results,
                    is_empty_length_retry=is_empty_length_retry,
                )

    request_sizes: list[int] = []
    remaining = args.samples_per_token
    while remaining > 0:
        size = min(choices_per_request, remaining)
        request_sizes.append(size)
        remaining -= size
    run_requests(request_sizes, request_max_tokens=args.api_max_tokens)

    if completion_policy == "exact":
        while len(sample_ids) < args.samples_per_token and refill_rounds < max_refill_rounds:
            # A one-native-token Claude 4.5 response can consist entirely of
            # replayed trailing assistant-prefill whitespace. Repeating the
            # same max_tokens=1 request cannot expose the continuation behind
            # that whitespace and previously multiplied every affected MC
            # step by max_sample_refill_rounds. Once this exact failure mode is
            # observed, move directly to the adaptive 2/4/8-token budgets.
            # Other models and non-empty request failures retain the ordinary
            # same-length refill policy.
            if (
                is_claude_45_model(
                    getattr(getattr(client, "args", None), "model", None)
                )
                and empty_length_samples > 0
            ):
                claude_45_empty_refill_bypass = True
                break
            refill_rounds += 1
            used_single_choice_fallback = used_single_choice_fallback or choices_per_request > 1
            missing = args.samples_per_token - len(sample_ids)
            run_requests([1] * missing, request_max_tokens=args.api_max_tokens)

        if len(sample_ids) < args.samples_per_token and empty_length_samples:
            for retry_max_tokens in retry_budgets:
                if len(sample_ids) >= args.samples_per_token:
                    break
                empty_length_retry_rounds += 1
                used_single_choice_fallback = (
                    used_single_choice_fallback or choices_per_request > 1
                )
                missing = args.samples_per_token - len(sample_ids)
                run_requests(
                    [1] * missing,
                    request_max_tokens=retry_max_tokens,
                    is_empty_length_retry=True,
                )

        if (
            empty_length_retry_rounds > 0
            and len(sample_ids) < args.samples_per_token
        ):
            empty_length_retry_exhausted = True
            record_client_empty_length_retry(client, exhaustions=1)

        if len(sample_ids) < args.samples_per_token:
            raise RuntimeError(
                f"Expected exactly {args.samples_per_token} valid next-token samples, "
                f"but collected {len(sample_ids)} after {refill_rounds} refill rounds "
                f"({empty} unmapped, {raw_empty} raw empty, "
                f"{canonical_eos_projections} canonical EOS projections, "
                f"{invalid_empty} invalid empty, {unmappable_nonempty} unmappable non-empty, "
                f"{missing_choices} missing choices, {request_failures} request failures; "
                f"{empty_length_samples} empty-length choices, "
                f"{empty_length_retry_requests} adaptive retry requests, "
                f"{empty_length_retry_recoveries} adaptive retry recoveries, "
                f"{empty_length_retry_rounds} adaptive retry rounds; "
                f"finish_reasons={dict(sorted(finish_reason_counts.items()))}, "
                f"native_finish_reasons={dict(sorted(native_finish_reason_counts.items()))}, "
                "canonical_eos_reasons="
                f"{dict(sorted(canonical_eos_projection_reason_counts.items()))}, "
                "response_cache_statuses="
                f"{dict(sorted(response_cache_status_counts.items()))}, "
                f"request_errors={dict(sorted(request_error_counts.items()))})."
            )

    sample_ids = sample_ids[: args.samples_per_token]
    sample_texts = sample_texts[: args.samples_per_token]
    sample_raw_texts = sample_raw_texts[: args.samples_per_token]
    if not len(sample_raw_texts) == len(sample_texts) == len(sample_ids):
        raise RuntimeError(
            "Sample token IDs and raw/normalized completion texts became misaligned."
        )

    return sample_ids, {
        "sampled_completion_text_counts": dict(
            sorted(Counter(sample_texts).items())
        ),
        "sampled_raw_completion_text_counts": dict(
            sorted(Counter(sample_raw_texts).items())
        ),
        "valid_samples": len(sample_ids),
        "empty_samples": empty,
        "raw_empty_samples": raw_empty,
        "invalid_empty_samples": invalid_empty,
        "unmappable_nonempty_samples": unmappable_nonempty,
        "canonical_eos_projections": canonical_eos_projections,
        "claude_45_whitespace_overlap_choices": (
            claude_45_whitespace_overlap_choices
        ),
        "claude_45_whitespace_overlap_chars": (
            claude_45_whitespace_overlap_chars
        ),
        "multi_token_samples": multi_token,
        "failed_samples": failures,
        "missing_choices": missing_choices,
        "request_failures": request_failures,
        "requested_samples": requested_samples,
        "returned_choices": returned_choices,
        "refill_rounds": refill_rounds,
        "empty_length_samples": empty_length_samples,
        "empty_length_retry_requests": empty_length_retry_requests,
        "empty_length_retry_recoveries": empty_length_retry_recoveries,
        "empty_length_retry_rounds": empty_length_retry_rounds,
        "empty_length_retry_exhausted": empty_length_retry_exhausted,
        "claude_45_empty_refill_bypass": claude_45_empty_refill_bypass,
        "reasoning_choices": reasoning_choices,
        "reasoning_tokens": reasoning_tokens,
        "used_single_choice_fallback": used_single_choice_fallback,
        "exact_samples": len(sample_ids) == args.samples_per_token,
        "actual_provider_counts": dict(provider_counts),
        "actual_response_model_counts": dict(response_model_counts),
        "finish_reason_counts": dict(finish_reason_counts),
        "native_finish_reason_counts": dict(native_finish_reason_counts),
        "canonical_eos_projection_reason_counts": dict(
            canonical_eos_projection_reason_counts
        ),
        "response_cache_status_counts": dict(response_cache_status_counts),
        "sample_max_tokens_choice_counts": {
            str(key): value
            for key, value in sorted(sample_max_tokens_choice_counts.items())
        },
        "request_error_counts": dict(request_error_counts),
        "missing_router_metadata_choices": missing_router_metadata_choices,
        "max_routing_attempt": max_routing_attempt,
    }


def ids_to_log_probs(
    sampled_token_ids: list[int],
    vocab_size: int,
    observed_alpha: float,
    floor_mass: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not sampled_token_ids:
        raise ValueError("No valid sampled token ids for this position.")
    tensor = torch.tensor([sampled_token_ids], dtype=torch.long)
    return sampled_ids_to_log_probs(
        sampled_token_ids=tensor,
        vocab_size=vocab_size,
        observed_alpha=observed_alpha,
        floor_mass=floor_mass,
        dtype=dtype,
    )[0]


def build_risk_gate_payload(
    risk_gate,
    tokenizer,
    question: str,
    answer_ids: torch.Tensor,
    base_token_ids: torch.Tensor,
) -> dict:
    if risk_gate is None:
        return {}
    if answer_ids.dim() != 2 or answer_ids.shape[0] != 1:
        raise ValueError("answer_ids must have shape [1, seq_len].")
    if base_token_ids.dim() != 1:
        raise ValueError("base_token_ids must have shape [seq_len].")

    target_ids = answer_ids[0].detach().cpu().long().tolist()
    current_ids = base_token_ids.detach().cpu().long().tolist()
    prompts = [question] * len(current_ids)
    answer_prefixes = []
    for position, token_id in enumerate(current_ids):
        prefix_ids = target_ids[:position] + [int(token_id)]
        answer_prefixes.append(tokenizer.decode(prefix_ids, skip_special_tokens=False))

    scores = risk_gate.score_prefixes(prompts, answer_prefixes)
    mask = scores < risk_gate.threshold
    return {
        "risk_gate_mask": mask.unsqueeze(0).cpu().bool(),
        "risk_gate_scores": scores.unsqueeze(0).cpu().float(),
        "risk_gate_token_ids": base_token_ids.unsqueeze(0).detach().cpu().long(),
    }


def get_sampled_logprobs_openrouter(
    client: OpenRouterClient,
    tokenizer,
    question: str,
    answer: str,
    args: argparse.Namespace,
    risk_gate=None,
) -> Optional[dict]:
    answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)
    if args.max_answer_tokens is not None:
        answer_token_ids = answer_token_ids[: args.max_answer_tokens]
    if not answer_token_ids:
        return None

    vocab_size = len(tokenizer)
    store_dtype = resolve_torch_dtype(args.store_dtype)
    if args.sample_only_risk_active:
        if risk_gate is None:
            raise ValueError("--sample_only_risk_active requires --risk_gate_checkpoint.")
        return get_risk_active_sampled_logprobs_openrouter(
            client=client,
            tokenizer=tokenizer,
            question=question,
            answer_token_ids=answer_token_ids,
            answer_text=answer,
            vocab_size=vocab_size,
            store_dtype=store_dtype,
            args=args,
            risk_gate=risk_gate,
        )

    log_prob_rows = []
    sampled_completion_text_counts = []
    sampled_raw_completion_text_counts = []
    base_token_ids = []
    valid_counts = []
    empty_counts = []
    raw_empty_counts = []
    invalid_empty_counts = []
    unmappable_nonempty_counts = []
    canonical_eos_projection_counts = []
    claude_45_whitespace_overlap_choice_counts = []
    claude_45_whitespace_overlap_char_counts = []
    multi_token_counts = []
    failed_counts = []
    missing_choice_counts = []
    request_failure_counts = []
    requested_sample_counts = []
    returned_choice_counts = []
    refill_round_counts = []
    empty_length_sample_counts = []
    empty_length_retry_request_counts = []
    empty_length_retry_recovery_counts = []
    empty_length_retry_exhausted_flags = []
    reasoning_choice_counts = []
    reasoning_token_counts = []
    single_choice_fallback_flags = []
    actual_provider_counts: Counter[str] = Counter()
    actual_response_model_counts: Counter[str] = Counter()
    finish_reason_counts: Counter[str] = Counter()
    native_finish_reason_counts: Counter[str] = Counter()
    canonical_eos_projection_reason_counts: Counter[str] = Counter()
    sample_response_cache_status_counts: Counter[str] = Counter()
    sample_max_tokens_choice_counts: Counter[str] = Counter()
    request_error_counts: Counter[str] = Counter()
    missing_router_metadata_choices = 0
    max_routing_attempt = 0
    deterministic_failures = 0
    deterministic_empty_length_retry_attempts = 0
    deterministic_empty_length_retry_recoveries = 0
    deterministic_empty_length_retry_exhaustions = 0
    calls_before = client.calls
    request_attempts_before = int(getattr(client, "request_attempts", 0))
    mc_requested_samples_before = int(getattr(client, "mc_requested_samples", 0))
    cost_before = client.total_cost
    provider_call_counts_before = Counter(client.provider_call_counts)
    response_model_call_counts_before = Counter(client.response_model_call_counts)
    response_cache_status_counts_before = Counter(client.response_cache_status_counts)
    api_max_tokens_call_counts_before = Counter(
        getattr(client, "api_max_tokens_call_counts", {})
    )
    empty_length_retry_attempts_before = int(
        getattr(client, "empty_length_retry_attempts", 0)
    )
    empty_length_retry_recoveries_before = int(
        getattr(client, "empty_length_retry_recoveries", 0)
    )
    empty_length_retry_exhaustions_before = int(
        getattr(client, "empty_length_retry_exhaustions", 0)
    )
    reasoning_response_calls_before = int(
        getattr(client, "reasoning_response_calls", 0)
    )
    reasoning_message_choices_before = int(
        getattr(client, "reasoning_message_choices", 0)
    )
    reasoning_tokens_before = int(getattr(client, "reasoning_tokens", 0))
    missing_router_metadata_calls_before = client.missing_router_metadata_calls
    prefix_texts = [
        tokenizer.decode(answer_token_ids[:position], skip_special_tokens=False)
        for position in range(len(answer_token_ids))
    ]

    def sample_one_position(position: int):
        if args.max_api_calls is not None and client.calls >= args.max_api_calls:
            raise RuntimeError(
                f"Reached --max_api_calls={args.max_api_calls}; stopping before more API spend."
            )
        prefix_text = prefix_texts[position]

        sampled_ids, sample_stats = sample_position_token_ids(
            client=client,
            tokenizer=tokenizer,
            question=question,
            prefix_text=prefix_text,
            args=args,
        )
        if not sampled_ids:
            raise RuntimeError(f"No valid sampled tokens at answer position {position}.")

        log_prob_row = ids_to_log_probs(
            sampled_token_ids=sampled_ids,
            vocab_size=vocab_size,
            observed_alpha=args.observed_alpha,
            floor_mass=args.floor_mass,
            dtype=store_dtype,
        )

        base_id = None
        deterministic_failure = 0
        deterministic_info = None
        if risk_gate is not None or bool(
            getattr(args, "store_deterministic_base_tokens", False)
        ):
            base_id, deterministic_info = query_next_token_id(
                client=client,
                tokenizer=tokenizer,
                question=question,
                prefix_text=prefix_text,
                temperature=0.0,
                top_p=1.0,
                max_tokens=args.api_max_tokens,
            )
            if base_id is None:
                deterministic_failure = 1
                if args.sample_completion_policy == "exact":
                    raise RuntimeError(
                        "Deterministic next-token query returned no valid token at "
                        f"answer position {position} after "
                        f"{int(deterministic_info.get('empty_length_retry_attempts', 0))} "
                        "adaptive empty-length retries "
                        f"(finish_reason={deterministic_info.get('finish_reason')!r}, "
                        "native_finish_reason="
                        f"{deterministic_info.get('native_finish_reason')!r})."
                    )
                base_id = sampled_ids[0]
        return (
            position,
            log_prob_row,
            sample_stats,
            base_id,
            deterministic_failure,
            deterministic_info,
        )

    rows_by_position: list[Optional[torch.Tensor]] = [None] * len(answer_token_ids)
    stats_by_position: list[Optional[dict]] = [None] * len(answer_token_ids)
    base_ids_by_position: list[Optional[int]] = [None] * len(answer_token_ids)
    deterministic_infos_by_position: list[Optional[dict]] = [None] * len(
        answer_token_ids
    )
    if args.parallel_positions <= 1:
        iterator = range(len(answer_token_ids))
        for position in tqdm(iterator, desc="API token positions", leave=False):
            (
                position,
                row,
                stats,
                base_id,
                deterministic_failure,
                deterministic_info,
            ) = sample_one_position(position)
            rows_by_position[position] = row
            stats_by_position[position] = stats
            base_ids_by_position[position] = base_id
            deterministic_infos_by_position[position] = deterministic_info
            deterministic_failures += deterministic_failure
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_positions) as executor:
            futures = {
                executor.submit(sample_one_position, position): position
                for position in range(len(answer_token_ids))
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="API token positions",
                leave=False,
            ):
                (
                    position,
                    row,
                    stats,
                    base_id,
                    deterministic_failure,
                    deterministic_info,
                ) = future.result()
                rows_by_position[position] = row
                stats_by_position[position] = stats
                base_ids_by_position[position] = base_id
                deterministic_infos_by_position[position] = deterministic_info
                deterministic_failures += deterministic_failure

    for position in range(len(answer_token_ids)):
        row = rows_by_position[position]
        stats = stats_by_position[position]
        if row is None or stats is None:
            raise RuntimeError(f"Missing sampled row for answer position {position}.")
        log_prob_rows.append(row)
        sampled_completion_text_counts.append(
            dict(stats["sampled_completion_text_counts"])
        )
        sampled_raw_completion_text_counts.append(
            dict(stats["sampled_raw_completion_text_counts"])
        )
        valid_counts.append(stats["valid_samples"])
        empty_counts.append(stats["empty_samples"])
        raw_empty_counts.append(stats["raw_empty_samples"])
        invalid_empty_counts.append(stats["invalid_empty_samples"])
        unmappable_nonempty_counts.append(stats["unmappable_nonempty_samples"])
        canonical_eos_projection_counts.append(stats["canonical_eos_projections"])
        claude_45_whitespace_overlap_choice_counts.append(
            stats["claude_45_whitespace_overlap_choices"]
        )
        claude_45_whitespace_overlap_char_counts.append(
            stats["claude_45_whitespace_overlap_chars"]
        )
        multi_token_counts.append(stats["multi_token_samples"])
        failed_counts.append(stats["failed_samples"])
        missing_choice_counts.append(stats["missing_choices"])
        request_failure_counts.append(stats["request_failures"])
        requested_sample_counts.append(stats["requested_samples"])
        returned_choice_counts.append(stats["returned_choices"])
        refill_round_counts.append(stats["refill_rounds"])
        empty_length_sample_counts.append(stats["empty_length_samples"])
        empty_length_retry_request_counts.append(
            stats["empty_length_retry_requests"]
        )
        empty_length_retry_recovery_counts.append(
            stats["empty_length_retry_recoveries"]
        )
        empty_length_retry_exhausted_flags.append(
            stats["empty_length_retry_exhausted"]
        )
        reasoning_choice_counts.append(stats["reasoning_choices"])
        reasoning_token_counts.append(stats["reasoning_tokens"])
        single_choice_fallback_flags.append(stats["used_single_choice_fallback"])
        actual_provider_counts.update(stats["actual_provider_counts"])
        actual_response_model_counts.update(stats["actual_response_model_counts"])
        finish_reason_counts.update(stats["finish_reason_counts"])
        native_finish_reason_counts.update(stats["native_finish_reason_counts"])
        canonical_eos_projection_reason_counts.update(
            stats["canonical_eos_projection_reason_counts"]
        )
        sample_response_cache_status_counts.update(
            stats["response_cache_status_counts"]
        )
        sample_max_tokens_choice_counts.update(
            stats["sample_max_tokens_choice_counts"]
        )
        request_error_counts.update(stats["request_error_counts"])
        missing_router_metadata_choices += stats["missing_router_metadata_choices"]
        max_routing_attempt = max(max_routing_attempt, stats["max_routing_attempt"])
        if risk_gate is not None or bool(
            getattr(args, "store_deterministic_base_tokens", False)
        ):
            base_id = base_ids_by_position[position]
            if base_id is None:
                raise RuntimeError(f"Missing risk-gate base token for answer position {position}.")
            base_token_ids.append(base_id)
            deterministic_info = deterministic_infos_by_position[position] or {}
            attempts = int(
                deterministic_info.get("empty_length_retry_attempts", 0)
            )
            deterministic_empty_length_retry_attempts += attempts
            deterministic_empty_length_retry_recoveries += int(
                bool(deterministic_info.get("empty_length_retry_recovered"))
            )
            deterministic_empty_length_retry_exhaustions += int(
                bool(deterministic_info.get("empty_length_retry_exhausted"))
            )

    labels = torch.tensor(answer_token_ids, dtype=torch.long).unsqueeze(0)
    log_probs = torch.stack(log_prob_rows, dim=0).unsqueeze(0).cpu()
    result = {
        "log_probs": log_probs,
        "labels": labels.cpu(),
        "sampled_completion_text_counts": sampled_completion_text_counts,
        "sampled_raw_completion_text_counts": sampled_raw_completion_text_counts,
        "sampled_prefix_texts": list(prefix_texts),
        "prompt_text": question,
        "answer_text": answer,
        "valid_sample_counts": torch.tensor(valid_counts, dtype=torch.long).unsqueeze(0),
        "empty_sample_counts": torch.tensor(empty_counts, dtype=torch.long).unsqueeze(0),
        "raw_empty_sample_counts": torch.tensor(
            raw_empty_counts, dtype=torch.long
        ).unsqueeze(0),
        "invalid_empty_sample_counts": torch.tensor(
            invalid_empty_counts, dtype=torch.long
        ).unsqueeze(0),
        "unmappable_nonempty_sample_counts": torch.tensor(
            unmappable_nonempty_counts, dtype=torch.long
        ).unsqueeze(0),
        "canonical_eos_projection_counts": torch.tensor(
            canonical_eos_projection_counts, dtype=torch.long
        ).unsqueeze(0),
        "claude_45_whitespace_overlap_choice_counts": torch.tensor(
            claude_45_whitespace_overlap_choice_counts,
            dtype=torch.long,
        ).unsqueeze(0),
        "claude_45_whitespace_overlap_char_counts": torch.tensor(
            claude_45_whitespace_overlap_char_counts,
            dtype=torch.long,
        ).unsqueeze(0),
        "multi_token_sample_counts": torch.tensor(multi_token_counts, dtype=torch.long).unsqueeze(0),
        "failed_sample_counts": torch.tensor(failed_counts, dtype=torch.long).unsqueeze(0),
        "missing_choice_counts": torch.tensor(
            missing_choice_counts, dtype=torch.long
        ).unsqueeze(0),
        "request_failure_counts": torch.tensor(
            request_failure_counts, dtype=torch.long
        ).unsqueeze(0),
        "requested_sample_counts": torch.tensor(
            requested_sample_counts, dtype=torch.long
        ).unsqueeze(0),
        "returned_choice_counts": torch.tensor(
            returned_choice_counts, dtype=torch.long
        ).unsqueeze(0),
        "sample_refill_round_counts": torch.tensor(
            refill_round_counts, dtype=torch.long
        ).unsqueeze(0),
        "empty_length_sample_counts": torch.tensor(
            empty_length_sample_counts, dtype=torch.long
        ).unsqueeze(0),
        "empty_length_retry_request_counts": torch.tensor(
            empty_length_retry_request_counts, dtype=torch.long
        ).unsqueeze(0),
        "empty_length_retry_recovery_counts": torch.tensor(
            empty_length_retry_recovery_counts, dtype=torch.long
        ).unsqueeze(0),
        "empty_length_retry_exhausted_flags": torch.tensor(
            empty_length_retry_exhausted_flags, dtype=torch.bool
        ).unsqueeze(0),
        "reasoning_choice_counts": torch.tensor(
            reasoning_choice_counts, dtype=torch.long
        ).unsqueeze(0),
        "reasoning_token_counts": torch.tensor(
            reasoning_token_counts, dtype=torch.long
        ).unsqueeze(0),
        "single_choice_fallback_flags": torch.tensor(
            single_choice_fallback_flags, dtype=torch.bool
        ).unsqueeze(0),
        "metadata": {
            "payload_schema_version": 2,
            "sample_representation": "completion_text_counter_v1",
            "completion_text_semantics": "continuation_after_prefill_strip",
            "raw_completion_text_semantics": "openrouter_message_content_before_prefill_strip",
            "empty_completion_text_semantics": "canonical_eos",
            "materialized_log_probs_tokenizer": args.tokenizer_name,
            "max_answer_tokens": args.max_answer_tokens,
            "source": "sampled_openrouter",
            "model": args.model,
            "tokenizer_name": args.tokenizer_name,
            "samples_per_token": args.samples_per_token,
            "sample_completion_policy": args.sample_completion_policy,
            "max_sample_refill_rounds": args.max_sample_refill_rounds,
            "sample_temperature": args.sample_temperature,
            "top_p": args.top_p,
            "omit_top_p": bool(getattr(args, "omit_top_p", False)),
            "observed_alpha": args.observed_alpha,
            "floor_mass": args.floor_mass,
            "api_max_tokens": args.api_max_tokens,
            "empty_length_retry_max_tokens": getattr(
                args, "empty_length_retry_max_tokens", None
            ),
            "max_empty_length_retry_rounds": int(
                getattr(args, "max_empty_length_retry_rounds", 0)
            ),
            "empty_response_token": args.empty_response_token,
            "canonical_eos_token_id": (
                int(tokenizer.eos_token_id)
                if getattr(tokenizer, "eos_token_id", None) is not None
                else None
            ),
            "reasoning_mode": args.reasoning_mode,
            "thinking_budget": getattr(args, "thinking_budget", None),
            "qwen_hard_no_think_prefill": bool(
                getattr(args, "qwen_hard_no_think_prefill", False)
            ),
            "claude_45_whitespace_prefill_fix": (
                CLAUDE_45_WHITESPACE_PREFILL_FIX
                if is_claude_45_model(args.model)
                else None
            ),
            "reject_reasoning_tokens": bool(
                getattr(args, "reject_reasoning_tokens", False)
            ),
            "provider_order": list(args.provider_order) if args.provider_order else None,
            "provider_allow_fallbacks": args.provider_allow_fallbacks,
            "router_metadata_requested": bool(getattr(args, "router_metadata", False)),
            "disable_openrouter_response_cache": bool(
                getattr(args, "disable_openrouter_response_cache", False)
            ),
            "actual_provider_counts": dict(actual_provider_counts),
            "actual_response_model_counts": dict(actual_response_model_counts),
            "sample_finish_reason_counts": dict(finish_reason_counts),
            "sample_native_finish_reason_counts": dict(native_finish_reason_counts),
            "canonical_eos_projection_reason_counts": dict(
                canonical_eos_projection_reason_counts
            ),
            "sample_response_cache_status_counts": dict(
                sample_response_cache_status_counts
            ),
            "sample_max_tokens_choice_counts": dict(
                sorted(sample_max_tokens_choice_counts.items())
            ),
            "request_error_counts": dict(request_error_counts),
            "actual_provider_call_counts": dict(
                Counter(client.provider_call_counts) - provider_call_counts_before
            ),
            "actual_response_model_call_counts": dict(
                Counter(client.response_model_call_counts)
                - response_model_call_counts_before
            ),
            "response_cache_status_call_counts": dict(
                Counter(client.response_cache_status_counts)
                - response_cache_status_counts_before
            ),
            "api_max_tokens_call_counts": {
                str(key): value
                for key, value in sorted(
                    (
                        Counter(getattr(client, "api_max_tokens_call_counts", {}))
                        - api_max_tokens_call_counts_before
                    ).items()
                )
            },
            "empty_length_retry_attempts": int(
                getattr(client, "empty_length_retry_attempts", 0)
            )
            - empty_length_retry_attempts_before,
            "empty_length_retry_recoveries": int(
                getattr(client, "empty_length_retry_recoveries", 0)
            )
            - empty_length_retry_recoveries_before,
            "empty_length_retry_exhaustions": int(
                getattr(client, "empty_length_retry_exhaustions", 0)
            )
            - empty_length_retry_exhaustions_before,
            "reasoning_response_calls": int(
                getattr(client, "reasoning_response_calls", 0)
            )
            - reasoning_response_calls_before,
            "reasoning_message_choices": int(
                getattr(client, "reasoning_message_choices", 0)
            )
            - reasoning_message_choices_before,
            "reasoning_tokens": int(getattr(client, "reasoning_tokens", 0))
            - reasoning_tokens_before,
            "missing_router_metadata_choices": missing_router_metadata_choices,
            "missing_router_metadata_calls": (
                client.missing_router_metadata_calls - missing_router_metadata_calls_before
            ),
            "max_routing_attempt": max_routing_attempt,
            "cache_configuration_fingerprint": getattr(
                args, "cache_configuration_fingerprint", None
            ),
            "api_calls": client.calls - calls_before,
            "api_request_attempts": int(getattr(client, "request_attempts", 0))
            - request_attempts_before,
            "mc_requested_samples": int(
                getattr(client, "mc_requested_samples", 0)
            )
            - mc_requested_samples_before,
            "api_cost": client.total_cost - cost_before,
            "deterministic_failures": deterministic_failures,
            "deterministic_empty_length_retry_attempts": (
                deterministic_empty_length_retry_attempts
            ),
            "deterministic_empty_length_retry_recoveries": (
                deterministic_empty_length_retry_recoveries
            ),
            "deterministic_empty_length_retry_exhaustions": (
                deterministic_empty_length_retry_exhaustions
            ),
        },
    }

    if risk_gate is not None or bool(
        getattr(args, "store_deterministic_base_tokens", False)
    ):
        base_tensor = torch.tensor(base_token_ids, dtype=torch.long)
        result["deterministic_completion_texts"] = [
            str((info or {}).get("completion_text", tokenizer.decode([base_id])))
            for info, base_id in zip(deterministic_infos_by_position, base_token_ids)
        ]
        result["deterministic_raw_completion_texts"] = [
            str((info or {}).get("raw_completion_text", ""))
            for info in deterministic_infos_by_position
        ]
        if risk_gate is not None:
            result.update(
                build_risk_gate_payload(
                    risk_gate=risk_gate,
                    tokenizer=tokenizer,
                    question=question,
                    answer_ids=labels,
                    base_token_ids=base_tensor,
                )
            )
            result["metadata"].update(
                {
                    "risk_gate_checkpoint": args.risk_gate_checkpoint,
                    "risk_gate_threshold": args.risk_gate_threshold,
                    "risk_gate_model_name": args.risk_gate_model_name,
                    "risk_gate_local_files_only": args.risk_gate_local_files_only,
                    "risk_gate_token_source": "openrouter_temperature0",
                }
            )
        else:
            result["risk_gate_token_ids"] = base_tensor.unsqueeze(0)
            result["metadata"]["deterministic_token_source"] = (
                "openrouter_temperature0"
            )

    return result


def get_risk_active_sampled_logprobs_openrouter(
    client: OpenRouterClient,
    tokenizer,
    question: str,
    answer_token_ids: list[int],
    answer_text: str,
    vocab_size: int,
    store_dtype: torch.dtype,
    args: argparse.Namespace,
    risk_gate,
) -> Optional[dict]:
    base_token_ids = []
    deterministic_failures = 0
    deterministic_empty_length_retry_attempts = 0
    deterministic_empty_length_retry_recoveries = 0
    deterministic_empty_length_retry_exhaustions = 0
    calls_before = client.calls
    request_attempts_before = int(getattr(client, "request_attempts", 0))
    mc_requested_samples_before = int(getattr(client, "mc_requested_samples", 0))
    cost_before = client.total_cost
    provider_call_counts_before = Counter(client.provider_call_counts)
    response_model_call_counts_before = Counter(client.response_model_call_counts)
    response_cache_status_counts_before = Counter(client.response_cache_status_counts)
    api_max_tokens_call_counts_before = Counter(
        getattr(client, "api_max_tokens_call_counts", {})
    )
    empty_length_retry_attempts_before = int(
        getattr(client, "empty_length_retry_attempts", 0)
    )
    empty_length_retry_recoveries_before = int(
        getattr(client, "empty_length_retry_recoveries", 0)
    )
    empty_length_retry_exhaustions_before = int(
        getattr(client, "empty_length_retry_exhaustions", 0)
    )
    missing_router_metadata_calls_before = client.missing_router_metadata_calls

    def scan_one_position(position: int):
        prefix_text = tokenizer.decode(answer_token_ids[:position], skip_special_tokens=False)
        base_id, deterministic_info = query_next_token_id(
            client=client,
            tokenizer=tokenizer,
            question=question,
            prefix_text=prefix_text,
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.api_max_tokens,
        )
        deterministic_failure = 0
        if base_id is None:
            deterministic_failure = 1
            if args.sample_completion_policy == "exact":
                raise RuntimeError(
                    "Deterministic risk-scan query returned no valid token at "
                    f"answer position {position} after "
                    f"{int(deterministic_info.get('empty_length_retry_attempts', 0))} "
                    "adaptive empty-length retries "
                    f"(finish_reason={deterministic_info.get('finish_reason')!r}, "
                    "native_finish_reason="
                    f"{deterministic_info.get('native_finish_reason')!r})."
                )
            base_id = answer_token_ids[position]
        return position, base_id, deterministic_failure, deterministic_info

    base_ids_by_position: list[Optional[int]] = [None] * len(answer_token_ids)
    deterministic_infos_by_position: list[Optional[dict]] = [None] * len(
        answer_token_ids
    )
    if args.parallel_positions <= 1:
        for position in tqdm(range(len(answer_token_ids)), desc="API risk scan", leave=False):
            if args.max_api_calls is not None and client.calls >= args.max_api_calls:
                raise RuntimeError(
                    f"Reached --max_api_calls={args.max_api_calls}; stopping before more API spend."
                )
            (
                position,
                base_id,
                deterministic_failure,
                deterministic_info,
            ) = scan_one_position(position)
            base_ids_by_position[position] = base_id
            deterministic_infos_by_position[position] = deterministic_info
            deterministic_failures += deterministic_failure
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_positions) as executor:
            futures = {
                executor.submit(scan_one_position, position): position
                for position in range(len(answer_token_ids))
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="API risk scan",
                leave=False,
            ):
                (
                    position,
                    base_id,
                    deterministic_failure,
                    deterministic_info,
                ) = future.result()
                base_ids_by_position[position] = base_id
                deterministic_infos_by_position[position] = deterministic_info
                deterministic_failures += deterministic_failure

    for position, base_id in enumerate(base_ids_by_position):
        if base_id is None:
            raise RuntimeError(f"Missing risk-scan token for answer position {position}.")
        base_token_ids.append(base_id)
        deterministic_info = deterministic_infos_by_position[position] or {}
        deterministic_empty_length_retry_attempts += int(
            deterministic_info.get("empty_length_retry_attempts", 0)
        )
        deterministic_empty_length_retry_recoveries += int(
            bool(deterministic_info.get("empty_length_retry_recovered"))
        )
        deterministic_empty_length_retry_exhaustions += int(
            bool(deterministic_info.get("empty_length_retry_exhausted"))
        )

    full_labels = torch.tensor(answer_token_ids, dtype=torch.long).unsqueeze(0)
    base_tensor = torch.tensor(base_token_ids, dtype=torch.long)
    risk_payload = build_risk_gate_payload(
        risk_gate=risk_gate,
        tokenizer=tokenizer,
        question=question,
        answer_ids=full_labels,
        base_token_ids=base_tensor,
    )
    full_mask = risk_payload["risk_gate_mask"][0].bool()
    active_positions = torch.nonzero(full_mask, as_tuple=False).flatten().tolist()
    if not active_positions:
        return None

    log_prob_rows = []
    sampled_completion_text_counts = []
    sampled_raw_completion_text_counts = []
    sampled_prefix_texts = []
    valid_counts = []
    empty_counts = []
    raw_empty_counts = []
    invalid_empty_counts = []
    unmappable_nonempty_counts = []
    canonical_eos_projection_counts = []
    claude_45_whitespace_overlap_choice_counts = []
    claude_45_whitespace_overlap_char_counts = []
    multi_token_counts = []
    failed_counts = []
    missing_choice_counts = []
    request_failure_counts = []
    requested_sample_counts = []
    returned_choice_counts = []
    refill_round_counts = []
    empty_length_sample_counts = []
    empty_length_retry_request_counts = []
    empty_length_retry_recovery_counts = []
    empty_length_retry_exhausted_flags = []
    single_choice_fallback_flags = []
    actual_provider_counts: Counter[str] = Counter()
    actual_response_model_counts: Counter[str] = Counter()
    finish_reason_counts: Counter[str] = Counter()
    native_finish_reason_counts: Counter[str] = Counter()
    canonical_eos_projection_reason_counts: Counter[str] = Counter()
    sample_response_cache_status_counts: Counter[str] = Counter()
    sample_max_tokens_choice_counts: Counter[str] = Counter()
    request_error_counts: Counter[str] = Counter()
    missing_router_metadata_choices = 0
    max_routing_attempt = 0
    for position in tqdm(active_positions, desc="API active token positions", leave=False):
        if args.max_api_calls is not None and client.calls >= args.max_api_calls:
            raise RuntimeError(
                f"Reached --max_api_calls={args.max_api_calls}; stopping before more API spend."
            )
        prefix_text = tokenizer.decode(answer_token_ids[:position], skip_special_tokens=False)
        sampled_ids, sample_stats = sample_position_token_ids(
            client=client,
            tokenizer=tokenizer,
            question=question,
            prefix_text=prefix_text,
            args=args,
        )
        if not sampled_ids:
            raise RuntimeError(f"No valid sampled tokens at active answer position {position}.")
        valid_counts.append(sample_stats["valid_samples"])
        sampled_completion_text_counts.append(
            dict(sample_stats["sampled_completion_text_counts"])
        )
        sampled_raw_completion_text_counts.append(
            dict(sample_stats["sampled_raw_completion_text_counts"])
        )
        sampled_prefix_texts.append(prefix_text)
        empty_counts.append(sample_stats["empty_samples"])
        raw_empty_counts.append(sample_stats["raw_empty_samples"])
        invalid_empty_counts.append(sample_stats["invalid_empty_samples"])
        unmappable_nonempty_counts.append(
            sample_stats["unmappable_nonempty_samples"]
        )
        canonical_eos_projection_counts.append(
            sample_stats["canonical_eos_projections"]
        )
        claude_45_whitespace_overlap_choice_counts.append(
            sample_stats["claude_45_whitespace_overlap_choices"]
        )
        claude_45_whitespace_overlap_char_counts.append(
            sample_stats["claude_45_whitespace_overlap_chars"]
        )
        multi_token_counts.append(sample_stats["multi_token_samples"])
        failed_counts.append(sample_stats["failed_samples"])
        missing_choice_counts.append(sample_stats["missing_choices"])
        request_failure_counts.append(sample_stats["request_failures"])
        requested_sample_counts.append(sample_stats["requested_samples"])
        returned_choice_counts.append(sample_stats["returned_choices"])
        refill_round_counts.append(sample_stats["refill_rounds"])
        empty_length_sample_counts.append(sample_stats["empty_length_samples"])
        empty_length_retry_request_counts.append(
            sample_stats["empty_length_retry_requests"]
        )
        empty_length_retry_recovery_counts.append(
            sample_stats["empty_length_retry_recoveries"]
        )
        empty_length_retry_exhausted_flags.append(
            sample_stats["empty_length_retry_exhausted"]
        )
        single_choice_fallback_flags.append(sample_stats["used_single_choice_fallback"])
        actual_provider_counts.update(sample_stats["actual_provider_counts"])
        actual_response_model_counts.update(sample_stats["actual_response_model_counts"])
        finish_reason_counts.update(sample_stats["finish_reason_counts"])
        native_finish_reason_counts.update(
            sample_stats["native_finish_reason_counts"]
        )
        canonical_eos_projection_reason_counts.update(
            sample_stats["canonical_eos_projection_reason_counts"]
        )
        sample_response_cache_status_counts.update(
            sample_stats["response_cache_status_counts"]
        )
        sample_max_tokens_choice_counts.update(
            sample_stats["sample_max_tokens_choice_counts"]
        )
        request_error_counts.update(sample_stats["request_error_counts"])
        missing_router_metadata_choices += sample_stats["missing_router_metadata_choices"]
        max_routing_attempt = max(max_routing_attempt, sample_stats["max_routing_attempt"])
        log_prob_rows.append(
            ids_to_log_probs(
                sampled_token_ids=sampled_ids,
                vocab_size=vocab_size,
                observed_alpha=args.observed_alpha,
                floor_mass=args.floor_mass,
                dtype=store_dtype,
            )
        )

    active_index_tensor = torch.tensor(active_positions, dtype=torch.long)
    labels = torch.tensor(
        [answer_token_ids[position] for position in active_positions],
        dtype=torch.long,
    ).unsqueeze(0)
    log_probs = torch.stack(log_prob_rows, dim=0).unsqueeze(0).cpu()
    active_deterministic_completion_texts = [
        str(
            (deterministic_infos_by_position[position] or {}).get(
                "completion_text",
                tokenizer.decode([base_token_ids[position]]),
            )
        )
        for position in active_positions
    ]
    active_deterministic_raw_completion_texts = [
        str(
            (deterministic_infos_by_position[position] or {}).get(
                "raw_completion_text",
                "",
            )
        )
        for position in active_positions
    ]
    return {
        "log_probs": log_probs,
        "labels": labels.cpu(),
        "sampled_completion_text_counts": sampled_completion_text_counts,
        "sampled_raw_completion_text_counts": sampled_raw_completion_text_counts,
        "sampled_prefix_texts": sampled_prefix_texts,
        "prompt_text": question,
        "answer_text": answer_text,
        "deterministic_completion_texts": active_deterministic_completion_texts,
        "deterministic_raw_completion_texts": (
            active_deterministic_raw_completion_texts
        ),
        "risk_gate_mask": torch.ones_like(labels, dtype=torch.bool),
        "risk_gate_scores": risk_payload["risk_gate_scores"][:, active_index_tensor].cpu(),
        "risk_gate_token_ids": risk_payload["risk_gate_token_ids"][:, active_index_tensor].cpu(),
        "risk_gate_active_positions": active_index_tensor.unsqueeze(0),
        "valid_sample_counts": torch.tensor(valid_counts, dtype=torch.long).unsqueeze(0),
        "empty_sample_counts": torch.tensor(empty_counts, dtype=torch.long).unsqueeze(0),
        "raw_empty_sample_counts": torch.tensor(
            raw_empty_counts, dtype=torch.long
        ).unsqueeze(0),
        "invalid_empty_sample_counts": torch.tensor(
            invalid_empty_counts, dtype=torch.long
        ).unsqueeze(0),
        "unmappable_nonempty_sample_counts": torch.tensor(
            unmappable_nonempty_counts, dtype=torch.long
        ).unsqueeze(0),
        "canonical_eos_projection_counts": torch.tensor(
            canonical_eos_projection_counts, dtype=torch.long
        ).unsqueeze(0),
        "claude_45_whitespace_overlap_choice_counts": torch.tensor(
            claude_45_whitespace_overlap_choice_counts,
            dtype=torch.long,
        ).unsqueeze(0),
        "claude_45_whitespace_overlap_char_counts": torch.tensor(
            claude_45_whitespace_overlap_char_counts,
            dtype=torch.long,
        ).unsqueeze(0),
        "multi_token_sample_counts": torch.tensor(multi_token_counts, dtype=torch.long).unsqueeze(0),
        "failed_sample_counts": torch.tensor(failed_counts, dtype=torch.long).unsqueeze(0),
        "missing_choice_counts": torch.tensor(
            missing_choice_counts, dtype=torch.long
        ).unsqueeze(0),
        "request_failure_counts": torch.tensor(
            request_failure_counts, dtype=torch.long
        ).unsqueeze(0),
        "requested_sample_counts": torch.tensor(
            requested_sample_counts, dtype=torch.long
        ).unsqueeze(0),
        "returned_choice_counts": torch.tensor(
            returned_choice_counts, dtype=torch.long
        ).unsqueeze(0),
        "sample_refill_round_counts": torch.tensor(
            refill_round_counts, dtype=torch.long
        ).unsqueeze(0),
        "empty_length_sample_counts": torch.tensor(
            empty_length_sample_counts, dtype=torch.long
        ).unsqueeze(0),
        "empty_length_retry_request_counts": torch.tensor(
            empty_length_retry_request_counts, dtype=torch.long
        ).unsqueeze(0),
        "empty_length_retry_recovery_counts": torch.tensor(
            empty_length_retry_recovery_counts, dtype=torch.long
        ).unsqueeze(0),
        "empty_length_retry_exhausted_flags": torch.tensor(
            empty_length_retry_exhausted_flags, dtype=torch.bool
        ).unsqueeze(0),
        "single_choice_fallback_flags": torch.tensor(
            single_choice_fallback_flags, dtype=torch.bool
        ).unsqueeze(0),
        "metadata": {
            "payload_schema_version": 2,
            "sample_representation": "completion_text_counter_v1",
            "completion_text_semantics": "continuation_after_prefill_strip",
            "raw_completion_text_semantics": "openrouter_message_content_before_prefill_strip",
            "empty_completion_text_semantics": "canonical_eos",
            "materialized_log_probs_tokenizer": args.tokenizer_name,
            "max_answer_tokens": args.max_answer_tokens,
            "source": "sampled_openrouter",
            "sample_only_risk_active": True,
            "model": args.model,
            "tokenizer_name": args.tokenizer_name,
            "samples_per_token": args.samples_per_token,
            "sample_completion_policy": args.sample_completion_policy,
            "max_sample_refill_rounds": args.max_sample_refill_rounds,
            "sample_temperature": args.sample_temperature,
            "top_p": args.top_p,
            "observed_alpha": args.observed_alpha,
            "floor_mass": args.floor_mass,
            "api_max_tokens": args.api_max_tokens,
            "empty_length_retry_max_tokens": getattr(
                args, "empty_length_retry_max_tokens", None
            ),
            "max_empty_length_retry_rounds": int(
                getattr(args, "max_empty_length_retry_rounds", 0)
            ),
            "empty_response_token": args.empty_response_token,
            "canonical_eos_token_id": (
                int(tokenizer.eos_token_id)
                if getattr(tokenizer, "eos_token_id", None) is not None
                else None
            ),
            "reasoning_mode": args.reasoning_mode,
            "claude_45_whitespace_prefill_fix": (
                CLAUDE_45_WHITESPACE_PREFILL_FIX
                if is_claude_45_model(args.model)
                else None
            ),
            "provider_order": list(args.provider_order) if args.provider_order else None,
            "provider_allow_fallbacks": args.provider_allow_fallbacks,
            "router_metadata_requested": bool(getattr(args, "router_metadata", False)),
            "disable_openrouter_response_cache": bool(
                getattr(args, "disable_openrouter_response_cache", False)
            ),
            "actual_provider_counts": dict(actual_provider_counts),
            "actual_response_model_counts": dict(actual_response_model_counts),
            "sample_finish_reason_counts": dict(finish_reason_counts),
            "sample_native_finish_reason_counts": dict(native_finish_reason_counts),
            "canonical_eos_projection_reason_counts": dict(
                canonical_eos_projection_reason_counts
            ),
            "sample_response_cache_status_counts": dict(
                sample_response_cache_status_counts
            ),
            "sample_max_tokens_choice_counts": dict(
                sorted(sample_max_tokens_choice_counts.items())
            ),
            "request_error_counts": dict(request_error_counts),
            "actual_provider_call_counts": dict(
                Counter(client.provider_call_counts) - provider_call_counts_before
            ),
            "actual_response_model_call_counts": dict(
                Counter(client.response_model_call_counts)
                - response_model_call_counts_before
            ),
            "response_cache_status_call_counts": dict(
                Counter(client.response_cache_status_counts)
                - response_cache_status_counts_before
            ),
            "api_max_tokens_call_counts": {
                str(key): value
                for key, value in sorted(
                    (
                        Counter(getattr(client, "api_max_tokens_call_counts", {}))
                        - api_max_tokens_call_counts_before
                    ).items()
                )
            },
            "empty_length_retry_attempts": int(
                getattr(client, "empty_length_retry_attempts", 0)
            )
            - empty_length_retry_attempts_before,
            "empty_length_retry_recoveries": int(
                getattr(client, "empty_length_retry_recoveries", 0)
            )
            - empty_length_retry_recoveries_before,
            "empty_length_retry_exhaustions": int(
                getattr(client, "empty_length_retry_exhaustions", 0)
            )
            - empty_length_retry_exhaustions_before,
            "missing_router_metadata_choices": missing_router_metadata_choices,
            "missing_router_metadata_calls": (
                client.missing_router_metadata_calls - missing_router_metadata_calls_before
            ),
            "max_routing_attempt": max_routing_attempt,
            "cache_configuration_fingerprint": getattr(
                args, "cache_configuration_fingerprint", None
            ),
            "api_calls": client.calls - calls_before,
            "api_request_attempts": int(getattr(client, "request_attempts", 0))
            - request_attempts_before,
            "mc_requested_samples": int(
                getattr(client, "mc_requested_samples", 0)
            )
            - mc_requested_samples_before,
            "api_cost": client.total_cost - cost_before,
            "deterministic_failures": deterministic_failures,
            "deterministic_empty_length_retry_attempts": (
                deterministic_empty_length_retry_attempts
            ),
            "deterministic_empty_length_retry_recoveries": (
                deterministic_empty_length_retry_recoveries
            ),
            "deterministic_empty_length_retry_exhaustions": (
                deterministic_empty_length_retry_exhaustions
            ),
            "original_answer_tokens": len(answer_token_ids),
            "risk_active_tokens": len(active_positions),
            "risk_gate_checkpoint": args.risk_gate_checkpoint,
            "risk_gate_threshold": args.risk_gate_threshold,
            "risk_gate_model_name": args.risk_gate_model_name,
            "risk_gate_local_files_only": args.risk_gate_local_files_only,
            "risk_gate_token_source": "openrouter_temperature0",
        },
    }


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive.")
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative.")
    if not args.dataset_prompt_field or not args.dataset_answer_field:
        raise ValueError("Local dataset field names must be non-empty.")
    if args.end_index is not None and args.end_index <= args.start_index:
        raise ValueError("--end_index must be greater than --start_index when provided.")
    if (
        args.end_index is not None
        and args.max_samples > args.end_index - args.start_index
    ):
        raise ValueError(
            "--max_samples cannot exceed the locked --start_index/--end_index row count."
        )
    if args.max_answer_tokens is not None and args.max_answer_tokens <= 0:
        raise ValueError("--max_answer_tokens must be positive when provided.")
    if args.samples_per_token <= 0:
        raise ValueError("--samples_per_token must be positive.")
    if args.api_max_tokens <= 0:
        raise ValueError("--api_max_tokens must be positive.")
    if args.thinking_budget is not None and args.thinking_budget < -1:
        raise ValueError("--thinking_budget must be -1 or greater when provided.")
    if args.parallel_requests <= 0:
        raise ValueError("--parallel_requests must be positive.")
    if args.parallel_positions <= 0:
        raise ValueError("--parallel_positions must be positive.")
    if args.sample_choices_per_request <= 0:
        raise ValueError("--sample_choices_per_request must be positive.")
    if args.max_sample_refill_rounds < 0:
        raise ValueError("--max_sample_refill_rounds must be non-negative.")
    if args.max_empty_length_retry_rounds < 0:
        raise ValueError("--max_empty_length_retry_rounds must be non-negative.")
    for field in (
        "max_api_calls",
        "max_api_request_attempts",
        "max_mc_requested_samples",
    ):
        value = getattr(args, field, None)
        if value is not None and value <= 0:
            raise ValueError(f"--{field} must be positive when provided.")
    if args.empty_length_retry_max_tokens is not None:
        if args.empty_length_retry_max_tokens <= 0:
            raise ValueError("--empty_length_retry_max_tokens must be positive.")
        if (
            args.max_empty_length_retry_rounds > 0
            and args.empty_length_retry_max_tokens <= args.api_max_tokens
        ):
            raise ValueError(
                "--empty_length_retry_max_tokens must exceed --api_max_tokens "
                "when adaptive retries are enabled."
            )
    if (
        args.max_empty_length_retry_rounds > 0
        and args.empty_length_retry_max_tokens is None
    ):
        raise ValueError(
            "--max_empty_length_retry_rounds requires "
            "--empty_length_retry_max_tokens."
        )
    if (
        args.write_cache_manifest_only
        and args.cache_manifest_policy != "require"
    ):
        raise ValueError(
            "--write_cache_manifest_only requires --cache_manifest_policy=require."
        )

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_tokenizer(
        args.tokenizer_name,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
        revision=args.tokenizer_revision,
        fix_mistral_regex=args.fix_mistral_regex,
    )
    if (
        args.empty_response_token == "stop_eos"
        and getattr(tokenizer, "eos_token_id", None) is None
    ):
        raise ValueError(
            "--empty_response_token=stop_eos requires a tokenizer with eos_token_id."
        )
    cache_configuration = build_cache_configuration(args, tokenizer)
    args.cache_configuration_fingerprint = ensure_cache_manifest(
        output_dir=args.output_dir,
        configuration=cache_configuration,
        policy=args.cache_manifest_policy,
    )
    if args.write_cache_manifest_only:
        print(
            "Created or validated cache manifest at "
            f"{Path(args.output_dir) / 'cache_manifest.json'} "
            f"(fingerprint={args.cache_configuration_fingerprint})."
        )
        return

    api_key = resolve_api_key(args)
    client = OpenRouterClient(args, api_key)
    risk_gate = load_risk_gate(args)

    train_data = load_training_data(args)
    existing_files = [name for name in os.listdir(args.output_dir) if name.endswith(".pt")]
    saved_count = len(existing_files)
    if saved_count >= args.max_samples:
        print(f"Output dir already has {saved_count} cache files; nothing to do.")
        return

    progress = tqdm(total=args.max_samples - saved_count, desc="OpenRouter sampled caches")
    dataset_stop = (
        min(len(train_data), args.end_index)
        if args.end_index is not None
        else len(train_data)
    )
    for dataset_idx in range(args.start_index, dataset_stop):
        if saved_count >= args.max_samples:
            break
        item = train_data[dataset_idx]
        question = item[args.dataset_prompt_field]
        answer = item[args.dataset_answer_field]
        api_question = question + "\n/no_think" if args.append_no_think else question
        data_hash = hashlib.md5((question + answer).encode("utf-8")).hexdigest()
        output_path = os.path.join(args.output_dir, f"{data_hash}.pt")
        if os.path.exists(output_path):
            continue

        try:
            result = get_sampled_logprobs_openrouter(
                client=client,
                tokenizer=tokenizer,
                question=api_question,
                answer=answer,
                args=args,
                risk_gate=risk_gate,
            )
        except FatalOpenRouterResponseError as exc:
            if not (
                getattr(args, "skip_provider_moderation_rejections", False)
                and is_provider_moderation_rejection(exc)
            ):
                raise
            record_provider_moderation_skip(
                args.output_dir,
                dataset_idx=dataset_idx,
                question=question,
                answer=answer,
            )
            print(
                "skipped_provider_moderation="
                f"dataset_idx:{dataset_idx},reason:provider_moderation_http_403",
                flush=True,
            )
            continue
        if result is None:
            continue
        example_metadata = {
            "dataset_idx": int(dataset_idx),
            "data_sha256": hashlib.sha256(
                (question + answer).encode("utf-8")
            ).hexdigest(),
            "cache_configuration_fingerprint": args.cache_configuration_fingerprint,
        }
        if "source_dataset_idx" in item:
            example_metadata["source_dataset_idx"] = int(
                item["source_dataset_idx"]
            )
        result.setdefault("metadata", {}).update(example_metadata)
        torch.save(result, output_path)
        saved_count += 1
        progress.update(1)
        print(
            f"saved={saved_count} api_calls={client.calls} "
            f"api_request_attempts={client.request_attempts} "
            f"mc_requested_samples={client.mc_requested_samples} "
            f"approx_cost=${client.total_cost:.6f}",
            flush=True,
        )

    progress.close()
    if args.cache_manifest_policy == "require" and saved_count != args.max_samples:
        raise RuntimeError(
            f"Strict cache run expected {args.max_samples} files but produced {saved_count} "
            f"within dataset indices [{args.start_index}, {dataset_stop})."
        )
    print(
        f"Processed {saved_count} samples. API calls={client.calls}, "
        f"API request attempts={client.request_attempts}, "
        f"MC requested samples={client.mc_requested_samples}, "
        f"approx_cost=${client.total_cost:.6f}. Saved in {args.output_dir}."
    )


if __name__ == "__main__":
    main()
