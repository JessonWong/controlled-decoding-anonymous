import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from benchmark_data import PromptRecord, collect_prompt_records as load_prompt_records
from modeling_biasnet import BiasNet
from context_encoder import FrozenContextEncoder, validate_contract
from risk_gate import PrefixRiskGate
from mc_reconstruction import (
    chat_template_sha256,
    FLOOR_LOGPROB,
    LOG_COUNT,
    MC_INPUT_REPRESENTATIONS,
    fuse_proxy_logits_with_mc_counts,
    sampled_ids_to_log_counts,
    validate_log_count_alpha,
)


PROXY_FUSION_MODE = "proxy_dirichlet_v1"
STATIC_MC_PRIOR_MODES = (
    "uniform_dirichlet_v1",
    "global_unigram_dirichlet_v1",
)
PROXY_CHAT_TEMPLATE_PROTOCOL = "messages_for_prefix_qwen_v1"
# Same message builder without the Qwen empty-think prefill, for targets with no
# /no_think switch. A separate identity so a cache built one way can never satisfy
# a runtime configured the other way.
PROXY_CHAT_TEMPLATE_PROTOCOL_PLAIN = "messages_for_prefix_plain_v1"

def expected_proxy_chat_template_protocol(args) -> str:
    """Protocol identity implied by the runtime's own prefill setting."""

    return (
        PROXY_CHAT_TEMPLATE_PROTOCOL
        if getattr(args, "qwen_hard_no_think_prefill", False)
        else PROXY_CHAT_TEMPLATE_PROTOCOL_PLAIN
    )
PROXY_VOCAB_TAIL_POLICY = "crop_to_shared_vocab_before_softmax"

TRAINING_DIR = Path(__file__).resolve().parent / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from pre_logits_sampled_openrouter import (  # noqa: E402
    CLAUDE_45_WHITESPACE_PREFILL_FIX,
    FatalOpenRouterResponseError,
    OpenRouterClient,
    ids_to_log_probs,
    is_claude_45_model,
    load_tokenizer,
    messages_for_prefix,
    parse_bool,
    query_next_token_id,
    resolve_api_key,
    sample_position_token_ids,
    strip_prefill_with_audit,
)
from anytime_stopping_features import (  # noqa: E402
    FEATURE_NAMES as ANYTIME_FEATURE_NAMES,
    build_state_features as build_anytime_state_features,
    top2_statistics as anytime_top2_statistics,
)
from anytime_proxy_mc import (  # noqa: E402
    BIASNET_AUTOCAST_DTYPE,
    BIASNET_AUTOCAST_ENABLED,
    BIASNET_PARAMETER_DTYPE,
    POLICY_SPEC_HASH_FIELD,
    PORTABLE_STOPPER_PROTOCOL,
    TARGET_PROTOCOL_METADATA_FIELDS,
    policy_spec_payload_sha256,
    portable_linear_stopper_confidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run BiasNet generation against an OpenRouter chat model.")
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
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument(
        "--benchmark",
        choices=["advbench", "harmbench", "sorrybench"],
        default=None,
        help="Load prompts from a benchmark source instead of --prompt/--prompt_file.",
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        default=None,
        help="Local benchmark file (HarmBench CSV or SORRY-Bench JSONL).",
    )
    parser.add_argument(
        "--benchmark_mutation",
        type=str,
        default=None,
        help="SORRY-Bench mutation suffix, for example slang or atbash.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N collected prompts.")
    parser.add_argument(
        "--begin",
        type=int,
        default=0,
        help="Inclusive index into the collected prompt set, applied before --limit.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive index into the collected prompt set, applied before --limit.",
    )
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--resume", action="store_true", help="Append to an existing JSONL and skip prompts already written.")
    parser.add_argument("--progress_steps", action="store_true", help="Print token-level progress without generated text.")
    parser.add_argument(
        "--stop_on_mc_failure",
        action="store_true",
        help="End the current completion instead of aborting when MC next-token estimation returns no valid samples.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument(
        "--initial_prefix",
        type=str,
        default="",
        help=(
            "Visible assistant prefix inserted before controlled generation. "
            "The prefix is returned with the completion but does not consume the "
            "--max_new_tokens control budget."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--mask_non_eos_special_tokens",
        action="store_true",
        help="Prevent a proxy-tokenizer BiasNet from emitting special tokens other than EOS.",
    )
    parser.add_argument("--biasnet_ckpt", type=str, default=None)
    parser.add_argument(
        "--biasnet_max_tokens",
        type=int,
        default=None,
        help=(
            "Optional maximum number of controlled generation steps on which BiasNet "
            "may be applied. Later steps use the deterministic base token."
        ),
    )
    parser.add_argument(
        "--bootstrap_biasnet_ckpt",
        type=str,
        default=None,
        help="Optional BiasNet used only for the first controlled generation steps.",
    )
    parser.add_argument(
        "--bootstrap_biasnet_tokens",
        type=int,
        default=0,
        help="Number of initial controlled steps handled by --bootstrap_biasnet_ckpt.",
    )
    parser.add_argument("--biasnet_dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mc_samples_per_token", type=int, default=50)
    parser.add_argument(
        "--mc_independent_views",
        type=int,
        default=1,
        help=(
            "Number of independent MC sample views evaluated by the checkpoint at "
            "each controlled token. Each view contains --mc_samples_per_token samples; "
            "the resulting per-view distributions are averaged in probability space."
        ),
    )
    parser.add_argument("--mc_sample_temperature", type=float, default=1.0)
    parser.add_argument("--mc_top_p", type=float, default=1.0)
    parser.add_argument("--mc_observed_alpha", type=float, default=0.1)
    parser.add_argument("--mc_floor_mass", type=float, default=1e-4)
    parser.add_argument("--context_device", default=None,
                        help="Frozen context encoder device; defaults to the BiasNet device.")
    parser.add_argument("--context_encoder_model", default=None,
                        help="Optional relocated encoder path, verified against checkpoint identity.")
    parser.add_argument("--context_encoder_tokenizer", default=None)
    parser.add_argument("--context_local_files_only", action="store_true")
    parser.add_argument(
        "--mc_static_prior_mode",
        choices=("none", *STATIC_MC_PRIOR_MODES),
        default=None,
        help=(
            "Optional fixed Dirichlet prior used instead of the raw MC floor. "
            "Defaults to the BiasNet checkpoint declaration."
        ),
    )
    parser.add_argument(
        "--mc_static_prior_strength",
        type=float,
        default=None,
        help="Positive Dirichlet strength for --mc_static_prior_mode.",
    )
    parser.add_argument(
        "--mc_static_prior_path",
        type=str,
        default=None,
        help="Path to a saved log-probability vector for the global-unigram prior.",
    )
    parser.add_argument(
        "--mc_input_representation",
        choices=MC_INPUT_REPRESENTATIONS,
        default=None,
        help=(
            "BiasNet MC input semantics. Defaults to the checkpoint declaration; "
            "legacy checkpoints default to floor_logprob."
        ),
    )
    parser.add_argument(
        "--mc_log_count_alpha",
        type=float,
        default=None,
        help="Positive alpha for log_count. Defaults to the checkpoint declaration.",
    )
    parser.add_argument(
        "--mc_base_score_representation",
        choices=MC_INPUT_REPRESENTATIONS,
        default=None,
        help=(
            "Fixed score baseline for residual decoding. Defaults to the "
            "checkpoint declaration or the input representation for legacy checkpoints."
        ),
    )
    parser.add_argument(
        "--proxy_model_name_or_path",
        type=str,
        default=None,
        help=(
            "Opt in to dense local-proxy/MC fusion using this causal LM. The "
            "proxy checkpoint and tokenizer must exactly match the fusion metadata "
            "stored in the BiasNet checkpoint."
        ),
    )
    parser.add_argument("--proxy_model_revision", type=str, default=None)
    parser.add_argument(
        "--proxy_temperature",
        type=float,
        default=1.0,
        help="Calibration temperature for local proxy logits.",
    )
    parser.add_argument(
        "--proxy_prior_strength",
        type=float,
        default=1.0,
        help="Dirichlet prior strength assigned to the local proxy distribution.",
    )
    parser.add_argument(
        "--proxy_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--proxy_quantization",
        choices=["none"],
        default="none",
        help="Proxy weight quantization mode; the MVP supports unquantized weights only.",
    )
    parser.add_argument(
        "--proxy_device",
        type=str,
        default=None,
        help="Local proxy device. Defaults to the BiasNet inference device.",
    )
    parser.add_argument("--proxy_trust_remote_code", action="store_true")
    parser.add_argument(
        "--proxy_local_files_only",
        action="store_true",
        help="Require the local proxy and its tokenizer to exist in the HF cache.",
    )
    parser.add_argument(
        "--anytime_policy_spec",
        type=str,
        default=None,
        help=(
            "Frozen development-calibrated anytime stopping policy. When set, "
            "proxy+MC sampling follows its incremental K schedule instead of "
            "always requesting --mc_samples_per_token choices."
        ),
    )
    parser.add_argument(
        "--anytime_policy_name",
        type=str,
        default="delta_0.05",
        help="Named operating point from --anytime_policy_spec.",
    )
    parser.add_argument(
        "--restrict_to_observed_support",
        action="store_true",
        help=(
            "Restrict BiasNet's choice to tokens actually sampled at this position. "
            "Those strings are ones the target itself emitted here, so splicing one in "
            "cannot create text the target would never tokenize that way. Needed when "
            "the proxy vocabulary is not the target's own."
        ),
    )
    parser.add_argument(
        "--require_faithful_support",
        action="store_true",
        help=(
            "With --restrict_to_observed_support, additionally drop support entries "
            "whose token does not decode back to a full continuation the target "
            "returned. Multi-token returns are recorded as their first token only, so "
            "such an entry is a prefix the target never emitted on its own."
        ),
    )
    parser.add_argument("--store_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--api_key_env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--api_key_file", type=str, default=None)
    parser.add_argument("--api_url", type=str, default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--site_url", type=str, default="https://anonymous.invalid")
    parser.add_argument("--app_name", type=str, default="Anonymous decoding")
    parser.add_argument("--request_timeout", type=float, default=90.0)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument(
        "--max_api_request_attempts",
        type=int,
        default=None,
        help=(
            "Hard global cap on outbound HTTP attempts, including failed and "
            "retried requests. The request that would cross the cap is not sent."
        ),
    )
    parser.add_argument(
        "--max_mc_requested_samples",
        type=int,
        default=None,
        help=(
            "Hard global cap on requested MC choices, including refill choices. "
            "The batch that would cross the cap is not sent."
        ),
    )
    parser.add_argument(
        "--max_reasoning_retries",
        type=int,
        default=0,
        help=(
            "Discard and retry a provider response that exposes reasoning while "
            "--reject_reasoning_tokens is active. Zero preserves fail-immediate behavior."
        ),
    )
    parser.add_argument(
        "--reasoning_retry_sleep",
        type=float,
        default=0.0,
        help="Seconds to wait before retrying a discarded reasoning response.",
    )
    parser.add_argument(
        "--reasoning_fallback_temperature",
        type=float,
        default=None,
        help=(
            "Opt-in temperature for a visible-only fallback after a temperature=0 "
            "request exhausts --max_reasoning_retries. The same prefix, routing, "
            "token budget, and strict reasoning rejection are preserved."
        ),
    )
    parser.add_argument(
        "--reasoning_fallback_mode",
        choices=["effort_none"],
        default=None,
        help=(
            "Opt in to a request-level reasoning-control fallback after a "
            "temperature=0 request exhausts --max_reasoning_retries. The "
            "primary --reasoning_mode must be enabled_false; fallback requests "
            "use effort_none at the same temperature. Mutually exclusive with "
            "--reasoning_fallback_temperature."
        ),
    )
    parser.add_argument(
        "--max_reasoning_fallback_retries",
        type=int,
        default=0,
        help=(
            "Additional reasoning-response retries after the first fallback "
            "request. Requires --reasoning_fallback_temperature or "
            "--reasoning_fallback_mode."
        ),
    )
    parser.add_argument("--parallel_requests", type=int, default=8)
    parser.add_argument(
        "--sample_choices_per_request",
        type=int,
        default=1,
        help="Request this many sampled choices per OpenRouter call for MC reconstruction.",
    )
    parser.add_argument(
        "--sample_completion_policy",
        choices=["partial", "exact"],
        default="partial",
        help="Accept partial MC samples or refill/fail unless exactly MCN valid samples are obtained.",
    )
    parser.add_argument(
        "--max_sample_refill_rounds",
        type=int,
        default=2,
        help="Maximum single-choice refill rounds for exact MC sampling.",
    )
    parser.add_argument("--api_max_tokens", type=int, default=2)
    parser.add_argument(
        "--empty_length_retry_max_tokens",
        type=int,
        default=None,
        help=(
            "Optional elevated max_tokens cap for retrying an empty response whose "
            "normalized finish_reason is length. Disabled unless "
            "--max_empty_length_retry_rounds is positive."
        ),
    )
    parser.add_argument(
        "--max_empty_length_retry_rounds",
        type=int,
        default=0,
        help=(
            "Maximum adaptive retry rounds for empty length-truncated responses. "
            "Zero preserves the legacy behavior."
        ),
    )
    parser.add_argument(
        "--empty_response_token",
        choices=["skip", "eos", "stop_eos"],
        default="skip",
        help=(
            "How to map empty OpenRouter completions. stop_eos maps only a normalized "
            "finish_reason=stop to the tokenizer's canonical EOS token."
        ),
    )
    parser.add_argument(
        "--disable_openrouter_response_cache",
        action="store_true",
        help="Send X-OpenRouter-Cache: false so stochastic MC requests cannot be replayed.",
    )
    parser.add_argument("--delay_seconds", type=float, default=0.0)
    parser.add_argument(
        "--reasoning_mode",
        choices=["enabled_false", "effort_none", "omit"],
        default="enabled_false",
    )
    parser.add_argument(
        "--append_no_think",
        action="store_true",
        help="Append Qwen's /no_think soft switch to user prompts before API calls.",
    )
    parser.add_argument(
        "--qwen_hard_no_think_prefill",
        action="store_true",
        help=(
            "Prefix every Qwen assistant continuation with its canonical empty think "
            "block followed by the current visible answer prefix."
        ),
    )
    parser.add_argument(
        "--reject_reasoning_tokens",
        action="store_true",
        help=(
            "Fail closed when a next-token request returns message reasoning or "
            "reports nonzero completion reasoning tokens."
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
        help="Record OpenRouter's actual provider/model routing in each output row.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--risk_gate_checkpoint", type=str, default=None)
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
        "--risk_gate_prompt_source",
        choices=["target", "dataset"],
        default="target",
        help=(
            "Prompt text supplied to the risk gate. 'target' uses the exact target/API "
            "prompt (including an appended /no_think switch); 'dataset' uses the "
            "unmodified benchmark prompt while leaving target requests unchanged."
        ),
    )
    parser.add_argument(
        "--risk_gate_mode",
        choices=["hard", "soft"],
        default="hard",
        help=(
            "How risk scores control the BiasNet residual. 'hard' preserves the binary "
            "threshold gate; 'soft' scales the residual with a sigmoid around the threshold."
        ),
    )
    parser.add_argument(
        "--risk_gate_soft_temperature",
        type=float,
        default=0.05,
        help=(
            "Sigmoid temperature for --risk_gate_mode soft. Smaller values approach "
            "the hard threshold."
        ),
    )
    parser.add_argument(
        "--risk_gate_warmup_tokens",
        type=int,
        default=0,
        help="Apply BiasNet without risk-gate blocking for the first N sampled generation steps.",
    )
    parser.add_argument(
        "--risk_gate_min_scale",
        type=float,
        default=0.0,
        help=(
            "Skip MC sampling and use the deterministic base token when the soft-gate "
            "residual scale is at or below this value. Zero preserves legacy behavior."
        ),
    )
    parser.add_argument(
        "--risk_gate_latch_off",
        action="store_true",
        help=(
            "Permanently disable BiasNet after consecutive gate scores reach the "
            "configured latch threshold."
        ),
    )
    parser.add_argument(
        "--risk_gate_latch_patience",
        type=int,
        default=2,
        help="Consecutive high-risk gate decisions required before latching BiasNet off.",
    )
    parser.add_argument(
        "--risk_gate_latch_threshold",
        type=float,
        default=None,
        help=(
            "Risk score at or above which a latch-off decision counts. Defaults to "
            "--risk_gate_threshold."
        ),
    )
    parser.add_argument(
        "--risk_gate_handoff_on_latch",
        action="store_true",
        help=(
            "After BiasNet latches off, generate the remaining token budget in one "
            "OpenRouter continuation request using the established assistant prefix."
        ),
    )
    parser.add_argument(
        "--risk_gate_speculative_draft",
        action="store_true",
        help=(
            "After consecutive base-only cutoff decisions, draft multiple base-model "
            "tokens in one request, verify every drafted prefix with the risk gate, "
            "and roll back at the first token that requires BiasNet."
        ),
    )
    parser.add_argument(
        "--risk_gate_speculative_min_base_streak",
        type=int,
        default=2,
        help="Consecutive below-min-scale base tokens required before speculative drafting.",
    )
    parser.add_argument(
        "--risk_gate_speculative_draft_tokens",
        type=int,
        default=80,
        help="Maximum tokens requested by each speculative base-model draft.",
    )
    parser.add_argument(
        "--risk_gate_trace",
        action="store_true",
        help="Store per-token risk score, residual scale, latch state, and handoff metadata.",
    )
    parser.add_argument(
        "--generation_trace",
        action="store_true",
        help=(
            "Store the same per-token runtime audit even when no risk gate is loaded. "
            "This is useful for base-only and fixed BiasNet-budget ablations."
        ),
    )
    return parser.parse_args()


def collect_prompt_records(args: argparse.Namespace) -> list[PromptRecord]:
    return load_prompt_records(
        prompt=args.prompt,
        prompts=args.prompts,
        prompt_file=args.prompt_file,
        benchmark=args.benchmark,
        benchmark_file=args.benchmark_file,
        benchmark_mutation=args.benchmark_mutation,
    )


def collect_prompts(args: argparse.Namespace) -> list[str]:
    """Compatibility wrapper returning only prompt text."""

    return [record.prompt for record in collect_prompt_records(args)]


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def resolve_device(value: Optional[str]) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_biasnet(path: Optional[str], device: torch.device, dtype_name: str) -> Optional[BiasNet]:
    if not path:
        return None
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    model = BiasNet.from_pretrained(path, map_location="cpu")
    model = model.to(device=device, dtype=dtype)
    model.set_up_proj()
    model.eval()
    return model


def tokenizer_mapping_sha256(tokenizer) -> str:
    """Hash the exact token-to-id map using the cache/materializer schema."""

    vocabulary = tokenizer.get_vocab()
    tokens = sorted(
        ([int(token_id), str(token)] for token, token_id in vocabulary.items()),
        key=lambda item: (item[0], item[1]),
    )
    payload = {
        "schema": "token_to_id_v1",
        "vocab_size": len(tokenizer),
        "tokens": tokens,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_shared_tokenizer_mapping(shared_tokenizer, proxy_tokenizer) -> str:
    """Fail closed unless both tokenizers define one identical shared ID space."""

    shared_size = len(shared_tokenizer)
    proxy_size = len(proxy_tokenizer)
    if proxy_size != shared_size:
        raise ValueError(
            "Proxy tokenizer vocabulary size does not match the OpenRouter/shared "
            f"tokenizer: proxy={proxy_size}, shared={shared_size}."
        )
    shared_vocab = {
        str(token): int(token_id)
        for token, token_id in shared_tokenizer.get_vocab().items()
    }
    proxy_vocab = {
        str(token): int(token_id)
        for token, token_id in proxy_tokenizer.get_vocab().items()
    }
    if shared_vocab != proxy_vocab:
        only_shared = sorted(set(shared_vocab) - set(proxy_vocab))[:3]
        only_proxy = sorted(set(proxy_vocab) - set(shared_vocab))[:3]
        mismatched = sorted(
            token
            for token in set(shared_vocab).intersection(proxy_vocab)
            if shared_vocab[token] != proxy_vocab[token]
        )[:3]
        raise ValueError(
            "Proxy tokenizer token-to-ID mapping does not exactly match the "
            "OpenRouter/shared tokenizer "
            f"(only_shared={only_shared}, only_proxy={only_proxy}, "
            f"mismatched_ids={mismatched})."
        )
    # Chat-template agreement is NOT checked here. Requiring proxy == shared only
    # makes sense when proxy and target are the same family. For a black-box target
    # the shared tokenizer is a borrowed proxy vocabulary whose template has nothing
    # to do with the target. What actually has to hold is that the proxy renders
    # prompts the same way it did when the cache was built, which is asserted
    # against the checkpoint's recorded proxy_chat_template_sha256 instead.
    for field in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        shared_value = getattr(shared_tokenizer, field, None)
        proxy_value = getattr(proxy_tokenizer, field, None)
        if shared_value != proxy_value:
            raise ValueError(
                f"Proxy tokenizer {field} does not match the shared tokenizer: "
                f"proxy={proxy_value!r}, shared={shared_value!r}."
            )
    expected_ids = set(range(shared_size))
    actual_ids = set(shared_vocab.values())
    if actual_ids != expected_ids:
        raise ValueError(
            "Shared tokenizer IDs must be contiguous over [0, len(tokenizer)); "
            f"missing={sorted(expected_ids - actual_ids)[:3]}, "
            f"extra={sorted(actual_ids - expected_ids)[:3]}."
        )
    shared_hash = tokenizer_mapping_sha256(shared_tokenizer)
    proxy_hash = tokenizer_mapping_sha256(proxy_tokenizer)
    if proxy_hash != shared_hash:
        raise ValueError(
            "Proxy and shared tokenizer hashes differ despite mapping validation: "
            f"proxy={proxy_hash}, shared={shared_hash}."
        )
    return shared_hash


def _proxy_torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported proxy dtype: {name}.")


@dataclass
class LocalProxyRuntime:
    model: Any
    tokenizer: Any
    device: torch.device
    model_name_or_path: str
    model_revision: Optional[str]
    tokenizer_sha256: str
    chat_template_sha256: str
    shared_chat_template_sha256: str
    shared_vocab_size: int
    proxy_vocab_size: int
    proxy_dtype: str
    proxy_quantization: str = "none"
    supports_logits_to_keep: bool = False
    calls: int = 0
    total_latency_seconds: float = 0.0
    total_input_tokens: int = 0

    def configuration(self, args: argparse.Namespace) -> dict[str, Any]:
        return {
            "mc_fusion_mode": PROXY_FUSION_MODE,
            "proxy_model_name_or_path": self.model_name_or_path,
            "proxy_model_revision": self.model_revision,
            "proxy_tokenizer_sha256": self.tokenizer_sha256,
            "proxy_temperature": float(args.proxy_temperature),
            "proxy_prior_strength": float(args.proxy_prior_strength),
            "proxy_chat_template_protocol": expected_proxy_chat_template_protocol(args),
            "shared_vocab_size": self.shared_vocab_size,
            "proxy_vocab_size": self.proxy_vocab_size,
            "proxy_vocab_tail_policy": PROXY_VOCAB_TAIL_POLICY,
            "proxy_dtype": self.proxy_dtype,
            "proxy_quantization": self.proxy_quantization,
            "proxy_device": str(self.device),
            "forward_mode": "full_prefix_no_kv_cache",
        }

    def next_token_logits(
        self,
        prompt: str,
        prefix_text: str,
        *,
        qwen_hard_no_think_prefill: bool,
    ) -> torch.Tensor:
        """Run one full-prefix proxy forward under the OpenRouter text protocol."""

        messages = messages_for_prefix(
            prompt,
            prefix_text,
            qwen_hard_no_think_prefill=qwen_hard_no_think_prefill,
        )
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_tensors": "pt",
        }
        if messages[-1]["role"] == "assistant":
            template_kwargs["continue_final_message"] = True
        else:
            template_kwargs["add_generation_prompt"] = True
        encoded = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        if isinstance(encoded, dict):
            input_ids = encoded["input_ids"]
        elif hasattr(encoded, "input_ids"):
            # Transformers may return a BatchEncoding, which exposes the
            # tensor as an attribute but is not guaranteed to subclass dict.
            input_ids = encoded.input_ids
        else:
            input_ids = encoded
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] <= 0:
            raise RuntimeError(
                "Proxy chat template must produce one non-empty [1, sequence] input."
            )
        input_ids = input_ids.to(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            if self.supports_logits_to_keep:
                outputs = self.model(input_ids=input_ids, logits_to_keep=1)
            else:
                outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[0, -1].detach()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        if logits.dim() != 1 or logits.shape[0] != self.proxy_vocab_size:
            raise RuntimeError(
                "Proxy output vocabulary changed after loading: "
                f"runtime={tuple(logits.shape)}, expected=({self.proxy_vocab_size},)."
            )
        self.calls += 1
        self.total_latency_seconds += elapsed
        self.total_input_tokens += int(input_ids.shape[1])
        return logits


def load_local_proxy(
    args: argparse.Namespace,
    shared_tokenizer,
    default_device: torch.device,
) -> Optional[LocalProxyRuntime]:
    """Load and strictly align the opt-in local causal-LM proxy."""

    model_name = getattr(args, "proxy_model_name_or_path", None)
    if not model_name:
        return None
    import inspect
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.proxy_device) if args.proxy_device else default_device
    common_kwargs = {
        "revision": args.proxy_model_revision,
        "trust_remote_code": bool(args.proxy_trust_remote_code),
        "local_files_only": bool(args.proxy_local_files_only),
    }
    proxy_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        **common_kwargs,
    )
    tokenizer_sha256 = validate_shared_tokenizer_mapping(
        shared_tokenizer,
        proxy_tokenizer,
    )
    proxy_chat_template_sha = chat_template_sha256(proxy_tokenizer)
    shared_chat_template_sha = chat_template_sha256(shared_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=_proxy_torch_dtype(args.proxy_dtype),
        **common_kwargs,
    )
    model = model.to(device)
    model.eval()
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Proxy model does not expose a causal-LM output embedding.")
    proxy_vocab_size = int(output_embeddings.weight.shape[0])
    effective_proxy_dtype = str(output_embeddings.weight.dtype).replace("torch.", "")
    if effective_proxy_dtype != str(args.proxy_dtype):
        raise ValueError(
            "Proxy effective dtype does not match --proxy_dtype: "
            f"runtime={effective_proxy_dtype!r}, requested={args.proxy_dtype!r}."
        )
    shared_vocab_size = len(shared_tokenizer)
    if proxy_vocab_size < shared_vocab_size:
        raise ValueError(
            "Proxy output vocabulary is smaller than the shared vocabulary: "
            f"proxy={proxy_vocab_size}, shared={shared_vocab_size}."
        )
    supports_logits_to_keep = (
        "logits_to_keep" in inspect.signature(model.forward).parameters
    )
    return LocalProxyRuntime(
        model=model,
        tokenizer=proxy_tokenizer,
        device=device,
        model_name_or_path=str(model_name),
        model_revision=args.proxy_model_revision,
        tokenizer_sha256=tokenizer_sha256,
        chat_template_sha256=proxy_chat_template_sha,
        shared_chat_template_sha256=shared_chat_template_sha,
        shared_vocab_size=shared_vocab_size,
        proxy_vocab_size=proxy_vocab_size,
        proxy_dtype=effective_proxy_dtype,
        proxy_quantization=str(args.proxy_quantization),
        supports_logits_to_keep=supports_logits_to_keep,
    )


def resolve_mc_input_configuration(
    args: argparse.Namespace,
    models: list[Optional[BiasNet]],
) -> None:
    """Resolve and fail-closed validate MC features against BiasNet configs."""

    active_models = [model for model in models if model is not None]
    declared_representations = {
        getattr(model.config, "mc_input_representation", None)
        for model in active_models
    }
    if len(declared_representations) > 1:
        raise ValueError(
            "Main and bootstrap BiasNet checkpoints declare different MC input "
            f"representations: {declared_representations}."
        )
    declared = next(iter(declared_representations), None)
    # Checkpoints predating this feature used the legacy materialized log-prob
    # representation.  Only that backward-compatible case may omit the field.
    declared = declared or FLOOR_LOGPROB
    requested = args.mc_input_representation or declared
    if requested != declared:
        raise ValueError(
            "MC input representation does not match the BiasNet checkpoint: "
            f"requested={requested}, checkpoint={declared}."
        )

    declared_base_representations = {
        getattr(model.config, "mc_base_score_representation", None)
        or getattr(model.config, "mc_input_representation", None)
        or FLOOR_LOGPROB
        for model in active_models
    }
    if len(declared_base_representations) > 1:
        raise ValueError(
            "Main and bootstrap BiasNet checkpoints declare different MC base "
            f"score representations: {declared_base_representations}."
        )
    declared_base = next(iter(declared_base_representations), declared)
    requested_base = (
        getattr(args, "mc_base_score_representation", None) or declared_base
    )
    if requested_base != declared_base:
        raise ValueError(
            "MC base score representation does not match the BiasNet checkpoint: "
            f"requested={requested_base}, checkpoint={declared_base}."
        )
    declared_interfaces = {
        getattr(model.config, "mc_score_interface", "shared_v1")
        for model in active_models
    }
    if len(declared_interfaces) > 1:
        raise ValueError(
            "Main and bootstrap BiasNet checkpoints declare different MC score "
            f"interfaces: {declared_interfaces}."
        )
    declared_interface = next(iter(declared_interfaces), "shared_v1")
    expected_interface = (
        "dual_v1" if requested != requested_base else "shared_v1"
    )
    if declared_interface != expected_interface:
        raise ValueError(
            "BiasNet checkpoint MC score interface is inconsistent with its "
            f"feature/base representations: {declared_interface} != {expected_interface}."
        )

    architecture_fields = (
        "input_projection_mode",
        "input_hidden_normalization",
        "input_layer_norm_eps",
        "count_sketch_hashes",
        "count_sketch_seed",
        "count_sketch_input_centering",
    )
    for field in architecture_fields:
        values = {getattr(model.config, field, None) for model in active_models}
        if len(values) > 1:
            raise ValueError(
                f"Main and bootstrap BiasNet checkpoints disagree on {field}: {values}."
            )

    if LOG_COUNT in {requested, requested_base}:
        declared_alphas = {
            getattr(model.config, "mc_log_count_alpha", None)
            for model in active_models
        }
        if None in declared_alphas or len(declared_alphas) != 1:
            raise ValueError(
                "Every log_count BiasNet checkpoint must declare one matching "
                "mc_log_count_alpha."
            )
        declared_alpha = validate_log_count_alpha(next(iter(declared_alphas)))
        requested_alpha = (
            declared_alpha
            if args.mc_log_count_alpha is None
            else validate_log_count_alpha(args.mc_log_count_alpha)
        )
        if requested_alpha != declared_alpha:
            raise ValueError(
                "MC log-count alpha does not match the BiasNet checkpoint: "
                f"requested={requested_alpha}, checkpoint={declared_alpha}."
            )
        if args.sample_completion_policy != "exact":
            raise ValueError("log_count BiasNet inference requires exact MC sampling.")
        declared_sample_counts = {
            getattr(model.config, "mc_samples_per_token", None)
            for model in active_models
        }
        if None in declared_sample_counts or len(declared_sample_counts) != 1:
            raise ValueError(
                "Every log_count BiasNet checkpoint must declare one matching "
                "mc_samples_per_token value."
            )
        declared_samples = int(next(iter(declared_sample_counts)))
        if int(args.mc_samples_per_token) != declared_samples:
            raise ValueError(
                "MC sample count does not match the BiasNet checkpoint: "
                f"requested={args.mc_samples_per_token}, checkpoint={declared_samples}."
            )
        if requested_base == LOG_COUNT and args.temperature != 0:
            raise ValueError(
                "log_count BiasNet inference currently requires greedy decoding "
                "(--temperature 0); full-vocabulary sampling would allocate mass "
                "to the zero-baseline unseen coordinates."
            )
        args.mc_log_count_alpha = declared_alpha
    else:
        if args.mc_log_count_alpha is not None:
            raise ValueError(
                "--mc_log_count_alpha is only valid with --mc_input_representation=log_count."
            )
    args.mc_input_representation = requested
    args.mc_base_score_representation = requested_base

    if declared_interface == "dual_v1":
        schema_versions = {
            getattr(model.config, "mc_estimator_schema_version", None)
            for model in active_models
        }
        if schema_versions != {2}:
            raise ValueError(
                "dual_v1 BiasNet checkpoints require mc_estimator_schema_version=2."
            )
        runtime_fields = {
            "mc_observed_alpha": float(args.mc_observed_alpha),
            "mc_floor_mass": float(args.mc_floor_mass),
            "mc_sample_temperature": float(args.mc_sample_temperature),
            "mc_top_p": float(args.mc_top_p),
            "mc_completion_policy": str(args.sample_completion_policy),
        }
        for field, runtime_value in runtime_fields.items():
            checkpoint_values = {
                getattr(model.config, field, None) for model in active_models
            }
            if len(checkpoint_values) != 1 or None in checkpoint_values:
                raise ValueError(
                    f"Every dual_v1 checkpoint must declare one matching {field}."
                )
            checkpoint_value = next(iter(checkpoint_values))
            if isinstance(runtime_value, float):
                matches = math.isclose(
                    float(checkpoint_value), runtime_value, rel_tol=0.0, abs_tol=1e-12
                )
            else:
                matches = str(checkpoint_value) == runtime_value
            if not matches:
                raise ValueError(
                    f"Runtime {field} does not match the BiasNet checkpoint: "
                    f"runtime={runtime_value!r}, checkpoint={checkpoint_value!r}."
                )


PROXY_FUSION_CHECKPOINT_FIELDS = (
    "mc_fusion_mode",
    "proxy_model_name_or_path",
    "proxy_model_revision",
    "proxy_tokenizer_sha256",
    "proxy_temperature",
    "proxy_prior_strength",
    "proxy_chat_template_protocol",
    "shared_vocab_size",
    "proxy_vocab_size",
    "proxy_vocab_tail_policy",
    "proxy_dtype",
    "proxy_quantization",
)


def _one_checkpoint_value(models: list[BiasNet], field: str) -> Any:
    values = [getattr(model.config, field, None) for model in models]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(
            "Main and bootstrap BiasNet checkpoints disagree on proxy fusion "
            f"field {field}: {values}."
        )
    return first


def validate_support_restriction(args: argparse.Namespace) -> None:
    """Reject combinations where the support restriction would silently not apply.

    The speculative-draft path picks its controlled token through a separate decode
    site that does not consult the observed support. Failing here beats appearing to
    restrict decoding while that path quietly ignores it.
    """

    if not getattr(args, "restrict_to_observed_support", False):
        if getattr(args, "require_faithful_support", False):
            raise ValueError(
                "--require_faithful_support has no effect without "
                "--restrict_to_observed_support."
            )
        return
    if getattr(args, "risk_gate_speculative_draft", False):
        raise ValueError(
            "--restrict_to_observed_support is not supported together with "
            "--risk_gate_speculative_draft: the speculative decode site does not "
            "consult the observed support."
        )


def validate_anytime_runtime(
    args: argparse.Namespace,
    policy: Optional["AnytimePolicy"],
    models: list[Optional[BiasNet]],
    runtime_device: Optional[torch.device] = None,
) -> None:
    active_models = [model for model in models if model is not None]
    declares_anytime = any(
        getattr(model.config, "anytime_schema_version", None) is not None
        for model in active_models
    )
    if policy is None:
        if declares_anytime:
            raise ValueError(
                "An anytime BiasNet checkpoint requires --anytime_policy_spec."
            )
        return
    if len(active_models) != 1:
        raise ValueError("Anytime stopping requires exactly one BiasNet checkpoint.")
    model = active_models[0]
    if getattr(args, "risk_gate_checkpoint", None):
        raise ValueError("Anytime stopping replaces, and cannot use, the risk gate.")
    if getattr(args, "biasnet_max_tokens", None) is not None:
        raise ValueError("Anytime stopping cannot use --biasnet_max_tokens.")
    if getattr(args, "initial_prefix", ""):
        raise ValueError(
            "Anytime stopping was calibrated from response position zero and "
            "cannot use --initial_prefix."
        )
    if float(args.temperature) != 0.0:
        raise ValueError("Anytime stopping requires greedy --temperature 0.")
    if args.sample_completion_policy != "exact":
        raise ValueError("Anytime stopping requires exact MC sampling.")
    if int(getattr(args, "mc_independent_views", 1)) != 1:
        raise ValueError("Anytime stopping supports exactly one nested MC view.")
    if not bool(getattr(args, "disable_openrouter_response_cache", False)):
        raise ValueError("Anytime stopping requires the OpenRouter cache to be disabled.")
    if int(getattr(args, "sample_choices_per_request", 1)) != 1:
        raise ValueError(
            "The validated anytime runtime currently requires one choice per request."
        )
    if getattr(args, "restrict_to_observed_support", False):
        raise ValueError(
            "Anytime K=0 stopping is incompatible with observed-support restriction."
        )
    if getattr(args, "require_faithful_support", False):
        raise ValueError("Anytime stopping cannot require sampled support.")
    if getattr(args, "mask_non_eos_special_tokens", False):
        raise ValueError(
            "Anytime stopping must use the same unmasked action space as calibration."
        )
    if not getattr(args, "proxy_model_name_or_path", None):
        raise ValueError("Anytime stopping requires the calibrated local proxy.")
    if str(getattr(args, "biasnet_dtype", "")) != policy.biasnet_parameter_dtype:
        raise ValueError(
            "Anytime runtime --biasnet_dtype does not match calibration: "
            f"{getattr(args, 'biasnet_dtype', None)!r} != "
            f"{policy.biasnet_parameter_dtype!r}."
        )
    parameter_dtypes = {
        str(parameter.dtype).replace("torch.", "")
        for parameter in model.parameters()
    }
    if parameter_dtypes != {policy.biasnet_parameter_dtype}:
        raise ValueError(
            "Loaded anytime BiasNet parameter dtypes do not match calibration: "
            f"{sorted(parameter_dtypes)!r} != "
            f"{policy.biasnet_parameter_dtype!r}."
        )
    if runtime_device is None:
        runtime_device = next(model.parameters()).device
    if policy.biasnet_autocast_enabled and runtime_device.type != "cuda":
        raise ValueError(
            "Anytime runtime requires CUDA because its calibrated BiasNet "
            "autocast protocol is enabled."
        )
    if int(args.max_new_tokens) > policy.position_normalizer + 1:
        raise ValueError(
            "--max_new_tokens exceeds the frozen anytime position range."
        )
    expected_runtime = {
        "mc_sample_temperature": float(args.mc_sample_temperature),
        "mc_top_p": float(args.mc_top_p),
        "mc_completion_policy": str(args.sample_completion_policy),
    }
    for field, runtime_value in expected_runtime.items():
        checkpoint_value = getattr(model.config, field, None)
        if isinstance(runtime_value, float):
            matches = checkpoint_value is not None and math.isclose(
                float(checkpoint_value), runtime_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = checkpoint_value == runtime_value
        if not matches:
            raise ValueError(
                f"Anytime runtime {field} does not match the checkpoint: "
                f"{runtime_value!r} != {checkpoint_value!r}."
            )

    def optional_list(value):
        return None if value is None else list(value)

    runtime_target_protocol = {
        "target_protocol_schema_version": 1,
        "target_api_url": str(args.api_url),
        "target_model": str(args.model),
        "target_provider_order": optional_list(
            getattr(args, "provider_order", None)
        ),
        "target_provider_allow_fallbacks": getattr(
            args, "provider_allow_fallbacks", None
        ),
        "target_provider_quantizations": optional_list(
            getattr(args, "provider_quantizations", None)
        ),
        "target_append_no_think": bool(
            getattr(args, "append_no_think", False)
        ),
        "target_qwen_hard_no_think_prefill": bool(
            getattr(args, "qwen_hard_no_think_prefill", False)
        ),
        "target_api_max_tokens": int(args.api_max_tokens),
        "target_empty_response_token": str(args.empty_response_token),
        "target_disable_openrouter_response_cache": bool(
            getattr(args, "disable_openrouter_response_cache", False)
        ),
        "target_sample_choices_per_request": int(
            getattr(args, "sample_choices_per_request", 1)
        ),
        "target_max_sample_refill_rounds": int(
            getattr(args, "max_sample_refill_rounds", 2)
        ),
        "target_empty_length_retry_max_tokens": getattr(
            args, "empty_length_retry_max_tokens", None
        ),
        "target_max_empty_length_retry_rounds": int(
            getattr(args, "max_empty_length_retry_rounds", 0)
        ),
        "target_reasoning_mode": str(
            getattr(args, "reasoning_mode", "enabled_false")
        ),
        "target_reject_reasoning_tokens": bool(
            getattr(args, "reject_reasoning_tokens", False)
        ),
        "target_max_reasoning_retries": int(
            getattr(args, "max_reasoning_retries", 0)
        ),
        "target_reasoning_fallback_temperature": getattr(
            args, "reasoning_fallback_temperature", None
        ),
        "target_reasoning_fallback_mode": getattr(
            args, "reasoning_fallback_mode", None
        ),
        "target_max_reasoning_fallback_retries": int(
            getattr(args, "max_reasoning_fallback_retries", 0)
        ),
    }
    if runtime_target_protocol != policy.target_protocol:
        differences = {
            field: {
                "runtime": runtime_target_protocol.get(field),
                "calibrated": policy.target_protocol.get(field),
            }
            for field in sorted(
                set(runtime_target_protocol) | set(policy.target_protocol)
            )
            if runtime_target_protocol.get(field)
            != policy.target_protocol.get(field)
        }
        raise ValueError(
            "Anytime target sampling protocol does not match calibration: "
            f"{differences}."
        )


def resolve_proxy_fusion_configuration(
    args: argparse.Namespace,
    models: list[Optional[BiasNet]],
) -> None:
    """Fail closed on legacy/fused checkpoint and runtime mismatches."""

    active_models = [model for model in models if model is not None]
    proxy_requested = bool(getattr(args, "proxy_model_name_or_path", None))
    if not active_models:
        if proxy_requested:
            raise ValueError("--proxy_model_name_or_path requires a BiasNet checkpoint.")
        return

    declared_mode = _one_checkpoint_value(active_models, "mc_fusion_mode")
    if declared_mode is None:
        partial_fields = [
            field
            for field in PROXY_FUSION_CHECKPOINT_FIELDS[1:]
            if any(getattr(model.config, field, None) is not None for model in active_models)
        ]
        if partial_fields:
            raise ValueError(
                "BiasNet checkpoint contains partial proxy fusion metadata but no "
                f"mc_fusion_mode: {partial_fields}."
            )
        if proxy_requested:
            raise ValueError(
                "A local proxy cannot be used with a legacy BiasNet checkpoint; "
                f"the checkpoint must declare mc_fusion_mode={PROXY_FUSION_MODE!r}."
            )
        return
    if declared_mode != PROXY_FUSION_MODE:
        raise ValueError(f"Unsupported checkpoint mc_fusion_mode: {declared_mode!r}.")
    if not proxy_requested:
        raise ValueError(
            f"BiasNet checkpoint requires {PROXY_FUSION_MODE}; provide "
            "--proxy_model_name_or_path."
        )

    for field in PROXY_FUSION_CHECKPOINT_FIELDS:
        value = _one_checkpoint_value(active_models, field)
        if field != "proxy_model_revision" and value is None:
            raise ValueError(
                f"Every fused BiasNet checkpoint must declare {field}."
            )

    temperature = float(getattr(args, "proxy_temperature", 1.0))
    prior_strength = float(getattr(args, "proxy_prior_strength", 1.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("--proxy_temperature must be finite and positive.")
    if not math.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("--proxy_prior_strength must be finite and positive.")

    expected_values = {
        "mc_fusion_mode": PROXY_FUSION_MODE,
        "proxy_model_name_or_path": str(args.proxy_model_name_or_path),
        "proxy_model_revision": getattr(args, "proxy_model_revision", None),
        "proxy_temperature": temperature,
        "proxy_prior_strength": prior_strength,
        "proxy_chat_template_protocol": expected_proxy_chat_template_protocol(args),
        "proxy_vocab_tail_policy": PROXY_VOCAB_TAIL_POLICY,
        "proxy_dtype": str(args.proxy_dtype),
        "proxy_quantization": str(args.proxy_quantization),
    }
    for field, runtime_value in expected_values.items():
        checkpoint_value = _one_checkpoint_value(active_models, field)
        if isinstance(runtime_value, float):
            matches = math.isclose(
                float(checkpoint_value), runtime_value, rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = checkpoint_value == runtime_value
        if not matches:
            raise ValueError(
                f"Runtime {field} does not match the BiasNet checkpoint: "
                f"runtime={runtime_value!r}, checkpoint={checkpoint_value!r}."
            )
    if FLOOR_LOGPROB not in {
        getattr(args, "mc_input_representation", FLOOR_LOGPROB),
        getattr(args, "mc_base_score_representation", FLOOR_LOGPROB),
    }:
        raise ValueError(
            "proxy_dirichlet_v1 requires floor_logprob as a BiasNet feature or "
            "base-score representation."
        )


def resolve_static_prior_configuration(
    args: argparse.Namespace,
    models: list[Optional[BiasNet]],
) -> None:
    """Resolve fixed no-proxy Dirichlet priors recorded by a BiasNet checkpoint."""

    active_models = [model for model in models if model is not None]
    declared_modes = {
        getattr(model.config, "mc_static_prior_mode", None)
        for model in active_models
    }
    if len(declared_modes) > 1:
        raise ValueError(
            "Main and bootstrap BiasNet checkpoints disagree on static MC prior mode: "
            f"{declared_modes}."
        )
    declared_mode = next(iter(declared_modes), None)
    if declared_mode is not None and declared_mode not in STATIC_MC_PRIOR_MODES:
        raise ValueError(f"Unsupported checkpoint static MC prior mode: {declared_mode!r}.")
    requested_mode = args.mc_static_prior_mode
    if requested_mode is None:
        requested_mode = declared_mode or "none"
    if requested_mode != (declared_mode or "none"):
        raise ValueError(
            "MC static prior mode does not match the BiasNet checkpoint: "
            f"requested={requested_mode}, checkpoint={declared_mode or 'none'}."
        )
    args.mc_static_prior_mode = requested_mode
    if requested_mode == "none":
        if args.mc_static_prior_strength is not None or args.mc_static_prior_path is not None:
            raise ValueError("Static prior strength/path require a static prior mode.")
        args.mc_static_prior_strength = None
        args.mc_static_prior_path = None
        return

    declared_strengths = {
        getattr(model.config, "mc_static_prior_strength", None)
        for model in active_models
    }
    if len(declared_strengths) > 1 or None in declared_strengths:
        raise ValueError("Static-prior checkpoints must declare one prior strength.")
    declared_strength = float(next(iter(declared_strengths)))
    requested_strength = (
        declared_strength
        if args.mc_static_prior_strength is None
        else float(args.mc_static_prior_strength)
    )
    if not math.isfinite(requested_strength) or requested_strength <= 0.0:
        raise ValueError("--mc_static_prior_strength must be finite and positive.")
    if not math.isclose(requested_strength, declared_strength, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "MC static prior strength does not match the BiasNet checkpoint: "
            f"requested={requested_strength}, checkpoint={declared_strength}."
        )
    args.mc_static_prior_strength = requested_strength
    if requested_mode == "global_unigram_dirichlet_v1":
        declared_paths = {
            getattr(model.config, "mc_static_prior_path", None)
            for model in active_models
        }
        if len(declared_paths) != 1 or None in declared_paths:
            raise ValueError("Global-unigram checkpoints must declare a prior path.")
        declared_path = str(next(iter(declared_paths)))
        requested_path = declared_path if args.mc_static_prior_path is None else str(args.mc_static_prior_path)
        if Path(requested_path).expanduser().resolve() != Path(declared_path).expanduser().resolve():
            raise ValueError(
                "MC static prior path does not match the BiasNet checkpoint: "
                f"requested={requested_path}, checkpoint={declared_path}."
            )
        args.mc_static_prior_path = str(Path(requested_path).expanduser().resolve())
    else:
        if args.mc_static_prior_path is not None:
            raise ValueError("Uniform static prior does not use --mc_static_prior_path.")
        args.mc_static_prior_path = None


def load_static_prior_log_probs(
    args: argparse.Namespace,
    vocab_size: int,
    device: torch.device,
) -> None:
    """Load and validate the fixed prior vector used by online MC scoring."""

    mode = getattr(args, "mc_static_prior_mode", "none")
    if mode == "none":
        args._mc_static_prior_log_probs = None
        return
    if mode == "uniform_dirichlet_v1":
        args._mc_static_prior_log_probs = torch.full(
            (vocab_size,), -math.log(vocab_size), dtype=torch.float32, device=device
        )
        return
    path = Path(args.mc_static_prior_path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    values = payload.get("log_probs") if isinstance(payload, dict) else payload
    if not isinstance(values, torch.Tensor) or values.dim() != 1 or values.numel() != vocab_size:
        raise ValueError(f"Static prior {path} must contain a [{vocab_size}] log_probs vector.")
    values = values.float()
    if not torch.isfinite(values).all().item():
        raise ValueError(f"Static prior {path} contains non-finite log probabilities.")
    normalizer = torch.logsumexp(values, dim=0)
    if not math.isclose(float(normalizer), 0.0, rel_tol=0.0, abs_tol=1e-4):
        raise ValueError(f"Static prior {path} is not normalized: logsumexp={float(normalizer)}")
    args._mc_static_prior_log_probs = values.to(device)


def apply_static_prior_to_mc_samples(
    sample_ids: list[int],
    args: argparse.Namespace,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return posterior-predictive log probabilities for a fixed prior."""

    prior_log_probs = getattr(args, "_mc_static_prior_log_probs", None)
    if prior_log_probs is None:
        raise RuntimeError("Static prior was not loaded before MC scoring.")
    counts = torch.bincount(
        torch.as_tensor(sample_ids, dtype=torch.long, device=device),
        minlength=vocab_size,
    ).unsqueeze(0)
    return fuse_proxy_logits_with_mc_counts(
        prior_log_probs.unsqueeze(0),
        counts,
        temperature=1.0,
        prior_strength=float(args.mc_static_prior_strength),
        dtype=dtype,
    )[0]


def validate_proxy_runtime_against_checkpoints(
    proxy: Optional[LocalProxyRuntime],
    models: list[Optional[BiasNet]],
) -> None:
    active_models = [model for model in models if model is not None]
    if proxy is None:
        return
    # A fused BiasNet learned its residual over features rendered with one specific
    # chat template. Newer caches record that template's hash, which is the property
    # that actually has to hold at inference. Checkpoints predating the field fall
    # back to the legacy proxy-equals-shared requirement so their guarantee is
    # unchanged.
    recorded_template = _one_checkpoint_value(active_models, "proxy_chat_template_sha256")
    if recorded_template is None:
        if proxy.chat_template_sha256 != proxy.shared_chat_template_sha256:
            raise ValueError(
                "Legacy fused checkpoint records no proxy_chat_template_sha256, so the "
                "proxy tokenizer chat template must equal the shared tokenizer's: "
                f"proxy={proxy.chat_template_sha256!r}, "
                f"shared={proxy.shared_chat_template_sha256!r}."
            )
    elif proxy.chat_template_sha256 != recorded_template:
        raise ValueError(
            "Runtime proxy chat template does not match the one recorded when the "
            f"fused cache was built: runtime={proxy.chat_template_sha256!r}, "
            f"checkpoint={recorded_template!r}."
        )

    runtime_values = {
        "proxy_tokenizer_sha256": proxy.tokenizer_sha256,
        "shared_vocab_size": proxy.shared_vocab_size,
        "proxy_vocab_size": proxy.proxy_vocab_size,
        "proxy_dtype": proxy.proxy_dtype,
        "proxy_quantization": proxy.proxy_quantization,
    }
    for field, runtime_value in runtime_values.items():
        checkpoint_value = _one_checkpoint_value(active_models, field)
        if checkpoint_value != runtime_value:
            raise ValueError(
                f"Runtime {field} does not match the BiasNet checkpoint: "
                f"runtime={runtime_value!r}, checkpoint={checkpoint_value!r}."
            )


def load_risk_gate(args: argparse.Namespace, default_device: torch.device) -> Optional[PrefixRiskGate]:
    if not args.risk_gate_checkpoint:
        return None
    device = torch.device(args.risk_gate_device) if args.risk_gate_device else default_device
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


def generation_token(
    log_probs: torch.Tensor,
    temperature: float,
    forbidden_token_ids: Optional[list[int]] = None,
) -> int:
    if forbidden_token_ids:
        log_probs = log_probs.clone()
        log_probs[forbidden_token_ids] = -torch.inf
    if temperature <= 0:
        return int(torch.argmax(log_probs, dim=-1).item())
    probs = torch.softmax(log_probs.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def proxy_forbidden_token_ids(tokenizer, enabled: bool) -> list[int]:
    if not enabled:
        return []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    return sorted(
        {
            int(token_id)
            for token_id in (getattr(tokenizer, "all_special_ids", None) or [])
            if eos_token_id is None or int(token_id) != int(eos_token_id)
        }
    )


@dataclass(frozen=True)
class BiasMCInputs:
    features: torch.Tensor
    base_scores: torch.Tensor
    independent_views: int = 1


@dataclass(frozen=True)
class AnytimePolicy:
    source_path: str
    source_sha256: str
    payload_sha256: str
    checkpoint_config_sha256: str
    name: str
    budgets: tuple[int, ...]
    threshold: float
    allowed_action_disagreement: float
    position_normalizer: int
    biasnet_parameter_dtype: str
    biasnet_autocast_enabled: bool
    biasnet_autocast_dtype: str
    target_protocol: dict[str, Any]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    coefficient: tuple[float, ...]
    intercept: float

    def confidence(self, features: Iterable[float]) -> float:
        values = tuple(float(value) for value in features)
        if len(values) != len(ANYTIME_FEATURE_NAMES):
            raise ValueError(
                "Anytime feature vector has the wrong width: "
                f"{len(values)} != {len(ANYTIME_FEATURE_NAMES)}."
            )
        return portable_linear_stopper_confidence(
            values,
            scaler_mean=self.scaler_mean,
            scaler_scale=self.scaler_scale,
            coefficient=self.coefficient,
            intercept=self.intercept,
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            POLICY_SPEC_HASH_FIELD: self.payload_sha256,
            "checkpoint_config_sha256": self.checkpoint_config_sha256,
            "policy_name": self.name,
            "budgets": list(self.budgets),
            "threshold": self.threshold,
            "allowed_action_disagreement": self.allowed_action_disagreement,
            "position_normalizer": self.position_normalizer,
            "biasnet_parameter_dtype": self.biasnet_parameter_dtype,
            "biasnet_autocast_enabled": self.biasnet_autocast_enabled,
            "biasnet_autocast_dtype": self.biasnet_autocast_dtype,
            # Deprecated alias retained for early schema-v2 audit readers.
            "biasnet_runtime_dtype": self.biasnet_parameter_dtype,
            "target_protocol": dict(self.target_protocol),
            "feature_names": list(ANYTIME_FEATURE_NAMES),
            "stopper": "standard_scaler_logistic_regression_v1",
        }


@dataclass(frozen=True)
class AnytimeEstimate:
    scores: torch.Tensor
    samples_used: int
    requested_choices: int
    stage_trace: tuple[dict[str, Any], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_anytime_policy(
    args: argparse.Namespace,
    bias_model: Optional[BiasNet],
    bootstrap_bias_model: Optional[BiasNet],
) -> Optional[AnytimePolicy]:
    policy_value = getattr(args, "anytime_policy_spec", None)
    if not policy_value:
        return None
    if bias_model is None:
        raise ValueError("--anytime_policy_spec requires --biasnet_ckpt.")
    if bootstrap_bias_model is not None:
        raise ValueError(
            "Anytime stopping does not support --bootstrap_biasnet_ckpt."
        )
    policy_path = Path(policy_value).expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Missing anytime policy specification: {policy_path}")
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("Unsupported anytime policy schema version.")
    recorded_payload_hash = payload.get(POLICY_SPEC_HASH_FIELD)
    actual_payload_hash = policy_spec_payload_sha256(payload)
    if recorded_payload_hash != actual_payload_hash:
        raise ValueError("Anytime policy payload hash mismatch.")
    if payload.get("feature_names") != list(ANYTIME_FEATURE_NAMES):
        raise ValueError("Anytime policy feature schema does not match the runtime.")
    checkpoint_path = Path(args.biasnet_ckpt).expanduser().resolve()
    expected_hash = payload.get("checkpoint_weights_sha256")
    actual_hash = _sha256_file(checkpoint_path / "pytorch_model.bin")
    if expected_hash != actual_hash:
        raise ValueError("Anytime policy was calibrated for a different BiasNet.")
    expected_config_hash = payload.get("checkpoint_config_sha256")
    actual_config_hash = _sha256_file(checkpoint_path / "config.json")
    if expected_config_hash != actual_config_hash:
        raise ValueError(
            "Anytime policy was calibrated with a different BiasNet config."
        )
    config = bias_model.config
    if getattr(config, "anytime_schema_version", None) != 1:
        raise ValueError("BiasNet does not declare anytime_schema_version=1.")
    if not bool(getattr(config, "mc_sample_count_conditioning", False)):
        raise ValueError("Anytime BiasNet is not conditioned on the actual K.")
    budgets = tuple(int(value) for value in payload.get("budgets") or ())
    if (
        not budgets
        or budgets[0] != 0
        or tuple(sorted(set(budgets))) != budgets
    ):
        raise ValueError("Anytime policy budgets must be increasing and start at K=0.")
    declared_budgets = tuple(
        int(value) for value in (getattr(config, "mc_sample_budgets", None) or ())
    )
    if budgets != declared_budgets:
        raise ValueError("Anytime policy and BiasNet budget schedules differ.")
    max_samples = int(getattr(config, "mc_max_samples_per_token", 0) or 0)
    if budgets[-1] != max_samples:
        raise ValueError("Anytime policy does not terminate at the BiasNet maximum K.")
    if int(getattr(args, "mc_samples_per_token", max_samples)) != max_samples:
        raise ValueError(
            "--mc_samples_per_token must equal the anytime checkpoint maximum K."
        )
    checkpoint_fingerprint = getattr(
        config, "proxy_mc_manifest_configuration_fingerprint", None
    )
    if payload.get(
        "cache_manifest_configuration_fingerprint"
    ) != checkpoint_fingerprint:
        raise ValueError("Anytime policy and BiasNet cache fingerprints differ.")
    policy_name = str(getattr(args, "anytime_policy_name", "delta_0.05"))
    operating_point = (payload.get("policies") or {}).get(policy_name)
    if not isinstance(operating_point, dict):
        raise ValueError(f"Unknown anytime policy operating point: {policy_name!r}.")
    if operating_point.get("status") != "selected_on_legacy_validation":
        raise ValueError(
            f"Anytime operating point {policy_name!r} was not feasible on validation."
        )
    estimator_metadata = payload.get("estimator_metadata")
    if not isinstance(estimator_metadata, dict):
        raise ValueError("Anytime policy has no frozen estimator metadata.")
    missing_target_fields = [
        field
        for field in TARGET_PROTOCOL_METADATA_FIELDS
        if field not in estimator_metadata
    ]
    if missing_target_fields:
        raise ValueError(
            "Anytime policy is missing target-protocol metadata: "
            f"{missing_target_fields}."
        )
    target_protocol = {
        field: estimator_metadata[field]
        for field in TARGET_PROTOCOL_METADATA_FIELDS
    }
    biasnet_parameter_dtype = payload.get("biasnet_parameter_dtype")
    legacy_runtime_dtype = payload.get("biasnet_runtime_dtype")
    if (
        biasnet_parameter_dtype != BIASNET_PARAMETER_DTYPE
        or legacy_runtime_dtype != biasnet_parameter_dtype
    ):
        raise ValueError("Anytime policy requires float32 BiasNet parameters.")
    biasnet_autocast_enabled = payload.get("biasnet_autocast_enabled")
    biasnet_autocast_dtype = payload.get("biasnet_autocast_dtype")
    if (
        biasnet_autocast_enabled is not BIASNET_AUTOCAST_ENABLED
        or biasnet_autocast_dtype != BIASNET_AUTOCAST_DTYPE
    ):
        raise ValueError("Unsupported anytime BiasNet autocast protocol.")
    linear = payload.get("linear_stopper")
    if not isinstance(linear, dict) or linear.get("schema_version") != 2:
        raise ValueError("Anytime policy has no portable linear stopper.")
    if linear.get("scoring_protocol") != PORTABLE_STOPPER_PROTOCOL:
        raise ValueError("Unsupported anytime portable stopper protocol.")
    if linear.get("classes") != [0, 1]:
        raise ValueError("Anytime stopper must be a binary [0, 1] classifier.")
    width = len(ANYTIME_FEATURE_NAMES)

    def finite_vector(field: str, *, positive: bool = False) -> tuple[float, ...]:
        vector = tuple(float(value) for value in (linear.get(field) or ()))
        if len(vector) != width or any(not math.isfinite(value) for value in vector):
            raise ValueError(f"Invalid anytime stopper vector: {field}.")
        if positive and any(value <= 0.0 for value in vector):
            raise ValueError(f"Anytime stopper {field} values must be positive.")
        return vector

    threshold = float(operating_point.get("threshold"))
    allowed_error = float(operating_point.get("allowed_action_disagreement"))
    position_normalizer = int(payload.get("position_normalizer", 0))
    intercept = float(linear.get("intercept"))
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Anytime threshold must be finite and non-negative.")
    if not math.isfinite(allowed_error) or not 0.0 <= allowed_error < 1.0:
        raise ValueError("Anytime disagreement budget must lie in [0, 1).")
    if position_normalizer <= 0:
        raise ValueError("Anytime position normalizer must be positive.")
    if not math.isfinite(intercept):
        raise ValueError("Anytime stopper intercept must be finite.")
    return AnytimePolicy(
        source_path=str(policy_path),
        source_sha256=_sha256_file(policy_path),
        payload_sha256=str(recorded_payload_hash),
        checkpoint_config_sha256=str(expected_config_hash),
        name=policy_name,
        budgets=budgets,
        threshold=threshold,
        allowed_action_disagreement=allowed_error,
        position_normalizer=position_normalizer,
        biasnet_parameter_dtype=str(biasnet_parameter_dtype),
        biasnet_autocast_enabled=bool(biasnet_autocast_enabled),
        biasnet_autocast_dtype=str(biasnet_autocast_dtype),
        target_protocol=target_protocol,
        scaler_mean=finite_vector("scaler_mean"),
        scaler_scale=finite_vector("scaler_scale", positive=True),
        coefficient=finite_vector("coefficient"),
        intercept=intercept,
    )


MC_ENSEMBLE_SCHEMA_VERSION = 1
MC_ENSEMBLE_AGGREGATION = "mean_probability_v1"


def mc_ensemble_audit(
    args: argparse.Namespace,
    anytime_policy: Optional[AnytimePolicy] = None,
) -> dict[str, Any]:
    """Describe the MC ensemble without changing per-view checkpoint semantics."""

    views = int(getattr(args, "mc_independent_views", 1))
    samples_per_view = int(getattr(args, "mc_samples_per_token", 50))
    if anytime_policy is not None:
        return {
            "schema_version": 2,
            "mode": "anytime_nested_single_view_v1",
            "independent_views": views,
            "budget_schedule": list(anytime_policy.budgets),
            "maximum_samples_per_view": samples_per_view,
            "fixed_requested_samples_per_token": None,
            "feature_construction": "nested_prefix_at_actual_k",
            "base_score_construction": "proxy_mc_fusion_at_actual_k",
            "aggregation": "single_view",
            "independence_mechanism": "separate_openrouter_calls_no_seed",
            "openrouter_response_cache_disabled": bool(
                getattr(args, "disable_openrouter_response_cache", False)
            ),
        }
    return {
        "schema_version": MC_ENSEMBLE_SCHEMA_VERSION,
        "independent_views": views,
        "samples_per_view": samples_per_view,
        "total_requested_samples_per_token": views * samples_per_view,
        "feature_construction": "checkpoint_native_per_view",
        "base_score_construction": "checkpoint_native_per_view",
        "per_view_score": "base_plus_scaled_residual",
        "aggregation": MC_ENSEMBLE_AGGREGATION,
        "independence_mechanism": "separate_openrouter_calls_no_seed",
        "openrouter_response_cache_disabled": bool(
            getattr(args, "disable_openrouter_response_cache", False)
        ),
        "single_view_compatibility": "legacy_scores",
    }


def validate_mc_ensemble_args(args: argparse.Namespace) -> None:
    """Fail closed when an opt-in ensemble cannot provide equal independent views."""

    views = int(getattr(args, "mc_independent_views", 1))
    if views <= 0:
        raise ValueError("--mc_independent_views must be positive.")
    if views == 1:
        return
    if getattr(args, "sample_completion_policy", "partial") != "exact":
        raise ValueError(
            "Multiple --mc_independent_views require "
            "--sample_completion_policy exact."
        )
    if not bool(getattr(args, "disable_openrouter_response_cache", False)):
        raise ValueError(
            "Multiple --mc_independent_views require "
            "--disable_openrouter_response_cache."
        )


def _mc_sampling_args(
    args: argparse.Namespace, samples_per_token: int
) -> argparse.Namespace:
    return argparse.Namespace(
        samples_per_token=int(samples_per_token),
        sample_temperature=args.mc_sample_temperature,
        top_p=args.mc_top_p,
        api_max_tokens=args.api_max_tokens,
        parallel_requests=args.parallel_requests,
        sample_choices_per_request=args.sample_choices_per_request,
        sample_completion_policy=args.sample_completion_policy,
        max_sample_refill_rounds=args.max_sample_refill_rounds,
        empty_length_retry_max_tokens=args.empty_length_retry_max_tokens,
        max_empty_length_retry_rounds=args.max_empty_length_retry_rounds,
        empty_response_token=args.empty_response_token,
        delay_seconds=args.delay_seconds,
        qwen_hard_no_think_prefill=bool(
            getattr(args, "qwen_hard_no_think_prefill", False)
        ),
        reject_reasoning_tokens=bool(
            getattr(args, "reject_reasoning_tokens", False)
        ),
    )


@torch.no_grad()
def estimate_anytime_scores(
    *,
    client: OpenRouterClient,
    tokenizer,
    prompt: str,
    prefix_text: str,
    args: argparse.Namespace,
    device: torch.device,
    proxy: LocalProxyRuntime,
    bias_model: BiasNet,
    base_token_id: int,
    position_id: int,
    policy: AnytimePolicy,
    support_out: Optional[list[int]] = None,
) -> AnytimeEstimate:
    """Incrementally sample one nested proxy-MC path until the frozen policy stops."""

    if position_id > policy.position_normalizer:
        raise ValueError(
            "Generation position exceeds the frozen anytime feature range: "
            f"{position_id} > {policy.position_normalizer}."
        )
    proxy_logits = proxy.next_token_logits(
        prompt,
        prefix_text,
        qwen_hard_no_think_prefill=bool(
            getattr(args, "qwen_hard_no_think_prefill", False)
        ),
    )
    all_sample_ids: list[int] = []
    sampled_text_counts: Counter[str] = Counter()
    previous_actions: Optional[torch.Tensor] = None
    stable_stages = torch.zeros(1, dtype=torch.long, device=device)
    proxy_top_ids = None
    proxy_top_probability = None
    proxy_gap = None
    selected_scores = None
    stage_trace: list[dict[str, Any]] = []
    requested_choices = 0
    use_amp = policy.biasnet_autocast_enabled
    if use_amp and device.type != "cuda":
        raise ValueError(
            "Anytime BiasNet autocast is frozen on but the inference device is not CUDA."
        )
    max_samples = policy.budgets[-1]

    for stage_index, budget in enumerate(policy.budgets):
        delta = budget - len(all_sample_ids)
        sampler_stats = None
        if delta < 0:
            raise RuntimeError("Anytime MC path exceeded its current budget.")
        if delta:
            sample_ids, sampler_stats = sample_position_token_ids(
                client=client,
                tokenizer=tokenizer,
                question=prompt,
                prefix_text=prefix_text,
                args=_mc_sampling_args(args, delta),
            )
            if len(sample_ids) != delta:
                raise RuntimeError(
                    f"Anytime stage expected {delta} exact samples, got {len(sample_ids)}."
                )
            all_sample_ids.extend(int(token_id) for token_id in sample_ids)
            requested_choices += int(
                sampler_stats.get("requested_samples", delta)
            )
            sampled_text_counts.update(
                sampler_stats.get("sampled_completion_text_counts") or {}
            )
        if len(all_sample_ids) != budget:
            raise RuntimeError("Anytime nested sampling produced the wrong cumulative K.")

        counts = torch.bincount(
            torch.as_tensor(
                all_sample_ids, dtype=torch.long, device=proxy_logits.device
            ),
            minlength=len(tokenizer),
        ).unsqueeze(0)
        fused_scores = fuse_proxy_logits_with_mc_counts(
            proxy_logits.unsqueeze(0),
            counts,
            temperature=float(args.proxy_temperature),
            prior_strength=float(args.proxy_prior_strength),
            dtype=torch.float32,
        ).to(device)
        positions = torch.tensor([position_id], dtype=torch.long, device=device)
        sample_counts = torch.tensor([budget], dtype=torch.long, device=device)
        with torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            residual = bias_model(
                fused_scores,
                position_ids=(
                    positions
                    if int(getattr(bias_model, "num_position_buckets", 0) or 0) > 0
                    else None
                ),
                mc_sample_counts=sample_counts,
            )
            controller_scores = fused_scores + residual
        fused_ids, fused_probability, fused_gap = anytime_top2_statistics(
            fused_scores
        )
        action_ids, action_probability, action_gap = anytime_top2_statistics(
            controller_scores
        )
        if stage_index == 0:
            proxy_top_ids = fused_ids
            proxy_top_probability = fused_probability
            proxy_gap = fused_gap
        assert proxy_top_ids is not None
        assert proxy_top_probability is not None and proxy_gap is not None
        current_stable = (
            torch.zeros_like(action_ids, dtype=torch.bool)
            if previous_actions is None
            else action_ids.eq(previous_actions)
        )
        stable_stages = torch.where(
            current_stable,
            stable_stages + 1,
            torch.zeros_like(stable_stages),
        )
        state_features = build_anytime_state_features(
            budget=budget,
            max_samples=max_samples,
            positions=positions,
            proxy_top_ids=proxy_top_ids,
            proxy_top_probability=proxy_top_probability,
            proxy_gap=proxy_gap,
            fused_top_ids=fused_ids,
            fused_top_probability=fused_probability,
            fused_gap=fused_gap,
            action_ids=action_ids,
            action_probability=action_probability,
            action_gap=action_gap,
            base_token_ids=torch.tensor(
                [base_token_id], dtype=torch.long, device=device
            ),
            counts=counts.to(device=device, dtype=torch.float32),
            previous_actions=previous_actions,
            stable_stages=stable_stages,
            stage_index=stage_index,
            position_normalizer=policy.position_normalizer,
        )
        feature_values = tuple(float(value) for value in state_features[0].cpu())
        forced_full_budget = budget == max_samples
        confidence = None if forced_full_budget else policy.confidence(feature_values)
        should_stop = forced_full_budget or confidence >= policy.threshold
        stage_trace.append(
            {
                "stage": stage_index,
                "budget": budget,
                "delta_samples": delta,
                "action_token_id": int(action_ids.item()),
                "confidence": confidence,
                "threshold": policy.threshold,
                "stopped": bool(should_stop),
                "stop_reason": (
                    "full_budget" if forced_full_budget else "confidence"
                ) if should_stop else None,
                "features": dict(zip(ANYTIME_FEATURE_NAMES, feature_values)),
                "sampler_requested_samples": (
                    int(sampler_stats.get("requested_samples", delta))
                    if sampler_stats is not None
                    else 0
                ),
                "sampler_valid_samples": (
                    int(sampler_stats.get("valid_samples", delta))
                    if sampler_stats is not None
                    else 0
                ),
            }
        )
        selected_scores = controller_scores
        previous_actions = action_ids
        if should_stop:
            break

    assert selected_scores is not None
    if support_out is not None:
        candidate_ids = sorted(set(all_sample_ids))
        raw_support_size = len(candidate_ids)
        if getattr(args, "require_faithful_support", False):
            candidate_ids = [
                token_id
                for token_id in candidate_ids
                if tokenizer.decode([token_id]) in sampled_text_counts
            ]
        support_out[:] = candidate_ids
        args._last_raw_support_size = raw_support_size
        args._last_support_size = len(candidate_ids)
    return AnytimeEstimate(
        scores=selected_scores,
        samples_used=len(all_sample_ids),
        requested_choices=requested_choices,
        stage_trace=tuple(stage_trace),
    )


def estimate_log_probs(
    client: OpenRouterClient,
    tokenizer,
    prompt: str,
    prefix_text: str,
    args: argparse.Namespace,
    device: torch.device,
    proxy: Optional[LocalProxyRuntime] = None,
    support_out: Optional[list[int]] = None,
) -> torch.Tensor | BiasMCInputs:
    sample_args = _mc_sampling_args(args, args.mc_samples_per_token)
    independent_views = int(getattr(args, "mc_independent_views", 1))
    if independent_views <= 0:
        raise ValueError("--mc_independent_views must be positive.")
    sample_id_views: list[list[int]] = []
    sampled_texts: dict[str, int] = {}
    requested_choices = 0
    for view_index in range(independent_views):
        sample_ids, stats = sample_position_token_ids(
            client=client,
            tokenizer=tokenizer,
            question=prompt,
            prefix_text=prefix_text,
            args=sample_args,
        )
        if not sample_ids:
            if independent_views == 1:
                raise RuntimeError(
                    "No valid OpenRouter samples for prefix length "
                    f"{len(prefix_text)}: {stats}"
                )
            raise RuntimeError(
                "No valid OpenRouter samples for MC view "
                f"{view_index + 1}/{independent_views} at prefix length "
                f"{len(prefix_text)}: {stats}"
            )
        sample_id_views.append(sample_ids)
        sampled_texts.update(stats.get("sampled_completion_text_counts") or {})
        requested_choices += int(stats.get("requested_samples", len(sample_ids)))

    args._last_mc_valid_samples = sum(len(ids) for ids in sample_id_views)
    args._last_mc_requested_choices = requested_choices

    if support_out is not None:
        # Union across MC views, but keep only ids that round-trip: decoding the id
        # must reproduce a string the target actually returned. When a returned
        # continuation needs more than one proxy token the pipeline records just the
        # first, so that id decodes to a PREFIX the target never emitted on its own,
        # and appending it strands the target mid-token.
        candidate_ids = sorted({int(i) for ids in sample_id_views for i in ids})
        raw_support_size = len(candidate_ids)
        if getattr(args, "require_faithful_support", False):
            # Round-trip test: the id must decode back to a string the target really
            # returned. Ids recorded from a multi-token return decode to a prefix that
            # the target never emitted alone, and appending one strands it mid-token.
            candidate_ids = [
                i for i in candidate_ids if tokenizer.decode([i]) in sampled_texts
            ]
        support_out[:] = candidate_ids
        # stash sizes so an over-strict filter shows up instead of silently no-opping
        args._last_raw_support_size = raw_support_size
        args._last_support_size = len(candidate_ids)

    dtype = torch.float16 if args.store_dtype == "float16" else torch.float32
    input_representation = getattr(
        args, "mc_input_representation", FLOOR_LOGPROB
    )
    base_representation = getattr(
        args, "mc_base_score_representation", None
    ) or input_representation
    fused_floor_scores = None
    if proxy is not None:
        proxy_logits = proxy.next_token_logits(
            prompt,
            prefix_text,
            qwen_hard_no_think_prefill=bool(
                getattr(args, "qwen_hard_no_think_prefill", False)
            ),
        )
        mc_counts = torch.stack(
            [
                torch.bincount(
                    torch.as_tensor(sample_ids, dtype=torch.long),
                    minlength=len(tokenizer),
                )
                for sample_ids in sample_id_views
            ]
        ).to(proxy_logits.device)
        expanded_proxy_logits = proxy_logits.unsqueeze(0).expand(
            len(sample_id_views), -1
        )
        fused_floor_scores = fuse_proxy_logits_with_mc_counts(
            expanded_proxy_logits,
            mc_counts,
            temperature=float(args.proxy_temperature),
            prior_strength=float(args.proxy_prior_strength),
            dtype=dtype,
        ).to(device)
    feature_rows = []
    base_rows = []
    for view_index, sample_ids in enumerate(sample_id_views):
        sampled_tensor = torch.tensor([sample_ids], dtype=torch.long)
        count_scores = None
        floor_scores = None
        if LOG_COUNT in {input_representation, base_representation}:
            count_scores = sampled_ids_to_log_counts(
                sampled_token_ids=sampled_tensor,
                vocab_size=len(tokenizer),
                alpha=args.mc_log_count_alpha,
                dtype=dtype,
            )[0]
        if FLOOR_LOGPROB in {input_representation, base_representation}:
            if fused_floor_scores is not None:
                floor_scores = fused_floor_scores[view_index]
            else:
                floor_scores = ids_to_log_probs(
                    sampled_token_ids=sample_ids,
                    vocab_size=len(tokenizer),
                    observed_alpha=args.mc_observed_alpha,
                    floor_mass=args.mc_floor_mass,
                    dtype=dtype,
                )
                if getattr(args, "mc_static_prior_mode", "none") != "none":
                    floor_scores = apply_static_prior_to_mc_samples(
                        sample_ids=sample_ids,
                        args=args,
                        vocab_size=len(tokenizer),
                        dtype=dtype,
                        device=device,
                    )
        feature_scores = (
            count_scores
            if input_representation == LOG_COUNT
            else floor_scores
        )
        base_scores = (
            count_scores
            if base_representation == LOG_COUNT
            else floor_scores
        )
        assert feature_scores is not None and base_scores is not None
        feature_rows.append(feature_scores)
        base_rows.append(base_scores)

    feature_batch = torch.stack(feature_rows).to(device)
    base_batch = torch.stack(base_rows).to(device)
    if input_representation == base_representation and independent_views == 1:
        # Preserve the legacy return type and exact single-view score path.
        return feature_batch
    return BiasMCInputs(
        features=feature_batch,
        base_scores=base_batch,
        independent_views=independent_views,
    )


def _mean_probability_scores(per_view_scores: torch.Tensor) -> torch.Tensor:
    """Average normalized per-view predictions and return their log scores."""

    if per_view_scores.dim() != 2 or per_view_scores.shape[0] <= 1:
        raise ValueError("MC probability ensembling requires at least two score rows.")
    view_log_probs = torch.log_softmax(per_view_scores.float(), dim=-1)
    return (
        torch.logsumexp(view_log_probs, dim=0, keepdim=True)
        - math.log(per_view_scores.shape[0])
    )


def context_contract_for_models(models, anytime_policy=None):
    contracts = []
    for model in models:
        if model is None or not getattr(model, "context_conditioning", False):
            continue
        contract = getattr(model.config, "context_encoder_contract", None)
        validate_contract(contract)
        if contract["hidden_size"] != model.config.context_dim:
            raise ValueError("Context encoder hidden size differs from BiasNet context_dim.")
        contracts.append(contract)
    if contracts and anytime_policy is not None:
        raise ValueError("Context conditioning currently supports fixed-budget MC, not anytime policies.")
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Main and bootstrap BiasNets require different context encoder contracts.")
    return contracts[0] if contracts else None


def apply_bias_model(
    bias_model: BiasNet,
    bias_input: torch.Tensor | BiasMCInputs,
    residual_scale: float = 1.0,
    position_id: Optional[int] = None,
    context_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = next(bias_model.parameters()).dtype
    if isinstance(bias_input, BiasMCInputs):
        model_input = bias_input.features.to(dtype=dtype)
        base_scores = bias_input.base_scores.to(dtype=dtype)
        independent_views = int(bias_input.independent_views)
    else:
        model_input = bias_input.to(dtype=dtype)
        base_scores = model_input
        independent_views = 1
    if independent_views <= 0:
        raise ValueError("BiasMCInputs.independent_views must be positive.")
    if model_input.dim() != 2 or base_scores.shape != model_input.shape:
        raise ValueError(
            "BiasNet MC feature and base score tensors must have matching "
            "[views, vocab_size] shapes."
        )
    if model_input.shape[0] != independent_views:
        raise ValueError(
            "BiasMCInputs.independent_views does not match its tensor rows: "
            f"{independent_views} != {model_input.shape[0]}."
        )
    kwargs = {}
    if int(getattr(bias_model, "num_position_buckets", 0) or 0) > 0:
        kwargs["position_ids"] = position_id
    if getattr(bias_model, "context_conditioning", False):
        expected = (1, bias_model.config.context_dim)
        if not isinstance(context_features, torch.Tensor) or tuple(context_features.shape) != expected:
            raise ValueError(f"Online context_features must have shape {expected} for one prefix.")
        # Every independent MC view describes the same pre-action state.
        kwargs["context_features"] = context_features.expand(independent_views, -1)
    elif context_features is not None:
        raise ValueError("Context supplied to a BiasNet without context conditioning.")
    residual = bias_model(model_input, **kwargs)
    per_view_scores = base_scores + residual_scale * residual
    if independent_views == 1:
        # Keep the pre-ensemble path bit-compatible by avoiding normalization.
        return per_view_scores
    return _mean_probability_scores(per_view_scores)


def risk_gate_scale_from_score(
    score: float,
    threshold: float,
    mode: str = "hard",
    soft_temperature: float = 0.05,
) -> float:
    """Convert a finite risk score into the BiasNet residual scale."""
    if not math.isfinite(score):
        raise ValueError(f"Risk gate returned a non-finite score: {score}.")
    if not math.isfinite(threshold):
        raise ValueError(f"Risk gate threshold must be finite, got {threshold}.")
    if mode == "hard":
        return float(score < threshold)
    if mode != "soft":
        raise ValueError(f"Unknown risk gate mode: {mode}.")
    if soft_temperature <= 0:
        raise ValueError("Soft risk-gate temperature must be positive.")

    # Stable sigmoid((threshold - score) / temperature).
    scaled_score = (float(threshold) - score) / soft_temperature
    if scaled_score >= 0:
        return 1.0 / (1.0 + math.exp(-scaled_score))
    exp_score = math.exp(scaled_score)
    return exp_score / (1.0 + exp_score)


def risk_gate_score_and_scale(
    risk_gate: Optional[PrefixRiskGate],
    prompt: str,
    answer_prefix: str,
    mode: str = "hard",
    soft_temperature: float = 0.05,
) -> tuple[Optional[float], float]:
    """Return the raw prefix-risk score and its BiasNet residual scale."""
    if risk_gate is None:
        return None, 1.0
    score = float(risk_gate.score_prefixes([prompt], [answer_prefix])[0].item())
    scale = risk_gate_scale_from_score(
        score,
        threshold=float(risk_gate.threshold),
        mode=mode,
        soft_temperature=soft_temperature,
    )
    return score, scale


def risk_gate_bias_scale(
    risk_gate: Optional[PrefixRiskGate],
    prompt: str,
    answer_prefix: str,
    mode: str = "hard",
    soft_temperature: float = 0.05,
) -> float:
    """Return the BiasNet residual scale implied by a prefix-risk score."""
    return risk_gate_score_and_scale(
        risk_gate,
        prompt,
        answer_prefix,
        mode=mode,
        soft_temperature=soft_temperature,
    )[1]


def gate_allows_bias(
    risk_gate: Optional[PrefixRiskGate],
    prompt: str,
    answer_prefix: str,
) -> bool:
    return risk_gate_bias_scale(risk_gate, prompt, answer_prefix, mode="hard") > 0.0


def validate_handoff_response(
    response,
    tokenizer,
    prefix_text: str,
    remaining_tokens: int,
    *,
    reject_reasoning_tokens: bool = False,
    allow_local_token_overflow: bool = False,
    requested_model: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Validate a multi-token handoff response and return continuation-only text."""
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
    reasoning_detected = bool(
        response.reasoning is not None
        or response.reasoning_details is not None
        or int(response.reasoning_tokens or 0) > 0
    )
    if reject_reasoning_tokens and reasoning_detected:
        raise FatalOpenRouterResponseError(
            "OpenRouter returned reasoning during a visible multi-token handoff "
            f"(generation_id={response.generation_id!r}, "
            f"reasoning_tokens={int(response.reasoning_tokens or 0)}, "
            f"reasoning_field_present={response.reasoning is not None}, "
            "reasoning_details_present="
            f"{response.reasoning_details is not None})."
        )
    if response.choice_error is not None:
        encoded_error = json.dumps(
            response.choice_error,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        raise FatalOpenRouterResponseError(
            "OpenRouter returned a choice-level error during gate handoff"
            f" (generation_id={response.generation_id!r}, "
            f"finish_reason={response.finish_reason!r}): {encoded_error[:500]}"
        )
    if (
        normalized_finish_reason in {"content_filter", "error"}
        or normalized_native_finish_reason in {"content_filter", "error"}
    ):
        raise FatalOpenRouterResponseError(
            "OpenRouter returned a fatal gate-handoff choice"
            f" (generation_id={response.generation_id!r}, "
            f"finish_reason={response.finish_reason!r}, "
            f"native_finish_reason={response.native_finish_reason!r})."
        )

    routed_or_requested_model = (
        response.model
        if is_claude_45_model(response.model)
        else requested_model
    )
    continuation, prefill_audit = strip_prefill_with_audit(
        response.content,
        prefix_text,
        model_name=routed_or_requested_model,
    )
    if not continuation:
        if (
            normalized_finish_reason == "length"
            or normalized_native_finish_reason == "length"
        ):
            raise RuntimeError(
                "OpenRouter returned an empty length-truncated gate handoff "
                f"(generation_id={response.generation_id!r})."
            )
        if (
            normalized_finish_reason != "stop"
            and normalized_native_finish_reason != "stop"
        ):
            raise RuntimeError(
                "OpenRouter returned an empty gate handoff without a stop "
                f"finish reason (generation_id={response.generation_id!r}, "
                f"finish_reason={response.finish_reason!r}, "
                "native_finish_reason="
                f"{response.native_finish_reason!r})."
            )

    isolated_continuation_token_ids = tokenizer.encode(
        continuation,
        add_special_tokens=False,
    )
    prefix_token_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    combined_token_ids = tokenizer.encode(
        f"{prefix_text}{continuation}",
        add_special_tokens=False,
    )
    boundary_continuation_tokens = max(
        0,
        len(combined_token_ids) - len(prefix_token_ids),
    )
    local_token_overflow = max(
        0,
        boundary_continuation_tokens - int(remaining_tokens),
    )
    if local_token_overflow and not allow_local_token_overflow:
        raise RuntimeError(
            "Gate handoff exceeded its remaining local token budget: "
            f"{boundary_continuation_tokens} > {remaining_tokens}."
        )
    return continuation, {
        "finish_reason": response.finish_reason,
        "native_finish_reason": response.native_finish_reason,
        "response_chars": len(response.content),
        "continuation_chars": len(continuation),
        **prefill_audit,
        "response_model": response.model,
        "generation_id": response.generation_id,
        "response_cache_status": response.response_cache_status,
        "request_max_tokens": response.request_max_tokens,
        "request_temperature": response.request_temperature,
        "request_reasoning_mode": response.request_reasoning_mode,
        "reasoning_fallback_used": bool(response.reasoning_fallback_used),
        "continuation_local_tokens": boundary_continuation_tokens,
        "continuation_local_token_overflow": local_token_overflow,
        "continuation_isolated_local_tokens": len(
            isolated_continuation_token_ids
        ),
        "remaining_tokens_requested": int(remaining_tokens),
        "reasoning_detected": reasoning_detected,
        "reasoning_tokens": int(response.reasoning_tokens or 0),
    }


def generate_handoff_continuation(
    client: OpenRouterClient,
    tokenizer,
    prompt: str,
    prefix_text: str,
    remaining_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, dict[str, Any]]:
    """Generate the post-latch continuation in one prefix-preserving request."""
    if remaining_tokens <= 0:
        raise ValueError("remaining_tokens must be positive for gate handoff.")
    response = client.generate(
        messages=messages_for_prefix(
            prompt,
            prefix_text,
            qwen_hard_no_think_prefill=bool(
                getattr(
                    getattr(client, "args", None),
                    "qwen_hard_no_think_prefill",
                    False,
                )
            ),
        ),
        temperature=temperature,
        top_p=top_p,
        max_tokens=remaining_tokens,
    )
    continuation, audit = validate_handoff_response(
        response,
        tokenizer,
        prefix_text,
        remaining_tokens,
        reject_reasoning_tokens=bool(
            getattr(
                getattr(client, "args", None),
                "reject_reasoning_tokens",
                False,
            )
        ),
        requested_model=getattr(getattr(client, "args", None), "model", None),
    )
    return continuation, audit


def generate_speculative_draft(
    client: OpenRouterClient,
    tokenizer,
    prompt: str,
    prefix_text: str,
    remaining_tokens: int,
    draft_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[int], dict[str, Any]]:
    """Generate a bounded base-model draft and reconstruct local token IDs."""
    requested_tokens = min(int(remaining_tokens), int(draft_tokens))
    if requested_tokens <= 0:
        raise ValueError("Speculative draft token budget must be positive.")
    response = client.generate(
        messages=messages_for_prefix(
            prompt,
            prefix_text,
            qwen_hard_no_think_prefill=bool(
                getattr(
                    getattr(client, "args", None),
                    "qwen_hard_no_think_prefill",
                    False,
                )
            ),
        ),
        temperature=temperature,
        top_p=top_p,
        max_tokens=requested_tokens,
    )
    continuation, audit = validate_handoff_response(
        response,
        tokenizer,
        prefix_text,
        requested_tokens,
        reject_reasoning_tokens=bool(
            getattr(
                getattr(client, "args", None),
                "reject_reasoning_tokens",
                False,
            )
        ),
        requested_model=getattr(getattr(client, "args", None), "model", None),
        # The provider enforces this budget in its native tokenizer.  A proxy
        # tokenizer can encode the same text into slightly more tokens; the
        # local draft is truncated to requested_tokens immediately below.
        allow_local_token_overflow=True,
    )
    token_ids = tokenizer.encode(continuation, add_special_tokens=False)
    isolated_count = len(token_ids)
    if isolated_count > requested_tokens:
        # Boundary tokenization can differ from the provider's native tokenization.
        # Keep the local controller within its explicit generation budget.
        token_ids = token_ids[:requested_tokens]
    audit.update(
        {
            "requested_tokens": requested_tokens,
            "draft_local_token_count": len(token_ids),
            "draft_isolated_local_token_count": isolated_count,
            "draft_local_tokens_truncated": max(
                0,
                isolated_count - len(token_ids),
            ),
        }
    )
    return [int(token_id) for token_id in token_ids], audit


def choose_final_token(
    biased_log_probs: torch.Tensor,
    args: argparse.Namespace,
    observed_support: list[int],
    forbidden_token_ids: Optional[list[int]],
) -> tuple[int, bool]:
    """Pick the emitted token, optionally confined to the sampled support.

    generate_one has a no-gate fast path and a gated path; both must apply this
    identically, so the logic lives here rather than being duplicated.

    Returns (token_id, was_off_support).
    """

    unrestricted = generation_token(
        biased_log_probs, args.temperature, forbidden_token_ids
    )
    if not getattr(args, "restrict_to_observed_support", False):
        return unrestricted, False
    if not observed_support:
        raise ValueError(
            "--restrict_to_observed_support was requested but the observed support is "
            "empty; refusing to silently fall back to unrestricted decoding."
        )
    if unrestricted in observed_support:
        return unrestricted, False
    masked = torch.full_like(biased_log_probs, float("-inf"))
    keep = torch.tensor(observed_support, dtype=torch.long, device=masked.device)
    masked[keep] = biased_log_probs[keep]
    return generation_token(masked, args.temperature, forbidden_token_ids), True


def _mc_requested_choice_count(client: OpenRouterClient) -> Optional[int]:
    value = getattr(client, "mc_requested_samples", None)
    return None if value is None else int(value)


def _mc_requested_choice_delta(
    client: OpenRouterClient,
    before: Optional[int],
    *,
    fallback: int,
) -> int:
    after = _mc_requested_choice_count(client)
    if before is None or after is None:
        return int(fallback)
    if after < before:
        raise RuntimeError("OpenRouter MC requested-choice counter moved backwards.")
    return after - before


def generate_one(
    client: OpenRouterClient,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
    device: torch.device,
    bias_model: Optional[BiasNet],
    risk_gate: Optional[PrefixRiskGate],
    risk_gate_prompt: Optional[str] = None,
    bootstrap_bias_model: Optional[BiasNet] = None,
    generation_audit: Optional[dict[str, Any]] = None,
    proxy: Optional[LocalProxyRuntime] = None,
    anytime_policy: Optional[AnytimePolicy] = None,
    context_encoder: Optional[FrozenContextEncoder] = None,
) -> str:
    context_contract = context_contract_for_models(
        [bias_model, bootstrap_bias_model], anytime_policy,
    )
    if context_contract is not None:
        if context_encoder is None or context_encoder.contract != context_contract:
            raise ValueError("Generation requires the checkpoint's frozen context encoder.")

    def context_for(model, prefix):
        if model is None or not getattr(model, "context_conditioning", False):
            return None
        return context_encoder.encode([prompt], [prefix])

    context_calls_before = context_encoder.calls if context_encoder is not None else 0
    context_tokens_before = context_encoder.input_tokens if context_encoder is not None else 0
    context_latency_before = context_encoder.latency_seconds if context_encoder is not None else 0.0
    gate_prompt = prompt if risk_gate_prompt is None else risk_gate_prompt
    initial_prefix = getattr(args, "initial_prefix", "") or ""
    generated_ids: list[int] = (
        [
            int(token_id)
            for token_id in tokenizer.encode(
                initial_prefix,
                add_special_tokens=False,
            )
        ]
        if initial_prefix
        else []
    )
    initial_prefix_token_count = len(generated_ids)
    bootstrap_bias_tokens = int(
        getattr(args, "bootstrap_biasnet_tokens", 0)
    )
    biasnet_max_tokens = getattr(args, "biasnet_max_tokens", None)
    if biasnet_max_tokens is not None:
        biasnet_max_tokens = int(biasnet_max_tokens)
    eos_id = tokenizer.eos_token_id
    forbidden_proxy_token_ids = proxy_forbidden_token_ids(
        tokenizer,
        bool(getattr(args, "mask_non_eos_special_tokens", False)),
    )
    handoff_continuation = ""

    min_scale = float(getattr(args, "risk_gate_min_scale", 0.0))
    latch_enabled = bool(getattr(args, "risk_gate_latch_off", False))
    latch_patience = int(getattr(args, "risk_gate_latch_patience", 2))
    configured_latch_threshold = getattr(
        args,
        "risk_gate_latch_threshold",
        None,
    )
    latch_threshold = (
        float(configured_latch_threshold)
        if configured_latch_threshold is not None
        else (
            float(risk_gate.threshold)
            if risk_gate is not None
            else None
        )
    )
    handoff_enabled = bool(
        getattr(args, "risk_gate_handoff_on_latch", False)
    )
    speculative_enabled = bool(
        getattr(args, "risk_gate_speculative_draft", False)
    )
    speculative_min_base_streak = int(
        getattr(args, "risk_gate_speculative_min_base_streak", 2)
    )
    speculative_draft_tokens = int(
        getattr(args, "risk_gate_speculative_draft_tokens", 80)
    )
    latched_off = False
    consecutive_high_risk = 0
    consecutive_speculative_base = 0
    latch_step: Optional[int] = None
    mc_steps = 0
    mc_positive_sample_steps = 0
    mc_valid_samples_total = 0
    mc_requested_choices_total = 0
    anytime_controller_steps = 0
    zero_sample_biasnet_steps = 0
    anytime_budget_histogram: Counter[int] = Counter()
    off_support_steps = 0
    raw_support_total = 0
    faithful_support_total = 0
    empty_support_steps = 0
    base_only_steps = 0
    scored_steps = 0
    warmup_steps = 0
    below_min_scale_steps = 0
    handoff_audit: Optional[dict[str, Any]] = None
    speculative_draft_calls = 0
    speculative_generated_tokens = 0
    speculative_verified_tokens = 0
    speculative_accepted_tokens = 0
    speculative_rejected_tokens = 0
    speculative_rollbacks = 0
    step_trace: list[dict[str, Any]] = []
    speculative_trace: list[dict[str, Any]] = []
    proxy_calls_before_generation = proxy.calls if proxy is not None else 0
    proxy_latency_before_generation = (
        proxy.total_latency_seconds if proxy is not None else 0.0
    )
    proxy_tokens_before_generation = (
        proxy.total_input_tokens if proxy is not None else 0
    )

    if generation_audit is not None:
        generation_audit.clear()
        generation_audit.update(
            {
                "schema_version": 1,
                "configuration": {
                    "context_encoder_contract": context_contract,
                    "requested_model": getattr(args, "model", None),
                    "tokenizer_name": getattr(args, "tokenizer_name", None),
                    "fix_mistral_regex": bool(
                        getattr(args, "fix_mistral_regex", False)
                    ),
                    "mc_input_representation": getattr(
                        args, "mc_input_representation", FLOOR_LOGPROB
                    ),
                    "mc_base_score_representation": getattr(
                        args, "mc_base_score_representation", None
                    ) or getattr(args, "mc_input_representation", FLOOR_LOGPROB),
                    "mc_score_interface": (
                        "dual_v1"
                        if (getattr(args, "mc_base_score_representation", None)
                            or getattr(args, "mc_input_representation", FLOOR_LOGPROB))
                        != getattr(args, "mc_input_representation", FLOOR_LOGPROB)
                        else "shared_v1"
                    ),
                    "mc_log_count_alpha": (
                        getattr(args, "mc_log_count_alpha", None)
                        if getattr(args, "mc_input_representation", FLOOR_LOGPROB)
                        == LOG_COUNT
                        else None
                    ),
                    "mc_samples_per_token": int(
                        getattr(args, "mc_samples_per_token", 50)
                    ),
                    "mc_independent_views": int(
                        getattr(args, "mc_independent_views", 1)
                    ),
                    "mc_ensemble": mc_ensemble_audit(args, anytime_policy),
                    "mc_sample_temperature": float(
                        getattr(args, "mc_sample_temperature", 1.0)
                    ),
                    "mc_top_p": float(getattr(args, "mc_top_p", 1.0)),
                    "mc_observed_alpha": float(
                        getattr(args, "mc_observed_alpha", 0.1)
                    ),
                    "mc_floor_mass": float(
                        getattr(args, "mc_floor_mass", 1e-4)
                    ),
                    "mc_static_prior": (
                        {
                            "mode": getattr(args, "mc_static_prior_mode", "none"),
                            "strength": getattr(args, "mc_static_prior_strength", None),
                            "path": getattr(args, "mc_static_prior_path", None),
                        }
                        if getattr(args, "mc_static_prior_mode", "none") != "none"
                        else None
                    ),
                    "mc_store_dtype": getattr(args, "store_dtype", None),
                    "mc_completion_policy": getattr(
                        args, "sample_completion_policy", "partial"
                    ),
                    "biasnet_checkpoint": getattr(args, "biasnet_ckpt", None),
                    "biasnet_input_projection_mode": (
                        getattr(bias_model, "input_projection_mode", None)
                        if bias_model is not None
                        else None
                    ),
                    "biasnet_input_hidden_normalization": (
                        getattr(bias_model, "input_hidden_normalization", None)
                        if bias_model is not None
                        else None
                    ),
                    "biasnet_input_layer_norm_eps": (
                        getattr(bias_model, "input_layer_norm_eps", None)
                        if bias_model is not None
                        else None
                    ),
                    "biasnet_count_sketch_input_centering": (
                        getattr(bias_model, "count_sketch_input_centering", None)
                        if bias_model is not None
                        else None
                    ),
                    "biasnet_max_tokens": biasnet_max_tokens,
                    "anytime_stopping": (
                        anytime_policy.configuration()
                        if anytime_policy is not None
                        else None
                    ),
                    "bootstrap_biasnet_checkpoint": getattr(
                        args,
                        "bootstrap_biasnet_ckpt",
                        None,
                    ),
                    "bootstrap_biasnet_tokens": bootstrap_bias_tokens,
                    "initial_prefix": getattr(args, "initial_prefix", "") or "",
                    "initial_prefix_token_count": initial_prefix_token_count,
                    "risk_gate_checkpoint": getattr(
                        args,
                        "risk_gate_checkpoint",
                        None,
                    ),
                    "risk_gate_prompt_source": getattr(
                        args,
                        "risk_gate_prompt_source",
                        "target",
                    ),
                    "mode": getattr(args, "risk_gate_mode", "hard"),
                    "threshold": (
                        float(risk_gate.threshold)
                        if risk_gate is not None
                        else None
                    ),
                    "soft_temperature": float(
                        getattr(args, "risk_gate_soft_temperature", 0.05)
                    ),
                    "warmup_tokens": int(
                        getattr(args, "risk_gate_warmup_tokens", 0)
                    ),
                    "min_scale": min_scale,
                    "latch_off": latch_enabled,
                    "latch_patience": latch_patience,
                    "latch_threshold": latch_threshold,
                    "handoff_on_latch": handoff_enabled,
                    "speculative_draft": speculative_enabled,
                    "speculative_min_base_streak": speculative_min_base_streak,
                    "speculative_draft_tokens": speculative_draft_tokens,
                    "reasoning_mode": getattr(
                        args,
                        "reasoning_mode",
                        None,
                    ),
                    "max_reasoning_retries": int(
                        getattr(args, "max_reasoning_retries", 0)
                    ),
                    "reasoning_retry_sleep": float(
                        getattr(args, "reasoning_retry_sleep", 0.0)
                    ),
                    "reasoning_fallback_temperature": getattr(
                        args,
                        "reasoning_fallback_temperature",
                        None,
                    ),
                    "reasoning_fallback_mode": getattr(
                        args,
                        "reasoning_fallback_mode",
                        None,
                    ),
                    "max_reasoning_fallback_retries": int(
                        getattr(args, "max_reasoning_fallback_retries", 0)
                    ),
                    "scored_token_source": (
                        "deterministic_base_candidate_with_audited_"
                        "reasoning_fallback"
                        if (
                            getattr(
                                args,
                                "reasoning_fallback_temperature",
                                None,
                            )
                            is not None
                            or getattr(
                                args,
                                "reasoning_fallback_mode",
                                None,
                            )
                            is not None
                        )
                        else "deterministic_base_candidate"
                    ),
                },
                "steps": step_trace,
                "speculative_drafts": speculative_trace,
            }
        )
        if proxy is not None:
            generation_audit["configuration"]["proxy_fusion"] = (
                proxy.configuration(args)
            )

    generation_steps = args.max_new_tokens
    for _step in range(generation_steps):
        controlled_token_count = len(generated_ids) - initial_prefix_token_count
        if controlled_token_count >= generation_steps:
            break
        step_index = controlled_token_count
        step_number = step_index + 1
        active_bias_model = (
            bootstrap_bias_model
            if bootstrap_bias_model is not None
            and step_index < bootstrap_bias_tokens
            else bias_model
        )
        biasnet_budget_exhausted = bool(
            biasnet_max_tokens is not None
            and step_index >= biasnet_max_tokens
        )
        prefix_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        if (
            active_bias_model is not None
            and anytime_policy is not None
            and not biasnet_budget_exhausted
        ):
            if proxy is None:
                raise RuntimeError("Anytime stopping requires a loaded local proxy.")
            try:
                base_token_id, base_info = query_next_token_id(
                    client=client,
                    tokenizer=tokenizer,
                    question=prompt,
                    prefix_text=prefix_text,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=args.api_max_tokens,
                )
            except FatalOpenRouterResponseError:
                raise
            except RuntimeError as exc:
                if not args.stop_on_mc_failure:
                    raise
                print(f"base_failure_step={step_number} reason={exc}", flush=True)
                break
            if base_token_id is None:
                exc = RuntimeError(
                    "Deterministic next-token query returned no valid token for "
                    f"anytime feature construction at step {step_number}."
                )
                if not args.stop_on_mc_failure:
                    raise exc
                print(f"base_failure_step={step_number} reason={exc}", flush=True)
                break
            requested_choices_before = _mc_requested_choice_count(client)
            try:
                estimate = estimate_anytime_scores(
                    client=client,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    prefix_text=prefix_text,
                    args=args,
                    device=device,
                    proxy=proxy,
                    bias_model=active_bias_model,
                    base_token_id=int(base_token_id),
                    position_id=step_index,
                    policy=anytime_policy,
                )
            except FatalOpenRouterResponseError:
                raise
            except RuntimeError as exc:
                if not args.stop_on_mc_failure:
                    raise
                print(f"mc_failure_step={step_number} reason={exc}", flush=True)
                break
            final_token_id = generation_token(
                estimate.scores[0], args.temperature, forbidden_proxy_token_ids
            )
            stop_stage = estimate.stage_trace[-1]
            stop_budget = int(estimate.samples_used)
            requested_choices = _mc_requested_choice_delta(
                client,
                requested_choices_before,
                fallback=estimate.requested_choices,
            )
            if requested_choices != estimate.requested_choices:
                raise RuntimeError(
                    "Anytime sampler audit disagrees with the OpenRouter client "
                    f"counter: {estimate.requested_choices} != {requested_choices}."
                )
            generated_ids.append(final_token_id)
            anytime_controller_steps += 1
            mc_valid_samples_total += stop_budget
            mc_requested_choices_total += requested_choices
            if stop_budget > 0:
                mc_steps += 1
                mc_positive_sample_steps += 1
            else:
                zero_sample_biasnet_steps += 1
            anytime_budget_histogram[stop_budget] += 1
            if generation_audit is not None:
                step_trace.append(
                    {
                        "step": step_number,
                        "base_token_id": int(base_token_id),
                        "final_token_id": int(final_token_id),
                        "risk_score": None,
                        "raw_bias_scale": 1.0,
                        "effective_bias_scale": 1.0,
                        "forced_warmup": False,
                        "high_risk": None,
                        "consecutive_high_risk": 0,
                        "latch_triggered": False,
                        "latched_off": False,
                        "used_biasnet": True,
                        "decision": f"anytime_stop_k{stop_budget}",
                        "anytime_stop_budget": stop_budget,
                        "anytime_stop_reason": stop_stage["stop_reason"],
                        "anytime_valid_mc_samples": stop_budget,
                        "anytime_requested_mc_choices": requested_choices,
                        "anytime_stages": list(estimate.stage_trace),
                        "base_empty_length_retry_attempts": int(
                            base_info.get("empty_length_retry_attempts", 0)
                        ),
                        "base_request_temperature": base_info.get(
                            "request_temperature"
                        ),
                        "base_request_reasoning_mode": base_info.get(
                            "request_reasoning_mode"
                        ),
                        "base_reasoning_fallback_used": bool(
                            base_info.get("reasoning_fallback_used", False)
                        ),
                        "base_reasoning_fallback_activations": int(
                            base_info.get("reasoning_fallback_activations", 0)
                        ),
                        "base_reasoning_fallback_result_used": bool(
                            base_info.get("reasoning_fallback_result_used", False)
                        ),
                    }
                )
            if args.progress_steps:
                print(
                    f"token_step={step_number}/{args.max_new_tokens} "
                    f"anytime_k={stop_budget} api_calls={client.calls} "
                    f"approx_cost=${client.total_cost:.6f}",
                    flush=True,
                )
            if eos_id is not None and final_token_id == eos_id:
                break
            continue
        if (
            active_bias_model is not None
            and risk_gate is None
            and anytime_policy is None
            and not biasnet_budget_exhausted
        ):
            step_context = context_for(active_bias_model, prefix_text)
            configured_valid_samples = (
                int(getattr(args, "mc_independent_views", 1))
                * int(getattr(args, "mc_samples_per_token", 50))
            )
            requested_choices_before = _mc_requested_choice_count(client)
            try:
                observed_support: list[int] = []
                bias_input = estimate_log_probs(
                    client=client,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    prefix_text=prefix_text,
                    args=args,
                    device=device,
                    proxy=proxy,
                    support_out=observed_support,
                )
            except FatalOpenRouterResponseError:
                raise
            except RuntimeError as exc:
                if not args.stop_on_mc_failure:
                    raise
                print(f"mc_failure_step={step_number} reason={exc}", flush=True)
                break
            biased_log_probs = apply_bias_model(
                active_bias_model,
                bias_input,
                position_id=step_index,
                context_features=step_context,
            )
            final_token_id, was_off_support = choose_final_token(
                biased_log_probs[0], args, observed_support, forbidden_proxy_token_ids
            )
            off_support_steps += int(was_off_support)
            raw_support_total += int(getattr(args, "_last_raw_support_size", 0) or 0)
            faithful_support_total += int(getattr(args, "_last_support_size", 0) or 0)
            generated_ids.append(final_token_id)
            mc_steps += 1
            mc_positive_sample_steps += 1
            observed_valid_samples = int(
                getattr(args, "_last_mc_valid_samples", configured_valid_samples)
            )
            observed_requested_choices = int(
                getattr(
                    args,
                    "_last_mc_requested_choices",
                    configured_valid_samples,
                )
            )
            mc_valid_samples_total += observed_valid_samples
            mc_requested_choices_total += _mc_requested_choice_delta(
                client,
                requested_choices_before,
                fallback=observed_requested_choices,
            )
            if generation_audit is not None:
                step_trace.append(
                    {
                        "step": step_number,
                        "base_token_id": None,
                        "final_token_id": int(final_token_id),
                        "risk_score": None,
                        "raw_bias_scale": 1.0,
                        "effective_bias_scale": 1.0,
                        "forced_warmup": False,
                        "high_risk": None,
                        "consecutive_high_risk": 0,
                        "latch_triggered": False,
                        "latched_off": False,
                        "used_biasnet": True,
                        "decision": "dense_bias_no_gate",
                        "base_empty_length_retry_attempts": 0,
                        "base_request_temperature": None,
                        "base_request_reasoning_mode": None,
                        "base_reasoning_fallback_used": False,
                        "base_reasoning_fallback_activations": 0,
                        "base_reasoning_fallback_result_used": False,
                    }
                )
            if args.progress_steps:
                print(
                    f"token_step={step_number}/{args.max_new_tokens} "
                    f"api_calls={client.calls} approx_cost=${client.total_cost:.6f}",
                    flush=True,
                )
            if eos_id is not None and final_token_id == eos_id:
                break
            continue

        try:
            base_token_id, _base_info = query_next_token_id(
                client=client,
                tokenizer=tokenizer,
                question=prompt,
                prefix_text=prefix_text,
                temperature=0.0,
                top_p=1.0,
                max_tokens=args.api_max_tokens,
            )
        except FatalOpenRouterResponseError:
            raise
        except RuntimeError as exc:
            if not args.stop_on_mc_failure:
                raise
            print(f"base_failure_step={step_number} reason={exc}", flush=True)
            break
        if base_token_id is None:
            if args.sample_completion_policy == "exact":
                exc = RuntimeError(
                    "Deterministic next-token query returned no valid token at "
                    f"generation step {step_number} after "
                    f"{int(_base_info.get('empty_length_retry_attempts', 0))} "
                    "adaptive empty-length retries "
                    f"(finish_reason={_base_info.get('finish_reason')!r}, "
                    "native_finish_reason="
                    f"{_base_info.get('native_finish_reason')!r})."
                )
                if not args.stop_on_mc_failure:
                    raise exc
                print(f"base_failure_step={step_number} reason={exc}", flush=True)
            break

        final_token_id = base_token_id
        base_answer_prefix = tokenizer.decode(
            generated_ids + [base_token_id],
            skip_special_tokens=False,
        )
        gate_forced_active = (
            risk_gate is not None
            and not latched_off
            and step_index < getattr(args, "risk_gate_warmup_tokens", 0)
        )
        gate_score: Optional[float] = None
        high_risk: Optional[bool] = None
        raw_bias_scale = 1.0
        effective_bias_scale = 1.0
        latch_triggered = False
        decision = "bias"

        if active_bias_model is None:
            raw_bias_scale = 0.0
            effective_bias_scale = 0.0
            decision = "base_no_biasnet"
        elif biasnet_budget_exhausted:
            raw_bias_scale = 0.0
            effective_bias_scale = 0.0
            decision = "biasnet_budget_base"
        elif risk_gate is not None:
            if latched_off:
                raw_bias_scale = 0.0
                effective_bias_scale = 0.0
                decision = "latched_base"
            elif gate_forced_active:
                warmup_steps += 1
                consecutive_high_risk = 0
                decision = "warmup_bias"
            else:
                gate_score, raw_bias_scale = risk_gate_score_and_scale(
                    risk_gate,
                    gate_prompt,
                    base_answer_prefix,
                    mode=getattr(args, "risk_gate_mode", "hard"),
                    soft_temperature=getattr(
                        args,
                        "risk_gate_soft_temperature",
                        0.05,
                    ),
                )
                scored_steps += 1
                high_risk = bool(
                    latch_threshold is not None
                    and gate_score >= latch_threshold
                )
                if high_risk:
                    consecutive_high_risk += 1
                else:
                    consecutive_high_risk = 0

                if (
                    latch_enabled
                    and consecutive_high_risk >= latch_patience
                ):
                    latched_off = True
                    latch_triggered = True
                    latch_step = step_number
                    effective_bias_scale = 0.0
                    decision = "latch_base"
                elif raw_bias_scale <= min_scale:
                    effective_bias_scale = 0.0
                    below_min_scale_steps += 1
                    decision = "below_min_scale_base"
                else:
                    effective_bias_scale = raw_bias_scale
                    decision = "soft_bias" if raw_bias_scale < 1.0 else "bias"

        used_biasnet = False
        if active_bias_model is not None and effective_bias_scale > 0.0:
            step_context = context_for(active_bias_model, prefix_text)
            configured_valid_samples = (
                int(getattr(args, "mc_independent_views", 1))
                * int(getattr(args, "mc_samples_per_token", 50))
            )
            requested_choices_before = _mc_requested_choice_count(client)
            try:
                observed_support: list[int] = []
                bias_input = estimate_log_probs(
                    client=client,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    prefix_text=prefix_text,
                    args=args,
                    device=device,
                    proxy=proxy,
                    support_out=observed_support,
                )
            except FatalOpenRouterResponseError:
                raise
            except RuntimeError as exc:
                if not args.stop_on_mc_failure:
                    raise
                print(f"mc_failure_step={step_number} reason={exc}", flush=True)
                break
            biased_log_probs = apply_bias_model(
                active_bias_model,
                bias_input,
                residual_scale=effective_bias_scale,
                position_id=step_index,
                context_features=step_context,
            )
            final_token_id, was_off_support = choose_final_token(
                biased_log_probs[0], args, observed_support, forbidden_proxy_token_ids
            )
            off_support_steps += int(was_off_support)
            raw_support_total += int(getattr(args, "_last_raw_support_size", 0) or 0)
            faithful_support_total += int(getattr(args, "_last_support_size", 0) or 0)
            used_biasnet = True
            mc_steps += 1
            mc_positive_sample_steps += 1
            observed_valid_samples = int(
                getattr(args, "_last_mc_valid_samples", configured_valid_samples)
            )
            observed_requested_choices = int(
                getattr(
                    args,
                    "_last_mc_requested_choices",
                    configured_valid_samples,
                )
            )
            mc_valid_samples_total += observed_valid_samples
            mc_requested_choices_total += _mc_requested_choice_delta(
                client,
                requested_choices_before,
                fallback=observed_requested_choices,
            )
        else:
            base_only_steps += 1

        generated_ids.append(final_token_id)
        if decision == "below_min_scale_base":
            consecutive_speculative_base += 1
        else:
            consecutive_speculative_base = 0
        if generation_audit is not None:
            step_trace.append(
                {
                    "step": step_number,
                    "base_token_id": int(base_token_id),
                    "final_token_id": int(final_token_id),
                    "risk_score": gate_score,
                    "raw_bias_scale": float(raw_bias_scale),
                    "effective_bias_scale": float(effective_bias_scale),
                    "forced_warmup": bool(gate_forced_active),
                    "high_risk": high_risk,
                    "consecutive_high_risk": int(consecutive_high_risk),
                    "latch_triggered": bool(latch_triggered),
                    "latched_off": bool(latched_off),
                    "used_biasnet": bool(used_biasnet),
                    "decision": decision,
                    "base_empty_length_retry_attempts": int(
                        _base_info.get("empty_length_retry_attempts", 0)
                    ),
                    "base_request_temperature": _base_info.get(
                        "request_temperature"
                    ),
                    "base_request_reasoning_mode": _base_info.get(
                        "request_reasoning_mode"
                    ),
                    "base_reasoning_fallback_used": bool(
                        _base_info.get("reasoning_fallback_used", False)
                    ),
                    "base_reasoning_fallback_activations": int(
                        _base_info.get("reasoning_fallback_activations", 0)
                    ),
                    "base_reasoning_fallback_result_used": bool(
                        _base_info.get("reasoning_fallback_result_used", False)
                    ),
                }
            )
        if args.progress_steps:
            print(
                f"token_step={len(generated_ids) - initial_prefix_token_count}/"
                f"{args.max_new_tokens} "
                f"api_calls={client.calls} approx_cost=${client.total_cost:.6f}",
                flush=True,
            )
        if eos_id is not None and final_token_id == eos_id:
            break
        if latch_triggered and handoff_enabled:
            remaining_tokens = generation_steps - (
                len(generated_ids) - initial_prefix_token_count
            )
            if remaining_tokens > 0:
                handoff_prefix = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=False,
                )
                handoff_continuation, handoff_audit = (
                    generate_handoff_continuation(
                        client=client,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        prefix_text=handoff_prefix,
                        remaining_tokens=remaining_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                )
                handoff_audit.update(
                    {
                        "occurred": True,
                        "trigger_step": step_number,
                        "prefix_token_count": len(generated_ids),
                    }
                )
            break

        if (
            speculative_enabled
            and risk_gate is not None
            and not latched_off
            and consecutive_speculative_base >= speculative_min_base_streak
        ):
            remaining_tokens = generation_steps - (
                len(generated_ids) - initial_prefix_token_count
            )
            if remaining_tokens <= 0:
                break
            draft_prefix_ids = list(generated_ids)
            draft_prefix_text = tokenizer.decode(
                draft_prefix_ids,
                skip_special_tokens=False,
            )
            draft_ids, draft_audit = generate_speculative_draft(
                client=client,
                tokenizer=tokenizer,
                prompt=prompt,
                prefix_text=draft_prefix_text,
                remaining_tokens=remaining_tokens,
                draft_tokens=speculative_draft_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            speculative_draft_calls += 1
            speculative_generated_tokens += len(draft_ids)
            draft_audit.update(
                {
                    "draft_index": speculative_draft_calls,
                    "trigger_after_step": (
                        len(draft_prefix_ids) - initial_prefix_token_count
                    ),
                    "base_streak_at_trigger": consecutive_speculative_base,
                }
            )
            if not draft_ids:
                draft_audit.update(
                    {
                        "verified_tokens": 0,
                        "accepted_tokens": 0,
                        "rejected_tokens": 0,
                        "rollback": False,
                        "terminal_empty_stop": True,
                    }
                )
                speculative_trace.append(draft_audit)
                break

            answer_prefixes = [
                tokenizer.decode(
                    draft_prefix_ids + draft_ids[: offset + 1],
                    skip_special_tokens=False,
                )
                for offset in range(len(draft_ids))
            ]
            draft_scores = [
                float(value)
                for value in risk_gate.score_prefixes(
                    [gate_prompt] * len(answer_prefixes),
                    answer_prefixes,
                ).reshape(-1).tolist()
            ]
            draft_scales = [
                risk_gate_scale_from_score(
                    score,
                    threshold=float(risk_gate.threshold),
                    mode=getattr(args, "risk_gate_mode", "hard"),
                    soft_temperature=getattr(
                        args,
                        "risk_gate_soft_temperature",
                        0.05,
                    ),
                )
                for score in draft_scores
            ]
            speculative_verified_tokens += len(draft_ids)
            violation_offset = next(
                (
                    offset
                    for offset, scale in enumerate(draft_scales)
                    if scale > min_scale
                ),
                None,
            )
            accepted_count = (
                len(draft_ids)
                if violation_offset is None
                else violation_offset
            )
            rejected_count = len(draft_ids) - accepted_count
            speculative_accepted_tokens += accepted_count
            speculative_rejected_tokens += rejected_count
            if violation_offset is not None:
                speculative_rollbacks += 1
            draft_audit.update(
                {
                    "verified_tokens": len(draft_ids),
                    "accepted_tokens": accepted_count,
                    "rejected_tokens": rejected_count,
                    "rollback": violation_offset is not None,
                    "rollback_offset": violation_offset,
                    "risk_scores": draft_scores,
                    "bias_scales": draft_scales,
                }
            )
            speculative_trace.append(draft_audit)

            for offset in range(accepted_count):
                token_id = int(draft_ids[offset])
                generated_ids.append(token_id)
                score = draft_scores[offset]
                raw_scale = draft_scales[offset]
                scored_steps += 1
                base_only_steps += 1
                below_min_scale_steps += 1
                consecutive_speculative_base += 1
                step_trace.append(
                    {
                        "step": (
                            len(generated_ids) - initial_prefix_token_count
                        ),
                        "base_token_id": token_id,
                        "final_token_id": token_id,
                        "risk_score": score,
                        "raw_bias_scale": float(raw_scale),
                        "effective_bias_scale": 0.0,
                        "forced_warmup": False,
                        "high_risk": bool(
                            latch_threshold is not None
                            and score >= latch_threshold
                        ),
                        "consecutive_high_risk": 0,
                        "latch_triggered": False,
                        "latched_off": False,
                        "used_biasnet": False,
                        "decision": "speculative_below_min_scale_base",
                        "base_source": "multi_token_draft",
                        "speculative_draft_index": speculative_draft_calls,
                        "base_empty_length_retry_attempts": 0,
                        "base_request_temperature": draft_audit.get(
                            "request_temperature"
                        ),
                        "base_request_reasoning_mode": draft_audit.get(
                            "request_reasoning_mode"
                        ),
                        "base_reasoning_fallback_used": bool(
                            draft_audit.get("reasoning_fallback_used", False)
                        ),
                        "base_reasoning_fallback_activations": int(
                            draft_audit.get("reasoning_fallback_used", False)
                        ),
                        "base_reasoning_fallback_result_used": bool(
                            draft_audit.get("reasoning_fallback_used", False)
                        ),
                    }
                )

            if violation_offset is None:
                normalized_finish = str(
                    draft_audit.get("finish_reason") or ""
                ).casefold()
                normalized_native_finish = str(
                    draft_audit.get("native_finish_reason") or ""
                ).casefold()
                if (
                    len(generated_ids) - initial_prefix_token_count
                    >= generation_steps
                    or len(draft_ids) < int(draft_audit["requested_tokens"])
                    or normalized_finish == "stop"
                    or normalized_native_finish == "stop"
                ):
                    break
                continue

            candidate_id = int(draft_ids[violation_offset])
            candidate_prefix_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=False,
            )
            candidate_score = draft_scores[violation_offset]
            candidate_scale = draft_scales[violation_offset]
            rollback_bias_model = (
                bootstrap_bias_model
                if bootstrap_bias_model is not None
                and len(generated_ids) - initial_prefix_token_count < bootstrap_bias_tokens
                else bias_model
            )
            step_context = context_for(rollback_bias_model, candidate_prefix_text)
            configured_valid_samples = (
                int(getattr(args, "mc_independent_views", 1))
                * int(getattr(args, "mc_samples_per_token", 50))
            )
            requested_choices_before = _mc_requested_choice_count(client)
            try:
                bias_input = estimate_log_probs(
                    client=client,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    prefix_text=candidate_prefix_text,
                    args=args,
                    device=device,
                    proxy=proxy,
                )
            except FatalOpenRouterResponseError:
                raise
            except RuntimeError as exc:
                if not args.stop_on_mc_failure:
                    raise
                print(
                    f"mc_failure_step={len(generated_ids) + 1} reason={exc}",
                    flush=True,
                )
                break
            biased_log_probs = apply_bias_model(
                rollback_bias_model,
                bias_input,
                residual_scale=candidate_scale,
                position_id=(
                    len(generated_ids) - initial_prefix_token_count
                ),
                context_features=step_context,
            )
            controlled_id = generation_token(
                biased_log_probs[0],
                args.temperature,
                forbidden_proxy_token_ids,
            )
            generated_ids.append(controlled_id)
            scored_steps += 1
            mc_steps += 1
            mc_positive_sample_steps += 1
            observed_valid_samples = int(
                getattr(args, "_last_mc_valid_samples", configured_valid_samples)
            )
            observed_requested_choices = int(
                getattr(
                    args,
                    "_last_mc_requested_choices",
                    configured_valid_samples,
                )
            )
            mc_valid_samples_total += observed_valid_samples
            mc_requested_choices_total += _mc_requested_choice_delta(
                client,
                requested_choices_before,
                fallback=observed_requested_choices,
            )
            consecutive_speculative_base = 0
            step_trace.append(
                {
                    "step": len(generated_ids) - initial_prefix_token_count,
                    "base_token_id": candidate_id,
                    "final_token_id": int(controlled_id),
                    "risk_score": candidate_score,
                    "raw_bias_scale": float(candidate_scale),
                    "effective_bias_scale": float(candidate_scale),
                    "forced_warmup": False,
                    "high_risk": bool(
                        latch_threshold is not None
                        and candidate_score >= latch_threshold
                    ),
                    "consecutive_high_risk": 0,
                    "latch_triggered": False,
                    "latched_off": False,
                    "used_biasnet": True,
                    "decision": "speculative_rollback_soft_bias",
                    "base_source": "multi_token_draft",
                    "speculative_draft_index": speculative_draft_calls,
                    "base_empty_length_retry_attempts": 0,
                    "base_request_temperature": draft_audit.get(
                        "request_temperature"
                    ),
                    "base_request_reasoning_mode": draft_audit.get(
                        "request_reasoning_mode"
                    ),
                    "base_reasoning_fallback_used": bool(
                        draft_audit.get("reasoning_fallback_used", False)
                    ),
                    "base_reasoning_fallback_activations": int(
                        draft_audit.get("reasoning_fallback_used", False)
                    ),
                    "base_reasoning_fallback_result_used": bool(
                        draft_audit.get("reasoning_fallback_used", False)
                    ),
                }
            )
            if eos_id is not None and controlled_id == eos_id:
                break

    if generation_audit is not None:
        generation_audit["summary"] = {
            "context_encoder_calls": (context_encoder.calls - context_calls_before) if context_encoder is not None else 0,
            "context_encoder_input_tokens": (context_encoder.input_tokens - context_tokens_before) if context_encoder is not None else 0,
            "context_encoder_latency_seconds": (context_encoder.latency_seconds - context_latency_before) if context_encoder is not None else 0.0,
            "context_encoder_forward_mode": "full_prefix_no_kv_cache" if context_contract is not None else None,
            "controlled_token_steps": (
                len(generated_ids) - initial_prefix_token_count
            ),
            "initial_prefix_token_count": initial_prefix_token_count,
            "warmup_steps": warmup_steps,
            "risk_scored_steps": scored_steps,
            "mc_steps": mc_steps,
            "mc_positive_sample_steps": mc_positive_sample_steps,
            "off_support_steps": off_support_steps,
            "empty_support_steps": empty_support_steps,
            "raw_support_total": raw_support_total,
            "faithful_support_total": faithful_support_total,
            # Preserve the historical fixed-path value for downstream readers.
            # Anytime has no legacy fixed-K meaning, so its alias reports the
            # actual requested choices; the two unambiguous fields below should
            # be preferred by new readers.
            "mc_requested_samples": (
                mc_requested_choices_total
                if anytime_policy is not None
                else (
                    mc_steps
                    * int(getattr(args, "mc_independent_views", 1))
                    * int(getattr(args, "mc_samples_per_token", 50))
                )
            ),
            "mc_valid_samples": mc_valid_samples_total,
            "mc_requested_choices": mc_requested_choices_total,
            "anytime_controller_steps": anytime_controller_steps,
            "anytime_base_token_queries": anytime_controller_steps,
            "zero_sample_biasnet_steps": zero_sample_biasnet_steps,
            "anytime_budget_histogram": {
                str(budget): int(count)
                for budget, count in sorted(anytime_budget_histogram.items())
            },
            "mean_samples_per_anytime_step": (
                mc_valid_samples_total / anytime_controller_steps
                if anytime_controller_steps
                else None
            ),
            "mean_valid_mc_samples_per_anytime_step": (
                mc_valid_samples_total / anytime_controller_steps
                if anytime_controller_steps
                else None
            ),
            "mean_requested_mc_choices_per_anytime_step": (
                mc_requested_choices_total / anytime_controller_steps
                if anytime_controller_steps
                else None
            ),
            "base_only_steps": base_only_steps,
            "below_min_scale_steps": below_min_scale_steps,
            "latch_step": latch_step,
            "final_latched_off": bool(latched_off),
            "handoff_occurred": handoff_audit is not None,
            "speculative_draft_calls": speculative_draft_calls,
            "speculative_generated_tokens": speculative_generated_tokens,
            "speculative_verified_tokens": speculative_verified_tokens,
            "speculative_accepted_tokens": speculative_accepted_tokens,
            "speculative_rejected_tokens": speculative_rejected_tokens,
            "speculative_rollbacks": speculative_rollbacks,
        }
        if proxy is not None:
            generation_audit["summary"].update(
                {
                    "proxy_calls": proxy.calls - proxy_calls_before_generation,
                    "proxy_latency_seconds": (
                        proxy.total_latency_seconds
                        - proxy_latency_before_generation
                    ),
                    "proxy_input_tokens": (
                        proxy.total_input_tokens - proxy_tokens_before_generation
                    ),
                    "proxy_forward_mode": "full_prefix_no_kv_cache",
                }
            )
        generation_audit["handoff"] = (
            handoff_audit
            if handoff_audit is not None
            else {"occurred": False}
        )

    controlled_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )
    return f"{controlled_text}{handoff_continuation}".strip()


def append_jsonl(
    path: str,
    prompts: list[str],
    completions: list[str],
    record_metadata: Optional[list[dict]] = None,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for index, (prompt, completion) in enumerate(zip(prompts, completions)):
            row = {"prompt": prompt, "completion": completion}
            if record_metadata is not None:
                row.update(record_metadata[index])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def crash_dump_path(output_json: str) -> str:
    """Sidecar path for partial generations abandoned by a fatal error."""

    base, _ = os.path.splitext(output_json)
    return f"{base}.crash.jsonl"


def append_crash_dump(
    path: str,
    *,
    prompt: str,
    tokenizer,
    generation_audit: Optional[dict[str, Any]],
    error: BaseException,
    record_metadata: Optional[dict] = None,
) -> None:
    """Persist the prefix generated before a fatal error aborted the prompt.

    generate_one binds its step trace into generation_audit before the first
    token, so the partial trace survives the exception. This is written beside
    the generation JSONL rather than into it: a crashed prompt is not a result,
    and must not be counted by --resume or by downstream audits that require
    exactly one record per benchmark row.
    """

    steps = list((generation_audit or {}).get("steps") or [])
    final_ids = [
        int(step["final_token_id"])
        for step in steps
        if step.get("final_token_id") is not None
    ]
    base_ids = [
        int(step["base_token_id"])
        for step in steps
        if step.get("base_token_id") is not None
    ]
    row = {
        "prompt": prompt,
        "error_type": type(error).__name__,
        "error": str(error),
        "steps_completed": len(steps),
        "partial_completion": tokenizer.decode(
            final_ids, skip_special_tokens=False
        ),
        "partial_completion_unbiased": tokenizer.decode(
            base_ids, skip_special_tokens=False
        ),
        "risk_gate_runtime": generation_audit,
    }
    if record_metadata is not None:
        row["record_metadata"] = record_metadata
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def count_jsonl_records(path: str) -> int:
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def validate_empty_length_retry_args(args: argparse.Namespace) -> None:
    retry_max_tokens = args.empty_length_retry_max_tokens
    retry_rounds = args.max_empty_length_retry_rounds
    if retry_max_tokens is not None and retry_max_tokens <= 0:
        raise ValueError("--empty_length_retry_max_tokens must be positive when provided.")
    if retry_rounds < 0:
        raise ValueError("--max_empty_length_retry_rounds must be non-negative.")
    if retry_rounds > 0:
        if retry_max_tokens is None:
            raise ValueError(
                "--empty_length_retry_max_tokens is required when "
                "--max_empty_length_retry_rounds is positive."
            )
        if retry_max_tokens <= args.api_max_tokens:
            raise ValueError(
                "--empty_length_retry_max_tokens must exceed --api_max_tokens "
                "when adaptive retries are enabled."
            )


def validate_risk_gate_runtime_args(args: argparse.Namespace) -> None:
    """Fail closed on inconsistent stateful runtime-gate configurations."""
    biasnet_max_tokens = getattr(args, "biasnet_max_tokens", None)
    if biasnet_max_tokens is not None:
        if int(biasnet_max_tokens) <= 0:
            raise ValueError("--biasnet_max_tokens must be positive when provided.")
        if not getattr(args, "biasnet_ckpt", None):
            raise ValueError("--biasnet_max_tokens requires --biasnet_ckpt.")
        if getattr(args, "risk_gate_checkpoint", None):
            raise ValueError(
                "--biasnet_max_tokens cannot be combined with --risk_gate_checkpoint."
            )

    threshold = float(args.risk_gate_threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("--risk_gate_threshold must be finite and in [0, 1].")
    if args.risk_gate_warmup_tokens < 0:
        raise ValueError("--risk_gate_warmup_tokens must be non-negative.")
    soft_temperature = float(args.risk_gate_soft_temperature)
    if (
        args.risk_gate_mode == "soft"
        and (
            not math.isfinite(soft_temperature)
            or soft_temperature <= 0.0
        )
    ):
        raise ValueError(
            "--risk_gate_soft_temperature must be finite and positive in soft mode."
        )

    min_scale = float(args.risk_gate_min_scale)
    if not math.isfinite(min_scale) or not 0.0 <= min_scale <= 1.0:
        raise ValueError("--risk_gate_min_scale must be finite and in [0, 1].")
    if args.risk_gate_latch_patience <= 0:
        raise ValueError("--risk_gate_latch_patience must be positive.")

    latch_threshold = args.risk_gate_latch_threshold
    if latch_threshold is not None:
        latch_threshold = float(latch_threshold)
        if (
            not math.isfinite(latch_threshold)
            or not 0.0 <= latch_threshold <= 1.0
        ):
            raise ValueError(
                "--risk_gate_latch_threshold must be finite and in [0, 1]."
            )
        if not args.risk_gate_latch_off:
            raise ValueError(
                "--risk_gate_latch_threshold requires --risk_gate_latch_off."
            )
    if args.risk_gate_handoff_on_latch and not args.risk_gate_latch_off:
        raise ValueError(
            "--risk_gate_handoff_on_latch requires --risk_gate_latch_off."
        )

    speculative_enabled = bool(
        getattr(args, "risk_gate_speculative_draft", False)
    )
    speculative_min_base_streak = int(
        getattr(args, "risk_gate_speculative_min_base_streak", 2)
    )
    speculative_draft_tokens = int(
        getattr(args, "risk_gate_speculative_draft_tokens", 80)
    )
    if speculative_min_base_streak <= 0:
        raise ValueError(
            "--risk_gate_speculative_min_base_streak must be positive."
        )
    if speculative_draft_tokens <= 0:
        raise ValueError(
            "--risk_gate_speculative_draft_tokens must be positive."
        )
    if speculative_enabled:
        if min_scale <= 0.0:
            raise ValueError(
                "--risk_gate_speculative_draft requires a positive "
                "--risk_gate_min_scale."
            )
        if args.risk_gate_latch_off or args.risk_gate_handoff_on_latch:
            raise ValueError(
                "Speculative drafting cannot be combined with latch-off/handoff."
            )

    stateful_gate_requested = bool(
        min_scale > 0.0
        or args.risk_gate_latch_off
        or args.risk_gate_handoff_on_latch
        or speculative_enabled
        or args.risk_gate_trace
    )
    if stateful_gate_requested and not args.biasnet_ckpt:
        raise ValueError(
            "Stateful risk-gate controls require --biasnet_ckpt."
        )
    if stateful_gate_requested and not args.risk_gate_checkpoint:
        raise ValueError(
            "Stateful risk-gate controls require --risk_gate_checkpoint."
        )


def snapshot_openrouter_audit(client: OpenRouterClient) -> dict:
    return {
        "calls": int(client.calls),
        "request_attempts": int(getattr(client, "request_attempts", 0)),
        "mc_requested_samples": int(
            getattr(client, "mc_requested_samples", 0)
        ),
        "cost": float(client.total_cost),
        "provider_call_counts": dict(client.provider_call_counts),
        "response_model_call_counts": dict(client.response_model_call_counts),
        "response_cache_status_counts": dict(client.response_cache_status_counts),
        "missing_router_metadata_calls": int(client.missing_router_metadata_calls),
        "api_max_tokens_call_counts": dict(
            getattr(client, "api_max_tokens_call_counts", {})
        ),
        "empty_length_retry_attempts": int(
            getattr(client, "empty_length_retry_attempts", 0)
        ),
        "empty_length_retry_recoveries": int(
            getattr(client, "empty_length_retry_recoveries", 0)
        ),
        "empty_length_retry_exhaustions": int(
            getattr(client, "empty_length_retry_exhaustions", 0)
        ),
        "reasoning_response_calls": int(
            getattr(client, "reasoning_response_calls", 0)
        ),
        "reasoning_message_choices": int(
            getattr(client, "reasoning_message_choices", 0)
        ),
        "reasoning_tokens": int(getattr(client, "reasoning_tokens", 0)),
        "reasoning_retry_attempts": int(
            getattr(client, "reasoning_retry_attempts", 0)
        ),
        "reasoning_retry_recoveries": int(
            getattr(client, "reasoning_retry_recoveries", 0)
        ),
        "reasoning_retry_exhaustions": int(
            getattr(client, "reasoning_retry_exhaustions", 0)
        ),
        "reasoning_fallback_activations": int(
            getattr(client, "reasoning_fallback_activations", 0)
        ),
        "reasoning_fallback_response_calls": int(
            getattr(client, "reasoning_fallback_response_calls", 0)
        ),
        "reasoning_fallback_recoveries": int(
            getattr(client, "reasoning_fallback_recoveries", 0)
        ),
        "reasoning_fallback_exhaustions": int(
            getattr(client, "reasoning_fallback_exhaustions", 0)
        ),
        "reasoning_fallback_provider_call_counts": dict(
            getattr(client, "reasoning_fallback_provider_call_counts", {})
        ),
        "reasoning_fallback_response_model_call_counts": dict(
            getattr(
                client,
                "reasoning_fallback_response_model_call_counts",
                {},
            )
        ),
    }


def _count_delta(current: dict, before: dict, field_name: str) -> dict:
    result = {}
    for key, value in current.items():
        delta = int(value) - int(before.get(key, 0))
        if delta < 0:
            raise RuntimeError(f"{field_name}[{key!r}] decreased during generation.")
        if delta > 0:
            result[str(key)] = delta
    for key, value in before.items():
        if key not in current and int(value) != 0:
            raise RuntimeError(f"{field_name}[{key!r}] disappeared during generation.")
    return result


def _integer_delta(current: int, before: int, field_name: str) -> int:
    delta = int(current) - int(before)
    if delta < 0:
        raise RuntimeError(f"{field_name} decreased during generation.")
    return delta


def build_openrouter_routing_metadata(
    client: OpenRouterClient,
    args: argparse.Namespace,
    before: dict,
    anytime_policy: Optional[AnytimePolicy] = None,
) -> dict:
    current = snapshot_openrouter_audit(client)
    return {
        "requested_model": args.model,
        "requested_provider_order": args.provider_order,
        "requested_provider_quantizations": getattr(
            args, "provider_quantizations", None
        ),
        "provider_allow_fallbacks": args.provider_allow_fallbacks,
        "router_metadata_requested": bool(args.router_metadata),
        "openrouter_response_cache_disabled": bool(
            args.disable_openrouter_response_cache
        ),
        "empty_response_token": args.empty_response_token,
        "api_max_tokens": int(args.api_max_tokens),
        "empty_length_retry_max_tokens": args.empty_length_retry_max_tokens,
        "max_empty_length_retry_rounds": int(
            args.max_empty_length_retry_rounds
        ),
        "qwen_hard_no_think_prefill": bool(
            getattr(args, "qwen_hard_no_think_prefill", False)
        ),
        "claude_45_whitespace_prefill_fix": (
            CLAUDE_45_WHITESPACE_PREFILL_FIX
            if is_claude_45_model(args.model)
            else None
        ),
        "mask_non_eos_special_tokens": bool(
            getattr(args, "mask_non_eos_special_tokens", False)
        ),
        "initial_prefix": getattr(args, "initial_prefix", "") or "",
        "bootstrap_biasnet_checkpoint": getattr(
            args,
            "bootstrap_biasnet_ckpt",
            None,
        ),
        "bootstrap_biasnet_tokens": int(
            getattr(args, "bootstrap_biasnet_tokens", 0)
        ),
        "biasnet_max_tokens": getattr(args, "biasnet_max_tokens", None),
        "mc_input_representation": getattr(
            args, "mc_input_representation", FLOOR_LOGPROB
        ),
        "mc_base_score_representation": getattr(
            args, "mc_base_score_representation", None
        ) or getattr(args, "mc_input_representation", FLOOR_LOGPROB),
        "mc_score_interface": (
            "dual_v1"
            if (getattr(args, "mc_base_score_representation", None)
                or getattr(args, "mc_input_representation", FLOOR_LOGPROB))
            != getattr(args, "mc_input_representation", FLOOR_LOGPROB)
            else "shared_v1"
        ),
        "mc_log_count_alpha": (
            getattr(args, "mc_log_count_alpha", None)
            if getattr(args, "mc_input_representation", FLOOR_LOGPROB) == LOG_COUNT
            else None
        ),
        "mc_samples_per_token": int(getattr(args, "mc_samples_per_token", 50)),
        "mc_independent_views": int(
            getattr(args, "mc_independent_views", 1)
        ),
        "mc_ensemble": mc_ensemble_audit(args, anytime_policy),
        "mc_sample_temperature": float(
            getattr(args, "mc_sample_temperature", 1.0)
        ),
        "mc_top_p": float(getattr(args, "mc_top_p", 1.0)),
        "mc_observed_alpha": float(getattr(args, "mc_observed_alpha", 0.1)),
        "mc_floor_mass": float(getattr(args, "mc_floor_mass", 1e-4)),
        "mc_store_dtype": getattr(args, "store_dtype", None),
        "mc_completion_policy": getattr(
            args, "sample_completion_policy", "partial"
        ),
        "reasoning_mode": getattr(args, "reasoning_mode", None),
        "reject_reasoning_tokens": bool(
            getattr(args, "reject_reasoning_tokens", False)
        ),
        "max_reasoning_retries": int(
            getattr(args, "max_reasoning_retries", 0)
        ),
        "reasoning_retry_sleep": float(
            getattr(args, "reasoning_retry_sleep", 0.0)
        ),
        "reasoning_fallback_temperature": getattr(
            args,
            "reasoning_fallback_temperature",
            None,
        ),
        "reasoning_fallback_mode": getattr(
            args,
            "reasoning_fallback_mode",
            None,
        ),
        "max_reasoning_fallback_retries": int(
            getattr(args, "max_reasoning_fallback_retries", 0)
        ),
        "max_api_request_attempts": getattr(
            args, "max_api_request_attempts", None
        ),
        "max_mc_requested_samples": getattr(
            args, "max_mc_requested_samples", None
        ),
        "anytime_policy_spec": getattr(args, "anytime_policy_spec", None),
        "anytime_policy_name": getattr(args, "anytime_policy_name", None),
        "actual_provider_call_counts": _count_delta(
            current["provider_call_counts"],
            before["provider_call_counts"],
            "provider_call_counts",
        ),
        "actual_response_model_call_counts": _count_delta(
            current["response_model_call_counts"],
            before["response_model_call_counts"],
            "response_model_call_counts",
        ),
        "response_cache_status_call_counts": _count_delta(
            current["response_cache_status_counts"],
            before["response_cache_status_counts"],
            "response_cache_status_counts",
        ),
        "api_max_tokens_call_counts": _count_delta(
            current["api_max_tokens_call_counts"],
            before["api_max_tokens_call_counts"],
            "api_max_tokens_call_counts",
        ),
        "empty_length_retry_attempts": _integer_delta(
            current["empty_length_retry_attempts"],
            before["empty_length_retry_attempts"],
            "empty_length_retry_attempts",
        ),
        "empty_length_retry_recoveries": _integer_delta(
            current["empty_length_retry_recoveries"],
            before["empty_length_retry_recoveries"],
            "empty_length_retry_recoveries",
        ),
        "empty_length_retry_exhaustions": _integer_delta(
            current["empty_length_retry_exhaustions"],
            before["empty_length_retry_exhaustions"],
            "empty_length_retry_exhaustions",
        ),
        "reasoning_response_calls": _integer_delta(
            current["reasoning_response_calls"],
            before["reasoning_response_calls"],
            "reasoning_response_calls",
        ),
        "reasoning_message_choices": _integer_delta(
            current["reasoning_message_choices"],
            before["reasoning_message_choices"],
            "reasoning_message_choices",
        ),
        "reasoning_tokens": _integer_delta(
            current["reasoning_tokens"],
            before["reasoning_tokens"],
            "reasoning_tokens",
        ),
        "reasoning_retry_attempts": _integer_delta(
            current["reasoning_retry_attempts"],
            before["reasoning_retry_attempts"],
            "reasoning_retry_attempts",
        ),
        "reasoning_retry_recoveries": _integer_delta(
            current["reasoning_retry_recoveries"],
            before["reasoning_retry_recoveries"],
            "reasoning_retry_recoveries",
        ),
        "reasoning_retry_exhaustions": _integer_delta(
            current["reasoning_retry_exhaustions"],
            before["reasoning_retry_exhaustions"],
            "reasoning_retry_exhaustions",
        ),
        "reasoning_fallback_activations": _integer_delta(
            current["reasoning_fallback_activations"],
            before["reasoning_fallback_activations"],
            "reasoning_fallback_activations",
        ),
        "reasoning_fallback_response_calls": _integer_delta(
            current["reasoning_fallback_response_calls"],
            before["reasoning_fallback_response_calls"],
            "reasoning_fallback_response_calls",
        ),
        "reasoning_fallback_recoveries": _integer_delta(
            current["reasoning_fallback_recoveries"],
            before["reasoning_fallback_recoveries"],
            "reasoning_fallback_recoveries",
        ),
        "reasoning_fallback_exhaustions": _integer_delta(
            current["reasoning_fallback_exhaustions"],
            before["reasoning_fallback_exhaustions"],
            "reasoning_fallback_exhaustions",
        ),
        "reasoning_fallback_provider_call_counts": _count_delta(
            current["reasoning_fallback_provider_call_counts"],
            before["reasoning_fallback_provider_call_counts"],
            "reasoning_fallback_provider_call_counts",
        ),
        "reasoning_fallback_response_model_call_counts": _count_delta(
            current["reasoning_fallback_response_model_call_counts"],
            before["reasoning_fallback_response_model_call_counts"],
            "reasoning_fallback_response_model_call_counts",
        ),
        "missing_router_metadata_calls": _integer_delta(
            current["missing_router_metadata_calls"],
            before["missing_router_metadata_calls"],
            "missing_router_metadata_calls",
        ),
        "api_calls": _integer_delta(
            current["calls"],
            before["calls"],
            "api_calls",
        ),
        "api_request_attempts": _integer_delta(
            current["request_attempts"],
            before["request_attempts"],
            "api_request_attempts",
        ),
        "mc_requested_samples": _integer_delta(
            current["mc_requested_samples"],
            before["mc_requested_samples"],
            "mc_requested_samples",
        ),
        "api_cost": current["cost"] - before["cost"],
        "max_routing_attempt_cumulative": client.max_routing_attempt,
    }


def validate_reasoning_retry_args(args: argparse.Namespace) -> None:
    """Fail closed on invalid or ambiguous reasoning-retry policies."""

    max_reasoning_retries = int(getattr(args, "max_reasoning_retries", 0))
    reasoning_retry_sleep = float(getattr(args, "reasoning_retry_sleep", 0.0))
    max_fallback_retries = int(
        getattr(args, "max_reasoning_fallback_retries", 0)
    )
    fallback_temperature = getattr(
        args,
        "reasoning_fallback_temperature",
        None,
    )
    fallback_mode = getattr(args, "reasoning_fallback_mode", None)
    reject_reasoning = bool(getattr(args, "reject_reasoning_tokens", False))
    primary_mode = getattr(args, "reasoning_mode", "enabled_false")

    if max_reasoning_retries < 0:
        raise ValueError("--max_reasoning_retries must be non-negative.")
    if not math.isfinite(reasoning_retry_sleep) or reasoning_retry_sleep < 0:
        raise ValueError(
            "--reasoning_retry_sleep must be finite and non-negative."
        )
    if max_reasoning_retries > 0 and not reject_reasoning:
        raise ValueError(
            "--max_reasoning_retries requires --reject_reasoning_tokens."
        )
    if max_fallback_retries < 0:
        raise ValueError(
            "--max_reasoning_fallback_retries must be non-negative."
        )
    if fallback_temperature is not None and fallback_mode is not None:
        raise ValueError(
            "--reasoning_fallback_temperature and --reasoning_fallback_mode "
            "are mutually exclusive."
        )
    fallback_configured = (
        fallback_temperature is not None or fallback_mode is not None
    )
    if max_fallback_retries > 0 and not fallback_configured:
        raise ValueError(
            "--max_reasoning_fallback_retries requires "
            "--reasoning_fallback_temperature or --reasoning_fallback_mode."
        )
    if fallback_temperature is not None:
        fallback_temperature = float(fallback_temperature)
        if not math.isfinite(fallback_temperature) or fallback_temperature <= 0:
            raise ValueError(
                "--reasoning_fallback_temperature must be finite and positive."
            )
    if fallback_mode is not None:
        if fallback_mode != "effort_none":
            raise ValueError(
                "--reasoning_fallback_mode currently supports only effort_none."
            )
        if primary_mode != "enabled_false":
            raise ValueError(
                "--reasoning_fallback_mode=effort_none requires "
                "--reasoning_mode=enabled_false."
            )
        if max_fallback_retries > 64:
            raise ValueError(
                "--max_reasoning_fallback_retries cannot exceed 64 with "
                "--reasoning_fallback_mode."
            )
    if fallback_configured:
        if not reject_reasoning:
            raise ValueError(
                "A reasoning fallback requires --reject_reasoning_tokens."
            )
        if max_reasoning_retries <= 0:
            raise ValueError(
                "A reasoning fallback requires positive "
                "--max_reasoning_retries."
            )


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be positive.")
    if args.api_max_tokens <= 0:
        raise ValueError("--api_max_tokens must be positive.")
    if args.mc_samples_per_token <= 0:
        raise ValueError("--mc_samples_per_token must be positive.")
    for field in ("max_api_request_attempts", "max_mc_requested_samples"):
        value = getattr(args, field, None)
        if value is not None and int(value) <= 0:
            raise ValueError(f"--{field} must be positive when provided.")
    validate_mc_ensemble_args(args)
    if args.parallel_requests <= 0:
        raise ValueError("--parallel_requests must be positive.")
    if args.sample_choices_per_request <= 0:
        raise ValueError("--sample_choices_per_request must be positive.")
    if args.max_sample_refill_rounds < 0:
        raise ValueError("--max_sample_refill_rounds must be non-negative.")
    validate_reasoning_retry_args(args)
    if args.bootstrap_biasnet_tokens < 0:
        raise ValueError("--bootstrap_biasnet_tokens must be non-negative.")
    if args.bootstrap_biasnet_tokens > 0 and not args.bootstrap_biasnet_ckpt:
        raise ValueError(
            "--bootstrap_biasnet_tokens requires --bootstrap_biasnet_ckpt."
        )
    if args.bootstrap_biasnet_ckpt and args.bootstrap_biasnet_tokens <= 0:
        raise ValueError(
            "--bootstrap_biasnet_ckpt requires positive --bootstrap_biasnet_tokens."
        )
    if args.bootstrap_biasnet_ckpt and not args.biasnet_ckpt:
        raise ValueError(
            "--bootstrap_biasnet_ckpt requires the main --biasnet_ckpt."
        )
    validate_empty_length_retry_args(args)
    validate_risk_gate_runtime_args(args)

    prompt_records = collect_prompt_records(args)
    if args.begin < 0:
        raise ValueError("--begin must be non-negative.")
    if args.end is not None and args.end <= args.begin:
        raise ValueError("--end must be greater than --begin.")
    prompt_records = prompt_records[args.begin : args.end]
    prompts = [record.prompt for record in prompt_records]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive when provided.")
        prompt_records = prompt_records[: args.limit]
        prompts = prompts[: args.limit]
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(
        args.tokenizer_name,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
        revision=args.tokenizer_revision,
        fix_mistral_regex=args.fix_mistral_regex,
    )
    api_key = resolve_api_key(args)
    client = OpenRouterClient(args, api_key)
    bias_model = load_biasnet(args.biasnet_ckpt, device, args.biasnet_dtype)
    bootstrap_bias_model = load_biasnet(
        args.bootstrap_biasnet_ckpt,
        device,
        args.biasnet_dtype,
    )
    resolve_mc_input_configuration(
        args,
        [bias_model, bootstrap_bias_model],
    )
    anytime_policy = load_anytime_policy(
        args, bias_model, bootstrap_bias_model
    )
    context_contract = context_contract_for_models(
        [bias_model, bootstrap_bias_model], anytime_policy,
    )
    context_encoder = (
        FrozenContextEncoder.from_contract(
            context_contract, device=args.context_device or device,
            model_override=args.context_encoder_model,
            tokenizer_override=args.context_encoder_tokenizer,
            local_files_only=args.context_local_files_only,
        ) if context_contract is not None else None
    )
    validate_anytime_runtime(
        args,
        anytime_policy,
        [bias_model, bootstrap_bias_model],
        runtime_device=device,
    )
    validate_support_restriction(args)
    resolve_proxy_fusion_configuration(
        args,
        [bias_model, bootstrap_bias_model],
    )
    resolve_static_prior_configuration(
        args,
        [bias_model, bootstrap_bias_model],
    )
    for name, model in (
        ("main", bias_model),
        ("bootstrap", bootstrap_bias_model),
    ):
        if model is not None and len(tokenizer) != model.vocab_size:
            raise ValueError(
                f"{name} BiasNet vocabulary does not match the tokenizer: "
                f"checkpoint={model.vocab_size}, tokenizer={len(tokenizer)}."
            )
    proxy = load_local_proxy(args, tokenizer, device)
    validate_proxy_runtime_against_checkpoints(
        proxy,
        [bias_model, bootstrap_bias_model],
    )
    load_static_prior_log_probs(args, len(tokenizer), device)
    risk_gate = load_risk_gate(args, device) if bias_model is not None else None

    completed = count_jsonl_records(args.output_json) if args.resume else 0
    if completed:
        print(f"Resuming from {completed} existing records in {args.output_json}.", flush=True)
        prompt_records = prompt_records[completed:]
        prompts = prompts[completed:]
    elif os.path.exists(args.output_json):
        os.remove(args.output_json)

    for batch_index, batch_records in enumerate(chunked(prompt_records, 1), start=completed + 1):
        batch = [record.prompt for record in batch_records]
        generation_batch = [
            prompt + "\n/no_think" if args.append_no_think else prompt
            for prompt in batch
        ]
        if args.progress_steps:
            print(f"starting_prompt={batch_index}/{completed + len(prompts)}", flush=True)
        audit_before = snapshot_openrouter_audit(client)
        trace_enabled = bool(
            args.risk_gate_trace
            or args.generation_trace
            or proxy is not None
            or anytime_policy is not None
            or context_encoder is not None
        )
        generation_audits: list[Optional[dict[str, Any]]] = [
            {} if trace_enabled else None
            for _ in batch
        ]
        completions = []
        for prompt_offset, (prompt, generation_audit) in enumerate(
            zip(generation_batch, generation_audits)
        ):
            generation_started = time.perf_counter()
            try:
                completion = generate_one(
                    client=client,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    risk_gate_prompt=(
                        batch_records[prompt_offset].prompt
                        if getattr(args, "risk_gate_prompt_source", "target")
                        == "dataset"
                        else prompt
                    ),
                    args=args,
                    device=device,
                    bias_model=bias_model,
                    bootstrap_bias_model=bootstrap_bias_model,
                    risk_gate=risk_gate,
                    generation_audit=generation_audit,
                    proxy=proxy,
                    anytime_policy=anytime_policy,
                    context_encoder=context_encoder,
                )
            except FatalOpenRouterResponseError as exc:
                # The prefix reached before the abort is the most diagnostic
                # part of a crashed run, so keep it instead of losing the whole
                # attempt. The prompt itself stays unwritten and will be retried
                # by --resume.
                dump_path = crash_dump_path(args.output_json)
                append_crash_dump(
                    dump_path,
                    prompt=prompt,
                    tokenizer=tokenizer,
                    generation_audit=generation_audit,
                    error=exc,
                    record_metadata=dict(
                        batch_records[prompt_offset].metadata
                    ),
                )
                print(
                    f"crash_dump_written={dump_path} "
                    f"steps={len((generation_audit or {}).get('steps') or [])}",
                    flush=True,
                )
                raise
            if generation_audit is not None:
                generation_audit["wall_time_seconds"] = (
                    time.perf_counter() - generation_started
                )
            completions.append(completion)
        record_metadata = [dict(record.metadata) for record in batch_records]
        if args.router_metadata or trace_enabled:
            if args.router_metadata:
                routing_metadata = build_openrouter_routing_metadata(
                    client,
                    args,
                    audit_before,
                    anytime_policy,
                )
                for metadata in record_metadata:
                    metadata["openrouter_routing"] = dict(routing_metadata)
            if trace_enabled:
                for metadata, generation_audit in zip(
                    record_metadata,
                    generation_audits,
                ):
                    metadata["risk_gate_runtime"] = generation_audit
        append_jsonl(args.output_json, batch, completions, record_metadata)
        print(
            f"written={sum(1 for _ in open(args.output_json, encoding='utf-8'))} "
            f"api_calls={client.calls} approx_cost=${client.total_cost:.6f}",
            flush=True,
        )

    print(
        f"Done. records={completed + len(prompts)} api_calls={client.calls} "
        f"approx_cost=${client.total_cost:.6f}"
    )


if __name__ == "__main__":
    main()
