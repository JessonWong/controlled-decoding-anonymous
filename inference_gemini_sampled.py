import argparse
import importlib.util
import json
import os
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import torch
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions, ThinkingConfig
from benchmark_data import collect_prompt_records
from modeling_biasnet import BiasNet
from risk_gate import PrefixRiskGate
from inference_openrouter import risk_gate_scale_from_score, risk_gate_score_and_scale

TRAINING_DIR = Path(__file__).resolve().parent / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from pre_logits_sampled_openweight import sampled_ids_to_log_probs  # noqa: E402
from pre_logits_sampled_openrouter import load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BiasNet generation with MC-sampled Gemini next-token estimates."
    )
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--tokenizer_name", default="google/gemma-3-1b-pt")
    parser.add_argument("--tokenizer_use_fast", action="store_true")
    parser.add_argument(
        "--distribution_vocab_size",
        type=int,
        default=None,
        help="Distribution/BiasNet size. Defaults to tokenizer.vocab_size.",
    )
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument(
        "--benchmark",
        choices=["advbench", "harmbench", "sorrybench"],
        default=None,
        help="Load prompts from a benchmark source instead of --prompt_file.",
    )
    parser.add_argument(
        "--benchmark_file",
        default=None,
        help="Local benchmark file (HarmBench CSV or SORRY-Bench JSONL).",
    )
    parser.add_argument(
        "--benchmark_mutation",
        default=None,
        help="SORRY-Bench mutation suffix, for example slang or atbash.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--begin",
        type=int,
        default=0,
        help="Inclusive index within the records selected by --limit.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Exclusive index within the records selected by --limit.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--biasnet_ckpt", default=None)
    parser.add_argument("--biasnet_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mc_samples_per_token", type=int, default=50)
    parser.add_argument("--mc_sample_temperature", type=float, default=1.0)
    parser.add_argument("--mc_top_p", type=float, default=1.0)
    parser.add_argument("--mc_observed_alpha", type=float, default=0.1)
    parser.add_argument("--mc_floor_mass", type=float, default=1e-4)
    parser.add_argument("--candidate_count", type=int, default=8)
    parser.add_argument("--parallel_requests", type=int, default=8)
    parser.add_argument("--api_max_output_tokens", type=int, default=1)
    parser.add_argument(
        "--empty_response_token",
        choices=["skip", "stop_eos"],
        default="skip",
        help="Optionally project an empty STOP candidate to canonical tokenizer EOS.",
    )
    parser.add_argument("--thinking_budget", type=int, default=0)
    parser.add_argument("--max_batch_calls", type=int, default=40)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--retry_sleep", type=float, default=1.0)
    parser.add_argument(
        "--request_timeout_ms",
        type=int,
        default=60_000,
        help="Per-request Gemini HTTP timeout in milliseconds.",
    )
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_key_env", default="GEMINI_API_KEY")
    parser.add_argument("--api_key_file", default=None)
    parser.add_argument("--progress_steps", action="store_true")
    parser.add_argument("--stop_on_mc_failure", action="store_true")
    parser.add_argument("--risk_gate_checkpoint", default=None)
    parser.add_argument("--risk_gate_threshold", type=float, default=0.1)
    parser.add_argument("--risk_gate_batch_size", type=int, default=8)
    parser.add_argument("--risk_gate_max_length", type=int, default=None)
    parser.add_argument(
        "--risk_gate_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--risk_gate_device", default=None)
    parser.add_argument("--risk_gate_model_name", default=None)
    parser.add_argument("--risk_gate_load_in_4bit", action="store_true")
    parser.add_argument("--risk_gate_local_files_only", action="store_true")
    parser.add_argument("--risk_gate_mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--risk_gate_soft_temperature", type=float, default=0.05)
    parser.add_argument("--risk_gate_warmup_tokens", type=int, default=0)
    parser.add_argument("--risk_gate_min_scale", type=float, default=0.0)
    parser.add_argument("--risk_gate_speculative_draft", action="store_true")
    parser.add_argument("--risk_gate_speculative_min_base_streak", type=int, default=2)
    parser.add_argument("--risk_gate_speculative_draft_tokens", type=int, default=80)
    parser.add_argument("--risk_gate_trace", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_api_key_from_file(path: str) -> Optional[str]:
    spec = importlib.util.spec_from_file_location("juli_api_key", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY_ALTERNATE",
        "GEMINI_API_KEY_SECONDARY",
        "gemini_api_key",
        "api_key",
    ):
        value = getattr(module, name, None)
        if value:
            return str(value)
    return None


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    env_value = os.environ.get(args.api_key_env)
    if env_value:
        return env_value
    if args.api_key_file:
        file_value = read_api_key_from_file(args.api_key_file)
        if file_value:
            return file_value
    raise ValueError("Gemini API key not found.")


def format_contents(question: str, answer_prefix: str) -> list[dict]:
    contents = [{"role": "user", "parts": [{"text": question}]}]
    if answer_prefix:
        contents.append({"role": "model", "parts": [{"text": answer_prefix}]})
    return contents


def candidate_text(candidate) -> str:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return ""
    return "".join((getattr(part, "text", "") or "") for part in parts)


class GeminiMCSampler:
    def __init__(self, args: argparse.Namespace, tokenizer) -> None:
        self.args = args
        self.tokenizer = tokenizer
        self.vocab_size = (
            int(args.distribution_vocab_size)
            if getattr(args, "distribution_vocab_size", None) is not None
            else int(tokenizer.vocab_size)
        )
        if self.vocab_size <= 1 or self.vocab_size > len(tokenizer):
            raise ValueError(
                f"Invalid distribution vocab size {self.vocab_size} for "
                f"tokenizer length {len(tokenizer)}."
            )
        self.client = genai.Client(
            api_key=resolve_api_key(args),
            http_options=HttpOptions(timeout=args.request_timeout_ms),
        )
        self.total_calls = 0
        self.total_valid = 0
        self.total_empty = 0
        self.total_multi = 0
        self.total_failed_calls = 0
        self._lock = threading.Lock()

    def _request(
        self,
        question: str,
        prefix: str,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        candidate_count: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ):
        config = GenerateContentConfig(
            max_output_tokens=(
                self.args.api_max_output_tokens
                if max_output_tokens is None
                else int(max_output_tokens)
            ),
            temperature=(
                self.args.mc_sample_temperature
                if temperature is None
                else temperature
            ),
            top_p=self.args.mc_top_p if top_p is None else top_p,
            candidate_count=(
                self.args.candidate_count
                if candidate_count is None
                else candidate_count
            ),
            thinking_config=ThinkingConfig(thinking_budget=self.args.thinking_budget),
            safety_settings=[
                {
                    "category": category,
                    "threshold": "BLOCK_NONE",
                }
                for category in (
                    "HARM_CATEGORY_HARASSMENT",
                    "HARM_CATEGORY_HATE_SPEECH",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "HARM_CATEGORY_CIVIC_INTEGRITY",
                )
            ],
        )
        for attempt in range(self.args.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.args.model,
                    contents=format_contents(question, prefix),
                    config=config,
                )
                with self._lock:
                    self.total_calls += 1
                return response
            except Exception:
                with self._lock:
                    self.total_failed_calls += 1
                if attempt >= self.args.max_retries:
                    raise
                time.sleep(self.args.retry_sleep * (2**attempt) + random.random())
        raise RuntimeError("Unreachable Gemini retry state.")

    def sample_ids(self, question: str, prefix: str) -> tuple[list[int], dict]:
        sampled_ids: list[int] = []
        stats = Counter()
        while (
            len(sampled_ids) < self.args.mc_samples_per_token
            and stats["calls"] < self.args.max_batch_calls
        ):
            remaining_calls = self.args.max_batch_calls - stats["calls"]
            remaining_samples = self.args.mc_samples_per_token - len(sampled_ids)
            wave_size = min(
                self.args.parallel_requests,
                remaining_calls,
                max(1, remaining_samples),
            )
            stats["calls"] += wave_size
            with ThreadPoolExecutor(max_workers=wave_size) as executor:
                futures = [
                    executor.submit(self._request, question, prefix)
                    for _ in range(wave_size)
                ]
                for future in as_completed(futures):
                    try:
                        response = future.result()
                    except Exception:
                        stats["failed_calls"] += 1
                        continue
                    candidates = response.candidates or []
                    if not candidates:
                        stats["no_candidates"] += 1
                        continue
                    for candidate in candidates:
                        text = candidate_text(candidate)
                        if not text:
                            finish_reason = str(
                                getattr(candidate, "finish_reason", "")
                            ).strip().casefold()
                            if (
                                self.args.empty_response_token == "stop_eos"
                                and "stop" in finish_reason
                                and self.tokenizer.eos_token_id is not None
                            ):
                                sampled_ids.append(
                                    int(self.tokenizer.eos_token_id)
                                )
                                stats["canonical_eos_projected"] += 1
                                self.total_valid += 1
                                if (
                                    len(sampled_ids)
                                    >= self.args.mc_samples_per_token
                                ):
                                    break
                                continue
                            stats["empty"] += 1
                            self.total_empty += 1
                            continue
                        token_ids = self.tokenizer.encode(
                            text, add_special_tokens=False
                        )
                        if not token_ids:
                            stats["empty_local_tokens"] += 1
                            continue
                        if len(token_ids) != 1:
                            stats["multi_token_first_used"] += 1
                            self.total_multi += 1
                        token_id = int(token_ids[0])
                        if token_id < 0 or token_id >= self.vocab_size:
                            stats["out_of_vocab"] += 1
                            continue
                        sampled_ids.append(token_id)
                        self.total_valid += 1
                        if len(sampled_ids) >= self.args.mc_samples_per_token:
                            break
        stats["valid"] = len(sampled_ids)
        return sampled_ids, dict(stats)

    def deterministic_token_id(
        self, question: str, prefix: str
    ) -> tuple[Optional[int], dict[str, Any]]:
        response = self._request(
            question,
            prefix,
            temperature=0.0,
            top_p=1.0,
            candidate_count=1,
        )
        candidates = response.candidates or []
        info: dict[str, Any] = {
            "candidate_count": len(candidates),
            "finish_reason": (
                str(getattr(candidates[0], "finish_reason", ""))
                if candidates
                else None
            ),
        }
        if not candidates:
            return None, info
        text = candidate_text(candidates[0])
        info["text"] = text
        if not text:
            finish_reason = str(
                getattr(candidates[0], "finish_reason", "")
            ).strip().casefold()
            if (
                self.args.empty_response_token == "stop_eos"
                and "stop" in finish_reason
                and self.tokenizer.eos_token_id is not None
            ):
                info["canonical_eos_projected"] = True
                return int(self.tokenizer.eos_token_id), info
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        info["local_token_count"] = len(token_ids)
        if not token_ids:
            return None, info
        return int(token_ids[0]), info

    def draft_token_ids(
        self,
        question: str,
        prefix: str,
        remaining_tokens: int,
        draft_tokens: int,
    ) -> tuple[list[int], dict[str, Any]]:
        requested_tokens = min(int(remaining_tokens), int(draft_tokens))
        if requested_tokens <= 0:
            raise ValueError("Speculative draft token budget must be positive.")
        response = self._request(
            question,
            prefix,
            temperature=0.0,
            top_p=1.0,
            candidate_count=1,
            max_output_tokens=requested_tokens,
        )
        candidates = response.candidates or []
        finish_reason = (
            str(getattr(candidates[0], "finish_reason", ""))
            if candidates
            else None
        )
        text = candidate_text(candidates[0]) if candidates else ""
        token_ids = self.tokenizer.encode(text, add_special_tokens=False) if text else []
        isolated_count = len(token_ids)
        token_ids = token_ids[:requested_tokens]
        return [int(token_id) for token_id in token_ids], {
            "requested_tokens": requested_tokens,
            "finish_reason": finish_reason,
            "response_chars": len(text),
            "draft_local_token_count": len(token_ids),
            "draft_isolated_local_token_count": isolated_count,
            "draft_local_tokens_truncated": max(0, isolated_count - len(token_ids)),
        }


def resolve_device(value: Optional[str]) -> torch.device:
    if value:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_biasnet(
    checkpoint: Optional[str], device: torch.device, dtype_name: str
) -> Optional[BiasNet]:
    if not checkpoint:
        return None
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    model = BiasNet.from_pretrained(checkpoint, map_location="cpu")
    model = model.to(device=device, dtype=dtype)
    model.set_up_proj()
    model.eval()
    return model


def load_risk_gate(
    args: argparse.Namespace, default_device: torch.device
) -> Optional[PrefixRiskGate]:
    if not args.risk_gate_checkpoint:
        return None
    device = (
        torch.device(args.risk_gate_device)
        if args.risk_gate_device
        else default_device
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
        local_files_only=args.risk_gate_local_files_only,
    )


def choose_token(log_probs: torch.Tensor, temperature: float) -> int:
    if temperature <= 0:
        return int(log_probs.argmax(dim=-1).item())
    probs = torch.softmax(log_probs.float() / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def generate_one(
    prompt: str,
    sampler: GeminiMCSampler,
    tokenizer,
    biasnet: Optional[BiasNet],
    risk_gate: Optional[PrefixRiskGate],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[str, dict, Optional[dict[str, Any]]]:
    generated = ""
    step_stats = Counter()
    trace_steps: list[dict[str, Any]] = []
    speculative_trace: list[dict[str, Any]] = []
    dtype = torch.float16 if args.biasnet_dtype == "float16" else torch.float32
    stop_reason = "max_new_tokens"
    speculative_enabled = bool(args.risk_gate_speculative_draft)
    speculative_base_streak = 0
    speculative_draft_calls = 0
    speculative_generated_tokens = 0
    speculative_verified_tokens = 0
    speculative_accepted_tokens = 0
    speculative_rejected_tokens = 0
    speculative_rollbacks = 0

    def append_token(token_id: int) -> bool:
        nonlocal generated, stop_reason
        if token_id == tokenizer.eos_token_id:
            stop_reason = "eos"
            return False
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if not token_text:
            stop_reason = "empty_local_decode"
            return False
        generated += token_text
        return True

    while len(trace_steps) < args.max_new_tokens:
        step = len(trace_steps)
        base_token_id: Optional[int] = None
        gate_score: Optional[float] = None
        raw_scale = 1.0
        effective_scale = 1.0
        decision = "bias"
        forced_warmup = risk_gate is not None and step < args.risk_gate_warmup_tokens

        if risk_gate is not None:
            base_token_id, base_info = sampler.deterministic_token_id(prompt, generated)
            if base_token_id is None:
                stop_reason = "invalid_deterministic_base_token"
                step_stats["invalid_deterministic_base_token"] += 1
                break
            if forced_warmup:
                decision = "warmup_bias"
            else:
                base_answer_prefix = generated + tokenizer.decode(
                    [base_token_id], skip_special_tokens=False
                )
                gate_score, raw_scale = risk_gate_score_and_scale(
                    risk_gate,
                    prompt,
                    base_answer_prefix,
                    mode=args.risk_gate_mode,
                    soft_temperature=args.risk_gate_soft_temperature,
                )
                if raw_scale <= args.risk_gate_min_scale:
                    effective_scale = 0.0
                    decision = "below_min_scale_base"
                else:
                    effective_scale = raw_scale
                    decision = "soft_bias" if raw_scale < 1.0 else "bias"

        if risk_gate is not None and effective_scale <= 0.0:
            token_id = int(base_token_id)
            step_stats["base_only_steps"] += 1
            valid_samples = 0
            sample_support = 0
        else:
            sample_ids, stats = sampler.sample_ids(prompt, generated)
            step_stats.update(stats)
            if len(sample_ids) < args.mc_samples_per_token:
                stop_reason = "mc_failure"
                if args.stop_on_mc_failure:
                    break
                raise RuntimeError(
                    f"Only {len(sample_ids)}/{args.mc_samples_per_token} usable Gemini "
                    f"samples at generation step {step}: {stats}"
                )
            sampled = torch.tensor([sample_ids], dtype=torch.long)
            log_probs = sampled_ids_to_log_probs(
                sampled_token_ids=sampled,
                vocab_size=sampler.vocab_size,
                observed_alpha=args.mc_observed_alpha,
                floor_mass=args.mc_floor_mass,
                dtype=dtype,
            ).to(device)
            final_log_probs = log_probs
            if biasnet is not None:
                residual = biasnet(log_probs)
                final_log_probs = log_probs + effective_scale * residual
            token_id = choose_token(final_log_probs, args.temperature)
            step_stats["mc_steps"] += 1
            valid_samples = len(sample_ids)
            sample_support = len(set(sample_ids))
        trace_steps.append(
            {
                "step": step + 1,
                "base_token_id": base_token_id,
                "final_token_id": token_id,
                "risk_score": gate_score,
                "raw_bias_scale": raw_scale,
                "effective_bias_scale": effective_scale,
                "forced_warmup": forced_warmup,
                "used_biasnet": bool(
                    biasnet is not None and effective_scale > 0.0
                ),
                "decision": decision,
                "base_source": "single_token_query",
            }
        )
        if decision == "below_min_scale_base":
            speculative_base_streak += 1
        else:
            speculative_base_streak = 0
        if not append_token(token_id):
            break
        if args.progress_steps:
            print(
                f"step={len(trace_steps)}/{args.max_new_tokens} valid={valid_samples} "
                f"support={sample_support}",
                flush=True,
            )

        if not (
            speculative_enabled
            and risk_gate is not None
            and speculative_base_streak
            >= args.risk_gate_speculative_min_base_streak
        ):
            continue

        remaining_tokens = args.max_new_tokens - len(trace_steps)
        if remaining_tokens <= 0:
            break
        draft_ids, draft_audit = sampler.draft_token_ids(
            prompt,
            generated,
            remaining_tokens,
            args.risk_gate_speculative_draft_tokens,
        )
        speculative_draft_calls += 1
        speculative_generated_tokens += len(draft_ids)
        draft_audit.update(
            {
                "draft_index": speculative_draft_calls,
                "trigger_after_step": len(trace_steps),
                "base_streak_at_trigger": speculative_base_streak,
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
            stop_reason = "speculative_empty"
            break

        answer_prefixes = [
            generated
            + tokenizer.decode(draft_ids[: offset + 1], skip_special_tokens=False)
            for offset in range(len(draft_ids))
        ]
        draft_scores = [
            float(value)
            for value in risk_gate.score_prefixes(
                [prompt] * len(answer_prefixes), answer_prefixes
            )
            .reshape(-1)
            .tolist()
        ]
        draft_scales = [
            risk_gate_scale_from_score(
                score,
                threshold=float(risk_gate.threshold),
                mode=args.risk_gate_mode,
                soft_temperature=args.risk_gate_soft_temperature,
            )
            for score in draft_scores
        ]
        speculative_verified_tokens += len(draft_ids)
        violation_offset = next(
            (
                offset
                for offset, scale in enumerate(draft_scales)
                if scale > args.risk_gate_min_scale
            ),
            None,
        )
        accepted_count = (
            len(draft_ids) if violation_offset is None else violation_offset
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

        terminal = False
        for offset in range(accepted_count):
            accepted_id = int(draft_ids[offset])
            trace_steps.append(
                {
                    "step": len(trace_steps) + 1,
                    "base_token_id": accepted_id,
                    "final_token_id": accepted_id,
                    "risk_score": draft_scores[offset],
                    "raw_bias_scale": draft_scales[offset],
                    "effective_bias_scale": 0.0,
                    "forced_warmup": False,
                    "used_biasnet": False,
                    "decision": "speculative_below_min_scale_base",
                    "base_source": "multi_token_draft",
                    "speculative_draft_index": speculative_draft_calls,
                }
            )
            step_stats["base_only_steps"] += 1
            speculative_base_streak += 1
            if not append_token(accepted_id):
                terminal = True
                break
        if terminal:
            break

        if violation_offset is None:
            finish_reason = str(draft_audit.get("finish_reason") or "").casefold()
            if (
                len(trace_steps) >= args.max_new_tokens
                or len(draft_ids) < int(draft_audit["requested_tokens"])
                or "stop" in finish_reason
            ):
                stop_reason = "speculative_complete"
                break
            continue

        candidate_id = int(draft_ids[violation_offset])
        candidate_scale = draft_scales[violation_offset]
        sample_ids, stats = sampler.sample_ids(prompt, generated)
        step_stats.update(stats)
        if len(sample_ids) < args.mc_samples_per_token:
            stop_reason = "mc_failure"
            if args.stop_on_mc_failure:
                break
            raise RuntimeError(
                f"Only {len(sample_ids)}/{args.mc_samples_per_token} usable Gemini "
                f"samples at speculative rollback step {len(trace_steps) + 1}: {stats}"
            )
        sampled = torch.tensor([sample_ids], dtype=torch.long)
        log_probs = sampled_ids_to_log_probs(
            sampled_token_ids=sampled,
            vocab_size=sampler.vocab_size,
            observed_alpha=args.mc_observed_alpha,
            floor_mass=args.mc_floor_mass,
            dtype=dtype,
        ).to(device)
        final_log_probs = log_probs
        if biasnet is not None:
            final_log_probs = log_probs + candidate_scale * biasnet(log_probs)
        controlled_id = choose_token(final_log_probs, args.temperature)
        step_stats["mc_steps"] += 1
        trace_steps.append(
            {
                "step": len(trace_steps) + 1,
                "base_token_id": candidate_id,
                "final_token_id": controlled_id,
                "risk_score": draft_scores[violation_offset],
                "raw_bias_scale": candidate_scale,
                "effective_bias_scale": candidate_scale,
                "forced_warmup": False,
                "used_biasnet": biasnet is not None,
                "decision": "speculative_rollback_soft_bias",
                "base_source": "multi_token_draft",
                "speculative_draft_index": speculative_draft_calls,
            }
        )
        speculative_base_streak = 0
        if not append_token(controlled_id):
            break
    runtime = None
    if args.risk_gate_trace:
        runtime = {
            "schema_version": 1,
            "configuration": {
                "requested_model": args.model,
                "tokenizer_name": args.tokenizer_name,
                "biasnet_checkpoint": args.biasnet_ckpt,
                "risk_gate_checkpoint": args.risk_gate_checkpoint,
                "mode": args.risk_gate_mode,
                "threshold": args.risk_gate_threshold,
                "soft_temperature": args.risk_gate_soft_temperature,
                "warmup_tokens": args.risk_gate_warmup_tokens,
                "min_scale": args.risk_gate_min_scale,
                "speculative_draft": speculative_enabled,
                "speculative_min_base_streak": (
                    args.risk_gate_speculative_min_base_streak
                ),
                "speculative_draft_tokens": args.risk_gate_speculative_draft_tokens,
                "mc_samples_per_token": args.mc_samples_per_token,
                "parallel_requests": args.parallel_requests,
                "candidate_count": args.candidate_count,
                "api_max_output_tokens": args.api_max_output_tokens,
                "request_timeout_ms": getattr(args, "request_timeout_ms", None),
                "scored_token_source": "deterministic_base_candidate",
            },
            "steps": trace_steps,
            "speculative_drafts": speculative_trace,
            "summary": {
                "controlled_token_steps": len(trace_steps),
                "mc_steps": int(step_stats["mc_steps"]),
                "base_only_steps": int(step_stats["base_only_steps"]),
                "speculative_draft_calls": speculative_draft_calls,
                "speculative_generated_tokens": speculative_generated_tokens,
                "speculative_verified_tokens": speculative_verified_tokens,
                "speculative_accepted_tokens": speculative_accepted_tokens,
                "speculative_rejected_tokens": speculative_rejected_tokens,
                "speculative_rollbacks": speculative_rollbacks,
                "stop_reason": stop_reason,
                "non_whitespace_chars": len("".join(generated.split())),
                "ends_in_whitespace": bool(generated and generated[-1].isspace()),
            },
        }
    return generated, dict(step_stats), runtime


def main() -> None:
    args = parse_args()
    if args.mc_samples_per_token <= 0:
        raise ValueError("--mc_samples_per_token must be positive.")
    if args.candidate_count <= 0 or args.candidate_count > 8:
        raise ValueError("--candidate_count must be in [1, 8].")
    if args.parallel_requests <= 0:
        raise ValueError("--parallel_requests must be positive.")
    if args.request_timeout_ms <= 0:
        raise ValueError("--request_timeout_ms must be positive.")
    if args.begin < 0:
        raise ValueError("--begin must be non-negative.")
    if args.end is not None and args.end <= args.begin:
        raise ValueError("--end must be greater than --begin.")
    if args.risk_gate_speculative_min_base_streak <= 0:
        raise ValueError("--risk_gate_speculative_min_base_streak must be positive.")
    if args.risk_gate_speculative_draft_tokens <= 0:
        raise ValueError("--risk_gate_speculative_draft_tokens must be positive.")
    if args.risk_gate_speculative_draft:
        if args.risk_gate_min_scale <= 0:
            raise ValueError(
                "--risk_gate_speculative_draft requires --risk_gate_min_scale > 0."
            )
        if not args.biasnet_ckpt or not args.risk_gate_checkpoint:
            raise ValueError(
                "Speculative generation requires both BiasNet and Risk Gate checkpoints."
            )
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tokenizer = load_tokenizer(
        args.tokenizer_name, use_fast=args.tokenizer_use_fast
    )
    device = resolve_device(args.device)
    biasnet = load_biasnet(args.biasnet_ckpt, device, args.biasnet_dtype)
    sampler = GeminiMCSampler(args, tokenizer)
    if (
        biasnet is not None
        and int(biasnet.config.vocab_size) != sampler.vocab_size
    ):
        raise ValueError(
            f"BiasNet vocab_size={biasnet.config.vocab_size}, "
            f"distribution size={sampler.vocab_size}"
        )
    risk_gate = load_risk_gate(args, device) if biasnet is not None else None
    prompt_records = collect_prompt_records(
        prompt_file=None if args.benchmark else args.prompt_file,
        benchmark=args.benchmark,
        benchmark_file=args.benchmark_file,
        benchmark_mutation=args.benchmark_mutation,
        limit=args.limit,
    )
    prompt_records = prompt_records[args.begin : args.end]
    prompts = [record.prompt for record in prompt_records]

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(json.loads(line)["prompt"])
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for index, record in enumerate(prompt_records):
            prompt = record.prompt
            if prompt in completed:
                continue
            calls_before = sampler.total_calls
            failed_calls_before = sampler.total_failed_calls
            started = time.perf_counter()
            completion, stats, runtime = generate_one(
                prompt, sampler, tokenizer, biasnet, risk_gate, device, args
            )
            record = {
                "prompt": prompt,
                "completion": completion,
                "mc_stats": stats,
                "elapsed_sec": time.perf_counter() - started,
                "api_calls": sampler.total_calls - calls_before,
                "api_failed_attempts": (
                    sampler.total_failed_calls - failed_calls_before
                ),
            }
            record.update(prompt_records[index].metadata)
            if runtime is not None:
                record["risk_gate_runtime"] = runtime
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                f"generated={index + 1}/{len(prompts)} chars={len(completion)} "
                f"elapsed_sec={record['elapsed_sec']:.1f}",
                flush=True,
            )
    print(
        f"Finished. calls={sampler.total_calls} valid={sampler.total_valid} "
        f"empty={sampler.total_empty} multi={sampler.total_multi} "
        f"failed_calls={sampler.total_failed_calls} output={output_path}"
    )


if __name__ == "__main__":
    main()
