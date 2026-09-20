"""White-box BiasNet generation against a locally loaded open-weight chat model.

This mirrors the soft-gated runtime in ``inference_openrouter.py`` step for step
-- warmup forcing, sigmoid residual scaling, minimum-scale fallback, greedy
decoding -- but replaces the black-box Monte Carlo estimator with the model's own
next-token distribution.  ``--logprob_source mc`` reproduces the Monte Carlo
estimator locally from the same logits, so an exact/MC pair of runs differs only
in the estimator and not in the serving stack.

Speculative drafting is intentionally absent: the gate is scored at every step,
which is the setting most favourable to BiasNet.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark_data import collect_prompt_records as load_prompt_records  # noqa: E402
from modeling_biasnet import BiasNet  # noqa: E402
from mc_reconstruction import fuse_proxy_logits_with_mc_counts  # noqa: E402
from foreign_proxy import ForeignProxyRuntime, load_or_build_first_token_map  # noqa: E402
from risk_gate import PrefixRiskGate  # noqa: E402

QWEN_HARD_NO_THINK_PREFILL = "<think>\n\n</think>\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--tokenizer_name", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--benchmark", default=None, choices=["advbench", "harmbench", "sorrybench"])
    parser.add_argument("--benchmark_file", default=None)
    parser.add_argument("--benchmark_mutation", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--torch_dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--device", default=None, help="Device for BiasNet and the risk gate.")
    parser.add_argument("--append_no_think", action="store_true")
    parser.add_argument("--qwen_hard_no_think_prefill", action="store_true")
    parser.add_argument("--biasnet_ckpt", default=None)
    parser.add_argument("--biasnet_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--logprob_source",
        choices=["exact", "mc", "fused"],
        default="exact",
        help=(
            "Distribution handed to BiasNet: the model's exact log-probs, a local MC "
            "estimate, or a Dirichlet fusion of dense proxy logits with MC counts."
        ),
    )
    parser.add_argument("--proxy_model_name_or_path", default=None)
    parser.add_argument("--proxy_torch_dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--proxy_device", default=None)
    parser.add_argument("--proxy_temperature", type=float, default=2.0)
    parser.add_argument("--proxy_prior_strength", type=float, default=8.0)
    parser.add_argument("--proxy_local_files_only", action="store_true")
    parser.add_argument("--proxy_tokenizer_name_or_path", default=None)
    parser.add_argument(
        "--proxy_projection",
        choices=["auto", "none", "first_token"],
        default="auto",
        help=(
            "How a proxy that does not share the target tokenizer is mapped into "
            "the target vocabulary. auto projects only when the vocabularies differ."
        ),
    )
    parser.add_argument("--proxy_first_token_map", default=None)
    parser.add_argument("--proxy_floor_logit", type=float, default=-30.0)
    parser.add_argument("--mc_samples_per_token", type=int, default=50)
    parser.add_argument("--mc_sample_temperature", type=float, default=1.0)
    parser.add_argument("--mc_top_p", type=float, default=1.0)
    parser.add_argument("--mc_observed_alpha", type=float, default=0.1)
    parser.add_argument("--mc_floor_mass", type=float, default=1e-4)
    parser.add_argument("--mc_seed", type=int, default=42)
    parser.add_argument("--risk_gate_checkpoint", default=None)
    parser.add_argument("--risk_gate_threshold", type=float, default=0.1)
    parser.add_argument("--risk_gate_mode", choices=["hard", "soft"], default="soft")
    parser.add_argument("--risk_gate_soft_temperature", type=float, default=0.05)
    parser.add_argument("--risk_gate_min_scale", type=float, default=0.01)
    parser.add_argument("--risk_gate_warmup_tokens", type=int, default=3)
    parser.add_argument("--risk_gate_batch_size", type=int, default=8)
    parser.add_argument("--risk_gate_max_length", type=int, default=None)
    parser.add_argument(
        "--risk_gate_dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto"
    )
    parser.add_argument("--risk_gate_device", default=None)
    parser.add_argument("--risk_gate_model_name", default=None)
    parser.add_argument("--risk_gate_load_in_4bit", action="store_true")
    parser.add_argument("--risk_gate_trust_remote_code", action="store_true")
    parser.add_argument("--risk_gate_local_files_only", action="store_true")
    parser.add_argument("--risk_gate_trace", action="store_true")
    parser.add_argument("--progress_steps", action="store_true")
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def risk_gate_scale_from_score(score: float, threshold: float, mode: str, soft_temperature: float) -> float:
    """Same residual-scale mapping as the OpenRouter runtime."""

    if not math.isfinite(score):
        raise ValueError(f"Risk gate returned a non-finite score: {score}.")
    if mode == "hard":
        return float(score < threshold)
    if soft_temperature <= 0:
        raise ValueError("--risk_gate_soft_temperature must be positive.")
    scaled = (float(threshold) - score) / soft_temperature
    if scaled >= 0:
        return 1.0 / (1.0 + math.exp(-scaled))
    exp_score = math.exp(scaled)
    return exp_score / (1.0 + exp_score)


def mc_log_probs_from_logits(
    logits: torch.Tensor,
    args: argparse.Namespace,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Reproduce the API Monte Carlo estimator from a single logit row."""

    scaled = logits.float() / args.mc_sample_temperature
    if args.mc_top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > args.mc_top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        sampled_sorted = torch.multinomial(
            torch.softmax(sorted_logits, dim=-1),
            num_samples=args.mc_samples_per_token,
            replacement=True,
            generator=generator,
        )
        sampled = sorted_indices.gather(dim=-1, index=sampled_sorted)
    else:
        sampled = torch.multinomial(
            torch.softmax(scaled, dim=-1),
            num_samples=args.mc_samples_per_token,
            replacement=True,
            generator=generator,
        )

    vocab_size = logits.shape[-1]
    unique_ids, counts = torch.unique(sampled.reshape(-1), sorted=False, return_counts=True)
    count_row = torch.zeros(vocab_size, dtype=torch.int64, device=logits.device)
    count_row[unique_ids] = counts.to(torch.int64)
    observed_count = int(unique_ids.numel())
    unseen_count = vocab_size - observed_count
    if unseen_count <= 0:
        unseen_log_prob = float("-inf")
        observed_mass = 1.0
    else:
        unseen_log_prob = math.log(args.mc_floor_mass / unseen_count)
        observed_mass = 1.0 - args.mc_floor_mass
    row = torch.full((vocab_size,), unseen_log_prob, dtype=torch.float32, device=logits.device)
    denom = float(args.mc_samples_per_token) + args.mc_observed_alpha * float(observed_count)
    observed_probs = observed_mass * (counts.float() + args.mc_observed_alpha) / denom
    row[unique_ids] = observed_probs.log()
    return row, count_row


def load_biasnet(path: Optional[str], device: torch.device, dtype_name: str) -> Optional[BiasNet]:
    if not path:
        return None
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    model = BiasNet.from_pretrained(path, map_location="cpu")
    model = model.to(device=device, dtype=dtype)
    model.set_up_proj()
    model.eval()
    return model


def load_risk_gate(args: argparse.Namespace, default_device: torch.device) -> Optional[PrefixRiskGate]:
    if args.risk_gate_checkpoint is None:
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


def build_prefix_text(tokenizer, prompt: str, hard_no_think: bool) -> str:
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if hard_no_think:
        prefix = prefix + QWEN_HARD_NO_THINK_PREFILL
    return prefix


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    args: argparse.Namespace,
    bias_model: Optional[BiasNet],
    risk_gate: Optional[PrefixRiskGate],
    vocab_size: int,
    mc_generator: Optional[torch.Generator],
    generation_audit: Optional[dict[str, Any]],
    proxy_model=None,
    foreign_proxy=None,
) -> tuple[str, dict[str, Any]]:
    device = next(model.parameters()).device
    bias_device = next(bias_model.parameters()).device if bias_model is not None else device
    prefix_text = build_prefix_text(tokenizer, prompt, args.qwen_hard_no_think_prefill)
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    proxy_device = next(proxy_model.parameters()).device if proxy_model is not None else None
    proxy_input_ids = (
        torch.tensor([prefix_ids], dtype=torch.long, device=proxy_device)
        if proxy_model is not None and foreign_proxy is None
        else None
    )
    # A foreign proxy cannot consume the target's ids, so it is re-rendered from
    # text each step against the same assistant prefix the target is producing.
    foreign_lead = QWEN_HARD_NO_THINK_PREFILL if args.qwen_hard_no_think_prefill else ""
    proxy_past = None

    eos_token_id = tokenizer.eos_token_id
    generated_ids: list[int] = []
    past_key_values = None
    step_trace: list[dict[str, Any]] = []
    mc_steps = 0
    base_only_steps = 0
    warmup_steps = 0
    scored_steps = 0
    below_min_scale_steps = 0
    controlled_token_steps = 0

    for step_index in range(args.max_new_tokens):
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1, :vocab_size].float()

        proxy_logits = None
        if foreign_proxy is not None:
            answer_prefix = foreign_lead + tokenizer.decode(
                generated_ids, skip_special_tokens=False
            )
            proxy_logits = foreign_proxy.next_token_log_probs(prompt, answer_prefix)
        elif proxy_model is not None:
            proxy_out = proxy_model(
                input_ids=proxy_input_ids, past_key_values=proxy_past, use_cache=True
            )
            proxy_past = proxy_out.past_key_values
            # Keep the padded tail; fuse_proxy_logits_with_mc_counts crops it.
            proxy_logits = proxy_out.logits[0, -1, :].float()
        base_log_probs = torch.log_softmax(logits, dim=-1)
        base_token_id = int(torch.argmax(base_log_probs).item())
        final_token_id = base_token_id

        gate_score: Optional[float] = None
        raw_bias_scale = 1.0
        effective_bias_scale = 1.0
        decision = "bias"
        gate_forced_active = (
            risk_gate is not None and step_index < args.risk_gate_warmup_tokens
        )

        if bias_model is None:
            raw_bias_scale = 0.0
            effective_bias_scale = 0.0
            decision = "base_no_biasnet"
        elif risk_gate is not None:
            if gate_forced_active:
                warmup_steps += 1
                decision = "warmup_bias"
            else:
                base_answer_prefix = tokenizer.decode(
                    generated_ids + [base_token_id], skip_special_tokens=False
                )
                gate_score = float(
                    risk_gate.score_prefixes([prompt], [base_answer_prefix])[0].item()
                )
                scored_steps += 1
                raw_bias_scale = risk_gate_scale_from_score(
                    gate_score,
                    threshold=args.risk_gate_threshold,
                    mode=args.risk_gate_mode,
                    soft_temperature=args.risk_gate_soft_temperature,
                )
                if raw_bias_scale <= args.risk_gate_min_scale:
                    effective_bias_scale = 0.0
                    below_min_scale_steps += 1
                    decision = "below_min_scale_base"
                else:
                    effective_bias_scale = raw_bias_scale
                    decision = "soft_bias" if raw_bias_scale < 1.0 else "bias"

        used_biasnet = False
        if bias_model is not None and effective_bias_scale > 0.0:
            if args.logprob_source == "mc":
                bias_input, _ = mc_log_probs_from_logits(logits, args, mc_generator)
            elif args.logprob_source == "fused":
                _, mc_counts = mc_log_probs_from_logits(logits, args, mc_generator)
                bias_input = fuse_proxy_logits_with_mc_counts(
                    proxy_logits.to(logits.device).unsqueeze(0),
                    mc_counts.unsqueeze(0),
                    temperature=args.proxy_temperature,
                    prior_strength=args.proxy_prior_strength,
                )[0]
            else:
                bias_input = base_log_probs
            bias_input = bias_input.to(device=bias_device).unsqueeze(0)
            bias_dtype = next(bias_model.parameters()).dtype
            model_input = bias_input.to(dtype=bias_dtype)
            if int(getattr(bias_model, "num_position_buckets", 0) or 0) > 0:
                residual = bias_model(model_input, position_ids=step_index)
            else:
                residual = bias_model(model_input)
            biased_scores = model_input + effective_bias_scale * residual
            if args.temperature <= 0:
                final_token_id = int(torch.argmax(biased_scores[0]).item())
            else:
                probs = torch.softmax(biased_scores[0].float() / args.temperature, dim=-1)
                final_token_id = int(torch.multinomial(probs, num_samples=1).item())
            used_biasnet = True
            mc_steps += 1
        else:
            base_only_steps += 1

        if final_token_id != base_token_id:
            controlled_token_steps += 1

        if generation_audit is not None:
            step_trace.append(
                {
                    "step": step_index,
                    "base_token_id": base_token_id,
                    "base_token": tokenizer.decode([base_token_id]),
                    "final_token_id": final_token_id,
                    "final_token": tokenizer.decode([final_token_id]),
                    "risk_score": gate_score,
                    "raw_bias_scale": float(raw_bias_scale),
                    "effective_bias_scale": float(effective_bias_scale),
                    "forced_warmup": bool(gate_forced_active),
                    "used_biasnet": bool(used_biasnet),
                    "decision": decision,
                }
            )

        if eos_token_id is not None and final_token_id == eos_token_id:
            break
        generated_ids.append(final_token_id)
        input_ids = torch.tensor([[final_token_id]], dtype=torch.long, device=device)
        if proxy_model is not None and foreign_proxy is None:
            proxy_input_ids = torch.tensor(
                [[final_token_id]], dtype=torch.long, device=proxy_device
            )

    completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    summary = {
        "steps": len(generated_ids),
        "mc_steps": mc_steps,
        "base_only_steps": base_only_steps,
        "warmup_steps": warmup_steps,
        "scored_steps": scored_steps,
        "below_min_scale_steps": below_min_scale_steps,
        "controlled_token_steps": controlled_token_steps,
    }
    if generation_audit is not None:
        generation_audit["steps"] = step_trace
    return completion, summary


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_prompt_records(
        prompt=None,
        prompts=None,
        prompt_file=args.prompt_file,
        benchmark=args.benchmark,
        benchmark_file=args.benchmark_file,
        benchmark_mutation=args.benchmark_mutation,
        limit=args.limit,
    )

    completed = 0
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            completed = sum(1 for line in handle if line.strip())
        records = records[completed:]
    elif output_path.exists():
        output_path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    default_device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # BiasNet is built before the target model on purpose. set_up_proj() takes the
    # pseudo-inverse of a [hidden_size, vocab_size] head, which needs several GB of
    # transient SVD workspace; doing it first lets that memory be released before a
    # large target model is resident, instead of having to fit both peaks at once.
    bias_model = load_biasnet(args.biasnet_ckpt, default_device, args.biasnet_dtype)
    if bias_model is not None and bias_model.vocab_size != vocab_size:
        raise ValueError(
            f"BiasNet vocabulary {bias_model.vocab_size} does not match tokenizer {vocab_size}."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=resolve_torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    risk_gate = load_risk_gate(args, default_device)

    proxy_model = None
    foreign_proxy = None
    proxy_projection_mode = "none"
    if args.logprob_source == "fused":
        if not args.proxy_model_name_or_path:
            raise ValueError("--logprob_source fused requires --proxy_model_name_or_path.")
        proxy_device = torch.device(args.proxy_device) if args.proxy_device else default_device
        proxy_model = AutoModelForCausalLM.from_pretrained(
            args.proxy_model_name_or_path,
            dtype=resolve_torch_dtype(args.proxy_torch_dtype),
            local_files_only=args.proxy_local_files_only,
        ).to(proxy_device)
        proxy_model.eval()
        proxy_vocab = proxy_model.get_output_embeddings().weight.shape[0]
        proxy_tokenizer = AutoTokenizer.from_pretrained(
            args.proxy_tokenizer_name_or_path or args.proxy_model_name_or_path,
            local_files_only=args.proxy_local_files_only,
        )
        # A proxy that shares the target's token->id map can be fed the target's
        # ids directly.  Anything else is projected into the target vocabulary,
        # which is what makes a cross-family proxy usable at all.
        shares_vocab = proxy_tokenizer.get_vocab() == tokenizer.get_vocab()
        if args.proxy_projection == "first_token" or (
            args.proxy_projection == "auto" and not shares_vocab
        ):
            first_token_ids = load_or_build_first_token_map(
                args.proxy_first_token_map,
                tokenizer,
                proxy_tokenizer,
                vocab_size,
            )
            foreign_proxy = ForeignProxyRuntime(
                proxy_model,
                proxy_tokenizer,
                first_token_ids,
                proxy_device,
                floor_logit=args.proxy_floor_logit,
            )
            proxy_projection_mode = "first_token"
        elif proxy_vocab < vocab_size:
            raise ValueError(
                f"Proxy output vocabulary {proxy_vocab} is smaller than the shared "
                f"vocabulary {vocab_size}; pass --proxy_projection first_token to "
                f"project a foreign proxy into the target vocabulary instead."
            )

    mc_generator = None
    if args.logprob_source in ("mc", "fused"):
        mc_generator = torch.Generator(device=next(model.parameters()).device)
        mc_generator.manual_seed(args.mc_seed)

    configuration = {
        "model_name_or_path": args.model_name_or_path,
        "biasnet_checkpoint": args.biasnet_ckpt,
        "logprob_source": args.logprob_source,
        "mc_samples_per_token": (
            args.mc_samples_per_token if args.logprob_source in ("mc", "fused") else None
        ),
        "proxy_model_name_or_path": args.proxy_model_name_or_path,
        "proxy_projection_mode": proxy_projection_mode,
        "proxy_temperature": args.proxy_temperature if args.logprob_source == "fused" else None,
        "proxy_prior_strength": (
            args.proxy_prior_strength if args.logprob_source == "fused" else None
        ),
        "risk_gate_mode": args.risk_gate_mode,
        "risk_gate_threshold": args.risk_gate_threshold,
        "risk_gate_soft_temperature": args.risk_gate_soft_temperature,
        "risk_gate_min_scale": args.risk_gate_min_scale,
        "warmup_tokens": args.risk_gate_warmup_tokens,
        "qwen_hard_no_think_prefill": bool(args.qwen_hard_no_think_prefill),
        "append_no_think": bool(args.append_no_think),
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "speculative_draft": False,
    }

    with output_path.open("a", encoding="utf-8") as handle:
        for offset, record in enumerate(records, start=completed + 1):
            model_prompt = record.prompt + "\n/no_think" if args.append_no_think else record.prompt
            if args.progress_steps:
                print(f"starting_prompt={offset}/{completed + len(records)}", flush=True)
            started = time.perf_counter()
            audit: Optional[dict[str, Any]] = {} if args.risk_gate_trace else None
            completion, summary = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=model_prompt,
                args=args,
                bias_model=bias_model,
                risk_gate=risk_gate,
                vocab_size=vocab_size,
                mc_generator=mc_generator,
                generation_audit=audit,
                proxy_model=proxy_model,
                foreign_proxy=foreign_proxy,
            )
            row: dict[str, Any] = {
                "prompt": record.prompt,
                "completion": completion,
                "risk_gate_runtime": {"configuration": configuration, "summary": summary},
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            if record.metadata:
                row["metadata"] = dict(record.metadata)
            if audit is not None:
                row["generation_audit"] = audit
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if args.progress_steps:
                print(
                    f"finished_prompt={offset} steps={summary['steps']} "
                    f"biasnet_steps={summary['mc_steps']} "
                    f"controlled={summary['controlled_token_steps']}",
                    flush=True,
                )

    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
