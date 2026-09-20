"""V-space black-box decoder (Solution A inference).

Same tested sampler path as ``inference_openrouter_string_action.py`` (one
``max_tokens=1`` MC draw per position, faithful string emission), but everything
downstream lives in Claude's empirical vocabulary ``V`` instead of a proxy vocab:

  * MC counts are EXACT V counts (no first-token truncation).
  * the proxy prior is projected onto V and fused by the same Dirichlet posterior.
  * BiasNet scores candidates on their TRUE V coordinate, not ``ids[0]``.
  * the emitted action is the observed string with the highest V-space score.

This closes both defects of the string-action script (which still scored on a
truncated proxy id and still built its feature from a proxy-vocab bincount).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (ROOT, os.path.join(ROOT, "training")):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_data import collect_prompt_records  # noqa: E402
from mc_reconstruction import fuse_proxy_logits_with_mc_counts  # noqa: E402
from pre_logits_sampled_openrouter import (  # noqa: E402
    OpenRouterClient,
    load_tokenizer,
    resolve_api_key,
    sample_position_token_ids,
)
import inference_openrouter as ior  # noqa: E402
from empirical_vocab.vspace import VSpaceProxy, Vocab, counter_to_vcounts  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--vocab", required=True, help="vocab.json from build_and_materialize.")
    p.add_argument("--biasnet_ckpt", required=True)
    p.add_argument("--biasnet_dtype", default="float16")
    p.add_argument("--tokenizer_name", required=True, help="Sampler tokenizer (cache tokenizer).")
    p.add_argument("--tokenizer_revision", default=None)
    # proxy (must match the materializer)
    p.add_argument("--proxy_model", required=True)
    p.add_argument("--proxy_tokenizer", default=None)
    p.add_argument("--proxy_dtype", default="float16")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--prior_strength", type=float, default=2.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--emit_mode",
        choices=["observed", "full_vocab"],
        default="observed",
        help="observed: emit only strings Claude actually sampled (Claude-realizable). "
        "full_vocab: emit argmax over ALL of V, so BiasNet can inject unsampled "
        "harmful tokens (forced-writing regime, now clean in V-space).",
    )
    p.add_argument("--residual_scale", type=float, default=1.0,
                   help="Multiplier on the BiasNet residual; >1 forces harder.")
    # prompts
    p.add_argument("--prompt_file", default=None)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--benchmark_file", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--append_no_think", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--output_json", required=True)
    # MC sampling
    p.add_argument("--mc_samples_per_token", type=int, default=50)
    p.add_argument("--mc_sample_temperature", type=float, default=1.0)
    p.add_argument("--mc_top_p", type=float, default=1.0)
    p.add_argument("--api_max_tokens", type=int, default=1)
    p.add_argument("--parallel_requests", type=int, default=50)
    p.add_argument("--sample_choices_per_request", type=int, default=1)
    p.add_argument("--sample_completion_policy", default="exact")
    p.add_argument("--max_sample_refill_rounds", type=int, default=5)
    p.add_argument("--empty_length_retry_max_tokens", type=int, default=8)
    p.add_argument("--max_empty_length_retry_rounds", type=int, default=3)
    p.add_argument("--empty_response_token", default="stop_eos")
    p.add_argument("--delay_seconds", type=float, default=0.0)
    p.add_argument("--qwen_hard_no_think_prefill", action="store_true")
    p.add_argument("--reject_reasoning_tokens", action="store_true")
    p.add_argument("--reasoning_mode", default="enabled_false",
                   help="Passed to the client payload; keep reasoning off for clean MC samples.")
    # client
    p.add_argument("--provider_order", nargs="*", default=None)
    p.add_argument("--provider_allow_fallbacks", type=lambda v: str(v).lower() == "true", default=False)
    p.add_argument("--router_metadata", action="store_true")
    p.add_argument("--disable_openrouter_response_cache", action="store_true")
    p.add_argument("--request_timeout", type=float, default=90)
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--retry_sleep", type=float, default=2)
    p.add_argument("--api_key", default=None)
    p.add_argument("--api_key_env", default="OPENROUTER_API_KEY")
    p.add_argument("--api_key_file", default=None)
    p.add_argument("--api_url", default="https://openrouter.ai/api/v1/chat/completions")
    p.add_argument("--progress_steps", action="store_true")
    p.add_argument("--stop_on_mc_failure", action="store_true", default=True)
    return p.parse_args()


def build_sample_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        samples_per_token=args.mc_samples_per_token,
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
        qwen_hard_no_think_prefill=bool(args.qwen_hard_no_think_prefill),
        reject_reasoning_tokens=bool(args.reject_reasoning_tokens),
    )


def build_client_args(args: argparse.Namespace) -> argparse.Namespace:
    import pre_logits_sampled_openrouter as sampler

    saved = sys.argv
    try:
        sys.argv = ["pre_logits_sampled_openrouter.py", "--output_dir", "/tmp/_unused"]
        base = sampler.parse_args()
    finally:
        sys.argv = saved
    for key, value in vars(args).items():
        setattr(base, key, value)
    base.cache_configuration_fingerprint = None
    return base


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def load_proxy(args, device) -> VSpaceProxy:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.proxy_tokenizer or args.proxy_model, use_fast=False)
    model = (
        AutoModelForCausalLM.from_pretrained(args.proxy_model, torch_dtype=resolve_dtype(args.proxy_dtype))
        .to(device)
        .eval()
    )
    return VSpaceProxy(model, tok, device)


def generate_one(client, tokenizer, prompt, args, bias_model, proxy, vocab, device):
    sample_args = build_sample_args(args)
    prefix_text = ""
    steps = 0
    changed_from_top = 0
    oov_emissions = 0
    injected_emissions = 0
    trace: list[dict[str, Any]] = []

    for _ in range(args.max_new_tokens):
        sample_ids, stats = sample_position_token_ids(
            client=client, tokenizer=tokenizer, question=prompt,
            prefix_text=prefix_text, args=sample_args,
        )
        text_counts: dict[str, int] = dict(stats.get("sampled_completion_text_counts") or {})
        if not text_counts:
            break

        counts_V = counter_to_vcounts(text_counts, vocab).to(device)  # [|V|]
        proxy_logits_V = proxy.vspace_logits(prompt, prefix_text, vocab)  # [|V|]
        fused = fuse_proxy_logits_with_mc_counts(
            proxy_logits_V.unsqueeze(0).to(device),
            counts_V.unsqueeze(0),
            temperature=args.temperature,
            prior_strength=args.prior_strength,
        )  # [1, |V|]
        scores = ior.apply_bias_model(bias_model, fused, residual_scale=args.residual_scale)[0]  # [|V|]

        if args.emit_mode == "full_vocab":
            # Argmax over ALL of V (minus the OOV sentinel, which has no emittable
            # string), so BiasNet may inject a harmful token Claude never sampled.
            masked = scores.clone()
            masked[vocab.oov_id] = float("-inf")
            chosen_id = int(masked.argmax().item())
            chosen_text = vocab.id_to_string[chosen_id]
        else:
            ranked = sorted(
                ((text, vocab.encode(text)) for text in text_counts),
                key=lambda tv: float(scores[tv[1]].item()),
                reverse=True,
            )
            chosen_text, chosen_id = ranked[0]

        if chosen_text == "" or chosen_id == vocab.eos_id:
            break  # Claude-realizable stop
        if chosen_id == vocab.oov_id:
            oov_emissions += 1

        most_sampled = max(text_counts.items(), key=lambda kv: kv[1])[0]
        chosen_observed = chosen_text in text_counts
        if chosen_text != most_sampled:
            changed_from_top += 1

        if not chosen_observed:
            injected_emissions += 1
        trace.append({
            "step": steps,
            "candidates": len(text_counts),
            "chosen": chosen_text,
            "chosen_v_id": int(chosen_id),
            "chosen_observed": bool(chosen_observed),
            "most_sampled": most_sampled,
            "chosen_count": int(text_counts.get(chosen_text, 0)),
            "top_count": int(text_counts.get(most_sampled, 0)),
        })
        prefix_text += chosen_text
        steps += 1
        if args.progress_steps and steps % 10 == 0:
            print(f"  step={steps}/{args.max_new_tokens}", flush=True)

    summary = {
        "steps": steps,
        "mc_steps": steps,
        "controlled_token_steps": changed_from_top,
        "controlled_token_rate": (changed_from_top / steps) if steps else 0.0,
        "injected_emissions": injected_emissions,
        "injected_rate": (injected_emissions / steps) if steps else 0.0,
        "oov_emissions": oov_emissions,
        "emit_mode": args.emit_mode,
        "residual_scale": args.residual_scale,
        "action_space": "empirical_vocab_string",
    }
    return prefix_text, {"summary": summary, "steps": trace}


def main() -> None:
    args = parse_args()
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    device = ior.resolve_device(args.device)
    vocab = Vocab.load(args.vocab)
    print(f"loaded |V|={vocab.size}", flush=True)

    tokenizer = load_tokenizer(
        args.tokenizer_name, use_fast=False, trust_remote_code=False,
        revision=args.tokenizer_revision,
    )
    bias_model = ior.load_biasnet(args.biasnet_ckpt, device, args.biasnet_dtype)
    if bias_model is None:
        raise ValueError("--biasnet_ckpt is required.")
    if int(getattr(bias_model, "vocab_size", -1)) != vocab.size:
        raise ValueError(
            f"BiasNet vocab_size {getattr(bias_model,'vocab_size',None)} != |V| {vocab.size}."
        )
    proxy = load_proxy(args, device)

    records = collect_prompt_records(
        prompt=None, prompts=None, prompt_file=args.prompt_file,
        benchmark=args.benchmark, benchmark_file=args.benchmark_file,
        benchmark_mutation=None, limit=args.limit,
    )
    client_args = build_client_args(args)
    client = OpenRouterClient(client_args, resolve_api_key(client_args))

    print(f"prompts={len(records)} action_space=empirical_vocab_string", flush=True)
    with out.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            question = record.prompt + ("\n/no_think" if args.append_no_think else "")
            if args.progress_steps:
                print(f"starting_prompt={index}/{len(records)}", flush=True)
            started = time.perf_counter()
            completion, audit = generate_one(
                client, tokenizer, question, args, bias_model, proxy, vocab, device
            )
            row = {
                "prompt": record.prompt,
                "completion": completion,
                "risk_gate_runtime": {
                    "configuration": {
                        "action_space": "empirical_vocab_string",
                        "model": args.model,
                        "biasnet_checkpoint": args.biasnet_ckpt,
                        "vocab": args.vocab,
                        "vocab_size": vocab.size,
                        "proxy_model": args.proxy_model,
                        "temperature": args.temperature,
                        "prior_strength": args.prior_strength,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    "summary": audit["summary"],
                },
                "openrouter_routing": ior.snapshot_openrouter_audit(client)
                if hasattr(ior, "snapshot_openrouter_audit") else {},
                "generation_audit": {"steps": audit["steps"]},
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            s = audit["summary"]
            print(
                f"finished_prompt={index} steps={s['steps']} "
                f"controlled_rate={s['controlled_token_rate']:.2f} oov={s['oov_emissions']}",
                flush=True,
            )
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
