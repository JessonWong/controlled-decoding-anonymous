"""Standalone black-box generation whose action is a SAMPLED STRING, not a token id.

Why this exists
---------------
`inference_openrouter.py` decodes by choosing a proxy-vocabulary token id and
appending that token's string. When the proxy tokenizer is not the target's own,
a continuation the target returned may need several proxy tokens; the pipeline
records only the first, so the appended string is a PREFIX the target never
emitted alone. Measured on Claude Haiku 4.5 with a Qwen3 proxy vocabulary: 12.2%
of samples are recorded truncated, producing text like
`I canSure,t help create mi sin formation`.

Restricting the choice to "faithful" single-token continuations does not fix it:
4.7% of positions then have NO legal action at all and 30.6% have at most one,
which leaves BiasNet no decision to make.

This script keeps BiasNet's scoring in token space -- that is what it was trained
on -- but makes the emitted action the full continuation text the target actually
returned. Scoring may use a truncated id; emission never does.

Nothing here modifies the existing pipeline; it imports from it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch

ROOT = os.path.abspath(os.path.dirname(__file__))
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer_name", required=True)
    p.add_argument("--tokenizer_revision", default=None)
    p.add_argument("--prompt_file", default=None)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--benchmark_file", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output_json", required=True)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--biasnet_ckpt", required=True)
    p.add_argument("--biasnet_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--device", default=None)
    # proxy fusion
    p.add_argument("--proxy_model_name_or_path", default=None)
    p.add_argument("--proxy_model_revision", default=None)
    p.add_argument("--proxy_temperature", type=float, default=2.0)
    p.add_argument("--proxy_prior_strength", type=float, default=2.0)
    p.add_argument("--proxy_dtype", default="float16")
    p.add_argument("--proxy_quantization", default="none")
    p.add_argument("--proxy_device", default=None)
    p.add_argument("--proxy_trust_remote_code", action="store_true")
    p.add_argument("--proxy_local_files_only", action="store_true")
    # MC sampling (mirrors the main runtime)
    p.add_argument("--mc_samples_per_token", type=int, default=50)
    p.add_argument("--mc_sample_temperature", type=float, default=1.0)
    p.add_argument("--mc_top_p", type=float, default=1.0)
    p.add_argument("--mc_observed_alpha", type=float, default=0.1)
    p.add_argument("--mc_floor_mass", type=float, default=1e-4)
    p.add_argument("--api_max_tokens", type=int, default=1)
    p.add_argument("--parallel_requests", type=int, default=8)
    p.add_argument("--sample_choices_per_request", type=int, default=1)
    p.add_argument("--sample_completion_policy", default="exact")
    p.add_argument("--max_sample_refill_rounds", type=int, default=5)
    p.add_argument("--empty_length_retry_max_tokens", type=int, default=8)
    p.add_argument("--max_empty_length_retry_rounds", type=int, default=4)
    p.add_argument("--empty_response_token", default="stop_eos")
    p.add_argument("--delay_seconds", type=float, default=0.0)
    p.add_argument("--reasoning_mode", default="enabled_false")
    p.add_argument("--reject_reasoning_tokens", action="store_true")
    p.add_argument("--qwen_hard_no_think_prefill", action="store_true")
    p.add_argument("--append_no_think", action="store_true")
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
    p.add_argument("--stop_on_mc_failure", action="store_true")
    return p.parse_args()


def build_client_args(args: argparse.Namespace) -> argparse.Namespace:
    """Start from the sampler's own defaults, then overlay ours.

    OpenRouterClient reads fields this script never declares (site_url and
    friends). Hand-listing them is how the first attempt failed after 300 dead
    requests, so take the full default namespace from the parser that owns the
    client and override only what we set.
    """

    import pre_logits_sampled_openrouter as sampler

    saved = sys.argv
    try:
        sys.argv = ["pre_logits_sampled_openrouter.py", "--output_dir", "/tmp/_unused"]
        base = sampler.parse_args()
    finally:
        sys.argv = saved
    for key, value in vars(args).items():
        setattr(base, key, value)
    # main() sets this in the sampler after writing a cache manifest; there is no
    # manifest when generating, and the client only forwards it as request metadata.
    base.cache_configuration_fingerprint = None
    return base


def preflight(client, tokenizer, prompt: str, args: argparse.Namespace) -> None:
    """Make one sampled position before the real loop.

    A malformed client namespace otherwise surfaces only after a full round of
    failed requests, once per position.
    """

    ids, stats = sample_position_token_ids(
        client=client, tokenizer=tokenizer, question=prompt,
        prefix_text="", args=build_sample_args(args),
    )
    if not ids:
        raise RuntimeError(f"Preflight sampling returned nothing: {stats}")
    print(f"preflight_ok samples={len(ids)} distinct_texts="
          f"{len(stats.get('sampled_completion_text_counts') or {})}", flush=True)


def build_sample_args(args: argparse.Namespace) -> argparse.Namespace:
    """The subset sample_position_token_ids reads, named as it expects."""

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


def score_candidate_strings(
    scores: torch.Tensor, text_counts: dict[str, int], tokenizer
) -> list[tuple[str, int, float]]:
    """Rank the sampled continuation strings by BiasNet's token-space score.

    A string is scored through its FIRST proxy token, which is the coordinate
    BiasNet was trained to move. The string itself is what gets emitted, so a
    multi-token continuation is ranked by a truncated id but never appended
    truncated.
    """

    ranked: list[tuple[str, int, float]] = []
    for text in text_counts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        ranked.append((text, ids[0], float(scores[ids[0]].item())))
    ranked.sort(key=lambda row: row[2], reverse=True)
    return ranked


def generate_one_string_action(
    client, tokenizer, prompt: str, args: argparse.Namespace,
    bias_model, proxy, device: torch.device,
) -> tuple[str, dict[str, Any]]:
    sample_args = build_sample_args(args)
    prefix_text = ""
    steps = 0
    multi_token_emissions = 0
    changed_from_top_sample = 0
    trace: list[dict[str, Any]] = []

    for _ in range(args.max_new_tokens):
        sample_ids, stats = sample_position_token_ids(
            client=client, tokenizer=tokenizer, question=prompt,
            prefix_text=prefix_text, args=sample_args,
        )
        if not sample_ids:
            if args.stop_on_mc_failure:
                break
            raise RuntimeError(f"No valid samples at prefix length {len(prefix_text)}")
        text_counts: dict[str, int] = dict(stats.get("sampled_completion_text_counts") or {})
        if not text_counts:
            break

        counts = torch.bincount(
            torch.tensor(sample_ids, dtype=torch.long), minlength=len(tokenizer)
        ).to(device)
        if proxy is not None:
            proxy_logits = proxy.next_token_logits(
                prompt, prefix_text,
                qwen_hard_no_think_prefill=bool(args.qwen_hard_no_think_prefill),
            )
            feature = fuse_proxy_logits_with_mc_counts(
                proxy_logits.unsqueeze(0).to(device),
                counts.unsqueeze(0),
                temperature=args.proxy_temperature,
                prior_strength=args.proxy_prior_strength,
            )
        else:
            total = counts.sum().clamp_min(1)
            probs = (counts.float() + args.mc_observed_alpha) / (
                total + args.mc_observed_alpha * len(tokenizer)
            )
            feature = probs.log().unsqueeze(0)

        scores = ior.apply_bias_model(bias_model, feature, residual_scale=1.0)[0]
        ranked = score_candidate_strings(scores, text_counts, tokenizer)
        if not ranked:
            break
        chosen_text, chosen_id, _ = ranked[0]

        most_sampled = max(text_counts.items(), key=lambda kv: kv[1])[0]
        if chosen_text != most_sampled:
            changed_from_top_sample += 1
        if len(tokenizer.encode(chosen_text, add_special_tokens=False)) > 1:
            multi_token_emissions += 1

        trace.append({
            "step": steps,
            "candidates": len(ranked),
            "chosen": chosen_text,
            "most_sampled": most_sampled,
            "chosen_is_multi_token": len(
                tokenizer.encode(chosen_text, add_special_tokens=False)
            ) > 1,
        })
        # The emitted action is the string itself, never a detokenized prefix of it.
        prefix_text += chosen_text
        steps += 1
        if args.progress_steps and steps % 10 == 0:
            print(f"  token_step={steps}/{args.max_new_tokens}", flush=True)

    summary = {
        "steps": steps,
        "mc_steps": steps,
        "base_only_steps": 0,
        "controlled_token_steps": changed_from_top_sample,
        "multi_token_emissions": multi_token_emissions,
        "action_space": "sampled_completion_string",
    }
    return prefix_text, {"summary": summary, "steps": trace}


def main() -> None:
    args = parse_args()
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    records = collect_prompt_records(
        prompt=None, prompts=None, prompt_file=args.prompt_file,
        benchmark=args.benchmark, benchmark_file=args.benchmark_file,
        benchmark_mutation=None, limit=args.limit,
    )
    device = ior.resolve_device(args.device)
    # Same loader the sampler uses, so token ids match the cache exactly.
    tokenizer = load_tokenizer(
        args.tokenizer_name,
        use_fast=False,
        trust_remote_code=False,
        revision=args.tokenizer_revision,
    )

    bias_model = ior.load_biasnet(args.biasnet_ckpt, device, args.biasnet_dtype)
    if bias_model is None:
        raise ValueError("--biasnet_ckpt is required.")
    proxy = ior.load_local_proxy(args, tokenizer, device)
    client_args = build_client_args(args)
    client = OpenRouterClient(client_args, resolve_api_key(client_args))

    print(f"prompts={len(records)} action_space=sampled_completion_string", flush=True)
    preflight(
        client, tokenizer,
        records[0].prompt + ("\n/no_think" if args.append_no_think else ""),
        args,
    )
    with out.open("a", encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            question = record.prompt + ("\n/no_think" if args.append_no_think else "")
            if args.progress_steps:
                print(f"starting_prompt={index}/{len(records)}", flush=True)
            started = time.perf_counter()
            completion, audit = generate_one_string_action(
                client, tokenizer, question, args, bias_model, proxy, device
            )
            row = {
                "prompt": record.prompt,
                "completion": completion,
                "risk_gate_runtime": {
                    "configuration": {
                        "action_space": "sampled_completion_string",
                        "model": args.model,
                        "biasnet_checkpoint": args.biasnet_ckpt,
                        "proxy_model_name_or_path": args.proxy_model_name_or_path,
                        "proxy_temperature": args.proxy_temperature,
                        "proxy_prior_strength": args.proxy_prior_strength,
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
                f"multi_token_emissions={s['multi_token_emissions']} "
                f"changed_from_top={s['controlled_token_steps']}",
                flush=True,
            )
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
