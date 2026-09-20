"""Online BiasNet decoding in Qwen coordinates plus empirical string actions.

The target API is sampled with exact MC at every step. Continuations that are
one Qwen token retain their native id; multi-token continuations use the fixed
extension sidecar built from the training cache. Extension actions append their
literal text and the next API call re-tokenizes the complete prefix with the
unchanged Qwen tokenizer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (ROOT, os.path.join(ROOT, "training")):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_data import collect_prompt_records  # noqa: E402
from empirical_vocab.hybrid_qwen import (  # noqa: E402
    HybridQwenVocab,
    counter_to_hybrid_counts,
)
import inference_openrouter as ior  # noqa: E402
from pre_logits_sampled_openrouter import (  # noqa: E402
    OpenRouterClient,
    load_tokenizer,
    resolve_api_key,
    sample_position_token_ids,
)
from pre_logits_sampled_openweight import (  # noqa: E402
    sampled_ids_to_log_probs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--biasnet_ckpt", required=True)
    parser.add_argument("--biasnet_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--tokenizer_name", required=True)
    parser.add_argument("--tokenizer_revision", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--emit_mode", choices=["observed", "full_vocab"], default="full_vocab")
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--benchmark_file", default=None)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append_no_think", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--mc_samples_per_token", type=int, default=50)
    parser.add_argument("--mc_sample_temperature", type=float, default=1.0)
    parser.add_argument("--mc_top_p", type=float, default=1.0)
    parser.add_argument("--mc_observed_alpha", type=float, default=0.1)
    parser.add_argument("--mc_floor_mass", type=float, default=1e-4)
    parser.add_argument("--api_max_tokens", type=int, default=1)
    parser.add_argument("--parallel_requests", type=int, default=8)
    parser.add_argument("--sample_choices_per_request", type=int, default=1)
    parser.add_argument("--sample_completion_policy", choices=["partial", "exact"], default="exact")
    parser.add_argument("--max_sample_refill_rounds", type=int, default=5)
    parser.add_argument("--empty_length_retry_max_tokens", type=int, default=8)
    parser.add_argument("--max_empty_length_retry_rounds", type=int, default=4)
    parser.add_argument("--empty_response_token", default="stop_eos")
    parser.add_argument("--delay_seconds", type=float, default=0.0)
    parser.add_argument("--qwen_hard_no_think_prefill", action="store_true")
    parser.add_argument("--reject_reasoning_tokens", action="store_true")
    parser.add_argument("--reasoning_mode", default="enabled_false")

    parser.add_argument("--provider_order", nargs="*", default=None)
    parser.add_argument("--provider_allow_fallbacks", type=lambda value: str(value).lower() == "true", default=False)
    parser.add_argument("--router_metadata", action="store_true")
    parser.add_argument("--disable_openrouter_response_cache", action="store_true")
    parser.add_argument("--request_timeout", type=float, default=90)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_key_env", default="OPENROUTER_API_KEY")
    parser.add_argument("--api_key_file", default=None)
    parser.add_argument("--api_url", default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--progress_steps", action="store_true")
    return parser.parse_args()


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


def scores_for_counter(
    text_counts: dict[str, int],
    vocab: HybridQwenVocab,
    tokenizer,
    bias_model,
    device: torch.device,
    *,
    observed_alpha: float,
    floor_mass: float,
    residual_scale: float,
    position_id: int,
) -> tuple[torch.Tensor, Counter[int]]:
    action_counts = counter_to_hybrid_counts(text_counts, vocab, tokenizer)
    sampled = torch.tensor(
        [[action for action, count in sorted(action_counts.items()) for _ in range(count)]],
        dtype=torch.long,
    )
    base_scores = sampled_ids_to_log_probs(
        sampled_token_ids=sampled,
        vocab_size=vocab.size,
        observed_alpha=observed_alpha,
        floor_mass=floor_mass,
        dtype=torch.float32,
    ).to(device)
    scores = ior.apply_bias_model(
        bias_model,
        base_scores,
        residual_scale=residual_scale,
        position_id=position_id,
    )[0]
    return scores, action_counts


def choose_observed(
    text_counts: dict[str, int],
    scores: torch.Tensor,
    vocab: HybridQwenVocab,
    tokenizer,
) -> tuple[str, int]:
    """Choose an actually sampled string, preserving OOV text faithfully."""

    ranked = sorted(
        (
            (
                float(scores[vocab.encode_action(text, tokenizer)].item()),
                int(count),
                text,
                vocab.encode_action(text, tokenizer),
            )
            for text, count in text_counts.items()
        ),
        key=lambda item: (-item[0], -item[1], item[2]),
    )
    _, _, text, action_id = ranked[0]
    return text, int(action_id)


def generate_one(client, tokenizer, prompt, args, bias_model, vocab, device):
    sample_args = build_sample_args(args)
    valid_mask = vocab.valid_action_mask(tokenizer, device=device)
    prefix_text = ""
    changed_from_top = 0
    injected_emissions = 0
    extension_emissions = 0
    observed_oov_emissions = 0
    trace: list[dict[str, Any]] = []

    for position in range(args.max_new_tokens):
        _, stats = sample_position_token_ids(
            client=client,
            tokenizer=tokenizer,
            question=prompt,
            prefix_text=prefix_text,
            args=sample_args,
        )
        text_counts = dict(stats.get("sampled_completion_text_counts") or {})
        if not text_counts:
            break
        scores, action_counts = scores_for_counter(
            text_counts,
            vocab,
            tokenizer,
            bias_model,
            device,
            observed_alpha=args.mc_observed_alpha,
            floor_mass=args.mc_floor_mass,
            residual_scale=args.residual_scale,
            position_id=position,
        )

        if args.emit_mode == "full_vocab":
            masked = scores.masked_fill(~valid_mask, float("-inf"))
            chosen_id = int(masked.argmax().item())
            chosen_text = vocab.decode_action(chosen_id, tokenizer)
            if chosen_text is None:
                raise RuntimeError(f"Selected non-emittable hybrid action {chosen_id}.")
        else:
            chosen_text, chosen_id = choose_observed(
                text_counts, scores, vocab, tokenizer
            )

        if chosen_text == "" or chosen_id == vocab.eos_token_id:
            trace.append(
                {
                    "step": position,
                    "chosen": chosen_text,
                    "chosen_action_id": chosen_id,
                    "stop": True,
                }
            )
            break

        most_sampled = sorted(text_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        chosen_observed = chosen_text in text_counts
        if chosen_text != most_sampled:
            changed_from_top += 1
        if not chosen_observed:
            injected_emissions += 1
        if vocab.extension_start_id <= chosen_id < vocab.extension_start_id + vocab.extension_count:
            extension_emissions += 1
        if chosen_id == vocab.oov_id:
            observed_oov_emissions += 1

        trace.append(
            {
                "step": position,
                "candidates": len(text_counts),
                "chosen": chosen_text,
                "chosen_action_id": chosen_id,
                "chosen_observed": chosen_observed,
                "chosen_count": int(text_counts.get(chosen_text, 0)),
                "most_sampled": most_sampled,
                "top_count": int(text_counts[most_sampled]),
                "extension_action": bool(
                    vocab.extension_start_id
                    <= chosen_id
                    < vocab.extension_start_id + vocab.extension_count
                ),
                "observed_action_support": int(action_counts.get(chosen_id, 0)),
                "mc_stats": {
                    "valid_samples": stats.get("valid_samples"),
                    "reasoning_choices": stats.get("reasoning_choices"),
                    "actual_provider_counts": stats.get("actual_provider_counts"),
                    "actual_response_model_counts": stats.get("actual_response_model_counts"),
                },
            }
        )
        prefix_text += chosen_text
        if args.progress_steps and (position + 1) % 10 == 0:
            print(f"  step={position + 1}/{args.max_new_tokens}", flush=True)

    steps = sum(not item.get("stop", False) for item in trace)
    return prefix_text, {
        "summary": {
            "steps": steps,
            "mc_steps": len(trace),
            "controlled_token_steps": changed_from_top,
            "controlled_token_rate": changed_from_top / steps if steps else 0.0,
            "injected_emissions": injected_emissions,
            "injected_rate": injected_emissions / steps if steps else 0.0,
            "extension_emissions": extension_emissions,
            "extension_rate": extension_emissions / steps if steps else 0.0,
            "observed_oov_emissions": observed_oov_emissions,
            "emit_mode": args.emit_mode,
            "residual_scale": args.residual_scale,
            "action_space": "qwen_plus_empirical_extension",
        },
        "steps": trace,
    }


def main() -> None:
    args = parse_args()
    if args.begin < 0:
        raise ValueError("--begin must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it.")

    device = ior.resolve_device(args.device)
    vocab = HybridQwenVocab.load(args.vocab)
    tokenizer = load_tokenizer(
        args.tokenizer_name,
        use_fast=False,
        trust_remote_code=False,
        revision=args.tokenizer_revision,
    )
    if len(tokenizer) != vocab.base_vocab_size:
        raise ValueError(
            f"Tokenizer size {len(tokenizer)} != hybrid base size {vocab.base_vocab_size}."
        )
    if int(tokenizer.eos_token_id) != vocab.eos_token_id:
        raise ValueError(
            f"Tokenizer EOS {tokenizer.eos_token_id} != hybrid EOS {vocab.eos_token_id}."
        )
    bias_model = ior.load_biasnet(args.biasnet_ckpt, device, args.biasnet_dtype)
    if bias_model is None or int(getattr(bias_model, "vocab_size", -1)) != vocab.size:
        raise ValueError(
            f"BiasNet vocabulary {getattr(bias_model, 'vocab_size', None)} "
            f"!= hybrid vocabulary {vocab.size}."
        )

    records = collect_prompt_records(
        prompt=None,
        prompts=None,
        prompt_file=args.prompt_file,
        benchmark=args.benchmark,
        benchmark_file=args.benchmark_file,
        benchmark_mutation=None,
        limit=None,
    )
    records = records[args.begin :]
    if args.limit is not None:
        records = records[: args.limit]
    client_args = build_client_args(args)
    client = OpenRouterClient(client_args, resolve_api_key(client_args))
    mode = "w" if args.overwrite else "x"
    with output.open(mode, encoding="utf-8") as handle:
        for index, record in enumerate(records, start=1):
            question = record.prompt + ("\n/no_think" if args.append_no_think else "")
            before = ior.snapshot_openrouter_audit(client)
            started = time.perf_counter()
            completion, audit = generate_one(
                client, tokenizer, question, args, bias_model, vocab, device
            )
            row = {
                "prompt": record.prompt,
                "completion": completion,
                "risk_gate_runtime": {
                    "configuration": {
                        "action_space": "qwen_plus_empirical_extension",
                        "model": args.model,
                        "biasnet_checkpoint": args.biasnet_ckpt,
                        "vocab": args.vocab,
                        "base_vocab_size": vocab.base_vocab_size,
                        "extension_capacity": vocab.extension_capacity,
                        "extension_count": vocab.extension_count,
                        "total_vocab_size": vocab.size,
                        "max_new_tokens": args.max_new_tokens,
                    },
                    "summary": audit["summary"],
                },
                "openrouter_routing": ior.build_openrouter_routing_metadata(
                    client, client_args, before
                ),
                "generation_audit": {"steps": audit["steps"]},
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            summary = audit["summary"]
            print(
                f"finished_prompt={index}/{len(records)} steps={summary['steps']} "
                f"controlled={summary['controlled_token_rate']:.2f} "
                f"injected={summary['injected_rate']:.2f} "
                f"extension={summary['extension_rate']:.2f}",
                flush=True,
            )
    print(f"wrote={output}", flush=True)


if __name__ == "__main__":
    main()
