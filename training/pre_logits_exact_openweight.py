"""Cache exact teacher-forced next-token log-probabilities from a locally loaded
open-weight chat model.

The prompt construction, answer tokenization, vocabulary, risk-gate scoring and
on-disk layout deliberately mirror ``training/pre_logits_sampled_openrouter.py``
so that an exact cache is a drop-in replacement for an API Monte Carlo cache in
``training/train_biasnet.py``.

Passing ``--mc_output_dir`` additionally writes a Monte Carlo cache estimated
from the *same* forward pass and annotated with the *same* risk scores.  The two
caches then differ only in the next-token estimator, which isolates Monte Carlo
sampling noise from every other part of the attack.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

QWEN_HARD_NO_THINK_PREFILL = "<think>\n\n</think>\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache exact (and optionally Monte Carlo) next-token log-probability "
            "estimates from a locally loaded chat model."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Optional tokenizer identifier when it differs from the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for the exact dense log-probability cache.",
    )
    parser.add_argument(
        "--mc_output_dir",
        type=str,
        default=None,
        help=(
            "Optional second directory receiving a Monte Carlo cache built from "
            "the same forward pass and the same risk scores."
        ),
    )
    parser.add_argument("--dataset_name", type=str, default="LLM-LAT/harmful-dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--start_index", type=int, default=100)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=41)
    parser.add_argument(
        "--max_answer_tokens",
        type=int,
        default=None,
        help=(
            "Cap on cached answer tokens. Unset caches the whole target answer; "
            "a silent default here would quietly truncate an 'exact' cache."
        ),
    )
    parser.add_argument("--samples_per_token", type=int, default=50)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--observed_alpha", type=float, default=0.1)
    parser.add_argument("--floor_mass", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--torch_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--store_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--append_no_think",
        action="store_true",
        help="Append Qwen's /no_think soft switch to each user prompt.",
    )
    parser.add_argument(
        "--qwen_hard_no_think_prefill",
        action="store_true",
        help=(
            "Prefill an empty <think></think> block at the start of the assistant "
            "turn, matching the hard-prefill OpenRouter caches."
        ),
    )
    parser.add_argument("--risk_gate_checkpoint", type=str, default=None)
    parser.add_argument("--risk_gate_threshold", type=float, default=0.1)
    parser.add_argument("--risk_gate_batch_size", type=int, default=8)
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
    parser.add_argument("--risk_gate_local_files_only", action="store_true")
    return parser.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def load_risk_gate(args):
    if args.risk_gate_checkpoint is None:
        return None
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


def build_assistant_prefix(tokenizer, question: str, hard_no_think: bool) -> str:
    """Return the chat-formatted text that precedes the first answer token."""

    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if hard_no_think:
        prefix = prefix + QWEN_HARD_NO_THINK_PREFILL
    return prefix


def sample_token_ids(
    logits: torch.Tensor,
    samples_per_token: int,
    temperature: float,
    top_p: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if samples_per_token <= 0:
        raise ValueError("--samples_per_token must be positive.")
    if temperature <= 0:
        raise ValueError("--sample_temperature must be positive.")
    if top_p <= 0 or top_p > 1:
        raise ValueError("--top_p must be in the interval (0, 1].")

    scaled_logits = logits.float() / temperature
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        sampled_sorted = torch.multinomial(
            torch.softmax(sorted_logits, dim=-1),
            num_samples=samples_per_token,
            replacement=True,
            generator=generator,
        )
        return sorted_indices.gather(dim=-1, index=sampled_sorted)

    return torch.multinomial(
        torch.softmax(scaled_logits, dim=-1),
        num_samples=samples_per_token,
        replacement=True,
        generator=generator,
    )


def sampled_ids_to_log_probs(
    sampled_token_ids: torch.Tensor,
    vocab_size: int,
    observed_alpha: float,
    floor_mass: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the API estimator: smoothed observed mass plus a uniform floor."""

    if sampled_token_ids.dim() != 2:
        raise ValueError("sampled_token_ids must have shape [seq_len, samples_per_token].")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1.")
    if observed_alpha < 0:
        raise ValueError("--observed_alpha must be non-negative.")
    if floor_mass < 0 or floor_mass >= 1:
        raise ValueError("--floor_mass must be in the interval [0, 1).")

    seq_len, samples_per_token = sampled_token_ids.shape
    rows = torch.empty((seq_len, vocab_size), dtype=torch.float32)
    counts_matrix = torch.zeros((seq_len, vocab_size), dtype=torch.int32)
    for row_idx in range(seq_len):
        row_samples = sampled_token_ids[row_idx].detach().cpu().long()
        unique_ids, counts = torch.unique(row_samples, sorted=False, return_counts=True)
        observed_count = unique_ids.numel()
        unseen_count = vocab_size - observed_count
        if unseen_count <= 0:
            unseen_log_prob = -torch.inf
            observed_mass = 1.0
        else:
            unseen_log_prob = torch.log(
                torch.tensor(floor_mass / unseen_count, dtype=torch.float32)
            )
            observed_mass = 1.0 - floor_mass
        rows[row_idx].fill_(unseen_log_prob)
        denom = float(samples_per_token) + observed_alpha * float(observed_count)
        observed_probs = observed_mass * (counts.float() + observed_alpha) / denom
        rows[row_idx, unique_ids] = observed_probs.log()
        counts_matrix[row_idx, unique_ids] = counts.to(torch.int32)

    return rows.to(dtype), counts_matrix


@torch.no_grad()
def build_risk_gate_payload(
    risk_gate,
    tokenizer,
    question: str,
    answer_ids: torch.Tensor,
    base_token_ids: torch.Tensor,
) -> dict:
    if risk_gate is None:
        return {}
    target_ids = answer_ids[0].detach().cpu().long().tolist()
    current_ids = base_token_ids.detach().cpu().long().tolist()
    prompts = [question] * len(current_ids)
    answer_prefixes = [
        tokenizer.decode(target_ids[:position] + [int(token_id)], skip_special_tokens=False)
        for position, token_id in enumerate(current_ids)
    ]
    scores = risk_gate.score_prefixes(prompts, answer_prefixes)
    return {
        "risk_gate_mask": (scores < risk_gate.threshold).unsqueeze(0).cpu().bool(),
        "risk_gate_scores": scores.unsqueeze(0).cpu().float(),
        "risk_gate_token_ids": base_token_ids.unsqueeze(0).detach().cpu().long(),
    }


@torch.no_grad()
def teacher_forced_logits(
    model,
    tokenizer,
    prefix_text: str,
    answer_token_ids: list[int],
    vocab_size: int,
) -> torch.Tensor:
    """Return next-token logits over the tokenizer vocabulary, one row per answer token."""

    device = model_device(model)
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    input_ids = torch.tensor(
        [list(prefix_ids) + list(answer_token_ids)], dtype=torch.long, device=device
    )
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[0, len(prefix_ids) - 1 : input_ids.shape[1] - 1, :]
    if logits.shape[0] != len(answer_token_ids):
        raise ValueError("Teacher-forced logits do not align with the answer tokens.")
    # The API can only ever emit real tokens, so drop the model's padded
    # vocabulary slots before normalising rather than after.
    return logits[:, :vocab_size].float()


def write_manifest(output_dir: str, configuration: dict) -> None:
    path = os.path.join(output_dir, "cache_manifest.json")
    payload = {"configuration": configuration, "manifest_schema_version": 1}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive.")
    if args.max_answer_tokens is not None and args.max_answer_tokens <= 0:
        raise ValueError("--max_answer_tokens must be positive when provided.")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mc_output_dir is not None:
        os.makedirs(args.mc_output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=resolve_torch_dtype(args.torch_dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    device = model_device(model)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    risk_gate = load_risk_gate(args)
    store_dtype = resolve_torch_dtype(args.store_dtype)

    shared_configuration = {
        "append_no_think": bool(args.append_no_think),
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "end_index": args.end_index,
        "max_answer_tokens": args.max_answer_tokens,
        "max_samples": args.max_samples,
        "model_name_or_path": args.model_name_or_path,
        "qwen_hard_no_think_prefill": bool(args.qwen_hard_no_think_prefill),
        "risk_gate_checkpoint": args.risk_gate_checkpoint,
        "risk_gate_threshold": args.risk_gate_threshold,
        "seed": args.seed,
        "start_index": args.start_index,
        "store_dtype": args.store_dtype,
        "tokenizer_name": args.tokenizer_name or args.model_name_or_path,
        "torch_dtype": args.torch_dtype,
        "vocab_size": vocab_size,
    }
    write_manifest(args.output_dir, {**shared_configuration, "source": "exact_openweight"})
    if args.mc_output_dir is not None:
        write_manifest(
            args.mc_output_dir,
            {
                **shared_configuration,
                "source": "sampled_openweight_local",
                "samples_per_token": args.samples_per_token,
                "sample_temperature": args.sample_temperature,
                "top_p": args.top_p,
                "observed_alpha": args.observed_alpha,
                "floor_mass": args.floor_mass,
            },
        )

    dataset = load_dataset(args.dataset_name)[args.dataset_split]
    dataset_stop = min(len(dataset), args.end_index) if args.end_index is not None else len(dataset)

    saved_count = len([name for name in os.listdir(args.output_dir) if name.endswith(".pt")])
    progress = tqdm(total=max(args.max_samples - saved_count, 0), desc="Exact log-prob caches")
    target_ranks: list[int] = []
    for dataset_idx in range(args.start_index, dataset_stop):
        if saved_count >= args.max_samples:
            break
        item = dataset[dataset_idx]
        question = item["prompt"]
        answer = item["rejected"]
        data_hash = hashlib.md5((question + answer).encode("utf-8")).hexdigest()
        exact_path = os.path.join(args.output_dir, f"{data_hash}.pt")
        mc_path = (
            os.path.join(args.mc_output_dir, f"{data_hash}.pt")
            if args.mc_output_dir is not None
            else None
        )
        if os.path.exists(exact_path) and (mc_path is None or os.path.exists(mc_path)):
            continue

        model_question = question + "\n/no_think" if args.append_no_think else question
        answer_token_ids = tokenizer.encode(answer, add_special_tokens=False)
        if args.max_answer_tokens is not None:
            answer_token_ids = answer_token_ids[: args.max_answer_tokens]
        if not answer_token_ids:
            continue

        prefix_text = build_assistant_prefix(
            tokenizer, model_question, args.qwen_hard_no_think_prefill
        )
        logits = teacher_forced_logits(
            model, tokenizer, prefix_text, answer_token_ids, vocab_size
        )
        labels = torch.tensor([answer_token_ids], dtype=torch.long)
        base_token_ids = logits.argmax(dim=-1)
        risk_payload = build_risk_gate_payload(
            risk_gate=risk_gate,
            tokenizer=tokenizer,
            question=question,
            answer_ids=labels,
            base_token_ids=base_token_ids.cpu(),
        )

        exact_log_probs = torch.log_softmax(logits, dim=-1)
        target_log_probs = exact_log_probs.gather(1, labels[0].to(logits.device).unsqueeze(1))
        target_rank = (exact_log_probs > target_log_probs).sum(dim=-1)
        target_ranks.extend(target_rank.cpu().tolist())

        exact_payload = {
            "log_probs": exact_log_probs.to(store_dtype).unsqueeze(0).cpu(),
            "labels": labels,
            "metadata": {
                "source": "exact_openweight",
                "model_name_or_path": args.model_name_or_path,
                "torch_dtype": args.torch_dtype,
                "vocab_size": vocab_size,
                "qwen_hard_no_think_prefill": bool(args.qwen_hard_no_think_prefill),
                "append_no_think": bool(args.append_no_think),
                "dataset_index": dataset_idx,
            },
        }
        exact_payload.update(risk_payload)
        if risk_gate is not None:
            exact_payload["metadata"].update(
                {
                    "risk_gate_checkpoint": args.risk_gate_checkpoint,
                    "risk_gate_threshold": args.risk_gate_threshold,
                    "risk_gate_token_source": "base_argmax",
                }
            )
        torch.save(exact_payload, exact_path)

        if mc_path is not None:
            sampled_ids = sample_token_ids(
                logits=logits,
                samples_per_token=args.samples_per_token,
                temperature=args.sample_temperature,
                top_p=args.top_p,
                generator=generator,
            )
            mc_log_probs, mc_counts = sampled_ids_to_log_probs(
                sampled_token_ids=sampled_ids,
                vocab_size=vocab_size,
                observed_alpha=args.observed_alpha,
                floor_mass=args.floor_mass,
                dtype=store_dtype,
            )
            mc_payload = {
                "log_probs": mc_log_probs.unsqueeze(0),
                "labels": labels,
                "mc_counts": mc_counts.unsqueeze(0),
                "valid_sample_counts": torch.full_like(labels, args.samples_per_token),
                "metadata": {
                    "source": "sampled_openweight_local",
                    "model_name_or_path": args.model_name_or_path,
                    "torch_dtype": args.torch_dtype,
                    "vocab_size": vocab_size,
                    "samples_per_token": args.samples_per_token,
                    "sample_temperature": args.sample_temperature,
                    "top_p": args.top_p,
                    "observed_alpha": args.observed_alpha,
                    "floor_mass": args.floor_mass,
                    "sample_completion_policy": "exact",
                    "qwen_hard_no_think_prefill": bool(args.qwen_hard_no_think_prefill),
                    "append_no_think": bool(args.append_no_think),
                    "dataset_index": dataset_idx,
                },
            }
            mc_payload.update(risk_payload)
            if risk_gate is not None:
                mc_payload["metadata"].update(
                    {
                        "risk_gate_checkpoint": args.risk_gate_checkpoint,
                        "risk_gate_threshold": args.risk_gate_threshold,
                        "risk_gate_token_source": "base_argmax",
                    }
                )
            torch.save(mc_payload, mc_path)

        saved_count += 1
        progress.update(1)

    progress.close()
    if target_ranks:
        ranks = torch.tensor(target_ranks, dtype=torch.float32)
        print(
            "target_rank_mean=" + f"{ranks.mean().item():.3f}",
            "target_is_argmax_rate=" + f"{(ranks == 0).float().mean().item():.4f}",
            "target_in_top50_rate=" + f"{(ranks < 50).float().mean().item():.4f}",
        )
    if saved_count < args.max_samples:
        print(f"Warning: only saved {saved_count} files before the dataset ended.", file=sys.stderr)
    print(f"Processed {saved_count} samples. Exact log-probs saved in {args.output_dir}.")


if __name__ == "__main__":
    main()
