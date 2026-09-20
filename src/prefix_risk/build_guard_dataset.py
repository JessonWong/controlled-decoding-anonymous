from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data import format_prompt_with_prefix


@dataclass(frozen=True)
class GuardScore:
    prefix_tokens: int
    unsafe_prob: float
    unsafe: bool
    raw_output: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scheme-B prefix labels with Llama Guard.")
    parser.add_argument("--dataset-name", default="LLM-LAT/harmful-dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--target-model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--guard-model-name", default="meta-llama/Llama-Guard-3-8B")
    parser.add_argument("--fields", default="rejected,chosen")
    parser.add_argument(
        "--trusted-safe-fields",
        default="chosen",
        help="Comma-separated fields to label as 0 without scanning; use an empty string to scan all fields.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-answer-tokens", type=int, default=None)
    parser.add_argument("--scan-stride", type=int, default=4)
    parser.add_argument("--output-stride", type=int, default=1)
    parser.add_argument("--max-output-prefixes", type=int, default=None)
    parser.add_argument("--unsafe-threshold", type=float, default=0.5)
    parser.add_argument("--decision-mode", choices=["generate", "first-token-prob"], default="generate")
    parser.add_argument("--guard-max-new-tokens", type=int, default=16)
    parser.add_argument("--guard-max-length", type=int, default=2048)
    parser.add_argument("--guard-batch-size", type=int, default=8)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def prefix_lengths(answer_token_count: int, stride: int) -> list[int]:
    if answer_token_count <= 0:
        return []
    if stride <= 1:
        return list(range(1, answer_token_count + 1))
    lengths = {1, answer_token_count}
    lengths.update(range(stride, answer_token_count + 1, stride))
    return sorted(lengths)


def cap_prefix_lengths(lengths: list[int], max_output_prefixes: int | None) -> list[int]:
    if max_output_prefixes is None or len(lengths) <= max_output_prefixes:
        return lengths
    if max_output_prefixes <= 1:
        return [lengths[-1]]

    capped = {lengths[0], lengths[-1]}
    for i in range(max_output_prefixes):
        index = round(i * (len(lengths) - 1) / max(1, max_output_prefixes - 1))
        capped.add(lengths[index])
    return sorted(capped)


def format_guard_response_input(tokenizer: Any, prompt: str, response: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    return f"User: {prompt}\n\nAssistant: {response}\n\nSafety:"


class LlamaGuardScorer:
    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype | str,
        load_in_4bit: bool,
        device: torch.device,
        max_length: int,
        batch_size: int,
        unsafe_threshold: float,
        decision_mode: str,
        max_new_tokens: int,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        safe_ids = self._first_token_ids(("safe", " safe"))
        unsafe_ids = self._first_token_ids(("unsafe", " unsafe"))
        if not safe_ids or not unsafe_ids:
            raise ValueError("Could not resolve Llama Guard safe/unsafe token ids.")
        self.safe_token_ids = safe_ids
        self.unsafe_token_ids = unsafe_ids

        model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not load_in_4bit:
            self.model.to(device)
            self.device = device
        else:
            self.device = next(self.model.parameters()).device
        self.model.eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.unsafe_threshold = unsafe_threshold
        self.decision_mode = decision_mode
        self.max_new_tokens = max_new_tokens

    def _first_token_ids(self, labels: tuple[str, ...]) -> list[int]:
        token_ids = []
        for label in labels:
            ids = self.tokenizer(label, add_special_tokens=False).input_ids
            if ids:
                token_ids.append(ids[0])
        return sorted(set(token_ids))

    @torch.no_grad()
    def score(self, prompt: str, responses: list[str]) -> list[tuple[float, bool, str | None]]:
        results: list[tuple[float, bool, str | None]] = []
        for start in range(0, len(responses), self.batch_size):
            chunk = responses[start : start + self.batch_size]
            texts = [
                format_guard_response_input(self.tokenizer, prompt=prompt, response=response)
                for response in chunk
            ]
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            if self.decision_mode == "generate":
                outputs = self.model.generate(
                    **batch,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                generated = outputs[:, batch["input_ids"].shape[1] :]
                for generated_ids in generated:
                    raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    normalized = raw_output.lower()
                    unsafe = normalized.startswith("unsafe")
                    safe = normalized.startswith("safe")
                    if not unsafe and not safe:
                        unsafe = "unsafe" in normalized.split()
                    results.append((1.0 if unsafe else 0.0, unsafe, raw_output))
                continue

            outputs = self.model(**batch, use_cache=False)

            positions = torch.arange(batch["attention_mask"].size(1), device=self.device).unsqueeze(0)
            last_token_index = (batch["attention_mask"].long() * positions).max(dim=1).values
            batch_index = torch.arange(batch["input_ids"].size(0), device=self.device)
            next_token_logits = outputs.logits[batch_index, last_token_index]
            safe_logits = torch.logsumexp(next_token_logits[:, self.safe_token_ids].float(), dim=-1)
            unsafe_logits = torch.logsumexp(next_token_logits[:, self.unsafe_token_ids].float(), dim=-1)
            safe_unsafe_logits = torch.stack([safe_logits, unsafe_logits], dim=-1)
            chunk_probs = torch.softmax(safe_unsafe_logits.float(), dim=-1)[:, 1]
            for unsafe_prob in chunk_probs.detach().cpu().tolist():
                results.append((unsafe_prob, unsafe_prob >= self.unsafe_threshold, None))
        return results

    def classify(self, prompt: str, prefixes_by_len: dict[int, str]) -> list[GuardScore]:
        lengths = sorted(prefixes_by_len)
        scores = self.score(prompt, [prefixes_by_len[length] for length in lengths])
        return [
            GuardScore(prefix_tokens=length, unsafe_prob=unsafe_prob, unsafe=unsafe, raw_output=raw_output)
            for length, (unsafe_prob, unsafe, raw_output) in zip(lengths, scores)
        ]


def find_first_unsafe_token(
    scorer: LlamaGuardScorer,
    prompt: str,
    answer_ids: list[int],
    target_tokenizer: Any,
    scan_stride: int,
) -> tuple[int | None, list[GuardScore]]:
    coarse_lengths = prefix_lengths(len(answer_ids), scan_stride)
    score_by_len: dict[int, GuardScore] = {}
    first_coarse_unsafe = None

    for start in range(0, len(coarse_lengths), scorer.batch_size):
        chunk_lengths = coarse_lengths[start : start + scorer.batch_size]
        prefixes_by_len = {
            length: target_tokenizer.decode(answer_ids[:length], skip_special_tokens=False)
            for length in chunk_lengths
        }
        for score in scorer.classify(prompt, prefixes_by_len):
            score_by_len[score.prefix_tokens] = score
        first_coarse_unsafe = next(
            (length for length in chunk_lengths if score_by_len[length].unsafe),
            None,
        )
        if first_coarse_unsafe is not None:
            break

    if first_coarse_unsafe is None:
        return None, [score_by_len[length] for length in sorted(score_by_len)]

    previous_scanned = 0
    for length in coarse_lengths:
        if length >= first_coarse_unsafe:
            break
        previous_scanned = length

    refine_lengths = list(range(previous_scanned + 1, first_coarse_unsafe + 1))
    missing_refine = [length for length in refine_lengths if length not in score_by_len]
    if missing_refine:
        refine_prefixes = {
            length: target_tokenizer.decode(answer_ids[:length], skip_special_tokens=False)
            for length in missing_refine
        }
        for score in scorer.classify(prompt, refine_prefixes):
            score_by_len[score.prefix_tokens] = score

    first_unsafe = next(
        (length for length in refine_lengths if score_by_len[length].unsafe),
        first_coarse_unsafe,
    )
    return first_unsafe, [score_by_len[length] for length in sorted(score_by_len)]


def write_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path) if args.summary_path else output_path.with_suffix(".summary.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name, use_fast=True)
    if target_tokenizer.pad_token is None:
        target_tokenizer.pad_token = target_tokenizer.eos_token

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    scorer = LlamaGuardScorer(
        model_name=args.guard_model_name,
        dtype=resolve_dtype(args.dtype),
        load_in_4bit=args.load_in_4bit,
        device=device,
        max_length=args.guard_max_length,
        batch_size=args.guard_batch_size,
        unsafe_threshold=args.unsafe_threshold,
        decision_mode=args.decision_mode,
        max_new_tokens=args.guard_max_new_tokens,
    )

    raw = load_dataset(args.dataset_name, split=args.split)
    if args.max_rows is not None:
        raw = raw.select(range(min(args.max_rows, len(raw))))

    fields = tuple(field.strip() for field in args.fields.split(",") if field.strip())
    trusted_safe_fields = {
        field.strip() for field in args.trusted_safe_fields.split(",") if field.strip()
    }
    required = {"prompt", *fields}
    missing = required.difference(raw.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    stats = {
        "answers": 0,
        "answers_with_guard_unsafe": 0,
        "prefix_examples": 0,
        "positive_prefix_examples": 0,
        "negative_prefix_examples": 0,
    }

    with output_path.open("w", encoding="utf-8") as out_handle, summary_path.open(
        "w", encoding="utf-8"
    ) as summary_handle:
        for row_id, row in enumerate(tqdm(raw, desc="labeling rows")):
            prompt = str(row["prompt"] or "").strip()
            if not prompt:
                continue

            for field in fields:
                answer = str(row[field] or "").strip()
                if not answer:
                    continue

                answer_ids = target_tokenizer(answer, add_special_tokens=False).input_ids
                if args.max_answer_tokens is not None:
                    answer_ids = answer_ids[: args.max_answer_tokens]
                if not answer_ids:
                    continue

                if field in trusted_safe_fields:
                    first_unsafe_token = None
                    guard_scores: list[GuardScore] = []
                    answer_decision_mode = "trusted_safe_field"
                else:
                    first_unsafe_token, guard_scores = find_first_unsafe_token(
                        scorer=scorer,
                        prompt=prompt,
                        answer_ids=answer_ids,
                        target_tokenizer=target_tokenizer,
                        scan_stride=args.scan_stride,
                    )
                    answer_decision_mode = args.decision_mode
                stats["answers"] += 1
                if first_unsafe_token is not None:
                    stats["answers_with_guard_unsafe"] += 1

                write_jsonl_record(
                    summary_handle,
                    {
                        "row_id": row_id,
                        "source_field": field,
                        "prompt": prompt,
                        "answer_tokens": len(answer_ids),
                        "first_unsafe_token": first_unsafe_token,
                        "unsafe_threshold": args.unsafe_threshold,
                        "scan_stride": args.scan_stride,
                        "decision_mode": answer_decision_mode,
                        "trusted_safe_field": field in trusted_safe_fields,
                        "guard_scores": [
                            {
                                "prefix_tokens": score.prefix_tokens,
                                "unsafe_prob": score.unsafe_prob,
                                "unsafe": score.unsafe,
                                "raw_output": score.raw_output,
                            }
                            for score in guard_scores
                        ],
                    },
                )

                output_lengths = cap_prefix_lengths(
                    prefix_lengths(len(answer_ids), args.output_stride),
                    args.max_output_prefixes,
                )
                for prefix_len in output_lengths:
                    answer_prefix = target_tokenizer.decode(
                        answer_ids[:prefix_len],
                        skip_special_tokens=False,
                    )
                    label = (
                        1.0
                        if first_unsafe_token is not None and prefix_len >= first_unsafe_token
                        else 0.0
                    )
                    text = format_prompt_with_prefix(
                        tokenizer=target_tokenizer,
                        prompt=prompt,
                        answer_prefix=answer_prefix,
                        use_chat_template=not args.no_chat_template,
                    )
                    write_jsonl_record(
                        out_handle,
                        {
                            "text": text,
                            "label": label,
                            "label_source": "llama_guard_first_unsafe_boundary",
                            "row_id": row_id,
                            "prompt": prompt,
                            "answer_prefix": answer_prefix,
                            "source_field": field,
                            "prefix_tokens": prefix_len,
                            "answer_tokens": len(answer_ids),
                            "first_unsafe_token": first_unsafe_token,
                            "guard_model_name": args.guard_model_name,
                            "unsafe_threshold": args.unsafe_threshold,
                            "guard_decision_mode": answer_decision_mode,
                        },
                    )
                    stats["prefix_examples"] += 1
                    if label >= 0.5:
                        stats["positive_prefix_examples"] += 1
                    else:
                        stats["negative_prefix_examples"] += 1

    metadata = {
        "args": vars(args),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "stats": stats,
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
