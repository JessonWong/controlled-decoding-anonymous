from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .build_guard_dataset import LlamaGuardScorer, resolve_dtype as resolve_guard_dtype
from .data import PrefixExample, load_prefix_examples_jsonl, split_examples
from .model import PrefixRiskModel, RiskHead, RiskHeadConfig
from .train import auroc, resolve_dtype as resolve_risk_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a risk head checkpoint against live Llama Guard.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labeled-jsonl", required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--fields", default="rejected,chosen", help="Comma-separated source fields to compare.")
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--risk-batch-size", type=int, default=8)
    parser.add_argument("--guard-model-name", default="meta-llama/Llama-Guard-3-8B")
    parser.add_argument("--guard-batch-size", type=int, default=8)
    parser.add_argument("--guard-max-length", type=int, default=2048)
    parser.add_argument("--guard-max-new-tokens", type=int, default=16)
    parser.add_argument("--guard-decision-mode", choices=["generate", "first-token-prob"], default="generate")
    parser.add_argument("--guard-unsafe-threshold", type=float, default=0.5)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--max-disagreements", type=int, default=20)
    return parser.parse_args()


def load_risk_backbone(model_name: str, args: argparse.Namespace) -> tuple[Any, torch.device]:
    model_kwargs: dict[str, Any] = {"torch_dtype": resolve_risk_dtype(args.dtype)}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    backbone = AutoModel.from_pretrained(model_name, **model_kwargs)
    if not args.load_in_4bit:
        backbone.to(device)
    else:
        device = next(backbone.parameters()).device
    return backbone, device


@torch.no_grad()
def score_risk_head(
    checkpoint_dir: Path,
    examples: list[PrefixExample],
    args: argparse.Namespace,
) -> list[float]:
    with (checkpoint_dir / "risk_head_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    model_name = args.model_name or config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    backbone, device = load_risk_backbone(model_name, args)
    layer_indices = tuple(config["layer_indices"])
    head_config = RiskHeadConfig(
        hidden_size=config["hidden_size"],
        layer_indices=layer_indices,
        head_hidden_size=config["head_hidden_size"],
        dropout=config["dropout"],
    )
    head = RiskHead(head_config).to(device)
    checkpoint = torch.load(checkpoint_dir / "risk_head.pt", map_location=device)
    head.load_state_dict(checkpoint["head_state_dict"])
    model = PrefixRiskModel(backbone=backbone, head=head, layer_indices=layer_indices)
    model.eval()

    scores: list[float] = []
    texts = [example.text for example in examples]
    for batch_texts in tqdm(
        DataLoader(texts, batch_size=args.risk_batch_size, shuffle=False),
        desc="risk head",
    ):
        encoded = tokenizer(
            list(batch_texts),
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded)
        scores.extend(torch.sigmoid(logits).detach().float().cpu().tolist())

    del model, head, backbone
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def score_llama_guard(examples: list[PrefixExample], args: argparse.Namespace) -> list[dict[str, Any]]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    scorer = LlamaGuardScorer(
        model_name=args.guard_model_name,
        dtype=resolve_guard_dtype(args.dtype),
        load_in_4bit=args.load_in_4bit,
        device=device,
        max_length=args.guard_max_length,
        batch_size=args.guard_batch_size,
        unsafe_threshold=args.guard_unsafe_threshold,
        decision_mode=args.guard_decision_mode,
        max_new_tokens=args.guard_max_new_tokens,
    )

    groups: dict[str, list[tuple[int, PrefixExample]]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[example.prompt].append((index, example))

    results: list[dict[str, Any] | None] = [None] * len(examples)
    for prompt, group in tqdm(groups.items(), desc="llama guard prompts"):
        responses = [example.answer_prefix for _, example in group]
        scores = scorer.score(prompt, responses)
        for (index, _), (unsafe_prob, unsafe, raw_output) in zip(group, scores):
            results[index] = {
                "unsafe_prob": unsafe_prob,
                "unsafe": unsafe,
                "raw_output": raw_output,
            }

    missing = [index for index, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"Llama Guard did not return scores for {len(missing)} examples.")
    return [result for result in results if result is not None]


def binary_summary(preds: list[int], labels: list[int], probs: list[float] | None = None) -> dict[str, float]:
    tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
    tn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 0)
    fp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 0)
    fn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 1)
    total = max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    result = {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    if probs is not None:
        result["auroc"] = auroc(torch.tensor(probs), torch.tensor(labels, dtype=torch.float32))
    return result


def truncate(text: str, limit: int = 300) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint)
    fields = {field.strip() for field in args.fields.split(",") if field.strip()}
    examples = load_prefix_examples_jsonl(args.labeled_jsonl)
    _, eval_examples = split_examples(examples, args.eval_fraction, args.seed)
    eval_examples = [example for example in eval_examples if example.source_field in fields]

    rng = random.Random(args.seed)
    rng.shuffle(eval_examples)
    if args.max_examples and len(eval_examples) > args.max_examples:
        eval_examples = eval_examples[: args.max_examples]

    risk_probs = score_risk_head(checkpoint_dir, eval_examples, args)
    guard_results = score_llama_guard(eval_examples, args)

    risk_preds = [int(prob >= args.risk_threshold) for prob in risk_probs]
    guard_labels = [int(result["unsafe"]) for result in guard_results]
    dataset_labels = [int(example.label >= 0.5) for example in eval_examples]

    by_field: dict[str, dict[str, Any]] = {}
    for field in sorted(fields):
        indices = [index for index, example in enumerate(eval_examples) if example.source_field == field]
        if not indices:
            continue
        by_field[field] = {
            "examples": len(indices),
            "dataset_positive": sum(dataset_labels[index] for index in indices),
            "guard_unsafe": sum(guard_labels[index] for index in indices),
            "risk_vs_guard": binary_summary(
                [risk_preds[index] for index in indices],
                [guard_labels[index] for index in indices],
                [risk_probs[index] for index in indices],
            ),
            "dataset_vs_guard": binary_summary(
                [dataset_labels[index] for index in indices],
                [guard_labels[index] for index in indices],
            ),
        }

    disagreement_examples = []
    for index, (example, risk_prob, risk_pred, guard_label, guard_result) in enumerate(
        zip(eval_examples, risk_probs, risk_preds, guard_labels, guard_results)
    ):
        if risk_pred == guard_label:
            continue
        disagreement_examples.append(
            {
                "index": index,
                "row_id": example.row_id,
                "source_field": example.source_field,
                "prefix_tokens": example.prefix_tokens,
                "answer_tokens": example.answer_tokens,
                "dataset_label": dataset_labels[index],
                "risk_prob": risk_prob,
                "risk_pred": risk_pred,
                "guard_unsafe": guard_label,
                "guard_raw_output": guard_result["raw_output"],
                "prompt": truncate(example.prompt),
                "answer_prefix": truncate(example.answer_prefix),
            }
        )

    checkpoint = torch.load(checkpoint_dir / "risk_head.pt", map_location="cpu")
    result = {
        "checkpoint": str(checkpoint_dir),
        "checkpoint_step": checkpoint.get("step"),
        "examples": len(eval_examples),
        "field_counts": dict(Counter(example.source_field for example in eval_examples)),
        "dataset_positive": sum(dataset_labels),
        "guard_unsafe": sum(guard_labels),
        "risk_vs_guard": binary_summary(risk_preds, guard_labels, risk_probs),
        "dataset_vs_guard": binary_summary(dataset_labels, guard_labels),
        "by_field": by_field,
        "disagreements": disagreement_examples[: args.max_disagreements],
        "disagreement_count": len(disagreement_examples),
    }

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
