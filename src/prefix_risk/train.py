from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

from .data import (
    PrefixRiskCollator,
    PrefixRiskDataset,
    build_prefix_examples,
    load_prefix_examples_jsonl,
    split_examples,
)
from .model import PrefixRiskModel, RiskHead, RiskHeadConfig, parse_layer_indices, validate_layer_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a harmful-continuation prefix risk head.")
    parser.add_argument("--model-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--dataset-name", default="LLM-LAT/harmful-dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--labeled-jsonl", default=None, help="Prebuilt prefix-level JSONL dataset.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="-1", help="Comma-separated hidden-state indices, e.g. -1 or -1,-8,-16.")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--prefixes-per-answer", type=int, default=3)
    parser.add_argument("--min-prefix-tokens", type=int, default=8)
    parser.add_argument("--min-prefix-ratio", type=float, default=0.10)
    parser.add_argument("--max-prefix-ratio", type=float, default=1.00)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--head-hidden-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits.detach().float().cpu())
    y = labels.detach().float().cpu()
    preds = (probs >= 0.5).float()

    tp = ((preds == 1) & (y == 1)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()
    total = max(1, len(y))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc(probs, y),
    }


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positives = labels == 1
    negatives = labels == 0
    n_pos = int(positives.sum().item())
    n_neg = int(negatives.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores, dtype=torch.float32)

    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    rank_sum_pos = ranks[positives].sum().item()
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@torch.no_grad()
def evaluate(
    model: PrefixRiskModel,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    logits_list = []
    labels_list = []

    for batch in dataloader:
        batch = move_batch(batch, device)
        labels = batch.pop("labels").to(model.head_device)
        logits = model(**batch)
        loss = loss_fn(logits, labels)
        losses.append(loss.detach().float().cpu())
        logits_list.append(logits.detach().float().cpu())
        labels_list.append(labels.detach().float().cpu())

    if not losses:
        return {}

    logits_all = torch.cat(logits_list)
    labels_all = torch.cat(labels_list)
    metrics = binary_metrics(logits_all, labels_all)
    metrics["loss"] = torch.stack(losses).mean().item()
    return metrics


def save_checkpoint(
    output_dir: Path,
    model: PrefixRiskModel,
    tokenizer: Any,
    args: argparse.Namespace,
    metrics: dict[str, float] | None,
    step: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    checkpoint = {
        "head_state_dict": model.head.state_dict(),
        "step": step,
        "metrics": metrics or {},
    }
    torch.save(checkpoint, output_dir / "risk_head.pt")

    config = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "labeled_jsonl": args.labeled_jsonl,
        "layer_indices": list(model.layer_indices),
        "hidden_size": model.backbone.config.hidden_size,
        "head_hidden_size": args.head_hidden_size,
        "dropout": args.dropout,
        "max_length": args.max_length,
        "use_chat_template": not args.no_chat_template,
        "prefixes_per_answer": args.prefixes_per_answer,
        "min_prefix_tokens": args.min_prefix_tokens,
        "min_prefix_ratio": args.min_prefix_ratio,
        "max_prefix_ratio": args.max_prefix_ratio,
    }
    with (output_dir / "risk_head_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def load_backbone_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, torch.device]:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = resolve_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {"torch_dtype": dtype}

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"

    backbone = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    if not args.load_in_4bit:
        backbone.to(device)
    else:
        first_param = next(backbone.parameters())
        device = first_param.device

    return backbone, tokenizer, device


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)

    backbone, tokenizer, device = load_backbone_and_tokenizer(args)
    layer_indices = parse_layer_indices(args.layers)

    # One tiny forward pass validates user-provided hidden-state indices before
    # building the dataset.
    probe = tokenizer("probe", return_tensors="pt").to(device)
    with torch.no_grad():
        probe_outputs = backbone(**probe, output_hidden_states=True, use_cache=False)
    validate_layer_indices(layer_indices, len(probe_outputs.hidden_states))

    if args.labeled_jsonl:
        examples = load_prefix_examples_jsonl(args.labeled_jsonl)
    else:
        examples = build_prefix_examples(
            tokenizer=tokenizer,
            dataset_name=args.dataset_name,
            split=args.split,
            max_rows=args.max_rows,
            prefixes_per_answer=args.prefixes_per_answer,
            min_prefix_tokens=args.min_prefix_tokens,
            min_prefix_ratio=args.min_prefix_ratio,
            max_prefix_ratio=args.max_prefix_ratio,
            use_chat_template=not args.no_chat_template,
            seed=args.seed,
        )
    train_examples, eval_examples = split_examples(examples, args.eval_fraction, args.seed)

    collator = PrefixRiskCollator(tokenizer, max_length=args.max_length)
    train_loader = DataLoader(
        PrefixRiskDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    eval_loader = None
    if eval_examples:
        eval_loader = DataLoader(
            PrefixRiskDataset(eval_examples),
            batch_size=args.eval_batch_size or args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
        )

    head_config = RiskHeadConfig(
        hidden_size=backbone.config.hidden_size,
        layer_indices=layer_indices,
        head_hidden_size=args.head_hidden_size,
        dropout=args.dropout,
    )
    head = RiskHead(head_config).to(device)
    model = PrefixRiskModel(backbone=backbone, head=head, layer_indices=layer_indices)

    optimizer = AdamW(model.head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_update_steps = max(1, update_steps_per_epoch * args.epochs)
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)
    loss_fn = nn.BCEWithLogitsLoss()

    print(
        json.dumps(
            {
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
                "layer_indices": layer_indices,
                "device": str(device),
                "total_update_steps": total_update_steps,
            },
            indent=2,
        )
    )

    global_step = 0
    best_eval_loss = float("inf")
    model.backbone.eval()
    model.head.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        running_loss = 0.0

        for step_in_epoch, batch in enumerate(progress, start=1):
            model.head.train()
            batch = move_batch(batch, device)
            labels = batch.pop("labels").to(model.head_device)

            logits = model(**batch)
            loss = loss_fn(logits, labels)
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            running_loss += loss.detach().float().item()

            should_step = (
                step_in_epoch % args.gradient_accumulation_steps == 0
                or step_in_epoch == len(train_loader)
            )
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_steps == 0:
                avg_loss = running_loss / max(1, args.log_steps * args.gradient_accumulation_steps)
                progress.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})
                running_loss = 0.0

            if eval_loader is not None and global_step % args.eval_steps == 0:
                metrics = evaluate(model, eval_loader, loss_fn, device)
                print({"step": global_step, "eval": metrics})
                if metrics and metrics["loss"] < best_eval_loss:
                    best_eval_loss = metrics["loss"]
                    save_checkpoint(output_dir / "best", model, tokenizer, args, metrics, global_step)

            if global_step % args.save_steps == 0:
                save_checkpoint(output_dir / "latest", model, tokenizer, args, None, global_step)

    final_metrics = evaluate(model, eval_loader, loss_fn, device) if eval_loader is not None else {}
    save_checkpoint(output_dir / "final", model, tokenizer, args, final_metrics, global_step)
    if final_metrics and final_metrics.get("loss", float("inf")) < best_eval_loss:
        save_checkpoint(output_dir / "best", model, tokenizer, args, final_metrics, global_step)

    print({"final": final_metrics, "checkpoint": str(output_dir / "final")})


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
