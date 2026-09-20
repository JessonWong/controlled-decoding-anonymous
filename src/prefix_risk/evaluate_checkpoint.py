from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .data import (
    PrefixRiskCollator,
    PrefixRiskDataset,
    build_prefix_examples,
    load_prefix_examples_jsonl,
    split_examples,
)
from .model import PrefixRiskModel, RiskHead, RiskHeadConfig
from .train import evaluate, resolve_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved prefix risk checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labeled-jsonl", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_backbone(model_name: str, args: argparse.Namespace) -> tuple[Any, torch.device]:
    model_kwargs: dict[str, Any] = {"torch_dtype": resolve_dtype(args.dtype)}
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


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint)
    with (checkpoint_dir / "risk_head_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    model_name = args.model_name or config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.labeled_jsonl:
        examples = load_prefix_examples_jsonl(args.labeled_jsonl)
    else:
        examples = build_prefix_examples(
            tokenizer=tokenizer,
            dataset_name=args.dataset_name or config["dataset_name"],
            split=args.split or config["split"],
            prefixes_per_answer=config.get("prefixes_per_answer", 3),
            min_prefix_tokens=config.get("min_prefix_tokens", 8),
            min_prefix_ratio=config.get("min_prefix_ratio", 0.10),
            max_prefix_ratio=config.get("max_prefix_ratio", 1.00),
            use_chat_template=config.get("use_chat_template", True),
            seed=args.seed,
        )
    _, eval_examples = split_examples(examples, args.eval_fraction, args.seed)

    backbone, device = load_backbone(model_name, args)
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

    collator = PrefixRiskCollator(tokenizer, max_length=args.max_length or config["max_length"])
    eval_loader = DataLoader(
        PrefixRiskDataset(eval_examples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    metrics = evaluate(model, eval_loader, nn.BCEWithLogitsLoss(), device)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_dir),
                "checkpoint_step": checkpoint.get("step"),
                "eval_examples": len(eval_examples),
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
