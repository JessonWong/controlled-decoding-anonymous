from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .data import format_prompt_with_prefix
from .model import PrefixRiskModel, RiskHead, RiskHeadConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score harmful-continuation risk for one prefix.")
    parser.add_argument("--checkpoint", required=True, help="Directory containing risk_head.pt and risk_head_config.json.")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--partial-answer", required=True)
    parser.add_argument("--model-name", default=None, help="Override the backbone from the checkpoint config.")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


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

    backbone, device = load_backbone(model_name, args)
    layer_indices = tuple(config["layer_indices"])
    head_config = RiskHeadConfig(
        hidden_size=config["hidden_size"],
        layer_indices=layer_indices,
        head_hidden_size=config["head_hidden_size"],
        dropout=config["dropout"],
    )
    head = RiskHead(head_config).to(device)
    state = torch.load(checkpoint_dir / "risk_head.pt", map_location=device)
    head.load_state_dict(state["head_state_dict"])

    model = PrefixRiskModel(backbone=backbone, head=head, layer_indices=layer_indices)
    model.eval()

    text = format_prompt_with_prefix(
        tokenizer=tokenizer,
        prompt=args.instruction,
        answer_prefix=args.partial_answer,
        use_chat_template=config.get("use_chat_template", True),
    )
    batch = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=args.max_length or config["max_length"],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**batch)
        probability = torch.sigmoid(logits)[0].item()

    print(json.dumps({"harmful_continuation_risk": probability}, indent=2))


if __name__ == "__main__":
    main()
