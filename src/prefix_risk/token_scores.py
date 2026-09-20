from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from .data import build_prefix_examples, format_prompt_with_prefix, split_examples
from .model import PrefixRiskModel, RiskHead, RiskHeadConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show per-answer-token prefix risk scores for one eval sample.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--field", choices=["rejected", "chosen"], default="rejected")
    parser.add_argument("--sample-index", type=int, default=0, help="Index among eval rows that contain the selected field.")
    parser.add_argument("--prefer-short", action="store_true", help="Sort eval candidates by answer token count first.")
    parser.add_argument("--max-answer-tokens", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-name", default=None)
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


def batch_score_texts(
    model: PrefixRiskModel,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    loader = DataLoader(texts, batch_size=batch_size, shuffle=False)
    model.eval()

    with torch.no_grad():
        for batch_texts in loader:
            encoded = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**encoded)
            scores.extend(torch.sigmoid(logits).detach().float().cpu().tolist())

    return scores


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint)
    with (checkpoint_dir / "risk_head_config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    model_name = args.model_name or config["model_name"]
    dataset_name = args.dataset_name or config["dataset_name"]
    split = args.split or config["split"]
    max_rows = args.max_rows

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    raw = load_dataset(dataset_name, split=split)
    if max_rows is not None:
        raw = raw.select(range(min(max_rows, len(raw))))

    examples = build_prefix_examples(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        split=split,
        max_rows=max_rows,
        prefixes_per_answer=config.get("prefixes_per_answer", 1),
        min_prefix_tokens=config.get("min_prefix_tokens", 8),
        min_prefix_ratio=config.get("min_prefix_ratio", 0.10),
        max_prefix_ratio=config.get("max_prefix_ratio", 1.00),
        use_chat_template=config.get("use_chat_template", True),
        seed=args.seed,
    )
    _, eval_examples = split_examples(examples, args.eval_fraction, args.seed)
    candidate_row_ids = sorted({example.row_id for example in eval_examples if example.source_field == args.field})
    candidates = [
        row_id
        for row_id in candidate_row_ids
        if str(raw[row_id].get("prompt") or "").strip()
        and str(raw[row_id].get(args.field) or "").strip()
    ]
    if not candidates:
        raise ValueError(f"No eval rows with field={args.field!r}.")

    if args.prefer_short:
        candidates.sort(
            key=lambda row_id: len(
                tokenizer(str(raw[row_id][args.field]).strip(), add_special_tokens=False).input_ids
            )
        )

    row_id = candidates[args.sample_index % len(candidates)]
    row = raw[row_id]
    prompt = str(row["prompt"]).strip()
    answer = str(row[args.field]).strip()
    all_answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    answer_ids = all_answer_ids[: args.max_answer_tokens]

    prefixes: list[str] = []
    tokens: list[str] = []
    for end in range(1, len(answer_ids) + 1):
        prefix = tokenizer.decode(answer_ids[:end], skip_special_tokens=False)
        token_text = tokenizer.decode([answer_ids[end - 1]], skip_special_tokens=False)
        prefixes.append(prefix)
        tokens.append(token_text)

    texts = [
        format_prompt_with_prefix(
            tokenizer=tokenizer,
            prompt=prompt,
            answer_prefix=prefix,
            use_chat_template=config.get("use_chat_template", True),
        )
        for prefix in prefixes
    ]

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

    scores = batch_score_texts(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=config["max_length"],
        batch_size=args.batch_size,
        device=device,
    )

    result = {
        "row_id": row_id,
        "field": args.field,
        "label": 1 if args.field == "rejected" else 0,
        "prompt": prompt,
        "answer_prefix_token_count": len(answer_ids),
        "answer_total_token_count": len(all_answer_ids),
        "truncated": len(answer_ids) < len(all_answer_ids),
        "token_scores": [
            {"i": index + 1, "token": token, "score": score}
            for index, (token, score) in enumerate(zip(tokens, scores))
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
