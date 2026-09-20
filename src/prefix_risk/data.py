from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class PrefixExample:
    text: str
    label: float
    row_id: int
    prompt: str
    answer_prefix: str
    source_field: str
    prefix_tokens: int
    answer_tokens: int


def format_prompt_with_prefix(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    answer_prefix: str,
    use_chat_template: bool = True,
) -> str:
    """Format a user instruction plus partial assistant answer.

    For chat models, we use the tokenizer chat template for the user turn plus
    the assistant generation header, then append the partial answer without an
    end-of-turn token.
    """

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            return rendered + answer_prefix
        except Exception:
            # Fall through to a plain format for tokenizers with broken or
            # incomplete chat templates.
            pass

    return f"### Instruction:\n{prompt}\n\n### Response:\n{answer_prefix}"


def choose_prefix_lengths(
    answer_token_count: int,
    prefixes_per_answer: int,
    min_tokens: int,
    min_ratio: float,
    max_ratio: float,
) -> list[int]:
    if answer_token_count <= 0:
        return []

    lower = min(
        answer_token_count,
        max(1, min_tokens, int(round(answer_token_count * min_ratio))),
    )
    upper = max(lower, min(answer_token_count, int(answer_token_count * max_ratio)))

    if prefixes_per_answer <= 1:
        return [upper]

    if upper == lower:
        return [upper]

    lengths = []
    for i in range(prefixes_per_answer):
        ratio = i / max(1, prefixes_per_answer - 1)
        length = round(lower + ratio * (upper - lower))
        lengths.append(max(1, min(answer_token_count, length)))

    return sorted(set(lengths))


def build_prefix_examples(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "LLM-LAT/harmful-dataset",
    split: str = "train",
    max_rows: int | None = None,
    prefixes_per_answer: int = 3,
    min_prefix_tokens: int = 8,
    min_prefix_ratio: float = 0.10,
    max_prefix_ratio: float = 1.00,
    use_chat_template: bool = True,
    seed: int = 42,
) -> list[PrefixExample]:
    raw = load_dataset(dataset_name, split=split)
    if max_rows is not None:
        raw = raw.select(range(min(max_rows, len(raw))))

    required = {"prompt", "rejected", "chosen"}
    missing = required.difference(raw.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    examples: list[PrefixExample] = []
    fields_and_labels = (("rejected", 1.0), ("chosen", 0.0))

    for row_id, row in enumerate(raw):
        prompt = str(row["prompt"] or "").strip()
        if not prompt:
            continue

        for field, label in fields_and_labels:
            answer = str(row[field] or "").strip()
            if not answer:
                continue

            answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
            prefix_lengths = choose_prefix_lengths(
                answer_token_count=len(answer_ids),
                prefixes_per_answer=prefixes_per_answer,
                min_tokens=min_prefix_tokens,
                min_ratio=min_prefix_ratio,
                max_ratio=max_prefix_ratio,
            )

            for prefix_len in prefix_lengths:
                prefix_ids = answer_ids[:prefix_len]
                answer_prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False)
                text = format_prompt_with_prefix(
                    tokenizer=tokenizer,
                    prompt=prompt,
                    answer_prefix=answer_prefix,
                    use_chat_template=use_chat_template,
                )
                examples.append(
                    PrefixExample(
                        text=text,
                        label=label,
                        row_id=row_id,
                        prompt=prompt,
                        answer_prefix=answer_prefix,
                        source_field=field,
                        prefix_tokens=prefix_len,
                        answer_tokens=len(answer_ids),
                    )
                )

    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples


def split_examples(
    examples: list[PrefixExample],
    eval_fraction: float,
    seed: int,
) -> tuple[list[PrefixExample], list[PrefixExample]]:
    if not examples:
        raise ValueError("No prefix examples were built.")

    groups: dict[int, list[PrefixExample]] = {}
    for example in examples:
        groups.setdefault(example.row_id, []).append(example)

    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    eval_group_count = max(1, int(len(group_ids) * eval_fraction)) if eval_fraction > 0 else 0
    if eval_group_count == 0:
        return examples, []

    eval_group_ids = set(group_ids[:eval_group_count])
    train: list[PrefixExample] = []
    eval_: list[PrefixExample] = []
    for group_id, group_examples in groups.items():
        if group_id in eval_group_ids:
            eval_.extend(group_examples)
        else:
            train.extend(group_examples)

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def load_prefix_examples_jsonl(path: str | Path) -> list[PrefixExample]:
    examples: list[PrefixExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                examples.append(
                    PrefixExample(
                        text=str(record["text"]),
                        label=float(record["label"]),
                        row_id=int(record["row_id"]),
                        prompt=str(record.get("prompt", "")),
                        answer_prefix=str(record.get("answer_prefix", "")),
                        source_field=str(record.get("source_field", record.get("field", ""))),
                        prefix_tokens=int(record["prefix_tokens"]),
                        answer_tokens=int(record["answer_tokens"]),
                    )
                )
            except KeyError as error:
                raise ValueError(f"{path}:{line_number} is missing required key {error!s}") from error

    if not examples:
        raise ValueError(f"No prefix examples were loaded from {path}.")
    return examples


class PrefixRiskDataset(Dataset[PrefixExample]):
    def __init__(self, examples: Iterable[PrefixExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PrefixExample:
        return self.examples[index]


class PrefixRiskCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[PrefixExample]) -> dict[str, Any]:
        texts = [example.text for example in examples]
        labels = torch.tensor([example.label for example in examples], dtype=torch.float32)
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch
