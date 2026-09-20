"""Qwen vocabulary augmented with fixed-capacity empirical string actions.

The black-box API returns continuation *strings*.  If a returned string maps to
exactly one Qwen token, that native Qwen id is retained.  Strings that map to
zero or multiple Qwen tokens are assigned deterministic ids in a fixed-size
extension region.  The extension ids are output actions only: callers should
decode them through this sidecar vocabulary, append the resulting text, and
re-tokenize the new context with the unchanged Qwen tokenizer.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "hybrid_qwen_extension_v1"


@dataclass(frozen=True)
class HybridQwenVocab:
    """Fixed Qwen base coordinates plus a partially occupied extension tail."""

    base_vocab_size: int
    extension_capacity: int
    extension_strings: tuple[str, ...]
    eos_token_id: int

    def __post_init__(self) -> None:
        if self.base_vocab_size <= 1:
            raise ValueError("base_vocab_size must be greater than one.")
        if self.extension_capacity <= 0:
            raise ValueError("extension_capacity must be positive.")
        if len(self.extension_strings) > self.extension_capacity:
            raise ValueError("Occupied extension strings exceed extension capacity.")
        if len(set(self.extension_strings)) != len(self.extension_strings):
            raise ValueError("Extension strings must be unique.")
        if any(text == "" for text in self.extension_strings):
            raise ValueError("The empty string is represented by EOS, not an extension.")
        if not 0 <= self.eos_token_id < self.base_vocab_size:
            raise ValueError("eos_token_id must be inside the Qwen base vocabulary.")

    @property
    def extension_start_id(self) -> int:
        return self.base_vocab_size

    @property
    def extension_count(self) -> int:
        return len(self.extension_strings)

    @property
    def oov_id(self) -> int:
        return self.base_vocab_size + self.extension_capacity

    @property
    def size(self) -> int:
        return self.oov_id + 1

    @property
    def string_to_extension_id(self) -> dict[str, int]:
        return {
            text: self.extension_start_id + offset
            for offset, text in enumerate(self.extension_strings)
        }

    def encode_action(self, text: str, tokenizer) -> int:
        """Map one returned API string into the hybrid action coordinates."""

        if text == "":
            return self.eos_token_id
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 1:
            token_id = int(token_ids[0])
            if not 0 <= token_id < self.base_vocab_size:
                raise ValueError(
                    f"Tokenizer emitted id {token_id} outside base vocabulary "
                    f"[0, {self.base_vocab_size})."
                )
            return token_id
        return self.string_to_extension_id.get(text, self.oov_id)

    def decode_action(self, action_id: int, tokenizer) -> str | None:
        """Decode an action; ``None`` denotes an unused extension or OOV slot."""

        action_id = int(action_id)
        if 0 <= action_id < self.base_vocab_size:
            return tokenizer.decode([action_id], skip_special_tokens=False)
        offset = action_id - self.extension_start_id
        if 0 <= offset < self.extension_count:
            return self.extension_strings[offset]
        if self.extension_count <= offset < self.extension_capacity:
            return None
        if action_id == self.oov_id:
            return None
        raise ValueError(f"Action id {action_id} outside hybrid vocabulary.")

    def to_json(
        self,
        *,
        tokenizer_name_or_path: str,
        extension_frequencies: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        frequencies = extension_frequencies or {}
        return {
            "schema": SCHEMA,
            "tokenizer_name_or_path": tokenizer_name_or_path,
            "base_vocab_size": self.base_vocab_size,
            "extension_start_id": self.extension_start_id,
            "extension_capacity": self.extension_capacity,
            "extension_count": self.extension_count,
            "unused_extension_slots": self.extension_capacity - self.extension_count,
            "eos_token_id": self.eos_token_id,
            "oov_id": self.oov_id,
            "total_vocab_size": self.size,
            "extensions": [
                {
                    "id": self.extension_start_id + offset,
                    "text": text,
                    "training_count": int(frequencies.get(text, 0)),
                }
                for offset, text in enumerate(self.extension_strings)
            ],
        }

    def save(
        self,
        path: str | Path,
        *,
        tokenizer_name_or_path: str,
        extension_frequencies: dict[str, int] | None = None,
    ) -> None:
        Path(path).write_text(
            json.dumps(
                self.to_json(
                    tokenizer_name_or_path=tokenizer_name_or_path,
                    extension_frequencies=extension_frequencies,
                ),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "HybridQwenVocab":
        """Load and strictly validate a serialized hybrid vocabulary."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise ValueError(
                f"Unsupported hybrid vocabulary schema {payload.get('schema')!r}; "
                f"expected {SCHEMA!r}."
            )
        extensions = payload.get("extensions")
        if not isinstance(extensions, list):
            raise ValueError("Hybrid vocabulary extensions must be a list.")
        base_vocab_size = int(payload["base_vocab_size"])
        extension_capacity = int(payload["extension_capacity"])
        expected_ids = list(
            range(base_vocab_size, base_vocab_size + len(extensions))
        )
        actual_ids = [int(item["id"]) for item in extensions]
        if actual_ids != expected_ids:
            raise ValueError("Hybrid extension ids must be contiguous and ordered.")
        vocab = cls(
            base_vocab_size=base_vocab_size,
            extension_capacity=extension_capacity,
            extension_strings=tuple(str(item["text"]) for item in extensions),
            eos_token_id=int(payload["eos_token_id"]),
        )
        serialized_contract = {
            "extension_start_id": vocab.extension_start_id,
            "extension_count": vocab.extension_count,
            "unused_extension_slots": (
                vocab.extension_capacity - vocab.extension_count
            ),
            "oov_id": vocab.oov_id,
            "total_vocab_size": vocab.size,
        }
        for key, expected in serialized_contract.items():
            if int(payload.get(key, -1)) != expected:
                raise ValueError(
                    f"Hybrid vocabulary {key}={payload.get(key)!r} does not "
                    f"match the derived value {expected}."
                )
        return vocab

    def valid_action_mask(self, tokenizer, *, device=None):
        """Return the actions that can be emitted by full-vocabulary decoding.

        Unoccupied extension slots, OOV, and non-EOS tokenizer special tokens
        have no faithful output action and are therefore always excluded.
        """

        import torch

        mask = torch.zeros(self.size, dtype=torch.bool, device=device)
        mask[: self.base_vocab_size] = True
        if self.extension_count:
            mask[
                self.extension_start_id : self.extension_start_id
                + self.extension_count
            ] = True
        for token_id in getattr(tokenizer, "all_special_ids", ()):
            token_id = int(token_id)
            if token_id != self.eos_token_id and 0 <= token_id < self.base_vocab_size:
                mask[token_id] = False
        mask[self.eos_token_id] = True
        mask[self.oov_id] = False
        return mask


def empirical_extension_frequencies(
    completion_counters: Iterable[dict[str, int]], tokenizer
) -> Counter[str]:
    """Count only non-empty strings that are not one native Qwen token."""

    frequencies: Counter[str] = Counter()
    for counter in completion_counters:
        for text, count in counter.items():
            count = int(count)
            if count < 0:
                raise ValueError("Completion counter values must be non-negative.")
            if text == "" or count == 0:
                continue
            if len(tokenizer.encode(text, add_special_tokens=False)) != 1:
                frequencies[text] += count
    return frequencies


def build_hybrid_qwen_vocab(
    completion_counters: Iterable[dict[str, int]],
    tokenizer,
    *,
    extension_capacity: int,
) -> tuple[HybridQwenVocab, Counter[str]]:
    """Build a deterministic frequency-ranked extension dictionary."""

    if extension_capacity <= 0:
        raise ValueError("extension_capacity must be positive.")
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("The base tokenizer must define eos_token_id.")
    frequencies = empirical_extension_frequencies(completion_counters, tokenizer)
    ranked = sorted(frequencies, key=lambda text: (-frequencies[text], text))
    vocab = HybridQwenVocab(
        base_vocab_size=len(tokenizer),
        extension_capacity=int(extension_capacity),
        extension_strings=tuple(ranked[:extension_capacity]),
        eos_token_id=int(eos_token_id),
    )
    return vocab, frequencies


def counter_to_hybrid_counts(
    counter: dict[str, int], vocab: HybridQwenVocab, tokenizer
) -> Counter[int]:
    """Map a text counter to sparse hybrid ids while preserving total mass."""

    counts: Counter[int] = Counter()
    for text, count in counter.items():
        count = int(count)
        if count < 0:
            raise ValueError("Completion counter values must be non-negative.")
        counts[vocab.encode_action(text, tokenizer)] += count
    if sum(counts.values()) != sum(int(value) for value in counter.values()):
        raise AssertionError("Hybrid action mapping did not preserve sample mass.")
    return counts
