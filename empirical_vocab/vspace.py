"""Empirical-vocabulary (V-space) primitives for Solution A.

Motivation
----------
The black-box target (Claude Haiku 4.5) returns only TEXT.  The legacy pipeline
re-tokenizes each returned string with a *proxy* tokenizer and keeps the FIRST
proxy token id (``content_to_token_id`` -> ``token_ids[0]``), which truncates
~12% of Claude's multi-token returns onto a prefix coordinate the target never
emitted alone.  Both the MC counts and the emitted action are then corrupted.

Solution A replaces the proxy vocabulary with Claude's OWN empirical vocabulary
``V`` = the set of continuation strings Claude actually returned (harvested from
the raw ``sampled_completion_text_counts`` counters), unioned with the reference
answer's per-token strings so every training label is representable in ``V``.

In ``V`` every ``max_tokens=1`` sample is exactly one coordinate, so the MC
counts become EXACT (no truncation, no dead positions) and the emitted action is
always a real Claude string.  Only the proxy PRIOR must be projected onto ``V``;
that projection is a computation we fully control locally.

This module is deliberately dependency-light: numpy/torch + a HF tokenizer/model.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch


# Reserved sentinel strings that must never collide with a real Claude return.
# The raw sampler records a canonical end-of-stream event as the empty string.
EOS_STRING = "\x00<VSPACE_EOS>"
OOV_STRING = "\x00<VSPACE_OOV>"


@dataclass
class Vocab:
    """An ordered empirical vocabulary with O(1) string<->id lookup."""

    id_to_string: list[str]
    string_to_id: dict[str, int]
    eos_id: int
    oov_id: int

    @property
    def size(self) -> int:
        return len(self.id_to_string)

    def encode(self, text: str) -> int:
        """Map a returned continuation string to its V coordinate.

        The empty string is the sampler's canonical EOS event.  Anything not in
        ``V`` (only possible at inference on held-out prompts) maps to OOV.
        """

        if text == "":
            return self.eos_id
        return self.string_to_id.get(text, self.oov_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "id_to_string": self.id_to_string,
            "eos_id": self.eos_id,
            "oov_id": self.oov_id,
            "size": self.size,
            "eos_string": EOS_STRING,
            "oov_string": OOV_STRING,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json(), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        id_to_string = list(payload["id_to_string"])
        string_to_id = {s: i for i, s in enumerate(id_to_string)}
        return cls(
            id_to_string=id_to_string,
            string_to_id=string_to_id,
            eos_id=int(payload["eos_id"]),
            oov_id=int(payload["oov_id"]),
        )


def build_vocab(
    completion_counters: Iterable[dict[str, int]],
    label_strings: Iterable[str],
    *,
    max_size: Optional[int] = None,
    min_count: int = 1,
) -> Vocab:
    """Build ``V`` from observed continuation counters and answer label strings.

    ``V`` always contains every ``label_strings`` entry (so no training label is
    unrepresentable) plus the most frequent observed continuations up to
    ``max_size``.  Two reserved coordinates (EOS, OOV) are appended last.

    Ordering is deterministic: labels first (in first-seen order), then observed
    strings by descending frequency with the string as a tie-breaker, then the
    two sentinels.  A fixed order keeps the count-sketch input hash reproducible.
    """

    freq: Counter[str] = Counter()
    for counter in completion_counters:
        for text, count in counter.items():
            if text == "":
                continue  # EOS handled by the reserved sentinel
            freq[text] += int(count)

    ordered: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        if text in ("", EOS_STRING, OOV_STRING) or text in seen:
            return
        seen.add(text)
        ordered.append(text)

    for text in label_strings:
        _add(text)

    remaining = [
        (text, count)
        for text, count in freq.items()
        if text not in seen and count >= int(min_count)
    ]
    remaining.sort(key=lambda kv: (-kv[1], kv[0]))
    budget = None if max_size is None else max_size - len(ordered) - 2
    if budget is not None and budget < 0:
        budget = 0
    for text, _ in (remaining if budget is None else remaining[:budget]):
        _add(text)

    ordered.append(EOS_STRING)
    eos_id = len(ordered) - 1
    ordered.append(OOV_STRING)
    oov_id = len(ordered) - 1

    string_to_id = {s: i for i, s in enumerate(ordered)}
    return Vocab(
        id_to_string=ordered,
        string_to_id=string_to_id,
        eos_id=eos_id,
        oov_id=oov_id,
    )


def counter_to_vcounts(counter: dict[str, int], vocab: Vocab) -> torch.Tensor:
    """Exact integer counts over ``V`` for one sampled position.

    Every sample lands on exactly one coordinate (its own string, EOS for the
    empty string, or OOV for an unseen held-out string).  The total is preserved,
    so ``counts.sum() == sum(counter.values())`` always holds -- this is what the
    trainer validates against ``valid_sample_counts``.
    """

    counts = torch.zeros(vocab.size, dtype=torch.long)
    for text, count in counter.items():
        counts[vocab.encode(text)] += int(count)
    return counts


class VSpaceProxy:
    """Projects a local proxy language model onto the empirical vocabulary ``V``.

    ``vspace_logits`` returns one score per ``V`` coordinate to be consumed as
    the ``proxy_logits`` argument of ``fuse_proxy_logits_with_mc_counts`` (the
    fuse step re-applies temperature and a softmax over ``V``).

    Cut 1 uses a FIRST-proxy-token approximation: ``score(s) = logP(first proxy
    token of s | prefix)``.  This is O(one proxy forward per position), matches
    the information the legacy pipeline already relied on, and is a strict
    improvement here only because the COUNTS and the ACTION space are now exact.
    A later cut can swap in a full-string teacher-forced probability without
    touching callers.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        *,
        floor_logit: float = -30.0,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.floor_logit = float(floor_logit)
        # First proxy token id per V coordinate, precomputed once and reused
        # across all positions.  -1 marks an unmappable coordinate (OOV, or a
        # string the proxy tokenizer encodes to nothing).
        self._first_token_ids: Optional[torch.Tensor] = None
        self._first_token_vocab_size: Optional[int] = None

    def _ensure_first_token_ids(self, vocab: Vocab) -> torch.Tensor:
        if (
            self._first_token_ids is not None
            and self._first_token_vocab_size == vocab.size
        ):
            return self._first_token_ids
        eos = getattr(self.tokenizer, "eos_token_id", None)
        ids: list[int] = []
        for text in vocab.id_to_string:
            if text == EOS_STRING:
                ids.append(int(eos) if eos is not None else -1)
            elif text == OOV_STRING:
                ids.append(-1)
            else:
                encoded = self.tokenizer.encode(text, add_special_tokens=False)
                ids.append(int(encoded[0]) if encoded else -1)
        tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
        self._first_token_ids = tensor
        self._first_token_vocab_size = vocab.size
        return tensor

    def _render_input_ids(self, prompt: str, prefix_text: str) -> torch.Tensor:
        """Render (user prompt, assistant prefix) the way the target was sampled."""

        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            base = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            base = prompt
        text = base + prefix_text
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def vspace_logits(self, prompt: str, prefix_text: str, vocab: Vocab) -> torch.Tensor:
        """Return a ``[V]`` float tensor of first-token proxy log-probs."""

        first_token_ids = self._ensure_first_token_ids(vocab)
        input_ids = self._render_input_ids(prompt, prefix_text)
        out = self.model(input_ids=input_ids)
        last_logits = out.logits[0, -1].float()
        log_probs = torch.log_softmax(last_logits, dim=-1)

        mappable = first_token_ids >= 0
        gather_ids = first_token_ids.clamp_min(0)
        scores = log_probs[gather_ids]
        scores = torch.where(
            mappable,
            scores,
            torch.full_like(scores, self.floor_logit),
        )
        return scores
