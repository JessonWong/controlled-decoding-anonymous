"""Project a CROSS-FAMILY proxy language model into the target's native vocabulary.

The Dirichlet fusion in :mod:`mc_reconstruction` needs a dense prior ``q`` over
the target's ``V`` coordinates.  The existing pipeline gets ``q`` from a proxy
that shares the target's tokenizer, so ``q`` is just the proxy's softmax.  That
makes the proxy same-family by construction, which confounds the claim that the
proxy only supplies a plausible dense distribution rather than a target prior.

This module removes that constraint with the same first-token approximation
``empirical_vocab.vspace.VSpaceProxy`` already uses for the empirical
vocabulary, applied to the target's FULL token vocabulary instead:

    score(target token t) = log P_proxy(first proxy token of decode(t) | prefix)

so any proxy, from any family, yields a dense ``[V_target]`` vector.  Collisions
(several target tokens sharing a first proxy token) and unmappable coordinates
are expected; ``fuse_proxy_logits_with_mc_counts`` re-applies a temperature and
a log-softmax, which renormalises over the target vocabulary.  That is the same
convention ``vspace_logits`` is consumed under, so the two lines stay comparable.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import torch

UNMAPPABLE = -1


def tokenizer_fingerprint(tokenizer) -> str:
    """Stable hash of a tokenizer's token->id map, matching the repo's convention."""

    vocab = tokenizer.get_vocab()
    payload = json.dumps(vocab, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_first_token_map(
    target_tokenizer,
    proxy_tokenizer,
    target_vocab_size: int,
    *,
    verbose: bool = True,
) -> torch.Tensor:
    """First proxy-token id for every target token id (``UNMAPPABLE`` when none).

    The target's EOS is mapped to the proxy's EOS so the stop coordinate stays a
    stop coordinate.  Every other special token is left unmappable: its surface
    form ("<|im_start|>") is Qwen-specific markup that no foreign proxy models.
    """

    target_special = set(target_tokenizer.all_special_ids or [])
    target_eos = target_tokenizer.eos_token_id
    proxy_eos = proxy_tokenizer.eos_token_id

    strings = target_tokenizer.batch_decode(
        [[i] for i in range(target_vocab_size)], skip_special_tokens=False
    )

    ids = torch.full((target_vocab_size,), UNMAPPABLE, dtype=torch.long)
    mapped = 0
    for index, text in enumerate(strings):
        if index == target_eos:
            if proxy_eos is not None:
                ids[index] = int(proxy_eos)
                mapped += 1
            continue
        if index in target_special or not text:
            continue
        encoded = proxy_tokenizer.encode(text, add_special_tokens=False)
        if encoded:
            ids[index] = int(encoded[0])
            mapped += 1
        if verbose and index % 20000 == 0 and index:
            print(f"  first_token_map {index}/{target_vocab_size}", flush=True)

    if verbose:
        distinct = int(torch.unique(ids[ids >= 0]).numel())
        print(
            f"first_token_map built: mapped={mapped}/{target_vocab_size} "
            f"({mapped / target_vocab_size:.4f}) distinct_proxy_ids={distinct} "
            f"collision_ratio={1.0 - distinct / max(mapped, 1):.4f}",
            flush=True,
        )
    return ids


def load_or_build_first_token_map(
    path: Optional[str],
    target_tokenizer,
    proxy_tokenizer,
    target_vocab_size: int,
    *,
    verbose: bool = True,
) -> torch.Tensor:
    """Cache the map on disk; rebuild whenever either tokenizer changes."""

    target_sha = tokenizer_fingerprint(target_tokenizer)
    proxy_sha = tokenizer_fingerprint(proxy_tokenizer)
    if path and os.path.exists(path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("target_tokenizer_sha256") == target_sha
            and payload.get("proxy_tokenizer_sha256") == proxy_sha
            and int(payload.get("target_vocab_size", -1)) == int(target_vocab_size)
        ):
            if verbose:
                print(f"first_token_map loaded from {path}", flush=True)
            return payload["first_token_ids"]
        if verbose:
            print(f"first_token_map at {path} is stale; rebuilding", flush=True)

    ids = build_first_token_map(
        target_tokenizer, proxy_tokenizer, target_vocab_size, verbose=verbose
    )
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "first_token_ids": ids,
                "target_tokenizer_sha256": target_sha,
                "proxy_tokenizer_sha256": proxy_sha,
                "target_vocab_size": int(target_vocab_size),
            },
            path,
        )
    return ids


def project_log_probs(
    proxy_log_probs: torch.Tensor,
    first_token_ids: torch.Tensor,
    *,
    floor_logit: float = -30.0,
) -> torch.Tensor:
    """Gather ``[..., V_proxy]`` proxy log-probs into ``[..., V_target]``."""

    ids = first_token_ids.to(proxy_log_probs.device)
    mappable = ids >= 0
    gathered = proxy_log_probs.index_select(-1, ids.clamp_min(0))
    return torch.where(mappable, gathered, torch.full_like(gathered, floor_logit))


class ForeignProxyRuntime:
    """A cross-family proxy that answers in the TARGET's coordinate system.

    The proxy is re-rendered from TEXT at every step (its own chat template plus
    the answer text produced so far), because a foreign tokenizer cannot consume
    the target's token ids.  That costs a full forward per step instead of an
    incremental one, which is immaterial next to a 32B target.
    """

    def __init__(
        self,
        model,
        tokenizer,
        first_token_ids: torch.Tensor,
        device: torch.device,
        *,
        floor_logit: float = -30.0,
        max_prefix_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.first_token_ids = first_token_ids.to(device)
        self.device = device
        self.floor_logit = float(floor_logit)
        self.max_prefix_tokens = int(max_prefix_tokens)

    def render(self, prompt: str, answer_prefix: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            base = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            base = prompt
        return base + answer_prefix

    @torch.no_grad()
    def next_token_log_probs(self, prompt: str, answer_prefix: str) -> torch.Tensor:
        """``[V_target]`` projected log-probs for the next target token."""

        text = self.render(prompt, answer_prefix)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > self.max_prefix_tokens:
            ids = ids[-self.max_prefix_tokens :]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids)
        log_probs = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        return project_log_probs(
            log_probs, self.first_token_ids, floor_logit=self.floor_logit
        )

    @torch.no_grad()
    def teacher_forced_log_probs(
        self,
        prompt: str,
        answer_prefixes: list[str],
        *,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """``[len(answer_prefixes), V_target]`` — one projected row per prefix.

        Each row is an independent forward on the exact prefix text, so no
        cross-tokenizer prefix-alignment approximation enters the cache.
        """

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        rows: list[torch.Tensor] = []
        for start in range(0, len(answer_prefixes), batch_size):
            chunk = answer_prefixes[start : start + batch_size]
            encoded = [
                self.tokenizer.encode(self.render(prompt, p), add_special_tokens=False)[
                    -self.max_prefix_tokens :
                ]
                for p in chunk
            ]
            width = max(len(e) for e in encoded)
            input_ids = torch.full(
                (len(encoded), width), pad_id, dtype=torch.long, device=self.device
            )
            attention = torch.zeros(
                (len(encoded), width), dtype=torch.long, device=self.device
            )
            last = []
            for row, seq in enumerate(encoded):
                input_ids[row, : len(seq)] = torch.tensor(seq, device=self.device)
                attention[row, : len(seq)] = 1
                last.append(len(seq) - 1)
            out = self.model(input_ids=input_ids, attention_mask=attention)
            picked = out.logits[torch.arange(len(encoded), device=self.device), torch.tensor(last, device=self.device)]
            log_probs = torch.log_softmax(picked.float(), dim=-1)
            rows.append(
                project_log_probs(
                    log_probs, self.first_token_ids, floor_logit=self.floor_logit
                ).cpu()
            )
        return torch.cat(rows, dim=0)
