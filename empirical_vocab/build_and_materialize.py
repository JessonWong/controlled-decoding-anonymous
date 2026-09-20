"""Build Claude's empirical vocabulary V and materialize a V-space fusion cache.

Reads a raw completion-text-counter cache (the ``*_with_..._raw_responses`` dir
whose payloads carry ``sampled_completion_text_counts`` + ``answer_text``) and
emits a ``proxy_dirichlet_v1`` cache indexed in V instead of the proxy vocab.

Why this is the whole point of Solution A:
  * MC counts become EXACT -- every sample is one V coordinate, so the ~12%
    first-token truncation of the proxy path disappears.
  * every training label is representable in V (answer tokens are added to V).
  * the proxy is projected onto V (first-token log-prob, cut 1) and fused with
    the exact counts by the SAME Dirichlet posterior the proxy path uses.

The output payloads are drop-in for ``training/train_biasnet.py`` (vocab-agnostic
``vocab_size = log_probs.shape[2]``) with ``--input_projection_mode count_sketch``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (ROOT, os.path.join(ROOT, "training")):
    if path not in sys.path:
        sys.path.insert(0, path)

from mc_reconstruction import fuse_proxy_logits_with_mc_counts  # noqa: E402
from empirical_vocab.vspace import (  # noqa: E402
    VSpaceProxy,
    Vocab,
    build_vocab,
    counter_to_vcounts,
)

PROXY_MC_FUSION_MODE = "proxy_dirichlet_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", required=True, help="Raw text-counter cache dir.")
    p.add_argument("--output_dir", required=True, help="V-space cache output dir.")
    p.add_argument("--vocab_out", required=True, help="Where to write vocab.json.")
    p.add_argument(
        "--label_tokenizer",
        required=True,
        help="Tokenizer used to decode answer labels into V strings "
        "(must be the cache's own tokenizer, e.g. HuggingFaceTB/SmolLM2-360M).",
    )
    p.add_argument("--proxy_model", default=None, help="Local proxy model path/id.")
    p.add_argument("--proxy_tokenizer", default=None, help="Proxy tokenizer (defaults to proxy_model).")
    p.add_argument("--proxy_revision", default="vspace_proxy_v1")
    p.add_argument("--proxy_dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--prior_strength", type=float, default=2.0)
    p.add_argument("--max_vocab", type=int, default=None, help="Cap |V| (labels always kept).")
    p.add_argument("--min_count", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--store_dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--limit", type=int, default=None, help="Only process first N records.")
    p.add_argument(
        "--no_proxy",
        action="store_true",
        help="CPU smoke test: skip the proxy, use a uniform prior (q = softmax(0)).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = sorted(input_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files in {input_dir}.")
    if args.limit is not None:
        files = files[: args.limit]
    existing = sorted(output_dir.glob("*.pt")) if output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(f"{output_dir} has {len(existing)} .pt; pass --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    label_tok = AutoTokenizer.from_pretrained(args.label_tokenizer, use_fast=False)

    print(f"Loading {len(files)} raw records ...", flush=True)
    payloads = [torch.load(f, map_location="cpu", weights_only=False) for f in files]

    # ---- Pass 1: build V from every counter + every answer label string -------
    all_counters: list[dict[str, int]] = []
    all_label_strings: list[str] = []
    per_record_labels: list[list[str]] = []
    for payload in payloads:
        counters = payload["sampled_completion_text_counts"]
        labels = payload["labels"][0].tolist()
        all_counters.extend(counters)
        label_strings = [label_tok.decode([int(t)], skip_special_tokens=False) for t in labels]
        per_record_labels.append(label_strings)
        all_label_strings.extend(label_strings)

    vocab = build_vocab(
        all_counters,
        all_label_strings,
        max_size=args.max_vocab,
        min_count=args.min_count,
    )
    vocab.save(args.vocab_out)
    print(f"|V| = {vocab.size} (eos={vocab.eos_id} oov={vocab.oov_id})", flush=True)

    # Coverage diagnostics: what fraction of raw sampled mass lands on a real V id.
    total_mass = 0
    covered_mass = 0
    for counter in all_counters:
        for text, count in counter.items():
            total_mass += count
            if text == "" or text in vocab.string_to_id:
                covered_mass += count
    label_in_v = sum(
        1 for ls in all_label_strings if ls == "" or ls in vocab.string_to_id
    )
    print(
        f"coverage: {covered_mass}/{total_mass} = {covered_mass/max(total_mass,1):.4f} of "
        f"sampled mass in V; labels_in_V={label_in_v}/{len(all_label_strings)}",
        flush=True,
    )

    # ---- Proxy -----------------------------------------------------------------
    proxy: VSpaceProxy | None = None
    device = torch.device(args.device if not args.no_proxy else "cpu")
    if not args.no_proxy:
        from transformers import AutoModelForCausalLM

        proxy_tok_name = args.proxy_tokenizer or args.proxy_model
        proxy_tok = AutoTokenizer.from_pretrained(proxy_tok_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.proxy_model, torch_dtype=resolve_dtype(args.proxy_dtype)
        ).to(device).eval()
        proxy = VSpaceProxy(model, proxy_tok, device)

    store_dtype = resolve_dtype(args.store_dtype)

    # ---- Pass 2: materialize each record in V-space ----------------------------
    summaries = []
    for rec_index, (path, payload) in enumerate(zip(files, payloads)):
        counters = payload["sampled_completion_text_counts"]
        label_strings = per_record_labels[rec_index]
        seq = len(counters)
        prompt = payload.get("prompt_text", "")
        prefixes = payload["sampled_prefix_texts"]

        counts_V = torch.stack(
            [counter_to_vcounts(dict(c), vocab) for c in counters], dim=0
        )  # [seq, |V|] long
        labels_V = torch.tensor(
            [vocab.encode(ls) for ls in label_strings], dtype=torch.long
        )  # [seq]
        valid = counts_V.sum(dim=-1)  # [seq] -- exact, matches counts by construction

        if proxy is not None:
            q_rows = [
                proxy.vspace_logits(prompt, str(prefixes[i]), vocab)
                for i in range(seq)
            ]
            proxy_logits = torch.stack(q_rows, dim=0).to(device)  # [seq, |V|]
        else:
            proxy_logits = torch.zeros((seq, vocab.size), dtype=torch.float32)

        fused = fuse_proxy_logits_with_mc_counts(
            proxy_logits,
            counts_V.to(proxy_logits.device),
            temperature=args.temperature,
            prior_strength=args.prior_strength,
            dtype=store_dtype,
        )  # [seq, |V|] log-probs

        src_meta = dict(payload.get("metadata") or {})
        out_meta = {
            **src_meta,
            "mc_fusion_mode": PROXY_MC_FUSION_MODE,
            "proxy_model_name_or_path": str(args.proxy_model or "uniform_smoke"),
            "proxy_model_revision": str(args.proxy_revision),
            "proxy_tokenizer_sha256": _sha(str(args.proxy_tokenizer or args.proxy_model or "uniform")),
            "proxy_temperature": float(args.temperature),
            "proxy_prior_strength": float(args.prior_strength),
            "proxy_chat_template_protocol": "vspace_first_token_v1",
            "shared_vocab_size": int(vocab.size),
            "proxy_vocab_size": int(vocab.size),
            "proxy_vocab_tail_policy": "vspace_no_tail",
            "proxy_dtype": str(args.proxy_dtype),
            "proxy_quantization": "none",
            "vspace_vocab_size": int(vocab.size),
            "vspace_eos_id": int(vocab.eos_id),
            "vspace_oov_id": int(vocab.oov_id),
            # observed_alpha/floor_mass/sample_temperature/top_p/sample_completion_policy
            # are inherited from src_meta and satisfy the trainer's MC checks.
        }

        out = {
            "log_probs": fused.unsqueeze(0).cpu(),
            "mc_counts": counts_V.unsqueeze(0).cpu().to(torch.long),
            "labels": labels_V.unsqueeze(0).cpu(),
            "valid_sample_counts": valid.unsqueeze(0).cpu().to(torch.long),
            "prompt_text": prompt,
            "answer_text": payload.get("answer_text", ""),
            "sampled_prefix_texts": prefixes,
            "sampled_completion_text_counts": counters,
            "metadata": out_meta,
        }
        dest = output_dir / path.name
        tmp = output_dir / f".{path.name}.tmp"
        torch.save(out, tmp)
        os.replace(tmp, dest)

        multi = int((labels_V == vocab.oov_id).sum())
        summaries.append(
            {
                "record": path.name,
                "seq": seq,
                "valid_min": int(valid.min()),
                "valid_max": int(valid.max()),
                "labels_oov": multi,
                "fused_finite": bool(torch.isfinite(fused).all()),
            }
        )
        print(
            f"[{rec_index+1}/{len(files)}] {path.name} seq={seq} "
            f"valid[{int(valid.min())},{int(valid.max())}] labels_oov={multi} "
            f"fused_finite={bool(torch.isfinite(fused).all())}",
            flush=True,
        )

    manifest = {
        "schema": "vspace_proxy_dirichlet_v1",
        "input_dir": str(input_dir.resolve()),
        "vocab_size": vocab.size,
        "records": len(summaries),
        "temperature": args.temperature,
        "prior_strength": args.prior_strength,
        "proxy_model": args.proxy_model,
        "no_proxy": args.no_proxy,
        "sampled_mass_coverage": covered_mass / max(total_mass, 1),
        "labels_in_v": label_in_v / max(len(all_label_strings), 1),
        "summaries": summaries,
    }
    (output_dir / "vspace_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("MANIFEST:", json.dumps({k: v for k, v in manifest.items() if k != "summaries"}))


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
