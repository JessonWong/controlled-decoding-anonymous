"""Build no-proxy MC caches with a fixed Dirichlet prior.

The raw MC50 cache only assigns one shared floor probability to every token
that was not sampled.  This utility keeps the sampled counts but replaces that
floor distribution with either a uniform prior or a static unigram prior
estimated from the training cache.  It also performs a small, label-free
held-out MC-event NLL selection of the prior strength before materialising the
cache used by BiasNet.

The source cache is deliberately kept separate from the counts cache.  The
current raw cache is compact and legacy-compatible, while the matching fused
cache already contains exact ``mc_counts`` tensors for the same filenames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mc_reconstruction import fuse_proxy_logits_with_mc_counts
from training.eval_proxy_mc_fusion import deterministic_binomial_split


DEFAULT_KAPPAS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
DEFAULT_GLOBAL_SMOOTHING = (0.001, 0.01, 0.05, 0.1)
STATIC_MODES = ("uniform_dirichlet_v1", "global_unigram_dirichlet_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--counts_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=STATIC_MODES, required=True)
    parser.add_argument("--held_out_file_count", type=int, default=8)
    parser.add_argument("--held_out_split_seed", default="qwen3-dual-rank-v1")
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--split_seeds", default="0,1,2,3,4")
    parser.add_argument("--kappas", default=",".join(str(x) for x in DEFAULT_KAPPAS))
    parser.add_argument(
        "--global_smoothing",
        default=",".join(str(x) for x in DEFAULT_GLOBAL_SMOOTHING),
        help="Uniform mixture fractions used to keep the static unigram prior nonzero.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def held_out_names(names: list[str], count: int, seed: str) -> list[str]:
    if count <= 0 or count >= len(names):
        raise ValueError("held_out_file_count must leave at least one training file")
    scored = sorted(
        names,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest(),
    )
    return sorted(scored[:count])


def parse_floats(value: str, name: str) -> list[float]:
    try:
        result = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {value!r}") from exc
    if not result or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique values")
    if any(not math.isfinite(x) or x <= 0 for x in result):
        raise ValueError(f"{name} must contain finite positive values")
    return result


def load_sparse_counts(counts_dir: Path, names: list[str]) -> tuple[int, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    vocab_size: int | None = None
    for name in names:
        path = counts_dir / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        counts = payload.get("mc_counts")
        if not isinstance(counts, torch.Tensor):
            raise ValueError(f"{path} has no mc_counts tensor")
        if counts.dim() != 3 or counts.shape[0] != 1:
            raise ValueError(f"{path} mc_counts must have shape [1, rows, vocab]")
        if counts.dtype == torch.bool or counts.is_floating_point() or (counts < 0).any():
            raise ValueError(f"{path} mc_counts must be non-negative integers")
        if vocab_size is None:
            vocab_size = int(counts.shape[-1])
        elif int(counts.shape[-1]) != vocab_size:
            raise ValueError("All counts caches must share one vocabulary size")
        rows = []
        for row in counts[0].long().cpu():
            ids = torch.nonzero(row, as_tuple=False).flatten()
            rows.append((ids, row[ids].clone()))
        records.append(
            {
                "name": name,
                "rows": rows,
                "file_sum": counts[0].long().sum(dim=0, dtype=torch.float64),
            }
        )
        del payload, counts
    if vocab_size is None:
        raise ValueError("No count records found")
    return vocab_size, records


def evaluate_candidate(
    records: list[dict[str, Any]],
    eval_names: set[str],
    vocab_size: int,
    mode: str,
    kappa: float,
    split_seeds: list[int],
    fraction: float,
    global_log_probs: torch.Tensor | None,
) -> dict[str, float]:
    total_nll = 0.0
    total_events = 0
    unseen_nll = 0.0
    unseen_events = 0
    for record in records:
        if record["name"] not in eval_names:
            continue
        for row_index, (ids, counts) in enumerate(record["rows"]):
            for seed in split_seeds:
                calibration, heldout = deterministic_binomial_split(
                    counts,
                    fraction,
                    seed=seed,
                    row_key=f"{record['name']}:{row_index}",
                )
                calibration_total = int(calibration.sum().item())
                if calibration_total <= 0:
                    continue
                for item in torch.nonzero(heldout, as_tuple=False).flatten().tolist():
                    token_id = int(ids[item])
                    held_count = int(heldout[item])
                    count = int(calibration[item])
                    if mode == "uniform_dirichlet_v1":
                        q = 1.0 / vocab_size
                    else:
                        assert global_log_probs is not None
                        q = float(global_log_probs[token_id].exp().item())
                    probability = (count + kappa * q) / (calibration_total + kappa)
                    nll = -math.log(max(probability, 1e-300))
                    total_nll += held_count * nll
                    total_events += held_count
                    if count == 0:
                        unseen_nll += held_count * nll
                        unseen_events += held_count
    if total_events <= 0:
        raise ValueError("No held-out MC events were available for selection")
    return {
        "mean_nll": total_nll / total_events,
        "perplexity": math.exp(total_nll / total_events),
        "unseen_mean_nll": unseen_nll / unseen_events if unseen_events else float("nan"),
        "events": float(total_events),
        "unseen_events": float(unseen_events),
    }


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    counts_dir = Path(args.counts_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not raw_dir.is_dir() or not counts_dir.is_dir():
        raise FileNotFoundError("raw_dir and counts_dir must be directories")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in raw_dir.glob("*.pt"))
    count_names = sorted(path.name for path in counts_dir.glob("*.pt"))
    if names != count_names:
        raise ValueError("raw_dir and counts_dir must contain matching .pt filenames")
    held_names = held_out_names(names, args.held_out_file_count, args.held_out_split_seed)
    train_names = [name for name in names if name not in set(held_names)]
    kappas = parse_floats(args.kappas, "kappas")
    split_seeds = [int(part.strip()) for part in args.split_seeds.split(",") if part.strip()]
    if not split_seeds:
        raise ValueError("split_seeds must not be empty")
    smoothings = parse_floats(args.global_smoothing, "global_smoothing")
    vocab_size, records = load_sparse_counts(counts_dir, names)

    global_log_probs = None
    selected_smoothing = None
    if args.mode == "global_unigram_dirichlet_v1":
        train_set = set(train_names)
        global_counts = torch.zeros(vocab_size, dtype=torch.float64)
        for record in records:
            if record["name"] in train_set:
                global_counts += record["file_sum"]
        total = float(global_counts.sum().item())
        if total <= 0:
            raise ValueError("Training records contain no MC counts")
        candidates = []
        for smoothing in smoothings:
            q = (1.0 - smoothing) * (global_counts / total) + smoothing / vocab_size
            candidates.append((smoothing, q.log()))
    else:
        candidates = [(None, None)]

    selection_rows = []
    best = None
    for smoothing, prior_log_probs in candidates:
        for kappa in kappas:
            metrics = evaluate_candidate(
                records,
                set(held_names),
                vocab_size,
                args.mode,
                kappa,
                split_seeds,
                float(args.calibration_fraction),
                prior_log_probs,
            )
            row = {"smoothing": smoothing, "kappa": kappa, **metrics}
            selection_rows.append(row)
            key = (metrics["mean_nll"], kappa, smoothing or 0.0)
            if best is None or key < best[0]:
                best = (key, row, prior_log_probs)
    assert best is not None
    selected = best[1]
    global_prior_path = None
    global_prior_sha = None
    if args.mode == "global_unigram_dirichlet_v1":
        assert best[2] is not None
        global_prior_path = output_dir / "global_unigram_prior.pt"
        torch.save(
            {
                "log_probs": best[2].to(torch.float32),
                "vocab_size": vocab_size,
                "mode": args.mode,
                "smoothing": selected["smoothing"],
                "source_files": train_names,
            },
            global_prior_path,
        )
        global_prior_sha = sha256_file(global_prior_path)
        global_log_probs = best[2]

    prior_logits = (
        torch.zeros(vocab_size, dtype=torch.float32)
        if args.mode == "uniform_dirichlet_v1"
        else global_log_probs.to(torch.float32)
    )
    selected_kappa = float(selected["kappa"])
    for record in records:
        raw_path = raw_dir / record["name"]
        raw_payload = torch.load(raw_path, map_location="cpu", weights_only=False)
        source = raw_payload.get("log_probs")
        if not isinstance(source, torch.Tensor) or source.dim() != 3 or source.shape[0] != 1:
            raise ValueError(f"{raw_path} has invalid log_probs")
        output_rows = []
        for ids, counts in record["rows"]:
            dense_counts = torch.zeros(vocab_size, dtype=torch.long)
            dense_counts[ids] = counts
            fused = fuse_proxy_logits_with_mc_counts(
                prior_logits.unsqueeze(0),
                dense_counts.unsqueeze(0),
                temperature=1.0,
                prior_strength=selected_kappa,
                dtype=torch.float32,
            )[0]
            output_rows.append(fused)
        transformed = torch.stack(output_rows).unsqueeze(0).to(source.dtype)
        if transformed.shape != source.shape:
            raise ValueError(f"Transformed shape mismatch for {raw_path}")
        raw_payload["log_probs"] = transformed
        metadata = dict(raw_payload.get("metadata") or {})
        metadata["mc_fusion_mode"] = None
        metadata["mc_static_prior_mode"] = args.mode
        metadata["mc_static_prior_strength"] = selected_kappa
        metadata["mc_static_prior_path"] = str(global_prior_path) if global_prior_path else None
        metadata["mc_static_prior_sha256"] = global_prior_sha
        metadata["mc_static_prior_smoothing"] = selected.get("smoothing")
        metadata["mc_static_prior_selection"] = "heldout_mc_event_nll"
        raw_payload["metadata"] = metadata
        torch.save(raw_payload, output_dir / record["name"])
        del raw_payload, source, transformed

    manifest = {
        "schema_version": 1,
        "mode": args.mode,
        "raw_dir": str(raw_dir),
        "counts_dir": str(counts_dir),
        "output_dir": str(output_dir),
        "vocab_size": vocab_size,
        "records": len(names),
        "held_out_files": held_names,
        "train_files": train_names,
        "split_seeds": split_seeds,
        "calibration_fraction": float(args.calibration_fraction),
        "selected": selected,
        "selection_grid": selection_rows,
        "global_prior_path": str(global_prior_path) if global_prior_path else None,
        "global_prior_sha256": global_prior_sha,
    }
    (output_dir / "static_prior_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
