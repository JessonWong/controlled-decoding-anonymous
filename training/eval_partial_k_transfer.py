"""Evaluate zero-training partial-K transfer of a fixed-MC50 BiasNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling_biasnet import BiasNet
from training.anytime_proxy_mc import load_anytime_cache, parse_budgets
from training.eval_anytime_stopping import (
    _fixed_budget_results,
    _row_indices_for_records,
    evaluate_replay_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="0,4,8,16,32,50")
    parser.add_argument("--replays", type=int, default=16)
    parser.add_argument("--replay_seed", type=int, default=20260825)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    cache = load_anytime_cache(args.cache_dir)
    budgets = parse_budgets(args.budgets, max_samples=cache.max_samples)
    split_config = json.loads(
        (Path(args.split_checkpoint) / "config.json").read_text(encoding="utf-8")
    )
    splits = {
        "historical_train": split_config["anytime_train_files"],
        "legacy_validation": split_config["anytime_test_files"],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiasNet.from_pretrained(args.checkpoint, map_location="cpu").to(device)
    model.set_up_proj()
    model.eval()
    train_indices = _row_indices_for_records(cache, splits["historical_train"])
    position_normalizer = int(cache.position_ids[train_indices].max().item())
    results = {}
    for split_name, files in splits.items():
        row_indices = _row_indices_for_records(cache, files)
        paths = evaluate_replay_paths(
            cache=cache,
            row_indices=row_indices,
            model=model,
            budgets=budgets,
            replays=args.replays,
            replay_seed=args.replay_seed,
            batch_size=args.batch_size,
            device=device,
            position_normalizer=position_normalizer,
        )
        full_actions = paths["actions"][:, :, -1]
        if not np.all(full_actions == full_actions[:1]):
            raise RuntimeError("Full-count actions changed across replay permutations.")
        paths["reference_actions"] = full_actions[0]
        results[split_name] = {
            "records": len(files),
            "rows": int(row_indices.numel()),
            "fixed_budget_results": _fixed_budget_results(paths, budgets),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "fixed_mc50_checkpoint_partial_k_transfer",
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "budgets": list(budgets),
                "replays": args.replays,
                "replay_seed": args.replay_seed,
                "position_normalizer": position_normalizer,
                "splits": results,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": "saved", "path": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
