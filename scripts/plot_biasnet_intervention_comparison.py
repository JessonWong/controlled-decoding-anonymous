#!/usr/bin/env python3
"""Combine cached BiasNet panels (b) and (c) across target models.

This utility consumes the summary and per-position JSONL emitted by
``plot_biasnet_intervention.py``.  It only restyles already-computed
statistics, so it neither loads BiasNet checkpoints nor calls a target API.
Histogram bins and plot limits are shared across rows to make the model
comparison visually meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine BiasNet KL-distribution and token-level KL panels."
    )
    parser.add_argument(
        "--input",
        action="append",
        nargs=2,
        metavar=("MODEL_LABEL", "SUMMARY_JSON"),
        required=True,
        help="Model label and source intervention summary; specify exactly twice.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--output_stem", default="biasnet_intervention_bc_comparison"
    )
    parser.add_argument("--histogram_bins", type=int, default=50)
    parser.add_argument("--example_max_tokens", type=int, default=32)
    parser.add_argument(
        "--kl_direction",
        choices=("biased_to_base", "base_to_biased"),
        default="biased_to_base",
    )
    parser.add_argument("--axis_label_fontsize", type=float, default=15.0)
    parser.add_argument("--tick_fontsize", type=float, default=12.0)
    parser.add_argument("--panel_label_fontsize", type=float, default=16.0)
    parser.add_argument("--subplot_caption_fontsize", type=float, default=18.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"No position rows found: {path}")
    return rows


def display_token(text: str, limit: int = 13) -> str:
    normalized = (
        str(text)
        .replace("\r", "↵")
        .replace("\n", "↵")
        .replace("\t", "⇥")
        .replace(" ", "·")
    )
    visible = "".join(
        character
        if character.isascii() or character in {"·", "↵", "⇥"}
        else f"U+{ord(character):04X}"
        for character in normalized
    )
    if not visible:
        visible = "∅"
    return visible if len(visible) <= limit else visible[: limit - 1] + "…"


def resolve_positions_path(summary: dict[str, Any], summary_path: Path) -> Path:
    value = (summary.get("outputs") or {}).get("positions_jsonl")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Summary has no outputs.positions_jsonl: {summary_path}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = summary_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_run(model_label: str, summary_value: str, metric_key: str) -> dict[str, Any]:
    summary_path = Path(summary_value).expanduser().resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = read_object(summary_path)
    positions_path = resolve_positions_path(summary, summary_path)
    rows = read_jsonl(positions_path)
    example_file = (summary.get("plot") or {}).get("example_file")
    example_rows = sorted(
        (row for row in rows if row.get("record_file") == example_file),
        key=lambda row: int(row["token_position"]),
    )
    if not example_rows:
        raise ValueError(f"No rows found for example {example_file!r}: {positions_path}")
    for row in rows:
        value = row.get(metric_key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"Invalid {metric_key} in {positions_path}")
    return {
        "model_label": model_label,
        "summary_path": summary_path,
        "positions_path": positions_path,
        "summary": summary,
        "rows": rows,
        "example_rows": example_rows,
    }


def plot_comparison(runs: list[dict[str, Any]], args: argparse.Namespace) -> None:
    metric_key = (
        "kl_biased_to_base"
        if args.kl_direction == "biased_to_base"
        else "kl_base_to_biased"
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{args.output_stem}.png"
    pdf_path = output_dir / f"{args.output_stem}.pdf"
    summary_path = output_dir / f"{args.output_stem}.summary.json"
    for path in (png_path, pdf_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite output: {path}")

    all_values = [
        np.asarray([row[metric_key] for row in run["rows"]], dtype=np.float64)
        for run in runs
    ]
    histogram_max = max(float(values.max()) for values in all_values)
    histogram_edges = np.linspace(0.0, histogram_max, args.histogram_bins + 1)
    histogram_counts = [
        np.histogram(values, bins=histogram_edges)[0] for values in all_values
    ]
    histogram_ymax = max(int(counts.max()) for counts in histogram_counts) * 1.07

    examples = [run["example_rows"][: args.example_max_tokens] for run in runs]
    example_ymax = max(
        float(row[metric_key]) for example in examples for row in example
    ) * 1.07

    plt.rcParams.update(
        {
            "font.size": args.tick_fontsize,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "font.family": "Droid Sans",
        }
    )
    figure, axes = plt.subplots(len(runs), 2, figsize=(13.2, 4.0 * len(runs)))
    if len(runs) == 1:
        axes = np.asarray([axes])
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.94,
        wspace=0.24,
        hspace=0.58,
    )
    panel_index = 0
    for run_index, (run, values, example) in enumerate(
        zip(runs, all_values, examples)
    ):
        histogram_axis, token_axis = axes[run_index]
        histogram_axis.hist(
            values,
            bins=histogram_edges,
            color="#72B7B2",
            edgecolor="white",
            linewidth=0.35,
        )
        histogram_axis.set_xlim(0.0, histogram_max)
        histogram_axis.set_ylim(0.0, histogram_ymax)
        histogram_axis.set_xlabel(
            "KL Divergence", fontsize=args.axis_label_fontsize
        )
        histogram_axis.set_ylabel("Frequency", fontsize=args.axis_label_fontsize)
        histogram_axis.tick_params(axis="both", labelsize=args.tick_fontsize)
        histogram_axis.grid(axis="y", alpha=0.25, linewidth=0.6)

        x = np.arange(len(example))
        token_values = [row[metric_key] for row in example]
        token_labels = [display_token(row["label_token_text"]) for row in example]
        scales = np.asarray(
            [row.get("residual_scale", 1.0) for row in example], dtype=np.float64
        )
        colors = np.where(scales > 0.0, "#F58518", "#BAB0AC")
        token_axis.bar(x, token_values, color=colors, width=0.82)
        token_axis.set_xticks(
            x,
            token_labels,
            rotation=62,
            ha="right",
            fontsize=args.tick_fontsize,
        )
        token_axis.set_xlabel("")
        token_axis.set_ylabel(
            "KL Divergence", fontsize=args.axis_label_fontsize
        )
        token_axis.tick_params(axis="y", labelsize=args.tick_fontsize)
        token_axis.set_ylim(0.0, example_ymax)
        token_axis.grid(axis="y", alpha=0.25, linewidth=0.6)

        for axis in (histogram_axis, token_axis):
            axis.set_title(
                run["model_label"],
                fontsize=args.subplot_caption_fontsize,
                fontweight="bold",
                pad=10,
            )
            panel_index += 1
            axis.text(
                -0.12,
                1.08,
                f"({chr(96 + panel_index)})",
                transform=axis.transAxes,
                fontsize=args.panel_label_fontsize,
                fontweight="bold",
                va="top",
            )
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    output_summary = {
        "schema_version": 1,
        "methodology": {
            "api_calls": 0,
            "biasnet_replay_repeated": False,
            "source": "previously computed per-position intervention JSONL",
            "kl_direction": args.kl_direction,
            "shared_histogram_bin_edges": True,
            "shared_histogram_limits": True,
            "shared_example_kl_limits": True,
        },
        "plot": {
            "layout": f"{len(runs)}x2",
            "columns": [
                "KL-divergence frequency histogram",
                "per-token KL divergence",
            ],
            "histogram_bins": args.histogram_bins,
            "example_max_tokens": args.example_max_tokens,
            "captions": "per-panel model captions only",
            "axis_label_fontsize": args.axis_label_fontsize,
            "tick_fontsize": args.tick_fontsize,
            "panel_label_fontsize": args.panel_label_fontsize,
            "subplot_caption_fontsize": args.subplot_caption_fontsize,
            "subplot_captions": [
                [run["model_label"], run["model_label"]] for run in runs
            ],
            "axis_labels": {
                "distribution_x": "KL Divergence",
                "distribution_y": "Frequency",
                "token_x": "",
                "token_y": "KL Divergence",
            },
        },
        "inputs": [
            {
                "model_label": run["model_label"],
                "experiment_label": (run["summary"].get("plot") or {}).get(
                    "experiment_label"
                ),
                "summary_json": str(run["summary_path"]),
                "positions_jsonl": str(run["positions_path"]),
                "record_count": (run["summary"].get("aggregate") or {}).get(
                    "record_count"
                ),
                "token_position_count": len(run["rows"]),
                "example_file": (run["summary"].get("plot") or {}).get(
                    "example_file"
                ),
            }
            for run in runs
        ],
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(
        json.dumps(output_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_summary["outputs"], sort_keys=True))


def main() -> None:
    args = parse_args()
    if len(args.input) != 2:
        raise ValueError("--input must be specified exactly twice")
    positive_values = {
        "histogram_bins": args.histogram_bins,
        "example_max_tokens": args.example_max_tokens,
        "axis_label_fontsize": args.axis_label_fontsize,
        "tick_fontsize": args.tick_fontsize,
        "panel_label_fontsize": args.panel_label_fontsize,
        "subplot_caption_fontsize": args.subplot_caption_fontsize,
        "dpi": args.dpi,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"These values must be positive: {', '.join(invalid)}")
    metric_key = (
        "kl_biased_to_base"
        if args.kl_direction == "biased_to_base"
        else "kl_base_to_biased"
    )
    runs = [load_run(label, summary, metric_key) for label, summary in args.input]
    plot_comparison(runs, args)


if __name__ == "__main__":
    main()
