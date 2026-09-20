#!/usr/bin/env python3
"""Plot Figure-3-style BiasNet intervention diagnostics from a cached run.

The script replays BiasNet on the exact dense score vectors stored in an MC
cache.  It never calls the target model API.  Consequently, token positions
are conditioned on the cached (teacher-forced) response prefixes rather than
on a newly generated trajectory.

By default, four panels mirror the analysis in Figure 3 of the JULI paper:

* top-k log probabilities at the first response position, before and after
  BiasNet;
* the distribution of per-position KL(biased || base);
* per-token KL on one deterministic example record; and
* per-token top-k replacements on that record.  A replacement count is
  ``k - |topk_base intersect topk_biased|``, i.e. half the cardinality of the
  true symmetric difference for two sets of equal size.

Alongside PNG/PDF figures, the script writes per-position JSONL statistics and
a provenance-rich JSON summary so that the plot can be audited or restyled
without rerunning BiasNet.

Use ``--panels bc`` for a compact two-panel figure containing only the global
KL histogram and the example's token-level KL trace.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modeling_biasnet import BiasNet  # noqa: E402


DEFAULT_STEM = "biasnet_intervention"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Figure-3-style intervention plots by replaying BiasNet on "
            "cached MC score vectors."
        )
    )
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--biasnet_ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--output_stem", default=DEFAULT_STEM)
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        help=(
            "Tokenizer path/name used only for plot labels. By default it is "
            "read from cache metadata; token IDs are used if loading fails."
        ),
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Forwarded to AutoTokenizer."
    )
    parser.add_argument(
        "--local_files_only", action="store_true", help="Forwarded to AutoTokenizer."
    )
    parser.add_argument(
        "--file_split",
        choices=("all", "checkpoint_train", "checkpoint_held_out"),
        default="all",
        help="Select cache files using held_out_files recorded by the checkpoint.",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=100,
        help="Maximum selected records; 0 means no limit (default: 100).",
    )
    parser.add_argument(
        "--max_tokens_per_record",
        type=int,
        default=0,
        help="Maximum positions per record; 0 means every cached position.",
    )
    parser.add_argument("--position_batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--example_file",
        default=None,
        help="Cache filename/path to use in panels (a), (c), and (d).",
    )
    parser.add_argument(
        "--example_strategy",
        choices=("first", "first_argmax_change", "max_first_kl"),
        default="first_argmax_change",
        help="Deterministic example selection when --example_file is omitted.",
    )
    parser.add_argument("--example_max_tokens", type=int, default=32)
    parser.add_argument(
        "--respect_risk_gate",
        action="store_true",
        help=(
            "Scale the residual with the checkpoint's cached hard/soft risk gate. "
            "By default the plot isolates BiasNet itself (residual scale 1)."
        ),
    )
    parser.add_argument(
        "--device", default="auto", help="Torch device (default: auto)."
    )
    parser.add_argument(
        "--model_dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--kl_direction",
        choices=("biased_to_base", "base_to_biased"),
        default="biased_to_base",
        help="KL direction drawn in panels (b) and (c); both are saved to JSONL.",
    )
    parser.add_argument("--histogram_bins", type=int, default=50)
    parser.add_argument(
        "--panels",
        choices=("abcd", "bc"),
        default="abcd",
        help="Render the original four panels or only panels (b) and (c).",
    )
    parser.add_argument(
        "--histogram_max_quantile",
        type=float,
        default=1.0,
        help="Optional upper quantile for the displayed histogram x range.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def require_positive(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[value]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 BiasNet replay on CPU is unsupported; use float32.")
    return dtype


def cache_files(cache_dir: Path) -> list[Path]:
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")
    files = sorted(
        path
        for path in cache_dir.glob("*.pt")
        if path.name != "global_unigram_prior.pt"
    )
    if not files:
        raise FileNotFoundError(f"No cache .pt records found in {cache_dir}")
    return files


def select_files(
    files: list[Path], config: dict[str, Any], split: str, max_records: int
) -> list[Path]:
    held_out_value = config.get("held_out_files")
    held_out = set(held_out_value or [])
    if split != "all" and not isinstance(held_out_value, list):
        raise ValueError(
            f"--file_split={split} requires held_out_files in checkpoint config."
        )
    if split == "checkpoint_train":
        files = [path for path in files if path.name not in held_out]
    elif split == "checkpoint_held_out":
        missing = sorted(held_out - {path.name for path in files})
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} checkpoint held-out files are absent from the cache; "
                f"first missing file: {missing[0]}"
            )
        files = [path for path in files if path.name in held_out]
    if max_records > 0:
        files = files[:max_records]
    if not files:
        raise ValueError(f"No records remain after selecting split {split!r}.")
    return files


def resolve_example_path(value: str, files: list[Path], cache_dir: Path) -> Path:
    requested = Path(value).expanduser()
    if not requested.is_absolute():
        direct = cache_dir / requested
        requested = direct if direct.exists() else requested
    requested = requested.resolve()
    by_resolved = {path.resolve(): path for path in files}
    if requested not in by_resolved:
        raise ValueError(
            "--example_file must be one of the records selected by --file_split "
            f"and --max_records: {requested}"
        )
    return by_resolved[requested]


def first_record_metadata(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Cache record must be a dict: {path}")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"Cache metadata must be a dict: {path}")
    return payload, metadata


def load_tokenizer(
    name: str | None,
    metadata: dict[str, Any],
    *,
    trust_remote_code: bool,
    local_files_only: bool,
):
    resolved_name = name
    if resolved_name is None:
        resolved_name = metadata.get("materialized_log_probs_tokenizer") or metadata.get(
            "tokenizer_name"
        )
    if not resolved_name:
        print("tokenizer=unavailable labels=token_ids", flush=True)
        return None, None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            resolved_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except Exception as exc:  # Plotting remains possible without text labels.
        print(
            f"tokenizer_load_warning name={resolved_name!r} "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None, str(resolved_name)
    return tokenizer, str(resolved_name)


def decode_token(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return f"#{token_id}"
    try:
        text = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if text:
        return text
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    return str(token) if token is not None else f"#{token_id}"


def display_token(text: str, limit: int = 18) -> str:
    normalized = (
        str(text)
        .replace("\r", "↵")
        .replace("\n", "↵")
        .replace("\t", "⇥")
        .replace(" ", "·")
    )
    # Cluster nodes do not consistently install CJK/emoji fonts. Keep the
    # underlying Unicode text in JSONL, but make every plotted label portable.
    visible = "".join(
        character
        if character.isascii() or character in {"·", "↵", "⇥"}
        else f"U+{ord(character):04X}"
        for character in normalized
    )
    if not visible:
        visible = "∅"
    if len(visible) > limit:
        visible = visible[: limit - 1] + "…"
    return visible


def distribution_metrics(
    base_scores: torch.Tensor, biased_scores: torch.Tensor, top_k: int
) -> dict[str, torch.Tensor]:
    """Return full-vocabulary per-row intervention metrics in float32."""

    if base_scores.ndim != 2 or biased_scores.shape != base_scores.shape:
        raise ValueError("base_scores and biased_scores must match [batch, vocab].")
    if base_scores.shape[-1] <= 0:
        raise ValueError("Score tensors must have a non-empty vocabulary.")
    top_k = min(int(top_k), int(base_scores.shape[-1]))
    require_positive(top_k, "top_k")

    base_log_probs = torch.log_softmax(base_scores.float(), dim=-1)
    biased_log_probs = torch.log_softmax(biased_scores.float(), dim=-1)
    base_probs = base_log_probs.exp()
    biased_probs = biased_log_probs.exp()
    kl_biased_to_base = torch.sum(
        biased_probs * (biased_log_probs - base_log_probs), dim=-1
    ).clamp_min_(0.0)
    kl_base_to_biased = torch.sum(
        base_probs * (base_log_probs - biased_log_probs), dim=-1
    ).clamp_min_(0.0)

    base_top_values, base_top_ids = torch.topk(base_log_probs, k=top_k, dim=-1)
    biased_top_values, biased_top_ids = torch.topk(
        biased_log_probs, k=top_k, dim=-1
    )
    base_boundary = base_top_values[:, -1:]
    biased_boundary = biased_top_values[:, -1:]
    base_boundary_ties = torch.sum(base_log_probs == base_boundary, dim=-1)
    biased_boundary_ties = torch.sum(biased_log_probs == biased_boundary, dim=-1)
    base_boundary_slots = top_k - torch.sum(
        base_log_probs > base_boundary, dim=-1
    )
    biased_boundary_slots = top_k - torch.sum(
        biased_log_probs > biased_boundary, dim=-1
    )
    intersection = (
        base_top_ids.unsqueeze(-1) == biased_top_ids.unsqueeze(-2)
    ).any(dim=-1).sum(dim=-1)
    replacements = top_k - intersection
    return {
        "base_log_probs": base_log_probs,
        "biased_log_probs": biased_log_probs,
        "kl_biased_to_base": kl_biased_to_base,
        "kl_base_to_biased": kl_base_to_biased,
        "base_top_ids": base_top_ids,
        "base_top_values": base_top_values,
        "biased_top_ids": biased_top_ids,
        "biased_top_values": biased_top_values,
        "topk_replacements": replacements,
        "topk_symmetric_difference": 2 * replacements,
        "base_topk_boundary_ties": base_boundary_ties,
        "biased_topk_boundary_ties": biased_boundary_ties,
        "base_topk_ambiguous": base_boundary_ties > base_boundary_slots,
        "biased_topk_ambiguous": biased_boundary_ties > biased_boundary_slots,
    }


def gate_scales(
    payload: dict[str, Any], config: dict[str, Any], length: int, enabled: bool
) -> torch.Tensor:
    if not enabled:
        return torch.ones(length, dtype=torch.float32)
    training_mode = str(config.get("risk_gate_training", "none"))
    warmup = int(config.get("risk_gate_warmup_tokens", 0) or 0)
    threshold = float(config.get("risk_gate_threshold", 0.0))
    positions = torch.arange(length)
    forced = positions < warmup
    scores = payload.get("risk_gate_scores")
    mask = payload.get("risk_gate_mask")
    if training_mode == "runtime_soft":
        if not isinstance(scores, torch.Tensor):
            raise ValueError("runtime_soft gate replay requires risk_gate_scores.")
        temperature = float(config.get("risk_gate_soft_temperature", 0.05))
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("Checkpoint risk_gate_soft_temperature must be positive.")
        flattened = scores.reshape(-1)[:length].float()
        scales = torch.sigmoid((threshold - flattened) / temperature)
        scales[forced] = 1.0
        minimum = float(config.get("risk_gate_min_scale", 0.0) or 0.0)
        scales[scales < minimum] = 0.0
        return scales
    if isinstance(scores, torch.Tensor):
        active = scores.reshape(-1)[:length].float() < threshold
    elif isinstance(mask, torch.Tensor):
        active = mask.reshape(-1)[:length].bool()
    else:
        raise ValueError("Hard risk-gate replay requires risk_gate_scores or risk_gate_mask.")
    return torch.logical_or(active, forced).float()


def validate_cache_contract(
    payload: dict[str, Any], config: dict[str, Any], path: Path
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    features = payload.get("log_probs")
    labels = payload.get("labels")
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError(f"{path}: log_probs must have shape [1, positions, vocab].")
    if features.shape[0] != 1:
        raise ValueError(f"{path}: only one cached response per record is supported.")
    if not isinstance(labels, torch.Tensor) or tuple(labels.shape) != tuple(
        features.shape[:2]
    ):
        raise ValueError(f"{path}: labels must match log_probs leading dimensions.")
    vocab_size = int(config.get("vocab_size", 0) or 0)
    if features.shape[-1] != vocab_size:
        raise ValueError(
            f"{path}: cache vocabulary {features.shape[-1]} != checkpoint {vocab_size}."
        )
    feature_representation = config.get("mc_input_representation")
    base_representation = config.get("mc_base_score_representation")
    if base_representation not in (None, feature_representation):
        raise ValueError(
            "This cache exposes only log_probs, but the checkpoint uses different "
            "BiasNet feature and base-score representations: "
            f"{feature_representation!r} vs {base_representation!r}."
        )
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata must be a dict.")
    expected_mode = config.get("mc_static_prior_mode") or "none"
    actual_mode = metadata.get("mc_static_prior_mode") or "none"
    if expected_mode != actual_mode:
        raise ValueError(
            f"{path}: cache/checkpoint static prior mismatch: "
            f"{actual_mode!r} != {expected_mode!r}."
        )
    expected_strength = config.get("mc_static_prior_strength")
    actual_strength = metadata.get("mc_static_prior_strength")
    if expected_strength is not None:
        if actual_strength is None or not math.isclose(
            float(expected_strength),
            float(actual_strength),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{path}: cache/checkpoint static-prior strength mismatch: "
                f"{actual_strength!r} != {expected_strength!r}."
            )
    return features[0], labels[0], metadata


def model_forward(
    model: BiasNet,
    features: torch.Tensor,
    position_ids: torch.Tensor,
    payload: dict[str, Any],
) -> torch.Tensor:
    kwargs: dict[str, Any] = {}
    if int(getattr(model, "num_position_buckets", 0) or 0) > 0:
        kwargs["position_ids"] = position_ids
    if bool(getattr(model, "mc_sample_count_conditioning", False)):
        counts = payload.get("valid_sample_counts")
        if not isinstance(counts, torch.Tensor):
            raise ValueError("Sample-count-conditioned BiasNet requires valid_sample_counts.")
        kwargs["mc_sample_counts"] = counts.reshape(-1)[position_ids.cpu()].to(
            features.device
        )
    if bool(getattr(model, "context_conditioning", False)):
        context = payload.get("context_features")
        if not isinstance(context, torch.Tensor):
            raise ValueError(
                "Context-conditioned BiasNet replay requires context_features in each cache."
            )
        kwargs["context_features"] = context.reshape(
            -1, context.shape[-1]
        )[position_ids.cpu()].to(device=features.device, dtype=features.dtype)
    return model(features, **kwargs)


def iter_chunks(length: int, size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, length, size):
        yield start, min(start + size, length)


def example_rank(
    strategy: str, record_order: int, first_row: dict[str, Any]
) -> tuple[float, ...]:
    if strategy == "first":
        return (float(-record_order),)
    if strategy == "first_argmax_change":
        changed = int(bool(first_row["argmax_changed"]))
        # Prefer the first changed record; if none changed, prefer the first record.
        return (float(changed), float(-record_order))
    return (float(first_row["kl_biased_to_base"]), float(-record_order))


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty metric.")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def prior_label(config: dict[str, Any], metadata: dict[str, Any]) -> str:
    samples = metadata.get("samples_per_token")
    mc_label = f"MC{samples}" if isinstance(samples, int) else "MC"
    mode = config.get("mc_static_prior_mode") or metadata.get("mc_static_prior_mode")
    strength = config.get("mc_static_prior_strength")
    if mode == "global_unigram_dirichlet_v1":
        prior = "global-unigram"
    elif mode == "uniform_dirichlet_v1":
        prior = "global-uniform"
    elif mode:
        prior = str(mode)
    else:
        prior = "no static prior"
    suffix = f", κ={float(strength):g}" if strength is not None else ""
    return f"{mc_label} + {prior}{suffix}"


def plot_topk_pair(
    figure,
    subplot_spec,
    example: dict[str, Any],
    tokenizer,
    top_k: int,
    experiment_label: str,
) -> None:
    grid = subplot_spec.subgridspec(2, 1, hspace=0.65)
    panels = (
        ("base_top_ids", "base_top_values", f"{experiment_label} (base)"),
        ("biased_top_ids", "biased_top_values", "After BiasNet"),
    )
    colors = ("#4C78A8", "#E45756")
    for index, (ids_key, values_key, label) in enumerate(panels):
        axis = figure.add_subplot(grid[index, 0])
        token_ids = example["positions"][0][ids_key][:top_k]
        values = example["positions"][0][values_key][:top_k]
        ambiguity_key = (
            "base_topk_ambiguous" if index == 0 else "biased_topk_ambiguous"
        )
        if example["positions"][0][ambiguity_key]:
            label += " (kth-place tie)"
        labels = [
            display_token(decode_token(tokenizer, token_id), 12)
            for token_id in token_ids
        ]
        axis.bar(np.arange(len(values)), values, color=colors[index], width=0.78)
        axis.set_xticks(np.arange(len(values)), labels, rotation=42, ha="right")
        axis.set_ylabel("Log probability")
        axis.set_title(label, fontsize=10, pad=4)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    # A panel label on the upper child reads naturally for the nested subplot.
    figure.axes[-2].text(
        -0.13,
        1.22,
        "(a)",
        transform=figure.axes[-2].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def plot_figure(
    output_png: Path,
    output_pdf: Path,
    all_rows: list[dict[str, Any]],
    example: dict[str, Any],
    tokenizer,
    args: argparse.Namespace,
    experiment_label: str,
) -> None:
    metric_key = (
        "kl_biased_to_base"
        if args.kl_direction == "biased_to_base"
        else "kl_base_to_biased"
    )
    kl_math = (
        r"$D_{KL}(p_{biased}\,\Vert\,p_{base})$"
        if args.kl_direction == "biased_to_base"
        else r"$D_{KL}(p_{base}\,\Vert\,p_{biased})$"
    )
    values = np.asarray([row[metric_key] for row in all_rows], dtype=np.float64)
    example_positions = example["positions"][: args.example_max_tokens]
    x = np.arange(len(example_positions))
    token_labels = [
        display_token(row["label_token_text"], 13) for row in example_positions
    ]

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "font.family": "Droid Sans",
        }
    )
    if args.panels == "bc":
        figure, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
        figure.subplots_adjust(
            left=0.07, right=0.98, bottom=0.28, top=0.75, wspace=0.24
        )
        histogram_axis, kl_axis = axes
        histogram_xlabel = "KL Divergence"
        histogram_ylabel = "Frequency"
        kl_xlabel = ""
        kl_ylabel = "KL Divergence"
    else:
        figure = plt.figure(figsize=(13.2, 8.0))
        figure.subplots_adjust(
            left=0.07, right=0.98, bottom=0.15, top=0.88, wspace=0.27, hspace=0.5
        )
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
        plot_topk_pair(
            figure, grid[0, 0], example, tokenizer, args.top_k, experiment_label
        )
        histogram_axis = figure.add_subplot(grid[0, 1])
        kl_axis = figure.add_subplot(grid[1, 0])
        histogram_xlabel = kl_math
        histogram_ylabel = "Token-position frequency"
        kl_xlabel = "Cached response token"
        kl_ylabel = kl_math

    upper = float(np.quantile(values, args.histogram_max_quantile))
    histogram_values = (
        values[values <= upper]
        if args.histogram_max_quantile < 1.0
        else values
    )
    histogram_axis.hist(
        histogram_values,
        bins=args.histogram_bins,
        color="#72B7B2",
        edgecolor="white",
        linewidth=0.35,
    )
    histogram_axis.set_xlabel(histogram_xlabel)
    histogram_axis.set_ylabel(histogram_ylabel)
    histogram_axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    if args.histogram_max_quantile < 1.0:
        histogram_axis.set_title(
            f"Displayed through q={args.histogram_max_quantile:g} (x ≤ {upper:.3g})",
            fontsize=9,
        )
    histogram_axis.text(
        -0.12,
        1.06,
        "(b)",
        transform=histogram_axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    example_kl = [row[metric_key] for row in example_positions]
    gate_scales_values = np.asarray(
        [row["residual_scale"] for row in example_positions], dtype=np.float64
    )
    kl_colors = np.where(gate_scales_values > 0.0, "#F58518", "#BAB0AC")
    kl_axis.bar(x, example_kl, color=kl_colors, width=0.82)
    kl_axis.set_xticks(x, token_labels, rotation=62, ha="right")
    kl_axis.set_xlabel(kl_xlabel)
    kl_axis.set_ylabel(kl_ylabel)
    kl_axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    kl_axis.text(
        -0.12,
        1.06,
        "(c)",
        transform=kl_axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )

    if args.panels == "abcd":
        difference_axis = figure.add_subplot(grid[1, 1])
        replacements = [row["topk_replacements"] for row in example_positions]
        ambiguous = np.asarray(
            [
                row["base_topk_ambiguous"] or row["biased_topk_ambiguous"]
                for row in example_positions
            ],
            dtype=bool,
        )
        difference_colors = np.where(ambiguous, "#BAB0AC", "#54A24B")
        difference_axis.bar(x, replacements, color=difference_colors, width=0.82)
        difference_axis.set_xticks(x, token_labels, rotation=62, ha="right")
        difference_axis.set_xlabel("Cached response token")
        difference_axis.set_ylabel(f"Top-{args.top_k} replacements")
        difference_axis.set_ylim(0, args.top_k + 0.5)
        difference_axis.set_yticks(
            np.arange(0, args.top_k + 1, max(1, args.top_k // 5))
        )
        difference_axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        if ambiguous.any():
            difference_axis.text(
                0.99,
                0.96,
                "gray: top-k boundary tie",
                transform=difference_axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#666666",
            )
        difference_axis.text(
            -0.12,
            1.06,
            "(d)",
            transform=difference_axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    title = args.title or f"BiasNet intervention under {experiment_label}"
    subtitle = (
        f"Teacher-forced cache replay · example {example['record_file']} · "
        f"{len(all_rows):,} token positions"
    )
    figure.suptitle(f"{title}\n{subtitle}", fontsize=14)
    figure.savefig(output_png, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    require_positive(args.position_batch_size, "position_batch_size")
    require_positive(args.top_k, "top_k")
    require_positive(args.example_max_tokens, "example_max_tokens")
    require_positive(args.histogram_bins, "histogram_bins")
    if not 0.0 < args.histogram_max_quantile <= 1.0:
        raise ValueError("histogram_max_quantile must be in (0, 1].")
    if args.max_records < 0 or args.max_tokens_per_record < 0:
        raise ValueError(
            "max_records and max_tokens_per_record must be non-negative."
        )

    cache_dir = args.cache_dir.expanduser().resolve()
    checkpoint = args.biasnet_ckpt.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "pytorch_model.bin"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"BiasNet checkpoint must contain config.json and pytorch_model.bin: {checkpoint}"
        )
    config = read_json(config_path)
    if args.top_k > int(config.get("vocab_size", 0) or 0):
        raise ValueError(
            f"top_k={args.top_k} exceeds checkpoint vocabulary size "
            f"{config.get('vocab_size')}."
        )
    files = select_files(
        cache_files(cache_dir), config, args.file_split, args.max_records
    )
    first_payload, first_metadata = first_record_metadata(files[0])
    tokenizer, tokenizer_name = load_tokenizer(
        args.tokenizer_name,
        first_metadata,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer is not None and len(tokenizer) != int(config["vocab_size"]):
        raise ValueError(
            f"Tokenizer vocabulary {len(tokenizer)} != checkpoint vocabulary "
            f"{config['vocab_size']}."
        )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.model_dtype, device)
    print(
        f"loading_biasnet checkpoint={checkpoint} device={device} dtype={dtype}",
        flush=True,
    )
    model = BiasNet.from_pretrained(str(checkpoint), map_location="cpu")
    model = model.to(device=device, dtype=dtype)
    model.set_up_proj()
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / f"{args.output_stem}.positions.jsonl"
    summary_path = output_dir / f"{args.output_stem}.summary.json"
    png_path = output_dir / f"{args.output_stem}.png"
    pdf_path = output_dir / f"{args.output_stem}.pdf"

    explicit_example = (
        resolve_example_path(args.example_file, files, cache_dir)
        if args.example_file
        else None
    )
    all_rows: list[dict[str, Any]] = []
    selected_example: dict[str, Any] | None = None
    selected_rank: tuple[float, ...] | None = None
    metadata_modes: set[Any] = set()
    metadata_strengths: set[Any] = set()
    sample_counts: set[Any] = set()

    with torch.inference_mode():
        for record_order, path in enumerate(files):
            payload = (
                first_payload
                if record_order == 0
                else torch.load(path, map_location="cpu", weights_only=False)
            )
            features_cpu, labels_cpu, metadata = validate_cache_contract(
                payload, config, path
            )
            length = int(features_cpu.shape[0])
            if args.max_tokens_per_record > 0:
                length = min(length, args.max_tokens_per_record)
            if length <= 0:
                raise ValueError(f"{path}: cache record has no token positions.")
            scales_cpu = gate_scales(payload, config, length, args.respect_risk_gate)
            metadata_modes.add(metadata.get("mc_static_prior_mode"))
            metadata_strengths.add(metadata.get("mc_static_prior_strength"))
            sample_counts.add(metadata.get("samples_per_token"))
            record_rows: list[dict[str, Any]] = []

            for start, end in iter_chunks(length, args.position_batch_size):
                features = features_cpu[start:end].to(device=device, dtype=dtype)
                positions = torch.arange(start, end, device=device, dtype=torch.long)
                residual = model_forward(model, features, positions, payload)
                scales = scales_cpu[start:end].to(device=device, dtype=torch.float32)
                biased_scores = (
                    features.float() + scales.unsqueeze(-1) * residual.float()
                )
                metrics = distribution_metrics(
                    features.float(), biased_scores, args.top_k
                )
                if not (
                    torch.isfinite(metrics["kl_biased_to_base"]).all()
                    and torch.isfinite(metrics["kl_base_to_biased"]).all()
                ):
                    raise ValueError(
                        f"{path}: non-finite intervention metric in positions "
                        f"{start}:{end}."
                    )

                labels = labels_cpu[start:end].tolist()
                base_target = metrics["base_log_probs"].gather(
                    1,
                    torch.as_tensor(
                        labels, device=device, dtype=torch.long
                    ).unsqueeze(-1),
                )[:, 0]
                biased_target = metrics["biased_log_probs"].gather(
                    1,
                    torch.as_tensor(
                        labels, device=device, dtype=torch.long
                    ).unsqueeze(-1),
                )[:, 0]
                for local_index, label_id in enumerate(labels):
                    position = start + local_index
                    base_ids = metrics["base_top_ids"][local_index].cpu().tolist()
                    biased_ids = metrics["biased_top_ids"][local_index].cpu().tolist()
                    base_values = (
                        metrics["base_top_values"][local_index].cpu().tolist()
                    )
                    biased_values = (
                        metrics["biased_top_values"][local_index].cpu().tolist()
                    )
                    row = {
                        "record_file": path.name,
                        "record_order": record_order,
                        "dataset_index": metadata.get("dataset_idx"),
                        "token_position": position,
                        "label_token_id": int(label_id),
                        "label_token_text": decode_token(tokenizer, int(label_id)),
                        "residual_scale": float(scales_cpu[position]),
                        "gate_active": bool(scales_cpu[position] > 0.0),
                        "base_top1_id": int(base_ids[0]),
                        "base_top1_text": decode_token(tokenizer, int(base_ids[0])),
                        "biased_top1_id": int(biased_ids[0]),
                        "biased_top1_text": decode_token(tokenizer, int(biased_ids[0])),
                        "argmax_changed": bool(base_ids[0] != biased_ids[0]),
                        "target_base_log_probability": float(base_target[local_index]),
                        "target_biased_log_probability": float(
                            biased_target[local_index]
                        ),
                        "target_log_probability_delta": float(
                            biased_target[local_index] - base_target[local_index]
                        ),
                        "kl_biased_to_base": float(
                            metrics["kl_biased_to_base"][local_index]
                        ),
                        "kl_base_to_biased": float(
                            metrics["kl_base_to_biased"][local_index]
                        ),
                        "topk_replacements": int(
                            metrics["topk_replacements"][local_index]
                        ),
                        "topk_symmetric_difference": int(
                            metrics["topk_symmetric_difference"][local_index]
                        ),
                        "base_topk_boundary_ties": int(
                            metrics["base_topk_boundary_ties"][local_index]
                        ),
                        "biased_topk_boundary_ties": int(
                            metrics["biased_topk_boundary_ties"][local_index]
                        ),
                        "base_topk_ambiguous": bool(
                            metrics["base_topk_ambiguous"][local_index]
                        ),
                        "biased_topk_ambiguous": bool(
                            metrics["biased_topk_ambiguous"][local_index]
                        ),
                        "base_top_ids": [int(value) for value in base_ids],
                        "base_top_values": [float(value) for value in base_values],
                        "biased_top_ids": [int(value) for value in biased_ids],
                        "biased_top_values": [float(value) for value in biased_values],
                    }
                    record_rows.append(row)
                    all_rows.append(row)

            candidate = {
                "record_file": path.name,
                "record_order": record_order,
                "dataset_index": metadata.get("dataset_idx"),
                "prompt_text": payload.get("prompt_text"),
                "answer_text": payload.get("answer_text"),
                "positions": record_rows,
            }
            if explicit_example is not None:
                if path.resolve() == explicit_example.resolve():
                    selected_example = candidate
                    selected_rank = (math.inf,)
            else:
                rank = example_rank(
                    args.example_strategy, record_order, record_rows[0]
                )
                if selected_rank is None or rank > selected_rank:
                    selected_example = candidate
                    selected_rank = rank
            print(
                f"record={record_order + 1}/{len(files)} file={path.name} "
                f"positions={length}",
                flush=True,
            )

    if selected_example is None:
        raise RuntimeError("Failed to select an example record.")

    # Keep top-k vectors in the audit JSONL: they make panel (a) and the set
    # metric exactly reproducible without loading the 500+ MB checkpoint again.
    with stats_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")

    kl_biased_to_base = [row["kl_biased_to_base"] for row in all_rows]
    kl_base_to_biased = [row["kl_base_to_biased"] for row in all_rows]
    replacements = [float(row["topk_replacements"]) for row in all_rows]
    argmax_changed = sum(bool(row["argmax_changed"]) for row in all_rows)
    active = sum(bool(row["gate_active"]) for row in all_rows)
    target_improved = sum(
        float(row["target_log_probability_delta"]) > 0.0 for row in all_rows
    )
    topk_ambiguous = sum(
        bool(row["base_topk_ambiguous"] or row["biased_topk_ambiguous"])
        for row in all_rows
    )
    experiment = prior_label(config, first_metadata)
    summary = {
        "schema_version": 1,
        "methodology": {
            "conditioning": "teacher_forced_cached_response_prefixes",
            "api_calls": 0,
            "base_distribution": "softmax(cached_base_scores)",
            "biased_distribution": (
                "softmax(cached_base_scores + residual_scale * BiasNet(features))"
            ),
            "topk_replacements": "k - cardinality(intersection(base_topk, biased_topk))",
            "topk_symmetric_difference": "2 * topk_replacements",
            "topk_tie_policy": (
                "torch.topk; boundary-tie ambiguity is flagged per position and "
                "shown in gray in panel (d)"
            ),
            "default_kl": "KL(biased || base)",
        },
        "inputs": {
            "cache_dir": str(cache_dir),
            "biasnet_checkpoint": str(checkpoint),
            "tokenizer": tokenizer_name,
            "file_split": args.file_split,
            "selected_files": [path.name for path in files],
            "respect_risk_gate": bool(args.respect_risk_gate),
            "checkpoint_static_prior_mode": config.get("mc_static_prior_mode"),
            "checkpoint_static_prior_strength": config.get("mc_static_prior_strength"),
            "cache_static_prior_modes": sorted(str(value) for value in metadata_modes),
            "cache_static_prior_strengths": sorted(
                str(value) for value in metadata_strengths
            ),
            "cache_samples_per_token": sorted(str(value) for value in sample_counts),
            "device": str(device),
            "model_dtype": str(dtype),
        },
        "plot": {
            "panels": args.panels,
            "experiment_label": experiment,
            "top_k": args.top_k,
            "kl_direction": args.kl_direction,
            "example_file": selected_example["record_file"],
            "example_dataset_index": selected_example["dataset_index"],
            "example_selection": (
                "explicit_file" if explicit_example else args.example_strategy
            ),
            "example_prompt": selected_example["prompt_text"],
            "example_answer": selected_example["answer_text"],
            "example_positions_plotted": min(
                args.example_max_tokens, len(selected_example["positions"])
            ),
        },
        "aggregate": {
            "record_count": len(files),
            "token_position_count": len(all_rows),
            "kl_biased_to_base": summarize(kl_biased_to_base),
            "kl_base_to_biased": summarize(kl_base_to_biased),
            "topk_replacements": summarize(replacements),
            "argmax_changed_count": argmax_changed,
            "argmax_changed_rate": argmax_changed / len(all_rows),
            "target_log_probability_improved_count": target_improved,
            "target_log_probability_improved_rate": target_improved / len(all_rows),
            "topk_boundary_ambiguous_count": topk_ambiguous,
            "topk_boundary_ambiguous_rate": topk_ambiguous / len(all_rows),
            "gate_active_count": active,
            "gate_active_rate": active / len(all_rows),
        },
        "outputs": {
            "png": str(png_path),
            "pdf": str(pdf_path),
            "positions_jsonl": str(stats_path),
            "summary_json": str(summary_path),
        },
    }
    plot_figure(
        png_path,
        pdf_path,
        all_rows,
        selected_example,
        tokenizer,
        args,
        experiment,
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(summary),
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")
    print(json.dumps(summary["outputs"], ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main(parse_args())
