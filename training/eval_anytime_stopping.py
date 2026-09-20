"""Calibrate and evaluate an anytime proxy-MC stopping policy.

The calibration phase uses group-out-of-fold predictions over development
records and writes a frozen policy specification.  The test phase only loads
that specification and evaluates the checkpoint's locked record split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mc_reconstruction import fuse_proxy_logits_with_mc_counts
from modeling_biasnet import BiasNet
from training.anytime_proxy_mc import (
    AnytimeProxyMCCache,
    BIASNET_AUTOCAST_DTYPE,
    BIASNET_AUTOCAST_ENABLED,
    BIASNET_PARAMETER_DTYPE,
    POLICY_SPEC_HASH_FIELD,
    PORTABLE_STOPPER_PROTOCOL,
    TARGET_PROTOCOL_METADATA_FIELDS,
    load_anytime_cache,
    parse_budgets,
    permute_events,
    policy_spec_payload_sha256,
    portable_linear_stopper_confidence,
)
from training.train_anytime_biasnet import _mixed_prefix_counts
from training.anytime_stopping_features import (
    FEATURE_NAMES,
    build_state_features as _build_state_features,
    top2_statistics as _top2_statistics,
)


EVALUATION_SCHEMA_VERSION = 2
# Deprecated policy alias retained so the first schema-v2 artifacts remain
# self-describing to readers that consumed the earlier field name.
BIASNET_RUNTIME_DTYPE = BIASNET_PARAMETER_DTYPE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate or test a K-conditioned anytime stopping policy."
    )
    parser.add_argument("--phase", choices=("calibrate", "test"), required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument(
        "--test_cache_dir",
        default=None,
        help=(
            "Fresh cache used only by --phase test. It must differ from the "
            "development cache frozen in policy_spec.json."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--reference_checkpoint",
        required=True,
        help="Original fixed-MC50 controller used as the quality reference.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--budgets", default="0,4,8,16,32,50")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--development_replays", type=int, default=4)
    parser.add_argument("--validation_replays", type=int, default=16)
    parser.add_argument("--test_replays", type=int, default=16)
    parser.add_argument("--replay_seed", type=int, default=20260824)
    parser.add_argument("--error_budgets", default="0.01,0.025,0.05,0.10")
    parser.add_argument(
        "--min_beneficial_flip_recall",
        type=float,
        default=0.95,
        help=(
            "Minimum validation recall of reference-MC50 beneficial flips for "
            "a stopping operating point."
        ),
    )
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--bootstrap_replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Debug-only row cap; prohibited in the locked test phase.",
    )
    return parser.parse_args()


def _parse_error_budgets(value: str) -> tuple[float, ...]:
    budgets = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not budgets or any(not math.isfinite(value) or value < 0 or value >= 1 for value in budgets):
        raise ValueError("Error budgets must be finite values in [0, 1).")
    if tuple(sorted(set(budgets))) != budgets:
        raise ValueError("Error budgets must be strictly increasing and unique.")
    return budgets


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


ESTIMATOR_METADATA_FIELDS = (
    "mc_fusion_mode",
    "proxy_temperature",
    "proxy_prior_strength",
    "proxy_model_name_or_path",
    "proxy_model_revision",
    "proxy_tokenizer_sha256",
    "proxy_chat_template_protocol",
    "shared_vocab_size",
    "proxy_vocab_size",
    "proxy_vocab_tail_policy",
    "proxy_dtype",
    "proxy_quantization",
    "sample_temperature",
    "top_p",
    "sample_completion_policy",
    "observed_alpha",
    "floor_mass",
    *TARGET_PROTOCOL_METADATA_FIELDS,
)


def _estimator_metadata(cache: AnytimeProxyMCCache) -> dict[str, Any]:
    return {
        field: cache.metadata.get(field) for field in ESTIMATOR_METADATA_FIELDS
    } | {"max_samples": cache.max_samples, "vocab_size": cache.vocab_size}


def _load_checkpoint_config(checkpoint: Path) -> dict:
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("anytime_schema_version") != 1:
        raise ValueError("Checkpoint does not declare anytime_schema_version=1.")
    if not config.get("mc_sample_count_conditioning"):
        raise ValueError("Checkpoint is not conditioned on the actual MC sample count.")
    return config


def _row_indices_for_records(
    cache: AnytimeProxyMCCache, record_names: Sequence[str]
) -> torch.Tensor:
    wanted = set(record_names)
    record_ids = [
        index for index, name in enumerate(cache.record_names) if name in wanted
    ]
    if len(record_ids) != len(wanted):
        missing = sorted(wanted - set(cache.record_names))
        raise ValueError(f"Split records are absent from the cache: {missing}")
    mask = torch.zeros(cache.num_rows, dtype=torch.bool)
    for record_id in record_ids:
        mask |= cache.record_ids.eq(record_id)
    return torch.nonzero(mask, as_tuple=False).flatten()


def _record_names_per_row(
    cache: AnytimeProxyMCCache, row_indices: torch.Tensor
) -> np.ndarray:
    return np.asarray(
        [cache.record_names[int(cache.record_ids[index])] for index in row_indices],
        dtype=object,
    )


@torch.no_grad()
def evaluate_replay_paths(
    *,
    cache: AnytimeProxyMCCache,
    row_indices: torch.Tensor,
    model: BiasNet,
    budgets: tuple[int, ...],
    replays: int,
    replay_seed: int,
    batch_size: int,
    device: torch.device,
    position_normalizer: int | None = None,
) -> dict[str, Any]:
    if row_indices.numel() == 0:
        raise ValueError("The evaluation split contains no rows.")
    if replays <= 0 or batch_size <= 0:
        raise ValueError("replays and batch_size must be positive.")
    row_events = cache.events[row_indices]
    row_keys = [cache.row_keys[int(index)] for index in row_indices]
    rows = int(row_indices.numel())
    stages = len(budgets)
    features = np.empty((replays, rows, stages, len(FEATURE_NAMES)), dtype=np.float32)
    actions = np.empty((replays, rows, stages), dtype=np.int64)
    observed_max_position = int(cache.position_ids[row_indices].max().item())
    if position_normalizer is None:
        position_normalizer = observed_max_position
    position_normalizer = int(position_normalizer)
    if position_normalizer <= 0:
        raise ValueError("position_normalizer must be positive.")
    if observed_max_position > position_normalizer:
        raise ValueError(
            "The evaluation split contains a position beyond the frozen "
            f"normalizer: {observed_max_position} > {position_normalizer}."
        )
    temperature = float(cache.metadata["proxy_temperature"])
    prior_strength = float(cache.metadata["proxy_prior_strength"])
    use_amp = device.type == "cuda"

    for replicate in range(replays):
        replay_events = permute_events(
            row_events,
            row_keys,
            replay_seed=replay_seed,
            replicate=replicate,
        )
        previous_actions = torch.full((rows,), -1, dtype=torch.long)
        stable_stages = torch.zeros(rows, dtype=torch.long)
        proxy_ids_all = torch.empty(rows, dtype=torch.long)
        proxy_probability_all = torch.empty(rows, dtype=torch.float32)
        proxy_gap_all = torch.empty(rows, dtype=torch.float32)
        for stage_index, budget in enumerate(budgets):
            for start in range(0, rows, batch_size):
                stop = min(start + batch_size, rows)
                source_ids = row_indices[start:stop]
                proxy_logits = cache.proxy_logits[source_ids].to(
                    device=device, non_blocking=True
                )
                positions = cache.position_ids[source_ids].to(device)
                base_token_ids = cache.base_token_ids[source_ids].to(device)
                batch_events = replay_events[start:stop].to(device)
                sample_counts = torch.full(
                    (stop - start,), budget, device=device, dtype=torch.long
                )
                counts = _mixed_prefix_counts(
                    batch_events,
                    sample_counts,
                    vocab_size=cache.vocab_size,
                    dtype=torch.float32,
                )
                fused_scores = fuse_proxy_logits_with_mc_counts(
                    proxy_logits,
                    counts,
                    temperature=temperature,
                    prior_strength=prior_strength,
                    dtype=torch.float32,
                )
                fused_ids, fused_probability, fused_gap = _top2_statistics(
                    fused_scores
                )
                if stage_index == 0:
                    proxy_ids = fused_ids
                    proxy_probability = fused_probability
                    proxy_gap = fused_gap
                    proxy_ids_all[start:stop] = proxy_ids.cpu()
                    proxy_probability_all[start:stop] = proxy_probability.cpu()
                    proxy_gap_all[start:stop] = proxy_gap.cpu()
                else:
                    proxy_ids = proxy_ids_all[start:stop].to(device)
                    proxy_probability = proxy_probability_all[start:stop].to(device)
                    proxy_gap = proxy_gap_all[start:stop].to(device)
                with torch.amp.autocast(
                    "cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    residual = model(
                        fused_scores,
                        position_ids=positions,
                        mc_sample_counts=sample_counts,
                    )
                    outputs = fused_scores + residual
                action_ids, action_probability, action_gap = _top2_statistics(outputs)
                prior_actions = (
                    None
                    if stage_index == 0
                    else previous_actions[start:stop].to(device)
                )
                if prior_actions is None:
                    current_stable = torch.zeros_like(action_ids, dtype=torch.bool)
                else:
                    current_stable = action_ids.eq(prior_actions)
                batch_stable_stages = stable_stages[start:stop].to(device)
                batch_stable_stages = torch.where(
                    current_stable,
                    batch_stable_stages + 1,
                    torch.zeros_like(batch_stable_stages),
                )
                state_features = _build_state_features(
                    budget=budget,
                    max_samples=cache.max_samples,
                    positions=positions,
                    proxy_top_ids=proxy_ids,
                    proxy_top_probability=proxy_probability,
                    proxy_gap=proxy_gap,
                    fused_top_ids=fused_ids,
                    fused_top_probability=fused_probability,
                    fused_gap=fused_gap,
                    action_ids=action_ids,
                    action_probability=action_probability,
                    action_gap=action_gap,
                    base_token_ids=base_token_ids,
                    counts=counts,
                    previous_actions=prior_actions,
                    stable_stages=batch_stable_stages,
                    stage_index=stage_index,
                    position_normalizer=position_normalizer,
                )
                features[replicate, start:stop, stage_index] = (
                    state_features.float().cpu().numpy()
                )
                actions[replicate, start:stop, stage_index] = action_ids.cpu().numpy()
                previous_actions[start:stop] = action_ids.cpu()
                stable_stages[start:stop] = batch_stable_stages.cpu()
        print(
            json.dumps(
                {"stage": "replay", "replicate": replicate + 1, "replays": replays},
                sort_keys=True,
            )
        )

    labels = cache.labels[row_indices].cpu().numpy()
    base_token_ids = cache.base_token_ids[row_indices].cpu().numpy()
    record_names = _record_names_per_row(cache, row_indices)
    full_actions = actions[:, :, -1]
    agreement = actions == full_actions[:, :, None]
    return {
        "features": features,
        "actions": actions,
        "agreement": agreement,
        "labels": labels,
        "base_token_ids": base_token_ids,
        "record_names": record_names,
        "row_keys": np.asarray(row_keys, dtype=object),
        "position_normalizer": position_normalizer,
    }


@torch.no_grad()
def evaluate_reference_actions(
    *,
    cache: AnytimeProxyMCCache,
    row_indices: torch.Tensor,
    model: BiasNet,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Evaluate the original fixed-MC50 controller on aggregate full counts."""

    actions = np.empty(int(row_indices.numel()), dtype=np.int64)
    temperature = float(cache.metadata["proxy_temperature"])
    prior_strength = float(cache.metadata["proxy_prior_strength"])
    use_amp = device.type == "cuda"
    full_budgets = torch.full(
        (int(row_indices.numel()),), cache.max_samples, dtype=torch.long
    )
    for start in range(0, int(row_indices.numel()), batch_size):
        stop = min(start + batch_size, int(row_indices.numel()))
        source_ids = row_indices[start:stop]
        proxy_logits = cache.proxy_logits[source_ids].to(
            device=device, non_blocking=True
        )
        events = cache.events[source_ids].to(device=device, non_blocking=True)
        counts = _mixed_prefix_counts(
            events,
            full_budgets[start:stop].to(device),
            vocab_size=cache.vocab_size,
            dtype=torch.float32,
        )
        fused_scores = fuse_proxy_logits_with_mc_counts(
            proxy_logits,
            counts,
            temperature=temperature,
            prior_strength=prior_strength,
            dtype=torch.float32,
        )
        positions = cache.position_ids[source_ids].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            if int(getattr(model, "num_position_buckets", 0) or 0) > 0:
                residual = model(fused_scores, position_ids=positions)
            else:
                residual = model(fused_scores)
            outputs = fused_scores + residual
        actions[start:stop] = outputs.argmax(dim=-1).cpu().numpy()
    return actions


def _make_stopper(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        ),
    )


def _flatten_development_states(paths: dict[str, Any]):
    features = paths["features"][:, :, :-1]
    labels = paths["agreement"][:, :, :-1].astype(np.int64)
    replays, rows, stages, width = features.shape
    flat_features = features.reshape(replays * rows * stages, width)
    flat_labels = labels.reshape(-1)
    groups = np.broadcast_to(
        paths["record_names"][None, :, None], (replays, rows, stages)
    ).reshape(-1)
    return flat_features, flat_labels, groups


def fit_group_oof_stopper(
    paths: dict[str, Any], *, cv_folds: int, seed: int
) -> tuple[Any, np.ndarray, dict[str, float]]:
    features, labels, groups = _flatten_development_states(paths)
    unique_groups = np.unique(groups)
    folds = min(int(cv_folds), len(unique_groups))
    if folds < 2:
        raise ValueError("At least two development records are required for group CV.")
    oof = np.empty(labels.shape[0], dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    for train_ids, validation_ids in splitter.split(features, labels, groups):
        model = _make_stopper(seed)
        model.fit(features[train_ids], labels[train_ids])
        oof[validation_ids] = model.predict_proba(features[validation_ids])[:, 1]
    final_model = _make_stopper(seed)
    final_model.fit(features, labels)
    metrics = {
        "state_count": int(labels.size),
        "positive_rate": float(labels.mean()),
        "oof_brier": float(np.mean((oof - labels) ** 2)),
        "oof_accuracy_at_half": float(np.mean((oof >= 0.5) == labels)),
        "group_count": int(unique_groups.size),
        "cv_folds": folds,
    }
    shape = paths["features"][:, :, :-1].shape[:-1]
    return final_model, oof.reshape(shape), metrics


def _select_stage(confidence: np.ndarray, threshold: float) -> np.ndarray:
    eligible = confidence >= threshold
    has_stop = eligible.any(axis=-1)
    first = eligible.argmax(axis=-1)
    return np.where(has_stop, first, confidence.shape[-1]).astype(np.int64)


def _policy_arrays(
    paths: dict[str, Any], budgets: tuple[int, ...], selected_stage: np.ndarray
) -> dict[str, np.ndarray]:
    actions = np.take_along_axis(
        paths["actions"], selected_stage[..., None], axis=-1
    )[..., 0]
    full_actions = paths["actions"][:, :, -1]
    labels = np.broadcast_to(paths["labels"][None, :], actions.shape)
    base = np.broadcast_to(paths["base_token_ids"][None, :], actions.shape)
    samples = np.asarray(budgets, dtype=np.float64)[selected_stage]
    full_correct = full_actions == labels
    chosen_correct = actions == labels
    beneficial = (base != labels) & full_correct
    arrays = {
        "samples": samples,
        "disagreement": (actions != full_actions).astype(np.float64),
        "chosen_correct": chosen_correct.astype(np.float64),
        "full_correct": full_correct.astype(np.float64),
        "accuracy_regret": (
            full_correct.astype(np.float64) - chosen_correct.astype(np.float64)
        ),
        "beneficial": beneficial.astype(np.float64),
        "beneficial_preserved": (beneficial & chosen_correct).astype(np.float64),
    }
    reference_actions = paths.get("reference_actions")
    if reference_actions is not None:
        reference = np.broadcast_to(reference_actions[None, :], actions.shape)
        reference_correct = reference == labels
        reference_beneficial = (base != labels) & reference_correct
        arrays.update(
            {
                "reference_disagreement": (
                    actions != reference
                ).astype(np.float64),
                "anytime_full_reference_disagreement": (
                    full_actions != reference
                ).astype(np.float64),
                "reference_correct": reference_correct.astype(np.float64),
                "reference_accuracy_regret": (
                    reference_correct.astype(np.float64)
                    - chosen_correct.astype(np.float64)
                ),
                "reference_beneficial": reference_beneficial.astype(np.float64),
                "reference_beneficial_preserved": (
                    reference_beneficial & chosen_correct
                ).astype(np.float64),
            }
        )
    return arrays


def _aggregate_metrics(arrays: dict[str, np.ndarray], max_samples: int) -> dict[str, Any]:
    beneficial_total = arrays["beneficial"].sum()
    samples = arrays["samples"]
    metrics = {
        "mean_samples": float(samples.mean()),
        "median_samples": float(np.median(samples)),
        "p90_samples": float(np.quantile(samples, 0.9)),
        "sample_reduction": float(1.0 - samples.mean() / max_samples),
        "action_disagreement": float(arrays["disagreement"].mean()),
        "action_agreement": float(1.0 - arrays["disagreement"].mean()),
        "label_accuracy": float(arrays["chosen_correct"].mean()),
        "full_mc_label_accuracy": float(arrays["full_correct"].mean()),
        "label_accuracy_regret": float(arrays["accuracy_regret"].mean()),
        "beneficial_flip_recall": (
            float(arrays["beneficial_preserved"].sum() / beneficial_total)
            if beneficial_total
            else None
        ),
        "beneficial_flip_replay_instances": int(beneficial_total),
        "beneficial_flip_unique_rows": int(
            arrays["beneficial"].astype(bool).any(axis=0).sum()
        ),
    }
    if "reference_disagreement" in arrays:
        reference_beneficial_total = arrays["reference_beneficial"].sum()
        metrics.update(
            {
                "reference_action_disagreement": float(
                    arrays["reference_disagreement"].mean()
                ),
                "anytime_k50_reference_action_disagreement": float(
                    arrays["anytime_full_reference_disagreement"].mean()
                ),
                "reference_mc50_label_accuracy": float(
                    arrays["reference_correct"].mean()
                ),
                "reference_label_accuracy_regret": float(
                    arrays["reference_accuracy_regret"].mean()
                ),
                "reference_beneficial_flip_recall": (
                    float(
                        arrays["reference_beneficial_preserved"].sum()
                        / reference_beneficial_total
                    )
                    if reference_beneficial_total
                    else None
                ),
                "reference_beneficial_flip_replay_instances": int(
                    reference_beneficial_total
                ),
                "reference_beneficial_flip_unique_rows": int(
                    arrays["reference_beneficial"].astype(bool).any(axis=0).sum()
                ),
            }
        )
    return metrics


def _threshold_candidates(confidence: np.ndarray) -> np.ndarray:
    flat = confidence.reshape(-1)
    quantiles = np.quantile(flat, np.linspace(0.0, 1.0, 501))
    return np.unique(
        np.concatenate(
            (
                np.linspace(0.0, 1.0, 501),
                quantiles,
                np.asarray([np.nextafter(1.0, 2.0)]),
            )
        )
    )


def calibrate_thresholds(
    *,
    paths: dict[str, Any],
    confidence: np.ndarray,
    budgets: tuple[int, ...],
    error_budgets: tuple[float, ...],
    min_beneficial_flip_recall: float = 0.0,
) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {}
    candidates = _threshold_candidates(confidence)
    evaluated = []
    for threshold in candidates:
        stages = _select_stage(confidence, float(threshold))
        arrays = _policy_arrays(paths, budgets, stages)
        metrics = _aggregate_metrics(arrays, budgets[-1])
        evaluated.append((float(threshold), metrics))
    for allowed_error in error_budgets:
        disagreement_field = (
            "reference_action_disagreement"
            if "reference_actions" in paths
            else "action_disagreement"
        )
        recall_field = (
            "reference_beneficial_flip_recall"
            if "reference_actions" in paths
            else "beneficial_flip_recall"
        )
        feasible = [
            (threshold, metrics)
            for threshold, metrics in evaluated
            if metrics[disagreement_field] <= allowed_error
            and (
                metrics[recall_field] is None
                or metrics[recall_field] >= min_beneficial_flip_recall
            )
        ]
        if not feasible:
            selected[f"delta_{allowed_error:g}"] = {
                "status": "infeasible_on_legacy_validation",
                "allowed_action_disagreement": allowed_error,
                "disagreement_metric": disagreement_field,
                "minimum_beneficial_flip_recall": min_beneficial_flip_recall,
            }
            continue
        threshold, metrics = min(
            feasible,
            key=lambda item: (item[1]["mean_samples"], item[0]),
        )
        selected[f"delta_{allowed_error:g}"] = {
            "status": "selected_on_legacy_validation",
            "allowed_action_disagreement": allowed_error,
            "disagreement_metric": disagreement_field,
            "minimum_beneficial_flip_recall": min_beneficial_flip_recall,
            "threshold": threshold,
            **metrics,
        }
    return selected


def _fixed_budget_results(paths: dict[str, Any], budgets: tuple[int, ...]) -> dict:
    results = {}
    for stage, budget in enumerate(budgets):
        selected = np.full(paths["actions"].shape[:2], stage, dtype=np.int64)
        results[str(budget)] = _aggregate_metrics(
            _policy_arrays(paths, budgets, selected), budgets[-1]
        )
    return results


def _cluster_bootstrap_intervals(
    *,
    paths: dict[str, Any],
    arrays: dict[str, np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    record_names = paths["record_names"]
    unique = np.unique(record_names)
    if replicates <= 0:
        return {}
    row_ids = {
        name: np.nonzero(record_names == name)[0] for name in unique
    }
    metric_arrays = {
        "mean_samples": arrays["samples"],
        "action_disagreement": arrays["disagreement"],
        "label_accuracy_regret": arrays["accuracy_regret"],
    }
    if "reference_disagreement" in arrays:
        metric_arrays["reference_action_disagreement"] = arrays[
            "reference_disagreement"
        ]
        metric_arrays["reference_label_accuracy_regret"] = arrays[
            "reference_accuracy_regret"
        ]
    generator = np.random.default_rng(seed)
    draws = {key: [] for key in metric_arrays}
    for _ in range(replicates):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        sampled_rows = np.concatenate([row_ids[name] for name in sampled])
        for key, values in metric_arrays.items():
            # Pool the resampled clusters' rows so the bootstrap targets the
            # same row-weighted estimand as the reported point estimate.
            draws[key].append(float(values[:, sampled_rows].mean()))
    return {
        key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
        for key, values in draws.items()
    }


def _predict_confidence(linear_stopper: dict[str, Any], paths: dict[str, Any]) -> np.ndarray:
    """Score float32 replay states with the inference-visible scalar rule."""

    if (
        linear_stopper.get("schema_version") != 2
        or linear_stopper.get("scoring_protocol") != PORTABLE_STOPPER_PROTOCOL
        or linear_stopper.get("classes") != [0, 1]
    ):
        raise ValueError("Unsupported portable anytime stopper payload.")
    states = paths["features"][:, :, :-1]
    flat = states.reshape(-1, states.shape[-1])
    confidence = np.empty(flat.shape[0], dtype=np.float64)
    for index, features in enumerate(flat):
        confidence[index] = portable_linear_stopper_confidence(
            features,
            scaler_mean=linear_stopper["scaler_mean"],
            scaler_scale=linear_stopper["scaler_scale"],
            coefficient=linear_stopper["coefficient"],
            intercept=linear_stopper["intercept"],
        )
    return confidence.reshape(states.shape[:-1])


def _serialize_linear_stopper(model) -> dict[str, Any]:
    """Export the fitted pipeline as a portable, auditable linear rule."""

    scaler = model.named_steps.get("standardscaler")
    classifier = model.named_steps.get("logisticregression")
    if scaler is None or classifier is None:
        raise TypeError("Expected a StandardScaler + LogisticRegression pipeline.")
    if classifier.coef_.shape != (1, len(FEATURE_NAMES)):
        raise ValueError("The fitted stopper has an unexpected coefficient shape.")
    return {
        "schema_version": 2,
        "scoring_protocol": PORTABLE_STOPPER_PROTOCOL,
        "classes": classifier.classes_.astype(np.int64).tolist(),
        "scaler_mean": scaler.mean_.astype(np.float64).tolist(),
        "scaler_scale": scaler.scale_.astype(np.float64).tolist(),
        "coefficient": classifier.coef_[0].astype(np.float64).tolist(),
        "intercept": float(classifier.intercept_[0]),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.cv_folds <= 1:
        raise ValueError("batch_size must be positive and cv_folds must exceed one.")
    if args.development_replays <= 0 or args.validation_replays <= 0:
        raise ValueError("development/validation replay counts must be positive.")
    if not 0.0 <= args.min_beneficial_flip_recall <= 1.0:
        raise ValueError("--min_beneficial_flip_recall must lie in [0, 1].")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max_rows must be positive when provided.")
    if args.phase == "test" and args.max_rows is not None:
        raise ValueError("--max_rows is prohibited in the locked test phase.")
    error_budgets = _parse_error_budgets(args.error_budgets)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    reference_checkpoint = Path(args.reference_checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "policy_spec.json"
    stopper_path = output_dir / "stopper.joblib"
    output_path = output_dir / (
        "development_results.json" if args.phase == "calibrate" else "test_results.json"
    )
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite result: {output_path}")
    config = _load_checkpoint_config(checkpoint)
    if not (reference_checkpoint / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"Missing reference checkpoint: {reference_checkpoint}")

    frozen_policy_spec = None
    development_cache_dir = Path(args.cache_dir).expanduser().resolve()
    if args.phase == "calibrate":
        if args.test_cache_dir is not None:
            raise ValueError("--test_cache_dir is only valid in the test phase.")
        if policy_path.exists() or stopper_path.exists():
            raise FileExistsError("Refusing to overwrite calibrated policy artifacts.")
        cache_dir = development_cache_dir
    else:
        if args.test_cache_dir is None:
            raise ValueError("--phase test requires a genuinely fresh --test_cache_dir.")
        if not policy_path.is_file() or not stopper_path.is_file():
            raise FileNotFoundError("Run the calibration phase before the test phase.")
        frozen_policy_spec = json.loads(policy_path.read_text(encoding="utf-8"))
        if frozen_policy_spec.get("schema_version") != EVALUATION_SCHEMA_VERSION:
            raise ValueError("Unsupported frozen anytime policy schema version.")
        recorded_payload_hash = frozen_policy_spec.get(POLICY_SPEC_HASH_FIELD)
        if recorded_payload_hash != policy_spec_payload_sha256(frozen_policy_spec):
            raise ValueError("Frozen anytime policy payload hash mismatch.")
        cache_dir = Path(args.test_cache_dir).expanduser().resolve()
        if cache_dir == Path(
            frozen_policy_spec.get("development_cache_dir", "")
        ).expanduser().resolve():
            raise ValueError("Fresh test cache must differ from the development cache.")

    cache = load_anytime_cache(cache_dir)
    budgets = parse_budgets(args.budgets, max_samples=cache.max_samples)
    if list(budgets) != config.get("mc_sample_budgets"):
        raise ValueError("Requested budgets do not match the anytime checkpoint.")
    model = BiasNet.from_pretrained(checkpoint, map_location="cpu")
    reference_model = BiasNet.from_pretrained(
        reference_checkpoint, map_location="cpu"
    )
    parameter_dtypes = {
        str(parameter.dtype).replace("torch.", "")
        for parameter in model.parameters()
    }
    if parameter_dtypes != {BIASNET_PARAMETER_DTYPE}:
        raise ValueError(
            "Anytime calibration requires uniformly float32 BiasNet parameters; "
            f"found {sorted(parameter_dtypes)!r}."
        )
    if reference_model.vocab_size != cache.vocab_size:
        raise ValueError("Reference checkpoint and cache vocabularies differ.")
    if getattr(reference_model.config, "mc_input_representation", None) != "floor_logprob":
        raise ValueError("Reference checkpoint must use floor_logprob features.")
    if (
        getattr(reference_model.config, "mc_base_score_representation", None)
        != "floor_logprob"
    ):
        raise ValueError("Reference checkpoint must use a floor_logprob base.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if BIASNET_AUTOCAST_ENABLED and device.type != "cuda":
        raise ValueError(
            "Anytime calibration requires CUDA because its frozen BiasNet "
            "autocast protocol is enabled."
        )
    model = model.to(device)
    model.set_up_proj()
    model.eval()
    reference_model = reference_model.to(device)
    reference_model.set_up_proj()
    reference_model.eval()

    if args.phase == "calibrate":
        fit_files = config.get("anytime_train_files")
        validation_files = config.get("anytime_test_files")
        if not isinstance(fit_files, list) or not fit_files:
            raise ValueError("Checkpoint has no anytime_train_files split.")
        if not isinstance(validation_files, list) or not validation_files:
            raise ValueError("Checkpoint has no legacy validation split.")
        if set(fit_files) & set(validation_files):
            raise ValueError("Stopper-fit and legacy-validation files overlap.")
        fit_indices = _row_indices_for_records(cache, fit_files)
        validation_indices = _row_indices_for_records(cache, validation_files)
        if args.max_rows is not None:
            fit_indices = fit_indices[: args.max_rows]
            validation_indices = validation_indices[: args.max_rows]
        position_normalizer = int(cache.position_ids[fit_indices].max().item())
        fit_paths = evaluate_replay_paths(
            cache=cache,
            row_indices=fit_indices,
            model=model,
            budgets=budgets,
            replays=args.development_replays,
            replay_seed=args.replay_seed,
            batch_size=args.batch_size,
            device=device,
            position_normalizer=position_normalizer,
        )
        fit_paths["reference_actions"] = evaluate_reference_actions(
            cache=cache,
            row_indices=fit_indices,
            model=reference_model,
            batch_size=args.batch_size,
            device=device,
        )
        validation_paths = evaluate_replay_paths(
            cache=cache,
            row_indices=validation_indices,
            model=model,
            budgets=budgets,
            replays=args.validation_replays,
            replay_seed=args.replay_seed + 1,
            batch_size=args.batch_size,
            device=device,
            position_normalizer=position_normalizer,
        )
        validation_paths["reference_actions"] = evaluate_reference_actions(
            cache=cache,
            row_indices=validation_indices,
            model=reference_model,
            batch_size=args.batch_size,
            device=device,
        )
        stopper, _oof_confidence, stopper_metrics = fit_group_oof_stopper(
            fit_paths, cv_folds=args.cv_folds, seed=args.seed
        )
        linear_stopper = _serialize_linear_stopper(stopper)
        validation_confidence = _predict_confidence(
            linear_stopper, validation_paths
        )
        policies = calibrate_thresholds(
            paths=validation_paths,
            confidence=validation_confidence,
            budgets=budgets,
            error_budgets=error_budgets,
            min_beneficial_flip_recall=args.min_beneficial_flip_recall,
        )
        joblib.dump(stopper, stopper_path)
        policy_spec = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "checkpoint": str(checkpoint),
            "checkpoint_weights_sha256": _sha256_file(
                checkpoint / "pytorch_model.bin"
            ),
            "checkpoint_config_sha256": _sha256_file(checkpoint / "config.json"),
            "reference_checkpoint": str(reference_checkpoint),
            "reference_checkpoint_weights_sha256": _sha256_file(
                reference_checkpoint / "pytorch_model.bin"
            ),
            "reference_checkpoint_config_sha256": _sha256_file(
                reference_checkpoint / "config.json"
            ),
            "stopper_joblib_sha256": _sha256_file(stopper_path),
            "development_cache_dir": str(development_cache_dir),
            "development_cache_manifest_sha256": _sha256_file(
                development_cache_dir / "proxy_mc_manifest.json"
            ),
            "cache_manifest_configuration_fingerprint": cache.metadata.get(
                "manifest_configuration_fingerprint"
            ),
            "estimator_metadata": _estimator_metadata(cache),
            "biasnet_parameter_dtype": BIASNET_PARAMETER_DTYPE,
            "biasnet_autocast_enabled": BIASNET_AUTOCAST_ENABLED,
            "biasnet_autocast_dtype": BIASNET_AUTOCAST_DTYPE,
            "biasnet_runtime_dtype": BIASNET_RUNTIME_DTYPE,
            "budgets": list(budgets),
            "feature_names": list(FEATURE_NAMES),
            "position_normalizer": position_normalizer,
            "stopper_fit_files": fit_files,
            "legacy_validation_files": validation_files,
            "development_replays": args.development_replays,
            "legacy_validation_replays": args.validation_replays,
            "replay_seed": args.replay_seed,
            "stopper": "standard_scaler_logistic_regression_v1",
            "linear_stopper": linear_stopper,
            "stopper_metrics": stopper_metrics,
            "operating_point_semantics": (
                "empirical_legacy_validation_not_a_finite_sample_guarantee"
            ),
            "offline_estimand": (
                "teacher_forced_token_action_preservation_relative_to_original_mc50"
            ),
            "policies": policies,
        }
        policy_spec[POLICY_SPEC_HASH_FIELD] = policy_spec_payload_sha256(
            policy_spec
        )
        policy_path.write_text(
            json.dumps(policy_spec, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        result = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "phase": "calibrate",
            "stopper_fit_records": len(fit_files),
            "stopper_fit_rows": int(fit_indices.numel()),
            "legacy_validation_records": len(validation_files),
            "legacy_validation_rows": int(validation_indices.numel()),
            "development_replays": args.development_replays,
            "legacy_validation_replays": args.validation_replays,
            "stopper_fit_fixed_budget_results": _fixed_budget_results(
                fit_paths, budgets
            ),
            "legacy_validation_fixed_budget_results": _fixed_budget_results(
                validation_paths, budgets
            ),
            "stopper_metrics": stopper_metrics,
            "calibrated_policies": policies,
            POLICY_SPEC_HASH_FIELD: policy_spec[POLICY_SPEC_HASH_FIELD],
        }
    else:
        assert frozen_policy_spec is not None
        policy_spec = frozen_policy_spec
        expected_hashes = {
            "checkpoint_weights_sha256": checkpoint / "pytorch_model.bin",
            "checkpoint_config_sha256": checkpoint / "config.json",
            "reference_checkpoint_weights_sha256": (
                reference_checkpoint / "pytorch_model.bin"
            ),
            "reference_checkpoint_config_sha256": reference_checkpoint / "config.json",
            "stopper_joblib_sha256": stopper_path,
        }
        for field, path in expected_hashes.items():
            if policy_spec.get(field) != _sha256_file(path):
                raise ValueError(f"Frozen policy hash mismatch for {field}.")
        if policy_spec.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Frozen policy feature schema does not match.")
        if (
            policy_spec.get("biasnet_parameter_dtype")
            != BIASNET_PARAMETER_DTYPE
            or policy_spec.get("biasnet_runtime_dtype")
            != BIASNET_PARAMETER_DTYPE
        ):
            raise ValueError(
                "Frozen policy BiasNet parameter dtype does not match evaluation."
            )
        if (
            policy_spec.get("biasnet_autocast_enabled")
            is not BIASNET_AUTOCAST_ENABLED
            or policy_spec.get("biasnet_autocast_dtype")
            != BIASNET_AUTOCAST_DTYPE
        ):
            raise ValueError(
                "Frozen policy BiasNet autocast protocol does not match evaluation."
            )
        if policy_spec.get("budgets") != list(budgets):
            raise ValueError("Frozen policy budget schedule does not match.")
        if policy_spec.get("estimator_metadata") != _estimator_metadata(cache):
            raise ValueError("Fresh test cache estimator protocol does not match.")
        old_names = set(policy_spec.get("stopper_fit_files") or ()) | set(
            policy_spec.get("legacy_validation_files") or ()
        )
        overlap = sorted(old_names & set(cache.record_names))
        if overlap:
            raise ValueError(f"Fresh test cache reuses development records: {overlap}")
        record_names = cache.record_names
        row_indices = torch.arange(cache.num_rows)
        position_normalizer = int(policy_spec["position_normalizer"])
        paths = evaluate_replay_paths(
            cache=cache,
            row_indices=row_indices,
            model=model,
            budgets=budgets,
            replays=args.test_replays,
            replay_seed=args.replay_seed,
            batch_size=args.batch_size,
            device=device,
            position_normalizer=position_normalizer,
        )
        paths["reference_actions"] = evaluate_reference_actions(
            cache=cache,
            row_indices=row_indices,
            model=reference_model,
            batch_size=args.batch_size,
            device=device,
        )
        linear_stopper = policy_spec.get("linear_stopper")
        if not isinstance(linear_stopper, dict):
            raise ValueError("Frozen policy has no portable linear stopper.")
        confidence = _predict_confidence(linear_stopper, paths)
        policy_results = {}
        for policy_name, policy in policy_spec["policies"].items():
            if policy.get("status") != "selected_on_legacy_validation":
                policy_results[policy_name] = {"status": policy.get("status")}
                continue
            stages = _select_stage(confidence, float(policy["threshold"]))
            arrays = _policy_arrays(paths, budgets, stages)
            metrics = _aggregate_metrics(arrays, budgets[-1])
            metrics["cluster_bootstrap_95ci"] = _cluster_bootstrap_intervals(
                paths=paths,
                arrays=arrays,
                replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
            policy_results[policy_name] = metrics
        result = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "phase": "fresh_locked_test",
            "test_cache_dir": str(cache_dir),
            "test_cache_manifest_sha256": _sha256_file(
                cache_dir / "proxy_mc_manifest.json"
            ),
            "test_cache_configuration_fingerprint": cache.metadata.get(
                "manifest_configuration_fingerprint"
            ),
            "records": len(record_names),
            "rows": int(row_indices.numel()),
            "replays": args.test_replays,
            "fixed_budget_results": _fixed_budget_results(paths, budgets),
            "policy_results": policy_results,
        }
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stage": "saved", "path": str(output_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
