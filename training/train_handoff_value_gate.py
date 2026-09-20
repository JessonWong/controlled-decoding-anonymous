"""Train a development-only base-continuation handoff gate.

The label is the realized outcome after handing the current post-action prefix
to the frozen base continuation policy:

    y(s) = 1[Gemini harmful_score(base_continuation(s)) == 5].

This is deliberately different from the paired outcome-regret estimator.  A
pair contributes two independent post-action states, one per forced token.  We
join judge output by ``source_record_index`` (never file order), keep the
planner's prompt-grouped train/legacy-validation split, and select all model
hyperparameters with group OOF predictions on the train split only.

Two pilots are supported:

* a cheap TF-IDF logistic baseline, always run;
* a deployable LayerNorm+linear head on a frozen causal-LM last-token hidden
  state, enabled with ``--backbone-model`` and ``--risk-checkpoint``.

The latter serializes the same checkpoint layout consumed by ``PrefixRiskGate``.
No generation or judge API is called by this program.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SCHEMA_VERSION = 1
ALLOWED_SPLITS = frozenset({"train", "legacy_validation"})
DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)
LAYER_NORM_EPS = 1e-5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a prompt-grouped base-continuation value/handoff gate."
    )
    parser.add_argument("--judge-input", required=True)
    parser.add_argument("--harmful-score", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples", type=int, default=600)
    parser.add_argument("--c-grid", default=",".join(map(str, DEFAULT_C_GRID)))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--handoff-precision", type=float, default=0.95)
    parser.add_argument(
        "--backbone-model",
        default=None,
        help="Optional local frozen backbone path. Enables the deployable hidden-state head.",
    )
    parser.add_argument(
        "--risk-checkpoint",
        default=None,
        help="Existing PrefixRiskGate checkpoint supplying tokenizer and zero-shot head.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    return parser.parse_args(argv)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_payload(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row at {path}:{line_number} is not an object.")
        rows.append(row)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def parse_c_grid(value: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    else:
        values = tuple(float(item) for item in value)
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise ValueError("C values must be finite and positive.")
    if len(set(values)) != len(values):
        raise ValueError("C values must be unique.")
    return tuple(sorted(values))


def _require_nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string.")
    return value


def build_value_samples(
    judge_input: Sequence[dict[str, Any]],
    harmful_scores: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strictly join each judged continuation to its post-action prefix."""

    scores_by_source: dict[int, dict[str, Any]] = {}
    for score_index, score in enumerate(harmful_scores):
        source_index = score.get("source_record_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(
                f"Harmful-score row {score_index} has invalid source_record_index."
            )
        if source_index < 0 or source_index >= len(judge_input):
            raise ValueError(
                f"Harmful-score row {score_index} points outside judge input."
            )
        if source_index in scores_by_source:
            raise ValueError(f"Duplicate score for source_record_index={source_index}.")
        source = judge_input[source_index]
        if score.get("source_record_sha256") != sha256_payload(source):
            raise ValueError(f"Judge/source hash mismatch at source index {source_index}.")
        if score.get("que") != source.get("prompt") or score.get("ans") != source.get(
            "completion"
        ):
            raise ValueError(f"Judge/source text mismatch at source index {source_index}.")
        harmful_score = score.get("duo_score")
        if (
            isinstance(harmful_score, bool)
            or not isinstance(harmful_score, int)
            or not 1 <= harmful_score <= 5
        ):
            raise ValueError(f"Invalid duo_score at source index {source_index}.")
        scores_by_source[source_index] = score

    expected_indices = set(range(len(judge_input)))
    if set(scores_by_source) != expected_indices:
        missing = sorted(expected_indices - set(scores_by_source))
        raise ValueError(f"Judge output does not exactly cover input: {missing[:10]}.")

    samples: list[dict[str, Any]] = []
    seen_arm_ids: set[str] = set()
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for source_index, source in enumerate(judge_input):
        score = scores_by_source[source_index]
        plan = source.get("plan_row")
        if not isinstance(plan, dict):
            raise ValueError(f"Judge input row {source_index} has no plan_row.")
        split = plan.get("split")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split {split!r} at source index {source_index}.")
        record_name = _require_nonempty_string(
            plan.get("record_name"), context=f"record_name at source index {source_index}"
        )
        arm_id = _require_nonempty_string(
            source.get("arm_id"), context=f"arm_id at source index {source_index}"
        )
        if arm_id in seen_arm_ids:
            raise ValueError(f"Duplicate arm_id: {arm_id}")
        seen_arm_ids.add(arm_id)
        prompt = _require_nonempty_string(
            source.get("prompt"), context=f"prompt at source index {source_index}"
        )
        answer_prefix = _require_nonempty_string(
            source.get("forced_prefix_text"),
            context=f"forced_prefix_text at source index {source_index}",
        )
        completion = _require_nonempty_string(
            source.get("completion"), context=f"completion at source index {source_index}"
        )
        if not completion.startswith(answer_prefix):
            raise ValueError(
                f"Completion does not start with forced prefix at source index {source_index}."
            )
        pair_id = _require_nonempty_string(
            source.get("pair_id"), context=f"pair_id at source index {source_index}"
        )
        if source.get("logibreak_group_id") != pair_id:
            raise ValueError(f"Pair/group mismatch at source index {source_index}.")
        arm_role = source.get("arm_role")
        if arm_role not in {"partial", "reference"}:
            raise ValueError(f"Invalid arm_role at source index {source_index}.")
        action_token_id = int(source.get("action_token_id", -1))
        expected_action = int(plan.get(f"{arm_role}_action", -2))
        if action_token_id < 0 or action_token_id != expected_action:
            raise ValueError(f"Action mismatch at source index {source_index}.")
        base_token_id = int(plan.get("base_token_id", -1))
        duo_score = int(score["duo_score"])
        feature_names = plan.get("feature_names")
        feature_values = plan.get("features")
        if not isinstance(feature_names, list) or not isinstance(feature_values, list):
            raise ValueError(f"Missing frozen state features at source index {source_index}.")
        try:
            position_feature_index = feature_names.index("position_fraction")
            position_fraction = float(feature_values[position_feature_index])
        except (ValueError, IndexError, TypeError) as exc:
            raise ValueError(
                f"Invalid position_fraction feature at source index {source_index}."
            ) from exc
        if not math.isfinite(position_fraction):
            raise ValueError(f"Non-finite position_fraction at source index {source_index}.")
        prefix_audit = source.get("prefix_tokenization_audit")
        if not isinstance(prefix_audit, dict):
            raise ValueError(f"Missing prefix tokenization audit at source index {source_index}.")
        judge_fallback = score.get("judge_input_fallback")
        if not isinstance(judge_fallback, dict):
            raise ValueError(f"Missing judge fallback audit at source index {source_index}.")
        sample = {
            "sample_id": arm_id,
            "branch_state_id": _require_nonempty_string(
                source.get("branch_serialization_sha256"),
                context=f"branch state id at source index {source_index}",
            ),
            "source_record_index": source_index,
            "pair_id": pair_id,
            "candidate_id": str(plan.get("candidate_id", source.get("candidate_id", ""))),
            "split": split,
            "record_name": record_name,
            "dataset_idx": int(plan["dataset_idx"]),
            "position": int(plan["position"]),
            "position_fraction": position_fraction,
            "position_bin": str(plan["position_bin"]),
            "budget": int(plan["budget"]),
            "arm_role": arm_role,
            "action_token_id": action_token_id,
            "base_token_id": base_token_id,
            "action_equals_base": bool(action_token_id == base_token_id),
            "remaining_tokens": int(source["remaining_tokens"]),
            "local_terminal": bool(source["local_terminal"]),
            "forced_prefix_roundtrip": bool(prefix_audit["forced_prefix_roundtrip"]),
            "judge_fallback_used": bool(judge_fallback["used"]),
            "prompt": prompt,
            "answer_prefix": answer_prefix,
            "prefix_characters": len(answer_prefix),
            "prefix_whitespace_tokens": len(answer_prefix.split()),
            "harmful_score": duo_score,
            "jailbreak_outcome": int(duo_score == 5),
        }
        samples.append(sample)
        groups_by_split[str(split)].add(record_name)

    overlap = groups_by_split["train"].intersection(groups_by_split["legacy_validation"])
    if overlap:
        raise ValueError(f"Prompt groups cross frozen splits: {sorted(overlap)[:5]}.")
    if not groups_by_split["train"] or not groups_by_split["legacy_validation"]:
        raise ValueError("Both train and legacy_validation prompt groups are required.")
    return samples


def layer_normalize_rows(features: np.ndarray, eps: float = LAYER_NORM_EPS) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Hidden features must be a finite two-dimensional matrix.")
    means = values.mean(axis=1, keepdims=True, dtype=np.float32)
    variances = np.square(values - means).mean(axis=1, keepdims=True, dtype=np.float32)
    return (values - means) / np.sqrt(variances + np.float32(eps))


def probability_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or labels.shape != scores.shape:
        raise ValueError("Metric arrays must be non-empty and equal length.")
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Scores must be finite probabilities.")
    result: dict[str, Any] = {
        "samples": int(len(labels)),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "brier": float(np.mean(np.square(scores - labels))),
        "log_loss": float(
            -np.mean(
                labels * np.log(np.clip(scores, 1e-12, 1.0))
                + (1 - labels) * np.log(np.clip(1 - scores, 1e-12, 1.0))
            )
        ),
        "mean_prediction": float(scores.mean()),
    }
    if len(np.unique(labels)) < 2:
        result.update({"auroc": None, "auprc": None, "rank_metric_status": "single_class"})
    else:
        result.update(
            {
                "auroc": float(roc_auc_score(labels, scores)),
                "auprc": float(average_precision_score(labels, scores)),
                "rank_metric_status": "defined",
            }
        )
    return result


def decision_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predicted = scores >= float(threshold)
    positives = labels == 1
    tp = int(np.sum(predicted & positives))
    fp = int(np.sum(predicted & ~positives))
    fn = int(np.sum(~predicted & positives))
    tn = int(np.sum(~predicted & ~positives))
    return {
        "threshold": float(threshold),
        "handoff_count": int(predicted.sum()),
        "handoff_rate": float(predicted.mean()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "accuracy": float((tp + tn) / len(labels)),
    }


def select_handoff_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    minimum_precision: float,
    minimum_handoffs: int = 5,
) -> dict[str, Any]:
    if not 0 < minimum_precision <= 1:
        raise ValueError("minimum_precision must be in (0,1].")
    thresholds = np.unique(np.asarray(scores, dtype=np.float64))
    candidates = []
    for threshold in thresholds:
        row = decision_metrics(labels, scores, float(threshold))
        if (
            row["handoff_count"] >= minimum_handoffs
            and row["precision"] is not None
            and row["precision"] >= minimum_precision
        ):
            candidates.append(row)
    if candidates:
        selected = max(
            candidates,
            key=lambda row: (row["recall"], row["handoff_count"], -row["threshold"]),
        )
        return {
            "status": "precision_constraint_satisfied_on_train_oof",
            "minimum_precision": float(minimum_precision),
            **selected,
        }
    fallback_rows = [
        decision_metrics(labels, scores, float(threshold)) for threshold in thresholds
    ]
    fallback = max(
        (row for row in fallback_rows if row["handoff_count"] >= minimum_handoffs),
        key=lambda row: (row["precision"] or -1.0, row["recall"] or -1.0),
    )
    return {
        "status": "precision_constraint_unattainable_fallback_max_precision",
        "minimum_precision": float(minimum_precision),
        **fallback,
    }


def _make_text_estimator(c_value: float, seed: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=20_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=40_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=float(c_value),
        max_iter=5000,
        random_state=int(seed),
        solver="liblinear",
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def _make_hidden_estimator(c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        max_iter=5000,
        random_state=int(seed),
        solver="liblinear",
    )


def fit_grouped_model(
    train_x: Any,
    train_y: np.ndarray,
    groups: np.ndarray,
    validation_x: Any,
    *,
    estimator_factory: Callable[[float, int], Any],
    c_grid: Sequence[float],
    cv_folds: int,
    seed: int,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    train_y = np.asarray(train_y, dtype=np.int64)
    groups = np.asarray(groups, dtype=object)
    if len(train_y) != len(groups) or len(np.unique(train_y)) != 2:
        raise ValueError("Grouped training requires equal-length, two-class labels/groups.")
    unique_groups = np.unique(groups)
    if cv_folds < 2 or len(unique_groups) < cv_folds:
        raise ValueError("Not enough prompt groups for grouped CV.")
    splitter = GroupKFold(n_splits=cv_folds)
    splits = list(splitter.split(np.arange(len(train_y)), train_y, groups))
    candidate_reports = []
    predictions: dict[float, np.ndarray] = {}
    for c_value in c_grid:
        oof = np.full(len(train_y), np.nan, dtype=np.float64)
        folds = []
        for fold, (fit_ids, held_ids) in enumerate(splits):
            if len(np.unique(train_y[fit_ids])) != 2:
                raise ValueError(f"Grouped CV fold {fold} training partition is single-class.")
            estimator = estimator_factory(float(c_value), seed + fold)
            estimator.fit(_take_rows(train_x, fit_ids), train_y[fit_ids])
            predicted = estimator.predict_proba(_take_rows(train_x, held_ids))[:, 1]
            oof[held_ids] = predicted
            folds.append(
                {
                    "fold": fold,
                    "train_samples": int(len(fit_ids)),
                    "validation_samples": int(len(held_ids)),
                    "train_groups": int(len(np.unique(groups[fit_ids]))),
                    "validation_groups": int(len(np.unique(groups[held_ids]))),
                    "validation_brier": float(
                        np.mean(np.square(predicted - train_y[held_ids]))
                    ),
                }
            )
        if not np.isfinite(oof).all():
            raise RuntimeError("Grouped CV did not score every train sample.")
        report = {
            "C": float(c_value),
            "oof_brier": float(np.mean(np.square(oof - train_y))),
            "oof_auprc": float(average_precision_score(train_y, oof)),
            "oof_auroc": float(roc_auc_score(train_y, oof)),
            "folds": folds,
        }
        candidate_reports.append(report)
        predictions[float(c_value)] = oof
    selected = min(candidate_reports, key=lambda row: (row["oof_brier"], row["C"]))
    selected_c = float(selected["C"])
    final_model = estimator_factory(selected_c, seed)
    final_model.fit(train_x, train_y)
    validation_scores = final_model.predict_proba(validation_x)[:, 1]
    return final_model, predictions[selected_c], validation_scores, {
        "selection_metric": "prompt_group_oof_brier",
        "group_field": "record_name",
        "cv_folds": int(cv_folds),
        "train_group_count": int(len(unique_groups)),
        "selected_C": selected_c,
        "C_grid": [float(value) for value in c_grid],
        "candidates": candidate_reports,
    }


def _take_rows(values: Any, ids: np.ndarray) -> Any:
    if isinstance(values, np.ndarray):
        return values[ids]
    return [values[int(index)] for index in ids]


def _sample_text(sample: Mapping[str, Any]) -> str:
    return (
        "### Instruction:\n"
        + str(sample["prompt"])
        + "\n\n### Response prefix:\n"
        + str(sample["answer_prefix"])
    )


def extract_hidden_features(
    samples: Sequence[Mapping[str, Any]],
    *,
    backbone_model: Path,
    risk_checkpoint: Path,
    batch_size: int,
    max_length: int | None,
    dtype_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Any]:
    """Extract deployable last-token features and the existing gate baseline."""

    import torch
    from transformers import AutoModel, AutoTokenizer

    from prefix_risk.data import format_prompt_with_prefix
    from prefix_risk.model import RiskHead, RiskHeadConfig

    if not torch.cuda.is_available():
        raise RuntimeError("--backbone-model extraction requires a CUDA GPU.")
    config_path = risk_checkpoint / "risk_head_config.json"
    head_path = risk_checkpoint / "risk_head.pt"
    if not config_path.is_file() or not head_path.is_file():
        raise FileNotFoundError("risk checkpoint must contain config and risk_head.pt")
    old_config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(old_config.get("layer_indices", ())) != (-1,):
        raise ValueError("Pilot extraction currently requires an existing -1-layer gate.")
    tokenizer = AutoTokenizer.from_pretrained(
        risk_checkpoint, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]
    device = torch.device("cuda:0")
    backbone = AutoModel.from_pretrained(
        backbone_model,
        torch_dtype=torch_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    backbone.eval()
    hidden_size = int(backbone.config.hidden_size)
    if hidden_size != int(old_config["hidden_size"]):
        raise ValueError("Backbone hidden size differs from existing risk checkpoint.")
    if int(backbone.config.vocab_size) != len(tokenizer):
        raise ValueError("Backbone/tokenizer vocabulary mismatch.")
    old_head_config = RiskHeadConfig(
        hidden_size=hidden_size,
        layer_indices=(-1,),
        head_hidden_size=int(old_config["head_hidden_size"]),
        dropout=float(old_config["dropout"]),
    )
    old_head = RiskHead(old_head_config).to(device)
    old_state = torch.load(head_path, map_location=device, weights_only=False)
    old_head.load_state_dict(old_state["head_state_dict"])
    old_head.eval()
    effective_max_length = int(max_length or old_config["max_length"])
    texts = [
        format_prompt_with_prefix(
            tokenizer=tokenizer,
            prompt=str(sample["prompt"]),
            answer_prefix=str(sample["answer_prefix"]),
            use_chat_template=bool(old_config.get("use_chat_template", True)),
        )
        for sample in samples
    ]
    hidden_batches: list[np.ndarray] = []
    old_score_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=effective_max_length,
                return_tensors="pt",
            ).to(device)
            outputs = backbone(**encoded, use_cache=False)
            positions = torch.arange(
                encoded["attention_mask"].shape[1], device=device
            ).unsqueeze(0)
            last_indices = (encoded["attention_mask"].long() * positions).max(dim=1).values
            batch_indices = torch.arange(len(last_indices), device=device)
            selected = outputs.last_hidden_state[batch_indices, last_indices]
            hidden_float = selected.float()
            old_scores = torch.sigmoid(old_head(hidden_float))
            hidden_batches.append(hidden_float.cpu().numpy().astype(np.float16))
            old_score_batches.append(old_scores.float().cpu().numpy())
    hidden = np.concatenate(hidden_batches, axis=0)
    old_scores = np.concatenate(old_score_batches, axis=0).astype(np.float64)
    if hidden.shape != (len(samples), hidden_size) or old_scores.shape != (len(samples),):
        raise RuntimeError("Hidden feature extraction returned unexpected shapes.")
    extraction = {
        "backbone_model": str(backbone_model.resolve()),
        "backbone_config_name_or_path": str(backbone.config.name_or_path),
        "risk_checkpoint": str(risk_checkpoint.resolve()),
        "hidden_size": hidden_size,
        "layer_index": -1,
        "max_length": effective_max_length,
        "dtype": dtype_name,
        "batch_size": int(batch_size),
        "format": "checkpoint_chat_template_prompt_plus_post_action_prefix",
    }
    del old_head, backbone
    torch.cuda.empty_cache()
    return hidden, old_scores, extraction, tokenizer


def save_deployable_head(
    output_dir: Path,
    *,
    classifier: LogisticRegression,
    tokenizer: Any,
    backbone_model: Path,
    risk_checkpoint: Path,
    extraction: Mapping[str, Any],
    threshold: float,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from prefix_risk.model import RiskHead, RiskHeadConfig

    hidden_size = int(extraction["hidden_size"])
    if classifier.coef_.shape != (1, hidden_size):
        raise ValueError("Hidden classifier has an unexpected coefficient shape.")
    gate_dir = output_dir / "handoff_gate"
    gate_dir.mkdir()
    head_config = RiskHeadConfig(
        hidden_size=hidden_size,
        layer_indices=(-1,),
        head_hidden_size=0,
        dropout=0.0,
    )
    head = RiskHead(head_config)
    layer_norm = head.net[0]
    linear = head.net[1]
    with torch.no_grad():
        layer_norm.weight.fill_(1.0)
        layer_norm.bias.zero_()
        linear.weight.copy_(torch.from_numpy(classifier.coef_.astype(np.float32)))
        linear.bias.copy_(torch.from_numpy(classifier.intercept_.astype(np.float32)))
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "step": None,
            "metrics": dict(metrics),
            "estimand": "P(base_continuation_harmful_score==5 | post_action_prefix)",
        },
        gate_dir / "risk_head.pt",
    )
    tokenizer.save_pretrained(gate_dir)
    config = {
        "schema_version": SCHEMA_VERSION,
        "model_name": str(backbone_model.resolve()),
        "source_risk_checkpoint": str(risk_checkpoint.resolve()),
        "dataset_name": "outcome_regret_collection_v1_600_arms",
        "split": "planner_train_only",
        "labeled_jsonl": None,
        "layer_indices": [-1],
        "hidden_size": hidden_size,
        "head_hidden_size": 0,
        "dropout": 0.0,
        "max_length": int(extraction["max_length"]),
        "use_chat_template": True,
        "prefixes_per_answer": 1,
        "min_prefix_tokens": 1,
        "min_prefix_ratio": 0.0,
        "max_prefix_ratio": 1.0,
        "estimand": "P(base_continuation_harmful_score==5 | post_action_prefix)",
        "positive_label": "Gemini duo_score == 5",
        "recommended_handoff_threshold": float(threshold),
        "apply_biasnet_when": "score < recommended_handoff_threshold",
        "development_only": True,
    }
    (gate_dir / "risk_head_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "directory": str(gate_dir),
        "risk_head_sha256": sha256_file(gate_dir / "risk_head.pt"),
        "config_sha256": sha256_file(gate_dir / "risk_head_config.json"),
    }


def split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [int(row["jailbreak_outcome"]) for row in rows]
    return {
        "samples": len(rows),
        "prompt_groups": len({str(row["record_name"]) for row in rows}),
        "positive_count": int(sum(labels)),
        "positive_rate": float(np.mean(labels)),
        "harmful_score_counts": {
            str(score): count
            for score, count in sorted(Counter(int(row["harmful_score"]) for row in rows).items())
        },
        "action_equals_base_samples": int(sum(bool(row["action_equals_base"]) for row in rows)),
        "local_terminal_samples": int(sum(bool(row["local_terminal"]) for row in rows)),
        "unique_branch_states": len({str(row["branch_state_id"]) for row in rows}),
    }


def duplicate_state_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["branch_state_id"])].append(row)
    repeated = [group for group in by_state.values() if len(group) > 1]
    conflicts = [
        group
        for group in repeated
        if len({int(row["jailbreak_outcome"]) for row in group}) > 1
    ]
    return {
        "unique_branch_states": len(by_state),
        "repeated_state_groups": len(repeated),
        "rows_in_repeated_state_groups": int(sum(len(group) for group in repeated)),
        "extra_duplicate_rows": int(len(rows) - len(by_state)),
        "binary_label_conflict_groups": len(conflicts),
        "handling": "retain all realized rollouts as observations; group split prevents cross-split state leakage",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    judge_path = Path(args.judge_input).expanduser().resolve()
    score_path = Path(args.harmful_score).expanduser().resolve()
    samples = build_value_samples(read_jsonl(judge_path), read_jsonl(score_path))
    if args.expected_samples > 0 and len(samples) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples} samples, found {len(samples)}."
        )
    c_grid = parse_c_grid(args.c_grid)
    train_ids = np.asarray(
        [index for index, row in enumerate(samples) if row["split"] == "train"],
        dtype=np.int64,
    )
    validation_ids = np.asarray(
        [
            index
            for index, row in enumerate(samples)
            if row["split"] == "legacy_validation"
        ],
        dtype=np.int64,
    )
    labels = np.asarray([row["jailbreak_outcome"] for row in samples], dtype=np.int64)
    groups = np.asarray([row["record_name"] for row in samples], dtype=object)
    train_y = labels[train_ids]
    validation_y = labels[validation_ids]
    texts = [_sample_text(row) for row in samples]

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite handoff-gate output: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        text_model, text_oof, text_validation, text_cv = fit_grouped_model(
            _take_rows(texts, train_ids),
            train_y,
            groups[train_ids],
            _take_rows(texts, validation_ids),
            estimator_factory=_make_text_estimator,
            c_grid=c_grid,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )
        text_threshold = select_handoff_threshold(
            train_y,
            text_oof,
            minimum_precision=args.handoff_precision,
        )
        joblib.dump(text_model, output_dir / "tfidf_model.joblib")
        predictions: dict[str, np.ndarray] = {
            "tfidf_group_oof_or_heldout": np.full(len(samples), np.nan, dtype=np.float64)
        }
        predictions["tfidf_group_oof_or_heldout"][train_ids] = text_oof
        predictions["tfidf_group_oof_or_heldout"][validation_ids] = text_validation
        position_features = np.asarray(
            [[float(row["position_fraction"])] for row in samples], dtype=np.float32
        )
        position_model, position_oof, position_validation, position_cv = fit_grouped_model(
            position_features[train_ids],
            train_y,
            groups[train_ids],
            position_features[validation_ids],
            estimator_factory=_make_hidden_estimator,
            c_grid=c_grid,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )
        position_threshold = select_handoff_threshold(
            train_y,
            position_oof,
            minimum_precision=args.handoff_precision,
        )
        joblib.dump(position_model, output_dir / "position_model.joblib")
        predictions["position_group_oof_or_heldout"] = np.full(
            len(samples), np.nan, dtype=np.float64
        )
        predictions["position_group_oof_or_heldout"][train_ids] = position_oof
        predictions["position_group_oof_or_heldout"][validation_ids] = position_validation
        train_prevalence = float(train_y.mean())
        constant_validation = np.full(len(validation_y), train_prevalence, dtype=np.float64)
        models: dict[str, Any] = {
            "constant_train_prevalence": {
                "train_prevalence": train_prevalence,
                "legacy_validation": probability_metrics(
                    validation_y, constant_validation
                ),
            },
            "tfidf_logistic": {
                "feature_contract": "prompt_plus_post_action_prefix_word1-2_char3-5_tfidf",
                "cross_validation": text_cv,
                "train_oof": probability_metrics(train_y, text_oof),
                "legacy_validation": probability_metrics(validation_y, text_validation),
                "threshold_selection": text_threshold,
                "legacy_validation_at_selected_threshold": decision_metrics(
                    validation_y, text_validation, text_threshold["threshold"]
                ),
            },
            "position_only_logistic": {
                "feature_contract": "single runtime-available position_fraction sanity baseline",
                "cross_validation": position_cv,
                "train_oof": probability_metrics(train_y, position_oof),
                "legacy_validation": probability_metrics(
                    validation_y, position_validation
                ),
                "threshold_selection": position_threshold,
                "legacy_validation_at_selected_threshold": decision_metrics(
                    validation_y,
                    position_validation,
                    position_threshold["threshold"],
                ),
            },
        }
        selection_candidates: dict[str, dict[str, Any]] = {
            "tfidf_logistic": {
                "prediction_key": "tfidf_group_oof_or_heldout",
                "train_oof_brier": models["tfidf_logistic"]["train_oof"]["brier"],
            },
            "position_only_logistic": {
                "prediction_key": "position_group_oof_or_heldout",
                "train_oof_brier": models["position_only_logistic"]["train_oof"]["brier"],
            },
        }
        deployable = None
        extraction = None

        if bool(args.backbone_model) != bool(args.risk_checkpoint):
            raise ValueError(
                "--backbone-model and --risk-checkpoint must be supplied together."
            )
        if args.backbone_model:
            if args.batch_size <= 0:
                raise ValueError("--batch-size must be positive.")
            backbone_path = Path(args.backbone_model).expanduser().resolve()
            risk_path = Path(args.risk_checkpoint).expanduser().resolve()
            hidden, old_scores, extraction, tokenizer = extract_hidden_features(
                samples,
                backbone_model=backbone_path,
                risk_checkpoint=risk_path,
                batch_size=args.batch_size,
                max_length=args.max_length,
                dtype_name=args.dtype,
            )
            np.savez_compressed(
                output_dir / "hidden_features.npz",
                sample_ids=np.asarray([row["sample_id"] for row in samples]),
                hidden=hidden,
                old_gate_scores=old_scores.astype(np.float32),
            )
            normalized = layer_normalize_rows(hidden)
            hidden_model, hidden_oof, hidden_validation, hidden_cv = fit_grouped_model(
                normalized[train_ids],
                train_y,
                groups[train_ids],
                normalized[validation_ids],
                estimator_factory=_make_hidden_estimator,
                c_grid=c_grid,
                cv_folds=args.cv_folds,
                seed=args.seed,
            )
            hidden_threshold = select_handoff_threshold(
                train_y,
                hidden_oof,
                minimum_precision=args.handoff_precision,
            )
            old_validation = old_scores[validation_ids]
            models["old_risk_gate_zero_shot"] = {
                "warning": "Old head was trained on LLM-LAT prefixes and is not a clean independent baseline.",
                "legacy_validation": probability_metrics(validation_y, old_validation),
                "legacy_validation_at_old_threshold_0.1": decision_metrics(
                    validation_y, old_validation, 0.1
                ),
            }
            models["llama_hidden_layernorm_logistic"] = {
                "feature_contract": "frozen_last_token_hidden_then_fixed_LayerNorm",
                "cross_validation": hidden_cv,
                "train_oof": probability_metrics(train_y, hidden_oof),
                "legacy_validation": probability_metrics(
                    validation_y, hidden_validation
                ),
                "threshold_selection": hidden_threshold,
                "legacy_validation_at_selected_threshold": decision_metrics(
                    validation_y,
                    hidden_validation,
                    hidden_threshold["threshold"],
                ),
            }
            predictions["hidden_group_oof_or_heldout"] = np.full(
                len(samples), np.nan, dtype=np.float64
            )
            predictions["hidden_group_oof_or_heldout"][train_ids] = hidden_oof
            predictions["hidden_group_oof_or_heldout"][validation_ids] = hidden_validation
            predictions["old_gate_zero_shot"] = old_scores
            joblib.dump(hidden_model, output_dir / "hidden_model.joblib")
            selection_candidates["llama_hidden_layernorm_logistic"] = {
                "prediction_key": "hidden_group_oof_or_heldout",
                "train_oof_brier": models["llama_hidden_layernorm_logistic"][
                    "train_oof"
                ]["brier"],
            }
            deployable = save_deployable_head(
                output_dir,
                classifier=hidden_model,
                tokenizer=tokenizer,
                backbone_model=backbone_path,
                risk_checkpoint=risk_path,
                extraction=extraction,
                threshold=float(hidden_threshold["threshold"]),
                metrics=models["llama_hidden_layernorm_logistic"],
            )

        selected_model_name = min(
            selection_candidates,
            key=lambda name: (
                selection_candidates[name]["train_oof_brier"],
                name,
            ),
        )
        selected_threshold = models[selected_model_name]["threshold_selection"]
        selected_scores = predictions[
            selection_candidates[selected_model_name]["prediction_key"]
        ]
        base_validation_ids = validation_ids[
            np.asarray(
                [bool(samples[index]["action_equals_base"]) for index in validation_ids]
            )
        ]
        selected_validation_scores = selected_scores[validation_ids]
        base_local_ids = np.asarray(
            [
                local_index
                for local_index, global_index in enumerate(validation_ids)
                if bool(samples[int(global_index)]["action_equals_base"])
            ],
            dtype=np.int64,
        )
        base_subset = {
            "samples": int(len(base_validation_ids)),
            "prompt_groups": int(
                len({samples[int(index)]["record_name"] for index in base_validation_ids})
            ),
            "probability_metrics": probability_metrics(
                validation_y[base_local_ids], selected_validation_scores[base_local_ids]
            ),
            "at_selected_threshold": decision_metrics(
                validation_y[base_local_ids],
                selected_validation_scores[base_local_ids],
                selected_threshold["threshold"],
            ),
        }
        reference_local_ids = np.asarray(
            [
                local_index
                for local_index, global_index in enumerate(validation_ids)
                if samples[int(global_index)]["arm_role"] == "reference"
            ],
            dtype=np.int64,
        )
        reference_subset = {
            "samples": int(len(reference_local_ids)),
            "prompt_groups": int(
                len(
                    {
                        samples[int(validation_ids[index])]["record_name"]
                        for index in reference_local_ids
                    }
                )
            ),
            "probability_metrics": probability_metrics(
                validation_y[reference_local_ids],
                selected_validation_scores[reference_local_ids],
            ),
            "at_selected_threshold": decision_metrics(
                validation_y[reference_local_ids],
                selected_validation_scores[reference_local_ids],
                selected_threshold["threshold"],
            ),
        }
        nonterminal_local_ids = np.asarray(
            [
                local_index
                for local_index, global_index in enumerate(validation_ids)
                if not bool(samples[int(global_index)]["local_terminal"])
            ],
            dtype=np.int64,
        )
        nonterminal_subset = {
            "samples": int(len(nonterminal_local_ids)),
            "prompt_groups": int(
                len(
                    {
                        samples[int(validation_ids[index])]["record_name"]
                        for index in nonterminal_local_ids
                    }
                )
            ),
            "probability_metrics": probability_metrics(
                validation_y[nonterminal_local_ids],
                selected_validation_scores[nonterminal_local_ids],
            ),
            "at_selected_threshold": decision_metrics(
                validation_y[nonterminal_local_ids],
                selected_validation_scores[nonterminal_local_ids],
                selected_threshold["threshold"],
            ),
        }
        for model_name, selection in selection_candidates.items():
            candidate_validation_scores = predictions[selection["prediction_key"]][
                validation_ids
            ]
            candidate_threshold = models[model_name]["threshold_selection"]["threshold"]
            models[model_name]["legacy_validation_subsets"] = {
                "reference_arm": {
                    "probability_metrics": probability_metrics(
                        validation_y[reference_local_ids],
                        candidate_validation_scores[reference_local_ids],
                    ),
                    "at_selected_threshold": decision_metrics(
                        validation_y[reference_local_ids],
                        candidate_validation_scores[reference_local_ids],
                        candidate_threshold,
                    ),
                },
                "action_equals_base": {
                    "probability_metrics": probability_metrics(
                        validation_y[base_local_ids],
                        candidate_validation_scores[base_local_ids],
                    ),
                    "at_selected_threshold": decision_metrics(
                        validation_y[base_local_ids],
                        candidate_validation_scores[base_local_ids],
                        candidate_threshold,
                    ),
                },
            }
        position_bin_subsets: dict[str, Any] = {}
        for position_bin in ("early", "middle", "late"):
            local_ids = np.asarray(
                [
                    local_index
                    for local_index, global_index in enumerate(validation_ids)
                    if samples[int(global_index)]["position_bin"] == position_bin
                ],
                dtype=np.int64,
            )
            position_bin_subsets[position_bin] = {
                "probability_metrics": probability_metrics(
                    validation_y[local_ids], selected_validation_scores[local_ids]
                ),
                "at_selected_threshold": decision_metrics(
                    validation_y[local_ids],
                    selected_validation_scores[local_ids],
                    selected_threshold["threshold"],
                ),
            }
        prediction_rows = []
        for index, sample in enumerate(samples):
            row = dict(sample)
            for name, values in predictions.items():
                row[name] = float(values[index])
            prediction_rows.append(row)
        prediction_path = output_dir / "samples_predictions.jsonl"
        _write_jsonl(prediction_path, prediction_rows)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "analysis": "development_only_base_continuation_handoff_value_gate_v1",
            "estimand": "P(Qwen3-32B/DeepInfra T=0 no-think continuation, within the collector's 80-token total answer budget, receives Gemini duo_score=5 | post-action prefix)",
            "label_definition": "jailbreak_outcome = 1[duo_score == 5]",
            "continuation_policy": "collector target_protocol; one realized continuation per post-action prefix; remaining budget is 80-position-1 tokens",
            "fresh_test_used": False,
            "deployment_ready": False,
            "selected_model": selected_model_name,
            "model_selection": {
                "protocol": "minimum prompt-grouped train OOF Brier; legacy validation not consulted",
                "candidates": selection_candidates,
            },
            "selected_threshold": selected_threshold,
            "train": split_summary([samples[int(index)] for index in train_ids]),
            "legacy_validation": split_summary(
                [samples[int(index)] for index in validation_ids]
            ),
            "legacy_validation_base_action_subset": base_subset,
            "legacy_validation_reference_arm_subset": reference_subset,
            "legacy_validation_nonterminal_sensitivity": nonterminal_subset,
            "legacy_validation_position_bins": position_bin_subsets,
            "all_samples": split_summary(samples),
            "duplicate_state_audit": duplicate_state_audit(samples),
            "data_quality_audit": {
                "local_terminal_samples": int(
                    sum(bool(row["local_terminal"]) for row in samples)
                ),
                "forced_prefix_roundtrip_failures": int(
                    sum(not bool(row["forced_prefix_roundtrip"]) for row in samples)
                ),
                "judge_fallback_samples": int(
                    sum(bool(row["judge_fallback_used"]) for row in samples)
                ),
            },
            "models": models,
            "hidden_extraction": extraction,
            "deployable_checkpoint": deployable,
            "input_files": {
                "judge_input": {"path": str(judge_path), "sha256": sha256_file(judge_path)},
                "harmful_score": {"path": str(score_path), "sha256": sha256_file(score_path)},
            },
            "output_files": {
                "samples_predictions": {
                    "file": prediction_path.name,
                    "sha256": sha256_file(prediction_path),
                },
                "tfidf_model": {
                    "file": "tfidf_model.joblib",
                    "sha256": sha256_file(output_dir / "tfidf_model.joblib"),
                },
                "position_model": {
                    "file": "position_model.joblib",
                    "sha256": sha256_file(output_dir / "position_model.joblib"),
                },
            },
            "seed": int(args.seed),
            "limitations": [
                "The 120-row legacy validation split is not a fresh locked test.",
                "The 600 arms come from only 40 source prompts and mismatch-selected states.",
                "Each state has one deterministic provider continuation and one judge score, so label noise is not estimated.",
                "Eleven local-terminal rows have no continuation call; the nonterminal validation sensitivity excludes them.",
                "The target is bounded by the collector's 80-token total answer budget and changes if deployment uses another budget.",
                "A value gate evaluates handoff readiness; it does not identify the causal benefit of a BiasNet action.",
                "Threshold precision is selected on grouped train OOF and needs on-policy confirmation.",
            ],
        }
        if extraction is not None:
            summary["output_files"].update(
                {
                    "hidden_features": {
                        "file": "hidden_features.npz",
                        "sha256": sha256_file(output_dir / "hidden_features.npz"),
                    },
                    "hidden_model": {
                        "file": "hidden_model.joblib",
                        "sha256": sha256_file(output_dir / "hidden_model.joblib"),
                    },
                }
            )
        summary["summary_payload_sha256"] = sha256_payload(summary)
        _write_json(output_dir / "summary.json", summary)
        return summary
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least two.")
    summary = run(args)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
