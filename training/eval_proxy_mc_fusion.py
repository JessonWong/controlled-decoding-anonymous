"""Calibrate and evaluate proxy--Monte Carlo fusion without API calls.

The target-model samples in each row are split into calibration counts and
held-out counts.  Proxy temperature and Dirichlet prior strength are selected
*only* by held-out target-sample likelihood.  Answer labels are intentionally
kept in a separate diagnostic section and never enter model selection.

The preferred input is produced by ``materialize_proxy_mc_cache.py`` and has
``proxy_logits`` and ``mc_counts`` tensors shaped ``[1, rows, vocab]``.  The
loader also accepts ``proxy_log_probs`` (treated as a temperature-one proxy
distribution unless its metadata declares ``proxy_log_probs_temperature``)
and can recover MC counts from a legacy floor-log-probability cache when proxy
scores are supplied through ``--proxy_dir``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mc_reconstruction import floor_log_probs_to_mc_counts  # noqa: E402


SCHEMA_VERSION = 1
METHOD_MC = "mc_only_floor_smoothed"
METHOD_PROXY = "proxy_only"
METHOD_FUSED = "proxy_dirichlet_fused"
STRATA = ("all", "observed_in_calibration", "unobserved_in_calibration")


def _parse_csv_numbers(value: str, cast, *, name: str) -> list[Any]:
    try:
        parsed = [cast(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid {name}: {value!r}") from error
    if not parsed:
        raise argparse.ArgumentTypeError(f"{name} must not be empty.")
    if len(parsed) != len(set(parsed)):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates.")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline calibration of a local proxy prior against exact target-model "
            "Monte Carlo counts. No API calls or harmful labels are used for selection."
        )
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument(
        "--proxy_dir",
        default=None,
        help=(
            "Optional directory with filename-matched proxy_logits/proxy_log_probs. "
            "Useful when --input_dir is a legacy MC cache."
        ),
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--temperatures",
        default="0.5,0.75,1.0,1.25,1.5,2.0",
        help="Comma-separated positive proxy-temperature grid.",
    )
    parser.add_argument(
        "--prior_strengths",
        "--kappas",
        dest="prior_strengths",
        default="0.25,0.5,1,2,4,8,16,32,64",
        help="Comma-separated positive Dirichlet prior-strength grid.",
    )
    parser.add_argument(
        "--split_seeds",
        default="0,1,2,3,4",
        help="Comma-separated integer seeds for deterministic binomial splits.",
    )
    parser.add_argument("--calibration_fraction", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help=(
            "Device used only for dense proxy normalization. Artifacts are still "
            "streamed one file at a time through CPU memory."
        ),
    )
    parser.add_argument(
        "--mc_observed_alpha",
        type=float,
        default=None,
        help="Override legacy MC observed-token smoothing alpha.",
    )
    parser.add_argument(
        "--mc_floor_mass",
        type=float,
        default=None,
        help="Override legacy MC unseen-vocabulary floor mass.",
    )
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=None)
    return parser.parse_args()


def _validate_positive_grid(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must not be empty.")
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError(f"{name} must contain finite positive values.")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates.")
    return result


def _normalise_rows(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        return tensor[0]
    if tensor.dim() == 2:
        return tensor
    raise ValueError(f"{name} must have shape [1, rows, vocab] or [rows, vocab].")


def _normalise_ids(
    tensor: torch.Tensor | None, name: str, rows: int, vocab_size: int
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor when present.")
    if tensor.dim() == 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() != 1 or tensor.numel() != rows:
        raise ValueError(f"{name} must have shape [1, rows] or [rows].")
    tensor = tensor.long()
    if ((tensor < 0) | (tensor >= vocab_size)).any().item():
        raise ValueError(f"{name} contains IDs outside the shared vocabulary.")
    return tensor


def _metadata_number(
    metadata: Mapping[str, Any], names: Sequence[str], default: float | None
) -> float | None:
    for name in names:
        value = metadata.get(name)
        if value is not None:
            return float(value)
    return default


def _extract_counts(
    payload: Mapping[str, Any],
    path: Path,
    *,
    observed_alpha_override: float | None = None,
    floor_mass_override: float | None = None,
) -> torch.Tensor:
    counts = payload.get("mc_counts")
    if counts is not None:
        counts = _normalise_rows(counts, "mc_counts")
        if counts.dtype == torch.bool or counts.is_complex():
            raise TypeError(f"{path}: mc_counts must contain real integer counts.")
        if counts.is_floating_point():
            if not torch.isfinite(counts).all().item() or not torch.equal(
                counts, counts.round()
            ):
                raise ValueError(f"{path}: mc_counts must be finite and integer-valued.")
        counts = counts.long()
        if (counts < 0).any().item():
            raise ValueError(f"{path}: mc_counts must be non-negative.")
        return counts

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{path}: missing mc_counts and metadata for count recovery.")
    semantics = str(metadata.get("log_probs_semantics", ""))
    if "proxy_dirichlet" in semantics:
        raise ValueError(
            f"{path}: fused log_probs cannot be inverted to MC counts; mc_counts is required."
        )
    log_probs = payload.get("log_probs")
    sample_counts = payload.get("valid_sample_counts")
    observed_alpha = (
        float(observed_alpha_override)
        if observed_alpha_override is not None
        else _metadata_number(
            metadata, ("observed_alpha", "materialized_observed_alpha"), None
        )
    )
    floor_mass = (
        float(floor_mass_override)
        if floor_mass_override is not None
        else _metadata_number(
            metadata, ("floor_mass", "materialized_floor_mass"), None
        )
    )
    if log_probs is None or sample_counts is None:
        raise ValueError(
            f"{path}: legacy recovery requires log_probs and valid_sample_counts."
        )
    if observed_alpha is None or floor_mass is None:
        raise ValueError(f"{path}: legacy recovery metadata lacks alpha/floor mass.")
    rows = _normalise_rows(log_probs, "log_probs")
    return floor_log_probs_to_mc_counts(
        rows,
        sample_counts=sample_counts,
        observed_alpha=observed_alpha,
        floor_mass=floor_mass,
    )


def _extract_proxy_scores(
    payload: Mapping[str, Any], path: Path
) -> tuple[torch.Tensor, str, float]:
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    logits = payload.get("proxy_logits")
    if logits is not None:
        scores = _normalise_rows(logits, "proxy_logits")
        semantics = str(
            metadata.get("proxy_logits_semantics", "uncalibrated_raw_logits")
        )
        source_temperature = 1.0
    else:
        log_probs = payload.get("proxy_log_probs")
        if log_probs is None:
            raise ValueError(
                f"{path}: missing proxy_logits/proxy_log_probs. A fused log_probs "
                "tensor alone cannot be recalibrated."
            )
        scores = _normalise_rows(log_probs, "proxy_log_probs")
        semantics = str(
            metadata.get("proxy_log_probs_semantics", "normalised_proxy_log_probs")
        )
        source_temperature = float(
            metadata.get("proxy_log_probs_temperature", 1.0)
        )
        if not math.isfinite(source_temperature) or source_temperature <= 0.0:
            raise ValueError(f"{path}: invalid proxy_log_probs_temperature.")
    if not scores.is_floating_point():
        raise TypeError(f"{path}: proxy scores must be floating point.")
    if not torch.isfinite(scores).all().item():
        raise ValueError(f"{path}: proxy scores must be finite.")
    return scores, semantics, source_temperature


def deterministic_binomial_split(
    counts: torch.Tensor,
    calibration_fraction: float,
    *,
    seed: int,
    row_key: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split sparse per-token counts reproducibly and independently per row."""

    if counts.dim() != 1:
        raise ValueError("counts must be one-dimensional.")
    if counts.dtype == torch.bool or counts.is_complex():
        raise TypeError("counts must contain integer counts.")
    if counts.is_floating_point() and not torch.equal(counts, counts.round()):
        raise ValueError("counts must be integer-valued.")
    counts = counts.long().cpu()
    if (counts < 0).any().item():
        raise ValueError("counts must be non-negative.")
    calibration_fraction = float(calibration_fraction)
    if not math.isfinite(calibration_fraction) or not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between zero and one.")
    digest = hashlib.sha256(f"{int(seed)}\0{row_key}".encode("utf-8")).digest()
    derived_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed)
    probability = torch.full(counts.shape, calibration_fraction, dtype=torch.float32)
    calibration = torch.binomial(
        counts.float(), probability, generator=generator
    ).long()
    return calibration, counts - calibration


def _safe_metric_value(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


@dataclass
class NLLBucket:
    nll_sum: float = 0.0
    events: int = 0

    def add(self, log_probability: float, count: int = 1) -> None:
        count = int(count)
        if count <= 0:
            return
        self.events += count
        if math.isinf(self.nll_sum):
            return
        if not math.isfinite(log_probability):
            self.nll_sum = math.inf
        else:
            self.nll_sum += -float(log_probability) * count

    def merge(self, other: "NLLBucket") -> None:
        self.events += other.events
        self.nll_sum += other.nll_sum

    def finish(self) -> dict[str, Any]:
        mean_nll = self.nll_sum / self.events if self.events else math.nan
        perplexity = (
            math.exp(mean_nll)
            if math.isfinite(mean_nll) and mean_nll < math.log(sys.float_info.max)
            else math.inf
        )
        return {
            "events": self.events,
            "nll_sum": _safe_metric_value(self.nll_sum),
            "mean_nll": _safe_metric_value(mean_nll),
            "perplexity": _safe_metric_value(perplexity),
            "finite": math.isfinite(mean_nll),
        }


@dataclass
class AgreementBucket:
    examples: int = 0
    top1_matches: int = 0
    topk_matches: int = 0

    def add(self, target: int, predicted: Sequence[int]) -> None:
        self.examples += 1
        self.top1_matches += int(bool(predicted) and int(predicted[0]) == int(target))
        self.topk_matches += int(int(target) in {int(item) for item in predicted})

    def merge(self, other: "AgreementBucket") -> None:
        self.examples += other.examples
        self.top1_matches += other.top1_matches
        self.topk_matches += other.topk_matches

    def finish(self, top_k: int) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "top1_matches": self.top1_matches,
            "top1_agreement": self.top1_matches / self.examples if self.examples else None,
            f"top{top_k}_matches": self.topk_matches,
            f"top{top_k}_agreement": (
                self.topk_matches / self.examples if self.examples else None
            ),
        }


@dataclass
class MethodMetrics:
    predictive: dict[str, NLLBucket] = field(
        default_factory=lambda: {name: NLLBucket() for name in STRATA}
    )
    risk_gate: dict[str, AgreementBucket] = field(
        default_factory=lambda: {name: AgreementBucket() for name in STRATA}
    )
    labels: dict[str, NLLBucket] = field(
        default_factory=lambda: {name: NLLBucket() for name in STRATA}
    )

    def add_predictive(
        self,
        log_probabilities: Sequence[float],
        heldout_counts: Sequence[int],
        observed: Sequence[bool],
    ) -> None:
        for log_probability, count, was_observed in zip(
            log_probabilities, heldout_counts, observed
        ):
            self.predictive["all"].add(float(log_probability), int(count))
            stratum = (
                "observed_in_calibration"
                if bool(was_observed)
                else "unobserved_in_calibration"
            )
            self.predictive[stratum].add(float(log_probability), int(count))

    def add_gate(self, target: int, predicted: Sequence[int], observed: bool) -> None:
        self.risk_gate["all"].add(target, predicted)
        stratum = (
            "observed_in_calibration" if observed else "unobserved_in_calibration"
        )
        self.risk_gate[stratum].add(target, predicted)

    def add_label(self, log_probability: float, observed: bool) -> None:
        self.labels["all"].add(log_probability)
        stratum = (
            "observed_in_calibration" if observed else "unobserved_in_calibration"
        )
        self.labels[stratum].add(log_probability)

    def merge(self, other: "MethodMetrics") -> None:
        for name in STRATA:
            self.predictive[name].merge(other.predictive[name])
            self.risk_gate[name].merge(other.risk_gate[name])
            self.labels[name].merge(other.labels[name])

    def finish_evaluation(self, top_k: int) -> dict[str, Any]:
        return {
            "predictive_count_likelihood": {
                name: self.predictive[name].finish() for name in STRATA
            },
            "deterministic_risk_gate_agreement": {
                name: self.risk_gate[name].finish(top_k) for name in STRATA
            },
        }

    def finish_labels(self) -> dict[str, Any]:
        return {name: self.labels[name].finish() for name in STRATA}


@dataclass
class AcrossSeeds:
    seeds: tuple[int, ...]
    by_seed: dict[int, MethodMetrics] = field(init=False)
    aggregate: MethodMetrics = field(default_factory=MethodMetrics)

    def __post_init__(self) -> None:
        self.by_seed = {seed: MethodMetrics() for seed in self.seeds}

    def metrics(self, seed: int) -> MethodMetrics:
        return self.by_seed[seed]

    def finalise_aggregate(self) -> None:
        self.aggregate = MethodMetrics()
        for seed in self.seeds:
            self.aggregate.merge(self.by_seed[seed])

    def grid_summary(self) -> dict[str, Any]:
        aggregate = self.aggregate.predictive["all"].finish()
        return {
            "selection_mean_nll": aggregate["mean_nll"],
            "selection_perplexity": aggregate["perplexity"],
            "held_out_events": aggregate["events"],
            "finite": aggregate["finite"],
            "mean_nll_by_seed": {
                str(seed): self.by_seed[seed].predictive["all"].finish()["mean_nll"]
                for seed in self.seeds
            },
        }


def _stable_dense_topk(scores: torch.Tensor, k: int) -> list[int]:
    """Top-k with token-ID ascending as an explicit tie break."""

    k = min(int(k), int(scores.numel()))
    if k <= 0:
        return []
    values, _ = torch.topk(scores, k)
    threshold = values[-1]
    strict = torch.nonzero(scores > threshold, as_tuple=False).flatten().tolist()
    strict.sort(key=lambda token_id: (-float(scores[token_id]), int(token_id)))
    needed = k - len(strict)
    ties = torch.nonzero(scores == threshold, as_tuple=False).flatten().tolist()
    ties.sort()
    return [int(item) for item in strict + ties[:needed]]


def _smallest_unobserved(observed_ids: set[int], k: int, vocab_size: int) -> list[int]:
    result: list[int] = []
    token_id = 0
    while len(result) < k and token_id < vocab_size:
        if token_id not in observed_ids:
            result.append(token_id)
        token_id += 1
    return result


def _rank_candidates(scores: Mapping[int, float], k: int) -> list[int]:
    return [
        token_id
        for token_id, _ in sorted(
            scores.items(), key=lambda item: (-float(item[1]), int(item[0]))
        )[:k]
    ]


def _mc_log_probability(
    token_id: int,
    calibration: Mapping[int, int],
    calibration_total: int,
    vocab_size: int,
    observed_alpha: float,
    floor_mass: float,
) -> float:
    if calibration_total == 0:
        return -math.log(vocab_size)
    observed_count = len(calibration)
    count = calibration.get(int(token_id), 0)
    if count > 0:
        return (
            math.log1p(-floor_mass)
            + math.log(count + observed_alpha)
            - math.log(calibration_total + observed_alpha * observed_count)
        )
    if floor_mass == 0.0:
        return -math.inf
    return math.log(floor_mass) - math.log(vocab_size - observed_count)


def _mc_topk(
    calibration: Mapping[int, int],
    calibration_total: int,
    vocab_size: int,
    observed_alpha: float,
    floor_mass: float,
    top_k: int,
) -> list[int]:
    observed_ids = set(calibration)
    candidate_ids = list(observed_ids)
    candidate_ids.extend(_smallest_unobserved(observed_ids, top_k, vocab_size))
    scores = {
        token_id: _mc_log_probability(
            token_id,
            calibration,
            calibration_total,
            vocab_size,
            observed_alpha,
            floor_mass,
        )
        for token_id in candidate_ids
    }
    return _rank_candidates(scores, top_k)


def _fused_log_probability(
    log_q: float,
    count: int,
    calibration_total: int,
    prior_strength: float,
) -> float:
    log_count = math.log(count) if count > 0 else -math.inf
    numerator = float(
        torch.logaddexp(
            torch.tensor(log_count, dtype=torch.float64),
            torch.tensor(math.log(prior_strength) + log_q, dtype=torch.float64),
        ).item()
    )
    return numerator - math.log(calibration_total + prior_strength)


def _fused_topk(
    logits: torch.Tensor,
    log_normalizer: float,
    raw_score_scale: float,
    temperature: float,
    prior_strength: float,
    calibration: Mapping[int, int],
    proxy_top_ids: Sequence[int],
    top_k: int,
) -> list[int]:
    candidate_ids = set(int(item) for item in proxy_top_ids)
    candidate_ids.update(calibration)
    scores: dict[int, float] = {}
    for token_id in candidate_ids:
        log_q = (
            float(logits[token_id]) * raw_score_scale / temperature - log_normalizer
        )
        scores[token_id] = calibration.get(token_id, 0) + prior_strength * math.exp(
            log_q
        )
    return _rank_candidates(scores, top_k)


def _resolve_mc_parameters(
    metadata: Mapping[str, Any],
    observed_alpha_override: float | None,
    floor_mass_override: float | None,
) -> tuple[float, float]:
    observed_alpha = (
        observed_alpha_override
        if observed_alpha_override is not None
        else _metadata_number(
            metadata, ("observed_alpha", "materialized_observed_alpha"), 0.1
        )
    )
    floor_mass = (
        floor_mass_override
        if floor_mass_override is not None
        else _metadata_number(
            metadata, ("floor_mass", "materialized_floor_mass"), 1e-4
        )
    )
    assert observed_alpha is not None and floor_mass is not None
    if not math.isfinite(observed_alpha) or observed_alpha < 0.0:
        raise ValueError("MC observed_alpha must be finite and non-negative.")
    if not math.isfinite(floor_mass) or not 0.0 <= floor_mass < 1.0:
        raise ValueError("MC floor_mass must be in [0, 1).")
    return float(observed_alpha), float(floor_mass)


def _selection_key(item: tuple[Any, AcrossSeeds]) -> tuple[float, ...]:
    config, accumulator = item
    value = accumulator.aggregate.predictive["all"].finish()["mean_nll"]
    objective = float(value) if value is not None else math.inf
    if isinstance(config, tuple):
        return (objective, *(float(part) for part in config))
    return (objective, float(config))


def _manifest_summary(input_dir: Path) -> dict[str, Any] | None:
    for name in ("proxy_mc_manifest.json", "materialization_manifest.json", "cache_manifest.json"):
        path = input_dir / name
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"Unable to read input manifest {path}: {error}") from error
            return {"file": name, "content": value}
    return None


def evaluate_artifacts(
    input_dir: Path | str,
    *,
    temperatures: Sequence[float],
    prior_strengths: Sequence[float],
    split_seeds: Sequence[int],
    calibration_fraction: float = 0.5,
    top_k: int = 5,
    proxy_dir: Path | str | None = None,
    mc_observed_alpha: float | None = None,
    mc_floor_mass: float | None = None,
    max_records: int | None = None,
    max_rows: int | None = None,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    proxy_dir = Path(proxy_dir) if proxy_dir is not None else None
    temperatures = _validate_positive_grid(temperatures, "temperatures")
    prior_strengths = _validate_positive_grid(prior_strengths, "prior_strengths")
    seeds = tuple(int(seed) for seed in split_seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("split_seeds must be non-empty and unique.")
    if not math.isfinite(calibration_fraction) or not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be strictly between zero and one.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when provided.")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when provided.")
    if str(device) == "auto":
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(device)
    if compute_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must resolve to CPU or CUDA.")
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    temperature_tensor = torch.tensor(
        temperatures, device=compute_device, dtype=torch.float32
    )

    files = sorted(input_dir.glob("*.pt"))
    if max_records is not None:
        files = files[:max_records]
    if not files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}.")
    if proxy_dir is not None:
        missing = [path.name for path in files if not (proxy_dir / path.name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} filename-matched proxy artifacts: {missing[:5]}."
            )

    mc_metrics = AcrossSeeds(seeds)
    proxy_metrics = {temperature: AcrossSeeds(seeds) for temperature in temperatures}
    fused_metrics = {
        (temperature, prior_strength): AcrossSeeds(seeds)
        for temperature in temperatures
        for prior_strength in prior_strengths
    }
    records = 0
    rows_seen = 0
    total_target_samples = 0
    gate_rows = 0
    label_rows = 0
    vocab_size_seen: int | None = None
    proxy_semantics_seen: set[str] = set()
    proxy_source_temperatures: set[float] = set()
    mc_parameters_seen: set[tuple[float, float]] = set()
    metadata_identity: dict[str, set[str]] = {
        "model": set(),
        "provider_order": set(),
        "samples_per_token": set(),
        "sample_temperature": set(),
        "top_p": set(),
        "sample_completion_policy": set(),
        "proxy_model_name_or_path": set(),
        "proxy_model_revision": set(),
        "proxy_tokenizer_sha256": set(),
        "target_tokenizer_sha256": set(),
        "target_model": set(),
        "provider": set(),
        "proxy_chat_template_protocol": set(),
        "proxy_logits_dtype": set(),
    }

    stop = False
    for input_path in files:
        payload = torch.load(input_path, map_location="cpu", weights_only=False)
        proxy_payload = (
            torch.load(proxy_dir / input_path.name, map_location="cpu", weights_only=False)
            if proxy_dir is not None
            else payload
        )
        counts = _extract_counts(
            payload,
            input_path,
            observed_alpha_override=mc_observed_alpha,
            floor_mass_override=mc_floor_mass,
        )
        proxy_scores, proxy_semantics, source_temperature = _extract_proxy_scores(
            proxy_payload, proxy_dir / input_path.name if proxy_dir is not None else input_path
        )
        if proxy_scores.shape[0] != counts.shape[0]:
            raise ValueError(f"{input_path}: proxy/count row mismatch.")
        if proxy_scores.shape[1] < counts.shape[1]:
            raise ValueError(f"{input_path}: proxy vocabulary is smaller than mc_counts.")
        vocab_size = int(counts.shape[1])
        proxy_scores = proxy_scores[:, :vocab_size]
        if vocab_size_seen is None:
            vocab_size_seen = vocab_size
        elif vocab_size != vocab_size_seen:
            raise ValueError("All artifacts must use one shared vocabulary size.")

        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        proxy_metadata = proxy_payload.get("metadata")
        proxy_metadata = proxy_metadata if isinstance(proxy_metadata, Mapping) else {}
        observed_alpha, floor_mass = _resolve_mc_parameters(
            metadata, mc_observed_alpha, mc_floor_mass
        )
        mc_parameters_seen.add((observed_alpha, floor_mass))
        proxy_semantics_seen.add(proxy_semantics)
        proxy_source_temperatures.add(source_temperature)
        for key in metadata_identity:
            value = proxy_metadata.get(key, metadata.get(key))
            if value is not None:
                if isinstance(value, (list, tuple, dict)):
                    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False)
                else:
                    rendered = str(value)
                metadata_identity[key].add(rendered)

        rows = counts.shape[0]
        labels = _normalise_ids(payload.get("labels"), "labels", rows, vocab_size)
        gate_ids = _normalise_ids(
            payload.get("risk_gate_token_ids"),
            "risk_gate_token_ids",
            rows,
            vocab_size,
        )
        records += 1
        for row_index in range(rows):
            if max_rows is not None and rows_seen >= max_rows:
                stop = True
                break
            row_counts = counts[row_index]
            observed_ids = torch.nonzero(row_counts > 0, as_tuple=False).flatten()
            observed_counts = row_counts[observed_ids].long()
            row_total = int(observed_counts.sum().item())
            if row_total <= 0:
                raise ValueError(f"{input_path}: row {row_index} has no MC samples.")
            total_target_samples += row_total

            logits = proxy_scores[row_index].float()
            raw_score_scale = source_temperature
            ranking_scores = logits * raw_score_scale
            proxy_top_ids = _stable_dense_topk(ranking_scores, top_k)
            gate_id = int(gate_ids[row_index]) if gate_ids is not None else None
            label_id = int(labels[row_index]) if labels is not None else None
            temperature_context: dict[float, tuple[float, torch.Tensor, float | None, float | None]] = {}
            device_logits = logits.to(compute_device, non_blocking=True)
            # One compact [temperature, vocabulary] operation per row avoids
            # repeatedly launching full-vocabulary normalizations.  Only the
            # <=MC50 observed coordinates and optional diagnostics return to CPU.
            scaled_rows = device_logits.unsqueeze(0) * (
                raw_score_scale / temperature_tensor
            ).unsqueeze(1)
            log_normalizers = torch.logsumexp(scaled_rows, dim=-1)
            device_observed_ids = observed_ids.to(compute_device)
            log_q_observed_rows = (
                scaled_rows.index_select(1, device_observed_ids)
                - log_normalizers.unsqueeze(1)
            ).cpu()
            log_q_gate_rows = (
                (scaled_rows[:, gate_id] - log_normalizers).cpu()
                if gate_id is not None
                else None
            )
            log_q_label_rows = (
                (scaled_rows[:, label_id] - log_normalizers).cpu()
                if label_id is not None
                else None
            )
            log_normalizers_cpu = log_normalizers.cpu()
            for temperature_index, temperature in enumerate(temperatures):
                log_normalizer = float(log_normalizers_cpu[temperature_index].item())
                log_q_observed = log_q_observed_rows[temperature_index]
                log_q_gate = (
                    float(log_q_gate_rows[temperature_index].item())
                    if log_q_gate_rows is not None
                    else None
                )
                log_q_label = (
                    float(log_q_label_rows[temperature_index].item())
                    if log_q_label_rows is not None
                    else None
                )
                temperature_context[temperature] = (
                    log_normalizer,
                    log_q_observed,
                    log_q_gate,
                    log_q_label,
                )

            for seed in seeds:
                calibration_values, heldout_values = deterministic_binomial_split(
                    observed_counts,
                    calibration_fraction,
                    seed=seed,
                    row_key=f"{input_path.name}:{row_index}",
                )
                calibration = {
                    int(token_id): int(count)
                    for token_id, count in zip(observed_ids.tolist(), calibration_values.tolist())
                    if count > 0
                }
                calibration_total = sum(calibration.values())
                heldout_mask = heldout_values > 0
                heldout_ids = observed_ids[heldout_mask]
                heldout_counts = heldout_values[heldout_mask]
                heldout_calibration_counts = calibration_values[heldout_mask]
                heldout_observed = (heldout_calibration_counts > 0).tolist()

                mc_row = mc_metrics.metrics(seed)
                mc_log_probs = [
                    _mc_log_probability(
                        int(token_id),
                        calibration,
                        calibration_total,
                        vocab_size,
                        observed_alpha,
                        floor_mass,
                    )
                    for token_id in heldout_ids.tolist()
                ]
                mc_row.add_predictive(
                    mc_log_probs, heldout_counts.tolist(), heldout_observed
                )
                mc_top_ids = _mc_topk(
                    calibration,
                    calibration_total,
                    vocab_size,
                    observed_alpha,
                    floor_mass,
                    top_k,
                )
                if gate_id is not None:
                    mc_row.add_gate(gate_id, mc_top_ids, gate_id in calibration)
                if label_id is not None:
                    mc_row.add_label(
                        _mc_log_probability(
                            label_id,
                            calibration,
                            calibration_total,
                            vocab_size,
                            observed_alpha,
                            floor_mass,
                        ),
                        label_id in calibration,
                    )

                for temperature in temperatures:
                    (
                        log_normalizer,
                        log_q_observed,
                        log_q_gate,
                        log_q_label,
                    ) = temperature_context[temperature]
                    proxy_row = proxy_metrics[temperature].metrics(seed)
                    proxy_heldout_log_q = log_q_observed[heldout_mask].tolist()
                    proxy_row.add_predictive(
                        proxy_heldout_log_q, heldout_counts.tolist(), heldout_observed
                    )
                    if gate_id is not None:
                        proxy_row.add_gate(
                            gate_id, proxy_top_ids, gate_id in calibration
                        )
                    if label_id is not None:
                        assert log_q_label is not None
                        proxy_row.add_label(log_q_label, label_id in calibration)

                    for prior_strength in prior_strengths:
                        fused_row = fused_metrics[(temperature, prior_strength)].metrics(seed)
                        fused_heldout_log_probs = [
                            _fused_log_probability(
                                float(log_q),
                                int(calibration_count),
                                calibration_total,
                                prior_strength,
                            )
                            for log_q, calibration_count in zip(
                                proxy_heldout_log_q,
                                heldout_calibration_counts.tolist(),
                            )
                        ]
                        fused_row.add_predictive(
                            fused_heldout_log_probs,
                            heldout_counts.tolist(),
                            heldout_observed,
                        )
                        fused_top_ids = _fused_topk(
                            logits,
                            log_normalizer,
                            raw_score_scale,
                            temperature,
                            prior_strength,
                            calibration,
                            proxy_top_ids,
                            top_k,
                        )
                        if gate_id is not None:
                            fused_row.add_gate(
                                gate_id, fused_top_ids, gate_id in calibration
                            )
                        if label_id is not None:
                            assert log_q_label is not None
                            fused_row.add_label(
                                _fused_log_probability(
                                    log_q_label,
                                    calibration.get(label_id, 0),
                                    calibration_total,
                                    prior_strength,
                                ),
                                label_id in calibration,
                            )
            rows_seen += 1
            gate_rows += int(gate_id is not None)
            label_rows += int(label_id is not None)
            del logits, device_logits, scaled_rows, temperature_context
        del payload, proxy_payload, counts, proxy_scores
        if stop:
            break

    for accumulator in [mc_metrics, *proxy_metrics.values(), *fused_metrics.values()]:
        accumulator.finalise_aggregate()
    selected_proxy_temperature, selected_proxy = min(
        proxy_metrics.items(), key=_selection_key
    )
    (selected_temperature, selected_prior_strength), selected_fused = min(
        fused_metrics.items(), key=_selection_key
    )

    def selected_evaluation(accumulator: AcrossSeeds) -> dict[str, Any]:
        return {
            "aggregate_over_splits": accumulator.aggregate.finish_evaluation(top_k),
            "by_split_seed": {
                str(seed): accumulator.by_seed[seed].finish_evaluation(top_k)
                for seed in seeds
            },
        }

    def selected_labels(accumulator: AcrossSeeds) -> dict[str, Any]:
        return {
            "aggregate_over_splits": accumulator.aggregate.finish_labels(),
            "by_split_seed": {
                str(seed): accumulator.by_seed[seed].finish_labels() for seed in seeds
            },
        }

    proxy_grid = []
    for temperature, accumulator in sorted(proxy_metrics.items()):
        proxy_grid.append(
            {"temperature": temperature, **accumulator.grid_summary()}
        )
    fused_grid = []
    for (temperature, prior_strength), accumulator in sorted(fused_metrics.items()):
        fused_grid.append(
            {
                "temperature": temperature,
                "prior_strength": prior_strength,
                **accumulator.grid_summary(),
            }
        )

    input_manifest = _manifest_summary(input_dir)
    result = {
        "schema_version": SCHEMA_VERSION,
        "tool": "training/eval_proxy_mc_fusion.py",
        "selection": {
            "objective": "aggregate_held_out_mc_event_mean_nll",
            "uses_answer_labels": False,
            "uses_risk_gate_token_ids": False,
            "split_method": "per_token_binomial",
            "calibration_fraction": calibration_fraction,
            "split_seeds": list(seeds),
            "selected_proxy_temperature": selected_proxy_temperature,
            "selected_fusion": {
                "proxy_temperature": selected_temperature,
                "prior_strength": selected_prior_strength,
            },
        },
        "data": {
            "input_dir": str(input_dir.resolve()),
            "proxy_dir": str(proxy_dir.resolve()) if proxy_dir is not None else None,
            "records": records,
            "rows": rows_seen,
            "shared_vocab_size": vocab_size_seen,
            "dense_normalization_device": str(compute_device),
            "target_mc_samples": total_target_samples,
            "risk_gate_rows": gate_rows,
            "label_rows": label_rows,
            "proxy_score_semantics": sorted(proxy_semantics_seen),
            "proxy_score_source_temperatures": sorted(proxy_source_temperatures),
            "mc_smoothing_parameters": [
                {"observed_alpha": alpha, "floor_mass": floor}
                for alpha, floor in sorted(mc_parameters_seen)
            ],
            "identity_metadata": {
                key: sorted(values) for key, values in metadata_identity.items() if values
            },
            "input_manifest": input_manifest,
        },
        "method_definitions": {
            METHOD_MC: (
                "Calibration-count estimator using the cache's observed_alpha and "
                "floor_mass; a zero-calibration row falls back to uniform."
            ),
            METHOD_PROXY: "Temperature-scaled local proxy distribution without target counts.",
            METHOD_FUSED: "(calibration_counts + kappa * proxy_probability) / (N_cal + kappa).",
        },
        "evaluation": {
            METHOD_MC: {"hyperparameters": None, **selected_evaluation(mc_metrics)},
            METHOD_PROXY: {
                "hyperparameters": {"proxy_temperature": selected_proxy_temperature},
                **selected_evaluation(selected_proxy),
            },
            METHOD_FUSED: {
                "hyperparameters": {
                    "proxy_temperature": selected_temperature,
                    "prior_strength": selected_prior_strength,
                },
                **selected_evaluation(selected_fused),
            },
        },
        "calibration_grid": {
            "selection_note": (
                "Only held-out target MC event likelihood in this section is used "
                "to choose hyperparameters."
            ),
            METHOD_PROXY: proxy_grid,
            METHOD_FUSED: fused_grid,
        },
        "label_diagnostics": {
            "used_for_selection": False,
            "warning": (
                "Answer-label NLL is post-selection diagnostic evidence only and "
                "must not be used to tune proxy temperature or prior strength."
            ),
            METHOD_MC: selected_labels(mc_metrics),
            METHOD_PROXY: selected_labels(selected_proxy),
            METHOD_FUSED: selected_labels(selected_fused),
        },
    }
    return result


def main() -> None:
    args = parse_args()
    temperatures = _parse_csv_numbers(
        args.temperatures, float, name="temperatures"
    )
    prior_strengths = _parse_csv_numbers(
        args.prior_strengths, float, name="prior_strengths"
    )
    split_seeds = _parse_csv_numbers(args.split_seeds, int, name="split_seeds")
    result = evaluate_artifacts(
        args.input_dir,
        proxy_dir=args.proxy_dir,
        temperatures=temperatures,
        prior_strengths=prior_strengths,
        split_seeds=split_seeds,
        calibration_fraction=args.calibration_fraction,
        top_k=args.top_k,
        mc_observed_alpha=args.mc_observed_alpha,
        mc_floor_mass=args.mc_floor_mass,
        max_records=args.max_records,
        max_rows=args.max_rows,
        device=args.device,
    )
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "output_json": str(output_path.resolve()),
                "rows": result["data"]["rows"],
                "selected_proxy_temperature": result["selection"][
                    "selected_proxy_temperature"
                ],
                "selected_fusion": result["selection"]["selected_fusion"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
