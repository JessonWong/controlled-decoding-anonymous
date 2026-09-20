"""Monte Carlo representations used as BiasNet inputs.

The legacy representation materializes a normalized distribution over the
entire vocabulary and assigns a very small shared probability to every token
that was not sampled.  With a small Monte Carlo budget this makes the input
depend strongly on the sampled support: moving from zero to one observation
can be a many-logit jump.

The ``log_count`` representation instead stores ``log(1 + count / alpha)``.
Unobserved tokens are exactly zero and the zero-to-one jump is bounded by
``log(1 + 1 / alpha)``.  It is not intended to be a calibrated probability
distribution; it is a stable feature representation for BiasNet.
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn.functional as F


FLOOR_LOGPROB = "floor_logprob"
LOG_COUNT = "log_count"
MC_INPUT_REPRESENTATIONS = (FLOOR_LOGPROB, LOG_COUNT)


NO_CHAT_TEMPLATE_SENTINEL = "none"


def chat_template_sha256(tokenizer) -> str:
    """Identity of a tokenizer's chat template.

    A proxy-fused BiasNet learns a residual over features produced by rendering
    prompts with one specific template, so cache build and inference must agree on
    it. Hashing the template gives a comparable identity even when the target is a
    black box whose own tokenizer is unavailable. Tokenizers without a template
    (base models) hash to a fixed sentinel rather than colliding with a real one.
    """

    import hashlib

    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return NO_CHAT_TEMPLATE_SENTINEL
    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()


def fuse_proxy_logits_with_mc_counts(
    proxy_logits: torch.Tensor,
    mc_counts: torch.Tensor,
    *,
    temperature: float = 1.0,
    prior_strength: float = 1.0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fuse dense proxy logits with sparse target-model Monte Carlo counts.

    The proxy distribution is used as the base measure of a Dirichlet prior.
    For target-model counts ``c``, proxy probabilities ``q``, and prior
    strength ``kappa``, the returned posterior predictive distribution is

    ``p_i = (c_i + kappa * q_i) / (sum(c) + kappa)``.

    ``proxy_logits`` may include a padded output-head tail.  Its final axis is
    cropped to the MC/shared vocabulary *before* temperature scaling and
    normalization, so probability mass cannot leak into padded proxy tokens.
    All probability arithmetic is performed in float32 log-space, regardless
    of input or requested output dtype.

    Args:
        proxy_logits: Floating tensor shaped ``[..., proxy_vocab_size]``.
        mc_counts: Non-negative integer-valued tensor shaped
            ``[..., shared_vocab_size]``.  Leading dimensions must exactly
            match ``proxy_logits``.  A row containing no samples is valid and
            returns the calibrated proxy distribution for that row.
        temperature: Finite, positive proxy calibration temperature.
        prior_strength: Finite, positive Dirichlet prior strength ``kappa``.
        dtype: Optional floating output dtype.  Defaults to ``torch.float32``.

    Returns:
        Normalized log probabilities shaped like ``mc_counts``.
    """

    if not isinstance(proxy_logits, torch.Tensor):
        raise TypeError("proxy_logits must be a torch.Tensor.")
    if not isinstance(mc_counts, torch.Tensor):
        raise TypeError("mc_counts must be a torch.Tensor.")
    if proxy_logits.dim() < 1 or mc_counts.dim() < 1:
        raise ValueError(
            "proxy_logits and mc_counts must each have at least one dimension."
        )
    if proxy_logits.shape[:-1] != mc_counts.shape[:-1]:
        raise ValueError(
            "proxy_logits and mc_counts must have identical leading dimensions."
        )

    shared_vocab_size = mc_counts.shape[-1]
    if shared_vocab_size <= 0:
        raise ValueError("mc_counts must have a non-empty shared vocabulary.")
    if proxy_logits.shape[-1] < shared_vocab_size:
        raise ValueError(
            "proxy_logits vocabulary must be at least as large as the shared "
            "mc_counts vocabulary."
        )
    if proxy_logits.device != mc_counts.device:
        raise ValueError("proxy_logits and mc_counts must be on the same device.")
    if not proxy_logits.is_floating_point():
        raise TypeError("proxy_logits must have a floating-point dtype.")
    if mc_counts.dtype == torch.bool or mc_counts.is_complex():
        raise TypeError("mc_counts must contain real integer counts, not bool/complex.")
    if not torch.isfinite(proxy_logits).all().item():
        raise ValueError("proxy_logits must contain only finite values.")

    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a finite positive number.")
    if temperature > torch.finfo(torch.float32).max:
        raise ValueError("temperature must be representable in float32.")
    prior_strength = float(prior_strength)
    if not math.isfinite(prior_strength) or prior_strength <= 0.0:
        raise ValueError("prior_strength must be a finite positive number.")
    if prior_strength > torch.finfo(torch.float32).max:
        raise ValueError("prior_strength must be representable in float32.")

    if dtype is None:
        dtype = torch.float32
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype or None.")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be a floating-point dtype.")

    if mc_counts.is_floating_point() and not torch.isfinite(mc_counts).all().item():
        raise ValueError("mc_counts must contain only finite values.")
    if mc_counts.is_floating_point() and not torch.equal(
        mc_counts, mc_counts.round()
    ):
        raise ValueError("mc_counts must contain integer-valued counts.")
    counts_float = mc_counts.to(dtype=torch.float32)
    if not torch.isfinite(counts_float).all().item():
        raise ValueError("mc_counts must be representable as finite float32 values.")
    if (counts_float < 0).any().item():
        raise ValueError("mc_counts must be non-negative.")

    # Crop first: Qwen output heads may be padded beyond the tokenizer/shared
    # vocabulary, and those tail logits must not participate in normalization.
    shared_proxy_logits = proxy_logits[..., :shared_vocab_size].to(torch.float32)
    scaled_proxy_logits = shared_proxy_logits / temperature
    if not torch.isfinite(scaled_proxy_logits).all().item():
        raise ValueError(
            "temperature scaling produced non-finite proxy logits; use a larger "
            "temperature or finite, lower-magnitude logits."
        )
    proxy_log_probs = F.log_softmax(scaled_proxy_logits, dim=-1)
    if not torch.isfinite(proxy_log_probs).all().item():
        raise ValueError("proxy log-softmax produced non-finite values.")

    log_counts = torch.log(counts_float)
    log_prior_mass = proxy_log_probs + math.log(prior_strength)
    log_numerator = torch.logaddexp(log_counts, log_prior_mass)

    sample_totals = counts_float.sum(dim=-1, keepdim=True)
    denominator = sample_totals + prior_strength
    if not torch.isfinite(denominator).all().item():
        raise ValueError("mc_counts totals are too large for float32 fusion.")
    fused_log_probs = log_numerator - torch.log(denominator)
    if not torch.isfinite(fused_log_probs).all().item():
        raise ValueError("fusion produced non-finite log probabilities.")

    return fused_log_probs.to(dtype=dtype)


def validate_log_count_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not torch.isfinite(torch.tensor(alpha)) or alpha <= 0:
        raise ValueError("log_count_alpha must be a finite positive number.")
    return alpha


def sampled_ids_to_log_counts(
    sampled_token_ids: torch.Tensor,
    vocab_size: int,
    alpha: float = 1.0,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Map sampled ids to zero-baseline smoothed log-count features.

    Args:
        sampled_token_ids: Integer tensor shaped ``[rows, samples]``.
        vocab_size: Size of the tokenizer vocabulary.
        alpha: Pseudocount controlling the zero-to-one feature jump.
        dtype: Output dtype.
    """

    if sampled_token_ids.dim() != 2:
        raise ValueError("sampled_token_ids must have shape [rows, samples].")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1.")
    if sampled_token_ids.numel() == 0 or sampled_token_ids.shape[1] <= 0:
        raise ValueError("sampled_token_ids must contain at least one sample per row.")
    if sampled_token_ids.min().item() < 0 or sampled_token_ids.max().item() >= vocab_size:
        raise ValueError("sampled_token_ids contain ids outside the vocabulary.")
    alpha = validate_log_count_alpha(alpha)

    rows = torch.zeros(
        (sampled_token_ids.shape[0], vocab_size), dtype=torch.float32
    )
    ones = torch.ones_like(sampled_token_ids, dtype=torch.float32, device="cpu")
    rows.scatter_add_(
        1,
        sampled_token_ids.detach().cpu().long(),
        ones,
    )
    rows.div_(alpha).log1p_()
    return rows.to(dtype)


def floor_log_probs_to_mc_counts(
    log_probs: torch.Tensor,
    sample_counts: Union[int, torch.Tensor],
    observed_alpha: float,
    floor_mass: float,
    dtype: torch.dtype = torch.long,
    validate_recovered_counts: bool = True,
) -> torch.Tensor:
    """Recover integer MC counts from a legacy materialized log-prob cache.

    Legacy cache rows are invertible at observed coordinates because they were
    constructed as

    ``p_i = (1-floor_mass) * (count_i+observed_alpha) /
             (N+observed_alpha*K)``.

    The shared minimum value identifies the unseen coordinates.  Counts are
    rounded back to integers, which is robust to the float16 log-prob storage
    used by existing caches.
    """

    if log_probs.dim() < 2:
        raise ValueError("log_probs must have shape [..., vocab_size].")
    observed_alpha = float(observed_alpha)
    if not math.isfinite(observed_alpha) or observed_alpha < 0:
        raise ValueError("observed_alpha must be finite and non-negative.")
    floor_mass = float(floor_mass)
    if not math.isfinite(floor_mass) or floor_mass < 0 or floor_mass >= 1:
        raise ValueError("floor_mass must be in the interval [0, 1).")
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype.")
    dtype_probe = torch.empty((), dtype=dtype)
    if dtype == torch.bool or dtype_probe.is_floating_point() or dtype_probe.is_complex():
        raise TypeError("dtype must be an integer dtype.")

    original_shape = log_probs.shape
    rows = log_probs.detach().float().reshape(-1, original_shape[-1])
    if torch.isnan(rows).any().item() or torch.isposinf(rows).any().item():
        raise ValueError("Legacy log_probs must be finite for count recovery.")
    if floor_mass > 0 and not torch.isfinite(rows).all().item():
        raise ValueError("Legacy log_probs must be finite for count recovery.")

    if isinstance(sample_counts, bool):
        raise ValueError("sample_counts must be positive integers.")
    if isinstance(sample_counts, int):
        if sample_counts <= 0:
            raise ValueError("sample_counts must be positive.")
        per_row_samples = torch.full(
            (rows.shape[0],),
            float(sample_counts),
            dtype=torch.float32,
            device=rows.device,
        )
    else:
        supplied_samples = (
            torch.as_tensor(sample_counts)
            .detach()
            .to(device=rows.device, dtype=torch.float32)
            .reshape(-1)
        )
        if supplied_samples.numel() == 1:
            per_row_samples = supplied_samples.expand(rows.shape[0])
        elif supplied_samples.numel() == rows.shape[0]:
            per_row_samples = supplied_samples
        else:
            raise ValueError(
                "sample_counts must be scalar or have one value per log-prob row."
            )
        if not torch.isfinite(per_row_samples).all() or (per_row_samples <= 0).any():
            raise ValueError("sample_counts must contain finite positive values.")

    if floor_mass == 0:
        observed = torch.isfinite(rows)
    else:
        row_minima = rows.amin(dim=-1, keepdim=True)
        observed = rows > row_minima
    observed_per_row = observed.sum(dim=-1).float()
    if (observed_per_row <= 0).any().item():
        # A one-token support has one observed value above the shared unseen
        # floor.  Therefore no observed coordinate indicates an incompatible
        # cache rather than a valid degenerate MC row.
        raise ValueError(
            "Could not identify observed tokens in one or more legacy cache rows."
        )

    denominator = per_row_samples + observed_alpha * observed_per_row
    # Only exponentiate observed coordinates.  Existing Qwen/GLM MC50 caches
    # contain roughly 1--20 observed tokens in a 150k-token vocabulary, so a
    # dense exp() would spend almost all of its time on the shared floor.
    # Find the sparse support once for the entire batch.  Calling ``nonzero``
    # separately for every row launches one parallel full-vocabulary scan per
    # row.  On large-vocabulary caches (for example 80 x 151,669) that becomes
    # pathologically slow when PyTorch inherits a large CPU thread pool.  A
    # single 2-D nonzero preserves the exact float16 floor comparison while
    # making the remaining arithmetic proportional to observed support size.
    observed_coordinates = torch.nonzero(observed, as_tuple=False)
    observed_rows = observed_coordinates[:, 0]
    observed_tokens = observed_coordinates[:, 1]
    sparse_counts = (
        rows[observed_rows, observed_tokens].exp()
        * denominator[observed_rows]
        / (1.0 - floor_mass)
        - observed_alpha
    ).round_().clamp_min_(0.0)

    if validate_recovered_counts:
        recovered_totals = torch.zeros(
            rows.shape[0], dtype=torch.float32, device=rows.device
        )
        recovered_totals.scatter_add_(0, observed_rows, sparse_counts)
        # Match the legacy validation semantics, which converted each supplied
        # sample count with ``int(...)`` after checking it was positive.
        expected_totals = per_row_samples.trunc()
        mismatched_rows = torch.nonzero(
            recovered_totals != expected_totals, as_tuple=False
        ).flatten()
    else:
        mismatched_rows = torch.empty(0, dtype=torch.long, device=rows.device)

    if mismatched_rows.numel():
        bad_rows = [
            (
                int(row_index),
                int(recovered_totals[row_index].item()),
                int(per_row_samples[row_index].item()),
            )
            for row_index in mismatched_rows[:5].tolist()
        ]
        raise ValueError(
            "Failed to recover exact MC counts from legacy cache rows "
            f"(row, recovered, expected): {bad_rows}."
        )

    recovered_counts = torch.zeros_like(rows)
    recovered_counts[observed_rows, observed_tokens] = sparse_counts

    return recovered_counts.reshape(original_shape).to(dtype)


def floor_log_probs_to_log_counts(
    log_probs: torch.Tensor,
    sample_counts: Union[int, torch.Tensor],
    observed_alpha: float,
    floor_mass: float,
    log_count_alpha: float = 1.0,
    dtype: torch.dtype = torch.float32,
    validate_recovered_counts: bool = True,
) -> torch.Tensor:
    """Recover log-count features from a legacy materialized MC cache."""

    log_count_alpha = validate_log_count_alpha(log_count_alpha)
    counts = floor_log_probs_to_mc_counts(
        log_probs=log_probs,
        sample_counts=sample_counts,
        observed_alpha=observed_alpha,
        floor_mass=floor_mass,
        dtype=torch.long,
        validate_recovered_counts=validate_recovered_counts,
    )
    return torch.log1p(counts.to(torch.float32) / log_count_alpha).to(dtype)
