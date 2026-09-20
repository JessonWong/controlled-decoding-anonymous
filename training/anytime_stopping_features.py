"""Shared, inference-visible feature contract for anytime MC stopping."""

from __future__ import annotations

import math

import torch


FEATURE_NAMES = (
    "k_fraction",
    "log_k_fraction",
    "position_fraction",
    "proxy_top1_probability",
    "proxy_top1_gap",
    "fused_top1_probability",
    "fused_top1_gap",
    "controller_top1_probability",
    "controller_top1_gap",
    "mc_support_fraction",
    "mc_top1_share",
    "mc_top2_gap_share",
    "action_equals_proxy",
    "action_equals_base",
    "action_equals_fused",
    "action_stable_from_previous_stage",
    "stable_stage_fraction",
)


def top2_statistics(scores: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if scores.dim() != 2 or scores.shape[-1] < 2:
        raise ValueError("scores must have shape [batch, vocabulary>=2].")
    top_scores, top_ids = torch.topk(scores.float(), k=2, dim=-1)
    normalizer = torch.logsumexp(scores.float(), dim=-1)
    top_probability = torch.exp(top_scores[:, 0] - normalizer)
    gap = top_scores[:, 0] - top_scores[:, 1]
    return top_ids[:, 0], top_probability, gap


def build_state_features(
    *,
    budget: int,
    max_samples: int,
    positions: torch.Tensor,
    proxy_top_ids: torch.Tensor,
    proxy_top_probability: torch.Tensor,
    proxy_gap: torch.Tensor,
    fused_top_ids: torch.Tensor,
    fused_top_probability: torch.Tensor,
    fused_gap: torch.Tensor,
    action_ids: torch.Tensor,
    action_probability: torch.Tensor,
    action_gap: torch.Tensor,
    base_token_ids: torch.Tensor,
    counts: torch.Tensor,
    previous_actions: torch.Tensor | None,
    stable_stages: torch.Tensor,
    stage_index: int,
    position_normalizer: int,
) -> torch.Tensor:
    if max_samples <= 0 or position_normalizer <= 0:
        raise ValueError("Feature normalizers must be positive.")
    batch = counts.shape[0]
    if budget:
        count_top2 = torch.topk(counts, k=2, dim=-1).values
        support = counts.gt(0).sum(dim=-1).float()
        top1_share = count_top2[:, 0] / float(budget)
        top2_gap_share = (count_top2[:, 0] - count_top2[:, 1]) / float(budget)
    else:
        support = torch.zeros(batch, device=counts.device)
        top1_share = torch.zeros(batch, device=counts.device)
        top2_gap_share = torch.zeros(batch, device=counts.device)
    if previous_actions is None:
        stable = torch.zeros(batch, device=counts.device, dtype=torch.bool)
    else:
        stable = action_ids.eq(previous_actions)
    stable_fraction = stable_stages.float() / max(stage_index, 1)
    k_fraction = torch.full(
        (batch,), float(budget) / float(max_samples), device=counts.device
    )
    log_k_fraction = torch.full(
        (batch,),
        math.log1p(budget) / math.log1p(max_samples),
        device=counts.device,
    )
    features = torch.stack(
        (
            k_fraction,
            log_k_fraction,
            positions.float() / position_normalizer,
            proxy_top_probability,
            proxy_gap,
            fused_top_probability,
            fused_gap,
            action_probability,
            action_gap,
            support / float(max_samples),
            top1_share,
            top2_gap_share,
            action_ids.eq(proxy_top_ids).float(),
            action_ids.eq(base_token_ids).float(),
            action_ids.eq(fused_top_ids).float(),
            stable.float(),
            stable_fraction,
        ),
        dim=-1,
    )
    if features.shape != (batch, len(FEATURE_NAMES)):
        raise RuntimeError("Anytime feature construction produced an invalid shape.")
    return features
