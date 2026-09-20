import argparse
import hashlib
import json
import math
import os
from typing import Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoConfig, AutoModelForCausalLM

import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from modeling_biasnet import BiasConfig, BiasNet
from context_encoder import load_context_manifest, load_context_sidecar
from mc_reconstruction import (
    FLOOR_LOGPROB,
    LOG_COUNT,
    MC_INPUT_REPRESENTATIONS,
    floor_log_probs_to_log_counts,
    validate_log_count_alpha,
)


PROXY_MC_FUSION_MODE = "proxy_dirichlet_v1"
PROXY_MC_CONFIG_FIELDS = (
    "mc_fusion_mode",
    "proxy_model_name_or_path",
    "proxy_model_revision",
    "proxy_tokenizer_sha256",
    "proxy_temperature",
    "proxy_prior_strength",
    "proxy_chat_template_protocol",
    "shared_vocab_size",
    "proxy_vocab_size",
    "proxy_vocab_tail_policy",
    "proxy_dtype",
    "proxy_quantization",
)

# Recorded by newer caches only. Propagated when present so a fused checkpoint can
# assert at inference that the proxy renders prompts exactly as it did during the
# cache build; absent on older caches, which fall back to the legacy check.
PROXY_MC_OPTIONAL_FIELDS = ("proxy_chat_template_sha256",)
# Static-prior caches use the same legacy floor_logprob interface but need to
# carry their prior identity into the trained checkpoint so inference cannot
# silently fall back to the raw MC distribution.
STATIC_MC_PRIOR_FIELDS = (
    "mc_static_prior_mode",
    "mc_static_prior_strength",
    "mc_static_prior_path",
    "mc_static_prior_sha256",
    "mc_static_prior_smoothing",
    "mc_static_prior_selection",
)

class CachedLogitsDataset(Dataset):
    """Dataset backed by cached logits generated via pre_logits_openweight.py."""

    def __init__(
        self,
        data_dir: str,
        dtype: torch.dtype = torch.float32,
        max_tokens_per_sample: Optional[int] = None,
        risk_gate_training: str = "none",
        risk_gate_hard_from_scores: bool = False,
        risk_gate_inactive_weight: float = 0.1,
        risk_gate_threshold: float = 0.1,
        risk_gate_soft_temperature: float = 0.05,
        risk_gate_min_scale: float = 0.0,
        risk_gate_warmup_tokens: int = 0,
        always_train_first_n: int = 0,
        base_token_hard_weight: float = 1.0,
        sparse_boundary_start_position: int = 0,
        sparse_boundary_max_per_sample: int = 0,
        sparse_boundary_window_size: int = 1,
        drop_zero_weight_tokens: bool = False,
        mc_input_representation: str = FLOOR_LOGPROB,
        mc_base_score_representation: Optional[str] = None,
        mc_log_count_alpha: float = 1.0,
        file_names: Optional[List[str]] = None,
        context_cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        if risk_gate_training not in {"none", "hard", "soft", "runtime_soft"}:
            raise ValueError(
                "--risk_gate_training must be one of: none, hard, soft, "
                "runtime_soft."
            )
        if risk_gate_hard_from_scores and risk_gate_training != "hard":
            raise ValueError(
                "--risk_gate_hard_from_scores requires "
                "--risk_gate_training hard."
            )
        if risk_gate_inactive_weight < 0:
            raise ValueError("--risk_gate_inactive_weight must be non-negative.")
        if not torch.isfinite(torch.tensor(float(risk_gate_threshold))):
            raise ValueError("--risk_gate_threshold must be finite.")
        if risk_gate_soft_temperature <= 0:
            raise ValueError("--risk_gate_soft_temperature must be positive.")
        if risk_gate_min_scale < 0 or risk_gate_min_scale > 1:
            raise ValueError("--risk_gate_min_scale must be in the interval [0, 1].")
        if risk_gate_warmup_tokens < 0:
            raise ValueError("--risk_gate_warmup_tokens must be non-negative.")
        if always_train_first_n < 0:
            raise ValueError("--always_train_first_n must be non-negative.")
        if always_train_first_n > 0 and risk_gate_training == "none":
            raise ValueError(
                "--always_train_first_n requires --risk_gate_training hard or soft."
            )
        if always_train_first_n > 0 and risk_gate_training == "runtime_soft":
            raise ValueError(
                "Use --risk_gate_warmup_tokens instead of --always_train_first_n "
                "with --risk_gate_training runtime_soft."
            )
        if base_token_hard_weight < 0:
            raise ValueError("--base_token_hard_weight must be non-negative.")
        if sparse_boundary_start_position < 0:
            raise ValueError(
                "--sparse_boundary_start_position must be non-negative."
            )
        if sparse_boundary_max_per_sample < 0:
            raise ValueError(
                "--sparse_boundary_max_per_sample must be non-negative."
            )
        if sparse_boundary_window_size <= 0:
            raise ValueError(
                "--sparse_boundary_window_size must be positive."
            )
        if mc_input_representation not in MC_INPUT_REPRESENTATIONS:
            raise ValueError(
                "--mc_input_representation must be one of: "
                + ", ".join(MC_INPUT_REPRESENTATIONS)
            )
        if mc_base_score_representation is None:
            mc_base_score_representation = mc_input_representation
        if mc_base_score_representation not in MC_INPUT_REPRESENTATIONS:
            raise ValueError(
                "--mc_base_score_representation must be one of: "
                + ", ".join(MC_INPUT_REPRESENTATIONS)
            )
        mc_log_count_alpha = validate_log_count_alpha(mc_log_count_alpha)

        # A global static-prior cache stores its auxiliary unigram vector as a
        # sibling ``.pt`` file.  It is not a training sample (and has no
        # labels), so keep it out of the sample-file inventory.
        available_files = {
            fname: os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if fname.endswith(".pt") and fname != "global_unigram_prior.pt"
        }
        if file_names is None:
            selected_names = sorted(available_files)
        else:
            selected_names = list(file_names)
            if len(selected_names) != len(set(selected_names)):
                raise ValueError("file_names must not contain duplicates.")
            missing = sorted(set(selected_names) - set(available_files))
            if missing:
                raise FileNotFoundError(
                    f"Selected cache files do not exist in {data_dir}: {missing}."
                )
        file_paths: List[str] = [available_files[name] for name in selected_names]
        if not file_paths:
            raise FileNotFoundError(f"No .pt files found in {data_dir}.")

        if max_tokens_per_sample is not None and max_tokens_per_sample <= 0:
            raise ValueError("--sample_length must be a positive integer when provided.")

        self.dtype = dtype
        context_manifest = load_context_manifest(context_cache_dir) if context_cache_dir else None
        self.context_encoder_contract = context_manifest["contract"] if context_manifest else None
        context_chunks = []
        logits_chunks: List[torch.Tensor] = []
        feature_chunks: List[torch.Tensor] = []
        label_chunks: List[torch.Tensor] = []
        base_token_id_chunks: List[torch.Tensor] = []
        position_id_chunks: List[torch.Tensor] = []
        weight_chunks: List[torch.Tensor] = []
        residual_scale_chunks: List[torch.Tensor] = []
        target_observed_chunks: List[torch.Tensor] = []
        risk_mask_chunks: List[torch.Tensor] = []
        primary_mask_chunks: List[torch.Tensor] = []
        base_wrong_chunks: List[torch.Tensor] = []
        sparse_boundary_chunks: List[torch.Tensor] = []
        vocab_size: Optional[int] = None
        mc_sample_count_values: set[int] = set()
        mc_observed_alpha_values: set[float] = set()
        mc_floor_mass_values: set[float] = set()
        mc_sample_temperature_values: set[float] = set()
        mc_top_p_values: set[float] = set()
        mc_completion_policy_values: set[str] = set()
        proxy_fusion_metadata_values: dict[str, set[Any]] = {
            field: set() for field in PROXY_MC_CONFIG_FIELDS + PROXY_MC_OPTIONAL_FIELDS
        }
        static_prior_metadata_values: dict[str, set[Any]] = {
            field: set() for field in STATIC_MC_PRIOR_FIELDS
        }
        saw_proxy_fusion_cache = False
        saw_legacy_cache = False

        for path in file_paths:
            payload = torch.load(path, map_location="cpu")
            source_log_probs = payload["log_probs"]
            labels = payload["labels"]
            if source_log_probs.dim() != 3 or labels.dim() != 2:
                raise ValueError(
                    "Expected log_probs of shape [1, seq_len, vocab] and labels [1, seq_len]."
                )
            if source_log_probs.shape[0] != 1:
                raise ValueError("Only single-sample batches are supported in cached files.")
            context_slice = (
                load_context_sidecar(path, payload, context_cache_dir, context_manifest)
                if context_manifest is not None else None
            )
            if vocab_size is None:
                vocab_size = source_log_probs.shape[2]
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            for field in STATIC_MC_PRIOR_FIELDS:
                value = metadata.get(field)
                if value is not None:
                    if isinstance(value, (list, dict, set)):
                        raise ValueError(
                            f"{path} static-prior metadata {field} must be scalar."
                        )
                    static_prior_metadata_values[field].add(value)
            fusion_mode = metadata.get("mc_fusion_mode")
            if fusion_mode == PROXY_MC_FUSION_MODE:
                saw_proxy_fusion_cache = True
                # Validate the tensor that defines the MC feature before the
                # auxiliary provenance fields.  Besides producing the most
                # actionable error for a corrupt cache, this ensures a fused
                # distribution can never be mistaken for recoverable legacy
                # floor log-probabilities.
                if not isinstance(payload.get("mc_counts"), torch.Tensor):
                    raise ValueError(
                        f"{path} is a {PROXY_MC_FUSION_MODE} cache but has no "
                        "mc_counts tensor."
                    )
                for field in PROXY_MC_CONFIG_FIELDS:
                    value = metadata.get(field)
                    if value is None:
                        raise ValueError(
                            f"{path} proxy fusion metadata is missing {field}."
                        )
                    if isinstance(value, (list, dict, set)):
                        raise ValueError(
                            f"{path} proxy fusion metadata {field} must be scalar."
                        )
                    proxy_fusion_metadata_values[field].add(value)
                for field in PROXY_MC_OPTIONAL_FIELDS:
                    proxy_fusion_metadata_values[field].add(metadata.get(field))
            elif fusion_mode is None:
                saw_legacy_cache = True
            else:
                raise ValueError(
                    f"{path} has unsupported mc_fusion_mode={fusion_mode!r}."
                )
            # Preserve the source argmax before optional feature conversion.
            # This is only used for legacy caches without deterministic base
            # token ids; log-count is monotone on observed coordinates, but
            # keeping the source semantics here is less surprising.
            source_argmax = source_log_probs[0].argmax(dim=-1).long()
            count_scores = None
            needs_mc_estimator_metadata = (
                LOG_COUNT
                in {
                    mc_input_representation,
                    mc_base_score_representation,
                }
                or fusion_mode == PROXY_MC_FUSION_MODE
            )
            if needs_mc_estimator_metadata:
                if not metadata:
                    raise ValueError(
                        f"{path} does not contain metadata required to recover MC counts."
                    )
                observed_alpha = metadata.get("observed_alpha")
                floor_mass = metadata.get("floor_mass")
                valid_sample_counts = payload.get("valid_sample_counts")
                if observed_alpha is None or floor_mass is None:
                    raise ValueError(
                        f"{path} metadata must contain observed_alpha and floor_mass."
                    )
                if valid_sample_counts is None or valid_sample_counts.shape != labels.shape:
                    raise ValueError(
                        f"{path} must contain valid_sample_counts shaped like labels."
                    )
                mc_sample_count_values.update(
                    int(value)
                    for value in torch.unique(valid_sample_counts).tolist()
                )
                mc_observed_alpha_values.add(float(observed_alpha))
                mc_floor_mass_values.add(float(floor_mass))
                if metadata.get("sample_temperature") is None:
                    raise ValueError(f"{path} metadata is missing sample_temperature.")
                if metadata.get("top_p") is None:
                    raise ValueError(f"{path} metadata is missing top_p.")
                if metadata.get("sample_completion_policy") is None:
                    raise ValueError(
                        f"{path} metadata is missing sample_completion_policy."
                    )
                mc_sample_temperature_values.add(
                    float(metadata["sample_temperature"])
                )
                mc_top_p_values.add(float(metadata["top_p"]))
                mc_completion_policy_values.add(
                    str(metadata["sample_completion_policy"])
                )
                if fusion_mode == PROXY_MC_FUSION_MODE:
                    mc_counts = payload.get("mc_counts")
                    if not isinstance(mc_counts, torch.Tensor):
                        raise ValueError(
                            f"{path} is a {PROXY_MC_FUSION_MODE} cache but has no "
                            "mc_counts tensor."
                        )
                    if mc_counts.shape != source_log_probs.shape:
                        raise ValueError(
                            f"{path} mc_counts shape {tuple(mc_counts.shape)} does "
                            f"not match log_probs {tuple(source_log_probs.shape)}."
                        )
                    if mc_counts.dtype == torch.bool or mc_counts.is_floating_point():
                        raise ValueError(f"{path} mc_counts must use an integer dtype.")
                    if (mc_counts < 0).any():
                        raise ValueError(f"{path} mc_counts must be non-negative.")
                    if not torch.equal(
                        mc_counts.sum(dim=-1).cpu().long(),
                        valid_sample_counts.cpu().long(),
                    ):
                        raise ValueError(
                            f"{path} mc_counts totals do not match valid_sample_counts."
                        )
                    if LOG_COUNT in {
                        mc_input_representation,
                        mc_base_score_representation,
                    }:
                        count_scores = torch.log1p(
                            mc_counts.to(torch.float32) / mc_log_count_alpha
                        ).to(dtype)
                elif fusion_mode is None:
                    count_scores = floor_log_probs_to_log_counts(
                        log_probs=source_log_probs,
                        sample_counts=valid_sample_counts,
                        observed_alpha=float(observed_alpha),
                        floor_mass=float(floor_mass),
                        log_count_alpha=mc_log_count_alpha,
                        dtype=dtype,
                    )
            feature_scores = (
                count_scores
                if mc_input_representation == LOG_COUNT
                else source_log_probs
            )
            base_scores = (
                count_scores
                if mc_base_score_representation == LOG_COUNT
                else source_log_probs
            )
            assert feature_scores is not None and base_scores is not None
            feature_slice = feature_scores[0]
            logit_slice = base_scores[0]
            label_slice = labels[0]
            if fusion_mode == PROXY_MC_FUSION_MODE:
                target_observed_slice = mc_counts[0].gather(
                    1, label_slice.unsqueeze(1)
                ).squeeze(1).gt(0)
            elif count_scores is not None:
                target_observed_slice = count_scores[0].gather(
                    1, label_slice.unsqueeze(1)
                ).squeeze(1).gt(0)
            else:
                # Legacy dense caches do not retain exact MC support.
                target_observed_slice = torch.ones_like(
                    label_slice, dtype=torch.bool
                )
            base_token_ids = payload.get("risk_gate_token_ids")
            if base_token_ids is not None:
                if base_token_ids.shape != labels.shape:
                    raise ValueError(
                        f"{path} has risk_gate_token_ids shape "
                        f"{tuple(base_token_ids.shape)}, expected {tuple(labels.shape)}."
                    )
                base_token_id_slice = base_token_ids[0].long()
            else:
                # Open-weight and legacy caches may not carry the deterministic
                # base token. Their cached-logit argmax is the best available
                # competitor for hard-example weighting and margin training.
                base_token_id_slice = source_argmax
            risk_mask = payload.get("risk_gate_mask")
            risk_scores = payload.get("risk_gate_scores")
            derive_gate_from_scores = risk_gate_training == "runtime_soft" or (
                risk_gate_training == "hard" and risk_gate_hard_from_scores
            )
            if derive_gate_from_scores:
                if risk_scores is None:
                    raise ValueError(
                        f"{path} does not contain risk_gate_scores. Regenerate or "
                        "augment the cache with --risk_gate_checkpoint before "
                        "deriving the training gate from scores."
                    )
                if risk_scores.shape != labels.shape:
                    raise ValueError(
                        f"{path} has risk_gate_scores shape "
                        f"{tuple(risk_scores.shape)}, expected {tuple(labels.shape)}."
                    )
                risk_score_slice = risk_scores[0].float()
                if not torch.isfinite(risk_score_slice).all():
                    raise ValueError(f"{path} contains non-finite risk_gate_scores.")
                risk_mask_slice = risk_score_slice < float(risk_gate_threshold)
            elif risk_gate_training != "none":
                if risk_mask is None:
                    raise ValueError(
                        f"{path} does not contain risk_gate_mask. Regenerate the cache "
                        "with --risk_gate_checkpoint before using gated training."
                    )
                if risk_mask.shape != labels.shape:
                    raise ValueError(
                        f"{path} has risk_gate_mask shape {tuple(risk_mask.shape)}, "
                        f"expected {tuple(labels.shape)}."
                    )
                risk_mask_slice = risk_mask[0].bool()
            else:
                risk_mask_slice = torch.ones_like(label_slice, dtype=torch.bool)
            if max_tokens_per_sample is not None:
                slice_len = min(max_tokens_per_sample, logit_slice.shape[0])
                if slice_len == 0:
                    continue
                logit_slice = logit_slice[:slice_len]
                feature_slice = feature_slice[:slice_len]
                label_slice = label_slice[:slice_len]
                target_observed_slice = target_observed_slice[:slice_len]
                base_token_id_slice = base_token_id_slice[:slice_len]
                risk_mask_slice = risk_mask_slice[:slice_len]
                if derive_gate_from_scores:
                    risk_score_slice = risk_score_slice[:slice_len]
            positions = torch.arange(label_slice.numel())
            if context_slice is not None:
                context_chunks.append(context_slice[:label_slice.numel()])
            early_mask = positions < always_train_first_n
            if risk_gate_training == "hard":
                primary_mask = risk_mask_slice | early_mask
                weight_slice = primary_mask.float()
            elif risk_gate_training == "soft":
                inactive = torch.full_like(risk_mask_slice, float(risk_gate_inactive_weight), dtype=torch.float32)
                active = torch.ones_like(risk_mask_slice, dtype=torch.float32)
                weight_slice = torch.where(risk_mask_slice, active, inactive)
                weight_slice = torch.where(early_mask, active, weight_slice)
                primary_mask = risk_mask_slice | early_mask
                residual_scale_slice = torch.ones_like(weight_slice)
            elif risk_gate_training == "runtime_soft":
                residual_scale_slice = torch.sigmoid(
                    (float(risk_gate_threshold) - risk_score_slice)
                    / float(risk_gate_soft_temperature)
                )
                warmup_mask = positions < risk_gate_warmup_tokens
                residual_scale_slice = torch.where(
                    warmup_mask,
                    torch.ones_like(residual_scale_slice),
                    residual_scale_slice,
                )
                residual_scale_slice = torch.where(
                    residual_scale_slice <= float(risk_gate_min_scale),
                    torch.zeros_like(residual_scale_slice),
                    residual_scale_slice,
                )
                primary_mask = residual_scale_slice > 0
                # The residual scale already attenuates the gradient through the
                # biased logits. Weight each runtime-controlled position once
                # instead of applying the scale a second time through the loss.
                weight_slice = primary_mask.float()
            else:
                weight_slice = torch.ones_like(label_slice, dtype=torch.float32)
                primary_mask = torch.ones_like(label_slice, dtype=torch.bool)
                residual_scale_slice = torch.ones_like(weight_slice)
            if risk_gate_training == "hard":
                residual_scale_slice = torch.ones_like(weight_slice)
            base_wrong_slice = base_token_id_slice.ne(label_slice)
            boundary_candidates = (
                base_wrong_slice
                & primary_mask
                & positions.ge(int(sparse_boundary_start_position))
            )
            sparse_boundary_slice = torch.zeros_like(
                boundary_candidates, dtype=torch.bool
            )
            boundary_indices = torch.nonzero(
                boundary_candidates, as_tuple=False
            ).flatten()
            if sparse_boundary_max_per_sample > 0:
                boundary_indices = boundary_indices[
                    : int(sparse_boundary_max_per_sample)
                ]
            for boundary_index in boundary_indices.tolist():
                window_end = min(
                    boundary_index + int(sparse_boundary_window_size),
                    sparse_boundary_slice.numel(),
                )
                sparse_boundary_slice[boundary_index:window_end] = (
                    primary_mask[boundary_index:window_end]
                )
            if base_token_hard_weight != 1.0:
                hard_weights = torch.full_like(
                    weight_slice, float(base_token_hard_weight), dtype=torch.float32
                )
                weight_slice = weight_slice * torch.where(
                    base_wrong_slice,
                    hard_weights,
                    torch.ones_like(weight_slice),
                )
            logits_chunks.append(logit_slice.to(dtype))
            feature_chunks.append(feature_slice.to(dtype))
            label_chunks.append(label_slice.long())
            base_token_id_chunks.append(base_token_id_slice)
            position_id_chunks.append(positions.long())
            weight_chunks.append(weight_slice.float())
            residual_scale_chunks.append(residual_scale_slice.float())
            target_observed_chunks.append(target_observed_slice.bool())
            primary_mask_chunks.append(primary_mask)
            base_wrong_chunks.append(base_wrong_slice)
            sparse_boundary_chunks.append(sparse_boundary_slice)
            if risk_gate_training != "none":
                risk_mask_chunks.append(risk_mask_slice)

        assert vocab_size is not None
        if not logits_chunks:
            raise ValueError(
                "No tokens available after applying --sample_length. "
                "Increase the value or remove the restriction."
            )
        self.vocab_size = vocab_size
        self.logits = torch.cat(logits_chunks, dim=0).contiguous()
        self.feature_logits = torch.cat(feature_chunks, dim=0).contiguous()
        self.context_features = torch.cat(context_chunks).contiguous() if context_chunks else None
        self.labels = torch.cat(label_chunks, dim=0).contiguous()
        self.base_token_ids = torch.cat(base_token_id_chunks, dim=0).contiguous()
        self.position_ids = torch.cat(position_id_chunks, dim=0).contiguous()
        self.weights = torch.cat(weight_chunks, dim=0).contiguous()
        self.residual_scales = torch.cat(
            residual_scale_chunks, dim=0
        ).contiguous()
        self.target_observed = torch.cat(
            target_observed_chunks, dim=0
        ).contiguous()
        self.sparse_boundary_mask = torch.cat(
            sparse_boundary_chunks, dim=0
        ).contiguous()
        self.num_source_tokens = self.labels.size(0)
        self.runtime_controlled_tokens = int(
            self.residual_scales.gt(0).sum().item()
        )
        self.runtime_residual_scale_mean = float(
            self.residual_scales.mean().item()
        )
        if drop_zero_weight_tokens:
            keep = self.weights > 0
            if not keep.any():
                raise ValueError("No positive-weight tokens remain after gated filtering.")
            self.logits = self.logits[keep].contiguous()
            self.feature_logits = self.feature_logits[keep].contiguous()
            if self.context_features is not None:
                self.context_features = self.context_features[keep].contiguous()
            self.labels = self.labels[keep].contiguous()
            self.base_token_ids = self.base_token_ids[keep].contiguous()
            self.position_ids = self.position_ids[keep].contiguous()
            self.weights = self.weights[keep].contiguous()
            self.residual_scales = self.residual_scales[keep].contiguous()
            self.target_observed = self.target_observed[keep].contiguous()
            self.sparse_boundary_mask = self.sparse_boundary_mask[
                keep
            ].contiguous()
        self.num_tokens = self.labels.size(0)
        self.loss_weight_sum = float(self.weights.sum().item())
        self.risk_gate_training = risk_gate_training
        self.risk_gate_hard_from_scores = risk_gate_hard_from_scores
        self.always_train_first_n = always_train_first_n
        self.base_token_hard_weight = base_token_hard_weight
        self.sparse_boundary_start_position = sparse_boundary_start_position
        self.sparse_boundary_max_per_sample = sparse_boundary_max_per_sample
        self.sparse_boundary_window_size = sparse_boundary_window_size
        self.sparse_boundary_tokens = int(
            self.sparse_boundary_mask.sum().item()
        )
        self.mc_input_representation = mc_input_representation
        self.mc_base_score_representation = mc_base_score_representation
        self.mc_log_count_alpha = mc_log_count_alpha
        if saw_proxy_fusion_cache and saw_legacy_cache:
            raise ValueError(
                "Cannot mix proxy-fused and legacy MC caches in one training dataset."
            )
        if saw_proxy_fusion_cache:
            inconsistent_fusion = {
                field: sorted(values, key=str)
                for field, values in proxy_fusion_metadata_values.items()
                if len(values) != 1
            }
            if inconsistent_fusion:
                raise ValueError(
                    "Proxy-fused training requires one consistent fusion "
                    f"configuration across the cache: {inconsistent_fusion}."
                )
            for field, values in proxy_fusion_metadata_values.items():
                setattr(self, field, next(iter(values)))
        else:
            for field in PROXY_MC_CONFIG_FIELDS + PROXY_MC_OPTIONAL_FIELDS:
                setattr(self, field, None)
        inconsistent_static_prior = {
            field: sorted(values, key=str)
            for field, values in static_prior_metadata_values.items()
            if len(values) > 1
        }
        if inconsistent_static_prior:
            raise ValueError(
                "Static-prior training requires one consistent prior "
                f"configuration across the cache: {inconsistent_static_prior}"
            )
        for field, values in static_prior_metadata_values.items():
            setattr(self, field, next(iter(values)) if values else None)
        if (
            LOG_COUNT
            in {
                mc_input_representation,
                mc_base_score_representation,
            }
            or saw_proxy_fusion_cache
        ):
            if len(mc_sample_count_values) != 1:
                raise ValueError(
                    "MC-aware training requires one constant exact MC sample "
                    f"count across the cache; found {sorted(mc_sample_count_values)}."
                )
            self.mc_samples_per_token = next(iter(mc_sample_count_values))
            estimator_fields = {
                "mc_observed_alpha": mc_observed_alpha_values,
                "mc_floor_mass": mc_floor_mass_values,
                "mc_sample_temperature": mc_sample_temperature_values,
                "mc_top_p": mc_top_p_values,
                "mc_completion_policy": mc_completion_policy_values,
            }
            inconsistent = {
                key: sorted(values)
                for key, values in estimator_fields.items()
                if len(values) != 1
            }
            if inconsistent:
                raise ValueError(
                    "MC-aware training requires one consistent MC estimator "
                    f"configuration across the cache: {inconsistent}."
                )
            self.mc_observed_alpha = next(iter(mc_observed_alpha_values))
            self.mc_floor_mass = next(iter(mc_floor_mass_values))
            self.mc_sample_temperature = next(
                iter(mc_sample_temperature_values)
            )
            self.mc_top_p = next(iter(mc_top_p_values))
            self.mc_completion_policy = next(
                iter(mc_completion_policy_values)
            )
        else:
            self.mc_samples_per_token = None
            self.mc_observed_alpha = None
            self.mc_floor_mass = None
            self.mc_sample_temperature = None
            self.mc_top_p = None
            self.mc_completion_policy = None
        primary_mask_all = torch.cat(primary_mask_chunks, dim=0)
        base_wrong_all = torch.cat(base_wrong_chunks, dim=0)
        self.primary_tokens = int(primary_mask_all.sum().item())
        self.primary_base_wrong_tokens = int(
            (primary_mask_all & base_wrong_all).sum().item()
        )
        if risk_mask_chunks:
            risk_mask_all = torch.cat(risk_mask_chunks, dim=0)
            self.risk_active_tokens = int(risk_mask_all.sum().item())
            self.risk_active_rate = self.risk_active_tokens / max(
                self.num_source_tokens, 1
            )
        else:
            self.risk_active_tokens = self.num_tokens
            self.risk_active_rate = 1.0

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(
        self, index: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if index < 0 or index >= self.num_tokens:
            raise IndexError(f"Index {index} out of range for dataset of size {self.num_tokens}.")
        logits = self.logits[index]
        label = self.labels[index]
        weight = self.weights[index]
        base_token_id = self.base_token_ids[index]
        position_id = self.position_ids[index]
        residual_scale = self.residual_scales[index]
        feature_logits = self.feature_logits[index]
        target_observed = self.target_observed[index]
        sparse_boundary = self.sparse_boundary_mask[index]
        row = (
            logits,
            label,
            weight,
            base_token_id,
            position_id,
            residual_scale,
            feature_logits,
            target_observed,
            sparse_boundary,
        )
        return row if self.context_features is None else (*row, self.context_features[index])



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BiasNet on cached logits.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing cached .pt files.")
    parser.add_argument("--context_cache_dir", default=None,
                        help="Verified pre-action context sidecars; enables variant B.")
    parser.add_argument("--eval_context_cache_dir", default=None,
                        help="Context sidecars for a separate --eval_data_dir.")
    parser.add_argument("--context_bottleneck", type=int, default=128)
    parser.add_argument("--context_learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default=None,
        help=(
            "Optional held-out cache directory. It is evaluated before and after "
            "training but never sampled by the optimizer."
        ),
    )
    parser.add_argument(
        "--held_out_file_count",
        type=int,
        default=0,
        help=(
            "Deterministically reserve this many .pt records from --data_dir "
            "for held-out evaluation. Mutually exclusive with --eval_data_dir."
        ),
    )
    parser.add_argument(
        "--held_out_split_seed",
        type=str,
        default="biasnet-heldout-v1",
        help="Stable string seed used to hash cache filenames for the holdout split.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store the trained BiasNet checkpoint.")
    parser.add_argument("--base_model_name_or_path", type=str, default=None, help="Hugging Face model id or local path used to derive hidden/vocab sizes.")
    parser.add_argument("--base_model_revision", type=str, default=None, help="Optional pinned Hugging Face revision for the base model and copied LM head.")
    parser.add_argument("--hidden_size", type=int, default=None, help="Hidden size override. Required if base model config is not supplied.")
    parser.add_argument(
        "--output_head_hidden_size",
        type=int,
        default=None,
        help="Frozen output-head width; inserts a trainable semantic projection when it differs from hidden_size.",
    )
    parser.add_argument(
        "--lm_head_lora_rank",
        type=int,
        default=0,
        help=(
            "Rank of a trainable residual adapter over the frozen BiasNet "
            "LM head. Zero disables the adapter."
        ),
    )
    parser.add_argument(
        "--lm_head_lora_alpha",
        type=float,
        default=None,
        help=(
            "LoRA scaling numerator for the residual LM-head adapter; "
            "defaults to its rank."
        ),
    )
    parser.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size override. Defaults to cached logits vocab size.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_accumulation", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--sample_length",
        type=int,
        default=None,
        help="Limit training to the first N tokens from each cached sample (e.g., 20).",
    )
    parser.add_argument(
        "--risk_gate_training",
        choices=["none", "hard", "soft", "runtime_soft"],
        default="none",
        help=(
            "Use cached risk_gate_mask as token loss weights. "
            "'hard' trains only active gate positions; 'soft' gives inactive "
            "positions a small weight; 'runtime_soft' uses cached continuous "
            "risk scores to reproduce inference-time residual scaling."
        ),
    )
    parser.add_argument(
        "--risk_gate_inactive_weight",
        type=float,
        default=0.1,
        help="Inactive-token loss weight used with --risk_gate_training soft.",
    )
    parser.add_argument(
        "--risk_gate_hard_from_scores",
        action="store_true",
        help=(
            "For hard-gated training, ignore the cached binary mask and derive "
            "it as risk_gate_scores < --risk_gate_threshold. This keeps the "
            "training operating point aligned with inference."
        ),
    )
    parser.add_argument(
        "--risk_gate_threshold",
        type=float,
        default=0.1,
        help=(
            "Risk threshold used to derive residual scales with "
            "--risk_gate_training runtime_soft."
        ),
    )
    parser.add_argument(
        "--risk_gate_soft_temperature",
        type=float,
        default=0.05,
        help=(
            "Sigmoid temperature used with --risk_gate_training runtime_soft."
        ),
    )
    parser.add_argument(
        "--risk_gate_min_scale",
        type=float,
        default=0.0,
        help=(
            "Set runtime-soft residual scales at or below this value to zero, "
            "matching inference-time BiasNet bypass."
        ),
    )
    parser.add_argument(
        "--risk_gate_warmup_tokens",
        type=int,
        default=0,
        help=(
            "Force residual scale 1 for the first N target positions with "
            "--risk_gate_training runtime_soft."
        ),
    )
    parser.add_argument(
        "--always_train_first_n",
        type=int,
        default=0,
        help=(
            "Force the first N positions of every sample to weight 1 before "
            "base-token hard-example weighting. With gated training this forms "
            "the union: first N OR cached gate-active positions."
        ),
    )
    parser.add_argument(
        "--base_token_hard_weight",
        type=float,
        default=1.0,
        help=(
            "Multiply the loss weight where the cached deterministic base token "
            "differs from the target."
        ),
    )
    parser.add_argument(
        "--drop_zero_weight_tokens",
        action="store_true",
        help=(
            "Discard zero-weight tokens after constructing gated weights. This "
            "is loss-equivalent for hard gating and avoids needless projection work."
        ),
    )
    parser.add_argument(
        "--mc_input_representation",
        choices=MC_INPUT_REPRESENTATIONS,
        default=FLOOR_LOGPROB,
        help=(
            "BiasNet input semantics. 'floor_logprob' preserves the legacy "
            "materialized distribution; 'log_count' uses centered "
            "log1p(count/alpha) features recovered from the MC cache."
        ),
    )
    parser.add_argument(
        "--mc_log_count_alpha",
        type=float,
        default=1.0,
        help="Positive alpha for --mc_input_representation=log_count.",
    )
    parser.add_argument(
        "--mc_base_score_representation",
        choices=MC_INPUT_REPRESENTATIONS,
        default=None,
        help=(
            "Fixed score baseline to which BiasNet adds its residual. Defaults "
            "to --mc_input_representation for backward compatibility. Use "
            "floor_logprob to decouple log-count features from decoding scores."
        ),
    )
    parser.add_argument(
        "--input_projection_mode",
        choices=["lm_head_pinv", "count_sketch"],
        default="lm_head_pinv",
        help="Projection from MC feature space into the BiasNet hidden space.",
    )
    parser.add_argument(
        "--input_hidden_normalization",
        choices=["none", "layernorm"],
        default="none",
        help="Optional normalization after the MC feature projection.",
    )
    parser.add_argument("--count_sketch_hashes", type=int, default=4)
    parser.add_argument("--count_sketch_seed", type=int, default=42)
    parser.add_argument(
        "--count_sketch_input_centering",
        choices=["none", "row_min"],
        default="none",
        help=(
            "Optional preprocessing before CountSketch. 'row_min' subtracts "
            "the per-row floor log-probability so unobserved vocabulary "
            "coordinates become zero; it does not alter the cached base scores."
        ),
    )
    parser.add_argument(
        "--ce_loss_weight",
        type=float,
        default=1.0,
        help="Weight of the full-vocabulary cross-entropy objective.",
    )
    parser.add_argument(
        "--ce_first_n_tokens",
        type=int,
        default=None,
        help=(
            "Optional opening-only CE objective. When set, cross-entropy is "
            "applied only at zero-based positions smaller than N; by default "
            "CE applies at every loss-weighted position."
        ),
    )
    parser.add_argument(
        "--margin_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a ranking loss that pushes the target above the cached "
            "base token. Applied only where base token and target differ."
        ),
    )
    parser.add_argument(
        "--margin_value",
        type=float,
        default=1.0,
        help="Required target-over-base logit margin.",
    )
    parser.add_argument(
        "--base_margin_start_position",
        type=int,
        default=0,
        help=(
            "Apply target-vs-base margin loss only at positions greater than "
            "or equal to this zero-based position. The default 0 preserves "
            "the existing all-position behavior."
        ),
    )
    parser.add_argument(
        "--sparse_boundary_start_position",
        type=int,
        default=0,
        help=(
            "First zero-based position eligible for a sparse intervention "
            "boundary. A boundary must also be gate-controlled and have a "
            "deterministic base token different from the target."
        ),
    )
    parser.add_argument(
        "--sparse_boundary_max_per_sample",
        type=int,
        default=0,
        help=(
            "Keep only the first N eligible intervention boundaries in each "
            "cached sequence. Zero keeps every eligible boundary."
        ),
    )
    parser.add_argument(
        "--sparse_boundary_window_size",
        type=int,
        default=1,
        help=(
            "Number of consecutive gate-controlled positions beginning at each "
            "selected intervention boundary. The default 1 preserves the "
            "legacy single-position objective."
        ),
    )
    parser.add_argument(
        "--top1_margin_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for a structured hinge loss that pushes the target above "
            "the highest-scoring non-target token."
        ),
    )
    parser.add_argument(
        "--top1_margin_value",
        type=float,
        default=1.0,
        help="Required target-over-highest-non-target score margin.",
    )
    parser.add_argument(
        "--top1_margin_boundary_only",
        action="store_true",
        help=(
            "Apply target-vs-top1 margin only at the precomputed sparse "
            "intervention boundaries."
        ),
    )
    parser.add_argument(
        "--residual_l2_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for mean squared residual-logit preservation. This can "
            "teach BiasNet to remain inactive away from intervention boundaries."
        ),
    )
    parser.add_argument(
        "--residual_l2_start_position",
        type=int,
        default=0,
        help="First zero-based position eligible for residual L2 preservation.",
    )
    parser.add_argument(
        "--residual_l2_nonboundary_only",
        action="store_true",
        help="Exclude sparse intervention boundaries from residual L2 preservation.",
    )
    parser.add_argument(
        "--candidate_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight of sparse candidate listwise loss. Candidates are the target, "
            "deterministic base token, and top-K cached MC/base tokens."
        ),
    )
    parser.add_argument(
        "--candidate_top_k",
        type=int,
        default=32,
        help="Number of cached-score top candidates used by candidate loss.",
    )
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=None,
        help="Optional BiasNet checkpoint used to initialise all model weights.",
    )
    parser.add_argument(
        "--eval_before_training",
        action="store_true",
        help="Evaluate the initial model on the weighted cached dataset.",
    )
    parser.add_argument(
        "--position_buckets",
        type=int,
        default=0,
        help=(
            "Enable hidden-space position bias with this many buckets. Cached "
            "positions at or beyond the final bucket share that bucket."
        ),
    )
    parser.add_argument(
        "--position_only_epochs",
        type=int,
        default=0,
        help=(
            "For the first N epochs, freeze the existing BiasNet MLP and train "
            "only the position embedding."
        ),
    )
    parser.add_argument(
        "--init_lm_head",
        action="store_true",
        help="Initialise BiasNet.lm_head from the base model weights (deprecated; prefer --lm_head_init).",
    )
    parser.add_argument(
        "--lm_head_init",
        choices=["none", "copy", "optimize", "data_aware", "zeros", "ones"],
        default=None,
        help="Strategy for initialising BiasNet.lm_head. Defaults to 'copy' when --init_lm_head is used, otherwise 'none'.",
    )
    parser.add_argument(
        "--lm_head_optimize_lr",
        type=float,
        default=1e-5,
        help="Learning rate used when --lm_head_init=optimize.",
    )
    parser.add_argument(
        "--lm_head_optimize_batch_size",
        type=int,
        default=1024,
        help="Batch size used when --lm_head_init=optimize.",
    )
    parser.add_argument(
        "--lm_head_optimize_epochs",
        type=int,
        default=1,
        help="Number of epochs for the optimization-based lm_head initialiser.",
    )
    parser.add_argument(
        "--lm_head_optimize_init",
        choices=["random", "zeros", "ones"],
        default="random",
        help="Initial weight pattern used before optimisation when --lm_head_init=optimize.",
    )
    parser.add_argument(
        "--lm_head_data_sample_size",
        type=int,
        default=2048,
        help="Number of cached tokens sampled when --lm_head_init=data_aware.",
    )
    parser.add_argument(
        "--lm_head_data_scale",
        choices=["singular", "normalize", "none"],
        default="singular",
        help="Row scaling applied to PCA components for --lm_head_init=data_aware.",
    )
    parser.add_argument("--mixed_precision", action="store_true", help="Use torch.autocast for mixed precision on CUDA.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def optimize_lm_head_weights(
    vocab_size: int,
    hidden_size: int,
    device: torch.device,
    lr: float,
    batch_size: int,
    epochs: int,
    init_mode: str,
) -> torch.Tensor:
    """Optimise a vocab-sized weight matrix to minimise pairwise cosine similarity."""

    if init_mode == "zeros":
        init_tensor = torch.zeros(vocab_size, hidden_size, device=device, dtype=torch.float32)
    elif init_mode == "ones":
        init_tensor = torch.ones(vocab_size, hidden_size, device=device, dtype=torch.float32)
    elif init_mode == "random":
        init_tensor = torch.randn(vocab_size, hidden_size, device=device, dtype=torch.float32) * 0.01
    else:
        raise ValueError(f"Unsupported lm_head optimisation init mode: {init_mode}")

    weight = nn.Parameter(init_tensor)
    optimizer = torch.optim.Adam([weight], lr=lr)
    total_steps = (vocab_size + batch_size - 1) // batch_size

    for epoch in range(epochs):
        permutation = torch.randperm(vocab_size, device=device)
        for step in range(total_steps):
            start = step * batch_size
            end = min(start + batch_size, vocab_size)
            indices = permutation[start:end]

            optimizer.zero_grad(set_to_none=True)
            batch_vectors = weight[indices]
            normalized = F.normalize(batch_vectors, p=2, dim=1)
            similarity = normalized @ normalized.t()
            similarity.fill_diagonal_(0)
            loss = similarity.pow(2).mean()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        weight.copy_(F.normalize(weight, p=2, dim=1, eps=1e-12))

    return weight.detach().cpu()


def data_aware_lm_head_weights(
    dataset: CachedLogitsDataset,
    hidden_size: int,
    sample_size: int,
    seed: int,
    scale_mode: str,
) -> torch.Tensor:
    """Initialise lm_head using principal components estimated from cached logits."""

    total_tokens = dataset.num_tokens
    if total_tokens == 0:
        raise ValueError("CachedLogitsDataset is empty; cannot derive data-aware lm_head.")
    if sample_size < 2:
        raise ValueError("--lm_head_data_sample_size must be at least 2.")

    sample_size = min(sample_size, total_tokens)
    logits = dataset.logits
    device = logits.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    sample_indices = torch.randperm(total_tokens, generator=generator, device=device)[:sample_size]
    sample = logits.index_select(0, sample_indices).to(torch.float32)
    sample = sample - sample.mean(dim=0, keepdim=True)

    # return sample.transpose(0,1).contiguous().cpu()
    target_rank = min(hidden_size, sample_size - 1, sample.shape[1])
    print("hidden_size:", hidden_size)
    print("target_rank:", target_rank)
    if target_rank <= 0:
        raise ValueError(
            f"Unable to compute PCA with sample_size={sample_size}. "
            "Increase --lm_head_data_sample_size or provide more cached logits."
        )

    with torch.no_grad():
        _, singular_values, right_vecs = torch.pca_lowrank(sample, q=target_rank, center=False)
        components = right_vecs[:, :target_rank]

        if scale_mode == "singular":
            components = components * singular_values[:target_rank].unsqueeze(0)
        elif scale_mode == "normalize":
            components = F.normalize(components, p=2, dim=1)

        if target_rank < hidden_size:
            remainder = hidden_size - target_rank
            random_block = torch.randn(
                components.size(0),
                remainder,
                generator=generator,
                device=components.device,
                dtype=components.dtype,
            )
            # Orthonormalise the random block to avoid duplicating directions.
            random_block, _ = torch.linalg.qr(random_block, mode="reduced")
            components = torch.cat([components, random_block[:, :remainder]], dim=1)

    return components[:, :hidden_size].contiguous().cpu()


def prepare_biasnet(
    args: argparse.Namespace, dataset: CachedLogitsDataset, device: torch.device
) -> BiasNet:
    if args.base_model_name_or_path:
        base_config = AutoConfig.from_pretrained(
            args.base_model_name_or_path,
            revision=args.base_model_revision,
            trust_remote_code=True,
        )
        if args.hidden_size is None:
            hidden_size = base_config.hidden_size
        else:
            hidden_size = args.hidden_size
        vocab_size = args.vocab_size or base_config.vocab_size
    else:
        if args.hidden_size is None:
            raise ValueError("hidden_size must be provided when base_model_name_or_path is not supplied.")
        hidden_size = args.hidden_size
        vocab_size = args.vocab_size or dataset.vocab_size

    output_head_hidden_size = args.output_head_hidden_size or hidden_size

    if vocab_size != dataset.vocab_size:
        raise ValueError(
            f"BiasNet vocab_size={vocab_size} does not match cached logits "
            f"vocab_size={dataset.vocab_size}. The cache tokenizer and LM head "
            "must use exactly the same vocabulary coordinates."
        )

    init_mode = args.lm_head_init or ("copy" if args.init_lm_head else "none")
    if args.init_checkpoint:
        model = BiasNet.from_pretrained(args.init_checkpoint, map_location="cpu")
        if model.hidden_size != hidden_size or model.vocab_size != vocab_size:
            raise ValueError(
                "Initial checkpoint dimensions do not match requested BiasNet: "
                f"checkpoint=({model.hidden_size}, {model.vocab_size}) "
                f"requested=({hidden_size}, {vocab_size})."
            )
        print(f"Initialising BiasNet from checkpoint: {args.init_checkpoint}")
    else:
        config = BiasConfig(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            input_projection_mode=args.input_projection_mode,
            input_hidden_normalization=args.input_hidden_normalization,
            count_sketch_hashes=args.count_sketch_hashes,
            count_sketch_seed=args.count_sketch_seed,
            count_sketch_input_centering=args.count_sketch_input_centering,
            output_head_hidden_size=output_head_hidden_size,
            lm_head_lora_rank=args.lm_head_lora_rank,
            lm_head_lora_alpha=args.lm_head_lora_alpha,
        )
        model = BiasNet(config)

    if model.lm_head_lora_rank != args.lm_head_lora_rank:
        raise ValueError(
            "Initial checkpoint lm_head_lora_rank does not match the request: "
            f"{model.lm_head_lora_rank} != {args.lm_head_lora_rank}."
        )
    requested_lora_alpha = (
        float(args.lm_head_lora_rank)
        if args.lm_head_lora_alpha is None
        else float(args.lm_head_lora_alpha)
    )
    if (
        model.lm_head_lora_rank > 0
        and model.lm_head_lora_alpha != requested_lora_alpha
    ):
        raise ValueError(
            "Initial checkpoint lm_head_lora_alpha does not match the request: "
            f"{model.lm_head_lora_alpha} != {requested_lora_alpha}."
        )

    if model.input_projection_mode != args.input_projection_mode:
        raise ValueError(
            "Initial checkpoint input_projection_mode does not match the request: "
            f"{model.input_projection_mode} != {args.input_projection_mode}."
        )
    if model.input_hidden_normalization != args.input_hidden_normalization:
        raise ValueError(
            "Initial checkpoint input_hidden_normalization does not match the request: "
            f"{model.input_hidden_normalization} != {args.input_hidden_normalization}."
        )
    if model.input_projection_mode == "count_sketch":
        if model.count_sketch_hashes != args.count_sketch_hashes:
            raise ValueError(
                "Initial checkpoint count_sketch_hashes does not match the request: "
                f"{model.count_sketch_hashes} != {args.count_sketch_hashes}."
            )
        if model.count_sketch_seed != args.count_sketch_seed:
            raise ValueError(
                "Initial checkpoint count_sketch_seed does not match the request: "
                f"{model.count_sketch_seed} != {args.count_sketch_seed}."
            )
        if model.count_sketch_input_centering != args.count_sketch_input_centering:
            raise ValueError(
                "Initial checkpoint count_sketch_input_centering does not match "
                "the request: "
                f"{model.count_sketch_input_centering} != "
                f"{args.count_sketch_input_centering}."
            )
    elif args.count_sketch_input_centering != "none":
        raise ValueError(
            "--count_sketch_input_centering requires "
            "--input_projection_mode=count_sketch."
        )

    model.config.lm_head_init = init_mode
    model.config.base_model_name_or_path = args.base_model_name_or_path
    model.config.base_model_revision = args.base_model_revision
    model.config.output_head_hidden_size = output_head_hidden_size
    model.config.lm_head_lora_rank = model.lm_head_lora_rank
    model.config.lm_head_lora_alpha = (
        model.lm_head_lora_alpha if model.lm_head_lora_rank > 0 else None
    )
    model.config.init_checkpoint = args.init_checkpoint
    if args.position_buckets > 0:
        model.enable_position_bias(args.position_buckets)

    if args.init_checkpoint:
        pass
    elif init_mode == "copy":
        if not args.base_model_name_or_path:
            raise ValueError("--lm_head_init=copy requires --base_model_name_or_path to be set.")
        # For large models (notably Qwen3-32B), loading the entire causal LM
        # just to copy lm_head exceeds the 63 GB host limit.  Read only the
        # safetensors shard containing lm_head.weight instead.
        source_weight = None
        try:
            from pathlib import Path
            from safetensors import safe_open
            from huggingface_hub import hf_hub_download
            base_path = Path(args.base_model_name_or_path)
            index_path = base_path / "model.safetensors.index.json"
            if index_path.exists():
                index_local = str(index_path)
            else:
                index_local = hf_hub_download(
                    repo_id=args.base_model_name_or_path,
                    filename="model.safetensors.index.json",
                    revision=args.base_model_revision,
                )
            index_data = json.loads(Path(index_local).read_text())
            shard_name = index_data["weight_map"].get("lm_head.weight")
            if shard_name is None:
                raise KeyError("model index has no lm_head.weight")
            shard_path = base_path / shard_name
            if not shard_path.exists():
                shard_path = Path(hf_hub_download(
                    repo_id=args.base_model_name_or_path,
                    filename=shard_name,
                    revision=args.base_model_revision,
                ))
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                source_weight = handle.get_tensor("lm_head.weight")
            print(f"Loaded native lm_head.weight directly from {shard_name}: {tuple(source_weight.shape)}")
        except Exception as direct_error:
            print(f"Direct lm_head shard loading unavailable ({direct_error}); falling back to full model load.")
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model_name_or_path,
                revision=args.base_model_revision,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            output_embeddings = base_model.get_output_embeddings()
            if output_embeddings is None or not hasattr(output_embeddings, "weight"):
                raise ValueError("Base model does not expose output embedding weights.")
            source_weight = output_embeddings.weight.detach().cpu()
            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if source_weight.shape[1] != model.lm_head.weight.shape[1] or source_weight.shape[0] < model.lm_head.weight.shape[0]:
            raise ValueError(
                "Base output embedding shape does not match BiasNet lm_head: "
                f"{tuple(source_weight.shape)} != "
                f"{tuple(model.lm_head.weight.shape)}."
            )
        if source_weight.shape[0] != model.lm_head.weight.shape[0]:
            print(
                "Truncating base output embedding vocabulary from "
                f"{source_weight.shape[0]} to cache vocabulary {model.lm_head.weight.shape[0]}."
            )
            source_weight = source_weight[: model.lm_head.weight.shape[0]]
        with torch.no_grad():
            model.lm_head.weight.copy_(source_weight.float())
    elif init_mode == "optimize":
        optim_device = device if device.type == "cuda" else torch.device("cpu")
        print(
            f"Initialising BiasNet lm_head with optimisation on {optim_device}"
            f" (epochs={args.lm_head_optimize_epochs}, batch_size={args.lm_head_optimize_batch_size}, lr={args.lm_head_optimize_lr})."
        )
        optimized_weights = optimize_lm_head_weights(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            device=optim_device,
            lr=args.lm_head_optimize_lr,
            batch_size=args.lm_head_optimize_batch_size,
            epochs=args.lm_head_optimize_epochs,
            init_mode=args.lm_head_optimize_init,
        )
        with torch.no_grad():
            model.lm_head.weight.copy_(optimized_weights.to(model.lm_head.weight.dtype))
    elif init_mode == "data_aware":
        print(
            f"Initialising BiasNet lm_head with data-aware PCA over {args.lm_head_data_sample_size} logits."
        )
        data_weights = data_aware_lm_head_weights(
            dataset=dataset,
            hidden_size=hidden_size,
            sample_size=args.lm_head_data_sample_size,
            seed=args.seed,
            scale_mode=args.lm_head_data_scale,
        )
        with torch.no_grad():
            model.lm_head.weight.copy_(data_weights.to(model.lm_head.weight.dtype))
    elif init_mode == "zeros":
        with torch.no_grad():
            model.lm_head.weight.zero_()
    elif init_mode == "ones":
        with torch.no_grad():
            model.lm_head.weight.fill_(1.0)
    elif init_mode != "none":
        raise ValueError(f"Unknown lm_head initialisation mode: {init_mode}")

    # Build the large vocabulary pseudoinverse on the requested training
    # device. For Gemma-sized vocabularies this is prohibitively slow on CPU,
    # while the allocated GPU has ample memory for the exact same operation.
    model = model.to(device)
    context_contract = getattr(dataset, "context_encoder_contract", None)
    if context_contract is not None:
        existing_contract = getattr(model.config, "context_encoder_contract", None)
        if existing_contract is not None and existing_contract != context_contract:
            raise ValueError("Initial checkpoint and dataset use different context encoders.")
        model.enable_context_conditioning(context_contract["hidden_size"], args.context_bottleneck)
        model.config.context_encoder_contract = context_contract
        model.config.context_learning_rate = args.context_learning_rate
    elif model.context_conditioning:
        raise ValueError("Context-conditioned initial checkpoint requires --context_cache_dir.")
    model.train()
    model.set_up_proj()
    for param in model.lm_head.parameters():
        param.requires_grad = False
    return model


def top1_margin_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    margin_value: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-row target-vs-best-other hinge and best-other scores."""

    if outputs.dim() != 2 or outputs.shape[1] < 2:
        raise ValueError("outputs must have shape [batch, vocab>=2].")
    top_scores, top_ids = torch.topk(outputs, k=2, dim=-1)
    best_other = torch.where(
        top_ids[:, 0].eq(labels), top_scores[:, 1], top_scores[:, 0]
    )
    target_scores = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
    return F.relu(float(margin_value) - target_scores + best_other), best_other


def intervention_objective_masks(
    position_ids: torch.Tensor,
    ce_first_n_tokens: Optional[int] = None,
    base_margin_start_position: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return masks for opening CE and post-opening target-vs-base margin.

    Defaults reproduce the legacy objective: CE and base-margin are both
    enabled at every position. Restricting CE to an opening and starting the
    base margin afterwards lets BiasNet learn a short steering intervention
    without forcing the remainder of a teacher answer token-for-token.
    """

    if position_ids.dim() != 1:
        raise ValueError("position_ids must be one-dimensional.")
    if ce_first_n_tokens is not None and int(ce_first_n_tokens) < 0:
        raise ValueError("ce_first_n_tokens must be non-negative when set.")
    if int(base_margin_start_position) < 0:
        raise ValueError("base_margin_start_position must be non-negative.")
    ce_mask = torch.ones_like(position_ids, dtype=torch.bool)
    if ce_first_n_tokens is not None:
        ce_mask = position_ids.lt(int(ce_first_n_tokens))
    margin_mask = position_ids.ge(int(base_margin_start_position))
    return ce_mask, margin_mask


def candidate_listwise_loss(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    base_token_ids: torch.Tensor,
    candidate_top_k: int,
    proposal_scores: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sparse listwise loss over target/base and top-K cached-score candidates.

    The candidate set is built from the detached cached baseline, so training
    cannot manufacture negatives from the residual it is currently learning.
    Labels and deterministic base are always included, including when they are
    absent from the MC50 observed support.
    """
    if outputs.dim() != 2 or labels.dim() != 1:
        raise ValueError("outputs must be [batch,vocab] and labels [batch].")
    if base_token_ids.shape != labels.shape:
        raise ValueError("base_token_ids must have the same shape as labels.")
    if candidate_top_k <= 0:
        raise ValueError("candidate_top_k must be positive.")
    if proposal_scores is None:
        proposal_scores = outputs
    if proposal_scores.shape != outputs.shape:
        raise ValueError("proposal_scores must have the same shape as outputs.")
    top_k = min(int(candidate_top_k), proposal_scores.shape[1])
    # This is only a candidate proposal; gradients flow through gathered output
    # scores, not through the detached candidate selection.
    candidate_ids = torch.topk(proposal_scores.detach(), k=top_k, dim=-1).indices
    candidate_ids = torch.cat(
        [candidate_ids, labels[:, None], base_token_ids[:, None]], dim=1
    )
    candidate_scores = outputs.gather(1, candidate_ids)
    # The target is included by construction; duplicate candidates are harmless
    # only if removed from the denominator, otherwise repeated labels bias loss.
    # Target/base were appended after top-K; compact each row and locate the
    # target explicitly so duplicate candidates do not bias the denominator.
    losses = []
    for row_scores, row_ids, label in zip(candidate_scores, candidate_ids, labels):
        seen = set()
        ids = []
        scores = []
        for token_id, score in zip(row_ids.tolist(), row_scores):
            if token_id not in seen:
                seen.add(token_id)
                ids.append(token_id)
                scores.append(score)
        target_index = ids.index(int(label.item()))
        scores_tensor = torch.stack(scores)
        losses.append(-scores_tensor[target_index] + torch.logsumexp(scores_tensor, dim=0))
    return torch.stack(losses)


def context_batch_kwargs(batch, device):
    if len(batch) == 9:
        return {}
    if len(batch) != 10:
        raise ValueError("Expected 9 legacy fields or 10 context-conditioned fields.")
    return {"context_features": batch[9].to(device)}


def evaluate_cached_model(
    model: BiasNet,
    dataloader: DataLoader,
    criterion: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    model.eval()
    weighted_loss_sum = 0.0
    weighted_ce_sum = 0.0
    weighted_margin_loss_sum = 0.0
    weighted_top1_margin_loss_sum = 0.0
    weighted_candidate_loss_sum = 0.0
    weighted_residual_l2_loss_sum = 0.0
    sparse_boundary_weight_sum = 0.0
    sparse_boundary_correct = 0.0
    sparse_boundary_target_over_base_sum = 0.0
    sparse_boundary_target_beats_base = 0.0
    preserved_nonboundary_weight_sum = 0.0
    preserved_nonboundary_changed_from_base = 0.0
    weighted_correct = 0.0
    deterministic_base_correct = 0.0
    cached_argmax_correct = 0.0
    weight_total = 0.0
    hard_weight_total = 0.0
    hard_target_over_base_sum = 0.0
    hard_target_beats_base = 0.0
    target_observed_weight = 0.0
    target_unobserved_weight = 0.0
    target_observed_correct = 0.0
    target_unobserved_correct = 0.0
    hard_target_observed_weight = 0.0
    hard_target_unobserved_weight = 0.0
    hard_target_observed_beats_base = 0.0
    hard_target_unobserved_beats_base = 0.0
    residual_scale_sum = 0.0
    residual_scale_weighted_sum = 0.0
    token_total = 0

    with torch.no_grad():
        for batch in dataloader:
            (
                logits,
                labels,
                weights,
                base_token_ids,
                position_ids,
                residual_scales,
                feature_logits,
                target_observed,
                sparse_boundary,
            ) = batch[:9]
            logits = logits.to(device)
            feature_logits = feature_logits.to(device)
            labels = labels.to(device)
            weights = weights.to(device=device, dtype=torch.float32)
            base_token_ids = base_token_ids.to(device)
            position_ids = position_ids.to(device)
            residual_scales = residual_scales.to(
                device=device, dtype=torch.float32
            )
            target_observed = target_observed.to(device=device, dtype=torch.bool)
            sparse_boundary = sparse_boundary.to(
                device=device, dtype=torch.bool
            )
            with torch.cuda.amp.autocast(
                enabled=args.mixed_precision and device.type == "cuda"
            ):
                residuals = model(feature_logits, position_ids=position_ids,
                                  **context_batch_kwargs(batch, device))
                outputs = logits + residual_scales.unsqueeze(1) * residuals
                ce_mask, base_margin_mask = intervention_objective_masks(
                    position_ids,
                    ce_first_n_tokens=args.ce_first_n_tokens,
                    base_margin_start_position=args.base_margin_start_position,
                )
                per_token_ce = criterion(outputs, labels) * ce_mask
                target_scores = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
                base_scores = outputs.gather(
                    1, base_token_ids.unsqueeze(1)
                ).squeeze(1)
                base_wrong = base_token_ids.ne(labels)
                target_over_base = target_scores - base_scores
                per_token_margin = F.relu(
                    args.margin_value - target_over_base
                ) * base_wrong * base_margin_mask
                per_token_top1_margin, _ = top1_margin_loss(
                    outputs, labels, args.top1_margin_value
                )
                if args.top1_margin_boundary_only:
                    per_token_top1_margin = (
                        per_token_top1_margin * sparse_boundary
                    )
                per_token_candidate = candidate_listwise_loss(
                    outputs, labels, base_token_ids, args.candidate_top_k, logits
                )
                residual_l2_mask = position_ids.ge(
                    int(args.residual_l2_start_position)
                )
                if args.residual_l2_nonboundary_only:
                    residual_l2_mask = residual_l2_mask & ~sparse_boundary
                per_token_residual_l2 = (
                    residuals.float().square().mean(dim=-1)
                    * residual_l2_mask
                )
                per_token_loss = (
                    args.ce_loss_weight * per_token_ce
                    + args.margin_loss_weight * per_token_margin
                    + args.top1_margin_loss_weight * per_token_top1_margin
                    + args.candidate_loss_weight * per_token_candidate
                    + args.residual_l2_loss_weight * per_token_residual_l2
                )

            hard_weights = weights * base_wrong
            # target_observed comes from the dataset, which derives it from the
            # raw MC counts. Recomputing it here from feature_logits would lose
            # count=0 support under floor_logprob inputs, where the floor mass
            # makes zero and small counts indistinguishable.
            target_unobserved = ~target_observed
            weighted_loss_sum += (per_token_loss.float() * weights).sum().item()
            weighted_ce_sum += (per_token_ce.float() * weights).sum().item()
            weighted_margin_loss_sum += (
                per_token_margin.float() * weights
            ).sum().item()
            weighted_top1_margin_loss_sum += (
                per_token_top1_margin.float() * weights
            ).sum().item()
            weighted_candidate_loss_sum += (
                per_token_candidate.float() * weights
            ).sum().item()
            weighted_residual_l2_loss_sum += (
                per_token_residual_l2.float() * weights
            ).sum().item()
            sparse_boundary_weight_sum += (
                sparse_boundary.float() * weights
            ).sum().item()
            predictions = outputs.argmax(dim=-1)
            boundary_weights = weights * sparse_boundary.float()
            sparse_boundary_correct += (
                predictions.eq(labels).float() * boundary_weights
            ).sum().item()
            sparse_boundary_target_over_base_sum += (
                target_over_base.float() * boundary_weights
            ).sum().item()
            sparse_boundary_target_beats_base += (
                target_over_base.gt(0).float() * boundary_weights
            ).sum().item()
            preserved_nonboundary = (
                position_ids.ge(int(args.residual_l2_start_position))
                & ~sparse_boundary
            )
            preserved_nonboundary_weights = (
                weights * preserved_nonboundary.float()
            )
            preserved_nonboundary_weight_sum += (
                preserved_nonboundary_weights.sum().item()
            )
            preserved_nonboundary_changed_from_base += (
                predictions.ne(base_token_ids).float()
                * preserved_nonboundary_weights
            ).sum().item()
            weighted_correct += (
                predictions.eq(labels).float() * weights
            ).sum().item()
            deterministic_base_correct += (
                base_token_ids.eq(labels).float() * weights
            ).sum().item()
            cached_argmax_correct += (
                logits.argmax(dim=-1).eq(labels).float() * weights
            ).sum().item()
            weight_total += weights.sum().item()
            hard_weight_total += hard_weights.sum().item()
            hard_target_over_base_sum += (
                target_over_base.float() * hard_weights
            ).sum().item()
            hard_target_beats_base += (
                target_over_base.gt(0).float() * hard_weights
            ).sum().item()
            target_observed_weight += (
                target_observed.float() * weights
            ).sum().item()
            target_unobserved_weight += (
                target_unobserved.float() * weights
            ).sum().item()
            target_observed_correct += (
                outputs.argmax(dim=-1).eq(labels).float()
                * target_observed.float()
                * weights
            ).sum().item()
            target_unobserved_correct += (
                outputs.argmax(dim=-1).eq(labels).float()
                * target_unobserved.float()
                * weights
            ).sum().item()
            hard_target_observed_weight += (
                target_observed.float() * hard_weights
            ).sum().item()
            hard_target_unobserved_weight += (
                target_unobserved.float() * hard_weights
            ).sum().item()
            hard_target_observed_beats_base += (
                target_over_base.gt(0).float()
                * target_observed.float()
                * hard_weights
            ).sum().item()
            hard_target_unobserved_beats_base += (
                target_over_base.gt(0).float()
                * target_unobserved.float()
                * hard_weights
            ).sum().item()
            residual_scale_sum += residual_scales.sum().item()
            residual_scale_weighted_sum += (
                residual_scales * weights
            ).sum().item()
            token_total += residual_scales.numel()

    model.train()
    denominator = max(weight_total, 1.0)
    hard_denominator = max(hard_weight_total, 1.0)
    return {
        "weighted_loss": weighted_loss_sum / denominator,
        "weighted_ce": weighted_ce_sum / denominator,
        "weighted_margin_loss": weighted_margin_loss_sum / denominator,
        "weighted_top1_margin_loss": (
            weighted_top1_margin_loss_sum / denominator
        ),
        "weighted_candidate_loss": weighted_candidate_loss_sum / denominator,
        "weighted_residual_l2_loss": (
            weighted_residual_l2_loss_sum / denominator
        ),
        "sparse_boundary_weight_sum": sparse_boundary_weight_sum,
        "sparse_boundary_accuracy": (
            sparse_boundary_correct / max(sparse_boundary_weight_sum, 1.0)
        ),
        "sparse_boundary_target_over_base_margin": (
            sparse_boundary_target_over_base_sum
            / max(sparse_boundary_weight_sum, 1.0)
        ),
        "sparse_boundary_target_beats_base_rate": (
            sparse_boundary_target_beats_base
            / max(sparse_boundary_weight_sum, 1.0)
        ),
        "preserved_nonboundary_change_from_base_rate": (
            preserved_nonboundary_changed_from_base
            / max(preserved_nonboundary_weight_sum, 1.0)
        ),
        "preserved_nonboundary_weight_sum": preserved_nonboundary_weight_sum,
        "weighted_accuracy": weighted_correct / denominator,
        "deterministic_base_accuracy": deterministic_base_correct / denominator,
        "cached_argmax_accuracy": cached_argmax_correct / denominator,
        "hard_target_over_base_margin": (
            hard_target_over_base_sum / hard_denominator
        ),
        "hard_target_beats_base_rate": hard_target_beats_base / hard_denominator,
        "target_observed_weight": target_observed_weight,
        "target_unobserved_weight": target_unobserved_weight,
        "target_observed_accuracy": (
            target_observed_correct / max(target_observed_weight, 1.0)
        ),
        "target_unobserved_accuracy": (
            target_unobserved_correct / max(target_unobserved_weight, 1.0)
        ),
        "hard_target_observed_beats_base_rate": (
            hard_target_observed_beats_base
            / max(hard_target_observed_weight, 1.0)
        ),
        "hard_target_unobserved_beats_base_rate": (
            hard_target_unobserved_beats_base
            / max(hard_target_unobserved_weight, 1.0)
        ),
        "residual_scale_mean": residual_scale_sum / max(token_total, 1),
        "weighted_residual_scale_mean": (
            residual_scale_weighted_sum / denominator
        ),
        "loss_weight_sum": weight_total,
        "hard_loss_weight_sum": hard_weight_total,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if args.context_bottleneck <= 0 or not math.isfinite(args.context_learning_rate) or args.context_learning_rate <= 0:
        raise ValueError("Context bottleneck and learning rate must be positive and finite.")
    if args.eval_context_cache_dir and (not args.context_cache_dir or not args.eval_data_dir):
        raise ValueError("--eval_context_cache_dir needs --context_cache_dir and --eval_data_dir.")
    if args.context_cache_dir and args.eval_data_dir and not args.eval_context_cache_dir:
        raise ValueError("Separate evaluation caches require --eval_context_cache_dir.")

    if args.margin_loss_weight < 0:
        raise ValueError("--margin_loss_weight must be non-negative.")
    if args.ce_loss_weight < 0:
        raise ValueError("--ce_loss_weight must be non-negative.")
    if args.ce_first_n_tokens is not None and args.ce_first_n_tokens < 0:
        raise ValueError("--ce_first_n_tokens must be non-negative when set.")
    if args.base_margin_start_position < 0:
        raise ValueError("--base_margin_start_position must be non-negative.")
    if args.sparse_boundary_start_position < 0:
        raise ValueError(
            "--sparse_boundary_start_position must be non-negative."
        )
    if args.sparse_boundary_max_per_sample < 0:
        raise ValueError(
            "--sparse_boundary_max_per_sample must be non-negative."
        )
    if args.sparse_boundary_window_size <= 0:
        raise ValueError("--sparse_boundary_window_size must be positive.")
    if args.risk_gate_hard_from_scores and args.risk_gate_training != "hard":
        raise ValueError(
            "--risk_gate_hard_from_scores requires --risk_gate_training hard."
        )
    if args.top1_margin_loss_weight < 0:
        raise ValueError("--top1_margin_loss_weight must be non-negative.")
    if args.residual_l2_loss_weight < 0:
        raise ValueError("--residual_l2_loss_weight must be non-negative.")
    if args.residual_l2_start_position < 0:
        raise ValueError("--residual_l2_start_position must be non-negative.")
    if args.candidate_loss_weight < 0:
        raise ValueError("--candidate_loss_weight must be non-negative.")
    if args.candidate_top_k <= 0:
        raise ValueError("--candidate_top_k must be positive.")
    if args.margin_value < 0:
        raise ValueError("--margin_value must be non-negative.")
    if args.top1_margin_value < 0:
        raise ValueError("--top1_margin_value must be non-negative.")
    if (
        args.ce_loss_weight == 0
        and args.margin_loss_weight == 0
        and args.top1_margin_loss_weight == 0
        and args.candidate_loss_weight == 0
        and args.residual_l2_loss_weight == 0
    ):
        raise ValueError("At least one training loss weight must be positive.")
    if args.count_sketch_hashes <= 0:
        raise ValueError("--count_sketch_hashes must be positive.")
    if args.lm_head_lora_rank < 0:
        raise ValueError("--lm_head_lora_rank must be non-negative.")
    if args.lm_head_lora_alpha is not None and (
        not math.isfinite(args.lm_head_lora_alpha)
        or args.lm_head_lora_alpha <= 0
    ):
        raise ValueError("--lm_head_lora_alpha must be finite and positive.")
    if args.lm_head_lora_rank == 0 and args.lm_head_lora_alpha is not None:
        raise ValueError(
            "--lm_head_lora_alpha requires positive --lm_head_lora_rank."
        )
    if args.held_out_file_count < 0:
        raise ValueError("--held_out_file_count must be non-negative.")
    if args.held_out_file_count and args.eval_data_dir:
        raise ValueError(
            "--held_out_file_count and --eval_data_dir are mutually exclusive."
        )
    if args.position_buckets < 0:
        raise ValueError("--position_buckets must be non-negative.")
    if args.position_only_epochs < 0:
        raise ValueError("--position_only_epochs must be non-negative.")
    if args.position_only_epochs > args.epochs:
        raise ValueError("--position_only_epochs cannot exceed --epochs.")
    if args.position_only_epochs > 0 and args.position_buckets <= 0:
        raise ValueError(
            "--position_only_epochs requires positive --position_buckets."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.lm_head_init is None:
        args.lm_head_init = "copy" if args.init_lm_head else "none"

    train_file_names = None
    eval_file_names = None
    if args.held_out_file_count:
        all_file_names = sorted(
            name
            for name in os.listdir(args.data_dir)
            if name.endswith(".pt") and name != "global_unigram_prior.pt"
        )
        if args.held_out_file_count >= len(all_file_names):
            raise ValueError(
                "--held_out_file_count must leave at least one cache file for training."
            )
        scored_names = sorted(
            all_file_names,
            key=lambda name: hashlib.sha256(
                f"{args.held_out_split_seed}:{name}".encode("utf-8")
            ).hexdigest(),
        )
        eval_file_names = sorted(scored_names[: args.held_out_file_count])
        eval_name_set = set(eval_file_names)
        train_file_names = [
            name for name in all_file_names if name not in eval_name_set
        ]
        print(
            "Deterministic cache split: "
            + json.dumps(
                {
                    "seed": args.held_out_split_seed,
                    "train_file_count": len(train_file_names),
                    "held_out_file_count": len(eval_file_names),
                    "held_out_files": eval_file_names,
                },
                sort_keys=True,
            )
        )

    dataset = CachedLogitsDataset(
        args.data_dir,
        max_tokens_per_sample=args.sample_length,
        risk_gate_training=args.risk_gate_training,
        risk_gate_hard_from_scores=args.risk_gate_hard_from_scores,
        risk_gate_inactive_weight=args.risk_gate_inactive_weight,
        risk_gate_threshold=args.risk_gate_threshold,
        risk_gate_soft_temperature=args.risk_gate_soft_temperature,
        risk_gate_min_scale=args.risk_gate_min_scale,
        risk_gate_warmup_tokens=args.risk_gate_warmup_tokens,
        always_train_first_n=args.always_train_first_n,
        base_token_hard_weight=args.base_token_hard_weight,
        sparse_boundary_start_position=args.sparse_boundary_start_position,
        sparse_boundary_max_per_sample=args.sparse_boundary_max_per_sample,
        sparse_boundary_window_size=args.sparse_boundary_window_size,
        drop_zero_weight_tokens=args.drop_zero_weight_tokens,
        mc_input_representation=args.mc_input_representation,
        mc_base_score_representation=args.mc_base_score_representation,
        mc_log_count_alpha=args.mc_log_count_alpha,
        file_names=train_file_names,
        context_cache_dir=args.context_cache_dir,
    )
    eval_dataset = None
    if args.eval_data_dir or eval_file_names is not None:
        eval_data_dir = args.eval_data_dir or args.data_dir
        eval_dataset = CachedLogitsDataset(
            eval_data_dir,
            max_tokens_per_sample=args.sample_length,
            risk_gate_training=args.risk_gate_training,
            risk_gate_hard_from_scores=args.risk_gate_hard_from_scores,
            risk_gate_inactive_weight=args.risk_gate_inactive_weight,
            risk_gate_threshold=args.risk_gate_threshold,
            risk_gate_soft_temperature=args.risk_gate_soft_temperature,
            risk_gate_min_scale=args.risk_gate_min_scale,
            risk_gate_warmup_tokens=args.risk_gate_warmup_tokens,
            always_train_first_n=args.always_train_first_n,
            base_token_hard_weight=args.base_token_hard_weight,
            sparse_boundary_start_position=args.sparse_boundary_start_position,
            sparse_boundary_max_per_sample=args.sparse_boundary_max_per_sample,
            sparse_boundary_window_size=args.sparse_boundary_window_size,
            drop_zero_weight_tokens=args.drop_zero_weight_tokens,
            mc_input_representation=args.mc_input_representation,
            mc_base_score_representation=args.mc_base_score_representation,
            mc_log_count_alpha=args.mc_log_count_alpha,
            file_names=eval_file_names,
            context_cache_dir=args.eval_context_cache_dir or args.context_cache_dir,
        )
        estimator_fields = (
            "context_encoder_contract",
            "vocab_size",
            "mc_input_representation",
            "mc_base_score_representation",
            "mc_log_count_alpha",
            "mc_samples_per_token",
            "mc_observed_alpha",
            "mc_floor_mass",
            "mc_sample_temperature",
            "mc_top_p",
            "mc_completion_policy",
            *PROXY_MC_CONFIG_FIELDS,
            *STATIC_MC_PRIOR_FIELDS,
        )
        mismatches = {
            field: (getattr(dataset, field), getattr(eval_dataset, field))
            for field in estimator_fields
            if getattr(dataset, field) != getattr(eval_dataset, field)
        }
        if mismatches:
            raise ValueError(
                "Training and held-out caches have incompatible MC semantics: "
                f"{mismatches}."
            )
    print(
        f"Loaded {dataset.num_tokens}/{dataset.num_source_tokens} cached tokens "
        f"(loss_weight_sum={dataset.loss_weight_sum:.1f})."
    )
    if args.risk_gate_training != "none":
        print(
            f"Risk-gated training: mode={args.risk_gate_training}, "
            f"active_tokens={dataset.risk_active_tokens}/{dataset.num_source_tokens} "
            f"({dataset.risk_active_rate:.2%}), "
            f"first_n_or_active={dataset.primary_tokens}, "
            f"base_wrong_in_union={dataset.primary_base_wrong_tokens}."
        )
        print(
            "Sparse intervention boundaries: "
            f"tokens={dataset.sparse_boundary_tokens}, "
            f"start_position={args.sparse_boundary_start_position}, "
            f"max_per_sample={args.sparse_boundary_max_per_sample or 'all'}."
        )
        if args.risk_gate_training == "runtime_soft":
            print(
                "Runtime-soft residual training: "
                f"controlled_tokens={dataset.runtime_controlled_tokens}/"
                f"{dataset.num_source_tokens}, "
                f"mean_scale={dataset.runtime_residual_scale_mean:.6f}, "
                f"threshold={args.risk_gate_threshold}, "
                f"temperature={args.risk_gate_soft_temperature}, "
                f"min_scale={args.risk_gate_min_scale}, "
                f"warmup_tokens={args.risk_gate_warmup_tokens}."
            )
            if dataset.loss_weight_sum <= 0:
                raise ValueError(
                    "All cached tokens have zero loss weight. Lower the risk gate threshold, "
                    "lower --risk_gate_min_scale, use --risk_gate_training soft, "
                    "or regenerate the cache."
                )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_dataloader = (
        DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if eval_dataset is not None
        else None
    )

    model = prepare_biasnet(args, dataset, device).to(device)
    model.config.always_train_first_n = args.always_train_first_n
    model.config.risk_gate_training = args.risk_gate_training
    model.config.risk_gate_hard_from_scores = args.risk_gate_hard_from_scores
    model.config.risk_gate_inactive_weight = args.risk_gate_inactive_weight
    model.config.risk_gate_threshold = args.risk_gate_threshold
    model.config.risk_gate_soft_temperature = args.risk_gate_soft_temperature
    model.config.risk_gate_min_scale = args.risk_gate_min_scale
    model.config.risk_gate_warmup_tokens = args.risk_gate_warmup_tokens
    model.config.base_token_hard_weight = args.base_token_hard_weight
    model.config.drop_zero_weight_tokens = args.drop_zero_weight_tokens
    model.config.margin_loss_weight = args.margin_loss_weight
    model.config.margin_value = args.margin_value
    model.config.ce_loss_weight = args.ce_loss_weight
    model.config.ce_first_n_tokens = args.ce_first_n_tokens
    model.config.base_margin_start_position = args.base_margin_start_position
    model.config.sparse_boundary_start_position = (
        args.sparse_boundary_start_position
    )
    model.config.sparse_boundary_max_per_sample = (
        args.sparse_boundary_max_per_sample
    )
    model.config.sparse_boundary_window_size = args.sparse_boundary_window_size
    model.config.top1_margin_loss_weight = args.top1_margin_loss_weight
    model.config.top1_margin_value = args.top1_margin_value
    model.config.top1_margin_boundary_only = args.top1_margin_boundary_only
    model.config.residual_l2_loss_weight = args.residual_l2_loss_weight
    model.config.residual_l2_start_position = args.residual_l2_start_position
    model.config.residual_l2_nonboundary_only = (
        args.residual_l2_nonboundary_only
    )
    model.config.candidate_loss_weight = args.candidate_loss_weight
    model.config.candidate_top_k = args.candidate_top_k
    model.config.position_only_epochs = args.position_only_epochs
    model.config.mc_input_representation = args.mc_input_representation
    model.config.mc_base_score_representation = (
        dataset.mc_base_score_representation
    )
    model.config.mc_score_interface = (
        "dual_v1"
        if dataset.mc_base_score_representation
        != args.mc_input_representation
        else "shared_v1"
    )
    model.config.mc_log_count_alpha = (
        args.mc_log_count_alpha
        if LOG_COUNT
        in {
            args.mc_input_representation,
            dataset.mc_base_score_representation,
        }
        else None
    )
    model.config.mc_samples_per_token = dataset.mc_samples_per_token
    model.config.mc_observed_alpha = dataset.mc_observed_alpha
    model.config.mc_floor_mass = dataset.mc_floor_mass
    model.config.mc_sample_temperature = dataset.mc_sample_temperature
    model.config.mc_top_p = dataset.mc_top_p
    model.config.mc_completion_policy = dataset.mc_completion_policy
    model.config.mc_estimator_schema_version = (
        2 if model.config.mc_score_interface == "dual_v1" else 1
    )
    for field in PROXY_MC_CONFIG_FIELDS + PROXY_MC_OPTIONAL_FIELDS:
        setattr(model.config, field, getattr(dataset, field, None))
    for field in STATIC_MC_PRIOR_FIELDS:
        setattr(model.config, field, getattr(dataset, field, None))
    model.config.held_out_evaluation = eval_dataset is not None
    model.config.held_out_num_tokens = (
        eval_dataset.num_tokens if eval_dataset is not None else 0
    )
    model.config.held_out_split_seed = (
        args.held_out_split_seed if eval_file_names is not None else None
    )
    model.config.held_out_files = eval_file_names
    if getattr(model, "up_proj", None) is not None:
        model.up_proj = model.up_proj.to(device)

    position_only_active = args.position_only_epochs > 0
    if position_only_active:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("position_embedding.")
        print(
            "Position-only warmup: "
            f"epochs={args.position_only_epochs}, "
            f"buckets={model.num_position_buckets}."
        )

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters()
                        if not n.startswith(("lm_head.", "context_"))],
             "lr": args.learning_rate},
            {"params": [p for n, p in model.named_parameters() if n.startswith("context_")],
             "lr": args.context_learning_rate},
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision and device.type == "cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    cached_metrics = {}
    if args.eval_before_training:
        cached_metrics["initial"] = evaluate_cached_model(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            args=args,
            device=device,
        )
        print(
            "Initial cached metrics: "
            + json.dumps(cached_metrics["initial"], sort_keys=True)
        )
        if eval_dataloader is not None:
            cached_metrics["validation_initial"] = evaluate_cached_model(
                model=model,
                dataloader=eval_dataloader,
                criterion=criterion,
                args=args,
                device=device,
            )
            print(
                "Initial held-out metrics: "
                + json.dumps(
                    cached_metrics["validation_initial"], sort_keys=True
                )
            )

    # model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        if position_only_active and epoch == args.position_only_epochs:
            for name, param in model.named_parameters():
                param.requires_grad = not name.startswith("lm_head.")
            position_only_active = False
            print(f"Unfroze BiasNet MLP at epoch {epoch + 1}.")
        running_loss = 0.0
        running_ce_loss = 0.0
        running_margin_loss = 0.0
        running_top1_margin_loss = 0.0
        running_candidate_loss = 0.0
        running_residual_l2_loss = 0.0
        weighted_correct = 0.0
        running_weight = 0.0
        raw_correct = 0
        raw_total = 0
        num_batches = len(dataloader)
        for step, batch in enumerate(dataloader):
            (
                logits,
                labels,
                weights,
                base_token_ids,
                position_ids,
                residual_scales,
                feature_logits,
                _target_observed,
                sparse_boundary,
            ) = batch[:9]
            logits = logits.to(device)
            feature_logits = feature_logits.to(device)
            labels = labels.to(device)
            weights = weights.to(device=device, dtype=torch.float32)
            base_token_ids = base_token_ids.to(device)
            position_ids = position_ids.to(device)
            residual_scales = residual_scales.to(
                device=device, dtype=torch.float32
            )
            sparse_boundary = sparse_boundary.to(
                device=device, dtype=torch.bool
            )

            with torch.cuda.amp.autocast(enabled=args.mixed_precision and device.type == "cuda"):
                residuals = model(feature_logits, position_ids=position_ids,
                                  **context_batch_kwargs(batch, device))
                outputs = logits + residual_scales.unsqueeze(1) * residuals
                ce_mask, base_margin_mask = intervention_objective_masks(
                    position_ids,
                    ce_first_n_tokens=args.ce_first_n_tokens,
                    base_margin_start_position=args.base_margin_start_position,
                )
                if args.ce_loss_weight > 0:
                    per_token_ce = criterion(outputs, labels) * ce_mask
                else:
                    per_token_ce = torch.zeros_like(
                        labels, dtype=outputs.dtype
                    )
                target_scores = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
                base_scores = outputs.gather(1, base_token_ids.unsqueeze(1)).squeeze(1)
                base_wrong = base_token_ids.ne(labels)
                per_token_margin = F.relu(
                    args.margin_value - target_scores + base_scores
                ) * base_wrong * base_margin_mask
                per_token_top1_margin, _ = top1_margin_loss(
                    outputs, labels, args.top1_margin_value
                )
                if args.top1_margin_boundary_only:
                    per_token_top1_margin = (
                        per_token_top1_margin * sparse_boundary
                    )
                per_token_candidate = candidate_listwise_loss(
                    outputs, labels, base_token_ids, args.candidate_top_k, logits
                )
                residual_l2_mask = position_ids.ge(
                    int(args.residual_l2_start_position)
                )
                if args.residual_l2_nonboundary_only:
                    residual_l2_mask = residual_l2_mask & ~sparse_boundary
                per_token_residual_l2 = (
                    residuals.float().square().mean(dim=-1)
                    * residual_l2_mask
                )
                per_token_loss = (
                    args.ce_loss_weight * per_token_ce
                    + args.margin_loss_weight * per_token_margin
                    + args.top1_margin_loss_weight * per_token_top1_margin
                    + args.candidate_loss_weight * per_token_candidate
                    + args.residual_l2_loss_weight * per_token_residual_l2
                )
                weight_sum = weights.sum().clamp_min(1.0)
                weighted_loss_sum = (per_token_loss.float() * weights).sum()
                weighted_ce_loss_sum = (per_token_ce.float() * weights).sum()
                weighted_margin_loss_sum = (
                    per_token_margin.float() * weights
                ).sum()
                weighted_top1_margin_loss_sum = (
                    per_token_top1_margin.float() * weights
                ).sum()
                weighted_candidate_loss_sum = (
                    per_token_candidate.float() * weights
                ).sum()
                weighted_residual_l2_loss_sum = (
                    per_token_residual_l2.float() * weights
                ).sum()
                loss = weighted_loss_sum / weight_sum
                loss = loss / args.grad_accumulation

            scaler.scale(loss).backward()

            should_step = (step + 1) % args.grad_accumulation == 0 or (step + 1) == num_batches
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running_loss += weighted_loss_sum.detach().item()
            running_ce_loss += weighted_ce_loss_sum.detach().item()
            running_margin_loss += weighted_margin_loss_sum.detach().item()
            running_top1_margin_loss += (
                weighted_top1_margin_loss_sum.detach().item()
            )
            running_candidate_loss += weighted_candidate_loss_sum.detach().item()
            running_residual_l2_loss += (
                weighted_residual_l2_loss_sum.detach().item()
            )
            running_weight += weights.sum().item()
            preds = outputs.argmax(dim=-1)
            correct_mask = (preds == labels).float()
            weighted_correct += (correct_mask * weights).sum().item()
            raw_correct += (preds == labels).sum().item()
            raw_total += labels.numel()

        epoch_loss = running_loss / max(running_weight, 1.0)
        epoch_ce_loss = running_ce_loss / max(running_weight, 1.0)
        epoch_margin_loss = running_margin_loss / max(running_weight, 1.0)
        epoch_top1_margin_loss = (
            running_top1_margin_loss / max(running_weight, 1.0)
        )
        epoch_candidate_loss = running_candidate_loss / max(running_weight, 1.0)
        epoch_residual_l2_loss = (
            running_residual_l2_loss / max(running_weight, 1.0)
        )
        epoch_acc = weighted_correct / max(running_weight, 1.0)
        raw_acc = raw_correct / max(raw_total, 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss:.4f} "
            f"- ce: {epoch_ce_loss:.4f} - margin: {epoch_margin_loss:.4f} "
            f"- top1_margin: {epoch_top1_margin_loss:.4f} "
            f"- candidate: {epoch_candidate_loss:.4f} "
            f"- residual_l2: {epoch_residual_l2_loss:.4f} "
            f"- weighted_acc: {epoch_acc:.4f} - raw_acc: {raw_acc:.4f}"
        )

    cached_metrics["final"] = evaluate_cached_model(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        args=args,
        device=device,
    )
    print(
        "Final cached metrics: "
        + json.dumps(cached_metrics["final"], sort_keys=True)
    )
    if eval_dataloader is not None:
        cached_metrics["validation_final"] = evaluate_cached_model(
            model=model,
            dataloader=eval_dataloader,
            criterion=criterion,
            args=args,
            device=device,
        )
        print(
            "Final held-out metrics: "
            + json.dumps(cached_metrics["validation_final"], sort_keys=True)
        )
    model.save_pretrained(args.output_dir)
    with open(
        os.path.join(args.output_dir, "training_metrics.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(cached_metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved BiasNet checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
