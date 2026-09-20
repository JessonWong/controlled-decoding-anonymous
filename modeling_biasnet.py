from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
import torch
import torch.nn as nn
import os
import math


MC_SAMPLE_COUNT_CONDITIONING_AFFINE = "affine_log_fraction_v1"
MC_SAMPLE_COUNT_CONDITIONING_EXACT_ANCHOR = "exact_anchor_count_vector_v1"
MC_SAMPLE_COUNT_CONDITIONING_MODES = (
    MC_SAMPLE_COUNT_CONDITIONING_AFFINE,
    MC_SAMPLE_COUNT_CONDITIONING_EXACT_ANCHOR,
)
COUNT_SKETCH_INPUT_CENTERING_NONE = "none"
COUNT_SKETCH_INPUT_CENTERING_ROW_MIN = "row_min"
COUNT_SKETCH_INPUT_CENTERING_MODES = (
    COUNT_SKETCH_INPUT_CENTERING_NONE,
    COUNT_SKETCH_INPUT_CENTERING_ROW_MIN,
)


class BiasConfig(PretrainedConfig):
    model_type = "biaas_net"
    
    def __init__(
        self,
        hidden_size=None,
        vocab_size=None,
        num_position_buckets=0,
        input_projection_mode="lm_head_pinv",
        input_projection_top_k=128,
        input_hidden_normalization="none",
        input_layer_norm_eps=1e-5,
        count_sketch_hashes=4,
        count_sketch_seed=42,
        count_sketch_input_centering=COUNT_SKETCH_INPUT_CENTERING_NONE,
        output_head_hidden_size=None,
        lm_head_lora_rank=0,
        lm_head_lora_alpha=None,
        mc_sample_count_conditioning=False,
        mc_max_samples_per_token=None,
        mc_sample_count_conditioning_mode=MC_SAMPLE_COUNT_CONDITIONING_AFFINE,
        context_conditioning=False,
        context_dim=None,
        context_bottleneck=128,
        context_encoder_contract=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_position_buckets = int(num_position_buckets or 0)
        self.input_projection_mode = str(input_projection_mode)
        self.input_projection_top_k = int(input_projection_top_k)
        self.input_hidden_normalization = str(input_hidden_normalization)
        self.input_layer_norm_eps = float(input_layer_norm_eps)
        self.count_sketch_hashes = int(count_sketch_hashes)
        self.count_sketch_seed = int(count_sketch_seed)
        self.count_sketch_input_centering = str(count_sketch_input_centering)
        self.mc_sample_count_conditioning = bool(mc_sample_count_conditioning)
        self.mc_sample_count_conditioning_mode = str(
            mc_sample_count_conditioning_mode
        )
        self.mc_max_samples_per_token = (
            None
            if mc_max_samples_per_token is None
            else int(mc_max_samples_per_token)
        )
        # ``PretrainedConfig.save_pretrained`` instantiates an empty config to
        # compute generation-parameter defaults in recent Transformers.  Keep
        # that metadata-only construction valid; real BiasNet configs still
        # receive concrete hidden/vocabulary sizes from the trainer/loader.
        resolved_output_size = (
            output_head_hidden_size
            if output_head_hidden_size is not None
            else hidden_size
        )
        self.output_head_hidden_size = (
            None if resolved_output_size is None else int(resolved_output_size)
        )
        self.lm_head_lora_rank = int(lm_head_lora_rank or 0)
        self.lm_head_lora_alpha = (
            None if lm_head_lora_alpha is None else float(lm_head_lora_alpha)
        )
        self.context_conditioning = bool(context_conditioning)
        self.context_dim = None if context_dim is None else int(context_dim)
        self.context_bottleneck = int(context_bottleneck)
        self.context_encoder_contract = context_encoder_contract

class BiasNet(PreTrainedModel):
    config_class = BiasConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.output_head_hidden_size = int(
            getattr(config, "output_head_hidden_size", self.hidden_size)
        )
        self.intermediate_size = self.hidden_size // 2
        self.vocab_size = config.vocab_size
        self.layer1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.layer2 = nn.Linear(self.intermediate_size, self.intermediate_size)
        self.final_projection = nn.Linear(self.intermediate_size, self.hidden_size)
        self.lm_head = nn.Linear(
            self.output_head_hidden_size, self.vocab_size, bias=False
        )
        self.lm_head_lora_rank = int(
            getattr(config, "lm_head_lora_rank", 0) or 0
        )
        configured_lora_alpha = getattr(config, "lm_head_lora_alpha", None)
        self.lm_head_lora_alpha = (
            float(self.lm_head_lora_rank)
            if configured_lora_alpha is None
            else float(configured_lora_alpha)
        )
        if self.lm_head_lora_rank < 0:
            raise ValueError("lm_head_lora_rank must be non-negative.")
        if self.lm_head_lora_rank > 0:
            if (
                not math.isfinite(self.lm_head_lora_alpha)
                or self.lm_head_lora_alpha <= 0
            ):
                raise ValueError(
                    "lm_head_lora_alpha must be finite and positive when "
                    "lm_head_lora_rank is enabled."
                )
            # Preserve the global RNG stream so a zero-initialized LoRA run
            # has the same data order/dropout draws as its no-LoRA control.
            cpu_rng_state = torch.random.get_rng_state()
            try:
                self.lm_head_lora_a = nn.Linear(
                    self.output_head_hidden_size,
                    self.lm_head_lora_rank,
                    bias=False,
                )
                self.lm_head_lora_b = nn.Linear(
                    self.lm_head_lora_rank,
                    self.vocab_size,
                    bias=False,
                )
                # A zero-initialized output factor makes the adapter an exact
                # no-op at step zero while still allowing B to receive gradients.
                nn.init.kaiming_uniform_(
                    self.lm_head_lora_a.weight, a=math.sqrt(5)
                )
                nn.init.zeros_(self.lm_head_lora_b.weight)
            finally:
                torch.random.set_rng_state(cpu_rng_state)
            self.lm_head_lora_scale = (
                self.lm_head_lora_alpha / self.lm_head_lora_rank
            )
        if self.output_head_hidden_size != self.hidden_size:
            self.output_projection = nn.Linear(
                self.hidden_size, self.output_head_hidden_size, bias=False
            )
        self.input_projection_mode = getattr(
            config, "input_projection_mode", "lm_head_pinv"
        )
        self.input_projection_top_k = int(
            getattr(config, "input_projection_top_k", 128)
        )
        self.input_hidden_normalization = getattr(
            config, "input_hidden_normalization", "none"
        )
        if self.input_projection_mode not in {
            "lm_head_pinv",
            "count_sketch",
            "topk_lm_head",
        }:
            raise ValueError(
                "input_projection_mode must be 'lm_head_pinv', 'count_sketch', "
                "or 'topk_lm_head'."
            )
        if self.input_projection_top_k <= 0:
            raise ValueError("input_projection_top_k must be positive.")
        if (
            self.input_projection_mode == "topk_lm_head"
            and self.output_head_hidden_size != self.hidden_size
        ):
            raise ValueError(
                "topk_lm_head requires output_head_hidden_size == hidden_size."
            )
        if self.input_hidden_normalization not in {"none", "layernorm"}:
            raise ValueError(
                "input_hidden_normalization must be 'none' or 'layernorm'."
            )
        self.count_sketch_hashes = int(
            getattr(config, "count_sketch_hashes", 4)
        )
        self.count_sketch_seed = int(getattr(config, "count_sketch_seed", 42))
        self.count_sketch_input_centering = str(
            getattr(
                config,
                "count_sketch_input_centering",
                COUNT_SKETCH_INPUT_CENTERING_NONE,
            )
        )
        self.input_layer_norm_eps = float(
            getattr(config, "input_layer_norm_eps", 1e-5)
        )
        self.mc_sample_count_conditioning = bool(
            getattr(config, "mc_sample_count_conditioning", False)
        )
        # Checkpoints produced before conditioning modes were introduced used
        # the affine projection.  Keep that interpretation as the default so
        # their state-dict keys and forward behavior remain unchanged.
        self.mc_sample_count_conditioning_mode = str(
            getattr(
                config,
                "mc_sample_count_conditioning_mode",
                MC_SAMPLE_COUNT_CONDITIONING_AFFINE,
            )
        )
        configured_mc_max = getattr(config, "mc_max_samples_per_token", None)
        self.mc_max_samples_per_token = (
            None if configured_mc_max is None else int(configured_mc_max)
        )
        if not math.isfinite(self.input_layer_norm_eps) or self.input_layer_norm_eps <= 0:
            raise ValueError("input_layer_norm_eps must be finite and positive.")
        if self.count_sketch_hashes <= 0:
            raise ValueError("count_sketch_hashes must be positive.")
        if self.count_sketch_input_centering not in COUNT_SKETCH_INPUT_CENTERING_MODES:
            raise ValueError(
                "count_sketch_input_centering must be one of "
                f"{COUNT_SKETCH_INPUT_CENTERING_MODES}."
            )
        if (
            self.input_projection_mode != "count_sketch"
            and self.count_sketch_input_centering != COUNT_SKETCH_INPUT_CENTERING_NONE
        ):
            raise ValueError(
                "count_sketch_input_centering is only valid with "
                "input_projection_mode='count_sketch'."
            )
        if (
            self.mc_sample_count_conditioning_mode
            not in MC_SAMPLE_COUNT_CONDITIONING_MODES
        ):
            raise ValueError(
                "mc_sample_count_conditioning_mode must be one of "
                f"{MC_SAMPLE_COUNT_CONDITIONING_MODES}."
            )
        if self.mc_sample_count_conditioning:
            if (
                self.mc_max_samples_per_token is None
                or self.mc_max_samples_per_token <= 0
            ):
                raise ValueError(
                    "mc_max_samples_per_token must be positive when MC sample "
                    "count conditioning is enabled."
                )
            self._initialize_mc_sample_count_conditioning()
        if self.input_projection_mode == "count_sketch":
            token_ids = torch.arange(self.vocab_size, dtype=torch.int64)
            buckets = []
            signs = []
            modulus = 2_147_483_647
            for hash_index in range(self.count_sketch_hashes):
                offset = self.count_sketch_seed + 104_729 * (hash_index + 1)
                bucket = (
                    (token_ids * (1_103_515_245 + 2 * hash_index) + offset)
                    % modulus
                ) % self.hidden_size
                sign_bits = (
                    token_ids * (2_654_435_761 + 2 * hash_index)
                    + offset * 2 + 1
                ) & 1
                sign = sign_bits.to(torch.float32).mul_(2).sub_(1)
                buckets.append(bucket)
                signs.append(sign)
            # These buffers are deterministic from the config and need not add
            # several megabytes to every checkpoint.
            self.register_buffer(
                "input_hash_buckets", torch.stack(buckets), persistent=False
            )
            self.register_buffer(
                "input_hash_signs", torch.stack(signs), persistent=False
            )
        if self.input_hidden_normalization == "layernorm":
            # Affine-free normalization fixes the scale collapse without adding
            # a prompt-independent trainable shortcut.
            self.input_layer_norm = nn.LayerNorm(
                self.hidden_size,
                eps=self.input_layer_norm_eps,
                elementwise_affine=False,
            )
        self.num_position_buckets = int(
            getattr(config, "num_position_buckets", 0) or 0
        )
        if self.num_position_buckets > 0:
            self.position_embedding = nn.Embedding(
                self.num_position_buckets,
                self.hidden_size,
            )
            nn.init.zeros_(self.position_embedding.weight)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(0.1)
        self.context_conditioning = False
        if config.context_conditioning:
            self.enable_context_conditioning(config.context_dim, config.context_bottleneck)

    def enable_context_conditioning(self, context_dim, bottleneck=128):
        """Add a zero-output context branch without perturbing the RNG stream."""
        if context_dim is None or int(context_dim) <= 0 or int(bottleneck) <= 0:
            raise ValueError("context_dim and context_bottleneck must be positive.")
        context_dim, bottleneck = int(context_dim), int(bottleneck)
        if self.context_conditioning:
            if (self.config.context_dim, self.config.context_bottleneck) != (context_dim, bottleneck):
                raise ValueError("Existing context branch dimensions differ from the request.")
            return
        # Initialize on CPU under fork_rng, then move to the existing model.
        with torch.random.fork_rng(devices=[]):
            self.context_down = nn.Linear(context_dim, bottleneck, bias=False, device="cpu")
            self.context_up = nn.Linear(bottleneck, self.intermediate_size, bias=False, device="cpu")
            nn.init.zeros_(self.context_up.weight)
        self.context_down.to(device=self.layer1.weight.device, dtype=self.layer1.weight.dtype)
        self.context_up.to(device=self.layer1.weight.device, dtype=self.layer1.weight.dtype)
        self.context_conditioning = True
        self.config.context_conditioning = True
        self.config.context_dim = context_dim
        self.config.context_bottleneck = bottleneck

    def context_hidden_bias(self, features, reference):
        if not isinstance(features, torch.Tensor):
            raise ValueError("Context-conditioned BiasNet requires context_features.")
        expected = (*reference.shape[:-1], self.config.context_dim)
        if tuple(features.shape) != expected:
            raise ValueError(f"context_features must have shape {expected}, got {tuple(features.shape)}.")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("context_features must contain finite floating-point values.")
        # Affine-free LayerNorm in fp32, including for fp16 checkpoints.
        normalized = torch.nn.functional.layer_norm(
            features.detach().to(device=reference.device, dtype=torch.float32),
            (self.config.context_dim,),
        ).to(dtype=self.context_down.weight.dtype)
        return self.context_up(torch.nn.functional.silu(self.context_down(normalized)))
    
    def set_up_proj(self):
        if self.input_projection_mode != "lm_head_pinv":
            self.up_proj = None
            return
        if self.layer1.weight.dtype in [torch.float]:
            self.up_proj = torch.linalg.pinv(self.lm_head.weight.clone().detach().t())
        elif self.layer1.weight.dtype in [torch.half]:
            self.up_proj = torch.linalg.pinv(self.lm_head.weight.float().clone().detach().t()).half()
            print("half")
    def inverse_mapping(self, logits):
        if logits.shape[-1] != self.vocab_size:
            raise ValueError(
                f"Expected final input dimension {self.vocab_size}, got {logits.shape[-1]}."
            )
        if self.input_projection_mode == "count_sketch":
            leading_shape = logits.shape[:-1]
            flat = logits.reshape(-1, self.vocab_size)
            if (
                self.count_sketch_input_centering
                == COUNT_SKETCH_INPUT_CENTERING_ROW_MIN
            ):
                # Materialized floor-logprob features are dense: every
                # unobserved vocabulary coordinate carries the same finite
                # floor value. Hashing that shared offset makes the projected
                # background dominate the sparse MC signal. The floor is the
                # row minimum by construction, so subtract it before hashing;
                # unseen coordinates then become exact zeros and are omitted
                # by the sparse nonzero traversal below. Base scores are kept
                # separate by the trainer/runtime and remain uncentered.
                flat = flat - flat.amin(dim=-1, keepdim=True)
            row_ids, token_ids = torch.nonzero(flat, as_tuple=True)
            hidden_states = torch.zeros(
                (flat.shape[0], self.hidden_size),
                device=flat.device,
                dtype=flat.dtype,
            )
            if token_ids.numel() > 0:
                values = flat[row_ids, token_ids]
                for hash_index in range(self.count_sketch_hashes):
                    bucket_ids = self.input_hash_buckets[
                        hash_index, token_ids
                    ]
                    signed_values = values * self.input_hash_signs[
                        hash_index, token_ids
                    ].to(dtype=values.dtype)
                    hidden_states.index_put_(
                        (row_ids, bucket_ids), signed_values, accumulate=True
                    )
                hidden_states = hidden_states / math.sqrt(
                    self.count_sketch_hashes
                )
            hidden_states = hidden_states.reshape(
                *leading_shape, self.hidden_size
            )
        elif self.input_projection_mode == "topk_lm_head":
            # Exact local logits are dense, so CountSketch has to traverse the
            # whole vocabulary and a full LM-head pseudoinverse is needlessly
            # expensive.  Renormalising the top-k probabilities and averaging
            # their frozen output embeddings gives a compact, deterministic
            # distribution representation without storing another projection.
            leading_shape = logits.shape[:-1]
            flat = logits.reshape(-1, self.vocab_size)
            top_k = min(self.input_projection_top_k, self.vocab_size)
            top_values, top_ids = torch.topk(flat.float(), k=top_k, dim=-1)
            top_probabilities = torch.softmax(top_values, dim=-1).to(
                dtype=self.lm_head.weight.dtype
            )
            top_embeddings = torch.nn.functional.embedding(
                top_ids, self.lm_head.weight.detach()
            )
            hidden_states = torch.sum(
                top_probabilities.unsqueeze(-1) * top_embeddings,
                dim=-2,
            ).reshape(*leading_shape, self.hidden_size)
        else:
            if not hasattr(self, "up_proj") or self.up_proj is None:
                raise RuntimeError("set_up_proj() must be called before forward().")
            hidden_states = torch.matmul(logits, self.up_proj)
        if self.input_hidden_normalization == "layernorm":
            hidden_states = self.input_layer_norm(hidden_states.float()).to(
                dtype=logits.dtype
            )
        return hidden_states
    def enable_position_bias(self, num_position_buckets):
        num_position_buckets = int(num_position_buckets)
        if num_position_buckets <= 0:
            raise ValueError("num_position_buckets must be positive.")
        if hasattr(self, "position_embedding"):
            if self.num_position_buckets != num_position_buckets:
                raise ValueError(
                    "BiasNet already has a different number of position buckets: "
                    f"{self.num_position_buckets} != {num_position_buckets}."
                )
            return
        self.num_position_buckets = num_position_buckets
        self.config.num_position_buckets = num_position_buckets
        self.position_embedding = nn.Embedding(
            num_position_buckets,
            self.hidden_size,
            device=self.layer1.weight.device,
            dtype=self.layer1.weight.dtype,
        )
        nn.init.zeros_(self.position_embedding.weight)

    def _initialize_mc_sample_count_conditioning(self):
        factory_kwargs = {
            "device": self.layer1.weight.device,
            "dtype": self.layer1.weight.dtype,
        }
        if (
            self.mc_sample_count_conditioning_mode
            == MC_SAMPLE_COUNT_CONDITIONING_AFFINE
        ):
            self.mc_sample_count_projection = nn.Linear(
                1, self.hidden_size, **factory_kwargs
            )
            nn.init.zeros_(self.mc_sample_count_projection.weight)
            nn.init.zeros_(self.mc_sample_count_projection.bias)
        else:
            self.mc_sample_count_vector = nn.Parameter(
                torch.zeros(self.hidden_size, **factory_kwargs)
            )

    def enable_mc_sample_count_conditioning(
        self,
        max_samples_per_token,
        *,
        mode=MC_SAMPLE_COUNT_CONDITIONING_AFFINE,
    ):
        max_samples_per_token = int(max_samples_per_token)
        mode = str(mode)
        if max_samples_per_token <= 0:
            raise ValueError("max_samples_per_token must be positive.")
        if mode not in MC_SAMPLE_COUNT_CONDITIONING_MODES:
            raise ValueError(
                f"mode must be one of {MC_SAMPLE_COUNT_CONDITIONING_MODES}."
            )
        if self.mc_sample_count_conditioning:
            if self.mc_max_samples_per_token != max_samples_per_token:
                raise ValueError(
                    "BiasNet already uses a different maximum MC sample count: "
                    f"{self.mc_max_samples_per_token} != {max_samples_per_token}."
                )
            if self.mc_sample_count_conditioning_mode != mode:
                raise ValueError(
                    "BiasNet already uses a different MC sample-count "
                    f"conditioning mode: {self.mc_sample_count_conditioning_mode} "
                    f"!= {mode}."
                )
            return
        self.mc_sample_count_conditioning = True
        self.mc_sample_count_conditioning_mode = mode
        self.mc_max_samples_per_token = max_samples_per_token
        self.config.mc_sample_count_conditioning = True
        self.config.mc_sample_count_conditioning_mode = mode
        self.config.mc_max_samples_per_token = max_samples_per_token
        self._initialize_mc_sample_count_conditioning()

    def mc_sample_count_hidden_bias(self, mc_sample_counts, hidden_states):
        if not self.mc_sample_count_conditioning:
            return torch.zeros_like(hidden_states)
        if mc_sample_counts is None:
            raise ValueError(
                "mc_sample_counts are required for an anytime BiasNet checkpoint."
            )
        sample_counts = torch.as_tensor(
            mc_sample_counts,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        if not torch.isfinite(sample_counts).all():
            raise ValueError("mc_sample_counts must be finite.")
        if (sample_counts < 0).any():
            raise ValueError("mc_sample_counts must be non-negative.")
        if (sample_counts > self.mc_max_samples_per_token).any():
            raise ValueError(
                "mc_sample_counts cannot exceed mc_max_samples_per_token."
            )
        if sample_counts.dim() == 0:
            sample_counts = sample_counts.reshape(1)
        normalized_counts = torch.log1p(sample_counts) / math.log1p(
            self.mc_max_samples_per_token
        )
        normalized_counts = normalized_counts.unsqueeze(-1)
        if (
            self.mc_sample_count_conditioning_mode
            == MC_SAMPLE_COUNT_CONDITIONING_AFFINE
        ):
            count_bias = self.mc_sample_count_projection(
                normalized_counts.to(
                    dtype=self.mc_sample_count_projection.weight.dtype
                )
            )
        else:
            # Explicitly replace the full-budget factor with zero instead of
            # relying on floating-point log division to produce exactly one.
            # Thus a trained count vector cannot perturb the original Kmax
            # controller at all.
            remaining_fraction = 1.0 - normalized_counts
            remaining_fraction = torch.where(
                sample_counts.unsqueeze(-1).eq(self.mc_max_samples_per_token),
                torch.zeros_like(remaining_fraction),
                remaining_fraction,
            )
            count_bias = remaining_fraction.to(
                dtype=self.mc_sample_count_vector.dtype
            ) * self.mc_sample_count_vector
        if count_bias.shape != hidden_states.shape:
            try:
                count_bias = count_bias.expand_as(hidden_states)
            except RuntimeError as exc:
                raise ValueError(
                    "mc_sample_counts must be scalar or match the BiasNet batch shape."
                ) from exc
        return count_bias.to(dtype=hidden_states.dtype)

    def position_hidden_bias(self, position_ids, hidden_states):
        if not hasattr(self, "position_embedding"):
            return torch.zeros_like(hidden_states)
        if position_ids is None:
            raise ValueError(
                "position_ids are required for a position-aware BiasNet checkpoint."
            )
        position_ids = torch.as_tensor(
            position_ids,
            device=hidden_states.device,
            dtype=torch.long,
        )
        position_ids = position_ids.clamp(
            min=0,
            max=self.num_position_buckets - 1,
        )
        position_bias = self.position_embedding(position_ids)
        if position_bias.dim() == 1 and hidden_states.dim() == 2:
            position_bias = position_bias.unsqueeze(0)
        return position_bias

    def forward(self, logits, position_ids=None, mc_sample_counts=None, context_features=None):
        x = self.inverse_mapping(logits)
        if self.mc_sample_count_conditioning:
            x = x + self.mc_sample_count_hidden_bias(mc_sample_counts, x)
        x = self.layer1(x)
        if self.context_conditioning:
            x = x + self.context_hidden_bias(context_features, x)
        elif context_features is not None:
            raise ValueError("context_features supplied to a checkpoint without context conditioning.")
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.final_projection(x)
        if hasattr(self, "position_embedding"):
            x = x + self.position_hidden_bias(position_ids, x)
        if hasattr(self, "output_projection"):
            x = self.output_projection(x)
        logits = self.lm_head(x)
        if self.lm_head_lora_rank > 0:
            logits = logits + self.lm_head_lora_scale * self.lm_head_lora_b(
                self.lm_head_lora_a(x)
            )
        return logits
    def save_pretrained(self, save_dir, **kwargs):
        os.makedirs(save_dir, exist_ok=True)
        self.config.save_pretrained(save_dir)
        model_state = {
            'bias_network': self.state_dict(),
            'config': self.config
        }
        model_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(model_state, model_path)
    @classmethod
    def from_pretrained(cls, save_dir, map_location="cpu"):
        config = BiasConfig.from_pretrained(save_dir)
        model = cls(config)
        model_path = os.path.join(save_dir, "pytorch_model.bin")
        if os.path.exists(model_path):
            model_state = torch.load(model_path, map_location=map_location, weights_only=False)
            model.load_state_dict(model_state['bias_network'])
        return model
