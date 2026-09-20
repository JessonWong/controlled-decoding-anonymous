from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prefix_risk.data import format_prompt_with_prefix
from prefix_risk.model import (
    PrefixRiskModel,
    RiskHead,
    RiskHeadConfig,
    validate_layer_indices,
)


def resolve_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def validate_backbone_compatibility(
    backbone_config,
    risk_config: dict,
    tokenizer_vocab_size: int,
    backbone_name: str,
) -> None:
    expected_hidden_size = int(risk_config["hidden_size"])
    actual_hidden_size = int(getattr(backbone_config, "hidden_size", -1))
    if actual_hidden_size != expected_hidden_size:
        raise ValueError(
            f"Risk-gate backbone {backbone_name} has hidden_size={actual_hidden_size}; "
            f"the trained head requires {expected_hidden_size}."
        )
    backbone_vocab_size = int(getattr(backbone_config, "vocab_size", -1))
    if backbone_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"Risk-gate backbone {backbone_name} has vocab_size={backbone_vocab_size}; "
            f"the checkpoint tokenizer has {tokenizer_vocab_size}."
        )
    hidden_state_count = int(getattr(backbone_config, "num_hidden_layers", 0)) + 1
    validate_layer_indices(tuple(risk_config["layer_indices"]), hidden_state_count)


class PrefixRiskGate:
    """Scores candidate next-token prefixes and returns a BiasNet mask."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: torch.device,
        threshold: float = 0.1,
        top_k: int = 50,
        batch_size: int = 16,
        max_length: Optional[int] = None,
        dtype: str = "auto",
        model_name: Optional[str] = None,
        load_in_4bit: bool = False,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ):
        self.checkpoint = Path(checkpoint)
        self.threshold = threshold
        self.top_k = top_k
        self.batch_size = batch_size

        with (self.checkpoint / "risk_head_config.json").open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)

        self.max_length = max_length or int(self.config["max_length"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            use_fast=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: dict[str, object] = {
            "torch_dtype": resolve_dtype(dtype),
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        if load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"

        backbone_name = model_name or self.config["model_name"]
        backbone_config = AutoConfig.from_pretrained(
            backbone_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        validate_backbone_compatibility(
            backbone_config=backbone_config,
            risk_config=self.config,
            tokenizer_vocab_size=len(self.tokenizer),
            backbone_name=str(backbone_name),
        )
        backbone = AutoModel.from_pretrained(
            backbone_name,
            config=backbone_config,
            **model_kwargs,
        )
        if load_in_4bit:
            self.device = next(backbone.parameters()).device
        else:
            self.device = device
            backbone.to(self.device)

        layer_indices = tuple(self.config["layer_indices"])
        head_config = RiskHeadConfig(
            hidden_size=self.config["hidden_size"],
            layer_indices=layer_indices,
            head_hidden_size=self.config["head_hidden_size"],
            dropout=self.config["dropout"],
        )
        head = RiskHead(head_config).to(self.device)
        state = torch.load(self.checkpoint / "risk_head.pt", map_location=self.device)
        head.load_state_dict(state["head_state_dict"])

        self.model = PrefixRiskModel(backbone=backbone, head=head, layer_indices=layer_indices)
        self.model.eval()

    @torch.no_grad()
    def score_prefixes(self, prompts: list[str], answer_prefixes: list[str]) -> torch.Tensor:
        if len(prompts) != len(answer_prefixes):
            raise ValueError("prompts and answer_prefixes must have the same length.")
        if not prompts:
            return torch.empty(0, dtype=torch.float32)

        scores: list[torch.Tensor] = []
        use_chat_template = bool(self.config.get("use_chat_template", True))
        for start in range(0, len(prompts), self.batch_size):
            prompt_chunk = prompts[start : start + self.batch_size]
            answer_chunk = answer_prefixes[start : start + self.batch_size]
            texts = [
                format_prompt_with_prefix(
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    answer_prefix=answer_prefix,
                    use_chat_template=use_chat_template,
                )
                for prompt, answer_prefix in zip(prompt_chunk, answer_chunk)
            ]
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**encoded)
            scores.append(torch.sigmoid(logits.detach().float()).cpu())

        return torch.cat(scores, dim=0)

    def _candidate_token_ids(
        self,
        log_probs: torch.Tensor,
        candidate_token_ids: Optional[list[list[int]]] = None,
    ) -> list[list[int]]:
        if candidate_token_ids is not None:
            return candidate_token_ids

        vocab_size = log_probs.size(-1)
        if self.top_k is None or self.top_k <= 0 or self.top_k >= vocab_size:
            all_ids = list(range(vocab_size))
            return [all_ids for _ in range(log_probs.size(0))]

        topk = torch.topk(log_probs.detach(), k=self.top_k, dim=-1).indices.cpu()
        return [[int(token_id) for token_id in row.tolist()] for row in topk]

    @torch.no_grad()
    def build_step_mask(
        self,
        prompts: list[str],
        token_ids: torch.Tensor,
        answer_prefix_builder: Callable[[int, int], str],
        finished_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a row-level mask for applying BiasNet after base-token scoring."""
        if token_ids.dim() != 1:
            raise ValueError("token_ids must have shape [batch_size].")
        if len(prompts) != token_ids.size(0):
            raise ValueError("prompts length must match token_ids batch size.")

        rows: list[int] = []
        score_prompts: list[str] = []
        answer_prefixes: list[str] = []
        for batch_idx, token_id_tensor in enumerate(token_ids.detach().cpu()):
            if finished_mask is not None and bool(finished_mask[batch_idx].item()):
                continue
            token_id = int(token_id_tensor.item())
            rows.append(batch_idx)
            score_prompts.append(prompts[batch_idx])
            answer_prefixes.append(answer_prefix_builder(batch_idx, token_id))

        mask = torch.zeros(token_ids.size(0), dtype=torch.bool)
        if not score_prompts:
            return mask

        scores = self.score_prefixes(score_prompts, answer_prefixes)
        keep = scores < self.threshold
        for batch_idx, keep_row in zip(rows, keep.tolist()):
            if keep_row:
                mask[batch_idx] = True
        return mask

    @torch.no_grad()
    def build_mask(
        self,
        log_probs: torch.Tensor,
        prompts: list[str],
        answer_prefix_builder: Callable[[int, int], str],
        finished_mask: Optional[torch.Tensor] = None,
        candidate_token_ids: Optional[list[list[int]]] = None,
    ) -> torch.Tensor:
        if log_probs.dim() != 2:
            raise ValueError("log_probs must have shape [batch, vocab_size].")
        if len(prompts) != log_probs.size(0):
            raise ValueError("prompts length must match log_probs batch size.")

        candidates = self._candidate_token_ids(log_probs, candidate_token_ids)
        if len(candidates) != log_probs.size(0):
            raise ValueError("candidate_token_ids length must match log_probs batch size.")
        rows: list[int] = []
        token_ids: list[int] = []
        score_prompts: list[str] = []
        answer_prefixes: list[str] = []
        vocab_size = log_probs.size(-1)

        for batch_idx, batch_token_ids in enumerate(candidates):
            if finished_mask is not None and bool(finished_mask[batch_idx].item()):
                continue
            for token_id in batch_token_ids:
                if token_id < 0 or token_id >= vocab_size:
                    continue
                rows.append(batch_idx)
                token_ids.append(int(token_id))
                score_prompts.append(prompts[batch_idx])
                answer_prefixes.append(answer_prefix_builder(batch_idx, int(token_id)))

        mask = torch.zeros_like(log_probs, dtype=torch.bool)
        if not score_prompts:
            return mask

        scores = self.score_prefixes(score_prompts, answer_prefixes)
        keep = scores < self.threshold
        for batch_idx, token_id, keep_token in zip(rows, token_ids, keep.tolist()):
            if keep_token:
                mask[batch_idx, token_id] = True
        return mask
