from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RiskHeadConfig:
    hidden_size: int
    layer_indices: tuple[int, ...]
    head_hidden_size: int = 1024
    dropout: float = 0.10

    @property
    def input_size(self) -> int:
        return self.hidden_size * len(self.layer_indices)


class RiskHead(nn.Module):
    def __init__(self, config: RiskHeadConfig):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(config.input_size)]
        if config.head_hidden_size > 0:
            layers.extend(
                [
                    nn.Linear(config.input_size, config.head_hidden_size),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.head_hidden_size, 1),
                ]
            )
        else:
            layers.append(nn.Linear(config.input_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class PrefixRiskModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: RiskHead,
        layer_indices: tuple[int, ...],
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.layer_indices = layer_indices

        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    @property
    def head_device(self) -> torch.device:
        return next(self.head.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Backbone did not return hidden states.")

        positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
        last_token_index = (attention_mask.long() * positions).max(dim=1).values
        batch_index = torch.arange(input_ids.size(0), device=last_token_index.device)

        features = []
        for layer_index in self.layer_indices:
            layer_hidden = hidden_states[layer_index]
            selected = layer_hidden[batch_index.to(layer_hidden.device), last_token_index.to(layer_hidden.device)]
            features.append(selected)

        combined = torch.cat(features, dim=-1)
        combined = combined.to(device=self.head_device, dtype=torch.float32)
        return self.head(combined)


def parse_layer_indices(raw: str) -> tuple[int, ...]:
    indices = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not indices:
        raise ValueError("At least one layer index is required.")
    return indices


def validate_layer_indices(layer_indices: tuple[int, ...], hidden_state_count: int) -> None:
    for index in layer_indices:
        normalized = index if index >= 0 else hidden_state_count + index
        if normalized < 0 or normalized >= hidden_state_count:
            raise ValueError(
                f"Layer index {index} is out of range for {hidden_state_count} hidden-state tensors."
            )
