"""
Segmentation probe: a trainable MLP head over precomputed token features.

A "token" is one square patch of the target frame. The head runs on each token
independently: "feature_dim -> 1", a foreground logit per token.
Reshaping those logits to the token grid (and optionally upsampling) gives a mask.

The model operates purely on **precomputed embeddings** from the probe-feature
cache -- it does not hold a backbone. Backbones live in "src.backbones" and are
run once, ahead of time, by the extraction script; training/inference only read
their cached outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}


@dataclass
class HeadConfig:
    """Convert json config to a data class."""
    feature_dim: int = 768 # Backbone output dim per patch
    num_classes: int = 1  # 1 => binary foreground/background segmentation
    hidden_dims: Sequence[int] = field(default_factory=tuple) # MLP layers
    activation: str = "gelu"
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadConfig":
        return cls(
            feature_dim=int(data["feature_dim"]),
            num_classes=int(data.get("num_classes", 1)),
            hidden_dims=tuple(int(value) for value in data.get("hidden_dims", ())),
            activation=str(data.get("activation", "gelu")),
            dropout=float(data.get("dropout", 0.0)),
        )


class MLPHead(nn.Module):
    """MLP Per-token binary segmentation classifier."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        if config.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if config.num_classes < 1:
            raise ValueError("num_classes must be positive")
        if config.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {config.activation!r}; "
                f"supported: {sorted(_ACTIVATIONS)}"
            )
        self.config = config # Data class object with "from json" config
        activation = _ACTIVATIONS[config.activation]

        # Build MLP layers
        layers: list[nn.Module] = []
        in_dim = config.feature_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(activation())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.num_classes))
        self.net = nn.Sequential(*layers)

    @property
    def is_linear(self) -> bool:
        return len(self.config.hidden_dims) == 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[..., feature_dim] -> [..., num_classes] applied per token."""
        if features.shape[-1] != self.config.feature_dim:
            raise ValueError(f"Expected last dim {self.config.feature_dim}, got {features.shape[-1]}")
        return self.net(features)


class SegmentationProbe(nn.Module):
    """Trainable per-token MLP head over precomputed token features."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.head = MLPHead(config)

    @property
    def num_classes(self) -> int:
        return self.head.config.num_classes

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """[B, N, D] -> [B, N, num_classes] per-token logits."""
        if spatial_tokens.ndim != 3:
            raise ValueError(f"spatial_tokens must be [B, N, D], got {tuple(spatial_tokens.shape)}")
        return self.head(spatial_tokens) # Apply the MLPHead forward pass

    def logit_grid(self, spatial_tokens: torch.Tensor, token_grid: tuple[int, int]) -> torch.Tensor:
        """Per-token logits reshaped to 2D: [B, num_classes, grid_h, grid_w]."""
        batch = spatial_tokens.shape[0]
        grid_h, grid_w = token_grid
        logits = self.forward(spatial_tokens)  # [B, N, C]
        return logits.transpose(1, 2).reshape(batch, self.num_classes, grid_h, grid_w)

    def predict_mask(self, spatial_tokens: torch.Tensor, token_grid: tuple[int, int], *,
        output_size: tuple[int, int] | None = None) -> torch.Tensor:
        """
        Foreground probabilities at grid resolution or upsampled to "output_size".
        Used for masking visualizations.

        Binary (num_classes == 1) returns sigmoid probabilities;
        multiclass returns softmax probabilities over the class dimension.
        """
        grid_logits = self.logit_grid(spatial_tokens, token_grid)
        if output_size is not None:
            grid_logits = F.interpolate(grid_logits, size=output_size, mode="bilinear", align_corners=False)
        if self.num_classes == 1:
            return torch.sigmoid(grid_logits)
        return torch.softmax(grid_logits, dim=1) # [B, num_classes, grid_h, grid_w]


def build_probe(config: dict[str, Any]) -> SegmentationProbe:
    """Build a SegmentationProbe object from a config model block."""
    return SegmentationProbe(HeadConfig.from_dict(config))
