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
    """Configuration for the probe head (parsed from the config file)."""
    feature_dim: int  # Backbone output dim per patch
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


def feature_statistics_from_moments(
    sum_: torch.Tensor, sumsq: torch.Tensor, count: int, *, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension (mean, std) from an accumulated sum / sum-of-squares / count.

    Lets statistics be computed by streaming batch-by-batch (accumulating "sum_"
    and "sumsq" as each batch is read) instead of materializing every token
    vector in memory at once, which matters at segmentation's scale: a train
    split is millions of individual tokens, not one vector per window.
    """
    if count < 2:
        raise ValueError(f"Need at least 2 token vectors to compute statistics, got {count}")
    mean = sum_ / count
    variance = (sumsq / count - mean * mean).clamp_min(0.0)  # clamp: fp cancellation can dip slightly below 0
    return mean.to(torch.float32), variance.sqrt().clamp_min(eps).to(torch.float32)


def feature_statistics(vectors: torch.Tensor, *, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension (mean, std) of [n, D] token features, std floored at eps."""
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError(f"Need at least 2 token vectors as [n, D], got {tuple(vectors.shape)}")
    vectors = vectors.to(torch.float64)
    return feature_statistics_from_moments(
        vectors.sum(dim=0), (vectors * vectors).sum(dim=0), vectors.shape[0], eps=eps
    )


class SegmentationProbe(nn.Module):
    """Trainable per-token MLP head over precomputed token features."""

    # Declared for the type checker: register_buffer creates these at runtime, so
    # without the annotations they look like attributes defined outside __init__.
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(config.feature_dim))
        self.register_buffer("feature_std", torch.ones(config.feature_dim))
        self.head = MLPHead(config)

    def set_feature_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install standardization statistics, which must come from the TRAIN split. Once."""
        expected = self.head.config.feature_dim
        if tuple(mean.shape) != (expected,) or tuple(std.shape) != (expected,):
            raise ValueError(
                f"Statistics must be [{expected}], got mean {tuple(mean.shape)} "
                f"std {tuple(std.shape)}"
            )
        # copy_ writes into the registered buffers instead of rebinding the attributes,
        # so the module keeps its own tensors: .to(device) still moves them, and the
        # statistics survive a later device change.
        self.feature_mean.copy_(mean.detach())
        self.feature_std.copy_(std.detach())

    def apply_normalization(self, tokens: torch.Tensor) -> torch.Tensor:
        """Standardize with the installed train-split statistics.

        Broadcasts over the last dim, so it works on both the flat [sum_N, D] tokens
        the training/eval loop feeds the head directly and the [B, N, D] batches
        logit_grid/predict_mask use.
        """
        return (tokens - self.feature_mean) / self.feature_std

    @property
    def num_classes(self) -> int:
        return self.head.config.num_classes

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """[..., D] -> [..., num_classes] per-token logits ([N, D] or [B, N, D])."""
        if spatial_tokens.ndim not in (2, 3):
            raise ValueError(f"spatial_tokens must be [N, D] or [B, N, D], got {tuple(spatial_tokens.shape)}")
        return self.head(self.apply_normalization(spatial_tokens))

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
