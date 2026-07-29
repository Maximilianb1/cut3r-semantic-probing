"""Segmentation probe: a frozen backbone plus a trainable MLP head.

The head is applied independently to every spatial token, turning per-token
features into per-token foreground logits. Reshaping those to the token grid and
upsampling gives a pixel mask.

Terminology (LLM_GUIDE guardrail): with ``hidden_dims=()`` the head is a genuine
**linear probe**; with one or more hidden layers it is a nonlinear **MLP probe**.
The class does not pretend a multi-layer head is linear -- ``head.is_linear``
reports which it is.

Two usage modes share one head:

- **Cached (default for training):** features are precomputed in the probe cache,
  so construct with ``feature_dim`` and call ``forward(spatial_tokens)``.
- **End-to-end (inference/analysis):** attach a live ``backbone`` and call
  ``extract_and_predict(frame_rows, dataset_root=...)``; the backbone stays
  frozen and only the head runs with gradients.
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
    feature_dim: int
    num_classes: int = 1  # 1 => binary foreground/background segmentation
    hidden_dims: Sequence[int] = field(default_factory=tuple)
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
    """Per-token classifier. Linear when ``hidden_dims`` is empty, else an MLP."""

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
        self.config = config
        activation = _ACTIVATIONS[config.activation]
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
        """``[..., feature_dim] -> [..., num_classes]`` applied per token."""
        if features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"Expected last dim {self.config.feature_dim}, got {features.shape[-1]}"
            )
        return self.net(features)


class SegmentationProbe(nn.Module):
    """Frozen backbone (optional) + trainable per-token MLP head."""

    def __init__(self, config: HeadConfig, *, backbone: Any | None = None) -> None:
        super().__init__()
        self.head = MLPHead(config)
        self.backbone = backbone  # kept out of nn parameters; frozen if present

    @property
    def num_classes(self) -> int:
        return self.head.config.num_classes

    def forward(self, spatial_tokens: torch.Tensor) -> torch.Tensor:
        """``[B, N, D] -> [B, N, num_classes]`` per-token logits."""
        if spatial_tokens.ndim != 3:
            raise ValueError(
                f"spatial_tokens must be [B, N, D], got {tuple(spatial_tokens.shape)}"
            )
        return self.head(spatial_tokens)

    def logit_grid(
        self, spatial_tokens: torch.Tensor, token_grid: tuple[int, int]
    ) -> torch.Tensor:
        """Per-token logits reshaped to ``[B, num_classes, grid_h, grid_w]``."""
        batch = spatial_tokens.shape[0]
        grid_h, grid_w = token_grid
        logits = self.forward(spatial_tokens)  # [B, N, C]
        return logits.transpose(1, 2).reshape(batch, self.num_classes, grid_h, grid_w)

    def predict_mask(
        self,
        spatial_tokens: torch.Tensor,
        token_grid: tuple[int, int],
        *,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Foreground probabilities at grid resolution or upsampled to ``output_size``.

        Binary (``num_classes == 1``) returns sigmoid probabilities; multiclass
        returns softmax probabilities over the class dimension.
        """
        grid_logits = self.logit_grid(spatial_tokens, token_grid)
        if output_size is not None:
            grid_logits = F.interpolate(
                grid_logits, size=output_size, mode="bilinear", align_corners=False
            )
        if self.num_classes == 1:
            return torch.sigmoid(grid_logits)
        return torch.softmax(grid_logits, dim=1)

    @torch.inference_mode()
    def extract_and_predict(
        self, frame_rows: list[dict[str, Any]], *, dataset_root: str
    ) -> dict[str, torch.Tensor]:
        """End-to-end: run the attached frozen backbone, then the head."""
        if self.backbone is None:
            raise RuntimeError("No backbone attached; construct with backbone=... ")
        extraction = self.backbone.extract_window(frame_rows, dataset_root=dataset_root)
        spatial = extraction.features.spatial_tokens[None]  # [1, N, D]
        grid = extraction.features.token_grid
        probability = self.predict_mask(spatial, grid)
        return {
            "probability": probability,
            "target_labels": extraction.target_labels()[None],
            "token_grid": torch.tensor(grid),
        }


def build_probe(config: dict[str, Any], *, backbone: Any | None = None) -> SegmentationProbe:
    """Build a :class:`SegmentationProbe` from a config ``model`` block."""
    return SegmentationProbe(HeadConfig.from_dict(config), backbone=backbone)
