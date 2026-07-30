"""Shared backbone contract for semantic probing.

A *backbone* is a frozen feature extractor that turns one CO3Dv2 six-frame
window into a standardized set of features for the **target frame** (the sixth
frame, index ``5``; see the Stage 0 window contract). Every backbone -- trained
CUT3R, randomly initialized CUT3R, or DINOv2 -- exposes the same
``extract_window`` interface and returns the same :class:`BackboneFeatures`
container, so the segmentation and classification probes can be written once and
run against any backbone.

Design notes:

- Each backbone owns its own preprocessing and therefore its own token grid.
  CUT3R uses the released 512 / patch-16 geometry; DINOv2 uses a patch-14
  geometry at its own input size. Features and the pooled mask target are always
  expressed on *that backbone's* grid, never assumed to match another backbone.
- ``spatial_tokens`` are pixel-grid aligned and drive segmentation and per-pixel
  classification. ``global_tokens`` are a non-spatial summary and drive pooled
  image classification. This mirrors the CUT3R ``image_tokens`` /
  ``state_tokens`` distinction and must not be conflated (see ADR 0003).
- Nothing here trains: backbones run frozen with gradients disabled. The only
  trainable component is the probe head added downstream.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

TARGET_FRAME_INDEX = 5
WINDOW_LENGTH = 6


@dataclass
class BackboneFeatures:
    """Standardized frozen features for one window's target frame.

    Attributes:
        spatial_tokens: ``[num_tokens, spatial_dim]`` float32 tokens aligned to
            ``token_grid`` (row-major, ``num_tokens == grid_h * grid_w``).
        token_grid: ``(grid_h, grid_w)`` describing the spatial layout.
        global_tokens: ``[num_global, global_dim]`` float32 non-spatial summary
            tokens (e.g. DINOv2 CLS -> ``num_global == 1``; CUT3R persistent
            state -> ``num_global == 768``).
        frame_id: The target frame's CO3D frame id.
    """

    spatial_tokens: torch.Tensor
    token_grid: tuple[int, int]
    global_tokens: torch.Tensor
    frame_id: str

    @property
    def spatial_dim(self) -> int:
        return int(self.spatial_tokens.shape[-1])

    @property
    def global_dim(self) -> int:
        return int(self.global_tokens.shape[-1])

    def validate(self) -> None:
        if self.spatial_tokens.ndim != 2:
            raise ValueError(
                "spatial_tokens must have shape [num_tokens, spatial_dim], got "
                f"{tuple(self.spatial_tokens.shape)}"
            )
        if self.global_tokens.ndim != 2:
            raise ValueError(
                "global_tokens must have shape [num_global, global_dim], got "
                f"{tuple(self.global_tokens.shape)}"
            )
        if len(self.token_grid) != 2 or any(value < 1 for value in self.token_grid):
            raise ValueError(f"token_grid must be two positive ints, got {self.token_grid}")
        grid_h, grid_w = self.token_grid
        if self.spatial_tokens.shape[0] != grid_h * grid_w:
            raise ValueError(
                f"spatial token count {self.spatial_tokens.shape[0]} does not match "
                f"token grid {self.token_grid}"
            )
        if self.spatial_tokens.shape[-1] < 1 or self.global_tokens.shape[-1] < 1:
            raise ValueError("Feature dimensions must be positive")
        if not torch.isfinite(self.spatial_tokens).all():
            raise ValueError("spatial_tokens contain NaN or infinity")
        if not torch.isfinite(self.global_tokens).all():
            raise ValueError("global_tokens contain NaN or infinity")

    def spatial_grid(self) -> torch.Tensor:
        """Return spatial tokens reshaped to ``[grid_h, grid_w, spatial_dim]``."""
        grid_h, grid_w = self.token_grid
        return self.spatial_tokens.reshape(grid_h, grid_w, self.spatial_dim)


@dataclass
class WindowExtraction:
    """One backbone's output for a window: features plus the aligned target mask.

    ``target_mask`` is the target frame's foreground mask **after this backbone's
    own resize/crop**, at the post-transform pixel resolution, so pooling it to
    ``features.token_grid`` yields a segmentation target that lines up with
    ``features.spatial_tokens``. It is kept single-channel ``[H, W]``.
    """

    features: BackboneFeatures
    target_mask: torch.Tensor

    def validate(self) -> None:
        self.features.validate()
        if self.target_mask.ndim != 2:
            raise ValueError(
                f"target_mask must be [H, W], got {tuple(self.target_mask.shape)}"
            )
        if not torch.isfinite(self.target_mask).all():
            raise ValueError("target_mask contains NaN or infinity")

    def target_labels(self, *, threshold: float = 0.5) -> torch.Tensor:
        """Pool the aligned target mask to per-token labels on the feature grid."""
        return pool_mask_to_grid(
            self.target_mask, self.features.token_grid, threshold=threshold
        )


@dataclass
class TrajectoryExtraction:
    """Full six-state CUT3R trajectory plus the target-frame aligned mask.

    Used only by the CUT3R-trained backbone, which is cached with the same
    structure as Stage 0 (grid and latent per state): ``image_tokens`` and
    ``state_tokens`` for all six timesteps. ``target_mask`` is the sixth frame's
    mask under the same geometry as ``image_tokens[5]`` so it pools onto that
    grid for the segmentation label.
    """

    image_tokens: torch.Tensor  # [6, 1, grid_h * grid_w, channels]
    state_tokens: torch.Tensor  # [6, 1, state_tokens, channels]
    token_grid: tuple[int, int]
    frame_ids: list[str]
    target_mask: torch.Tensor  # [H, W]

    def validate(self, *, expected_timesteps: int = WINDOW_LENGTH) -> None:
        if self.image_tokens.ndim != 4 or self.state_tokens.ndim != 4:
            raise ValueError("trajectory tensors must be 4-D [T, B, tokens, channels]")
        if self.image_tokens.shape[0] != expected_timesteps:
            raise ValueError("Unexpected number of image-token timesteps")
        if self.state_tokens.shape[0] != expected_timesteps:
            raise ValueError("Unexpected number of state-token timesteps")
        if len(self.frame_ids) != expected_timesteps:
            raise ValueError("frame_ids count does not match trajectory timesteps")
        grid_h, grid_w = self.token_grid
        if self.image_tokens.shape[-2] != grid_h * grid_w:
            raise ValueError("image token count does not match token_grid")
        if self.target_mask.ndim != 2:
            raise ValueError("target_mask must be [H, W]")
        for tensor in (self.image_tokens, self.state_tokens, self.target_mask):
            if not torch.isfinite(tensor).all():
                raise ValueError("trajectory contains NaN or infinity")

    def target_labels(self, *, threshold: float = 0.5) -> torch.Tensor:
        return pool_mask_to_grid(self.target_mask, self.token_grid, threshold=threshold)


@dataclass
class BackboneConfig:
    """Configuration for :func:`build_backbone`.

    ``kind`` selects the implementation. ``weights`` selects trained vs random
    for CUT3R. ``options`` carries implementation-specific settings (paths,
    variant names, random-init strategy, device).
    """

    kind: str
    weights: str = "trained"
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackboneConfig:
        if "kind" not in data:
            raise ValueError("Backbone config requires a 'kind' field")
        return cls(
            kind=str(data["kind"]),
            weights=str(data.get("weights", "trained")),
            options=dict(data.get("options", {})),
        )


class Backbone(abc.ABC):
    """Frozen per-window feature extractor with a uniform interface."""

    #: Short, stable identifier recorded in cache provenance (e.g. "cut3r-trained").
    name: str

    @abc.abstractmethod
    def extract_window(
        self,
        frame_rows: list[dict[str, Any]],
        *,
        dataset_root: str,
    ) -> WindowExtraction:
        """Extract target-frame features and aligned target mask for one window.

        ``frame_rows`` are ordered ``frames.parquet`` rows (length
        :data:`WINDOW_LENGTH`) carrying at least ``frame_id``, ``image_relpath``,
        and ``mask_relpath``. Implementations load and preprocess frames with
        their own geometry and return features plus the target-frame mask under
        the same geometry (so the mask pools onto the feature grid).
        """

    @abc.abstractmethod
    def provenance(self) -> dict[str, Any]:
        """Return a JSON-serializable record binding features to this backbone."""


def pool_mask_to_grid(
    mask: torch.Tensor,
    token_grid: tuple[int, int],
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Pool a binary/greyscale mask to a per-token foreground label grid.

    Args:
        mask: ``[H, W]`` or ``[1, 1, H, W]`` tensor. Values may be {0,1}, {0,255},
            or probabilities; they are normalized to [0, 1] by their own maximum
            when the maximum exceeds 1.
        token_grid: ``(grid_h, grid_w)`` target resolution.
        threshold: A token is foreground when the mean foreground fraction inside
            its patch is ``>= threshold``.

    Returns:
        ``[grid_h, grid_w]`` float32 tensor of {0.0, 1.0} labels.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 4:
        if mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ValueError(f"Expected a single-channel mask, got {tuple(mask.shape)}")
    else:
        raise ValueError(f"mask must be [H, W] or [1, 1, H, W], got {tuple(mask.shape)}")
    probability = mask.to(dtype=torch.float32)
    maximum = float(probability.max().item()) if probability.numel() else 0.0
    if maximum > 1.0:
        probability = probability / maximum
    foreground = (probability >= threshold).to(dtype=torch.float32)
    grid_h, grid_w = token_grid
    pooled = F.adaptive_avg_pool2d(foreground, output_size=(grid_h, grid_w))
    return (pooled[0, 0] >= threshold).to(dtype=torch.float32)
