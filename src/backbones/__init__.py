"""Shared frozen backbones for CUT3R semantic probing.

Public API::

    from src.backbones import build_backbone, Backbone, BackboneFeatures

``build_backbone`` maps a config dict to a concrete frozen backbone. The three
backbones under study share the :class:`Backbone` contract so the segmentation
and classification probes are written once:

- ``{"kind": "cut3r", "weights": "trained"}``
- ``{"kind": "cut3r", "weights": "random"}``  (untrained control)
- ``{"kind": "dinov2"}``                        (vision-model anchor)
"""

from __future__ import annotations

from typing import Any

from src.backbones.base import (
    Backbone,
    BackboneConfig,
    BackboneFeatures,
    TrajectoryExtraction,
    WindowExtraction,
    pool_mask_to_grid,
)

__all__ = [
    "Backbone",
    "BackboneConfig",
    "BackboneFeatures",
    "TrajectoryExtraction",
    "WindowExtraction",
    "pool_mask_to_grid",
    "build_backbone",
]


def build_backbone(config: dict[str, Any] | BackboneConfig) -> Backbone:
    """Instantiate a backbone from a config dict (or :class:`BackboneConfig`)."""
    if isinstance(config, BackboneConfig):
        resolved = config
    else:
        resolved = BackboneConfig.from_dict(config)
    options = dict(resolved.options)

    if resolved.kind == "cut3r":
        from src.backbones.cut3r import Cut3rBackbone

        return Cut3rBackbone(weights=resolved.weights, **options)
    if resolved.kind == "dinov2":
        from src.backbones.dinov2 import Dinov2Backbone

        if resolved.weights != "trained":
            raise ValueError("DINOv2 only supports trained weights")
        return Dinov2Backbone(**options)
    raise ValueError(f"Unknown backbone kind: {resolved.kind!r}")
