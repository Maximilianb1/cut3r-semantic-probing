"""DINOv2 backbone (advanced frozen self-supervised baseline).

DINOv2 has no recurrent state and does not consume the six-frame window: it
encodes the target frame independently. It uses its own patch-14 geometry and
ImageNet normalization, so its token grid differs from CUT3R's. The target mask
is resized with the *same* geometry so it pools onto the DINOv2 patch grid.

- ``spatial_tokens`` <- ``x_norm_patchtokens`` ``[num_patches, dim]``.
- ``global_tokens`` <- ``x_norm_clstoken`` ``[1, dim]``.

The heavy model is loaded lazily via ``torch.hub`` (default
``dinov2_vitb14``, dim 768 to match CUT3R). Loading downloads weights over the
network on first use; the variant and its source are recorded in provenance, so a
result always states which DINOv2 produced it. A ``model_loader`` callable can be
injected (e.g. in tests) to avoid the network entirely.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

from src.backbones.base import (
    TARGET_FRAME_INDEX,
    Backbone,
    BackboneFeatures,
    WindowExtraction,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_PATCH_SIZE = 14


def _target_grid(width: int, height: int, *, image_size: int) -> tuple[int, int]:
    """Aspect-preserving grid whose pixel sizes are multiples of the patch size."""
    scale = image_size / max(width, height)
    resized_w = max(_PATCH_SIZE, int(round(width * scale)) // _PATCH_SIZE * _PATCH_SIZE)
    resized_h = max(_PATCH_SIZE, int(round(height * scale)) // _PATCH_SIZE * _PATCH_SIZE)
    return resized_h, resized_w


def _load_target(root: str, frame_row: dict[str, Any]) -> tuple[Image.Image, Image.Image]:
    from pathlib import Path

    from src.embeddings.input import _safe_path  # reuse the root-escape guard

    base = Path(root)
    image_path = _safe_path(base, frame_row["image_relpath"])
    mask_path = _safe_path(base, frame_row["mask_relpath"])
    with Image.open(image_path) as handle:
        image = handle.copy().convert("RGB")
    with Image.open(mask_path) as handle:
        mask = handle.copy().convert("L")
    if image.size != mask.size:
        raise ValueError(f"RGB and mask sizes differ: {image.size} vs {mask.size}")
    return image, mask


def default_torch_hub_loader(variant: str) -> torch.nn.Module:
    return torch.hub.load("facebookresearch/dinov2", variant)


class Dinov2Backbone(Backbone):
    def __init__(
        self,
        *,
        variant: str = "dinov2_vitb14",
        image_size: int = 518,
        device: str = "cuda",
        model_loader: Callable[[str], torch.nn.Module] | None = None,
    ) -> None:
        self.name = f"dinov2-{variant}"
        self.variant = variant
        self._image_size = int(image_size)
        if self._image_size % _PATCH_SIZE:
            raise ValueError(f"image_size must be a multiple of {_PATCH_SIZE}")
        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DINOv2 backbone requested CUDA but it is unavailable")
        loader = model_loader or default_torch_hub_loader
        self._model = loader(variant)
        self._model.to(self._device)
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)

    def _preprocess(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        grid_pixels = _target_grid(*image.size, image_size=self._image_size)
        resized_h, resized_w = grid_pixels
        resized_image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        resized_mask = mask.resize((resized_w, resized_h), Image.Resampling.NEAREST)
        array = np.asarray(resized_image, dtype=np.float32) / 255.0
        array = (array - np.asarray(_IMAGENET_MEAN)) / np.asarray(_IMAGENET_STD)
        image_tensor = torch.from_numpy(
            np.transpose(array, (2, 0, 1)).astype(np.float32)
        )[None]
        mask_tensor = torch.from_numpy(np.asarray(resized_mask, dtype=np.float32))
        token_grid = (resized_h // _PATCH_SIZE, resized_w // _PATCH_SIZE)
        return image_tensor, mask_tensor, token_grid

    def extract_window(
        self,
        frame_rows: list[dict[str, Any]],
        *,
        dataset_root: str,
    ) -> WindowExtraction:
        target_row = frame_rows[TARGET_FRAME_INDEX]
        image, mask = _load_target(dataset_root, target_row)
        image_tensor, mask_tensor, token_grid = self._preprocess(image, mask)
        image_tensor = image_tensor.to(self._device)
        with torch.inference_mode():
            outputs = self._model.forward_features(image_tensor)
        patch_tokens = outputs["x_norm_patchtokens"][0].float().cpu().contiguous()
        cls_token = outputs["x_norm_clstoken"][0].float().cpu().reshape(1, -1).contiguous()
        features = BackboneFeatures(
            spatial_tokens=patch_tokens,
            token_grid=token_grid,
            global_tokens=cls_token,
            frame_id=target_row["frame_id"],
        )
        extraction = WindowExtraction(features=features, target_mask=mask_tensor)
        extraction.validate()
        return extraction

    def provenance(self) -> dict[str, Any]:
        return {
            "backbone": self.name,
            "weights": "trained",
            "variant": self.variant,
            "patch_size": _PATCH_SIZE,
            "image_size": self._image_size,
            "normalization": "imagenet",
            "source": "torch.hub:facebookresearch/dinov2",
        }
