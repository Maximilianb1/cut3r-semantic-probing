from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from src.data.transforms import normalized_rgb_array, transform_rgb_mask


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Frame path escapes the dataset root: {relative}") from error
    return path


def prepare_image_window(
    frame_rows: list[dict[str, Any]],
    *,
    dataset_root: str | Path,
    input_size: int = 512,
    patch_size: int = 16,
    square_ok: bool = False,
) -> tuple[list[dict[str, Any]], list[torch.Tensor], tuple[int, int]]:
    """Load a window and build CUT3R-compatible image-only view dictionaries."""
    if len(frame_rows) != 6:
        raise ValueError("The Stage 0 contract requires exactly six frame rows")
    root = Path(dataset_root)
    views: list[dict[str, Any]] = []
    masks: list[torch.Tensor] = []
    token_grid: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None
    for timestep, frame in enumerate(frame_rows):
        image_path = _safe_path(root, frame["image_relpath"])
        mask_path = _safe_path(root, frame["mask_relpath"])
        with Image.open(image_path) as image_handle:
            image = image_handle.copy()
        with Image.open(mask_path) as mask_handle:
            mask = mask_handle.copy()
        image, mask, transform = transform_rgb_mask(
            image,
            mask,
            input_size=input_size,
            patch_size=patch_size,
            square_ok=square_ok,
        )
        recorded_transform = frame.get("spatial_transform_json")
        if recorded_transform is not None and json.loads(recorded_transform) != (
            transform.to_dict()
        ):
            raise ValueError(
                f"Runtime transform disagrees with manifest for {frame['frame_id']}"
            )
        if token_grid is None:
            token_grid = transform.token_grid
            output_size = image.size
        elif transform.token_grid != token_grid or image.size != output_size:
            raise ValueError(
                "All frames in one cached window must share a token grid; "
                f"got {token_grid} and {transform.token_grid}"
            )
        image_tensor = torch.from_numpy(normalized_rgb_array(image))[None]
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy())[
            None, None
        ]
        height, width = image_tensor.shape[-2:]
        views.append(
            {
                "img": image_tensor,
                "ray_map": torch.full((1, 6, height, width), torch.nan),
                "true_shape": torch.tensor([[height, width]], dtype=torch.int32),
                "idx": timestep,
                "instance": frame["frame_id"],
                "camera_pose": torch.eye(4, dtype=torch.float32)[None],
                "img_mask": torch.tensor([True]),
                "ray_mask": torch.tensor([False]),
                "update": torch.tensor([True]),
                "reset": torch.tensor([False]),
            }
        )
        masks.append(mask_tensor)
    assert token_grid is not None
    return views, masks, token_grid


def move_views_to_device(
    views: list[dict[str, Any]], device: torch.device | str
) -> None:
    for view in views:
        for key, value in list(view.items()):
            if isinstance(value, torch.Tensor):
                view[key] = value.to(device)
