from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.transforms import compute_cut3r_transform, transform_rgb_mask


def test_square_image_uses_cut3r_center_crop() -> None:
    plan = compute_cut3r_transform(
        800, 800, input_size=512, patch_size=16, square_ok=False
    )
    assert (plan.resized_width, plan.resized_height) == (512, 512)
    assert (plan.output_width, plan.output_height) == (512, 384)
    assert (plan.crop_left, plan.crop_top, plan.crop_right, plan.crop_bottom) == (
        0,
        64,
        512,
        448,
    )
    assert plan.token_grid == (24, 32)


def test_landscape_image_preserves_geometry_and_mask_alignment() -> None:
    rgb = np.zeros((600, 800, 3), dtype=np.uint8)
    mask = np.zeros((600, 800), dtype=np.uint8)
    rgb[150:450, 200:600, 0] = 255
    mask[150:450, 200:600] = 255
    transformed_rgb, transformed_mask, plan = transform_rgb_mask(
        Image.fromarray(rgb), Image.fromarray(mask), input_size=512
    )
    assert transformed_rgb.size == transformed_mask.size == (512, 384)
    red = np.asarray(transformed_rgb)[:, :, 0] > 200
    foreground = np.asarray(transformed_mask) > 127
    intersection = np.logical_and(red, foreground).sum()
    union = np.logical_or(red, foreground).sum()
    assert intersection / union > 0.98
    assert plan.token_grid == (24, 32)


def test_rgb_mask_size_mismatch_fails_loudly() -> None:
    with pytest.raises(ValueError, match="sizes differ"):
        transform_rgb_mask(Image.new("RGB", (100, 100)), Image.new("L", (99, 100)))
