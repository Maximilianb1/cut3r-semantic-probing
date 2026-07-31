from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class SpatialTransform:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int
    output_width: int
    output_height: int
    input_size: int
    patch_size: int
    square_ok: bool

    @property
    def token_grid(self) -> tuple[int, int]:
        return (
            self.output_height // self.patch_size,
            self.output_width // self.patch_size,
        )

    def to_dict(self) -> dict[str, int | bool]:
        return asdict(self)


def compute_cut3r_transform(
    width: int,
    height: int,
    *,
    input_size: int = 512,
    patch_size: int = 16,
    square_ok: bool = False,
) -> SpatialTransform:
    """Reproduce CUT3R's aspect-preserving resize and centered patch-aligned crop."""
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image size: {width}x{height}")
    if input_size not in (224, 512):
        raise ValueError(
            "The released CUT3R preprocessing supports input_size 224 or 512"
        )
    scale_target = (
        round(input_size * max(width / height, height / width))
        if input_size == 224
        else input_size
    )
    scale = scale_target / max(width, height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    center_x, center_y = resized_width // 2, resized_height // 2
    if input_size == 224:
        half_width = half_height = min(center_x, center_y)
    else:
        half_width = ((2 * center_x) // patch_size) * (patch_size // 2)
        half_height = ((2 * center_y) // patch_size) * (patch_size // 2)
        if not square_ok and resized_width == resized_height:
            half_height = (3 * half_width) // 4
    left, top = center_x - half_width, center_y - half_height
    right, bottom = center_x + half_width, center_y + half_height
    output_width, output_height = right - left, bottom - top
    if output_width % patch_size or output_height % patch_size:
        raise AssertionError("CUT3R output dimensions must be divisible by patch size")
    return SpatialTransform(
        original_width=width,
        original_height=height,
        resized_width=resized_width,
        resized_height=resized_height,
        crop_left=left,
        crop_top=top,
        crop_right=right,
        crop_bottom=bottom,
        output_width=output_width,
        output_height=output_height,
        input_size=input_size,
        patch_size=patch_size,
        square_ok=square_ok,
    )


def _apply(image: Image.Image, plan: SpatialTransform, *, is_mask: bool) -> Image.Image:
    interpolation = (
        Image.Resampling.NEAREST
        if is_mask
        else (
            Image.Resampling.LANCZOS
            if max(plan.original_width, plan.original_height) > plan.input_size
            else Image.Resampling.BICUBIC
        )
    )
    resized = image.resize((plan.resized_width, plan.resized_height), interpolation)
    return resized.crop(
        (plan.crop_left, plan.crop_top, plan.crop_right, plan.crop_bottom)
    )


def transform_rgb_mask(
    image: Image.Image,
    mask: Image.Image,
    *,
    input_size: int = 512,
    patch_size: int = 16,
    square_ok: bool = False,
) -> tuple[Image.Image, Image.Image, SpatialTransform]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    mask = mask.convert("L")
    if image.size != mask.size:
        raise ValueError(
            "RGB and mask sizes differ before preprocessing: "
            f"{image.size} vs {mask.size}"
        )
    plan = compute_cut3r_transform(
        *image.size, input_size=input_size, patch_size=patch_size, square_ok=square_ok
    )
    transformed_image = _apply(image, plan, is_mask=False)
    transformed_mask = _apply(mask, plan, is_mask=True)
    if transformed_image.size != transformed_mask.size:
        raise AssertionError("RGB and mask transformations diverged")
    return transformed_image, transformed_mask, plan


def normalized_rgb_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.transpose(array * 2.0 - 1.0, (2, 0, 1))


def binary_mask_array(mask: Image.Image, threshold: float = 0.5) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Mask threshold must be in [0, 1]")
    probability = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return probability >= threshold
