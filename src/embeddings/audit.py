from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw, ImageFont
from safetensors.torch import load_file, save_file

from src.common.io import atomic_write_json, load_json, sha256_file
from src.data.co3d import load_jgzip
from src.data.transforms import transform_rgb_mask
from src.embeddings.cache import load_trajectory


AUDIT_REFERENCE_SCHEMA_VERSION = "stage0-window-audit-reference-v1"


def _select_cache_row(
    cache_dir: str | Path, *, window_id: str | None, window_index: int
) -> dict[str, Any]:
    rows = pq.read_table(Path(cache_dir) / "index.parquet").to_pylist()
    if not rows:
        raise ValueError("Cache index is empty")
    if window_id is not None:
        matches = [row for row in rows if row["window_id"] == window_id]
        if len(matches) != 1:
            raise KeyError(f"Expected one cache row for {window_id}, found {len(matches)}")
        return matches[0]
    if window_index < 0 or window_index >= len(rows):
        raise IndexError(
            f"Window index {window_index} is outside cache range [0, {len(rows) - 1}]"
        )
    return rows[window_index]


def export_audit_reference(
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    window_id: str | None = None,
    window_index: int = 0,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    row = _select_cache_row(
        cache_dir, window_id=window_id, window_index=window_index
    )
    shard_path = cache_dir / row["shard"]
    actual_shard_sha256 = sha256_file(shard_path)
    if actual_shard_sha256 != row["shard_sha256"]:
        raise ValueError(f"Cache shard SHA-256 mismatch: {row['shard']}")
    trajectory = load_trajectory(cache_dir, row["window_id"])
    tensor_path = output_dir / "reference.safetensors"
    save_file(
        {
            "image_tokens": trajectory.image_tokens.contiguous(),
            "state_tokens": trajectory.state_tokens.contiguous(),
        },
        tensor_path,
    )
    reference = {
        "schema_version": AUDIT_REFERENCE_SCHEMA_VERSION,
        "window_id": row["window_id"],
        "frame_ids": trajectory.frame_ids,
        "token_grid": list(trajectory.token_grid),
        "image_shape": list(trajectory.image_tokens.shape),
        "state_shape": list(trajectory.state_tokens.shape),
        "dtype": "float16",
        "source_sha256": json.loads(row["source_sha256_json"]),
        "source_cache": {
            "metadata_sha256": sha256_file(cache_dir / "metadata.json"),
            "index_sha256": sha256_file(cache_dir / "index.parquet"),
            "shard": row["shard"],
            "shard_sha256": actual_shard_sha256,
        },
        "reference_tensor_sha256": sha256_file(tensor_path),
    }
    atomic_write_json(output_dir / "reference.json", reference)
    return reference


def load_audit_reference(
    reference_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    reference_dir = Path(reference_dir)
    metadata = load_json(reference_dir / "reference.json")
    if metadata.get("schema_version") != AUDIT_REFERENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported audit-reference schema")
    tensor_path = reference_dir / "reference.safetensors"
    if sha256_file(tensor_path) != metadata["reference_tensor_sha256"]:
        raise ValueError("Audit-reference tensor SHA-256 mismatch")
    tensors = load_file(tensor_path, device="cpu")
    if set(tensors) != {"image_tokens", "state_tokens"}:
        raise ValueError("Audit reference has unexpected tensor keys")
    for name, shape_key in (
        ("image_tokens", "image_shape"),
        ("state_tokens", "state_shape"),
    ):
        tensor = tensors[name]
        if list(tensor.shape) != list(metadata[shape_key]):
            raise ValueError(f"Audit-reference shape mismatch for {name}")
        if tensor.dtype != torch.float16 or not torch.isfinite(tensor).all():
            raise ValueError(f"Audit-reference tensor is invalid: {name}")
    return metadata, tensors


def resolve_reference_frames(
    reference: dict[str, Any], dataset_root: str | Path
) -> list[dict[str, Any]]:
    frame_ids = list(reference["frame_ids"])
    categories = {identifier.split("/", 1)[0] for identifier in frame_ids}
    if len(categories) != 1:
        raise ValueError("One audit window must contain exactly one category")
    category = categories.pop()
    root = Path(dataset_root)
    annotations = load_jgzip(root / category / "frame_annotations.jgz")
    wanted = set(frame_ids)
    rows: dict[str, dict[str, Any]] = {}
    for annotation in annotations:
        identifier = (
            f"{category}/{annotation['sequence_name']}:"
            f"{int(annotation['frame_number'])}"
        )
        if identifier not in wanted:
            continue
        image = annotation["image"]
        mask = annotation["mask"]
        height, width = (int(value) for value in image["size"])
        rows[identifier] = {
            "frame_id": identifier,
            "category": category,
            "sequence_id": str(annotation["sequence_name"]),
            "frame_number": int(annotation["frame_number"]),
            "image_relpath": str(image["path"]),
            "mask_relpath": str(mask["path"]),
            "original_height": height,
            "original_width": width,
        }
    missing = [identifier for identifier in frame_ids if identifier not in rows]
    if missing:
        raise KeyError(f"CO3D metadata is missing audit frames: {missing}")
    ordered = [rows[identifier] for identifier in frame_ids]
    for row in ordered:
        expected = reference["source_sha256"][row["frame_id"]]
        for kind, key in (("image", "image_relpath"), ("mask", "mask_relpath")):
            path = root / row[key]
            if not path.is_file():
                raise FileNotFoundError(f"Missing audit source file: {path}")
            actual = sha256_file(path)
            if actual != expected[kind]:
                raise ValueError(f"Source SHA-256 mismatch for {row['frame_id']}/{kind}")
    return ordered


def trajectory_statistics(tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("image_tokens", "state_tokens"):
        value = tensors[name].float()
        result[name] = {
            "shape": list(value.shape),
            "dtype": str(tensors[name].dtype),
            "all_finite": bool(torch.isfinite(value).all()),
            "mean_absolute_value_by_timestep": [
                float(value[timestep].abs().mean())
                for timestep in range(value.shape[0])
            ],
            "adjacent_mean_absolute_difference": [
                float((value[timestep] - value[timestep - 1]).abs().mean())
                for timestep in range(1, value.shape[0])
            ],
            "first_to_last_mean_absolute_difference": float(
                (value[-1] - value[0]).abs().mean()
            ),
        }
    return result


def compare_to_reference(
    reference_tensors: dict[str, torch.Tensor],
    *,
    image_tokens: torch.Tensor,
    state_tokens: torch.Tensor,
) -> dict[str, Any]:
    fresh = {
        "image_tokens": image_tokens.detach().cpu().to(torch.float16),
        "state_tokens": state_tokens.detach().cpu().to(torch.float16),
    }
    maxima: dict[str, float] = {}
    for name in ("image_tokens", "state_tokens"):
        if fresh[name].shape != reference_tensors[name].shape:
            raise ValueError(f"Fresh/reference shape mismatch for {name}")
        difference = (fresh[name].float() - reference_tensors[name].float()).abs()
        maxima[name] = float(difference.max())
        if not torch.equal(fresh[name], reference_tensors[name]):
            raise ValueError(
                f"Fresh CUT3R output differs from cached {name}; max={maxima[name]}"
            )
    return {"exactly_equal": True, "maximum_absolute_difference": maxima}


def _label(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 24), "#111111")
    canvas.paste(image, (0, 24))
    ImageDraw.Draw(canvas).text((6, 5), text, fill="white", font=ImageFont.load_default())
    return canvas


def _contact_sheet(images: list[Image.Image], *, columns: int) -> Image.Image:
    if not images:
        raise ValueError("Contact sheet requires at least one image")
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "#181818")
    for index, image in enumerate(images):
        x = (index % columns) * width + (width - image.width) // 2
        y = (index // columns) * height + (height - image.height) // 2
        sheet.paste(image, (x, y))
    return sheet


def render_inputs_and_features(
    reference_dir: str | Path,
    dataset_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    reference, tensors = load_audit_reference(reference_dir)
    rows = resolve_reference_frames(reference, dataset_root)
    transformed: list[Image.Image] = []
    root = Path(dataset_root)
    for timestep, row in enumerate(rows, start=1):
        with Image.open(root / row["image_relpath"]) as handle:
            rgb = handle.copy()
        with Image.open(root / row["mask_relpath"]) as handle:
            mask = handle.copy()
        rgb, mask, plan = transform_rgb_mask(rgb, mask)
        if list(plan.token_grid) != list(reference["token_grid"]):
            raise ValueError("Source transform token grid disagrees with cache")
        overlay = rgb.copy()
        red = Image.new("RGB", rgb.size, (255, 40, 40))
        alpha = mask.convert("L").point(lambda value: 92 if value >= 128 else 0)
        overlay.paste(red, mask=alpha)
        transformed.append(_label(overlay.resize((288, 216)), f"t={timestep} input + mask"))

    tokens = tensors["image_tokens"][:, 0].float()
    flattened = tokens.reshape(-1, tokens.shape[-1])
    centered = flattened - flattened.mean(dim=0, keepdim=True)
    torch.manual_seed(0)
    _u, _s, components = torch.pca_lowrank(centered, q=3, center=False, niter=4)
    projected = centered @ components[:, :3]
    low = torch.quantile(projected, 0.01, dim=0)
    high = torch.quantile(projected, 0.99, dim=0)
    projected = ((projected - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)
    grid_height, grid_width = (int(v) for v in reference["token_grid"])
    feature_images: list[Image.Image] = []
    count = grid_height * grid_width
    for timestep in range(6):
        rgb = (
            projected[timestep * count : (timestep + 1) * count]
            .reshape(grid_height, grid_width, 3)
            .mul(255)
            .byte()
            .numpy()
        )
        feature = Image.fromarray(rgb, mode="RGB").resize(
            (288, 216), Image.Resampling.NEAREST
        )
        feature_images.append(_label(feature, f"t={timestep + 1} image-token PCA"))
    sheet = _contact_sheet(transformed + feature_images, columns=6)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    stats = trajectory_statistics(tensors)
    result = {
        "window_id": reference["window_id"],
        "frame_ids": reference["frame_ids"],
        "source_files_sha256_verified": True,
        "reference_tensor_sha256_verified": True,
        "token_grid": reference["token_grid"],
        "statistics": stats,
        "visualization": str(output_path),
    }
    atomic_write_json(output_path.with_suffix(".json"), result)
    return result


def save_point_cloud_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("PLY points/colors must both have shape [N, 3]")
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            handle.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def render_point_cloud_projections(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    size: int = 560,
) -> None:
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if len(points) < 10:
        raise ValueError("Too few finite reconstruction points")
    center = np.median(points, axis=0)
    radius = np.linalg.norm(points - center, axis=1)
    keep = radius <= np.quantile(radius, 0.98)
    points = points[keep]
    colors = colors[keep]
    panels: list[Image.Image] = []
    for label, axes in (("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2))):
        xy = points[:, axes]
        low = np.quantile(xy, 0.01, axis=0)
        high = np.quantile(xy, 0.99, axis=0)
        span = np.maximum(high - low, 1e-8)
        normalized = np.clip((xy - low) / span, 0, 1)
        pixels = np.rint(normalized * (size - 1)).astype(np.int32)
        pixels[:, 1] = size - 1 - pixels[:, 1]
        canvas = np.full((size, size, 3), 20, dtype=np.uint8)
        canvas[pixels[:, 1], pixels[:, 0]] = colors.astype(np.uint8)
        panels.append(_label(Image.fromarray(canvas), f"CUT3R point cloud: {label}"))
    _contact_sheet(panels, columns=3).save(path)
