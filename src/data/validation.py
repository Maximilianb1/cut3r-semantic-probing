from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from src.common.io import load_json, sha256_file
from src.common.tables import read_parquet
from src.data.transforms import (
    binary_mask_array,
    compute_cut3r_transform,
    transform_rgb_mask,
)


def _safe_dataset_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Manifest path escapes the dataset root: {relative}"
        ) from error
    return candidate


def validate_manifests(
    manifest_dir: str | Path,
    dataset_root: str | Path | None = None,
    *,
    inspect_files: bool = False,
) -> dict[str, Any]:
    directory = Path(manifest_dir)
    paths = {
        name: directory / f"{name}.parquet"
        for name in ("sequences", "frames", "windows")
    }
    summary_path = directory / "summary.json"
    for path in [*paths.values(), summary_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing manifest artifact: {path}")
    summary = load_json(summary_path)
    for name, path in paths.items():
        expected = summary["manifest_sha256"][name]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"Manifest hash mismatch for {name}: expected {expected}, got {actual}"
            )
    sequences = read_parquet(paths["sequences"])
    frames = read_parquet(paths["frames"])
    windows = read_parquet(paths["windows"])
    preprocessing = summary["preprocessing"]
    target_frame_ids = {window["target_frame_id"] for window in windows}
    target_masks_inspected = 0

    selected_sequences = [row for row in sequences if row["selected"]]
    seen_sequence_split: dict[tuple[str, str], str] = {}
    seen_sequence_rows: set[tuple[str, str, str]] = set()
    for row in selected_sequences:
        key = (row["category"], row["sequence_id"])
        row_key = (*key, row["split"])
        if row_key in seen_sequence_rows:
            raise ValueError(f"Duplicate selected sequence row: {row_key}")
        seen_sequence_rows.add(row_key)
        previous = seen_sequence_split.setdefault(key, row["split"])
        if previous != row["split"]:
            raise ValueError(
                f"Sequence leakage for {key}: {previous} and {row['split']}"
            )

    frame_by_id: dict[str, dict[str, Any]] = {}
    frame_numbers: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    for row in frames:
        if row["frame_id"] in frame_by_id:
            raise ValueError(f"Duplicate frame ID: {row['frame_id']}")
        frame_by_id[row["frame_id"]] = row
        key = (row["category"], row["sequence_id"])
        number = int(row["frame_number"])
        if number in frame_numbers[key]:
            raise ValueError(f"Duplicate frame number {number} in sequence {key}")
        frame_numbers[key].add(number)
        expected_transform = compute_cut3r_transform(
            int(row["original_width"]),
            int(row["original_height"]),
            input_size=int(preprocessing["input_size"]),
            patch_size=int(preprocessing["patch_size"]),
            square_ok=bool(preprocessing["square_ok"]),
        ).to_dict()
        if json.loads(row["spatial_transform_json"]) != expected_transform:
            raise ValueError(
                f"Recorded spatial transform is invalid for {row['frame_id']}"
            )
        if dataset_root is not None:
            root = Path(dataset_root)
            resolved: dict[str, Path] = {}
            for field in ("image_relpath", "mask_relpath"):
                resolved[field] = _safe_dataset_path(root, row[field])
                if not resolved[field].is_file():
                    raise FileNotFoundError(
                        f"Manifest path does not exist: {resolved[field]}"
                    )
            if inspect_files:
                with Image.open(resolved["image_relpath"]) as image:
                    image.verify()
                with Image.open(resolved["mask_relpath"]) as mask:
                    mask.verify()
                with Image.open(resolved["image_relpath"]) as image:
                    image_size = ImageOps.exif_transpose(image).size
                with Image.open(resolved["mask_relpath"]) as mask:
                    mask_size = mask.size
                expected_size = (
                    int(row["original_width"]),
                    int(row["original_height"]),
                )
                if image_size != expected_size or mask_size != expected_size:
                    raise ValueError(
                        f"File dimensions disagree for {row['frame_id']}: "
                        f"annotation={expected_size}, image={image_size}, "
                        f"mask={mask_size}"
                    )
                if row["frame_id"] in target_frame_ids:
                    with Image.open(resolved["image_relpath"]) as image_handle:
                        image = image_handle.copy()
                    with Image.open(resolved["mask_relpath"]) as mask_handle:
                        mask = mask_handle.copy()
                    _image, transformed_mask, _plan = transform_rgb_mask(
                        image,
                        mask,
                        input_size=int(preprocessing["input_size"]),
                        patch_size=int(preprocessing["patch_size"]),
                        square_ok=bool(preprocessing["square_ok"]),
                    )
                    if not binary_mask_array(
                        transformed_mask,
                        threshold=float(preprocessing["mask_threshold"]),
                    ).any():
                        raise ValueError(
                            f"Transformed target mask is empty for {row['frame_id']}"
                        )
                    target_masks_inspected += 1

    seen_windows: set[str] = set()
    window_counts: Counter[str] = Counter()
    used_frames_by_sequence: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for window in windows:
        window_id = window["window_id"]
        if window_id in seen_windows:
            raise ValueError(f"Duplicate window ID: {window_id}")
        seen_windows.add(window_id)
        ids = list(window["frame_ids"])
        numbers = [int(number) for number in window["frame_numbers"]]
        if len(ids) != 6 or len(numbers) != 6 or int(window["window_length"]) != 6:
            raise ValueError(f"Window {window_id} does not contain exactly six frames")
        if numbers != sorted(numbers) or len(set(numbers)) != 6:
            raise ValueError(f"Window {window_id} is not strictly ordered")
        if window["target_frame_id"] != ids[-1] or int(window["target_timestep"]) != 6:
            raise ValueError(f"Window {window_id} target is not timestep six")
        sequence_key = (window["category"], window["sequence_id"])
        reused = used_frames_by_sequence[sequence_key].intersection(ids)
        if reused:
            raise ValueError(
                f"Windows reuse frames in sequence {sequence_key}: {sorted(reused)}"
            )
        used_frames_by_sequence[sequence_key].update(ids)
        for item in ids:
            if item not in frame_by_id:
                raise ValueError(f"Window {window_id} references unknown frame {item}")
            frame = frame_by_id[item]
            for field in ("category", "sequence_id", "split"):
                if frame[field] != window[field]:
                    raise ValueError(f"Window/frame {field} mismatch for {window_id}")
        window_counts[window["split"]] += 1
    if len(windows) != int(summary["window_count"]):
        raise ValueError("Summary window count does not match windows manifest")
    if inspect_files and target_masks_inspected != len(windows):
        raise ValueError("Not every window target mask was inspected")
    if len(frames) != int(summary["selected_frame_count"]):
        raise ValueError("Summary frame count does not match frames manifest")
    if len(selected_sequences) != int(summary["selected_sequence_count"]):
        raise ValueError("Summary sequence count does not match sequences manifest")
    categories = sorted({row["category"] for row in sequences})
    if categories != list(summary["categories"]):
        raise ValueError("Summary categories do not match sequence manifest")
    if len(categories) != int(summary["category_count"]):
        raise ValueError("Summary category count does not match sequence manifest")
    for split in ("train", "val", "test"):
        expected = summary["counts_by_split"][split]
        actual = {
            "sequences": sum(row["split"] == split for row in selected_sequences),
            "frames": sum(row["split"] == split for row in frames),
            "windows": window_counts[split],
        }
        if actual != expected:
            raise ValueError(f"Summary split counts do not match for {split}")
    return {
        "valid": True,
        "selected_sequences": len(selected_sequences),
        "frames": len(frames),
        "windows": len(windows),
        "windows_by_split": dict(sorted(window_counts.items())),
        "files_inspected": len(frames) if inspect_files else 0,
        "target_masks_inspected": target_masks_inspected,
    }
