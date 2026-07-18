from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.common.tables import read_parquet
from src.data.co3d import assert_disjoint_splits, build_manifests
from src.data.validation import validate_manifests


def test_build_and_validate_synthetic_manifests(
    synthetic_co3d: tuple[Path, Path, Path],
) -> None:
    root, config_path, manifest_dir = synthetic_co3d
    summary = build_manifests(config_path)
    assert summary["selected_sequence_count"] == 3
    assert summary["selected_frame_count"] == 36
    assert summary["window_count"] == 6
    result = validate_manifests(manifest_dir, dataset_root=root, inspect_files=True)
    assert result == {
        "valid": True,
        "selected_sequences": 3,
        "frames": 36,
        "windows": 6,
        "windows_by_split": {"test": 2, "train": 2, "val": 2},
        "files_inspected": 36,
        "target_masks_inspected": 6,
    }
    windows = read_parquet(manifest_dir / "windows.parquet")
    assert all(len(window["frame_ids"]) == 6 for window in windows)


def test_manifest_generation_is_deterministic(
    synthetic_co3d: tuple[Path, Path, Path],
) -> None:
    _root, config_path, _manifest_dir = synthetic_co3d
    first = build_manifests(config_path)
    second = build_manifests(config_path)
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_split_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="Sequence leakage"):
        assert_disjoint_splits({"train": {"same"}, "val": {"same"}, "test": set()})


def test_expected_category_count_is_enforced(
    synthetic_co3d: tuple[Path, Path, Path],
) -> None:
    _root, config_path, _manifest_dir = synthetic_co3d
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dataset"]["expected_category_count"] = 2
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Expected 2 categories, found 1"):
        build_manifests(config_path)


def test_manifest_hash_tampering_is_rejected(
    synthetic_co3d: tuple[Path, Path, Path],
) -> None:
    _root, config_path, manifest_dir = synthetic_co3d
    build_manifests(config_path)
    summary_path = manifest_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["manifest_sha256"]["frames"] = "0" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest hash mismatch for frames"):
        validate_manifests(manifest_dir)


def test_empty_transformed_target_mask_is_rejected(
    synthetic_co3d: tuple[Path, Path, Path],
) -> None:
    root, config_path, manifest_dir = synthetic_co3d
    build_manifests(config_path)
    target_id = read_parquet(manifest_dir / "windows.parquet")[0]["target_frame_id"]
    target = next(
        row
        for row in read_parquet(manifest_dir / "frames.parquet")
        if row["frame_id"] == target_id
    )
    Image.new("L", (64, 48), 0).save(root / target["mask_relpath"])
    with pytest.raises(ValueError, match="Transformed target mask is empty"):
        validate_manifests(manifest_dir, dataset_root=root, inspect_files=True)
