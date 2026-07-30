from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image


def _write_jgzip(path: Path, value: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(value, handle)


@pytest.fixture
def synthetic_co3d(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "co3d"
    category = "ball"
    category_dir = root / category
    (category_dir / "set_lists").mkdir(parents=True)
    sequence_annotations = []
    frame_annotations = []
    split_sequences: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split in split_sequences:
        for sequence_number in range(2):
            sequence_id = f"{split}_sequence_{sequence_number}"
            split_sequences[split].append(sequence_id)
            sequence_annotations.append(
                {
                    "sequence_name": sequence_id,
                    "category": category,
                    "video": None,
                    "point_cloud": None,
                    "viewpoint_quality_score": 0.9,
                }
            )
            image_dir = category_dir / sequence_id / "images"
            mask_dir = category_dir / sequence_id / "masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            for frame_number in range(12):
                relative_image = (
                    f"{category}/{sequence_id}/images/frame{frame_number:06d}.jpg"
                )
                relative_mask = (
                    f"{category}/{sequence_id}/masks/frame{frame_number:06d}.png"
                )
                rgb = np.zeros((48, 64, 3), dtype=np.uint8)
                rgb[12:36, 16:48] = [200, 40, 20]
                mask = np.zeros((48, 64), dtype=np.uint8)
                mask[12:36, 16:48] = 255
                Image.fromarray(rgb).save(root / relative_image)
                Image.fromarray(mask).save(root / relative_mask)
                frame_annotations.append(
                    {
                        "sequence_name": sequence_id,
                        "frame_number": frame_number,
                        "frame_timestamp": frame_number / 30.0,
                        "image": {"path": relative_image, "size": [48, 64]},
                        "mask": {
                            "path": relative_mask,
                            "mass": float(mask.sum() / 255),
                        },
                        "viewpoint": {
                            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            "T": [0, 0, 0],
                            "focal_length": [1, 1],
                            "principal_point": [0, 0],
                            "intrinsics_format": "ndc_norm_image_bounds",
                        },
                        "meta": None,
                    }
                )
    _write_jgzip(category_dir / "sequence_annotations.jgz", sequence_annotations)
    _write_jgzip(category_dir / "frame_annotations.jgz", frame_annotations)
    files = {
        "set_lists_fewview_train.json": {"train": split_sequences["train"]},
        "set_lists_fewview_dev.json": {"val": split_sequences["val"]},
        "set_lists_fewview_test.json": {"test": split_sequences["test"]},
    }
    for filename, assignments in files.items():
        payload = {"train": [], "val": [], "test": []}
        for key, sequence_ids in assignments.items():
            payload[key] = [
                [sequence_id, 0, "unused.jpg"] for sequence_id in sequence_ids
            ]
        (category_dir / "set_lists" / filename).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    manifest_dir = tmp_path / "manifests"
    config = {
        "dataset": {
            "version": "synthetic-co3d-v2",
            "root": str(root),
            "categories": [category],
            "expected_category_count": 1,
            "require_viewpoint": True,
            "require_local_files": True,
        },
        "sampling": {
            "seed": 7,
            "window_length": 6,
            "windows_per_sequence": 2,
            "sequence_caps": {"train": 1, "val": 1, "test": 1},
        },
        "preprocessing": {
            "input_size": 224,
            "patch_size": 16,
            "square_ok": False,
            "mask_threshold": 0.5,
        },
        "output": {"manifest_dir": str(manifest_dir)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return root, config_path, manifest_dir
