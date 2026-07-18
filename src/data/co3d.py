from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.io import atomic_write_json, load_yaml, sha256_file
from src.common.tables import write_parquet_atomic
from src.data.transforms import compute_cut3r_transform
from src.data.windows import choose_sequences, generate_ordered_windows

OFFICIAL_SPLIT_FILES = {
    "train": ("set_lists_fewview_train.json", "train"),
    "val": ("set_lists_fewview_dev.json", "val"),
    "test": ("set_lists_fewview_test.json", "test"),
}
MANIFEST_SCHEMA_VERSION = "stage0-manifest-v1"
CO3DV2_231130_CATEGORIES = frozenset(
    {
        "apple",
        "backpack",
        "ball",
        "banana",
        "baseballbat",
        "baseballglove",
        "bench",
        "bicycle",
        "book",
        "bottle",
        "bowl",
        "broccoli",
        "cake",
        "car",
        "carrot",
        "cellphone",
        "chair",
        "couch",
        "cup",
        "donut",
        "frisbee",
        "hairdryer",
        "handbag",
        "hotdog",
        "hydrant",
        "keyboard",
        "kite",
        "laptop",
        "microwave",
        "motorcycle",
        "mouse",
        "orange",
        "parkingmeter",
        "pizza",
        "plant",
        "remote",
        "sandwich",
        "skateboard",
        "stopsign",
        "suitcase",
        "teddybear",
        "toaster",
        "toilet",
        "toybus",
        "toyplane",
        "toytrain",
        "toytruck",
        "tv",
        "umbrella",
        "vase",
        "wineglass",
    }
)


def load_jgzip(path: str | Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def load_official_sequence_splits(category_dir: str | Path) -> dict[str, set[str]]:
    category_dir = Path(category_dir)
    result: dict[str, set[str]] = {}
    for split, (filename, key) in OFFICIAL_SPLIT_FILES.items():
        path = category_dir / "set_lists" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing official CO3Dv2 set list: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if key not in payload or not isinstance(payload[key], list):
            raise ValueError(f"Set list {path} has no list named {key!r}")
        result[split] = {str(row[0]) for row in payload[key]}
    assert_disjoint_splits(result)
    return result


def assert_disjoint_splits(splits: dict[str, set[str]]) -> None:
    names = sorted(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = splits[left] & splits[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ValueError(
                    f"Sequence leakage between {left} and {right}: {len(overlap)} "
                    f"overlap(s), including {examples}"
                )


def discover_categories(dataset_root: str | Path) -> list[str]:
    root = Path(dataset_root)
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir()
        and (child / "frame_annotations.jgz").is_file()
        and (child / "sequence_annotations.jgz").is_file()
    )


def frame_id(category: str, sequence_id: str, number: int) -> str:
    return f"{category}/{sequence_id}:{number}"


def _dataset_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(
            f"Annotation path escapes the dataset root: {relative}"
        ) from error
    return candidate


def _viewpoint_json(viewpoint: Any) -> str | None:
    if viewpoint is None:
        return None
    return json.dumps(viewpoint, sort_keys=True, separators=(",", ":"))


def _validate_config(config: dict[str, Any]) -> None:
    required = {"dataset", "sampling", "preprocessing", "output"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Stage 0 configuration is missing: {', '.join(missing)}")
    if int(config["sampling"].get("window_length", 0)) != 6:
        raise ValueError("The accepted Stage 0 protocol requires six-frame windows")
    caps = config["sampling"].get("sequence_caps", {})
    for split in ("train", "val", "test"):
        if split not in caps:
            raise ValueError(f"Missing sequence cap for split {split}")
        if caps[split] is not None and (
            isinstance(caps[split], bool) or int(caps[split]) < 1
        ):
            raise ValueError(f"Sequence cap for {split} must be positive or null")
    if int(config["sampling"].get("windows_per_sequence", 0)) < 1:
        raise ValueError("windows_per_sequence must be positive")
    categories = config["dataset"].get("categories", "all")
    if categories != "all" and (
        not isinstance(categories, list)
        or not categories
        or not all(isinstance(category, str) and category for category in categories)
    ):
        raise ValueError("dataset.categories must be 'all' or a non-empty string list")
    if categories != "all" and len(categories) != len(set(categories)):
        raise ValueError("dataset.categories must not contain duplicates")
    preprocessing = config["preprocessing"]
    mask_threshold = float(preprocessing["mask_threshold"])
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("preprocessing.mask_threshold must be in [0, 1]")
    compute_cut3r_transform(
        640,
        480,
        input_size=int(preprocessing["input_size"]),
        patch_size=int(preprocessing["patch_size"]),
        square_ok=bool(preprocessing["square_ok"]),
    )


def build_manifests(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    _validate_config(config)
    dataset_cfg = config["dataset"]
    sampling_cfg = config["sampling"]
    root = Path(dataset_cfg["root"])
    if not root.is_dir():
        raise FileNotFoundError(f"CO3Dv2 root does not exist: {root}")
    configured_categories = dataset_cfg.get("categories", "all")
    categories = (
        discover_categories(root)
        if configured_categories == "all"
        else sorted(configured_categories)
    )
    if not categories:
        raise ValueError("No CO3Dv2 categories were selected or discovered")
    expected_category_count = dataset_cfg.get("expected_category_count")
    if expected_category_count is not None and len(categories) != int(
        expected_category_count
    ):
        raise ValueError(
            f"Expected {expected_category_count} categories, found {len(categories)}"
        )
    if dataset_cfg.get("version") == "CO3Dv2-v2_231130":
        unknown = sorted(set(categories) - CO3DV2_231130_CATEGORIES)
        if unknown:
            raise ValueError(
                "Unknown CO3Dv2-v2_231130 categories: " + ", ".join(unknown)
            )
        if configured_categories == "all" and set(categories) != (
            CO3DV2_231130_CATEGORIES
        ):
            missing = sorted(CO3DV2_231130_CATEGORIES - set(categories))
            raise ValueError(
                "Full CO3Dv2-v2_231130 run is missing official categories: "
                + ", ".join(missing)
            )
    missing_categories = [
        category for category in categories if not (root / category).is_dir()
    ]
    if missing_categories:
        raise FileNotFoundError(
            f"Missing category directories: {', '.join(missing_categories)}"
        )

    seed = int(sampling_cfg.get("seed", 0))
    window_length = int(sampling_cfg["window_length"])
    windows_per_sequence = int(sampling_cfg["windows_per_sequence"])
    sequence_caps = sampling_cfg["sequence_caps"]
    require_viewpoint = bool(dataset_cfg.get("require_viewpoint", True))
    require_local_files = bool(dataset_cfg.get("require_local_files", True))
    preprocessing_cfg = config["preprocessing"]

    sequence_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    for category in categories:
        category_dir = root / category
        splits = load_official_sequence_splits(category_dir)
        split_by_sequence = {
            sequence_id: split
            for split, sequence_ids in splits.items()
            for sequence_id in sequence_ids
        }
        sequence_annotations = load_jgzip(category_dir / "sequence_annotations.jgz")
        frame_annotations = load_jgzip(category_dir / "frame_annotations.jgz")
        annotated_frame_counts = Counter(
            row["sequence_name"] for row in frame_annotations
        )
        sequence_annotation_by_id = {
            row["sequence_name"]: row for row in sequence_annotations
        }
        annotated_frames: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for annotation in frame_annotations:
            sequence_id = annotation["sequence_name"]
            split = split_by_sequence.get(sequence_id)
            if split is None:
                continue
            image = annotation.get("image")
            mask = annotation.get("mask")
            viewpoint = annotation.get("viewpoint")
            reason: str | None = None
            if not image or not image.get("path") or not image.get("size"):
                reason = "missing_image_annotation"
            elif not mask or not mask.get("path"):
                reason = "missing_mask_annotation"
            elif require_viewpoint and viewpoint is None:
                reason = "missing_viewpoint_annotation"
            image_path = (
                _dataset_path(root, image["path"])
                if image and image.get("path")
                else None
            )
            mask_path = (
                _dataset_path(root, mask["path"]) if mask and mask.get("path") else None
            )
            if reason is None and require_local_files and not image_path.is_file():
                reason = "missing_image_file"
            if reason is None and require_local_files and not mask_path.is_file():
                reason = "missing_mask_file"
            if reason is not None:
                rejection_counts[reason] += 1
                continue
            height, width = (int(value) for value in image["size"])
            number = int(annotation["frame_number"])
            mask_mass = None if mask.get("mass") is None else float(mask["mass"])
            spatial_transform = compute_cut3r_transform(
                width,
                height,
                input_size=int(preprocessing_cfg["input_size"]),
                patch_size=int(preprocessing_cfg["patch_size"]),
                square_ok=bool(preprocessing_cfg["square_ok"]),
            )
            annotated_frames[sequence_id].append(
                {
                    "frame_id": frame_id(category, sequence_id, number),
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "category": category,
                    "sequence_id": sequence_id,
                    "split": split,
                    "frame_number": number,
                    "frame_timestamp": float(annotation["frame_timestamp"]),
                    "image_relpath": str(image["path"]),
                    "mask_relpath": str(mask["path"]),
                    "original_height": height,
                    "original_width": width,
                    "mask_mass": mask_mass,
                    "soft_foreground_fraction": (
                        None if mask_mass is None else mask_mass / float(height * width)
                    ),
                    "spatial_transform_json": json.dumps(
                        spatial_transform.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "viewpoint_json": _viewpoint_json(viewpoint),
                }
            )

        for split in ("train", "val", "test"):
            eligible = sorted(
                sequence_id
                for sequence_id in splits[split]
                if len(annotated_frames.get(sequence_id, [])) >= window_length
            )
            selected = set(
                choose_sequences(
                    eligible,
                    cap=None
                    if sequence_caps[split] is None
                    else int(sequence_caps[split]),
                    seed=seed,
                    category=category,
                    split=split,
                )
            )
            for sequence_id in sorted(splits[split]):
                annotation = sequence_annotation_by_id.get(sequence_id, {})
                usable_count = len(annotated_frames.get(sequence_id, []))
                is_selected = sequence_id in selected
                reason = None
                if usable_count < window_length:
                    reason = "fewer_than_six_usable_frames"
                elif not is_selected:
                    reason = "sequence_cap"
                sequence_rows.append(
                    {
                        "schema_version": MANIFEST_SCHEMA_VERSION,
                        "category": category,
                        "sequence_id": sequence_id,
                        "split": split,
                        "viewpoint_quality_score": annotation.get(
                            "viewpoint_quality_score"
                        ),
                        "annotated_frame_count": annotated_frame_counts[sequence_id],
                        "usable_frame_count": usable_count,
                        "selected": is_selected,
                        "exclusion_reason": reason,
                    }
                )
                if not is_selected:
                    if reason:
                        rejection_counts[reason] += 1
                    continue
                frames = sorted(
                    annotated_frames[sequence_id],
                    key=lambda row: (row["frame_number"], row["frame_id"]),
                )
                frame_rows.extend(frames)
                window_rows.extend(
                    generate_ordered_windows(
                        frames,
                        window_length=window_length,
                        windows_per_sequence=windows_per_sequence,
                    )
                )

    if not frame_rows or not window_rows:
        raise ValueError("The selected configuration produced no frames or windows")
    output_dir = Path(config["output"]["manifest_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences_path = output_dir / "sequences.parquet"
    frames_path = output_dir / "frames.parquet"
    windows_path = output_dir / "windows.parquet"
    write_parquet_atomic(sequences_path, sequence_rows)
    write_parquet_atomic(frames_path, frame_rows)
    write_parquet_atomic(windows_path, window_rows)
    summary = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_version": dataset_cfg.get("version", "CO3Dv2"),
        "categories": categories,
        "category_count": len(categories),
        "selected_sequence_count": sum(bool(row["selected"]) for row in sequence_rows),
        "selected_frame_count": len(frame_rows),
        "window_count": len(window_rows),
        "counts_by_split": {
            split: {
                "sequences": sum(
                    row["selected"] and row["split"] == split for row in sequence_rows
                ),
                "frames": sum(row["split"] == split for row in frame_rows),
                "windows": sum(row["split"] == split for row in window_rows),
            }
            for split in ("train", "val", "test")
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "manifest_sha256": {
            "sequences": sha256_file(sequences_path),
            "frames": sha256_file(frames_path),
            "windows": sha256_file(windows_path),
        },
        "sampling": {
            "seed": seed,
            "window_length": window_length,
            "windows_per_sequence": windows_per_sequence,
            "sequence_caps": sequence_caps,
        },
        "preprocessing": preprocessing_cfg,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary
