from __future__ import annotations

import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
import yaml

from src.common.tables import read_parquet
from src.data.co3d import build_manifests
from src.data.co3d_selective import (
    extract_verified_metadata_archive,
    find_required_members,
    in_memory_archive_opener,
    materialize_required_members,
    plan_selective_download,
    run_selective_download,
)


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with ZipFile(payload, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    return payload.getvalue()


def test_selective_plan_matches_manifest_window_sampling(
    synthetic_co3d: tuple[Path, Path, Path],
) -> None:
    _root, config_path, manifest_dir = synthetic_co3d
    summary = build_manifests(config_path)
    plan = plan_selective_download(config_path)
    manifest_windows = read_parquet(manifest_dir / "windows.parquet")
    assert plan["counts"] == {
        "sequences": 3,
        "windows": 6,
        "frames": 36,
        "files": 72,
    }
    assert {row["window_id"] for row in plan["selected_windows"]} == {
        row["window_id"] for row in manifest_windows
    }
    assert plan["counts"]["windows"] == summary["window_count"]
    assert len({row["path"] for row in plan["required_files"]}) == 72


def test_selective_plan_rejects_unbounded_caps(
    synthetic_co3d: tuple[Path, Path, Path],
) -> None:
    _root, config_path, _manifest_dir = synthetic_co3d
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["sampling"]["sequence_caps"]["train"] = None
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="finite sequence cap for train"):
        plan_selective_download(config_path)


def test_metadata_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "ball_000.zip"
    archive_path.write_bytes(_zip_bytes({"ball/set_lists/../../escape.txt": b"unsafe"}))
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        extract_verified_metadata_archive(
            archive_path, dataset_root=tmp_path / "dataset", category="ball"
        )
    assert not (tmp_path / "escape.txt").exists()


def test_find_and_materialize_remote_members_is_resumable(tmp_path: Path) -> None:
    first_url = "https://example.test/ball_001.zip"
    second_url = "https://example.test/ball_002.zip"
    image_path = "ball/sequence/images/frame000001.jpg"
    mask_path = "ball/sequence/masks/frame000001.png"
    archives = {
        first_url: _zip_bytes({image_path: b"jpeg-bytes", "unused": b"ignore"}),
        second_url: _zip_bytes({mask_path: b"png-bytes"}),
    }
    opener = in_memory_archive_opener(archives)
    sources = find_required_members(
        [first_url, second_url],
        [image_path, mask_path],
        archive_opener=opener,
    )
    assert set(sources) == {image_path, mask_path}
    first = materialize_required_members(
        sources, dataset_root=tmp_path / "dataset", archive_opener=opener
    )
    assert {row["status"] for row in first} == {"downloaded"}
    second = materialize_required_members(
        sources, dataset_root=tmp_path / "dataset", archive_opener=opener
    )
    assert {row["status"] for row in second} == {"verified_existing"}
    assert all(len(row["sha256"]) == 64 for row in second)


def test_materialize_rejects_corrupt_existing_file(tmp_path: Path) -> None:
    url = "https://example.test/ball_001.zip"
    relative = "ball/sequence/images/frame000001.jpg"
    archives = {url: _zip_bytes({relative: b"expected"})}
    opener = in_memory_archive_opener(archives)
    sources = find_required_members([url], [relative], archive_opener=opener)
    destination = tmp_path / "dataset" / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="size mismatch|CRC mismatch"):
        materialize_required_members(
            sources, dataset_root=tmp_path / "dataset", archive_opener=opener
        )


def test_missing_required_remote_member_is_rejected() -> None:
    url = "https://example.test/ball_001.zip"
    opener = in_memory_archive_opener({url: _zip_bytes({"unrelated": b"value"})})
    with pytest.raises(FileNotFoundError, match="Could not locate 1"):
        find_required_members([url], ["ball/missing.jpg"], archive_opener=opener)


def test_selective_download_modes_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        run_selective_download(
            "not-read.yaml",
            plan_only=True,
            index_only=True,
        )


def test_materialize_rejects_remote_member_metadata_change(tmp_path: Path) -> None:
    url = "https://example.test/ball_001.zip"
    relative = "ball/sequence/images/frame000001.jpg"
    original = in_memory_archive_opener({url: _zip_bytes({relative: b"original"})})
    changed = in_memory_archive_opener({url: _zip_bytes({relative: b"changed"})})
    sources = find_required_members([url], [relative], archive_opener=original)
    with pytest.raises(ValueError, match="metadata changed"):
        materialize_required_members(
            sources,
            dataset_root=tmp_path / "dataset",
            archive_opener=changed,
        )
