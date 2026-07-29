from __future__ import annotations

from typing import Any

import pytest
import torch

from src.backbones import BackboneConfig, build_backbone
from src.backbones.base import (
    Backbone,
    BackboneFeatures,
    TrajectoryExtraction,
    WindowExtraction,
    pool_mask_to_grid,
)
from src.backbones.probe_cache import (
    EmbeddingSample,
    category_index_map,
    category_vocabulary,
    extract_to_cache,
    load_embedding_sample,
    load_probe_index,
    verify_probe_cache,
)

_GRID = (4, 4)
_DIM = 8
# Two real CO3D categories so the vocabulary lookup succeeds.
_CATEGORIES = ["apple", "ball"]


def _top_half_mask() -> torch.Tensor:
    mask = torch.zeros(_GRID[0] * 8, _GRID[1] * 8)
    mask[: (_GRID[0] * 8) // 2, :] = 1.0
    return mask


class _TargetOnlyBackbone(Backbone):
    name = "fake-target"

    def extract_window(self, frame_rows: list[dict[str, Any]], *, dataset_root: str) -> WindowExtraction:
        num = _GRID[0] * _GRID[1]
        features = BackboneFeatures(
            spatial_tokens=torch.randn(num, _DIM),
            token_grid=_GRID,
            global_tokens=torch.randn(1, _DIM),
            frame_id=frame_rows[-1]["frame_id"],
        )
        return WindowExtraction(features=features, target_mask=_top_half_mask())

    def provenance(self) -> dict[str, Any]:
        return {"backbone": self.name}


class _TrajectoryBackbone(Backbone):
    name = "fake-trajectory"

    def extract_window(self, frame_rows, *, dataset_root):  # pragma: no cover - unused
        raise NotImplementedError

    def extract_trajectory(self, frame_rows: list[dict[str, Any]], *, dataset_root: str) -> TrajectoryExtraction:
        num = _GRID[0] * _GRID[1]
        return TrajectoryExtraction(
            image_tokens=torch.randn(6, 1, num, _DIM),
            state_tokens=torch.randn(6, 1, _DIM, _DIM),
            token_grid=_GRID,
            frame_ids=[row["frame_id"] for row in frame_rows],
            target_mask=_top_half_mask(),
        )

    def provenance(self) -> dict[str, Any]:
        return {"backbone": self.name}


def _windows_and_frames(count: int = 6):
    windows = [
        {
            "window_id": f"w{i}",
            "frame_ids": [f"{i}_{j}" for j in range(6)],
            "category": _CATEGORIES[i % 2],
            "sequence_id": f"seq{i}",
            "split": "train" if i < 4 else "val",
        }
        for i in range(count)
    ]
    frame_by_id = {
        fid: {"frame_id": fid, "image_relpath": "x", "mask_relpath": "y"}
        for window in windows
        for fid in window["frame_ids"]
    }
    return windows, frame_by_id


def test_pool_mask_to_grid_top_half_foreground() -> None:
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., :16, :] = 255
    labels = pool_mask_to_grid(mask, (4, 4), threshold=0.5)
    assert torch.equal(labels.sum(dim=1), torch.tensor([4.0, 4.0, 0.0, 0.0]))


def test_vocabulary_is_deterministic_and_indexed() -> None:
    vocab = category_vocabulary()
    assert len(vocab) == 51 and vocab == sorted(vocab)
    index_of = category_index_map()
    assert vocab[index_of["apple"]] == "apple"


@pytest.mark.parametrize("layout,backbone", [("target_only", _TargetOnlyBackbone()), ("trajectory", _TrajectoryBackbone())])
def test_embedding_cache_round_trip(tmp_path, layout, backbone) -> None:
    windows, frame_by_id = _windows_and_frames()
    result = extract_to_cache(
        backbone, layout=layout, windows=windows, frame_by_id=frame_by_id,
        dataset_root=".", cache_dir=tmp_path, contract={}, windows_per_shard=4,
        verify_source_hashes=False,
    )
    assert result["written"] == 6
    assert result["verification"]["valid"] and result["verification"]["layout"] == layout
    rows = load_probe_index(tmp_path)
    sample = load_embedding_sample(tmp_path, rows[0])
    assert isinstance(sample, EmbeddingSample)
    assert sample.layout == layout
    assert sample.target_spatial().shape == (16, _DIM)  # layout-agnostic accessor
    assert sample.seg_labels.shape == _GRID
    assert float(sample.seg_labels.mean()) == 0.5  # top half foreground
    assert category_vocabulary()[sample.category_index] == sample.category
    # Resuming skips completed windows.
    again = extract_to_cache(
        backbone, layout=layout, windows=windows, frame_by_id=frame_by_id,
        dataset_root=".", cache_dir=tmp_path, contract={}, windows_per_shard=4,
        verify_source_hashes=False,
    )
    assert again["written"] == 0 and again["skipped"] == 6


def test_embedding_cache_detects_corruption(tmp_path) -> None:
    windows, frame_by_id = _windows_and_frames()
    extract_to_cache(
        _TargetOnlyBackbone(), layout="target_only", windows=windows, frame_by_id=frame_by_id,
        dataset_root=".", cache_dir=tmp_path, contract={}, verify_source_hashes=False,
    )
    shard = next(tmp_path.glob("shard-*.safetensors"))
    shard.write_bytes(shard.read_bytes() + b"corruption")
    with pytest.raises(ValueError):
        verify_probe_cache(tmp_path)


def test_extract_rejects_unknown_category(tmp_path) -> None:
    windows, frame_by_id = _windows_and_frames()
    windows[0]["category"] = "not-a-co3d-category"
    with pytest.raises(KeyError):
        extract_to_cache(
            _TargetOnlyBackbone(), layout="target_only", windows=windows, frame_by_id=frame_by_id,
            dataset_root=".", cache_dir=tmp_path, contract={}, verify_source_hashes=False,
        )


def _build_stage0_trajectory_cache(tmp_path):
    """A minimal real Stage 0 trajectory cache + on-disk CO3D frames for reuse tests."""
    import numpy as np
    from PIL import Image

    from src.common.io import sha256_file
    from src.embeddings.cache import FeatureCacheWriter
    from src.embeddings.types import FeatureTrajectory

    dataset_root = tmp_path / "co3d"
    (dataset_root / "seq0").mkdir(parents=True)
    frame_ids = [f"seq0/f{j}" for j in range(6)]
    frame_by_id = {}
    source_sha256 = {}
    top_half = np.zeros((512, 512), dtype=np.uint8)
    top_half[:256, :] = 255
    for frame_id in frame_ids:
        image_rel = f"{frame_id}_rgb.png"
        mask_rel = f"{frame_id}_mask.png"
        Image.new("RGB", (512, 512), (120, 120, 120)).save(dataset_root / image_rel)
        Image.fromarray(top_half, mode="L").save(dataset_root / mask_rel)
        frame_by_id[frame_id] = {"frame_id": frame_id, "image_relpath": image_rel, "mask_relpath": mask_rel}
        source_sha256[frame_id] = {
            "image": sha256_file(dataset_root / image_rel),
            "mask": sha256_file(dataset_root / mask_rel),
        }
    token_grid = (24, 32)  # CUT3R 512/patch-16 geometry for a square image
    num = token_grid[0] * token_grid[1]
    trajectory = FeatureTrajectory(
        image_tokens=torch.randn(6, 1, num, 768),
        state_tokens=torch.randn(6, 1, 768, 768),
        frame_ids=frame_ids,
        token_grid=token_grid,
    )
    traj_dir = tmp_path / "stage0"
    with FeatureCacheWriter(traj_dir, contract={}) as writer:
        writer.add("w0", trajectory, source_sha256=source_sha256)
    window = {
        "window_id": "w0", "frame_ids": frame_ids, "category": "apple",
        "sequence_id": "seq0", "split": "train",
    }
    return traj_dir, str(dataset_root), [window], frame_by_id, token_grid


def test_reuse_trajectory_cache_attaches_labels(tmp_path) -> None:
    from src.backbones.probe_cache import attach_labels_from_trajectory_cache

    traj_dir, dataset_root, windows, frame_by_id, token_grid = _build_stage0_trajectory_cache(tmp_path)
    result = attach_labels_from_trajectory_cache(
        traj_dir, windows=windows, frame_by_id=frame_by_id, dataset_root=dataset_root,
        cache_dir=tmp_path / "v2", contract={},
    )
    assert result["written"] == 1
    assert result["verification"]["valid"] and result["verification"]["layout"] == "trajectory"
    sample = load_embedding_sample(tmp_path / "v2", load_probe_index(tmp_path / "v2")[0])
    assert sample.layout == "trajectory"
    assert sample.seg_labels.shape == token_grid
    assert 0.0 < float(sample.seg_labels.mean()) < 1.0  # top-half mask -> mixed labels
    assert sample.target_spatial().shape == (token_grid[0] * token_grid[1], 768)


def test_reuse_rejects_mask_that_changed_since_extraction(tmp_path) -> None:
    from PIL import Image

    from src.backbones.probe_cache import attach_labels_from_trajectory_cache

    traj_dir, dataset_root, windows, frame_by_id, _ = _build_stage0_trajectory_cache(tmp_path)
    # Tamper with the target-frame mask so its SHA no longer matches the cache record.
    target_mask_rel = frame_by_id[windows[0]["frame_ids"][5]]["mask_relpath"]
    Image.new("L", (512, 512), 0).save(f"{dataset_root}/{target_mask_rel}")
    with pytest.raises(ValueError):
        attach_labels_from_trajectory_cache(
            traj_dir, windows=windows, frame_by_id=frame_by_id, dataset_root=dataset_root,
            cache_dir=tmp_path / "v2", contract={},
        )


def test_build_backbone_dispatch_and_validation() -> None:
    with pytest.raises(ValueError):
        build_backbone({"kind": "does-not-exist"})
    with pytest.raises(ValueError):
        build_backbone(BackboneConfig(kind="dinov2", weights="random"))
