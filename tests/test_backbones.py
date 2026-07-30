"""Tests for the shared backbone contract and the embedding cache.

What this file covers:

- the :class:`Backbone` **contract** every backbone must satisfy - feature/grid
  validation, and pooling a target-frame mask onto the token grid;
- the **embedding cache** both backbones' layouts write to - round trip, resume,
  corruption detection, and the category vocabulary/index binding;
- the **reuse path** that attaches labels to existing Stage 0 embeddings,
  including its guard against a mask that changed since extraction;
- the parts of the real baseline backbones that run **without weights or data**:
  DINOv2's patch-14 geometry and its extraction path (via an injected fake model),
  and CUT3R's seeded random re-initialization.

What it deliberately does not cover: running real CUT3R or DINOv2 weights. Those
need a GPU, the pinned checkpoint / hub download, and CO3D files, so the contract
is exercised with lightweight fake backbones here and the real models are
validated on the VM during extraction.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from src.backbones import BackboneConfig, build_backbone
from src.backbones.base import (
    TARGET_FRAME_INDEX,
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
    load_target_tokens,
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


@pytest.mark.parametrize(
    "layout,backbone", [("target_only", _TargetOnlyBackbone()), ("trajectory", _TrajectoryBackbone())]
)
def test_load_target_tokens_matches_whole_entry_read(tmp_path, layout, backbone) -> None:
    """The probe read path must return exactly what the whole-entry read returns.

    ``load_target_tokens`` slices the target frame out inside the shard read to
    avoid transferring the other five states and the state latents. That is only a
    valid optimization if the values are identical - and if it picks state 5 and no
    other. The trajectory fixture fills every state with different random values,
    so reading the wrong one cannot pass.
    """
    windows, frame_by_id = _windows_and_frames()
    extract_to_cache(
        backbone, layout=layout, windows=windows, frame_by_id=frame_by_id,
        dataset_root=".", cache_dir=tmp_path, contract={}, windows_per_shard=4,
        verify_source_hashes=False,
    )
    for row in load_probe_index(tmp_path):
        whole = load_embedding_sample(tmp_path, row)
        spatial, labels = load_target_tokens(tmp_path, row)
        assert torch.equal(spatial, whole.target_spatial())
        assert torch.equal(labels, whole.seg_labels)
        assert spatial.shape == (_GRID[0] * _GRID[1], _DIM)
        assert labels.shape == _GRID
        if layout == "trajectory":
            # Specifically state 5, not merely "some state".
            matching = [t for t in range(6) if torch.equal(whole.image_tokens[t, 0], spatial)]
            assert matching == [TARGET_FRAME_INDEX]


def test_load_target_tokens_rejects_grid_disagreement(tmp_path) -> None:
    """A row whose token_grid contradicts the stored tensors must fail loudly."""
    windows, frame_by_id = _windows_and_frames()
    extract_to_cache(
        _TargetOnlyBackbone(), layout="target_only", windows=windows, frame_by_id=frame_by_id,
        dataset_root=".", cache_dir=tmp_path, contract={}, windows_per_shard=4,
        verify_source_hashes=False,
    )
    row = dict(load_probe_index(tmp_path)[0], token_grid=[_GRID[0] + 1, _GRID[1]])
    with pytest.raises(ValueError, match="disagree with token_grid"):
        load_target_tokens(tmp_path, row)


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


def test_dinov2_grid_is_aspect_preserving_multiple_of_patch() -> None:
    """DINOv2 uses a 14-pixel patch, so its grid geometry differs from CUT3R's."""
    from src.backbones.dinov2 import _PATCH_SIZE, _target_grid

    for width, height in ((640, 480), (480, 640), (800, 800)):
        grid_h, grid_w = _target_grid(width, height, image_size=518)
        assert grid_h % _PATCH_SIZE == 0 and grid_w % _PATCH_SIZE == 0
        assert max(grid_h, grid_w) <= 518
        # Landscape stays landscape, portrait stays portrait.
        assert (width > height) == (grid_w > grid_h) or width == height


def test_dinov2_backbone_extracts_without_downloading_weights(tmp_path) -> None:
    """The DINOv2 extraction path is exercised with an injected fake model.

    Real weights need a torch.hub download, so the backbone accepts a
    ``model_loader``; this checks the preprocessing, token grid, and
    mask alignment without any network access.
    """
    import numpy as np
    from PIL import Image

    from src.backbones.dinov2 import _PATCH_SIZE, Dinov2Backbone

    dataset_root = tmp_path / "co3d"
    dataset_root.mkdir()
    mask_array = np.zeros((224, 224), dtype=np.uint8)
    mask_array[:112, :] = 255  # top half foreground
    Image.new("RGB", (224, 224), (10, 20, 30)).save(dataset_root / "rgb.png")
    Image.fromarray(mask_array, mode="L").save(dataset_root / "mask.png")
    frame_rows = [
        {"frame_id": f"f{i}", "image_relpath": "rgb.png", "mask_relpath": "mask.png"}
        for i in range(6)
    ]

    class _FakeDinov2(torch.nn.Module):
        def forward_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
            height, width = image.shape[-2:]
            patches = (height // _PATCH_SIZE) * (width // _PATCH_SIZE)
            return {
                "x_norm_patchtokens": torch.randn(1, patches, 768),
                "x_norm_clstoken": torch.randn(1, 768),
            }

    backbone = Dinov2Backbone(
        image_size=224, device="cpu", model_loader=lambda variant: _FakeDinov2()
    )
    extraction = backbone.extract_window(frame_rows, dataset_root=str(dataset_root))
    grid_h, grid_w = extraction.features.token_grid
    assert extraction.features.spatial_tokens.shape == (grid_h * grid_w, 768)
    assert extraction.features.global_tokens.shape == (1, 768)
    assert extraction.features.frame_id == "f5"  # the window's target frame
    labels = extraction.target_labels()
    assert labels.shape == (grid_h, grid_w)
    assert float(labels.mean()) == pytest.approx(0.5, abs=0.1)  # top half foreground
    assert backbone.provenance()["patch_size"] == _PATCH_SIZE


def test_random_cut3r_reinitialization_is_seeded_and_changes_weights() -> None:
    """The random baseline must be reproducible and must actually re-initialize."""
    from src.backbones.cut3r import randomize_weights

    def _model(seed: int) -> torch.nn.Module:
        torch.manual_seed(1234)  # identical starting weights every time
        module = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
        randomize_weights(module, seed=seed, strategy="reset_parameters")
        return module

    torch.manual_seed(1234)
    original = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 4))
    same_a, same_b, different = _model(7), _model(7), _model(8)

    assert torch.equal(same_a[0].weight, same_b[0].weight)  # same seed reproduces
    assert not torch.equal(same_a[0].weight, different[0].weight)  # seed matters
    assert not torch.equal(same_a[0].weight, original[0].weight)  # weights changed
    with pytest.raises(ValueError):
        randomize_weights(original, seed=0, strategy="not-a-strategy")


def test_build_backbone_dispatch_and_validation() -> None:
    with pytest.raises(ValueError):
        build_backbone({"kind": "does-not-exist"})
    with pytest.raises(ValueError):
        build_backbone(BackboneConfig(kind="dinov2", weights="random"))
