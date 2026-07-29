from __future__ import annotations

from typing import Any

import pytest
import torch

from model_segmentation import HeadConfig, MLPHead, SegmentationProbe, build_probe
from segmentation_dataset import ProbeCacheDataset, assert_sequence_disjoint, collate_windows
from src.backbones.base import Backbone, BackboneFeatures, WindowExtraction
from src.backbones.probe_cache import extract_to_cache
from segmentation_inference import run_inference
from train_segmentation import train_from_config


class _SeparableBackbone(Backbone):
    """Features linearly separable by the top/bottom-half mask, so a probe can learn."""

    name = "separable"

    def __init__(self, *, grid: tuple[int, int] = (4, 4), dim: int = 4) -> None:
        self.grid = grid
        self.dim = dim

    def extract_window(self, frame_rows: list[dict[str, Any]], *, dataset_root: str) -> WindowExtraction:
        grid_h, grid_w = self.grid
        tokens = torch.zeros(grid_h * grid_w, self.dim)
        for row in range(grid_h):
            signal = 1.0 if row < grid_h / 2 else -1.0
            for col in range(grid_w):
                tokens[row * grid_w + col, 0] = signal
        features = BackboneFeatures(
            spatial_tokens=tokens,
            token_grid=self.grid,
            global_tokens=torch.ones(1, self.dim),
            frame_id=frame_rows[-1]["frame_id"],
        )
        mask = torch.zeros(grid_h * 8, grid_w * 8)
        mask[: (grid_h * 8) // 2, :] = 1.0  # top half foreground -> top rows are class 1
        return WindowExtraction(features=features, target_mask=mask)

    def provenance(self) -> dict[str, Any]:
        return {"backbone": self.name}


def _build_cache(tmp_path, *, count: int = 8):
    windows = [
        {
            "window_id": f"w{i}",
            "frame_ids": [f"{i}_{j}" for j in range(6)],
            "category": "apple" if i % 2 else "ball",
            "sequence_id": f"seq{i}",
            "split": "train" if i < count - 2 else "val",
        }
        for i in range(count)
    ]
    frame_by_id = {
        fid: {"frame_id": fid, "image_relpath": "x", "mask_relpath": "y"}
        for window in windows
        for fid in window["frame_ids"]
    }
    extract_to_cache(
        _SeparableBackbone(), layout="target_only", windows=windows, frame_by_id=frame_by_id,
        dataset_root=".", cache_dir=tmp_path, contract={}, windows_per_shard=4,
        verify_source_hashes=False,
    )
    return tmp_path


def test_mlp_head_linear_vs_nonlinear() -> None:
    linear = MLPHead(HeadConfig(feature_dim=8, hidden_dims=()))
    assert linear.is_linear
    assert linear(torch.randn(3, 5, 8)).shape == (3, 5, 1)
    mlp = MLPHead(HeadConfig(feature_dim=8, hidden_dims=(16,), num_classes=1))
    assert not mlp.is_linear
    assert mlp(torch.randn(7, 8)).shape == (7, 1)


def test_probe_logit_grid_shape() -> None:
    probe = build_probe({"feature_dim": 8, "num_classes": 1})
    logits = probe.logit_grid(torch.randn(2, 16, 8), (4, 4))
    assert logits.shape == (2, 1, 4, 4)
    with torch.no_grad():
        probability = probe.predict_mask(torch.randn(2, 16, 8), (4, 4), output_size=(64, 64))
    assert probability.shape == (2, 1, 64, 64)
    assert float(probability.min()) >= 0.0 and float(probability.max()) <= 1.0


def test_collate_concatenates_variable_grids() -> None:
    batch = [
        {"spatial": torch.randn(16, 8), "labels": torch.zeros(16), "count": 16, "token_grid": (4, 4), "window_id": "a", "category": "apple"},
        {"spatial": torch.randn(9, 8), "labels": torch.ones(9), "count": 9, "token_grid": (3, 3), "window_id": "b", "category": "ball"},
    ]
    collated = collate_windows(batch)
    assert collated["spatial"].shape == (25, 8)
    assert collated["counts"].tolist() == [16, 9]


def test_sequence_disjoint_guard(tmp_path) -> None:
    cache = _build_cache(tmp_path)
    train_set = ProbeCacheDataset(cache, split="train")
    val_set = ProbeCacheDataset(cache, split="val")
    assert_sequence_disjoint(train_set, val_set)  # disjoint by construction


def test_training_learns_separable_signal(tmp_path) -> None:
    cache = _build_cache(tmp_path)
    config = {
        "experiment": "smoke",
        "probe_cache": {"dir": str(cache)},
        "model": {"feature_dim": 4, "num_classes": 1, "hidden_dims": []},
        "training": {"epochs": 60, "batch_size": 4, "lr": 0.05, "seed": 0, "device": "cpu"},
        "splits": {"train": "train", "val": "val"},
        "output": {"dir": str(tmp_path / "runs")},
    }
    result = train_from_config(config)
    assert result["is_linear_probe"] is True
    final = result["final_val"]
    assert final["token_accuracy"] > 0.95
    assert final["macro_foreground_iou"] > 0.9
    assert (tmp_path / "runs" / "metrics.json").is_file()
    assert (tmp_path / "runs" / "head.pt").is_file()


def test_inference_reloads_trained_probe(tmp_path) -> None:
    cache = _build_cache(tmp_path)
    config = {
        "experiment": "smoke",
        "probe_cache": {"dir": str(cache)},
        "model": {"feature_dim": 4, "num_classes": 1, "hidden_dims": []},
        "training": {"epochs": 60, "batch_size": 4, "lr": 0.05, "seed": 0, "device": "cpu"},
        "splits": {"train": "train", "val": "val"},
        "output": {"dir": str(tmp_path / "runs")},
    }
    train_from_config(config)
    result = run_inference(
        config, split="val", save_dir=str(tmp_path / "infer"), save_masks=True
    )
    assert result["metrics"]["macro_foreground_iou"] > 0.9
    assert len(result["per_window_iou"]) == result["windows"]
    assert (tmp_path / "infer" / "inference-val.json").is_file()
    assert (tmp_path / "infer" / "masks-val.pt").is_file()
