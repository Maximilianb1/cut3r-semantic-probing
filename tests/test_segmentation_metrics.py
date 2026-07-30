from __future__ import annotations

import sys
from pathlib import Path

import torch


_SEGMENTATION_DIR = Path(__file__).resolve().parents[1] / "segmentation_validation"
if str(_SEGMENTATION_DIR) not in sys.path:
    sys.path.insert(0, str(_SEGMENTATION_DIR))

from train_segmentation import BinaryMetrics


def _batch_stub() -> dict[str, object]:
    return {
        "counts": torch.tensor([4], dtype=torch.int64),
        "categories": ["apple"],
        "token_grids": [(2, 2)],
        "window_ids": ["w0"],
    }


def test_binary_metrics_skips_cpu_transfer_without_mask_collection(monkeypatch) -> None:
    cpu_calls: list[torch.Tensor] = []
    original_cpu = torch.Tensor.cpu

    def _spy_cpu(self: torch.Tensor, *args, **kwargs):
        cpu_calls.append(self)
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", _spy_cpu)

    metrics = BinaryMetrics(collect_windows=False, collect_masks=False)
    logits = torch.tensor([2.0, -2.0, 1.0, -1.0], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    metrics.update(logits, labels, _batch_stub(), loss=0.1)

    assert cpu_calls == []


def test_binary_metrics_collect_masks_moves_saved_masks_to_cpu(monkeypatch) -> None:
    cpu_calls: list[torch.Tensor] = []
    original_cpu = torch.Tensor.cpu

    def _spy_cpu(self: torch.Tensor, *args, **kwargs):
        cpu_calls.append(self)
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", _spy_cpu)

    metrics = BinaryMetrics(collect_windows=True, collect_masks=True)
    logits = torch.tensor([2.0, -2.0, 1.0, -1.0], dtype=torch.float32)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    metrics.update(logits, labels, _batch_stub(), loss=0.1)

    result = metrics.result()
    assert len(cpu_calls) >= 2
    assert result["windows"] == 1
    assert len(result["per_window"]) == 1
    assert result["per_window"][0]["predicted_labels"].device.type == "cpu"
    assert result["per_window"][0]["target_labels"].device.type == "cpu"