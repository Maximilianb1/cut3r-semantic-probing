"""Tests for the segmentation probe's metrics.

``BinaryMetrics`` produces the reported macro foreground IoU, so the arithmetic
is pinned here against hand-computed values rather than only exercised
end-to-end. It also covers the empty-target convention and the mask-collection
path that inference uses.
"""

from __future__ import annotations

import pytest
import torch

from src.segmentation.train_segmentation import BinaryMetrics


def _batch(counts, categories, grids, window_ids) -> dict[str, object]:
    return {
        "counts": torch.tensor(counts, dtype=torch.int64),
        "categories": list(categories),
        "token_grids": list(grids),
        "window_ids": list(window_ids),
    }


def _logits(labels) -> torch.Tensor:
    """Logits that decode to exactly ``labels`` under the fixed 0-threshold."""
    return torch.tensor([2.0 if value else -2.0 for value in labels], dtype=torch.float32)


def test_macro_iou_averages_windows_and_micro_iou_pools_tokens() -> None:
    # Window A: predict [1,1,0,0] against [1,0,0,0] -> intersection 1, union 2, IoU 0.5
    # Window B: predict [1,0]     against [1,1]     -> intersection 1, union 2, IoU 0.5
    metrics = BinaryMetrics()
    metrics.update(
        _logits([1, 1, 0, 0, 1, 0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0]),
        _batch([4, 2], ["apple", "bench"], [(2, 2), (1, 2)], ["a", "b"]),
    )
    result = metrics.result()

    assert result["windows"] == 2
    assert result["macro_foreground_iou"] == pytest.approx(0.5)
    # Pooled over tokens: intersection 2, union 4.
    assert result["micro_foreground_iou"] == pytest.approx(0.5)
    # 4 of 6 tokens predicted correctly.
    assert result["token_accuracy"] == pytest.approx(4 / 6)


def test_macro_and_micro_iou_diverge_when_windows_hold_uneven_foreground() -> None:
    # A large window scored perfectly and a small one scored zero: the window
    # average is 0.5, but pooling tokens weights the large window far more.
    metrics = BinaryMetrics()
    metrics.update(
        _logits([1, 1, 1, 1, 0, 0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        _batch([4, 2], ["apple", "bench"], [(2, 2), (1, 2)], ["big", "small"]),
    )
    result = metrics.result()

    assert result["macro_foreground_iou"] == pytest.approx(0.5)
    assert result["micro_foreground_iou"] == pytest.approx(4 / 5)


def test_per_category_iou_averages_within_a_category_before_reporting_it() -> None:
    metrics = BinaryMetrics()
    # Two apple windows, IoU 1.0 and 0.0; one bench window, IoU 1.0.
    metrics.update(
        _logits([1, 1, 0, 0, 1, 1]),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        _batch([2, 2, 2], ["apple", "apple", "bench"], [(1, 2)] * 3, ["a", "b", "c"]),
    )
    result = metrics.result()

    assert result["per_category_iou"] == pytest.approx({"apple": 0.5, "bench": 1.0})
    # The category mean weights categories equally, unlike the window macro.
    assert result["mean_category_iou"] == pytest.approx(0.75)
    assert result["macro_foreground_iou"] == pytest.approx(2 / 3)


def test_a_window_with_no_foreground_and_no_prediction_scores_one() -> None:
    # The empty/empty convention: an undefined 0/0 IoU is scored 1.0, not 0.0.
    metrics = BinaryMetrics()
    metrics.update(
        _logits([0, 0]),
        torch.tensor([0.0, 0.0]),
        _batch([2], ["apple"], [(1, 2)], ["empty"]),
    )
    result = metrics.result()

    assert result["macro_foreground_iou"] == pytest.approx(1.0)
    assert result["micro_foreground_iou"] == pytest.approx(1.0)


def test_loss_is_token_weighted_across_uneven_batches() -> None:
    metrics = BinaryMetrics()
    metrics.update(_logits([1] * 4), torch.ones(4), _batch([4], ["apple"], [(2, 2)], ["a"]), loss=1.0)
    metrics.update(_logits([1]), torch.ones(1), _batch([1], ["apple"], [(1, 1)], ["b"]), loss=6.0)

    # (1.0*4 + 6.0*1) / 5, not the mean of the two batch means (3.5).
    assert metrics.result()["loss"] == pytest.approx(2.0)


def test_per_window_records_appear_only_when_collection_is_requested() -> None:
    batch = _batch([4], ["apple"], [(2, 2)], ["w0"])
    labels = torch.tensor([1.0, 0.0, 0.0, 0.0])

    quiet = BinaryMetrics()
    quiet.update(_logits([1, 1, 0, 0]), labels, batch)
    assert "per_window" not in quiet.result()

    collecting = BinaryMetrics(collect_windows=True)
    collecting.update(_logits([1, 1, 0, 0]), labels, batch)
    record = collecting.result()["per_window"][0]
    assert record["window_id"] == "w0"
    assert record["category"] == "apple"
    assert record["token_grid"] == [2, 2]
    assert record["foreground_iou"] == pytest.approx(0.5)
    # Masks are extra weight; they are not saved unless inference asks for them.
    assert "predicted_labels" not in record


def test_collected_masks_are_reshaped_to_the_token_grid_and_moved_to_cpu() -> None:
    metrics = BinaryMetrics(collect_windows=True, collect_masks=True)
    metrics.update(
        _logits([1, 1, 0, 0]),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        _batch([4], ["apple"], [(2, 2)], ["w0"]),
    )
    record = metrics.result()["per_window"][0]

    assert record["predicted_labels"].shape == (2, 2)
    assert record["target_labels"].shape == (2, 2)
    assert record["predicted_labels"].device.type == "cpu"
    assert record["target_labels"].device.type == "cpu"
    assert record["predicted_labels"].tolist() == [[1.0, 1.0], [0.0, 0.0]]


def test_masks_are_not_moved_off_device_when_they_are_not_collected(monkeypatch) -> None:
    # The evaluation pass runs on GPU; a per-batch .cpu() there would cost a
    # synchronisation for masks nothing is going to read.
    calls: list[torch.Tensor] = []
    original = torch.Tensor.cpu
    monkeypatch.setattr(
        torch.Tensor, "cpu", lambda self, *a, **k: (calls.append(self), original(self, *a, **k))[1]
    )

    metrics = BinaryMetrics(collect_windows=False, collect_masks=False)
    metrics.update(
        _logits([1, 0, 1, 0]),
        torch.tensor([1.0, 0.0, 1.0, 0.0]),
        _batch([4], ["apple"], [(2, 2)], ["w0"]),
    )

    assert calls == []
