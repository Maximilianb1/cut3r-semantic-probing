from __future__ import annotations

import torch

from src.segmentation.analysis.build_score_comparison import precision_recall


def test_precision_recall_includes_micro_iou(tmp_path) -> None:
    # Two windows with known tp/fp/fn/tn so micro-IoU can be hand-checked.
    masks = {
        "w0": {
            "predicted_labels": torch.tensor([1, 1, 0, 0], dtype=torch.int64),
            "target_labels": torch.tensor([1, 0, 0, 0], dtype=torch.int64),
        },
        "w1": {
            "predicted_labels": torch.tensor([1, 0], dtype=torch.int64),
            "target_labels": torch.tensor([1, 1], dtype=torch.int64),
        },
    }
    torch.save(masks, tmp_path / "masks-test.pt")

    result = precision_recall(tmp_path, "test")

    # tp=2 (w0 idx0, w1 idx0), fp=1 (w0 idx1), fn=1 (w1 idx1), tn=2 (w0 idx2,3)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 1
    assert result["tn"] == 2
    assert result["micro_iou"] == 2 / (2 + 1 + 1)
    assert result["precision"] == 2 / (2 + 1)
    assert result["recall"] == 2 / (2 + 1)
