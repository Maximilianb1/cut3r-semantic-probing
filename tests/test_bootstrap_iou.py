"""Tests for the sequence-cluster bootstrap on segmentation macro-IoU."""

from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.bootstrap_iou import (
    assert_paired_clusters,
    bootstrap_iou_ci,
    bootstrap_iou_difference,
    iou_clusters,
)


def _inference(rows: list[dict]) -> dict:
    return {"per_window_iou": rows}


def test_iou_clusters_averages_a_sequences_windows_before_the_macro_average() -> None:
    # seq-A: two windows, IoU 1.0 and 0.0 -> sequence_iou 0.5
    # seq-B: one window, IoU 0.8 -> sequence_iou 0.8
    inference = _inference([
        {"window_id": "w0", "sequence_id": "seq-A", "foreground_iou": 1.0},
        {"window_id": "w1", "sequence_id": "seq-A", "foreground_iou": 0.0},
        {"window_id": "w2", "sequence_id": "seq-B", "foreground_iou": 0.8},
    ])
    clusters = iou_clusters(inference)

    assert clusters.sequence_ids == ("seq-A", "seq-B")
    assert clusters.sequence_iou == pytest.approx([0.5, 0.8])
    # Macro-IoU here weights seq-A and seq-B equally (0.5, 0.8), NOT the 3
    # underlying windows equally (which would give (1+0+0.8)/3 = 0.6) - that
    # equal-weight-per-sequence property is the whole point of clustering.
    assert clusters.macro_iou == pytest.approx(0.65)


def test_iou_clusters_rejects_duplicate_window_ids() -> None:
    inference = _inference([
        {"window_id": "w0", "sequence_id": "seq-A", "foreground_iou": 1.0},
        {"window_id": "w0", "sequence_id": "seq-A", "foreground_iou": 0.0},
    ])
    with pytest.raises(ValueError, match="Duplicate window_id"):
        iou_clusters(inference)


def test_assert_paired_clusters_rejects_mismatched_sequences() -> None:
    left = iou_clusters(_inference([{"window_id": "w0", "sequence_id": "seq-A", "foreground_iou": 1.0}]))
    right = iou_clusters(_inference([{"window_id": "w1", "sequence_id": "seq-B", "foreground_iou": 1.0}]))
    with pytest.raises(ValueError, match="disagree"):
        assert_paired_clusters(left, right)


def test_bootstrap_ci_collapses_to_a_point_when_every_sequence_agrees() -> None:
    # Zero variance across sequences -> every resample gives the same mean.
    inference = _inference([
        {"window_id": f"w{i}", "sequence_id": f"seq-{i}", "foreground_iou": 0.7}
        for i in range(5)
    ])
    clusters = iou_clusters(inference)
    result = bootstrap_iou_ci(clusters, samples=500, seed=1)

    assert result["clusters"] == 5
    assert result["macro_iou"]["estimate"] == pytest.approx(0.7)
    assert result["macro_iou"]["ci95"][0] == pytest.approx(0.7)
    assert result["macro_iou"]["ci95"][1] == pytest.approx(0.7)


def test_bootstrap_ci_widens_with_more_within_sequence_variance() -> None:
    # Same 4 sequences, but one version has each sequence's windows agreeing
    # (low variance) and the other has them split evenly between 0 and 1 - the
    # sequence-mean macro-IoU is identical (0.5) in both, but resampling
    # sequences should show a wider CI for the noisier one.
    generator = np.random.default_rng(0)
    agreeing = _inference([
        {"window_id": f"a{i}", "sequence_id": f"seq-{i}", "foreground_iou": 0.5}
        for i in range(20)
    ])
    noisy_rows = []
    for i in range(20):
        value = 0.0 if i % 2 == 0 else 1.0
        noisy_rows.append({"window_id": f"n{i}", "sequence_id": f"seq-{i}", "foreground_iou": value})
    noisy = _inference(noisy_rows)

    agreeing_ci = bootstrap_iou_ci(iou_clusters(agreeing), samples=5000, seed=2)
    noisy_ci = bootstrap_iou_ci(iou_clusters(noisy), samples=5000, seed=2)

    agreeing_width = agreeing_ci["macro_iou"]["ci95"][1] - agreeing_ci["macro_iou"]["ci95"][0]
    noisy_width = noisy_ci["macro_iou"]["ci95"][1] - noisy_ci["macro_iou"]["ci95"][0]
    assert noisy_width > agreeing_width
    del generator


def test_paired_difference_is_zero_when_target_equals_reference() -> None:
    inference = _inference([
        {"window_id": f"w{i}", "sequence_id": f"seq-{i}", "foreground_iou": 0.3 + 0.1 * i}
        for i in range(6)
    ])
    clusters = iou_clusters(inference)
    result = bootstrap_iou_difference(clusters, clusters, samples=2000, seed=3)

    assert result["macro_iou_difference"]["estimate"] == pytest.approx(0.0)
    assert result["macro_iou_difference"]["ci95"][0] == pytest.approx(0.0, abs=1e-9)
    assert result["macro_iou_difference"]["ci95"][1] == pytest.approx(0.0, abs=1e-9)


def test_paired_difference_recovers_a_known_gap() -> None:
    # Target beats reference by exactly 0.2 on every sequence. Same window_ids
    # for both - a paired comparison means the two runs were scored on the
    # exact same physical test windows, only the model differs.
    target = iou_clusters(_inference([
        {"window_id": f"w{i}", "sequence_id": f"seq-{i}", "foreground_iou": 0.6}
        for i in range(10)
    ]))
    reference = iou_clusters(_inference([
        {"window_id": f"w{i}", "sequence_id": f"seq-{i}", "foreground_iou": 0.4}
        for i in range(10)
    ]))
    result = bootstrap_iou_difference(target, reference, samples=2000, seed=4)

    assert result["macro_iou_difference"]["estimate"] == pytest.approx(0.2)
    assert result["macro_iou_difference"]["ci95"][0] == pytest.approx(0.2, abs=1e-9)
    assert result["macro_iou_difference"]["ci95"][1] == pytest.approx(0.2, abs=1e-9)
