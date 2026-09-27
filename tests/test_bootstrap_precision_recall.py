"""Tests for the sequence-cluster bootstrap on segmentation precision/recall."""

from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.bootstrap_precision_recall import (
    assert_paired_clusters,
    bootstrap_precision_recall_ci,
    bootstrap_precision_recall_difference,
    precision_recall_clusters,
)


def _inference(rows: list[dict]) -> dict:
    return {"per_window_iou": rows}


def test_precision_recall_clusters_pools_counts_per_sequence() -> None:
    # seq-A: two windows (tp=8,fp=2,fn=0) and (tp=2,fp=0,fn=8) -> pooled tp=10, fp=2, fn=8
    inference = _inference([
        {"window_id": "w0", "sequence_id": "seq-A", "category": "apple", "tp": 8, "fp": 2, "fn": 0},
        {"window_id": "w1", "sequence_id": "seq-A", "category": "apple", "tp": 2, "fp": 0, "fn": 8},
        {"window_id": "w2", "sequence_id": "seq-B", "category": "bowl", "tp": 5, "fp": 5, "fn": 5},
    ])
    clusters = precision_recall_clusters(inference)

    assert clusters.sequence_ids == ("seq-A", "seq-B")
    assert clusters.categories == ("apple", "bowl")
    assert list(clusters.tp) == [10, 5]
    assert list(clusters.fp) == [2, 5]
    assert list(clusters.fn) == [8, 5]
    # Pooled precision/recall over ALL sequences (tp=15, fp=7, fn=13), not an
    # average of the two sequences' own precision/recall.
    assert clusters.precision == pytest.approx(15 / 22)
    assert clusters.recall == pytest.approx(15 / 28)


def test_precision_recall_clusters_rejects_duplicate_window_ids() -> None:
    inference = _inference([
        {"window_id": "w0", "sequence_id": "seq-A", "category": "apple", "tp": 1, "fp": 0, "fn": 0},
        {"window_id": "w0", "sequence_id": "seq-A", "category": "apple", "tp": 1, "fp": 0, "fn": 0},
    ])
    with pytest.raises(ValueError, match="Duplicate window_id"):
        precision_recall_clusters(inference)


def test_precision_recall_clusters_rejects_a_sequence_that_changes_category() -> None:
    inference = _inference([
        {"window_id": "w0", "sequence_id": "seq-A", "category": "apple", "tp": 1, "fp": 0, "fn": 0},
        {"window_id": "w1", "sequence_id": "seq-A", "category": "bowl", "tp": 1, "fp": 0, "fn": 0},
    ])
    with pytest.raises(ValueError, match="changes category"):
        precision_recall_clusters(inference)


def test_assert_paired_clusters_rejects_mismatched_sequences() -> None:
    left = precision_recall_clusters(_inference(
        [{"window_id": "w0", "sequence_id": "seq-A", "category": "apple", "tp": 1, "fp": 0, "fn": 0}]
    ))
    right = precision_recall_clusters(_inference(
        [{"window_id": "w1", "sequence_id": "seq-B", "category": "apple", "tp": 1, "fp": 0, "fn": 0}]
    ))
    with pytest.raises(ValueError, match="disagree"):
        assert_paired_clusters(left, right)


def test_bootstrap_ci_collapses_to_a_point_when_every_sequence_agrees() -> None:
    # Every sequence has the same tp/fp/fn -> zero variance across sequences.
    inference = _inference([
        {"window_id": f"w{i}", "sequence_id": f"seq-{i}", "category": "apple", "tp": 8, "fp": 2, "fn": 2}
        for i in range(5)
    ])
    clusters = precision_recall_clusters(inference)
    result = bootstrap_precision_recall_ci(clusters, samples=500, seed=1)

    assert result["clusters"] == 5
    assert result["precision"]["estimate"] == pytest.approx(0.8)
    assert result["precision"]["ci95"][0] == pytest.approx(0.8)
    assert result["precision"]["ci95"][1] == pytest.approx(0.8)
    assert result["recall"]["estimate"] == pytest.approx(0.8)


def test_paired_difference_is_zero_when_target_equals_reference() -> None:
    inference = _inference([
        {"window_id": f"w{i}", "sequence_id": f"seq-{i}", "category": "apple",
         "tp": 5 + i, "fp": 2, "fn": 1}
        for i in range(6)
    ])
    clusters = precision_recall_clusters(inference)
    result = bootstrap_precision_recall_difference(clusters, clusters, samples=2000, seed=3)

    assert result["precision_difference"]["estimate"] == pytest.approx(0.0)
    assert result["precision_difference"]["ci95"][0] == pytest.approx(0.0, abs=1e-9)
    assert result["precision_difference"]["ci95"][1] == pytest.approx(0.0, abs=1e-9)
    assert result["recall_difference"]["estimate"] == pytest.approx(0.0)


def test_bootstrap_ci_is_zero_not_nan_when_no_sequence_has_a_predicted_positive() -> None:
    # tp=fp=0 on every sequence -> precision's pooled denominator is always
    # zero, on every resample, not just some -- a valid outcome (a probe that
    # predicted no foreground at all), which SequencePrecisionRecallClusters
    # .precision defines as 0.0. The resampling used to divide by that zero
    # sum directly and produce NaN, poisoning the whole CI.
    inference = _inference([
        {"window_id": f"w{i}", "sequence_id": f"seq-{i}", "category": "apple", "tp": 0, "fp": 0, "fn": 5}
        for i in range(5)
    ])
    clusters = precision_recall_clusters(inference)
    result = bootstrap_precision_recall_ci(clusters, samples=500, seed=1)

    assert result["precision"]["estimate"] == 0.0
    assert not np.isnan(result["precision"]["ci95"]).any()
    assert result["precision"]["ci95"] == pytest.approx([0.0, 0.0])
    assert result["recall"]["estimate"] == pytest.approx(0.0)


def test_paired_difference_recovers_a_known_gap() -> None:
    # Target has zero false positives (perfect precision); reference has fp
    # equal to tp on every sequence (precision 0.5) - same recall on both
    # sides (fn=0 everywhere), isolating the precision gap.
    target = precision_recall_clusters(_inference([
        {"window_id": f"w{i}", "sequence_id": f"seq-{i}", "category": "apple", "tp": 10, "fp": 0, "fn": 0}
        for i in range(10)
    ]))
    reference = precision_recall_clusters(_inference([
        {"window_id": f"w{i}", "sequence_id": f"seq-{i}", "category": "apple", "tp": 10, "fp": 10, "fn": 0}
        for i in range(10)
    ]))
    result = bootstrap_precision_recall_difference(target, reference, samples=2000, seed=4)

    assert result["precision_difference"]["estimate"] == pytest.approx(0.5)
    assert result["precision_difference"]["ci95"][0] == pytest.approx(0.5, abs=1e-9)
    assert result["precision_difference"]["ci95"][1] == pytest.approx(0.5, abs=1e-9)
    assert result["recall_difference"]["estimate"] == pytest.approx(0.0)


def test_paired_difference_is_zero_not_nan_when_neither_side_predicts_a_positive() -> None:
    # Same all-zero-denominator case as the CI test, but through
    # _difference_distribution's separate target/reference division.
    clusters = precision_recall_clusters(_inference([
        {"window_id": f"w{i}", "sequence_id": f"seq-{i}", "category": "apple", "tp": 0, "fp": 0, "fn": 5}
        for i in range(5)
    ]))
    result = bootstrap_precision_recall_difference(clusters, clusters, samples=500, seed=5)

    assert result["precision_difference"]["estimate"] == 0.0
    assert not np.isnan(result["precision_difference"]["ci95"]).any()
    assert result["precision_difference"]["ci95"] == pytest.approx([0.0, 0.0])
