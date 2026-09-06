"""Tests for the threshold-free AUC-ROC / AUC-PR reimplementation.

Pinned against hand-computed values (see the derivations in the PR/commit
that added this) since there's no scikit-learn in this project to check
against directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.segmentation.curve_metrics import average_precision, roc_auc


def test_roc_auc_matches_hand_computed_value_with_no_ties() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    # Concordant pairs: (0.1,0.35) (0.1,0.8) (0.4,0.8) = 3 of 4 neg-pos pairs.
    assert roc_auc(scores, labels) == pytest.approx(0.75)


def test_roc_auc_is_one_for_perfect_separation_and_zero_when_reversed() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(scores, labels) == pytest.approx(1.0)
    assert roc_auc(scores, labels[::-1].copy()) == pytest.approx(0.0)


def test_roc_auc_is_one_half_when_all_scores_are_tied() -> None:
    # No discriminative information at all -> chance-level AUC.
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0])
    assert roc_auc(scores, labels) == pytest.approx(0.5)


def test_roc_auc_is_nan_when_only_one_class_is_present() -> None:
    assert np.isnan(roc_auc(np.array([0.1, 0.9]), np.array([1, 1])))
    assert np.isnan(roc_auc(np.array([0.1, 0.9]), np.array([0, 0])))


def test_average_precision_matches_hand_computed_value_with_no_ties() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    labels = np.array([0, 0, 1, 1])
    # Descending: 0.8(tp,P=1,R=.5) 0.4(fp,P=.5,R=.5) 0.35(tp,P=2/3,R=1) 0.1(fp,P=.5,R=1)
    # AP = .5*1 + 0*.5 + .5*(2/3) + 0 = 5/6
    assert average_precision(scores, labels) == pytest.approx(5 / 6)


def test_average_precision_is_one_for_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert average_precision(scores, labels) == pytest.approx(1.0)


def test_average_precision_does_not_depend_on_tie_order() -> None:
    # Same scores/labels, tied 0.5-score pair given in opposite label order:
    # a per-sample (ungrouped) implementation would give different answers
    # depending on this order; the grouped one must not.
    scores = np.array([0.5, 0.5, 0.9, 0.2])
    first_order = average_precision(scores, np.array([1, 0, 1, 0]))
    second_order = average_precision(scores, np.array([0, 1, 1, 0]))
    assert first_order == pytest.approx(second_order)


def test_average_precision_is_nan_when_there_are_no_positives() -> None:
    assert np.isnan(average_precision(np.array([0.1, 0.9]), np.array([0, 0])))
