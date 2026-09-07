"""Threshold-free binary-classification metrics: AUC-ROC and AUC-PR (average
precision), computed straight from raw scores + labels.

Reimplemented on numpy instead of pulling in scikit-learn for two functions -
this project has no other use for it. Both match scikit-learn's definitions
exactly, including its tie-handling (ties are resolved by average rank for
ROC, and grouped by distinct score for PR so the result never depends on the
arbitrary order two equal-scoring tokens happen to appear in).

These exist because the probe uses one fixed decision threshold (logit > 0)
across all backbones by design - so IoU/accuracy conflate "is the information
in the embedding" with "is 0 the right cutoff for this backbone's raw score
scale." AUC-ROC/AUC-PR strip the threshold out entirely and answer only the
first question.
"""

from __future__ import annotations

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks over ``values``, tied values given their group's mean rank."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    n = len(values)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # +1 for 1-based ranks
        i = j + 1
    return ranks


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, via the Mann-Whitney U statistic.

    Equivalent to the probability that a random positive scores higher than a
    random negative (0.5 if the score carries no information, 1.0 for perfect
    separation). Returns NaN if ``labels`` is all one class - AUC is undefined
    without both classes present.
    """
    positive = labels.astype(bool)
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    sum_ranks_positive = ranks[positive].sum()
    return float((sum_ranks_positive - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision).

    Sorts descending by score, groups tied scores together (precision/recall
    are evaluated once per distinct score, using the cumulative counts through
    the end of that group), then sums ``(recall_k - recall_{k-1}) * precision_k``.
    Returns NaN if there are no positives - precision/recall are undefined
    without at least one.
    """
    positive = labels.astype(bool)
    n_pos = int(positive.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    scores_sorted = scores[order]
    positive_sorted = positive[order]

    tp_cumulative = np.cumsum(positive_sorted)
    fp_cumulative = np.cumsum(~positive_sorted)

    # Last index of each run of equal (tied) scores, so ties share one
    # precision/recall point computed from their combined counts.
    group_ends = np.flatnonzero(np.diff(scores_sorted) != 0)
    group_ends = np.concatenate((group_ends, [len(scores_sorted) - 1]))

    tp_at_group = tp_cumulative[group_ends]
    fp_at_group = fp_cumulative[group_ends]
    precision = tp_at_group / (tp_at_group + fp_at_group)
    recall = tp_at_group / n_pos

    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))
