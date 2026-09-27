"""Sequence-cluster bootstrap intervals for segmentation foreground precision/recall.

Same experimental unit as ``bootstrap_iou.py`` (a complete CO3D sequence, not
an overlapping window) but a different statistic: precision and recall are
ratios of *pooled* token counts, not averages of per-window ratios - a window
with few foreground tokens gives an unstable per-window precision/recall, so
each resample sums tp/fp/fn across a sampled sequence's windows first and only
then takes the ratio. This mirrors ``_ratio_distribution`` in
``src/classification/bootstrap_accuracy.py``, which pools window_correct/
window_total per sequence for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .bootstrap_iou import derive_seed, summarize

__all__ = [
    "SequencePrecisionRecallClusters",
    "precision_recall_clusters",
    "assert_paired_clusters",
    "bootstrap_precision_recall_ci",
    "bootstrap_precision_recall_difference",
]


@dataclass(frozen=True)
class SequencePrecisionRecallClusters:
    """Per-sequence pooled confusion counts - sufficient statistics for
    sequence-cluster precision/recall."""

    sequence_ids: tuple[str, ...]
    categories: tuple[str, ...]
    window_ids: tuple[tuple[str, ...], ...]
    tp: np.ndarray
    fp: np.ndarray
    fn: np.ndarray

    @property
    def precision(self) -> float:
        tp, fp = float(self.tp.sum()), float(self.fp.sum())
        return tp / (tp + fp) if (tp + fp) else 0.0

    @property
    def recall(self) -> float:
        tp, fn = float(self.tp.sum()), float(self.fn.sum())
        return tp / (tp + fn) if (tp + fn) else 0.0


def precision_recall_clusters(inference: dict[str, Any]) -> SequencePrecisionRecallClusters:
    """Convert one inference-<split>.json's ``per_window_iou`` (which now also
    carries per-window ``tp``/``fp``/``fn``) into one pooled-count cluster
    record per sequence."""
    rows = inference.get("per_window_iou")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Inference requires a non-empty per_window_iou list")

    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_windows: set[str] = set()
    for row in rows:
        window_id = str(row["window_id"])
        if window_id in seen_windows:
            raise ValueError(f"Duplicate window_id in inference: {window_id}")
        seen_windows.add(window_id)
        grouped.setdefault(str(row["sequence_id"]), []).append(row)

    sequence_ids: list[str] = []
    categories: list[str] = []
    window_ids: list[tuple[str, ...]] = []
    tp: list[int] = []
    fp: list[int] = []
    fn: list[int] = []
    for sequence_id, records in sorted(grouped.items()):
        record_categories = {str(record["category"]) for record in records}
        if len(record_categories) != 1:
            raise ValueError(
                f"Sequence {sequence_id!r} changes category: {sorted(record_categories)}"
            )
        sequence_ids.append(sequence_id)
        categories.append(next(iter(record_categories)))
        window_ids.append(tuple(sorted(str(record["window_id"]) for record in records)))
        tp.append(sum(int(record["tp"]) for record in records))
        fp.append(sum(int(record["fp"]) for record in records))
        fn.append(sum(int(record["fn"]) for record in records))

    return SequencePrecisionRecallClusters(
        sequence_ids=tuple(sequence_ids),
        categories=tuple(categories),
        window_ids=tuple(window_ids),
        tp=np.asarray(tp, dtype=np.int64),
        fp=np.asarray(fp, dtype=np.int64),
        fn=np.asarray(fn, dtype=np.int64),
    )


def assert_paired_clusters(
    left: SequencePrecisionRecallClusters, right: SequencePrecisionRecallClusters
) -> None:
    """Require the exact same sequence/category/window test observations."""
    for name in ("sequence_ids", "categories", "window_ids"):
        if getattr(left, name) != getattr(right, name):
            raise ValueError(f"Paired inferences disagree on {name}")


def _safe_ratio(numerator_sum: np.ndarray, denominator_sum: np.ndarray) -> np.ndarray:
    """``numerator_sum / denominator_sum``, 0.0 where the denominator pools to
    zero -- some resampled sequence-cluster sets have no predicted positives
    at all (a valid outcome, not a bootstrap failure), matching
    ``SequencePrecisionRecallClusters.precision``/``.recall``'s own convention
    rather than letting that draw poison the distribution with NaN."""
    return np.divide(
        numerator_sum, denominator_sum,
        out=np.zeros_like(numerator_sum, dtype=np.float64), where=denominator_sum != 0,
    )


def _ratio_distribution(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    samples: int,
    generator: np.random.Generator,
    batch_size: int = 1_000,
) -> np.ndarray:
    count = len(numerator)
    if count < 2 or len(denominator) != count:
        raise ValueError("At least two aligned sequence clusters are required")
    result = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = generator.integers(0, count, size=(stop - start, count))
        result[start:stop] = _safe_ratio(
            numerator[indices].sum(axis=1), denominator[indices].sum(axis=1)
        )
    return result


def bootstrap_precision_recall_ci(
    clusters: SequencePrecisionRecallClusters, *, samples: int = 20_000, seed: int = 20260825,
) -> dict[str, Any]:
    """Percentile CIs on pooled precision/recall from the sequence-cluster bootstrap."""
    count = len(clusters.sequence_ids)
    if count < 2:
        raise ValueError("At least two sequence clusters are required")
    generator = np.random.default_rng(seed)
    precision_distribution = _ratio_distribution(
        clusters.tp, clusters.tp + clusters.fp, samples=samples, generator=generator
    )
    recall_distribution = _ratio_distribution(
        clusters.tp, clusters.tp + clusters.fn, samples=samples, generator=generator
    )
    return {
        "method": "sequence-cluster percentile bootstrap (pooled counts)",
        "samples": samples,
        "seed": seed,
        "clusters": count,
        "precision": summarize(precision_distribution, clusters.precision),
        "recall": summarize(recall_distribution, clusters.recall),
    }


def _difference_distribution(
    target_numerator: np.ndarray,
    target_denominator: np.ndarray,
    reference_numerator: np.ndarray,
    reference_denominator: np.ndarray,
    *,
    samples: int,
    seed: int,
    paired: bool,
    batch_size: int = 1_000,
) -> np.ndarray:
    count = len(target_denominator)
    if count < 2:
        raise ValueError("At least two sequence clusters are required")
    generator = np.random.default_rng(seed)
    result = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        target_indices = generator.integers(0, count, size=(stop - start, count))
        reference_indices = target_indices if paired else generator.integers(0, count, size=(stop - start, count))
        target = _safe_ratio(
            target_numerator[target_indices].sum(axis=1), target_denominator[target_indices].sum(axis=1)
        )
        reference = _safe_ratio(
            reference_numerator[reference_indices].sum(axis=1), reference_denominator[reference_indices].sum(axis=1)
        )
        result[start:stop] = target - reference
    return result


def bootstrap_precision_recall_difference(
    target: SequencePrecisionRecallClusters, reference: SequencePrecisionRecallClusters, *,
    samples: int = 20_000, seed: int = 20260825, paired: bool = True,
) -> dict[str, Any]:
    """Bootstrap target-minus-reference precision/recall on aligned sequence clusters.

    ``paired=True`` is the valid primary comparison (same resample draw for
    both, so shared within-sequence noise cancels). ``paired=False``
    independently resamples the two runs and is kept as a sensitivity check.
    """
    assert_paired_clusters(target, reference)
    precision_distribution = _difference_distribution(
        target.tp, target.tp + target.fp, reference.tp, reference.tp + reference.fp,
        samples=samples, seed=seed, paired=paired,
    )
    recall_distribution = _difference_distribution(
        target.tp, target.tp + target.fn, reference.tp, reference.tp + reference.fn,
        samples=samples, seed=derive_seed(seed, "recall"), paired=paired,
    )
    method = "paired" if paired else "unpaired"
    return {
        "method": f"{method} sequence-cluster percentile bootstrap (pooled counts)",
        "samples": samples,
        "seed": seed,
        "clusters": len(target.sequence_ids),
        "precision_difference": summarize(precision_distribution, target.precision - reference.precision),
        "recall_difference": summarize(recall_distribution, target.recall - reference.recall),
    }
