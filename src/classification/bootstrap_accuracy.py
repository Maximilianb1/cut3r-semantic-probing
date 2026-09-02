"""Sequence-cluster bootstrap intervals for classification accuracy.

The independent experimental unit is a CO3D sequence, not an overlapping window.
Every resample therefore draws sequence clusters with replacement and carries all of
their windows together. Paired comparisons reuse the same cluster draw for both
models, which preserves their within-sequence error correlation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class SequenceAccuracyClusters:
    """Sufficient statistics for window- and sequence-level accuracy."""

    sequence_ids: tuple[str, ...]
    categories: tuple[str, ...]
    window_ids: tuple[tuple[str, ...], ...]
    window_correct: np.ndarray
    window_total: np.ndarray
    sequence_correct: np.ndarray

    @property
    def window_accuracy(self) -> float:
        return float(self.window_correct.sum() / self.window_total.sum())

    @property
    def sequence_accuracy(self) -> float:
        return float(self.sequence_correct.mean())


def accuracy_clusters(inference: dict[str, Any]) -> SequenceAccuracyClusters:
    """Convert one probability inference into one cluster record per sequence."""
    labels = tuple(str(value) for value in inference.get("label_space", ()))
    windows = inference.get("per_window")
    if not labels or not isinstance(windows, list) or not windows:
        raise ValueError("Inference requires a non-empty label_space and per_window list")
    label_set = set(labels)
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_windows: set[str] = set()
    for window in windows:
        window_id = str(window["window_id"])
        if window_id in seen_windows:
            raise ValueError(f"Duplicate window_id in inference: {window_id}")
        seen_windows.add(window_id)
        probabilities = window.get("class_probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != label_set:
            raise ValueError(f"Probability labels disagree for window {window_id}")
        values = np.asarray([float(probabilities[label]) for label in labels])
        if not np.isfinite(values).all() or not np.isclose(values.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Invalid probability vector for window {window_id}")
        grouped.setdefault(str(window["sequence_id"]), []).append(window)

    sequence_ids: list[str] = []
    categories: list[str] = []
    window_ids: list[tuple[str, ...]] = []
    window_correct: list[int] = []
    window_total: list[int] = []
    sequence_correct: list[int] = []
    for sequence_id, records in sorted(grouped.items()):
        record_categories = {str(record["category"]) for record in records}
        if len(record_categories) != 1:
            raise ValueError(
                f"Sequence {sequence_id!r} changes category: {sorted(record_categories)}"
            )
        category = next(iter(record_categories))
        probabilities = np.asarray([
            [float(record["class_probabilities"][label]) for label in labels]
            for record in records
        ])
        targets = np.asarray([labels.index(category)] * len(records))
        predictions = probabilities.argmax(axis=1)
        sequence_prediction = int(probabilities.mean(axis=0).argmax())
        sequence_ids.append(sequence_id)
        categories.append(category)
        window_ids.append(tuple(sorted(str(record["window_id"]) for record in records)))
        window_correct.append(int((predictions == targets).sum()))
        window_total.append(len(records))
        sequence_correct.append(int(labels[sequence_prediction] == category))

    return SequenceAccuracyClusters(
        sequence_ids=tuple(sequence_ids),
        categories=tuple(categories),
        window_ids=tuple(window_ids),
        window_correct=np.asarray(window_correct, dtype=np.int64),
        window_total=np.asarray(window_total, dtype=np.int64),
        sequence_correct=np.asarray(sequence_correct, dtype=np.int64),
    )


def assert_paired_clusters(
    left: SequenceAccuracyClusters, right: SequenceAccuracyClusters
) -> None:
    """Require the exact same sequence/category/window test observations."""
    for name in ("sequence_ids", "categories", "window_ids"):
        if getattr(left, name) != getattr(right, name):
            raise ValueError(f"Paired inferences disagree on {name}")
    if not np.array_equal(left.window_total, right.window_total):
        raise ValueError("Paired inferences disagree on windows per sequence")


def _ratio_distribution(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    samples: int,
    generator: np.random.Generator,
    batch_size: int = 1_000,
) -> np.ndarray:
    if samples < 1:
        raise ValueError("samples must be positive")
    count = len(numerator)
    if count < 2 or len(denominator) != count:
        raise ValueError("At least two aligned sequence clusters are required")
    result = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        indices = generator.integers(0, count, size=(stop - start, count))
        sampled_numerator = numerator[indices].sum(axis=1)
        sampled_denominator = denominator[indices].sum(axis=1)
        result[start:stop] = sampled_numerator / sampled_denominator
    return result


def _summary(distribution: np.ndarray, point: float) -> dict[str, Any]:
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    return {
        "estimate": float(point),
        "ci95": [float(lower), float(upper)],
        "bootstrap_standard_error": float(distribution.std(ddof=1)),
    }


def bootstrap_accuracy_ci(
    clusters: SequenceAccuracyClusters,
    *,
    samples: int = 20_000,
    seed: int = 20260825,
) -> dict[str, Any]:
    """Marginal percentile CIs from the classical sequence-cluster bootstrap."""
    generator = np.random.default_rng(seed)
    window_distribution = _ratio_distribution(
        clusters.window_correct,
        clusters.window_total,
        samples=samples,
        generator=generator,
    )
    sequence_distribution = _ratio_distribution(
        clusters.sequence_correct,
        np.ones_like(clusters.sequence_correct),
        samples=samples,
        generator=generator,
    )
    return {
        "method": "nonparametric percentile sequence-cluster bootstrap",
        "samples": samples,
        "seed": seed,
        "clusters": len(clusters.sequence_ids),
        "window_accuracy": _summary(window_distribution, clusters.window_accuracy),
        "sequence_accuracy": _summary(sequence_distribution, clusters.sequence_accuracy),
    }


def _difference_distribution(
    target_numerator: np.ndarray,
    reference_numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    samples: int,
    seed: int,
    paired: bool,
    batch_size: int = 1_000,
) -> np.ndarray:
    count = len(denominator)
    if count < 2:
        raise ValueError("At least two sequence clusters are required")
    generator = np.random.default_rng(seed)
    result = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        target_indices = generator.integers(0, count, size=(stop - start, count))
        reference_indices = (
            target_indices
            if paired
            else generator.integers(0, count, size=(stop - start, count))
        )
        target = target_numerator[target_indices].sum(axis=1) / denominator[
            target_indices
        ].sum(axis=1)
        reference = reference_numerator[reference_indices].sum(axis=1) / denominator[
            reference_indices
        ].sum(axis=1)
        result[start:stop] = target - reference
    return result


def bootstrap_accuracy_difference(
    target: SequenceAccuracyClusters,
    reference: SequenceAccuracyClusters,
    *,
    samples: int = 20_000,
    seed: int = 20260825,
    paired: bool = True,
) -> dict[str, Any]:
    """Bootstrap target-minus-reference accuracy on aligned sequence clusters.

    ``paired=True`` is the valid primary comparison for models evaluated on the same
    test sequences. ``paired=False`` independently resamples the two models and is
    retained as the requested classical/unpaired sensitivity analysis.
    """
    assert_paired_clusters(target, reference)
    window_distribution = _difference_distribution(
        target.window_correct,
        reference.window_correct,
        target.window_total,
        samples=samples,
        seed=seed,
        paired=paired,
    )
    ones = np.ones_like(target.sequence_correct)
    sequence_distribution = _difference_distribution(
        target.sequence_correct,
        reference.sequence_correct,
        ones,
        samples=samples,
        seed=seed + 1,
        paired=paired,
    )
    method = (
        "paired percentile sequence-cluster bootstrap"
        if paired
        else "unpaired percentile sequence-cluster bootstrap"
    )
    return {
        "method": method,
        "samples": samples,
        "seed": seed,
        "clusters": len(target.sequence_ids),
        "window_accuracy_difference": _summary(
            window_distribution, target.window_accuracy - reference.window_accuracy
        ),
        "sequence_accuracy_difference": _summary(
            sequence_distribution, target.sequence_accuracy - reference.sequence_accuracy
        ),
    }
