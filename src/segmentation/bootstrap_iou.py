"""Sequence-cluster bootstrap intervals for segmentation macro-IoU.

Mirrors ``src/classification/bootstrap_accuracy.py``: the independent
experimental unit is a CO3D sequence (one physical object), not a window - a
sequence contributes up to 4 windows in this project's test split
(``docs/data/stage0-protocol.md``), and those windows share the same object
instance and scene, so their errors are correlated even though they don't
share raw frames. Every resample therefore draws sequence clusters with
replacement, carrying all of a sequence's windows with it. Paired comparisons
reuse the same cluster draw for both models, preserving their within-sequence
error correlation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np


def derive_seed(base_seed: int, key: str) -> int:
    """Deterministic per-``key`` seed derived from one shared ``base_seed``.

    Lets every backbone's CI and every pairwise comparison draw from its own
    resample stream (so a correlated-noise bug in one can't leak into another)
    while the whole report still reproduces exactly from one recorded seed.
    """
    digest = hashlib.sha256(f"{base_seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class SequenceIoUClusters:
    """Sufficient statistics for sequence-level macro-IoU."""

    sequence_ids: tuple[str, ...]
    window_ids: tuple[tuple[str, ...], ...]
    sequence_iou: np.ndarray  # mean foreground IoU of that sequence's windows

    @property
    def macro_iou(self) -> float:
        """Mean over sequences of that sequence's mean window IoU - the
        sequence-cluster analog of the project's window-level macro-IoU."""
        return float(self.sequence_iou.mean())


def iou_clusters(inference: dict[str, Any]) -> SequenceIoUClusters:
    """Convert one inference-<split>.json's ``per_window_iou`` into one
    IoU-cluster record per sequence."""
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
    window_ids: list[tuple[str, ...]] = []
    sequence_iou: list[float] = []
    for sequence_id, records in sorted(grouped.items()):
        sequence_ids.append(sequence_id)
        window_ids.append(tuple(sorted(str(record["window_id"]) for record in records)))
        sequence_iou.append(float(np.mean([float(record["foreground_iou"]) for record in records])))

    return SequenceIoUClusters(
        sequence_ids=tuple(sequence_ids),
        window_ids=tuple(window_ids),
        sequence_iou=np.asarray(sequence_iou, dtype=np.float64),
    )


def assert_paired_clusters(left: SequenceIoUClusters, right: SequenceIoUClusters) -> None:
    """Require the exact same sequence/window test observations."""
    for name in ("sequence_ids", "window_ids"):
        if getattr(left, name) != getattr(right, name):
            raise ValueError(f"Paired inferences disagree on {name}")


def _summary(distribution: np.ndarray, point: float) -> dict[str, Any]:
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    return {
        "estimate": float(point),
        "ci95": [float(lower), float(upper)],
        "bootstrap_standard_error": float(distribution.std(ddof=1)),
    }


def bootstrap_iou_ci(
    clusters: SequenceIoUClusters, *, samples: int = 20_000, seed: int = 20260825,
) -> dict[str, Any]:
    """Percentile CI on macro-IoU from the sequence-cluster bootstrap."""
    count = len(clusters.sequence_ids)
    if count < 2:
        raise ValueError("At least two sequence clusters are required")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, count, size=(samples, count))
    distribution = clusters.sequence_iou[indices].mean(axis=1)
    return {
        "method": "sequence-cluster percentile bootstrap",
        "samples": samples,
        "seed": seed,
        "clusters": count,
        "macro_iou": _summary(distribution, clusters.macro_iou),
    }


def bootstrap_iou_difference(
    target: SequenceIoUClusters, reference: SequenceIoUClusters, *,
    samples: int = 20_000, seed: int = 20260825, paired: bool = True,
) -> dict[str, Any]:
    """Bootstrap target-minus-reference macro-IoU on aligned sequence clusters.

    ``paired=True`` is the valid primary comparison for two runs evaluated on
    the same test sequences (same resample draw for both, so shared
    within-sequence noise cancels in the difference). ``paired=False``
    independently resamples the two runs and is kept as a sensitivity check.
    """
    assert_paired_clusters(target, reference)
    count = len(target.sequence_ids)
    if count < 2:
        raise ValueError("At least two sequence clusters are required")
    generator = np.random.default_rng(seed)
    target_indices = generator.integers(0, count, size=(samples, count))
    reference_indices = target_indices if paired else generator.integers(0, count, size=(samples, count))
    distribution = (
        target.sequence_iou[target_indices].mean(axis=1)
        - reference.sequence_iou[reference_indices].mean(axis=1)
    )
    method = "paired" if paired else "unpaired"
    return {
        "method": f"{method} sequence-cluster percentile bootstrap",
        "samples": samples,
        "seed": seed,
        "clusters": count,
        "macro_iou_difference": _summary(distribution, target.macro_iou - reference.macro_iou),
    }
