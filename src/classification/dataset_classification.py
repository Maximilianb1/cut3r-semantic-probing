"""
Feed cached classification data to the training loop.

Same probe-feature cache the segmentation probe reads, different target: here a
window has **one** label - its CO3D category - instead of one label per token.

Two things simplify this than the segmentation dataset. The label comes from the
cache **index**, so no tensor is read to get it; and each window is pooled to a single
``[D]`` vector, so batches stack into a plain rectangle and need none of the
variable-token-count bookkeeping.

Splits (train / val / test) are read straight from the cache rows, which carry the
manifest's **sequence-level** assignment - this module never invents splits.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from src.backbones.probe_cache import (
    category_vocabulary,
    load_probe_index,
    load_target_features,
)

from .model_classification import FeatureSource, Pooling, pool_tokens


class ProbeCacheClassificationDataset(Dataset):
    """
    One split's windows from a probe cache, as (pooled features, category) pairs.

    Each item is one window's target frame, reduced to a single vector by
    "pooling" over "source" - the two together are the Stage 2 representation choice.
    """

    def __init__(self, cache_dir: str | Path, *, source: FeatureSource,
        pooling: Pooling = "mean", split: str | None = None,
        categories: list[str] | None = None,
        label_space: Sequence[str] | None = None) -> None:
        """
        Select the cache rows for one split (and optionally some categories).

        "label_space" names the categories the labels index into. ``None`` means the
        fixed 51-category vocabulary, so labels are vocabulary indices (and the cache's
        "category_index" is cross-checked for consistency). Passing the categories a cache
        actually holds instead yields contiguous labels over just those - see :func:`cache_categories`.
        """
        self.cache_dir = Path(cache_dir)
        self.label_space = tuple(label_space) if label_space is not None else category_names()
        self._label_of = {name: index for index, name in enumerate(self.label_space)}
        if source not in ("image_tokens", "state_tokens"):
            raise ValueError(f"Unknown source {source!r}; expected 'image_tokens' or 'state_tokens'")
        self.source = source
        self.pooling = pooling
        # The cache calls these "spatial" and "global"; translate once, here.
        self.kind = "spatial" if source == "image_tokens" else "global"
        rows = load_probe_index(self.cache_dir)

        # Filter by split
        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        # Filter by category
        if categories is not None:
            allowed = set(categories)
            rows = [row for row in rows if row["category"] in allowed]

        # Dataset is empty
        if not rows:
            raise ValueError(f"No probe-cache windows for split={split!r} categories={categories!r} in {self.cache_dir}")
        missing = sorted({row["category"] for row in rows} - set(self._label_of))
        if missing:
            raise ValueError(
                f"Categories {missing} are in the cache but not in the label space "
                f"{list(self.label_space)[:5]}...; a window would have no label"
            )
        if self.label_space == category_names():
            # Labels are looked up by name, so the cache's own category_index must agree
            # with this repo's vocabulary or every label would be silently off.
            disagreeing = {
                row["category"]: (int(row["category_index"]), self._label_of[row["category"]])
                for row in rows
                if int(row["category_index"]) != self._label_of[row["category"]]
            }
            if disagreeing:
                raise ValueError(
                    f"Cache {self.cache_dir} disagrees with the 51-category vocabulary: "
                    f"{disagreeing} (cache index, vocabulary index); the cache was built "
                    "against a different vocabulary"
                )
        self.rows = rows

    def __len__(self) -> int:
        """Number of windows in this split - i.e., how many examples an epoch sees."""
        return len(self.rows)

    def sequence_ids(self) -> set[str]:
        """The CO3D sequences this split draws from, deduplicated (leakage check)."""
        return {row["sequence_id"] for row in self.rows}

    def categories(self) -> list[str]:
        """The CO3D categories present in this split, sorted, each listed once."""
        return sorted({row["category"] for row in self.rows})

    def class_counts(self) -> dict[str, int]:
        """Windows per category, for reading imbalance off a run before trusting it."""
        counts: dict[str, int] = {}
        for row in self.rows:
            counts[row["category"]] = counts.get(row["category"], 0) + 1
        return dict(sorted(counts.items()))

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Load window "index": its pooled target-frame features and category.

        Returns one window as a dict:

        - "features" - [D] float32, the pooled representation.
        - "label" - int64 scalar, the category's index in "self.label_space" (the fixed
          51-category vocabulary unless the caller narrowed it).
        - "window_id" / "category" - identifiers for per-window metrics.

        No custom collate function is needed: once pooled every item has the same shape,
        so PyTorch's default_collate stacks a batch correctly - "features" into [B, D],
        "label" into [B], and the two strings into lists of B. It preserves order, so row
        i of every batch entry describes the same window, which is what the per-category
        metrics rely on.
        """
        row = self.rows[index]
        tokens = load_target_features(self.cache_dir, row, kind=self.kind)  # [tokens, D]
        return {
            "features": pool_tokens(tokens, self.pooling),  # [D]
            "label": torch.tensor(self._label_of[row["category"]], dtype=torch.long),
            "window_id": row["window_id"],
            "category": row["category"],
        }


def assert_sequence_disjoint(*datasets: ProbeCacheClassificationDataset) -> None:
    """
    Fail loudly if any two datasets share a CO3D sequence (leakage guard).
    """
    seen: dict[str, int] = {}
    for position, dataset in enumerate(datasets):
        for sequence_id in dataset.sequence_ids():
            if sequence_id in seen and seen[sequence_id] != position:
                raise ValueError(
                    f"Sequence {sequence_id!r} appears in more than one split; "
                    "train/val/test must be sequence-disjoint"
                )
            seen[sequence_id] = position


def cache_categories(cache_dir: str | Path) -> tuple[str, ...]:
    """The categories a cache actually holds, sorted - the "present" label space.

    Read from the whole index, not one split, so train, val and test agree on what
    label 7 means. Sorted by name, which for this vocabulary is also sorted by
    vocabulary index.
    """
    return tuple(sorted({row["category"] for row in load_probe_index(Path(cache_dir))}))


@lru_cache(maxsize=1)
def category_names() -> tuple[str, ...]:
    """The fixed 51-category vocabulary, index-aligned with "category_index".

    Cached because it is immutable and rebuilding it re-sorts 51 strings; a tuple rather
    than a list so a caller cannot mutate the shared, cached value.
    """
    return tuple(category_vocabulary())
