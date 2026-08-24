"""
Feed cached segmentation data to the training loop.

This file is the bridge between the embedding files on disk (the "probe-feature
cache") and the trainer. It hands over one window's target-frame features + labels
at a time, and bundles several of them into a batch.
Contains dataset construction and batching logic.

Vocabulary (used throughout the segmentation code):

- **sequence** - one CO3D video of a single object, filmed from many viewpoints.
- **window** - an ordered group of **6 frames** sampled from a sequence. It is the
  unit the cache is keyed by, and here it is one training example.
- **target frame** - the window's 6th (last) frame; the only frame we segment.
- **token / patch** - a fixed square block of the target frame (16x16 for CUT3R,
  14x14 for DINOv2). Each token carries one feature vector and gets one 0/1 label,
  so a frame is a grid of tokens, not individual pixels. At CUT3R-512 a 4:3 frame
  gives 24x32 = 768 tokens and a 3:2 frame 21x32 = 672, so the grid varies with
  both the backbone and the frame's aspect ratio.

Why the custom batching in: func:`collate_windows`: the token-grid size differs
from window to window (the crop depends on the frame's aspect ratio), so windows
cannot be stacked into a neat rectangle. Instead, we pour every window's tokens
into one flat ``[sum_N, D]`` pile and keep a ``counts`` vector, so the tokens can
later be split back into their per-window groups.

Splits (train / val / test) are read straight from the cache rows, which carry
the manifest's **sequence-level** assignment - this module never invents splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from src.backbones.probe_cache import load_probe_index, load_target_tokens


def _filter_rows(
    rows: list[dict[str, Any]], *, split: str | None, categories: list[str] | None
) -> list[dict[str, Any]]:
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    if categories is not None:
        allowed = set(categories)
        rows = [row for row in rows if row["category"] in allowed]
    return rows


class ProbeCacheDataset(Dataset):
    """
    One split's windows from a probe cache.
    Each item is one window's **target frame** - its grid tokens and its mask.

    Construction only reads the lightweight cache index (the table of contents)
    and filters it by "split" / "categories" - the tensors are loaded lazily in
    '__getitem__', one window at a time, and sliced down to the target frame
    inside the shard read.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        split: str | None = None,
        categories: list[str] | None = None,
    ) -> None:
        """
        Select the cache rows for one split (and optionally some categories).

        "split" alone is the normal call: "categories" defaults to "None", which keeps all 51.
        "split=None" likewise means every split.

        Raises ``ValueError`` when the filters leave no windows, so an empty
        dataset can never be trained or evaluated on silently.
        """
        # Reads index.parquet only - a few hundred KB of bookkeeping, no tensors.
        self.cache_dir = Path(cache_dir)
        rows = _filter_rows(load_probe_index(self.cache_dir), split=split, categories=categories)
        if not rows: # Dataset is empty
            raise ValueError(
                f"No probe-cache windows for split={split!r} categories={categories!r} "
                f"in {self.cache_dir}")
        self.rows = rows

    def __len__(self) -> int:
        """Number of windows (rows) in this split."""
        return len(self.rows)

    def sequence_ids(self) -> set[str]:
        """
        The CO3D sequences this split draws from, deduplicated.
        Used by: func:`assert_sequence_disjoint` for assuring no leakage.
        """
        return {row["sequence_id"] for row in self.rows}

    def categories(self) -> list[str]:
        """The CO3D categories present in this split, sorted, each listed once."""
        return sorted({row["category"] for row in self.rows})

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Load the "index" window's target-frame image-token features and mask labels.

        Returns one window as a dict, where "N" is its token count and "D" the
        backbone's feature width:
        """
        row = self.rows[index]
        spatial, labels = load_target_tokens(self.cache_dir, row)  # [N, D], [grid_h, grid_w]
        return {
            "spatial": spatial, # [N, D], one feature vector per token
            "labels": labels.reshape(-1),  # [N], one 0/1 per token
            "count": spatial.shape[0],  # N, this window's token count
            "token_grid": tuple(int(value) for value in row["token_grid"]), # (grid_h, grid_w), ints, with (grid_h * grid_w == N)
            "window_id": row["window_id"], # the cache's window key, for per-window reporting
            "category": row["category"], # CO3D category, for per-category reporting
            "category_index": int(row["category_index"]), # CO3D category index, for per-category reporting
        }


class CombinedProbeCacheDataset(Dataset):
    """
    One split's windows unioned across **several** probe caches.

    Built for expanded training (30->100 videos/category, plus the "leftover" windows):
    combine the *train* rows of the original, leftover, and cap100-new-train
    caches into one training set, while val/test stay on a single
    :class:`ProbeCacheDataset` (the original cache) so the score stays
    comparable to a single-cache baseline. See ``docs/data/part-a-cache-layout.md``.

    Each row remembers which cache directory it came from - shard filenames
    are only unique *within* one cache, so windows are read from their own
    cache directory rather than physically merging shards (the cache-layout
    doc's own warning against copying shard folders together).
    """

    def __init__(
        self,
        cache_dirs: Sequence[str | Path],
        *,
        split: str | None = None,
        categories: list[str] | None = None,
    ) -> None:
        if not cache_dirs:
            raise ValueError("cache_dirs must be non-empty")
        self.entries: list[tuple[Path, dict[str, Any]]] = []
        for cache_dir in cache_dirs:
            cache_dir = Path(cache_dir)
            rows = _filter_rows(load_probe_index(cache_dir), split=split, categories=categories)
            self.entries.extend((cache_dir, row) for row in rows)
        if not self.entries:
            raise ValueError(
                f"No probe-cache windows for split={split!r} categories={categories!r} "
                f"in any of {list(cache_dirs)}"
            )

    def __len__(self) -> int:
        return len(self.entries)

    def sequence_ids(self) -> set[str]:
        """The CO3D sequences this split draws from, deduplicated across caches."""
        return {row["sequence_id"] for _, row in self.entries}

    def categories(self) -> list[str]:
        """The CO3D categories present in this split, sorted, each listed once."""
        return sorted({row["category"] for _, row in self.entries})

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Same shape as :meth:`ProbeCacheDataset.__getitem__`, for one window."""
        cache_dir, row = self.entries[index]
        spatial, labels = load_target_tokens(cache_dir, row)  # [N, D], [grid_h, grid_w]
        return {
            "spatial": spatial,
            "labels": labels.reshape(-1),
            "count": spatial.shape[0],
            "token_grid": tuple(int(value) for value in row["token_grid"]),
            "window_id": row["window_id"],
            "category": row["category"],
            "category_index": int(row["category_index"]),
        }


### Dataset utils ###

def collate_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Fuse a list of windows into one batch.

    Because windows have different token counts, we concatenate all their tokens
    into one flat pile rather than stacking a rectangle.

    Returns the batch as a dict, for "B" windows holding "sum_N" tokens in total.
    """
    spatial = torch.cat([item["spatial"] for item in batch], dim=0)
    labels = torch.cat([item["labels"] for item in batch], dim=0)
    counts = torch.tensor([item["count"] for item in batch], dtype=torch.long)
    return {
        "spatial": spatial,  # [sum_N, D]
        "labels": labels,  # [sum_N]
        "counts": counts,  # [B]
        "token_grids": [item["token_grid"] for item in batch],
        "window_ids": [item["window_id"] for item in batch],
        "categories": [item["category"] for item in batch],
    }


def assert_sequence_disjoint(*datasets: ProbeCacheDataset | CombinedProbeCacheDataset) -> None:
    """Fail loudly if any two datasets share a CO3D sequence (leakage guard).

    Sequence overlap *within* one dataset argument is fine and expected -
    :class:`CombinedProbeCacheDataset` deliberately reuses the original cache's
    train sequences (the "leftover" partition draws extra windows from the
    *same* videos). Only overlap *across* different positional arguments -
    e.g. a training dataset and a val/test dataset - is leakage.
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
