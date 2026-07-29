"""
Feed cached segmentation data to the training loop.

This file is the bridge between the embedding files on disk (the "probe-feature
cache") and the trainer. It hands over one window's features + labels at a time,
and bundles several of them into a batch.

Vocabulary (used throughout the segmentation code):

- **sequence** - one CO3D video of a single object, filmed from many viewpoints.
- **window** - an ordered group of **6 frames** sampled from a sequence. It is the
  unit the cache is keyed by, and here it is one training example.
- **target frame** - the window's 6th (last) frame; the only frame we segment.
- **token / patch** - a 16x16 block of the target frame. Each token carries one
  feature vector and gets one 0/1 label, so a frame is a grid of tokens
  (e.g. 24x32 = 768 tokens), not individual pixels.

Why the custom batching in :func:`collate_windows`: the token-grid size differs
from window to window (the crop depends on the frame's aspect ratio), so windows
cannot be stacked into a neat rectangle. Instead we pour every window's tokens
into one flat ``[sum_N, D]`` pile and keep a ``counts`` list, so the tokens can
later be split back into their per-window groups.

Splits (train / val / test) are read straight from the cache rows, which carry
the manifest's **sequence-level** assignment - this module never invents splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.backbones.probe_cache import load_embedding_sample, load_probe_index


class ProbeCacheDataset(Dataset):
    """
    One split's windows from a probe cache; each item is one window.

    Construction only reads the lightweight cache index (the table of contents)
    and filters it by "split" / "categories" - the heavy tensors are loaded
    lazily in '__getitem__', one window at a time.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        split: str | None = None,
        categories: list[str] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        rows = load_probe_index(self.cache_dir)
        if split is not None:
            rows = [row for row in rows if row["split"] == split]
        if categories is not None:
            allowed = set(categories)
            rows = [row for row in rows if row["category"] in allowed]
        if not rows:
            raise ValueError(
                f"No probe-cache windows for split={split!r} in {self.cache_dir}"
            )
        self.rows = rows
        self._shard_cache: dict[str, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def sequence_ids(self) -> set[str]:
        return {row["sequence_id"] for row in self.rows}

    def categories(self) -> list[str]:
        return sorted({row["category"] for row in self.rows})

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load window ``index``: its target-frame token features and labels.

        ``spatial`` is ``[N, D]`` (N tokens, each a D-vector). ``labels`` is the
        target-frame mask pooled to the token grid, flattened to ``[N]`` so label
        ``j`` lines up with token ``j``. The rest is per-window bookkeeping.
        """
        row = self.rows[index]
        sample = load_embedding_sample(self.cache_dir, row)
        spatial = sample.target_spatial()  # [N, D]; same call works for any backbone layout
        return {
            "spatial": spatial,
            "labels": sample.seg_labels.reshape(-1),  # [N], one 0/1 per token
            "count": spatial.shape[0],  # N, this window's token count
            "token_grid": sample.token_grid,  # (grid_h, grid_w)
            "window_id": sample.window_id,
            "category": sample.category,
            "category_index": sample.category_index,
        }


def collate_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Fuse a list of windows into one batch.

    Because windows have different token counts, we concatenate all their tokens
    into one flat pile (``spatial`` ``[sum_N, D]``, ``labels`` ``[sum_N]``) rather
    than stacking a rectangle. ``counts`` records how many tokens each window
    contributed, so the pile can be split back into its per-window groups later.
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


def assert_sequence_disjoint(*datasets: ProbeCacheDataset) -> None:
    """Fail loudly if any two datasets share a CO3D sequence (leakage guard)."""
    seen: dict[str, int] = {}
    for position, dataset in enumerate(datasets):
        for sequence_id in dataset.sequence_ids():
            if sequence_id in seen and seen[sequence_id] != position:
                raise ValueError(
                    f"Sequence {sequence_id!r} appears in more than one split; "
                    "train/val/test must be sequence-disjoint"
                )
            seen[sequence_id] = position
