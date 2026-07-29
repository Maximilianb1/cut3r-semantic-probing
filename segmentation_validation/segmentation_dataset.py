"""Dataset and collation over the probe-feature cache for segmentation.

Token grids vary per window (CUT3R's crop depends on aspect ratio), so windows
cannot be stacked into a rectangular batch. Instead each window contributes its
tokens, and :func:`collate_windows` concatenates tokens across the batch into
``[sum_N, D]`` with a ``counts`` vector, so:

- the per-token loss sees one flat ``[sum_N, D]`` tensor, and
- image-level metrics regroup tokens per window via ``counts``.

Splits come straight from the cache rows, which carry the manifest's
**sequence-level** train/val/test assignment; this module never re-splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.backbones.probe_cache import load_embedding_sample, load_probe_index


class ProbeCacheDataset(Dataset):
    """Windows from one probe cache (optionally filtered to one split)."""

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
        row = self.rows[index]
        sample = load_embedding_sample(self.cache_dir, row)
        spatial = sample.target_spatial()  # [N, D], layout-agnostic
        return {
            "spatial": spatial,
            "labels": sample.seg_labels.reshape(-1),  # [N]
            "count": spatial.shape[0],
            "token_grid": sample.token_grid,
            "window_id": sample.window_id,
            "category": sample.category,
            "category_index": sample.category_index,
        }


def collate_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate window tokens into flat tensors, keeping per-window grouping."""
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
