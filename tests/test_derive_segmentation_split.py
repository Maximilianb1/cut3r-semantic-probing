"""Tests for segmentation's independent stratified split derivation.

Covers ``scripts/derive_segmentation_split.py``: the stratified 80/10/10
sequence-level split (no classification cache involved - see the script's
module docstring for why) and its composition with the per-category floor
patch from ``src/data/split_overrides.py``.
"""

from __future__ import annotations

import pytest
import torch

from scripts.derive_segmentation_split import _category_map, _stratified_split, derive
from src.backbones.probe_cache import TARGET_ONLY, EmbeddingCacheWriter, EmbeddingSample, category_index_map

_DIM = 4
_GRID = (2, 2)


def _write_cache(cache_dir, rows: list[dict]) -> None:
    """One window per row: {window_id, category, sequence_id}."""
    index_of = category_index_map()
    with EmbeddingCacheWriter(cache_dir, contract={"layout": TARGET_ONLY}, windows_per_shard=64) as writer:
        for row in rows:
            writer.add(
                EmbeddingSample(
                    window_id=row["window_id"], category=row["category"],
                    category_index=index_of[row["category"]], sequence_id=row["sequence_id"],
                    split="train", layout=TARGET_ONLY, token_grid=_GRID,
                    frame_ids=[f"{row['window_id']}_{j}" for j in range(6)],
                    seg_labels=torch.zeros(*_GRID),
                    spatial_tokens=torch.randn(_GRID[0] * _GRID[1], _DIM),
                    global_tokens=torch.randn(1, _DIM),
                ),
                source_sha256={},
            )


### _stratified_split ###


def test_stratified_split_partitions_every_sequence_once() -> None:
    category_of = {f"s{i}": "apple" for i in range(20)}
    split_of = _stratified_split(category_of, train_frac=0.8, val_frac=0.1, seed=1)
    assert set(split_of) == set(category_of)
    counts = {"train": 0, "val": 0, "test": 0}
    for split in split_of.values():
        counts[split] += 1
    assert counts == {"train": 16, "val": 2, "test": 2}


def test_stratified_split_is_per_category_independent() -> None:
    # 20 apples, 10 balls: each category hits its own 80/10/10, not a pooled one.
    category_of = {f"a{i}": "apple" for i in range(20)}
    category_of.update({f"b{i}": "ball" for i in range(10)})
    split_of = _stratified_split(category_of, train_frac=0.8, val_frac=0.1, seed=1)
    apple_counts = {"train": 0, "val": 0, "test": 0}
    ball_counts = {"train": 0, "val": 0, "test": 0}
    for sequence_id, split in split_of.items():
        (apple_counts if sequence_id.startswith("a") else ball_counts)[split] += 1
    assert apple_counts == {"train": 16, "val": 2, "test": 2}
    assert ball_counts == {"train": 8, "val": 1, "test": 1}


def test_stratified_split_is_deterministic_and_seed_sensitive() -> None:
    category_of = {f"s{i}": "apple" for i in range(30)}
    first = _stratified_split(category_of, train_frac=0.8, val_frac=0.1, seed=42)
    second = _stratified_split(category_of, train_frac=0.8, val_frac=0.1, seed=42)
    third = _stratified_split(category_of, train_frac=0.8, val_frac=0.1, seed=43)
    assert first == second
    assert first != third


def test_stratified_split_ignores_dict_iteration_order() -> None:
    category_of = {f"s{i}": "apple" for i in range(30)}
    reordered = dict(reversed(category_of.items()))
    assert _stratified_split(category_of, train_frac=0.8, val_frac=0.1, seed=7) == (
        _stratified_split(reordered, train_frac=0.8, val_frac=0.1, seed=7)
    )


### _category_map ###


def test_category_map_unions_across_caches(tmp_path) -> None:
    original = tmp_path / "original"
    leftover = tmp_path / "leftover"
    _write_cache(original, [{"window_id": "o0", "category": "apple", "sequence_id": "seqA"}])
    _write_cache(leftover, [{"window_id": "l0", "category": "ball", "sequence_id": "seqB"}])
    category_of = _category_map([original, leftover])
    assert category_of == {"seqA": "apple", "seqB": "ball"}


def test_category_map_raises_on_cross_cache_category_conflict(tmp_path) -> None:
    original = tmp_path / "original"
    cap100 = tmp_path / "cap100"
    _write_cache(original, [{"window_id": "o0", "category": "apple", "sequence_id": "seqA"}])
    _write_cache(cap100, [{"window_id": "c0", "category": "ball", "sequence_id": "seqA"}])
    with pytest.raises(ValueError, match="disagree"):
        _category_map([original, cap100])


### derive ###


def test_derive_applies_floor_on_top_of_the_fresh_split(tmp_path) -> None:
    # "apple" has only 3 sequences - too few for the stratified split to ever
    # place one in test, so the floor promotion must do it.
    rows = [{"window_id": f"e{i}", "category": "apple", "sequence_id": f"seqE{i}"} for i in range(3)]
    cache = tmp_path / "cache"
    _write_cache(cache, rows)
    payload = derive(
        segmentation_caches=[cache], train_frac=0.8, val_frac=0.1,
        floors={"test": 1, "val": 1}, seed=20260731, priority=["test", "val"],
    )
    counts = {"train": 0, "val": 0, "test": 0}
    for split in payload["split_override"].values():
        counts[split] += 1
    assert counts["test"] >= 1 and counts["val"] >= 1
    assert payload["schema_version"] == "segmentation-split-override-v2"
    assert payload["sources"]["segmentation_caches"] == [str(cache)]
