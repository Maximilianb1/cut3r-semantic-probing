"""Tests for the segmentation probe-cache datasets.

Covers :class:`CombinedProbeCacheDataset` (expanded training: unioning several
caches' train rows - original + leftover + cap100-new-train) and
``train_segmentation.build_datasets``, the config-driven switch between a
single cache (the original baseline, unchanged) and a combined one (expanded
training).
"""

from __future__ import annotations

import pytest
import torch

from src.backbones.probe_cache import TARGET_ONLY, EmbeddingCacheWriter, EmbeddingSample, category_index_map
from src.segmentation.dataset_segmentation import (
    CombinedProbeCacheDataset,
    ProbeCacheDataset,
    assert_sequence_disjoint,
)
from src.segmentation.train_segmentation import build_datasets

_DIM = 8
_GRID = (2, 2)


def _write_cache(cache_dir, rows: list[dict], *, contract: dict | None = None) -> None:
    """Write one window per ``rows`` entry: {window_id, category, sequence_id, split}."""
    index_of = category_index_map()
    contract = {"layout": TARGET_ONLY, **(contract or {})}
    with EmbeddingCacheWriter(cache_dir, contract=contract, windows_per_shard=64) as writer:
        for row in rows:
            writer.add(
                EmbeddingSample(
                    window_id=row["window_id"], category=row["category"],
                    category_index=index_of[row["category"]], sequence_id=row["sequence_id"],
                    split=row["split"], layout=TARGET_ONLY, token_grid=_GRID,
                    frame_ids=[f"{row['window_id']}_{j}" for j in range(6)],
                    seg_labels=torch.zeros(*_GRID),
                    spatial_tokens=torch.randn(_GRID[0] * _GRID[1], _DIM),
                    global_tokens=torch.randn(1, _DIM),
                ),
                source_sha256={},
            )


@pytest.fixture()
def original_cache(tmp_path):
    directory = tmp_path / "original"
    _write_cache(directory, [
        {"window_id": "o0", "category": "apple", "sequence_id": "seqA", "split": "train"},
        {"window_id": "o1", "category": "ball", "sequence_id": "seqB", "split": "train"},
        {"window_id": "o2", "category": "apple", "sequence_id": "seqC", "split": "val"},
    ])
    return directory


@pytest.fixture()
def leftover_cache(tmp_path):
    # Same sequences as `original_cache`'s train rows - extra windows from
    # previously-unused frames of the same videos, not new sequences.
    directory = tmp_path / "leftover"
    _write_cache(directory, [
        {"window_id": "l0", "category": "apple", "sequence_id": "seqA", "split": "train"},
    ])
    return directory


@pytest.fixture()
def cap100_cache(tmp_path):
    # New, disjoint sequences (train-only, per the cache-layout doc).
    directory = tmp_path / "cap100"
    _write_cache(directory, [
        {"window_id": "c0", "category": "ball", "sequence_id": "seqD", "split": "train"},
        {"window_id": "c1", "category": "apple", "sequence_id": "seqE", "split": "train"},
    ])
    return directory


def test_combined_dataset_unions_train_rows_across_caches(original_cache, leftover_cache, cap100_cache) -> None:
    combined = CombinedProbeCacheDataset(
        [original_cache, leftover_cache, cap100_cache], split="train"
    )
    assert len(combined) == 5  # o0, o1, l0, c0, c1 - val row o2 excluded
    assert combined.sequence_ids() == {"seqA", "seqB", "seqD", "seqE"}
    assert combined.categories() == ["apple", "ball"]
    item = combined[0]
    assert item["spatial"].shape == (_GRID[0] * _GRID[1], _DIM)
    assert item["labels"].shape == (_GRID[0] * _GRID[1],)


def test_combined_dataset_requires_at_least_one_cache_dir() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CombinedProbeCacheDataset([], split="train")


def test_combined_dataset_empty_selection_raises(original_cache) -> None:
    with pytest.raises(ValueError, match="No probe-cache windows"):
        CombinedProbeCacheDataset([original_cache], split="test")


def test_sequence_disjoint_ignores_overlap_within_one_combined_dataset(
    original_cache, leftover_cache
) -> None:
    """seqA appears in both original-train and leftover-train - not leakage."""
    combined_train = CombinedProbeCacheDataset([original_cache, leftover_cache], split="train")
    val = ProbeCacheDataset(original_cache, split="val")
    assert_sequence_disjoint(combined_train, val)  # must not raise


def test_sequence_disjoint_still_catches_cross_split_leakage(tmp_path, original_cache) -> None:
    leaking = tmp_path / "leaking"
    _write_cache(leaking, [
        {"window_id": "x0", "category": "apple", "sequence_id": "seqC", "split": "train"},
    ])  # seqC is original_cache's val sequence
    combined_train = CombinedProbeCacheDataset([original_cache, leaking], split="train")
    val = ProbeCacheDataset(original_cache, split="val")
    with pytest.raises(ValueError, match="more than one split"):
        assert_sequence_disjoint(combined_train, val)


def _config(cache, *, train_dirs=None):
    config = {"probe_cache": {"dir": str(cache)}, "splits": {"train": "train", "val": "val"}}
    if train_dirs is not None:
        config["probe_cache"]["train_dirs"] = [str(d) for d in train_dirs]
    return config


def test_build_datasets_without_train_dirs_matches_the_original_shape(original_cache) -> None:
    train_set, val_set, record = build_datasets(_config(original_cache))
    assert isinstance(train_set, ProbeCacheDataset)
    assert len(train_set) == 2 and len(val_set) == 1
    assert set(record.keys()) == {"dir", "metadata"}
    assert record["dir"] == str(original_cache)


def test_build_datasets_with_train_dirs_combines_and_records_all_sources(
    original_cache, leftover_cache, cap100_cache
) -> None:
    train_set, val_set, record = build_datasets(
        _config(original_cache, train_dirs=[original_cache, leftover_cache, cap100_cache])
    )
    assert isinstance(train_set, CombinedProbeCacheDataset)
    assert len(train_set) == 5
    assert len(val_set) == 1
    assert set(record.keys()) == {"val_dir", "train_dirs"}
    assert record["val_dir"]["dir"] == str(original_cache)
    assert [entry["dir"] for entry in record["train_dirs"]] == [
        str(original_cache), str(leftover_cache), str(cap100_cache)
    ]


def test_build_datasets_rejects_expanded_train_that_leaks_into_val(
    tmp_path, original_cache
) -> None:
    leaking = tmp_path / "leaking2"
    _write_cache(leaking, [
        {"window_id": "y0", "category": "apple", "sequence_id": "seqC", "split": "train"},
    ])
    with pytest.raises(ValueError, match="more than one split"):
        build_datasets(_config(original_cache, train_dirs=[original_cache, leaking]))


### split_override ###


def test_dataset_split_override_relabels_without_mutating_row_split(original_cache) -> None:
    # seqA's windows are "train" in the cache; relabel seqA to "test" for reading.
    override = {"seqA": "test"}
    promoted = ProbeCacheDataset(original_cache, split="test", split_override=override)
    assert {row["window_id"] for row in promoted.rows} == {"o0"}
    assert promoted.sequence_ids() == {"seqA"}
    # The train split, filtered under the same override, must no longer see seqA.
    remaining_train = ProbeCacheDataset(original_cache, split="train", split_override=override)
    assert remaining_train.sequence_ids() == {"seqB"}
    # A dataset built without the override still sees the cache's original split.
    unpatched = ProbeCacheDataset(original_cache, split="train")
    assert unpatched.sequence_ids() == {"seqA", "seqB"}


def test_combined_dataset_split_override_applies_across_every_cache_dir(
    original_cache, cap100_cache
) -> None:
    # seqD lives only in cap100_cache, recorded there as "train"; promote it to "val".
    override = {"seqD": "val"}
    combined = CombinedProbeCacheDataset(
        [original_cache, cap100_cache], split="val", split_override=override
    )
    assert combined.sequence_ids() == {"seqC", "seqD"}  # seqC was already val


def _config_with_cache_dirs(cache_dirs, *, split_override_path=None):
    # No "dir" needed in this mode - every split reads from the pooled cache_dirs.
    config = {
        "probe_cache": {"cache_dirs": [str(d) for d in cache_dirs]},
        "splits": {"train": "train", "val": "val"},
    }
    if split_override_path is not None:
        config["probe_cache"]["split_override_path"] = str(split_override_path)
    return config


def test_build_datasets_cache_dirs_mode_pools_every_split_across_every_dir(
    original_cache, leftover_cache, cap100_cache
) -> None:
    train_set, val_set, record = build_datasets(
        _config_with_cache_dirs([original_cache, leftover_cache, cap100_cache])
    )
    assert isinstance(train_set, CombinedProbeCacheDataset)
    assert isinstance(val_set, CombinedProbeCacheDataset)
    assert len(train_set) == 5  # o0, o1, l0, c0, c1
    assert len(val_set) == 1  # o2
    assert record["cache_dirs"] == [str(original_cache), str(leftover_cache), str(cap100_cache)]
    assert record["split_override_path"] is None


def test_build_datasets_cache_dirs_mode_honors_split_override_path(
    tmp_path, original_cache, cap100_cache
) -> None:
    import json

    # Promote seqD (native split "train", lives only in cap100_cache) into "val".
    override_path = tmp_path / "override.json"
    override_path.write_text(json.dumps({"split_override": {"seqD": "val"}}), encoding="utf-8")

    train_set, val_set, record = build_datasets(
        _config_with_cache_dirs(
            [original_cache, cap100_cache], split_override_path=override_path
        )
    )
    assert val_set.sequence_ids() == {"seqC", "seqD"}
    assert "seqD" not in train_set.sequence_ids()
    assert record["split_override_path"] == str(override_path)


def test_build_datasets_train_dirs_and_cache_dirs_are_mutually_exclusive(original_cache) -> None:
    config = _config(original_cache, train_dirs=[original_cache])
    config["probe_cache"]["cache_dirs"] = [str(original_cache)]
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_datasets(config)


def test_split_override_path_without_cache_dirs_raises(tmp_path, original_cache) -> None:
    import json

    from src.segmentation.train_segmentation import resolve_probe_cache_dataset

    override_path = tmp_path / "override.json"
    override_path.write_text(json.dumps({"split_override": {}}), encoding="utf-8")
    config = _config(original_cache)
    config["probe_cache"]["split_override_path"] = str(override_path)
    with pytest.raises(ValueError, match="requires probe_cache.cache_dirs"):
        resolve_probe_cache_dataset(config, split="train", categories=None)
