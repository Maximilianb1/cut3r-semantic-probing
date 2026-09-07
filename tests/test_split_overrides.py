"""Tests for the per-category floor patch applied on top of an existing split.

Pure logic, no cache/data needed - see ``src/data/split_overrides.py``.
"""

from __future__ import annotations

import pytest

from src.data.split_overrides import apply_category_floor


def test_no_promotion_when_floors_already_met() -> None:
    base_split = {
        "b0": "train", "b1": "val", "b2": "test", "b3": "test", "b4": "test",
    }
    category_of = {sequence_id: "ball" for sequence_id in base_split}
    override, report = apply_category_floor(
        base_split, category_of, floors={"test": 2, "val": 1}, seed=1
    )
    assert override == base_split
    assert report["categories"] == {}


def test_promotion_prefers_val_donors_before_train() -> None:
    base_split = {"c0": "train", "c1": "val", "c2": "val", "c3": "test"}
    category_of = {sequence_id: "cup" for sequence_id in base_split}
    override, report = apply_category_floor(
        base_split, category_of, floors={"test": 2}, seed=7
    )
    test_sequences = {seq for seq, split in override.items() if split == "test"}
    assert "c3" in test_sequences  # the original test sequence always stays test
    assert len(test_sequences) == 2
    promoted = test_sequences - {"c3"}
    assert promoted <= {"c1", "c2"}  # promoted from val, never from train (c0 stays train)
    assert override["c0"] == "train"
    assert report["categories"]["cup"]["promotions"]["test"] == sorted(promoted)


def test_test_floor_is_filled_before_val_floor_and_val_backfills_from_train() -> None:
    # apple: 6 train (a0-a5), 2 val (a6,a7), 1 test (a8). floors: test=3, val=1.
    base_split = {f"a{i}": "train" for i in range(6)}
    base_split.update({"a6": "val", "a7": "val", "a8": "test"})
    category_of = {sequence_id: "apple" for sequence_id in base_split}
    override, report = apply_category_floor(
        base_split, category_of, floors={"test": 3, "val": 1},
        seed=3, priority=("test", "val"),
    )
    counts = {"train": 0, "val": 0, "test": 0}
    for split in override.values():
        counts[split] += 1
    assert counts == {"train": 5, "val": 1, "test": 3}
    # Filling test's floor (3) exhausts both original val sequences.
    assert override["a6"] == "test" and override["a7"] == "test"
    assert override["a8"] == "test"
    # val's floor (1) then had to be backfilled from train, since val had nothing left.
    backfilled = [seq for seq in ("a0", "a1", "a2", "a3", "a4", "a5") if override[seq] == "val"]
    assert len(backfilled) == 1


def test_shortfall_is_reported_not_raised_when_category_is_too_small() -> None:
    base_split = {"e0": "train"}
    category_of = {"e0": "egg"}
    override, report = apply_category_floor(
        base_split, category_of, floors={"test": 2}, seed=1
    )
    assert override["e0"] == "test"  # promotes everything it can
    assert report["categories"]["egg"]["shortfall"] == {"test": 1}


def test_result_is_deterministic_across_calls() -> None:
    base_split = {"c0": "train", "c1": "val", "c2": "val", "c3": "test"}
    category_of = {sequence_id: "cup" for sequence_id in base_split}
    first, _ = apply_category_floor(base_split, category_of, floors={"test": 2}, seed=99)
    second, _ = apply_category_floor(base_split, category_of, floors={"test": 2}, seed=99)
    assert first == second


def test_mismatched_sequence_universe_raises() -> None:
    with pytest.raises(ValueError, match="same sequences"):
        apply_category_floor(
            {"a": "train"}, {"a": "cat", "b": "cat"}, floors={"test": 1}, seed=1
        )


def test_unknown_floor_split_raises() -> None:
    with pytest.raises(ValueError, match="Unknown floor split"):
        apply_category_floor({"a": "train"}, {"a": "cat"}, floors={"bogus": 1}, seed=1)


def test_negative_floor_raises() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        apply_category_floor({"a": "train"}, {"a": "cat"}, floors={"test": -1}, seed=1)
