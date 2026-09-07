"""
Layer a per-category floor on top of an existing sequence split.

Segmentation derives its own train/val/test sequence membership
(``scripts/derive_segmentation_split.py``), then promotes just enough
additional sequences into a thin split - typically ``test`` - so no
category's score rests on too few sequences. Nothing here touches a
cache: callers apply the resulting mapping
at read time via ``split_override`` on the segmentation datasets
(``src/segmentation/dataset_segmentation.py``).

Promotion is deterministic (SHA-256 ranked, like ``choose_sequences`` in
``src/data/windows.py``) and prefers the least disruptive donor split first -
by default pulling from ``val`` before ``train``, so filling a thin test split
does not remove data from the pools used to update the probe's weights unless
there is no other choice.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from src.data.windows import deterministic_rank

KNOWN_SPLITS = ("train", "val", "test")
DEFAULT_PRIORITY = ("test", "val")
DEFAULT_PROMOTION_ORDER = ("val", "train")


def apply_category_floor(
    base_split: Mapping[str, str],
    category_of: Mapping[str, str],
    *,
    floors: Mapping[str, int],
    seed: int,
    priority: Sequence[str] = DEFAULT_PRIORITY,
    promotion_order: Sequence[str] = DEFAULT_PROMOTION_ORDER,
) -> tuple[dict[str, str], dict[str, Any]]:
    """
    Return ``(overridden_split, report)``.

    ``base_split`` and ``category_of`` must cover exactly the same sequence
    IDs (fail loud on any mismatch, rather than silently ignoring an unknown
    sequence). ``floors`` maps a split name to the minimum number of sequences
    every category must have in that split, e.g. ``{"test": 5, "val": 5}``.

    For each category, in ``priority`` order (default: fill ``test`` first,
    then ``val``), if a split is short of its floor, sequences are pulled from
    ``promotion_order`` splits (default: ``val`` before ``train``) - ranked
    deterministically so the result depends only on ``seed``, never on dict
    or filesystem iteration order. A category too small to clear a floor even
    after using every available sequence is left short and recorded in
    ``report["categories"][category]["shortfall"]`` rather than raised on -
    this mirrors the project's existing stance that some CO3D categories are
    just small (ADR 0002), not a bug to crash over.

    Every sequence keeps its original split unless a category's floor
    required moving it; the vast majority of sequences pass through
    unchanged, which is the point - this is a targeted patch on top of an
    existing split, not a fresh one.
    """
    if set(base_split) != set(category_of):
        symmetric_difference = sorted(set(base_split) ^ set(category_of))
        raise ValueError(
            "base_split and category_of must cover exactly the same sequences; "
            f"{len(symmetric_difference)} disagree, e.g. {symmetric_difference[:5]}"
        )
    unknown_floor_splits = sorted(set(floors) - set(KNOWN_SPLITS))
    if unknown_floor_splits:
        raise ValueError(f"Unknown floor split(s): {unknown_floor_splits}")
    for split, count in floors.items():
        if count < 0:
            raise ValueError(f"Floor for {split!r} must be >= 0, got {count}")

    current = dict(base_split)
    sequences_by_category: dict[str, list[str]] = {}
    for sequence_id, category in category_of.items():
        sequences_by_category.setdefault(category, []).append(sequence_id)
    for sequences in sequences_by_category.values():
        sequences.sort()

    categories_report: dict[str, Any] = {}
    for category in sorted(sequences_by_category):
        sequences = sequences_by_category[category]
        promotions: dict[str, list[str]] = {}
        shortfall: dict[str, int] = {}
        for target_split in priority:
            floor = floors.get(target_split)
            if floor is None:
                continue
            held = [seq for seq in sequences if current[seq] == target_split]
            deficit = floor - len(held)
            if deficit <= 0:
                continue
            promoted: list[str] = []
            for donor_split in promotion_order:
                if deficit <= 0:
                    break
                if donor_split == target_split:
                    continue
                donors = sorted(
                    (seq for seq in sequences if current[seq] == donor_split),
                    key=lambda seq: deterministic_rank(
                        seed, category, f"promote-{target_split}", seq
                    ),
                )
                take = donors[:deficit]
                for seq in take:
                    current[seq] = target_split
                promoted.extend(take)
                deficit -= len(take)
            if promoted:
                promotions[target_split] = promoted
            if deficit > 0:
                shortfall[target_split] = deficit
        if promotions or shortfall:
            categories_report[category] = {
                "promotions": promotions,
                "shortfall": shortfall,
            }

    report = {
        "seed": seed,
        "floors": dict(floors),
        "priority": list(priority),
        "promotion_order": list(promotion_order),
        "categories": categories_report,
    }
    return current, report
