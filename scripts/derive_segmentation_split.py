"""Derive segmentation's shared-split override.

Reuses classification's ``fullunion-resplit`` sequence-to-split membership as
segmentation's train/val/test boundary, then patches in a per-category floor
(default: at least N sequences/category in test, then val) so no category's
score rests on too few sequences. See ``src/data/split_overrides.py`` for the
promotion logic and ``src/segmentation/train_segmentation.py``'s
``resolve_probe_cache_dataset`` for how the result is consumed.

Reads only cache ``index.parquet`` files (sequence_id/category/split columns) -
no tensors, no GPU, no CO3D download. Run once real caches are reachable:

    python -m scripts.derive_segmentation_split \\
      --classification-cache ${CUT3R_TRAINED_RESPLIT_CACHE} \\
      --segmentation-cache ${CUT3R_CACHE_ROOT}/probe/cut3r-trained \\
      --segmentation-cache ${CUT3R_CACHE_ROOT}/probe/cut3r-trained-leftover \\
      --segmentation-cache ${CUT3R_CACHE_ROOT}/probe/cut3r-trained-cap100-new-train \\
      --test-floor 5 --val-floor 5 --seed 20260731 \\
      --output configs/segmentation_split_override.json

One backbone's caches are enough for ``--segmentation-cache``: sequence/category
membership does not depend on which backbone extracted the embeddings, only
``--classification-cache`` and ``--segmentation-cache`` need not be the same
backbone either, for the same reason.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.backbones.probe_cache import load_probe_index
from src.common.io import atomic_write_json
from src.data.split_overrides import apply_category_floor

# Matches the val/test cap ADR 0002/0004 already used for segmentation (5/category);
# kept as the default here so every category's reported IoU rests on at least this
# many sequences. Override with --test-floor/--val-floor if you want a different value.
DEFAULT_CATEGORY_FLOOR = 5


def _sequence_maps(cache_dirs: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    """Read (sequence_id -> split, sequence_id -> category) across ``cache_dirs``.

    Disagreement across the caches in this list (same sequence, different
    recorded split or category) is a data-integrity problem, not something to
    silently resolve - it fails loudly with the offending sequence IDs.
    """
    split_of: dict[str, str] = {}
    category_of: dict[str, str] = {}
    conflicts: list[str] = []
    for cache_dir in cache_dirs:
        for row in load_probe_index(cache_dir):
            sequence_id, category, split = row["sequence_id"], row["category"], row["split"]
            if sequence_id in category_of and category_of[sequence_id] != category:
                conflicts.append(
                    f"{sequence_id}: category {category_of[sequence_id]!r} vs {category!r} in {cache_dir}"
                )
            if sequence_id in split_of and split_of[sequence_id] != split:
                conflicts.append(
                    f"{sequence_id}: split {split_of[sequence_id]!r} vs {split!r} in {cache_dir}"
                )
            category_of[sequence_id] = category
            split_of[sequence_id] = split
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} sequence(s) disagree across {cache_dirs}: {conflicts[:5]}"
        )
    return split_of, category_of


def derive(
    *,
    classification_caches: list[Path],
    segmentation_caches: list[Path],
    floors: dict[str, int],
    seed: int,
    priority: list[str],
) -> dict[str, Any]:
    class_split, class_category = _sequence_maps(classification_caches)
    seg_split_native, seg_category = _sequence_maps(segmentation_caches)

    shared = set(class_category) & set(seg_category)
    disagreeing = sorted(s for s in shared if class_category[s] != seg_category[s])
    if disagreeing:
        raise ValueError(
            f"{len(disagreeing)} sequence(s) have a different category between the "
            f"classification and segmentation caches, e.g. {disagreeing[:5]}"
        )

    # Classification's membership is the shared boundary; a segmentation sequence
    # classification's resplit never saw (e.g. not part of that union) falls back
    # to its own native manifest split rather than being dropped.
    base_split = {**seg_split_native, **class_split}
    category_of = {**seg_category, **class_category}
    gap_filled = sorted(set(seg_split_native) - set(class_split))

    override, report = apply_category_floor(
        base_split, category_of, floors=floors, seed=seed, priority=priority
    )

    return {
        "schema_version": "segmentation-split-override-v1",
        "seed": seed,
        "floors": floors,
        "priority": priority,
        "sources": {
            "classification_caches": [str(path) for path in classification_caches],
            "segmentation_caches": [str(path) for path in segmentation_caches],
        },
        "gap_filled_sequences": gap_filled,
        "split_override": override,
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--classification-cache", action="append", required=True, dest="classification_caches",
        type=Path, help="A classification fullunion-resplit cache dir; repeatable",
    )
    parser.add_argument(
        "--segmentation-cache", action="append", required=True, dest="segmentation_caches",
        type=Path, help="A segmentation probe-feature cache dir (original/leftover/cap100-new-train); repeatable",
    )
    parser.add_argument(
        "--test-floor", type=int, default=DEFAULT_CATEGORY_FLOOR,
        help=f"Minimum sequences/category in test (default: {DEFAULT_CATEGORY_FLOOR})",
    )
    parser.add_argument(
        "--val-floor", type=int, default=DEFAULT_CATEGORY_FLOOR,
        help=f"Minimum sequences/category in val (default: {DEFAULT_CATEGORY_FLOOR})",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--priority", default="test,val",
        help="Comma-separated order to fill floors in (earlier = higher priority)",
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    floors: dict[str, int] = {}
    if arguments.test_floor is not None:
        floors["test"] = arguments.test_floor
    if arguments.val_floor is not None:
        floors["val"] = arguments.val_floor
    if not floors:
        raise ValueError("At least one of --test-floor / --val-floor is required")

    payload = derive(
        classification_caches=arguments.classification_caches,
        segmentation_caches=arguments.segmentation_caches,
        floors=floors,
        seed=arguments.seed,
        priority=arguments.priority.split(","),
    )
    atomic_write_json(arguments.output, payload)

    touched = payload["report"]["categories"]
    shortfalls = {
        category: entry["shortfall"] for category, entry in touched.items() if entry["shortfall"]
    }
    print(
        f"wrote {arguments.output}: {len(payload['split_override'])} sequences, "
        f"{len(touched)} category(ies) needed promotion"
    )
    if payload["gap_filled_sequences"]:
        print(
            f"note: {len(payload['gap_filled_sequences'])} segmentation sequence(s) were not in "
            "the classification cache(s); kept their native split"
        )
    if shortfalls:
        print(f"warning: floor NOT met even after pooling for: {shortfalls}")


if __name__ == "__main__":
    main()
