"""Derive segmentation's own independent train/val/test sequence split.

Assigns every sequence in the full window union (original + leftover +
cap100-new-train caches) to train/val/test with a stratified 80/10/10 split
per category, then patches in a per-category floor (default: at least N
sequences/category in test, then val) so no category's score rests on too
few sequences. See ``src/data/split_overrides.py`` for the promotion logic and
``src/segmentation/train_segmentation.py``'s ``resolve_probe_cache_dataset``
for how the result is consumed.

This does not reuse classification's sequence split - that split (an
independent 80/10/10 resplit classification ran over the same window union)
was never committed to this repo in any form (script or cache), and is not
recoverable from git history, this VM, or the team drive. This script derives
segmentation's own split by the same method (stratified 80/10/10 at sequence
level) instead, using the same tie-breaking convention as the rest of this
package's sequence selection (``deterministic_rank`` - SHA-256 ranked, not a
seeded PRNG - so the order sequences happen to be read in never matters).

Consequence: a sequence may land in segmentation's train split and
classification's test split (or vice versa) - the two tasks are no longer
guaranteed to share the exact same boundary. Within segmentation alone the
split is still clean (stratified, sequence-disjoint, floor-patched).

Reads only cache ``index.parquet`` files (sequence_id/category columns) - no
tensors, no GPU, no CO3D download. Run once real caches are reachable:

    python -m scripts.derive_segmentation_split \\
      --segmentation-cache ${CUT3R_CACHE_ROOT}/probe/cut3r-trained \\
      --segmentation-cache ${CUT3R_CACHE_ROOT}/probe/cut3r-trained-leftover \\
      --segmentation-cache ${CUT3R_CACHE_ROOT}/probe/cut3r-trained-cap100-new-train \\
      --test-floor 5 --val-floor 5 --seed 20260731 \\
      --output configs/segmentation_split_override.json

One backbone's caches are enough for ``--segmentation-cache``: sequence/category
membership does not depend on which backbone extracted the embeddings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.backbones.probe_cache import load_probe_index
from src.common.io import atomic_write_json
from src.data.split_overrides import apply_category_floor
from src.data.windows import deterministic_rank

# Matches the val/test cap ADR 0002/0004 already used for segmentation (5/category);
# kept as the default here so every category's reported IoU rests on at least this
# many sequences. Override with --test-floor/--val-floor if you want a different value.
DEFAULT_CATEGORY_FLOOR = 5
RESPLIT_PURPOSE = "independent-resplit"


def _category_map(cache_dirs: list[Path]) -> dict[str, str]:
    """Read (sequence_id -> category) across ``cache_dirs``.

    Disagreement across the caches in this list (same sequence, different
    recorded category) is a data-integrity problem, not something to silently
    resolve - it fails loudly with the offending sequence IDs.
    """
    category_of: dict[str, str] = {}
    conflicts: list[str] = []
    for cache_dir in cache_dirs:
        for row in load_probe_index(cache_dir):
            sequence_id, category = row["sequence_id"], row["category"]
            if sequence_id in category_of and category_of[sequence_id] != category:
                conflicts.append(
                    f"{sequence_id}: category {category_of[sequence_id]!r} vs {category!r} in {cache_dir}"
                )
            category_of[sequence_id] = category
    if conflicts:
        raise ValueError(
            f"{len(conflicts)} sequence(s) disagree across {cache_dirs}: {conflicts[:5]}"
        )
    return category_of


def _stratified_split(
    category_of: dict[str, str], *, train_frac: float, val_frac: float, seed: int
) -> dict[str, str]:
    """
    Independent 80/10/10-style split, stratified per category, at sequence level.

    Within each category, sequences are ranked by :func:`deterministic_rank`
    (SHA-256 of seed/category/purpose/sequence_id) rather than shuffled with a
    seeded PRNG - the result depends only on the seed, never on dict or
    filesystem iteration order, and never on which RNG algorithm a future
    caller's Python/NumPy version happens to default to.
    """
    sequences_by_category: dict[str, list[str]] = {}
    for sequence_id, category in category_of.items():
        sequences_by_category.setdefault(category, []).append(sequence_id)

    split_of: dict[str, str] = {}
    for category in sorted(sequences_by_category):
        ranked = sorted(
            sequences_by_category[category],
            key=lambda seq: deterministic_rank(seed, category, RESPLIT_PURPOSE, seq),
        )
        n = len(ranked)
        n_train = round(n * train_frac)
        n_val = min(round(n * val_frac), n - n_train)
        for sequence_id in ranked[:n_train]:
            split_of[sequence_id] = "train"
        for sequence_id in ranked[n_train:n_train + n_val]:
            split_of[sequence_id] = "val"
        for sequence_id in ranked[n_train + n_val:]:
            split_of[sequence_id] = "test"
    return split_of


def derive(
    *,
    segmentation_caches: list[Path],
    train_frac: float,
    val_frac: float,
    floors: dict[str, int],
    seed: int,
    priority: list[str],
) -> dict[str, Any]:
    category_of = _category_map(segmentation_caches)
    base_split = _stratified_split(
        category_of, train_frac=train_frac, val_frac=val_frac, seed=seed
    )
    override, report = apply_category_floor(
        base_split, category_of, floors=floors, seed=seed, priority=priority
    )

    return {
        "schema_version": "segmentation-split-override-v2",
        "method": "independent stratified 80/10/10 sequence-level split per category, "
        "then a per-category floor promotion - see this script's module docstring",
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "floors": floors,
        "priority": priority,
        "sources": {
            "segmentation_caches": [str(path) for path in segmentation_caches],
        },
        "split_override": override,
        "report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--segmentation-cache", action="append", required=True, dest="segmentation_caches",
        type=Path, help="A segmentation probe-feature cache dir (original/leftover/cap100-new-train); repeatable",
    )
    parser.add_argument("--train-frac", type=float, default=0.8, help="Train share per category (default: 0.8)")
    parser.add_argument("--val-frac", type=float, default=0.1, help="Val share per category (default: 0.1)")
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

    if not 0.0 < arguments.train_frac < 1.0:
        raise ValueError("--train-frac must be in (0, 1)")
    if not 0.0 < arguments.val_frac < 1.0:
        raise ValueError("--val-frac must be in (0, 1)")
    if arguments.train_frac + arguments.val_frac >= 1.0:
        raise ValueError("--train-frac + --val-frac must leave a positive test share")

    floors: dict[str, int] = {}
    if arguments.test_floor is not None:
        floors["test"] = arguments.test_floor
    if arguments.val_floor is not None:
        floors["val"] = arguments.val_floor
    if not floors:
        raise ValueError("At least one of --test-floor / --val-floor is required")

    payload = derive(
        segmentation_caches=arguments.segmentation_caches,
        train_frac=arguments.train_frac,
        val_frac=arguments.val_frac,
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
    if shortfalls:
        print(f"warning: floor NOT met even after pooling for: {shortfalls}")


if __name__ == "__main__":
    main()
