"""Short check: does a category's test-IoU just track how many training
windows it had, or does something else (object difficulty, clutter, size)
matter more? Correlates per-category training-window counts (from the real
combined train cache) against each backbone's per-category test IoU.

If representation (window count) drove performance, categories with more
training windows should score higher, consistently across backbones. If the
correlation is weak, the per-category differences seen elsewhere (e.g. the
CUT3R-trained vs DINOv2 comparison) are probably about the categories'
visual properties, not how much of them the head happened to train on.

Run example (on the VM, where the probe caches live):
    python -m src.segmentation.analysis.build_category_representation_check \
        --config src/segmentation/configs/cut3r_trained_expanded.yaml \
        --experiments-root src/segmentation/experiments \
        --backbones cut3r_trained cut3r_random dinov2 --run-suffix=-expanded-bestval --split test
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.backbones.probe_cache import load_probe_index

from ..train_segmentation import load_config
from .runs import BACKBONE_COLOR, DISPLAY_NAME, load_inference, resolve_run_dir

_MARKERS = {"cut3r_trained": "o", "cut3r_random": "^", "dinov2": "s"}


def train_window_counts(train_dirs: list[str]) -> dict[str, int]:
    """Per-category count of train-split windows across one or more caches."""
    counts: Counter[str] = Counter()
    for cache_dir in train_dirs:
        for row in load_probe_index(Path(cache_dir)):
            if row["split"] == "train":
                counts[row["category"]] += 1
    return dict(counts)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Ranks with ties assigned their average rank (matches scipy's rankdata(method="average"))."""
    sorter = np.argsort(x, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    x_sorted = x[sorter]
    is_new_group = np.r_[True, x_sorted[1:] != x_sorted[:-1]]
    group_id = np.cumsum(is_new_group) - 1
    group_sizes = np.bincount(group_id)
    group_start = np.r_[0, np.cumsum(group_sizes)[:-1]]
    avg_rank_per_group = group_start + (group_sizes + 1) / 2.0
    return avg_rank_per_group[group_id][inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation, computed by hand (tie-averaged ranks) to avoid a scipy dependency."""
    return pearson(_rankdata_average(x), _rankdata_average(y))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                         help="one backbone's *_expanded.yaml, read only for probe_cache.train_dirs/dir")
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=None,
                         help="default: <experiments-root>/../category-representation-check.png")
    args = parser.parse_args()

    config = load_config(args.config)
    train_dirs = config["probe_cache"].get("train_dirs") or [config["probe_cache"]["dir"]]
    counts = train_window_counts([str(d) for d in train_dirs])
    categories = sorted(counts)
    print(f"Train dirs: {train_dirs}")
    print(f"{len(categories)} categories, {sum(counts.values())} total train windows")
    print("\nMost- and least-represented categories:")
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    for cat, c in ranked[:3] + [("...", None)] + ranked[-3:]:
        print(f"  {cat:15s} {c if c is not None else ''}")

    per_backbone_iou = {}
    for b in args.backbones:
        exp_dir = resolve_run_dir(args.experiments_root, b, args.run_suffix)
        data = load_inference(exp_dir, args.split)
        per_backbone_iou[b] = data["metrics"]["per_category_iou"]

    x = np.array([counts[c] for c in categories], dtype=float)
    print(f"\nCorrelation between train-window count and {args.split} per-category IoU:")
    correlations = {}
    for b in args.backbones:
        y = np.array([per_backbone_iou[b][c] for c in categories], dtype=float)
        r_pearson, r_spearman = pearson(x, y), spearman(x, y)
        correlations[b] = (r_pearson, r_spearman)
        print(f"  {DISPLAY_NAME.get(b, b):15s} pearson={r_pearson:+.3f}  spearman={r_spearman:+.3f}")

    # Linear, not log: the real range is ~181-400 (under 2.5x), too narrow to
    # need log scale. A small horizontal jitter (fixed seed) separates
    # categories landing on the same count. A broken x-axis (not one
    # continuous linear axis) because one category (parkingmeter, ~181) sits
    # far from the rest (~370-400): a single continuous axis would waste most
    # of its width on the empty gap between them and squeeze the actual
    # 25-category cluster -- where the "no correlation" finding lives -- into
    # a thin sliver.
    jitter_rng = np.random.default_rng(20260729)
    x_jittered = x + jitter_rng.uniform(-2.5, 2.5, size=len(x))
    low_cutoff = 250.0  # splits the outlier(s) below this from the main cluster above it

    fig, (ax_low, ax_high) = plt.subplots(
        1, 2, figsize=(6.5, 4.2), sharey=True,
        gridspec_kw={"width_ratios": [1, 4], "wspace": 0.08},
    )
    for b in args.backbones:
        y = np.array([per_backbone_iou[b][c] for c in categories], dtype=float)
        label = DISPLAY_NAME.get(b, b)
        style = dict(color=BACKBONE_COLOR.get(b, "#888"), marker=_MARKERS.get(b, "o"),
                     s=28, alpha=0.75, edgecolor="white", linewidth=0.5)
        ax_low.scatter(x_jittered[x < low_cutoff], y[x < low_cutoff], **style)
        ax_high.scatter(x_jittered[x >= low_cutoff], y[x >= low_cutoff], label=label, **style)

    ax_low.set_xlim(x[x < low_cutoff].min() - 15, x[x < low_cutoff].max() + 15)
    ax_high.set_xlim(x[x >= low_cutoff].min() - 8, x[x >= low_cutoff].max() + 8)
    ax_low.set_ylim(0, 1.05)
    ax_low.set_ylabel(f"{args.split.capitalize()} foreground IoU")
    fig.supxlabel("Train windows in category (jittered; axis broken -- one category sits far from the rest)", fontsize=9)

    # Diagonal break marks, the standard way to show a deliberately-cut axis.
    ax_low.spines["right"].set_visible(False)
    ax_high.spines["left"].set_visible(False)
    ax_high.tick_params(left=False)
    d = 0.02
    kwargs = dict(transform=ax_low.transAxes, color="k", clip_on=False, linewidth=1)
    ax_low.plot((1 - d, 1 + d), (-d, d), **kwargs)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_high.transAxes)
    ax_high.plot((-d, d), (-d, d), **kwargs)
    ax_high.plot((-d, d), (1 - d, 1 + d), **kwargs)

    for ax in (ax_low, ax_high):
        ax.grid(linewidth=0.3, alpha=0.5)
        ax.spines["top"].set_visible(False)
    ax_high.spines["right"].set_visible(False)
    fig.suptitle("Per-category test IoU vs. training-data representation", fontsize=10, fontweight="bold")
    ax_high.legend(frameon=True, framealpha=0.85, edgecolor="#ccc", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    output = args.output or (args.experiments_root.parent / "category-representation-check.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"\nSaved figure -> {output}")


if __name__ == "__main__":
    main()
