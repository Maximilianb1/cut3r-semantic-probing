"""Finding 3 -- is category difficulty backbone-agnostic?

Heatmap: rows = 26 categories (sorted by mean rank across the 6 runs,
hardest at top), columns = the 6 backbone x head-capacity runs, cell =
test-split per-category IoU. Captioned with Spearman rank correlations,
computed manually (average-rank Pearson-on-ranks, no scipy dependency --
matching this project's existing no-scikit-learn stance in curve_metrics.py):

- among the 4 trained-backbone runs (cut3r_trained x {linear,mlp}, dinov2 x
  {linear,mlp}): pairwise rho, min/max reported.
- cut3r_random (either probe) vs. every trained run: pairwise rho, min/max
  reported.

The two ranges should separate cleanly if difficulty is dataset/geometry
driven (shared ranking whenever there's real signal to rank with) rather than
embedding-content driven (which would show equally weak agreement
everywhere, including among the trained runs).

Run:
    python -m src.segmentation.analysis.build_category_difficulty_heatmap
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .figures import _apply_rc
from .runs import DISPLAY_NAME, load_inference, resolve_report_dir


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via average ranks + Pearson on those ranks."""
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(values), dtype=np.float64)
        # Average-rank tie correction.
        sorted_vals = values[order]
        i = 0
        while i < len(sorted_vals):
            j = i
            while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
            i = j + 1
        return ranks

    ra, rb = rank(np.asarray(a, dtype=np.float64)), rank(np.asarray(b, dtype=np.float64))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("reports/segmentation"))
    parser.add_argument("--backbones", nargs="+", default=["cut3r_random", "cut3r_trained", "dinov2"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=None,
                         help="default: <reports-root>/comparison/category-difficulty-heatmap.png")
    args = parser.parse_args()

    run_keys = [(b, p) for b in args.backbones for p in ("linear", "mlp")]
    per_category = {}
    for b, p in run_keys:
        inf = load_inference(resolve_report_dir(args.reports_root, b, p), args.split)
        per_category[(b, p)] = inf["metrics"]["per_category_iou"]

    categories = sorted(set.intersection(*(set(d.keys()) for d in per_category.values())))
    matrix = np.array([[per_category[key][c] for key in run_keys] for c in categories])  # [n_cat, n_run]

    # Sort rows by mean rank across the trained-backbone runs only (hardest = lowest
    # IoU rank = top of plot). cut3r_random is excluded from the sort -- it's shown
    # for comparison, but including it would let its (weakly correlated) ranking
    # drag categories' positions away from where they actually sit for the trained
    # backbones, which is what this figure is meant to illustrate.
    trained_col_idx = [i for i, (b, _) in enumerate(run_keys) if b != "cut3r_random"]
    ranks_per_run = np.apply_along_axis(lambda col: col.argsort().argsort(), 0, matrix)
    mean_rank = ranks_per_run[:, trained_col_idx].mean(axis=1)
    order = np.argsort(mean_rank)  # ascending: lowest mean rank (hardest) first
    matrix_sorted = matrix[order]
    categories_sorted = [categories[i] for i in order]

    # ---- Spearman correlations ----
    trained_keys = [k for k in run_keys if k[0] != "cut3r_random"]
    random_keys = [k for k in run_keys if k[0] == "cut3r_random"]

    def col(key):
        return matrix[:, run_keys.index(key)]

    trained_pairs = list(itertools.combinations(trained_keys, 2))
    trained_rhos = [spearman_rho(col(a), col(b)) for a, b in trained_pairs]
    random_vs_trained_rhos = [spearman_rho(col(r), col(t)) for r in random_keys for t in trained_keys]

    print("=== Finding 3: per-category IoU Spearman rank correlation ===")
    print(f"Among {len(trained_keys)} trained-backbone runs ({len(trained_pairs)} pairs): "
          f"rho in [{min(trained_rhos):.2f}, {max(trained_rhos):.2f}]")
    print(f"cut3r_random vs. any trained run ({len(random_vs_trained_rhos)} pairs): "
          f"rho in [{min(random_vs_trained_rhos):.2f}, {max(random_vs_trained_rhos):.2f}]")

    # ---- Heatmap ----
    _apply_rc()
    col_labels = [f"{DISPLAY_NAME.get(b, b)}\n{p}" for b, p in run_keys]
    fig, ax = plt.subplots(figsize=(5.6, 0.28 * len(categories_sorted) + 1.6))
    im = ax.imshow(matrix_sorted, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7)
    ax.set_yticks(range(len(categories_sorted)))
    ax.set_yticklabels(categories_sorted, fontsize=7)
    for i in range(matrix_sorted.shape[0]):
        for j in range(matrix_sorted.shape[1]):
            val = matrix_sorted[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5,
                    color="black" if 0.25 < val < 0.75 else "white")
    ax.set_title(f"Per-category test IoU by run (rows: hardest -> easiest)", fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Foreground IoU", fontsize=8)

    caption = (
        f"Spearman rho -- trained-backbone pairs: [{min(trained_rhos):.2f}, {max(trained_rhos):.2f}]; "
        f"cut3r_random vs. trained: [{min(random_vs_trained_rhos):.2f}, {max(random_vs_trained_rhos):.2f}]"
    )
    fig.text(0.5, -0.01, caption, ha="center", fontsize=7.5, style="italic")
    fig.tight_layout()

    output = args.output or (args.reports_root / "comparison" / "category-difficulty-heatmap.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure -> {output}")


if __name__ == "__main__":
    main()
