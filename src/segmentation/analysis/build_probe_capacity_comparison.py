"""Does a backbone's segmentation signal need the [512] MLP head, or is it
already (near-)linearly decodable? Compares the `mlp` (hidden_dims: [512])
and `linear` (hidden_dims: []) heads, holding backbone and data fixed, on
already-computed inference-<split>.json -- no re-training, no re-inference.

Both heads are evaluated on the identical test windows (same cache, same
split -- only the head differs), so this is a genuinely paired comparison:
the per-window IoU delta is bootstrapped directly, the same way
build_score_comparison.py bootstraps cross-backbone deltas.

Saves two figures:
- probe-capacity-comparison.png: a grouped bar chart, two bars per backbone
  (mlp vs. linear macro-IoU), so all three backbones' capacity ablations read
  together in one place.
- probe-capacity-per-category.png: one MLP-vs-linear scatter panel per
  backbone (one point per category, y=x reference line), to check whether
  the aggregate capacity gap is spread evenly across categories or driven by
  a few categories collapsing under the linear head -- the bar chart alone
  can't distinguish those two stories.

Run example (after training + inference with --checkpoint-selection best_val
for both the *_expanded_mlp.yaml and *_expanded_linear.yaml config of each
backbone):

    python -m src.segmentation.analysis.build_probe_capacity_comparison \
        --experiments-root src/segmentation/experiments \
        --backbones cut3r_trained cut3r_random dinov2 \
        --mlp-run-suffix=-expanded-bestval --linear-run-suffix=-expanded-linear-bestval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .build_score_comparison import bootstrap_ci_mean
from .runs import BACKBONE_COLOR, DISPLAY_NAME, load_inference, load_per_window_iou, resolve_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--mlp-run-suffix", default="-expanded-bestval",
                         help="run_suffix for the [512] MLP runs (segmentation-<backbone><suffix>)")
    parser.add_argument("--linear-run-suffix", default="-expanded-linear-bestval",
                         help="run_suffix for the linear (hidden_dims: []) runs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, default=None,
                         help="default: <experiments-root>/../probe-capacity-comparison.png")
    args = parser.parse_args()

    generator = np.random.default_rng(args.seed)

    mlp_dirs = {b: resolve_run_dir(args.experiments_root, b, args.mlp_run_suffix) for b in args.backbones}
    linear_dirs = {b: resolve_run_dir(args.experiments_root, b, args.linear_run_suffix) for b in args.backbones}

    mlp_iou = {b: {w: row["foreground_iou"] for w, row in load_per_window_iou(mlp_dirs[b], args.split).items()}
               for b in args.backbones}
    linear_iou = {b: {w: row["foreground_iou"] for w, row in load_per_window_iou(linear_dirs[b], args.split).items()}
                  for b in args.backbones}

    print(f"=== Probe-capacity ablation: [512] MLP vs. linear ([]) head, paired per-window ({args.split}) ===")
    results = {}
    for b in args.backbones:
        window_ids = sorted(set(mlp_iou[b]) & set(linear_iou[b]))
        if not window_ids:
            raise ValueError(
                f"No common {args.split} windows between {mlp_dirs[b]} and {linear_dirs[b]} for {b}; "
                "check that both runs share the same cache/split"
            )
        mlp_values = np.array([mlp_iou[b][w] for w in window_ids])
        linear_values = np.array([linear_iou[b][w] for w in window_ids])
        deltas = mlp_values - linear_values

        mlp_mean, mlp_lo, mlp_hi = bootstrap_ci_mean(mlp_values, n_boot=args.n_boot, generator=generator)
        linear_mean, linear_lo, linear_hi = bootstrap_ci_mean(linear_values, n_boot=args.n_boot, generator=generator)
        delta_mean, delta_lo, delta_hi = bootstrap_ci_mean(deltas, n_boot=args.n_boot, generator=generator)
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        significant = not (delta_lo <= 0.0 <= delta_hi)

        results[b] = {
            "n_windows": len(window_ids),
            "mlp": (mlp_mean, mlp_lo, mlp_hi),
            "linear": (linear_mean, linear_lo, linear_hi),
            "delta": (delta_mean, delta_lo, delta_hi),
            "wins": wins, "losses": losses, "significant": significant,
        }
        name = DISPLAY_NAME.get(b, b)
        print(f"\n{name} ({len(window_ids)} windows):")
        print(f"  mlp    macro_iou={mlp_mean:.4f}  95% CI [{mlp_lo:.4f}, {mlp_hi:.4f}]")
        print(f"  linear macro_iou={linear_mean:.4f}  95% CI [{linear_lo:.4f}, {linear_hi:.4f}]")
        print(f"  delta (mlp - linear)={delta_mean:+.4f}  95% CI [{delta_lo:+.4f}, {delta_hi:+.4f}]  "
              f"wins/losses={wins}/{losses}  {'SIGNIFICANT' if significant else 'not significant'}")

    # ---- One combined figure: two bars per backbone (mlp vs. linear), so
    # all three capacity ablations read together in one place. Same visual
    # language as build_score_comparison.py's macro-micro chart (hatch marks
    # the "other" bar within a color-coded backbone pair). ----
    order = sorted(args.backbones, key=lambda b: -results[b]["mlp"][0])
    x = np.arange(len(order))
    width = 0.32
    colors = [BACKBONE_COLOR.get(b, "#888") for b in order]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    mlp_means = [results[b]["mlp"][0] for b in order]
    mlp_hi = [results[b]["mlp"][2] for b in order]
    mlp_err = [[results[b]["mlp"][0] - results[b]["mlp"][1] for b in order],
               [results[b]["mlp"][2] - results[b]["mlp"][0] for b in order]]
    linear_means = [results[b]["linear"][0] for b in order]
    linear_hi = [results[b]["linear"][2] for b in order]
    linear_err = [[results[b]["linear"][0] - results[b]["linear"][1] for b in order],
                  [results[b]["linear"][2] - results[b]["linear"][0] for b in order]]

    ax.bar(x - width / 2, mlp_means, width=width * 0.9, yerr=mlp_err, capsize=3,
           color=colors, edgecolor="white", zorder=3, label="MLP [512]")
    ax.bar(x + width / 2, linear_means, width=width * 0.9, yerr=linear_err, capsize=3,
           color=colors, alpha=0.55, edgecolor="white", zorder=3, hatch="//", label="Linear []")

    # Label position follows each bar's own CI upper bound (not a fixed offset
    # from the mean), so a wide CI never lets the label collide with its cap.
    for xi, m, hi in zip(x, mlp_means, mlp_hi):
        ax.text(xi - width / 2, hi + 0.02, f"{m:.3f}", ha="center", fontsize=7, fontweight="bold")
    for xi, m, hi in zip(x, linear_means, linear_hi):
        ax.text(xi + width / 2, hi + 0.02, f"{m:.3f}", ha="center", fontsize=7, fontweight="bold")

    n_windows = next(iter(results.values()))["n_windows"]
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAME.get(b, b) for b in order])
    ax.set_ylabel(f"{args.split.capitalize()} macro-IoU")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Probe-capacity ablation: MLP vs. linear head ({n_windows} windows, 95% CI)",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="#cccccc", fancybox=False)
    fig.tight_layout()

    output = args.output or (args.experiments_root.parent / "probe-capacity-comparison.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure -> {output}")

    # ---- Per-category scatter: does the capacity gap hit every category
    # evenly, or is it concentrated in a few? Points on the y=x line are
    # capacity-invariant (implicit signal); points below it are categories
    # where the MLP's extra capacity is doing real work. ----
    fig_cat, axes_cat = plt.subplots(1, len(order), figsize=(4.0 * len(order), 4.0), sharex=True, sharey=True)
    if len(order) == 1:
        axes_cat = [axes_cat]
    outlier_threshold = 0.2  # annotate only categories with a big MLP-vs-linear gap, to avoid label clutter
    for ax_cat, b in zip(axes_cat, order):
        mlp_cat = load_inference(mlp_dirs[b], args.split)["metrics"]["per_category_iou"]
        linear_cat = load_inference(linear_dirs[b], args.split)["metrics"]["per_category_iou"]
        categories = sorted(set(mlp_cat) & set(linear_cat))
        xs = np.array([mlp_cat[c] for c in categories])
        ys = np.array([linear_cat[c] for c in categories])

        ax_cat.plot([0, 1], [0, 1], "--", color="#999", linewidth=1, zorder=1)
        ax_cat.scatter(xs, ys, color=BACKBONE_COLOR.get(b, "#888"), edgecolor="white",
                       linewidth=0.5, s=36, zorder=3, alpha=0.85)
        for cat, xi, yi in zip(categories, xs, ys):
            if xi - yi > outlier_threshold:
                ax_cat.annotate(cat, (xi, yi), fontsize=6, xytext=(3, -3),
                                 textcoords="offset points", color="#555")

        ax_cat.set_xlim(-0.02, 1.02)
        ax_cat.set_ylim(-0.02, 1.02)
        ax_cat.set_title(DISPLAY_NAME.get(b, b), fontweight="bold", fontsize=10)
        ax_cat.set_xlabel("MLP [512] category IoU")
        ax_cat.grid(linewidth=0.3, alpha=0.5, zorder=0)
        ax_cat.set_aspect("equal")
        ax_cat.spines["top"].set_visible(False)
        ax_cat.spines["right"].set_visible(False)
    axes_cat[0].set_ylabel("Linear [] category IoU")

    fig_cat.suptitle(f"Per-category IoU: MLP vs. linear head ({args.split}, "
                      f"labels = categories >{outlier_threshold:.1f} below the diagonal)",
                      fontsize=10, fontweight="bold")
    fig_cat.tight_layout(rect=(0, 0, 1, 0.94))

    category_output = args.output.with_name("probe-capacity-per-category.png") if args.output else \
        (args.experiments_root.parent / "probe-capacity-per-category.png")
    category_output.parent.mkdir(parents=True, exist_ok=True)
    fig_cat.savefig(category_output, dpi=200, bbox_inches="tight")
    print(f"Saved figure -> {category_output}")


if __name__ == "__main__":
    main()
