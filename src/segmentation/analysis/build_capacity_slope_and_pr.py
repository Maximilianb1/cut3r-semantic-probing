"""Finding 2 -- linear-vs-MLP gap, and where the improvement comes from.

Two figures across all three backbones (linear + MLP each):

- capacity-macro-iou-bars.png: grouped bar chart, two bars per backbone
  (Linear vs. MLP macro-IoU, sequence-cluster bootstrap 95% CI as error
  bars). A tall gap between a backbone's two bars means the MLP's extra
  capacity mattered; near-identical bars mean the backbone's signal was
  already close to linearly readable. (A slope/line chart was tried first --
  with CUT3R-trained and DINOv2 landing almost exactly on top of each other,
  the lines were too close to read as two backbones instead of one; bars with
  their own error whiskers separate cleanly even when two point estimates
  nearly coincide.)
- capacity-confusion-counts.png: small multiples, one panel per backbone,
  x = {TP, FP, FN} with a Linear and an MLP bar in each group (own y-scale
  per panel -- CUT3R-random's counts and the trained backbones' counts live
  on very different scales, and this chart is about the within-backbone
  shift, not cross-backbone magnitude). TN is left out on purpose: it's the
  background cell, ~4M and uninteresting by construction, and would flatten
  every other bar to invisible (the same reason this project's comparison
  charts already never plot token_accuracy). Precision/recall are printed as
  a one-line annotation per panel instead of their own bars -- they're a
  *ratio* of these same three counts, and showing the counts directly is what
  makes the mechanism legible: CUT3R-random's MLP mostly cuts FP (60.8k ->
  27.8k) while TP barely grows -- cleaning up false positives, not finding
  new objects -- whereas the two trained backbones' TP/FP/FN bars barely move
  at all (already near their ceiling at linear capacity). Counts come
  straight from each inference-<split>.json's tp/fp/fn (no masks needed).

Run:
    python -m src.segmentation.analysis.build_capacity_slope_and_pr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..bootstrap_iou import bootstrap_iou_ci, derive_seed, iou_clusters
from .figures import _apply_rc
from .runs import BACKBONE_COLOR, DISPLAY_NAME, load_inference, resolve_report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("reports/segmentation"))
    parser.add_argument("--backbones", nargs="+", default=["cut3r_random", "cut3r_trained", "dinov2"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="default: <reports-root>/comparison")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.reports_root / "comparison")
    _apply_rc()

    run_dirs = {(b, p): resolve_report_dir(args.reports_root, b, p)
                for b in args.backbones for p in ("linear", "mlp")}
    inferences = {k: load_inference(d, args.split) for k, d in run_dirs.items()}
    clusters = {k: iou_clusters(inf) for k, inf in inferences.items()}

    print(f"=== Finding 2: macro-IoU, linear vs. MLP, sequence-cluster bootstrap 95% CI ({args.split}) ===")
    ci: dict[tuple[str, str], tuple[float, float, float]] = {}
    for key, cl in clusters.items():
        seed = derive_seed(args.seed, f"capacity-ci/{key[0]}/{key[1]}")
        result = bootstrap_iou_ci(cl, samples=args.n_boot, seed=seed)
        mean, (lo, hi) = result["macro_iou"]["estimate"], result["macro_iou"]["ci95"]
        ci[key] = (mean, lo, hi)
        print(f"{DISPLAY_NAME.get(key[0], key[0]):15s} {key[1]:7s} macro_iou={mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    # ---- Grouped bar chart: two bars per backbone (Linear vs. MLP) ----
    order = sorted(args.backbones, key=lambda b: -ci[(b, "mlp")][0])
    x = np.arange(len(order))
    width = 0.32
    colors = [BACKBONE_COLOR.get(b, "#888") for b in order]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    linear_means = [ci[(b, "linear")][0] for b in order]
    linear_hi = [ci[(b, "linear")][2] for b in order]
    linear_err = [[ci[(b, "linear")][0] - ci[(b, "linear")][1] for b in order],
                  [ci[(b, "linear")][2] - ci[(b, "linear")][0] for b in order]]
    mlp_means = [ci[(b, "mlp")][0] for b in order]
    mlp_hi = [ci[(b, "mlp")][2] for b in order]
    mlp_err = [[ci[(b, "mlp")][0] - ci[(b, "mlp")][1] for b in order],
               [ci[(b, "mlp")][2] - ci[(b, "mlp")][0] for b in order]]

    ax.bar(x - width / 2, linear_means, width=width * 0.9, yerr=linear_err, capsize=3,
           color=colors, alpha=0.55, edgecolor="white", zorder=3, hatch="//", label="Linear")
    ax.bar(x + width / 2, mlp_means, width=width * 0.9, yerr=mlp_err, capsize=3,
           color=colors, edgecolor="white", zorder=3, label="MLP [512]")

    for xi, m, hi in zip(x, linear_means, linear_hi):
        ax.text(xi - width / 2, hi + 0.02, f"{m:.3f}", ha="center", fontsize=7, fontweight="bold")
    for xi, m, hi in zip(x, mlp_means, mlp_hi):
        ax.text(xi + width / 2, hi + 0.02, f"{m:.3f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAME.get(b, b) for b in order])
    ax.set_ylabel(f"{args.split.capitalize()} macro-IoU")
    ax.set_ylim(0, 1.08)
    ax.set_title("Probe capacity: macro-IoU, Linear vs. MLP (95% bootstrap CI)", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, framealpha=0.85, edgecolor="#cccccc", fancybox=False)
    fig.tight_layout()
    bars_output = output_dir / "capacity-macro-iou-bars.png"
    bars_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(bars_output, dpi=200)
    print(f"\nSaved figure -> {bars_output}")

    # ---- Confusion counts: one panel per backbone, x = {TP, FP, FN} ----
    print(f"\n=== Finding 2: confusion counts, linear vs. MLP (global, all tokens, {args.split}) ===")
    counts: dict[tuple[str, str], dict[str, int]] = {}
    pr: dict[tuple[str, str], tuple[float, float]] = {}
    for key, inf in inferences.items():
        m = inf["metrics"]
        counts[key] = {"tp": m["tp"], "fp": m["fp"], "fn": m["fn"]}
        pr[key] = (m["foreground_precision"], m["foreground_recall"])
        print(f"{DISPLAY_NAME.get(key[0], key[0]):15s} {key[1]:7s} "
              f"tp={m['tp']} fp={m['fp']} fn={m['fn']}  "
              f"precision={m['foreground_precision']:.4f} recall={m['foreground_recall']:.4f}")

    fig2, axes2 = plt.subplots(1, len(order), figsize=(3.4 * len(order), 4.2))
    if len(order) == 1:
        axes2 = [axes2]
    cell_x = np.arange(3)  # {TP, FP, FN}
    cell_width = 0.32
    for ax, b in zip(axes2, order):
        color = BACKBONE_COLOR.get(b, "#888")
        lin, mlp = counts[(b, "linear")], counts[(b, "mlp")]
        lin_vals = [lin["tp"], lin["fp"], lin["fn"]]
        mlp_vals = [mlp["tp"], mlp["fp"], mlp["fn"]]

        bars_lin = ax.bar(cell_x - cell_width / 2, lin_vals, width=cell_width * 0.9,
                           color=color, alpha=0.55, edgecolor="white", linewidth=0.6,
                           zorder=3, hatch="//", label="Linear")
        bars_mlp = ax.bar(cell_x + cell_width / 2, mlp_vals, width=cell_width * 0.9,
                           color=color, edgecolor="white", linewidth=0.6, zorder=3, label="MLP [512]")
        y_top = max(lin_vals + mlp_vals)
        for bar, val in zip(list(bars_lin) + list(bars_mlp), lin_vals + mlp_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * y_top,
                    f"{val / 1000:.1f}k", ha="center", va="bottom", fontsize=6.5)

        p_lin, r_lin = pr[(b, "linear")]
        p_mlp, r_mlp = pr[(b, "mlp")]
        ax.set_title(
            f"{DISPLAY_NAME.get(b, b)}\nP: {p_lin:.2f}$\\to${p_mlp:.2f}   R: {r_lin:.2f}$\\to${r_mlp:.2f}",
            fontweight="bold", fontsize=9,
        )
        ax.set_xticks(cell_x)
        ax.set_xticklabels(["TP", "FP", "FN"])
        ax.set_ylim(0, y_top * 1.15)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes2[0].set_ylabel("Token count")
    axes2[-1].legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="#cccccc",
                      fancybox=False, fontsize=7)
    fig2.suptitle(f"Confusion counts: Linear vs. MLP ({args.split}, global, all tokens; TN omitted -- background-dominated)",
                  fontsize=10.5, fontweight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.90))
    counts_output = output_dir / "capacity-confusion-counts.png"
    fig2.savefig(counts_output, dpi=200)
    print(f"Saved figure -> {counts_output}")


if __name__ == "__main__":
    main()
