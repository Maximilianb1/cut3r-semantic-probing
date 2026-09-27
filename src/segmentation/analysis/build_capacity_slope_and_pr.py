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
- capacity-pr-shift.png: one shared precision-recall plane, one arrow per
  backbone from its Linear to its MLP operating point. Arrow direction shows
  whether MLP capacity mainly cut FP (rightward, precision up) or FN
  (upward, recall up). DINOv2 and CUT3R-trained land almost on top of each
  other near the ceiling, so a small inset zooms into just that cluster.

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
    parser.add_argument("--n-boot", type=int, default=20000)
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

    # ---- PR shift: one shared plane, one arrow per backbone (Linear -> MLP) ----
    print(f"\n=== Finding 2: precision/recall shift, linear vs. MLP (global, all tokens, {args.split}) ===")
    pr: dict[tuple[str, str], tuple[float, float]] = {}
    for key, inf in inferences.items():
        m = inf["metrics"]
        pr[key] = (m["foreground_precision"], m["foreground_recall"])
        print(f"{DISPLAY_NAME.get(key[0], key[0]):15s} {key[1]:7s} "
              f"precision={m['foreground_precision']:.4f} recall={m['foreground_recall']:.4f}")

    fig2 = plt.figure(figsize=(6.4, 5.2))
    ax = fig2.add_axes((0.11, 0.10, 0.85, 0.78))

    def _draw_arrows(target_ax, backbones: list[str], *, label_fontsize: float) -> None:
        for b in backbones:
            color = BACKBONE_COLOR.get(b, "#888")
            p_lin, r_lin = pr[(b, "linear")]
            p_mlp, r_mlp = pr[(b, "mlp")]
            target_ax.annotate(
                "", xy=(r_mlp, p_mlp), xytext=(r_lin, p_lin),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16, shrinkA=0, shrinkB=0),
                zorder=3,
            )
            target_ax.scatter([r_lin], [p_lin], s=26, facecolor="white", edgecolor=color, linewidth=1.6, zorder=4)
            target_ax.scatter([r_mlp], [p_mlp], s=42, facecolor=color, edgecolor="white", linewidth=0.8, zorder=4)
            target_ax.annotate(
                DISPLAY_NAME.get(b, b), xy=(r_mlp, p_mlp), xytext=(4, 4), textcoords="offset points",
                fontsize=label_fontsize, fontweight="bold", color=color,
            )

    _draw_arrows(ax, order, label_fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Foreground recall  (TP / (TP + FN))")
    ax.set_ylabel("Foreground precision  (TP / (TP + FP))")
    ax.grid(linewidth=0.3, alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.scatter([], [], s=26, facecolor="white", edgecolor="#555", linewidth=1.6, label="Linear")
    ax.scatter([], [], s=42, facecolor="#555", edgecolor="white", linewidth=0.8, label="MLP [512]")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="#cccccc", fancybox=False, fontsize=8)

    # Zoom inset for backbones already near the P/R ceiling on both endpoints.
    ceiling = 0.7
    near_ceiling = [b for b in order
                     if min(pr[(b, "linear")][0], pr[(b, "mlp")][0], pr[(b, "linear")][1], pr[(b, "mlp")][1]) > ceiling]
    if len(near_ceiling) >= 2:
        pts = [pr[(b, p)] for b in near_ceiling for p in ("linear", "mlp")]
        p_lo, p_hi = min(v[0] for v in pts), max(v[0] for v in pts)
        r_lo, r_hi = min(v[1] for v in pts), max(v[1] for v in pts)
        pad_p, pad_r = max(0.02, (p_hi - p_lo) * 0.35), max(0.02, (r_hi - r_lo) * 0.35)
        axins = ax.inset_axes((0.04, 0.70, 0.34, 0.28))
        _draw_arrows(axins, near_ceiling, label_fontsize=6.5)
        axins.set_xlim(r_lo - pad_r, r_hi + pad_r)
        axins.set_ylim(p_lo - pad_p, p_hi + pad_p)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.grid(linewidth=0.3, alpha=0.4, zorder=0)
        for spine in axins.spines.values():
            spine.set_edgecolor("#999999")
        ax.indicate_inset_zoom(axins, edgecolor="#999999")

    fig2.suptitle(f"Precision/recall shift: Linear -> MLP ({args.split}, global, all tokens)",
                  fontsize=10.5, fontweight="bold", y=0.985)
    counts_output = output_dir / "capacity-pr-shift.png"
    fig2.savefig(counts_output, dpi=200)
    print(f"Saved figure -> {counts_output}")


if __name__ == "__main__":
    main()
