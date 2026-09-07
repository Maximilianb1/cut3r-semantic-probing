"""Finding 5 -- learning curves, with best_val_epoch marked.

Two side-by-side panels (linear probes / MLP probes), 3 lines each (one per
backbone), val macro-foreground-IoU vs. epoch, with a marker at each run's
best_val_epoch (metrics.json's `best_val_epoch` -- the checkpoint actually
selected and reported, not whichever epoch training happened to stop on).

Splitting into two panels by probe capacity (rather than one 6-line overlay)
makes the outlier shapes easy to read at a glance: cut3r_random-linear should
visibly peak early then fall for the rest of the budget, and dinov2-mlp
should peak early then mildly overfit -- both easier to see against
same-capacity peers than against all 6 lines at once.

No CI: this is a single training run per config (one seed each), not a
resampled statistic -- there's nothing to bootstrap here without repeated
training runs, which this project doesn't do.

Run:
    python -m src.segmentation.analysis.build_learning_curve_panels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .figures import _apply_rc
from .runs import BACKBONE_COLOR, DISPLAY_NAME, load_metrics, resolve_report_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("reports/segmentation"))
    parser.add_argument("--backbones", nargs="+", default=["cut3r_random", "cut3r_trained", "dinov2"])
    parser.add_argument("--metric", default="macro_foreground_iou")
    parser.add_argument("--output", type=Path, default=None,
                         help="default: <reports-root>/comparison/learning-curve-panels.png")
    args = parser.parse_args()

    _apply_rc()
    fig, (ax_lin, ax_mlp) = plt.subplots(1, 2, figsize=(9.5, 4.2), sharey=True)

    print("=== Finding 5: best_val_epoch per run ===")
    for ax, probe, title in ((ax_lin, "linear", "Linear probes"), (ax_mlp, "mlp", "MLP [512] probes")):
        for b in args.backbones:
            metrics = load_metrics(resolve_report_dir(args.reports_root, b, probe))
            history = metrics["history"]
            best_epoch = metrics["best_val_epoch"]
            epochs = [h["epoch"] for h in history]
            values = [h["val"][args.metric] for h in history]
            color = BACKBONE_COLOR.get(b, "#888")
            name = DISPLAY_NAME.get(b, b)

            ax.plot(epochs, values, marker="o", markersize=2.5, linewidth=1.6, color=color, label=name)
            best_value = next(h["val"][args.metric] for h in history if h["epoch"] == best_epoch)
            ax.scatter([best_epoch], [best_value], marker="*", s=140, color=color,
                       edgecolor="black", linewidth=0.6, zorder=5)
            print(f"{name:15s} {probe:7s} best_val_epoch={best_epoch:2d}  "
                  f"best_val_{args.metric}={best_value:.4f}  final={values[-1]:.4f}")

        ax.set_xlabel("Epoch")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(linewidth=0.3, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_lin.set_ylabel(f"Val {args.metric.replace('_', ' ')}")
    ax_lin.legend(loc="lower right", frameon=True, framealpha=0.85, edgecolor="#ccc", fancybox=False, fontsize=7)
    fig.suptitle("Validation learning curves (* = best_val_epoch, the checkpoint reported)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output = args.output or (args.reports_root / "comparison" / "learning-curve-panels.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"\nSaved figure -> {output}")


if __name__ == "__main__":
    main()
