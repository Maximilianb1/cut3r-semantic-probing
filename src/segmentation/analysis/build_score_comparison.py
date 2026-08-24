"""Precision/recall, bootstrap CI, and a paired per-window comparison across
backbones -- straight from already-computed inference-<split>.json /
masks-<split>.pt. No re-training, no re-inference.

Backbones sharing the same manifest are scored on the *identical* test
windows, so treating their aggregate IoUs as independent throws away
information: this script joins on window_id and bootstraps the paired
per-window deltas, which is a stronger test of "is this gap real" than
comparing two point estimates.

Saves three figures, grouped by what's meaningful to read together:
- macro-iou-ci.png: macro-IoU alone with its bootstrap CI. Duplicated as its
  own figure (not just inside macro-micro-iou.png below) because it's the
  ADR/primary metric, cited on its own in write-ups.
- macro-micro-iou.png: macro-IoU + micro-IoU, the "overlap quality" pair.
  They can disagree -- macro gives every window equal weight, micro is
  dominated by whichever windows have the most foreground tokens -- so both
  are worth keeping rather than picking one.
- precision-recall.png: precision + recall, the "prediction bias" pair --
  diagnoses over- vs under-prediction, a different question from "how good."
Token accuracy is deliberately never plotted here: its ~0.78 always-background
baseline and tendency to move opposite real IoU (see EXP-005) make it
misleading next to overlap metrics.

Run example (after train_segmentation.py --checkpoint-selection best_val +
inference_segmentation.py --save-masks have produced metrics for each backbone):

    python -m src.segmentation.analysis.build_score_comparison \
        --experiments-root src/segmentation/experiments \
        --backbones cut3r_trained cut3r_random dinov2 --run-suffix=-expanded-bestval
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .runs import BACKBONE_COLOR, DISPLAY_NAME, load_masks, load_per_window_iou, resolve_run_dir


def precision_recall(run_dir: Path, split: str) -> dict[str, float]:
    """Global precision/recall/micro-IoU over every token in every window of ``split``.

    Micro-IoU (tp / (tp+fp+fn)) is the token-weighted counterpart to the
    per-window-averaged macro-IoU bootstrapped below: it shares the same
    tp/fp/fn already computed here for precision/recall, so it comes for free.
    """
    masks = load_masks(run_dir, split)
    tp = fp = fn = tn = 0
    for entry in masks.values():
        pred = entry["predicted_labels"].reshape(-1).bool()
        gt = entry["target_labels"].reshape(-1).bool()
        tp += int((pred & gt).sum())
        fp += int((pred & ~gt).sum())
        fn += int((~pred & gt).sum())
        tn += int((~pred & ~gt).sum())
    return {
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "micro_iou": tp / (tp + fp + fn) if (tp + fp + fn) else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def bootstrap_ci_mean(
    values: np.ndarray, *, n_boot: int, generator: np.random.Generator
) -> tuple[float, float, float]:
    """(mean, 2.5th percentile, 97.5th percentile) over ``n_boot`` resamples."""
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = generator.choice(values, size=n, replace=True).mean()
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--run-suffix", default="", help="e.g. -expanded-bestval to match segmentation-<backbone>-expanded-bestval dirs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", type=Path, default=None,
                         help="Where to save the standalone macro-IoU CI figure (default: <experiments-root>/../macro-iou-ci.png)")
    parser.add_argument("--overlap-output", type=Path, default=None,
                         help="Where to save the macro+micro IoU figure "
                              "(default: <experiments-root>/../macro-micro-iou.png)")
    parser.add_argument("--pr-output", type=Path, default=None,
                         help="Where to save the precision/recall figure "
                              "(default: <experiments-root>/../precision-recall.png)")
    parser.add_argument("--csv-output", type=Path, default=None,
                         help="Where to save the full per-window comparison CSV "
                              "(default: <experiments-root>/../per-window-comparison.csv)")
    args = parser.parse_args()

    generator = np.random.default_rng(args.seed)
    run_dirs = {b: resolve_run_dir(args.experiments_root, b, args.run_suffix) for b in args.backbones}

    per_window_rows = {b: load_per_window_iou(run_dirs[b], args.split) for b in args.backbones}
    per_window = {b: {w: row["foreground_iou"] for w, row in rows.items()} for b, rows in per_window_rows.items()}
    window_ids = sorted(set.intersection(*(set(d.keys()) for d in per_window.values())))
    print(f"Common {args.split} windows across all backbones: {len(window_ids)}")

    print("\n=== Precision / recall (global, all tokens) ===")
    pr = {}
    for b in args.backbones:
        result = precision_recall(run_dirs[b], args.split)
        pr[b] = result
        print(f"{DISPLAY_NAME.get(b, b):15s} precision={result['precision']:.4f} recall={result['recall']:.4f} "
              f"micro_iou={result['micro_iou']:.4f} (tp={result['tp']} fp={result['fp']} fn={result['fn']} tn={result['tn']})")

    print(f"\n=== Macro-IoU bootstrap 95% CI ({args.n_boot} resamples over {len(window_ids)} windows) ===")
    ci = {}
    for b in args.backbones:
        values = np.array([per_window[b][w] for w in window_ids])
        mean, lo, hi = bootstrap_ci_mean(values, n_boot=args.n_boot, generator=generator)
        ci[b] = (mean, lo, hi)
        print(f"{DISPLAY_NAME.get(b, b):15s} macro_iou={mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    print("\n=== Paired per-window comparison (bootstrap 95% CI on mean delta) ===")
    for i, a in enumerate(args.backbones):
        for b in args.backbones[i + 1:]:
            deltas = np.array([per_window[a][w] - per_window[b][w] for w in window_ids])
            mean, lo, hi = bootstrap_ci_mean(deltas, n_boot=args.n_boot, generator=generator)
            wins = int((deltas > 0).sum())
            losses = int((deltas < 0).sum())
            significant = not (lo <= 0.0 <= hi)
            print(f"{DISPLAY_NAME.get(a, a)} - {DISPLAY_NAME.get(b, b):15s} mean_delta={mean:+.4f}  "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]  wins/losses={wins}/{losses}  "
                  f"{'SIGNIFICANT' if significant else 'not significant'}")

    order = sorted(args.backbones, key=lambda b: -ci[b][0])
    x = np.arange(len(order))
    means = [ci[b][0] for b in order]
    hi_ci = [ci[b][2] for b in order]
    lo_err = [ci[b][0] - ci[b][1] for b in order]
    hi_err = [ci[b][2] - ci[b][0] for b in order]
    colors = [BACKBONE_COLOR.get(b, "#888") for b in order]

    # ---- Standalone macro-IoU (duplicated below inside the overlap chart too) ----
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(x, means, yerr=[lo_err, hi_err], capsize=4, color=colors, edgecolor="white", width=0.6, zorder=3)
    for xi, m, hi in zip(x, means, hi_ci):
        ax.text(xi, hi + 0.035, f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAME.get(b, b) for b in order])
    ax.set_ylabel(f"{args.split.capitalize()} macro-IoU")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{args.split.capitalize()} macro-IoU, 95% bootstrap CI ({len(window_ids)} windows)",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    output = args.output or (args.experiments_root.parent / "macro-iou-ci.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"\nSaved figure -> {output}")

    # ---- Overlap quality: macro-IoU + micro-IoU, both as plain point values.
    # No CI on either here: a fair CI for micro-IoU would need the same per-window
    # resampling as macro's (its ~59k tokens are clustered within 101 windows, not
    # independent, so a naive per-token bootstrap would understate its uncertainty)
    # -- not implemented, so neither bar claims a CI in this combined view. Macro's
    # CI still lives on its own in macro-iou-ci.png above. ----
    micro_values = [pr[b]["micro_iou"] for b in order]
    fig_ov, ax_ov = plt.subplots(figsize=(6.0, 3.6))
    width = 0.32
    ax_ov.bar(x - width / 2, means, width=width * 0.9,
              color=colors, edgecolor="white", zorder=3, label="Macro-IoU")
    ax_ov.bar(x + width / 2, micro_values, width=width * 0.9, color=colors, alpha=0.55,
              edgecolor="white", zorder=3, label="Micro-IoU", hatch="//")
    for xi, m in zip(x, means):
        ax_ov.text(xi - width / 2, m + 0.03, f"{m:.3f}", ha="center", fontsize=7, fontweight="bold")
    for xi, m in zip(x, micro_values):
        ax_ov.text(xi + width / 2, m + 0.03, f"{m:.3f}", ha="center", fontsize=7, fontweight="bold")
    ax_ov.set_xticks(x)
    ax_ov.set_xticklabels([DISPLAY_NAME.get(b, b) for b in order])
    ax_ov.set_ylabel(f"{args.split.capitalize()} IoU")
    ax_ov.set_ylim(0, 1.0)
    ax_ov.set_title(f"{args.split.capitalize()} macro- vs micro-IoU ({len(window_ids)} windows)",
                     fontsize=10, fontweight="bold")
    ax_ov.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax_ov.spines["top"].set_visible(False)
    ax_ov.spines["right"].set_visible(False)
    ax_ov.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="#cccccc", fancybox=False)
    fig_ov.tight_layout()
    overlap_output = args.overlap_output or (args.experiments_root.parent / "macro-micro-iou.png")
    overlap_output.parent.mkdir(parents=True, exist_ok=True)
    fig_ov.savefig(overlap_output, dpi=200)
    print(f"Saved figure -> {overlap_output}")

    # ---- Prediction bias: precision + recall only. Deliberately excludes
    # micro-IoU (now in the overlap chart above) and token accuracy (misleading
    # baseline -- see this script's module docstring). ----
    pr_order = sorted(args.backbones, key=lambda b: -pr[b]["micro_iou"])
    pr_metrics = ("precision", "recall")
    pr_labels = {"precision": "Precision", "recall": "Recall"}
    pr_palette = {"precision": "#6BAED6", "recall": "#FDAE6B"}
    fig_pr, ax_pr = plt.subplots(figsize=(5.0, 3.6))
    x_pr = np.arange(len(pr_order))
    bar_width = 0.7 / len(pr_metrics)
    for idx, metric in enumerate(pr_metrics):
        values = [pr[b][metric] for b in pr_order]
        offset = (idx - (len(pr_metrics) - 1) / 2) * bar_width
        bars = ax_pr.bar(
            x_pr + offset, values, width=bar_width * 0.88,
            label=pr_labels[metric], color=pr_palette[metric],
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        for bar, val in zip(bars, values):
            ax_pr.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold",
            )
    ax_pr.set_xticks(x_pr)
    ax_pr.set_xticklabels([DISPLAY_NAME.get(b, b) for b in pr_order])
    ax_pr.set_ylabel("Score")
    ax_pr.set_ylim(0, 1.12)
    ax_pr.set_title(f"{args.split.capitalize()} precision / recall (global, all tokens)",
                     fontsize=10, fontweight="bold")
    ax_pr.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax_pr.spines["top"].set_visible(False)
    ax_pr.spines["right"].set_visible(False)
    ax_pr.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="#cccccc", fancybox=False)
    fig_pr.tight_layout()
    pr_output = args.pr_output or (args.experiments_root.parent / "precision-recall.png")
    pr_output.parent.mkdir(parents=True, exist_ok=True)
    fig_pr.savefig(pr_output, dpi=200)
    print(f"Saved figure -> {pr_output}")

    # Full per-window comparison, one row per window: every backbone's IoU plus
    # every pairwise delta. This is the detailed data behind the printed
    # summaries above -- kept as a separate file rather than inlined into an
    # experiment write-up, since a 101-row table belongs in results/, not in a
    # summary document.
    csv_output = args.csv_output or (args.experiments_root.parent / "per-window-comparison.csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    pair_names = [(a, b) for i, a in enumerate(args.backbones) for b in args.backbones[i + 1:]]
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["window_id", "category"] + [f"{b}_iou" for b in args.backbones] + [
            f"delta_{a}_minus_{b}" for a, b in pair_names
        ]
        writer.writerow(header)
        for w in window_ids:
            category = per_window_rows[args.backbones[0]][w]["category"]
            ious = [per_window[b][w] for b in args.backbones]
            deltas = [per_window[a][w] - per_window[b][w] for a, b in pair_names]
            writer.writerow([w, category] + ious + deltas)
    print(f"Saved per-window comparison CSV -> {csv_output} ({len(window_ids)} rows)")


if __name__ == "__main__":
    main()
