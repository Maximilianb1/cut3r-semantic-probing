"""Precision/recall, bootstrap CI, and a paired per-window comparison across
backbones -- straight from already-computed inference-<split>.json /
masks-<split>.pt. No re-training, no re-inference.

Backbones sharing the same manifest are scored on the *identical* test
windows, so treating their aggregate IoUs as independent throws away
information: this script joins on window_id and bootstraps the paired
per-window deltas, which is a stronger test of "is this gap real" than
comparing two point estimates.

Run example (after train_segmentation.py --checkpoint-selection best_val +
inference_segmentation.py --save-masks have produced metrics for each backbone):

    python -m src.segmentation.analysis.build_bootstrap_ci \
        --experiments-root src/segmentation/experiments \
        --backbones cut3r_trained cut3r_random dinov2 --run-suffix=-expanded-bestval
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_DISPLAY_NAME = {
    "cut3r_trained": "CUT3R-trained",
    "cut3r_random": "CUT3R-random",
    "dinov2": "DINOv2",
}


def load_per_window(run_dir: Path, split: str) -> dict[str, dict[str, float | str]]:
    data = json.loads((run_dir / f"inference-{split}.json").read_text(encoding="utf-8"))
    return {row["window_id"]: row for row in data["per_window_iou"]}


def precision_recall(run_dir: Path, split: str) -> dict[str, float]:
    """Global precision/recall over every token in every window of ``split``."""
    masks = torch.load(run_dir / f"masks-{split}.pt", weights_only=False)
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
                         help="Where to save the macro-IoU CI figure (default: <experiments-root>/../macro-iou-ci.png)")
    parser.add_argument("--csv-output", type=Path, default=None,
                         help="Where to save the full per-window comparison CSV "
                              "(default: <experiments-root>/../per-window-comparison.csv)")
    args = parser.parse_args()

    generator = np.random.default_rng(args.seed)
    run_dirs = {b: args.experiments_root / f"segmentation-{b}{args.run_suffix}" for b in args.backbones}

    per_window_rows = {b: load_per_window(run_dirs[b], args.split) for b in args.backbones}
    per_window = {b: {w: row["foreground_iou"] for w, row in rows.items()} for b, rows in per_window_rows.items()}
    window_ids = sorted(set.intersection(*(set(d.keys()) for d in per_window.values())))
    print(f"Common {args.split} windows across all backbones: {len(window_ids)}")

    print("\n=== Precision / recall (global, all tokens) ===")
    pr = {}
    for b in args.backbones:
        result = precision_recall(run_dirs[b], args.split)
        pr[b] = result
        print(f"{_DISPLAY_NAME.get(b, b):15s} precision={result['precision']:.4f} recall={result['recall']:.4f} "
              f"(tp={result['tp']} fp={result['fp']} fn={result['fn']} tn={result['tn']})")

    print(f"\n=== Macro-IoU bootstrap 95% CI ({args.n_boot} resamples over {len(window_ids)} windows) ===")
    ci = {}
    for b in args.backbones:
        values = np.array([per_window[b][w] for w in window_ids])
        mean, lo, hi = bootstrap_ci_mean(values, n_boot=args.n_boot, generator=generator)
        ci[b] = (mean, lo, hi)
        print(f"{_DISPLAY_NAME.get(b, b):15s} macro_iou={mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    print("\n=== Paired per-window comparison (bootstrap 95% CI on mean delta) ===")
    for i, a in enumerate(args.backbones):
        for b in args.backbones[i + 1:]:
            deltas = np.array([per_window[a][w] - per_window[b][w] for w in window_ids])
            mean, lo, hi = bootstrap_ci_mean(deltas, n_boot=args.n_boot, generator=generator)
            wins = int((deltas > 0).sum())
            losses = int((deltas < 0).sum())
            significant = not (lo <= 0.0 <= hi)
            print(f"{_DISPLAY_NAME.get(a, a)} - {_DISPLAY_NAME.get(b, b):15s} mean_delta={mean:+.4f}  "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}]  wins/losses={wins}/{losses}  "
                  f"{'SIGNIFICANT' if significant else 'not significant'}")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    order = sorted(args.backbones, key=lambda b: -ci[b][0])
    x = np.arange(len(order))
    means = [ci[b][0] for b in order]
    hi_ci = [ci[b][2] for b in order]
    lo_err = [ci[b][0] - ci[b][1] for b in order]
    hi_err = [ci[b][2] - ci[b][0] for b in order]
    palette = ["#6BAED6", "#74C476", "#FC9999", "#FDAE6B", "#B5A8D4"]
    colors = [palette[i % len(palette)] for i in range(len(order))]
    ax.bar(x, means, yerr=[lo_err, hi_err], capsize=4, color=colors, edgecolor="white", width=0.6, zorder=3)
    for xi, m, hi in zip(x, means, hi_ci):
        ax.text(xi, hi + 0.035, f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_DISPLAY_NAME.get(b, b) for b in order])
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
