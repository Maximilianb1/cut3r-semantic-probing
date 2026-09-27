"""Finding 1 -- random-vs-trained gap, and trained-CUT3R ~= DINOv2.

macro-iou-ci-linear.png: sequence-cluster bootstrap 95% CI on macro-IoU
(macro over *windows*, not categories -- see bootstrap_iou.py), linear-probe
runs only (the cleanest comparison: no MLP capacity to potentially paper over
a weak backbone).

An earlier version of this script also plotted pooled AUROC/AUPRC bars next
to this one, as the threshold-free counterpart to IoU's threshold-dependent
view -- dropped because the bar chart looked like a near-duplicate of this
one (same 3-backbone ranking, no CI to distinguish it), and the numbers it
would show are already in the headline results table in
reports/segmentation/README.md. If AUROC/AUPRC need a figure of their own
later, Finding 4's real ROC/PR curves (blocked on saving per-token scores,
see reports/segmentation/README.md) are the more informative next step
rather than resurrecting this bar chart.

Reads straight from the committed reports/segmentation/<backbone>-<probe>/
{metrics.json,inference-test.json} -- no masks needed, no re-inference.

Run:
    python -m src.segmentation.analysis.build_headline_gap
"""

from __future__ import annotations

import argparse
from pathlib import Path

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

    run_dirs = {b: resolve_report_dir(args.reports_root, b, "linear") for b in args.backbones}
    inferences = {b: load_inference(run_dirs[b], args.split) for b in args.backbones}
    clusters = {b: iou_clusters(inferences[b]) for b in args.backbones}

    print(f"=== Finding 1: linear-probe macro-IoU, sequence-cluster bootstrap 95% CI ({args.split}) ===")
    ci = {}
    for b in args.backbones:
        seed = derive_seed(args.seed, f"headline-ci/{b}")
        result = bootstrap_iou_ci(clusters[b], samples=args.n_boot, seed=seed)
        mean, (lo, hi) = result["macro_iou"]["estimate"], result["macro_iou"]["ci95"]
        ci[b] = (mean, lo, hi)
        print(f"{DISPLAY_NAME.get(b, b):15s} macro_iou={mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
              f"({len(clusters[b].sequence_ids)} sequences)")

    order = sorted(args.backbones, key=lambda b: -ci[b][0])
    _apply_rc()
    import matplotlib.pyplot as plt

    x = np.arange(len(order))
    means = [ci[b][0] for b in order]
    hi_ci = [ci[b][2] for b in order]
    lo_err = [ci[b][0] - ci[b][1] for b in order]
    hi_err = [ci[b][2] - ci[b][0] for b in order]
    colors = [BACKBONE_COLOR.get(b, "#888") for b in order]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.bar(x, means, yerr=[lo_err, hi_err], capsize=4, color=colors, edgecolor="white", width=0.55, zorder=3)
    for xi, m, hi in zip(x, means, hi_ci):
        ax.text(xi, hi + 0.03, f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAME.get(b, b) for b in order])
    ax.set_ylabel(f"{args.split.capitalize()} macro-IoU")
    ax.set_ylim(0, 1.0)
    n_windows = inferences[order[0]]["windows"]
    ax.set_title(f"Linear probe: macro-IoU, 95% bootstrap CI ({n_windows} windows)",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    iou_output = output_dir / "macro-iou-ci-linear.png"
    iou_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(iou_output, dpi=200)
    print(f"Saved figure -> {iou_output}")


if __name__ == "__main__":
    main()
