"""Overlay validation learning curves for all six backbone x head-capacity
combinations (CUT3R-trained / CUT3R-random / DINOv2, each linear and MLP) in a
single figure, straight from already-computed metrics.json -- no re-training,
no re-inference.

Distinguishes representation quality (which backbone's curve sits highest)
from probe capacity/overfitting (how far apart a backbone's MLP and linear
curves are, and how fast each converges) in one place, instead of reading six
separate per-run training-curves.png figures side by side.

Run example (after train_segmentation.py --checkpoint-selection best_val has
produced metrics.json for each of the 6 <backbone>_<capacity> configs):

    python -m src.segmentation.analysis.build_unified_learning_curves \
        --experiments-root src/segmentation/experiments \
        --backbones cut3r-trained cut3r-random dinov2 \
        --mlp-run-suffix=-mlp --linear-run-suffix=-linear
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .figures import plot_unified_learning_curves
from .runs import BACKBONE_COLOR, DISPLAY_NAME, load_metrics, resolve_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--mlp-run-suffix", default="-mlp",
                         help="run_suffix for the [512] MLP runs (segmentation-<backbone><suffix>)")
    parser.add_argument("--linear-run-suffix", default="-linear",
                         help="run_suffix for the linear (hidden_dims: []) runs")
    parser.add_argument("--metric", default="macro_foreground_iou",
                         help="Which val metric (a key under history[i]['val']) to plot")
    parser.add_argument("--output", type=Path, default=None,
                         help="default: <experiments-root>/../unified-learning-curves.png")
    args = parser.parse_args()

    histories: dict[str, list] = {}
    colors: dict[str, str] = {}
    linestyles: dict[str, str] = {}
    for backbone in args.backbones:
        name = DISPLAY_NAME.get(backbone, backbone)
        color = BACKBONE_COLOR.get(backbone, "#888")
        for capacity, suffix, linestyle in (
            ("MLP", args.mlp_run_suffix, "-"),
            ("Linear", args.linear_run_suffix, "--"),
        ):
            run_dir = resolve_run_dir(args.experiments_root, backbone, suffix)
            label = f"{name} ({capacity})"
            histories[label] = load_metrics(run_dir)["history"]
            colors[label] = color
            linestyles[label] = linestyle

    output = args.output or (args.experiments_root.parent / "unified-learning-curves.png")
    plot_unified_learning_curves(
        histories,
        metric=args.metric,
        colors=colors,
        linestyles=linestyles,
        ylabel=f"Val {args.metric.replace('_', ' ')}",
        title="Validation Learning Curves -- All 6 Backbone x Head-Capacity Combinations",
        save_path=output,
    )
    print(f"Saved figure -> {output}")


if __name__ == "__main__":
    main()
