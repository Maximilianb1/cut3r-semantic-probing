"""Build the training-curve and per-category-IoU plots for one or more runs,
straight from their already-computed metrics.json / inference-<split>.json --
no re-training, no re-inference. Companion to build_qualitative_plots.py
(worst/best-5 grids); this covers the other two chart kinds in figures.py.

Run example (after train_segmentation.py --checkpoint-selection best_val +
inference_segmentation.py have produced metrics.json and inference-test.json
for each backbone):

    python -m src.segmentation.analysis.build_curves_and_iou \
        --experiments-root src/segmentation/experiments \
        --backbones cut3r_trained cut3r_random dinov2 --run-suffix -expanded-bestval
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .figures import plot_per_category_iou, plot_training_curves
from .runs import DISPLAY_NAME, load_inference, load_metrics, resolve_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--run-suffix", default="", help="e.g. -expanded-bestval to match segmentation-<backbone>-expanded-bestval dirs")
    parser.add_argument("--split", default="test", help="which inference-<split>.json to read for the per-category chart")
    args = parser.parse_args()

    for backbone in args.backbones:
        exp_dir = resolve_run_dir(args.experiments_root, backbone, args.run_suffix)
        metrics = load_metrics(exp_dir)
        inference = load_inference(exp_dir, args.split)
        name = DISPLAY_NAME.get(backbone, backbone)

        curves_path = exp_dir / "training-curves.png"
        plot_training_curves(metrics["history"], title=f"{name} -- Training Curves", save_path=curves_path)
        print(f"[{backbone}] saved training curves    -> {curves_path}")

        category_path = exp_dir / f"per-category-iou-{args.split}.png"
        plot_per_category_iou(
            inference["metrics"]["per_category_iou"], method_name=name, top_k=None, save_path=category_path,
        )
        print(f"[{backbone}] saved per-category IoU    -> {category_path}")


if __name__ == "__main__":
    main()
