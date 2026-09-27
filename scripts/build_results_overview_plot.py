"""
Results overview figure for the report: test scores of every backbone and head.

Two panels share one encoding (backbones on the x-axis, linear vs MLP head as
paired bars): (a) segmentation foreground macro-IoU, (b) classification accuracy
from state tokens (CUT3R state / DINOv2 CLS). Error bars are 95% sequence-cluster
bootstrap CIs, read with the estimates from the bootstrap CSVs in ``reports/``.

Run example:
    python scripts/build_results_overview_plot.py --output reports/results-overview.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

BACKBONES = ("cut3r_trained", "dinov2", "cut3r_random")
BACKBONE_NAMES = {
    "cut3r_trained": "CUT3R\ntrained",
    "dinov2": "DINOv2\nViT-B/14",
    "cut3r_random": "CUT3R\nrandom",
}
HEADS = ("linear", "mlp")
HEAD_NAMES = {"linear": "Linear probe", "mlp": "MLP probe"}
HEAD_COLORS = {"linear": "#4C72B0", "mlp": "#DD8452"}

CHANCE_ACCURACY = 1 / 26

Score = Tuple[float, float, float]  # estimate, CI lower, CI upper


def read_segmentation(path: Path) -> dict[tuple[str, str], Score]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["backbone"], row["probe"]): (
                float(row["estimate"]), float(row["ci95_lower"]), float(row["ci95_upper"])
            )
            for row in csv.DictReader(handle)
        }


def read_state_accuracy(path: Path) -> dict[tuple[str, str], Score]:
    scores = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["unit"] != "window_accuracy":
                continue
            backbone = row["dataset"].replace("-", "_")
            head = "mlp" if row["model"].startswith("mlp") else row["model"]
            scores[(backbone, head)] = (
                float(row["estimate"]), float(row["ci95_lower"]), float(row["ci95_upper"])
            )
    return scores


def _panel(axis, scores: dict[tuple[str, str], Score], title: str, ylabel: str,
           chance: float | None = None) -> None:
    positions = np.arange(len(BACKBONES))
    width = 0.36
    for offset, head in zip((-width / 2, width / 2), HEADS):
        values = [scores[(backbone, head)] for backbone in BACKBONES]
        heights = np.array([value[0] for value in values])
        errors = np.array([[h - v[1] for h, v in zip(heights, values)],
                           [v[2] - h for h, v in zip(heights, values)]])
        axis.bar(positions + offset, heights, width * 0.92, color=HEAD_COLORS[head],
                 label=HEAD_NAMES[head], zorder=2)
        axis.errorbar(positions + offset, heights, yerr=errors, fmt="none",
                      ecolor="#222222", elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=3)
        for x, height, value in zip(positions + offset, heights, values):
            axis.text(x, value[2] + 0.015, f"{height:.2f}", ha="center", va="bottom",
                      fontsize=6, color="#222222")
    if chance is not None:
        axis.axhline(chance, color="#666666", linestyle=(0, (3, 2)), linewidth=0.7, zorder=1,
                     label="Chance (1/26)")
    axis.set_xticks(positions, [BACKBONE_NAMES[backbone] for backbone in BACKBONES])
    axis.tick_params(axis="x", length=0)
    axis.set_xlim(-0.6, len(BACKBONES) - 0.4)
    axis.set_ylim(0, 1.05)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.yaxis.grid(True, color="#E5E5E5", linewidth=0.5, zorder=0)
    axis.set_axisbelow(True)
    axis.set_title(title, loc="left")
    axis.set_ylabel(ylabel)


def build_figure(segmentation_csv: Path, classification_csv: Path) -> plt.Figure:
    mpl.rcParams.update(_RC_PARAMS)
    segmentation = read_segmentation(segmentation_csv)
    state = read_state_accuracy(classification_csv)

    figure, axes = plt.subplots(1, 2, figsize=(5.0, 2.3), sharey=True,
                                gridspec_kw={"wspace": 0.12})
    _panel(axes[0], segmentation, "(a) Segmentation", "Macro-IoU  /  accuracy")
    _panel(axes[1], state, "(b) Classification, state tokens", "", chance=CHANCE_ACCURACY)
    handles, labels = axes[1].get_legend_handles_labels()
    handles, labels = handles[1:] + handles[:1], labels[1:] + labels[:1]  # bars, then chance
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
                  bbox_to_anchor=(0.5, 1.08), handlelength=1.2, columnspacing=1.5)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the report's results overview figure.")
    parser.add_argument("--segmentation-csv", type=Path,
                        default=Path("reports/segmentation/comparison/bootstrap-iou-ci.csv"))
    parser.add_argument("--classification-csv", type=Path,
                        default=Path("reports/classification/bootstrap-accuracy-ci.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/results-overview.png"))
    arguments = parser.parse_args()

    figure = build_figure(arguments.segmentation_csv, arguments.classification_csv)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output)
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
