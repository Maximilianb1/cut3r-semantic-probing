"""Shared run-loading helpers and backbone display names/colors for the
segmentation analysis scripts -- one place so a figure's backbone-to-color or
backbone-to-name mapping can't drift between scripts that plot the same
three backbones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ..bootstrap_iou import SequenceIoUClusters, iou_clusters

DISPLAY_NAME = {
    "cut3r_trained": "CUT3R-trained",
    "cut3r_random": "CUT3R-random",
    "dinov2": "DINOv2",
}

BACKBONE_COLOR = {
    "cut3r_trained": "#74C476",
    "cut3r_random": "#FC9999",
    "dinov2": "#6BAED6",
}


def resolve_run_dir(experiments_root: Path, backbone: str, run_suffix: str = "") -> Path:
    return experiments_root / f"segmentation-{backbone}{run_suffix}"


def resolve_report_dir(reports_root: Path, backbone: str, probe: str) -> Path:
    """Maps a backbone id (``cut3r_trained``, underscored) + probe (``linear``/``mlp``)
    to its promoted run directory under ``reports/segmentation/`` (hyphenated,
    no ``segmentation-`` prefix -- distinct from the gitignored
    ``resolve_run_dir`` working-directory layout, since only metrics.json /
    inference-<split>.json are committed here, never masks)."""
    return reports_root / f"{backbone.replace('_', '-')}-{probe}"


def load_metrics(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def load_inference(run_dir: Path, split: str = "test") -> dict[str, Any]:
    return json.loads((run_dir / f"inference-{split}.json").read_text(encoding="utf-8"))


def load_per_window_iou(run_dir: Path, split: str) -> dict[str, dict[str, Any]]:
    """Maps window_id -> its per_window_iou row, from one run's inference-<split>.json."""
    return {row["window_id"]: row for row in load_inference(run_dir, split)["per_window_iou"]}


def load_masks(run_dir: Path, split: str = "test") -> dict[str, Any]:
    return torch.load(run_dir / f"masks-{split}.pt", weights_only=True)


def filtered_iou_clusters(run_dir: Path, split: str, window_ids: set[str]) -> SequenceIoUClusters:
    """Sequence-cluster IoU stats built from only ``window_ids`` (typically the
    windows common to every run being compared), so a paired bootstrap sees
    the identical test windows on both sides."""
    inference = load_inference(run_dir, split)
    rows = [row for row in inference["per_window_iou"] if row["window_id"] in window_ids]
    return iou_clusters({"per_window_iou": rows})
