"""
Binary segmentation visualizations.
Run with: python -m src.segmentation.visualizations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# -- LaTeX-ready matplotlib defaults --

_RC_PARAMS: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}

# Stage 1 backbones
METHODS = ("CUT3R Trained", "CUT3R Random", "DINOv2")

# Scalar metrics for comparison
METRICS = (
    "token_accuracy",
    "macro_foreground_iou",
    "micro_foreground_iou",
    "mean_category_iou",
)

# Legend labels
_METRIC_LABELS: dict[str, str] = {
    "token_accuracy": "Token Acc.",
    "macro_foreground_iou": "Macro FG-IoU",
    "micro_foreground_iou": "Micro FG-IoU",
    "mean_category_iou": "Mean Cat. IoU",
}


def _apply_rc() -> None:
    mpl.rcParams.update(_RC_PARAMS)


# ---- 1. Qualitative grid ----

def plot_segmentation_results(
    frames: np.ndarray,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    *,
    max_rows: int = 4,
    overlay_alpha: float = 0.45,
    overlay_color: tuple[float, float, float] = (0.40, 0.70, 0.95),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots a 4-column grid: Input | Ground Truth | Prediction | Overlay."""
    _apply_rc()

    frames = _to_float_image(frames)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    pred_masks = np.asarray(pred_masks, dtype=np.float32)

    n = min(len(frames), max_rows)
    col_titles = ["Input Frame", "Ground Truth", "Prediction", "Overlay"]
    fig, axes = plt.subplots(
        n, 4, figsize=(7.0, 1.7 * n + 0.4),
        gridspec_kw={"wspace": 0.04, "hspace": 0.15},
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    for row in range(n):
        img, gt, pred = frames[row], gt_masks[row], pred_masks[row]

        axes[row, 0].imshow(img)
        axes[row, 1].imshow(gt, cmap="GnBu", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 2].imshow(pred, cmap="OrRd", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 3].imshow(_blend_overlay(img, pred, alpha=overlay_alpha, color=overlay_color))

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontweight="bold", pad=4)

    fig.suptitle("Binary Segmentation -- Qualitative Results", fontsize=11, fontweight="bold", y=1.01)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def _to_float_image(images: np.ndarray) -> np.ndarray:
    """Converts image to float32 [0, 1]."""
    images = np.asarray(images)
    if images.dtype == np.uint8:
        return images.astype(np.float32) / 255.0
    return np.clip(images.astype(np.float32), 0.0, 1.0)


def _blend_overlay(image, mask, *, alpha=0.35, color=(0.40, 0.70, 0.95)):
    """Overlays a mask on an RGB image."""
    overlay = image.copy()
    tint = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    fg = mask[..., np.newaxis] > 0.5
    overlay = np.where(fg, overlay * (1 - alpha) + tint * alpha, overlay)
    return np.clip(overlay, 0.0, 1.0)


# ---- 2. Baseline comparison bar chart ----

def plot_segmentation_baselines(
    results_dict: dict[str, dict[str, float]],
    *,
    metrics: Sequence[str] = METRICS,
    title: str = "Binary Segmentation -- Method Comparison",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots a grouped bar chart of metric scores per method."""
    _apply_rc()

    methods = list(results_dict.keys())
    n_methods, n_metrics = len(methods), len(metrics)

    palette = ["#6BAED6", "#FDAE6B", "#74C476", "#FC9999", "#B5A8D4"]
    bar_colors = [palette[i % len(palette)] for i in range(n_metrics)]

    x = np.arange(n_methods)
    bar_width = 0.7 / max(n_metrics, 1)

    fig, ax = plt.subplots(figsize=(max(5.5, 1.1 * n_methods * n_metrics), 3.5))

    for idx, metric in enumerate(metrics):
        values = [results_dict[m][metric] for m in methods]
        label = _METRIC_LABELS.get(metric, metric)
        offset = (idx - (n_metrics - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, width=bar_width * 0.88,
            label=label, color=bar_colors[idx],
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight="medium")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontweight="bold", pad=8)

    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    ax.grid(axis="y", which="major", linewidth=0.4, alpha=0.5, zorder=0)
    ax.grid(axis="y", which="minor", linewidth=0.2, alpha=0.3, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="#cccccc", fancybox=False)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


# ---- 3. Per-category IoU ----

def plot_per_category_iou(
    per_category_iou: dict[str, float],
    *,
    method_name: str = "CUT3R Trained",
    top_k: int | None = 15,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots a horizontal bar chart of per-category IoU scores."""
    _apply_rc()

    sorted_cats = sorted(per_category_iou.items(), key=lambda kv: kv[1])
    if top_k is not None and len(sorted_cats) > 2 * top_k:
        sorted_cats = sorted_cats[:top_k] + sorted_cats[-top_k:]

    names = [c for c, _ in sorted_cats]
    values = [v for _, v in sorted_cats]

    fig, ax = plt.subplots(figsize=(5.0, max(3.0, 0.28 * len(names))))

    cmap = mpl.colormaps["RdYlGn"]
    colors = [cmap(v) for v in values]
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.5, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=7)

    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Foreground IoU")
    ax.set_title(f"Per-Category IoU -- {method_name}", fontweight="bold", pad=8)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    ax.grid(axis="x", linewidth=0.3, alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


# ---- 4. Training curves ----

def plot_training_curves(
    history: list[dict[str, Any]],
    *,
    title: str = "Training Curves",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots training vs validation curves for Loss and IoU."""
    _apply_rc()

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train"]["loss"] for h in history]
    val_loss = [h["val"]["loss"] for h in history]
    train_iou = [h["train"]["macro_foreground_iou"] for h in history]
    val_iou = [h["val"]["macro_foreground_iou"] for h in history]

    fig, (ax_loss, ax_iou) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # Loss
    ax_loss.plot(epochs, train_loss, "o-", markersize=3, label="Train", color="#6BAED6")
    ax_loss.plot(epochs, val_loss, "s--", markersize=3, label="Val", color="#FDAE6B")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE Loss")
    ax_loss.set_title("Loss", fontweight="bold")
    ax_loss.legend(frameon=True, framealpha=0.8, edgecolor="#ccc", fancybox=False)
    ax_loss.grid(linewidth=0.3, alpha=0.5)
    ax_loss.spines["top"].set_visible(False)
    ax_loss.spines["right"].set_visible(False)

    # Macro Foreground IoU
    ax_iou.plot(epochs, train_iou, "o-", markersize=3, label="Train", color="#6BAED6")
    ax_iou.plot(epochs, val_iou, "s--", markersize=3, label="Val", color="#FDAE6B")
    ax_iou.set_xlabel("Epoch")
    ax_iou.set_ylabel("Macro FG-IoU")
    ax_iou.set_title("Foreground IoU", fontweight="bold")
    ax_iou.set_ylim(0, 1.05)
    ax_iou.legend(frameon=True, framealpha=0.8, edgecolor="#ccc", fancybox=False)
    ax_iou.grid(linewidth=0.3, alpha=0.5)
    ax_iou.spines["top"].set_visible(False)
    ax_iou.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


# ---- Fixture data ----

def _get_baselines(metrics_data) -> dict[str, dict[str, float]]:
    """Returns baseline metrics for bar charts."""
    return {
        # Placeholder -- replace when experiment finishes.
        "CUT3R Trained": {
            "token_accuracy":       0.92,
            "macro_foreground_iou": 0.73,
            "micro_foreground_iou": 0.76,
            "mean_category_iou":    0.70,
        },
        # Real data from metrics.json final_val.
        "CUT3R Random": {
            k: metrics_data["final_val"][k]
            for k in METRICS
        },
        # Placeholder -- replace when experiment finishes.
        "DINOv2": {
            "token_accuracy":       0.86,
            "macro_foreground_iou": 0.64,
            "micro_foreground_iou": 0.67,
            "mean_category_iou":    0.60,
        },
    }


def _generate_mock_frames(*, n_samples=4, height=224, width=224, rng):
    """Generates synthetic frames and masks for the qualitative grid."""
    frames = np.empty((n_samples, height, width, 3), dtype=np.float32)
    gt_masks = np.zeros((n_samples, height, width), dtype=np.float32)
    pred_masks = np.zeros((n_samples, height, width), dtype=np.float32)

    yy, xx = np.mgrid[:height, :width]

    for i in range(n_samples):
        base = rng.uniform(0.15, 0.85, size=3).astype(np.float32)
        frames[i] = np.clip(base + rng.uniform(-0.08, 0.08, (height, width, 3)).astype(np.float32), 0, 1)

        cy, cx = rng.integers(height // 4, 3 * height // 4), rng.integers(width // 4, 3 * width // 4)
        ry, rx = rng.integers(height // 6, height // 3), rng.integers(width // 6, width // 3)
        gt_masks[i] = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0).astype(np.float32)

        fg_color = rng.uniform(0.4, 1.0, size=3).astype(np.float32)
        fg = gt_masks[i][..., np.newaxis] > 0.5
        frames[i] = np.where(fg, frames[i] * 0.3 + fg_color * 0.7, frames[i])

        pred_masks[i] = np.where(rng.random((height, width)) < 0.08, 1.0 - gt_masks[i], gt_masks[i])

    return frames, gt_masks, pred_masks


# ---- CLI entry point ----

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate segmentation plots.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/visualizations"),
                        help="Output directory (default: artifacts/visualizations).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    fixtures_dir = Path(__file__).parent / "fixtures"
    
    with open(fixtures_dir / "metrics.json") as f:
        metrics_data = json.load(f)
        
    with open(fixtures_dir / "inference-test.json") as f:
        inference_data = json.load(f)

    # 1. Qualitative grid (synthetic frames)
    frames, gt_masks, pred_masks = _generate_mock_frames(rng=rng)
    seg_path = out / "segmentation_results.png"
    plot_segmentation_results(frames, gt_masks, pred_masks, save_path=seg_path)
    print(f"Saved qualitative grid       -> {seg_path}")

    # 2. Baseline comparison (real CUT3R Random, placeholder others)
    baselines = _get_baselines(metrics_data)
    bar_path = out / "baseline_comparison.png"
    plot_segmentation_baselines(baselines, save_path=bar_path)
    print(f"Saved baseline bar chart     -> {bar_path}")

    # 3. Per-category IoU (real test data from inference-test.json)
    cat_path = out / "per_category_iou.png"
    plot_per_category_iou(
        inference_data["metrics"]["per_category_iou"],
        method_name="CUT3R Random",
        save_path=cat_path,
    )
    print(f"Saved per-category IoU chart -> {cat_path}")

    # 4. Training curves (real 20-epoch history from metrics.json)
    curve_path = out / "training_curves.png"
    plot_training_curves(metrics_data["history"], title="CUT3R Random -- Training Curves", save_path=curve_path)
    print(f"Saved training curves        -> {curve_path}")

    plt.close("all")


if __name__ == "__main__":
    main()
