"""
Shared plot_* figure-rendering functions for the segmentation analysis scripts.
"""

from __future__ import annotations

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


# ---- 1b. Two-backbone comparison grid (same windows, two predictions) ----

def plot_backbone_comparison_grid(
    frames: np.ndarray,
    gt_masks: np.ndarray,
    pred_masks_a: np.ndarray,
    pred_masks_b: np.ndarray,
    captions: Sequence[str],
    *,
    name_a: str = "A",
    name_b: str = "B",
    max_rows: int = 10,
    title: str = "Backbone Comparison -- Largest Per-Window Deltas",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plots a 4-column grid: Input | Ground Truth | <name_a> Pred | <name_b> Pred.

    One row per window, for windows chosen elsewhere (e.g. the largest
    per-window IoU deltas between two backbones) -- this only renders what
    it's given, it does not rank or select windows itself.
    """
    _apply_rc()

    frames = _to_float_image(frames)
    gt_masks = np.asarray(gt_masks, dtype=np.float32)
    pred_masks_a = np.asarray(pred_masks_a, dtype=np.float32)
    pred_masks_b = np.asarray(pred_masks_b, dtype=np.float32)

    n = min(len(frames), max_rows)
    col_titles = ["Input Frame", "Ground Truth", f"{name_a}\nPrediction", f"{name_b}\nPrediction"]
    fig, axes = plt.subplots(
        n, 4, figsize=(9.0, 2.0 * n + 0.6),
        gridspec_kw={"wspace": 0.10, "hspace": 0.35},
    )
    if n == 1:
        axes = axes[np.newaxis, :]

    for row in range(n):
        img, gt, pred_a, pred_b = frames[row], gt_masks[row], pred_masks_a[row], pred_masks_b[row]

        axes[row, 0].imshow(img)
        axes[row, 1].imshow(gt, cmap="GnBu", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 2].imshow(pred_a, cmap="OrRd", vmin=0, vmax=1, interpolation="nearest")
        axes[row, 3].imshow(pred_b, cmap="OrRd", vmin=0, vmax=1, interpolation="nearest")

        axes[row, 0].set_ylabel(captions[row], fontsize=7, rotation=0, ha="right", va="center", labelpad=6)
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    for col, col_title in enumerate(col_titles):
        axes[0, col].set_title(col_title, fontweight="bold", fontsize=9, pad=4)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

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
