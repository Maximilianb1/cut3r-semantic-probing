"""
Figures for the classification probe, built from what a run actually wrote.

Every plot reads `metrics.json` and `inference-<split>.json` out of the experiments
tree. Nothing is simulated: a figure covers exactly the runs found on disk, and a run
whose cache is marked synthetic is stamped SYNTHETIC so a smoke-test figure cannot be
mistaken for a result.

Run example:
    python -m src.classification.visualizations --experiments-dir src/classification/experiments
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# LaTeX-ready defaults, matching src/segmentation/analysis/figures.py.
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

# Assigned in fixed order, never cycled per figure.
_PALETTE = ("#6BAED6", "#FDAE6B", "#74C476", "#FC9999", "#B5A8D4")

# Marker + line style carry the second dimension (train vs val, or which arm), so
# colour is free to identify the compared run. A lone run per backbone is named by
# backbone; multiple selected runs from one backbone are named by model configuration.
_TRAIN_STYLE = {"marker": "o", "linestyle": "-"}
_VAL_STYLE = {"marker": "s", "linestyle": "--"}
# Line weight and marker cadence: markers every few epochs keep a long run legible
# instead of turning the line into a bead chain.
_LINE = {"linewidth": 1.8, "markersize": 3.5, "markeredgewidth": 0}


def _markevery(count: int) -> int:
    """Show ~8 markers whatever the epoch count, so 20 and 200 epochs both read."""
    return max(1, count // 8)

# One hue, light -> dark: a confusion matrix encodes magnitude, never identity.
# GnBu is the ramp the segmentation figures use for a target mask.
_SEQUENTIAL = "GnBu"

_CURVE_METRICS = (
    ("loss", "Cross-entropy loss"),
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("macro_recall", "Macro recall"),
)


def _apply_rc() -> None:
    mpl.rcParams.update(_RC_PARAMS)


# ---- loading ----

class Run:
    """One run's files: the training record and, if present, an evaluation."""

    def __init__(self, directory: Path, split: str = "test") -> None:
        self.directory = directory
        self.metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        inference_path = directory / f"inference-{split}.json"
        self.inference = (
            json.loads(inference_path.read_text(encoding="utf-8"))
            if inference_path.is_file() else None
        )

    @property
    def source(self) -> str:
        return self.metrics["features"]["source"]

    @property
    def experiment(self) -> str:
        return str(self.metrics.get("experiment") or self.directory.name)

    @property
    def backbone(self) -> str:
        return str(self.metrics.get("backbone") or self.metrics.get("experiment"))

    @property
    def synthetic(self) -> bool:
        metadata = (self.metrics.get("probe_cache") or {}).get("metadata") or {}
        return bool(metadata.get("synthetic"))

    @property
    def label(self) -> str:
        suffix = " (SYNTHETIC)" if self.synthetic else ""
        marker = "fullunion-resplit80-10-10-"
        model = self.experiment.split(marker, 1)[-1]
        return f"{self.backbone} / {model}{suffix}"

    def history(self, side: str, metric: str) -> tuple[list[int], list[float]]:
        epochs, values = [], []
        for record in self.metrics.get("history", []):
            if metric in record[side]:
                epochs.append(record["epoch"])
                values.append(record[side][metric])
        return epochs, values


def discover_runs(
    experiments_dir: str | Path,
    split: str = "test",
    experiments: Sequence[str] | None = None,
) -> list[Run]:
    """Every run under ``<experiments_dir>/<source>/<experiment>/``, sorted."""
    root = Path(experiments_dir)
    runs = [
        Run(directory, split)
        for directory in sorted(root.glob("*/*"))
        if (directory / "metrics.json").is_file()
    ]
    if not runs:
        raise FileNotFoundError(
            f"No runs with a metrics.json under {root}; train something first"
        )
    if experiments:
        requested = set(experiments)
        runs = [run for run in runs if run.experiment in requested]
        missing = sorted(requested - {run.experiment for run in runs})
        if missing:
            raise FileNotFoundError(
                f"Requested experiments not found under {root}: {missing}"
            )
    return runs


def _series_name(run: Run, runs: Sequence[Run]) -> str:
    """Use backbone unless several selected runs need model-level distinction."""
    duplicate_backbone = sum(other.backbone == run.backbone for other in runs) > 1
    return run.label if duplicate_backbone else run.backbone


def _stamp_synthetic(figure: plt.Figure, runs: Iterable[Run]) -> None:
    """Mark a figure whose data came from a synthetic cache. It is not a result."""
    if any(run.synthetic for run in runs):
        figure.text(
            0.99, 0.01, "SYNTHETIC DATA - pipeline check, not a result",
            ha="right", va="bottom", fontsize=7, color="#b3261e", alpha=0.9,
        )


# ---- 1. confusion matrix ----

def plot_confusion_matrix(run: Run, *, normalize: bool = False,
                          save_path: str | Path | None = None) -> plt.Figure:
    """Confusion over the categories present, one matrix per run.

    Rows are the true category, columns the prediction, so a row reads "where did this
    category's windows go" and its diagonal cell is recall. Restricted to categories
    that appear in this split: the head has 51 outputs, and plotting all of them for a
    cache holding eight would be mostly empty cells.

    The colour scale is **window counts** by default, so a cell's shade and its number
    say the same thing. Pass ``normalize=True`` to shade by the row's share instead,
    which is the fairer read when categories have very different support - a rare
    category's total failure is otherwise a pale cell next to a common category's
    ordinary one. The counts are annotated either way.
    """
    _apply_rc()
    if run.inference is None:
        raise ValueError(f"{run.directory} has no inference file to plot")
    counts = run.inference["confusion"]
    present = sorted(run.inference["class_counts"])
    position = {name: index for index, name in enumerate(present)}
    vocabulary = _vocabulary(run)

    matrix = np.zeros((len(present), len(present)))
    off_grid = 0
    for entry in counts:
        true_name = vocabulary[entry["true"]]
        predicted_name = vocabulary[entry["predicted"]]
        if true_name in position and predicted_name in position:
            matrix[position[true_name], position[predicted_name]] += entry["count"]
        else:
            off_grid += entry["count"]

    shown = matrix / matrix.sum(axis=1, keepdims=True) if normalize else matrix
    figure, axes = plt.subplots(figsize=(0.45 * len(present) + 3.0,
                                         0.45 * len(present) + 2.4))
    image = axes.imshow(shown, cmap=_SEQUENTIAL, vmin=0, vmax=1 if normalize else None)

    # Flip the label to white only on the dark end of whatever scale is in use; a fixed
    # threshold would be white-on-pale as soon as the scale is counts rather than 0-1.
    dark_from = 0.55 * (shown.max() or 1.0)
    for row in range(len(present)):
        for column in range(len(present)):
            value = matrix[row, column]
            if value:
                # Direct labels: the cell colour alone should not have to carry the number.
                axes.text(column, row, f"{int(value)}", ha="center", va="center",
                          fontsize=7,
                          color="white" if shown[row, column] > dark_from else "#1a1a19")

    axes.set_xticks(range(len(present)), present, rotation=45, ha="right")
    axes.set_yticks(range(len(present)), present)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.set_title(f"{run.label} - {run.source}", fontweight="bold", pad=8)
    if off_grid:
        # A note, not part of the title: a long title collides with the colour bar's.
        axes.set_xlabel(
            "Predicted\n"
            f"({off_grid} prediction(s) into categories absent from this split)"
        )
    bar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    # Title above the bar rather than a rotated side label: it reads horizontally.
    bar.ax.set_title("Share of the row" if normalize else "Windows count",
                     fontsize=8, pad=6)
    if not normalize:
        # Counts are integers; a colour bar reading 0.5 windows would be nonsense.
        bar.locator = mpl.ticker.MaxNLocator(integer=True)
        bar.update_ticks()
    _stamp_synthetic(figure, [run])
    figure.tight_layout()
    _save(figure, save_path)
    return figure


def _vocabulary(run: Run) -> list[str]:
    """Category names by output index, as the run's own head numbered them.

    The run's ``label_space`` first: a head trained with ``label_space: present`` emits
    indices into just the categories its cache holds, so reading them against the full
    51-category vocabulary would name every prediction wrongly.
    """
    names = run.metrics.get("label_space")
    if names:
        return list(names)
    metadata = (run.metrics.get("probe_cache") or {}).get("metadata") or {}
    names = metadata.get("category_vocabulary")
    if names:
        return list(names)
    from src.backbones.probe_cache import category_vocabulary  # local: keeps import light
    return category_vocabulary()


# ---- 2 & 3. curves ----

def plot_curves_by_arm(runs: Sequence[Run], arm: str, *,
                       save_path: str | Path | None = None) -> plt.Figure:
    """One arm's curves: a 2x2 of loss, accuracy, macro F1, macro recall.

    Colour identifies the compared run. Train is the faded solid line, val the marked
    dashed one, so the gap between them - the thing worth seeing on a frozen-feature
    probe - reads directly off each panel.
    """
    _apply_rc()
    selected = [run for run in runs if run.source == arm]
    if not selected:
        raise ValueError(f"No runs with features.source={arm!r}")
    names = sorted({_series_name(run, selected) for run in selected})
    colour = {name: _PALETTE[index % len(_PALETTE)] for index, name in enumerate(names)}

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))
    for panel, (metric, pretty) in zip(axes.flat, _CURVE_METRICS):
        for run in selected:
            for side, style, alpha in (("train", _TRAIN_STYLE, 0.35), ("val", _VAL_STYLE, 1.0)):
                epochs, values = run.history(side, metric)
                if epochs:
                    panel.plot(
                        epochs,
                        values,
                        color=colour[_series_name(run, selected)],
                        alpha=alpha,
                        markevery=_markevery(len(epochs)),
                        **_LINE,
                        **style,
                    )
        _finish_panel(panel, pretty, metric)

    figure.legend(handles=_curve_legend(colour, ("train (faded)", "val")),
                  loc="lower center", ncol=len(colour) + 2, frameon=False,
                  bbox_to_anchor=(0.5, -0.02))
    figure.suptitle(f"Training curves - {arm}", fontsize=11, fontweight="bold")
    _stamp_synthetic(figure, selected)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    _save(figure, save_path)
    return figure


def plot_curves_train_val(runs: Sequence[Run], arm: str, *,
                          save_path: str | Path | None = None) -> plt.Figure:
    """One arm's curves with train and val in adjacent columns, one row per metric.

    The overlaid version in :func:`plot_curves_by_arm` distinguishes the two splits by
    opacity, which is hard to read where several backbones sit at similar values.
    Here each row shares a y-axis, so a backbone's train and val points are directly
    comparable across the pair and the generalization gap is the horizontal offset
    between the same colour in the two columns.
    """
    _apply_rc()
    selected = [run for run in runs if run.source == arm]
    if not selected:
        raise ValueError(f"No runs with features.source={arm!r}")
    names = sorted({_series_name(run, selected) for run in selected})
    colour = {name: _PALETTE[index % len(_PALETTE)] for index, name in enumerate(names)}

    figure, axes = plt.subplots(len(_CURVE_METRICS), 2, figsize=(9.0, 11.0),
                                sharex=True, sharey="row")
    for row, (metric, pretty) in enumerate(_CURVE_METRICS):
        for column, side in enumerate(("train", "val")):
            panel = axes[row][column]
            for run in selected:
                epochs, values = run.history(side, metric)
                if epochs:
                    style = _TRAIN_STYLE if side == "train" else _VAL_STYLE
                    panel.plot(epochs, values, color=colour[_series_name(run, selected)],
                               markevery=_markevery(len(epochs)), **_LINE, **style)
            _finish_panel(panel, pretty, metric)
            # The row already names the metric; the panel title names the split.
            panel.set_title("Train" if side == "train" else "Validation",
                            fontweight="bold", fontsize=9)
            if column:
                panel.set_ylabel("")

    figure.legend(handles=[mpl.lines.Line2D([], [], color=value, marker="o", label=name,
                                            **_LINE)
                           for name, value in colour.items()],
                  loc="lower center", ncol=len(colour), frameon=False,
                  bbox_to_anchor=(0.5, -0.01))
    figure.suptitle(f"Train vs validation - {arm}", fontsize=11, fontweight="bold")
    _stamp_synthetic(figure, selected)
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    _save(figure, save_path)
    return figure


def plot_curves_merged(runs: Sequence[Run], *,
                       save_path: str | Path | None = None) -> plt.Figure:
    """Both arms overlaid, validation only: colour = run, line style = arm.

    Validation only on purpose. Adding train would double the lines and bury the one
    comparison this figure exists to make.
    """
    _apply_rc()
    names = sorted({_series_name(run, runs) for run in runs})
    colour = {name: _PALETTE[index % len(_PALETTE)] for index, name in enumerate(names)}

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))
    for panel, (metric, pretty) in zip(axes.flat, _CURVE_METRICS):
        for run in runs:
            epochs, values = run.history("val", metric)
            if epochs:
                style = _TRAIN_STYLE if run.source == "image_tokens" else _VAL_STYLE
                panel.plot(epochs, values, color=colour[_series_name(run, runs)],
                           markevery=_markevery(len(epochs)), **_LINE, **style)
        _finish_panel(panel, pretty, metric)

    figure.legend(handles=_curve_legend(colour, ("image_tokens", "state_tokens")),
                  loc="lower center", ncol=len(colour) + 2, frameon=False,
                  bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("Validation curves - both feature sources", fontsize=11, fontweight="bold")
    _stamp_synthetic(figure, runs)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    _save(figure, save_path)
    return figure


def _finish_panel(panel: plt.Axes, pretty: str, metric: str) -> None:
    panel.set_xlabel("Epoch")
    panel.set_ylabel(pretty)
    panel.set_title(pretty, fontweight="bold", fontsize=9)
    # Epochs are whole numbers; matplotlib's default locator invents 2.5 and 7.5.
    panel.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    if metric == "loss":
        panel.set_ylim(bottom=0)
    else:
        panel.set_ylim(0, 1.02)
        panel.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
        panel.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
        panel.grid(which="minor", linewidth=0.2, alpha=0.25, linestyle="--", zorder=0)
    panel.grid(which="major", linewidth=0.4, alpha=0.45, zorder=0)
    panel.set_axisbelow(True)  # grid behind the curves, not through them
    for side in ("top", "right"):
        panel.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        panel.spines[side].set_color("#b8b8b3")
    panel.tick_params(colors="#5b5b57", length=3)


def _curve_legend(colour: dict[str, str], style_labels: tuple[str, str]) -> list[Any]:
    """Colour keys the backbone; the two line styles key whatever the figure varies."""
    handles = [mpl.lines.Line2D([], [], color=value, marker="o", label=name, **_LINE)
               for name, value in colour.items()]
    handles.append(mpl.lines.Line2D([], [], color="#9a9a95", alpha=0.6,
                                    label=style_labels[0], **_LINE, **_TRAIN_STYLE))
    handles.append(mpl.lines.Line2D([], [], color="#9a9a95",
                                    label=style_labels[1], **_LINE, **_VAL_STYLE))
    return handles


# ---- 4. per-category bars ----

def plot_per_category(run: Run, *, save_path: str | Path | None = None) -> plt.Figure:
    """Per-category precision / recall / F1, sorted by recall.

    Note on naming: per-category **accuracy** in a multiclass setting is recall
    (TP / support). The one-vs-rest accuracy - which would also count the true
    negatives - sits near 1.0 for every category and says nothing, so the sort key and
    the bar are recall.

    Support is printed per category because these are small numbers: a recall computed
    over four windows moves in steps of 0.25, and the reader needs to see that.
    """
    _apply_rc()
    source = run.inference or run.metrics
    metrics = source["metrics"] if run.inference else source["final_val"]
    counts = run.inference["class_counts"] if run.inference else run.metrics.get(
        "train_class_counts", {})

    recall = metrics["per_category_recall"]
    order = sorted(recall, key=lambda name: (-recall[name], name))
    series = (
        ("Precision", metrics["per_category_precision"], _PALETTE[0]),
        ("Recall", metrics["per_category_recall"], _PALETTE[1]),
        ("F1", metrics["per_category_f1"], _PALETTE[2]),
    )

    positions = np.arange(len(order))
    width = 0.26
    figure, axes = plt.subplots(figsize=(max(6.0, 0.75 * len(order)), 3.6))
    for index, (name, values, color) in enumerate(series):
        offset = (index - 1) * width
        bars = axes.bar(positions + offset, [values[c] for c in order], width * 0.9,
                        label=name, color=color, edgecolor="white", linewidth=0.6,
                        zorder=3)
        for bar in bars:  # direct labels: aqua is under 3:1 on this surface
            axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                      f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=6)

    chance = 1.0 / len(order)
    axes.axhline(chance, color="#5b5b57", linewidth=1, linestyle=":", zorder=2)
    axes.text(len(order) - 0.4, chance + 0.02, f"chance {chance:.2f}", fontsize=7,
              ha="right", color="#5b5b57")

    axes.set_xticks(positions, [f"{c}\n(n={counts.get(c, 0)})" for c in order])
    axes.set_ylim(0, 1.12)
    axes.set_ylabel("Score")
    axes.set_title(f"Per-category scores - {run.label} / {run.source}"
                   f" ({run.inference['split'] if run.inference else 'val'})",
                   fontweight="bold", pad=8)
    axes.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    axes.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    axes.grid(axis="y", which="major", linewidth=0.4, alpha=0.5, zorder=0)
    axes.grid(axis="y", which="minor", linewidth=0.2, alpha=0.3, linestyle="--", zorder=0)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="#cccccc", fancybox=False)
    _stamp_synthetic(figure, [run])
    figure.tight_layout()
    _save(figure, save_path)
    return figure


# ---- extras ----

def plot_summary(runs: Sequence[Run], *, save_path: str | Path | None = None) -> plt.Figure:
    """Every run on one axis: the comparison table as a figure.

    Grouped by backbone, one bar cluster per arm, so the two questions - which backbone,
    which representation - are both readable without flipping between files.
    """
    _apply_rc()
    scored = [run for run in runs if run.inference]
    if not scored:
        raise ValueError("No inference files found; run inference before plotting a summary")
    labels = [f"{run.label}\n{run.source}" for run in scored]
    series = (("Accuracy", "accuracy", _PALETTE[0]),
              ("Macro F1", "macro_f1", _PALETTE[1]),
              ("Macro recall", "macro_recall", _PALETTE[2]))

    positions = np.arange(len(scored))
    width = 0.26
    figure, axes = plt.subplots(figsize=(max(6.0, 1.6 * len(scored)), 3.6))
    for index, (name, key, color) in enumerate(series):
        values = [run.inference["metrics"][key] for run in scored]
        bars = axes.bar(positions + (index - 1) * width, values, width * 0.9, label=name,
                        color=color, edgecolor="white", linewidth=0.6, zorder=3)
        for bar in bars:
            axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                      f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=6.5)

    chances = {1.0 / len(run.inference["class_counts"]) for run in scored}
    if len(chances) == 1:
        chance = chances.pop()
        axes.axhline(chance, color="#5b5b57", linewidth=1, linestyle=":", zorder=2)
        axes.text(len(scored) - 0.4, chance + 0.02, f"chance {chance:.2f}", fontsize=7,
                  ha="right", color="#5b5b57")

    axes.set_xticks(positions, labels)
    axes.set_ylim(0, 1.12)
    axes.set_ylabel("Score")
    axes.set_title("Classification probe - all runs", fontweight="bold", pad=8)
    axes.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    axes.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))
    axes.grid(axis="y", which="major", linewidth=0.4, alpha=0.5, zorder=0)
    axes.grid(axis="y", which="minor", linewidth=0.2, alpha=0.3, linestyle="--", zorder=0)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.legend(loc="upper right", frameon=True, framealpha=0.85, edgecolor="#cccccc", fancybox=False)
    _stamp_synthetic(figure, scored)
    figure.tight_layout()
    _save(figure, save_path)
    return figure


def plot_top_confusions(run: Run, *, top: int = 12,
                        save_path: str | Path | None = None) -> plt.Figure:
    """The most frequent mistakes, as "true -> predicted" bars.

    A matrix over 51 categories is unreadable; this answers the question a reader
    actually has - which pairs does it mix up - directly, and stays legible however many
    categories there are.
    """
    _apply_rc()
    if run.inference is None:
        raise ValueError(f"{run.directory} has no inference file to plot")
    vocabulary = _vocabulary(run)
    mistakes = [
        (f"{vocabulary[e['true']]} -> {vocabulary[e['predicted']]}", e["count"])
        for e in run.inference["confusion"] if e["true"] != e["predicted"]
    ]
    mistakes.sort(key=lambda pair: pair[1], reverse=True)
    mistakes = mistakes[:top]

    figure, axes = plt.subplots(figsize=(6.0, max(2.4, 0.32 * max(len(mistakes), 1))))
    if mistakes:
        names = [name for name, _ in mistakes][::-1]
        values = [count for _, count in mistakes][::-1]
        bars = axes.barh(names, values, color=_PALETTE[0], edgecolor="white",
                         linewidth=0.6, zorder=3)
        for bar, value in zip(bars, values):
            axes.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                      str(value), va="center", fontsize=7)
        axes.set_xlabel("Windows")
    else:
        axes.text(0.5, 0.5, "No mistakes on this split", ha="center", va="center")
        axes.set_axis_off()
    axes.set_title(f"Most frequent confusions - {run.label} / {run.source}",
                   fontweight="bold", pad=8)
    axes.grid(axis="x", linewidth=0.3, alpha=0.5, zorder=0)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    _stamp_synthetic(figure, [run])
    figure.tight_layout()
    _save(figure, save_path)
    return figure


def _save(figure: plt.Figure, save_path: str | Path | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path)


# ---- CLI ----

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate classification probe figures.")
    parser.add_argument("--experiments-dir", type=Path,
                        default=Path("src/classification/experiments"),
                        help="Root holding <features.source>/<experiment>/ run directories.")
    parser.add_argument("--split", default="test", help="Which inference-<split>.json to read.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write figures (default: <experiments-dir>/figures).")
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Exact experiment name to include; repeat to select multiple runs.",
    )
    arguments = parser.parse_args()

    runs = discover_runs(
        arguments.experiments_dir,
        arguments.split,
        experiments=arguments.experiment or None,
    )
    out = arguments.output_dir or arguments.experiments_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(runs)} run(s) found under {arguments.experiments_dir}")

    for arm in sorted({run.source for run in runs}):
        plot_curves_by_arm(runs, arm, save_path=out / f"curves-{arm}.png")
        print(f"  curves {arm}")
        plot_curves_train_val(runs, arm, save_path=out / f"curves-train-val-{arm}.png")
        print(f"  curves train/val {arm}")
    plot_curves_merged(runs, save_path=out / "curves-merged.png")
    print("  curves merged")

    scored = [run for run in runs if run.inference]
    if scored:
        plot_summary(scored, save_path=out / "summary.png")
        print(f"  summary            -> {out / 'summary.png'}")
    for run in runs:
        stem = f"{run.source}-{run.experiment}".replace("/", "-")
        plot_per_category(run, save_path=out / f"per_category-{stem}.png")
        if run.inference:
            plot_confusion_matrix(run, save_path=out / f"confusion-{stem}.png")
            plot_top_confusions(run, save_path=out / f"top_confusions-{stem}.png")
        print(f"  {stem}")
    if any(run.synthetic for run in runs):
        print("NOTE: some runs used a SYNTHETIC cache; those figures are stamped as such.")
    plt.close("all")


if __name__ == "__main__":
    main()
