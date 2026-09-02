"""Build the held-out test report for the classification probes.

Reads one per-window probability CSV per run (written by
``inference_classification.py`` as ``inference-<split>-probabilities.csv``) and
writes the tables and figures reported for Stage 2:

- ``model-test-metrics.csv``            one row per run, window- and sequence-level
- ``model-test-per-class-metrics.csv``  one row per run and category
- ``bootstrap-accuracy-ci.csv``         sequence-cluster bootstrap CIs
- ``bootstrap-accuracy-differences.csv`` paired and unpaired run differences
- ``test-bootstrap-report.json``        the full record, including the protocol
- four PNGs used in the report and the talk

The experimental unit is a complete CO3D sequence, never an overlapping window,
so every interval resamples sequence clusters. See ``bootstrap_accuracy.py``.

Run example::

    python -m src.classification.build_test_report \
      --predictions-dir reports/classification/predictions \
      --output-dir reports/classification \
      --seeds reports/classification/bootstrap-seeds.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl

# This module only writes files, and it runs on headless VMs. Choose the backend
# before pyplot is imported so no display is ever required.
mpl.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .bootstrap_accuracy import (
    SequenceAccuracyClusters,
    accuracy_clusters,
    bootstrap_accuracy_ci,
    bootstrap_accuracy_difference,
)

BOOTSTRAP_SAMPLES = 20_000
BASE_SEED = 20260825
# Floor on the true-class probability inside the log, so one confidently wrong
# window cannot dominate the negative log-likelihood.
_LOG_EPSILON = 1e-15

# Display names, and the fixed left-to-right order of the bar charts.
_DATASET_ORDER = ("cut3r-trained", "cut3r-random", "dinov2")
_MODEL_ORDER = ("linear", "mlp512")
_DATASET_TITLE = {
    "cut3r-trained": "CUT3R-trained",
    "cut3r-random": "CUT3R-random",
    "dinov2": "DINOv2",
}
_MODEL_TITLE = {"linear": "Linear Adam", "mlp512": "MLP-512 Adam"}
# Colourblind-safe, assigned per dataset in fixed order and never cycled.
_DATASET_COLOUR = {
    "cut3r-trained": "#0173B2",
    "cut3r-random": "#D55E00",
    "dinov2": "#029E73",
}
_DIFFERENCE_COLOUR = "#D55E00"

_RC_PARAMS: dict[str, Any] = {
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
}


# --------------------------------------------------------------------------- input


def read_predictions(path: str | Path) -> dict[str, Any]:
    """Rebuild the inference record ``bootstrap_accuracy`` consumes from one CSV."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No prediction rows in {path}")
    prefix = "probability_"
    labels = tuple(
        name[len(prefix):] for name in rows[0] if name.startswith(prefix)
    )
    if not labels:
        raise ValueError(f"{path} has no probability_<category> columns")
    per_window = [
        {
            "window_id": row["window_id"],
            "sequence_id": row["sequence_id"],
            "category": row["category"],
            "predicted_category": row["predicted_category"],
            "class_probabilities": {
                label: float(row[prefix + label]) for label in labels
            },
        }
        for row in rows
    ]
    return {"label_space": list(labels), "per_window": per_window}


def discover_runs(predictions_dir: str | Path) -> dict[str, Path]:
    """``<dataset>_<model>.csv`` becomes the run label ``<dataset>/<model>``."""
    root = Path(predictions_dir)
    found: dict[str, Path] = {}
    for path in sorted(root.glob("*.csv")):
        dataset, separator, model = path.stem.rpartition("_")
        if not separator:
            raise ValueError(f"Cannot read a dataset/model label from {path.name}")
        found[f"{dataset}/{model}"] = path
    if not found:
        raise FileNotFoundError(f"No prediction CSVs under {root}")
    return found


def _ordered_labels(labels: Iterable[str]) -> list[str]:
    """Dataset order first, then head order; anything unknown sorts last."""
    def key(label: str) -> tuple[int, int, str]:
        dataset, _, model = label.partition("/")
        dataset_rank = (
            _DATASET_ORDER.index(dataset) if dataset in _DATASET_ORDER else len(_DATASET_ORDER)
        )
        model_rank = _MODEL_ORDER.index(model) if model in _MODEL_ORDER else len(_MODEL_ORDER)
        return (dataset_rank, model_rank, label)

    return sorted(labels, key=key)


# ------------------------------------------------------------------------- metrics


def _macro_scores(
    targets: np.ndarray, predictions: np.ndarray, labels: Sequence[str]
) -> dict[str, Any]:
    """Per-category precision/recall/F1 and their unweighted means."""
    precision, recall, f1 = {}, {}, {}
    for index, name in enumerate(labels):
        predicted = predictions == index
        actual = targets == index
        hit = int((predicted & actual).sum())
        precision[name] = hit / int(predicted.sum()) if predicted.any() else 0.0
        recall[name] = hit / int(actual.sum()) if actual.any() else 0.0
        total = precision[name] + recall[name]
        f1[name] = 2 * precision[name] * recall[name] / total if total else 0.0
    support = {name: int((targets == index).sum()) for index, name in enumerate(labels)}
    return {
        "macro_precision": float(np.mean(list(precision.values()))),
        "macro_recall": float(np.mean(list(recall.values()))),
        "macro_f1": float(np.mean(list(f1.values()))),
        "per_category_precision": precision,
        "per_category_recall": recall,
        "per_category_f1": f1,
        "per_category_support": support,
    }


def _expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 10
) -> float:
    """Equal-width binned |accuracy - confidence|, weighted by bin occupancy."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Right-closed bins so a confidence of exactly 1.0 lands in the last one.
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for bin_index in range(bins):
        selected = index == bin_index
        if not selected.any():
            continue
        weight = selected.mean()
        error += weight * abs(correct[selected].mean() - confidence[selected].mean())
    return float(error)


def _score_block(
    probabilities: np.ndarray, targets: np.ndarray, labels: Sequence[str], top_k: int
) -> dict[str, Any]:
    """Accuracy, top-k, macro scores, calibration, and true-class rank statistics."""
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == targets).astype(np.float64)
    true_probability = probabilities[np.arange(len(targets)), targets]
    confidence = probabilities.max(axis=1)
    # Rank 1 means the true class scored highest; ties broken against the model.
    rank = 1 + (probabilities > true_probability[:, None]).sum(axis=1)
    ranked = np.argsort(-probabilities, axis=1)[:, :top_k]
    top_k_hit = (ranked == targets[:, None]).any(axis=1)

    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    block = {
        "accuracy": float(correct.mean()),
        f"top{top_k}_accuracy": float(top_k_hit.mean()),
    }
    block.update(_macro_scores(targets, predictions, labels))
    block.update({
        "nll": float(-np.log(np.clip(true_probability, _LOG_EPSILON, None)).mean()),
        "brier_score": float(((probabilities - one_hot) ** 2).sum(axis=1).mean()),
        "ece_10bin": _expected_calibration_error(confidence, correct),
        "mean_confidence": float(confidence.mean()),
        "mean_true_probability": float(true_probability.mean()),
        "mean_true_class_rank": float(rank.mean()),
        "median_true_class_rank": float(np.median(rank)),
        "mrr": float((1.0 / rank).mean()),
    })
    return block


def score_run(inference: dict[str, Any], top_k: int = 5) -> dict[str, Any]:
    """Window-level and sequence-level scores for one run.

    A sequence prediction is the argmax of its windows' mean probability vector,
    the same aggregation ``bootstrap_accuracy.accuracy_clusters`` uses.
    """
    labels = [str(name) for name in inference["label_space"]]
    index_of = {name: position for position, name in enumerate(labels)}

    window_probabilities = np.asarray([
        [window["class_probabilities"][name] for name in labels]
        for window in inference["per_window"]
    ])
    window_targets = np.asarray(
        [index_of[window["category"]] for window in inference["per_window"]]
    )

    grouped: dict[str, list[int]] = {}
    for position, window in enumerate(inference["per_window"]):
        grouped.setdefault(str(window["sequence_id"]), []).append(position)
    sequence_ids = sorted(grouped)
    sequence_probabilities = np.asarray(
        [window_probabilities[grouped[key]].mean(axis=0) for key in sequence_ids]
    )
    sequence_targets = np.asarray([window_targets[grouped[key][0]] for key in sequence_ids])

    return {
        "label_space": labels,
        "windows": int(len(window_targets)),
        "sequences": int(len(sequence_targets)),
        "window_metrics": _score_block(window_probabilities, window_targets, labels, top_k),
        "sequence_metrics": _score_block(
            sequence_probabilities, sequence_targets, labels, top_k
        ),
    }


# ------------------------------------------------------------------------ bootstrap


def comparison_pairs(labels: Sequence[str], accuracy: dict[str, float]) -> list[tuple[str, str]]:
    """Within-representation head comparisons first, then across-representation ones.

    Across representations the higher-accuracy run is always the target, so every
    reported difference reads as a positive improvement over its reference.
    """
    datasets = _ordered_labels(labels)
    by_dataset: dict[str, list[str]] = {}
    for label in datasets:
        by_dataset.setdefault(label.partition("/")[0], []).append(label)

    pairs: list[tuple[str, str]] = []
    for members in by_dataset.values():
        baseline, *rest = members
        pairs.extend((other, baseline) for other in rest)

    models = sorted(
        {label.partition("/")[2] for label in datasets},
        key=lambda name: _MODEL_ORDER.index(name) if name in _MODEL_ORDER else len(_MODEL_ORDER),
    )
    names = list(by_dataset)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            for model in models:
                candidates = [f"{names[left]}/{model}", f"{names[right]}/{model}"]
                if not all(name in accuracy for name in candidates):
                    continue
                target, reference = sorted(candidates, key=lambda name: -accuracy[name])
                pairs.append((target, reference))
    return pairs


def _seed_for(key: str, recorded: dict[str, int], sequence: Iterable[int]) -> int:
    """A recorded seed reproduces a published run exactly; otherwise derive one."""
    if key in recorded:
        return int(recorded[key])
    return int(next(iter(sequence)))


# -------------------------------------------------------------------------- figures


def plot_accuracy_bars(
    model: str,
    labels: Sequence[str],
    intervals: dict[str, dict[str, Any]],
    chance: float,
    save_path: str | Path,
) -> None:
    """One bar per representation for a single head, annotated with its 95% CI."""
    selected = [label for label in labels if label.partition("/")[2] == model]
    estimates = [intervals[label]["window_accuracy"]["estimate"] for label in selected]
    lower = [intervals[label]["window_accuracy"]["ci95"][0] for label in selected]
    upper = [intervals[label]["window_accuracy"]["ci95"][1] for label in selected]
    errors = np.vstack([
        np.asarray(estimates) - np.asarray(lower),
        np.asarray(upper) - np.asarray(estimates),
    ])

    figure, axes = plt.subplots(figsize=(8.0, 5.4))
    positions = np.arange(len(selected))
    colours = [_DATASET_COLOUR.get(label.partition("/")[0], "#666666") for label in selected]
    axes.bar(positions, estimates, yerr=errors, color=colours, width=0.62,
             capsize=6, error_kw={"elinewidth": 2.0, "capthick": 2.0, "ecolor": "#222222"})
    for position, estimate, low, high in zip(positions, estimates, lower, upper):
        axes.annotate(
            f"accuracy = {estimate:.3f}\n95% CI [{low:.3f}, {high:.3f}]",
            (position, high),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=10,
        )
    axes.axhline(chance, color="#555555", linestyle=":", linewidth=1.6)
    axes.annotate(f"chance = {chance:.4f}", (1.0, chance), xycoords=("axes fraction", "data"),
                  textcoords="offset points", xytext=(-4, 4), ha="right",
                  fontsize=9, color="#8B2E2E")
    axes.set_xticks(positions)
    axes.set_xticklabels([_DATASET_TITLE.get(label.partition("/")[0], label) for label in selected])
    axes.set_xlabel("Frozen representation")
    axes.set_ylabel("Test accuracy")
    axes.set_ylim(0.0, 1.15)
    axes.set_title(f"{_MODEL_TITLE.get(model, model)}: test accuracy with 95% sequence-bootstrap CI")
    axes.grid(axis="x", visible=False)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    figure.savefig(save_path)
    plt.close(figure)


def plot_all_intervals(
    labels: Sequence[str], intervals: dict[str, dict[str, Any]], save_path: str | Path
) -> None:
    """Every run's window accuracy on one axis, so the three tiers read at a glance."""
    figure, axes = plt.subplots(figsize=(9.0, 4.8))
    positions = np.arange(len(labels))
    estimates = [intervals[label]["window_accuracy"]["estimate"] for label in labels]
    errors = np.vstack([
        [estimate - intervals[label]["window_accuracy"]["ci95"][0]
         for label, estimate in zip(labels, estimates)],
        [intervals[label]["window_accuracy"]["ci95"][1] - estimate
         for label, estimate in zip(labels, estimates)],
    ])
    colours = [_DATASET_COLOUR.get(label.partition("/")[0], "#666666") for label in labels]
    axes.errorbar(positions, estimates, yerr=errors, fmt="none", ecolor="#222222",
                  elinewidth=1.6, capsize=5, capthick=1.6, zorder=1)
    axes.scatter(positions, estimates, c=colours, s=64, zorder=2)
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=20, ha="right")
    axes.set_ylabel("Test window accuracy")
    axes.set_title("Sequence-cluster bootstrap: window accuracy with 95% CI")
    axes.grid(axis="x", visible=False)
    figure.savefig(save_path)
    plt.close(figure)


def plot_differences(
    comparisons: list[dict[str, Any]], save_path: str | Path
) -> None:
    """Paired differences with 95% CIs; a CI crossing zero is not a difference."""
    figure, axes = plt.subplots(figsize=(10.0, 5.0))
    positions = np.arange(len(comparisons))
    estimates = [item["paired"]["window_accuracy_difference"]["estimate"] for item in comparisons]
    lower = [item["paired"]["window_accuracy_difference"]["ci95"][0] for item in comparisons]
    upper = [item["paired"]["window_accuracy_difference"]["ci95"][1] for item in comparisons]
    errors = np.vstack([
        np.asarray(estimates) - np.asarray(lower),
        np.asarray(upper) - np.asarray(estimates),
    ])
    axes.errorbar(estimates, positions, xerr=errors, fmt="o", color=_DIFFERENCE_COLOUR,
                  ecolor=_DIFFERENCE_COLOUR, elinewidth=1.8, capsize=5, capthick=1.8,
                  markersize=7)
    axes.axvline(0.0, color="#222222", linestyle="--", linewidth=1.4)
    axes.set_yticks(positions)
    axes.set_yticklabels([f"{item['target']} − {item['reference']}" for item in comparisons])
    axes.set_xlabel("Paired test-accuracy difference with 95% CI")
    axes.set_title("Paired sequence-cluster bootstrap")
    axes.grid(axis="y", visible=False)
    figure.savefig(save_path)
    plt.close(figure)


# ---------------------------------------------------------------------------- write


def _metrics_rows(scores: dict[str, dict[str, Any]], epochs: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for label, score in scores.items():
        dataset, _, model = label.partition("/")
        row: dict[str, Any] = {
            "label": label,
            "dataset": dataset,
            "model": model,
            "selected_epoch": epochs.get(label, ""),
            "test_windows": score["windows"],
            "test_sequences": score["sequences"],
        }
        for side in ("window", "sequence"):
            for key, value in score[f"{side}_metrics"].items():
                if key.startswith("per_category_"):
                    continue
                row[f"{side}_{key}"] = value
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    predictions: dict[str, Path],
    output_dir: str | Path,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BASE_SEED,
    recorded_seeds: dict[str, int] | None = None,
    epochs: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Score every run, bootstrap it, and write every table and figure."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recorded_seeds = recorded_seeds or {}
    epochs = epochs or {}
    mpl.rcParams.update(_RC_PARAMS)

    labels = _ordered_labels(predictions)
    inferences = {label: read_predictions(predictions[label]) for label in labels}
    clusters: dict[str, SequenceAccuracyClusters] = {
        label: accuracy_clusters(inference) for label, inference in inferences.items()
    }
    scores = {label: score_run(inferences[label]) for label in labels}

    # Every run must have been evaluated on the exact same test observations, or no
    # paired comparison below is meaningful. Fail here rather than report a difference.
    reference = clusters[labels[0]]
    for label in labels[1:]:
        try:
            from .bootstrap_accuracy import assert_paired_clusters

            assert_paired_clusters(clusters[label], reference)
        except ValueError as error:
            raise ValueError(f"{label} was not evaluated on the same test set: {error}") from error

    derived = iter(
        int(child.generate_state(1)[0]) for child in np.random.SeedSequence(seed).spawn(512)
    )
    intervals = {
        label: bootstrap_accuracy_ci(
            clusters[label],
            samples=samples,
            seed=_seed_for(f"ci/{label}", recorded_seeds, derived),
        )
        for label in labels
    }

    accuracy = {label: scores[label]["window_metrics"]["accuracy"] for label in labels}
    comparisons = []
    for target, ref in comparison_pairs(labels, accuracy):
        pair_seed = _seed_for(f"difference/{target}|{ref}", recorded_seeds, derived)
        comparisons.append({
            "target": target,
            "reference": ref,
            "seed": pair_seed,
            "paired": bootstrap_accuracy_difference(
                clusters[target], clusters[ref], samples=samples, seed=pair_seed, paired=True
            ),
            "unpaired": bootstrap_accuracy_difference(
                clusters[target], clusters[ref], samples=samples, seed=pair_seed, paired=False
            ),
        })

    chance = 1.0 / len(scores[labels[0]]["label_space"])
    report = {
        "protocol": {
            "split": "test",
            "clusters": int(len(reference.sequence_ids)),
            "windows": int(reference.window_total.sum()),
            "bootstrap_samples": samples,
            "base_seed": seed,
            "confidence_level": 0.95,
            "interval": "percentile",
            "resampling_unit": "complete CO3D sequence",
            "classical": "each model resampled by sequence; paired structure ignored for differences",
            "paired": "the same sampled sequence indices are reused for both models",
            "chance_accuracy": chance,
        },
        "runs": {
            label: {"predictions": predictions[label].as_posix(), **scores[label]}
            for label in labels
        },
        "bootstrap_accuracy": intervals,
        "comparisons": comparisons,
        "exact_observation_pairing_passed": True,
    }
    (output / "test-bootstrap-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    _write_csv(output / "model-test-metrics.csv", _metrics_rows(scores, epochs))

    per_class_rows = []
    for label in labels:
        block = scores[label]["window_metrics"]
        for category in scores[label]["label_space"]:
            per_class_rows.append({
                "label": label,
                "category": category,
                "support": block["per_category_support"][category],
                "precision": block["per_category_precision"][category],
                "recall": block["per_category_recall"][category],
                "f1": block["per_category_f1"][category],
            })
    _write_csv(output / "model-test-per-class-metrics.csv", per_class_rows)

    interval_rows = []
    for label in labels:
        dataset, _, model = label.partition("/")
        for unit in ("window_accuracy", "sequence_accuracy"):
            block = intervals[label][unit]
            interval_rows.append({
                "label": label,
                "dataset": dataset,
                "model": model,
                "unit": unit,
                "estimate": block["estimate"],
                "ci95_lower": block["ci95"][0],
                "ci95_upper": block["ci95"][1],
                "bootstrap_standard_error": block["bootstrap_standard_error"],
                "clusters": intervals[label]["clusters"],
                "samples": intervals[label]["samples"],
                "seed": intervals[label]["seed"],
            })
    _write_csv(output / "bootstrap-accuracy-ci.csv", interval_rows)

    difference_rows = []
    for item in comparisons:
        for unit in ("window_accuracy_difference", "sequence_accuracy_difference"):
            paired, unpaired = item["paired"][unit], item["unpaired"][unit]
            difference_rows.append({
                "target": item["target"],
                "reference": item["reference"],
                "unit": unit,
                "estimate": paired["estimate"],
                "unpaired_ci95_lower": unpaired["ci95"][0],
                "unpaired_ci95_upper": unpaired["ci95"][1],
                "unpaired_bootstrap_standard_error": unpaired["bootstrap_standard_error"],
                "paired_ci95_lower": paired["ci95"][0],
                "paired_ci95_upper": paired["ci95"][1],
                "paired_bootstrap_standard_error": paired["bootstrap_standard_error"],
                "clusters": item["paired"]["clusters"],
                "samples": item["paired"]["samples"],
                "seed": item["seed"],
            })
    _write_csv(output / "bootstrap-accuracy-differences.csv", difference_rows)

    for model in sorted({label.partition("/")[2] for label in labels}):
        plot_accuracy_bars(
            model, labels, intervals, chance, output / f"bootstrap-test-accuracy-{model}.png"
        )
    plot_all_intervals(labels, intervals, output / "bootstrap-window-accuracy-ci.png")
    plot_differences(comparisons, output / "paired-bootstrap-window-accuracy-differences.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True,
                        help="Directory of <dataset>_<model>.csv per-window probability files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BASE_SEED,
                        help="Base seed; per-run seeds are spawned from it in report order.")
    parser.add_argument("--seeds", type=Path, default=None,
                        help="JSON of recorded per-run seeds, to reproduce a published run exactly.")
    parser.add_argument("--epochs", type=Path, default=None,
                        help="JSON mapping run label to the selected training epoch.")
    arguments = parser.parse_args()

    recorded = json.loads(arguments.seeds.read_text(encoding="utf-8")) if arguments.seeds else {}
    epochs = json.loads(arguments.epochs.read_text(encoding="utf-8")) if arguments.epochs else {}
    report = build_report(
        discover_runs(arguments.predictions_dir),
        arguments.output_dir,
        samples=arguments.samples,
        seed=arguments.seed,
        recorded_seeds=recorded,
        epochs=epochs,
    )
    for label, block in report["bootstrap_accuracy"].items():
        window = block["window_accuracy"]
        print(
            f"{label:<24} accuracy {window['estimate']:.4f} "
            f"95% CI [{window['ci95'][0]:.4f}, {window['ci95'][1]:.4f}]"
        )


if __name__ == "__main__":
    main()
