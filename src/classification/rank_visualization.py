"""Analyse where the true class ranks among every classification-head output.

The random-ranking baseline over ``C`` classes is explicit:

- each exact rank has probability ``1 / C``;
- top-k accuracy is ``k / C``;
- mean rank is ``(C + 1) / 2``;
- mean reciprocal rank is ``sum(1/r for r=1..C) / C``.

Run example::

    python -m src.classification.rank_visualization \
      --inference path/to/inference-val.json \
      --output-dir path/to/rank-analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


_COLORS = {
    "observed": "#D55E00",
    "chance": "#4D4D4D",
    "category": "#0072B2",
    "interval": "#BDBDBD",
}


def true_class_rank(window: dict[str, Any], label_space: Sequence[str]) -> int:
    """Return 1 + the number of classes scored above the true class.

    This is the usual descending competition rank. Ties at the true probability share
    the same optimistic rank; floating-point softmax outputs normally have no ties.
    """
    probabilities = window.get("class_probabilities")
    if not isinstance(probabilities, dict):
        raise ValueError(
            "Inference has no class_probabilities; rerun inference_classification "
            "so that per-window class probabilities are written"
        )
    expected = set(label_space)
    actual = set(probabilities)
    if actual != expected:
        raise ValueError(
            "A per-window probability mapping does not match label_space: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    category = window["category"]
    true_probability = float(probabilities[category])
    return 1 + sum(float(probabilities[name]) > true_probability for name in label_space)


def rank_records(inference: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten one inference JSON into one true-rank record per window."""
    label_space = inference["label_space"]
    records = []
    for window in inference["per_window"]:
        rank = true_class_rank(window, label_space)
        records.append({
            "window_id": window["window_id"],
            "sequence_id": window["sequence_id"],
            "category": window["category"],
            "predicted_category": window["predicted_category"],
            "true_class_probability": float(
                window["class_probabilities"][window["category"]]
            ),
            "true_class_rank": rank,
            "reciprocal_rank": 1.0 / rank,
            "correct": bool(window["correct"]),
        })
    return records


def _binomial_tail_log10(n: int, observed: int, probability: float) -> float:
    """Exact log10 P[X >= observed] for X ~ Binomial(n, probability)."""
    if not 0 <= observed <= n:
        raise ValueError(f"observed must be between 0 and {n}, got {observed}")
    if probability == 0:
        return 0.0 if observed == 0 else -math.inf
    if probability == 1:
        return 0.0
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(value + 1)
        - math.lgamma(n - value + 1)
        + value * math.log(probability)
        + (n - value) * math.log1p(-probability)
        for value in range(observed, n + 1)
    ]
    maximum = max(terms)
    log_tail = maximum + math.log(sum(math.exp(term - maximum) for term in terms))
    return log_tail / math.log(10)


def _cluster_bootstrap(
    records: Sequence[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    """95% CIs from resampling whole sequences, never individual windows."""
    by_sequence: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_sequence[record["sequence_id"]].append(int(record["true_class_rank"]))
    groups = [np.asarray(values, dtype=np.int16) for values in by_sequence.values()]
    generator = np.random.default_rng(seed)
    estimates = {"mean_rank": [], "mrr": [], "top1": [], "top5": []}
    for _ in range(samples):
        choices = generator.integers(0, len(groups), size=len(groups))
        ranks = np.concatenate([groups[index] for index in choices])
        estimates["mean_rank"].append(float(ranks.mean()))
        estimates["mrr"].append(float((1.0 / ranks).mean()))
        estimates["top1"].append(float((ranks <= 1).mean()))
        estimates["top5"].append(float((ranks <= 5).mean()))
    return {
        name: [float(value) for value in np.quantile(values, [0.025, 0.975])]
        for name, values in estimates.items()
    }


def summarize_ranks(
    records: Sequence[dict[str, Any]],
    *,
    num_classes: int,
    bootstrap_samples: int = 10_000,
    seed: int = 20260821,
) -> dict[str, Any]:
    """Observed rank metrics, uniform-rank chance values, and sequence-aware checks."""
    if not records:
        raise ValueError("Cannot summarize an empty inference")
    ranks = np.asarray([record["true_class_rank"] for record in records], dtype=np.int16)
    top_k = {
        str(k): float((ranks <= k).mean())
        for k in range(1, num_classes + 1)
    }
    chance_top_k = {str(k): k / num_classes for k in range(1, num_classes + 1)}
    chance_mean_rank = (num_classes + 1) / 2
    chance_mrr = sum(1.0 / rank for rank in range(1, num_classes + 1)) / num_classes

    by_sequence: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_sequence[record["sequence_id"]].append(int(record["true_class_rank"]))
    sequence_means = np.asarray(
        [np.mean(values) for values in by_sequence.values()], dtype=np.float64
    )
    better = int((sequence_means < chance_mean_rank).sum())
    worse = int((sequence_means > chance_mean_rank).sum())
    tied = int((sequence_means == chance_mean_rank).sum())
    non_tied = better + worse

    top1_count = int((ranks <= 1).sum())
    top5_count = int((ranks <= min(5, num_classes)).sum())
    return {
        "windows": len(records),
        "sequences": len(by_sequence),
        "num_classes": num_classes,
        "rank_counts": {
            str(rank): int((ranks == rank).sum())
            for rank in range(1, num_classes + 1)
        },
        "mean_rank": float(ranks.mean()),
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
        "top_k_accuracy": top_k,
        "chance": {
            "rank_probability": 1.0 / num_classes,
            "mean_rank": chance_mean_rank,
            "median_rank": chance_mean_rank,
            "mrr": chance_mrr,
            "top_k_accuracy": chance_top_k,
        },
        "cluster_bootstrap_95_ci": _cluster_bootstrap(
            records, samples=bootstrap_samples, seed=seed
        ),
        "window_binomial_tests": {
            "note": "Exact random-ranking tails; windows within a sequence are correlated.",
            "top1_count": top1_count,
            "top1_log10_p": _binomial_tail_log10(
                len(ranks), top1_count, 1.0 / num_classes
            ),
            "top5_count": top5_count,
            "top5_log10_p": _binomial_tail_log10(
                len(ranks), top5_count, min(5, num_classes) / num_classes
            ),
        },
        "sequence_sign_test": {
            "better_than_chance_mean_rank": better,
            "worse_than_chance_mean_rank": worse,
            "ties_excluded": tied,
            "non_tied_sequences": non_tied,
            "one_sided_log10_p": _binomial_tail_log10(non_tied, better, 0.5),
        },
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "unit": "sequence"},
    }


def category_rank_rows(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-category rank metrics, sorted from lowest to highest mean rank."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for record in records:
        grouped[record["category"]].append(int(record["true_class_rank"]))
    rows = []
    for category, values in grouped.items():
        ranks = np.asarray(values)
        rows.append({
            "category": category,
            "windows": len(ranks),
            "mean_rank": float(ranks.mean()),
            "median_rank": float(np.median(ranks)),
            "mrr": float((1.0 / ranks).mean()),
            "top1_accuracy": float((ranks <= 1).mean()),
            "top5_accuracy": float((ranks <= 5).mean()),
        })
    return sorted(rows, key=lambda row: (row["mean_rank"], row["category"]))


def _apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_rank_dashboard(
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    category_rows: Sequence[dict[str, Any]],
    *,
    save_path: str | Path,
    split_label: str = "validation",
    experiment_label: str = "classification probe",
) -> None:
    """Four complementary rank views, each carrying its random-ranking baseline."""
    _apply_style()
    ranks = np.asarray([record["true_class_rank"] for record in records])
    classes = int(summary["num_classes"])
    window_count = len(ranks)
    independent_units = int(summary["sequences"])
    ks = np.arange(1, classes + 1)
    observed = np.asarray([summary["top_k_accuracy"][str(k)] for k in ks])
    chance = ks / classes
    # Conservative visual null: one independent draw per sequence, not per window.
    null_se = np.sqrt(chance * (1 - chance) / independent_units)

    figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.3))

    histogram = np.bincount(ranks, minlength=classes + 1)[1:] / window_count
    axes[0, 0].bar(ks, histogram, color=_COLORS["observed"], width=0.82)
    axes[0, 0].axhline(1 / classes, color=_COLORS["chance"], linestyle="--",
                       label=f"Chance: 1/{classes} = {1/classes:.3f}")
    axes[0, 0].set_title("Exact rank of the true class", fontweight="bold")
    axes[0, 0].set_xlabel("True-class rank (1 is best)")
    axes[0, 0].set_ylabel(f"Share of {split_label} windows")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].fill_between(
        ks,
        np.clip(chance - 1.96 * null_se, 0, 1),
        np.clip(chance + 1.96 * null_se, 0, 1),
        color=_COLORS["interval"], alpha=0.35,
        label="Sequence-level chance 95% band",
    )
    axes[0, 1].plot(ks, chance, color=_COLORS["chance"], linestyle="--", label="Chance k/26")
    axes[0, 1].plot(ks, observed, color=_COLORS["observed"], marker="o",
                    markersize=3, label="Observed")
    axes[0, 1].set_title("Cumulative top-k accuracy", fontweight="bold")
    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("P(true-class rank ≤ k)")
    axes[0, 1].set_ylim(0, 1.03)
    axes[0, 1].legend(frameon=False)

    lift = observed / chance
    axes[1, 0].plot(ks, lift, color=_COLORS["observed"], marker="o", markersize=3)
    axes[1, 0].axhline(1.0, color=_COLORS["chance"], linestyle="--", label="Chance")
    axes[1, 0].set_title("Top-k lift over random ranking", fontweight="bold")
    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Observed / chance")
    axes[1, 0].legend(frameon=False)

    shown = list(reversed(category_rows))
    names = [row["category"] for row in shown]
    values = [row["mean_rank"] for row in shown]
    axes[1, 1].barh(names, values, color=_COLORS["category"])
    axes[1, 1].axvline(
        summary["chance"]["mean_rank"], color=_COLORS["chance"], linestyle="--",
        label=f"Chance mean rank: {summary['chance']['mean_rank']:.1f}",
    )
    axes[1, 1].set_title("Mean true-class rank by category", fontweight="bold")
    axes[1, 1].set_xlabel("Mean rank (lower is better)")
    axes[1, 1].legend(frameon=False, loc="lower right")

    for axis in axes.flat:
        axis.grid(alpha=0.25, zorder=0)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    top1 = summary["top_k_accuracy"]["1"]
    top5 = summary["top_k_accuracy"][str(min(5, classes))]
    sign = summary["sequence_sign_test"]
    figure.suptitle(
        f"True-class rank on {split_label} — {experiment_label}\n"
        f"mean rank {summary['mean_rank']:.2f} vs {summary['chance']['mean_rank']:.1f} chance; "
        f"MRR {summary['mrr']:.3f} vs {summary['chance']['mrr']:.3f}; "
        f"top-1 {top1:.3f}, top-5 {top5:.3f}\n"
        f"{sign['better_than_chance_mean_rank']}/{sign['non_tied_sequences']} non-tied sequences "
        f"beat chance mean rank (one-sided sign test p=10^{sign['one_sided_log10_p']:.1f})",
        fontsize=11, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(save_path)
    plt.close(figure)


def plot_rank_histogram(
    records: Sequence[dict[str, Any]],
    num_classes: int,
    *,
    save_path: str | Path,
    split_label: str = "validation",
) -> None:
    """Standalone exact-rank distribution against uniform chance."""
    _apply_style()
    ranks = np.asarray([record["true_class_rank"] for record in records])
    ks = np.arange(1, num_classes + 1)
    shares = np.bincount(ranks, minlength=num_classes + 1)[1:] / len(ranks)
    figure, axes = plt.subplots(figsize=(8.0, 3.8))
    axes.bar(ks, shares, color=_COLORS["observed"], width=0.82, label="Observed")
    axes.axhline(1 / num_classes, color=_COLORS["chance"], linestyle="--",
                 label=f"Uniform chance = 1/{num_classes}")
    axes.set_xlabel("True-class rank (1 is best)")
    axes.set_ylabel(f"Share of {split_label} windows")
    axes.set_title("Distribution of the true class rank", fontweight="bold")
    axes.grid(axis="y", alpha=0.25)
    axes.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(save_path)
    plt.close(figure)


def plot_top_k(
    summary: dict[str, Any], *, save_path: str | Path
) -> None:
    """Standalone cumulative rank curve and random-ranking baseline."""
    _apply_style()
    classes = int(summary["num_classes"])
    independent_units = int(summary["sequences"])
    ks = np.arange(1, classes + 1)
    observed = np.asarray([summary["top_k_accuracy"][str(k)] for k in ks])
    chance = ks / classes
    # Conservative visual null: one independent draw per sequence, not per window.
    null_se = np.sqrt(chance * (1 - chance) / independent_units)
    figure, axes = plt.subplots(figsize=(7.2, 4.2))
    axes.fill_between(
        ks,
        np.clip(chance - 1.96 * null_se, 0, 1),
        np.clip(chance + 1.96 * null_se, 0, 1),
        color=_COLORS["interval"], alpha=0.4,
        label="Sequence-level chance 95% band",
    )
    axes.plot(ks, chance, color=_COLORS["chance"], linestyle="--", label="Chance k/26")
    axes.plot(ks, observed, color=_COLORS["observed"], marker="o", markersize=3,
              label="Observed")
    axes.set_xlabel("k")
    axes.set_ylabel("Top-k accuracy")
    axes.set_ylim(0, 1.03)
    axes.set_title("Is the true class ranked above chance?", fontweight="bold")
    axes.grid(alpha=0.25)
    axes.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(save_path)
    plt.close(figure)


def write_rank_analysis(
    inference_path: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260821,
) -> dict[str, Any]:
    """Compute, save, and plot a complete true-class-rank analysis."""
    inference_path = Path(inference_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    split_label = str(inference.get("split", "evaluation"))
    experiment_label = str(inference.get("experiment", "classification probe"))
    records = rank_records(inference)
    summary = summarize_ranks(
        records,
        num_classes=len(inference["label_space"]),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    summary.update({
        "experiment": inference.get("experiment"),
        "split": inference.get("split"),
        "inference_path": str(inference_path),
    })
    categories = category_rank_rows(records)

    with (output_dir / f"true-class-ranks-{split_label}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "true-class-rank-by-category.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(categories[0]))
        writer.writeheader()
        writer.writerows(categories)
    (output_dir / "true-class-rank-summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plot_rank_dashboard(
        records, summary, categories,
        save_path=output_dir / "true-class-rank-dashboard.png",
        split_label=split_label,
        experiment_label=experiment_label,
    )
    plot_rank_histogram(
        records, len(inference["label_space"]),
        save_path=output_dir / "true-class-rank-histogram.png",
        split_label=split_label,
    )
    plot_top_k(summary, save_path=output_dir / "top-k-vs-chance.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, required=True,
                        help="An inference-<split>.json containing class probabilities")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260821)
    arguments = parser.parse_args()
    summary = write_rank_analysis(
        arguments.inference,
        arguments.output_dir,
        bootstrap_samples=arguments.bootstrap_samples,
        seed=arguments.seed,
    )
    sign = summary["sequence_sign_test"]
    print(
        f"windows={summary['windows']} classes={summary['num_classes']} "
        f"mean-rank={summary['mean_rank']:.3f} "
        f"MRR={summary['mrr']:.3f} "
        f"top1={summary['top_k_accuracy']['1']:.3f} "
        f"top5={summary['top_k_accuracy']['5']:.3f} "
        f"sequence-sign-p=10^{sign['one_sided_log10_p']:.2f}"
    )


if __name__ == "__main__":
    main()
