"""Build the held-out test report for the segmentation probes.

Mirrors ``src/classification/build_test_report.py``: reads each run's
already-computed ``reports/segmentation/<backbone>-<probe>/inference-<split>.json``
(never re-trains, never re-runs inference) and writes:

- ``bootstrap-iou-ci.csv``                    sequence-cluster bootstrap CI on macro-IoU, one row per run
- ``bootstrap-iou-per-category.csv``          the same CI restricted to one category's sequences
- ``bootstrap-iou-differences.csv``           paired and unpaired run differences, plus window win/loss/tie
- ``bootstrap-precision-recall-ci.csv``       sequence-cluster bootstrap CI on pooled precision/recall
- ``bootstrap-precision-recall-differences.csv``  paired and unpaired precision/recall differences
- ``test-bootstrap-report.json``              the full record, including the protocol

The experimental unit is a complete CO3D sequence, never an overlapping
window, so every interval resamples sequence clusters. See ``bootstrap_iou.py``
and ``bootstrap_precision_recall.py``.

Run example::

    python -m src.segmentation.build_test_report \
      --reports-root reports/segmentation \
      --output-dir reports/segmentation/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .analysis.runs import load_inference, resolve_report_dir
from .bootstrap_iou import (
    SequenceIoUClusters,
    assert_paired_clusters,
    bootstrap_iou_ci,
    bootstrap_iou_difference,
    derive_seed,
    iou_clusters,
)
from .bootstrap_precision_recall import (
    SequencePrecisionRecallClusters,
    assert_paired_clusters as assert_paired_precision_recall_clusters,
    bootstrap_precision_recall_ci,
    bootstrap_precision_recall_difference,
    precision_recall_clusters,
)

BOOTSTRAP_SAMPLES = 20_000
BASE_SEED = 20260825
MIN_CATEGORY_SEQUENCES = 5

_BACKBONE_ORDER = ("cut3r_trained", "cut3r_random", "dinov2")
_PROBE_ORDER = ("linear", "mlp")


# --------------------------------------------------------------------------- input


def discover_runs(reports_root: str | Path, split: str) -> dict[str, Path]:
    """``<backbone>/<probe>`` -> its run directory, for every combination
    that has a committed ``inference-<split>.json``."""
    reports_root = Path(reports_root)
    found: dict[str, Path] = {}
    for backbone in _BACKBONE_ORDER:
        for probe in _PROBE_ORDER:
            run_dir = resolve_report_dir(reports_root, backbone, probe)
            if (run_dir / f"inference-{split}.json").is_file():
                found[f"{backbone}/{probe}"] = run_dir
    if not found:
        raise FileNotFoundError(f"No inference-{split}.json found under {reports_root}")
    return found


def _ordered_labels(labels: Sequence[str]) -> list[str]:
    """Backbone order first, then probe order; anything unknown sorts last."""
    def key(label: str) -> tuple[int, int, str]:
        backbone, _, probe = label.partition("/")
        backbone_rank = (
            _BACKBONE_ORDER.index(backbone) if backbone in _BACKBONE_ORDER else len(_BACKBONE_ORDER)
        )
        probe_rank = _PROBE_ORDER.index(probe) if probe in _PROBE_ORDER else len(_PROBE_ORDER)
        return (backbone_rank, probe_rank, label)

    return sorted(labels, key=key)


# ------------------------------------------------------------------------ bootstrap


def comparison_pairs(labels: Sequence[str], macro_iou: dict[str, float]) -> list[tuple[str, str]]:
    """Within-backbone probe-capacity comparisons first, then across-backbone
    ones at matching capacity.

    Across backbones the higher-IoU run is always the target, so every
    reported difference reads as a positive gain over its reference.
    """
    by_backbone: dict[str, list[str]] = {}
    for label in labels:
        by_backbone.setdefault(label.partition("/")[0], []).append(label)

    pairs: list[tuple[str, str]] = []
    for members in by_backbone.values():
        baseline, *rest = members  # already probe-ordered: linear first
        pairs.extend((other, baseline) for other in rest)

    names = list(by_backbone)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            for probe in _PROBE_ORDER:
                candidates = [f"{names[left]}/{probe}", f"{names[right]}/{probe}"]
                if not all(name in macro_iou for name in candidates):
                    continue
                target, reference = sorted(candidates, key=lambda name: -macro_iou[name])
                pairs.append((target, reference))
    return pairs


def window_win_loss_tie(
    target_windows: dict[str, float], reference_windows: dict[str, float]
) -> dict[str, int]:
    """Tally how many shared test windows each run wins, on per-window IoU."""
    shared = sorted(set(target_windows) & set(reference_windows))
    wins = losses = ties = 0
    for window_id in shared:
        target_iou, reference_iou = target_windows[window_id], reference_windows[window_id]
        if target_iou > reference_iou:
            wins += 1
        elif target_iou < reference_iou:
            losses += 1
        else:
            ties += 1
    return {
        "windows_compared": len(shared),
        "target_wins": wins,
        "reference_wins": losses,
        "ties": ties,
    }


def _validate_shared_observations(
    clusters: dict[str, SequenceIoUClusters],
    pr_clusters: dict[str, SequencePrecisionRecallClusters],
    labels: Sequence[str],
) -> None:
    """Every run must have been evaluated on the exact same test observations,
    or no paired comparison below is meaningful. Fail here rather than report
    a difference."""
    iou_reference = clusters[labels[0]]
    pr_reference = pr_clusters[labels[0]]
    for label in labels[1:]:
        try:
            assert_paired_clusters(clusters[label], iou_reference)
            assert_paired_precision_recall_clusters(pr_clusters[label], pr_reference)
        except ValueError as error:
            raise ValueError(f"{label} was not evaluated on the same test set: {error}") from error


def _precision_recall_comparison(
    pr_clusters: dict[str, SequencePrecisionRecallClusters], target: str, reference: str, *,
    samples: int, seed: int,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "paired": bootstrap_precision_recall_difference(
            pr_clusters[target], pr_clusters[reference], samples=samples, seed=seed, paired=True
        ),
        "unpaired": bootstrap_precision_recall_difference(
            pr_clusters[target], pr_clusters[reference], samples=samples, seed=seed, paired=False
        ),
    }


def _filter_by_category(clusters: SequenceIoUClusters, category: str) -> SequenceIoUClusters:
    """The sequence-cluster subset belonging to one category, for a
    per-category bootstrap CI. A sequence belongs to exactly one category
    (``iou_clusters`` enforces this), so this is a pure subset, not a re-pool."""
    indices = [position for position, name in enumerate(clusters.categories) if name == category]
    return SequenceIoUClusters(
        sequence_ids=tuple(clusters.sequence_ids[i] for i in indices),
        categories=tuple(clusters.categories[i] for i in indices),
        window_ids=tuple(clusters.window_ids[i] for i in indices),
        sequence_iou=clusters.sequence_iou[indices],
    )


# ---------------------------------------------------------------------------- write


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _precision_recall_ci_rows(
    labels: Sequence[str], intervals: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One row per run per metric (precision, recall) -- long format, so the
    CSV stays a fixed shape regardless of which metrics are reported."""
    rows = []
    for label in labels:
        backbone, _, probe = label.partition("/")
        block = intervals[label]
        for metric in ("precision", "recall"):
            summary = block[metric]
            rows.append({
                "label": label,
                "backbone": backbone,
                "probe": probe,
                "metric": metric,
                "estimate": summary["estimate"],
                "ci95_lower": summary["ci95"][0],
                "ci95_upper": summary["ci95"][1],
                "bootstrap_standard_error": summary["bootstrap_standard_error"],
                "clusters": block["clusters"],
                "samples": block["samples"],
                "seed": block["seed"],
            })
    return rows


def _precision_recall_difference_rows(comparisons: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per comparison pair per metric (precision, recall)."""
    rows = []
    for item in comparisons:
        pr = item["precision_recall"]
        paired, unpaired = pr["paired"], pr["unpaired"]
        for metric in ("precision", "recall"):
            paired_block = paired[f"{metric}_difference"]
            unpaired_block = unpaired[f"{metric}_difference"]
            rows.append({
                "target": item["target"],
                "reference": item["reference"],
                "metric": metric,
                "estimate": paired_block["estimate"],
                "paired_ci95_lower": paired_block["ci95"][0],
                "paired_ci95_upper": paired_block["ci95"][1],
                "paired_bootstrap_standard_error": paired_block["bootstrap_standard_error"],
                "unpaired_ci95_lower": unpaired_block["ci95"][0],
                "unpaired_ci95_upper": unpaired_block["ci95"][1],
                "unpaired_bootstrap_standard_error": unpaired_block["bootstrap_standard_error"],
                "clusters": paired["clusters"],
                "samples": paired["samples"],
                "seed": pr["seed"],
            })
    return rows


def build_report(
    reports_root: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BASE_SEED,
    min_category_sequences: int = MIN_CATEGORY_SEQUENCES,
) -> dict[str, Any]:
    """Load every committed run, bootstrap it, and write every table."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_runs(reports_root, split)
    labels = _ordered_labels(run_dirs)
    inferences = {label: load_inference(run_dirs[label], split) for label in labels}
    clusters = {label: iou_clusters(inferences[label]) for label in labels}
    pr_clusters = {label: precision_recall_clusters(inferences[label]) for label in labels}
    window_iou = {
        label: {
            str(row["window_id"]): float(row["foreground_iou"])
            for row in inferences[label]["per_window_iou"]
        }
        for label in labels
    }

    _validate_shared_observations(clusters, pr_clusters, labels)
    reference = clusters[labels[0]]

    intervals = {
        label: bootstrap_iou_ci(
            clusters[label], samples=samples, seed=derive_seed(seed, f"ci/{label}")
        )
        for label in labels
    }
    pr_intervals = {
        label: bootstrap_precision_recall_ci(
            pr_clusters[label], samples=samples, seed=derive_seed(seed, f"pr-ci/{label}")
        )
        for label in labels
    }

    macro_iou = {label: clusters[label].macro_iou for label in labels}
    comparisons = []
    for target, ref in comparison_pairs(labels, macro_iou):
        pair_seed = derive_seed(seed, f"difference/{target}|{ref}")
        comparisons.append({
            "target": target,
            "reference": ref,
            "seed": pair_seed,
            "paired": bootstrap_iou_difference(
                clusters[target], clusters[ref], samples=samples, seed=pair_seed, paired=True
            ),
            "unpaired": bootstrap_iou_difference(
                clusters[target], clusters[ref], samples=samples, seed=pair_seed, paired=False
            ),
            "window_tally": window_win_loss_tie(window_iou[target], window_iou[ref]),
            "precision_recall": _precision_recall_comparison(
                pr_clusters, target, ref, samples=samples,
                seed=derive_seed(seed, f"pr-difference/{target}|{ref}"),
            ),
        })

    categories = sorted(set(reference.categories))
    per_category: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        rows = []
        for category in categories:
            filtered = _filter_by_category(clusters[label], category)
            if len(filtered.sequence_ids) < min_category_sequences:
                continue
            result = bootstrap_iou_ci(
                filtered, samples=samples, seed=derive_seed(seed, f"category/{label}/{category}")
            )
            rows.append({
                "category": category,
                "sequences": len(filtered.sequence_ids),
                **result["macro_iou"],
            })
        per_category[label] = rows

    report = {
        "protocol": {
            "split": split,
            "clusters": len(reference.sequence_ids),
            "windows": sum(len(ids) for ids in reference.window_ids),
            "bootstrap_samples": samples,
            "base_seed": seed,
            "confidence_level": 0.95,
            "interval": "percentile",
            "resampling_unit": "complete CO3D sequence",
            "paired": "the same sampled sequence indices are reused for both runs",
            "classical": "each run resampled independently; paired structure ignored for differences",
            "min_category_sequences": min_category_sequences,
        },
        "runs": {
            label: {
                "run_dir": run_dirs[label].as_posix(),
                "macro_iou": macro_iou[label],
                "clusters": len(clusters[label].sequence_ids),
            }
            for label in labels
        },
        "bootstrap_iou": intervals,
        "bootstrap_precision_recall": pr_intervals,
        "comparisons": comparisons,
        "per_category": per_category,
        "exact_observation_pairing_passed": True,
    }
    (output / "test-bootstrap-report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    ci_rows = []
    for label in labels:
        backbone, _, probe = label.partition("/")
        block = intervals[label]["macro_iou"]
        ci_rows.append({
            "label": label,
            "backbone": backbone,
            "probe": probe,
            "estimate": block["estimate"],
            "ci95_lower": block["ci95"][0],
            "ci95_upper": block["ci95"][1],
            "bootstrap_standard_error": block["bootstrap_standard_error"],
            "clusters": intervals[label]["clusters"],
            "samples": intervals[label]["samples"],
            "seed": intervals[label]["seed"],
        })
    _write_csv(output / "bootstrap-iou-ci.csv", ci_rows)

    category_rows = []
    for label in labels:
        backbone, _, probe = label.partition("/")
        for row in per_category[label]:
            category_rows.append({
                "label": label,
                "backbone": backbone,
                "probe": probe,
                "category": row["category"],
                "sequences": row["sequences"],
                "estimate": row["estimate"],
                "ci95_lower": row["ci95"][0],
                "ci95_upper": row["ci95"][1],
                "bootstrap_standard_error": row["bootstrap_standard_error"],
            })
    _write_csv(output / "bootstrap-iou-per-category.csv", category_rows)

    difference_rows = []
    for item in comparisons:
        paired = item["paired"]["macro_iou_difference"]
        unpaired = item["unpaired"]["macro_iou_difference"]
        tally = item["window_tally"]
        difference_rows.append({
            "target": item["target"],
            "reference": item["reference"],
            "estimate": paired["estimate"],
            "paired_ci95_lower": paired["ci95"][0],
            "paired_ci95_upper": paired["ci95"][1],
            "paired_bootstrap_standard_error": paired["bootstrap_standard_error"],
            "unpaired_ci95_lower": unpaired["ci95"][0],
            "unpaired_ci95_upper": unpaired["ci95"][1],
            "unpaired_bootstrap_standard_error": unpaired["bootstrap_standard_error"],
            "windows_compared": tally["windows_compared"],
            "target_wins": tally["target_wins"],
            "reference_wins": tally["reference_wins"],
            "ties": tally["ties"],
            "clusters": item["paired"]["clusters"],
            "samples": item["paired"]["samples"],
            "seed": item["seed"],
        })
    _write_csv(output / "bootstrap-iou-differences.csv", difference_rows)

    _write_csv(output / "bootstrap-precision-recall-ci.csv", _precision_recall_ci_rows(labels, pr_intervals))
    _write_csv(
        output / "bootstrap-precision-recall-differences.csv",
        _precision_recall_difference_rows(comparisons),
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("reports/segmentation"))
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="default: <reports-root>/comparison")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BASE_SEED,
                         help="Base seed; every run/comparison/category seed is derived from it.")
    parser.add_argument("--min-category-sequences", type=int, default=MIN_CATEGORY_SEQUENCES,
                         help="Skip a category's bootstrap CI below this many test sequences.")
    arguments = parser.parse_args()

    output_dir = arguments.output_dir or (arguments.reports_root / "comparison")
    report = build_report(
        arguments.reports_root,
        output_dir,
        split=arguments.split,
        samples=arguments.samples,
        seed=arguments.seed,
        min_category_sequences=arguments.min_category_sequences,
    )
    for label, block in report["bootstrap_iou"].items():
        macro = block["macro_iou"]
        pr = report["bootstrap_precision_recall"][label]
        print(
            f"{label:<20} macro_iou {macro['estimate']:.4f}  "
            f"95% CI [{macro['ci95'][0]:.4f}, {macro['ci95'][1]:.4f}]  "
            f"precision {pr['precision']['estimate']:.4f}  recall {pr['recall']['estimate']:.4f}"
        )
    for item in report["comparisons"]:
        difference = item["paired"]["macro_iou_difference"]
        tally = item["window_tally"]
        pr_difference = item["precision_recall"]["paired"]
        print(
            f"{item['target']} - {item['reference']:<20} "
            f"diff {difference['estimate']:+.4f} "
            f"95% CI [{difference['ci95'][0]:+.4f}, {difference['ci95'][1]:+.4f}]  "
            f"wins {tally['target_wins']}/{tally['windows_compared']}  "
            f"precision_diff {pr_difference['precision_difference']['estimate']:+.4f}  "
            f"recall_diff {pr_difference['recall_difference']['estimate']:+.4f}"
        )


if __name__ == "__main__":
    main()
