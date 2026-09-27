"""Tests for the segmentation held-out test report builder."""

from __future__ import annotations

import json

import pytest

from src.segmentation.build_test_report import (
    build_report,
    comparison_pairs,
    discover_runs,
    window_win_loss_tie,
)

CATEGORIES = ("apple", "bench", "cup", "kite")


def _write_inference(run_dir, rows, *, split="test"):
    """One inference-<split>.json in the format inference_segmentation.py writes."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"split": split, "windows": len(rows), "metrics": {}, "per_window_iou": rows}
    (run_dir / f"inference-{split}.json").write_text(json.dumps(payload), encoding="utf-8")


def _confusion_counts(iou, *, tp=100, tn=200):
    """tp/fp/fn that reproduce ``iou`` under ``iou = tp / (tp + fp + fn)``,
    splitting the error evenly between false positives and false negatives."""
    if iou <= 0:
        return {"tp": 0, "fp": tp, "fn": tp, "tn": tn}
    error = tp * (1 / iou - 1)
    fp = fn = round(error / 2)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _rows(iou_pattern, *, sequences=6, windows=3, window_prefix="w"):
    """Windows across ``sequences`` sequences, categories cycling through CATEGORIES."""
    rows = []
    for sequence in range(sequences):
        category = CATEGORIES[sequence % len(CATEGORIES)]
        for window in range(windows):
            iou = iou_pattern[(sequence * windows + window) % len(iou_pattern)]
            rows.append({
                "window_id": f"{window_prefix}-{sequence}-{window}",
                "sequence_id": f"seq-{sequence}",
                "category": category,
                "foreground_iou": iou,
                **_confusion_counts(iou),
            })
    return rows


def test_discover_runs_finds_every_committed_run(tmp_path):
    _write_inference(tmp_path / "cut3r-trained-linear", _rows([0.5]))
    _write_inference(tmp_path / "cut3r-random-mlp", _rows([0.2]))

    found = discover_runs(tmp_path, split="test")

    assert set(found) == {"cut3r_trained/linear", "cut3r_random/mlp"}


def test_discover_runs_ignores_a_backbone_without_the_requested_split(tmp_path):
    _write_inference(tmp_path / "cut3r-trained-linear", _rows([0.5]), split="test")

    assert discover_runs(tmp_path, split="test")
    with pytest.raises(FileNotFoundError):
        discover_runs(tmp_path, split="val")


def test_comparison_pairs_orders_probes_within_a_backbone_then_across_backbones():
    labels = [
        "cut3r_trained/linear",
        "cut3r_trained/mlp",
        "cut3r_random/linear",
        "cut3r_random/mlp",
    ]
    macro_iou = {
        "cut3r_trained/linear": 0.73,
        "cut3r_trained/mlp": 0.79,
        "cut3r_random/linear": 0.19,
        "cut3r_random/mlp": 0.29,
    }

    assert comparison_pairs(labels, macro_iou) == [
        ("cut3r_trained/mlp", "cut3r_trained/linear"),
        ("cut3r_random/mlp", "cut3r_random/linear"),
        # The stronger backbone is always the target, so the difference is positive.
        ("cut3r_trained/linear", "cut3r_random/linear"),
        ("cut3r_trained/mlp", "cut3r_random/mlp"),
    ]


def test_window_win_loss_tie_counts_correctly():
    target = {"w0": 0.8, "w1": 0.3, "w2": 0.5, "w3": 0.1}
    reference = {"w0": 0.6, "w1": 0.3, "w2": 0.9, "w3": 0.1}

    tally = window_win_loss_tie(target, reference)

    assert tally == {
        "windows_compared": 4,
        "target_wins": 1,
        "reference_wins": 1,
        "ties": 2,
    }


def test_build_report_writes_every_table(tmp_path):
    reports_root = tmp_path / "reports"
    _write_inference(
        reports_root / "cut3r-trained-linear", _rows([0.9, 0.8, 0.3], sequences=8)
    )
    _write_inference(
        reports_root / "cut3r-trained-mlp", _rows([0.95, 0.85, 0.4], sequences=8)
    )

    output = tmp_path / "comparison"
    report = build_report(
        reports_root, output, samples=200, seed=7, min_category_sequences=2
    )

    for name in (
        "bootstrap-iou-ci.csv",
        "bootstrap-iou-per-category.csv",
        "bootstrap-iou-differences.csv",
        "bootstrap-precision-recall-ci.csv",
        "bootstrap-precision-recall-differences.csv",
        "test-bootstrap-report.json",
    ):
        assert (output / name).is_file(), name

    written = json.loads((output / "test-bootstrap-report.json").read_text(encoding="utf-8"))
    assert written["protocol"]["clusters"] == 8
    assert len(report["comparisons"]) == 1

    interval = report["bootstrap_iou"]["cut3r_trained/linear"]["macro_iou"]
    assert interval["ci95"][0] <= interval["estimate"] <= interval["ci95"][1]

    pr_interval = report["bootstrap_precision_recall"]["cut3r_trained/linear"]
    for metric in ("precision", "recall"):
        block = pr_interval[metric]
        assert block["ci95"][0] <= block["estimate"] <= block["ci95"][1]

    comparison = report["comparisons"][0]
    assert comparison["target"] == "cut3r_trained/mlp"
    assert comparison["reference"] == "cut3r_trained/linear"
    assert comparison["window_tally"]["windows_compared"] == 24
    assert "precision_difference" in comparison["precision_recall"]["paired"]
    assert "recall_difference" in comparison["precision_recall"]["paired"]

    # Every category (2 sequences each, at min_category_sequences=2) gets a row.
    assert len(report["per_category"]["cut3r_trained/linear"]) == len(CATEGORIES)


def test_build_report_refuses_runs_evaluated_on_different_test_sets(tmp_path):
    reports_root = tmp_path / "reports"
    _write_inference(reports_root / "cut3r-trained-linear", _rows([0.5], sequences=6))
    _write_inference(reports_root / "cut3r-trained-mlp", _rows([0.5], sequences=5))

    with pytest.raises(ValueError, match="not evaluated on the same test set"):
        build_report(reports_root, tmp_path / "comparison", samples=50)


def test_build_report_is_deterministic_for_a_fixed_seed(tmp_path):
    reports_root = tmp_path / "reports"
    _write_inference(
        reports_root / "cut3r-trained-linear", _rows([0.9, 0.8, 0.3], sequences=8)
    )
    _write_inference(
        reports_root / "cut3r-trained-mlp", _rows([0.95, 0.85, 0.4], sequences=8)
    )

    first = build_report(reports_root, tmp_path / "a", samples=300, seed=42)
    second = build_report(reports_root, tmp_path / "b", samples=300, seed=42)

    assert first["bootstrap_iou"] == second["bootstrap_iou"]
    assert first["bootstrap_precision_recall"] == second["bootstrap_precision_recall"]
