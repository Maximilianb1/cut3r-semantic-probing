"""Tests for the Stage 2 held-out test report builder."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src.classification.build_test_report import (
    build_report,
    comparison_pairs,
    discover_runs,
    read_predictions,
    score_run,
)

CATEGORIES = ("apple", "bench", "cup", "kite")


def _write_predictions(path, rows):
    """One inference-<split>-probabilities.csv, in the writer's column order."""
    fieldnames = [
        "window_id",
        "sequence_id",
        "category",
        "predicted_category",
        "predicted_probability",
        "correct",
        *(f"probability_{name}" for name in CATEGORIES),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rows(accuracy_pattern, *, sequences=6, windows=3, seed=0):
    """Build predictions whose per-sequence correctness follows ``accuracy_pattern``."""
    generator = np.random.default_rng(seed)
    rows = []
    for sequence in range(sequences):
        category = CATEGORIES[sequence % len(CATEGORIES)]
        for window in range(windows):
            correct = accuracy_pattern[(sequence * windows + window) % len(accuracy_pattern)]
            probabilities = generator.uniform(0.01, 0.05, size=len(CATEGORIES))
            winner = CATEGORIES.index(category) if correct else (
                (CATEGORIES.index(category) + 1) % len(CATEGORIES)
            )
            probabilities[winner] = 1.0
            probabilities = probabilities / probabilities.sum()
            predicted = CATEGORIES[int(probabilities.argmax())]
            rows.append({
                "window_id": f"window-{sequence}-{window}",
                "sequence_id": f"seq-{sequence}",
                "category": category,
                "predicted_category": predicted,
                "predicted_probability": float(probabilities.max()),
                "correct": predicted == category,
                **{
                    f"probability_{name}": float(probabilities[index])
                    for index, name in enumerate(CATEGORIES)
                },
            })
    return rows


def test_read_predictions_recovers_the_label_space_and_windows(tmp_path):
    path = tmp_path / "cut3r-trained_linear.csv"
    _write_predictions(path, _rows([True]))
    inference = read_predictions(path)

    assert inference["label_space"] == list(CATEGORIES)
    assert len(inference["per_window"]) == 18
    first = inference["per_window"][0]
    assert set(first["class_probabilities"]) == set(CATEGORIES)
    assert first["sequence_id"] == "seq-0"


def test_discover_runs_reads_the_label_from_the_filename(tmp_path):
    for name in ("cut3r-trained_linear.csv", "dinov2_mlp512.csv"):
        _write_predictions(tmp_path / name, _rows([True]))

    assert set(discover_runs(tmp_path)) == {"cut3r-trained/linear", "dinov2/mlp512"}


def test_discover_runs_rejects_a_filename_without_a_model_suffix(tmp_path):
    _write_predictions(tmp_path / "dinov2.csv", _rows([True]))

    with pytest.raises(ValueError, match="dataset/model label"):
        discover_runs(tmp_path)


def test_score_run_counts_windows_and_sequences_separately(tmp_path):
    path = tmp_path / "cut3r-trained_linear.csv"
    _write_predictions(path, _rows([True, True, False]))
    score = score_run(read_predictions(path))

    assert score["windows"] == 18
    assert score["sequences"] == 6
    # Two of every three windows are right, and the sequence vote carries them.
    assert score["window_metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert score["sequence_metrics"]["accuracy"] == pytest.approx(1.0)
    assert score["window_metrics"]["top5_accuracy"] == pytest.approx(1.0)


def test_comparison_pairs_orders_heads_within_a_backbone_then_across_backbones():
    labels = [
        "cut3r-trained/linear",
        "cut3r-trained/mlp512",
        "cut3r-random/linear",
        "cut3r-random/mlp512",
    ]
    accuracy = {
        "cut3r-trained/linear": 0.7,
        "cut3r-trained/mlp512": 0.68,
        "cut3r-random/linear": 0.21,
        "cut3r-random/mlp512": 0.24,
    }

    assert comparison_pairs(labels, accuracy) == [
        ("cut3r-trained/mlp512", "cut3r-trained/linear"),
        ("cut3r-random/mlp512", "cut3r-random/linear"),
        # The stronger representation is always the target, so the difference is positive.
        ("cut3r-trained/linear", "cut3r-random/linear"),
        ("cut3r-trained/mlp512", "cut3r-random/mlp512"),
    ]


def test_build_report_writes_every_table_and_figure(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    _write_predictions(predictions / "cut3r-trained_linear.csv", _rows([True, True, False], seed=1))
    _write_predictions(predictions / "cut3r-trained_mlp512.csv", _rows([True, False, False], seed=2))

    output = tmp_path / "report"
    report = build_report(discover_runs(predictions), output, samples=200, seed=7)

    for name in (
        "model-test-metrics.csv",
        "model-test-per-class-metrics.csv",
        "bootstrap-accuracy-ci.csv",
        "bootstrap-accuracy-differences.csv",
        "test-bootstrap-report.json",
        "bootstrap-test-accuracy-linear.png",
        "bootstrap-test-accuracy-mlp512.png",
        "bootstrap-window-accuracy-ci.png",
        "paired-bootstrap-window-accuracy-differences.png",
    ):
        assert (output / name).is_file(), name

    written = json.loads((output / "test-bootstrap-report.json").read_text(encoding="utf-8"))
    assert written["protocol"]["clusters"] == 6
    assert written["protocol"]["chance_accuracy"] == pytest.approx(1 / len(CATEGORIES))
    assert len(report["comparisons"]) == 1
    interval = report["bootstrap_accuracy"]["cut3r-trained/linear"]["window_accuracy"]
    assert interval["ci95"][0] <= interval["estimate"] <= interval["ci95"][1]


def test_build_report_refuses_runs_evaluated_on_different_test_sets(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    _write_predictions(predictions / "cut3r-trained_linear.csv", _rows([True], sequences=6))
    _write_predictions(predictions / "cut3r-trained_mlp512.csv", _rows([True], sequences=5))

    with pytest.raises(ValueError, match="not evaluated on the same test set"):
        build_report(discover_runs(predictions), tmp_path / "report", samples=50)


def test_recorded_seeds_reproduce_a_published_interval_exactly(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    _write_predictions(predictions / "cut3r-trained_linear.csv", _rows([True, False], seed=3))

    seeds = {"ci/cut3r-trained/linear": 123456}
    first = build_report(
        discover_runs(predictions), tmp_path / "a", samples=500, recorded_seeds=seeds
    )
    # A different base seed must not move an interval whose seed was recorded.
    second = build_report(
        discover_runs(predictions), tmp_path / "b", samples=500, seed=999, recorded_seeds=seeds
    )

    assert first["bootstrap_accuracy"] == second["bootstrap_accuracy"]
