"""Tests for the classification probe's metrics, pooling and feature contract.

The properties worth pinning: macro recall and macro F1 diverge from accuracy under
class imbalance, macro F1 averages per-category F1s rather than the macro means, and the
representation is never defaulted (it is one arm of the Stage 2 comparison).

Category indices are the fixed 51-category vocabulary: 0 apple, 1 backpack, 2 ball. A
batch's names must match its label indices, as they always do in real data.
"""

from __future__ import annotations

import pytest
import torch

from src.classification.model_classification import (
    ClassificationProbe,
    HeadConfig,
    build_probe,
    pool_tokens,
    resolve_features,
)
from src.classification.train_classification import MulticlassMetrics


def _batch(categories: list[str]) -> dict[str, object]:
    """A collated batch, as default_collate produces it: singular keys, lists of B."""
    return {"category": categories, "window_id": [f"w{i}" for i in range(len(categories))]}


def test_pooling_reduces_tokens_to_one_vector() -> None:
    tokens = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    assert torch.equal(pool_tokens(tokens, "mean"), torch.tensor([2.0, 4.0]))
    assert torch.equal(pool_tokens(tokens, "max"), torch.tensor([3.0, 6.0]))


def test_unknown_pooling_raises_rather_than_defaulting() -> None:
    with pytest.raises(ValueError, match="Unknown pooling"):
        pool_tokens(torch.zeros(2, 2), "median")


def test_features_source_is_required() -> None:
    """Choosing the representation is a compared variable, so it must be explicit."""
    with pytest.raises(ValueError, match="explicit 'source'"):
        resolve_features({})
    with pytest.raises(ValueError, match="explicit 'source'"):
        resolve_features({"features": {"pooling": "mean"}})
    with pytest.raises(ValueError, match="Unknown features.source"):
        resolve_features({"features": {"source": "cls_token"}})
    assert resolve_features({"features": {"source": "state_tokens"}}) == {
        "source": "state_tokens", "pooling": "mean", "normalize": "standardize",
    }


def test_probe_rejects_a_binary_head() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        ClassificationProbe(HeadConfig(feature_dim=8, num_classes=1))


def test_probe_maps_one_vector_per_window_to_class_logits() -> None:
    probe = build_probe({"feature_dim": 8, "num_classes": 51, "hidden_dims": []})
    assert probe.forward(torch.randn(4, 8)).shape == (4, 51)
    with pytest.raises(ValueError, match=r"\[B, D\]"):
        probe.forward(torch.randn(4, 3, 8))  # a token axis has no meaning after pooling


def test_predict_returns_ranked_categories() -> None:
    probe = build_probe({"feature_dim": 8, "num_classes": 51, "hidden_dims": []})
    probabilities, indices = probe.predict(torch.randn(2, 8), top_k=3)
    assert probabilities.shape == indices.shape == (2, 3)
    assert torch.all(probabilities[:, 0] >= probabilities[:, 1])  # best first


def test_macro_metrics_expose_an_ignored_minority_class() -> None:
    """9 of 10 windows are 'apple' and the probe always says 'apple'.

    Accuracy reads 0.90 and looks fine. Macro recall reads 0.50 and macro F1 0.474,
    because the minority category is never found. That gap is why accuracy alone is
    not reported.
    """
    metrics = MulticlassMetrics(top_k=2)
    logits = torch.zeros(10, 51)
    logits[:, 0] = 5.0  # always predict category 0 ("apple")
    labels = torch.tensor([0] * 9 + [2])  # 0 apple, 2 ball
    metrics.update(logits, labels, _batch(["apple"] * 9 + ["ball"]))
    result = metrics.result()
    assert result["accuracy"] == pytest.approx(0.9)
    assert result["macro_recall"] == pytest.approx(0.5)
    assert result["per_category_recall"] == {"apple": 1.0, "ball": 0.0}
    # apple: predicted 10 times, right 9 -> precision 0.9, recall 1.0, F1 0.947
    # ball : never predicted        -> precision 0.0, recall 0.0, F1 0.0
    assert result["per_category_precision"]["apple"] == pytest.approx(0.9)
    assert result["per_category_precision"]["ball"] == 0.0
    assert result["per_category_f1"]["apple"] == pytest.approx(2 * 0.9 * 1.0 / 1.9)
    assert result["macro_precision"] == pytest.approx(0.45)
    assert result["macro_f1"] == pytest.approx((2 * 0.9 * 1.0 / 1.9) / 2)
    assert result["categories_present"] == 2


def test_macro_f1_averages_per_category_f1_not_the_macro_means() -> None:
    """F1(mean P, mean R) is a different number, and the wrong one."""
    metrics = MulticlassMetrics()
    logits = torch.zeros(4, 51)
    logits[0:3, 0] = 5.0   # predict apple three times
    logits[3, 2] = 5.0     # predict ball once
    labels = torch.tensor([0, 0, 2, 2])  # apple, apple, ball, ball
    metrics.update(logits, labels, _batch(["apple", "apple", "ball", "ball"]))
    r = metrics.result()
    # apple: tp 2 / predicted 3 -> P .667, recall 2/2 = 1.0,  F1 .8
    # ball : tp 1 / predicted 1 -> P 1.0,  recall 1/2 = 0.5,  F1 .667
    expected_macro_f1 = (0.8 + 2 * 1.0 * 0.5 / 1.5) / 2
    assert r["macro_f1"] == pytest.approx(expected_macro_f1)
    naive = 2 * r["macro_precision"] * r["macro_recall"] / (r["macro_precision"] + r["macro_recall"])
    assert r["macro_f1"] != pytest.approx(naive), "must not be F1 of the averaged P and R"


def test_micro_scores_equal_accuracy_so_are_not_reported() -> None:
    """Single-label multiclass: micro P = micro R = micro F1 = accuracy."""
    metrics = MulticlassMetrics()
    logits = torch.zeros(3, 51)
    logits[:, 0] = 5.0
    metrics.update(logits, torch.tensor([0, 0, 2]), _batch(["apple", "apple", "ball"]))
    result = metrics.result()
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert not any(key.startswith("micro") for key in result), sorted(result)


def test_absent_categories_are_omitted_not_scored_zero() -> None:
    metrics = MulticlassMetrics()
    metrics.update(torch.zeros(1, 51), torch.tensor([0]), _batch(["apple"]))
    result = metrics.result()
    assert list(result["per_category_recall"]) == ["apple"], "only present categories count"
    assert list(result["per_category_f1"]) == ["apple"]
    assert result["categories_present"] == 1


def test_top_k_accuracy_counts_a_near_miss() -> None:
    metrics = MulticlassMetrics(top_k=2)
    logits = torch.tensor([[1.0, 2.0, 0.0]])  # ranked: class 1, then class 0
    metrics.update(logits, torch.tensor([0]), _batch(["apple"]))
    result = metrics.result()
    assert result["accuracy"] == 0.0
    assert result["top2_accuracy"] == 1.0


def test_loss_is_example_weighted_across_uneven_batches() -> None:
    metrics = MulticlassMetrics()
    metrics.update(torch.zeros(3, 51), torch.zeros(3, dtype=torch.long),
                   _batch(["apple"] * 3), loss=1.0)
    metrics.update(torch.zeros(1, 51), torch.zeros(1, dtype=torch.long),
                   _batch(["apple"]), loss=5.0)
    # (1.0*3 + 5.0*1) / 4, not the mean of the two batch means (3.0)
    assert metrics.result()["loss"] == pytest.approx(2.0)


def test_loss_absent_when_none_was_fed() -> None:
    metrics = MulticlassMetrics()
    metrics.update(torch.zeros(1, 51), torch.zeros(1, dtype=torch.long), _batch(["apple"]))
    assert "loss" not in metrics.result()
