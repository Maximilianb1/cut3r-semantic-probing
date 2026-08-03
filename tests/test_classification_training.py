"""End-to-end tests for the classification training path.

Covers the three things that are easy to get subtly wrong and invisible when wrong:
the output dimension follows the fixed category vocabulary (not the categories a cache
happens to hold), standardization statistics come from the train split and travel with
the checkpoint, and the balanced sampler evens out category frequency.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from src.backbones.probe_cache import (
    TARGET_ONLY,
    EmbeddingCacheWriter,
    EmbeddingSample,
    category_index_map,
    category_vocabulary,
)
from src.classification.dataset_classification import (
    ProbeCacheClassificationDataset,
    cache_categories,
)
from src.classification.inference_classification import run_inference
from src.classification.model_classification import (
    ClassificationProbe,
    HeadConfig,
    feature_statistics,
    resolve_features,
)
from src.classification.train_classification import (
    build_balanced_sampler,
    run_directory,
    train_from_config,
)

_DIM = 8
_GRID = (2, 2)


def _write_cache(cache_dir, counts: dict[str, int], split: str = "train", start: int = 0) -> int:
    """Write ``counts[category]`` windows per category into ``cache_dir``."""
    index_of = category_index_map()
    written = start
    with EmbeddingCacheWriter(cache_dir, contract={"layout": TARGET_ONLY},
                              windows_per_shard=64) as writer:
        for category, count in counts.items():
            for _ in range(count):
                writer.add(
                    EmbeddingSample(
                        window_id=f"w{written}", category=category,
                        category_index=index_of[category], sequence_id=f"seq{written}",
                        split=split, layout=TARGET_ONLY, token_grid=_GRID,
                        frame_ids=[f"{written}_{j}" for j in range(6)],
                        seg_labels=torch.zeros(*_GRID),
                        spatial_tokens=torch.randn(_GRID[0] * _GRID[1], _DIM) + index_of[category],
                        global_tokens=torch.randn(1, _DIM),
                    ),
                    source_sha256={},
                )
                written += 1
    return written


@pytest.fixture()
def cache(tmp_path):
    directory = tmp_path / "cache"
    written = _write_cache(directory, {"apple": 6, "ball": 2}, split="train")
    _write_cache(directory, {"apple": 2, "ball": 2}, split="val", start=written)
    return directory


def _config(cache, tmp_path, **training):
    return {
        "experiment": "test",
        "probe_cache": {"dir": str(cache)},
        "features": {"source": "image_tokens", "pooling": "mean", "normalize": "standardize"},
        "model": {"feature_dim": _DIM, "hidden_dims": []},
        "training": {"epochs": 1, "batch_size": 4, "device": "cpu", "seed": 0,
                     "progress": False, **training},
        "splits": {"train": "train", "val": "val"},
        "output": {"dir": str(tmp_path / "run")},
    }


def test_output_dim_follows_the_vocabulary_not_the_cache(cache, tmp_path) -> None:
    """The cache holds 2 categories; labels are still indices into all 51."""
    result = train_from_config(_config(cache, tmp_path))
    assert result["model"]["num_classes"] == len(category_vocabulary()) == 51


def test_a_wrong_output_dim_is_rejected(cache, tmp_path) -> None:
    config = _config(cache, tmp_path)
    config["model"]["num_classes"] = 2  # the categories present, not the vocabulary
    with pytest.raises(ValueError, match="a 51-category label space"):
        train_from_config(config)


def test_label_space_present_narrows_the_head_to_the_cache(cache, tmp_path) -> None:
    """``present`` sizes the head to the categories the cache holds, not all 51."""
    config = _config(cache, tmp_path)
    config["model"]["label_space"] = "present"
    result = train_from_config(config)
    assert result["model"]["num_classes"] == 2  # this cache holds apple and ball
    assert result["label_space"] == ["apple", "ball"]
    assert "label_space" not in result["model"], "selects the space; not a head hyperparameter"

    checkpoint = torch.load(run_directory(config, resolve_features(config)) / "head.pt",
                            weights_only=True)
    assert checkpoint["label_space"] == ["apple", "ball"]
    assert checkpoint["head_state_dict"]["net.0.weight"].shape == (2, _DIM)


def test_label_space_defaults_to_the_full_vocabulary(cache, tmp_path) -> None:
    result = train_from_config(_config(cache, tmp_path))
    assert result["model"]["num_classes"] == 51
    assert result["label_space"] == list(category_vocabulary())


def test_an_unknown_label_space_is_rejected(cache, tmp_path) -> None:
    config = _config(cache, tmp_path)
    config["model"]["label_space"] = "present_in_train"
    with pytest.raises(ValueError, match="Unknown model.label_space"):
        train_from_config(config)


def test_present_labels_are_contiguous_not_vocabulary_indices(cache) -> None:
    """apple is 0 and ball is 2 in the vocabulary; under ``present`` they are 0 and 1."""
    present = ProbeCacheClassificationDataset(
        cache, source="image_tokens", split="train", label_space=cache_categories(cache)
    )
    assert sorted({int(present[i]["label"]) for i in range(len(present))}) == [0, 1]

    full = ProbeCacheClassificationDataset(cache, source="image_tokens", split="train")
    assert sorted({int(full[i]["label"]) for i in range(len(full))}) == [0, 2]


def test_inference_refuses_a_head_from_a_different_label_space(cache, tmp_path) -> None:
    """A 2-way head read as 51-way would name every prediction wrongly."""
    trained = _config(cache, tmp_path)
    trained["model"]["label_space"] = "present"
    train_from_config(trained)
    checkpoint = run_directory(trained, resolve_features(trained)) / "head.pt"

    evaluated = _config(cache, tmp_path)  # defaults back to the 51-category vocabulary
    with pytest.raises(ValueError, match="label space"):
        run_inference(evaluated, checkpoint=checkpoint, split="val")


def test_statistics_are_saved_with_the_checkpoint(cache, tmp_path) -> None:
    config = _config(cache, tmp_path)
    train_from_config(config)
    directory = run_directory(config, resolve_features(config))
    checkpoint = torch.load(directory / "head.pt", weights_only=True)
    assert checkpoint["features"]["normalize"] == "standardize"
    assert checkpoint["feature_mean"].shape == (_DIM,)
    assert checkpoint["feature_std"].shape == (_DIM,)
    assert torch.all(checkpoint["feature_std"] > 0)


def test_statistics_come_from_train_only(cache, tmp_path) -> None:
    """Recomputing them per split at evaluation would leak that split into its score."""
    config = _config(cache, tmp_path)
    train_from_config(config)
    saved = torch.load(run_directory(config, resolve_features(config)) / "head.pt",
                       weights_only=True)
    train_set = ProbeCacheClassificationDataset(cache, source="image_tokens", split="train")
    pooled = torch.stack([train_set[i]["features"] for i in range(len(train_set))])
    mean, _ = feature_statistics(pooled)
    assert torch.allclose(saved["feature_mean"], mean, atol=1e-5)

    val_set = ProbeCacheClassificationDataset(cache, source="image_tokens", split="val")
    val_pooled = torch.stack([val_set[i]["features"] for i in range(len(val_set))])
    val_mean, _ = feature_statistics(val_pooled)
    assert not torch.allclose(saved["feature_mean"], val_mean, atol=1e-3), (
        "train and val means coincide, so this cannot distinguish the two"
    )


def test_standardization_actually_normalizes() -> None:
    probe = ClassificationProbe(HeadConfig(feature_dim=4, num_classes=51),
                                normalize="standardize")
    vectors = torch.randn(50, 4) * 7.0 + 3.0
    probe.set_feature_statistics(*feature_statistics(vectors))
    standardized = probe.apply_normalization(vectors)
    assert torch.allclose(standardized.mean(dim=0), torch.zeros(4), atol=1e-5)
    assert torch.allclose(standardized.std(dim=0), torch.ones(4), atol=1e-5)


def test_standardization_without_statistics_refuses_to_run() -> None:
    probe = ClassificationProbe(HeadConfig(feature_dim=4, num_classes=51),
                                normalize="standardize")
    with pytest.raises(RuntimeError, match="no statistics were installed"):
        probe.forward(torch.randn(2, 4))


def test_constant_dimension_does_not_divide_by_zero() -> None:
    vectors = torch.randn(10, 3)
    vectors[:, 1] = 2.5  # a dead dimension, as a frozen backbone can produce
    _, std = feature_statistics(vectors)
    assert std[1] > 0
    assert torch.isfinite((vectors - vectors.mean(0)) / std).all()


def test_balanced_sampler_evens_out_category_frequency(cache) -> None:
    train_set = ProbeCacheClassificationDataset(cache, source="image_tokens", split="train")
    assert train_set.class_counts() == {"apple": 6, "ball": 2}  # 3:1 imbalance

    generator = torch.Generator().manual_seed(0)
    drawn = Counter(
        train_set.rows[index]["category"]
        for _ in range(200)
        for index in build_balanced_sampler(train_set, generator)
    )
    ratio = drawn["apple"] / drawn["ball"]
    assert 0.8 < ratio < 1.25, f"expected roughly equal draws, got {dict(drawn)}"


def test_sampler_is_opt_in_and_training_only(cache, tmp_path) -> None:
    result = train_from_config(_config(cache, tmp_path, balanced_sampler=True))
    assert result["training"]["balanced_sampler"] is True
    # val must keep the real distribution, so its window count is unchanged
    assert result["val_windows"] == 4


def test_statistics_ignore_the_balanced_sampler(cache, tmp_path) -> None:
    """The sampler draws with replacement; statistics must still describe the split.

    Iterating the training loader would count rare windows several times and miss
    others, so head.pt would carry numbers that depend on the sampler rather than on
    the data - and two runs differing only in the sampler flag would standardize
    differently.
    """
    plain = _config(cache, tmp_path / "plain")
    train_from_config(plain)
    without = torch.load(
        run_directory(plain, resolve_features(plain)) / "head.pt", weights_only=True
    )

    sampled = _config(cache, tmp_path / "sampled", balanced_sampler=True)
    train_from_config(sampled)
    with_sampler = torch.load(
        run_directory(sampled, resolve_features(sampled)) / "head.pt", weights_only=True
    )

    assert torch.allclose(without["feature_mean"], with_sampler["feature_mean"])
    assert torch.allclose(without["feature_std"], with_sampler["feature_std"])

    # ...and they are the statistics of the split itself, not of any resampling.
    train_set = ProbeCacheClassificationDataset(cache, source="image_tokens", split="train")
    pooled = torch.stack([train_set[i]["features"] for i in range(len(train_set))])
    mean, std = feature_statistics(pooled)
    assert torch.allclose(with_sampler["feature_mean"], mean, atol=1e-5)
    assert torch.allclose(with_sampler["feature_std"], std, atol=1e-5)
