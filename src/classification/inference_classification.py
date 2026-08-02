"""
Run a trained classification probe on a held-out split (inference / evaluation).

Counterpart to train_classification.py: reloads the trained MLP head saved as "head.pt",
rebuilds the: class:`ClassificationProbe`, and evaluates it on a chosen
split of the probe-feature cache -- reporting the same accuracy, macro recall and
macro F1, plus per-window predictions and a confusion matrix.

Run example: python -m src.classification.inference_classification --config src/classification/configs/cut3r_trained.yaml --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.backbones.probe_cache import load_probe_index

from .model_classification import FeatureSpec, HeadConfig, build_probe, resolve_features
from .dataset_classification import ProbeCacheClassificationDataset
from .train_classification import (
    _resolve_device,
    evaluate_multiclass,
    load_config,
    probe_cache_provenance,
    resolve_model_config,
    run_directory,
)


def load_trained_probe(config: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, FeatureSpec]:
    """
    Rebuild the probe and load the trained head from ``head.pt``.

    The **checkpoint** defines both the architecture and the representation it was trained on;
    the config is only cross-checked against it.

    Returns (model, features) where "features" is the recorded {"source", "pooling"}.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_cfg = checkpoint["model_config"]
    # Same derivation training used, so a config that omits num_classes still compares.
    config_model = resolve_model_config(config) if config.get("model") is not None else None
    if config_model is not None and HeadConfig.from_dict(config_model) != HeadConfig.from_dict(model_cfg):
        raise ValueError(
            f"Config model block {config_model} disagrees with the head saved in "
            f"{checkpoint_path} ({model_cfg}); retrain or evaluate the matching checkpoint"
        )
    features = checkpoint.get("features")
    if features is None:  # checkpoint from before features were recorded
        raise ValueError(
            f"{checkpoint_path} does not record which features it was trained on; "
            "retrain so the representation is part of the checkpoint"
        )
    requested = resolve_features(config)
    if requested != features:
        raise ValueError(
            f"Config asks for features {requested} but {checkpoint_path} was trained on "
            f"{features}; those are different representations, so the score would not "
            "describe the trained probe"
        )
    features = FeatureSpec(**features)  # a plain dict comes back from torch.load
    model = build_probe(model_cfg, normalize=features["normalize"])
    model.head.load_state_dict(checkpoint["head_state_dict"])
    if model.normalize != "none":
        # The train-split statistics, reused verbatim. Recomputing them here would let
        # the evaluated split's own statistics into its score.
        if "feature_mean" not in checkpoint or "feature_std" not in checkpoint:
            raise ValueError(
                f"{checkpoint_path} asks for normalize={model.normalize!r} but carries no "
                "statistics; retrain so they are saved with the head"
            )
        model.set_feature_statistics(checkpoint["feature_mean"], checkpoint["feature_std"])
    model.to(device).eval()
    return model, features


def assert_not_trained_on(cache_dir: str | Path, config: dict[str, Any], dataset: ProbeCacheClassificationDataset,
    split: str) -> None:
    """Fail loudly if the evaluated split shares CO3D sequences with the train split."""
    train_split = (config.get("splits") or {}).get("train", "train")
    if train_split == split:
        return
    categories = config.get("categories")
    allowed = None if categories is None else set(categories)
    trained_sequences = {
        row["sequence_id"]
        for row in load_probe_index(Path(cache_dir))
        if row["split"] == train_split and (allowed is None or row["category"] in allowed)
    }
    overlap = sorted(trained_sequences & dataset.sequence_ids())
    if overlap:
        raise ValueError(
            f"Split {split!r} shares {len(overlap)} sequence(s) with the training split "
            f"{train_split!r} (e.g. {overlap[:3]}); evaluation would not be held out"
        )


def run_inference(config: dict[str, Any], *, checkpoint: str | Path | None = None, split: str = "test",
    device: str | None = None, save_dir: str | Path | None = None) -> dict[str, Any]:
    """Evaluate "split" with the trained head and return the run record."""
    training = config.get("training", {})
    resolved_device = _resolve_device(device or training.get("device", "cpu"))
    # Same derivation training used, so the arm's own directory is found automatically.
    run_dir = run_directory(config, resolve_features(config))
    checkpoint_path = Path(checkpoint) if checkpoint else (run_dir or Path()) / "head.pt"
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Trained head not found at {checkpoint_path}; run train_classification.py first")
    model, features = load_trained_probe(config, checkpoint_path, resolved_device)

    cache_dir = config["probe_cache"]["dir"]
    dataset = ProbeCacheClassificationDataset(
        cache_dir, source=features["source"], pooling=features["pooling"], split=split,
        categories=config.get("categories"),
    )
    assert_not_trained_on(cache_dir, config, dataset, split)
    loader = DataLoader(dataset, batch_size=int(training.get("batch_size", 32)), shuffle=False)
    metrics = evaluate_multiclass(
        model, loader, resolved_device, top_k=int(training.get("top_k", 5)),
        collect_windows=True,
        desc=f"inference {split}" if training.get("progress", True) else None,
    )
    per_window = metrics.pop("per_window")
    confusion = metrics.pop("confusion")

    result = {
        "experiment": config.get("experiment", "classification"),
        "split": split,
        "checkpoint": str(checkpoint_path),
        # Same provenance metrics.json carries, so an evaluation file also states which
        # cache and which representation produced it.
        "probe_cache": {"dir": str(cache_dir), "metadata": probe_cache_provenance(cache_dir)},
        "features": features,
        "windows": len(dataset),
        "class_counts": dataset.class_counts(),
        "metrics": metrics,
        "confusion": confusion,
        "per_window": per_window,
    }
    if save_dir:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"inference-{split}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="head.pt (defaults to output.dir/head.pt)")
    parser.add_argument("--split", default="test", help="Cache split to evaluate (default: test)")
    parser.add_argument("--device", default=None,
                        help="Override device (default: config training.device)")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="Where to write the inference JSON")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    result = run_inference(
        config,
        checkpoint=arguments.checkpoint,
        split=arguments.split,
        device=arguments.device,
        save_dir=arguments.save_dir or run_directory(config, resolve_features(config)),
    )
    metrics = result["metrics"]
    top_k_key = next(key for key in metrics if key.startswith("top"))
    print(
        f"[{result['experiment']}] split={result['split']} windows={result['windows']}  "
        f"acc {metrics['accuracy']:.4f}  "
        f"{top_k_key} {metrics[top_k_key]:.4f}  "
        f"macro P/R/F1 {metrics['macro_precision']:.4f}/{metrics['macro_recall']:.4f}/{metrics['macro_f1']:.4f}  "
        f"({result['features']['source']}/{result['features']['pooling']})"
    )


if __name__ == "__main__":
    main()
