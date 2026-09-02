"""
Run a trained segmentation probe on a held-out split (test).

This is the counterpart to "train_segmentation.py`": it reloads the trained MLP
head saved as "head.pt", rebuilds the: class:`SegmentationProbe`, and evaluates
it on a chosen split of the probe-feature cache -- reporting the same token
accuracy and foreground IoU (macro / micro / per-category), plus optional
per-window predictions and grid-resolution predicted masks.

Run example: python -m src.segmentation.inference_segmentation \n  --config src/segmentation/configs/cut3r_trained_partial_mlp.yaml --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.backbones.probe_cache import load_probe_index

from .model_segmentation import HeadConfig, build_probe
from .dataset_segmentation import ProbeCacheDataset, collate_windows
from .train_segmentation import (
    _resolve_device,
    evaluate_binary,
    load_config,
    probe_cache_provenance,
)


def load_trained_probe(config: dict[str, Any], checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    """
    Rebuild the probe and load the trained head from "head.pt".
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_cfg = checkpoint["model_config"]
    config_model = config.get("model")
    if config_model is not None and HeadConfig.from_dict(config_model) != HeadConfig.from_dict(model_cfg):
        raise ValueError(
            f"Config model block {config_model} disagrees with the head saved in "
            f"{checkpoint_path} ({model_cfg}); retrain or evaluate the matching checkpoint"
        )
    model = build_probe(model_cfg)
    model.head.load_state_dict(checkpoint["head_state_dict"])
    model.to(device).eval()
    return model


def assert_not_trained_on(cache_dir: str | Path, config: dict[str, Any], dataset: ProbeCacheDataset, split: str) -> None:
    """
    Fail loudly if the evaluated split shares CO3D sequences with the train split.

    Checks ``probe_cache.train_dirs`` when set (an expanded combined training set:
    original + leftover + cap100-new-train), not just ``cache_dir``, so evaluating
    against a combined-data run is checked against everything it actually trained
    on, not only the single val/test cache.
    """
    train_split = (config.get("splits") or {}).get("train", "train")
    if train_split == split:
        return
    categories = config.get("categories")
    allowed = None if categories is None else set(categories)
    probe_cache = config.get("probe_cache") or {}
    train_dirs = probe_cache.get("train_dirs")
    if train_dirs is None:
        train_dirs = [cache_dir]
    elif not isinstance(train_dirs, (list, tuple)) or len(train_dirs) == 0:
        raise ValueError("probe_cache.train_dirs must be a non-empty list when provided")
    trained_sequences: set[str] = set()
    for train_dir in train_dirs:
        trained_sequences |= {
            row["sequence_id"]
            for row in load_probe_index(Path(train_dir))
            if row["split"] == train_split and (allowed is None or row["category"] in allowed)
        }
    overlap = sorted(trained_sequences & dataset.sequence_ids())
    if overlap:
        raise ValueError(
            f"Split {split!r} shares {len(overlap)} sequence(s) with the training split "
            f"{train_split!r} (e.g. {overlap[:3]}); evaluation would not be held out"
        )


def run_inference(config: dict[str, Any], *, checkpoint: str | Path | None = None,
    split: str = "test", device: str | None = None,
    save_dir: str | Path | None = None, save_masks: bool = False) -> dict[str, Any]:
    resolved_device = _resolve_device(device or config.get("training", {}).get("device", "cpu"))
    checkpoint_path = Path(checkpoint) if checkpoint else Path(config["output"]["dir"]) / "head.pt"
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Trained head not found at {checkpoint_path}; run train_segmentation.py first")
    model = load_trained_probe(config, checkpoint_path, resolved_device)

    cache_dir = config["probe_cache"]["dir"]
    dataset = ProbeCacheDataset(cache_dir, split=split, categories=config.get("categories"))
    assert_not_trained_on(cache_dir, config, dataset, split)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 16)),
        shuffle=False,
        collate_fn=collate_windows,
    )
    # One batched pass produces the metrics, the per-window IoUs, and (only when
    # requested) the label grids - so the dataset is never iterated twice.
    metrics = evaluate_binary(
        model, loader, resolved_device, collect_windows=True, collect_masks=save_masks,
        desc=f"inference {split}" if config.get("training", {}).get("progress", True) else None,
    )
    per_window = metrics.pop("per_window")

    result = {
        "experiment": config.get("experiment", "segmentation"),
        "split": split,
        "checkpoint": str(checkpoint_path),
        # Same provenance metrics.json carries, so an evaluation file also states which
        # cache produced it - including "synthetic: true" for a smoke run, which
        # otherwise looks identical to a real result.
        "probe_cache": {"dir": str(cache_dir), "metadata": probe_cache_provenance(cache_dir)},
        "windows": len(dataset),
        "metrics": metrics,
        "per_window_iou": [
            {"window_id": p["window_id"], "category": p["category"], "foreground_iou": p["foreground_iou"]}
            for p in per_window
        ],
    }
    if save_dir:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"inference-{split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if save_masks:
            torch.save(
                {
                    p["window_id"]: {
                        "predicted_labels": p["predicted_labels"],
                        "target_labels": p["target_labels"],
                        "token_grid": p["token_grid"],
                    }
                    for p in per_window
                },
                path / f"masks-{split}.pt",
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML (or JSON) config")
    parser.add_argument("--checkpoint", type=Path, default=None, help="head.pt (defaults to output/dir/head.pt)")
    parser.add_argument("--split", default="test", help="Cache split to evaluate (default: test)")
    parser.add_argument("--device", default=None, help="Override device (default: config training.device)")
    parser.add_argument("--save-dir", type=Path, default=None, help="Where to write inference JSON / masks")
    parser.add_argument("--save-masks", action="store_true", help="Also dump per-window predicted/target grids")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    result = run_inference(
        config,
        checkpoint=arguments.checkpoint,
        split=arguments.split,
        device=arguments.device,
        save_dir=arguments.save_dir or config.get("output", {}).get("dir"),
        save_masks=arguments.save_masks,
    )
    metrics = result["metrics"]
    print(
        f"[{result['experiment']}] split={result['split']} windows={result['windows']}  "
        f"macro_IoU {metrics['macro_foreground_iou']:.4f}  "
        f"micro_IoU {metrics['micro_foreground_iou']:.4f}  "
        f"token_acc {metrics['token_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
