"""Run a trained segmentation probe on a held-out split (inference / evaluation).

This is the counterpart to ``train_segmentation.py``: it reloads the trained MLP
head saved as ``head.pt``, rebuilds the :class:`SegmentationProbe`, and evaluates
it on a chosen split of the probe-feature cache -- reporting the same token
accuracy and foreground IoU (macro / micro / per-category), plus optional
per-window predictions and grid-resolution predicted masks.

The backbone is not involved here: features come from the probe-feature cache
and only the trained head runs. Run from inside ``segmentation_validation/``::

    python segmentation_inference.py --config configs/cut3r_trained.json --split test

Grid-resolution or upsampled masks come from ``SegmentationProbe.predict_mask``
on the cached tokens (see ``predict_windows``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from model_segmentation import build_probe
from segmentation_dataset import ProbeCacheDataset, collate_windows
from train_segmentation import _resolve_device, evaluate_binary


def load_trained_probe(
    config: dict[str, Any], checkpoint_path: str | Path, device: torch.device
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = config.get("model") or checkpoint["model_config"]
    model = build_probe(model_cfg)
    model.head.load_state_dict(checkpoint["head_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict_windows(
    model: torch.nn.Module, dataset: ProbeCacheDataset, device: torch.device
) -> list[dict[str, Any]]:
    """Per-window predicted foreground label grids and IoU vs. the target."""
    predictions: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]
        spatial = item["spatial"].to(device)[None]  # [1, N, D]
        grid = item["token_grid"]
        logits = model.logit_grid(spatial, grid)[0, 0]  # [h, w]
        predicted = (logits > 0.0).to(torch.float32).cpu()
        target = item["labels"].reshape(grid).cpu()
        intersection = float(((predicted == 1) & (target == 1)).sum().item())
        union = float(((predicted == 1) | (target == 1)).sum().item())
        predictions.append(
            {
                "window_id": item["window_id"],
                "category": item["category"],
                "token_grid": list(grid),
                "foreground_iou": 1.0 if union == 0.0 else intersection / union,
                "predicted_labels": predicted,
                "target_labels": target,
            }
        )
    return predictions


def run_inference(
    config: dict[str, Any],
    *,
    checkpoint: str | Path | None = None,
    split: str = "test",
    device: str | None = None,
    save_dir: str | Path | None = None,
    save_masks: bool = False,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device or config.get("training", {}).get("device", "cpu"))
    checkpoint_path = Path(checkpoint) if checkpoint else Path(config["output"]["dir"]) / "head.pt"
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"Trained head not found at {checkpoint_path}; run train_segmentation.py first"
        )
    model = load_trained_probe(config, checkpoint_path, resolved_device)

    dataset = ProbeCacheDataset(
        config["probe_cache"]["dir"], split=split, categories=config.get("categories")
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("training", {}).get("batch_size", 16)),
        shuffle=False,
        collate_fn=collate_windows,
    )
    metrics = evaluate_binary(model, loader, resolved_device)
    per_window = predict_windows(model, dataset, resolved_device)

    result = {
        "experiment": config.get("experiment", "segmentation"),
        "split": split,
        "checkpoint": str(checkpoint_path),
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
    parser.add_argument("--config", required=True, type=Path, help="Path to a JSON config")
    parser.add_argument("--checkpoint", type=Path, default=None, help="head.pt (defaults to output/dir/head.pt)")
    parser.add_argument("--split", default="test", help="Cache split to evaluate (default: test)")
    parser.add_argument("--device", default=None, help="Override device (default: config training.device)")
    parser.add_argument("--save-dir", type=Path, default=None, help="Where to write inference JSON / masks")
    parser.add_argument("--save-masks", action="store_true", help="Also dump per-window predicted/target grids")
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
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
