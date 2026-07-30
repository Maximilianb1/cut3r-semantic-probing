"""Train and evaluate the binary segmentation probe over cached features.

This is the shared driver for all three backbones; only the config differs. The
backbone is never run here - it was run ahead of time and its outputs live in the
probe-feature cache, so training only fits the small MLP head.

Reading guide - what one run does (see :func:`train_from_config`):

1. Read the JSON config and pick device / seed.
2. Build the train and val datasets from the cache, and assert their CO3D
   sequences don't overlap (no leakage).
3. Build the head (:class:`SegmentationProbe`) and an Adam optimizer over it.
4. Each epoch: for every batch of tokens, predict a foreground logit per token,
   compute per-token ``BCEWithLogitsLoss`` against the labels, and step.
5. After each epoch, evaluate on val (:func:`evaluate_binary`) - foreground IoU
   (macro over windows, micro over tokens, per-category) plus token accuracy, all
   at token / patch-grid resolution. Finally save ``metrics.json`` and ``head.pt``.

Run from inside ``segmentation_validation/`` after installing from the repo root::

    python train_segmentation.py --config configs/cut3r_trained.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from model_segmentation import build_probe
from segmentation_dataset import ProbeCacheDataset, assert_sequence_disjoint, collate_windows


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is false")
    return torch.device(name)


def _split_by_counts(values: torch.Tensor, counts: torch.Tensor) -> list[torch.Tensor]:
    return list(torch.split(values, counts.tolist()))


@torch.no_grad()
def evaluate_binary(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    collect_windows: bool = False,
    collect_masks: bool = False,
) -> dict[str, Any]:
    """Token accuracy + foreground IoU (macro/micro/per-category) for binary probes.

    All metrics are at **token / patch-grid resolution** (the mask was pooled to
    the backbone's token grid), not full pixel resolution. A token is predicted
    foreground when its logit ``> 0`` (i.e. ``sigmoid(logit) > 0.5``).

    Convention: a window whose prediction and target are both entirely background
    (foreground union ``== 0``) scores IoU ``1.0``. This is an open choice (see
    the README) and can inflate macro-IoU for foreground-free windows.

    Set ``collect_windows`` to also return a ``per_window`` list (id, category,
    IoU), and ``collect_masks`` to include each window's predicted/target label
    grids. Both are off by default so per-epoch training history stays small;
    inference turns them on to avoid a second pass over the data.
    """
    model.eval()
    per_window_iou: list[float] = []
    per_category_iou: dict[str, list[float]] = {}
    windows: list[dict[str, Any]] = []
    global_intersection = global_union = 0.0
    correct = total = 0
    for batch in loader:
        spatial = batch["spatial"].to(device)
        labels = batch["labels"].to(device)
        # Apply the head directly to the flat [sum_N, D] token pile. We call
        # model.head (not model.forward, which expects a [B, N, D] batch) because
        # collation already flattened all windows' tokens into one dimension.
        logits = model.head(spatial).squeeze(-1)  # [sum_N]
        prediction = (logits > 0.0).to(torch.float32)
        correct += float((prediction == labels).sum().item())
        total += int(labels.numel())
        preds = _split_by_counts(prediction.cpu(), batch["counts"])
        gts = _split_by_counts(labels.cpu(), batch["counts"])
        for position, (pred, gt, category) in enumerate(
            zip(preds, gts, batch["categories"])
        ):
            intersection = float(((pred == 1) & (gt == 1)).sum().item())
            union = float(((pred == 1) | (gt == 1)).sum().item())
            global_intersection += intersection
            global_union += union
            iou = 1.0 if union == 0.0 else intersection / union
            per_window_iou.append(iou)
            per_category_iou.setdefault(category, []).append(iou)
            if collect_windows:
                grid = tuple(batch["token_grids"][position])
                record = {
                    "window_id": batch["window_ids"][position],
                    "category": category,
                    "token_grid": list(grid),
                    "foreground_iou": iou,
                }
                if collect_masks:
                    record["predicted_labels"] = pred.reshape(grid)
                    record["target_labels"] = gt.reshape(grid)
                windows.append(record)
    macro_iou = sum(per_window_iou) / len(per_window_iou) if per_window_iou else 0.0
    micro_iou = 1.0 if global_union == 0.0 else global_intersection / global_union
    category_iou = {
        category: sum(values) / len(values) for category, values in per_category_iou.items()
    }
    metrics = {
        "token_accuracy": correct / total if total else 0.0,
        "macro_foreground_iou": macro_iou,
        "micro_foreground_iou": micro_iou,
        "mean_category_iou": (
            sum(category_iou.values()) / len(category_iou) if category_iou else 0.0
        ),
        "per_category_iou": category_iou,
        "windows": len(per_window_iou),
    }
    if collect_windows:
        metrics["per_window"] = windows
    return metrics


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Train the binary segmentation probe from a parsed config dict.

    Builds the probe-cache datasets (sequence-disjoint train/val), trains only
    the MLP head with per-token BCE while the backbone stays frozen/precomputed,
    evaluates on val each epoch, and (if ``output.dir`` is set) writes
    ``metrics.json`` and ``head.pt``. Returns the run record incl. per-epoch
    history. Note: ``head.pt`` is the **final** epoch's head, not the best-val one.
    """
    training = config.get("training", {})
    splits = config.get("splits", {"train": "train", "val": "val"})
    model_cfg = dict(config["model"])
    device = _resolve_device(str(training.get("device", "cpu")))
    seed = int(training.get("seed", 0))
    torch.manual_seed(seed)

    if int(model_cfg.get("num_classes", 1)) != 1:
        raise NotImplementedError(
            "train_segmentation currently implements the binary (num_classes=1) probe"
        )

    cache_dir = config["probe_cache"]["dir"]
    categories = config.get("categories")
    train_set = ProbeCacheDataset(cache_dir, split=splits["train"], categories=categories)
    val_set = ProbeCacheDataset(cache_dir, split=splits["val"], categories=categories)
    assert_sequence_disjoint(train_set, val_set)

    train_loader = DataLoader(
        train_set,
        batch_size=int(training.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=collate_windows,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(training.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=collate_windows,
    )

    model = build_probe(model_cfg).to(device)
    pos_weight = training.get("pos_weight")
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=None if pos_weight is None else torch.tensor(float(pos_weight), device=device)
    )
    optimizer = torch.optim.Adam(
        model.head.parameters(),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )

    history: list[dict[str, Any]] = []
    epochs = int(training.get("epochs", 10))
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for batch in train_loader:
            spatial = batch["spatial"].to(device)
            labels = batch["labels"].to(device)
            logits = model.head(spatial).squeeze(-1)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * labels.numel()
            seen += int(labels.numel())
        metrics = evaluate_binary(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": epoch_loss / seen if seen else 0.0, "val": metrics}
        history.append(record)
        print(
            f"epoch {epoch:3d}  loss {record['train_loss']:.4f}  "
            f"val macro_IoU {metrics['macro_foreground_iou']:.4f}  "
            f"token_acc {metrics['token_accuracy']:.4f}"
        )

    result = {
        "experiment": config.get("experiment", "segmentation"),
        "backbone": config.get("backbone"),
        "model": model_cfg,
        "is_linear_probe": model.head.is_linear,
        "seed": seed,
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "history": history,
        "final_val": history[-1]["val"] if history else None,
    }
    output_dir = config.get("output", {}).get("dir")
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        # Save the trained head so segmentation_inference.py can reload the probe.
        torch.save(
            {"head_state_dict": model.head.state_dict(), "model_config": model_cfg},
            path / "head.pt",
        )
        result["checkpoint"] = str(path / "head.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a JSON config")
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    train_from_config(config)


if __name__ == "__main__":
    main()
