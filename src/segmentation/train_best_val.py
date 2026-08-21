"""Best-val variant of src.segmentation.train_segmentation.

Reuses the committed trainer's building blocks unchanged (model, dataset,
optimizer, BinaryMetrics/evaluate_binary); it only adds checkpoint-on-
improvement. train_segmentation.py itself always saves the *final* epoch's
head (documented there); this driver instead tracks validation macro-IoU
across training and saves whichever epoch was best, alongside the final
epoch's head for reference. See EXP-005 for the run this produced.

Run example:
    python -m src.segmentation.train_best_val \
        --config src/segmentation/configs/cut3r_trained.yaml \
        --output-dir src/segmentation/experiments/segmentation-cut3r-trained-bestval
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset_segmentation import ProbeCacheDataset, assert_sequence_disjoint, collate_windows
from .model_segmentation import build_probe
from .train_segmentation import (
    BinaryMetrics,
    _build_optimizer,
    _progress,
    _resolve_device,
    evaluate_binary,
    load_config,
    probe_cache_provenance,
)


def train_best_val(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Same training loop as train_from_config, plus checkpoint-on-improvement.

    Returns the run record; also writes metrics.json, head.pt (best-val) and
    head-last.pt (final epoch) under output_dir.
    """
    training = config.get("training", {})
    splits = config.get("splits", {"train": "train", "val": "val"})
    model_cfg = dict(config["model"])
    device = _resolve_device(str(training.get("device", "cpu")))
    seed = int(training.get("seed", 0))
    torch.manual_seed(seed)

    cache_dir = config["probe_cache"]["dir"]
    categories = config.get("categories")
    cache_metadata = probe_cache_provenance(cache_dir)

    train_set = ProbeCacheDataset(cache_dir, split=splits["train"], categories=categories)
    val_set = ProbeCacheDataset(cache_dir, split=splits["val"], categories=categories)
    assert_sequence_disjoint(train_set, val_set)

    batch_size = int(training.get("batch_size", 16))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=int(training.get("num_workers", 0)), collate_fn=collate_windows)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=int(training.get("num_workers", 0)), collate_fn=collate_windows)

    model = build_probe(model_cfg).to(device)
    pos_weight = training.get("pos_weight")
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=None if pos_weight is None else torch.tensor(float(pos_weight), device=device)
    )
    optimizer = _build_optimizer(model.head.parameters(), training)

    history: list[dict[str, Any]] = []
    epochs = int(training.get("epochs", 10))
    progress = bool(training.get("progress", True))
    best_val_macro_iou = -1.0
    best_val_epoch: int | None = None
    best_state_dict = None

    epoch_bar = _progress(range(1, epochs + 1), "epochs" if progress else None, unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_metrics = BinaryMetrics()
        batch_bar = _progress(train_loader, f"epoch {epoch}/{epochs} train" if progress else None, unit="batch")
        for batch in batch_bar:
            spatial = batch["spatial"].to(device)
            labels = batch["labels"].to(device)
            logits = model.head(spatial).squeeze(-1)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if progress:
                batch_bar.set_postfix(loss=f"{float(loss.item()):.4f}")
            train_metrics.update(logits.detach(), labels, batch, loss=float(loss.item()))
        train = train_metrics.result()
        val = evaluate_binary(model, val_loader, device, loss_fn=loss_fn,
                               desc=f"epoch {epoch}/{epochs} val" if progress else None)
        history.append({"epoch": epoch, "train": train, "val": val})
        tqdm.write(
            f"epoch {epoch:3d}  loss {train['loss']:.4f}/{val['loss']:.4f}  "
            f"macro_IoU {train['macro_foreground_iou']:.4f}/{val['macro_foreground_iou']:.4f}  "
            f"acc {train['token_accuracy']:.4f}/{val['token_accuracy']:.4f}  [train/val]"
            + ("  *new best val*" if val["macro_foreground_iou"] > best_val_macro_iou else "")
        )
        if val["macro_foreground_iou"] > best_val_macro_iou:
            best_val_macro_iou = val["macro_foreground_iou"]
            best_val_epoch = epoch
            best_state_dict = copy.deepcopy(model.head.state_dict())

    last_state_dict = model.head.state_dict()

    result = {
        "experiment": config.get("experiment", "segmentation"),
        "backbone": config.get("backbone"),
        "probe_cache": {"dir": str(cache_dir), "metadata": cache_metadata},
        "model": model_cfg,
        "is_linear_probe": model.head.is_linear,
        "training": dict(training),
        "seed": seed,
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "history": history,
        "final_train": history[-1]["train"] if history else None,
        "final_val": history[-1]["val"] if history else None,
        "best_val_epoch": best_val_epoch,
        "best_val_macro_iou": best_val_macro_iou,
        "checkpoint_selection": "best_val",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save({"head_state_dict": best_state_dict, "model_config": model_cfg}, output_dir / "head.pt")
    torch.save({"head_state_dict": last_state_dict, "model_config": model_cfg}, output_dir / "head-last.pt")
    result["checkpoint"] = str(output_dir / "head.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    result = train_best_val(config, arguments.output_dir)
    print(
        f"[{result['experiment']}] best_val_epoch={result['best_val_epoch']} "
        f"best_val_macro_iou={result['best_val_macro_iou']:.4f}"
    )


if __name__ == "__main__":
    main()
