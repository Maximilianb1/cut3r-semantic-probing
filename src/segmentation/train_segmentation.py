"""
Train and evaluate the binary segmentation probe over cached features.

This is the shared driver for all three backbones; only the config differs. The
backbone is never run here, and this workspace never extracts features: the
probe-feature cache is an **input** built ahead of time by the Stage 0 tooling
("scripts/extract_probe_features.py"), so training only fits the small MLP head.

Checkpoint selection is controlled by "training.checkpoint_selection" in the
config (or "--checkpoint-selection" to override it): "last" (default) saves the
final epoch's head as head.pt. "best_val" tracks validation macro-IoU across
training and saves the best epoch as head.pt, plus the final epoch as
head-last.pt for reference.

Run example: python -m src.segmentation.train_segmentation --config src/segmentation/configs/cut3r_trained.yaml
Best-val example: python -m src.segmentation.train_segmentation --config src/segmentation/configs/cut3r_trained.yaml \
    --checkpoint-selection best_val --output-dir src/segmentation/experiments/segmentation-cut3r-trained-bestval
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

from src.common.io import load_json, load_yaml

from .model_segmentation import build_probe
from .dataset_segmentation import (
    CombinedProbeCacheDataset,
    ProbeCacheDataset,
    assert_sequence_disjoint,
    collate_windows,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Read a probe config and resolve its ${ENV_VAR} path references.

    The configs point at the probe-feature cache through ${CUT3R_CACHE_ROOT}
    rather than a machine-specific path, so the value has to be expanded before use.
    """
    path = Path(path)
    if path.suffix not in (".yaml", ".yml"):
        raise ValueError(f"Config must be .yaml or .yml, got {path.suffix!r}: {path}")
    return load_yaml(path)


def probe_cache_provenance(cache_dir: str | Path) -> dict[str, Any]:
    """
    The cache's own metadata.json - which backbone/layout produced it.

    Recorded into ``metrics.json`` so a result is always traceable to the exact
    cache it was trained on, instead of trusting the config's ``backbone`` label.
    """
    path = Path(cache_dir) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Probe-feature cache metadata is missing: {path}. This workspace consumes "
            "caches; build one first with scripts/extract_probe_features.py"
        )
    return load_json(path)


def build_datasets(
    config: dict[str, Any],
) -> tuple[ProbeCacheDataset | CombinedProbeCacheDataset, ProbeCacheDataset, dict[str, Any]]:
    """
    Build the train/val probe-cache datasets and their provenance record.

    Reads ``probe_cache.dir`` for val (always a single cache). Train reads the
    same ``dir`` too, unless ``probe_cache.train_dirs`` (a list) is set, in which
    case train is the union of those caches' train rows instead (expanded training:
    original + leftover + cap100-new-train) - val/test stay on ``dir`` alone, so
    the score stays comparable to a single-cache baseline. Asserts train/val stay
    sequence-disjoint either way.

    Returns ``(train_set, val_set, probe_cache_record)``. When ``train_dirs`` is
    absent, ``probe_cache_record`` is exactly ``{"dir": ..., "metadata": ...}``,
    unchanged from before this existed. When present, it is
    ``{"val_dir": {...}, "train_dirs": [{...}, ...]}``.
    """
    splits = config.get("splits", {"train": "train", "val": "val"})
    categories = config.get("categories")
    probe_cache_cfg = config["probe_cache"]
    cache_dir = probe_cache_cfg["dir"]
    train_dirs = probe_cache_cfg.get("train_dirs")
    if train_dirs is not None and not isinstance(train_dirs, (list, tuple)):
        raise TypeError(
            f"probe_cache.train_dirs must be a list/tuple of cache dirs, got {type(train_dirs).__name__}"
        )

    val_set = ProbeCacheDataset(cache_dir, split=splits["val"], categories=categories)
    val_record = {"dir": str(cache_dir), "metadata": probe_cache_provenance(cache_dir)}

    if train_dirs is not None:
        train_set = CombinedProbeCacheDataset(train_dirs, split=splits["train"], categories=categories)
        probe_cache_record = {
            "val_dir": val_record,
            "train_dirs": [
                {"dir": str(d), "metadata": probe_cache_provenance(d)} for d in train_dirs
            ],
        }
    else:
        train_set = ProbeCacheDataset(cache_dir, split=splits["train"], categories=categories)
        probe_cache_record = val_record

    assert_sequence_disjoint(train_set, val_set)
    return train_set, val_set, probe_cache_record


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is false")
    return torch.device(name)


def _split_by_counts(values: torch.Tensor, counts: torch.Tensor) -> list[torch.Tensor]:
    return list(torch.split(values, counts.tolist()))


def _progress(iterable: Any, desc: str | None, **options: Any) -> Any:
    """A tqdm bar when "desc" is given, otherwise the plain iterable.

    Inner bars use leave=False so finished epochs collapse and the per-epoch summary
    lines stay readable. Bars go to stderr, so redirecting stdout still captures the
    summaries cleanly. disable=None makes tqdm draw only on a real terminal, so a
    redirected VM run does not fill its log with thousands of refresh lines.
    """
    if desc is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False, dynamic_ncols=True, disable=None, **options)


_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}


def _build_optimizer(parameters: Any, training: dict[str, Any]) -> torch.optim.Optimizer:
    """
    Optimizer named by "training.optimizer" over the head's parameters only.
    """
    name = str(training.get("optimizer", "adam")).lower()
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {name!r}; supported: {sorted(_OPTIMIZERS)}")
    options: dict[str, Any] = {
        "lr": float(training.get("lr", 1e-3)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
    }
    if name == "sgd":
        options["momentum"] = float(training.get("momentum", 0.0))
    return _OPTIMIZERS[name](parameters, **options)


class BinaryMetrics:
    """
    Streaming loss + token accuracy + foreground IoU over one pass of the data.

    Training and evaluation both feed this, so a train number and a val number are
    produced by identical arithmetic and can be compared directly.

    All metrics are at **token / patch-grid resolution**.
    A token is predicted foreground when its logit > 0.
    """

    def __init__(self, *, collect_windows: bool = False, collect_masks: bool = False) -> None:
        self.collect_windows = collect_windows
        self.collect_masks = collect_masks
        self.per_window_iou: list[float] = []
        self.per_category_iou: dict[str, list[float]] = {}
        self.windows: list[dict[str, Any]] = []
        self.global_intersection = self.global_union = 0.0 # For IoU calculation
        self.correct = self.total = 0 # For token accuracy calculation
        self.loss_sum: float | None = None # Token-weighted, so uneven batches average right

    def update(self, logits: torch.Tensor, labels: torch.Tensor, batch: dict[str, Any],
        *, loss: float | None = None) -> None:
        """Fold one batch in. "loss" is that batch's mean loss, if it was computed."""
        prediction = (logits > 0.0).to(torch.float32)  # fixed 0.5 across backbones, by design: no per-backbone tuning

        # Accuracy
        self.correct += float((prediction == labels).sum().item())
        self.total += int(labels.numel())

        # Loss: batch mean -> batch total, so the pass average is per token
        if loss is not None:
            self.loss_sum = (self.loss_sum or 0.0) + loss * labels.numel()

        # IoU
        preds = _split_by_counts(prediction, batch["counts"])
        gts = _split_by_counts(labels, batch["counts"])
        for position, (pred, gt, category) in enumerate(zip(preds, gts, batch["categories"])):
            intersection = float(((pred == 1) & (gt == 1)).sum().item())
            union = float(((pred == 1) | (gt == 1)).sum().item())
            self.global_intersection += intersection
            self.global_union += union
            iou = 1.0 if union == 0.0 else intersection / union
            self.per_window_iou.append(iou)
            self.per_category_iou.setdefault(category, []).append(iou)
            if self.collect_windows:
                grid = tuple(batch["token_grids"][position])
                record = {
                    "window_id": batch["window_ids"][position],
                    "category": category,
                    "token_grid": list(grid),
                    "foreground_iou": iou,
                }
                if self.collect_masks:
                    record["predicted_labels"] = pred.reshape(grid).cpu()
                    record["target_labels"] = gt.reshape(grid).cpu()
                self.windows.append(record)

    def result(self) -> dict[str, Any]:
        """The aggregated metrics. "loss" appears only if a loss was fed in."""
        macro_iou = sum(self.per_window_iou) / len(self.per_window_iou) if self.per_window_iou else 0.0
        micro_iou = 1.0 if self.global_union == 0.0 else self.global_intersection / self.global_union
        category_iou = {c: sum(v) / len(v) for c, v in self.per_category_iou.items()}
        metrics: dict[str, Any] = {
            "token_accuracy": self.correct / self.total if self.total else 0.0,
            "macro_foreground_iou": macro_iou,
            "micro_foreground_iou": micro_iou,
            "mean_category_iou": (sum(category_iou.values()) / len(category_iou) if category_iou else 0.0),
            "per_category_iou": category_iou,
            "windows": len(self.per_window_iou),
        }
        if self.loss_sum is not None:
            metrics["loss"] = self.loss_sum / self.total if self.total else 0.0
        if self.collect_windows:
            metrics["per_window"] = self.windows
        return metrics


@torch.no_grad()
def evaluate_binary(model: torch.nn.Module, loader: DataLoader, device: torch.device, *,
    loss_fn: torch.nn.Module | None = None, collect_windows: bool = False,
    collect_masks: bool = False, desc: str | None = None) -> dict[str, Any]:
    """
    Evaluate a binary probe over "loader": the same metrics the training pass reports.

    Pass "loss_fn" to also get the split's loss, so train and val loss are directly
    comparable (same criterion, including any pos_weight). "desc" labels a tqdm bar;
    without it the pass is silent.
    """
    model.eval()
    metrics = BinaryMetrics(collect_windows=collect_windows, collect_masks=collect_masks)
    for batch in _progress(loader, desc, unit="batch"):
        spatial = batch["spatial"].to(device)
        labels = batch["labels"].to(device)
        logits = model.head(spatial).squeeze(-1)  # [sum_N]
        loss = None if loss_fn is None else float(loss_fn(logits, labels).item())
        metrics.update(logits, labels, batch, loss=loss)
    return metrics.result()


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Train the binary segmentation probe from a parsed config dict.

    Builds the probe-cache datasets, trains only the MLP head with per-token BCE
    while the backbone stays frozen/precomputed.
    Returns the run record incl. per-epoch history.

    "training.checkpoint_selection" picks what "head.pt" holds: "last" (default)
    is the final epoch's head. "best_val" instead tracks validation macro-IoU
    across training and saves the best epoch as "head.pt", plus the final epoch
    as "head-last.pt" for reference; the run record then also carries
    "best_val_epoch"/"best_val_macro_iou"/"checkpoint_selection".

    Each history entry is {"epoch", "train", "val"} and both sides carry the same
    keys - loss, token_accuracy, macro/micro/mean-category IoU - so the gap between
    them is readable per epoch.
    """
    training = config.get("training", {})
    model_cfg = dict(config["model"])
    device = _resolve_device(str(training.get("device", "cpu")))
    seed = int(training.get("seed", 0))
    torch.manual_seed(seed)

    if int(model_cfg.get("num_classes", 1)) != 1:
        raise NotImplementedError("train_segmentation currently implements the binary (num_classes=1) probe")

    checkpoint_selection = str(training.get("checkpoint_selection", "last")).lower()
    if checkpoint_selection not in ("last", "best_val"):
        raise ValueError(
            f"Unknown training.checkpoint_selection {checkpoint_selection!r}; expected 'last' or 'best_val'"
        )

    train_set, val_set, probe_cache_record = build_datasets(config)

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
        batch_bar = _progress(
            train_loader, f"epoch {epoch}/{epochs} train" if progress else None, unit="batch"
        )
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
            # Train metrics come free from the forward pass that was needed anyway.
            # They are running values - the weights move within the epoch - whereas
            # val is a snapshot of the epoch's final weights, so a small train/val
            # gap early in training is expected and not yet overfitting.
            train_metrics.update(logits.detach(), labels, batch, loss=float(loss.item()))
        train = train_metrics.result()
        val = evaluate_binary(
            model, val_loader, device, loss_fn=loss_fn,
            desc=f"epoch {epoch}/{epochs} val" if progress else None,
        )
        history.append({"epoch": epoch, "train": train, "val": val})
        is_new_best = val["macro_foreground_iou"] > best_val_macro_iou
        # tqdm.write, not print, so the summary does not land on top of a live bar.
        tqdm.write(
            f"epoch {epoch:3d}  loss {train['loss']:.4f}/{val['loss']:.4f}  "
            f"macro_IoU {train['macro_foreground_iou']:.4f}/{val['macro_foreground_iou']:.4f}  "
            f"acc {train['token_accuracy']:.4f}/{val['token_accuracy']:.4f}  [train/val]"
            + ("  *new best val*" if checkpoint_selection == "best_val" and is_new_best else "")
        )
        if checkpoint_selection == "best_val" and is_new_best:
            best_val_macro_iou = val["macro_foreground_iou"]
            best_val_epoch = epoch
            best_state_dict = copy.deepcopy(model.head.state_dict())

    result = {
        "experiment": config.get("experiment", "segmentation"),
        "backbone": config.get("backbone"),
        "probe_cache": probe_cache_record,
        "model": model_cfg,
        "is_linear_probe": model.head.is_linear,
        # The full training block, so metrics.json records the optimizer, lr,
        # schedule-free epochs and pos_weight the numbers came from.
        "training": dict(training),
        "seed": seed,
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "history": history,
        "final_train": history[-1]["train"] if history else None,
        "final_val": history[-1]["val"] if history else None,
    }
    if checkpoint_selection == "best_val":
        if best_state_dict is None:
            raise RuntimeError(
                "checkpoint_selection='best_val' but no epoch produced a valid checkpoint "
                "(best_val_macro_iou stayed at its -1.0 initial value); check training.epochs "
                "and val metrics for NaN or a zero-epoch run"
            )
        result["best_val_epoch"] = best_val_epoch
        result["best_val_macro_iou"] = best_val_macro_iou
        result["checkpoint_selection"] = "best_val"

    output_dir = config.get("output", {}).get("dir")
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        # Save the trained head so inference_segmentation.py can reload the probe.
        if checkpoint_selection == "best_val":
            torch.save({"head_state_dict": best_state_dict, "model_config": model_cfg}, path / "head.pt")
            torch.save(
                {"head_state_dict": model.head.state_dict(), "model_config": model_cfg}, path / "head-last.pt"
            )
        else:
            torch.save({"head_state_dict": model.head.state_dict(), "model_config": model_cfg}, path / "head.pt")
        result["checkpoint"] = str(path / "head.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config (.yaml/.yml)")
    parser.add_argument(
        "--checkpoint-selection", choices=["last", "best_val"], default=None,
        help="Override the config's training.checkpoint_selection (default: last).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override the config's output.dir."
    )
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.checkpoint_selection is not None:
        config.setdefault("training", {})["checkpoint_selection"] = arguments.checkpoint_selection
    if arguments.output_dir is not None:
        config.setdefault("output", {})["dir"] = str(arguments.output_dir)
    result = train_from_config(config)
    if result.get("checkpoint_selection") == "best_val":
        print(
            f"[{result['experiment']}] best_val_epoch={result['best_val_epoch']} "
            f"best_val_macro_iou={result['best_val_macro_iou']:.4f}"
        )


if __name__ == "__main__":
    main()
