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

Run example: python -m src.segmentation.train_segmentation --config src/segmentation/configs/cut3r_trained_mlp.yaml
Best-val example: python -m src.segmentation.train_segmentation --config src/segmentation/configs/cut3r_trained_mlp.yaml \
    --checkpoint-selection best_val
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.io import load_json, load_yaml

from .curve_metrics import average_precision, roc_auc
from .model_segmentation import build_probe, feature_statistics_from_moments
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
            f"Probe-feature cache metadata is missing: {path}. This package consumes "
            "caches; build one first with scripts/extract_probe_features.py"
        )
    return load_json(path)


def probe_cache_provenance_record(config: dict[str, Any]) -> dict[str, Any]:
    """
    Describe which cache(s) a run's ``probe_cache`` config points at, for
    recording into ``metrics.json`` / ``inference-<split>.json``. Same shape
    ``build_datasets`` uses for its record: ``{"dir": ..., "metadata": ...}``
    normally, ``{"cache_dirs": [...], "split_override_path": ...}`` in
    shared-split mode.
    """
    probe_cache_cfg = config["probe_cache"]
    cache_dirs = probe_cache_cfg.get("cache_dirs")
    if cache_dirs is not None:
        return {
            "cache_dirs": [str(d) for d in cache_dirs],
            "split_override_path": probe_cache_cfg.get("split_override_path"),
        }
    cache_dir = probe_cache_cfg["dir"]
    return {"dir": str(cache_dir), "metadata": probe_cache_provenance(cache_dir)}


def _load_split_override(path: str | Path | None) -> dict[str, str] | None:
    """Load a ``{sequence_id: split}`` patch written by
    ``scripts/derive_segmentation_split.py`` (see ``src/data/split_overrides.py``)."""
    if path is None:
        return None
    payload = load_json(path)
    override = payload.get("split_override")
    if not isinstance(override, dict):
        raise ValueError(f"{path} has no 'split_override' mapping")
    return {str(sequence_id): str(split) for sequence_id, split in override.items()}


def resolve_probe_cache_dataset(
    config: dict[str, Any], *, split: str, categories: list[str] | None
) -> ProbeCacheDataset | CombinedProbeCacheDataset:
    """
    Build the dataset for one split, honoring ``probe_cache.cache_dirs`` /
    ``probe_cache.split_override_path`` when set - the general-purpose version
    of the per-mode logic ``build_datasets`` inlines for train/val. Used by
    ``inference_segmentation.py`` too, so the test split and the leak check both
    see the same pooled cache set and the same override as training did.

    ``probe_cache.cache_dirs`` (all available caches, pooled - e.g. original +
    leftover + cap100-new-train) makes every split searchable across every cache,
    which a promoted sequence may require: ``split_override_path`` can relabel a
    sequence that physically lives in a train-only cache (like cap100-new-train)
    as ``test``, and a single ``dir`` would never find its rows. Absent
    ``cache_dirs``, this falls back to ``probe_cache.train_dirs`` for the
    configured train split (unchanged expanded-training behavior) and to
    ``probe_cache.dir`` otherwise - the two pre-existing modes.
    """
    probe_cache_cfg = config["probe_cache"]
    split_override = _load_split_override(probe_cache_cfg.get("split_override_path"))
    cache_dirs = probe_cache_cfg.get("cache_dirs")
    if split_override is not None and cache_dirs is None:
        raise ValueError(
            "probe_cache.split_override_path requires probe_cache.cache_dirs "
            "(a sequence promoted into a split may live in any pooled cache, "
            "not just probe_cache.dir)"
        )
    if cache_dirs is not None:
        return CombinedProbeCacheDataset(
            cache_dirs, split=split, categories=categories, split_override=split_override
        )
    train_split = (config.get("splits") or {}).get("train", "train")
    train_dirs = probe_cache_cfg.get("train_dirs")
    if train_dirs is not None and split == train_split:
        return CombinedProbeCacheDataset(train_dirs, split=split, categories=categories)
    return ProbeCacheDataset(probe_cache_cfg["dir"], split=split, categories=categories)


def build_datasets(
    config: dict[str, Any],
) -> tuple[ProbeCacheDataset | CombinedProbeCacheDataset, ProbeCacheDataset, dict[str, Any]]:
    """
    Build the train/val probe-cache datasets and their provenance record.

    Three modes, picked by what ``probe_cache`` sets:

    - ``dir`` alone (original behavior): train and val both read the single cache.
    - ``dir`` + ``train_dirs`` (expanded training): train is the union of those
      caches' train rows (original + leftover + cap100-new-train); val stays on
      ``dir`` alone so the score stays comparable to a single-cache baseline.
    - ``dir`` + ``cache_dirs`` (+ optional ``split_override_path``): train and val
      both pool every cache in ``cache_dirs`` and, if given, relabel sequences per
      the override - this is the shared-split mode, for reusing classification's
      train/val/test sequence membership (plus segmentation's own per-category
      floor) instead of the caches' own recorded split. See
      ``resolve_probe_cache_dataset`` and ``scripts/derive_segmentation_split.py``.

    Asserts train/val stay sequence-disjoint in every mode.

    Returns ``(train_set, val_set, probe_cache_record)``. Record shape depends on
    mode: ``{"dir": ..., "metadata": ...}`` for the first, ``{"val_dir": {...},
    "train_dirs": [{...}, ...]}`` for the second, ``{"cache_dirs": [...],
    "split_override_path": ...}`` for the third.
    """
    splits = config.get("splits", {"train": "train", "val": "val"})
    categories = config.get("categories")
    probe_cache_cfg = config["probe_cache"]
    train_dirs = probe_cache_cfg.get("train_dirs")
    cache_dirs = probe_cache_cfg.get("cache_dirs")
    if train_dirs is not None and not isinstance(train_dirs, (list, tuple)):
        raise TypeError(
            f"probe_cache.train_dirs must be a list/tuple of cache dirs, got {type(train_dirs).__name__}"
        )
    if cache_dirs is not None and not isinstance(cache_dirs, (list, tuple)):
        raise TypeError(
            f"probe_cache.cache_dirs must be a list/tuple of cache dirs, got {type(cache_dirs).__name__}"
        )
    if train_dirs is not None and cache_dirs is not None:
        raise ValueError("probe_cache.train_dirs and probe_cache.cache_dirs are mutually exclusive")

    if cache_dirs is not None:
        # No probe_cache.dir needed in this mode: every split is read from the
        # pooled cache_dirs instead.
        train_set = resolve_probe_cache_dataset(config, split=splits["train"], categories=categories)
        val_set = resolve_probe_cache_dataset(config, split=splits["val"], categories=categories)
        probe_cache_record = {
            "cache_dirs": [str(d) for d in cache_dirs],
            "split_override_path": probe_cache_cfg.get("split_override_path"),
        }
        assert_sequence_disjoint(train_set, val_set)
        return train_set, val_set, probe_cache_record

    cache_dir = probe_cache_cfg["dir"]
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
    Streaming loss + token accuracy + foreground/background IoU + confusion
    counts over one pass of the data.

    Training and evaluation both feed this, so a train number and a val number are
    produced by identical arithmetic and can be compared directly.

    All metrics are at **token / patch-grid resolution**.
    A token is predicted foreground when its logit > 0.

    Foreground IoU (macro/micro/per-category) is the project's own established
    metric and stays as-is. Alongside it this also tracks the background class
    symmetrically, so a standard 2-class mIoU (``mean_iou``) and mAcc
    (``mean_class_accuracy``, the mean of foreground- and background-recall) can
    be reported without depending on ``token_accuracy`` - which is dominated by
    the always-large background class and known to move opposite real IoU.
    Raw ``tp``/``fp``/``fn``/``tn`` are also kept so precision/recall/mAcc can be
    recomputed post-hoc without needing ``--save-masks``.

    Also accumulates raw logits (pooled, and per-category) to report AUC-ROC
    and average precision (``curve_metrics.py``) - threshold-free counterparts
    to the IoU/accuracy metrics above, which all depend on the fixed logit > 0
    cutoff. Since that cutoff is never tuned per backbone, AUC isolates "is the
    information in the embedding" from "is 0 the right cutoff for this
    backbone's raw score scale."
    """

    def __init__(self, *, collect_windows: bool = False, collect_masks: bool = False) -> None:
        self.collect_windows = collect_windows
        self.collect_masks = collect_masks
        self.per_window_iou: list[float] = []
        self.per_category_iou: dict[str, list[float]] = {}
        self.per_window_background_iou: list[float] = []
        self.per_category_background_iou: dict[str, list[float]] = {}
        self.windows: list[dict[str, Any]] = []
        self.global_intersection = self.global_union = 0.0 # For foreground IoU
        self.global_bg_intersection = self.global_bg_union = 0.0 # For background IoU
        self.correct = self.total = 0 # For token accuracy calculation
        self.tp = self.fp = self.fn = self.tn = 0 # Confusion counts, foreground = positive
        self.loss_sum: float | None = None # Token-weighted, so uneven batches average right
        # Raw logits, kept on-device and concatenated/moved to CPU once in
        # result() rather than per batch here - a per-batch .cpu() would add a
        # synchronization to every training step for a number only read once
        # per pass.
        self.scores: list[torch.Tensor] = []
        self.score_labels: list[torch.Tensor] = []
        self.category_scores: dict[str, list[torch.Tensor]] = {}
        self.category_labels: dict[str, list[torch.Tensor]] = {}

    def update(self, logits: torch.Tensor, labels: torch.Tensor, batch: dict[str, Any],
        *, loss: float | None = None) -> None:
        """Fold one batch in. "loss" is that batch's mean loss, if it was computed."""
        prediction = (logits > 0.0).to(torch.float32)  # fixed 0.5 across backbones, by design: no per-backbone tuning

        # Accuracy + confusion counts
        self.correct += float((prediction == labels).sum().item())
        self.total += int(labels.numel())
        self.tp += int(((prediction == 1) & (labels == 1)).sum().item())
        self.fp += int(((prediction == 1) & (labels == 0)).sum().item())
        self.fn += int(((prediction == 0) & (labels == 1)).sum().item())
        self.tn += int(((prediction == 0) & (labels == 0)).sum().item())

        # Loss: batch mean -> batch total, so the pass average is per token
        if loss is not None:
            self.loss_sum = (self.loss_sum or 0.0) + loss * labels.numel()

        # Raw scores for AUC-ROC / AUC-PR - pooled globally and per-category.
        # Stays on-device here; see the on-device comment in __init__.
        self.scores.append(logits.detach())
        self.score_labels.append(labels.detach())

        # IoU
        preds = _split_by_counts(prediction, batch["counts"])
        gts = _split_by_counts(labels, batch["counts"])
        score_windows = _split_by_counts(logits.detach(), batch["counts"])
        for position, (pred, gt, score, category) in enumerate(
            zip(preds, gts, score_windows, batch["categories"])
        ):
            self.category_scores.setdefault(category, []).append(score)
            self.category_labels.setdefault(category, []).append(gt)
            intersection = float(((pred == 1) & (gt == 1)).sum().item())
            union = float(((pred == 1) | (gt == 1)).sum().item())
            self.global_intersection += intersection
            self.global_union += union
            iou = 1.0 if union == 0.0 else intersection / union
            self.per_window_iou.append(iou)
            self.per_category_iou.setdefault(category, []).append(iou)

            bg_intersection = float(((pred == 0) & (gt == 0)).sum().item())
            bg_union = float(((pred == 0) | (gt == 0)).sum().item())
            self.global_bg_intersection += bg_intersection
            self.global_bg_union += bg_union
            bg_iou = 1.0 if bg_union == 0.0 else bg_intersection / bg_union
            self.per_window_background_iou.append(bg_iou)
            self.per_category_background_iou.setdefault(category, []).append(bg_iou)

            if self.collect_windows:
                grid = tuple(batch["token_grids"][position])
                record = {
                    "window_id": batch["window_ids"][position],
                    "sequence_id": batch["sequence_ids"][position],
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

        macro_bg_iou = (sum(self.per_window_background_iou) / len(self.per_window_background_iou)
                         if self.per_window_background_iou else 0.0)
        micro_bg_iou = 1.0 if self.global_bg_union == 0.0 else self.global_bg_intersection / self.global_bg_union
        category_bg_iou = {c: sum(v) / len(v) for c, v in self.per_category_background_iou.items()}

        foreground_precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        foreground_recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        background_recall = self.tn / (self.tn + self.fp) if (self.tn + self.fp) else 0.0

        # Single CPU transfer for the whole pass, not one per batch (see __init__).
        all_scores = torch.cat(self.scores).cpu().numpy() if self.scores else np.array([])
        all_score_labels = torch.cat(self.score_labels).cpu().numpy() if self.score_labels else np.array([])
        auroc = roc_auc(all_scores, all_score_labels)
        auprc = average_precision(all_scores, all_score_labels)

        category_score_arrays = {
            c: (torch.cat(self.category_scores[c]).cpu().numpy(), torch.cat(self.category_labels[c]).cpu().numpy())
            for c in self.category_scores
        }
        category_auroc = {c: roc_auc(scores, labels_) for c, (scores, labels_) in category_score_arrays.items()}
        category_auprc = {
            c: average_precision(scores, labels_) for c, (scores, labels_) in category_score_arrays.items()
        }
        # A category with only one class present (e.g. an all-background test
        # window) has undefined AUC - excluded from the macro average rather
        # than treated as 0 or 1, since neither is a meaningful score there.
        valid_category_auroc = [v for v in category_auroc.values() if not np.isnan(v)]
        valid_category_auprc = [v for v in category_auprc.values() if not np.isnan(v)]

        metrics: dict[str, Any] = {
            "token_accuracy": self.correct / self.total if self.total else 0.0,
            "macro_foreground_iou": macro_iou,
            "micro_foreground_iou": micro_iou,
            "mean_category_iou": (sum(category_iou.values()) / len(category_iou) if category_iou else 0.0),
            "per_category_iou": category_iou,
            "macro_background_iou": macro_bg_iou,
            "micro_background_iou": micro_bg_iou,
            "mean_category_background_iou": (
                sum(category_bg_iou.values()) / len(category_bg_iou) if category_bg_iou else 0.0
            ),
            "per_category_background_iou": category_bg_iou,
            # Standard 2-class mIoU: macro-averaged, foreground and background weighted equally.
            "mean_iou": (macro_iou + macro_bg_iou) / 2,
            "foreground_precision": foreground_precision,
            "foreground_recall": foreground_recall,
            "background_recall": background_recall,
            # mAcc: mean of per-class recall, unlike token_accuracy which is
            # dominated by whichever class has more tokens.
            "mean_class_accuracy": (foreground_recall + background_recall) / 2,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            # Threshold-free: pooled over every test token, and per-category
            # macro-averaged (mirrors macro-IoU) so large categories don't dominate.
            "auroc": auroc,
            "auprc": auprc,
            "per_category_auroc": category_auroc,
            "per_category_auprc": category_auprc,
            "mean_category_auroc": (
                sum(valid_category_auroc) / len(valid_category_auroc) if valid_category_auroc else float("nan")
            ),
            "mean_category_auprc": (
                sum(valid_category_auprc) / len(valid_category_auprc) if valid_category_auprc else float("nan")
            ),
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
        logits = model(spatial).squeeze(-1)  # [sum_N]
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
        raise NotImplementedError("train_segmentation implements the binary (num_classes=1) probe")

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

    progress = bool(training.get("progress", True))
    model = build_probe(model_cfg).to(device)

    # Standardization statistics from the TRAIN split only, over individual tokens
    # (not pooled per-window vectors, unlike classification) - segmentation trains
    # per token. Computing them over train+val, or per split at evaluation time,
    # would leak the evaluated split into its own score.
    #
    # Streamed batch-by-batch (running sum / sum-of-squares) rather than
    # concatenated into one [total_tokens, D] tensor first: a full train split is
    # millions of tokens, so materializing them all at once would be many GB.
    statistics_loader = DataLoader(
        train_set,
        batch_size=int(training.get("batch_size", 16)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
        collate_fn=collate_windows,
    )
    feature_dim = int(model_cfg["feature_dim"])
    token_sum = torch.zeros(feature_dim, dtype=torch.float64)
    token_sumsq = torch.zeros(feature_dim, dtype=torch.float64)
    token_count = 0
    for batch in _progress(
        statistics_loader, "feature statistics" if progress else None, unit="batch"
    ):
        spatial = batch["spatial"].to(torch.float64)
        token_sum += spatial.sum(dim=0)
        token_sumsq += (spatial * spatial).sum(dim=0)
        token_count += spatial.shape[0]
    mean, std = feature_statistics_from_moments(token_sum, token_sumsq, token_count)
    model.set_feature_statistics(mean.to(device), std.to(device))

    pos_weight = training.get("pos_weight")
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=None if pos_weight is None else torch.tensor(float(pos_weight), device=device)
    )
    optimizer = _build_optimizer(model.head.parameters(), training)

    history: list[dict[str, Any]] = []
    epochs = int(training.get("epochs", 10))
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
            logits = model(spatial).squeeze(-1)
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
        # The normalization statistics travel with the weights (unchanged across
        # epochs, since they're computed once from the train split before training
        # starts), so evaluation applies the same transform instead of recomputing
        # it from the evaluated split.
        checkpoint_extra = {
            "feature_mean": model.feature_mean.detach().cpu(),
            "feature_std": model.feature_std.detach().cpu(),
        }
        if checkpoint_selection == "best_val":
            torch.save(
                {"head_state_dict": best_state_dict, "model_config": model_cfg, **checkpoint_extra},
                path / "head.pt",
            )
            torch.save(
                {"head_state_dict": model.head.state_dict(), "model_config": model_cfg, **checkpoint_extra},
                path / "head-last.pt",
            )
        else:
            torch.save(
                {"head_state_dict": model.head.state_dict(), "model_config": model_cfg, **checkpoint_extra},
                path / "head.pt",
            )
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
