"""
Train and evaluate the multiclass classification probe over cached features.

This is the shared driver for all three backbones; only the config differs. The
backbone is never run here, and this package never extracts features: the
probe-feature cache is an **input** built ahead of time by the Stage 0 tooling
("scripts/extract_probe_features.py"), so training only fits the small MLP head.

There is one config per backbone **and arm** - the arm is the open ADR 0003 comparison,
so a run has to name it.

Run example: python -m src.classification.train_classification --config src/classification/configs/cut3r_trained_state.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.common.io import load_json, load_yaml

from .model_classification import (
    FeatureSpec,
    build_probe,
    feature_statistics,
    resolve_features,
)
from .dataset_classification import (
    ProbeCacheClassificationDataset,
    assert_sequence_disjoint,
    cache_categories,
    category_names,
)

# Optimizers selectable from the config; an unknown name is rejected, never defaulted.
_OPTIMIZERS = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
_CHECKPOINT_SELECTIONS = {"final", "best_val_macro_f1"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a probe config and resolve its ${ENV_VAR} path references."""
    path = Path(path)
    if path.suffix not in (".yaml", ".yml"):
        raise ValueError(f"Config must be .yaml or .yml, got {path.suffix!r}: {path}")
    return load_yaml(path)


def probe_cache_provenance(cache_dir: str | Path) -> dict[str, Any]:
    """The cache's own metadata.json - which backbone/layout produced it."""
    path = Path(cache_dir) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Probe-feature cache metadata is missing: {path}. This package consumes "
            "caches; build one first with scripts/extract_probe_features.py"
        )
    return load_json(path)


def _resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is false")
    return torch.device(name)


def _progress(iterable: Any, desc: str | None, **options: Any) -> Any:
    """A tqdm bar when "desc" is given, otherwise the plain iterable."""
    if desc is None:
        return iterable
    return tqdm(iterable, desc=desc, leave=False, dynamic_ncols=True, disable=None, **options)


def build_balanced_sampler(dataset: ProbeCacheClassificationDataset, generator: torch.Generator | None = None) -> WeightedRandomSampler:
    """Sample windows with probability 1 / (windows in that category).
    Drawn **with replacement** over len(dataset).

    Sampling touches the training split only. Val and test must keep the real
    distribution, or their metrics would describe a population that does not exist.
    """
    counts = dataset.class_counts()
    weights = [1.0 / counts[row["category"]] for row in dataset.rows]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


def run_directory(config: dict[str, Any], features: FeatureSpec) -> Path | None:
    """Where a run's files go: ``<output.dir>/<features.source>/<experiment>``.

    The feature source is part of the path rather than something to remember, so the two
    arms of the comparison cannot land in the same directory and overwrite each other -
    and a directory always says which representation produced what is in it.

    Returns ``None`` when the config sets no ``output.dir``, which means "do not write".
    """
    base = config.get("output", {}).get("dir")
    if not base:
        return None
    return Path(base) / features["source"] / str(config.get("experiment", "classification"))


def resolve_label_space(config: dict[str, Any], cache_dir: str | Path) -> tuple[str, ...]:
    """The categories the head's outputs stand for, named by ``model.label_space``.

    - ``vocabulary`` (default): all 51 CO3D categories. Labels are vocabulary indices, so
      a head trained on one cache is directly comparable to a head trained on another -
      including a later cache that adds categories.
    - ``present``: only the categories this cache holds, relabelled contiguously. The
      correctly specified model when a cache covers a subset, at the cost of that
      comparability, so it is opt-in rather than the default.

    Measured on the Part-A cache (26 of 51 categories), ``present`` is worth roughly
    +0.006 val macro F1 and leaves the train/val gap unchanged: the outputs it removes
    were never discriminating between the categories that are there.
    """
    mode = str((config.get("model") or {}).get("label_space", "vocabulary"))
    if mode == "vocabulary":
        return category_names()
    if mode == "present":
        return cache_categories(cache_dir)
    raise ValueError(
        f"Unknown model.label_space {mode!r}; supported: 'vocabulary' (all 51) or "
        "'present' (the categories this cache holds)"
    )


def resolve_model_config(config: dict[str, Any], label_space: Sequence[str]) -> dict[str, Any]:
    """
    The model block with num_classes filled in from the label space.
    Shared by training and inference, so both build the same head from the same config.
    """
    model_cfg = dict(config["model"])
    model_cfg.pop("label_space", None)  # selects the space; not a head hyperparameter
    size = len(label_space)
    model_cfg.setdefault("num_classes", size)
    if int(model_cfg["num_classes"]) != size:
        raise ValueError(
            f"model.num_classes is {model_cfg['num_classes']} but labels are indices into "
            f"a {size}-category label space; leave it unset or set it to {size}"
        )
    return model_cfg


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


class MulticlassMetrics:
    """
    Streaming loss, accuracy, recall, and F1 over one pass of the data.

    Reported, and why each earns its place in single-label multiclass:

    - accuracy: fraction of windows whose top choice is right.
    - macro_recall: mean per-category recall, i.e. balanced accuracy.
                    Says whether every category is found, not just the common ones.
    - macro_precision: mean per-category precision.
    - macro_f1: mean of the **per-category** F1s
    - per_category_{precision,recall,f1}: the breakdown behind the averages.
    """

    def __init__(self, *, top_k: int = 5, collect_windows: bool = False,
                 vocabulary: Sequence[str] | None = None) -> None:
        self.top_k = top_k
        self.collect_windows = collect_windows # Keep window-level metrics
        self.correct = self.total = 0
        self.top_k_correct = 0
        self.loss_sum: float | None = None  # example-weighted; batches can differ in size
        self.true_positive: dict[str, int] = {}   # predicted c and it was c
        self.support: dict[str, int] = {}         # windows whose true category is c
        self.predicted_count: dict[str, int] = {}  # times c was predicted (precision's denominator)
        self.confusion: dict[tuple[int, int], int] = {}
        self.windows: list[dict[str, Any]] = []
        # Resolved once per pass, not once per batch. Names the head's outputs, so it
        # must be the label space training used - not always the full vocabulary.
        self.vocabulary = tuple(vocabulary) if vocabulary is not None else category_names()

    def update(self, logits: torch.Tensor, labels: torch.Tensor, batch: dict[str, Any],
        *, loss: float | None = None) -> None:
        """Update the metrics per batch."""
        ranked = logits.topk(min(self.top_k, logits.shape[1]), dim=1).indices  # [B, k]
        predicted = ranked[:, 0]
        probabilities = torch.softmax(logits, dim=1) if self.collect_windows else None
        hit = (predicted == labels)
        self.correct += int(hit.sum().item())
        self.total += int(labels.numel())
        self.top_k_correct += int((ranked == labels.unsqueeze(1)).any(dim=1).sum().item())

        # Mean loss -> total, so the pass average is per example
        if loss is not None:
            self.loss_sum = (self.loss_sum or 0.0) + loss * labels.numel()

        for position, category in enumerate(batch["category"]):
            predicted_category = self.vocabulary[int(predicted[position])]
            self.support[category] = self.support.get(category, 0) + 1
            self.predicted_count[predicted_category] = (self.predicted_count.get(predicted_category, 0) + 1)
            if bool(hit[position]):
                self.true_positive[category] = self.true_positive.get(category, 0) + 1
            key = (int(labels[position]), int(predicted[position]))
            self.confusion[key] = self.confusion.get(key, 0) + 1
            if self.collect_windows:
                assert probabilities is not None
                class_probabilities = {
                    name: float(probabilities[position, class_index])
                    for class_index, name in enumerate(self.vocabulary)
                }
                self.windows.append({
                    "window_id": batch["window_id"][position],
                    "sequence_id": batch["sequence_id"][position],
                    "category": category,
                    "predicted_category": predicted_category,
                    "predicted_probability": class_probabilities[predicted_category],
                    "class_probabilities": class_probabilities,
                    "correct": bool(hit[position]),
                })

    def result(self) -> dict[str, Any]:
        """The aggregated metrics. "loss" appears only if a loss was fed in."""
        precision: dict[str, float] = {}
        recall: dict[str, float] = {}
        f1: dict[str, float] = {}
        for category, support in sorted(self.support.items()):
            hits = self.true_positive.get(category, 0)
            predicted = self.predicted_count.get(category, 0)
            p = hits / predicted if predicted else 0.0
            r = hits / support  # support > 0 by construction
            precision[category] = p
            recall[category] = r
            f1[category] = 2 * p * r / (p + r) if (p + r) else 0.0

        def mean(values: dict[str, float]) -> float:
            return sum(values.values()) / len(values) if values else 0.0

        metrics: dict[str, Any] = {
            "accuracy": self.correct / self.total if self.total else 0.0,
            f"top{self.top_k}_accuracy": self.top_k_correct / self.total if self.total else 0.0,
            "macro_precision": mean(precision),
            "macro_recall": mean(recall),  # == balanced accuracy
            "macro_f1": mean(f1),  # mean of per-category F1s, not F1 of the macro means
            "per_category_precision": precision,
            "per_category_recall": recall,
            "per_category_f1": f1,
            "categories_present": len(recall),
            "windows": self.total,
        }
        if self.loss_sum is not None:
            metrics["loss"] = self.loss_sum / self.total if self.total else 0.0
        if self.collect_windows:
            metrics["per_window"] = self.windows
            metrics["confusion"] = [
                {"true": true, "predicted": predicted, "count": count}
                for (true, predicted), count in sorted(self.confusion.items())
            ]
        return metrics


@torch.no_grad()
def evaluate_multiclass(model: torch.nn.Module, loader: DataLoader, device: torch.device, *,
    loss_fn: torch.nn.Module | None = None, top_k: int = 5, collect_windows: bool = False,
    desc: str | None = None, vocabulary: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Evaluate a classification probe over "loader": the same metrics the training pass reports.
    """
    model.eval()
    metrics = MulticlassMetrics(top_k=top_k, collect_windows=collect_windows,
                                vocabulary=vocabulary)
    for batch in _progress(loader, desc, unit="batch"):
        pooled = batch["features"].to(device)
        labels = batch["label"].to(device)
        logits = model(pooled)  # [B, num_classes]
        loss = None if loss_fn is None else float(loss_fn(logits, labels).item())
        metrics.update(logits.cpu(), labels.cpu(), batch, loss=loss)
    return metrics.result()


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Train the multiclass classification probe from a parsed config dict.
    """
    training = config.get("training", {})
    splits = config.get("splits", {"train": "train", "val": "val"})
    features = resolve_features(config)
    source, pooling = features["source"], features["pooling"]
    device = _resolve_device(str(training.get("device", "cpu")))
    seed = int(training.get("seed", 0))
    torch.manual_seed(seed)

    cache_dir = config["probe_cache"]["dir"]
    categories = config.get("categories")
    # Read up front, so a cache without provenance fails now rather than after every
    # epoch has already run.
    cache_metadata = probe_cache_provenance(cache_dir)
    label_space = resolve_label_space(config, cache_dir)
    model_cfg = resolve_model_config(config, label_space)
    dataset_options = {"source": source, "pooling": pooling, "categories": categories,
                       "label_space": label_space}
    train_set = ProbeCacheClassificationDataset(cache_dir, split=splits["train"], **dataset_options)
    val_set = ProbeCacheClassificationDataset(cache_dir, split=splits["val"], **dataset_options)
    assert_sequence_disjoint(train_set, val_set)
    if bool(training.get("preload_features", False)):
        # Pool each frozen representation once. Re-reading a 768x768 state tensor on
        # every epoch is unnecessary when the resulting 768-vector is only 3 KiB.
        train_set.preload_features()
        val_set.preload_features()

    loader_options = {
        "batch_size": int(training.get("batch_size", 32)),
        "num_workers": int(training.get("num_workers", 0)),
    }
    # A sampler replaces shuffling, so the two are mutually exclusive. Training only:
    # val and test keep the real category distribution.
    balanced = bool(training.get("balanced_sampler", False))
    sampler = build_balanced_sampler(train_set) if balanced else None
    train_loader = DataLoader(train_set, shuffle=not balanced, sampler=sampler, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    model = build_probe(model_cfg, normalize=features["normalize"]).to(device)
    if features["normalize"] != "none":
        # Statistics from the TRAIN split only. Computing them over train+val, or per
        # split at evaluation time, would leak the evaluated split into its own score.
        #
        # Deliberately NOT train_loader: with balanced_sampler that draws with
        # replacement, so the statistics would describe the resampled distribution -
        # rare windows counted several times, others missing - and head.pt would carry
        # numbers that depend on the sampler rather than on the data. One sequential
        # pass over the dataset visits each window exactly once.
        statistics_loader = DataLoader(train_set, shuffle=False, **loader_options)
        pooled = torch.cat(
            [batch["features"] for batch in _progress(
                statistics_loader,
                "feature statistics" if training.get("progress", True) else None,
                unit="batch")],
            dim=0,
        )
        mean, std = feature_statistics(pooled)
        model.set_feature_statistics(mean.to(device), std.to(device))
    # Class imbalance is handled by training.balanced_sampler, not by weighting the
    # loss: two corrections for one problem would compound.
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=float(training.get("label_smoothing", 0.0))
    )
    optimizer = _build_optimizer(model.head.parameters(), training)
    top_k = int(training.get("top_k", 5))

    history: list[dict[str, Any]] = []
    epochs = int(training.get("epochs", 10))
    if epochs < 1:
        raise ValueError("training.epochs must be at least 1")
    progress = bool(training.get("progress", True))
    checkpoint_policy = str(training.get("checkpoint_selection", "final"))
    if checkpoint_policy not in _CHECKPOINT_SELECTIONS:
        raise ValueError(
            f"Unknown training.checkpoint_selection {checkpoint_policy!r}; supported: "
            f"{sorted(_CHECKPOINT_SELECTIONS)}"
        )
    selected_head_state: dict[str, torch.Tensor] | None = None
    selected_epoch: int | None = None
    selected_val: dict[str, Any] | None = None
    for epoch in _progress(range(1, epochs + 1), "epochs" if progress else None, unit="epoch"):
        model.train()
        train_metrics = MulticlassMetrics(top_k=top_k, vocabulary=label_space)
        batch_bar = _progress(
            train_loader, f"epoch {epoch}/{epochs} train" if progress else None, unit="batch"
        )
        for batch in batch_bar:
            pooled = batch["features"].to(device)
            labels = batch["label"].to(device)
            logits = model(pooled)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if progress:
                batch_bar.set_postfix(loss=f"{float(loss.item()):.4f}")
            # Train metrics come free from the forward pass that was needed anyway.
            # They are running values - the weights move within the epoch - whereas val
            # is a snapshot of the epoch's final weights.
            train_metrics.update(logits.detach().cpu(), labels.cpu(), batch,
                                 loss=float(loss.item()))
        train = train_metrics.result()
        val = evaluate_multiclass(
            model, val_loader, device, loss_fn=loss_fn, top_k=top_k,
            desc=f"epoch {epoch}/{epochs} val" if progress else None,
            vocabulary=label_space,
        )
        history.append({"epoch": epoch, "train": train, "val": val})
        if checkpoint_policy == "best_val_macro_f1" and (
            selected_val is None or val["macro_f1"] > selected_val["macro_f1"]
        ):
            # state_dict tensors otherwise keep referencing the live module and would
            # silently become the final epoch as training continues. Clone to CPU now.
            selected_head_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.head.state_dict().items()
            }
            selected_epoch = epoch
            selected_val = val
        # tqdm.write, not print, so the summary does not land on top of a live bar.
        tqdm.write(
            f"epoch {epoch:3d}  loss {train['loss']:.4f}/{val['loss']:.4f}  "
            f"acc {train['accuracy']:.4f}/{val['accuracy']:.4f}  "
            f"macroF1 {train['macro_f1']:.4f}/{val['macro_f1']:.4f}  "
            f"recall {train['macro_recall']:.4f}/{val['macro_recall']:.4f}  [train/val]"
        )

    if checkpoint_policy == "final":
        selected_head_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.head.state_dict().items()
        }
        selected_epoch = history[-1]["epoch"]
        selected_val = history[-1]["val"]
    if selected_head_state is None or selected_epoch is None or selected_val is None:
        raise RuntimeError("No checkpoint was selected during training")

    selection = {
        "policy": checkpoint_policy,
        "epoch": selected_epoch,
        "val_macro_f1": selected_val["macro_f1"],
        "val_accuracy": selected_val["accuracy"],
    }
    result = {
        "experiment": config.get("experiment", "classification"),
        "backbone": config.get("backbone"),
        "probe_cache": {"dir": str(cache_dir), "metadata": cache_metadata},
        # The representation this result came from - the open Stage 2 comparison, so it
        # is recorded rather than assumed.
        "features": dict(features),
        "model": model_cfg,
        # What each output stands for. Without it a 26-way head's logits are unreadable.
        "label_space": list(label_space),
        "is_linear_probe": model.head.is_linear,
        "training": dict(training),
        "seed": seed,
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "train_class_counts": train_set.class_counts(),
        "history": history,
        "final_train": history[-1]["train"] if history else None,
        "final_val": history[-1]["val"] if history else None,
        "checkpoint_selection": selection,
    }
    path = run_directory(config, features)
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        # Save the trained head so inference_classification.py can reload the probe.
        torch.save(
            {
                "head_state_dict": selected_head_state,
                "model_config": model_cfg,
                "features": dict(features),
                "epoch": selected_epoch,
                "checkpoint_selection": selection,
                # Travels with the weights: output i means label_space[i], and inference
                # cannot name a prediction without it.
                "label_space": list(label_space),
                # The normalization travels with the weights, so evaluation applies the
                # same transform instead of recomputing it from the evaluated split.
                "feature_mean": model.feature_mean.detach().cpu(),
                "feature_std": model.feature_std.detach().cpu(),
            },
            path / "head.pt",
        )
        result["checkpoint"] = str(path / "head.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a YAML config")
    arguments = parser.parse_args()
    train_from_config(load_config(arguments.config))


if __name__ == "__main__":
    main()
