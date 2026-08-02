"""
Classification probe: a trainable MLP head over one pooled vector per window.

Classifies the whole target frame: the window's tokens are pooled to a single [D] vector, and the head
maps it to one logit per CO3D category ("feature_dim -> num_classes").

Pooling is defined here rather than in the dataset because *which* representation to
pool is a compared variable, not a fixed detail: pooled frame-6 image tokens vs. pooled
state after frame 6.

The model operates purely on **precomputed embeddings** from the probe-feature cache, it does not hold a backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence, TypedDict, cast

import torch
import torch.nn as nn

FeatureSource = Literal["image_tokens", "state_tokens"] # Embedding type
Pooling = Literal["mean", "max"] # Pooling type
Normalization = Literal["standardize", "none"] # is_normalization


class FeatureSpec(TypedDict):
    """The representation a run uses; recorded in metrics.json and head.pt."""
    source: FeatureSource
    pooling: Pooling
    normalize: Normalization


_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}

# Ways to reduce a window's [tokens, D] block to one [D] vector.
_POOLINGS = {
    "mean": lambda tokens: tokens.mean(dim=0),
    "max": lambda tokens: tokens.max(dim=0).values,
}

# Which cached tensor to pool. Both are per-target-frame.
_SOURCES = ("image_tokens", "state_tokens")

# What to do to the pooled vector before the head sees it.
_NORMALIZATIONS = ("standardize", "none")


@dataclass
class HeadConfig:
    """Configuration for the classification head (parsed from the config file)."""
    feature_dim: int  # Backbone outputs dim per token, unchanged by pooling
    num_classes: int  # CO3D categories - MUST BE DEFINED
    hidden_dims: Sequence[int] = field(default_factory=tuple)  # MLP layers
    activation: str = "gelu"
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HeadConfig":
        if "num_classes" not in data:
            raise KeyError("model.num_classes is required for classification (51 for the full CO3D vocabulary)")
        return cls(
            feature_dim=int(data["feature_dim"]),
            num_classes=int(data["num_classes"]),
            hidden_dims=tuple(int(value) for value in data.get("hidden_dims", ())),
            activation=str(data.get("activation", "gelu")),
            dropout=float(data.get("dropout", 0.0)),
        )


class MLPHead(nn.Module):
    """Per-vector multiclass classifier."""

    def __init__(self, config: HeadConfig) -> None:
        super().__init__()
        if config.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if config.num_classes < 2:
            raise ValueError(f"num_classes must be at least 2 for classification, got {config.num_classes}")
        if config.activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation {config.activation!r}; supported: {sorted(_ACTIVATIONS)}")
        self.config = config
        activation = _ACTIVATIONS[config.activation]

        # Build MLP layers
        layers: list[nn.Module] = []
        in_dim = config.feature_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(activation())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, config.num_classes))
        self.net = nn.Sequential(*layers)

    @property
    def is_linear(self) -> bool:
        """True when there are no hidden layers, i.e., a genuine linear probe."""
        return len(self.config.hidden_dims) == 0

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """[B, feature_dim] -> [B, num_classes]"""
        if features.shape[-1] != self.config.feature_dim:
            raise ValueError(f"Expected last dim {self.config.feature_dim}, got {features.shape[-1]}")
        return self.net(features)


def pool_tokens(tokens: torch.Tensor, pooling: Pooling) -> torch.Tensor:
    """[tokens, D] -> [D] by the named reduction."""
    if pooling not in _POOLINGS:
        raise ValueError(f"Unknown pooling {pooling!r}; supported: {sorted(_POOLINGS)}")
    if tokens.ndim != 2:
        raise ValueError(f"Expected [tokens, D], got {tuple(tokens.shape)}")
    return _POOLINGS[pooling](tokens)


def resolve_features(config: dict[str, Any]) -> FeatureSpec:
    """The {source, pooling, normalize} a config asks for."""
    features = config.get("features")
    if not isinstance(features, dict) or "source" not in features:
        raise ValueError(
            "Config needs a 'features' block with an explicit 'source': "
            f"one of {list(_SOURCES)}. Which representation to classify from is an "
            "open comparison, so it is not defaulted."
        )
    source = str(features["source"])
    if source not in _SOURCES:
        raise ValueError(f"Unknown features.source {source!r}; supported: {list(_SOURCES)}")
    pooling = str(features.get("pooling", "mean"))
    if pooling not in _POOLINGS:
        raise ValueError(f"Unknown features.pooling {pooling!r}; supported: {sorted(_POOLINGS)}")
    normalize = str(features.get("normalize", "standardize"))
    if normalize not in _NORMALIZATIONS:
        raise ValueError(f"Unknown features.normalize {normalize!r}; supported: {list(_NORMALIZATIONS)}")
    # cast, not assert: the membership checks above already restrict each value to its
    # Literal, but the checker cannot see that through the dict lookups.
    return FeatureSpec(
        source=cast(FeatureSource, source),
        pooling=cast(Pooling, pooling),
        normalize=cast(Normalization, normalize),
    )


def feature_statistics(vectors: torch.Tensor, *, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension (mean, std) of [n, D] pooled features, std floored at eps."""
    if vectors.ndim != 2 or vectors.shape[0] < 2:
        raise ValueError(f"Need at least 2 pooled vectors as [n, D], got {tuple(vectors.shape)}")
    return vectors.mean(dim=0), vectors.std(dim=0).clamp_min(eps)


class ClassificationProbe(nn.Module):
    """Trainable per-window MLP head over pooled, precomputed token features."""

    # Declared for the type checker: register_buffer creates these at runtime, so
    # without the annotations they look like attributes defined outside __init__.
    feature_mean: torch.Tensor
    feature_std: torch.Tensor

    def __init__(self, config: HeadConfig, *, normalize: Normalization = "none") -> None:
        super().__init__()
        if normalize not in _NORMALIZATIONS:
            raise ValueError(f"Unknown normalize {normalize!r}; supported: {list(_NORMALIZATIONS)}")
        self.normalize = normalize
        self.register_buffer("feature_mean", torch.zeros(config.feature_dim))
        self.register_buffer("feature_std", torch.ones(config.feature_dim))
        self.statistics_set = normalize == "none"  # "none" needs none
        self.head = MLPHead(config)  # Validates feature_dim / num_classes / activation

    def set_feature_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Install standardization statistics, which must come from the TRAIN split. Once."""
        expected = self.head.config.feature_dim
        if tuple(mean.shape) != (expected,) or tuple(std.shape) != (expected,):
            raise ValueError(
                f"Statistics must be [{expected}], got mean {tuple(mean.shape)} "
                f"std {tuple(std.shape)}"
            )
        # copy_ writes into the registered buffers instead of rebinding the attributes,
        # so the module keeps its own tensors: .to(device) still moves them, and the
        # statistics survive a later device change.
        self.feature_mean.copy_(mean.detach())
        self.feature_std.copy_(std.detach())
        self.statistics_set = True

    def apply_normalization(self, pooled: torch.Tensor) -> torch.Tensor:
        """Standardize with the installed train-split statistics, if enabled. Every batch."""
        if self.normalize == "none":
            return pooled
        if not self.statistics_set:
            raise RuntimeError(
                "normalize='standardize' but no statistics were installed; call "
                "set_feature_statistics() with train-split values first"
            )
        return (pooled - self.feature_mean) / self.feature_std

    @property
    def num_classes(self) -> int:
        return self.head.config.num_classes

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """[B, D] -> [B, num_classes] class logits, one row per window."""
        if pooled.ndim != 2:
            raise ValueError(f"pooled features must be [B, D], got {tuple(pooled.shape)}")
        return self.head(self.apply_normalization(pooled))

    @torch.no_grad()
    def predict(self, pooled: torch.Tensor, *, top_k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """(probabilities [B, top_k], category indices [B, top_k]), the best first."""
        probabilities = torch.softmax(self.forward(pooled), dim=1)
        values, indices = probabilities.topk(min(top_k, self.num_classes), dim=1)
        return values, indices


def build_probe(config: dict[str, Any], *, normalize: Normalization = "none") -> ClassificationProbe:
    """Build a: class:`ClassificationProbe` from a config model block."""
    return ClassificationProbe(HeadConfig.from_dict(config), normalize=normalize)
