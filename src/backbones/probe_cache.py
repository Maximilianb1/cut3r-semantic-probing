"""Unified embedding cache for the three probing backbones.

One on-disk format, two embedding *layouts*, and both task labels in every entry
so probe training reads any backbone uniformly:

- ``layout="trajectory"`` (CUT3R-trained): the full six-state structure, same as
  Stage 0 -- ``image_tokens [6,1,N,C]`` (grid per state) and
  ``state_tokens [6,1,S,C]`` (latent per state).
- ``layout="target_only"`` (CUT3R-random, DINOv2): only the last state --
  ``spatial_tokens [N,C]`` (grid) and ``global_tokens [M,C]`` (latent).

Every entry also stores:

- ``seg_labels [grid_h, grid_w]`` uint8 {0,1} -- segmentation target, the target
  frame's mask pooled to *this backbone's* token grid.
- ``category`` + ``category_index`` -- classification target, from a fixed
  vocabulary (``metadata.json``) shared across all three caches.

Splits stay at the manifest's CO3D **sequence** level; this module never
re-splits. Writes are atomic, resumable, and SHA-256 verified.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from src.backbones.base import (
    TARGET_FRAME_INDEX,
    Backbone,
    TrajectoryExtraction,
    WindowExtraction,
)
from src.common.io import (
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_file,
)
from src.common.tables import read_parquet, write_parquet_atomic
from src.data.co3d import CO3DV2_231130_CATEGORIES

PROBE_CACHE_SCHEMA_VERSION = "probe-features-v2"
TRAJECTORY = "trajectory"
TARGET_ONLY = "target_only"


def category_vocabulary() -> list[str]:
    """Deterministic 51-category vocabulary (index = position). Sorted, stable."""
    return sorted(CO3DV2_231130_CATEGORIES)


def category_index_map() -> dict[str, int]:
    return {category: index for index, category in enumerate(category_vocabulary())}


@dataclass
class EmbeddingSample:
    """One window's cached embeddings and both task labels."""

    window_id: str
    category: str
    category_index: int
    sequence_id: str
    split: str
    layout: str
    token_grid: tuple[int, int]
    frame_ids: list[str]
    seg_labels: torch.Tensor  # [grid_h, grid_w]
    # trajectory layout:
    image_tokens: torch.Tensor | None = None  # [6, 1, N, C]
    state_tokens: torch.Tensor | None = None  # [6, 1, S, C]
    # target_only layout:
    spatial_tokens: torch.Tensor | None = None  # [N, C]
    global_tokens: torch.Tensor | None = None  # [M, C]

    def target_spatial(self) -> torch.Tensor:
        """Target-frame spatial (grid) tokens ``[N, C]`` regardless of layout."""
        if self.layout == TRAJECTORY:
            return self.image_tokens[TARGET_FRAME_INDEX, 0]  # type: ignore[index]
        return self.spatial_tokens  # type: ignore[return-value]

    def target_global(self) -> torch.Tensor:
        """Target-frame global (latent) tokens ``[M, C]`` regardless of layout."""
        if self.layout == TRAJECTORY:
            return self.state_tokens[TARGET_FRAME_INDEX, 0]  # type: ignore[index]
        return self.global_tokens  # type: ignore[return-value]

    @classmethod
    def from_window_extraction(
        cls,
        *,
        window: dict[str, Any],
        extraction: WindowExtraction,
        category_index: int,
        mask_threshold: float,
    ) -> "EmbeddingSample":
        return cls(
            window_id=window["window_id"],
            category=window["category"],
            category_index=category_index,
            sequence_id=window["sequence_id"],
            split=window["split"],
            layout=TARGET_ONLY,
            token_grid=extraction.features.token_grid,
            frame_ids=[extraction.features.frame_id],
            seg_labels=extraction.target_labels(threshold=mask_threshold),
            spatial_tokens=extraction.features.spatial_tokens,
            global_tokens=extraction.features.global_tokens,
        )

    @classmethod
    def from_trajectory(
        cls,
        *,
        window: dict[str, Any],
        extraction: TrajectoryExtraction,
        category_index: int,
        mask_threshold: float,
    ) -> "EmbeddingSample":
        return cls(
            window_id=window["window_id"],
            category=window["category"],
            category_index=category_index,
            sequence_id=window["sequence_id"],
            split=window["split"],
            layout=TRAJECTORY,
            token_grid=extraction.token_grid,
            frame_ids=list(extraction.frame_ids),
            seg_labels=extraction.target_labels(threshold=mask_threshold),
            image_tokens=extraction.image_tokens,
            state_tokens=extraction.state_tokens,
        )


def _require_fields(row: dict[str, Any], fields: Iterable[str]) -> None:
    missing = [name for name in fields if name not in row or row[name] is None]
    if missing:
        raise KeyError(f"Manifest row is missing required fields: {missing}")


class EmbeddingCacheWriter:
    """Atomic, resumable, shard-based writer for the embedding cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        contract: dict[str, Any],
        windows_per_shard: int = 32,
    ) -> None:
        if windows_per_shard < 1:
            raise ValueError("windows_per_shard must be positive")
        self.directory = Path(cache_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.layout = contract["layout"]
        if self.layout not in (TRAJECTORY, TARGET_ONLY):
            raise ValueError(f"Unknown cache layout: {self.layout!r}")
        self.contract = {
            "probe_cache_schema_version": PROBE_CACHE_SCHEMA_VERSION,
            "category_vocabulary": category_vocabulary(),
            **contract,
        }
        self.windows_per_shard = windows_per_shard
        self.metadata_path = self.directory / "metadata.json"
        self.index_path = self.directory / "index.parquet"
        self.index_rows = read_parquet(self.index_path) if self.index_path.is_file() else []
        self.completed = {row["window_id"] for row in self.index_rows}
        self.buffer: list[tuple[dict[str, Any], dict[str, torch.Tensor]]] = []
        if self.metadata_path.is_file():
            existing = load_json(self.metadata_path)
            if canonical_json_bytes(existing) != canonical_json_bytes(self.contract):
                raise ValueError("Existing embedding-cache metadata is incompatible")
        else:
            atomic_write_json(self.metadata_path, self.contract)

    def contains(self, window_id: str) -> bool:
        return window_id in self.completed

    def add(self, sample: EmbeddingSample, *, source_sha256: dict[str, dict[str, str]]) -> bool:
        if sample.layout != self.layout:
            raise ValueError(
                f"Sample layout {sample.layout!r} differs from cache layout {self.layout!r}"
            )
        if self.contains(sample.window_id):
            return False
        if any(row["window_id"] == sample.window_id for row, _ in self.buffer):
            raise ValueError(f"Window {sample.window_id} added twice before flushing")
        grid_h, grid_w = sample.token_grid
        if tuple(sample.seg_labels.shape) != (grid_h, grid_w):
            raise ValueError("seg_labels shape does not match token_grid")
        tensors: dict[str, torch.Tensor] = {
            "seg_labels": sample.seg_labels.detach().to("cpu", torch.uint8).contiguous()
        }
        row: dict[str, Any] = {
            "window_id": sample.window_id,
            "layout": sample.layout,
            "category": sample.category,
            "category_index": int(sample.category_index),
            "sequence_id": sample.sequence_id,
            "split": sample.split,
            "token_grid": [grid_h, grid_w],
            "frame_ids": list(sample.frame_ids),
            "source_sha256_json": canonical_json_bytes(source_sha256).decode("ascii"),
        }
        if sample.layout == TRAJECTORY:
            image = sample.image_tokens.detach().to("cpu", torch.float16).contiguous()  # type: ignore[union-attr]
            state = sample.state_tokens.detach().to("cpu", torch.float16).contiguous()  # type: ignore[union-attr]
            if image.shape[-2] != grid_h * grid_w:
                raise ValueError("image_tokens grid does not match token_grid")
            tensors["image_tokens"] = image
            tensors["state_tokens"] = state
            row["image_shape"] = list(image.shape)
            row["state_shape"] = list(state.shape)
        else:
            spatial = sample.spatial_tokens.detach().to("cpu", torch.float16).contiguous()  # type: ignore[union-attr]
            global_tokens = sample.global_tokens.detach().to("cpu", torch.float16).contiguous()  # type: ignore[union-attr]
            if spatial.shape[0] != grid_h * grid_w:
                raise ValueError("spatial_tokens count does not match token_grid")
            tensors["spatial"] = spatial
            tensors["global"] = global_tokens
            row["spatial_shape"] = list(spatial.shape)
            row["global_shape"] = list(global_tokens.shape)
        self.buffer.append((row, tensors))
        if len(self.buffer) >= self.windows_per_shard:
            self.flush()
        return True

    def flush(self) -> None:
        if not self.buffer:
            return
        existing = [int(Path(r["shard"]).stem.removeprefix("shard-")) for r in self.index_rows]
        shard_number = 0 if not existing else max(existing) + 1
        shard_name = f"shard-{shard_number:05d}.safetensors"
        shard_path = self.directory / shard_name
        if shard_path.exists():
            raise FileExistsError(f"Refusing to overwrite cache shard: {shard_path}")
        tensors: dict[str, torch.Tensor] = {}
        pending: list[dict[str, Any]] = []
        for item_number, (row, item_tensors) in enumerate(self.buffer):
            prefix = f"item_{item_number:04d}"
            stored = {name: f"{prefix}.{name}" for name in item_tensors}
            for name, tensor in item_tensors.items():
                tensors[stored[name]] = tensor
            new_row = {**row, "shard": shard_name}
            for name, key in stored.items():
                new_row[f"{name}_key"] = key
            pending.append(new_row)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.directory, prefix=f".{shard_name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            save_file(tensors, temporary)
            reloaded = load_file(temporary, device="cpu")
            for key, expected in tensors.items():
                actual = reloaded.get(key)
                if actual is None or actual.shape != expected.shape:
                    raise ValueError(f"Embedding cache round-trip failed for {key}")
            os.replace(temporary, shard_path)
        finally:
            temporary.unlink(missing_ok=True)
        shard_hash = sha256_file(shard_path)
        for row in pending:
            row["shard_sha256"] = shard_hash
        self.index_rows.extend(pending)
        write_parquet_atomic(self.index_path, self.index_rows)
        self.completed.update(row["window_id"] for row in pending)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "EmbeddingCacheWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()


def load_probe_index(cache_dir: str | Path) -> list[dict[str, Any]]:
    return read_parquet(Path(cache_dir) / "index.parquet")


def load_embedding_sample(cache_dir: str | Path, row: dict[str, Any]) -> EmbeddingSample:
    """Load one window's tensors from its shard.

    Reads **only this window's tensors** via ``safe_open`` rather than
    materializing the whole shard. A shard holds many windows (32 by default), so
    a full ``load_file`` per window would read tens to hundreds of megabytes to
    return a few, once per window per epoch. Lazy reads keep training I/O
    proportional to what is actually used and stay efficient under shuffling.
    """
    directory = Path(cache_dir)
    grid = tuple(int(value) for value in row["token_grid"])
    wanted = ["seg_labels_key"] + (
        ["image_tokens_key", "state_tokens_key"]
        if row["layout"] == TRAJECTORY
        else ["spatial_key", "global_key"]
    )
    with safe_open(directory / row["shard"], framework="pt", device="cpu") as handle:
        tensors = {row[field]: handle.get_tensor(row[field]) for field in wanted}
    common = {
        "window_id": row["window_id"],
        "category": row["category"],
        "category_index": int(row["category_index"]),
        "sequence_id": row["sequence_id"],
        "split": row["split"],
        "layout": row["layout"],
        "token_grid": grid,  # type: ignore[arg-type]
        "frame_ids": list(row["frame_ids"]),
        "seg_labels": tensors[row["seg_labels_key"]].float(),
    }
    if row["layout"] == TRAJECTORY:
        return EmbeddingSample(
            **common,
            image_tokens=tensors[row["image_tokens_key"]].float(),
            state_tokens=tensors[row["state_tokens_key"]].float(),
        )
    return EmbeddingSample(
        **common,
        spatial_tokens=tensors[row["spatial_key"]].float(),
        global_tokens=tensors[row["global_key"]].float(),
    )


def load_target_tokens(
    cache_dir: str | Path, row: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read only what a per-token probe needs: target-frame grid tokens + labels.

    Returns ``(spatial [N, C], seg_labels [grid_h, grid_w])``, both float32.

    :func:`load_embedding_sample` returns a window's *whole* entry. For the
    ``trajectory`` layout that is all six states' grid tokens ``[6,1,N,768]``
    **and** all six persistent-state latents ``[6,1,768,768]``. At CUT3R-512 with
    a 24x32 grid (``N`` = 768) that is 13.5 MiB in fp16, of which a segmentation
    probe uses one grid: 1.1 MiB. Here the target frame is sliced out *inside* the
    shard read, so the other five states and every state latent are never
    transferred -- ``6 * (N + 768) / N`` fewer bytes per window per epoch, i.e.
    12x at ``N`` = 768 and 12.9x at the 21x32 grid of a 3:2 frame.

    Values are identical to ``load_embedding_sample(...).target_spatial()``; only
    the amount of I/O differs. Use :func:`load_embedding_sample` when a caller
    genuinely needs the full trajectory or the state latents.
    """
    directory = Path(cache_dir)
    grid_h, grid_w = (int(value) for value in row["token_grid"])
    with safe_open(directory / row["shard"], framework="pt", device="cpu") as handle:
        labels = handle.get_tensor(row["seg_labels_key"]).float()
        if row["layout"] == TRAJECTORY:
            # get_slice reads just this sub-range of the tensor's bytes.
            window = handle.get_slice(row["image_tokens_key"])[
                TARGET_FRAME_INDEX : TARGET_FRAME_INDEX + 1, 0:1
            ]
            spatial = window.reshape(window.shape[-2], window.shape[-1]).float()
        else:
            # target_only already stores just the target frame; ``global`` is
            # simply not requested.
            spatial = handle.get_tensor(row["spatial_key"]).float()
    if tuple(labels.shape) != (grid_h, grid_w) or spatial.shape[0] != grid_h * grid_w:
        raise ValueError(
            f"Cached shapes disagree with token_grid {(grid_h, grid_w)} for "
            f"{row['window_id']}: spatial {tuple(spatial.shape)}, labels {tuple(labels.shape)}"
        )
    return spatial, labels


def load_target_features(
    cache_dir: str | Path, row: dict[str, Any], *, kind: str = "spatial"
) -> torch.Tensor:
    """One target-frame tensor, for probes that do not need the segmentation labels.

    ``kind="spatial"`` returns the grid tokens ``[N, C]`` (CUT3R
    ``image_tokens[5,0]``, DINOv2 patch tokens). ``kind="global"`` returns the latent
    ``[M, C]`` (CUT3R ``state_tokens[5,0]``, the persistent recurrent memory, or
    DINOv2's CLS token).

    Both are sliced inside the shard read, so a classification probe reading only the
    state latents never transfers the grid tokens, and vice versa. Which of the two a
    classification probe reads is a **compared variable** (see
    ``src/classification/README.md``), which is why this exposes both rather than
    picking one.

    The global latent is *not* pixel-aligned and must never be reshaped to a
    segmentation grid (ADR 0003).
    """
    if kind not in ("spatial", "global"):
        raise ValueError(f"kind must be 'spatial' or 'global', got {kind!r}")
    directory = Path(cache_dir)
    trajectory = row["layout"] == TRAJECTORY
    if trajectory:
        key = row["image_tokens_key"] if kind == "spatial" else row["state_tokens_key"]
    else:
        key = row["spatial_key"] if kind == "spatial" else row["global_key"]
    with safe_open(directory / row["shard"], framework="pt", device="cpu") as handle:
        if trajectory:
            window = handle.get_slice(key)[TARGET_FRAME_INDEX : TARGET_FRAME_INDEX + 1, 0:1]
            tensor = window.reshape(window.shape[-2], window.shape[-1]).float()
        else:
            tensor = handle.get_tensor(key).float()
    if tensor.ndim != 2:
        raise ValueError(
            f"Expected a [tokens, channels] tensor for {row['window_id']} {kind}, "
            f"got {tuple(tensor.shape)}"
        )
    if kind == "spatial":
        grid_h, grid_w = (int(value) for value in row["token_grid"])
        if tensor.shape[0] != grid_h * grid_w:
            raise ValueError(
                f"Cached spatial tokens disagree with token_grid {(grid_h, grid_w)} for "
                f"{row['window_id']}: {tuple(tensor.shape)}"
            )
    return tensor


def verify_probe_cache(cache_dir: str | Path) -> dict[str, Any]:
    directory = Path(cache_dir)
    if not (directory / "metadata.json").is_file() or not (directory / "index.parquet").is_file():
        raise FileNotFoundError("Embedding cache requires metadata.json and index.parquet")
    metadata = load_json(directory / "metadata.json")
    if metadata.get("probe_cache_schema_version") != PROBE_CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported embedding-cache schema version")
    vocabulary = metadata.get("category_vocabulary") or []
    rows = read_parquet(directory / "index.parquet")
    if not rows:
        raise ValueError("Embedding cache index is empty")
    present = {path.name for path in directory.glob("shard-*.safetensors")}
    indexed = {row["shard"] for row in rows}
    if present != indexed:
        raise ValueError(
            f"Embedding cache shards disagree with index: unindexed={sorted(present - indexed)}, "
            f"missing={sorted(indexed - present)}"
        )
    seen: set[str] = set()
    for shard_name in sorted(indexed):
        shard_rows = [row for row in rows if row["shard"] == shard_name]
        hashes = {row["shard_sha256"] for row in shard_rows}
        if len(hashes) != 1 or sha256_file(directory / shard_name) not in hashes:
            raise ValueError(f"SHA-256 mismatch for embedding cache shard {shard_name}")
        for row in shard_rows:
            if row["window_id"] in seen:
                raise ValueError(f"Duplicate embedding cache window: {row['window_id']}")
            seen.add(row["window_id"])
            index = int(row["category_index"])
            if vocabulary and not (0 <= index < len(vocabulary) and vocabulary[index] == row["category"]):
                raise ValueError(f"category_index/category mismatch for {row['window_id']}")
            sample = load_embedding_sample(directory, row)
            grid_h, grid_w = sample.token_grid
            if tuple(sample.seg_labels.shape) != (grid_h, grid_w):
                raise ValueError(f"seg_labels/grid mismatch for {row['window_id']}")
            if sample.target_spatial().shape[0] != grid_h * grid_w:
                raise ValueError(f"spatial/grid mismatch for {row['window_id']}")
            for tensor in (sample.target_spatial(), sample.target_global()):
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite features for {row['window_id']}")
    return {"valid": True, "windows": len(rows), "shards": len(indexed), "layout": metadata["layout"]}


def _target_source_sha256(
    frame_by_id: dict[str, dict[str, Any]], target_frame_id: str, dataset_root: Path
) -> dict[str, dict[str, str]]:
    row = frame_by_id[target_frame_id]
    return {
        target_frame_id: {
            "image": sha256_file(dataset_root / row["image_relpath"]),
            "mask": sha256_file(dataset_root / row["mask_relpath"]),
        }
    }


def extract_to_cache(
    backbone: Backbone,
    *,
    layout: str,
    windows: list[dict[str, Any]],
    frame_by_id: dict[str, dict[str, Any]],
    dataset_root: str,
    cache_dir: str | Path,
    contract: dict[str, Any],
    mask_threshold: float = 0.5,
    windows_per_shard: int = 32,
    limit_windows: int | None = None,
    verify_source_hashes: bool = True,
) -> dict[str, Any]:
    """Run a backbone over windows and write the embedding cache for one layout."""
    if layout not in (TRAJECTORY, TARGET_ONLY):
        raise ValueError(f"Unknown layout: {layout!r}")
    if layout == TRAJECTORY and not hasattr(backbone, "extract_trajectory"):
        raise ValueError("Trajectory layout requires a backbone with extract_trajectory")
    index_of = category_index_map()
    root = Path(dataset_root)
    selected = windows if limit_windows is None else windows[:limit_windows]
    full_contract = {"layout": layout, "backbone": backbone.provenance(),
                     "mask_threshold": float(mask_threshold), **contract}
    written = skipped = 0
    with EmbeddingCacheWriter(
        cache_dir, contract=full_contract, windows_per_shard=windows_per_shard
    ) as writer:
        for window in selected:
            _require_fields(window, ("window_id", "frame_ids", "category", "sequence_id", "split"))
            if window["category"] not in index_of:
                raise KeyError(f"Category {window['category']!r} is not in the vocabulary")
            if writer.contains(window["window_id"]):
                skipped += 1
                continue
            frame_rows = [frame_by_id[frame_id] for frame_id in window["frame_ids"]]
            category_index = index_of[window["category"]]
            if layout == TRAJECTORY:
                extraction = backbone.extract_trajectory(frame_rows, dataset_root=dataset_root)  # type: ignore[attr-defined]
                sample = EmbeddingSample.from_trajectory(
                    window=window, extraction=extraction,
                    category_index=category_index, mask_threshold=mask_threshold,
                )
                target_frame_id = extraction.frame_ids[TARGET_FRAME_INDEX]
            else:
                extraction = backbone.extract_window(frame_rows, dataset_root=dataset_root)
                sample = EmbeddingSample.from_window_extraction(
                    window=window, extraction=extraction,
                    category_index=category_index, mask_threshold=mask_threshold,
                )
                target_frame_id = extraction.features.frame_id
            source_sha256 = (
                _target_source_sha256(frame_by_id, target_frame_id, root)
                if verify_source_hashes
                else {}
            )
            writer.add(sample, source_sha256=source_sha256)
            written += 1
    return {"written": written, "skipped": skipped, "verification": verify_probe_cache(cache_dir)}


def attach_labels_from_trajectory_cache(
    trajectory_cache_dir: str | Path,
    *,
    windows: list[dict[str, Any]],
    frame_by_id: dict[str, dict[str, Any]],
    dataset_root: str,
    cache_dir: str | Path,
    contract: dict[str, Any],
    input_size: int = 512,
    patch_size: int = 16,
    square_ok: bool = False,
    mask_threshold: float = 0.5,
    windows_per_shard: int = 32,
    limit_windows: int | None = None,
    verify_source_hashes: bool = True,
) -> dict[str, Any]:
    """Build the CUT3R-trained v2 cache by REUSING existing Stage 0 embeddings.

    No GPU and no CUT3R re-run: the trajectory embeddings come straight from the
    audited Stage 0 cache; only the labels are attached. The target mask is loaded
    from CO3D with the exact CUT3R transform and pooled to the token grid, and its
    SHA-256 is cross-checked against the hash the Stage 0 cache recorded -- so the
    label provably comes from the same mask bytes the embedding was extracted on.
    """
    import numpy as np
    from PIL import Image

    from src.data.transforms import transform_rgb_mask
    from src.embeddings.cache import load_trajectory
    from src.embeddings.input import _safe_path

    index_of = category_index_map()
    root = Path(dataset_root)
    traj_dir = Path(trajectory_cache_dir)
    recorded = {
        row["window_id"]: json.loads(row["source_sha256_json"])
        for row in read_parquet(traj_dir / "index.parquet")
    }
    full_contract = {
        "layout": TRAJECTORY,
        "backbone": {"backbone": "cut3r-trained", "weights": "trained", "source": "existing_embeddings"},
        "mask_threshold": float(mask_threshold),
        "trajectory_cache_metadata_sha256": sha256_file(traj_dir / "metadata.json"),
        **contract,
    }
    selected = windows if limit_windows is None else windows[:limit_windows]
    written = skipped = 0
    with EmbeddingCacheWriter(
        cache_dir, contract=full_contract, windows_per_shard=windows_per_shard
    ) as writer:
        for window in selected:
            _require_fields(window, ("window_id", "frame_ids", "category", "sequence_id", "split"))
            if window["category"] not in index_of:
                raise KeyError(f"Category {window['category']!r} is not in the vocabulary")
            if writer.contains(window["window_id"]):
                skipped += 1
                continue
            trajectory = load_trajectory(traj_dir, window["window_id"])
            target_frame_id = trajectory.frame_ids[TARGET_FRAME_INDEX]
            frame_row = frame_by_id[target_frame_id]
            image_path = _safe_path(root, frame_row["image_relpath"])
            mask_path = _safe_path(root, frame_row["mask_relpath"])
            with Image.open(image_path) as handle:
                image = handle.copy()
            with Image.open(mask_path) as handle:
                mask = handle.copy()
            _image, transformed_mask, transform = transform_rgb_mask(
                image, mask, input_size=input_size, patch_size=patch_size, square_ok=square_ok
            )
            if transform.token_grid != trajectory.token_grid:
                raise ValueError(
                    f"Transform grid {transform.token_grid} disagrees with cached grid "
                    f"{trajectory.token_grid} for {window['window_id']}"
                )
            mask_hash = sha256_file(mask_path)
            if verify_source_hashes:
                expected = recorded.get(window["window_id"], {}).get(target_frame_id, {}).get("mask")
                if expected is not None and expected != mask_hash:
                    raise ValueError(
                        f"Mask SHA-256 for {window['window_id']} does not match the hash the "
                        "Stage 0 embedding recorded; label and embedding would be misaligned"
                    )
            extraction = TrajectoryExtraction(
                image_tokens=trajectory.image_tokens.float(),
                state_tokens=trajectory.state_tokens.float(),
                token_grid=trajectory.token_grid,
                frame_ids=list(trajectory.frame_ids),
                target_mask=torch.from_numpy(np.asarray(transformed_mask, dtype=np.float32)),
            )
            sample = EmbeddingSample.from_trajectory(
                window=window, extraction=extraction,
                category_index=index_of[window["category"]], mask_threshold=mask_threshold,
            )
            source_sha256 = {target_frame_id: {"image": sha256_file(image_path), "mask": mask_hash}}
            writer.add(sample, source_sha256=source_sha256)
            written += 1
    return {"written": written, "skipped": skipped, "verification": verify_probe_cache(cache_dir)}
