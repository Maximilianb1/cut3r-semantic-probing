from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from src.common.io import (
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_file,
)
from src.common.tables import read_parquet, write_parquet_atomic
from src.embeddings.types import REPRESENTATION_CONTRACT_VERSION, FeatureTrajectory

CACHE_SCHEMA_VERSION = "stage0-cache-v1"


class FeatureCacheWriter:
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
        self.contract = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "representation_contract_version": REPRESENTATION_CONTRACT_VERSION,
            **contract,
        }
        self.windows_per_shard = windows_per_shard
        self.metadata_path = self.directory / "metadata.json"
        self.index_path = self.directory / "index.parquet"
        self.index_rows = (
            read_parquet(self.index_path) if self.index_path.is_file() else []
        )
        self.completed = {row["window_id"] for row in self.index_rows}
        self.buffer: list[tuple[str, FeatureTrajectory, dict[str, dict[str, str]]]] = []
        self._check_or_create_metadata()
        referenced_shards = {row["shard"] for row in self.index_rows}
        orphaned = sorted(
            path.name
            for path in self.directory.glob("shard-*.safetensors")
            if path.name not in referenced_shards
        )
        if orphaned:
            raise RuntimeError(
                "Cache contains shards absent from its index; review or remove the "
                f"incomplete files before resuming: {', '.join(orphaned)}"
            )
        if self.index_rows:
            verify_cache(self.directory)

    def _check_or_create_metadata(self) -> None:
        if self.metadata_path.is_file():
            existing = load_json(self.metadata_path)
            if canonical_json_bytes(existing) != canonical_json_bytes(self.contract):
                raise ValueError(
                    "Existing cache metadata is incompatible with this "
                    "extraction contract"
                )
        else:
            atomic_write_json(self.metadata_path, self.contract)

    def contains(self, window_id: str) -> bool:
        return window_id in self.completed

    def add(
        self,
        window_id: str,
        trajectory: FeatureTrajectory,
        *,
        source_sha256: dict[str, dict[str, str]] | None = None,
    ) -> bool:
        if self.contains(window_id):
            return False
        if any(buffered_id == window_id for buffered_id, *_ in self.buffer):
            raise ValueError(f"Window {window_id} was added twice before flushing")
        trajectory.validate(expected_timesteps=6)
        self.buffer.append((window_id, trajectory.cpu_float16(), source_sha256 or {}))
        if len(self.buffer) >= self.windows_per_shard:
            self.flush()
        return True

    def flush(self) -> None:
        if not self.buffer:
            return
        existing_numbers = [
            int(Path(row["shard"]).stem.removeprefix("shard-"))
            for row in self.index_rows
        ]
        shard_number = 0 if not existing_numbers else max(existing_numbers) + 1
        shard_name = f"shard-{shard_number:05d}.safetensors"
        shard_path = self.directory / shard_name
        if shard_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing cache shard: {shard_path}"
            )
        tensors: dict[str, torch.Tensor] = {}
        pending_rows: list[dict[str, Any]] = []
        for item_number, (window_id, trajectory, source_sha256) in enumerate(
            self.buffer
        ):
            prefix = f"item_{item_number:04d}"
            image_key = f"{prefix}.image_tokens"
            state_key = f"{prefix}.state_tokens"
            tensors[image_key] = trajectory.image_tokens
            tensors[state_key] = trajectory.state_tokens
            pending_rows.append(
                {
                    "window_id": window_id,
                    "shard": shard_name,
                    "image_key": image_key,
                    "state_key": state_key,
                    "frame_ids": trajectory.frame_ids,
                    "token_grid": list(trajectory.token_grid),
                    "image_shape": list(trajectory.image_tokens.shape),
                    "state_shape": list(trajectory.state_tokens.shape),
                    "dtype": "float16",
                    "source_sha256_json": canonical_json_bytes(source_sha256).decode(
                        "ascii"
                    ),
                }
            )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.directory, prefix=f".{shard_name}.", suffix=".tmp"
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            save_file(tensors, temporary)
            loaded = load_file(temporary, device="cpu")
            for key, expected in tensors.items():
                actual = loaded.get(key)
                if (
                    actual is None
                    or actual.shape != expected.shape
                    or actual.dtype != expected.dtype
                ):
                    raise ValueError(f"Cache shard round-trip failed for tensor {key}")
                if not torch.isfinite(actual).all():
                    raise ValueError(f"Cache shard contains non-finite tensor {key}")
            os.replace(temporary, shard_path)
        finally:
            temporary.unlink(missing_ok=True)
        shard_hash = sha256_file(shard_path)
        for row in pending_rows:
            row["shard_sha256"] = shard_hash
        self.index_rows.extend(pending_rows)
        write_parquet_atomic(self.index_path, self.index_rows)
        self.completed.update(row["window_id"] for row in pending_rows)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> FeatureCacheWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()


def load_trajectory(cache_dir: str | Path, window_id: str) -> FeatureTrajectory:
    directory = Path(cache_dir)
    rows = [
        row
        for row in read_parquet(directory / "index.parquet")
        if row["window_id"] == window_id
    ]
    if len(rows) != 1:
        raise KeyError(f"Expected one cache entry for {window_id}, found {len(rows)}")
    row = rows[0]
    tensors = load_file(directory / row["shard"], device="cpu")
    trajectory = FeatureTrajectory(
        image_tokens=tensors[row["image_key"]],
        state_tokens=tensors[row["state_key"]],
        frame_ids=list(row["frame_ids"]),
        token_grid=tuple(int(value) for value in row["token_grid"]),
    )
    trajectory.validate(expected_timesteps=6)
    return trajectory


def verify_cache(cache_dir: str | Path) -> dict[str, Any]:
    directory = Path(cache_dir)
    metadata_path = directory / "metadata.json"
    index_path = directory / "index.parquet"
    if not metadata_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("Cache requires metadata.json and index.parquet")
    metadata = load_json(metadata_path)
    if metadata.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("Unsupported cache schema version")
    rows = read_parquet(index_path)
    if not rows:
        raise ValueError("Cache index is empty")
    by_shard: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        if row["window_id"] in seen:
            raise ValueError(f"Duplicate cache window ID: {row['window_id']}")
        seen.add(row["window_id"])
        source_hashes = json.loads(row["source_sha256_json"])
        if not isinstance(source_hashes, dict):
            raise ValueError(f"Invalid source hashes for {row['window_id']}")
        if metadata.get("source_file_hashes") == "sha256" and set(source_hashes) != set(
            row["frame_ids"]
        ):
            raise ValueError(f"Incomplete source hashes for {row['window_id']}")
        for frame_id, hashes in source_hashes.items():
            if (
                frame_id not in row["frame_ids"]
                or not isinstance(hashes, dict)
                or set(hashes) != {"image", "mask"}
            ):
                raise ValueError(f"Invalid source hash entry for {row['window_id']}")
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes.values()
            ):
                raise ValueError(f"Invalid source SHA-256 for {row['window_id']}")
        by_shard[row["shard"]].append(row)
    indexed_shards = set(by_shard)
    present_shards = {path.name for path in directory.glob("shard-*.safetensors")}
    if present_shards != indexed_shards:
        raise ValueError(
            "Cache shard files disagree with its index: "
            f"unindexed={sorted(present_shards - indexed_shards)}, "
            f"missing={sorted(indexed_shards - present_shards)}"
        )
    tensor_count = 0
    for shard_name, shard_rows in sorted(by_shard.items()):
        shard_path = directory / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing cache shard: {shard_path}")
        hashes = {row["shard_sha256"] for row in shard_rows}
        if len(hashes) != 1 or sha256_file(shard_path) not in hashes:
            raise ValueError(f"SHA-256 mismatch for cache shard {shard_name}")
        tensors = load_file(shard_path, device="cpu")
        expected_keys = {
            row[key_field]
            for row in shard_rows
            for key_field in ("image_key", "state_key")
        }
        if set(tensors) != expected_keys:
            raise ValueError(
                f"Shard {shard_name} tensor keys disagree with the cache index"
            )
        for row in shard_rows:
            if row["dtype"] != "float16":
                raise ValueError(f"Invalid indexed dtype for {row['window_id']}")
            trajectory = FeatureTrajectory(
                image_tokens=tensors[row["image_key"]],
                state_tokens=tensors[row["state_key"]],
                frame_ids=list(row["frame_ids"]),
                token_grid=tuple(int(value) for value in row["token_grid"]),
            )
            trajectory.validate(expected_timesteps=6)
            for key_field, shape_field in (
                ("image_key", "image_shape"),
                ("state_key", "state_shape"),
            ):
                key = row[key_field]
                if key not in tensors:
                    raise KeyError(f"Shard {shard_name} is missing tensor {key}")
                tensor = tensors[key]
                if list(tensor.shape) != list(row[shape_field]):
                    raise ValueError(f"Shape mismatch for {key}")
                if tensor.dtype != torch.float16 or not torch.isfinite(tensor).all():
                    raise ValueError(f"Invalid dtype or values for {key}")
                tensor_count += 1
    return {
        "valid": True,
        "windows": len(rows),
        "shards": len(by_shard),
        "tensors": tensor_count,
        "metadata_sha256": sha256_file(metadata_path),
        "index_sha256": sha256_file(index_path),
    }


def compare_caches(
    left_dir: str | Path,
    right_dir: str | Path,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    if atol < 0 or rtol < 0:
        raise ValueError("Comparison tolerances must be non-negative")
    verify_cache(left_dir)
    verify_cache(right_dir)
    left_metadata = load_json(Path(left_dir) / "metadata.json")
    right_metadata = load_json(Path(right_dir) / "metadata.json")
    if canonical_json_bytes(left_metadata) != canonical_json_bytes(right_metadata):
        raise ValueError("Cache extraction contracts differ")
    left_ids = {
        row["window_id"] for row in read_parquet(Path(left_dir) / "index.parquet")
    }
    right_ids = {
        row["window_id"] for row in read_parquet(Path(right_dir) / "index.parquet")
    }
    if left_ids != right_ids:
        raise ValueError("Cache window ID sets differ")
    left_sources = {
        row["window_id"]: row["source_sha256_json"]
        for row in read_parquet(Path(left_dir) / "index.parquet")
    }
    right_sources = {
        row["window_id"]: row["source_sha256_json"]
        for row in read_parquet(Path(right_dir) / "index.parquet")
    }
    if left_sources != right_sources:
        raise ValueError("Cache source-file hashes differ")
    maxima = {"image_tokens": 0.0, "state_tokens": 0.0}
    for window_id in sorted(left_ids):
        left = load_trajectory(left_dir, window_id)
        right = load_trajectory(right_dir, window_id)
        if left.frame_ids != right.frame_ids or left.token_grid != right.token_grid:
            raise ValueError(f"Cache trajectory metadata differs for {window_id}")
        for field in ("image_tokens", "state_tokens"):
            left_tensor = getattr(left, field).float()
            right_tensor = getattr(right, field).float()
            if left_tensor.shape != right_tensor.shape:
                raise ValueError(f"Cache tensor shape differs for {window_id}/{field}")
            difference = (left_tensor - right_tensor).abs()
            maxima[field] = max(maxima[field], float(difference.max().item()))
            if not torch.allclose(
                left_tensor, right_tensor, atol=atol, rtol=rtol, equal_nan=False
            ):
                raise ValueError(
                    f"Cache values differ beyond tolerance for {window_id}/{field}; "
                    f"maximum absolute difference={float(difference.max().item())}"
                )
    return {
        "equal_within_tolerance": True,
        "windows": len(left_ids),
        "atol": atol,
        "rtol": rtol,
        "max_absolute_difference": maxima,
    }
