from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.common.tables import read_parquet, write_parquet_atomic
from src.embeddings.cache import (
    FeatureCacheWriter,
    compare_caches,
    load_trajectory,
    verify_cache,
)
from src.embeddings.types import FeatureTrajectory


def _trajectory(offset: float = 0.0) -> FeatureTrajectory:
    return FeatureTrajectory(
        image_tokens=torch.arange(6 * 1 * 4 * 3, dtype=torch.float32).reshape(
            6, 1, 4, 3
        )
        + offset,
        state_tokens=torch.arange(6 * 1 * 2 * 3, dtype=torch.float32).reshape(
            6, 1, 2, 3
        )
        + offset,
        frame_ids=[f"frame-{index}" for index in range(6)],
        token_grid=(2, 2),
    )


def test_cache_round_trip_resume_and_verification(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    contract = {"checkpoint_sha256": "abc", "manifest_sha256": {"windows": "def"}}
    with FeatureCacheWriter(
        cache_dir, contract=contract, windows_per_shard=2
    ) as writer:
        assert writer.add("window-a", _trajectory())
        assert writer.add("window-b", _trajectory(1))
    result = verify_cache(cache_dir)
    assert result["windows"] == 2
    assert result["shards"] == 1
    loaded = load_trajectory(cache_dir, "window-b")
    assert loaded.image_tokens.dtype == torch.float16
    assert torch.equal(loaded.image_tokens, _trajectory(1).image_tokens.half())
    with FeatureCacheWriter(
        cache_dir, contract=contract, windows_per_shard=2
    ) as writer:
        assert not writer.add("window-a", _trajectory())
        assert writer.add("window-c", _trajectory(2))
    assert verify_cache(cache_dir)["windows"] == 3


def test_incompatible_cache_contract_is_rejected(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    with FeatureCacheWriter(cache_dir, contract={"checkpoint": "a"}) as writer:
        writer.add("window-a", _trajectory())
    with pytest.raises(ValueError, match="incompatible"):
        FeatureCacheWriter(cache_dir, contract={"checkpoint": "b"})


def test_non_finite_trajectory_is_rejected(tmp_path: Path) -> None:
    trajectory = _trajectory()
    trajectory.image_tokens[0, 0, 0, 0] = torch.nan
    with (
        FeatureCacheWriter(tmp_path / "cache", contract={"checkpoint": "a"}) as writer,
        pytest.raises(ValueError, match="NaN"),
    ):
        writer.add("bad", trajectory)


def test_resume_uses_next_shard_number_when_index_has_a_gap(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    contract = {"checkpoint": "a"}
    with FeatureCacheWriter(
        cache_dir, contract=contract, windows_per_shard=1
    ) as writer:
        writer.add("window-a", _trajectory())
        writer.add("window-b", _trajectory(1))
    rows = read_parquet(cache_dir / "index.parquet")
    write_parquet_atomic(cache_dir / "index.parquet", rows[1:])
    (cache_dir / "shard-00000.safetensors").unlink()
    with FeatureCacheWriter(
        cache_dir, contract=contract, windows_per_shard=1
    ) as writer:
        writer.add("window-c", _trajectory(2))
    assert (cache_dir / "shard-00002.safetensors").is_file()
    assert verify_cache(cache_dir)["windows"] == 2


def test_orphaned_shard_is_rejected(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    contract = {"checkpoint": "a"}
    with FeatureCacheWriter(cache_dir, contract=contract) as writer:
        writer.add("window-a", _trajectory())
    (cache_dir / "shard-99999.safetensors").write_bytes(b"incomplete")
    with pytest.raises(ValueError, match="disagree with its index"):
        verify_cache(cache_dir)
    with pytest.raises(RuntimeError, match="absent from its index"):
        FeatureCacheWriter(cache_dir, contract=contract)


def test_independent_cache_comparison_detects_changes(tmp_path: Path) -> None:
    directories = [tmp_path / "left", tmp_path / "right"]
    for directory in directories:
        with FeatureCacheWriter(directory, contract={"checkpoint": "a"}) as writer:
            writer.add("window-a", _trajectory())
    assert compare_caches(*directories)["equal_within_tolerance"]
    changed = tmp_path / "changed"
    with FeatureCacheWriter(changed, contract={"checkpoint": "a"}) as writer:
        writer.add("window-a", _trajectory(1))
    with pytest.raises(ValueError, match="values differ"):
        compare_caches(directories[0], changed)


def test_required_source_hashes_must_cover_all_six_frames(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    with FeatureCacheWriter(
        cache_dir, contract={"source_file_hashes": "sha256"}
    ) as writer:
        writer.add("window-a", _trajectory())
    with pytest.raises(ValueError, match="Incomplete source hashes"):
        verify_cache(cache_dir)
