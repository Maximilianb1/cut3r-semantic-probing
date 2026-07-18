from __future__ import annotations

from pathlib import Path

from src.data.co3d import build_manifests
from src.embeddings.projection import project_cache_storage


def test_cache_projection_uses_manifest_token_grids(
    synthetic_co3d: tuple[Path, Path, Path],
) -> None:
    _root, config_path, manifest_dir = synthetic_co3d
    summary = build_manifests(config_path)
    result = project_cache_storage(
        manifest_dir,
        filesystem_path=manifest_dir,
        reserve_bytes=100,
        overhead_fraction=0.0,
        available_bytes=10**12,
    )
    assert result["windows"] == summary["window_count"] == 6
    assert result["minimum_spatial_tokens"] == 196
    assert result["maximum_spatial_tokens"] == 196
    expected_per_window = 6 * 196 * 768 * 2 + 6 * 768 * 768 * 2
    assert result["tensor_bytes"] == expected_per_window * 6
    assert result["projected_cache_bytes"] == result["tensor_bytes"]
    assert result["required_free_bytes"] == result["tensor_bytes"] + 100
    assert result["sufficient"] is True


def test_cache_projection_enforces_reserve(
    synthetic_co3d: tuple[Path, Path, Path],
) -> None:
    _root, config_path, manifest_dir = synthetic_co3d
    build_manifests(config_path)
    result = project_cache_storage(
        manifest_dir,
        filesystem_path=manifest_dir,
        reserve_bytes=1,
        overhead_fraction=0.05,
        available_bytes=1,
    )
    assert result["sufficient"] is False
