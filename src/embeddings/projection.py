from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from src.common.tables import read_parquet

TIMESTEPS = 6
CHANNELS = 768
PERSISTENT_STATE_TOKENS = 768
STORAGE_BYTES_PER_VALUE = 2


def project_cache_storage(
    manifest_dir: str | Path,
    *,
    filesystem_path: str | Path,
    reserve_bytes: int = 10 * 1024**3,
    overhead_fraction: float = 0.05,
    available_bytes: int | None = None,
) -> dict[str, Any]:
    if reserve_bytes < 0:
        raise ValueError("reserve_bytes must be non-negative")
    if overhead_fraction < 0:
        raise ValueError("overhead_fraction must be non-negative")
    directory = Path(manifest_dir)
    frames = read_parquet(directory / "frames.parquet")
    windows = read_parquet(directory / "windows.parquet")
    frame_by_id = {row["frame_id"]: row for row in frames}
    tensor_bytes = 0
    token_counts: list[int] = []
    for window in windows:
        window_token_counts: set[int] = set()
        for frame_id in window["frame_ids"]:
            try:
                frame = frame_by_id[frame_id]
            except KeyError as error:
                raise KeyError(
                    f"Window {window['window_id']} references unknown frame {frame_id}"
                ) from error
            plan = json.loads(frame["spatial_transform_json"])
            patch_size = int(plan["patch_size"])
            tokens = (int(plan["output_height"]) // patch_size) * (
                int(plan["output_width"]) // patch_size
            )
            window_token_counts.add(tokens)
        if len(window_token_counts) != 1:
            raise ValueError(
                f"Window {window['window_id']} has inconsistent token grids: "
                f"{sorted(window_token_counts)}"
            )
        spatial_tokens = window_token_counts.pop()
        token_counts.append(spatial_tokens)
        image_bytes = TIMESTEPS * spatial_tokens * CHANNELS * STORAGE_BYTES_PER_VALUE
        state_bytes = (
            TIMESTEPS * PERSISTENT_STATE_TOKENS * CHANNELS * STORAGE_BYTES_PER_VALUE
        )
        tensor_bytes += image_bytes + state_bytes
    projected_cache_bytes = math.ceil(tensor_bytes * (1.0 + overhead_fraction))
    if available_bytes is None:
        available_bytes = shutil.disk_usage(filesystem_path).free
    required_free_bytes = projected_cache_bytes + reserve_bytes
    return {
        "windows": len(windows),
        "minimum_spatial_tokens": min(token_counts) if token_counts else None,
        "maximum_spatial_tokens": max(token_counts) if token_counts else None,
        "tensor_bytes": tensor_bytes,
        "overhead_fraction": overhead_fraction,
        "projected_cache_bytes": projected_cache_bytes,
        "reserve_bytes": reserve_bytes,
        "required_free_bytes": required_free_bytes,
        "available_bytes": available_bytes,
        "sufficient": available_bytes >= required_free_bytes,
    }
