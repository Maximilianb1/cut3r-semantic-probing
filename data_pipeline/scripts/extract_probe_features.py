"""Extract the probe-feature (embedding + labels) cache for one backbone.

Config-driven, mirroring ``scripts.extract_features``. Runs one backbone over
the CO3Dv2 window manifests and writes an embedding cache containing, per window,
the backbone's embeddings and both task labels (segmentation grid + class index).

Layout is derived from the backbone (override with ``extraction.layout``):

- CUT3R-trained  -> ``trajectory``  (grid + latent for all six states)
- CUT3R-random   -> ``target_only`` (grid + latent, last state)
- DINOv2         -> ``target_only`` (grid + latent, last state)

Run one config at a time (from ``data_pipeline/``)::

    python -m scripts.extract_probe_features \
      --config configs/probe_features/cut3r_trained.yaml

Add ``--limit-windows N`` for a quick smoke, ``--device`` / ``--cache-dir`` to
override. Requires the CO3Dv2 manifests + data and (for CUT3R) the CUT3R repo +
checkpoint; intended to run on the GPU VM for the full 51-category set.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any

import torch

from src.backbones import build_backbone
from src.backbones.probe_cache import (
    TARGET_ONLY,
    TRAJECTORY,
    attach_labels_from_trajectory_cache,
    extract_to_cache,
)
from src.common.io import load_json, load_yaml, sha256_file
from src.common.tables import read_parquet
from src.data.validation import validate_manifests


def _load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config and resolve ``${ENV_VAR}`` references (e.g. ``${CO3D_ROOT}``)."""
    if path.suffix not in (".yaml", ".yml"):
        raise ValueError(f"Config must be .yaml or .yml, got {path.suffix!r}: {path}")
    return load_yaml(path)


def _default_layout(backbone_cfg: dict[str, Any]) -> str:
    kind = backbone_cfg.get("kind")
    weights = backbone_cfg.get("weights", "trained")
    if kind == "cut3r" and weights == "trained":
        return TRAJECTORY
    return TARGET_ONLY


def run(config_path: Path, *, limit_windows: int | None, device_override: str | None,
        cache_dir_override: str | None, manifest_dir_override: str | None = None) -> dict[str, Any]:
    config = _load_config(config_path)
    backbone_cfg = dict(config["backbone"])
    data_cfg = config["data"]
    extraction_cfg = config.get("extraction", {})

    manifest_dir = Path(manifest_dir_override or data_cfg["manifest_dir"])
    dataset_root = str(data_cfg["dataset_root"])
    frames_path = manifest_dir / "frames.parquet"
    windows_path = manifest_dir / "windows.parquet"
    for path in (frames_path, windows_path):
        if not path.is_file():
            raise FileNotFoundError(f"Manifest is missing: {path}")

    if extraction_cfg.get("validate_manifests", True):
        validate_manifests(
            manifest_dir,
            dataset_root=dataset_root,
            inspect_files=bool(extraction_cfg.get("inspect_files", False)),
        )

    source = extraction_cfg.get("source", "backbone")
    layout = extraction_cfg.get("layout") or _default_layout(backbone_cfg)
    mask_threshold = float(extraction_cfg.get("mask_threshold", config.get("mask_threshold", 0.5)))
    windows_per_shard = int(extraction_cfg.get("windows_per_shard", 32))
    cache_dir = cache_dir_override or config["probe_cache"]["dir"]
    limit = limit_windows if limit_windows is not None else extraction_cfg.get("limit_windows")
    verify_hashes = bool(extraction_cfg.get("verify_source_hashes", True))

    torch.manual_seed(int(extraction_cfg.get("seed", 0)))

    frames = read_parquet(frames_path)
    frame_by_id = {row["frame_id"]: row for row in frames}
    windows = read_parquet(windows_path)

    summary_path = manifest_dir / "summary.json"
    contract = {
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": (load_json(summary_path).get("manifest_sha256") if summary_path.is_file() else None),
        "stack": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
        },
    }

    started = time.perf_counter()
    if source == "existing_embeddings":
        # CUT3R-trained reuse path: no GPU, no re-extraction; attach labels only.
        options = backbone_cfg.get("options", {})
        result = attach_labels_from_trajectory_cache(
            extraction_cfg["trajectory_cache_dir"],
            windows=windows,
            frame_by_id=frame_by_id,
            dataset_root=dataset_root,
            cache_dir=cache_dir,
            contract=contract,
            input_size=int(options.get("input_size", 512)),
            patch_size=int(options.get("patch_size", 16)),
            square_ok=bool(options.get("square_ok", False)),
            mask_threshold=mask_threshold,
            windows_per_shard=windows_per_shard,
            limit_windows=limit,
            verify_source_hashes=verify_hashes,
        )
        layout = TRAJECTORY
        peak_cuda = None
    else:
        if str(device := (device_override or extraction_cfg.get("device")
                          or backbone_cfg.get("options", {}).get("device", "cuda"))).startswith("cuda") \
                and not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA but torch.cuda.is_available() is false")
        backbone_cfg.setdefault("options", {})["device"] = device
        contract["stack"]["device"] = (
            torch.cuda.get_device_name(0) if str(device).startswith("cuda") else "cpu"
        )
        backbone = build_backbone(backbone_cfg)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        result = extract_to_cache(
            backbone,
            layout=layout,
            windows=windows,
            frame_by_id=frame_by_id,
            dataset_root=dataset_root,
            cache_dir=cache_dir,
            contract=contract,
            mask_threshold=mask_threshold,
            windows_per_shard=windows_per_shard,
            limit_windows=limit,
            verify_source_hashes=verify_hashes,
        )
        peak_cuda = (
            torch.cuda.max_memory_allocated() if str(device).startswith("cuda") else None
        )
    elapsed = time.perf_counter() - started
    result["elapsed_seconds"] = elapsed
    result["seconds_per_written_window"] = (
        None if result["written"] == 0 else elapsed / result["written"]
    )
    result["peak_cuda_bytes"] = peak_cuda
    result["layout"] = layout
    result["cache_dir"] = str(cache_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--manifest-dir", default=None, help="Override data.manifest_dir (e.g. for part B)")
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        limit_windows=arguments.limit_windows,
        device_override=arguments.device,
        cache_dir_override=arguments.cache_dir,
        manifest_dir_override=arguments.manifest_dir,
    )
    print(
        f"layout={result['layout']} written={result['written']} skipped={result['skipped']} "
        f"elapsed={result['elapsed_seconds']:.1f}s "
        f"per_window={result['seconds_per_written_window']} "
        f"peak_cuda_bytes={result['peak_cuda_bytes']} "
        f"verification={result['verification']}"
    )
    print(f"cache_dir={result['cache_dir']}")


if __name__ == "__main__":
    main()
