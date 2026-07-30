from __future__ import annotations

import importlib
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from src.common.io import load_json, load_yaml, sha256_file
from src.common.tables import read_parquet
from src.data.validation import validate_manifests
from src.embeddings.cache import FeatureCacheWriter, verify_cache
from src.embeddings.checkpoint import trusted_cut3r_checkpoint
from src.embeddings.cut3r_adapter import Cut3rFeatureExtractor
from src.embeddings.cut3r_provenance import validate_cut3r_checkout
from src.embeddings.input import move_views_to_device, prepare_image_window


def load_cut3r_model(
    cut3r_root: str | Path,
    checkpoint: str | Path,
    device: torch.device,
    *,
    expected_checkpoint_sha256: str,
) -> tuple[Any, dict[str, Any]]:
    root = Path(cut3r_root).resolve()
    source = root / "src"
    if not (source / "dust3r" / "model.py").is_file():
        raise FileNotFoundError(f"CUT3R source tree not found under {source}")
    existing = sys.modules.get("dust3r.model")
    if existing is not None:
        existing_file = Path(existing.__file__).resolve()
        try:
            existing_file.relative_to(source.resolve())
        except ValueError as error:
            raise RuntimeError(
                "dust3r.model is already imported from a different source tree: "
                f"{existing_file}"
            ) from error
    existing_package = sys.modules.get("dust3r")
    if existing_package is not None:
        package_paths = [Path(path).resolve() for path in existing_package.__path__]
        if source.resolve() / "dust3r" not in package_paths:
            raise RuntimeError(
                "dust3r is already imported from a different source tree: "
                f"{package_paths}"
            )
    with trusted_cut3r_checkpoint(
        checkpoint, expected_sha256=expected_checkpoint_sha256
    ) as checkpoint_provenance:
        sys.path.insert(0, str(source))
        try:
            module = importlib.import_module("dust3r.model")
            module_path = Path(module.__file__).resolve()
            try:
                module_path.relative_to(source.resolve())
            except ValueError as error:
                raise RuntimeError(
                    "Imported dust3r.model from "
                    f"{module_path}, expected it under {source}"
                ) from error
            model = module.ARCroco3DStereo.from_pretrained(
                str(Path(checkpoint).resolve())
            )
        finally:
            if sys.path[0] == str(source):
                sys.path.pop(0)
    model.to(device)
    model.eval()
    return model, checkpoint_provenance


def run_extraction(
    config_path: str | Path,
    *,
    limit_windows: int | None = None,
    cache_directory: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    model_cfg = config["model"]
    preprocessing_cfg = config["preprocessing"]
    cache_cfg = config["cache"]
    cache_dir = (
        Path(cache_directory)
        if cache_directory is not None
        else Path(cache_cfg["directory"])
    )
    data_cfg = config["dataset"]
    manifest_dir = Path(config["output"]["manifest_dir"])
    frames_path = manifest_dir / "frames.parquet"
    windows_path = manifest_dir / "windows.parquet"
    summary_path = manifest_dir / "summary.json"
    for path in (frames_path, windows_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Extraction input is missing: {path}")
    validate_manifests(manifest_dir, dataset_root=data_cfg["root"], inspect_files=True)
    checkpoint = Path(model_cfg["checkpoint"])
    cut3r_root = Path(model_cfg["cut3r_root"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CUT3R checkpoint does not exist: {checkpoint}")
    requested_device = str(model_cfg.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA extraction was requested but torch.cuda.is_available() is false"
        )
    device = torch.device(requested_device)
    manifest_summary = load_json(summary_path)
    expected_commit = str(model_cfg["expected_commit"])
    cut3r_provenance = validate_cut3r_checkout(
        cut3r_root,
        expected_commit=expected_commit,
        expected_patch=model_cfg.get("compatibility_patch"),
        require_compiled_extension=True,
    )
    model, checkpoint_provenance = load_cut3r_model(
        cut3r_root,
        checkpoint,
        device,
        expected_checkpoint_sha256=str(model_cfg["checkpoint_sha256"]),
    )
    contract = {
        "config_sha256": sha256_file(config_path),
        "checkpoint_provenance": checkpoint_provenance,
        "cut3r_provenance": cut3r_provenance,
        "manifest_sha256": manifest_summary["manifest_sha256"],
        "input_size": int(preprocessing_cfg["input_size"]),
        "patch_size": int(preprocessing_cfg["patch_size"]),
        "square_ok": bool(preprocessing_cfg["square_ok"]),
        "mask_threshold": float(preprocessing_cfg["mask_threshold"]),
        "timesteps": [1, 2, 3, 4, 5, 6],
        "representations": ["image_tokens", "state_tokens"],
        "storage_dtype": "float16",
        "source_file_hashes": "sha256",
        "target_rgb_visible": True,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
            ),
        },
    }
    frames = read_parquet(frames_path)
    frame_by_id = {row["frame_id"]: row for row in frames}
    windows = read_parquet(windows_path)
    if limit_windows is not None:
        if limit_windows < 1:
            raise ValueError("limit_windows must be positive")
        windows = windows[:limit_windows]
    extractor = Cut3rFeatureExtractor(model)
    started = time.perf_counter()
    written = skipped = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with FeatureCacheWriter(
        cache_dir,
        contract=contract,
        windows_per_shard=int(cache_cfg.get("windows_per_shard", 32)),
    ) as writer:
        for window in windows:
            window_id = window["window_id"]
            if writer.contains(window_id):
                skipped += 1
                continue
            try:
                rows = [frame_by_id[frame_id] for frame_id in window["frame_ids"]]
            except KeyError as error:
                raise KeyError(
                    f"Window {window_id} references a frame missing from frames.parquet"
                ) from error
            root = Path(data_cfg["root"])
            source_sha256 = {
                row["frame_id"]: {
                    "image": sha256_file(root / row["image_relpath"]),
                    "mask": sha256_file(root / row["mask_relpath"]),
                }
                for row in rows
            }
            views, _masks, token_grid = prepare_image_window(
                rows,
                dataset_root=data_cfg["root"],
                input_size=contract["input_size"],
                patch_size=contract["patch_size"],
                square_ok=contract["square_ok"],
            )
            move_views_to_device(views, device)
            trajectory = extractor.extract(
                views, frame_ids=list(window["frame_ids"]), token_grid=token_grid
            )
            writer.add(window_id, trajectory, source_sha256=source_sha256)
            written += 1
    elapsed = time.perf_counter() - started
    verification = verify_cache(cache_dir)
    return {
        "written": written,
        "skipped": skipped,
        "elapsed_seconds": elapsed,
        "seconds_per_written_window": None if written == 0 else elapsed / written,
        "peak_cuda_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "verification": verification,
    }
