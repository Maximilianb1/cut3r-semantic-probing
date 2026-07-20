from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.common.io import load_yaml
from src.common.tables import read_parquet
from src.embeddings.audit import (
    compare_to_reference,
    export_audit_reference,
    load_audit_reference,
    render_inputs_and_features,
    render_point_cloud_projections,
    resolve_reference_frames,
    save_point_cloud_ply,
)
from src.embeddings.cut3r_adapter import Cut3rFeatureExtractor
from src.embeddings.extract import load_cut3r_model
from src.embeddings.input import move_views_to_device, prepare_image_window


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit one cached CUT3R window and optionally reconstruct it"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="Export one small cache reference")
    export.add_argument("--cache-dir", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--window-id")
    export.add_argument("--window-index", type=int, default=0)
    inspect = subparsers.add_parser(
        "inspect", help="Verify source mapping and render inputs/token PCA"
    )
    inspect.add_argument("--reference-dir", required=True)
    inspect.add_argument("--dataset-root", required=True)
    inspect.add_argument("--output", required=True)
    reconstruct = subparsers.add_parser(
        "reconstruct", help="Re-run CUT3R and render its 3D prediction"
    )
    reconstruct.add_argument("--reference-dir", required=True)
    reconstruct.add_argument("--config", required=True)
    reconstruct.add_argument("--output-dir", required=True)
    return parser


def _reconstruct(args: argparse.Namespace) -> dict[str, object]:
    reference, reference_tensors = load_audit_reference(args.reference_dir)
    config = load_yaml(args.config)
    dataset_root = config["dataset"]["root"]
    rows = resolve_reference_frames(reference, dataset_root)
    manifest_dir = Path(config["output"]["manifest_dir"])
    manifest_rows = {
        row["frame_id"]: row for row in read_parquet(manifest_dir / "frames.parquet")
    }
    rows = [manifest_rows[row["frame_id"]] for row in rows]
    preprocessing = config["preprocessing"]
    views, _masks, token_grid = prepare_image_window(
        rows,
        dataset_root=dataset_root,
        input_size=int(preprocessing["input_size"]),
        patch_size=int(preprocessing["patch_size"]),
        square_ok=bool(preprocessing["square_ok"]),
    )
    if list(token_grid) != list(reference["token_grid"]):
        raise ValueError("Fresh preprocessing token grid differs from reference")
    device = torch.device(config["model"].get("device", "cuda"))
    move_views_to_device(views, device)
    model, checkpoint_provenance = load_cut3r_model(
        config["model"]["cut3r_root"],
        config["model"]["checkpoint"],
        device,
        expected_checkpoint_sha256=config["model"]["checkpoint_sha256"],
    )
    fresh = Cut3rFeatureExtractor(model).extract(
        views,
        frame_ids=list(reference["frame_ids"]),
        token_grid=token_grid,
    )
    comparison = compare_to_reference(
        reference_tensors,
        image_tokens=fresh.image_tokens,
        state_tokens=fresh.state_tokens,
    )
    with torch.inference_mode():
        predictions = model(views).ress
    source_colors = [
        ((view["img"][0].permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5)
        .clip(0, 255)
        .astype(np.uint8)
        for view in views
    ]
    from dust3r.utils.camera import pose_encoding_to_camera
    from dust3r.utils.geometry import geotrf

    points_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for prediction, color in zip(predictions, source_colors, strict=True):
        pose = pose_encoding_to_camera(prediction["camera_pose"].clone())
        points = geotrf(pose, prediction["pts3d_in_self_view"])[0]
        confidence = prediction["conf_self"][0]
        threshold = torch.quantile(confidence, 0.5)
        selection = confidence >= threshold
        points_parts.append(points[selection].detach().cpu().numpy()[::4])
        color_parts.append(color[selection.detach().cpu().numpy()][::4])
    points = np.concatenate(points_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_point_cloud_ply(output_dir / "reconstruction.ply", points, colors)
    render_point_cloud_projections(
        output_dir / "reconstruction-projections.png", points, colors
    )
    result: dict[str, object] = {
        "window_id": reference["window_id"],
        "frame_ids": reference["frame_ids"],
        "fresh_features_match_cache_exactly": comparison,
        "checkpoint_provenance": checkpoint_provenance,
        "predicted_frames": len(predictions),
        "rendered_points": int(len(points)),
        "reconstruction_ply": str(output_dir / "reconstruction.ply"),
        "reconstruction_projections": str(
            output_dir / "reconstruction-projections.png"
        ),
    }
    from src.common.io import atomic_write_json

    atomic_write_json(output_dir / "reconstruction-audit.json", result)
    return result


def main() -> None:
    args = _parser().parse_args()
    if args.command == "export":
        result = export_audit_reference(
            args.cache_dir,
            args.output_dir,
            window_id=args.window_id,
            window_index=args.window_index,
        )
    elif args.command == "inspect":
        result = render_inputs_and_features(
            args.reference_dir, args.dataset_root, args.output
        )
    else:
        result = _reconstruct(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
