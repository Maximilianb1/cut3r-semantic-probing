"""Build worst-5 / best-5 qualitative segmentation grids (Input | GT | Pred | Overlay)
for each backbone's test-split inference output.

1. Reads inference-<split>.json to rank test windows by foreground IoU.
2. Traces each window's target frame to its image_relpath via the manifest
   (windows.parquet -> frames.parquet).
3. Downloads only the needed target-frame images via
   scripts.download_co3d_targeted's primitives (no re-sampling: exactly the
   files the manifest already names).
4. Reads predicted/target label grids from masks-<split>.pt (already produced
   by `python -m src.segmentation.inference_segmentation --save-masks`) --
   the model is never re-run.
5. Calls the existing plot_segmentation_results (this module) unchanged.

Run example:
    python -m src.segmentation.build_qualitative_plots \
        --manifest-dir ${CUT3R_ARTIFACT_ROOT}/manifests/full51-part-a-v1 \
        --dataset-root ${CO3D_ROOT} \
        --experiments-root src/segmentation/experiments \
        --backbones cut3r_trained cut3r_random dinov2 --run-suffix -bestval
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.common.tables import read_parquet
from src.data.co3d_selective import (
    DEFAULT_CHECKSUMS_URL,
    DEFAULT_LINKS_URL,
    _fetch_json,
    _remote_archive_opener,
    find_required_members,
    materialize_required_members,
)

from .visualizations import plot_segmentation_results


def pick_extremes(inference_json: Path, k: int = 5) -> tuple[list[dict], list[dict]]:
    data = json.loads(inference_json.read_text(encoding="utf-8"))
    rows = sorted(data["per_window_iou"], key=lambda r: r["foreground_iou"])
    return rows[:k], rows[-k:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--backbones", nargs="+", required=True)
    parser.add_argument("--run-suffix", default="", help="e.g. -bestval to match segmentation-<backbone>-bestval dirs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    windows = {w["window_id"]: w for w in read_parquet(args.manifest_dir / "windows.parquet")}
    frames = {f["frame_id"]: f for f in read_parquet(args.manifest_dir / "frames.parquet")}

    per_backbone_selection: dict[str, dict[str, list[dict]]] = {}
    needed_by_category: dict[str, set[str]] = defaultdict(set)
    window_to_relpath: dict[str, str] = {}

    for backbone in args.backbones:
        exp_dir = args.experiments_root / f"segmentation-{backbone}{args.run_suffix}"
        worst, best = pick_extremes(exp_dir / f"inference-{args.split}.json", args.k)
        per_backbone_selection[backbone] = {"worst": worst, "best": best}
        for row in worst + best:
            window = windows[row["window_id"]]
            frame = frames[window["target_frame_id"]]
            window_to_relpath[row["window_id"]] = frame["image_relpath"]
            needed_by_category[frame["category"]].add(frame["image_relpath"])

    total = sum(len(v) for v in needed_by_category.values())
    print(f"Need {total} unique target-frame images across {len(needed_by_category)} categories", flush=True)

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    links, _ = _fetch_json(DEFAULT_LINKS_URL, timeout=args.timeout)
    checksums, _ = _fetch_json(DEFAULT_CHECKSUMS_URL, timeout=args.timeout)
    full_links = links["full"]
    full_checksums = checksums["full"]
    opener = _remote_archive_opener(timeout=args.timeout, initial_buffer_size=16 * 1024 * 1024)

    for category, paths in needed_by_category.items():
        urls = full_links[category]
        metadata_name = f"{category}_000.zip"
        data_urls = [u for u in urls if u.rsplit("/", 1)[-1] != metadata_name]
        print(f"[{category}] fetching {len(paths)} image(s)", flush=True)
        found = find_required_members(data_urls, sorted(paths), archive_opener=opener,
                                       progress=lambda m: print(f"  {m}", flush=True))
        sources = {r: {**s, "official_archive_sha256": full_checksums[s["archive"]]} for r, s in found.items()}
        materialize_required_members(sources, dataset_root=args.dataset_root, archive_opener=opener,
                                      progress=lambda m: print(f"  {m}", flush=True))

    for backbone in args.backbones:
        exp_dir = args.experiments_root / f"segmentation-{backbone}{args.run_suffix}"
        masks = torch.load(exp_dir / f"masks-{args.split}.pt", weights_only=False)
        for label, rows in per_backbone_selection[backbone].items():
            frame_imgs, gts, preds, captions = [], [], [], []
            for row in rows:
                wid = row["window_id"]
                img = Image.open(args.dataset_root / window_to_relpath[wid]).convert("RGB")
                frame_imgs.append(np.array(img.resize((256, 256))))
                mask_entry = masks[wid]
                # Token grids vary in shape across windows (CO3D aspect ratios differ),
                # so resize to a common display size before stacking. Nearest-neighbor
                # keeps the per-token blockiness instead of blurring it away.
                gt_img = Image.fromarray(mask_entry["target_labels"].numpy().astype(np.uint8) * 255)
                pred_img = Image.fromarray(mask_entry["predicted_labels"].numpy().astype(np.uint8) * 255)
                gts.append(np.array(gt_img.resize((256, 256), Image.NEAREST)) / 255.0)
                preds.append(np.array(pred_img.resize((256, 256), Image.NEAREST)) / 255.0)
                captions.append(f"{row['category']} IoU={row['foreground_iou']:.3f}")
            save_path = exp_dir / f"qualitative-{label}{args.k}-{args.split}.png"
            plot_segmentation_results(
                np.stack(frame_imgs), np.stack(gts), np.stack(preds), max_rows=args.k,
                save_path=save_path,
            )
            print(f"[{backbone}] saved {label} -> {save_path}  ({', '.join(captions)})", flush=True)


if __name__ == "__main__":
    main()
