"""Build a side-by-side qualitative grid (Input | GT | A Pred | B Pred) for the
windows where two backbones' foreground IoU differs the most -- the actual
photos behind build_bootstrap_ci.py's paired per-window comparison.

1. Joins both backbones' inference-<split>.json on window_id, ranks by
   foreground-IoU delta (A minus B).
2. Traces the top/bottom-k windows' target frames via the manifest
   (windows.parquet -> frames.parquet) and downloads only those images
   (same targeted-fetch mechanism as build_qualitative_plots.py).
3. Reads both backbones' predicted/target label grids from their own
   masks-<split>.pt (already produced by inference_segmentation.py
   --save-masks) -- neither model is re-run.
4. Renders two grids via figures.plot_backbone_comparison_grid: one
   for windows favoring A, one for windows favoring B.

Run example:
    python -m src.segmentation.analysis.build_delta_comparison_plot \
        --manifest-dir ${CUT3R_ARTIFACT_ROOT}/manifests/full51-part-a-v1 \
        --dataset-root ${CO3D_ROOT} \
        --experiments-root src/segmentation/experiments \
        --backbone-a cut3r_trained --backbone-b dinov2 --run-suffix=-expanded-bestval
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

from .figures import plot_backbone_comparison_grid

_DISPLAY_NAME = {
    "cut3r_trained": "CUT3R-trained",
    "cut3r_random": "CUT3R-random",
    "dinov2": "DINOv2",
}


def load_per_window_iou(run_dir: Path, split: str) -> dict[str, dict]:
    data = json.loads((run_dir / f"inference-{split}.json").read_text(encoding="utf-8"))
    return {row["window_id"]: row for row in data["per_window_iou"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--backbone-a", required=True)
    parser.add_argument("--backbone-b", required=True)
    parser.add_argument("--run-suffix", default="", help="e.g. -expanded-bestval to match segmentation-<backbone>-expanded-bestval dirs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=5, help="windows shown per direction (top-k favoring each backbone)")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="default: <experiments-root>/..")
    args = parser.parse_args()

    name_a = _DISPLAY_NAME.get(args.backbone_a, args.backbone_a)
    name_b = _DISPLAY_NAME.get(args.backbone_b, args.backbone_b)
    dir_a = args.experiments_root / f"segmentation-{args.backbone_a}{args.run_suffix}"
    dir_b = args.experiments_root / f"segmentation-{args.backbone_b}{args.run_suffix}"

    rows_a = load_per_window_iou(dir_a, args.split)
    rows_b = load_per_window_iou(dir_b, args.split)
    common = sorted(set(rows_a) & set(rows_b))
    deltas = sorted(
        ((wid, rows_a[wid]["category"], rows_a[wid]["foreground_iou"], rows_b[wid]["foreground_iou"],
          rows_a[wid]["foreground_iou"] - rows_b[wid]["foreground_iou"]) for wid in common),
        key=lambda r: -r[4],
    )
    favors_a = deltas[: args.k]
    favors_b = list(reversed(deltas[-args.k :]))
    print(f"Common {args.split} windows: {len(common)}  |  top {args.k} each way selected")

    windows = {w["window_id"]: w for w in read_parquet(args.manifest_dir / "windows.parquet")}
    frames = {f["frame_id"]: f for f in read_parquet(args.manifest_dir / "frames.parquet")}

    selected = favors_a + favors_b
    window_to_relpath: dict[str, str] = {}
    needed_by_category: dict[str, set[str]] = defaultdict(set)
    for wid, category, _, _, _ in selected:
        window = windows[wid]
        frame = frames[window["target_frame_id"]]
        window_to_relpath[wid] = frame["image_relpath"]
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

    masks_a = torch.load(dir_a / f"masks-{args.split}.pt", weights_only=False)
    masks_b = torch.load(dir_b / f"masks-{args.split}.pt", weights_only=False)

    output_dir = args.output_dir or args.experiments_root.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    def _render(group: list[tuple], label: str) -> None:
        frame_imgs, gts, preds_a, preds_b, captions = [], [], [], [], []
        for wid, category, iou_a, iou_b, delta in group:
            img = Image.open(args.dataset_root / window_to_relpath[wid]).convert("RGB")
            frame_imgs.append(np.array(img.resize((256, 256))))
            # Ground truth is identical in both masks files (same window, same
            # manifest); read it from A's for convenience.
            gt_img = Image.fromarray(masks_a[wid]["target_labels"].numpy().astype(np.uint8) * 255)
            pred_a_img = Image.fromarray(masks_a[wid]["predicted_labels"].numpy().astype(np.uint8) * 255)
            pred_b_img = Image.fromarray(masks_b[wid]["predicted_labels"].numpy().astype(np.uint8) * 255)
            gts.append(np.array(gt_img.resize((256, 256), Image.NEAREST)) / 255.0)
            preds_a.append(np.array(pred_a_img.resize((256, 256), Image.NEAREST)) / 255.0)
            preds_b.append(np.array(pred_b_img.resize((256, 256), Image.NEAREST)) / 255.0)
            captions.append(f"{category}\n{name_a}={iou_a:.2f}\n{name_b}={iou_b:.2f}")
        save_path = output_dir / f"delta-favors-{label}-{args.split}.png"
        plot_backbone_comparison_grid(
            np.stack(frame_imgs), np.stack(gts), np.stack(preds_a), np.stack(preds_b), captions,
            name_a=name_a, name_b=name_b, max_rows=args.k,
            title=f"Largest {name_a} vs {name_b} deltas -- favors {_DISPLAY_NAME.get(label, label)}",
            save_path=save_path,
        )
        print(f"Saved -> {save_path}", flush=True)

    _render(favors_a, args.backbone_a)
    _render(favors_b, args.backbone_b)


if __name__ == "__main__":
    main()
