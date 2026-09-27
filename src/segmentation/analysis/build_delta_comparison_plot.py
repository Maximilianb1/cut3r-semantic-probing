"""Build a side-by-side qualitative grid (Input | GT | A Pred | B Pred) for the
windows where two backbones' inference disagrees most -- the actual photos
behind a paired per-window comparison.

1. Joins both backbones' inference-<split>.json on window_id.
2. Selects windows by --rank-by:
   - "iou" (default): largest foreground-IoU delta (A minus B), one grid
     favoring each direction.
   - "precision-gap": the finding-7 over-prediction signature -- B is far
     more precise than A while A's recall still matches or beats B's (within
     --recall-margin), i.e. same coverage, worse precision. Computed from
     each row's tp/fp/fn, ranked by precision gap, one grid (A vs B is a
     fixed direction, not two).
3. Traces the selected windows' target frames via the manifest
   (windows.parquet -> frames.parquet) and downloads only those images
   (same targeted-fetch mechanism as build_qualitative_plots.py).
4. Reads both backbones' predicted/target label grids from their own
   masks-<split>.pt (already produced by inference_segmentation.py
   --save-masks) -- neither model is re-run.
5. Renders the grid(s) via figures.plot_backbone_comparison_grid.

Run example (iou mode, both directions):
    python -m src.segmentation.analysis.build_delta_comparison_plot \
        --manifest-dir ${CUT3R_ARTIFACT_ROOT}/manifests/full51-part-a-v1 \
                       ${CUT3R_ARTIFACT_ROOT}/manifests/part-a-leftover-windows-v1 \
                       ${CUT3R_ARTIFACT_ROOT}/manifests/full51-part-a-cap100-new-train-v1 \
        --dataset-root ${CO3D_ROOT} \
        --experiments-root src/segmentation/experiments \
        --backbone-a cut3r-trained --backbone-b dinov2 --run-suffix=-mlp

Run example (finding 7's over-prediction signature, cut3r-trained vs dinov2):
    python -m src.segmentation.analysis.build_delta_comparison_plot \
        --manifest-dir ${CUT3R_ARTIFACT_ROOT}/manifests/full51-part-a-v1 \
                       ${CUT3R_ARTIFACT_ROOT}/manifests/part-a-leftover-windows-v1 \
                       ${CUT3R_ARTIFACT_ROOT}/manifests/full51-part-a-cap100-new-train-v1 \
        --dataset-root ${CO3D_ROOT} \
        --experiments-root src/segmentation/experiments \
        --backbone-a cut3r-trained --backbone-b dinov2 --run-suffix=-mlp \
        --rank-by precision-gap
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
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
from .runs import DISPLAY_NAME, load_masks, load_per_window_iou, resolve_run_dir


def _precision_recall(row: dict) -> tuple[float, float]:
    """Per-window foreground precision/recall from a per_window_iou row's tp/fp/fn."""
    tp, fp, fn = row["tp"], row["fp"], row["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir", required=True, type=Path, nargs="+",
        help="One or more manifest dirs (windows.parquet + frames.parquet). The current "
             "unified split pools three partitions (see docs/data/part-a-cache-layout.md) -- "
             "a window_id can live in any one of them, so pass all three, e.g. "
             "full51-part-a-v1 part-a-leftover-windows-v1 full51-part-a-cap100-new-train-v1.",
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--experiments-root", required=True, type=Path)
    parser.add_argument("--backbone-a", required=True)
    parser.add_argument("--backbone-b", required=True)
    parser.add_argument("--run-suffix", default="", help="e.g. -mlp to match segmentation-<backbone>-mlp dirs")
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=5, help="windows shown per direction (top-k favoring each backbone)")
    parser.add_argument(
        "--rank-by", choices=["iou", "precision-gap"], default="iou",
        help="iou (default): largest per-window foreground-IoU delta, one grid favoring each "
             "backbone. precision-gap: the finding-7 over-prediction signature -- windows where "
             "B is far more precise than A while A's recall still matches or beats B's (within "
             "--recall-margin); one grid, ranked by precision gap (A -> B is a fixed direction).",
    )
    parser.add_argument(
        "--recall-margin", type=float, default=0.05,
        help="precision-gap mode only: keep windows with recall_a >= recall_b - margin",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="default: <experiments-root>/..")
    args = parser.parse_args()

    name_a = DISPLAY_NAME.get(args.backbone_a, args.backbone_a)
    name_b = DISPLAY_NAME.get(args.backbone_b, args.backbone_b)
    dir_a = resolve_run_dir(args.experiments_root, args.backbone_a, args.run_suffix)
    dir_b = resolve_run_dir(args.experiments_root, args.backbone_b, args.run_suffix)

    rows_a = load_per_window_iou(dir_a, args.split)
    rows_b = load_per_window_iou(dir_b, args.split)
    common = sorted(set(rows_a) & set(rows_b))

    # Each selected item is (window_id, category, caption, sort_score) -- the
    # caption is fully built here since the two modes show different numbers.
    if args.rank_by == "iou":
        deltas = sorted(
            (
                (wid, rows_a[wid]["category"],
                 f"{name_a} IoU={rows_a[wid]['foreground_iou']:.2f}\n"
                 f"{name_b} IoU={rows_b[wid]['foreground_iou']:.2f}",
                 rows_a[wid]["foreground_iou"] - rows_b[wid]["foreground_iou"])
                for wid in common
            ),
            key=lambda r: -r[3],
        )
        groups = [
            (deltas[: args.k], f"delta-favors-{args.backbone_a}",
             f"Largest {name_a} vs {name_b} IoU deltas -- favors {name_a}"),
            (list(reversed(deltas[-args.k :])), f"delta-favors-{args.backbone_b}",
             f"Largest {name_a} vs {name_b} IoU deltas -- favors {name_b}"),
        ]
        print(f"Common {args.split} windows: {len(common)}  |  top {args.k} each way selected")
    else:
        scored = []
        for wid in common:
            precision_a, recall_a = _precision_recall(rows_a[wid])
            precision_b, recall_b = _precision_recall(rows_b[wid])
            if recall_a < recall_b - args.recall_margin:
                continue  # not a matched-recall case -- A is just worse here, not less precise
            if precision_b <= precision_a:
                continue  # not an over-prediction case -- A isn't actually less precise here
            caption = (
                f"{name_a} P={precision_a:.2f} R={recall_a:.2f}\n"
                f"{name_b} P={precision_b:.2f} R={recall_b:.2f}"
            )
            scored.append((wid, rows_a[wid]["category"], caption, precision_b - precision_a))
        scored.sort(key=lambda r: -r[3])
        groups = [
            (scored[: args.k], f"precision-gap-{args.backbone_a}-vs-{args.backbone_b}",
             f"{name_a} vs {name_b}: precision gap at matched-or-higher {name_a} recall"),
        ]
        print(
            f"Common {args.split} windows: {len(common)}  |  "
            f"{len(scored)} pass the recall guard (margin={args.recall_margin})  |  "
            f"top {args.k} selected"
        )

    # Pool all given manifests -- the current unified split draws windows from
    # three disjoint partitions (docs/data/part-a-cache-layout.md), so a given
    # window_id lives in exactly one of them, not necessarily the first.
    windows: dict[str, dict] = {}
    frames: dict[str, dict] = {}
    for manifest_dir in args.manifest_dir:
        windows.update({w["window_id"]: w for w in read_parquet(manifest_dir / "windows.parquet")})
        frames.update({f["frame_id"]: f for f in read_parquet(manifest_dir / "frames.parquet")})

    selected = [item for group, _, _ in groups for item in group]
    window_to_relpath: dict[str, str] = {}
    needed_by_category: dict[str, set[str]] = defaultdict(set)
    for wid, category, _, _ in selected:
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

    masks_a = load_masks(dir_a, args.split)
    masks_b = load_masks(dir_b, args.split)

    output_dir = args.output_dir or args.experiments_root.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    def _render(group: list[tuple], filename_label: str, title: str) -> None:
        frame_imgs, gts, preds_a, preds_b, captions = [], [], [], [], []
        for wid, category, caption, _ in group:
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
            captions.append(f"{category}\n{caption}")
        save_path = output_dir / f"{filename_label}-{args.split}.png"
        plot_backbone_comparison_grid(
            np.stack(frame_imgs), np.stack(gts), np.stack(preds_a), np.stack(preds_b), captions,
            name_a=name_a, name_b=name_b, max_rows=args.k,
            title=title,
            save_path=save_path,
        )
        print(f"Saved -> {save_path}", flush=True)

    for group, filename_label, title in groups:
        if group:
            _render(group, filename_label, title)
        else:
            print(f"Skipping {filename_label}: no windows passed the filter", flush=True)


if __name__ == "__main__":
    main()
