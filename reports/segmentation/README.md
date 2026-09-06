# Segmentation results

Foreground/background probe results for three frozen backbones, at two probe
capacities, over one shared held-out test split.

## Setup

- **Backbones** (frozen, never fine-tuned): CUT3R-trained (the pretrained 3D
  reconstruction model — the subject of the project), CUT3R-random (same
  architecture, untrained weights — the control for what pretraining
  contributes), DINOv2 ViT-B/14 (an off-the-shelf self-supervised model — the
  upper anchor for "backbone known to encode strong semantic content").
- **Probe**: a per-token head trained on cached features. `linear` =
  `hidden_dims: []`, a true linear probe. `mlp` = one 512-unit hidden layer.
- **Data**: 8,540 train / 1,075 val / 1,077 test windows, 26 CO3D categories,
  sequence-disjoint split (`configs/segmentation_split_override.json`, derived
  by `scripts/derive_segmentation_split.py`), features standardized per
  backbone. Same split, optimizer, epochs, and seed across all six runs — that
  is what makes them comparable to each other.
- **Checkpoint**: best validation macro-foreground-IoU across 20 epochs
  (`--checkpoint-selection best_val`), not whichever epoch training happened
  to stop on.

## Files

Per run (`<backbone>-<probe>/`): `metrics.json` (full per-epoch train/val
history, plus the cache provenance the run read, so a number is traceable to
the exact cache and backbone that produced it) and `inference-test.json`
(final test-set metrics, plus per-window IoU). No `masks-test.pt` is
committed here (`--save-masks` wasn't used for this rerun), so any figure
below is built from `metrics.json`/`inference-test.json` alone.

`comparison/` holds the cross-run figures backing the findings below, built
by `src/segmentation/analysis/` (see
[src/segmentation/README.md](../../src/segmentation/README.md) for the full
script list):

| Figure | Finding | Script |
|---|---|---|
| `macro-iou-ci-linear.png` | 1 — random-vs-trained gap | `build_headline_gap.py` |
| `capacity-macro-iou-bars.png` | 2 — linear-vs-MLP gap, per backbone | `build_capacity_slope_and_pr.py` |
| `capacity-confusion-counts.png` | 2 — where the MLP gap comes from (TP/FP/FN, not just the precision/recall ratio) | `build_capacity_slope_and_pr.py` |
| `category-difficulty-heatmap.png` | 3 — category difficulty is dataset-, not embedding-, driven | `build_category_difficulty_heatmap.py` |
| `learning-curve-panels.png` | 5 — best_val_epoch per run | `build_learning_curve_panels.py` |

Finding 4 (AUROC-vs-AUPRC gap explained via actual ROC/PR curves) has no
figure yet: it needs per-pixel/per-token scores, and `masks-<split>.pt` only
ever stores binarized predicted/target labels, not the underlying logits —
reconstructing it needs a change to `inference_segmentation.py` (save raw
scores, not just the thresholded mask) plus a re-run, not just a new script
over data already on hand.

## Headline results (test split, 1,077 windows)

| Backbone | Probe | mIoU | AUROC | AUPRC | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| CUT3R-random | linear | 0.185 | 0.652 | 0.340 | 0.38 | 0.27 |
| CUT3R-random | MLP | 0.287 | 0.774 | 0.548 | 0.63 | 0.34 |
| CUT3R-trained | linear | 0.729 | 0.962 | 0.887 | 0.81 | 0.88 |
| CUT3R-trained | MLP | 0.785 | 0.979 | 0.930 | 0.84 | 0.91 |
| DINOv2 | linear | 0.735 | 0.969 | 0.918 | 0.84 | 0.87 |
| DINOv2 | MLP | 0.782 | 0.976 | 0.937 | 0.87 | 0.88 |

mIoU = mean-category IoU (foreground, macro over the 26 categories).

## Findings

**1. CUT3R's pretraining, not its architecture, is what makes objects
identifiable in its embeddings.** Untrained CUT3R: mIoU 0.185, AUROC 0.652.
Trained CUT3R: mIoU 0.729, AUROC 0.962. IoU (threshold-dependent) and AUROC
(threshold-free) agree, so the gap isn't an artifact of where the 0.5 cutoff
falls.

**2. Trained CUT3R encodes object identity about as well as DINOv2, despite
never training on anything semantic.** Linear-probe mIoU: 0.729 (CUT3R) vs.
0.735 (DINOv2); their validation-selection scores are within 0.001 of each
other. CUT3R was trained purely for 3D pose/geometry regression.

**3. CUT3R-random's embeddings hold some usable signal, but it's tangled —
not the linearly-readable object location both trained backbones already
have.** Linear→MLP, CUT3R-random: precision 0.38→0.63, recall barely moves
(0.27→0.34) — the MLP mostly removes false positives, it doesn't find new
objects. The trained backbones already have recall 0.87–0.88 at linear
capacity; MLP only tightens precision.

**4. CUT3R-random can weakly rank pixels but can't commit to a confident
prediction; both trained backbones can.** AUROC 0.65–0.77 next to AUPRC only
0.34–0.55 for CUT3R-random — ranking foreground above background looks fine
when background is abundant, but confident, precise foreground predictions
are what's actually hard on a rare-positive task. The two metrics converge
for the trained backbones (DINOv2-MLP: 0.976 vs 0.937).

**5. Which categories are hard is a property of the dataset, not of any
specific embedding — as long as the embedding carries real signal.**
Per-category IoU correlates 0.80–0.94 among the four trained-backbone runs,
but only 0.30–0.47 between CUT3R-random and any trained run. Bench,
parking-meter, toyplane, baseball-bat, and wineglass are hard everywhere with
real signal; apple, bowl, frisbee, and toytruck are easy everywhere.

**6. CUT3R-random is the only backbone that runs out of real signal to learn
from — training past that point trades real segmentation quality for the
easy background-majority fit.** Best-validation epoch (of 20): CUT3R-random-
linear = 3 (then degrades for the rest of training), CUT3R-random-MLP = 16,
CUT3R-trained-linear = 18, CUT3R-trained-MLP = 17, DINOv2-linear = 20,
DINOv2-MLP = 10 (the one run that visibly overfits afterward, from its
unregularized 512-unit head).

## Reproducing

```bash
python -m src.segmentation.train_segmentation \
  --config src/segmentation/configs/<backbone>_<probe>.yaml --checkpoint-selection best_val
python -m src.segmentation.inference_segmentation \
  --config src/segmentation/configs/<backbone>_<probe>.yaml --split test \
  --checkpoint src/segmentation/experiments/segmentation-<backbone>-<probe>/head.pt
```

`<backbone>` is one of `cut3r_random`, `cut3r_trained`, `dinov2`; `<probe>` is
`linear` or `mlp`. See
[src/segmentation/README.md](../../src/segmentation/README.md) for the full
pipeline and the analysis scripts that turn a run's `metrics.json` /
`inference-test.json` into figures.
