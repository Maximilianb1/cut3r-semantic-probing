# Segmentation results

Metrics and figures for the binary foreground/background probe. Three frozen
backbones feed the same trainable head, which labels each patch token of the
target frame as foreground or background.

| Backbone | What it is |
|---|---|
| CUT3R-trained | The pretrained CUT3R 3D-reconstruction model, frozen — the subject of the project. |
| CUT3R-random | The same architecture with untrained weights — the control for what pretraining contributes. |
| DINOv2 ViT-B/14 | An off-the-shelf pretrained vision model — the upper anchor. |

Primary metric is foreground macro-IoU on the held-out test split, at token
resolution, reported at each run's best-validation checkpoint. 101 test
windows, 26 categories, seed `20260729`.

## Results

Test macro-IoU:

| Run | Head | Train windows | CUT3R-trained | CUT3R-random | DINOv2 |
|---|---|---:|---:|---:|---:|
| `baseline-30cap/` | MLP `[512]` | 3,054 | 0.7402 | 0.2298 | 0.7922 |
| `expanded-100cap/` | MLP `[512]` | ~10,000 | 0.7772 | 0.2772 | 0.8063 |
| `probe-capacity-ablation/` | linear `[]` | ~10,000 | 0.7401 | 0.1601 | 0.7382 |

On the expanded runs, macro and micro IoU agree for DINOv2 (0.806 / 0.805) and
CUT3R-trained (0.777 / 0.749) but diverge for CUT3R-random (0.277 / 0.223).
Precision and recall are 0.786 / 0.941 for CUT3R-trained, 0.891 / 0.894 for
DINOv2, and 0.40 / 0.34 for CUT3R-random.

## The three runs

- **`baseline-30cap/`** — the first three-way comparison, trained on up to 30
  sequences per category.
- **`expanded-100cap/`** — the same heads and recipe on roughly 3.3x more
  training data (up to 100 sequences per category), with validation and test
  held fixed so the scores stay comparable. `comparison/` holds the
  cross-backbone analysis.
- **`probe-capacity-ablation/`** — the same data again with a plain linear head
  instead of the `[512]` MLP. `comparison/` shows how much each backbone loses
  when the head loses its nonlinear capacity.

## Files

Per run, one directory per backbone:

| File | Contents |
|---|---|
| `metrics.json` | The full numeric result, plus the cache provenance the run read — so a number is traceable to the exact cache and backbone that produced it. |
| `training-curves.png` | Train and validation score per epoch. |
| `per-category-iou-test.png` | Test IoU broken down by object category. |
| `qualitative-best5-test.png`, `qualitative-worst5-test.png` | Input, ground truth, prediction, and overlay for the five best and five worst test windows by IoU. Not produced for the linear ablation. |

In `comparison/`:

| File | Contents |
|---|---|
| `macro-iou-ci.png` | Macro-IoU per backbone with its bootstrap 95% CI. |
| `macro-micro-iou.png` | Macro and micro IoU side by side, as point values. |
| `precision-recall.png` | Foreground precision and recall per backbone. |
| `per-window-comparison.csv` | One row per test window: every backbone's IoU and every pairwise delta. The data behind the paired comparison. |
| `delta-favors-*.png` | The windows where two backbones' IoU differs most, as photos. |
| `category-representation-check.png` | Per-category training-window count against per-category test IoU, to check whether coverage rather than visual difficulty drives the per-category spread. |
| `probe-capacity-comparison.png` | MLP against linear macro-IoU, two bars per backbone. |
| `probe-capacity-per-category.png` | The same comparison per category, one panel per backbone. |

Checkpoints, raw predicted-mask tensors, and per-window inference JSON are
working artifacts and are not committed; the commands that regenerate them are
in [docs/REPRODUCING.md](../../docs/REPRODUCING.md).

## Regenerating the figures

Every figure here is produced from an already-computed run by
`src/segmentation/analysis/` — no re-training and no re-inference:

```bash
python -m src.segmentation.analysis.build_curves_and_iou \
  --experiments-root src/segmentation/experiments \
  --backbones cut3r_trained cut3r_random dinov2 --run-suffix -expanded-bestval

python -m src.segmentation.analysis.build_score_comparison \
  --experiments-root src/segmentation/experiments \
  --backbones cut3r_trained cut3r_random dinov2 --run-suffix -expanded-bestval
```

See [src/segmentation/README.md](../../src/segmentation/README.md) for the rest
of the analysis scripts.
