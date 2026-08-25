# Segmentation results (for slides)

Figures and summary numbers behind three segmentation experiments, filtered
down to what's useful for a slide deck. Full write-ups with methodology and
interpretation are in `docs/experiments/`:

- [EXP-005](../../docs/experiments/EXP-005-part-a-seg-cut3r-unblocked.md) —
  `exp005-baseline-30cap/`
- [EXP-006](../../docs/experiments/EXP-006-part-a-seg-expanded-training.md) —
  `exp006-expanded-100cap/`
- [EXP-007](../../docs/experiments/EXP-007-part-a-seg-probe-capacity-ablation.md) —
  `exp007-probe-capacity-ablation/`

Model checkpoints, raw prediction tensors, and raw per-window inference
JSON are intentionally left out (they're VM-local working artifacts, not
report material — see each experiment doc for how to regenerate them).

## What's being compared

Three frozen backbones feed a segmentation head that predicts a
foreground/background mask per image patch. The head itself is trained;
the backbone is frozen (except CUT3R-random, see below).

| Backbone | What it is |
|---|---|
| DINOv2 ViT-B/14 | Off-the-shelf pretrained vision model — the upper-anchor baseline. |
| CUT3R-trained | The pretrained CUT3R 3D-reconstruction model, frozen — this project's actual subject: does its representation carry usable semantic segmentation signal? |
| CUT3R-random | Same CUT3R architecture, but with randomly re-initialized (untrained) weights — a lower-anchor control, to check the trained model is actually contributing something beyond architecture alone. |

**Primary metric:** macro-foreground-IoU on the held-out test split (higher
is better). Reported at each backbone's best validation-score checkpoint.

## The three experiments

- **exp005-baseline-30cap** — first clean three-way comparison, trained on
  30 video sequences/category.
- **exp006-expanded-100cap** — same heads/recipe, retrained on ~3.3x more
  training data (up to 100 sequences/category); `comparison/` holds the
  cross-backbone statistical/qualitative analysis (confidence intervals,
  precision/recall, per-window head-to-head).
- **exp007-probe-capacity-ablation** — does the result hold with a plain
  linear head instead of the small MLP head used in exp006? `comparison/`
  shows how much each backbone's score drops when the head loses its
  nonlinear capacity.

## Per-run files

- `metrics.json` — full numeric results for that run.
- `qualitative-best5-test.png` / `qualitative-worst5-test.png` — example
  predictions on the 5 best/worst test images (not produced for exp007).
- `training-curves.png` — train/val score over training epochs.
- `per-category-iou-test.png` — score broken down by object category.
