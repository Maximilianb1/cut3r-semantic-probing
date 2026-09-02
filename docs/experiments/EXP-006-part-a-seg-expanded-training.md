# EXP-006: Part-A expanded-training comparison

- Date: 2026-08-22
- Owner: Yam Ben-Tov
- Status: Completed
- Related issue/PR: none
- Code commit: `6a0f33b` on branch `seg/21-expand-part-a-data`

## Hypothesis

EXP-005 established one trustworthy baseline per backbone on the original
3,667-window Part-A cache. Two more data batches since became available for
the same manifest family: a "leftover" batch (extra windows from previously-
unused frames of the *same* training videos) and a "cap100-new-train" batch
(the training-video cap raised 30->100/category, 1,762 brand-new videos).
Question: does training the *exact same, unchanged* head on the union of all
three train partitions (~3.3x more windows) move the score, holding val/test
fixed for a direct comparison to EXP-005?

## Representation and model

Identical to EXP-005 — same three backbones, same head geometry
(`hidden_dims: [512]`, GELU, dropout 0.0, `pos_weight: null`), same optimizer
(Adam, lr 1e-3, weight_decay 0, batch_size 16, 20 epochs, seed `20260729`).
Only the training data changed.

## Data

- Train: union of three caches' `train`-split rows per backbone — original
  (3,054) + leftover (74) + cap100-new-train (6,901) = **10,029 windows**.
- Val/test: the **original** cache's rows only, unchanged — 512 val / 101
  test — so the score stays comparable to EXP-005.
- All six new caches (leftover + cap100-new-train, x3 backbones) confirmed
  `probe-features-v2` / `target_only`, integrity-verified after transfer.
- Leakage checks passed: `assert_sequence_disjoint` on the real combined
  train set vs val (train 2,541 sequences, val 130, zero overlap);
  `assert_not_trained_on` passed at inference time.

## Configuration

- New configs: `src/segmentation/configs/{cut3r_trained,cut3r_random,dinov2}_expanded_mlp.yaml`
  — only `probe_cache.train_dirs` and `output.dir` differ from their originals.
- New code enabling this: `CombinedProbeCacheDataset` +
  `train_segmentation.build_datasets` (opt-in via `probe_cache.train_dirs`;
  original configs/behavior unchanged when absent). Covered by
  `tests/test_segmentation_dataset.py`.
- Checkpoint selection: **best-val** (`train_segmentation.py --checkpoint-selection best_val`),
  matching EXP-005's recommendation, so this stays a clean "more data" comparison.
- `training.num_workers: 8` (was `0`): with 3.3x more windows, single-threaded
  per-window `safetensors` reads were the real bottleneck, not compute
  (~2.6x faster, bit-identical results — shuffle order is seed-fixed, no
  per-item randomness). GPU was ruled out: this VM's `torch.cuda.is_available()`
  is `False` (driver/build mismatch), and wouldn't have helped an I/O-bound
  workload regardless.
- Commands (per backbone):

  ```bash
  python -m src.segmentation.train_segmentation --config src/segmentation/configs/<backbone>_expanded_mlp.yaml \
    --checkpoint-selection best_val --output-dir src/segmentation/experiments/segmentation-<backbone>-expanded-bestval

  python -m src.segmentation.inference_segmentation --config src/segmentation/configs/<backbone>_expanded_mlp.yaml \
    --checkpoint src/segmentation/experiments/segmentation-<backbone>-expanded-bestval/head.pt \
    --split test --save-dir src/segmentation/experiments/segmentation-<backbone>-expanded-bestval --save-masks
  # repeat inference with --split val
  ```

## Metrics and success criteria

Same convention as EXP-003/EXP-005: **macro-foreground-IoU** on the test
split is the primary metric, compared against EXP-005 (DINOv2 0.7922,
CUT3R-trained 0.7402, CUT3R-random 0.2298).

## Results

| Backbone | Original (3,054 train) | Expanded (10,029 train) | Delta |
|---|---:|---:|---:|
| DINOv2 ViT-B/14 | 0.7922 | 0.8063 | +0.0141 |
| CUT3R-trained | 0.7402 | 0.7772 | +0.0370 |
| CUT3R-random | 0.2298 | 0.2772 | +0.0474 |

All three improved. CUT3R-trained's gap to DINOv2 narrowed from 0.052 to
0.029 in point-estimate terms — but see the statistical check below.

| Backbone | best_val_epoch | val macro-IoU | test macro-IoU | test micro-IoU | test tok-acc |
|---|---:|---:|---:|---:|---:|
| CUT3R-trained | 17 | 0.8540 | 0.7772 | 0.7492 | 0.9477 |
| CUT3R-random | 16 | 0.3068 | 0.2772 | 0.2232 | 0.8050 |
| DINOv2 | 3 | 0.8479 | 0.8063 | 0.8055 | 0.9637 |

### Per-category test macro-IoU

| Category | CUT3R-trained | CUT3R-random | DINOv2 |
|---|---:|---:|---:|
| apple | 0.846 | 0.397 | 0.753 |
| ball | 0.767 | 0.647 | 0.868 |
| baseballbat | 0.772 | 0.263 | 0.652 |
| bench | 0.903 | 0.260 | 0.812 |
| book | 0.719 | 0.208 | 0.918 |
| bowl | 0.708 | 0.018 | 0.680 |
| cake | 0.656 | 0.366 | 0.797 |
| carrot | 0.689 | 0.542 | 0.900 |
| chair | 0.817 | 0.283 | 0.755 |
| cup | 0.645 | 0.240 | 0.736 |
| frisbee | 0.637 | 0.384 | 0.786 |
| handbag | 0.871 | 0.445 | 0.768 |
| hydrant | 0.885 | 0.162 | 0.895 |
| kite | 0.836 | 0.451 | 0.847 |
| microwave | 0.699 | 0.006 | 0.627 |
| mouse | 0.752 | 0.307 | 0.876 |
| parkingmeter | 0.292 | 0.138 | 0.513 |
| plant | 0.952 | 0.557 | 0.906 |
| sandwich | 0.852 | 0.298 | 0.926 |
| stopsign | 0.755 | 0.175 | 0.878 |
| teddybear | 0.957 | 0.268 | 0.929 |
| toilet | 0.481 | 0.042 | 0.604 |
| toyplane | 0.745 | 0.132 | 0.807 |
| toytruck | 0.811 | 0.444 | 0.793 |
| umbrella | 0.975 | 0.322 | 0.949 |
| wineglass | 0.912 | 0.030 | 0.856 |

CUT3R-trained beats DINOv2 outright in 12 of 26 categories; no category-label
pattern explains the split (see the per-window analysis below). CUT3R-random's
qualitative worst-5 are still all exact 0.000 IoU — identical failure mode to
EXP-005 despite 3.3x more data, pointing at the representation, not data
volume, as its limiting factor. Regenerate this table, the qualitative
worst/best-5 grids, and the per-category bar charts via `analysis/build_curves_and_iou.py`
/ `analysis/build_qualitative_plots.py` if the underlying runs ever change.

### Statistical significance (paired bootstrap CI)

Run `python -m src.segmentation.analysis.build_score_comparison --experiments-root
src/segmentation/experiments --backbones cut3r_trained cut3r_random dinov2
--run-suffix=-expanded-bestval` for precision/recall, a bootstrap 95% CI on
macro-IoU, and a paired per-window comparison (all three backbones share the
identical 101 test windows, so their scores aren't independent samples).

The CUT3R-trained-vs-DINOv2 paired delta's 95% CI includes zero — not
significant, despite the 0.029 point-estimate gap and DINOv2 winning more
individual windows (61 vs 40). Both backbones beating CUT3R-random is
unambiguous (CIs exclude zero by a wide margin).

### Precision / recall

CUT3R-trained over-guesses foreground (recall 0.941 vs precision 0.786);
DINOv2 is balanced (0.891/0.894); CUT3R-random is weak on both (0.40/0.34) —
it isn't just imprecise, it isn't finding the object either.

### Macro vs. micro IoU

Macro and micro IoU are close for DINOv2 (0.806/0.805) and CUT3R-trained
(0.777/0.749) — no size bias — but diverge more for CUT3R-random
(0.277/0.223), meaning it does even worse on large-foreground windows than
its macro score alone suggests.

### Training curves

DINOv2's val IoU plateaus by epoch 3 (hence best-val epoch 3) while train
loss keeps dropping for the full 20 — mild overfitting past that point that
best-val selection already avoids. CUT3R-trained's val curve is fairly flat
from ~epoch 10 on, so its best-val checkpoint (epoch 17) is a late, marginal
pick off a near-plateau. CUT3R-random's train and val IoU are both still
rising at epoch 20 — unconverged, so its real ceiling is unknown until it's
trained longer (open question, see Interpretation below).

### Per-category ranking (sorted, not the alphabetical table above)

CUT3R-trained's and DINOv2's rankings have the same shape — umbrella/
teddybear/plant on top, parkingmeter/toilet on bottom. CUT3R-random's
ranking looks almost inverted: its best category (ball, 0.65) is only
middling for the other two, while categories both trained backbones score
highest on (microwave, bowl, wineglass) are its near-total failures
(≤0.03) — consistent with it not learning category semantics at all, but
picking up something else in the token statistics.

### Failure modes: CUT3R-trained vs. DINOv2

Side-by-side predictions on the windows where the two backbones disagree
most (`python -m src.segmentation.analysis.build_delta_comparison_plot
--backbone-a cut3r_trained --backbone-b dinov2 --run-suffix=-expanded-bestval`)
show every window CUT3R-trained loses badly, it loses the same way: a
diffuse blob spilling into background clutter. DINOv2's worst losses are
not one thing — outright mislocation on apple, under-coverage on
microwave/bench, its own scattered noise on both baseballbat windows.
"DINOv2 is the clean backbone" holds only on average. Consistent with this:
CUT3R-trained's and DINOv2's best-5 test windows overlap heavily (4 of 5
are the same windows) — the windows that are easy are easy for any
competent backbone, and the two backbones only diverge on hard cases.

### CUT3R-random qualitative results

Its best-5 test windows are still only coarse, blocky localizations, never
boundary-accurate; its worst-5 are near-total misses — predictions reduce
to a few scattered pixels or nothing, uncorrelated with the true object.

### Category dataset size vs. difficulty

`python -m src.segmentation.analysis.build_category_representation_check
--config src/segmentation/configs/cut3r_trained_expanded_mlp.yaml` — 25 of 26
categories sit in a narrow ~370-400-window band (cap100 nearly equalized
them; parkingmeter alone sits apart at ~181), yet test IoU still spans
nearly the full 0-1 range within that band (Spearman r only +0.10 to +0.33,
vs. inflated Pearson r of +0.53 to +0.63 driven by the single low-count
outlier). Per-category difficulty is about the category's visual
properties, not how much training data it got — so `pos_weight` (loss-side)
is a more promising next lever than more data collection for still-weak
categories.

## Interpretation

Supported:

- More training data helps all three backbones on this head/recipe. Modest
  for DINOv2 (+0.014, near its ceiling here) and CUT3R-trained (+0.037),
  largest in relative terms for CUT3R-random (+0.047, off a low base).
- Both CUT3R-trained and DINOv2 clearly beat CUT3R-random (paired CIs
  exclude zero by a wide margin) — trained representations carry real
  segmentation signal, independent of exactly how they compare to DINOv2.
- CUT3R-random's failure is not primarily a data problem: worst-5 failures
  are unchanged and several categories remain near-zero despite 3.3x more
  data — points at the (untrained) representation itself.

Not supported / still open:

- Whether CUT3R-trained actually trails DINOv2 — the paired per-window CI on
  that gap includes zero at this test-set size (101 windows).
- Whether CUT3R-random would plateau higher with a longer schedule (val
  curve still not stable at epoch 20 on either dataset).
- Cross-backbone IoU comparability (different native token grids) — open
  since EXP-003/EXP-005.
- The CUT3R-trained-vs-CUT3R-random comparison itself is confounded:
  CUT3R-trained's cache uses `layout: trajectory` (6-frame memory context)
  while CUT3R-random's uses `layout: target_only` (single frame, no
  context) — see `configs/probe_features/cut3r_random.yaml`. Re-extracting
  CUT3R-random with `layout: trajectory` would isolate the weights variable
  cleanly.

## Problems and deviations

- `analysis/build_qualitative_plots.py`/`analysis/build_curves_and_iou.py`/`analysis/build_score_comparison.py`/
  `analysis/build_delta_comparison_plot.py` all need `--run-suffix=-expanded-bestval`
  (equals-sign form) — space-separated fails argparse when the value starts
  with `-`.
- `num_workers` was raised `0`->`8` after an initial slow attempt; the whole
  3-backbone pipeline was restarted so all three runs here share identical
  infra settings.

## Next action

- Head/loss improvements (`pos_weight` first, per the category-representation
  check above) can now proceed with two real baselines to compare against.
- Re-extract CUT3R-random with `layout: trajectory` to remove the
  memory-context confound noted above.
- A 40-60 epoch CUT3R-random run, on the expanded data, to check whether it
  plateaus higher or keeps behaving like an untrained representation
  regardless of epoch/data budget.
