# EXP-007: Part-A segmentation probe-capacity ablation (linear vs. MLP head)

- Date: 2026-08-25
- Owner: Yam Ben-Tov with Claude Code
- Status: Completed
- Related issue/PR: none
- Code commit: `1adf73c` on branch `seg/21-expand-part-a-data`

## Hypothesis

EXP-006 established macro-IoU with the `[512]` MLP head on the expanded
training set for all three backbones. Every segmentation config has carried
an OPEN-decision comment since 2026-07-30
(`docs/sessions/2026-07-30-segmentation-probe-scope.md`) asking whether Stage
1 should report the linear head (`hidden_dims: []`) instead of `[512]` — and,
more importantly for the project's actual question, comparing backbones only
through an MLP head cannot separate "this backbone's frozen features
implicitly encode segmentation" from "the MLP head is constructing the
signal out of a less-informative representation." Question: holding backbone
and training data fixed, how much test macro-IoU does each backbone lose
when the head is stripped down to a true linear probe? A small relative loss
signals the segmentation signal is (near-)linearly decodable — genuinely
implicit in the representation; a large one signals the MLP's nonlinear
capacity is doing real work.

## Representation and model

- Same three backbones as EXP-005/EXP-006: CUT3R-trained, CUT3R-random,
  DINOv2 (`dinov2-vitb14`).
- Same expanded training data as EXP-006 (original + leftover +
  cap100-new-train, 10,029 train windows; val/test held fixed at the
  original cache's 512/101 windows).
- Head: two capacities compared per backbone, everything else (optimizer,
  lr, epochs, seed, `pos_weight`) unchanged from EXP-006 —
  `hidden_dims: [512]` (MLP, identical to EXP-006) and `hidden_dims: []`
  (true linear probe, `nn.Linear(768 -> 1)`).
- Frozen: the whole backbone. Trainable: the head only.

## Data

Identical to EXP-006: real combined train cache (10,029 windows / 26
categories), val/test on the original cache only (512 / 101 windows).
Leakage checks unchanged from EXP-006 (`assert_sequence_disjoint`,
`assert_not_trained_on`).

## Configuration

- New configs: `src/segmentation/configs/{cut3r_trained,cut3r_random,dinov2}_expanded_linear.yaml`
  — identical to the corresponding `*_expanded_mlp.yaml` (the config this
  project used for EXP-006, since renamed — see "Problems and deviations")
  except `model.hidden_dims: []`.
- Checkpoint selection: best-val, matching EXP-005/EXP-006.
- Seed: `20260729`, same as EXP-005/EXP-006.
- Hardware: Technion course VM (`mcvgpu2025s-0086`), CPU-only
  (`torch.cuda.is_available()` is `False` on this VM, same as EXP-006),
  `num_workers: 8`.
- Exact commands (per backbone):

  ```bash
  python -m src.segmentation.train_segmentation --config src/segmentation/configs/<backbone>_expanded_linear.yaml \
    --checkpoint-selection best_val --output-dir src/segmentation/experiments/segmentation-<backbone>-expanded-linear-bestval

  python -m src.segmentation.inference_segmentation --config src/segmentation/configs/<backbone>_expanded_linear.yaml \
    --checkpoint src/segmentation/experiments/segmentation-<backbone>-expanded-linear-bestval/head.pt \
    --split test --save-dir src/segmentation/experiments/segmentation-<backbone>-expanded-linear-bestval --save-masks
  # repeat inference with --split val
  ```

- Comparison and figures:

  ```bash
  python -m src.segmentation.analysis.build_probe_capacity_comparison \
    --experiments-root src/segmentation/experiments \
    --backbones cut3r_trained cut3r_random dinov2 \
    --mlp-run-suffix=-expanded-bestval --linear-run-suffix=-expanded-linear-bestval
  ```

## Metrics and success criteria

Same convention as EXP-003/005/006: macro-foreground-IoU on the test split
is the primary metric. The primary comparison here is the **paired
per-window delta** (mlp − linear) per backbone, since both heads are
evaluated on the identical 101 test windows — a stronger test than comparing
two independent point estimates (same method `build_score_comparison.py`
uses for cross-backbone deltas).

## Results

| Backbone | MLP `[512]` test macro-IoU | Linear `[]` test macro-IoU | Absolute delta | Relative drop | Paired 95% CI on delta | wins/losses (mlp/linear) |
|---|---:|---:|---:|---:|---:|---:|
| DINOv2 | 0.8063 | 0.7382 | +0.0681 | 8.4% | [+0.0467, +0.0919] | 80/19 |
| CUT3R-trained | 0.7772 | 0.7401 | +0.0371 | 4.8% | [+0.0173, +0.0571] | 68/27 |
| CUT3R-random | 0.2772 | 0.1601 | +0.1171 | 42.2% | [+0.0635, +0.1714] | 60/29 |

All three deltas are statistically significant: the paired bootstrap 95% CI
(10,000 resamples, seed `20260729`, 101 shared test windows) excludes zero
in every row.

| Backbone | best_val_epoch (linear) | val macro-IoU (linear) | test macro-IoU (linear) | test micro-IoU (linear) | test tok-acc (linear) |
|---|---:|---:|---:|---:|---:|
| CUT3R-trained | 18 | 0.8001 | 0.7401 | 0.7371 | 0.9457 |
| CUT3R-random | 17 | 0.1782 | 0.1601 | 0.1581 | 0.8226 |
| DINOv2 | 4 | 0.7994 | 0.7382 | 0.7515 | 0.9523 |

Artifacts:

- `results/pillar-c/probe-capacity-comparison.png` — combined bar figure
  (both heads, all three backbones, bootstrap CI).
- `results/pillar-c/probe-capacity-per-category.png` — per-category
  MLP-vs-linear scatter, one panel per backbone, y=x reference line.
- `results/pillar-c/experiments/segmentation-<backbone>-expanded-linear-bestval/`
  — full run artifacts (`head.pt`, `metrics.json`, `inference-{test,val}.json`,
  masks, `training-curves.png`, `per-category-iou-test.png`) for all three
  backbones.
- `results/pillar-c/probe-capacity-notes.md` — detailed per-category
  discussion behind the "Interpretation" section below.

## Interpretation

Supported:

- All three backbones retain a statistically significant fraction of their
  MLP-head macro-IoU under a strictly linear readout (CUT3R-random: 0.160 of
  0.277; CUT3R-trained: 0.740 of 0.777; DINOv2: 0.738 of 0.806) — none
  collapses to floor performance, so all three encode *some* linearly
  accessible foreground/background signal.
- The **relative** size of the capacity gap differentiates the backbones
  more informatively than the raw MLP scores alone: CUT3R-trained loses the
  smallest fraction of its score (4.8% relative) when stripped to a linear
  head, DINOv2 loses more (8.4%), and CUT3R-random loses by far the most
  (42.2%, nearly halved).
- Per-category (`probe-capacity-per-category.png`), DINOv2 and CUT3R-trained
  both track the y=x diagonal tightly across nearly all 26 categories — the
  capacity gap is a small, roughly uniform tax, not concentrated in a
  handful of categories. CUT3R-random's per-category scatter is far
  messier, with several categories (ball, plant, cake, carrot, chair,
  sandwich) collapsing well below the diagonal — consistent with an
  untrained representation where the MLP recovers signal via nonlinear
  capacity rather than the features already being close to linearly
  separable.
- This resolves, for this project's purposes, the OPEN decision flagged in
  every segmentation config since 2026-07-30: **CUT3R-trained has the
  strongest claim to "implicit" segmentation encoding** among the three
  backbones — not the one with the highest raw IoU (that is still DINOv2),
  but the one whose score survives the linear-only readout most intact,
  both in aggregate and per-category.

Not supported / still open:

- Whether CUT3R-trained's advantage in relative-drop terms would hold on
  Part-B data or a different train/test split — this ablation reuses
  EXP-006's fixed val/test windows only.
- Whether the linear probe's absolute numbers would change under a
  different `pos_weight` / loss weighting (still an open engineering choice
  per EXP-006 and the segmentation README).
- CUT3R-random's per-category collapse pattern (ball, plant, cake, carrot,
  chair, sandwich) is not yet explained by any shared category property
  (size, texture, shape) — flagged for a follow-up qualitative look, not
  concluded here.

## Problems and deviations

- Mid-session, all nine segmentation configs were renamed to a consistent
  `<backbone>_<data>_<capacity>.yaml` scheme (`partial`/`expanded` x
  `mlp`/`linear`) to remove the ambiguity that appeared once linear configs
  existed alongside the MLP ones. EXP-005 and EXP-006 are written against
  the old names and are left as-written (accurate to what was actually run
  at the time). Mapping: `<backbone>.yaml` → `<backbone>_partial_mlp.yaml`;
  `<backbone>_expanded.yaml` → `<backbone>_expanded_mlp.yaml`. Only the
  config **files** moved — EXP-006's already-completed VM run directories
  were not renamed (still `segmentation-<backbone>-expanded-bestval`).
- `build_probe_capacity_comparison.py` reuses `bootstrap_ci_mean` from
  `build_score_comparison.py` rather than duplicating it.

## Next action

- Investigate CUT3R-random's specific per-category collapse pattern
  (ball/plant/cake/carrot/chair/sandwich) qualitatively — check whether
  these share a visual property (round/smooth objects? small foreground
  fraction?).
- Decide, given the evidence above, whether the project's headline Stage 1
  comparison should report the MLP head (higher absolute numbers, current
  convention), foreground the linear-head result as the "implicit encoding"
  evidence, or report both.
- Re-run this ablation on Part-B data once available, to check whether the
  relative-drop ranking (CUT3R-trained < DINOv2 < CUT3R-random) is stable.
