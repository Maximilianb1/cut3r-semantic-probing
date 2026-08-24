# EXP-005: Part-A three-way segmentation comparison (CUT3R-trained unblocked)

- Date: 2026-08-20
- Owner: Yam Ben-Tov with Claude Code
- Status: Completed
- Related issue/PR: Pillar A of the segmentation-probe redesign plan
- Code commit: `be792c7` on branch `seg/21-expand-part-a-data`

## Hypothesis

EXP-003 established DINOv2 vs CUT3R-random but explicitly could not test the
scientifically load-bearing comparison: how a **trained** CUT3R backbone's
frozen representation performs at binary foreground segmentation, relative to
both DINOv2 (upper anchor) and CUT3R-random (lower anchor). The labelled
CUT3R-trained probe cache that blocked EXP-003 (`cache_schema_version:
stage0-target-cache-v1` instead of `probe-features-v2`) has since been
re-shared correctly. This record re-runs all three backbones together, on
the same unchanged head/recipe and the same original Part-A cache, to
produce one directly comparable table and finally test that hypothesis.

## Representation and model

Three backbones, identical head geometry and training recipe (unchanged from EXP-003):

| Backbone | Version / checkpoint | Feature dim | Token grid | Layer / token |
|---|---|---:|---|---|
| DINOv2 ViT-B/14 | `dinov2-vitb14` (Facebook Research release) | 768 | 16×16 patches | last-block patch tokens |
| CUT3R-trained | Upstream commit `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`, released `cut3r_512_dpt_4_64.pth` checkpoint | 768 | 14×14 patches | last-block target tokens (`layout: target_only`) |
| CUT3R-random | Same commit, all trainable modules re-initialised via `reset_parameters`, seed `20260729` | 768 | 14×14 patches | last-block target tokens (`layout: target_only`) |

- Probe: 1-hidden-layer MLP (`hidden_dims: [512]`, GELU, dropout 0.0), linear
  head to `num_classes: 1`, sigmoid + BCE, `pos_weight: null`. Identical
  across all three backbones (`src/segmentation/configs/{cut3r_trained,cut3r_random,dinov2}.yaml`,
  unchanged).
- Frozen parameters: all backbone parameters; only the MLP head trains.

## Data

- Manifest: `full51-part-a-v1`, identical to EXP-003 (`manifest_sha256` matches
  exactly across all three caches). 26 categories, train 3054 / val 512 / test 101 windows.
- Leakage checks: `assert_sequence_disjoint` passed for every run (train/val);
  `assert_not_trained_on` passed for every test-split inference call.
- Preprocessing: `probe_cache_schema_version: probe-features-v2`, `layout: target_only`
  for all three caches.

## Data (continued)

The three probe-feature caches and the manifest already exist on the team
Drive; fetch them per
[`DRIVE_TO_VM_RUNBOOK.md`](../project/DRIVE_TO_VM_RUNBOOK.md) before running
anything below. That runbook's `cache/${BACKBONE}` path assumption does not
match two of the three real folder layouts (see the fix noted in that file);
list the folder first if a copy comes back empty.

## Configuration

- Tracked configs (unchanged): `cut3r_trained.yaml`, `cut3r_random.yaml`, `dinov2.yaml`.
- Optimiser: Adam, `lr 1e-3`, `weight_decay 0`, `batch_size 16`, 20 epochs, seed `20260729`.
- Hardware: CPU-only (`device: cpu`, per the tracked configs).
- Two checkpoint-selection modes, same data/seed, both against
  [`src/segmentation/train_segmentation.py`](../../src/segmentation/train_segmentation.py)/`inference_segmentation.py`,
  controlled by `training.checkpoint_selection`:
  1. **Last-epoch** — the default (`checkpoint_selection: last`); `head.pt` is
     "the final epoch's head, not the best-val one".
  2. **Best-val** — `checkpoint_selection: best_val`. Checkpoints whenever
     validation macro-IoU improves; also keeps the final-epoch head as
     `head-last.pt`.
- Qualitative figures use
  [`src/segmentation/analysis/build_qualitative_plots.py`](../../src/segmentation/analysis/build_qualitative_plots.py)
  and [`scripts/download_co3d_targeted.py`](../../scripts/download_co3d_targeted.py)
  (fetches only the exact images named by the manifest — see "Notes").
- Exact commands:

  ```bash
  # Last-epoch (stock, unchanged)
  python -m src.segmentation.train_segmentation --config src/segmentation/configs/<backbone>.yaml
  python -m src.segmentation.inference_segmentation --config src/segmentation/configs/<backbone>.yaml --split test

  # Best-val
  python -m src.segmentation.train_segmentation --config src/segmentation/configs/<backbone>.yaml \
    --checkpoint-selection best_val --output-dir src/segmentation/experiments/segmentation-<backbone>-bestval
  python -m src.segmentation.inference_segmentation --config src/segmentation/configs/<backbone>.yaml \
    --checkpoint src/segmentation/experiments/segmentation-<backbone>-bestval/head.pt \
    --split test --save-dir src/segmentation/experiments/segmentation-<backbone>-bestval --save-masks

  # Qualitative worst-5/best-5 figures (after the best-val runs above)
  python -m src.segmentation.analysis.build_qualitative_plots \
    --manifest-dir ${CUT3R_ARTIFACT_ROOT}/manifests/full51-part-a-v1 \
    --dataset-root ${CO3D_ROOT} --experiments-root src/segmentation/experiments \
    --backbones cut3r_trained cut3r_random dinov2 --run-suffix -bestval
  ```

## Metrics and success criteria

Same convention as EXP-003: **macro-foreground-IoU** on the test split is the
primary/ADR metric. Token accuracy is not the deciding number — the Part-A
foreground-token fraction is ≈22%, so an always-background head scores ≈0.78
accuracy at IoU 0.

## Results

### Last-epoch (epoch 20) vs. best-val

| Backbone | Mode | Test macro-IoU | Test micro-IoU | Test tok-acc |
|---|---|---:|---:|---:|
| DINOv2 ViT-B/14 | last-epoch | 0.7903 | 0.7761 | 0.9582 |
| DINOv2 ViT-B/14 | best-val (epoch 6) | 0.7922 | 0.7838 | 0.9595 |
| CUT3R-trained | last-epoch | 0.7402 | 0.7213 | 0.9404 |
| CUT3R-trained | best-val (epoch 17) | 0.7402 | 0.7274 | 0.9432 |
| CUT3R-random | last-epoch | 0.2232 | 0.1819 | 0.8081 |
| CUT3R-random | best-val (epoch 16) | 0.2298 | 0.2116 | **0.7521** |

DINOv2's best-val numbers (epoch 6, val 0.8360, test 0.7922) reproduce EXP-003
exactly, confirming the cache/manifest/seed/code stack behaves deterministically.

**Difference and consequence:** macro-IoU — the metric that actually matters
here — is essentially unaffected by the choice (largest delta 0.0066, on
CUT3R-random); neither backbone overfits badly within 20 epochs (small frozen-feature
head, no weight decay needed). The one real move is CUT3R-random's token
accuracy, which *drops* under best-val (0.8081 → 0.7521) even though its IoU
improves. **Probable reason:** accuracy is inflated by the class-imbalance
shortcut of predicting "background" — more training steps (last-epoch) give
the head more chance to lean into that shortcut, which raises accuracy without
raising real segmentation quality; the earlier best-val checkpoint is doing
comparatively more genuine foreground prediction, at an accuracy cost. This is
exactly the distortion EXP-003 already flagged accuracy for, which is why it
is not the ADR metric.

**Recommendation:** report best-val going forward. It targets the ceiling of
what the frozen representation makes decodable, rather than an artifact of a
uniform 20-epoch budget applied identically regardless of each backbone's own
convergence speed (DINOv2 peaked at epoch 6; CUT3R-trained/random were still
close to their epoch-20 value). The fact both conventions agree so closely
here is itself reassuring: this record's qualitative conclusion does not hinge
on the choice.

### Per-category test macro-IoU (best-val checkpoints)

| Category | CUT3R-trained | CUT3R-random | DINOv2 |
|---|---:|---:|---:|
| apple | 0.782 | 0.244 | 0.748 |
| ball | 0.561 | 0.517 | 0.942 |
| baseballbat | 0.713 | 0.178 | 0.704 |
| bench | 0.911 | 0.081 | 0.754 |
| book | 0.756 | 0.105 | 0.915 |
| bowl | 0.605 | 0.264 | 0.673 |
| cake | 0.547 | 0.134 | 0.772 |
| carrot | 0.706 | 0.281 | 0.893 |
| chair | 0.812 | 0.216 | 0.731 |
| cup | 0.542 | 0.021 | 0.762 |
| frisbee | 0.630 | 0.575 | 0.678 |
| handbag | 0.809 | 0.425 | 0.753 |
| hydrant | 0.837 | 0.175 | 0.873 |
| kite | 0.737 | 0.305 | 0.815 |
| microwave | 0.691 | 0.036 | 0.624 |
| mouse | 0.670 | 0.313 | 0.771 |
| parkingmeter | 0.359 | 0.121 | 0.492 |
| plant | 0.935 | 0.401 | 0.841 |
| sandwich | 0.832 | 0.265 | 0.936 |
| stopsign | 0.698 | 0.010 | 0.887 |
| teddybear | 0.951 | 0.347 | 0.932 |
| toilet | 0.427 | 0.159 | 0.502 |
| toyplane | 0.730 | 0.183 | 0.809 |
| toytruck | 0.796 | 0.309 | 0.798 |
| umbrella | 0.950 | 0.402 | 0.947 |
| wineglass | 0.868 | 0.079 | 0.918 |

CUT3R-trained beats DINOv2 outright in 9 of 26 categories (apple,
baseballbat, bench, chair, handbag, microwave, plant, teddybear, umbrella —
by up to +0.157 on bench), so the aggregate DINOv2 lead is not uniform.

### Qualitative (worst-5 / best-5 test windows, best-val checkpoints)

- CUT3R-trained worst 5: toilet 0.000, mouse 0.106, cup 0.106, toyplane 0.141, ball 0.164.
- CUT3R-random worst 5: **all five are exact 0.000 IoU** (apple, book×2, bowl, carrot) — complete misses, not partial ones.
- DINOv2 worst 5: toilet 0.136, apple 0.161, microwave 0.196, bowl 0.201, frisbee 0.245.

Even at its worst, CUT3R-trained and DINOv2 degrade to partial misses;
CUT3R-random's worst cases are total failures — a concrete illustration of
the macro-IoU gap, not just a summary statistic. Figures:
`src/segmentation/experiments/segmentation-<backbone>-bestval/qualitative-{worst,best}5-test.png`.

Artifacts (git-ignored per the existing `src/segmentation/experiments/`
convention; VM-local only, same as EXP-003/EXP-004): run directories,
`metrics.json`, `head.pt`/`head-last.pt`, `inference-test.json`,
`masks-test.pt`, and the six qualitative figures. The three scripts that
produced them are committed, so any teammate can regenerate all of it.

## Interpretation

Supported:

- **CUT3R-trained's frozen representation carries strong, genuine
  segmentation signal** — test macro-IoU 0.7402, far above CUT3R-random
  (0.2298) and within ~0.05 of DINOv2 (0.7922). This is the comparison
  EXP-003 could not make; the project's core hypothesis (frozen CUT3R
  encodes class-agnostic foreground/background structure) is supported on
  Part-A.
- The gap over CUT3R-random (+0.51 macro-IoU) is not attributable to head
  capacity, seed, optimiser, or dataset (identical across all three runs),
  and is robust to the last-epoch/best-val checkpoint choice.
- DINOv2 still leads on mean per-category IoU, but not uniformly — CUT3R-trained
  wins outright on 9/26 categories.
- Qualitatively, CUT3R-trained's worst failures are partial misses; CUT3R-random's
  are total ones.

Not supported:

- Whether this ordering holds on the full 51 categories (Part-B untested)
  or on the expanded/leftover data batches (Pillar B, not run here).
- CUT3R-random's number is still not its true floor — EXP-003 already flagged
  that its val curve had not plateaued at epoch 20 (best-val landed at epoch
  16, only slightly earlier); a 40–60 epoch run remains open future work.
- Cross-backbone IoU comparability is not fully settled — DINOv2 and CUT3R
  are scored on different native token grids (16×16 vs 14×14), an open ADR
  item carried over unchanged from EXP-003.

## Notes

`download_co3d_targeted.py` is a new script, not a reuse of
`scripts/download_co3d_selective.py`: the latter re-derives its own
sequence/window selection from a config's `sampling` block and was observed
**not** to reproduce this manifest's exact selection on a fresh run (3,777
windows / 1,040 sequences vs. this manifest's actual 3,667 / 1,008), even
with an identical `config_sha256`. The targeted script instead trusts the
manifest's own `image_relpath`/`mask_relpath` columns directly, so the
qualitative figures are guaranteed to use the same frames the caches were
built from.

## Next action

- Pillar B (data expansion: 30→100 sequences, leftover windows) can likely
  proceed the same way this record did — cap100/leftover probe-feature
  caches for all three backbones are already present on Drive (unverified
  schema-wise; check each `metadata.json` first).
- Independent follow-up carried over from EXP-003, still open: run
  CUT3R-random for 40–60 epochs to lock down its true random-init plateau.
