# EXP-003: Part-A frozen-representation segmentation baselines

- Date: 2026-08-01
- Owner: Ron Bartal
- Status: Partially completed — DINOv2 and CUT3R-random runs completed; CUT3R-trained blocked pending re-shared probe cache
- Related issue/PR: follows PR #16; no PR opened this session
- Code commit: `1f45943` on branch `main`

## Hypothesis

For token-level foreground-object segmentation on the Part-A subset of CO3Dv2
(26 categories, 3667 windows), a frozen DINOv2 ViT-B/14 backbone with a
1-hidden-layer MLP probe yields test macro-foreground-IoU substantially above a
CUT3R backbone with randomly re-initialised weights and above token-level
random-guessing floors. The gap between the two quantifies how much of the
segmentation signal comes from the ImageNet-scale self-supervised pretraining
rather than from the head or the token grid.

The trained-CUT3R comparison — the scientifically load-bearing one — is not
covered by this record; see "Next action".

## Representation and model

Two backbones, identical head geometry and training recipe:

| Backbone | Version / checkpoint | Feature dim | Token grid | Layer / token |
|---|---|---:|---|---|
| DINOv2 ViT-B/14 | `dinov2-vitb14` (Facebook Research release) | 768 | 16×16 patches | last-block patch tokens |
| CUT3R-random | Upstream commit `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`, all trainable modules re-initialised via `torch.nn.Module.reset_parameters` under seed `20260729` (`random_init.strategy: reset_parameters` in the cache metadata) | 768 | 14×14 patches | last-block target tokens (`layout: target_only`) |

- Probe: 1-hidden-layer MLP (`hidden_dims: [512]`, GELU, dropout 0.0), followed
  by a linear head to `num_classes: 1` with sigmoid + BCE. `pos_weight: null`.
- Frozen parameters: all backbone parameters (features are pre-extracted into
  the probe cache before this experiment; only the MLP head is trained).
- The DINOv2 vs CUT3R token grids differ (16×16 vs 14×14). IoU is computed on
  each backbone's native grid; the cross-backbone comparison ADR (open) will
  decide whether to normalise for that.

## Data

- CO3Dv2 version: `v2_231130`.
- Manifest set: `full51-part-a-v1` under `artifacts/manifests/full51-part-a-v1/`.
- Manifest SHA-256 anchors (identical across all Part-A probe caches):
  - frames `8f3d9545e4c70998aefbe5bd78dd1c01be7dbe298b3bc10f7724bf962cec3496`;
  - sequences `7b7bf73eec1e4da84ad2a209003480b5c5adc600064e0490b87d30a010deae4c`;
  - windows `e499d9a8879a9f8d649eca0c8a28797f2cb846caff19e25b9d17195078f7b34c`.
- Splits, 26 CO3D categories (Part-A subset of the 51-category vocabulary):
  - train: 3054 windows;
  - val: 512 windows;
  - test: 101 windows.
- Leakage checks: manifest builder enforces disjoint sequence ids across
  train/val/test (see ADR 0002); no window straddles splits.
- Preprocessing version: probe caches
  `probe_cache_schema_version: probe-features-v2`, `layout: target_only`.
  Cache `checkpoint_provenance.sha256` matches the trust-anchor
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` for the
  CUT3R-random backbone (used to instantiate the module before
  `reset_parameters`); the DINOv2 cache carries the standard DINOv2 release
  provenance.

## Configuration

- Tracked configs (unchanged this experiment):
  - [`src/segmentation/configs/dinov2_partial_mlp.yaml`](../../src/segmentation/configs/dinov2_partial_mlp.yaml);
  - [`src/segmentation/configs/cut3r_random_partial_mlp.yaml`](../../src/segmentation/configs/cut3r_random_partial_mlp.yaml)
    (both renamed after this experiment; see EXP-007).
- Machine-local overrides under `local/*.vm.yaml` (excluded from git) only
  change `device: cpu → cuda` and `output.dir → ${CUT3R_ARTIFACT_ROOT}/…`.
- Optimiser: Adam, `lr 1e-3`, `weight_decay 0`, `batch_size 16`, 20 epochs.
- Seed: `20260729` for both `torch.manual_seed` and `random_init` provenance.
- Checkpointing: best-validation macro-foreground-IoU, saved to `head.pt`;
  last-epoch head kept alongside as `head-last.pt`. Driver:
  `local/train_best_val.py` (thin wrapper reusing `src/segmentation/`
  primitives; not committed).
- Hardware: NVIDIA A10-24Q on the Technion course VM,
  Python 3.11.15, torch 2.7.1+cu128.
- Exact commands, on the VM:

  ```bash
  # DINOv2
  python -m local.train_best_val --config local/dinov2.vm.bestval.yaml
  python -m src.segmentation.inference_segmentation \
    --config local/dinov2.vm.bestval.yaml --split val
  python -m src.segmentation.inference_segmentation \
    --config local/dinov2.vm.bestval.yaml --split test

  # CUT3R-random
  python -m local.train_best_val --config local/cut3r_random.vm.yaml
  python -m src.segmentation.inference_segmentation \
    --config local/cut3r_random.vm.yaml --split val
  python -m src.segmentation.inference_segmentation \
    --config local/cut3r_random.vm.yaml --split test
  ```

## Metrics and success criteria

- Primary metric: **macro-foreground-IoU** on the test split, evaluated at
  `mask_threshold 0.5`.
- Secondary: micro-IoU (token-weighted), token accuracy.
- Convention: token accuracy is reported for completeness but is **not**
  the ADR metric. The Part-A foreground-token fraction is ≈22 %, so a
  degenerate always-predict-background head yields token accuracy ≈ 0.78 at
  IoU 0.
- Success criterion for this record: report all six numbers (val + test) for
  each backbone from a bit-reproducible run under the seed above.

## Results

Best-val head (test row is the ADR value):

| Backbone            | Best-val epoch | Val macro-IoU | Test macro-IoU | Test micro-IoU | Test tok-acc |
|---------------------|---------------:|--------------:|---------------:|---------------:|-------------:|
| DINOv2 ViT-B/14     |              6 |        0.8360 |     **0.7922** |         0.7838 |       0.9595 |
| CUT3R-random        |             16 |        0.2579 |     **0.2134** |         0.1833 |       0.7416 |
| Δ (DINOv2 − random) |              — |       +0.5781 |     **+0.5788** |        +0.6005 |      +0.2179 |
| CUT3R-trained       |     — pending — |             — |              — |              — |            — |

Last-epoch (epoch 20) head, for reference:

| Backbone       | Val macro-IoU | Test macro-IoU | Test micro-IoU | Test tok-acc |
|----------------|--------------:|---------------:|---------------:|-------------:|
| DINOv2 ViT-B/14 |        0.8236 |         0.7903 |         0.7758 |       0.9582 |
| CUT3R-random    |        ~0.211 |          ~0.21 |          ~0.18 |       ~0.74  |

For DINOv2 the best-val vs last-epoch gap on test is ≈0.002 macro-IoU
(within n=101 noise). For CUT3R-random the val curve has not plateaued at
epoch 20 (train 0.267 vs val 0.211 macro-IoU), so a longer schedule would
raise the random floor somewhat while leaving DINOv2 essentially unchanged.

Determinism was verified twice: the two independent DINOv2 drivers
(`train_segmentation` and `train_best_val`) produced bit-identical per-epoch
loss and IoU sequences under the seed, and a rerun of `train_best_val` on
CUT3R-random reproduced every epoch's numbers bit-exactly — confirming the
head is retrained from scratch on each invocation (no accidental resumption
from a saved `head.pt`).

### Artifacts

All artifacts are on the Technion VM under `${CUT3R_ARTIFACT_ROOT}` and are
too large to commit. Structure:

```
${CUT3R_ARTIFACT_ROOT}/segmentation/
├── segmentation-dinov2/               # DINOv2, epoch-20 driver
│   ├── head.pt                        # last-epoch head
│   ├── metrics.json
│   ├── train.log
│   └── inference-{val,test}.json
├── segmentation-dinov2-bestval/       # DINOv2, best-val driver
│   ├── head.pt                        # best-val head (epoch 6)
│   ├── head-last.pt                   # last-epoch head (epoch 20)
│   ├── metrics.json                   # + best_val_epoch, best_val_macro_iou
│   ├── train.log
│   └── inference-{val,test}.json
└── segmentation-cut3r-random/         # CUT3R-random, best-val driver
    ├── head.pt                        # best-val head (epoch 16)
    ├── head-last.pt                   # last-epoch head (epoch 20)
    ├── metrics.json
    ├── train.log
    └── inference-{val,test}.json
```

`${CUT3R_ARTIFACT_ROOT}/from-drive/cut3r-random/cut3r_random.yaml` stashes the
collaborator's original feature-extraction config for provenance.

## Interpretation

Supported:

- On Part-A, a frozen DINOv2 ViT-B/14 with a 1-hidden-layer MLP probe reaches
  test macro-IoU 0.792. A CUT3R backbone with all parameters re-initialised
  under the same head and recipe reaches 0.213. The +0.58 gap is well outside
  the n=101 test-split noise band and outside best-val vs last-epoch
  variation for DINOv2 (≈0.002).
- The gap is not attributable to head capacity, seed, optimiser, or dataset
  (identical across both runs).
- Token accuracy alone would understate the gap by ~2.6× (0.960 vs 0.742,
  Δ 0.22) and would misleadingly suggest CUT3R-random "works" at 74 %; the
  actual segmentation quality at IoU 0.213 is close to the null baseline.

Not supported:

- Nothing here says anything about the **trained** CUT3R representation. The
  DINOv2-vs-random gap only bounds the head+recipe contribution from below;
  a trained CUT3R could plausibly land anywhere between 0.21 and above 0.79.
- The Part-A subset is 26 of 51 categories; whether the same ordering holds
  on the full 51 is untested. Part-B extraction and probing are future work.
- CUT3R-random's absolute number (0.213) is not the true random-floor for
  CUT3R — training had not plateaued at epoch 20. A longer schedule (40–60
  epochs) would move this number upward, but even the extrapolated plateau
  (≤ ~0.30 val macro-IoU) sits far below 0.79, so the qualitative conclusion
  is stable.

## Problems and deviations

- The shared folder intended as the CUT3R-trained probe cache turned out to be a Stage-0 *target feature cache*
  (`cache_schema_version: stage0-target-cache-v1`), missing the `split`,
  `seg_labels_key`, `category`, `sequence_id` columns that the segmentation
  loader (`ProbeCacheDataset`) requires. Blocker communicated to collaborator
  Max; no code adaptation attempted this experiment.
- No source files under `src/` were modified. All non-tracked assets live in
  `local/` on the VM and under `${CUT3R_ARTIFACT_ROOT}`.

## Next action

- **Blocker**: obtain a labelled CUT3R-trained probe cache (probe-features-v2,
  target-only layout, with `seg_labels`, `split`, `category`, `sequence_id`)
  from the collaborator, then rerun the exact commands above with
  `local/cut3r_trained.vm.yaml`. Wall-clock ≈5 minutes end to end.
  Done in [EXP-005](EXP-005-part-a-seg-cut3r-unblocked.md), which supersedes
  this record's numbers.
- Independent follow-up: run CUT3R-random for 40–60 epochs to lock down the
  true random-init plateau. Not on this experiment's critical path.
