# EXP-004: Part-A image-level classification probes

- Date: 2026-08-02
- Owner: Aviv Rabi
- Status: Completed
- Related issue/PR: follows #18 (Stage 2 classification probe)
- Code commit: this PR

## Hypothesis

If CUT3R's frozen features encode object category, a linear probe over them should score
well above chance, above a randomly initialised CUT3R, and above a random projection of
pixels. ADR 0003 additionally asks which representation carries more of it: the pooled
target-frame image tokens, or the persistent recurrent state after frame 6.

## Representation and model

- Backbones: `cut3r-trained`, `cut3r-random` (seed 20260729), `dinov2-vitb14`,
  `rgb-patch-random` (seed 20260731). All frozen; none is run here.
- Tokens: target frame only. `image_tokens` = grid tokens (CUT3R 576, DINOv2 740),
  mean-pooled. `state_tokens` = CUT3R's persistent state `[768, 768]`, mean-pooled; for
  DINOv2 and RGB-random there is no recurrent state, so that arm reads their `[1, 768]`
  global vector instead.
- Probe: **linear**, `nn.Linear(768 -> 26)`, `hidden_dims: []`. 19,994 parameters.
- Trainable: the head only. Features are read from the probe-feature cache.

## Data

- CO3Dv2, Part A: the 26 **even-indexed** categories of the 51-category vocabulary.
- Manifests: `full51-part-a-v1`; all four caches carry identical `manifest_sha256`
  (`windows e499d9a8...`), so the arms are comparable window for window.
- Split: 3054 train / 512 val / 101 test windows, over 1008 sequences (779/130/99).
  Assigned at **sequence** level by the manifest.
- Leakage checks: 0 sequences appear in more than one split (verified against
  `index.parquet`); `assert_sequence_disjoint` runs on every training call.
- Balance: 133-145 windows per category, ~30 training sequences each.
- Cache integrity: 116/116 shards present for all four caches.

## Configuration

- Config files: `src/classification/configs/{cut3r_trained,cut3r_random,dinov2,rgb_patch_random}_{image,state}.yaml`
- Seed: 20260731 (single seed - see Problems)
- Hardware: NVIDIA A10-24Q, torch 2.11.0+cu128
- Exact command, per config:

```
CUT3R_CACHE_ROOT=<cache root> python -m src.classification.train_classification --config <config>
CUT3R_CACHE_ROOT=<cache root> python -m src.classification.inference_classification --config <config> --split val
```

## Metrics and success criteria

Accuracy, macro precision/recall/F1, top-5 accuracy. Chance is **0.0385** (1/26).
Micro variants are not reported: in single-label multiclass they all equal accuracy.

Success = CUT3R-trained clearly above both random baselines. No threshold was set for
the DINOv2 comparison; DINOv2 is a reference point, not a target.

## Results

Validation, 512 windows. Test is untouched.

| Backbone | Arm | Accuracy | Macro F1 | Top-5 | Train acc |
|---|---|---:|---:|---:|---:|
| dinov2-vitb14 | state (CLS) | **0.9668** | 0.9663 | 1.0000 | 1.0000 |
| dinov2-vitb14 | image | 0.9473 | 0.9456 | 0.9922 | 1.0000 |
| cut3r-trained | **state** | **0.5684** | 0.5637 | 0.8457 | 0.9967 |
| cut3r-trained | image | 0.5117 | 0.5106 | 0.7910 | 0.9165 |
| cut3r-random | state | 0.1562 | 0.1487 | 0.3789 | 0.3870 |
| cut3r-random | image | 0.1465 | 0.1412 | 0.3828 | 0.3965 |
| rgb-patch-random | either | 0.0820 | 0.0608 | 0.2559 | 0.1300 |
| *chance* | | *0.0385* | | | |

Artifacts: run directories and figures under `src/classification/experiments/`
(git-ignored). Figures: confusion matrix, per-category bars, top confusions, train-vs-val
curves per arm, merged val curves, cross-run summary.

## Interpretation

**Supported.** CUT3R's trained features carry object-category information well above
chance and well above both null models: 0.5684 against 0.0385 chance, 0.1562 for a
randomly initialised CUT3R, and 0.0820 for a random projection of pixels. The ordering
trained > random-init > random-projection is what a "the training, not the architecture"
claim requires.

**Supported.** The persistent state beats the grid tokens on both CUT3R arms
(+0.0567 accuracy trained, +0.0097 random). The gap appears only in the trained model, so
it reflects what training put in the recurrent state rather than the tensor's shape.

**Supported, and worth stating plainly.** DINOv2 is far stronger (0.9668 vs 0.5684). On
this task, a self-supervised 2D backbone carries much more category information than a 3D
reconstruction backbone.

**Not supported: any claim about the size of these gaps.** One seed, and 512 val windows
from 130 sequences. Differences under roughly 3-4 points are not resolvable here; only
the between-backbone ordering is.

**Not supported: that the errors are semantic.** CUT3R's confusions are shape-based -
`baseballbat` to `carrot` (12 of 20), `apple` to `ball`, `toyplane` to `toytruck`. A
per-category spread from `hydrant` recall 1.00 to `toyplane` 0.06 sits behind the 0.5684
average.

**`rgb-patch-random` has only one arm.** Its stored global vector equals the mean of its
spatial tokens (median elementwise ratio 1.00000, max difference 0.0017), so both arms
feed identical inputs. Reported once, not twice.

## Problems and deviations

- **Single seed.** The 3-seed side experiment on the head width gave a per-seed spread of
  0.003-0.010 macro F1, which bounds what a single run can claim.
- **`head.pt` is the final epoch, not the best.** This is wrong in opposite directions for
  the two backbones: DINOv2 peaks at epoch 1-4 and decays, while CUT3R was still improving
  at epoch 30. The CUT3R numbers above are therefore a floor.
- **Tuning happened on val**, which is also the split reported. Test was reserved instead
  of being used as a second check.
- Ablations behind the configuration (all on `cut3r-trained`, val):

  | Choice | Alternative | Effect |
  |---|---|---|
  | 30 epochs | 10 epochs | **+0.084** accuracy (image arm) |
  | `normalize: standardize` | `none` | **+0.236** accuracy (image arm) |
  | `label_space: present` (26) | `vocabulary` (51) | +0.006 macro F1, 3 seeds; train/val gap unchanged |
  | `weight_decay: 0.0` | swept to 1.0 | best alternative +0.008, inside noise; >=1e-2 clearly worse |
  | `dropout: 0.0` | swept to 0.7 | monotonically worse, down to -0.11 accuracy |
  | linear head | `hidden_dims: [512]` | within 0.024, and train accuracy 1.0000 |

## Next action

1. Best-epoch checkpointing on val macro F1 - currently the single largest known error.
2. Repeat over 3-5 seeds before any gap is quoted with a number.
3. Train CUT3R past 30 epochs to find where it actually plateaus.
4. Part B (the 25 odd-indexed categories) doubles the independent sequence count, which
   matters more than any regularizer tried here: the train/val gap survived every one.
