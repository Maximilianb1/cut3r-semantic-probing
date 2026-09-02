# EXP-008: 26-way classification, linear vs. MLP head, three frozen representations

- Date: 2026-08-25
- Owner: Jeremy Jornet
- Status: Completed — these are the reported Stage 2 results
- Related PR: #23
- Supersedes: [EXP-004](EXP-004-part-a-classification.md), which reported the
  same probe on the un-unioned Part-A cache and on the validation split only

## Question

Do frozen CUT3R representations encode object category, and is that information
linearly available?

Two things are being separated. Whether the *representation* carries category
identity is answered by the gap between pretrained CUT3R and a random-initialised
CUT3R of the same architecture. Whether the information is *linearly available*
is answered by the gap between a linear head and an MLP head on the same
features. Neither question is answered by a single accuracy number.

## Data

The union of the three Part-A probe caches — original (3,667 windows), leftover
(124), and `cap100-new-train` (6,901) — 10,692 windows over 26 categories,
re-split 80/10/10 at **sequence** level under seed `20260731`. The inherited
split was 10,029 / 559 / 104, which leaves a test set too small to say anything;
the re-split gives 8,514 train / 1,114 val / 1,064 test windows over 280 test
sequences.

The split is sequence-level, so no two windows of one physical object appear on
both sides of it. `assert_sequence_disjoint` enforces this at load time and
fails the run rather than warning.

Chance accuracy is 1/26 = 0.0385.

## Configuration

- Representations: CUT3R-trained, CUT3R-random, DINOv2 ViT-B/14, all frozen.
- Feature: `state_tokens`, mean-pooled to one 768-vector per window, standardised.
- Heads: `hidden_dims: []` (linear) and `hidden_dims: [512]` (MLP), GELU, no dropout.
- Optimiser: Adam, lr 1e-3, weight decay 0, batch size 64, 100 epochs, seed `20260731`.
- Checkpoint: best validation macro-F1.
- Label space: `present` (26 outputs, the categories the cache holds).
- Configs: [`src/classification/configs/fullunion-resplit/`](../../src/classification/configs/fullunion-resplit/),
  `*_state_{linear,mlp512}_adam.yaml`.

The linear and MLP runs differ **only** in `model.hidden_dims`. That is what
makes the comparison a capacity ablation rather than a hyperparameter search.

## Result

Test window accuracy with 95% percentile sequence-cluster bootstrap CI,
20,000 resamples:

| Representation | Linear | MLP-512 |
|---|---|---|
| CUT3R-trained | 0.708 [0.663, 0.750] | 0.683 [0.636, 0.729] |
| CUT3R-random | 0.214 [0.173, 0.258] | 0.244 [0.203, 0.288] |
| DINOv2 ViT-B/14 | 0.953 [0.930, 0.973] | 0.960 [0.939, 0.978] |

Paired differences on the same test sequences (window level, 95% CI):

| Comparison | Difference | Reading |
|---|---|---|
| CUT3R-trained − CUT3R-random (linear) | +0.493 [0.443, 0.543] | pretraining, not architecture |
| DINOv2 − CUT3R-trained (linear) | +0.245 [0.203, 0.288] | a real remaining gap |
| MLP − linear, CUT3R-trained | −0.024 [−0.051, 0.002] | capacity buys nothing |
| MLP − linear, CUT3R-random | +0.030 [−0.005, 0.065] | not resolved either way |
| MLP − linear, DINOv2 | +0.007 [0.000, 0.015] | negligible |

Full tables, per-category metrics, and figures:
[reports/classification/](../../reports/classification/README.md).

## Interpretation

**CUT3R's persistent state encodes category identity.** 0.708 against 0.214 for
the same architecture with untrained weights, on identical data and an identical
head, is a difference no probe capacity can manufacture. The signal is in the
representation.

**It is close to linearly available.** The MLP is not better than the linear
head on CUT3R-trained; the paired interval barely crosses zero on the wrong
side. Combined with the same finding on segmentation
([EXP-007](EXP-007-part-a-seg-probe-capacity-ablation.md)), the picture is a
representation whose semantic content is already largely disentangled rather
than one that needs a nonlinear readout to unlock.

**A dedicated 2D vision model is still clearly ahead.** DINOv2 at 0.953 is
0.245 above CUT3R-trained, well outside the interval. Geometric pretraining
produces real semantics as a by-product; it does not close the gap to a model
trained for exactly this.

## Limitations

- **Head capacity is ablated at one point, not swept.** The claim is "an MLP-512
  at matched hyperparameters is not better", not "no head could do better".
  Regularised MLP variants in the same config directory reach a higher
  *validation* macro-F1 on CUT3R-trained (0.692 for `mlp512_dropout05_ls01_wd1e3`
  against 0.666 for `linear_adam`). We report the matched pair because that is
  the ablation; we do not claim the linear head is optimal.
- **CUT3R-trained and CUT3R-random caches use different layouts** (`trajectory`
  and `target_only`). The control isolates weights *and* layout, not weights
  alone. Re-extracting CUT3R-random with `trajectory` would remove the confound.
  It would have to make up a 0.49 gap to change the conclusion.
- **The union-and-resplit tooling is not in this repository.** The cache
  metadata records the protocol and seed; the script that applied them was
  VM-local. See the known-gap note in
  [docs/REPRODUCING.md](../REPRODUCING.md).
- **~41 test windows per category** (1,064 over 26). Per-category numbers in
  `model-test-per-class-metrics.csv` are indicative; the aggregate is what the
  intervals are computed on.

## Reproducing

The analysis stage runs from the committed per-window predictions with no
dataset and no GPU, and reproduces every table here bit for bit:

```bash
python -m src.classification.build_test_report \
  --predictions-dir reports/classification/predictions \
  --output-dir reports/classification \
  --seeds reports/classification/bootstrap-seeds.json \
  --epochs reports/classification/selected-epochs.json
```

Training commands are in [docs/REPRODUCING.md](../REPRODUCING.md).
