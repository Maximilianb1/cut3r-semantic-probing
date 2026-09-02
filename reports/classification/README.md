# Stage 2 - 26-way classification results

Held-out test results for the image-level classification probe, over three
frozen representations and two probe heads. These are the numbers reported in
the final report and the talk.

## Result

Test window accuracy, 1,064 windows from 280 held-out CO3D sequences, 26
categories, chance 0.0385. Intervals are 95% percentile sequence-cluster
bootstrap over 20,000 resamples.

| Representation | Linear head | MLP-512 head |
|---|---|---|
| CUT3R-trained | **0.708** [0.663, 0.750] | 0.683 [0.636, 0.729] |
| CUT3R-random | 0.214 [0.173, 0.258] | 0.244 [0.203, 0.288] |
| DINOv2 ViT-B/14 | **0.953** [0.930, 0.973] | 0.960 [0.939, 0.978] |

Three things follow, and only these three:

1. **CUT3R's pretrained state carries category identity.** 0.708 against a
   0.214 random-initialised control on the same architecture and the same data
   is a paired difference of +0.493 [0.443, 0.543] - the semantics come from
   3D pretraining, not from the probe.
2. **A 2D vision backbone is still clearly ahead.** DINOv2 beats CUT3R-trained
   by +0.245 [0.203, 0.288] paired. The gap is real, not a sampling artefact.
3. **Head capacity buys nothing on CUT3R.** At matched hyperparameters the
   MLP-512 is not better than the linear head: -0.024 [-0.051, 0.002] paired
   at window level. Whatever category information is present is already close
   to linearly available.

On (3), one honest caveat: the comparison above holds the optimiser and
regularisation fixed. Regularised MLP variants in `src/classification/configs/fullunion-resplit/`
do reach a higher *validation* macro-F1 for CUT3R-trained (0.692 for
`mlp512_dropout05_ls01_wd1e3` against 0.666 for `linear_adam`). We report the
matched-hyperparameter pair because that is the capacity ablation; we do not
claim a linear head is the best possible head.

## Files

| File | What it is |
|---|---|
| `predictions/<representation>_<head>.csv` | Per-window class probabilities on the test split. The input to everything below. |
| `model-test-metrics.csv` | One row per run: accuracy, top-5, macro precision/recall/F1, NLL, Brier, ECE, and true-class rank statistics, at window and sequence level. |
| `model-test-per-class-metrics.csv` | Per-category support, precision, recall, and F1. |
| `bootstrap-accuracy-ci.csv` | Sequence-cluster bootstrap point estimate, 95% CI, and standard error per run. |
| `bootstrap-accuracy-differences.csv` | Paired and unpaired differences between every reported pair of runs. |
| `test-bootstrap-report.json` | The complete record, including the resampling protocol. |
| `bootstrap-test-accuracy-{linear,mlp512}.png` | The two bar charts used in the talk. |
| `bootstrap-window-accuracy-ci.png` | All six runs on one axis. |
| `paired-bootstrap-window-accuracy-differences.png` | Every paired difference with its 95% CI. |
| `bootstrap-seeds.json`, `selected-epochs.json` | The RNG seeds and best-validation epochs behind the published run. |

## Why the bootstrap resamples sequences

CO3D windows from one sequence are four views of one physical object, so they
are not independent observations. Treating them as independent would shrink
every interval by roughly the square root of the windows-per-sequence factor.
Every interval here therefore resamples complete sequence clusters and carries
all of a sequence's windows together, and every difference reuses the same
cluster draw for both models so the within-sequence error correlation is kept.
See `src/classification/bootstrap_accuracy.py`.

## Reproducing

Everything in this directory except `predictions/` is regenerated from
`predictions/` alone:

```bash
python -m src.classification.build_test_report \
  --predictions-dir reports/classification/predictions \
  --output-dir reports/classification \
  --seeds reports/classification/bootstrap-seeds.json \
  --epochs reports/classification/selected-epochs.json
```

Passing `--seeds` reproduces the published tables bit for bit. Without it, the
per-run seeds are spawned from `--seed` and the point estimates are unchanged
while the interval endpoints move by Monte-Carlo error.

To reproduce `predictions/` itself you need the probe-feature caches; see
[docs/REPRODUCING.md](../../docs/REPRODUCING.md).
