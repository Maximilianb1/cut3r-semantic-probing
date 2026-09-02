# Classification results

Held-out test results for the image-level classification probe: three frozen
backbones, two probe heads, 26 CO3D categories.

| Backbone | What it is |
|---|---|
| CUT3R-trained | The pretrained CUT3R 3D-reconstruction model, frozen — the subject of the project. |
| CUT3R-random | The same architecture with untrained weights — the control for what pretraining contributes. |
| DINOv2 ViT-B/14 | An off-the-shelf pretrained vision model — the upper anchor. |

The two heads differ only in `model.hidden_dims` (`[]` against `[512]`), so the
comparison between them is a capacity ablation rather than a hyperparameter
search.

## Results

Test window accuracy over 1,064 windows from 280 held-out sequences, chance
0.0385. Intervals are 95% percentile sequence-cluster bootstrap, 20,000
resamples.

| Backbone | Linear head | MLP-512 head |
|---|---|---|
| CUT3R-trained | 0.708 [0.663, 0.750] | 0.683 [0.636, 0.729] |
| CUT3R-random | 0.214 [0.173, 0.258] | 0.244 [0.203, 0.288] |
| DINOv2 ViT-B/14 | 0.953 [0.930, 0.973] | 0.960 [0.939, 0.978] |

Paired differences on the same test sequences, window level, 95% CI:

| Comparison | Difference |
|---|---|
| CUT3R-trained − CUT3R-random, linear | +0.493 [0.443, 0.543] |
| DINOv2 − CUT3R-trained, linear | +0.245 [0.203, 0.288] |
| MLP − linear, CUT3R-trained | −0.024 [−0.051, 0.002] |
| MLP − linear, CUT3R-random | +0.030 [−0.005, 0.065] |
| MLP − linear, DINOv2 | +0.007 [0.000, 0.015] |

## Why the bootstrap resamples sequences

Windows from one CO3D sequence are several views of one physical object, so
they are not independent observations. Treating them as independent would
shrink every interval by roughly the square root of the windows-per-sequence
factor. Every interval here resamples complete sequence clusters and carries
all of a sequence's windows together; every difference reuses the same cluster
draw for both models, so the within-sequence error correlation is preserved.
See `src/classification/bootstrap_accuracy.py`.

## Files

| File | Contents |
|---|---|
| `predictions/<backbone>_<head>.csv` | Per-window class probabilities on the test split. The input to everything else here. |
| `model-test-metrics.csv` | One row per run: accuracy, top-5, macro precision/recall/F1, NLL, Brier, ECE, and true-class rank statistics, at window and sequence level. |
| `model-test-per-class-metrics.csv` | Per-category support, precision, recall, and F1. |
| `bootstrap-accuracy-ci.csv` | Point estimate, 95% CI, and bootstrap standard error per run. |
| `bootstrap-accuracy-differences.csv` | Paired and unpaired differences for every reported pair of runs. |
| `test-bootstrap-report.json` | The complete record, including the resampling protocol. |
| `bootstrap-test-accuracy-{linear,mlp512}.png` | Accuracy per backbone with its CI, one figure per head. |
| `bootstrap-window-accuracy-ci.png` | All six runs on one axis. |
| `paired-bootstrap-window-accuracy-differences.png` | Every paired difference with its 95% CI. |
| `bootstrap-seeds.json`, `selected-epochs.json` | The RNG seeds and best-validation epochs behind these runs. |

## Regenerating

Everything except `predictions/` is rebuilt from `predictions/` alone — no
dataset, no GPU, no model weights:

```bash
python -m src.classification.build_test_report \
  --predictions-dir reports/classification/predictions \
  --output-dir reports/classification \
  --seeds reports/classification/bootstrap-seeds.json \
  --epochs reports/classification/selected-epochs.json
```

With `--seeds` this reproduces the tables above exactly. Without it, per-run
seeds are spawned from `--seed`: the point estimates are unchanged and the
interval endpoints move by Monte-Carlo error.

To reproduce `predictions/` itself, see
[docs/REPRODUCING.md](../../docs/REPRODUCING.md).
