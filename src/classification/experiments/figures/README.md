# Classification figures

What `src/classification/visualizations.py` produces, how to read each one, and what
each is **not** evidence of. Regenerate with, from the repository root:

```bash
python -m src.classification.visualizations
```

It discovers every run under `experiments/<features.source>/<experiment>/`, so a figure
covers exactly the runs on disk — nothing is simulated or filled in. Output goes to
`experiments/figures/` (git-ignored, like the runs themselves).

## The figures

| File | One per | Shows |
|---|---|---|
| `curves-<source>.png` | feature source | loss, accuracy, macro F1, macro recall over epochs — train and val |
| `curves-merged.png` | all runs | validation only, both sources overlaid |
| `summary.png` | all runs | final test accuracy / macro F1 / macro recall, every run on one axis |
| `confusion-<source>-<backbone>.png` | run | true vs predicted counts |
| `per_category-<source>-<backbone>.png` | run | precision, recall, F1 per category, sorted by recall |
| `top_confusions-<source>-<backbone>.png` | run | the most frequent "true → predicted" mistakes |

## How to read them

**Curves.** Colour identifies the **backbone**; line style carries the second dimension —
in `curves-<source>.png` faded solid is train and marked dashed is val, while in
`curves-merged.png` solid is `image_tokens` and dashed is `state_tokens`. The gap between
train and val is the thing to watch: a probe on frozen features overfits quickly, and
`head.pt` stores the **final** epoch, so a widening gap means the saved head is not the
best one.

**Summary.** The dotted line is chance, `1 / categories present`. Quote it whenever you
quote an accuracy: at 8 categories chance is 0.125, so 0.15 is barely above guessing;
across all 51 it is 0.02.

**Confusion matrix.** Rows are the true category, columns the prediction, so a row reads
"where did this category's windows go" and the diagonal cell is that category's recall.
Colour is the **window count**, matching the number in the cell. Only categories present
in the split get axes — the head always has 51 outputs, so a cache holding eight would
otherwise be mostly empty. Predictions into categories absent from the split cannot be
placed on the grid, so their total is noted under the x-axis rather than dropped.

The function also takes `normalize=True`, which shades by each row's share instead. That
is the fairer read when support is very uneven, where a rare category's total failure is
otherwise a pale cell beside a common category's ordinary one.

**Per-category bars.** Sorted by recall, with each category's support printed as `n=`.
Note that per-category **accuracy is recall** in a multiclass setting — the one-vs-rest
version that also counts true negatives sits near 1.0 for everything and says nothing.
Read the `n=`: a recall over four windows moves in steps of 0.25.

**Top confusions.** The same information as the matrix, but legible when there are 51
categories. Answers "which pairs does it mix up" directly.

## What these figures cannot tell you

- **Whether a gap is real.** Every figure shows a single run. With ~20 test windows per
  category, differences of a few points are inside noise. Repeated seeds, or a paired
  test over `per_window`, are what would support "A beats B".
- **How confident the probe was.** `inference-<split>.json` records the predicted category
  but not its probability, so calibration and "confident mistakes" are not plottable yet.
  Adding it means re-running inference, not just re-plotting.
- **Anything at all, if the run was synthetic.** Figures built from a dummy cache carry
  `SYNTHETIC DATA - pipeline check, not a result` in the corner and `(SYNTHETIC)` in the
  title. That marker comes from the cache's own metadata, so it cannot be forgotten.

## Design

Palette, fonts and grid match `src/segmentation/visualizations.py`, so Stage 1 and Stage 2
figures can sit in one report. Two rules the code follows deliberately: colour identifies
an entity and never a split or a metric, and the confusion matrix uses a single-hue
sequential ramp because it encodes magnitude, not identity.
