# Evaluation protocol

How the reported numbers are aggregated, and where that differs from what was
originally planned.

## Segmentation: foreground IoU

For a predicted foreground mask `P` and ground-truth foreground mask `G`:

```text
IoU = |P and G| / |P or G|
```

1 for perfect overlap, 0 when non-empty masks do not overlap. A window with no
foreground at all scores 1.0 under the current empty/empty convention. That is a
choice, not a fact; such windows are rare in CO3D's object-centric sequences.

Masks are compared at **token / patch-grid resolution**, not full pixel
resolution: the ground-truth mask is pooled to the backbone's token grid, and
the probe predicts one label per token.

Three aggregations are computed, and all three are reported:

| Name | Definition |
|---|---|
| **macro-IoU** | mean IoU over evaluated windows — **the reported headline** |
| micro-IoU | one IoU over all tokens pooled together |
| per-category | mean over the windows of each category, reported separately |

Macro and micro are reported side by side on purpose. They agree for DINOv2
(0.806 / 0.805) and CUT3R-trained (0.777 / 0.749), which rules out a
foreground-size bias; they diverge for CUT3R-random (0.277 / 0.223), which says
it does worse still on large-foreground windows.

## Classification: accuracy at two levels

**Window accuracy** — one prediction per window; the reported headline.

**Sequence accuracy** — a sequence's windows have their probability vectors
averaged into one prediction for the whole sequence. Reported beside the window
number in every table.

Both are reported because a sequence is the independent physical unit while a
window is the unit the head actually sees.

## Uncertainty: resample sequences, never windows

Four windows drawn from one CO3D sequence are four views of one object. They are
not four independent observations, and an interval that treats them as such is
too narrow by roughly the square root of the windows-per-sequence factor.

Every interval in this project therefore resamples complete **sequence
clusters**, carrying all of a sequence's windows together. Every difference
between two runs reuses the same cluster draw for both, so the within-sequence
error correlation is preserved; the unpaired version is computed as a
sensitivity check and never used as the headline.

Implementations: `src/classification/bootstrap_accuracy.py` and
`src/segmentation/analysis/build_score_comparison.py`.

## Deviation from the original plan

The proposal called for a **category macro** — mean windows within a sequence,
sequences within a category, then categories with equal weight — as the primary
headline, on the grounds that categories hold unequal numbers of usable samples.

We report the window macro instead, with per-category breakdowns beside it. The
reason is sample size: the Part-A test split is 101 windows over 26 categories,
roughly four per category. A category macro over four-window estimates is
noisier than the window macro, not less biased, and it would place the same
weight on a category with two usable windows as on one with six.

The per-category tables are still reported in full — in
[reports/segmentation](../reports/segmentation/README.md) and
`reports/classification/model-test-per-class-metrics.csv` — and every experiment
record states the per-category counts, so the aggregation can be recomputed by a
reader who prefers the other convention.
