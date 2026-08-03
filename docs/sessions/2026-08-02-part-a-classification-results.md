# Session: Part-A classification results

- Date: 2026-08-02
- Author: Aviv Rabi
- Branch: classification-part-a-results
- Related issue/PR: follows #18
- Assistant/model, if used: Claude Code (Opus 5)

## Objective

Run the Stage 2 classification probe on the real Part-A probe-feature caches, settle the
open training choices with measurements rather than defaults, and commit the
configuration that produced the reported numbers.

## Context and inputs

- A third VM (`104.210.220.111`, A10-24Q) turned out to hold four **complete** probe
  caches — `cut3r-trained`, `cut3r-random`, `dinov2-vitb14`, `rgb-patch-random`, 116/116
  shards each, identical `manifest_sha256`. Stage 2 had been blocked on exactly this.
- Part A is the 26 **even-indexed** categories of the 51-category vocabulary; 3667
  windows over 1008 sequences, balanced to 133-145 windows per category.

## Work completed

- Files changed:
  - `dataset_classification.py`: `cache_categories()`, and a `label_space` argument.
    Labels are now looked up by category **name**, with a guard that fails loudly if a
    cache's own `category_index` disagrees with this repo's vocabulary.
  - `train_classification.py`: `resolve_label_space()`; `resolve_model_config()` now
    takes the label space; `MulticlassMetrics` takes the vocabulary that names its
    outputs; the space is written into `metrics.json` and `head.pt`.
  - `inference_classification.py`: reads the label space from the checkpoint and refuses
    a config that resolves to a different one.
  - `model_classification.py`: dropout now applies to a head with no hidden layers
    (previously the loop that inserted it never ran, so `dropout` was silently ignored
    for the linear probe). MLP layout is unchanged.
  - `visualizations.py`: `plot_curves_train_val` (train and val side by side, y shared
    per row); `_vocabulary()` prefers the run's own label space, without which a 26-way
    head's confusion matrix would be labelled with the wrong names.
  - `configs/`: replaced three configs with **eight** — one per backbone and arm. The old
    three could not express the two-arm comparison and `rgb-patch-random` had none.
  - `tests/test_classification_training.py`: six new cases for the label space.
- Artifacts: run directories and figures under `src/classification/experiments/`
  (git-ignored). Nothing generated is committed.

## Decisions

- Made:
  - **`label_space: present`** for Part A. The owner's argument was specification, not
    regularization, and that is the defensible one: measured over 3 seeds it gives
    +0.006 val macro F1 and leaves the train/val gap unchanged. It is an option, not a
    replacement — `vocabulary` remains the default so heads stay comparable across caches.
  - **30 epochs**, from 10. Worth +0.084 val accuracy on CUT3R image tokens; larger than
    every regularizer tried, combined.
  - **`weight_decay: 0.0`**, uniform. Swept `[1e-4 .. 1.0]`: the best alternative was
    +0.008 on one arm, inside what 512 val windows from 130 sequences resolve. Per-arm
    selection off a val maximum was rejected as val-fitting.
  - **`dropout: 0.0`.** Once it actually did something it monotonically hurt CUT3R and did
    nothing for DINOv2.
  - **Linear head reported**, not `[512]`. The MLP was within 0.024 and hit train accuracy
    1.0000.
  - **`balanced_sampler: false`** — Part A is balanced 1.09:1, so resampling adds noise
    and corrects nothing.
- Still open:
  - Best-epoch checkpointing. `head.pt` is the final epoch, which is wrong in opposite
    directions for the two backbones.
  - Seeds. Everything reported is one seed.
  - Where CUT3R plateaus past 30 epochs.
  - PCA on the input, the one untried lever aimed at the actual cause of the gap.

## Verification

| Command/check | Result |
|---|---|
| `python -m pytest -q` | 109 passed (103 before, plus 6 label-space cases) |
| 8 configs trained + val inference | All 8; inference reproduces training val exactly |
| Label-space correctness | 26 outputs, confusion indices 0-25, every predicted name inside the space, accuracy recomputed from `per_window` matches |
| Split leakage | 0 sequences span two splits, checked against `index.parquet` |
| Figures | 30, all from `metrics.json` / `inference-val.json` |
| `git status` | No cache, run directory, checkpoint, or image staged |

Results are in `docs/experiments/EXP-004-part-a-classification.md`. Headline: CUT3R-trained
state 0.5684 val accuracy against 0.0385 chance, 0.1562 CUT3R-random and 0.0820
RGB-random; DINOv2 0.9668.

## Human review of AI-assisted work

Pending. Reviewer should confirm: replacing three configs with eight is wanted; that
`weight_decay: 0.0` rather than per-arm bests is the right call; and that reporting val
while test stays unspent matches the intent.

## Next step

Best-epoch checkpointing, then repeat over 3-5 seeds. Neither changes the ordering, both
change what can be claimed about the size of the gaps.
