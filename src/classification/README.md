# Multiclass Classification

**Stage 2 — image-level object classification over the 51 CO3D categories.**

## Objective

Test whether frozen CUT3R representations encode **semantic object identity**, not just
figure-ground. Three frozen backbones are compared with the **same** head:
CUT3R-trained, CUT3R-random, and DINOv2.

The prediction is **global to the target frame**: every token in the frame is pooled
into one vector, and the head emits one distribution over the 51 categories for the
whole frame.

## Scope

This package **trains and evaluates the probe head, and nothing else**. No backbone is
loaded or run here: embeddings and labels are an *input*, read from a probe-feature
cache that already exists on disk. Producing those caches is the data pipeline's job —
see [`configs/probe_features/`](../../configs/probe_features/) and
[`scripts/extract_probe_features.py`](../../scripts/extract_probe_features.py).

## Which representation to classify from

ADR 0003 records that image-level classification should **compare** spatially pooled
`image_tokens[5,0]` with token-pooled `state_tokens[5,0]`:

| `features.source` | What it reads | Shape before pooling | Question it answers |
|---|---|---|---|
| `image_tokens` | CUT3R `image_tokens[5,0]` / DINOv2 patch tokens | `[N, 768]` | does the representation of *this frame* carry object identity? |
| `state_tokens` | CUT3R `state_tokens[5,0]` (recurrent memory) / DINOv2 CLS | `[M, 768]` | does what CUT3R *carries across frames* carry object identity? |

`features.source` is **required** — there is no default — and both `metrics.json` and
`head.pt` record which one produced a result. The state latent is not pixel-aligned and
must never be reshaped to a grid.

## How it works

1. **Input** — a probe-feature cache directory, one per backbone. Each entry already
   holds the window's `category` and `category_index`, so the **label costs no tensor
   read**; it comes from the cache index.
2. **`dataset_classification.py`** reads that cache, filters to one split, pools each
   window's chosen tensor to a single `[D]` vector, and stacks batches. Only the
   requested tensor is transferred — the grid tokens and the state latent are never
   both read.
3. **`train_classification.py`** trains only the MLP head with `CrossEntropyLoss`,
   evaluating on val each epoch and saving `head.pt` + `metrics.json`.
4. **`inference_classification.py`** reloads `head.pt` and evaluates a held-out split,
   writing per-window predictions and a confusion matrix.

Because a batch is one vector per window, there is none of the variable-token-count
collation the segmentation dataset needs.

### What the head's outputs stand for: `model.label_space`

`category_index` is a position in the sorted 51-category CO3D vocabulary — `0 apple`,
`1 backpack`, `2 ball`, … — **not** a position among the categories a cache happens to
hold. A cache with only 8 categories still yields indices like 40, so the output
dimension cannot simply be "however many categories are here".

Two label spaces are available, and `num_classes` is derived from whichever is chosen —
writing a different value is rejected rather than silently honoured:

| `model.label_space` | Outputs | Use when |
|---|---|---|
| `vocabulary` (default) | 51 | Heads must stay comparable across caches, including a later cache that adds categories. |
| `present` | one per category in the cache (26 for Part A) | The cache covers a subset and you want the correctly specified model. |

`present` is the Part-A default in the shipped configs. It is **not** an overfitting fix:
measured over 3 seeds it is worth about +0.006 val macro F1 and leaves the train/val gap
unchanged, because the outputs it removes were never discriminating between the
categories that are actually there — they only ever learned to say "not me". Its real
cost is comparability: a `present` head's output 7 means something different from a
`vocabulary` head's output 7, so the label space travels inside `head.pt` and inference
refuses a checkpoint whose space disagrees with the config.

Under `vocabulary`, categories absent from a split simply never appear as targets and the
macro metrics skip them.

### No custom collate

Pooling makes every item the same shape, so PyTorch's `default_collate` stacks a batch
on its own: `features` into `[B, D]`, `label` into `[B]`, and the two string fields into
lists of `B`. The segmentation package needs a custom collate because its token piles are
ragged; here there is nothing to reconcile, so there is no collate function to read.

Batch keys therefore stay **singular** — `batch["label"]`, `batch["category"]` — matching
the keys `__getitem__` returns. `default_collate` preserves order, so row `i` of every
entry describes the same window, which is what the per-category metrics and the confusion
matrix depend on.

## Metrics

| Metric | Reads |
|---|---|
| `accuracy` | fraction of frames whose top choice is right |
| `macro_recall` | mean per-category recall — is every category *found*? |
| `macro_precision` | mean per-category precision — is a category predicted only when really there? |
| `macro_f1` | mean of the **per-category** F1s |
| `top5_accuracy` | is the truth in the top five? (out of the label space, so 26 or 51) |
| `per_category_{precision,recall,f1}` | the breakdown behind the averages |

Accuracy alone misleads under class imbalance: it can look healthy while whole
categories are never predicted, which is exactly what the macro numbers expose.

Two things worth knowing when reading them. *Micro* precision, recall and F1 all equal
accuracy in single-label multiclass, so they are not reported as separate numbers.
`macro_f1` is the mean of the per-category F1s, **not** the F1 of `macro_precision` and
`macro_recall` — a different and less meaningful quantity.

Macro averages cover categories present in the split's targets; an absent category is
omitted rather than scored 0, which would silently drag the mean down. A present
category the probe never predicts still scores 0, which is the honest reading.

Chance level is `1 / categories present`, so quote it next to any accuracy: 51
categories put chance near 0.02, and a small subset puts it much higher.

## Part-A results

Full record, with every ablation behind the shipped configuration:
[`docs/experiments/EXP-004-part-a-classification.md`](../../docs/experiments/EXP-004-part-a-classification.md).
Validation, 512 windows, 26 categories, chance 0.0385:

| Backbone | Arm | Accuracy | Macro F1 |
|---|---|---:|---:|
| dinov2-vitb14 | state (CLS) | 0.9668 | 0.9663 |
| dinov2-vitb14 | image | 0.9473 | 0.9456 |
| cut3r-trained | state | 0.5684 | 0.5637 |
| cut3r-trained | image | 0.5117 | 0.5106 |
| cut3r-random | state | 0.1562 | 0.1487 |
| cut3r-random | image | 0.1465 | 0.1412 |
| rgb-patch-random | either | 0.0820 | 0.0608 |

CUT3R's trained features are far above both null models, and the persistent state beats
the grid tokens — the ADR 0003 comparison, answered on val. DINOv2 is far ahead of both.

**One seed, and val is 512 windows from 130 sequences: differences under ~3–4 points are
not resolvable.** Test is untouched.

> These are the first-pass numbers ([EXP-004](../../docs/experiments/EXP-004-part-a-classification.md)).
> The **reported** results run the same probe on the unioned, re-split cache with
> best-validation checkpoint selection and a held-out test split — see
> [EXP-008](../../docs/experiments/EXP-008-classification-linear-vs-mlp.md) and
> [reports/classification](../../reports/classification/README.md).

## Files

| File | Purpose |
|---|---|
| `model_classification.py` | `ClassificationProbe` = pooling + trainable MLP head (`hidden_dims=[]` gives a true linear probe). The head is defined here rather than shared with `src.segmentation`: the two answer different questions, and keeping them separate means neither can be changed by an edit meant for the other. |
| `dataset_classification.py` | `ProbeCacheClassificationDataset` over the probe-feature cache (one pooled vector + one label per window). |
| `train_classification.py` | Config-driven training loop; cross-entropy; asserts sequence-disjoint splits; saves `head.pt`. `training.checkpoint_selection` picks what `head.pt` holds: `final` (default) or `best_val_macro_f1`. |
| `inference_classification.py` | Reloads `head.pt` and evaluates a chosen split (default `test`); writes per-window predictions, a confusion matrix, and `inference-<split>-probabilities.csv`. |
| `bootstrap_accuracy.py` | Percentile bootstrap over complete CO3D **sequence** clusters — marginal intervals and paired/unpaired differences. Windows within a sequence are not independent observations, so windows are never the resampling unit. |
| `build_test_report.py` | Turns per-window probability CSVs into the tables and figures in `reports/classification/`. Refuses to compare runs that were not evaluated on identical test observations. |
| `rank_visualization.py` | Where the true class ranks among the head's outputs, against the explicit random-ranking baseline. |
| `configs/*.yaml` | One config per backbone **and arm** (`<backbone>_{image,state}.yaml`), the first-pass Part-A runs. |
| `configs/fullunion-resplit/*.yaml` | The reported runs: three representations x seven head/regularisation settings on the unioned, re-split cache. The two `*_{linear,mlp512}_adam.yaml` per representation differ **only** in `model.hidden_dims` — that is the capacity ablation. |
| `visualizations.py` | Figures built from the run outputs; see [Figures](#figures). |

Structure deliberately mirrors [`../segmentation/`](../segmentation/README.md). The
shared machinery (config loading, optimizer registry, progress bars, provenance) is
currently duplicated across the two stages; factoring it into one place is the obvious
follow-up once Stage 2 stabilises, and is not done here to avoid churning
freshly-merged Stage 1 code.

## Run the probe

```bash
python -m pip install -e ".[dev]"          # from repo root, once
```

```bash
python -m src.classification.train_classification --config src/classification/configs/cut3r_trained_state.yaml
```

```bash
python -m src.classification.inference_classification --config src/classification/configs/cut3r_trained_state.yaml --split val
```

There is one config per backbone **and arm** — `<backbone>_{image,state}.yaml`, eight in
all. The arm is the open ADR 0003 comparison, so a run states it rather than inheriting a
default, and both arms are reported.

Outputs land in `<output.dir>/<features.source>/<experiment>/`, holding `metrics.json`,
`head.pt`, and `inference-<split>.json`. The feature source is part of the path rather
than something to remember, so the two arms of the comparison cannot overwrite each
other and a directory always says which representation produced it.

That tree is git-ignored: it is working output, not a record. Promote a result worth
keeping to `docs/experiments/`.

`head.pt` records the head **and** the feature source it was trained on, so inference
refuses to evaluate it against a different representation — otherwise a config edit
could silently score state-token weights on pooled grid tokens.

## Figures

```bash
python -m src.classification.visualizations
```

Discovers every run under `experiments/<features.source>/<experiment>/` and writes to
`experiments/figures/`: curves per arm and merged, a summary across runs, and — per run —
a confusion matrix, per-category bars, and the top confusions.

Three conventions worth knowing before reading one:

- **Colour identifies the backbone**, never a split or a metric; line style carries
  train-vs-val (or which arm, in the merged figure).
- **Chance is drawn on the bar charts** as `1 / categories present` — 0.125 at eight
  categories, 0.02 across all 51 — because an accuracy means nothing without it.
- **Synthetic runs are stamped** in the corner and the title, from the cache's own
  metadata.

Two things the figures deliberately do not claim: a single run cannot support "A beats B"
(that needs seeds or a paired test over `per_window`), and confidence is not plotted
because `inference-<split>.json` does not record it yet.

One naming caveat worth repeating: the per-category bars plot **recall**, not accuracy.
Per-category accuracy in a multiclass setting would count true negatives too, which sits
near 1.0 for every category and says nothing.

## Test report

`build_test_report.py` is the analysis stage. It reads only
`inference-<split>-probabilities.csv` files, so it needs no cache, no weights, and no
GPU — and it is the reason the reported results can be re-derived from what is committed:

```bash
python -m src.classification.build_test_report   --predictions-dir reports/classification/predictions   --output-dir reports/classification   --seeds reports/classification/bootstrap-seeds.json   --epochs reports/classification/selected-epochs.json
```

Two design points. Every interval resamples **sequences**, because four windows of one
object are one observation, not four. And every difference reuses the same cluster draw
for both models, so the within-sequence error correlation survives into the paired
interval; the unpaired version is written out beside it as a sensitivity check, never as
the headline.

## Smoke test without real embeddings

`scripts/make_synthetic_probe_cache.py` writes a cache of **fake** embeddings and labels
in the real format. **No number from such a run means anything about the research
question** — the cache stamps `synthetic: true` into its `metadata.json`, which
propagates into `metrics.json` and the inference JSON.

The fixture gives each category its own direction (and puts it in the state latent as
well as the grid tokens), so one cache exercises both stages and both feature sources.
Build three caches under a local, git-ignored directory, from the repo root:

```bash
python -m scripts.make_synthetic_probe_cache --cache-dir src/classification/dummy_embeddings/probe/cut3r-trained --layout trajectory --grids "8x10,6x8" --categories 8 --seed 1
```

```bash
python -m scripts.make_synthetic_probe_cache --cache-dir src/classification/dummy_embeddings/probe/cut3r-random --grids "8x10,6x8" --noise 3.0 --category-signal 0.4 --categories 8 --seed 2
```

```bash
python -m scripts.make_synthetic_probe_cache --cache-dir src/classification/dummy_embeddings/probe/dinov2-vitb14 --grids "10x13,8x11" --noise 1.2 --category-signal 1.3 --categories 8 --seed 3
```

They are named as the configs expect, so point the cache root at that directory and the
real configs run unchanged:

```bash
CUT3R_CACHE_ROOT=src/classification/dummy_embeddings python -m src.classification.train_classification --config src/classification/configs/cut3r_trained_state.yaml
```

`--category-signal` sets how separable the categories are, so the three caches produce
deliberately different scores — useful for checking a comparison chart reacts, and
meaningless as a result. Delete the run directories before switching to real caches, so
a synthetic `metrics.json` never sits under the name a real run will reuse.
