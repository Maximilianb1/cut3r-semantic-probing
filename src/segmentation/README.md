# Segmentation Validation

**Stage 1 — binary, class-agnostic segmentation, scaled to all 51 CO3D categories.**

## Objective

A lightweight probe on frozen CUT3R `image_tokens` can separate foreground
object from background — but an earlier proof of concept only showed this on
a single CO3D category. This workspace re-runs that **binary**
(foreground-vs-background) probe across all **51 CO3D categories** to check
whether the ability is genuinely general, not a one-category fluke.

The task stays binary — each token's label is just `0/1`. Categories are used
only to (a) draw data from every object type and (b) break the IoU down
per-category in the metrics; the probe never predicts *which* category (that
is Stage 2).

Three frozen backbones share the **same** head: CUT3R-trained, CUT3R-random
(same architecture, untrained — the control), and DINOv2.

## Scope

This workspace **trains and evaluates the probe head only** — no backbone is
ever loaded here. Embeddings and labels are read from a probe-feature cache
that already exists on disk; producing that cache is the data pipeline's job
(see [`configs/probe_features/`](../../configs/probe_features/) and
[`scripts/extract_probe_features.py`](../../scripts/extract_probe_features.py)).

## How it works

1. **Input** — a probe-feature cache (`src.backbones.probe_cache` format), one
   per backbone. Each entry holds one window's target-frame tokens, the
   pooled binary mask, and the manifest's sequence-level split.
2. **`dataset_segmentation.py`** reads the cache, filters to one split, and
   collates windows of different token-grid sizes into one flat `[ΣN, D]`
   tensor plus a `counts` vector for regrouping into per-window metrics. It
   reads only the two tensors the probe needs, sliced straight out of the
   shard rather than loading the whole entry.
3. **`train_segmentation.py`** first computes per-feature mean/std from the
   **train split only** (streamed batch by batch, since a train split is
   millions of tokens), then trains the MLP head on standardized features
   with a per-token `BCEWithLogitsLoss`. Those statistics are saved inside
   `head.pt` alongside the weights, so evaluation always standardizes with
   the exact numbers training used — never statistics recomputed from
   whatever split is being scored. Training evaluates on the val split each
   epoch and writes `head.pt` + `metrics.json`.
4. **`inference_segmentation.py`** reloads `head.pt` (weights + standardization
   statistics) and evaluates a held-out split, with optional per-window
   predicted masks.

## Metrics

All metrics are computed at **token / patch-grid resolution** (the mask is
pooled to the backbone's token grid), not full pixel resolution:

- **IoU** — foreground and background, macro (per-window average) and micro
  (pooled over tokens), plus a per-category breakdown.
- **mIoU** (`mean_iou`) — the 2-class macro average.
- **mAcc** (`mean_class_accuracy`) — mean of foreground/background recall.
  Preferred over plain `token_accuracy`, which is dominated by the
  always-large background class.
- Precision, recall, and the raw confusion counts (`tp`/`fp`/`fn`/`tn`).
- **AUC-ROC** (`auroc`) and **AUC-PR** (`auprc`) — computed in
  `curve_metrics.py`, no scikit-learn needed. Every metric above depends on a
  fixed `logit > 0` decision threshold, used identically across backbones by
  design (no per-backbone tuning); AUC drops that threshold and asks only
  whether foreground/background scores are separable at all — it isolates
  "is the information in the embedding" from "is 0 the right cutoff for this
  backbone's raw score scale."

## Files

| File | Purpose |
|---|---|
| `model_segmentation.py` | `SegmentationProbe` — trainable per-token MLP head over cached features (`hidden_dims=[]` gives a true linear probe). Also holds and applies the standardization statistics. |
| `dataset_segmentation.py` | `ProbeCacheDataset` over the probe-feature cache (target-frame tokens + mask only), with collation for variable-size token grids. |
| `train_segmentation.py` | Config-driven training loop: computes train-only standardization, trains the head, tracks foreground IoU + token accuracy, asserts sequence-disjoint splits. `--checkpoint-selection` picks what `head.pt` holds: `last` (default) is the final epoch; `best_val` tracks validation macro-IoU and also keeps the final epoch as `head-last.pt`. |
| `inference_segmentation.py` | Reloads `head.pt` and evaluates a chosen split (default `test`); optional per-window masks via `--save-masks`. Each `per_window_iou` row also carries that window's `tp`/`fp`/`fn`/`tn`, so precision/recall can be bootstrapped from the committed inference file without re-running inference. |
| `bootstrap_iou.py` | `SequenceIoUClusters` (sequence-cluster sufficient statistics, with per-sequence category) plus `bootstrap_iou_ci`/`bootstrap_iou_difference` — the sequence-cluster bootstrap, mirroring `src/classification/bootstrap_accuracy.py`. Also holds `derive_seed`/`summarize`, the small helpers shared with `bootstrap_precision_recall.py`. |
| `bootstrap_precision_recall.py` | `SequencePrecisionRecallClusters` (per-sequence pooled `tp`/`fp`/`fn`) plus `bootstrap_precision_recall_ci`/`bootstrap_precision_recall_difference` — same sequence-cluster resampling as `bootstrap_iou.py`, but on pooled counts (ratio-of-sums), since precision/recall are ratios and per-window ratios get unstable on windows with few foreground tokens. Mirrors `_ratio_distribution` in `src/classification/bootstrap_accuracy.py`. |
| `build_test_report.py` | Consolidated held-out test report, mirroring `src/classification/build_test_report.py`: bootstrap CI and paired/unpaired differences (IoU and precision/recall) for every meaningful run pair, a per-category IoU CI breakdown, and a window win/loss/tie tally — all from the committed `inference-<split>.json` files, never re-running inference. Writes `reports/segmentation/comparison/{bootstrap-iou-ci.csv, bootstrap-iou-per-category.csv, bootstrap-iou-differences.csv, bootstrap-precision-recall-ci.csv, bootstrap-precision-recall-differences.csv, test-bootstrap-report.json}`. |
| `configs/*.yaml` | One config per `<backbone>_<capacity>.yaml`: backbone (`cut3r_trained`, `cut3r_random`, `dinov2`) × head capacity (`mlp` = one 512-unit hidden layer, `linear` = `hidden_dims: []`). All six read the same pooled cache via `probe_cache.cache_dirs`, relabeled by the shared `split_override_path`. Only `probe_cache`/`model.hidden_dims`/`output.dir` differ between them. |
| `analysis/` | Post-hoc scripts that turn already-computed `metrics.json`/`inference-<split>.json` into plots — never re-train or re-run inference. See below. |

### `analysis/`

The four figures actually cited in [reports/segmentation](../../reports/segmentation/README.md)
are built from committed `reports/segmentation/<backbone>-<probe>/` data by:

| File | Purpose |
|---|---|
| `build_headline_gap.py` | Linear-probe-only macro-IoU bootstrap CI across the three backbones — the random-vs-trained gap. |
| `build_capacity_slope_and_pr.py` | Linear-vs-MLP macro-IoU bars (bootstrap CI) plus a shared precision/recall shift plot, showing the mechanism behind the MLP gap (false-positive cleanup vs. new true positives). |
| `build_category_difficulty_heatmap.py` | Per-category test-IoU heatmap across all 6 runs, with a manual Spearman rank correlation (no scipy) to check whether category difficulty is shared across backbones. |
| `build_learning_curve_panels.py` | Val macro-IoU vs. epoch, split into linear/MLP panels, marking each run's `best_val_epoch`. |

Two more scripts still work (against the gitignored `experiments/` working
directory, using `masks-<split>.pt` from `inference_segmentation.py
--save-masks`) but aren't part of the current committed report, which skips
`--save-masks`:

| File | Purpose |
|---|---|
| `build_qualitative_plots.py` | Worst-5/best-5 test windows by IoU, as image grids per backbone. |
| `build_delta_comparison_plot.py` | Paired two-backbone grid on the windows where per-window IoU differs most. |

`figures.py` (shared plotting helpers) and `runs.py` (shared run-loading /
display-name helpers) back all of the above and aren't run directly.

## Configs

A config here holds only what training and evaluation read: which caches to
read (`probe_cache.cache_dirs`, plus `split_override_path`), the head
(`model`), the optimization (`training`), the split names (`splits`), and
where to write results (`output`). Extraction-side settings (backbone
weights, CO3D manifests, mask threshold) live with the extraction script
instead, in [`configs/probe_features/`](../../configs/probe_features/) — its
configs and `cache_dirs` here must agree on where each cache lives.

Each entry in `cache_dirs` is written as `${CUT3R_CACHE_ROOT}/probe/<name>`
rather than a machine-specific path, so export `CUT3R_CACHE_ROOT` before
running; an unset variable fails the run instead of silently pointing
somewhere wrong. `split_override_path`
(`configs/segmentation_split_override.json`, built by
`scripts/derive_segmentation_split.py`) relabels every window from the
pooled caches into one shared, sequence-disjoint train/val/test split,
independent of whichever split each cache originally recorded — this is what
makes all six runs directly comparable.

## Run the probe

```bash
python -m pip install -e ".[dev]"          # from repo root, once
```

```bash
python -m src.segmentation.train_segmentation --config src/segmentation/configs/cut3r_trained_mlp.yaml
```

```bash
python -m src.segmentation.inference_segmentation --config src/segmentation/configs/cut3r_trained_mlp.yaml --split test
```

To select the best-validation checkpoint instead of the final epoch:

```bash
python -m src.segmentation.train_segmentation --config src/segmentation/configs/cut3r_trained_mlp.yaml \
  --checkpoint-selection best_val
python -m src.segmentation.inference_segmentation --config src/segmentation/configs/cut3r_trained_mlp.yaml \
  --checkpoint src/segmentation/experiments/segmentation-cut3r-trained-mlp/head.pt --split test \
  --save-dir src/segmentation/experiments/segmentation-cut3r-trained-mlp --save-masks
```

Swap in any other `<backbone>_<capacity>.yaml` for the same two commands to
run a different backbone or head capacity.

Outputs land in `output.dir` — `src/segmentation/experiments/<experiment>/`,
holding `metrics.json`, `head.pt`, and (from inference)
`inference-<split>.json` plus `masks-<split>.pt` with `--save-masks`. That
directory is git-ignored: it is working output, not a record. Promote a
result worth keeping to `reports/`.

`metrics.json` records the cache's own `metadata.json` alongside the
results, so a number is always traceable to the exact cache (and backbone
provenance) it came from.

## Smoke test without real embeddings

`scripts/make_synthetic_probe_cache.py` writes a cache of **fake** embeddings
and masks in the real format, so the pipeline can run end to end without
CO3D, CUT3R weights, or a GPU. **No number from such a run means anything
about the research question** — the cache stamps `synthetic: true` into its
`metadata.json`, which propagates into `metrics.json`.

Each backbone's config pools three cache directories keyed by real CO3D
sequence IDs, and `make_synthetic_probe_cache.py` has no matching override
file, so running a real config unchanged against synthetic caches isn't a
one-command dry run. `tests/test_segmentation_dataset.py` exercises that same
`cache_dirs` + `split_override_path` combination against in-memory fixtures —
start there for a fast, no-GPU check of the dataset-building code.

To smoke-test training/inference themselves, write one synthetic cache per
backbone and point a trimmed config at just that directory (dropping
`split_override_path`, since synthetic sequence IDs won't appear in the real
override file):

```bash
python -m scripts.make_synthetic_probe_cache --cache-dir src/segmentation/dummy_embeddings/probe/cut3r-trained --layout trajectory --grids "8x10,6x8" --seed 1
CUT3R_CACHE_ROOT=src/segmentation/dummy_embeddings python -m src.segmentation.train_segmentation --config <a config with cache_dirs: [${CUT3R_CACHE_ROOT}/probe/cut3r-trained] and no split_override_path>
```

The fixture is deliberately noisy rather than separable, so a linear probe
lands between chance and perfect. Delete these run directories before
switching to real caches, so a synthetic `metrics.json` never sits under a
name a real run will reuse.

## Results

Reported results cover all three backbones at both head capacities (linear
and MLP) — six runs in total, every one using `checkpoint_selection:
best_val` and the same head/loss/optimization settings, so the comparison
is never confounded by mismatched training choices. See
[reports/segmentation](../../reports/segmentation/README.md) for metrics,
per-window results, and findings.
