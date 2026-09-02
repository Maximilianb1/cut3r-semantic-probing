# Segmentation Validation

**Stage 1 — binary, class-agnostic segmentation, scaled to all 51 CO3D categories.**

## Objective

The earlier proof of concept showed that a lightweight probe on frozen CUT3R
`image_tokens` can separate foreground object from background — but only on a
single CO3D category. This workspace re-runs that **binary** (foreground vs.
background) probe across the full set of **51 CO3D categories** to test whether
the ability is genuinely class-agnostic and general, rather than a one-category
fluke.

The task stays binary — the per-token label is just `0/1`. Categories are used
only to (a) draw data from every object type and (b) break the IoU down
per-category in the metrics; the probe never predicts a category (that is
Stage 2, classification).

Three frozen backbones are compared with the **same** head: CUT3R-trained,
CUT3R-random, and DINOv2.

## Scope

This workspace **trains and evaluates the probe head, and nothing else**. No
backbone is ever loaded or run here: embeddings and labels are an *input*, read
from a probe-feature cache that already exists on disk. Producing those caches is
the data pipeline's job — see
[`configs/probe_features/`](../../configs/probe_features/) and
[`scripts/extract_probe_features.py`](../../scripts/extract_probe_features.py).

## How it works

1. **Input** — a probe-feature cache directory (`src.backbones.probe_cache`
   format), one per backbone. Each entry holds one window's target-frame tokens,
   the pooled binary mask label, and the manifest's sequence-level split.
2. **`dataset_segmentation.py`** reads that cache, filters to one split, and
   collates windows of different token-grid sizes into one flat `[ΣN, D]` tensor
   plus a `counts` vector (so per-image metrics can regroup). Per window it reads
   only the two tensors the probe consumes — the target frame's grid tokens and
   the mask — slicing them out inside the shard read rather than loading the whole
   entry (~12x fewer bytes on the CUT3R-trained trajectory cache).
3. **`train_segmentation.py`** trains only the MLP head with a per-token
   `BCEWithLogitsLoss`, evaluating on the val split each epoch and saving
   `head.pt` + `metrics.json`.
4. **`inference_segmentation.py`** reloads `head.pt` and evaluates on a held-out
   split, with optional per-window predicted masks.

Metrics are foreground **IoU** (macro over windows, micro over tokens, and
per-category) plus token accuracy — all at **token / patch-grid resolution**
(the mask was pooled to the backbone's token grid), not full pixel resolution.

## Files

| File | Purpose |
|---|---|
| `model_segmentation.py` | `SegmentationProbe` = trainable per-token MLP head over cached features (`hidden_dims=[]` gives a true linear probe). |
| `dataset_segmentation.py` | `ProbeCacheDataset` over the probe-feature cache (target-frame tokens + mask only) + collation for variable-size token grids, keeping per-window grouping. |
| `train_segmentation.py` | Config-driven training loop; foreground IoU + token accuracy; asserts sequence-disjoint splits. `training.checkpoint_selection` (or `--checkpoint-selection`) picks what `head.pt` holds: `last` (default) is the final epoch's head; `best_val` tracks validation macro-IoU across training and also keeps the final epoch as `head-last.pt`. See [EXP-005](../../docs/experiments/EXP-005-part-a-seg-cut3r-unblocked.md). |
| `inference_segmentation.py` | Reloads `head.pt` and evaluates a chosen split (default `test`); optional per-window masks. |
| `configs/*.yaml` | One config per `<backbone>_<data>_<capacity>.yaml`: backbone (`cut3r_trained`, `cut3r_random`, `dinov2`) x data scale (`partial` = original 3,054-window train set, `expanded` = +leftover +cap100-new-train, ~10k windows) x head capacity (`mlp` = `[512]` hidden layer, `linear` = `hidden_dims: []`, true linear probe). Only `probe_cache`/`model.hidden_dims`/`output.dir` differ between them. |
| `analysis/` | Post-hoc scripts that turn already-computed `metrics.json`/`inference-<split>.json` into plots and reports — never re-train or re-run inference. See below. |

### `analysis/`

| File | Purpose |
|---|---|
| `figures.py` | Shared `plot_*` figure-rendering functions (image grids, bar charts, training curves) called by the scripts below. |
| `build_qualitative_plots.py` | Worst-5/best-5 test windows by IoU from an `inference-<split>.json`, fetches just their real CO3D photos (via `scripts/download_co3d_targeted.py`), and renders `figures.plot_segmentation_results` grids per backbone. |
| `build_delta_comparison_plot.py` | Paired two-backbone grid on the windows where their per-window IoU differs the most (via `figures.plot_backbone_comparison_grid`). |
| `build_curves_and_iou.py` | Per-backbone training-curve and per-category IoU bar plots (via `figures.plot_training_curves`/`plot_per_category_iou`). |
| `build_score_comparison.py` | Compares backbones on already-computed scores: a bootstrap 95% CI on macro-IoU and a paired per-window significance test, plus deterministic macro/micro-IoU and precision/recall comparisons (three figures; plots directly, not via `figures.py`). |
| `build_category_representation_check.py` | Correlates per-category training-window counts against per-category test IoU, to check whether representation (not just visual difficulty) drives per-category performance (plots directly, not via `figures.py`). |
| `build_probe_capacity_comparison.py` | Compares the `mlp` (`[512]`) vs. `linear` (`[]`) head on identical data per backbone: paired per-window bootstrap CI on the delta, a combined bar figure, and a per-category MLP-vs-linear scatter (one panel per backbone) to check whether the gap is spread evenly across categories or concentrated in a few (plots directly, not via `figures.py`). |

## Configs

A config here holds only what training and evaluation read: which cache to read
(`probe_cache.dir`), the head (`model`), the optimization (`training`), the split
names (`splits`), and where to write results (`output`). The extraction-side
settings (backbone weights, CO3D manifests, mask threshold) live with the script
that uses them, in
[`configs/probe_features/`](../../configs/probe_features/) —
the two files per backbone must agree on `probe_cache.dir`.

`probe_cache.dir` is written as `${CUT3R_CACHE_ROOT}/probe/<backbone>` rather
than a machine-specific path, so export `CUT3R_CACHE_ROOT` before running; an
unset variable fails the run instead of silently resolving to a wrong directory.
No other environment variable is needed — the CO3D files, the CUT3R checkpoint,
and the manifests are not touched by the probe.

## Run the probe

```bash
python -m pip install -e ".[dev]"          # from repo root, once
```

```bash
python -m src.segmentation.train_segmentation --config src/segmentation/configs/cut3r_trained_partial_mlp.yaml
```

```bash
python -m src.segmentation.inference_segmentation --config src/segmentation/configs/cut3r_trained_partial_mlp.yaml --split test
```

To select the best-validation checkpoint instead of the final epoch (see
[EXP-005](../../docs/experiments/EXP-005-part-a-seg-cut3r-unblocked.md)):

```bash
python -m src.segmentation.train_segmentation --config src/segmentation/configs/cut3r_trained_partial_mlp.yaml \
  --checkpoint-selection best_val --output-dir src/segmentation/experiments/segmentation-cut3r-trained-bestval
python -m src.segmentation.inference_segmentation --config src/segmentation/configs/cut3r_trained_partial_mlp.yaml \
  --checkpoint src/segmentation/experiments/segmentation-cut3r-trained-bestval/head.pt --split test \
  --save-dir src/segmentation/experiments/segmentation-cut3r-trained-bestval --save-masks
```

Swap in any other `<backbone>_<data>_<capacity>.yaml` for the same two commands
to run a different backbone, data scale, or head capacity.

Outputs land in `output.dir` — `src/segmentation/experiments/<experiment>/`, holding
`metrics.json`, `head.pt`, and (from inference) `inference-<split>.json` plus
`masks-<split>.pt` with `--save-masks`. That directory is git-ignored: it is working
output, not a record. Promote a result worth keeping to `docs/experiments/`.

`metrics.json` records the cache's own `metadata.json` alongside the results, so
a number is always traceable to the exact cache (and backbone provenance) it came
from.

## Smoke test without real embeddings

`scripts/make_synthetic_probe_cache.py` writes a cache of **fake** embeddings and
masks in the real format, so the pipeline can be run end to end without CO3D, CUT3R
weights, or a GPU. **No number from such a run means anything about the research
question** — the cache stamps `synthetic: true` into its `metadata.json`, which
propagates into `metrics.json`.

Build the three caches under a local, git-ignored directory (from the repo root):

```bash
python -m scripts.make_synthetic_probe_cache --cache-dir src/segmentation/dummy_embeddings/probe/cut3r-trained --layout trajectory --grids "8x10,6x8" --seed 1
```

```bash
python -m scripts.make_synthetic_probe_cache --cache-dir src/segmentation/dummy_embeddings/probe/cut3r-random --grids "8x10,6x8" --noise 3.0 --seed 2
```

```bash
python -m scripts.make_synthetic_probe_cache --cache-dir src/segmentation/dummy_embeddings/probe/dinov2-vitb14 --grids "10x13,8x11" --noise 1.2 --seed 3
```

They are named as the configs expect, so point the cache root at that directory and
the **real configs run unchanged** — no config edit, nothing to remember to revert:

```bash
CUT3R_CACHE_ROOT=src/segmentation/dummy_embeddings python -m src.segmentation.train_segmentation --config src/segmentation/configs/cut3r_trained_partial_mlp.yaml
```

Only the `partial` configs are smoke-testable this way -- `make_synthetic_probe_cache.py`
writes one cache dir, and `expanded` configs read three via `probe_cache.train_dirs`
(see `tests/test_segmentation_dataset.py` for that coverage instead).

The fixture is deliberately noisy rather than separable, so a linear probe lands
between chance and perfect and the IoU code is exercised on real values. Delete the
run directories before switching to real caches, so a synthetic `metrics.json` never
sits under the name a real run will reuse.

## Results and caveats

Reported results are [EXP-006](../../docs/experiments/EXP-006-part-a-seg-expanded-training.md)
(expanded training, MLP head) and
[EXP-007](../../docs/experiments/EXP-007-part-a-seg-probe-capacity-ablation.md)
(the same runs with a linear head). Figures and metrics are in
[reports/segmentation](../../reports/segmentation/README.md).

What the reported runs settled, and what they did not:

- `checkpoint_selection: best_val` is what every reported run uses, so a result is
  never whichever epoch training happened to stop on.
- The CUT3R-random cache uses `layout: target_only` while CUT3R-trained uses
  `layout: trajectory`. The control therefore isolates weights *and* layout, not
  weights alone — noted as a limitation rather than fixed, because it would have
  to close a ~0.5 IoU gap to matter.
- Class imbalance is untouched: no `pos_weight`, no resampling. The same head and
  the same loss are used for all three backbones, which is what makes them
  comparable.
- A foreground-free window scores IoU 1.0 under the current empty/empty
  convention. Such windows are rare in CO3D's object-centric sequences, but the
  convention is a choice, not a fact.

See [../backbones/README.md](../backbones/README.md) for what the random-weight
baseline does and does not establish.
