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
the data pipeline's job and is documented there —
[`data_pipeline/README.md`](../data_pipeline/README.md#probe-embedding-extraction).

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
| `train_segmentation.py` | Config-driven training loop; foreground IoU + token accuracy; asserts sequence-disjoint splits; saves `head.pt`. |
| `inference_segmentation.py` | Reloads `head.pt` and evaluates a chosen split (default `test`); optional per-window masks. |
| `configs/*.yaml` | One config per compared backbone: `cut3r_trained`, `cut3r_random`, `dinov2`. Identical heads; only the cache differs. |

## Configs

A config here holds only what training and evaluation read: which cache to read
(`probe_cache.dir`), the head (`model`), the optimization (`training`), the split
names (`splits`), and where to write results (`output`). The extraction-side
settings (backbone weights, CO3D manifests, mask threshold) live with the script
that uses them, in
[`data_pipeline/configs/probe_features/`](../data_pipeline/configs/probe_features/) —
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
cd segmentation_validation && python train_segmentation.py --config configs/cut3r_trained.yaml
```

```bash
cd segmentation_validation && python inference_segmentation.py --config configs/cut3r_trained.yaml --split test
```

`metrics.json` records the cache's own `metadata.json` alongside the results, so
a number is always traceable to the exact cache (and backbone provenance) it came
from.

## Open decisions

The random-CUT3R baseline meaning, the DINOv2 variant/dependency, and the
cross-backbone comparison protocol are unresolved and should be ratified by ADR
before results are reported. See
[../data_pipeline/src/backbones/README.md](../data_pipeline/src/backbones/README.md).

Still-open engineering choices in the training code (see `train_segmentation.py`):
best-vs-final-epoch checkpoint selection, class-imbalance handling (`pos_weight`),
and the empty-target IoU convention (a foreground-free window currently scores 1.0).
