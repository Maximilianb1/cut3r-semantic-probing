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

## How it works

1. **Embeddings + labels** are precomputed once per backbone into a probe-feature
   cache (`src.backbones.probe_cache`) by
   [`scripts/extract_probe_features.py`](../data_pipeline/scripts/extract_probe_features.py).
   Each entry holds the target-frame tokens **and** the pooled binary mask label.
2. **`segmentation_dataset.py`** reads that cache, filters to one split, and
   collates windows of different token-grid sizes into one flat `[ΣN, D]` tensor
   plus a `counts` vector (so per-image metrics can regroup).
3. **`train_segmentation.py`** trains only the MLP head with a per-token
   `BCEWithLogitsLoss` (the backbone is frozen and already cached), evaluating on
   the val split each epoch and saving `head.pt` + `metrics.json`.
4. **`segmentation_inference.py`** reloads `head.pt` and evaluates on a held-out
   split, with optional per-window predicted masks.

Metrics are foreground **IoU** (macro over windows, micro over tokens, and
per-category) plus token accuracy — all at **token / patch-grid resolution**
(the mask was pooled to the backbone's token grid), not full pixel resolution.

## Relationship to the data pipeline

This workspace consumes — and never modifies — the frozen Stage 0 artifacts in
[`../data_pipeline/`](../data_pipeline/README.md): the CO3Dv2 manifests
(category, sequence-level split, RGB/mask paths) and, for CUT3R-trained, the
existing embedding cache. Everything is installed as the shared package:

```python
from src.backbones import build_backbone
from src.backbones.probe_cache import load_probe_index, load_embedding_sample
```

## Files

| File | Purpose |
|---|---|
| `model_segmentation.py` | `SegmentationProbe` = frozen backbone (optional) + trainable per-token MLP head (`hidden_dims=[]` gives a true linear probe). |
| `segmentation_dataset.py` | `ProbeCacheDataset` over the probe-feature cache + collation for variable-size token grids, keeping per-window grouping. |
| `train_segmentation.py` | Config-driven training loop; foreground IoU + token accuracy; asserts sequence-disjoint splits; saves `head.pt`. |
| `segmentation_inference.py` | Reloads `head.pt` and evaluates on a chosen split (default `test`); optional per-window masks. |
| `configs/*.json` | One config per compared backbone: `cut3r_trained`, `cut3r_random`, `dinov2`. Identical heads; only the backbone and cache differ. |
| `tests/` | Synthetic-fixture tests for the model, dataset, and a training smoke. |

## Building the feature caches

Run once per backbone (produces the cache the training reads):

```bash
python -m scripts.extract_probe_features --config segmentation_validation/configs/<backbone>.json
```

The configs reference `${ENV}` paths (no hardcoded locations). Export the Stage 0
variables before running:

- **DINOv2:** `CO3D_ROOT`, `CUT3R_ARTIFACT_ROOT`, `CUT3R_CACHE_ROOT`
- **random-CUT3R:** those + `CUT3R_ROOT` (the checkpoint resolves to CUT3R's
  default `${CUT3R_ROOT}/src/cut3r_512_dpt_4_64.pth`)
- **CUT3R-trained:** reuses the existing Stage 0 cache — no GPU, no re-extraction

Full-51 has two manifest parts; run each config once per part into the **same**
cache dir, using `--manifest-dir "$CUT3R_ARTIFACT_ROOT/manifests/full51-part-b-v1"`
for part B.

## Run the probe

```bash
python -m pip install -e ".[dev]"          # from repo root, once
cd segmentation_validation
python train_segmentation.py --config configs/cut3r_trained.json
python segmentation_inference.py --config configs/cut3r_trained.json --split test
```

## Open decisions

The random-CUT3R baseline meaning, the DINOv2 variant/dependency, and the
cross-backbone comparison protocol are unresolved and should be ratified by ADR
before results are reported. See
[../data_pipeline/src/backbones/README.md](../data_pipeline/src/backbones/README.md).

Still-open engineering choices in the training code (see `train_segmentation.py`):
best-vs-final-epoch checkpoint selection, class-imbalance handling (`pos_weight`),
and the empty-target IoU convention (a foreground-free window currently scores 1.0).
