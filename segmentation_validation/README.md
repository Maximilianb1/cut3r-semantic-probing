# Segmentation Validation

**Stage 1 — binary segmentation, scaled to all categories.**

## Objective

The earlier proof of concept showed that a lightweight probe on frozen CUT3R
`image_tokens` can separate foreground object from background, but it was only
checked on a single CO3D category. This workspace extends and validates that
result across the full set of 51 CO3D categories using the already-extracted,
verified Stage 0 feature cache, to see whether the segmentation ability holds on
a larger and more diverse class set.

## Relationship to the data pipeline

This workspace consumes — and never modifies — the frozen Stage 0 artifacts in
[`../data_pipeline/`](../data_pipeline/README.md):

- the two immutable cache roots (`full51-part-a-v1`, `full51-part-b-v1`), loaded
  as a logical union, providing per-window `image_tokens` and `state_tokens`;
- the CO3Dv2 manifests for category, official train/validation/test split, and
  RGB/mask lookup.

Install the project once from the repository root (`python -m pip install -e
".[dev]"`), then import the pipeline as the shared package, for example:

```python
from src.embeddings.cache import load_trajectory, verify_cache
from src.common.tables import read_parquet
```

## Infrastructure

| File | Purpose |
|---|---|
| `model_segmentation.py` | `SegmentationProbe` = frozen backbone (from `src.backbones`) + trainable per-token MLP head (`hidden_dims=[]` gives a true linear probe). |
| `segmentation_dataset.py` | `ProbeCacheDataset` over the probe-feature cache + collation that concatenates variable-size token grids and keeps per-window grouping. |
| `train_segmentation.py` | Config-driven training loop; token accuracy + foreground IoU (macro/micro/per-category); asserts sequence-disjoint splits; saves the trained head to `head.pt`. |
| `segmentation_inference.py` | Reloads `head.pt` and evaluates the probe on a chosen split (default `test`); optional per-window masks. |
| `configs/*.json` | One config per compared backbone: `cut3r_trained`, `cut3r_random`, `dinov2`. Identical heads; only the backbone and cache differ. |
| `tests/` | Synthetic-fixture tests for the model, dataset, and a training smoke. |

### Feature caches

Probe training reads a **probe-feature cache** (`src.backbones.probe_cache`),
one per backbone, holding target-frame spatial/global tokens and the pooled
binary mask target. Build them once with:

- `extract_to_cache(backbone, layout=...)` — live backbone (DINOv2, random-CUT3R,
  target-only layout).
- `attach_labels_from_trajectory_cache(...)` — CUT3R-trained, reusing the existing
  Stage 0 trajectory cache (no GPU), verifying each mask SHA-256 against the cache.

Build them with `python -m scripts.extract_probe_features --config <one config>`.

Fill the `REPLACE_WITH_*` paths in the configs before running.

### Run

```bash
python -m pip install -e ".[dev]"          # from repo root, once
cd segmentation_validation
python train_segmentation.py --config configs/cut3r_trained.json
python segmentation_inference.py --config configs/cut3r_trained.json --split test
```

## Open decisions

The random-CUT3R baseline meaning, the DINOv2 variant/dependency, and the
cross-backbone comparison protocol are unresolved and should be ratified by ADR
before results are reported. See [../data_pipeline/src/backbones/README.md](../data_pipeline/src/backbones/README.md).
