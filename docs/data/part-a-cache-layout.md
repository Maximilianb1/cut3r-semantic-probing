# Part-A cache layout

Part-A exists as three storage partitions per backbone. They are partitions, not
different splits: the official `train`, `val`, and `test` assignment stays a
sequence-level field inside every cache index.

Use one backbone's three folders together. Never combine CUT3R-trained,
CUT3R-random, DINOv2, or RGB-patch tensors in the same feature arm.

## The three partitions

| Partition | Contents | Counts |
|---|---|---:|
| original (`part-a - only last frame`) | The capped Part-A cache, up to 30/5/5 train/val/test sequences per category. | 3,667 windows: 3,054 train, 512 val, 101 test |
| `part-a-leftover-*` | Windows built from frames the original cache did not use, drawn from the *same* sequences — not new sequences. | 124 windows: 74 train, 47 val, 3 test |
| `part-a-cap100-new-train` | The train-cap expansion from 30 to 100 sequences per category. Train rows only, zero sequence overlap with the original. | 1,762 sequences, 6,901 train windows |

All three use the same six-frame window protocol and the same Part-A manifest
family. The `cap100-new-train` manifest records zero sequence overlap with the
original Part-A manifest and was validated against the materialized RGB/mask
files.

Per its manifest `summary.json`, `cap100-new-train` selected 1,762 sequences and
42,288 frames (~24.0 frames per sequence), giving 6,901 windows at 4 windows per
sequence.

## Combining them safely

For expanded training, take the union of the `train` rows from the original
cache, the matching leftover cache, and the matching `cap100-new-train` cache.

Keep validation and test on the original cache alone. The leftover cache carries
extra val/test rows, and folding those in would define a different evaluation
set — the scores would no longer be comparable to the 30-cap runs.

The `*_expanded_mlp.yaml` and `*_expanded_linear.yaml` configs do exactly this
through `probe_cache.train_dirs`; see `CombinedProbeCacheDataset` in
`src/segmentation/dataset_segmentation.py` and `build_datasets` in
`src/segmentation/train_segmentation.py`.

## Layouts differ between partitions

Only the original cache uses `layout: trajectory` (context frames 1-5 plus the
target). Both `part-a-leftover-*` and `part-a-cap100-new-train` use
`layout: target_only`: the target frame's `image_tokens` and `state_tokens`, not
the six-timestep trajectory. Anything that needs the context frames can draw on
the original cache only.

## Schema

Read every cache through its own `index.parquet`, and check its `metadata.json`
first. Probe training expects `probe_cache_schema_version: probe-features-v2`,
with labels, categories, sequence IDs, and split fields. All nine caches
(three partitions x three backbones) report that schema, so no conversion is
needed before training. Each `cap100-new-train` trio shares one
`manifest_sha256`, and each `leftover` trio shares another.

Never merge caches by copying folders together: shard names like
`shard-00000.safetensors` are only unique within one cache directory and will
collide. Either read each cache through its own index, or build a merged cache
with globally unique shard names and combined provenance.

## Local layout

Fetch the caches from the published Drive folder (see
[REPRODUCING.md](../REPRODUCING.md)) into `${CUT3R_CACHE_ROOT}/probe/`, one
sibling directory per partition:

```text
${CUT3R_CACHE_ROOT}/probe/
  cut3r-trained/
  cut3r-trained-leftover/
  cut3r-trained-cap100-new-train/
  cut3r-random/
  cut3r-random-leftover/
  cut3r-random-cap100-new-train/
  dinov2-vitb14/
  dinov2-vitb14-leftover/
  dinov2-vitb14-cap100-new-train/
```

Manifests go under `${CUT3R_ARTIFACT_ROOT}/manifests/` as
`full51-part-a-cap100-new-train-v1/` and `part-a-leftover-windows-v1/` — one
copy each, shared by all three backbones rather than duplicated per backbone.
