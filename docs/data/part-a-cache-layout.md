# Part-A Cache Layout

This folder contains storage partitions of the Part-A data. The partitions are
not different scientific splits. The official `train`, `val`, and `test`
assignments remain sequence-level fields in each cache index.

Use the matching three folders for one backbone only. Do not combine CUT3R-trained,
CUT3R-random, DINOv2, or RGB-patch tensors in the same feature arm.

## Folder Roles

| Folder suffix | Contents | Current counts |
|---|---|---:|
| `part-a - only last frame` | Original capped Part-A cache. Up to 30/5/5 train/val/test sequences per category. | 3,667 windows: 3,054 train, 512 val, 101 test |
| `part-a-leftover-*` | Windows made from frames unused by the original cache. These are from existing selected sequences, not new sequences. | 124 windows: 74 train, 47 val, 3 test |
| `part-a-cap100-new-train` | New sequence expansion after increasing the train cap from 30 to 100. Train only. It has zero overlap with the original sequence IDs. | 1,762 sequences and 6,901 train windows |

The new CUT3R-trained folder is currently:

```text
full51-cut3r-trained-part-a-cap100-new-train/
```

The analogous random-CUT3R and DINOv2 folders can use the same manifest and
folder naming convention when their new caches are extracted.

### Provenance

`part-a-leftover-*` was extracted with `cut3r_trained_extract.yaml`
(`layout: target_only`, `mask_threshold: 0.5`, `windows_per_shard: 32`, seed
`20260729`). This config is VM/Drive-only; it is not checked into this repo.
The exact rule for which leftover frames became which windows is not
documented anywhere yet, beyond "previously-unused frames of the same
sequences."

`part-a-cap100-new-train`, per its manifest `summary.json`: 1,762 selected
sequences, 42,288 selected frames (~24.0 frames/sequence), 6,901 windows (4
windows/sequence).

## Safe Use

For expanded training, use the union of the `train` rows from:

1. the original cache;
2. the matching leftover cache; and
3. the matching `cap100-new-train` cache.

Keep the original validation and test rows for the primary comparison with the
previous experiments. The leftover cache contains extra validation/test rows,
but adding them creates a separate expanded-evaluation protocol. Do not silently
mix them into the old headline metrics.

Only the original `part-a - only last frame` cache uses `layout: trajectory`
(context frames 1-5 plus the target). Both `part-a-leftover-*` and
`part-a-cap100-new-train` use `layout: target_only`: the target
`image_tokens` and target `state_tokens`, not the six-timestep trajectory.
Any experiment needing the context frames, not just the target frame, can
only draw on the original cache.

## Schema Rules

Inspect every `metadata.json` before training. Probe training expects the common
`probe-features-v2` format with `index.parquet`, labels, categories, sequence IDs,
and split fields. The historical CUT3R-trained original target cache may be a
`stage0-target-cache-v1` cache and must first be converted to a labelled
`probe-features-v2` cache with the project's target-label attachment utility.

Do not concatenate Safetensors files by copying folders together. Shard names
such as `shard-00000.safetensors` can collide. Read each cache through its own
`index.parquet`, or create a new merged cache with globally unique shard names
and the combined metadata/provenance.

All three partitions use the same six-frame window protocol and Part-A manifest
family. The new manifest records zero sequence overlap with the original
Part-A manifest and was validated against the materialized RGB/mask files.

## Known Gaps

- No per-category breakdown exists in any `summary.json`; a per-category window
  count requires reading `windows.parquet` / `sequences.parquet` directly.
- No statistical summary (mean/std/value ranges) of the embeddings themselves
  exists anywhere; that would need to be computed from the Safetensors tensors.
- Resolved: all six new caches (`cap100-new-train` and `leftover`, all three
  backbones) checked directly from their Drive `metadata.json` on 2026-08-21 —
  every one reports `probe_cache_schema_version: probe-features-v2` and
  `layout: target_only`. No schema conversion is needed before training
  against any of them. Each `cap100-new-train` trio shares one
  `manifest_sha256`; each `leftover` trio shares a different one (both
  confirmed identical across backbones).

## Local cache layout (VM)

Fetched from Drive (`stage0-full51-v1/caches/`) into
`${CUT3R_CACHE_ROOT}/probe/`, alongside the existing `<backbone>/` (original)
folders, one sibling folder per batch — no shard renaming, since shard names
are only unique within one cache directory:

```text
${CUT3R_CACHE_ROOT}/probe/
  cut3r-trained/                        # original baseline
  cut3r-trained-leftover/
  cut3r-trained-cap100-new-train/
  cut3r-random/
  cut3r-random-leftover/
  cut3r-random-cap100-new-train/
  dinov2-vitb14/
  dinov2-vitb14-leftover/
  dinov2-vitb14-cap100-new-train/
```

Manifests (`${CUT3R_ARTIFACT_ROOT}/manifests/`): `full51-part-a-cap100-new-train-v1/`
(shared by all three `cap100-new-train` caches) and `part-a-leftover-windows-v1/`
(shared by all three `leftover` caches) — copied once each, not per backbone.

For expanded training, `src/segmentation/configs/{cut3r_trained,cut3r_random,dinov2}_expanded.yaml`
point `probe_cache.train_dirs` at the three matching folders above (val/test
stay on the original folder alone) — see `src/segmentation/dataset_segmentation.py`'s
`CombinedProbeCacheDataset` and `train_segmentation.build_datasets`.
