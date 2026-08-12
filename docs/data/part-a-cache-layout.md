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

## Safe Use

For expanded training, use the union of the `train` rows from:

1. the original cache;
2. the matching leftover cache; and
3. the matching `cap100-new-train` cache.

Keep the original validation and test rows for the primary comparison with the
previous experiments. The leftover cache contains extra validation/test rows,
but adding them creates a separate expanded-evaluation protocol. Do not silently
mix them into the old headline metrics.

The new cache contains only target-frame features. For CUT3R this means the
target `image_tokens` and target `state_tokens`, not the six-timestep trajectory.

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
