# Reproducing the results

The pipeline has four stages. Each is more expensive than the last, and each can
be entered independently, because the stage before it wrote its output to disk:

```
CO3Dv2 download  ->  frozen-backbone extraction  ->  probe training  ->  analysis
   (~hours)              (~1 day, 1 GPU)              (~minutes, CPU)     (seconds)
```

Two shortcuts, in order of how little they cost:

- **Only checking our numbers?** Go straight to *Analysis only*. It runs from
  what is committed here — no dataset, no GPU, no model weights.
- **Want to retrain the probes?** Download the published caches below. That
  skips stages 1 and 2 entirely and leaves a few minutes of CPU training.

Stages 1 and 2 are documented for completeness, and are only needed to rebuild
the caches from raw CO3D.

---

## 0. Install

```bash
python -m pip install -e ".[dev]"
pytest
```

Python 3.11 or newer. Run every command from the repository root, because
relative `configs/...` paths are resolved from there.

On Windows, put the virtual environment outside any cloud-synced folder
(OneDrive, Dropbox); the sync client corrupts `site-packages` during installs.

## 4. Analysis only (no data required)

Regenerate every classification table and figure from the committed per-window
predictions:

```bash
python -m src.classification.build_test_report \
  --predictions-dir reports/classification/predictions \
  --output-dir reports/classification \
  --seeds reports/classification/bootstrap-seeds.json \
  --epochs reports/classification/selected-epochs.json
```

With `--seeds` this reproduces the committed tables bit for bit. See
[reports/classification/README.md](../reports/classification/README.md).

The segmentation analysis scripts under `src/segmentation/analysis/` work the
same way, but they read the per-run `inference-<split>.json` files, which are
working artifacts and are not committed (see *What is not committed* below).
Their outputs are in [reports/segmentation/](../reports/segmentation/README.md).

## Published caches (skip stages 1 and 2)

Both expensive stages are already done. The extracted caches are published here:

**https://drive.google.com/drive/folders/1cmWe0S6F4444F3yL-9Y_sR-nHNh5LAKe**

Download the folders you need, place them under `${CUT3R_CACHE_ROOT}/probe/`
using the local names in the table below, and go straight to stage 3.

### `caches/` — what each folder is

**Probe-feature caches.** These are what probe training reads: per window, one
backbone's target-frame embeddings plus both task labels (the segmentation mask
pooled to that backbone's token grid, and the category index). Three partitions
of Part-A exist for each of the three backbones; they are storage partitions,
not different splits — the official `train`/`val`/`test` assignment is a
sequence-level field inside every cache index.

| Drive folder | Place at `${CUT3R_CACHE_ROOT}/probe/…` | Contents |
|---|---|---|
| `full51-cut3r_trained-part-a - only last frame` | `cut3r-trained/` | Original Part-A cache, up to 30/5/5 sequences per category: 3,667 windows (3,054 train, 512 val, 101 test). |
| `full51-cut3r_random_weights-part-a - only last frame` | `cut3r-random/` | The same windows, CUT3R with untrained weights. |
| `dinov2-vitb14-part-a-baseline-1 - only last frame` | `dinov2-vitb14/` | The same windows, DINOv2 ViT-B/14. |
| `part-a-leftover-cut3r-trained` | `cut3r-trained-leftover/` | Extra windows built from frames the original cache did not use, from the *same* sequences: 124 windows (74 train, 47 val, 3 test). |
| `part-a-leftover-cut3r-random` | `cut3r-random-leftover/` | The same, CUT3R-random. |
| `part-a-leftover-dinov2` | `dinov2-vitb14-leftover/` | The same, DINOv2. |
| `full51-cut3r-trained-part-a-cap100-new-train` | `cut3r-trained-cap100-new-train/` | The train-cap expansion from 30 to 100 sequences per category: 1,762 new sequences, 6,901 train windows, zero sequence overlap with the original cache. Train rows only. |
| `full51-cut3r_random_weights-part-a-cap100-new-train` | `cut3r-random-cap100-new-train/` | The same, CUT3R-random. |
| `dinov2-vitb14-part-a-cap100-new-train` | `dinov2-vitb14-cap100-new-train/` | The same, DINOv2. |

The `*_partial_mlp.yaml` configs read the first group only. The `*_expanded_*`
configs union all three of a backbone's train rows via `probe_cache.train_dirs`,
while validation and test stay on the original cache alone so the scores remain
comparable. Never mix two backbones' tensors into one feature arm.

**Stage-0 target-feature caches.** The raw extraction output over all 51
categories, before task labels were attached. Only needed to re-derive a probe
cache or to audit the extraction; probe training does not read them.

| Drive folder | Contents |
|---|---|
| `full51-cut3r_trained-part-a- main data - full` | CUT3R-trained, the 26 Part-A categories. |
| `full51-cut3r_trained-part-b -full` | CUT3R-trained, the 25 Part-B categories. |

Together they are 7,125 windows and about 83 GiB, which is why extraction ran as
two sequential shards.

`PART_A_CACHE_README.md` in that folder is an earlier copy of
[part-a-cache-layout.md](data/part-a-cache-layout.md); this repository's version
is the current one.

### `checksums/`

`full51-part-a-v1.sha256` and `full51-part-b-v1.sha256` are the per-file SHA-256
lists for the two Stage-0 caches. Verify after downloading:

```bash
sha256sum -c full51-part-a-v1.sha256
```

Every cache also carries its own `metadata.json`, binding it to the checkpoint,
upstream commit, configuration, and manifest hashes that produced it. Read that
before training against a cache — the probe expects the `probe-features-v2`
schema with `index.parquet`, labels, categories, sequence IDs, and split fields.

## 1. Data

CO3Dv2 is external and is never committed. The selective downloader fetches
only the categories and sequences a config names, and records per-file CRC and
SHA-256 provenance:

```bash
export CO3D_ROOT=/path/to/co3d

python -m scripts.download_co3d_selective --config configs/stage0/full51-part-a.yaml --plan-only
python -m scripts.download_co3d_selective --config configs/stage0/full51-part-a.yaml --index-only
python -m scripts.download_co3d_selective --config configs/stage0/full51-part-a.yaml
python -m scripts.build_manifests          --config configs/stage0/full51-part-a.yaml
```

`configs/stage0/full.yaml` deliberately refuses `categories: all` as an
accidental-download guard. Use `full51-part-a.yaml` and `full51-part-b.yaml`,
whose 26/25 category lists are disjoint and union to the official 51. They are
storage shards, not scientific splits.

The split protocol - official CO3D sequence splits, sequence-level isolation,
deterministic six-frame windows - is
[ADR 0002](decisions/0002-co3dv2-stage0-data-protocol.md) and
[docs/data/stage0-protocol.md](data/stage0-protocol.md).

## 2. Frozen-backbone extraction

Both CUT3R backbones need the audited upstream checkout and the compatibility
patch this repository carries:

```bash
export CUT3R_ROOT=/path/to/CUT3R
export CUT3R_CHECKPOINT=/path/to/cut3r_512_dpt_4_64.pth
export CUT3R_CACHE_ROOT=/path/to/caches
export CUT3R_ARTIFACT_ROOT=/path/to/artifacts

python -m scripts.apply_cut3r_compatibility_patch \
  --cut3r-root "$CUT3R_ROOT" --expected-commit 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf
python -m scripts.validate_checkpoint --config configs/stage0/full51-part-a.yaml --load-model
python -m scripts.project_cache_storage \
  --manifest-dir "$CUT3R_ARTIFACT_ROOT/manifests/full51-part-a-v1" \
  --filesystem-path "$CUT3R_CACHE_ROOT" --reserve-gib 10
```

Extraction pins the upstream revision and the checkpoint SHA-256, and fails on
any other checkout or checkpoint content. Then extract the probe-feature caches
one backbone at a time:

```bash
python -m scripts.extract_probe_features --config configs/probe_features/cut3r_trained.yaml
python -m scripts.extract_probe_features --config configs/probe_features/cut3r_random.yaml
python -m scripts.extract_probe_features --config configs/probe_features/dinov2.yaml
python -m scripts.validate_cache --cache-dir "$CUT3R_CACHE_ROOT/probe/cut3r-trained"
```

Both Full-51 cache roots together are about 83 GiB, so the two parts were run
sequentially. Cache layout, including the expanded Part-A partitions, is in
[docs/data/part-a-cache-layout.md](data/part-a-cache-layout.md).

## 3. Probe training

Backbones are frozen throughout. Only the head is trained; it reads cached
tensors and never loads a backbone.

### Segmentation (Stage 1)

```bash
export CUT3R_CACHE_ROOT=/path/to/caches

python -m src.segmentation.train_segmentation \
  --config src/segmentation/configs/cut3r_trained_expanded_mlp.yaml \
  --checkpoint-selection best_val \
  --output-dir src/segmentation/experiments/cut3r-trained-expanded-bestval

python -m src.segmentation.inference_segmentation \
  --config src/segmentation/configs/cut3r_trained_expanded_mlp.yaml \
  --checkpoint src/segmentation/experiments/cut3r-trained-expanded-bestval/head.pt \
  --split test --save-dir src/segmentation/experiments/cut3r-trained-expanded-bestval --save-masks
```

Swap the config for any other `<backbone>_<data>_<capacity>.yaml` to change
backbone, data scale, or head capacity. Training is deterministic under the
seed fixed in each config (`20260729`), so results are bit-identical rather
than approximately reproducible.

Without CO3D or a GPU you can still exercise the whole path on a synthetic
cache; see the smoke-test section of
[src/segmentation/README.md](../src/segmentation/README.md). Nothing measured
that way says anything about the research question, and the synthetic flag
propagates into every artifact.

### Classification (Stage 2)

```bash
export CUT3R_TRAINED_RESPLIT_CACHE=/path/to/cut3r-trained-resplit
export CUT3R_RANDOM_RESPLIT_CACHE=/path/to/cut3r-random-resplit
export DINOV2_RESPLIT_CACHE=/path/to/dinov2-resplit

python -m src.classification.train_classification \
  --config src/classification/configs/fullunion-resplit/cut3r_trained_state_linear_adam.yaml

python -m src.classification.inference_classification \
  --config src/classification/configs/fullunion-resplit/cut3r_trained_state_linear_adam.yaml \
  --checkpoint src/classification/experiments/fullunion-resplit_cut3r-trained/state_tokens/classification-cut3r-trained-fullunion-resplit80-10-10-linear-adam/head.pt \
  --split test
```

Inference writes `inference-test-probabilities.csv`, which is exactly the input
format `reports/classification/predictions/` holds, so stage 4 runs on it
directly.

The three classification caches are the union of the original, leftover, and
`cap100-new-train` Part-A caches, re-split 80/10/10 at sequence level under seed
`20260731`; the resulting cache metadata records that protocol.

## What is not committed

Datasets, embedding caches, backbone checkpoints, trained probe heads, raw
per-window mask tensors, and run directories. They are large, they are
regenerable from the commands above, and `.gitignore` keeps them out.

What is committed instead is everything needed to check the result: the configs
that produced each run, the metrics, the per-window classification predictions,
and the report figures.
