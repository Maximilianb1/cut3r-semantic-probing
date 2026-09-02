# Reproducing the results

The pipeline has four stages. Each one is more expensive than the last, and
each one can be entered independently, because the stage before it wrote its
output to disk:

```
CO3Dv2 download  ->  frozen-backbone extraction  ->  probe training  ->  analysis
   (~hours)              (~1 day, 1 GPU)              (~minutes, CPU)     (seconds)
```

**Start at stage 4 if you only want to check our numbers.** The analysis stage
runs from what is committed in this repository, with no dataset, no GPU, and no
model weights. Stages 1-3 need the CO3D dataset and a GPU.

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

> **Known gap.** The three classification caches are not the Part-A caches as
> extracted. They are the hard-link union of the original, leftover, and
> `cap100-new-train` Part-A caches, re-split 80/10/10 at sequence level because
> the inherited split left only 104 test windows. The resulting cache metadata
> records the protocol (`probe-cache-hardlink-union-v1`,
> `classification-derived-v1`), the seed (`20260731`), and the fractions
> (`val_fraction: 0.1`, `test_fraction: 0.1`), but the script that performed
> the union and the re-split was written on the VM and is not in this
> repository. Stage 3 for classification therefore cannot be reproduced
> byte-for-byte from this repository alone; stage 4 can, which is why the
> per-window predictions are committed.

## What is not committed

Datasets, embedding caches, backbone checkpoints, trained probe heads, raw
per-window mask tensors, and run directories. They are large, they are
regenerable from the commands above, and `.gitignore` keeps them out.

What is committed instead is everything needed to check the result: the
configs that produced each run, the metrics, the per-window classification
predictions, the report figures, and an experiment record per run under
[docs/experiments/](experiments/README.md).
