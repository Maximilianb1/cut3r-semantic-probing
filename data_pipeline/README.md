# Data Pipeline (Stage 0)

The data and embedding foundation for the project: CO3Dv2 preprocessing,
leakage-safe manifests, deterministic six-frame windows, CUT3R-aligned RGB/mask
transforms, the shared frozen backbones, and verified embedding caches.

This directory holds **code only**. Project-wide material lives at the repository
root: [README](../README.md), [PROJECT_STATUS](../PROJECT_STATUS.md),
[CONTRIBUTING](../CONTRIBUTING.md), and [docs/](../docs). No semantic probe is
trained here; Stage 1 lives in [`../segmentation_validation/`](../segmentation_validation/README.md).

## Contents

| Path | Purpose |
|---|---|
| `src/data/` | CO3Dv2 annotation parsing, official splits, six-frame windows, transforms, manifests, and validation |
| `src/embeddings/` | Pinned-upstream CUT3R adapter, cuRoPE compatibility provenance, six-timestep feature trajectories, extraction, and verified caching |
| `src/backbones/` | Shared frozen-backbone contract (CUT3R trained/random, DINOv2) and the probe embedding cache |
| `src/segmentation/`, `src/classification/`, `src/baselines/` | Placeholders for later probe and baseline code |
| `configs/` | Versioned experiment configurations |
| `scripts/` | Command-line entry points |
| `patches/` | The audited upstream CUT3R compatibility patch |
| `tests/` | Unit and integration tests |

## Where to run commands

Packaging lives at the repository root, so **install from the root** but run the
`python -m scripts.*` commands (and anything using a relative `configs/...` path)
from **inside `data_pipeline/`**. Downstream workspaces import this code as the
installed `src` / `scripts` packages.

```bash
python -m pip install -e ".[dev]"    # from the repository root
```

Set external paths. Never commit datasets, checkpoints, caches, VM credentials,
or generated artifacts.

On Azure, first verify that these paths are backed by a managed OS/data disk or
an explicitly approved persistent share. Never store repositories, checkpoints,
datasets, caches, manifests, or results on `/mnt` when it resolves to Azure's
`/dev/disk/azure/resource` disk; that disk is temporary and can be reinitialized
after stop/deallocate or host-maintenance events.

```bash
export CO3D_ROOT=/data/co3d
export CUT3R_ROOT=/work/CUT3R
export CUT3R_ARTIFACT_ROOT=/artifacts/cut3r-semantic
export CUT3R_CACHE_ROOT=/cache/cut3r-semantic
```

All Stage 0 configurations pin the released `cut3r_512_dpt_4_64.pth` bytes to
SHA-256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`.
Extraction rejects any other file before deserialization and uses a scoped
seven-type OmegaConf allowlist with PyTorch's restricted weights-only loader.

## Stage 0 quick start

Download only the files selected by the deterministic debug configuration.
Run the metadata plan, remote ZIP index, and payload materialization as separate
gates so disk use is known before downloading image data:

```bash
python -m scripts.download_co3d_selective \
  --config configs/stage0/debug.yaml --plan-only \
  > "$CUT3R_ARTIFACT_ROOT/debug-download-plan.json"
python -m scripts.download_co3d_selective \
  --config configs/stage0/debug.yaml --index-only \
  > "$CUT3R_ARTIFACT_ROOT/debug-download-index.json"
python -m scripts.download_co3d_selective \
  --config configs/stage0/debug.yaml \
  > "$CUT3R_ARTIFACT_ROOT/debug-download-result.json"
```

The downloader uses byte-range requests against the official full-release ZIPs;
it does not download whole multi-gigabyte category archives. Then build and
validate the debug manifests:

```bash
python -m scripts.build_manifests --config configs/stage0/debug.yaml
python -m scripts.validate_manifests \
  --manifest-dir "$CUT3R_ARTIFACT_ROOT/manifests/debug" \
  --dataset-root "$CO3D_ROOT" \
  --inspect-files
```

On the CUDA machine, independently extract the same window twice, compare every
cached tensor exactly, and verify a cache:

```bash
python -m scripts.apply_cut3r_compatibility_patch \
  --cut3r-root "$CUT3R_ROOT" \
  --expected-commit 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf
(cd "$CUT3R_ROOT/src/croco/models/curope" && \
  python setup.py build_ext --inplace)
python -m scripts.validate_checkpoint \
  --config configs/stage0/debug.yaml \
  --load-model
python -m scripts.extract_features \
  --config configs/stage0/debug.yaml \
  --limit-windows 1 \
  --cache-dir "$CUT3R_CACHE_ROOT/preflight-a"
python -m scripts.extract_features \
  --config configs/stage0/debug.yaml \
  --limit-windows 1 \
  --cache-dir "$CUT3R_CACHE_ROOT/preflight-b"
python -m scripts.compare_caches \
  --left "$CUT3R_CACHE_ROOT/preflight-a" \
  --right "$CUT3R_CACHE_ROOT/preflight-b"
python -m scripts.validate_cache --cache-dir "$CUT3R_CACHE_ROOT/preflight-a"
```

See [the Stage 0 protocol](../docs/data/stage0-protocol.md),
[ADR 0002](../docs/decisions/0002-co3dv2-stage0-data-protocol.md), and
[ADR 0003](../docs/decisions/0003-cut3r-trajectory-and-cache-contract.md) before
running the pilot or full extraction.

## Probe embedding extraction

`scripts/extract_probe_features.py` builds the per-backbone embedding caches the
probes train on. Each entry holds one window's target-frame tokens plus both task
labels (segmentation grid and category). See
[`src/backbones/README.md`](src/backbones/README.md) for the contract and the
open baseline decisions.

Run once per backbone, from this directory:

```bash
python -m scripts.extract_probe_features --config configs/probe_features/dinov2.yaml
```

Full-51 has two manifest parts. Run each config once per part into the **same**
cache directory, passing the part B manifests explicitly:

```bash
python -m scripts.extract_probe_features --config configs/probe_features/dinov2.yaml --manifest-dir "$CUT3R_ARTIFACT_ROOT/manifests/full51-part-b-v1"
```

The configs reference `${ENV}` paths (no hardcoded locations); export the Stage 0
variables before running:

- **DINOv2:** `CO3D_ROOT`, `CUT3R_ARTIFACT_ROOT`, `CUT3R_CACHE_ROOT`
- **random-CUT3R:** those plus `CUT3R_ROOT` (the checkpoint resolves to CUT3R's
  default `${CUT3R_ROOT}/src/cut3r_512_dpt_4_64.pth`)
- **CUT3R-trained:** reuses the existing Stage 0 cache — no GPU, no re-extraction

Each cache written here is what
[`segmentation_validation/`](../segmentation_validation/README.md) consumes; the
probe config of the same name must point at the same `probe_cache.dir`.

## Team cache access

The Full-51 Stage 0 representations are distributed as two immutable external
cache roots, `full51-part-a-v1` and `full51-part-b-v1`. They are category/storage
shards, not scientific splits, and must never be merged by copying their files
together. Teammates should download both to a non-synced local SSD, verify the
published SHA-256 manifests, validate each cache, and load them as a logical
union. The cache does not replace the CO3D manifests or RGB/mask files.

Authorized teammates can access the private
[Stage 0 Full-51 Google Drive folder](https://drive.google.com/drive/folders/1UttTnkxRlcz3H3K-Puv1VhVfcjrAZTzN).
The folder remains restricted; request Viewer access from Max if the link does
not open.

Follow the complete [Stage 0 Full-51 cache handoff](../docs/data/stage0-full51-cache-handoff.md)
for the private team-folder layout, upload/download commands, immutable identities,
verification gates, tensor shapes, timestep semantics, and loading example.

Each cached window contains two different tensors; use their exact names rather
than calling either one simply an "embedding":

| Cache field | Meaning | Cached shape | Primary later use |
|---|---|---|---|
| `image_tokens` | Spatial tokens for each current frame after interaction with the preceding recurrent state; the pose token is removed | `[6, 1, grid_height * grid_width, 768]` | Frame-6 segmentation and per-pixel classification; spatial pooling for image classification |
| `state_tokens` | The persistent recurrent memory after CUT3R has consumed and committed each frame | `[6, 1, 768, 768]` | Pooled image classification and temporal/state analysis; not spatial mask prediction |

Index `t` refers to `frame_ids[t]`. Therefore `image_tokens[5, 0]` describes
the sixth frame in its five-frame history, while `state_tokens[5, 0]` is the
memory after all six frames. Both trajectories are saved for later analysis.

The approved all-category run uses two storage shards with 30/5/5
train/validation/test sequence caps. Follow the
[Full-51 two-part runbook](../docs/project/FULL51_TWO_PART_RUNBOOK.md); Part A and
Part B are execution shards only and must be combined for later training and
evaluation.
