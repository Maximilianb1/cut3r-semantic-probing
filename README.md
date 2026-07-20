# CUT3R Semantic Probing

Deep Learning course project investigating whether frozen CUT3R representations encode:

1. class-agnostic foreground/background segmentation; and
2. semantic object identity across CO3D categories.

Stage 0 implementation now provides leakage-safe CO3Dv2 manifests, deterministic
six-frame windows, CUT3R-aligned RGB/mask transforms, frozen CUT3R trajectory
extraction, and a verified feature cache. The scientific decisions remain
proposed until team review; no semantic probe is trained in Stage 0.

## Stage 0 quick start

Install the project and development tools:

```bash
python -m pip install -e ".[dev]"
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
export CUT3R_CHECKPOINT=/models/cut3r_512_dpt_4_64.pth
export CUT3R_ARTIFACT_ROOT=/artifacts/cut3r-semantic
export CUT3R_CACHE_ROOT=/cache/cut3r-semantic
```

All Stage 0 configurations pin the released `cut3r_512_dpt_4_64.pth` bytes to
SHA-256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`.
Extraction rejects any other file before deserialization and uses a scoped
seven-type OmegaConf allowlist with PyTorch's restricted weights-only loader.

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

See [the Stage 0 protocol](docs/data/stage0-protocol.md),
[ADR 0002](docs/decisions/0002-co3dv2-stage0-data-protocol.md), and
[ADR 0003](docs/decisions/0003-cut3r-trajectory-and-cache-contract.md) before
running the pilot or full extraction.

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

Follow the complete [Stage 0 Full-51 cache handoff](docs/data/stage0-full51-cache-handoff.md)
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
[Full-51 two-part runbook](docs/project/FULL51_TWO_PART_RUNBOOK.md); Part A and
Part B are execution shards only and must be combined for later training and
evaluation.

## Project stages

These stages follow the project proposal and work-allocation document. Keep the
tables current when ownership changes or a pull request completes work. Use one
of these status values: `Not started`, `Planned`, `In progress`, `Blocked`,
`Done`, or `Deferred`. Detailed blockers and the current milestone belong in
[PROJECT_STATUS.md](PROJECT_STATUS.md).

### Stage 0 - Foundations

| Work item | Status | Current owner(s) | Previous contributor(s) | Done so far |
|---|---|---|---|---|
| Clarify what the random-initialized CUT3R baseline means and reconcile it with the course guidance | Planned | Aviv Rabi | - | The question is recorded; no technical definition has been accepted yet. |
| Preprocess CO3D for both segmentation and classification | Done | Max Bershtman | - | The deterministic Full-51 manifests cover all 51 categories with 30/5/5 train/validation/test sequence caps and sequence-level split isolation. |
| Define, extract, and cache the headless CUT3R representation ("embeddings") | In progress | Max Bershtman | Max Bershtman (earlier proof of concept) | Both immutable cache roots are extracted and verified: 7,125 windows, 14,250 tensors, and 83.053 GiB total. Private Drive publication is in progress. |
| Design the shared code, configuration, and experiment structure for Stages 1 and 2 | In progress | Team (specific owners TBD) | - | The repository structure, versioned Stage 0 configs, tests, and documentation workflow are implemented; later probe interfaces remain to be finalized. |

### Stage 1 - Binary segmentation

| Work item | Status | Current owner(s) | Previous contributor(s) | Done so far |
|---|---|---|---|---|
| Verify that a binary segmentation probe works on the selected CO3D data | Not started | TBD | Max Bershtman (earlier proof of concept) | The earlier single-sequence proof of concept was audited; validation on the new dataset split has not started. |
| Decide whether inference with an existing probe is sufficient or whether the MLP must be retrained | Not started | TBD | Max Bershtman (earlier proof of concept) | The decision is recorded as open; no approach has been approved. |
| Compare against random-weight and state-of-the-art baselines | Not started | TBD | - | Baseline definitions and the evaluation protocol must be agreed before implementation. |
| Analyze results and produce plots | Not started | TBD | Max Bershtman (earlier proof-of-concept report) | Experiment and metric templates exist; the new project has no results yet. |

### Stage 2 - Multiclass classification

| Work item | Status | Current owner(s) | Previous contributor(s) | Done so far |
|---|---|---|---|---|
| Train an MLP head for multiclass object classification | Not started | TBD | - | The task is defined in the proposal; implementation has not started. |
| Compare image-level classification with per-pixel classification and choose an architecture | Not started | TBD | - | The required comparison is recorded; no architecture has been selected. |
| Compare against random-weight and state-of-the-art baselines | Not started | TBD | - | Baseline definitions and the evaluation protocol remain open. |
| Analyze results and produce plots | Not started | TBD | - | Experiment and metric templates exist; the project has no classification results yet. |

### Stage 3 - Optional extension

| Work item | Status | Current owner(s) | Previous contributor(s) | Done so far |
|---|---|---|---|---|
| Make a go/no-go decision and define the optional extension, such as multi-object detection or classification | Deferred | Team | - | The stage is intentionally outside the required scope until Stages 0-2 are stable. |

### Project close-out

| Work item | Status | Current owner(s) | Previous contributor(s) | Done so far |
|---|---|---|---|---|
| Create the final architecture diagram | Not started | TBD | - | A location for report assets exists; the final architecture is not yet decided. |
| Prepare the course presentation | Not started | TBD | - | No presentation work has started. |
| Write and review the final report | Not started | Team (section owners TBD) | Max Bershtman (earlier proof-of-concept report) | The proposal and earlier report provide background material; final-project writing has not started. |

## Start here

1. Read [CONTRIBUTING.md](CONTRIBUTING.md).
2. Check [PROJECT_STATUS.md](PROJECT_STATUS.md).
3. Read the accepted records in [docs/decisions](docs/decisions/README.md).
4. Create or claim a GitHub issue.
5. Work on a short-lived branch and open a pull request.
6. Add a session note when handing work to another person or assistant.

## Repository map

| Path | Purpose |
|---|---|
| `src/data/` | CO3Dv2 annotation parsing, official splits, six-frame windows, transforms, manifests, and validation |
| `src/embeddings/` | Pinned-upstream CUT3R adapter, exact cuRoPE compatibility provenance, six-timestep feature trajectories, extraction, and verified caching |
| `src/segmentation/` | Binary segmentation probes and evaluation |
| `src/classification/` | Multiclass probes and evaluation |
| `src/baselines/` | Random-weight, simple trained, and advanced frozen baselines |
| `configs/` | Versioned experiment configurations |
| `tests/` | Unit and integration tests |
| `docs/decisions/` | Architectural and scientific decision records |
| `docs/sessions/` | Concise work-session and handoff notes |
| `docs/experiments/` | Reproducible experiment records |
| `docs/data/` | Dataset versions, manifests, splits, and provenance |
| `artifacts/` | Instructions and indexes for external large artifacts |

Large datasets, embeddings, checkpoints, and generated results must not be committed to Git.

## Team

- Ron Bartal
- Yam Ben-Tov
- Max Bershtman
- Lihi Bar-Tal
- Aviv Rabi
- Sixth member: to be added

GitHub handles and detailed ownership will be added when available.
