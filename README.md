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

```bash
export CO3D_ROOT=/data/co3d
export CUT3R_ROOT=/work/CUT3R
export CUT3R_CHECKPOINT=/models/cut3r_512_dpt_4_64.pth
export CUT3R_ARTIFACT_ROOT=/artifacts/cut3r-semantic
export CUT3R_CACHE_ROOT=/cache/cut3r-semantic
```

Build and validate the local debug manifests:

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
| Preprocess CO3D for both segmentation and classification | In progress | Max Bershtman | - | Leakage-safe manifests, exact shared RGB/mask transforms, and validation are implemented; real debug data and overlay review remain. |
| Define, extract, and cache the headless CUT3R representation ("embeddings") | In progress | Max Bershtman | Max Bershtman (earlier proof of concept) | The six-timestep image/state contract, adapter, verified cache, and provenance checks are implemented; real CUDA reproducibility and performance gates remain. |
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
