# CUT3R Semantic Probing

Deep Learning course project investigating whether frozen CUT3R representations encode:

1. class-agnostic foreground/background segmentation; and
2. semantic object identity across CO3D categories.

CUT3R is used as a frozen backbone. Only lightweight probe heads are trained, so
any semantic signal we measure comes from the representation, not from the probe.

## Repository layout

| Path | Purpose |
|---|---|
| `src/` | Importable package. Stage 0 subpackages: `src.backbones` (frozen CUT3R and DINOv2 wrappers, probe cache), `src.data` (CO3Dv2 manifests, transforms, deterministic windows), `src.embeddings` (extraction, cache, provenance), `src.common` (IO and table helpers). Stage 1 lives under `src.segmentation` (probe head, dataset, train/inference drivers). Placeholders for Stage 2 (`src.classification`) and shared baselines. See [src/backbones/README.md](src/backbones/README.md), [src/data/README.md](src/data/README.md), [src/embeddings/README.md](src/embeddings/README.md), and [src/segmentation/README.md](src/segmentation/README.md). |
| `scripts/` | One-off CLI utilities: manifest building, feature extraction, cache validation, CUT3R patch application. Installed as the `scripts.*` package so tests can invoke them via `python -m scripts.<name>`. See [scripts/README.md](scripts/README.md). |
| `tests/` | Test suite. Run from the repository root with `pytest`. See [tests/README.md](tests/README.md). |
| `configs/` | Versioned YAML configs. Stage 0 lives under `configs/stage0/`; probe-feature extraction under `configs/probe_features/`; Stage 1 probe-head configs live with the code under `src/segmentation/configs/`. |
| `patches/` | Upstream compatibility patches applied to third-party code (currently CUT3R). |
| `docs/` | Repository-wide records: ADRs, session notes, experiments, dataset provenance, and project protocol. |
| `artifacts/`, `reports/`, `notebooks/` | Registries and sources for external artifacts, the final report, and exploration. |
| `pyproject.toml` | Packaging. Installs `src.*` and `scripts.*` as top-level importable packages, and defines the pytest configuration. |
| `LLM_GUIDE.md` | Shared guide for AI-assisted work. |

Code lives in `src/` and `scripts/`; everything that describes the project as a
whole lives at the repository root.

## Getting started

Install once from the repository root:

```bash
python -m pip install -e ".[dev]"
```

This exposes the pipeline as top-level `src.*` and `scripts.*` packages, so every
workspace imports it the same way (for example
`from src.backbones import build_backbone`).

On Windows with the workspace inside OneDrive, follow
[docs/project/LOCAL_DEV_WINDOWS.md](docs/project/LOCAL_DEV_WINDOWS.md) instead
of `pip install` from the workspace root — the venv must live outside
OneDrive and Python 3.13 has a known numpy issue.

Run scripts and any command that uses a relative `configs/…` path from the
repository root. Run the test suite from the repository root:

```bash
pytest
```

Large datasets, embeddings, checkpoints, and generated results must never be
committed to Git.

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
| Verify that a binary segmentation probe works on the selected CO3D data | In progress | Aviv Rabi | Max Bershtman (earlier proof of concept) | The probe, dataset, training, and inference code exist and run end to end on a synthetic cache; no automated tests and no run on real embeddings yet. |
| Decide whether inference with an existing probe is sufficient or whether the MLP must be retrained | Not started | TBD | Max Bershtman (earlier proof of concept) | The decision is recorded as open; no approach has been approved. |
| Compare against random-weight and state-of-the-art baselines | In progress | Aviv Rabi | - | Shared backbones and a single extraction script cover CUT3R-trained, CUT3R-random, and DINOv2; the baseline definitions still need ADRs. |
| Analyze results and produce plots | Not started | TBD | Max Bershtman (earlier proof-of-concept report) | Experiment and metric templates exist; the new project has no results yet. |

### Stage 2 - Multiclass classification

| Work item | Status | Current owner(s) | Previous contributor(s) | Done so far |
|---|---|---|---|---|
| Train an MLP head for multiclass object classification | Not started | TBD | - | The embedding cache already stores per-window category labels; no classification probe is implemented. |
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

## Team

- Ron Bartal
- Yam Ben-Tov
- Max Bershtman
- Lihi Bar-Tal
- Aviv Rabi
- Sixth member: to be added

GitHub handles and detailed ownership will be added when available.
