# CUT3R Semantic Probing

Deep Learning course project investigating whether frozen CUT3R representations encode:

1. class-agnostic foreground/background segmentation; and
2. semantic object identity across CO3D categories.

This repository currently contains the project structure and collaboration process only. Implementation code will be added through reviewed pull requests.

## Project stages

| Stage | Goal | Planned areas |
|---|---|---|
| 0 - Foundations | Prepare CO3D and define reproducible embedding extraction | `src/data/`, `src/embeddings/` |
| 1 - Binary segmentation | Validate the prior proof of concept at larger scale and compare baselines | `src/segmentation/`, `src/baselines/` |
| 2 - Multiclass classification | Compare image-level and per-pixel classification probes | `src/classification/`, `src/baselines/` |
| 3 - Optional extension | Multi-object detection/classification, only after a recorded scope decision | To be decided |
| Close-out | Analysis, architecture diagram, presentation, and report | `docs/`, `reports/` |

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
| `src/data/` | CO3D discovery, filtering, manifests, and preprocessing |
| `src/embeddings/` | Frozen-backbone feature extraction and caching |
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
