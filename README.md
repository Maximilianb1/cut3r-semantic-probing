# CUT3R Semantic Probing

Deep Learning course project investigating whether frozen CUT3R representations encode:

1. class-agnostic foreground/background segmentation; and
2. semantic object identity across CO3D categories.

This repository currently contains the project structure and collaboration process only. Implementation code will be added through reviewed pull requests.

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
| Preprocess CO3D for both segmentation and classification | Planned | Max Bershtman | - | Dataset documentation and leakage-prevention rules exist; preprocessing code has not been implemented. |
| Define, extract, and cache the headless CUT3R representation ("embeddings") | Planned | Max Bershtman | Max Bershtman (earlier proof of concept) | The earlier extraction path was audited. The exact representation and cache format remain open decisions. |
| Design the shared code, configuration, and experiment structure for Stages 1 and 2 | In progress | Team (specific owners TBD) | - | The responsibility-based repository skeleton, configuration folders, tests, and documentation workflow are in place. Model interfaces are not yet defined. |

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
