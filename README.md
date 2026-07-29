# CUT3R Semantic Probing

Deep Learning course project investigating whether frozen CUT3R representations encode:

1. class-agnostic foreground/background segmentation; and
2. semantic object identity across CO3D categories.

## Repository layout

The repository is split into a completed foundation and the active work built on
top of it.

| Path | Purpose |
|---|---|
| `data_pipeline/` | Stage 0 foundation: CO3Dv2 preprocessing, leakage-safe manifests, deterministic six-frame windows, frozen CUT3R feature extraction, and the verified feature cache. This is prior work; see [data_pipeline/README.md](data_pipeline/README.md) and [data_pipeline/PROJECT_STATUS.md](data_pipeline/PROJECT_STATUS.md). |
| `segmentation_validation/` | Active work: extend the earlier single-category binary-segmentation proof of concept to all 51 CO3D categories using the cached CUT3R representations. See [segmentation_validation/README.md](segmentation_validation/README.md). |
| `pyproject.toml` | Root packaging. Installs the `data_pipeline` code as the importable `src` and `scripts` packages so downstream workspaces can reuse it. |
| `LLM_GUIDE.md` | Shared guide for AI-assisted work across the whole repository. |

## Getting started

Install once from the repository root:

```bash
python -m pip install -e ".[dev]"
```

This exposes the pipeline as top-level `src.*` and `scripts.*` packages, so both
`data_pipeline/` and `segmentation_validation/` import it the same way (for
example `from src.embeddings.cache import load_trajectory`).

Run pipeline scripts and any command that uses a relative `configs/...` path
from inside `data_pipeline/`. Run the test suite from the repository root:

```bash
pytest
```

## Where to read next

1. [LLM_GUIDE.md](LLM_GUIDE.md) — guardrails and workflow for any change.
2. [data_pipeline/README.md](data_pipeline/README.md) — the frozen foundation and cache contract.
3. [data_pipeline/PROJECT_STATUS.md](data_pipeline/PROJECT_STATUS.md) — current phase, blockers, and milestones.
4. [segmentation_validation/README.md](segmentation_validation/README.md) — the current task.
