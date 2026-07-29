# Session: Segmentation-validation infrastructure (backbones, probe cache, probe)

- Date: 2026-07-29
- Author: Aviv Rabi
- Branch: segmentation-validation
- Related issue/PR: (to be linked)
- Assistant/model, if used: Claude Code (Opus 4.8)

## Objective

Stand up the infrastructure to reproduce the earlier single-category binary
segmentation proof of concept across all 51 CO3D categories, and to compare
three frozen backbones (CUT3R-trained, CUT3R-random, DINOv2-trained) with one
shared probe. No results yet — infrastructure only.

## Context and inputs

- Repository was reorganized: prior Stage 0 work now lives under `data_pipeline/`;
  new work lives under `segmentation_validation/`. Packaging stays at the repo
  root and installs `data_pipeline` as the importable `src` / `scripts` packages.
- Grounded in the real Stage 0 APIs: `Cut3rFeatureExtractor`, `load_cut3r_model`,
  `prepare_image_window`, `transform_rgb_mask`, the trajectory cache
  (`load_trajectory`), and the window/frame manifest columns
  (`window_id`, `frame_ids`, `category`, `sequence_id`, `split`).

## Work completed

- Files changed / added:
  - `src/backbones/{base,cut3r,dinov2,probe_cache,__init__}.py` + `README.md`:
    shared frozen-backbone contract, CUT3R (trained/random) and DINOv2 backbones,
    a unified probe-feature cache, and `build_backbone`.
  - `segmentation_validation/model_segmentation.py`: `SegmentationProbe` = frozen
    backbone + per-token MLP head (linear when `hidden_dims=[]`).
  - `segmentation_validation/segmentation_dataset.py`: `ProbeCacheDataset` +
    variable-grid collation + sequence-disjoint guard.
  - `segmentation_validation/train_segmentation.py`: config-driven training loop
    with token accuracy and foreground IoU (macro/micro/per-category); saves the
    trained head to `head.pt`.
  - `segmentation_validation/segmentation_inference.py`: reloads the trained head
    and evaluates the probe on a chosen split; optional per-window mask export.
  - `segmentation_validation/configs/{cut3r_trained,cut3r_random,dinov2}.json`.
  - Tests: `data_pipeline/tests/test_backbones.py`,
    `segmentation_validation/tests/{conftest,test_segmentation}.py`.
- Artifacts produced: none committed (caches/runs are git-ignored external data).

## Decisions

- Made (engineering):
  - One `Backbone` contract returns target-frame `spatial_tokens` (segmentation)
    and `global_tokens` (classification) plus the geometry-aligned target mask.
  - Each backbone owns its preprocessing/grid; the mask is pooled onto that grid.
  - A single v2 embedding cache serves all three backbones, with two layouts and
    both task labels per entry: `trajectory` (CUT3R-trained: grid+latent per
    state) vs `target_only` (random-CUT3R, DINOv2: last state), plus `seg_labels`
    and `category`/`category_index`. `scripts/extract_probe_features.py` produces
    it. CUT3R-trained reuses the existing trajectory cache without a GPU via
    `attach_labels_from_trajectory_cache`, cross-checking each mask SHA-256;
    random-CUT3R and DINOv2 extract live via `extract_to_cache`.
  - Splits are the manifest's sequence-level assignment; training asserts
    sequence-disjoint splits (no adjacent-view leakage).
  - The head is honestly named: linear probe vs MLP probe (`head.is_linear`).
- Still open (scientific — must NOT be treated as settled; need ADRs):
  - Meaning of the **random-initialized CUT3R** baseline. Current
    `reset_parameters` re-init is a seeded placeholder only.
  - **DINOv2 variant + dependency** (`torch.hub dinov2_vitb14`); adds a dependency
    and network download.
  - **Cross-backbone comparison protocol** (CUT3R and DINOv2 use different native
    preprocessing/grids); what is held fixed for fairness is undecided.

## Verification

| Command/check | Result |
|---|---|
| `pip install -e ".[dev]"` (repo root) | Succeeds; `src`, `scripts`, `src.backbones` import |
| GPU build | torch 2.11.0+cu128, RTX 5080 (sm_120), CUDA fwd/bwd OK |
| `pytest` (full suite) | 58 passed |
| v2 cache round-trip both layouts (trajectory + target_only) + corruption | Pass |
| Reuse converter: attach labels from Stage 0 cache + mask-SHA mismatch guard | Pass |
| Training smoke on separable synthetic cache | Learns: macro IoU > 0.9, token acc > 0.95 |
| DINOv2 grid is aspect-preserving multiples of 14 | Confirmed |

Not run here (needs GPU + external CUT3R repo/checkpoint, DINOv2 weights, and
CO3D files): real feature extraction into probe caches and real probe training.

## Human review of AI-assisted work

Pending human review. Reviewer should confirm: the random-init placeholder is
not mistaken for an accepted baseline; the CUT3R `image_tokens[5]`/`state_tokens[5]`
mapping matches ADR 0003; and DINOv2 mask/patch alignment is correct on real data.

## Next step

Build the three probe-feature caches (Aviv) with
`python -m scripts.extract_probe_features --config <config>`: CUT3R-trained
reuses the Full-51 trajectory cache (`source: existing_embeddings`, no GPU),
CUT3R-random and DINOv2 extract live on the GPU VM; then run
`train_segmentation.py` per config. Open the baseline-definition ADRs before
reporting any comparison.
