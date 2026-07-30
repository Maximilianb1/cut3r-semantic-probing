# Session: Segmentation-validation infrastructure (backbones, probe cache, probe)

- Date: 2026-07-29
- Author: Aviv Rabi
- Branch: segmentation-validation
- Related issue/PR: (to be linked)
- Commits: `94c6997` (infrastructure + extraction), `bb65265` (probe simplification, docs, trajectory-layout test)
- Assistant/model, if used: Claude Code (Claude Fable 5)

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
  - `segmentation_validation/model_segmentation.py`: `SegmentationProbe` = a
    per-token MLP head over **precomputed cached tokens** (linear when
    `hidden_dims=[]`). The probe holds no backbone: backbones live in
    `src.backbones` and are run ahead of time by the extraction script.
  - `segmentation_validation/segmentation_dataset.py`: `ProbeCacheDataset` +
    variable-grid collation + sequence-disjoint guard.
  - `segmentation_validation/train_segmentation.py`: config-driven training loop
    with token accuracy and foreground IoU (macro/micro/per-category); saves the
    trained head to `head.pt`.
  - `segmentation_validation/segmentation_inference.py`: reloads the trained head
    and evaluates the probe on a chosen split; optional per-window mask export.
  - `segmentation_validation/configs/{cut3r_trained,cut3r_random,dinov2}.json`:
    paths use the Stage 0 `${ENV}` variables (resolved at config load, no
    hardcoded locations); the CUT3R checkpoint resolves to upstream's default
    `${CUT3R_ROOT}/src/cut3r_512_dpt_4_64.pth`, so no separate
    `CUT3R_CHECKPOINT` is needed. `--manifest-dir` overrides the manifest part.
  - Tests: `data_pipeline/tests/test_backbones.py`,
    `segmentation_validation/tests/{conftest,test_segmentation}.py`, including a
    training smoke on each cache layout so all three backbones' storage shapes
    are exercised end to end.
  - PR #10 review fixes: window reads now pull only their own tensors from a
    shard via `safe_open` (a full-shard read per window multiplied training I/O
    by the shard size), the trained head loads with `weights_only=True`, and
    inference derives metrics, per-window IoUs, and optional mask grids from a
    single batched pass instead of iterating the split twice. Added baseline
    backbone tests that need no weights or network (DINOv2 patch-14 geometry and
    extraction via an injected model; CUT3R seeded re-initialization).
  - Documentation pass over `segmentation_validation/`: shared vocabulary
    (sequence / window = 6 frames / target frame / token), a reading guide for the
    training loop, and the recorded conventions (decision threshold, patch-grid
    metric resolution, empty-target IoU).
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
  - **DINOv2 sees only the target frame** (it is a single-image encoder, so it has
    no multi-view state), while CUT3R's tokens are conditioned on the preceding
    five frames. Accepted for segmentation, where the target is one frame's mask;
    revisit for classification.
- Still open (repository layout; raised by ronbartal in PR #10, needs a team call):
  - The reorganization moved `docs/` under `data_pipeline/`, but it holds
    repo-wide governance (ADRs, CONTRIBUTING, session notes for every stage) and
    `data_pipeline/README.md` likewise describes the whole project. Candidate fix:
    move repo-wide docs back to the repository root and keep only
    pipeline-specific material under `data_pipeline/`.
- Still open (engineering; documented in the code and README, not yet implemented):
  - `head.pt` stores the **final** epoch, not the best-validation epoch.
  - **Class imbalance** is unmanaged; `pos_weight` exists but defaults to null.
  - **Empty-target IoU convention**: a window with no foreground currently scores
    IoU 1.0, which can inflate macro-IoU.

## Verification

| Command/check | Result |
|---|---|
| `pip install -e ".[dev]"` (repo root) | Succeeds; `src`, `scripts`, `src.backbones` import |
| GPU build | torch 2.11.0+cu128, RTX 5080 (sm_120), CUDA fwd/bwd OK |
| `pytest` (full suite) | 65 passed |
| Window read does not materialize its whole shard | Regression test (full-shard loader raises if used) |
| Inference macro-IoU equals the mean of its own per-window IoUs | Pass (single pass, metrics unchanged) |
| v2 cache round-trip both layouts (trajectory + target_only) + corruption | Pass |
| Reuse converter: attach labels from Stage 0 cache + mask-SHA mismatch guard | Pass |
| Training smoke on separable synthetic cache (both layouts) | Learns: macro IoU > 0.9, token acc > 0.95 |
| Train -> save `head.pt` -> reload -> inference round trip | Pass |
| DINOv2 grid is aspect-preserving multiples of 14 | Confirmed |
| Config `${ENV}` paths resolve; no placeholders remain | Confirmed |

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
