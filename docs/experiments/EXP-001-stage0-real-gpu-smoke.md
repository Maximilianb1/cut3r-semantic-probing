# EXP-001: Stage 0 real-data GPU smoke

- Date: 2026-07-18
- Owner: Max Bershtman
- Status: Completed (engineering smoke only)
- Related issue/PR: PRs #2, #3, and #4
- Code commit: `0e5af52e4b1738bdb838f224bd5cd2e1b80ec476`

## Objective

Verify on real CO3Dv2 files and the Technion GPU that the Stage 0 pipeline can
build leakage-checked six-frame manifests, apply aligned RGB/mask transforms,
load the pinned CUT3R checkpoint safely, save every image/state timestep, and
reproduce the resulting cache exactly.

This was not a probe-training or scientific evaluation run.

## Environment and model

- VM GPU: NVIDIA A10-24Q, 24,512 MiB reported VRAM.
- Driver/runtime: NVIDIA driver 570.211.01; CUDA 12.8.
- PyTorch: 2.7.1+cu128; torchvision 0.22.1+cu128.
- CUT3R commit: `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`.
- Compatibility policy: `curope-scalar-type-v1`.
- Checkpoint: `cut3r_512_dpt_4_64.pth`.
- Checkpoint SHA-256:
  `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`.
- Restricted load policy: `pytorch-weights-only-omegaconf-v1`.
- Model: `ARCroco3DStereo`, 793,307,858 parameters; all checkpoint keys matched.

## Data and manifest

The official CO3Dv2 `v2_231130` single-sequence data for `ball` and `chair`
occupied 434 MiB. It omitted the few-view training set lists, so the verified
full-release metadata-only archives were overlaid:

- `ball_000.zip` SHA-256:
  `f144eda00355309e88fe7097278f5fbbd3f5d816ccdb43fd9b2bea1e99d5e028`;
- `chair_000.zip` SHA-256:
  `78f09f2c92cce85af3f71b8c7a79dbc14142641648133bb60b927e3279666b79`.

The resulting external dataset occupied 618 MiB. `configs/stage0/debug.yaml`
generated a valid manifest with full file inspection:

- selected sequences: 2, both `ball` and official validation split;
- selected frames: 404;
- windows: 6, all validation;
- train/test windows: 0/0;
- frames manifest SHA-256:
  `c168c8f407778f919a38596ba543dfac4a47bc3b8590525e12b8893d2dd36242`;
- sequences manifest SHA-256:
  `8f56b837584cf85f89b1ce5d74c13b7823100f3de6fbf7622444409fe20a9a5e`;
- windows manifest SHA-256:
  `4911f89f952f2111ab973bc4ccd11bf033ed8659d24a1be536018c3089c2b0a7`.

All six transformed target overlays were reviewed. The available masks followed
the object boundaries, foreground objects remained inside the crops, aspect
ratios were preserved, and no target mask was empty. The montage contained no
chair example, so it did not establish category-diverse transform coverage.

## Representation audit

The independently audited first window was
`window-73f75f39785ac26ee186`.

- Image trajectory: float16 `[6, 1, 576, 768]`, finite at every timestep.
- State trajectory: float16 `[6, 1, 768, 768]`, finite at every timestep.
- Image adjacent-timestep mean absolute differences ranged from 0.036565 to
  0.049823; first-to-last difference was 0.072059.
- State adjacent-timestep mean absolute differences ranged from 0.198294 to
  0.313114; first-to-last difference was 0.543277.

The nonzero changes confirm that all six states were recorded after their
respective frames rather than one state being repeated.

## Reproducibility and performance

Two independent one-window cache directories produced identical metadata and
index hashes and passed `scripts.validate_cache`:

- metadata SHA-256:
  `94d8573a5a782fa30a3250051035ebdf5329763bb88d41d1b8684dabf988ad3b`;
- index SHA-256:
  `21cd873924a040dd57e3acd60238dc720f0732accd210f4c730ecfdbce963c73`;
- exact comparison: image maximum absolute difference 0.0; state maximum
  absolute difference 0.0 at `atol=0`, `rtol=0`;
- peak allocated CUDA memory: 3,429,500,928 bytes;
- observed cold/warm one-window elapsed times: 5.834/0.766 seconds;
- rounded disk usage: 12 MiB per one-window cache.

The six-window run wrote one verified shard containing 12 tensors:

- elapsed time: 3.409 seconds;
- mean: 0.568 seconds per written window;
- peak allocated CUDA memory: 3,454,282,752 bytes;
- rounded cache size: 71 MiB;
- index SHA-256:
  `b61774cbc4deeeabe964ede06b9a796de07d845e76359354282d9221e3c4b2ba`.

The first window in this multi-window cache was bit-for-bit identical to its
independent one-window cache for both representations. This also checks that
state from one window did not leak into the next extraction call.

## Interpretation and limitations

Supported:

- the real-file transform and cache pipeline works end to end;
- the pinned model/checkpoint/compatibility contract works on the A10 VM;
- image and persistent-state trajectories evolve across frames;
- repeated and single-vs-multi-window extraction are exactly deterministic in
  the pinned environment;
- preliminary smoke throughput and storage are small enough for the next gate.

Not supported:

- the configured Debug tier is not complete because `chair`, train, and test
  have no selected windows;
- no segmentation or classification probe was trained or evaluated;
- the six homogeneous portrait windows cannot project varied-grid cache sizes;
- the 100-window runtime/storage gate remains pending;
- no pilot/full extraction is authorized by this run.

## Next action

Acquire only the deterministically selected RGB/mask entries for both
categories and all official splits, preferably through verified HTTP range
access to the official archives. Rebuild the manifest, repeat category-diverse
overlay review, and run the 100-window gate before approving the pilot.
