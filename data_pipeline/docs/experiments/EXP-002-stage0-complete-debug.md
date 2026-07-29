# EXP-002: Complete Stage 0 Debug extraction

- Date: 2026-07-18
- Owner: Max Bershtman with Codex
- Status: Completed
- Configuration: `configs/stage0/debug.yaml`
- Dataset: CO3Dv2 `v2_231130`

## Objective

Complete the two-category, official-split Debug tier after the earlier
single-category engineering smoke, then measure real cache runtime, VRAM, and
storage before choosing the all-category extraction strategy. No probe was
trained and no scientific test metric was computed.

## Data and manifest result

The bounded selective downloader acquired the deterministic `ball` and `chair`
RGB/mask subset through official ZIP byte ranges. The initial manifest proposed
42 windows. Exact CUT3R preprocessing made target
`ball/563_81372_161289:196` empty at the 0.5 threshold, so the corrected
manifest builder excluded that window and its now-windowless sequence before
GPU work.

Final manifest counts:

- selected sequences: 17;
- selected frames: 430;
- windows: 41;
- `empty_transformed_target_mask`: 1;
- `no_valid_target_windows`: 1.

The 430 frame count exceeds the downloader's minimum required-frame count
because one already-present selected sequence had its complete frame set. This
does not change its deterministic windows.

## GPU and cache result

- GPU: NVIDIA A10-24Q;
- windows written/skipped: 41/0;
- image/state tensors: 82;
- shards: 6;
- extraction time: 19.035263829 seconds;
- mean time: 0.464274728 seconds per written window;
- peak allocated CUDA memory: 3,532,914,688 bytes;
- rounded cache size: 492 MiB;
- cache validation: passed;
- cache index SHA-256:
  `e0776d236d75d8622f324dd0351e79dcc9c72bf53e359cbfb96b64f6586942af`;
- cache metadata SHA-256:
  `e43beefd3b36f5c75d1ba08d598b8674bd5fd52d772010722f3f152f7e178c1d`.

## Decision enabled by this run

Measured storage is approximately 12 MiB/window. The owner chose to skip the
10-category Pilot and extract all 51 categories with 30/5/5
train/validation/test sequence caps in two sequential execution shards. Upper
projections are 4,160 windows/~49 GiB for Part A and 4,000 windows/~47 GiB for
Part B. Each cache must be copied off the VM, hash-verified, and only then
deleted before the next part.
