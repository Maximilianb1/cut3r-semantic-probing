# Stage 0 Debug completion and Full-51 handoff

- Date: 2026-07-18
- Owner: Max Bershtman with Codex
- Machine: Technion Azure VM `mcvgpu2025s-0047`
- Current milestone: Stage 0 Full-51 Part A acquisition/extraction

## Durable VM layout

The Azure `/mnt` resource disk was observed being reinitialized and must not be
used. Current durable paths are on the OS disk:

```text
/home/vmadmin/cut3r-stage0/repos/cut3r-semantic-probing
/home/vmadmin/cut3r-stage0/repos/CUT3R
/home/vmadmin/cut3r-stage0/datasets/co3dv2
/home/vmadmin/cut3r-stage0/checkpoints/cut3r_512_dpt_4_64.pth
/home/vmadmin/cut3r-stage0/cache
/home/vmadmin/cut3r-stage0/artifacts
```

The CUT3R checkout is commit
`8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf` plus the audited
`curope-scalar-type-v1` compatibility patch. The released checkpoint SHA-256 is
`45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`.
PyTorch 2.7.1+cu128 and the compiled cuRoPE extension passed CUDA tests on the
NVIDIA A10-24Q.

## Completed work

- PR #2: Stage 0 foundations, merged.
- PR #3: audited cuRoPE compatibility patch, merged.
- PR #4: hash-bound restricted checkpoint loading, merged.
- PR #5: real GPU engineering smoke record, merged.
- PR #6: selective downloader, valid-target filtering, and Full-51 configs;
  intended to be merged at the end of this session.
- Exact repeated one-window caches and single-vs-multi-window caches matched at
  zero tolerance for both image and state trajectories.
- The selective downloader safely acquired the complete Debug subset.
- The final Debug manifest contained 17 sequences, 430 frames, and 41 windows.
  One target mask became empty after exact CUT3R preprocessing; its window and
  now-windowless sequence were excluded before model loading.
- The final verified Debug cache is
  `/home/vmadmin/cut3r-stage0/cache/debug-valid-targets-v2`: 41 windows, 82
  tensors, six shards, 492 MiB, 19.035 seconds, 0.464 seconds/window, and
  3,532,914,688 peak allocated CUDA bytes.

## Accepted Full-51 decision

The project owner elected to skip the separate 10-category Pilot and cover all
51 categories in two sequential storage shards:

- Part A: 26 alternating alphabetic categories;
- Part B: the remaining 25 categories;
- caps per category: 30 train, 5 validation, 5 test sequences;
- up to four disjoint ordered six-frame windows per sequence;
- frame 6 remains the later supervised target;
- empty transformed target masks are excluded before GPU extraction.

The two category shards are not train/test splits. Official CO3D sequence
membership remains the only train/validation/test assignment. Later probe code
must load both caches and report window-micro, sequence-macro, category-macro,
and per-category results. Five validation/test sequences per category imply
noisy per-category estimates, so counts must accompany all metrics.

Upper projections from the Debug measurement:

| Part | Categories | Windows | Cache | Extraction only |
|---|---:|---:|---:|---:|
| A | 26 | 4,160 | about 49 GiB at the Debug grid mix | about 32 minutes |
| B | 25 | 4,000 | about 47 GiB at the Debug grid mix | about 31 minutes |

Actual totals will be smaller when categories lack the cap or targets are
invalid, but varied aspect ratios can increase image-token bytes per window.
The runbook therefore performs an exact manifest-token-grid projection with 5%
format overhead and a 10 GiB free-space reserve before extraction. Selective
download can take much longer than GPU extraction because it performs many HTTP
range requests.

## Next action

Follow [the two-part runbook](../project/FULL51_TWO_PART_RUNBOOK.md). Start Part
A inside `tmux`, leave the fail-fast command running overnight, and inspect its
logs after reconnecting. After completion, validate the cache, copy it to a
non-OneDrive Windows path with at least 120 GiB free, verify every file against
the generated SHA-256 list, and only then delete the Part A cache from the VM.

Do not delete manifests, logs, download provenance, or dataset files. Do not
combine Part A and Part B directories by copying shards together.
