# Session: Stage 0 data and representation foundations

- Date: 2026-07-18
- Author: Max Bershtman with Codex
- Branches: `codex/refine-project-stages`; `codex/curope-compat`;
  `codex/checkpoint-safe-load`
- Related issue/PR: Stage 0 foundations PR #2, compatibility PR #3, and
  checkpoint safe-loading PR #4 merged
- Assistant/model: Codex

## Objective

Implement the agreed inference-only Stage 0 foundation: official CO3Dv2
sequence splits, deterministic six-frame windows, CUT3R-aligned transforms,
all-timestep image/state extraction, and verified external caching.

## Work completed

- Added proposed ADRs 0002 and 0003.
- Added debug, pilot, and all-category configurations.
- Implemented official set-list parsing, SHA-ranked sequence caps, six-frame
  uniform/disjoint windows, Parquet manifests, summaries, and leakage checks.
- Implemented shared RGB/mask transform geometry matching released CUT3R
  preprocessing and recorded every per-frame transform in the manifest.
- Implemented an adapter around pinned CUT3R recurrent internals that saves
  final image tokens and committed persistent state after all six timesteps.
- Implemented atomic, resumable, hashed Safetensors shards with Parquet index
  and independent cache-to-cache tensor comparison. Index entries bind each
  window to exact RGB/mask SHA-256 values; extraction requires the pinned
  upstream commit plus the exact versioned compatibility patch.
- Added synthetic CO3Dv2, transform, window, fake-model trajectory, and cache
  tests.
- Documented evaluation aggregation and the credential-free Technion VM runbook.

## Decisions

- Stage 0 performs no probe training or CUT3R fine-tuning.
- Frame 6 remains the primary later target while all six feature timesteps are
  cached.
- Full extraction aims at all 51 categories with initial 50/10/10 sequence caps.
- Overall window, sequence-macro, category-macro, and per-category metrics will
  all be reported in later stages.

## Verification

- `python -m pytest -q`: 27 passed.
- Strict Ruff checks (`E,F,I,B,UP,SIM`) and Ruff format checks passed.
- `python -m compileall -q src scripts tests` passed.
- All seven Stage 0 CLI entry points returned valid help output.
- `git diff --check` and a local Markdown-link target audit passed.
- Every repository `README.md` was updated for the Stage 0 contract.
- Manual comparison with the pinned upstream recurrent path found and corrected
  the pose-token position to `(-1, -1)`; a pose-enabled regression test now
  guards it.
- Real CUDA preflight exposed upstream CUT3R issue #7 in cuRoPE. The project now
  carries an exact, provenance-checked `tokens.scalar_type()` compatibility
  patch and rejects any additional upstream changes.
- Real checkpoint preflight exposed PyTorch 2.6+'s restricted-loader default.
  The official checkpoint loaded with all state keys matching after its exact
  SHA-256 and seven static OmegaConf globals were verified. The project now
  enforces that hash-bound scoped loading policy without disabling weights-only
  loading globally.
- The course VM later stopped/recreated and reinitialized Azure's temporary
  `/mnt` resource disk, removing the reproducible repository clones, compiled
  extension, and downloaded checkpoint. No unique dataset, cache, or result had
  been produced. The runbook now prohibits persistent artifacts on the resource
  disk; the 79 GB free OS disk is reserved for code/checkpoint/control artifacts
  while approved persistent large-data storage is requested.
- A storage credential was exposed in diagnostic output. Its value was not
  recorded in the project; course-administrator rotation is required.
- The environment and checkpoint were rebuilt on the persistent OS disk. PR #4's
  scoped checkpoint policy loaded all 793,307,858 parameters on CUDA with the
  exact released checkpoint hash and all state keys matching.
- The official single-sequence archives for `ball` and `chair` were combined
  with their verified full-release metadata archives. The valid real manifest
  contained two `ball` validation sequences, 404 frames, and six windows; it
  contained no usable `chair`, train, or test windows and is explicitly
  `smoke-only`.
- Human review of all six target overlays found aligned masks, preserved aspect
  ratios, complete foreground objects, and no empty masks for the available
  ball frames.
- Two independent one-window extractions matched exactly at zero tolerance for
  every image/state value. Both caches passed hash, schema, shape, dtype,
  finiteness, and source-file checks.
- A six-window cache also reproduced the independently extracted first window
  exactly. It ran at 0.568 seconds/window, peaked at 3,454,282,752 CUDA bytes,
  and occupied 71 MiB. Full details are in
  [EXP-001](../experiments/EXP-001-stage0-real-gpu-smoke.md).

The local Windows workspace still has no dataset, checkpoint, or CUDA device;
real artifacts remain external on the Technion VM. The next data requirement is
a balanced `ball`/`chair` subset with nonempty official train, validation, and
test splits. Static type checkers were not available in the local environment;
runtime annotations, strict lint, compile checks, and tests were used instead.

## Human review of AI-assisted work

The team must review ADRs 0002 and 0003, the private CUT3R adapter contract, and
real debug RGB/mask overlays before accepting the decisions or launching the
large cache.

## Next step

This early-session next step was completed. Continue from the
[latest Full-51 handoff](2026-07-18-stage0-debug-full51-handoff.md).
