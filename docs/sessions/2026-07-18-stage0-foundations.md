# Session: Stage 0 data and representation foundations

- Date: 2026-07-18
- Author: Max Bershtman with Codex
- Branches: `codex/refine-project-stages`; `codex/curope-compat`
- Related issue/PR: Stage 0 foundations PR #2 (merged); compatibility PR pending
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

- `python -m pytest -q`: 24 passed.
- Strict Ruff checks (`E,F,I,B,UP,SIM`) and Ruff format checks passed.
- `python -m compileall -q src scripts tests` passed.
- All six Stage 0 CLI entry points returned valid help output.
- `git diff --check` and a local Markdown-link target audit passed.
- Every repository `README.md` was updated for the Stage 0 contract.
- Manual comparison with the pinned upstream recurrent path found and corrected
  the pose-token position to `(-1, -1)`; a pose-enabled regression test now
  guards it.
- Real CUDA preflight exposed upstream CUT3R issue #7 in cuRoPE. The project now
  carries an exact, provenance-checked `tokens.scalar_type()` compatibility
  patch and rejects any additional upstream changes.

Real CO3Dv2 and GPU extraction remain pending because this workspace contains
neither the dataset nor the released checkpoint and has no visible CUDA device.
Static type checkers were not available in the current environment; runtime
annotations, strict lint, compile checks, and tests were used instead.

## Human review of AI-assisted work

The team must review ADRs 0002 and 0003, the private CUT3R adapter contract, and
real debug RGB/mask overlays before accepting the decisions or launching the
large cache.

## Next step

Create/claim the Stage 0 GitHub issue, review this branch, then run the documented
debug and GPU preflight gates on real artifacts.
