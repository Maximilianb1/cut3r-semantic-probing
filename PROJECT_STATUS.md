# Project Status

Last updated: 2026-07-30

## Current phase

Stage 0 - Full-51 frozen representation extraction.

## Immediate objectives

- [ ] Confirm what the course expects by a random-initialized CUT3R baseline. Owner: Aviv
- [x] Propose the CO3Dv2 subset tiers, filtering criteria, official sequence splits, and deterministic window protocol. Owner: Max
- [x] Implement and test the manifest, transform, window, feature, and cache contracts on synthetic fixtures. Owner: Max
- [ ] Review and accept ADRs 0002 and 0003. Owner: team
- [x] Run a real CO3Dv2 smoke manifest and inspect target-mask overlays. Owner: Max
- [x] Run exact one-window reproducibility and six-window throughput preflights on the Technion GPU. Owner: Max
- [x] Obtain a real debug subset with both configured categories and official train/validation/test coverage. Owner: Max
- [x] Complete the valid-target Debug cache and record runtime, VRAM, and storage. Owner: Max
- [x] Extract, transfer, locally SHA-verify, and sample-audit Full-51 Part A. Owner: Max
- [x] Transfer and locally SHA-verify extracted Full-51 Part B. Owner: Max
- [ ] Publish both immutable cache roots and provenance to Max's private Drive folder shared read-only with the team. Owner: Max
- [ ] Approve the shared data-to-embedding interface after real GPU validation. Owner: team
- [ ] Assign Stage 1 and Stage 2 owners after the sixth member joins.

## Blockers and open questions

- Exact six-person roster and GitHub handles are incomplete.
- ADR 0002 and ADR 0003 are proposed but require team review before becoming accepted scientific contracts.
- The complete Debug tier produced 41 valid windows after one empty transformed
  target and its windowless sequence were excluded. Its verified 492 MiB cache
  ran at 0.464 seconds/window and peaked at 3,532,914,688 CUDA bytes.
- The combined Full-51 cache is 83.053 GiB and did not fit alongside both source
  payloads on the VM disk. Part A and Part B therefore ran sequentially; each
  cache was transferred to non-OneDrive local storage and SHA-verified.
- Actual extraction produced 3,667 Part A windows (43 GiB) and 3,458 Part B
  windows (41 GiB), for 7,125 windows across all 51 categories. Part A is
  locally verified and exactly reproduced in a six-frame GPU audit. Part B is
  cache-valid on the VM and all 111 transferred files match the published
  SHA-256 list locally. Both parts are ready for staged Drive publication.
- Five validation/test sequences per category provide limited independent
  per-category estimates; later reports must show counts and macro/micro views.
- Azure's `/mnt` resource disk is temporary and explicitly prohibited.
- The meaning and implementation of the random-initialization baseline requires clarification.
- Baseline models and comparison protocol are not yet selected.
- The `codebaseNdatapipeline-redesign` PR is in review and flattens `data_pipeline/` into the repo root. Teammates should rebase in-flight branches after it merges. Imports do not change (already root-relative `from src.…`); configs move from `data_pipeline/configs/stage0/` to `configs/stage0/`; Stage 0 tests/scripts/patches move to repo root. `segmentation_validation/` is intentionally not moved on this branch. A new reusable prompt `.github/prompts/fix_pr_comments.prompt.md` is also introduced. Details in [docs/sessions/2026-07-30-codebase-redesign.md](docs/sessions/2026-07-30-codebase-redesign.md).

## Milestones

| Milestone | Exit condition | Status |
|---|---|---|
| M0 - Foundations | Reproducible data manifest and embedding contract approved | In progress |
| M1 - Binary segmentation | Held-out evaluation with agreed baselines and metrics | Not started |
| M2 - Multiclass classification | Image-level and per-pixel approaches compared | Not started |
| M3 - Optional task | Go/no-go ADR accepted | Not started |
| M4 - Final delivery | Plots, architecture diagram, presentation, and report complete | Not started |

## Latest handoff

See the [current Full-51 cache handoff](docs/data/stage0-full51-cache-handoff.md),
the [Part A exact-feature and reconstruction audit](docs/sessions/2026-07-20-stage0-part-a-window-audit.md),
[EXP-001](docs/experiments/EXP-001-stage0-real-gpu-smoke.md), and
[EXP-002](docs/experiments/EXP-002-stage0-complete-debug.md).
