# Project Status

Last updated: 2026-07-18

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
- [ ] Extract and transfer Full-51 Part A. Owner: Max
- [ ] Extract and transfer Full-51 Part B. Owner: Max
- [ ] Approve the shared data-to-embedding interface after real GPU validation. Owner: team
- [ ] Assign Stage 1 and Stage 2 owners after the sixth member joins.

## Blockers and open questions

- Exact six-person roster and GitHub handles are incomplete.
- ADR 0002 and ADR 0003 are proposed but require team review before becoming accepted scientific contracts.
- The complete Debug tier produced 41 valid windows after one empty transformed
  target and its windowless sequence were excluded. Its verified 492 MiB cache
  ran at 0.464 seconds/window and peaked at 3,532,914,688 CUDA bytes.
- The combined Full-51 cache is projected near 96 GiB and cannot coexist on the
  VM disk. Part A and Part B must run sequentially, with each cache transferred
  to non-OneDrive local storage, hash-verified, and then removed from the VM.
- Five validation/test sequences per category provide limited independent
  per-category estimates; later reports must show counts and macro/micro views.
- Azure's `/mnt` resource disk is temporary and explicitly prohibited.
- The meaning and implementation of the random-initialization baseline requires clarification.
- Baseline models and comparison protocol are not yet selected.

## Milestones

| Milestone | Exit condition | Status |
|---|---|---|
| M0 - Foundations | Reproducible data manifest and embedding contract approved | In progress |
| M1 - Binary segmentation | Held-out evaluation with agreed baselines and metrics | Not started |
| M2 - Multiclass classification | Image-level and per-pixel approaches compared | Not started |
| M3 - Optional task | Go/no-go ADR accepted | Not started |
| M4 - Final delivery | Plots, architecture diagram, presentation, and report complete | Not started |

## Latest handoff

See the [latest durable handoff](docs/sessions/2026-07-18-stage0-debug-full51-handoff.md),
[EXP-001](docs/experiments/EXP-001-stage0-real-gpu-smoke.md), and
[EXP-002](docs/experiments/EXP-002-stage0-complete-debug.md).
