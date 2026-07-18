# Project Status

Last updated: 2026-07-18

## Current phase

Stage 0 - repository and experiment foundations.

## Immediate objectives

- [ ] Confirm what the course expects by a random-initialized CUT3R baseline. Owner: Aviv
- [x] Propose the CO3Dv2 subset tiers, filtering criteria, official sequence splits, and deterministic window protocol. Owner: Max
- [x] Implement and test the manifest, transform, window, feature, and cache contracts on synthetic fixtures. Owner: Max
- [ ] Review and accept ADRs 0002 and 0003. Owner: team
- [ ] Run the debug manifest against real CO3Dv2 data. Owner: Max
- [ ] Run one-window and 100-window preflight extraction on the Technion GPU. Owner: Max
- [ ] Approve the shared data-to-embedding interface after real GPU validation. Owner: team
- [ ] Assign Stage 1 and Stage 2 owners after the sixth member joins.

## Blockers and open questions

- Exact six-person roster and GitHub handles are incomplete.
- ADR 0002 and ADR 0003 are proposed but require team review before becoming accepted scientific contracts.
- The real CO3Dv2 debug files and released CUT3R checkpoint are not present in this workspace.
- The Technion VM's CUDA environment, disk capacity, runtime, and cache-size projection have not yet been measured.
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

See [the Stage 0 foundations session](docs/sessions/2026-07-18-stage0-foundations.md).
