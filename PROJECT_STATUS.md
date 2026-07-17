# Project Status

Last updated: 2026-07-17

## Current phase

Stage 0 - repository and experiment foundations.

## Immediate objectives

- [ ] Confirm what the course expects by a random-initialized CUT3R baseline. Owner: Aviv
- [ ] Define the CO3D subset, filtering criteria, and official train/validation/test split. Owner: Max
- [ ] Define the exact CUT3R representation(s) to extract and cache. Owner: Max
- [ ] Approve the shared data-to-embedding interface. Owner: team
- [ ] Assign Stage 1 and Stage 2 owners after the sixth member joins.

## Blockers and open questions

- Exact six-person roster and GitHub handles are incomplete.
- Representation choice is intentionally undecided: persistent state tokens versus final state-conditioned image tokens must not be conflated.
- The meaning and implementation of the random-initialization baseline requires clarification.
- Baseline models and comparison protocol are not yet selected.

## Milestones

| Milestone | Exit condition | Status |
|---|---|---|
| M0 - Foundations | Reproducible data manifest and embedding contract approved | Not started |
| M1 - Binary segmentation | Held-out evaluation with agreed baselines and metrics | Not started |
| M2 - Multiclass classification | Image-level and per-pixel approaches compared | Not started |
| M3 - Optional task | Go/no-go ADR accepted | Not started |
| M4 - Final delivery | Plots, architecture diagram, presentation, and report complete | Not started |

## Latest handoff

See [the initial repository session](docs/sessions/2026-07-17-initial-repository.md).
