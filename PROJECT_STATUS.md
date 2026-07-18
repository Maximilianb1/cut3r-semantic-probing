# Project Status

Last updated: 2026-07-18

## Current phase

Stage 0 - repository and experiment foundations.

## Immediate objectives

- [ ] Confirm what the course expects by a random-initialized CUT3R baseline. Owner: Aviv
- [x] Propose the CO3Dv2 subset tiers, filtering criteria, official sequence splits, and deterministic window protocol. Owner: Max
- [x] Implement and test the manifest, transform, window, feature, and cache contracts on synthetic fixtures. Owner: Max
- [ ] Review and accept ADRs 0002 and 0003. Owner: team
- [x] Run a real CO3Dv2 smoke manifest and inspect target-mask overlays. Owner: Max
- [x] Run exact one-window reproducibility and six-window throughput preflights on the Technion GPU. Owner: Max
- [ ] Obtain a real debug subset with both configured categories and nonempty train/validation/test splits. Owner: Max
- [ ] Run the 100-window performance preflight after the balanced debug/pilot data is available. Owner: Max
- [ ] Approve the shared data-to-embedding interface after real GPU validation. Owner: team
- [ ] Assign Stage 1 and Stage 2 owners after the sixth member joins.

## Blockers and open questions

- Exact six-person roster and GitHub handles are incomplete.
- ADR 0002 and ADR 0003 are proposed but require team review before becoming accepted scientific contracts.
- The official CO3Dv2 single-sequence subset plus verified full split metadata
  produced a valid engineering smoke manifest, but only two `ball` validation
  sequences had local usable frames: 404 frames and six windows. `chair`, train,
  and test coverage remain absent, so this cache cannot be used for probe
  training, category comparison, or scientific evaluation.
- Real Technion A10-24Q extraction is bit-for-bit reproducible for both image
  and state trajectories. The six-window run took 3.409 seconds (0.568
  seconds/window), peaked at 3,454,282,752 CUDA bytes, and occupied 71 MiB.
  These smoke measurements do not replace the required 100-window projection.
- The VM has about 76 GB free on its persistent OS disk, enough for code, the released
  checkpoint, manifests, logs, and small preflight caches. CO3Dv2 and large
  caches require an approved managed disk or student-writable persistent share;
  Azure's `/mnt` resource disk is explicitly prohibited.
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

See [the Stage 0 foundations session](docs/sessions/2026-07-18-stage0-foundations.md)
and [the real GPU smoke record](docs/experiments/EXP-001-stage0-real-gpu-smoke.md).
