# Project Status

Last updated: 2026-08-01

## Current phase

Stage 1 - Binary segmentation probe training on frozen representations.
Stage 0 Full-51 extraction is complete; Drive publication remains open.

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
- [x] Train and evaluate the Part-A DINOv2 segmentation probe on the Technion VM. Owner: Ron
- [x] Train and evaluate the Part-A CUT3R-random segmentation probe on the Technion VM. Owner: Ron
- [x] Train and evaluate the Part-A CUT3R-trained segmentation probe. Owner: Yam
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
- Part-A frozen-representation baselines are partially in: DINOv2 ViT-B/14 test macro-IoU 0.7922, CUT3R-random test macro-IoU 0.2134 (best-val heads, seed 20260729, n=101 test split). Details in [EXP-003](docs/experiments/EXP-003-part-a-segmentation-baselines.md).
- Resolved 2026-08-20: the CUT3R-trained artifact on Drive (folder `1aZ30CLLbw3Lwgs7KcLYNp4HYNi0uN2hI`) is now a correctly labelled `probe-features-v2` cache (re-shared between 2026-08-01 and 2026-08-11, by whoever re-shared it). Part-A frozen-representation baselines are now complete for all three backbones: DINOv2 test macro-IoU 0.7922, CUT3R-trained test macro-IoU 0.7402, CUT3R-random test macro-IoU 0.2298 (best-val heads, seed 20260729, n=101 test split). Details in [EXP-005](docs/experiments/EXP-005-part-a-cut3r-trained-unblocked.md).
- The `codebaseNdatapipeline-redesign` PR is in review and consolidates the codebase into a single-package monorepo. It (a) flattens `data_pipeline/` into the repo root (configs move from `data_pipeline/configs/stage0/` to `configs/stage0/`; Stage 0 tests/scripts/patches move to repo root; imports are unchanged — already root-relative `from src.…`), and (b) moves `segmentation_validation/` into `src/segmentation/` on top of Aviv & Lihi's PR #12 (probe head, dataset, and drivers are now importable via `src.segmentation.*`; the training entry point becomes `python -m src.segmentation.train_segmentation --config src/segmentation/configs/<backbone>.yaml`). A new reusable prompt `.github/prompts/fix_pr_comments.prompt.md` is also introduced. Teammates with in-flight branches should rebase after this PR merges. Details in [docs/sessions/2026-07-30-codebase-redesign.md](docs/sessions/2026-07-30-codebase-redesign.md).

## Milestones

| Milestone | Exit condition | Status |
|---|---|---|
| M0 - Foundations | Reproducible data manifest and embedding contract approved | In progress |
| M1 - Binary segmentation | Held-out evaluation with agreed baselines and metrics | In progress (all three Part-A baselines done; Part-B and data-expansion pillars remain) |
| M2 - Multiclass classification | Image-level and per-pixel approaches compared | Not started |
| M3 - Optional task | Go/no-go ADR accepted | Not started |
| M4 - Final delivery | Plots, architecture diagram, presentation, and report complete | Not started |

## Latest handoff

See the [current Full-51 cache handoff](docs/data/stage0-full51-cache-handoff.md),
the [Part A exact-feature and reconstruction audit](docs/sessions/2026-07-20-stage0-part-a-window-audit.md),
[EXP-001](docs/experiments/EXP-001-stage0-real-gpu-smoke.md),
[EXP-002](docs/experiments/EXP-002-stage0-complete-debug.md),
the [Part-A DINOv2 + CUT3R-random baseline session](docs/sessions/2026-08-01-part-a-baselines.md),
[EXP-003](docs/experiments/EXP-003-part-a-segmentation-baselines.md),
[EXP-005](docs/experiments/EXP-005-part-a-cut3r-trained-unblocked.md) (all three Part-A segmentation
baselines complete), and
the [Google Drive ↔ Technion VM data-transfer runbook](docs/project/DRIVE_TO_VM_RUNBOOK.md).
