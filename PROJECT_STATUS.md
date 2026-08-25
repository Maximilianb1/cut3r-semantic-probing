# Project Status

Last updated: 2026-08-25

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
- [x] Run the Part-A segmentation probe-capacity ablation (linear vs. `[512]` MLP head) across all three backbones. Owner: Yam
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
- Resolved 2026-08-20: the CUT3R-trained artifact on Drive (folder `1aZ30CLLbw3Lwgs7KcLYNp4HYNi0uN2hI`) is now a correctly labelled `probe-features-v2` cache. Part-A frozen-representation baselines are now complete for all three backbones: DINOv2 test macro-IoU 0.7922, CUT3R-trained test macro-IoU 0.7402, CUT3R-random test macro-IoU 0.2298 (best-val heads, seed 20260729, n=101 test split). Details in [EXP-005](docs/experiments/EXP-005-part-a-seg-cut3r-unblocked.md).
- Resolved 2026-08-22: expanded training (leftover + cap100-new-train batches added to the training set, ~3.3x more train windows, val/test held fixed) complete for all three backbones. All three improved: DINOv2 0.7922 -> 0.8063, CUT3R-trained 0.7402 -> 0.7772, CUT3R-random 0.2298 -> 0.2772 (test macro-IoU, best-val). CUT3R-random's worst-5 qualitative failures are unchanged (still complete 0.000-IoU misses) despite the extra data, pointing at the representation itself rather than data volume as its limiting factor. Details in [EXP-006](docs/experiments/EXP-006-part-a-seg-expanded-training.md).
- Resolved 2026-08-25: completed the probe-capacity ablation (linear `hidden_dims: []` vs. the `[512]` MLP head) that every segmentation config has flagged as an open decision since 2026-07-30, holding backbone and expanded-training data fixed. All three backbones keep a statistically significant fraction of their MLP score under a linear-only head, but by very different relative margins: CUT3R-trained -4.8%, DINOv2 -8.4%, CUT3R-random -42.2% (paired bootstrap 95% CIs all exclude zero, n=101 shared test windows). Per-category, CUT3R-trained's and DINOv2's drops are roughly uniform across categories; CUT3R-random's drop concentrates in six categories (carrot, ball, plant, sandwich, cake, chair) that collapse toward ~0 IoU under the linear head. This is the first direct evidence that CUT3R-trained's segmentation signal is the most linearly ("implicitly") decodable of the three, even though DINOv2 still has the higher raw MLP score. Details in [EXP-007](docs/experiments/EXP-007-part-a-seg-probe-capacity-ablation.md) and `results/pillar-c/`; which head capacity Stage 1's headline comparison should report is still an open decision (see EXP-007's Next action).
- The `codebaseNdatapipeline-redesign` PR is in review and consolidates the codebase into a single-package monorepo. It (a) flattens `data_pipeline/` into the repo root (configs move from `data_pipeline/configs/stage0/` to `configs/stage0/`; Stage 0 tests/scripts/patches move to repo root; imports are unchanged — already root-relative `from src.…`), and (b) moves `segmentation_validation/` into `src/segmentation/` on top of Aviv & Lihi's PR #12 (probe head, dataset, and drivers are now importable via `src.segmentation.*`; the training entry point becomes `python -m src.segmentation.train_segmentation --config src/segmentation/configs/<backbone>.yaml`). A new reusable prompt `.github/prompts/fix_pr_comments.prompt.md` is also introduced. Teammates with in-flight branches should rebase after this PR merges. Details in [docs/sessions/2026-07-30-codebase-redesign.md](docs/sessions/2026-07-30-codebase-redesign.md).

## Milestones

| Milestone | Exit condition | Status |
|---|---|---|
| M0 - Foundations | Reproducible data manifest and embedding contract approved | In progress |
| M1 - Binary segmentation | Held-out evaluation with agreed baselines and metrics | In progress (baseline + expanded-training + probe-capacity-ablation results done for all three Part-A backbones; the CO3D Part-B data half and further head/architecture improvements remain) |
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
[EXP-005](docs/experiments/EXP-005-part-a-seg-cut3r-unblocked.md) (all three Part-A segmentation
baselines complete),
[EXP-006](docs/experiments/EXP-006-part-a-seg-expanded-training.md) (expanded-training
results for all three backbones),
[EXP-007](docs/experiments/EXP-007-part-a-seg-probe-capacity-ablation.md) (linear-vs-MLP
probe-capacity ablation, `results/pillar-c/`),
[reports/segmentation/](reports/segmentation/README.md) (committed, slide-ready
figures and metrics for EXP-005/006/007), and
the [Google Drive ↔ Technion VM data-transfer runbook](docs/project/DRIVE_TO_VM_RUNBOOK.md).
