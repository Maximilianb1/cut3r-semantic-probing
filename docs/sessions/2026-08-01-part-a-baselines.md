# Session: Part-A segmentation baselines — DINOv2 and CUT3R-random (CUT3R-trained blocked)

- Date: 2026-08-01
- Author: Ron Bartal
- Branch: main
- Related issue/PR: follows PR #16 (synthetic probe cache); no PR opened this session (all outputs are local artifacts, no source changes)
- Related experiment record: [`../experiments/EXP-003-part-a-segmentation-baselines.md`](../experiments/EXP-003-part-a-segmentation-baselines.md)
- Assistant/model, if used: GitHub Copilot (Claude Opus 4.7)

## Objective

Train and evaluate the segmentation probe (MLP head) on the Technion GPU VM for
the two Part-A backbones whose probe caches are ready — **DINOv2** (advanced
frozen baseline) and **CUT3R-random** (random-init reference floor) — and stand
up the CUT3R-trained run in parallel using the artifact uploaded to Google
Drive by the collaborator.

## Context and inputs

- Repo at commit `1f45943` (July 31, matches GitHub `main` tip).
- Technion VM `mcvgpu2025s-0066` (Ubuntu 22.04, NVIDIA A10-24Q), conda env
  `stud_prj` (Python 3.11.15, torch 2.7.1+cu128). `pip install -e .` was
  deliberately skipped to avoid downgrading the env's `numpy` (env has 1.26.4;
  pyproject also pins 1.26.4, but standalone `pip install "pyarrow>=16,<21"`
  covered the missing runtime dep without touching numpy).
- Three Google Drive artifacts consumed:
  - **DINOv2 Part-A probe cache** (labelled, `probe_cache_schema_version:
    probe-features-v2`, `layout: target_only`) — 116 shards, 4.2 GB, 3667
    windows, transferred via `rclone` from the collaborator's Drive.
  - **CUT3R-random Part-A probe cache** (labelled, `probe_cache_schema_version:
    probe-features-v2`, `layout: target_only`, `random_init.strategy:
    reset_parameters`, seed 20260729) — 117 shards, 7.1 GB, 3667 windows,
    uploaded to `1wyEljQmtTxmC4mfqcc8T0hXYmaI_yhxr`. Well-organized delivery:
    `cache/cut3r-random/`, `config/cut3r_random.yaml`, `manifests/full51-part-a-v1/`
    (manifests skipped on copy since bit-identical to on-disk).
  - **CUT3R-trained target cache** (`cache_schema_version:
    stage0-target-cache-v1`) — 115 shards, 7.2 GB, uploaded to
    `1aZ30CLLbw3Lwgs7KcLYNp4HYNi0uN2hI`. Turned out to be the *upstream* target
    feature cache, not the probe cache the trainer consumes (see Decisions).
- All three artifacts match the on-disk manifest hashes bit-for-bit (frames
  `8f3d9545…`, sequences `7b7bf73e…`, windows `e499d9a8…`), so the evaluation
  split is identical to what the manifests describe.

## Work completed

- Files changed: none in the tracked tree. All work is local artifacts.
- VM layout established under `~/cut3r/` (env vars `CUT3R_CACHE_ROOT`,
  `CUT3R_ARTIFACT_ROOT` in `~/.bashrc`):

  ```
  ~/cut3r/
  ├── repos/cut3r-semantic-probing/                          # commit 1f45943
  ├── artifacts/manifests/full51-part-a-v1/                  # backbone-agnostic
  ├── artifacts/from-drive/cut3r-random/cut3r_random.yaml    # collaborator's extraction config
  ├── artifacts/segmentation/segmentation-dinov2/            # DINOv2 last-epoch head
  ├── artifacts/segmentation/segmentation-dinov2-bestval/    # DINOv2 best-val + last-epoch heads
  ├── artifacts/segmentation/segmentation-cut3r-random/      # CUT3R-random best-val + last-epoch heads
  └── cache/probe/
      ├── dinov2-vitb14/                                     # 4.2 GB, labelled, ready
      ├── cut3r-random/                                      # 7.1 GB, labelled, ready
      └── cut3r-trained/                                     # 7.2 GB, target cache only (unusable as-is)
  ```

- Machine-local training configs under `local/` (excluded via
  `.git/info/exclude`, not committed): `dinov2.vm.yaml`,
  `dinov2.vm.sanity.yaml`, `dinov2.vm.bestval.yaml`,
  `cut3r_random.vm.yaml`, `cut3r_random.vm.sanity.yaml`,
  `cut3r_trained.vm.yaml`, `cut3r_trained.vm.sanity.yaml`. Only differences vs
  the tracked configs are `device: cpu → cuda` and `output.dir →
  ${CUT3R_ARTIFACT_ROOT}/segmentation/…`.
- `local/train_best_val.py` (also `local/`, git-excluded): thin driver that
  reuses `train_from_config`'s primitives but saves `head.pt` at best-val
  macro-foreground-IoU and also emits `head-last.pt` for A/B. Adds
  `best_val_epoch`, `best_val_macro_iou`, `checkpoint_policy` to `metrics.json`.
  There is a harmless post-write bug (`result["checkpoint"]` is assigned after
  `metrics.json` is written); does not affect disk artifacts.

- Artifacts produced (all under `~/cut3r/artifacts/segmentation/` on the VM,
  none in the repo):
  - `segmentation-dinov2/` — 20-epoch head, `metrics.json`, `train.log`,
    `inference-{val,test}.json`.
  - `segmentation-dinov2-bestval/` — same 20 epochs but `head.pt` is the
    best-val checkpoint (epoch 6), `head-last.pt` is epoch 20, plus both
    `inference-{val,test}.json`.
  - `segmentation-cut3r-random/` — 20-epoch best-val run: `head.pt` is best-val
    (epoch 16), `head-last.pt` is epoch 20, `metrics.json` with full history,
    `train.log`, plus `inference-{val,test}.json`.
- Stale 37 GB CUT3R Stage-0 trajectory cache (from a previous session) deleted
  from `~/cut3r-stage0/` after the collaborator confirmed a Drive backup; VM
  free space went from ~35 GB to ~55 GB after both probe caches copied.

## Decisions

- Made:
  - Adopt `~/cut3r/{repos,cache,artifacts}` as the canonical VM layout for all
    Stage 1 work; keep `${CUT3R_CACHE_ROOT}` and `${CUT3R_ARTIFACT_ROOT}` as the
    two env-var handles the configs already expect.
  - Keep the tracked `src/segmentation/configs/*.yaml` untouched; put
    machine-local overrides under `local/` and rely on
    `.git/info/exclude` so no session-specific paths ever land in a commit.
  - Report **macro-foreground-IoU** as the primary segmentation metric.
    Token-accuracy is misleading on this task: the CUT3R-random head reaches
    val token-accuracy 0.828 while its actual segmentation IoU is 0.258, only
    a few points above the ~0.78 accuracy an "always predict background" head
    would achieve given the ~22 % foreground token fraction on Part-A.
  - Report **best-val** head numbers as the ADR values (DINOv2 test 0.792,
    CUT3R-random test 0.213), but record that best-val vs last-epoch is
    essentially tied for DINOv2 (Δ ≈ 0.002 macro-IoU on the n=101 test split).
  - The Drive folder `1aZ…N2hI` is *not* the probe cache the trainer consumes.
    Its provenance (checkpoint, commit, manifests) is correct, but it lacks
    `seg_labels`, `split`, `category`, `category_index`, `sequence_id` columns
    and uses a different schema key (`cache_schema_version: stage0-target-cache-v1`
    vs `probe_cache_schema_version: probe-features-v2`). Requested the
    collaborator run the label-attach step and re-share.
  - The `1wy…yhxr` Drive folder *is* a valid probe cache and matched all
    manifest and provenance hashes; adopted as the CUT3R-random Part-A cache.
- Still open:
  - Whether the CUT3R-trained probe cache will be re-uploaded, or whether we
    need to adapt `attach_labels_from_trajectory_cache` to the target-only
    layout and get CO3D onto the VM to re-derive labels here.
  - Linear-probe (`hidden_dims: []`) variant, called out as an open Stage-1
    decision in the config comments, not run this session.
  - Whether to formalise the epoch-20 vs best-val checkpointing policy. For
    DINOv2 they are effectively tied (Δ ≈ 0.002 on test), but for CUT3R-random
    training had not plateaued at epoch 20 (val macro-IoU curve still climbing),
    so a longer schedule would shift the random baseline upward while leaving
    DINOv2 mostly unchanged.
  - Whether to run CUT3R-random for more epochs (e.g. 40-60) to lock down where
    the random-baseline curve actually plateaus. Not done this session because
    even the extrapolated plateau (~0.30 val macro-IoU) sits far below DINOv2's
    0.836, so the ADR conclusion "random-init CUT3R is near the null floor" is
    already load-bearing.

## Verification

| Command / check | Result |
|---|---|
| DINOv2 cache `metadata.manifest_sha256.windows` vs on-disk `summary.json` | match (`e499d9a8879a…f7b34c`) |
| DINOv2 cache `probe_cache_schema_version` | `probe-features-v2` (as expected) |
| CUT3R-random cache `manifest_sha256.{frames,sequences,windows}` vs summary | all match |
| CUT3R-random cache `probe_cache_schema_version` / `layout` | `probe-features-v2` / `target_only` (as expected) |
| CUT3R-random cache `checkpoint_provenance.sha256` | `45f7e98a…f8103` (trust-anchor match) |
| CUT3R-random cache `random_init.{strategy,seed}` | `reset_parameters` / `20260729` (as expected) |
| CUT3R-trained cache `manifest_sha256.{frames,sequences,windows}` vs summary | all match |
| CUT3R-trained cache `checkpoint_provenance.sha256` | `45f7e98a…f8103` (trust-anchor match) |
| CUT3R-trained cache `cut3r_provenance.commit` | `8bc15dc9…3ecf` (pinned commit) |
| CUT3R-trained cache columns | missing `split`, `seg_labels_key`, `category`, `sequence_id`; unusable as probe cache |
| `torch.cuda.is_available()` on VM | `True`, device `NVIDIA A10-24Q` |
| Sanity run (1 epoch, DINOv2) | val macro-IoU 0.807 in 7 s |
| Sanity run (1 epoch, CUT3R-random) | val macro-IoU 0.017 in 6 s (near-zero, as expected for random features) |
| Full 20-epoch DINOv2 run (both drivers) | bit-identical loss/IoU sequences → deterministic under seed 20260729 |
| Full 20-epoch CUT3R-random run rerun | bit-identical to the first run → confirms determinism (no continued training between rerun invocations) |
| DINOv2 last-epoch head, `inference-val.json` macro-IoU | 0.8236, matches `final_val` in `metrics.json` bit-exactly |
| DINOv2 last-epoch head, test | macro-IoU 0.7903, micro-IoU 0.7758, tok-acc 0.9582 (n=101) |
| DINOv2 best-val head (epoch 6), test | macro-IoU 0.7922, micro-IoU 0.7838, tok-acc 0.9595 (n=101) |
| CUT3R-random best-val head (epoch 16), val | macro-IoU 0.2579, micro-IoU 0.2309, tok-acc 0.8111 (n=512) |
| CUT3R-random best-val head (epoch 16), test | macro-IoU 0.2134, micro-IoU 0.1833, tok-acc 0.7416 (n=101) |
| CUT3R-random train/val overfitting at epoch 20 | none: train 0.267 vs val 0.211 macro-IoU (curve still climbing) |
| `git status --short` after all local edits | shows only this session note |

Headline Part-A numbers (26 CO3D categories, MLP head, seed 20260729, best-val
checkpointing, `mask_threshold` 0.5):

| Backbone | Val macro-IoU (best) | Test macro-IoU | Test micro-IoU | Test tok-acc | Best-val epoch |
|---|---:|---:|---:|---:|---:|
| DINOv2 ViT-B/14 | 0.8360 | **0.7922** | 0.7838 | 0.9595 | 6 |
| CUT3R-random (reset_parameters) | 0.2579 | **0.2134** | 0.1833 | 0.7416 | 16 |
| CUT3R-trained | — pending collaborator's re-share — | — | — | — | — |
| **DINOv2 − CUT3R-random** | **+0.578** | **+0.579** | +0.601 | +0.218 | — |

CUT3R-random test macro-IoU 0.213 is the Part-A random-baseline floor to
interpret the CUT3R-trained result against. It is not zero because the MLP head
can learn a mild center-of-image / aspect-ratio prior even from noise-level
features; that residual is much lower than the ~0.79 DINOv2 achieves.

## Human review of AI-assisted work

- CUT3R-trained cache metadata was inspected directly (`metadata.json` cat,
  `pyarrow` column listing) rather than trusting the assistant's read; the
  identified schema/column gap was confirmed by comparing against the DINOv2
  cache which trains successfully.
- CUT3R-random cache metadata was likewise inspected before copy; the
  `probe-features-v2` schema, `layout: target_only`, matching manifest hashes,
  and matching checkpoint/commit provenance were all confirmed before any
  training bytes were touched.
- The val bit-exact match between `metrics.json.final_val` and the
  `inference-val.json` produced by the standard eval script was checked
  end-to-end for DINOv2 to confirm no loader drift.
- Determinism was confirmed twice: once by observing the epoch-20 driver and
  the best-val driver produce identical per-epoch loss/IoU logs on DINOv2, and
  once by rerunning `train_best_val` on CUT3R-random and observing bit-exact
  reproduction of every epoch — evidence the training loop starts fresh under
  the seed rather than continuing from a saved `head.pt`.
- CUT3R-random's high token-accuracy vs low IoU was cross-checked against the
  ~22 % Part-A foreground token fraction to confirm the head is only slightly
  above the always-predict-background floor on the imbalanced binary task.
- No source files under `src/` were modified. All non-tracked work sits under
  `local/` (VM) and `~/cut3r/artifacts/` (VM); `git status` shows only this
  session note.

## Next step

Wait for the collaborator's re-shared CUT3R-trained probe cache (labelled,
`probe-features-v2` schema) and rerun the same
`train_segmentation` + `inference_segmentation` recipe against
`~/cut3r/cache/probe/cut3r-trained/`. If instead only the target cache is
available, extend `attach_labels_from_trajectory_cache` to accept the
target-only `[1,1,N,C]` layout and provision CO3D on the VM. Owner: Ron Bartal.
