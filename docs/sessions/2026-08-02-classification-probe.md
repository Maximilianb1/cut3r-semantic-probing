# Session: Stage 2 classification probe

- Date: 2026-08-02
- Author: Aviv Rabi
- Branch: classification
- Related issue/PR: follows PR #16 (synthetic probe cache)
- Assistant/model, if used: Claude Code (Opus 5)

## Objective

Build the Stage 2 image-level classification probe with the same file structure as
Stage 1 segmentation, runnable end to end on synthetic caches before any real
probe-feature cache exists.

## Context and inputs

- `src/segmentation/` as merged in #12/#14/#16, used as the structural template.
- No probe-feature cache exists yet, so everything here was exercised against the
  synthetic generator from #16, extended for this stage.

## Work completed

- Files changed:
  - `src/classification/{model,dataset,train,inference}_classification.py` (new): the
    probe. One label per window, taken from the cache **index** (no tensor read); each
    window's tokens pooled to a single `[D]` vector, so batches are plain rectangles and
    need no custom collate. `default_collate` handles them.
  - `src/classification/visualizations.py` (new): confusion matrix, curves per feature
    source plus merged, per-category bars, a cross-run summary, and top confusions. All
    built from `metrics.json` / `inference-<split>.json`; nothing is simulated, and a run
    from a synthetic cache is stamped on the figure.
  - `src/classification/configs/{cut3r_trained,cut3r_random,dinov2}.yaml` (new).
  - `src/backbones/probe_cache.py`: added `load_target_features(..., kind="spatial"|
    "global")`. The existing fast loader reads grid tokens + mask only, so there was no
    way to read the state latents without pulling the whole 13.5 MiB entry.
  - `scripts/make_synthetic_probe_cache.py`: each category now gets its own direction, in
    both the grid tokens and the state latent, so one cache exercises both stages and
    both feature sources. New `--category-signal`.
  - `pyproject.toml`: **dependency added** — `matplotlib>=3.8,<4`. It was already imported
    by `src/segmentation/visualizations.py` but never declared, so a clean install could
    not run that module.
  - `tests/test_classification_{metrics,training}.py` (new): 23 cases.
  - READMEs for the package and for the figures directory.

- Artifacts produced: none committed. Dummy caches and every run directory and figure are
  git-ignored; only the figures README is tracked.

## Decisions

- Made:
  - **The head is duplicated, not shared with `src.segmentation`.** They answer different
    questions, and separating them means neither package's model changes through an edit
    meant for the other. Cost: "linear probe" is defined twice. Owner's call.
  - **`features.source` has no default** — `image_tokens` vs `state_tokens` is the open
    ADR 0003 comparison, so a run must state its arm; `metrics.json` and `head.pt` record
    it, and inference refuses a checkpoint trained on the other one.
  - **Output path derives from the arm**: `<output.dir>/<features.source>/<experiment>/`,
    so the two arms cannot overwrite each other.
  - **Output dim is the vocabulary size (51), always** — labels are indices into the fixed
    vocabulary, so a subset cache still yields index 40. Deriving it removed a config
    value that could silently be wrong.
  - **Standardization statistics come from the train split** and travel inside `head.pt`;
    inference reuses them rather than recomputing, which would leak the evaluated split
    into its own score.
  - **Imbalance is handled by `balanced_sampler` only.** Loss weighting was implemented,
    then removed at the owner's request: two corrections for one problem compound.
  - Metrics are accuracy, macro precision/recall/F1 and the per-category breakdown. Micro
    variants are *not* reported: in single-label multiclass they all equal accuracy.
- Still open:
  - Which arm heads the report — to be chosen on **val**, with test reported once.
  - Best-vs-final-epoch checkpointing (`head.pt` is the final epoch, and every synthetic
    run overfits hard).
  - Confidence is not recorded, so calibration figures are not yet possible.
  - Seeds/paired tests before any "A beats B" claim.

## Verification

| Command/check | Result |
|---|---|
| `python -m pytest -q` | 103 passed (81 before, plus 22 classification cases) |
| Full 2x3 grid (3 backbones x 2 feature sources) trained and evaluated on dummy caches | All six run end to end; outputs land under `experiments/<source>/<experiment>/` |
| Feature-spec mismatch | Evaluating an `image_tokens` checkpoint against a `state_tokens` config raises |
| Wrong `model.num_classes` | Rejected with the vocabulary-size message |
| Standardization | Train-split statistics saved in `head.pt`; standardized features have mean 0 / std 1; a probe without installed statistics refuses to run |
| Balanced sampler | With `apple: 6, ball: 2`, draws even out to ~1:1 |
| Figures | 22 files generated from the six runs; all stamped SYNTHETIC |
| `git add -A` | Stages 18 text files - no cache, run output, or image |

Every number produced in this session came from synthetic caches and means nothing about
the research question.

## Human review of AI-assisted work

Pending. Reviewer should confirm: the duplicated head is the wanted trade; no committed
config points at a dummy cache; and that reporting both feature arms (rather than
picking one silently) matches the intent of ADR 0003.

## Next step

Blocked on data, not code: the probe-feature caches still do not exist. Building them
needs CO3D RGB+masks and the Full-51 manifests, neither of which is on either VM (see
the 2026-08-01 VM setup). Once a cache exists, run the 2x3 grid over several seeds, pick
the arm on val, and report test once.
