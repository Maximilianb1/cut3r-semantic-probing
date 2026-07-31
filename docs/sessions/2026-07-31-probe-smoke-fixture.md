# Session: Synthetic probe cache and run-output location

- Date: 2026-07-31
- Author: Aviv Rabi
- Branch: segmentation-validation-3
- Related issue/PR: follows PR #12 (probe scope) and #14 (repository redesign)
- Assistant/model, if used: Claude Code (Opus 5)

## Objective

Be able to run the Stage 1 probe end to end before any real probe-feature cache
exists, using fake embeddings in the real cache format, and settle where run
outputs land after the #14 redesign.

## Context and inputs

- `src/segmentation/` after #14 flattened `segmentation_validation/` into it.
- No probe-feature cache exists yet, so the training and evaluation drivers had
  never been executed against a cache on disk — only against in-memory fixtures.

## Work completed

- Files changed:
  - `scripts/make_synthetic_probe_cache.py` (new): writes a probe-feature cache of
    fake embeddings and fake masks. The task is learnable but not separable — a
    token is `±w + N(0, noise²)` for a fixed random direction `w` — so a linear
    probe lands between chance and perfect and the IoU code runs on
    non-degenerate values. Token grids vary window to window, as in the real
    caches, which is what exercises the variable-token-count collation. Splits are
    assigned at sequence level so the leakage guards see a valid cache. Some
    windows are deliberately foreground-free, to exercise the empty-target IoU
    convention.
  - `src/segmentation/dummy_embeddings/` (git-ignored, local only): three
    caches, 100 windows each, named exactly as the real configs expect
    (`probe/cut3r-trained` trajectory, `probe/cut3r-random` and
    `probe/dinov2-vitb14` target-only), so pointing `CUT3R_CACHE_ROOT` at the
    directory runs the real configs unchanged. Different grids and noise per
    backbone so a comparison produces visibly different numbers.
  - `src/segmentation/experiments/` (git-ignored, local only): run outputs moved
    here from `runs/`, one directory per experiment.
  - `src/segmentation/configs/*.yaml`: `output.dir` now
    `src/segmentation/experiments/<experiment>`. `probe_cache.dir` stays
    `${CUT3R_CACHE_ROOT}/probe/<backbone>`; each config carries a comment showing
    how to smoke-test against the fake caches by setting that variable instead of
    editing the file.
  - `.gitignore`: exclude both new directories. They are local scratch, so nothing
    inside them is tracked at all.
  - `scripts/make_synthetic_probe_cache.py`, argument handling (PR #16 review): bad
    values used to fail far from their cause - `--windows-per-sequence 0` as a
    ZeroDivisionError, `--windows 0` inside the cache verifier, `--grids 8-10` as an
    int() error - and two failed silently: `--feature-dim 0` wrote a cache of
    zero-width features, and `--categories 99` quietly used 51. A shared `_problems()`
    now reports every unusable value at once; the CLI turns them into `parser.error`
    (exit 2 with usage) and `build()` raises `ValueError`, so importers get the same
    message. `--empty-fraction` was also *wrong in range*: the old "every
    int(1/fraction)-th window" rule made 0.6 mean every window. The count is now exact
    and evenly spread.
  - `src/segmentation/inference_segmentation.py`: `inference-<split>.json` now records
    the cache dir and its metadata, as `metrics.json` already did. Without it an
    evaluation file could not be told apart from a real one - the `synthetic: true`
    marker was absent exactly where a reader would look for it.
  - `tests/test_synthetic_probe_cache.py` (new): 21 cases over the rejected values, the
    grid parser, and the exact empty fraction.
  - `src/segmentation/README.md`: documents where outputs land and how to smoke-test
    against the synthetic caches. Kept in the existing README rather than in per-
    directory READMEs, so there is one discoverable place and no duplication of the
    generator's own docstring.

- Artifacts produced: none committed. The dummy caches (97 MB) and every run
  directory are local and ignored.

## Decisions

- Made:
  - Committed configs point at `${CUT3R_CACHE_ROOT}`, never at the dummy path. A
    config with a hard-coded fake path would let a teammate or the VM train on
    synthetic bytes and produce a `metrics.json` that looks real.
  - The fixture is noisy on purpose. A perfectly separable fixture reaches IoU 1.0
    and would hide a broken metric; a noisy one has a computable ceiling to check
    against.
  - Run outputs are working files, not records. Promoting a result means writing
    it up under `docs/experiments/`, not committing a run directory.
- Still open (unchanged here): everything on the PR #12 list — linear vs MLP head,
  the two baseline ADRs, the cross-backbone comparison protocol, best-vs-final
  epoch checkpointing, `pos_weight`, and the empty-target IoU convention.

## Verification

| Command/check | Result |
|---|---|
| `python -m scripts.make_synthetic_probe_cache` x3 | 100 windows each; `verify_probe_cache` valid for both layouts |
| All three real configs, train + inference, via `CUT3R_CACHE_ROOT=src/segmentation/dummy_embeddings` | Run end to end; outputs land in `src/segmentation/experiments/<experiment>/` |
| Config without the env variable set | Raises `Unresolved environment variable in configuration: ${CUT3R_CACHE_ROOT}/probe/dinov2-vitb14` |
| Probe learns what is learnable | At `noise=1.5` the Bayes-optimal token accuracy is `Φ(1/1.5)` = 0.7475; a linear probe reached 0.7269 |
| Trivial baselines measured on the dummy test split | Foreground fraction 0.206; all-foreground macro IoU 0.207; all-background 0.05 |
| `git add -A` after generating everything | Stages only `.gitignore`, the generator, three configs, the README and the session note — no cache, no run output |
| Every rejected CLI value | Actionable `error:` message, exit code 2; all problems listed at once |
| `--empty-fraction` accuracy | 0.0/0.05/0.25/0.6/1.0 give exactly 0/5/25/60/100 empty windows out of 100 |
| `inference-<split>.json` provenance | Carries `synthetic: true` and the cache metadata |
| `python -m pytest -q` | 81 passed (60 before, plus 21 generator-validation cases) |

Nothing here is evidence about the research question: the embeddings are fake. The
caches stamp `synthetic: true` in `metadata.json`, which propagates into
`metrics.json`.

## Human review of AI-assisted work

Pending. Reviewer should confirm that no committed config points at
`dummy_embeddings`, and that no cache or run artifact is tracked.

## Next step

Two things surfaced that need a decision before the real run (Aviv):

1. **Overfitting is visible in every dummy run** — train macro IoU 1.0000 and train
   loss ~0.012 while val loss climbs, with `hidden_dims: [512]` over 20 epochs.
   `head.pt` saves the final epoch, so inference would evaluate the most overfit
   weights. Best-val checkpoint selection is the fix and is still open.
2. **A trivial baseline belongs in the metrics.** "Predict everything foreground"
   scores the foreground token fraction (0.207 on the fixture), so an absolute IoU
   is not interpretable on its own. Consider reporting it alongside the probe score.
