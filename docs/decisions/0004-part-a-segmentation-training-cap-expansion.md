# ADR 0004: Part-A segmentation training-cap expansion

- Status: Proposed
- Date: 2026-08-25
- Owners: Yam Ben-Tov; project team review required
- Related issue/PR: #21

## Context

ADR 0002 caps the Full configuration at 30/5/5 train/val/test sequences per
category. EXP-005 trained Part-A segmentation probes (DINOv2, CUT3R-trained,
CUT3R-random) on that 30-sequence-per-category training partition. Two more
batches became available for the same manifest family: a "leftover" batch
(extra windows from previously-unused frames of the *same* 30 training
sequences) and a "cap100-new-train" batch (1,762 brand-new training
sequences, raising the effective per-category training cap from 30 to up to
100). EXP-006 trained the same unchanged heads on the union of all three
train partitions, holding val/test fixed on the original 5/5-sequence cache
so the score stayed comparable to EXP-005. This raises the training cap
beyond what ADR 0002 documents, without an ADR recording the change.

## Decision criteria

- must not cross the official train/dev/test sequence boundary ADR 0002
  established — val/test scope stays exactly as ADR 0002 defined it;
- must remain deterministic and reproducible (same selection rule as
  ADR 0002, no filesystem-order dependence);
- must fit the Technion VM's storage/compute budget;
- must be empirically justified, not applied on assumption.

## Options considered

### Keep the cap at 30 sequences/category

No ADR update needed, but leaves Part-A segmentation training data
artificially small relative to what CO3Dv2 makes available, and EXP-005
alone could not show whether the reported gaps between backbones were a
data-volume artifact.

### Raise the training cap to 100 sequences/category via an additive batch, val/test unchanged

More training data, and because sequence membership (train vs. dev vs. test)
is fixed by the official CO3D lists before any cap is applied (ADR 0002),
raising the train-only cap cannot cross the train/test boundary — leakage
risk stays zero by construction. Extracted as a separate, additive
"cap100-new-train" manifest/cache rather than replacing the original, so the
original 30-sequence baseline (EXP-005) remains reproducible on its own.

### Remove the training cap entirely

Uses all available official-train sequences per category. Rejected for now:
unbounded storage/compute cost on the Technion VM, and untested whether
returns beyond 100/category are worth the extraction cost. Left open as a
possible future ADR if 100/category proves insufficient.

## Decision

Raise the Part-A segmentation **training-only** cap from 30 to up to 100
sequences/category (actual count per category depends on how many official
train sequences exist for it; 1,762 total new sequences across the dataset).
Selection uses the same deterministic rule as ADR 0002 (SHA-256 ranking over
seed, category, split, sequence ID; `uniform-disjoint-v1` windowing, up to
four windows/sequence), applied to additional official-train sequences
beyond the original 30. Extracted as an additive `cap100-new-train`
manifest/cache, plus a separate `leftover` batch of extra windows from the
original 30 sequences' previously-unused frames. **Val/test caps stay at
5/5 sequences/category, unchanged from ADR 0002** — this decision only
deepens sampling within the train partition; it does not change split
scope or membership.

Training must union the `train` split rows across the original, leftover,
and cap100-new-train caches (`CombinedProbeCacheDataset`,
`train_segmentation.build_datasets`) while evaluating against the original
cache's val/test rows only, so results stay comparable to the ADR 0002
30-sequence baseline. `assert_sequence_disjoint` and `assert_not_trained_on`
must pass against the full set of training directories used, not just one.

## Rationale

Sequence membership is fixed by the official CO3D train/dev/test lists
before any per-category cap is applied (ADR 0002), so a train-only cap
increase cannot introduce train/dev/test leakage — this was also verified
operationally in this branch (`assert_sequence_disjoint` on the real
combined train set vs. val: train 2,541 sequences, val 130, zero overlap).
EXP-006 empirically confirmed the benefit: test macro-IoU improved for all
three backbones (DINOv2 0.7922 -> 0.8063, CUT3R-trained 0.7402 -> 0.7772,
CUT3R-random 0.2298 -> 0.2772), justifying the extraction and storage cost.

## Consequences

- Training storage/compute cost roughly tripled for Part-A segmentation
  (3,054 -> 10,029 train windows); future re-extractions or reruns must
  budget for this.
- ADR 0002's Full-configuration cap table (30/5/5) is now stale for the
  Part-A segmentation **training** partition specifically; val/test caps are
  unaffected and ADR 0002 remains authoritative for them. This ADR
  supplements, not supersedes, ADR 0002.
- Per-category training-window counts are no longer exactly balanced after
  the expansion (25 of 26 categories land in a ~370-400-window band;
  `parkingmeter` alone sits apart at ~181, per EXP-006's category-
  representation check) — tracked as a known limitation, not a blocker.
- Any other Part-A task (e.g. classification) that wants to reuse this
  expanded training cap must adopt it explicitly; it is not automatically
  in scope project-wide.

## Validation

- `assert_sequence_disjoint` / `assert_not_trained_on` pass against the full
  combined training set, not just the original cache (already run in this
  branch for EXP-006).
- `cap100-new-train` manifest `summary.json` counts (1,762 sequences, 42,288
  frames, 6,901 windows) match the deterministic re-generation, per ADR
  0002's manifest-regeneration-and-hash-comparison validation step.
- Empirical macro-IoU deltas reported for all three backbones in EXP-006,
  compared directly against the EXP-005 30-sequence baseline.
