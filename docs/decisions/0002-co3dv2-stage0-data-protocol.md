# ADR 0002: CO3Dv2 Stage 0 data protocol

- Status: Proposed
- Date: 2026-07-18
- Owners: Max Bershtman; project team review required
- Related issue/PR: Stage 0 foundations

## Context

The earlier proof of concept split odd and even frames from the same physical
object sequence. Adjacent views and backgrounds therefore crossed the training
and test boundary. Stages 1 and 2 need a shared, leakage-safe set of ordered
multi-view samples.

## Decision criteria

- prevent physical-object leakage;
- support segmentation and classification from the same samples;
- represent CO3Dv2 broadly while bounding GPU and storage cost;
- make selection independent of filesystem order;
- retain enough temporal information for later timestep analysis.

## Options considered

### Frame-level random or odd/even split

Cheap, but invalid for generalization because related views of the same object
cross splits.

### Custom 70/15/15 sequence split

Leakage-safe, but unnecessarily replaces CO3Dv2's published sequence sets.

### Official sequence sets with deterministic capped sampling

Leakage-safe, reproducible, compatible with all 51 categories, and adjustable
to available resources.

## Decision

- Use CO3Dv2 `v2_231130` annotations.
- Derive sequence membership from the official few-view train, dev, and test
  set lists. A sequence belongs to exactly one project split.
- Construct ordered six-frame samples. The sixth frame is the primary target.
- Select up to four non-overlapping windows per sequence with
  `uniform-disjoint-v1`: choose `6 * W` evenly spaced annotated frames across
  the sequence and divide them into `W` ordered groups of six, where
  `W = min(configured_windows, floor(usable_frames / 6))`.
- Rank sequences with SHA-256 over seed, category, split, and sequence ID before
  applying a per-category cap. Filesystem ordering has no effect.
- Debug uses `ball` and `chair`, three sequences per split, and up to three
  windows per sequence.
- Pilot remains available but is skipped after the complete Debug extraction
  supplied real runtime, VRAM, and storage measurements.
- Full uses all 51 categories with caps of 30/5/5 train/val/test sequences and
  up to four windows per sequence. Categories with fewer sequences use all
  available sequences.
- Execute Full as two disjoint 26/25-category storage shards. The shards are not
  train/test partitions and must be combined in later stages.
- Require RGB, foreground mask, and camera viewpoint annotations. Reject
  sequences with fewer than six usable frames and record every exclusion class.
- Do not run probe training or test-set evaluation in Stage 0.

## Rationale

Splitting before window construction prevents the same physical object from
appearing in multiple splits. Uniform, disjoint frames cover the sequence while
reducing redundant adjacent views. Caps keep categories with thousands of
sequences from dominating extraction cost while retaining all 51 categories in
the full configuration.

## Consequences

- Multiple windows from one sequence remain statistically related. Later
  evaluation must report window-micro, sequence-macro, category-macro, and
  per-category results.
- Rare categories such as `tv` have uncertain per-category estimates and must
  show their sequence counts.
- Full extraction requires sequential cache transfer because the combined
  measured projection exceeds the VM's persistent disk capacity.
- Blind target/raymap evaluation is deferred.

## Validation

- fail on any sequence split overlap;
- verify every window has six strictly ordered, unique frames from one split;
- regenerate manifests twice and compare hashes;
- visually inspect transformed RGB/mask overlays;
- require the full configuration to discover exactly 51 categories;
- decode and dimension-check every selected RGB/mask before GPU extraction;
- recompute and verify the per-frame recorded resize/crop plan;
- reject any frame-6 target whose transformed binary foreground mask is empty;
- record counts and rejection reasons in `summary.json`.
