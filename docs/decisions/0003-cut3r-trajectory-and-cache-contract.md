# ADR 0003: CUT3R trajectory and cache contract

- Date: 2026-07-18
- Author: Max Bershtman

## Context

The previous code called final per-view decoder tokens "the state" and attached
probe logic directly to multiple CUT3R forward paths. The project needs frozen,
headless representations whose meanings and provenance are explicit.

## Decision criteria

- distinguish persistent state from state-conditioned image tokens;
- preserve all six timesteps for later temporal analysis;
- keep upstream model logic intact apart from a versioned build-compatibility
  patch;
- support resumable, corruption-detecting extraction;
- prevent incompatible checkpoints or manifests from sharing a cache.

## Options considered

### Save only the sixth-frame image tokens

Smallest cache, but prevents analysis of information accumulation.

### Save only persistent state tokens

Compact and relevant to the recurrent-state claim, but not spatially aligned
for segmentation.

### Save both representations at all six timesteps

Larger, but directly supports the approved segmentation, image-level
classification, per-pixel classification, and temporal ablations.

## Decision

For each ordered six-frame window, cache:

- `image_tokens[t]`: final normalized spatial decoder tokens produced while
  processing frame `t`, after cross-attention with the prior persistent state;
  remove CUT3R's non-image pose token explicitly.
- `state_tokens[t]`: final normalized persistent state after frame `t` is
  committed according to CUT3R's update/reset masks.

Both tensors retain a batch dimension and are stacked over six timesteps.
CUT3R runs in evaluation and inference mode, and all parameters have gradients
disabled. The adapter mirrors the pinned upstream recurrent forward path and
does not copy the upstream model or add a semantic head.

Cache tensors use float16 Safetensors shards. A Parquet index maps stable window
IDs to tensor keys and records shapes, frame IDs, token grids, shard hashes, and
dtypes, plus SHA-256 values for all source RGB/mask files. `metadata.json` binds
the cache to checkpoint, pinned CUT3R commit and exact compatibility patch,
manifests, configuration, transform, runtime stack, and representation contract.
Writes are atomic,
resumable, and verified after every shard and before resuming.

Stage 0 configurations require upstream CUT3R commit
`8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf`, the revision against which the
private recurrent adapter was audited. Modern PyTorch requires the versioned
`curope-scalar-type-v1` patch documented by
[upstream CUT3R issue #7](https://github.com/CUT3R/CUT3R/issues/7), which replaces
the deprecated `tokens.type()`
dispatch argument with `tokens.scalar_type()`. Provenance validation accepts
that exact source transformation and no other tracked or untracked source
changes. Extraction also requires exactly one compiled in-place cuRoPE shared
object and records its filename and SHA-256; ignored native build intermediates
do not weaken the source-content check.

[PyTorch 2.6 changed `torch.load` to use its restricted weights-only unpickler
by default](https://docs.pytorch.org/docs/stable/notes/serialization.html#torch-load-with-weights-only-true).
The released CUT3R checkpoint includes OmegaConf configuration containers, so
Stage 0 pins its exact SHA-256 and statically requires exactly these seven
globals: `builtins.dict`, `collections.defaultdict`,
`omegaconf.base.ContainerMetadata`, `omegaconf.base.Metadata`,
`omegaconf.dictconfig.DictConfig`, `omegaconf.nodes.AnyNode`, and `typing.Any`.
Those types are added only within a `torch.serialization.safe_globals` context
around the upstream load call. The project does not set
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` or otherwise enable unrestricted pickle
loading globally.

## Rationale

This contract supports all required later probes without recomputing CUT3R and
keeps scientific claims tied to the actual tensor used. Hash binding makes
silent reuse of stale or incompatible features an error.

## Consequences

- At 512x384, both representations for six timesteps are near 10 MiB per
  window. The two Full-51 shards came to 42.768 and 40.285 GiB, which is why
  extraction ran as two sequential parts.
- The same raw frame can have different cached features in different windows;
  cache identity is the window plus timestep, not the frame alone.

## Validation

- assert feature, batch, timestep, spatial-grid, and channel compatibility;
- reject NaN or infinity;
- round-trip every shard and verify SHA-256;
- reject duplicate window IDs, missing tensors, shape changes, or incompatible
  metadata;
- reject a checkpoint hash mismatch or any pickle global outside the audited
  allowlist before model construction;
- run two independent extractions of the same debug window into different cache
  directories and compare every tensor with `scripts.compare_caches` using zero
  tolerance first; document any required nonzero tolerance and pinned stack.
