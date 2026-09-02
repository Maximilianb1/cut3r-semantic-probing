# Configurations

Versioned YAML configurations for every stage of the pipeline. Each file
describes a **conceptual tier** (its scientific meaning) together with the
**physical parameters** needed to reproduce the run.

## Directory layout

- `configs/stage0/` — Stage 0 (data preprocessing + frozen embedding
  extraction into per-window token caches).
- `configs/probe_features/` — probing backbone extraction: one file per
  backbone (`cut3r_random`, `cut3r_trained`, `dinov2`), consumed by
  `scripts/extract_probe_features.py`. These are extraction configs only —
  the probe head and its optimization are configured separately in
  `src/segmentation/configs/`, whose file of the same name must point
  at the same `probe_cache.dir`.
- Follow-up stages will live under sibling subdirectories
  (`configs/classification/`, etc.) once their code lands.

## Stage 0 tiers

`stage0/debug.yaml`, `pilot.yaml`, and `full.yaml` encode the conceptual dataset
tiers. The approved 51-category extraction is physically sharded into
`full51-part-a.yaml` and `full51-part-b.yaml` so one cache fits on the Technion
VM at a time. The two files contain disjoint 26/25-category lists whose union is
exactly the official 51 categories; they share 30/5/5 train/validation/test
sequence caps, four windows per sequence, seed, preprocessing, model, and
checkpoint policy. They are execution/storage shards, not scientific splits.

All Stage 0 tiers pin the audited upstream CUT3R Git revision plus the exact
`curope-scalar-type-v1` compatibility patch required by modern PyTorch.
They also pin the released 512 checkpoint SHA-256 and its audited restricted
loading policy. Extraction fails for any other checkout or checkpoint content.

Do not use `full.yaml` with the selective downloader because `categories: all`
is intentionally rejected as an accidental-download safeguard. Use the two
explicit `full51-part-*.yaml` configurations and retain both manifests/caches.

## Environment-variable resolution

Every host-specific path is written as `${NAME}` in YAML and resolved at load
time from the process environment. Configs never fall back to defaults:
`src/common/io.load_yaml` raises `ValueError("Unresolved environment variable
in configuration: …")` if any `${…}` reference is unset after
`os.path.expandvars`. This is deliberate — a missing variable is a machine
misconfiguration, not something to guess.

Stage 0 configs expect these variables to be exported before invoking a
script:

| Variable | Meaning |
|---|---|
| `CO3D_ROOT` | Root of the CO3Dv2-v2_231130 dataset checkout. |
| `CUT3R_ROOT` | Local clone of the audited CUT3R repository. |
| `CUT3R_CHECKPOINT` | Path to the released 512 checkpoint file. |
| `CUT3R_CACHE_ROOT` | Root under which extraction writes cache shards. |
| `CUT3R_ARTIFACT_ROOT` | Root under which manifests and reports are written. |

[`docs/REPRODUCING.md`](../docs/REPRODUCING.md) records the canonical export
block.

## Top-level schema

Every Stage 0 config contains these top-level keys. Each consumer reads the
keys it needs directly (`scripts/extract_features.py`,
`src/data/co3d.build_dataset`, `src/embeddings/extract`, etc.); there is no
centralized validator today, so **unknown keys are silently ignored**.
Keep the exact key names below to avoid confusing typos:

- `dataset` — CO3Dv2 subset selection: `version`, `root`, `categories`,
  `expected_category_count`, `require_viewpoint`, `require_local_files`.
- `sampling` — deterministic subsetting: `seed`, `window_length`,
  `windows_per_sequence`, `sequence_caps: {train, val, test}`.
- `preprocessing` — image and target normalisation: `input_size`, `patch_size`,
  `square_ok`, `mask_threshold`.
- `model` — frozen backbone identity: `cut3r_root`, `expected_commit`,
  `compatibility_patch`, `checkpoint`, `checkpoint_sha256`, `device`.
- `cache` — writer configuration: `directory`, `windows_per_shard`.
- `output` — manifest and report locations: `manifest_dir`.

A helper `src/common/io.reject_unknown_keys` exists but is not yet wired into
any config-loading path. Wiring it in is a small hardening follow-up.

## Adding or changing a config

Scientific changes require a **new versioned config file** and a documented
rationale (ADR or session note). Do not edit an existing tier file in-place —
downstream cache hashes and provenance depend on its exact content.

Common patterns:

- Adding a category subset for a debug run → new `debug-<name>.yaml`.
- Bumping the CUT3R revision or checkpoint → new tier under a versioned name
  and an ADR update.
- Sharding a run for VM storage → follow the `full51-part-a/b` split pattern.

## Related

- Provenance and cache identity: `docs/data/stage0-protocol.md`.
- Manifest and window contracts: ADRs 0002 and 0003 under `docs/decisions/`.
- Segmentation probe-head configs (twin of `configs/probe_features/`):
  `src/segmentation/configs/`. The two files must agree on
  `probe_cache.dir`.

