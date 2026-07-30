# Configurations

`stage0/` configures CO3D preprocessing and Stage 0 CUT3R feature extraction.
`probe_features/` configures `scripts/extract_probe_features.py`, one file per
probing backbone: which backbone to run, over which manifests, and which
probe-feature cache to write. These are extraction configs only — the probe head
and its optimization are configured separately in
`../../segmentation_validation/configs/`, whose file of the same name must point
at the same `probe_cache.dir`.

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

Missing path environment variables fail rather than falling back to a
machine-specific location. Scientific changes require a new versioned config
and documented rationale.

Do not use `full.yaml` with the selective downloader because `categories: all`
is intentionally rejected as an accidental-download safeguard. Use the two
explicit `full51-part-*.yaml` configurations and retain both manifests/caches.
