# Configurations

`stage0/debug.yaml`, `pilot.yaml`, and `full.yaml` encode the dataset tiers,
official split caps, six-frame sampler, released CUT3R 512 contract, cache
sharding, expected category counts, mask policy, and external paths.

All Stage 0 tiers pin the audited upstream CUT3R Git revision plus the exact
`curope-scalar-type-v1` compatibility patch required by modern PyTorch.
Extraction fails for any other checkout change.

Missing path environment variables fail rather than falling back to a
machine-specific location. Scientific changes require a new versioned config
and documented rationale.
