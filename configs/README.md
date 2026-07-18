# Configurations

`stage0/debug.yaml`, `pilot.yaml`, and `full.yaml` encode the dataset tiers,
official split caps, six-frame sampler, released CUT3R 512 contract, cache
sharding, expected category counts, mask policy, and external paths.

Missing path environment variables fail rather than falling back to a
machine-specific location. Scientific changes require a new versioned config
and documented rationale.
