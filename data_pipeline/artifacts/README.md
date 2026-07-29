# Artifact Registry

Large artifacts live outside Git. Each important artifact must be referenced from an experiment or session record with its type, stable location, producing commit/configuration, checksum when practical, owner, and retention policy.

Do not place the artifacts themselves in this directory.

Stage 0 artifacts include Parquet manifests, JSON summaries, Safetensors
shards, cache indexes, extraction logs, and preflight reports. Cache metadata
binds checkpoint, upstream commit, configuration, and manifest hashes. Record
stable locations without recording VM credentials.

The canonical external layout, immutable Full-51 cache identities, teammate
verification commands, and loading example are defined in the
[Stage 0 Full-51 cache handoff](../docs/data/stage0-full51-cache-handoff.md).
The current publication target is a privately owned My Drive folder shared with
teammates as Viewer. Both Part A and Part B passed independent local SHA-256
verification. Keep the `-staging` suffix until every cache and provenance file
has uploaded; then publish the immutable `stage0-full51-v1` folder with ordinary
teammate access read-only.
