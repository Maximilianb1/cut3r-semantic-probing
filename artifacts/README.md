# Artifact Registry

Large artifacts live outside Git. Each important artifact must be referenced from an experiment or session record with its type, stable location, producing commit/configuration, checksum when practical, owner, and retention policy.

Do not place the artifacts themselves in this directory.

Stage 0 artifacts include Parquet manifests, JSON summaries, Safetensors
shards, cache indexes, extraction logs, and preflight reports. Cache metadata
binds checkpoint, upstream commit, configuration, and manifest hashes. Record
stable locations without recording VM credentials.
