# Data Documentation

CO3D files are external and must not be committed.

Every dataset version used by the project should have:

- CO3D version and source URL;
- category and sequence identifiers;
- exact frame manifest;
- train/validation/test assignment;
- rationale for filtering;
- mask interpretation and thresholding;
- preprocessing configuration;
- checksum or stable version identifier;
- leakage checks, especially sequence and neighboring-frame overlap;
- storage location accessible to the team.

Commit manifests and metadata when small. Keep images, masks, depths, and caches in external storage.

## Current Stage 0 protocol

- [Data and feature protocol](stage0-protocol.md)
- [Full-51 cache publication and team handoff](stage0-full51-cache-handoff.md)
- [CO3Dv2 planning statistics](co3dv2-statistics.md)
- [ADR 0002](../decisions/0002-co3dv2-stage0-data-protocol.md)
- [ADR 0003](../decisions/0003-cut3r-trajectory-and-cache-contract.md)

Generated manifests consist of `sequences.parquet`, `frames.parquet`,
`windows.parquet`, and `summary.json`. Keep them external until their size and
privacy are reviewed; only suitably small summaries/manifests belong in Git.
