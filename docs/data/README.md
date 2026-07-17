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
