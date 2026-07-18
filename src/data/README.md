# Data Pipeline

Implemented Stage 0 responsibilities:

- parse CO3Dv2 annotations and official few-view set lists;
- fail on physical-sequence split overlap;
- select capped sequences with stable SHA-256 ranking;
- create deterministic ordered six-frame windows with frame 6 as target;
- record and reproduce CUT3R's resize/crop geometry for RGB and masks;
- write typed Parquet manifests and hashed JSON summaries;
- validate identities, paths, decoded dimensions, order, target semantics, and
  leakage.

Reusable modules live in `co3d.py`, `windows.py`, `transforms.py`, and
`validation.py`. See [the protocol](../../docs/data/stage0-protocol.md) and
[ADR 0002](../../docs/decisions/0002-co3dv2-stage0-data-protocol.md).
