# Data Documentation

CO3Dv2 is external. Images, masks, depths, and embedding caches are never
committed; only the protocols and the small summaries that describe them are.

| Document | What it covers |
|---|---|
| [`stage0-protocol.md`](stage0-protocol.md) | The subset, the filtering rules, the official sequence splits, the deterministic six-frame window protocol, and the cache contract. |
| [`co3dv2-statistics.md`](co3dv2-statistics.md) | The per-category counts the subset decisions were made from. |
| [`part-a-cache-layout.md`](part-a-cache-layout.md) | The three Part-A cache partitions, what may be unioned with what, and the schema rules for reading them. |

The scientific contracts behind these are
[ADR 0002](../decisions/0002-co3dv2-stage0-data-protocol.md) and
[ADR 0003](../decisions/0003-cut3r-trajectory-and-cache-contract.md).

A generated manifest is `sequences.parquet`, `frames.parquet`,
`windows.parquet`, and `summary.json`. Every cache carries a `metadata.json`
binding it to the checkpoint, upstream commit, configuration, and manifest
hashes that produced it, and every `metrics.json` copies that metadata in — so
a reported number is always traceable to the exact cache it came from.
