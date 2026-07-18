# Frozen CUT3R Feature Extraction

Stage 0 has a proposed, tested contract for two distinct representations at all
six timesteps: final normalized state-conditioned spatial image tokens without
the pose token, and committed final normalized persistent-state tokens.

`cut3r_adapter.py` reads these from an unmodified upstream model. `cache.py`
stores float16 Safetensors shards with a Parquet index, hashes, atomic writes,
resume support, integrity validation, and independent cache comparison.
`extract.py` orchestrates CUDA work.

The index hashes the exact source RGB/mask bytes for every window, while cache
metadata records the clean upstream commit, checkpoint, manifests, transform,
and runtime environment.

The adapter depends on pinned upstream private methods and requires a real GPU
preflight before [ADR 0003](../../docs/decisions/0003-cut3r-trajectory-and-cache-contract.md)
can be accepted. It never trains or attaches a semantic head.
