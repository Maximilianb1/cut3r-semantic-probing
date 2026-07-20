# Frozen CUT3R Feature Extraction

Stage 0 has a proposed, tested contract for two distinct representations at all
six timesteps:

| Cache field | Meaning | Shape per cached window | Use |
|---|---|---|---|
| `image_tokens` | Final normalized, state-conditioned spatial tokens for the current frame, without CUT3R's pose token | `[6, 1, grid_height * grid_width, 768]` | Spatial probes; optionally pool for image classification |
| `state_tokens` | Final normalized persistent memory after committing the current frame | `[6, 1, 768, 768]` | Pooled classification and recurrent-state analysis; not pixel-aligned |

At timestep `t`, `image_tokens[t]` belongs to `frame_ids[t]`, whereas
`state_tokens[t]` is the memory after that frame has been processed. The sixth
frame is index `5`; both complete trajectories are retained.

`cut3r_adapter.py` reads these from the pinned upstream model; the only upstream
source change permitted is the versioned cuRoPE build-compatibility patch. `cache.py`
stores float16 Safetensors shards with a Parquet index, hashes, atomic writes,
resume support, integrity validation, and independent cache comparison.
`extract.py` orchestrates CUDA work.

The index hashes the exact source RGB/mask bytes for every window, while cache
metadata records the audited upstream commit, exact cuRoPE compatibility patch,
checkpoint, manifests, transform, and runtime environment.

`checkpoint.py` verifies the released checkpoint's exact SHA-256 before
deserialization, statically rejects globals outside the audited seven-type
OmegaConf set, and exposes those types only through a scoped PyTorch
`safe_globals` context. This preserves the PyTorch 2.6+ weights-only security
default without modifying CUT3R's model loader or enabling unrestricted pickle
loading process-wide.

The adapter depends on pinned upstream private methods. Real A10 preflights and
the Full-51 extraction passed; [ADR 0003](../../docs/decisions/0003-cut3r-trajectory-and-cache-contract.md)
still requires team review before acceptance. The adapter never trains or
attaches a semantic head.
