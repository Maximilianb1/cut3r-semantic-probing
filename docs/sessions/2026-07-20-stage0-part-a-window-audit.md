# Stage 0 Full-51 Part A window audit

- Date: 2026-07-20
- Owner: Max Bershtman with Codex
- Assistant/model: Codex, GPT-5
- Status: Local cache/source checks complete; GPU reconstruction pending

## Objective

Inspect a very small real Full-51 Part A sample independently of probe training:
confirm that cache indexing maps one window to the intended ordered CO3Dv2
frames, verify the source and tensor bytes, show the corresponding transformed
inputs, and prepare a reproducible GPU re-run that compares fresh CUT3R features
exactly before rendering the model's actual 3D prediction.

## Sample and observed local results

- Local cache: `C:\cut3r-full51\full51-part-a-v1` (external, not committed).
- Window: `window-68f7d8eb5b94e9b984d7`.
- Sequence: `apple/110_13072_25709`.
- Ordered frame numbers: `1, 8, 15, 23, 30, 37`.
- Image trajectory: float16 `[6, 1, 576, 768]`.
- State trajectory: float16 `[6, 1, 768, 768]`.
- Token grid: `[32, 18]`.
- The referenced cache shard, exported reference tensor, and all six RGB plus
  six mask files passed their recorded SHA-256 checks.
- Both trajectories were finite. Image first-to-last mean absolute difference
  was `0.0454761`; state first-to-last mean absolute difference was `0.4841075`.
  Every adjacent-timestep difference was nonzero.

The six exact source pairs were selectively recovered from the pinned official
CO3Dv2 release into `C:\cut3r-stage0-audit\co3dv2`. The local inspection output
is under `C:\cut3r-stage0-audit\part-a-window-000` and is not committed.

## Implementation

- `src/embeddings/audit.py` validates/exports a small reference, resolves and
  hashes its CO3D sources, computes trajectory diagnostics, and renders inputs,
  shared-PCA spatial-token views, PLY data, and orthographic point-cloud views.
- `scripts/audit_cached_window.py` exposes `export`, `inspect`, and `reconstruct`
  commands.
- `tests/test_embedding_audit.py` covers reference round-trip integrity and the
  exact comparison gate.

No probe is trained. The shared-PCA token view is a diagnostic visualization,
not an inverse model and not a reconstruction. CUT3R's DPT head consumes four
decoder levels while the Stage 0 cache intentionally stores only the final
image tokens and committed state. Therefore an actual reconstruction requires
an independent forward pass from the same six source images on the pinned GPU.

## Checks run

```text
python -m pytest -q tests/test_embedding_audit.py tests/test_cache.py
9 passed
python -m compileall -q src scripts
passed
```

## Next step

Copy the 12 MiB exported reference to the Technion VM, run the `reconstruct`
subcommand with `configs/stage0/full51-part-a.yaml`, and copy back
`reconstruction-audit.json`, `reconstruction.ply`, and
`reconstruction-projections.png`. Completion requires exact zero-difference
image/state features and a visually inspected finite point cloud.
