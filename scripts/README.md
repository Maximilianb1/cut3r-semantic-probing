# Scripts

Run commands as modules from the repository root:

```bash
python -m scripts.download_co3d_selective --config configs/stage0/debug.yaml --plan-only
python -m scripts.download_co3d_selective --config configs/stage0/debug.yaml --index-only
python -m scripts.download_co3d_selective --config configs/stage0/debug.yaml
python -m scripts.download_co3d_targeted --frames-parquet /artifacts/manifests/full51-part-a-v1/frames.parquet --dataset-root /data/co3d
python -m scripts.build_manifests --config configs/stage0/debug.yaml
python -m scripts.project_cache_storage --manifest-dir /artifacts/manifests/full51-part-a-v1 --filesystem-path /cache --reserve-gib 10
python -m scripts.apply_cut3r_compatibility_patch --cut3r-root /work/CUT3R --expected-commit 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf
python -m scripts.validate_checkpoint --config configs/stage0/debug.yaml --load-model
python -m scripts.validate_manifests --manifest-dir /artifacts/manifests/debug --dataset-root /data/co3d --inspect-files
python -m scripts.extract_features --config configs/stage0/debug.yaml --limit-windows 1 --cache-dir /cache/preflight-a
python -m scripts.validate_cache --cache-dir /cache/cut3r-semantic/debug
python -m scripts.compare_caches --left /cache/preflight-a --right /cache/preflight-b
python -m scripts.audit_cached_window export --cache-dir /cache/full51-part-a-v1 --output-dir /artifacts/audits/part-a-window
python -m scripts.audit_cached_window inspect --reference-dir /artifacts/audits/part-a-window --dataset-root /data/co3d --output /artifacts/audits/part-a-window/inputs-and-features.png
python -m scripts.audit_cached_window reconstruct --reference-dir /artifacts/audits/part-a-window --config configs/stage0/full51-part-a.yaml --output-dir /artifacts/audits/part-a-window/reconstruction
```

`download_co3d_selective` accepts only explicit category lists and finite
train/validation/test sequence caps. `--plan-only` verifies and extracts the
small official metadata archives. `--index-only` additionally locates all
required members in the remote ZIPs and reports projected compressed and
uncompressed bytes. The full mode downloads those members and records ZIP CRC,
per-file SHA-256, source archive, and official container hash provenance under
config-specific records under `$CO3D_ROOT/.co3d-selective/`, preventing Part B
from overwriting Part A provenance. Use `full51-part-a.yaml` and
`full51-part-b.yaml` for the sequential all-category run.

`download_co3d_targeted` is the fetch-only counterpart: given a manifest that
already exists (a `frames.parquet`), it downloads exactly the
`image_relpath`/`mask_relpath` files that manifest names, with the same
per-file CRC/checksum verification, and nothing else. Use it instead of
`download_co3d_selective` when a trusted manifest already exists and a
`sampling`-based re-derivation is not guaranteed to reproduce it exactly
(this happened in practice on the Part-A manifest).

The cache override enables genuinely independent reproducibility runs. Reusable
and tested logic remains under `src/`.

`project_cache_storage` uses every selected window's recorded token grid to
project float16 image/state tensor bytes, adds configurable format overhead and
a free-space reserve, and exits nonzero before extraction if the cache would not
fit. The Full-51 runbook requires this gate because varied aspect ratios can use
more image tokens than the Debug measurement.

`audit_cached_window` exports one roughly 12 MiB reference rather than copying a
whole cache shard. `inspect` verifies the six source RGB/mask hashes and renders
the transformed inputs beside a shared-PCA view of the spatial tokens.
`reconstruct` must run on the pinned CUDA machine: it independently recomputes
the six image/state trajectories, requires exact float16 equality with the
reference, then invokes CUT3R's original DPT reconstruction head and writes a
colored point cloud plus three orthographic inspection views. The PCA image is
only a feature visualization; it is never described as a 3D reconstruction.
