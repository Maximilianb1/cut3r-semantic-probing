# Scripts

Run commands as modules from the repository root:

```bash
python -m scripts.download_co3d_selective --config configs/stage0/debug.yaml --plan-only
python -m scripts.download_co3d_selective --config configs/stage0/debug.yaml --index-only
python -m scripts.download_co3d_selective --config configs/stage0/debug.yaml
python -m scripts.build_manifests --config configs/stage0/debug.yaml
python -m scripts.apply_cut3r_compatibility_patch --cut3r-root /work/CUT3R --expected-commit 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf
python -m scripts.validate_checkpoint --config configs/stage0/debug.yaml --load-model
python -m scripts.validate_manifests --manifest-dir /artifacts/manifests/debug --dataset-root /data/co3d --inspect-files
python -m scripts.extract_features --config configs/stage0/debug.yaml --limit-windows 1 --cache-dir /cache/preflight-a
python -m scripts.validate_cache --cache-dir /cache/cut3r-semantic/debug
python -m scripts.compare_caches --left /cache/preflight-a --right /cache/preflight-b
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

The cache override enables genuinely independent reproducibility runs. Reusable
and tested logic remains under `src/`.
