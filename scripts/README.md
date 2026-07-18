# Scripts

Run commands as modules from the repository root:

```bash
python -m scripts.build_manifests --config configs/stage0/debug.yaml
python -m scripts.apply_cut3r_compatibility_patch --cut3r-root /work/CUT3R --expected-commit 8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf
python -m scripts.validate_checkpoint --config configs/stage0/debug.yaml --load-model
python -m scripts.validate_manifests --manifest-dir /artifacts/manifests/debug --dataset-root /data/co3d --inspect-files
python -m scripts.extract_features --config configs/stage0/debug.yaml --limit-windows 1 --cache-dir /cache/preflight-a
python -m scripts.validate_cache --cache-dir /cache/cut3r-semantic/debug
python -m scripts.compare_caches --left /cache/preflight-a --right /cache/preflight-b
```

The cache override enables genuinely independent reproducibility runs. Reusable
and tested logic remains under `src/`.
