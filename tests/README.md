# Tests

Run from the repository root:

```bash
pytest
```

Everything runs on synthetic fixtures — no CO3D download, no model weights, no
GPU. Real-data and GPU checks are extraction preflights, not tests, so nothing
here claims a result it did not compute.

| Area | Files | What is pinned |
|---|---|---|
| Data protocol | `test_manifests.py`, `test_windows.py`, `test_transforms.py`, `test_co3d_selective.py` | Deterministic seeded caps and windows, rejection of split overlap and of manifest tampering, CUT3R crop geometry and RGB/mask alignment, resumable and checksum-verified downloading. |
| Representations | `test_backbones.py`, `test_cut3r_adapter.py`, `test_cut3r_provenance.py`, `test_checkpoint.py` | The shared backbone contract, six-timestep feature semantics against an injected fake model, DINOv2 patch-14 geometry without downloading weights, seeded random re-initialization, the exact compatibility patch, and hash-bound checkpoint loading with a scoped allowlist. |
| Caching | `test_cache.py`, `test_cache_projection.py`, `test_embedding_audit.py`, `test_synthetic_probe_cache.py`, `test_full51_configs.py` | Cache round trip, resume, corruption and orphan detection, storage projection, exact float16 audit equality, fixture argument validation, and that the two Full-51 shards are disjoint and share one contract. |
| Segmentation probe | `test_segmentation_metrics.py`, `test_segmentation_dataset.py`, `test_build_score_comparison.py` | Macro and micro IoU against hand-computed values, the empty-target convention, unioned multi-cache training, and that leakage checks still fire across splits. |
| Classification probe | `test_classification_metrics.py`, `test_classification_training.py`, `test_classification_test_report.py` | Macro F1 averaged over per-category F1s, the label space travelling with the checkpoint, standardization statistics coming from train only, and the sequence-cluster bootstrap reproducing a recorded interval. |

Shared fixtures — the synthetic CO3Dv2 tree, manifests, and probe caches — are
in `conftest.py`.
