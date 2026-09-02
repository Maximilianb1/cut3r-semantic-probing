# Backbones

Shared frozen feature extractors for semantic probing. One contract, three
weight sources, reused by both segmentation and classification probes so the
probe code is written once.

## Contract

`build_backbone(config)` returns a `Backbone`. Every backbone implements:

```python
extraction = backbone.extract_window(frame_rows, dataset_root=root)
extraction.features       # BackboneFeatures: spatial_tokens [N, D], global_tokens [M, D], token_grid
extraction.target_mask    # [H, W] target-frame mask under this backbone's own geometry
extraction.target_labels()  # per-token foreground labels pooled onto token_grid
backbone.provenance()     # JSON-serializable identity recorded in the cache
```

- `spatial_tokens` are pixel-grid aligned → segmentation and per-pixel tasks.
- `global_tokens` are a non-spatial summary → pooled image classification.
- Each backbone owns its preprocessing and **its own token grid**; never assume
  two backbones share a grid. The target mask is returned under the same
  geometry as the features, so it pools onto the feature grid.
- Backbones are frozen (`eval`, `requires_grad_(False)`). The only trainable
  component is the probe head, added downstream.

## Backbones

| Config | Source | spatial ← | global ← | Notes |
|---|---|---|---|---|
| `{"kind":"cut3r","weights":"trained"}` | Stage 0 CUT3R path (reuses `Cut3rFeatureExtractor`) | `image_tokens[5,0]` | `state_tokens[5,0]` | 512 / patch-16 geometry |
| `{"kind":"cut3r","weights":"random"}` | Same architecture, seeded re-init | `image_tokens[5,0]` | `state_tokens[5,0]` | Untrained control |
| `{"kind":"dinov2"}` | `torch.hub` `dinov2_vitb14` | `x_norm_patchtokens` | `x_norm_clstoken` | patch-14, ImageNet normalization |

## What the two baselines establish

- **Random-initialized CUT3R** is the untrained control. `randomize_weights`
  resets every module exposing `reset_parameters` under a fixed seed, recorded
  as `options.random_init = {strategy, seed}` and written into the cache
  provenance. Holding architecture, data, head, and optimization fixed and
  removing only the pretrained weights is what isolates pretraining from
  architecture.
- **DINOv2 ViT-B/14** is the vision-model anchor: what a model trained for 2D
  visual representation achieves on the same task. 768-dim to match CUT3R, so
  the head is identical across the two. The variant and its `torch.hub` source
  are written into provenance, so a result always names the DINOv2 it used.

## What the comparison holds fixed

Across all three backbones: the same windows and the same sequence-level splits,
the same head architecture, loss, optimizer, and epoch budget, and the same
best-validation checkpoint rule.

Not held fixed: preprocessing and token grid. Each backbone keeps its own native
geometry — CUT3R at 512 / patch-16, DINOv2 at patch-14 — because resampling one
model onto the other's grid would degrade it rather than make the comparison
fairer. The target mask is pooled onto whichever grid the backbone produced, so
IoU is always measured against that backbone's own resolution.
