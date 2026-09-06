# Reports

The results themselves: the metrics, figures, and predictions behind the report
and the talk.

| Directory | Contents |
|---|---|
| [`segmentation/`](segmentation/README.md) | Stage 1. Three frozen backbones (CUT3R-trained, CUT3R-random, DINOv2) x two probe capacities (linear, MLP), on one shared sequence-disjoint data split — per-run `metrics.json` and `inference-test.json`. |
| [`classification/`](classification/README.md) | Stage 2. Per-window test predictions, the metric and bootstrap tables computed from them, and the figures. |

Nothing here is an opaque output. Everything in `classification/` is
regenerated from `classification/predictions/` by a committed script, and
everything in `segmentation/` names the config and command that produced it.

Model checkpoints, embedding caches, and raw prediction tensors are not here:
they are large, regenerable working artifacts. See
[docs/REPRODUCING.md](../docs/REPRODUCING.md).
