# Reports

The results themselves: the metrics, figures, and predictions behind the report
and the talk.

| Directory | Contents |
|---|---|
| [`segmentation/`](segmentation/README.md) | Stage 1. Three frozen backbones (CUT3R-trained, CUT3R-random, DINOv2) x two probe capacities (linear, MLP), on one shared sequence-disjoint data split — per-run `metrics.json` and `inference-test.json`. |
| [`classification/`](classification/README.md) | Stage 2. Per-window test predictions, the metric and bootstrap tables computed from them, and the figures. |

`results-overview.png` is the top-level summary figure referenced from the
repository README — segmentation macro-IoU and state-token classification
accuracy, backbone x probe-capacity, with 95% sequence-cluster bootstrap CIs.
Built by
[`scripts/build_results_overview_plot.py`](../scripts/README.md) from the
bootstrap CSVs already committed under `segmentation/comparison/` and
`classification/` — no cache, no GPU, no model weights.

Nothing here is an opaque output. Everything in `classification/` is
regenerated from `classification/predictions/` by a committed script, and
everything in `segmentation/` names the config and command that produced it.

Model checkpoints, embedding caches, and raw prediction tensors are not here:
they are large, regenerable working artifacts. See
[docs/REPRODUCING.md](../docs/REPRODUCING.md).
