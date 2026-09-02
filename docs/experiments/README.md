# Experiment Records

One record per run whose numbers appear in a figure, the report, or the talk.
Each states the configuration that produced it, the result, what it means, and
what it does not support.

| Record | Subject |
|---|---|
| [EXP-001](EXP-001-stage0-real-gpu-smoke.md) | Real-data GPU smoke: exact one-window reproducibility and throughput before committing to a 51-category extraction. |
| [EXP-002](EXP-002-stage0-complete-debug.md) | Complete Debug-tier extraction: runtime, VRAM, and storage measured end to end. |
| [EXP-003](EXP-003-part-a-segmentation-baselines.md) | First Part-A segmentation baselines (DINOv2, CUT3R-random). Superseded by EXP-005. |
| [EXP-004](EXP-004-part-a-classification.md) | First Part-A classification probes, validation split. Superseded by EXP-008. |
| [EXP-005](EXP-005-part-a-seg-cut3r-unblocked.md) | The complete three-way segmentation comparison, once the labelled CUT3R-trained cache existed. |
| [EXP-006](EXP-006-part-a-seg-expanded-training.md) | **Reported.** Segmentation on ~3.3x more training data, all three backbones. |
| [EXP-007](EXP-007-part-a-seg-probe-capacity-ablation.md) | **Reported.** Segmentation, linear head vs. MLP-512 on identical data. |
| [EXP-008](EXP-008-classification-linear-vs-mlp.md) | **Reported.** 26-way classification, linear head vs. MLP-512, three representations. |

Large outputs stay out of Git; each record names the configuration and commands
that regenerate them. See [../REPRODUCING.md](../REPRODUCING.md).
