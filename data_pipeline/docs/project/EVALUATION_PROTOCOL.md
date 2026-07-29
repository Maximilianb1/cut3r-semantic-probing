# Proposed evaluation aggregation for Stages 1 and 2

This document records the agreed reporting plan; Stage 0 does not calculate
scientific performance or train probes.

## Intersection over Union

For a predicted foreground mask `P` and ground-truth foreground mask `G`:

```text
IoU = number of pixels in both P and G / number of pixels in P or G
```

IoU is 1 for a perfect overlap and 0 when non-empty masks have no overlap. The
empty/empty convention must be fixed in the Stage 1 metric implementation and
reported explicitly.

## Aggregations to report

- **Window micro/overall:** mean metric over every evaluated target window.
- **Sequence macro:** mean each sequence's windows first, then mean sequences.
- **Category macro (primary headline):** mean windows within sequence, sequences
  within category, then categories with equal weight.
- **Per category:** metric, sequence/window count, dispersion or confidence
  interval, and representative failures.

For image-level classification, average a sequence's window logits or
probabilities before making one final sequence prediction. Also report raw
window accuracy as a secondary diagnostic.

The simple overall window average is valid and will be included. It cannot be
the only result because categories and physical sequences contain unequal
numbers of usable samples.
