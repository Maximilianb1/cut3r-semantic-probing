# Documentation

Long-form documentation that spans more than one directory. Code-level READMEs
(`src/*/README.md`, `scripts/README.md`, `configs/README.md`) explain the
directory they sit in.

| Path | What is in it |
|---|---|
| [`REPRODUCING.md`](REPRODUCING.md) | How to re-run the pipeline, from analysis-only to a full extraction. Start here. |
| [`evaluation-protocol.md`](evaluation-protocol.md) | How IoU and accuracy are aggregated, and why the headline is a category macro rather than a window mean. |
| [`data/`](data/README.md) | The CO3Dv2 subset, the window protocol, and the cache layouts. |
| [`decisions/`](decisions/README.md) | Decision records for the choices that constrain what the results can mean. |
| [`experiments/`](experiments/README.md) | One record per reported run: configuration, numbers, interpretation, limitations. |

Results themselves — figures, metrics, per-window predictions — live under
[`reports/`](../reports/README.md).
