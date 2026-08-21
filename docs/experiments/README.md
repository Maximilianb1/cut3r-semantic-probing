# Experiment Records

Create one record for every run or grouped sweep whose result may influence a decision, plot, presentation, or report.

Naming: `EXP-NNN-short-name.md`. Keep large outputs outside Git and link them using stable run IDs or paths.

Use [the template](template.md).

- [EXP-001: Stage 0 real-data GPU smoke](EXP-001-stage0-real-gpu-smoke.md)
- [EXP-002: Complete Stage 0 Debug extraction](EXP-002-stage0-complete-debug.md)
- [EXP-003: Part-A frozen-representation segmentation baselines](EXP-003-part-a-segmentation-baselines.md)
- [EXP-004: Part-A image-level classification probes](EXP-004-part-a-classification.md)
- [EXP-005: Part-A three-way segmentation comparison (CUT3R-trained unblocked)](EXP-005-part-a-cut3r-trained-unblocked.md)

Stage 0 manifest builds and feature extraction used only for engineering
verification belong in session notes and preflight artifacts. Create an
experiment record if a run influences category caps, representation choices,
reported performance, or later scientific conclusions.
