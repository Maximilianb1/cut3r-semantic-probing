# Project Scope

## Primary question

Do frozen CUT3R representations encode semantic object identity beyond class-agnostic foreground/background localization?

## Required work

- Preprocess and filter a documented CO3D subset.
- Extract and cache explicitly defined representations from a frozen CUT3R checkpoint.
- Validate binary segmentation across diverse CO3D categories.
- Train and evaluate multiclass classification probes.
- Compare image-level and per-pixel classification designs.
- Compare against agreed random-weight, simple trained, and advanced frozen baselines using the same data.
- Produce reproducible metrics, analyses, plots, an architecture diagram, presentation, and report.

## Optional work

Multi-object detection and classification require a separate go/no-go decision after the required stages are stable.

## Non-goals for the initial scaffold

- No implementation code.
- No unreviewed reproduction of the previous proof-of-concept claims.
- No decision yet about the exact CUT3R layer/token representation.
- No decision yet about baseline model families.
- No full CO3D download inside the repository.

## Scientific caution

Success of a trained nonlinear head shows that information is decodable from its input representation; it does not automatically show linear separability, causal reliance on 3D state, or category-general semantic understanding. Claims must match the probe, split, controls, and metrics actually used.
