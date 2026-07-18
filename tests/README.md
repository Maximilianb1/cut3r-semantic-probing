# Tests

The Stage 0 suite covers seeded caps, deterministic windows, leakage rejection,
synthetic CO3Dv2 manifests, CUT3R geometry and RGB/mask alignment, fake-model
six-timestep feature semantics, and cache round-trip/resume/integrity failures.
It also checks full synthetic RGB/mask decoding and independent cache equality.
The exact CUT3R compatibility patch and rejection of additional upstream
changes are regression-tested in a temporary Git repository.

Run `python -m pytest`. Real CO3Dv2 and released-checkpoint GPU checks remain
documented preflight gates rather than synthetic claims.
