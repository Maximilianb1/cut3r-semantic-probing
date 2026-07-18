# Work Breakdown

This reflects the current proposal and allocation document. Unassigned work stays `TBD` rather than being silently assigned.

## Stage 0 - Foundations

| Work item | Owner | Output |
|---|---|---|
| Clarify the course's intended random-initialization baseline | Aviv | Accepted decision or clarification note |
| CO3D preprocessing for segmentation and classification | Max | Implemented manifest/transform pipeline; ADR 0002 review and real-data validation pending |
| Define and cache CUT3R representations | Max | Implemented trajectory/cache pipeline; ADR 0003 review and GPU validation pending |
| Shared code and configuration design | Team | Proposed data-to-feature interfaces and versioned debug/pilot/full configurations |

## Stage 1 - Binary segmentation

| Work item | Owner |
|---|---|
| Reproduce and validate the segmentation probe on selected CO3D data | TBD |
| Determine whether inference is sufficient or the probe must be retrained | TBD |
| Compare random-weight and advanced baselines | TBD |
| Metrics, analysis, and plots | TBD |

## Stage 2 - Multiclass classification

| Work item | Owner |
|---|---|
| Train multiclass probe head | TBD |
| Compare image-level and per-pixel classification | TBD |
| Compare random-weight and advanced baselines | TBD |
| Metrics, analysis, and plots | TBD |

## Stage 3 - Optional

Plan only after required stages and an accepted go/no-go ADR.

## Close-out

| Work item | Owner |
|---|---|
| Architecture diagram | TBD |
| Presentation | TBD |
| Final report | Team; section owners TBD |
