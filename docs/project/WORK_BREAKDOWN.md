# Work Breakdown

This reflects the current proposal and allocation document. Unassigned work stays `TBD` rather than being silently assigned.

## Stage 0 - Foundations

| Work item | Owner | Output |
|---|---|---|
| Clarify the course's intended random-initialization baseline | Aviv | Accepted decision or clarification note |
| CO3D preprocessing for segmentation and classification | Max | Versioned manifest and preprocessing contract |
| Define and cache CUT3R representations | Max | Accepted representation ADR and cache specification |
| Shared code and configuration design | Team | Interface ADR |

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
