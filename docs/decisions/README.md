# Decision Records

The choices that constrain what the results are allowed to mean — splits,
window protocol, what counts as a representation, how much training data. Each
record states the context, the decision, and its consequences, so a number can
be traced to the contract it was produced under.

| ADR | Status | Decision |
|---|---|---|
| [0002](0002-co3dv2-stage0-data-protocol.md) | Proposed | Official CO3D sequence splits and deterministic six-frame windows, with sequence-level split isolation. |
| [0003](0003-cut3r-trajectory-and-cache-contract.md) | Proposed | What "the CUT3R representation" means: six-timestep image and state features, and the verified cache that holds them. |
| [0004](0004-part-a-segmentation-training-cap-expansion.md) | Proposed | Raise the Part-A segmentation training cap from 30 to 100 sequences per category; validation and test unchanged from ADR 0002. |

All three stayed `Proposed`: they were written and followed, but never formally
ratified by the team. The code enforces them regardless — split isolation is an
assertion, not a convention.
