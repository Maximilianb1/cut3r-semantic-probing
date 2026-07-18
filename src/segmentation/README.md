# Binary Segmentation

Future home of foreground/background probe training and evaluation. Stage 0
supplies six-timestep spatial tokens and masks but trains no head. Stage 1 will
start with a tiny overfit check, then use frame-6 image tokens and masks under
`docs/project/EVALUATION_PROTOCOL.md`.
