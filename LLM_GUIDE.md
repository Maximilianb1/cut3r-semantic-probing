# Shared Guide for AI-Assisted Work

This file is the single source of truth for Codex, Claude, Gemini, GitHub Copilot, and other
coding assistants. `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, and
`.github/copilot-instructions.md` are thin convention-filename redirects that
point here — **do not delete them**.

## Before changing anything

1. Read `README.md`, `PROJECT_STATUS.md`, and `CONTRIBUTING.md` at the repository root.
   Source code lives under `src/`; Stage 0 scripts live under `scripts/`; Stage 1
   segmentation code lives under `src/segmentation/`.
2. Read accepted ADRs relevant to the task.
3. Read the latest applicable session and experiment records.
4. Inspect the actual code path. Treat comments, reports, and previous assistant statements as claims to verify.
5. Keep the requested task within one issue and one branch.

## Project guardrails

- Do not implement an unresolved scientific choice as if it were settled.
- Do not call a nonlinear MLP a linear probe.
- Distinguish persistent CUT3R state tokens from state-conditioned per-view decoder tokens.
- Do not create frame-level splits that leak adjacent views across train and test without an accepted ADR.
- Never commit CO3D data, cached embeddings, model weights, secrets, or large generated artifacts.
- Never invent experiment results, tests, paths, dataset versions, or completed work.
- Do not change dependencies, data splits, metrics, or representation definitions without documenting the reason.

## Working style

- Act as a senior partner with critical thinking. Push back on your own
  reasoning, on the user's reasoning, and on prior claims from other humans or
  agents. Do not accept a decision as settled just because it is written down.
- Justify non-trivial decisions. Ground them in the codebase, an accepted ADR,
  or an authoritative external reference when useful. Cite the reference
  inline.
- Match the register to the audience:
  - Files read by humans (`README.md`, `PROJECT_STATUS.md`, ADRs, session
    notes, `docs/`): short, concise, plain prose.
  - Files read by agents (`.github/copilot-instructions.md`, `.github/prompts/*`,
    skill/subagent files): precise, imperative, agent-directed. Prefer bullet
    lists of rules over narrative.

## Required handoff

For substantial work, update or create a session note containing the objective, files changed, decisions, commands and real outcomes, artifact locations, next step, assistant/model used, and human verification.

Raw chat transcripts and private prompts are not required. Record only the information needed to reproduce and review the work.

## Quality bar

- Prefer configuration-driven experiments.
- Make randomness explicit and seedable.
- Fail loudly on invalid shapes, missing artifacts, and split overlap.
- Add the smallest useful test with each behavior change.
- Report successful and failed checks accurately.
