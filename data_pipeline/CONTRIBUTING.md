# Contributing

The goal of these rules is to make parallel work understandable and reversible.

## Workflow

1. Start from an issue with a clear result and acceptance criteria.
2. Pull the latest `main`.
3. Create a short-lived branch:
   - `data/<issue>-<description>`
   - `embed/<issue>-<description>`
   - `seg/<issue>-<description>`
   - `cls/<issue>-<description>`
   - `baseline/<issue>-<description>`
   - `docs/<issue>-<description>`
   - `fix/<issue>-<description>`
4. Keep commits focused. Do not mix unrelated cleanup with feature work.
5. Open a pull request using the repository template.
6. Request one teammate review for meaningful code, data, metric, split, or architecture changes.
7. Merge only after the documented checks pass. Prefer squash merge.

Do not develop directly on `main`, force-push `main`, or commit datasets, checkpoints, cached embeddings, credentials, or generated experiment folders.

## What must be recorded

- A scientific or architectural choice: add an ADR in `docs/decisions/`.
- A training or evaluation run used in analysis: add an experiment record in `docs/experiments/`.
- A handoff or substantial work session: add a note in `docs/sessions/`.
- Current phase, owners, or blockers changed: update `PROJECT_STATUS.md`.

Small mechanical edits do not need their own ADR or session note.

## Definition of done

A pull request is complete when:

- its scope matches the linked issue;
- implementation and configuration are separated;
- relevant tests or smoke checks pass;
- data split and metric changes are documented;
- generated artifacts have a stable external location or identifier;
- the PR explains any LLM assistance and the human verification performed;
- status, decision, experiment, and session records are updated when applicable.

## Review priorities

1. Scientific validity and data leakage
2. Reproducibility
3. Correctness
4. Tests and failure behavior
5. Maintainability and documentation
6. Performance

New dependencies require justification in the pull request. Before adapting CUT3R, CO3D, or baseline code, record its source, version/commit, and license compatibility.
