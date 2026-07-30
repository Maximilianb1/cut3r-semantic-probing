# Documentation

All long-form project documentation lives here. Code-level READMEs
(`src/*/README.md`, `scripts/README.md`, `configs/README.md`) explain the
directory they sit in; `docs/` is the home for everything that spans multiple
directories or outlives a single change — protocols, decisions, runbooks,
experiment records, and session handoffs.

## Layout

| Subfolder | Purpose | Filename convention |
|---|---|---|
| [`decisions/`](decisions/README.md) | Architecture / Analysis Decision Records for choices that affect interfaces, scientific validity, reproducibility, or multiple teammates. | `NNNN-short-name.md` (four-digit ADR number, monotonically increasing). |
| [`experiments/`](experiments/README.md) | One record per run or grouped sweep whose result may influence a decision, plot, presentation, or report. | `EXP-NNN-short-name.md`. |
| [`sessions/`](sessions/README.md) | Substantial-work notes, investigations, and human/agent handoff briefs. | `YYYY-MM-DD-short-description.md`. Append the GitHub issue number if names collide. |
| [`data/`](data/README.md) | CO3Dv2 and downstream dataset protocols, manifest formats, and cache-handoff documents. | Lowercase-hyphen, topic-first (`stage0-protocol.md`). |
| `project/` | Static scope, evaluation protocol, work breakdown, and machine-specific runbooks. Files here are meant to be stable references, not living notes. | `UPPER_SNAKE_CASE.md` (e.g. `SCOPE.md`, `TECHNION_VM_RUNBOOK.md`). |

## Where does a new document belong?

- **A choice that affects code, interfaces, or scientific validity** → new ADR
  under `decisions/`, discussed in the pull request that carries the change.
- **A run whose numbers might end up in a report or slide deck** → new file
  under `experiments/`.
- **A working session, investigation, or handoff between teammates or between
  a human and an assistant** → new file under `sessions/`. Trivial edits do
  not need a session note.
- **A dataset version, manifest schema, or cache-handoff artifact** → new file
  under `data/`.
- **A stable runbook or protocol reference that is expected to be re-read
  rather than appended to** → new file under `project/`.

When a document does not fit any of these buckets, ask before adding a new
subfolder. Root-level `README.md` and `PROJECT_STATUS.md` are the two
top-level living documents; everything else that survives a PR should live
under `docs/`.

## Related

- Repository governance: [ADR 0001](decisions/0001-repository-governance.md).
- Working-style rules for humans and AI assistants: [`LLM_GUIDE.md`](../LLM_GUIDE.md).
- Current phase and blockers: [`PROJECT_STATUS.md`](../PROJECT_STATUS.md).
