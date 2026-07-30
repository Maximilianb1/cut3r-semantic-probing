# Session: codebaseNdatapipeline-redesign plan

- Date: 2026-07-30
- Author: Ron Bartal
- Branch: `codebaseNdatapipeline-redesign`
- Related issue/PR: TBD
- Assistant/model, if used: GitHub Copilot (Claude Opus 4.7)

## Objective

Turn the current half-migrated layout (root + `data_pipeline/` +
`segmentation_validation/`) into a clean, single-package monorepo ready for
four+ people to collaborate on Stages 1–2 without stepping on each other. This
document is the working plan for the branch; sections below track proposed
changes with proposer attribution, difficulty, severity, and status.

---

## Working plan

## Ground rules

- **Do NOT touch `segmentation_validation/`** (Aviv & Lihi are working on it).
  Read-only findings go in §8; no edits, no moves.
- **Do NOT edit `data_pipeline/src/backbones/probe_cache.py`** (Aviv & Lihi
  co-own). Read-only findings in §9.
- Every other file/dir under root and under `data_pipeline/` is in scope.
- Land the plan first, then execute in the order in §10, one PR per section
  where possible (per `CONTRIBUTING.md` "one issue, one branch, focused
  commits").

## Legend

| Column | Values |
|---|---|
| Proposer | **Ron** (human) / **Agent** (Opus-4.7) |
| Difficulty | **S** (< 1 h mechanical) / **M** (1–4 h, needs care) / **L** (design + review) |
| Size | files touched / rough LOC |
| Severity | **low** (cosmetic) / **med** (friction, confusion) / **high** (blocks collaboration or correctness) |
| Status | **New** (proposed, unreviewed) / **Accepted** (agreed, not started) / **WIP** (in progress on this branch) / **PR-open** (implemented, awaiting review/merge) / **Done** (merged) / **Deferred** (out of scope for this branch) / **Read-only** (finding for another owner; we will not act) / **Rejected** (discussed and dropped) |

**Ron's blanket decisions on 2026-07-30:** items §1, §4, §5, §7 all accepted. Items 3+4 in "open questions" deferred. All items with Proposer=Agent that Ron confirmed via §4 flatten → **Accepted** (see per-row status).

---

## 1. Root-level meta files (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `LLM_GUIDE.md`)

**Context.** All three of `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` are 2-line
redirects to `LLM_GUIDE.md`. They *look* redundant, but each filename is the
**auto-discovery convention** of a different tool:

- `AGENTS.md` — OpenAI Codex CLI / Cursor / Aider convention
  (see [agents.md](https://agents.md/) — an open standard adopted by 20k+ repos).
- `CLAUDE.md` — Claude Code auto-loads this into every session
  ([Anthropic docs](https://docs.claude.com/en/docs/claude-code/memory)).
- `GEMINI.md` — Gemini CLI convention
  ([Google docs](https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md#context-files-hierarchical-instructional-context)).
- **Missing**: `.github/copilot-instructions.md` — GitHub Copilot's convention
  ([VS Code docs](https://code.visualstudio.com/docs/copilot/copilot-customization#_custom-instructions)).
  Since you and Aviv both use Copilot in VS Code, this is a real gap.

**Conclusion.** Keep the redirects — they are load-bearing, not repetition.
Add the Copilot one. Push the meat into `LLM_GUIDE.md`.

| # | Change | Proposer | Diff | Size | Sev | Status |
|---|---|---|---|---|---|---|
| 1.1 | **Keep** `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` at root as 1-line redirects. Document *why* in `LLM_GUIDE.md` ("these are convention filenames, do not delete"). | Agent | S | 1f / +3 LOC | low | WIP (Commit 2, `2698cd1`) |
| 1.2 | **Add** `.github/copilot-instructions.md` — same 1-line redirect to `LLM_GUIDE.md`. | Agent | S | 1f / +1 LOC | med | WIP (Commit 2, `2698cd1`) |
| 1.3 | Move all agent-shared guardrails into `LLM_GUIDE.md`; keep it as the single source of truth. Already true today — just re-audit after §4 renames land so paths in it stay correct. | Ron | S | 1f / ~10 LOC | low | WIP (Commit 2, `2698cd1`) |

---

## 2. `.github/` — instructions, prompts, and "skills"

**Context.** `.github/` today has `CODEOWNERS`, `pull_request_template.md`, and
three issue templates. Ron asked whether "skills / subagents" should live here
or be scattered. My recommendation:

- **Reusable, tool-agnostic guidance** (guardrails, definitions, glossary) →
  `LLM_GUIDE.md` at root. Already there.
- **Copilot-specific auto-loaded rules** → `.github/copilot-instructions.md`
  (§1.2).
- **Repeatable prompts** (like Ron's `fix_pr_comments`) → **`.github/prompts/*.prompt.md`**.
  This is the VS Code Copilot convention for reusable prompts
  ([docs](https://code.visualstudio.com/docs/copilot/copilot-customization#_prompt-files-experimental))
  and can also be checked in for Claude Code / other tools to reference.
- **Claude-specific subagents** (if we want them) → `.claude/agents/*.md`
  ([docs](https://docs.claude.com/en/docs/claude-code/sub-agents)).
  Only add when we have a concrete recurring workflow that benefits.

The distinction that matters: a **prompt file** is a saved instruction any human
or agent can invoke on demand. A **subagent** is a Claude-only mechanism that
gets its own context window and tool policy. `fix_pr_comments` is naturally a
prompt (short, tool-agnostic, invoked ad-hoc). We can promote it to a subagent
later if it grows a distinct tool policy (e.g. "may write, may not push").

| # | Change | Proposer | Diff | Size | Sev | Status |
|---|---|---|---|---|---|---|
| 2.1 | Create `.github/prompts/` with a README explaining the convention (target: Copilot prompt files, cross-tool usable). | Agent | S | 2f / +30 LOC | med | WIP (Commit 5). `.github/prompts/README.md` created — explains why the convention exists (per-tool auto-discovery), what belongs vs does not belong here, `.prompt.md` file format with minimal `description` frontmatter, and how to add a new prompt. |
| 2.2 | Author `.github/prompts/fix_pr_comments.prompt.md` per Ron's spec (see §11). | Ron | M | 1f / ~60 LOC | med | WIP (Commit 5). Created verbatim per §11: 6-step flow (ingest, read code, read comment, take-position-and-stop, negotiate, act) with non-negotiable ground rules (one thread at a time, never push/post unattended, never invent test results, honour `LLM_GUIDE.md`). Imperative bullets per LLM_GUIDE working-style rule for agent-facing files. |
| 2.3 | Do **not** create a `.claude/agents/` tree yet. Defer until a workflow actually needs its own tool policy. | Agent | — | 0 | low | Deferred |
| 2.4 | Review `CODEOWNERS` after §4 renames — path patterns will change. | Agent | S | 1f / ~5 LOC | med | No-op (Commit 5 audit). `CODEOWNERS` is a single glob `* @Maximilianb1` with no path patterns to update. Genuine gap surfaced: Aviv & Lihi should own `segmentation_validation/` and `src/backbones/probe_cache.py`, but their GitHub handles are not documented (README explicitly says "handles will be added when available"). Filed as follow-up: refresh `CODEOWNERS` once handles land. |

---

## 3. `docs/` — keep, but de-duplicate

**Context.** Ron asked if `docs/` is justified. Yes, and strongly:

- Course project deliverables (ADRs, experiment records, session handoffs) are
  **cross-cutting** and long-lived — scattering them under `data_pipeline/` and
  `src/segmentation/` would break the paper trail the LLM guide relies on.
- The alternative is Notion / a wiki, which the team isn't using.
- `docs/decisions/README.md` already indexes ADRs; that pattern must stay.

**Real problem** is not existence, it's **duplication**:

- `docs/project/SCOPE.md`, `EVALUATION_PROTOCOL.md`, `WORK_BREAKDOWN.md`,
  `FULL51_TWO_PART_RUNBOOK.md`, `TECHNION_VM_RUNBOOK.md` — some of these
  overlap with the root `README.md` "Project stages" table and with
  `PROJECT_STATUS.md`. Need a single audit pass to decide who owns each fact.
- The commit `94c6997` had moved `docs/` under `data_pipeline/`; the branch has
  it back at root. That's the right home — confirm and lock in.

| # | Change | Proposer | Diff | Size | Sev | Status |
|---|---|---|---|---|---|---|
| 3.1 | Keep `docs/` at repo root. Document intent in `docs/README.md` (missing today). | Ron | S | 1f / +40 LOC | med | Accepted |
| 3.2 | Audit `docs/project/*` vs root `README.md` + `PROJECT_STATUS.md`. Rule: **README** = quickstart + layout + stage tables; **PROJECT_STATUS** = current phase + blockers; **docs/project/** = static runbooks and protocol specs. Remove or shrink whatever duplicates the first two. | Agent | M | 5–6f / -100 LOC | med | Accepted |
| 3.3 | Add missing `docs/README.md` explaining the four subfolders (`decisions/`, `experiments/`, `sessions/`, `project/`) and the naming convention. | Agent | S | 1f / +30 LOC | low | Accepted |
| 3.4 | Leave `docs/decisions/`, `docs/sessions/`, `docs/experiments/` structure untouched — it works. | — | — | 0 | — | Accepted |

---

## 4. Source layout — **flatten to a single `src/`** (biggest change)

**Context.** Current layout has three overlapping roots:

- `data_pipeline/src/` (real code, installed as `src.*` via a `pyproject.toml` trick)
- `segmentation_validation/` (real code, **not** installed, imported via cwd)
- root `src/` (empty stub, leftover from the commit `94c6997` move)

The trick in `pyproject.toml`:

```toml
[tool.setuptools.packages.find]
where = ["data_pipeline"]
include = ["src*", "scripts*"]
```

means `probe_cache.py` writes `from src.backbones.base import ...` — the
imports are **already root-relative**. The `data_pipeline/` wrapper adds a
directory level but no isolation. That's the tell: the wrapper is architectural
dead weight.

### Ron's proposal (verbatim)

> `src/` should maybe contain `segmentation_validation/` renamed to
> `src/segmentation/`.

### My take (Agent)

Ron's instinct is right, but incomplete. The real fix is to **collapse
`data_pipeline/` into root** so there is **one** `src/` tree:

```
src/
  backbones/          # from data_pipeline/src/backbones/
  common/             # from data_pipeline/src/common/
  data/               # from data_pipeline/src/data/
  embeddings/         # from data_pipeline/src/embeddings/
  segmentation/       # from segmentation_validation/  (Aviv/Lihi own — DO NOT MOVE YET)
  classification/     # placeholder for Stage 2
  baselines/          # placeholder
scripts/              # from data_pipeline/scripts/
tests/                # from data_pipeline/tests/ + segmentation_validation/tests/
configs/              # from data_pipeline/configs/ + segmentation_validation/configs/
patches/              # from data_pipeline/patches/
```

Why this is better than "just rename `segmentation_validation/` to `src/segmentation/`":

1. **Imports do not change.** `probe_cache.py` already uses `from src.backbones...`.
   Flattening to root just makes the packaging honest.
2. **One test root.** `pytest` picks up everything, no `testpaths` gymnastics.
3. **One config root.** Stage 1 configs already reference Stage 0 paths; a
   single `configs/` with `stage0/`, `segmentation/`, `classification/`
   subfolders matches how experiments are actually organized.
4. **No cross-directory imports.** Currently `segmentation_validation/README.md`
   claims code is "installed as the shared package" — false, because it lives
   outside `data_pipeline/`. Flattening removes the lie.
5. **The `data_pipeline` name was wrong to begin with.** The dir also contains
   `src/backbones/` (a model wrapper), `src/embeddings/` (feature extractor),
   and Stage 0 configs — it is more than a data pipeline. Naming it that made
   Aviv's segmentation work feel like a "second project" instead of the next
   layer of one project.

**Important constraint.** Aviv & Lihi are actively editing `segmentation_validation/`.
We must **not** move that directory in this branch. The plan:

- Move everything else into root `src/`, `scripts/`, `tests/`, `configs/`.
- Leave `segmentation_validation/` where it is for now.
- After Aviv/Lihi merge their PR, a **follow-up branch** (`seg-into-src`) moves
  `segmentation_validation/` → `src/segmentation/` in one small commit, updates
  their configs' paths, and deletes the old dir.

| # | Change | Proposer | Diff | Size | Sev | Status |
|---|---|---|---|---|---|---|
| 4.1 | Move `data_pipeline/src/` → root `src/`. Delete empty root stubs first. | Agent + Ron | M | ~40f moved | high | WIP (Commit 3, staged) |
| 4.2 | Move `data_pipeline/scripts/` → root `scripts/`. | Agent | S | ~12f moved | high | WIP (Commit 3, staged) |
| 4.3 | Move `data_pipeline/tests/` → root `tests/`. | Agent | S | ~12f moved | high | WIP (Commit 3, staged) |
| 4.4 | Move `data_pipeline/configs/` → root `configs/stage0/`. | Agent | S | ~5f moved | high | WIP (Commit 3, staged) |
| 4.5 | Move `data_pipeline/patches/` → root `patches/`. | Agent | S | ~1f moved | med | WIP (Commit 3, staged) |
| 4.6 | Delete `data_pipeline/` (empty by now). | Agent | S | 1d | med | WIP (Commit 3, staged) |
| 4.7 | Update `pyproject.toml`: `where = ["."]`, `include = ["src*", "scripts*"]`, `testpaths = ["tests", "segmentation_validation/tests"]`. | Agent | S | 1f / ~4 LOC | high | WIP (Commit 3, staged; same edit as §5.1) |
| 4.8 | Update every `data_pipeline/…` path in root README, PROJECT_STATUS, CONTRIBUTING, LLM_GUIDE, docs/. | Agent | M | ~15f | med | WIP (Commit 4, staged). Only `README.md` had stale layout content — rewritten to describe root `src/`, `scripts/`, `tests/`, `configs/`, `patches/` and add a pointer to `docs/project/LOCAL_DEV_WINDOWS.md`. `PROJECT_STATUS.md`, `CONTRIBUTING.md`, `LLM_GUIDE.md`, and the three subdir READMEs (`notebooks/`, `reports/`, `artifacts/`) had no stale references — they were already layout-independent. Historical session notes and `segmentation_validation/README.md` left untouched per ground rules. |
| 4.9 | **Deferred** (post-merge with Aviv/Lihi): rename `segmentation_validation/` → `src/segmentation/`. Separate branch. | Ron | M | ~10f | med | Deferred |
| 4.10 | Delete the `src/segmentation/README.md` placeholder — it will collide with 4.9. **Deferred to the §4.9 PR**: that PR does the collision resolution as part of its own move; keeping the placeholder here preserves the "future home" signal and keeps this PR's diff purely mechanical (Ron, 2026-07-30). | Agent → Ron | S | 1f | low | Deferred |

---

## 5. Packaging (`pyproject.toml`)

After §4, this becomes trivial:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["src*", "scripts*"]

[tool.pytest.ini_options]
addopts = "-ra"
testpaths = ["tests", "segmentation_validation/tests"]
```

Once §4.9 lands, `segmentation_validation/tests` drops out and `include` may
add `segmentation_validation*` if we choose to keep the dir alive during
transition. Nothing else changes.

| # | Change | Proposer | Diff | Size | Sev | Status |
|---|---|---|---|---|---|---|
| 5.1 | Rewrite the two blocks above. | Agent | S | 1f / ~6 LOC | high | WIP (Commit 3, staged) |
| 5.2 | Verify `pip install -e ".[dev]"` from repo root exposes `src.backbones`, `src.data`, `scripts.build_manifests`. Run in a clean venv on Windows and on the Technion VM. | Ron | S | — | high | **Windows: WIP-passing.** Ron 2026-07-30: install + pytest run in Python 3.13 venv at `C:\dev\venvs\cut3r\` (moved outside OneDrive to dodge MAX_PATH); 54/55 tests pass; the 1 failure (`test_cut3r_provenance::test_compatibility_patch_is_applied_and_validated`) is a numpy-1.26.4-on-cp313 native crash (exit `0xC0000005`), unrelated to flatten. **Technion VM: not run yet.** |

---

## 6. Configs — one root, YAML everywhere

**Context.**

- `data_pipeline/configs/stage0/*.yaml` — YAML, `${ENV}` interpolation.
- `segmentation_validation/configs/*.json` — JSON.

Inconsistency causes real friction: two loaders, two schemas, no shared
resolver. YAML wins (already dominant, supports comments, already has env
interpolation).

| # | Change | Proposer | Diff | Size | Sev | Status |
|---|---|---|---|---|---|---|
| 6.1 | After §4.4, structure is `configs/stage0/*.yaml`. Add `configs/README.md` describing the schema and env-var resolution. | Agent | S | 1f / +40 LOC | med | WIP (Commit 6). Existing thin README rewritten to include: directory layout for future stages, env-var resolution mechanism (strict-fail via `src/common/io.expand_environment`), the five env vars each Stage 0 config expects, top-level schema per section (`dataset`, `sampling`, `preprocessing`, `model`, `cache`, `output`), and rules for adding new configs. Original tier-description content preserved. |
| 6.2 | **Flag only** (read-only): recommend Aviv/Lihi convert `segmentation_validation/configs/*.json` → YAML in their next PR, for consistency with Stage 0. Do not touch. | Agent | — | — | med | Read-only |
| 6.3 | Add `configs/segmentation/` and `configs/classification/` after §4.9 lands. Not now. | Ron | — | — | low | Deferred |

---

## 7. Stale / empty directories at root

These exist on disk but are not tracked in git — leftover from the commit
`94c6997` mass-move. They confuse the layout in editors:

- `src/`, `tests/`, `scripts/`, `configs/`, `patches/`, `notebooks/`,
  `reports/`, `artifacts/` — some tracked as README-only stubs, some untracked
  empty dirs.

| # | Change | Proposer | Diff | Size | Sev | Status |
|---|---|---|---|---|---|---|
| 7.1 | Delete the untracked empty stubs **before** §4 moves, so `git mv` targets are clean. | Agent | S | — | med | WIP (done pre-Commit 2; no git changes needed since dirs were untracked) |
| 7.2 | Keep `notebooks/`, `reports/`, `artifacts/` at root (they are registries for external artifacts — mentioned in README). Update their READMEs after §4 renames. | Agent | S | 3f / ~30 LOC | low | WIP (Commit 4). Audited: all three README bodies already describe purpose without referencing `data_pipeline/`; no edits needed. |

---

## 8. Read-only findings on `segmentation_validation/` (no edits)

Aviv & Lihi own this directory. Findings for their consideration only:

| # | Finding | Sev | Status |
|---|---|---|---|
| 8.1 | `README.md` says code is "installed as the shared package" — this is currently false; the directory is not in `pyproject.toml`'s discovery path. §4/§5 fix this. | med | Read-only |
| 8.2 | Configs are JSON while Stage 0 uses YAML with `${ENV}` interpolation. Consider aligning (§6.2). | med | Read-only |
| 8.3 | `SegmentationProbe` supports `num_classes > 1` (multiclass) but the README says "Stage 1 is strictly binary". Good forward-compat, but confirm there is at least one test covering the multiclass branch, or explicitly mark it experimental. | low | Read-only |
| 8.4 | `SegmentationProbe.backbone` is stored as a plain attribute (not registered as a submodule). This is intentional per the docstring (keeps it out of `.parameters()`), but means `model.to(device)` will **not** move it. Worth a one-line note in the docstring for the next reader. | low | Read-only |
| 8.5 | `train_segmentation.py`, `segmentation_inference.py`, `segmentation_dataset.py`, `model_segmentation.py` live at the top of the directory rather than under a package (no `__init__.py`). Once §4.9 moves this to `src/segmentation/`, package init + explicit exports will be needed. | med | Read-only |

### Handoff to Aviv & Lihi

- This branch **does not move or edit** `segmentation_validation/`. You keep
  working on it exactly as today.
- What changes around you: `data_pipeline/` is deleted. Its subdirs are now at
  root (`src/`, `scripts/`, `tests/`, `configs/`, `patches/`). Any config or
  import in your dir that referenced `data_pipeline/…` needs the prefix
  dropped. Grep-and-replace: `data_pipeline/` → `` (empty). We will do a
  best-effort scan in §4.8 but please double-check your JSON configs after
  rebase.
- The follow-up branch `seg-into-src` (§4.9) will rename
  `segmentation_validation/` → `src/segmentation/` *after* your PR merges. We
  will coordinate timing on Slack/GitHub before starting.

---

## 9. Read-only findings on `data_pipeline/src/backbones/probe_cache.py`

Aviv & Lihi co-own. **Do not edit on this branch.** The §4 flatten moves this
file to `src/backbones/probe_cache.py` via `git mv` — contents unchanged,
history preserved. Findings only:

| # | Finding | Sev | Status |
|---|---|---|---|
| 9.1 | Uses `from src.backbones.base import ...` — already root-relative. Compatible with §4 as-is; no import changes needed post-flatten. | — | Read-only |
| 9.2 | 550 LOC in one module. Long-term this is worth splitting into `layout_trajectory.py`, `layout_target_only.py`, `writer.py`, `reader.py`. Not urgent — mention next time the owners open it. | low | Read-only |
| 9.3 | Metadata schema (category vocab, layout tag, SHA-256) is defined inline. Consider a `probe_cache_schema.py` or a pydantic/dataclass model so classification (Stage 2) reuses it without copy-paste. | low | Read-only |

### Handoff to Aviv & Lihi

- This branch renames `data_pipeline/src/backbones/probe_cache.py` →
  `src/backbones/probe_cache.py` and **does not touch its contents**. Git records
  it as a rename, so `git log --follow` and `git blame` stay intact.
- When you rebase your in-flight branch on the merged main, git's rename
  detection will reapply your edits at the new path automatically. Expected
  outcome: no conflicts.
- Risk: if your working copy has a very large rewrite of the file (>50% content
  change), rename detection may miss and treat the change as delete + add,
  effectively discarding your edits. Mitigation: pause big rewrites until this
  PR merges, or rebase early. Ping Ron if unsure.
- Imports do not change (`from src.backbones.base import ...` works before and
  after).

---

## 10. Execution order

**Ron's decision (2026-07-30):** consolidate §§1–8 into a **single PR** on this
branch. My earlier three-PR split was over-cautious; the sections do not block
each other and none depend on Aviv/Lihi.

Justification for one PR:
- No overlap with `segmentation_validation/` or `probe_cache.py` — no merge
  conflict risk with in-flight work.
- All content is mechanical (moves, path fixes, README additions) except §3.2
  (docs de-dup, judgement call). §3.2 is the one section that could reasonably
  be split out for reviewability; keeping it in-PR is fine if it's the last
  commit, so reviewers can look at it independently.
- The whole branch **is** the "one issue, one branch" unit from
  `CONTRIBUTING.md`. Sub-splitting inside it is not required.

**Ordered commits inside the single PR** (each is `git`-atomic and reviewable):

1. **§7.1** — delete untracked empty root dirs.
2. **§1.1, §1.2, §1.3** — meta files + Copilot instructions + LLM_GUIDE audit.
3. **§4.1–4.7, §5.1** — the flatten (`git mv` per subdir, then
   `pyproject.toml`). One commit per subdir move keeps `git log --follow`
   working. §4.10 is deferred to the §4.9 PR (see below).
4. **§4.8, §7.2** — path fixes across README, PROJECT_STATUS, CONTRIBUTING,
   LLM_GUIDE, docs/, and root subdir READMEs.
5. **§5.2** — Ron verifies install + `pytest` on Windows and Technion VM.
   Result recorded in PR description, not code.
6. **§2.1, §2.2, §2.4** — `.github/prompts/` + `fix_pr_comments.prompt.md`
   (§11) + CODEOWNERS refresh.
7. **§6.1** — `configs/README.md`.
8. **§3.3, §3.2** — `docs/README.md` first, then the de-dup audit (which will
   reference §3.3's index). This is the judgement-heavy commit — keep it last
   so reviewers can look at it independently.

**Explicitly out of this PR / branch:**
- §4.9, §4.10 — the `segmentation_validation/` → `src/segmentation/`
  rename **and** the placeholder README delete. Follow-up branch `seg-into-src`
  after Aviv/Lihi merge.
- §6.2, §8.x, §9.x — filed as GitHub issues assigned to Aviv/Lihi.
- §2.3 — `.claude/agents/` tree deferred until a workflow needs it.

## Progress log

Living record of what has been implemented on this branch. Update at every
commit-readiness handoff. Uses the same Status vocabulary as §§1–9.

| # | Commit | Contents | Status |
|---|---|---|---|
| 1 | pre-branch cleanup | §7.1 | Done (untracked dirs; nothing to git-commit) |
| 2 | `2698cd1` | §1.1, §1.2, §1.3 — `.github/copilot-instructions.md`, LLM_GUIDE working-style section, plan landed as session note | Committed on branch |
| 3 | staged | §4.1–4.7, §5.1 — flatten `data_pipeline/` into root + `pyproject.toml` rewrite | Staged, awaiting Ron's `git commit` |
| 4 | staged | §4.8, §7.2 — path fixes across README, PROJECT_STATUS, CONTRIBUTING, LLM_GUIDE, docs/, subdir READMEs | Staged, awaiting Ron's `git commit`. Scope collapsed to a single file (`README.md`) after audit — the other targets were already layout-independent. |
| 5 | not started | §5.2 — Ron verifies install + pytest in Python 3.11+ venv on Windows and Technion VM | Windows part complete (54/55 pass; single failure is numpy-1.26.4 cp313 native crash, unrelated to flatten). Technion VM part still owed. |
| 6 | staged | §2.1, §2.2, §2.4 — `.github/prompts/` + `fix_pr_comments.prompt.md` (§11) + CODEOWNERS refresh | Staged, awaiting Ron's `git commit`. §2.4 turned out to be a no-op today (CODEOWNERS has no path patterns); genuine ownership refresh filed as a follow-up pending GitHub handles. |
| 7 | staged | §6.1 — `configs/README.md` rewritten with schema + env-var resolution | Staged, awaiting Ron's `git commit`. |
| 8 | not started | §3.3, §3.2 — `docs/README.md` + docs de-dup audit | Accepted |

**Deferred to follow-up branch `seg-into-src` (post-merge with Aviv/Lihi):**
§4.9, §4.10.

**Deferred to GitHub issues assigned to Aviv/Lihi:** §6.2, §8.1–8.5, §9.1–9.3.

**Deferred until a workflow needs it:** §2.3 (`.claude/agents/` tree).

### Follow-up notes surfaced during PR execution

- **Promote `fix_pr_comments` prompt to a skill if the team forgets to invoke
  it (Commit 5 decision).** Kept as a prompt for cross-tool portability and
  because the negotiate-first flow benefits from explicit user consent. If
  real usage shows the team defaults to "just fix what the reviewer said"
  without invoking the prompt, promote it to a Copilot skill or Claude
  subagent under `.claude/agents/` so it fires automatically on PR-comment
  context. Revisit after two or three PRs land.
- **`reject_unknown_keys` is dead code (Commit 6 audit).**
  `src/common/io.reject_unknown_keys` is defined but never called; today
  Stage 0 configs silently ignore typo'd top-level keys. Wire it into the
  config-loading path in a small hardening PR (needs an explicit list of
  allowed keys per section — candidate for a dataclass or pydantic model).
  Documented in `configs/README.md` so users know the gap exists.
- **`CODEOWNERS` refresh (Commit 5 audit).** Today `.github/CODEOWNERS` is
  `* @Maximilianb1`, which routes every review to Max. Once GitHub handles for
  the rest of the team are documented, add path-scoped ownership: Aviv & Lihi
  on `segmentation_validation/` and `src/backbones/probe_cache.py`; team-wide
  on `docs/` and `configs/`. Track as a GitHub issue.
- **numpy 1.26.4 + Python 3.13 (Windows) (Commit 3 §5.2).** No official cp313
  wheel; PyPI serves a MinGW-w64 build that warns "CRASHES ARE TO BE EXPECTED"
  and does crash (exit `0xC0000005`) inside
  `scripts.apply_cut3r_compatibility_patch`. Not a flatten regression. Options
  for a separate PR: (a) pin `python = ">=3.11,<3.13"` in `pyproject.toml` and
  standardize on 3.12, or (b) bump `numpy` to 2.x (needs a wider
  dep-compatibility check with torch/pyarrow). Track as a GitHub issue after
  this PR merges.
- **Windows workspace inside OneDrive is hostile to venvs (Commit 3 §5.2).**
  Two failures hit during setup: (1) pip's `setuptools 58 → 83` uninstall step
  fails `[Errno 22]` when OneDrive holds file handles; workaround was
  `python -m venv --upgrade-deps` (Python 3.13 no longer bundles setuptools).
  (2) `WinError 206 filename too long` on torch's nested license paths;
  workaround was moving the venv to `C:\dev\venvs\cut3r\` outside OneDrive.
  Both workarounds are documented in `docs/project/LOCAL_DEV_WINDOWS.md`
  (added in Commit 3) so Aviv, Lihi, and Max don't have to rediscover them.
  Long-term fix: enable Windows long-path support system-wide, or standardize
  on the Technion VM.

---

## 11. `fix_pr_comments` prompt spec (Ron)

Recorded here so §2.2 can be authored inside this PR without further clarification.

**Purpose.** A tool-agnostic prompt any agent (Copilot, Claude, Codex) can be
invoked with to help the user respond to and fix PR review comments.

**Behaviour, per comment thread:**

0. **Ingest source.** If the agent has direct GitHub access (MCP GitHub server,
   `gh` CLI, or equivalent), fetch the PR, thread, and diff directly. Otherwise
   ask the user to paste the reviewer's comment and the file+line range it
   points at.
1. **Read code context.** Load the relevant code region (from disk or from the
   fetched diff).
2. **Read the comment.** Include any code suggestion the reviewer attached.
3. **State a position.** Say clearly whether the comment is (a) correct and
   worth fixing, (b) partially correct, or (c) wrong / not applicable. Justify
   in one paragraph. **Wait for the user's response.**
4. **Negotiate.**
    - If user and agent agree → skip to 5.
    - If they disagree, the agent may **challenge the user's reasoning** with
      evidence (docs link, code reference, standard). It may equally **admit
      it was wrong** if the user's counter is stronger. Loop until agreement
      or until either side declares dead-end. On dead-end, the agent
      **explicitly logs "agree to disagree"** and follows the user's decision.
5. **Act.**
    - If the outcome is *fix*: apply the code change (matching repo style,
      running tests if configured), then draft a short "resolved by …" reply
      for the user to post.
    - If the outcome is *defend*: draft a polite, evidence-based reply to the
      reviewer for the user to post; do not change code.

**Non-goals / guardrails:**
- Never push commits or post GitHub comments unattended — always hand control
  back to the user before writing to the remote.
- Never invent test results or claim a fix works without running the relevant
  test.
- Respect all project guardrails in `LLM_GUIDE.md` (no split leakage, no
  renaming a nonlinear MLP to "linear probe", etc.).
- Handle one comment thread at a time; batch mode is opt-in.

## Attribution summary

| Proposer | Items |
|---|---|
| Ron | 1 (repetitive meta files?), 2 (`.github/` skills), 3 (need for `docs/`?), 4 (put `segmentation_validation` under `src/`) — plus 1.3, 2.2, 3.1, 4.9, 5.2, 6.3 |
| Agent (Opus-4.7) | Everything else: `.github/copilot-instructions.md` gap (1.2), prompts-vs-subagents distinction (2), docs de-duplication (3.2/3.3), **flatten `data_pipeline/` into root** (4.1–4.8, 4.10), packaging cleanup (5), config format convergence (6), stale-dir cleanup (7), read-only findings (8, 9), execution order (10) |

## Open questions — resolved 2026-07-30

1. **Flatten `data_pipeline/`?** ✅ Confirmed by Ron.
2. **`fix_pr_comments` spec?** ✅ Provided by Ron, recorded in §11.
3. **CI in this branch?** *Deferred.* Meaning: add `.github/workflows/*.yml`
   that runs `pytest` on every PR so the "58 tests pass" claim is machine-verified
   for every reviewer, not just on Aviv's laptop. Not essential to the
   redesign — the flatten does not depend on CI existing. Filed as a follow-up
   issue.
4. **`docs/project/` archival files?** *Deferred to §3.2 execution.* Meaning:
   during the docs de-dup audit, some content in `docs/project/*.md` may
   deserve to be kept verbatim even if it duplicates the README (e.g. the
   Technion VM runbook is a snapshot of a specific environment). Decision
   deferred until the audit commit — will ping Ron per file if unclear.

{
  "chat.tools.terminal.autoApprove": {
    "git": true,
    "Get-ChildItem": true,
    "Test-Path": true,
    "Measure-Object": true,
    "Select-Object": true,
    "Write-Host": true,
    "Remove-Item": false,
    "rm": false,
    "/^git (status|log|show|diff|ls-files|fetch|branch|checkout|mv|add)\\b/": true
  },
  "chat.tools.autoApprove": false
}
