# Session: Codebase and data-pipeline redesign

- Date: 2026-07-30
- Author: Ron Bartal
- Branch: `codebaseNdatapipeline-redesign`
- Related issue/PR: TBD (this branch)
- Assistant/model, if used: GitHub Copilot (Claude Opus 4.7)

## Objective

Turn the half-migrated layout (root + `data_pipeline/` + `segmentation_validation/`) into a single-package monorepo so 6 people can collaborate on Stages 1–2 without stepping on each other. Land the change in one PR, keep it mechanical. `segmentation_validation/` was originally deferred to a follow-up branch out of respect for Aviv & Lihi's in-flight work; once their PR #12 merged onto `main` mid-branch, this branch rebased onto it and absorbed the seg rename (§4.9), the placeholder-README delete (§4.10), and the `__init__.py` add (§8.5) so the layout ships in one PR.

## Context and inputs

- Layout before this branch had three overlapping roots: `data_pipeline/src/` (installed as the importable `src` via a `where = ["data_pipeline"]` packaging trick), `segmentation_validation/` (real code, not installed, imported via cwd), and an empty root `src/` stub. Imports across the codebase were already root-relative (`from src.…`), so the `data_pipeline/` wrapper was architectural dead weight rather than isolation.
- Configs were split between YAML with `${ENV}` interpolation (Stage 0) and JSON (segmentation), tests lived in two roots, and `data_pipeline/` was a misnomer — it hosted backbones, embeddings, scripts, patches, and Stage 0 configs, not only data.
- Ground rules honoured throughout the branch:
  - No edits or moves inside `segmentation_validation/` (Aviv & Lihi own it).
  - No content edits to `data_pipeline/src/backbones/probe_cache.py` (Aviv & Lihi co-own; `git mv` only).
  - All other repo content in scope.
- The full working plan and Ron ↔ agent negotiation record is preserved off-tree at `local/2026-07-30-codebase-redesign-plan.md` (git-ignored) and in git history at commit `2698cd1`.

## Work completed

Commits landed on the branch, in order:

| # | Commit | Contents |
|---|---|---|
| 1 | pre-branch cleanup | Deleted untracked empty stub directories at repo root so `git mv` targets in Commit 3 were clean. |
| 2 | `2698cd1` | Added `.github/copilot-instructions.md` (Copilot auto-discovery convention was missing). Extended `LLM_GUIDE.md` with a "Working style" section (register rules for human-facing vs agent-facing files; critical-thinking guidance; explicit "these convention filenames are load-bearing, do not delete" note about `AGENTS.md` / `CLAUDE.md` / `GEMINI.md`). Landed the full working plan as `docs/sessions/2026-07-30-codebase-redesign-plan.md` (Commit 8 later trimmed and renamed it). |
| 3 | `204b96a` | Flatten: `git mv` `data_pipeline/{src, scripts, tests, configs, patches}` → repo root; deleted `data_pipeline/`. Rewrote `pyproject.toml` to `where = ["."]`, `include = ["src*", "scripts*"]`, `testpaths = ["tests", "segmentation_validation/tests"]`. Added `docs/project/LOCAL_DEV_WINDOWS.md` capturing the venv-outside-OneDrive lesson (see Decisions → Still open, numpy/cp313 note). |
| 4 | `d4998f2` | Retargeted root `README.md` to the flattened layout: structure table and Getting-started rewritten; pointer to `docs/project/LOCAL_DEV_WINDOWS.md` added. Audited `PROJECT_STATUS.md`, `CONTRIBUTING.md`, `LLM_GUIDE.md`, and the `notebooks/` / `reports/` / `artifacts/` READMEs — all already layout-independent, no edits needed. Historical session notes and `segmentation_validation/README.md` left untouched. |
| 5 | `830f13f` | Added `.github/prompts/README.md` documenting the reusable-prompt convention (Copilot auto-discovery, cross-tool invocation via `@`). Authored `.github/prompts/fix_pr_comments.prompt.md` (six-step negotiate-then-fix flow per Appendix A). Audited `.github/CODEOWNERS` — no path patterns to update at that time; refresh queued in Commit 8 once handles were known. |
| 6 | `ffecd99` | Rewrote `configs/README.md` — env-var resolution semantics (strict-fail via `src/common/io.expand_environment`), the five Stage 0 env vars, top-level schema per section (`dataset`, `sampling`, `preprocessing`, `model`, `cache`, `output`), and rules for adding new configs. |
| 7 | `4f38844` | Added `docs/README.md` — subfolder layout table, filename conventions, "where does a new document belong" cookbook. |
| 8 | `060af1f` | Reshaped this session note to fit `docs/sessions/template.md`; preserved the long working plan off-tree; updated `PROJECT_STATUS.md`; refreshed `.github/CODEOWNERS` to add Ron as a co-owner. |
| — | rebase | Rebased the branch onto Aviv & Lihi's PR #12 (`eee10c9`, "Scope segmentation_validation to probe training and evaluation"). Real conflicts in three files: `pyproject.toml` (took our `where`/`include`; kept their new `tqdm` dep; `testpaths = ["tests"]` because their PR emptied `segmentation_validation/tests/`), `data_pipeline/README.md` (took our delete), `configs/README.md` (folded their new `probe_features/` paragraph into our rewritten Directory-layout and Related sections; corrected the stale "JSON today" claim about `segmentation_validation/configs/`). Git rename detection auto-carried their `probe_cache.py` (+42) and `extract_probe_features.py` (+12) edits onto the flattened paths. Their three new `configs/probe_features/*.yaml` files and `tests/test_backbones.py` and `tests/test_segmentation_metrics.py` landed at the flattened locations; the seg-metrics test needed one path fix (`parents[2]` → `parents[1]`) folded into Commit 3 via `--fixup` + `--autosquash`. Force-pushed with `--force-with-lease`. |
| 9 | (this commit) | Physical move `segmentation_validation/` → `src/segmentation/`: their four `*.py` drivers, three probe-head YAML configs, and 94-line README; deleted the pre-existing placeholder `src/segmentation/README.md`. Added `src/segmentation/__init__.py` so the package is importable. Converted the drivers' bare-name imports to relative package imports (`from .model_segmentation import …`). Retargeted every in-tree reference from `segmentation_validation/…` to `src/segmentation/…`: root `README.md`, `PROJECT_STATUS.md`, `LLM_GUIDE.md`, `configs/README.md`, `configs/probe_features/*.yaml` cross-link comments, `scripts/extract_probe_features.py` docstring, and the moved README's own command examples (`python -m src.segmentation.train_segmentation …`). Simplified `tests/test_segmentation_metrics.py` — the `sys.path.insert` hack is replaced by a direct `from src.segmentation.train_segmentation import BinaryMetrics`. Added `*.egg-info/` and `runs/` to `.gitignore`. |
| 10 | not started | §3.2 docs de-dup audit across `docs/project/*.md` vs root `README.md` and `PROJECT_STATUS.md`. Judgement-heavy; will pause on individual files where intent is unclear. |

What teammates will feel on rebase:

- Imports do not change for Stage 0 code — every module was already using `from src.…` root-relative paths, and `src.backbones.probe_cache` still lives at the same import path.
- Configs move from `data_pipeline/configs/stage0/` → `configs/stage0/`. Aviv & Lihi's new probe-feature extraction configs move from `data_pipeline/configs/probe_features/` → `configs/probe_features/`. Their probe-head configs move from `segmentation_validation/configs/` → `src/segmentation/configs/`.
- Stage 0 tests, scripts, patches move from `data_pipeline/{tests, scripts, patches}/` → repo root.
- `segmentation_validation/` no longer exists. All four drivers and the three probe-head configs live under `src/segmentation/`. Training and inference invocation changes from `cd segmentation_validation && python train_segmentation.py …` to `python -m src.segmentation.train_segmentation --config src/segmentation/configs/<backbone>.yaml` (run from repo root). Their internal imports are now relative (`from .model_segmentation import …`).
- One new reusable prompt is available: `.github/prompts/fix_pr_comments.prompt.md`. Copilot auto-discovers it; other tools invoke it via `@` (see `.github/prompts/README.md`).

## Decisions

Made:

- **Flatten `data_pipeline/` into the repo root**, not the smaller "rename `segmentation_validation/` under `src/`" change originally proposed. Justification: the `where = ["data_pipeline"]` packaging trick and the fact that imports were already root-relative showed the wrapper was architectural dead weight, not isolation.
- **Keep** `AGENTS.md`, `CLAUDE.md`, `GEMINI.md` as one-line redirects to `LLM_GUIDE.md`. Each filename is a different tool's auto-discovery convention (Codex / Cursor / Aider, Claude Code, Gemini CLI), not repetition. Documented in `LLM_GUIDE.md`.
- **Add** `.github/copilot-instructions.md` — real gap since the team uses Copilot in VS Code.
- **Prompt, not skill, for `fix_pr_comments`** — cross-tool portable; the negotiate-first flow benefits from explicit user consent, so user-triggered invocation is a feature. Revisit if the team forgets to invoke it (see Still open).
- **Do not create `.claude/agents/`** yet. Add it when a workflow needs its own tool policy.
- **Split the docs de-dup audit** (§3.2) off from the mechanical redesign work — it lands as its own Commit 10 so reviewers can look at judgement calls independently.
- **Absorb the segmentation rename** (§4.9 + §4.10 + §8.5) into this branch after Aviv & Lihi's PR #12 merged, rather than deferring to a `seg-into-src` follow-up. Justification: the follow-up would immediately conflict with itself (their probe-feature configs already reference `segmentation_validation/configs/`, the `configs/README.md` we wrote already documents that path, and the test we inherited already uses a `sys.path.insert` hack pointing at the old dir). Consolidating avoids two round-trips of cross-directory retargeting and gives teammates one atomic layout to rebase against.
- **Preserve the long working plan off-tree** at `local/2026-07-30-codebase-redesign-plan.md` (git-ignored via new `local/` entry in `.gitignore`) rather than tracking two versions of the same document. Git history at `2698cd1` also holds the original.

Still open:

- `reject_unknown_keys` in `src/common/io.py` is defined but never called; today Stage 0 configs silently ignore typo'd top-level keys. Wire it into the config-loading path in a small hardening PR. Needs an explicit list of allowed keys per section — candidate for a dataclass or pydantic model. Documented in `configs/README.md` so users know the gap exists.
- `.github/CODEOWNERS` path-scoped ownership for Aviv & Lihi (`segmentation_validation/` and `src/backbones/probe_cache.py`) is blocked on their handles being documented anywhere. Ron added in Commit 8; team-wide `docs/` and `configs/` scoping deferred to the follow-up refresh.
- numpy 1.26.4 on Python 3.13 (Windows) native crash in one test (`test_cut3r_provenance::test_compatibility_patch_is_applied_and_validated`, exit `0xC0000005`). Not a flatten regression. Fix options: pin `python = ">=3.11,<3.13"` in `pyproject.toml`, or bump `numpy` to 2.x (needs a wider dep-compat check with torch/pyarrow). Track as a GitHub issue.
- Promote `fix_pr_comments` from a prompt to a Copilot skill or Claude subagent if real usage shows the team defaults to fixing without invoking it. Revisit after two or three PRs land.

## Verification

| Command/check | Result |
|---|---|
| `pip install -e ".[dev]"` in Python 3.13 venv on Windows | Succeeds. Venv at `C:\dev\venvs\cut3r\` (moved outside OneDrive to avoid `MAX_PATH` on torch's nested license paths). |
| `pytest tests/` on Windows, same venv, before rebase | 54/55 pass. The one failure is the numpy-1.26.4-cp313 native crash noted above, unrelated to the flatten. |
| `pytest tests/` on Windows, same venv, after rebase + Commit 9 | 59 passed, 1 pre-existing failure (the same numpy crash). +5 tests are Aviv & Lihi's new `tests/test_backbones.py` and `tests/test_segmentation_metrics.py`. |
| `pip install -e ".[dev]"` + `pytest tests/` on the Technion VM | **Not yet run.** Owed by Ron before PR merge. |

## Human review of AI-assisted work

Ron reviewed every commit before it was made and drove multiple scope collapses — most notably Commit 4, where the plan originally listed ~15 files to retarget and the audit revealed only `README.md` needed layout-driven changes. Ron authored the `fix_pr_comments` prompt spec verbatim (Appendix A); the agent implemented it in `.github/prompts/fix_pr_comments.prompt.md`. Ron caught the Commit 6 draft claim that "unknown config keys are rejected" — grep showed `reject_unknown_keys` is defined but never called — and the `configs/README.md` wording was corrected before commit. Ron chose to keep `fix_pr_comments` as a prompt rather than a skill after the agent laid out the trade-offs.

## Next step

Commit 10 — §3.2 docs de-dup audit across `docs/project/*.md` (`SCOPE`, `EVALUATION_PROTOCOL`, `WORK_BREAKDOWN`, `FULL51_TWO_PART_RUNBOOK`, `TECHNION_VM_RUNBOOK`) against root `README.md` and `PROJECT_STATUS.md`. Rule of thumb: **README** = quickstart + layout + stage tables; **PROJECT_STATUS** = current phase + blockers; **docs/project/** = static runbooks and protocol specs. Remove or shrink whatever duplicates the first two. Owner: agent + Ron.

After Commit 10: Ron runs the Technion VM install + pytest sanity check, then PR opens.

---

## Appendix A — `fix_pr_comments` prompt spec

Recorded here as authored by Ron; implemented in `.github/prompts/fix_pr_comments.prompt.md`.

**Purpose.** A tool-agnostic prompt any agent (Copilot, Claude, Codex) can be invoked with to help the user respond to and fix PR review comments.

**Behaviour, per comment thread:**

0. **Ingest source.** If the agent has direct GitHub access (MCP GitHub server, `gh` CLI, or equivalent), fetch the PR, thread, and diff directly. Otherwise ask the user to paste the reviewer's comment and the file+line range it points at.
1. **Read code context.** Load the relevant code region (from disk or from the fetched diff).
2. **Read the comment.** Include any code suggestion the reviewer attached.
3. **State a position.** Say clearly whether the comment is (a) correct and worth fixing, (b) partially correct, or (c) wrong / not applicable. Justify in one paragraph. **Wait for the user's response.**
4. **Negotiate.**
    - If user and agent agree → skip to 5.
    - If they disagree, the agent may **challenge the user's reasoning** with evidence (docs link, code reference, standard). It may equally **admit it was wrong** if the user's counter is stronger. Loop until agreement or until either side declares dead-end. On dead-end, the agent **explicitly logs "agree to disagree"** and follows the user's decision.
5. **Act.**
    - If the outcome is *fix*: apply the code change (matching repo style, running tests if configured), then draft a short "resolved by …" reply for the user to post.
    - If the outcome is *defend*: draft a polite, evidence-based reply to the reviewer for the user to post; do not change code.

**Non-goals / guardrails:**

- Never push commits or post GitHub comments unattended — always hand control back to the user before writing to the remote.
- Never invent test results or claim a fix works without running the relevant test.
- Respect all project guardrails in `LLM_GUIDE.md`.
- Handle one comment thread at a time; batch mode is opt-in.

## Appendix B — Follow-up backlog

Items surfaced during the redesign work that intentionally do not land in this PR. Each should become its own issue or follow-up branch.

- **Promote `fix_pr_comments` from prompt to skill/subagent** if the team forgets to invoke it. Symptom to watch for: "just fix what the reviewer said" without a first-round position statement. Revisit after two or three PRs.
- **Wire `src/common/io.reject_unknown_keys` into config loading.** Needs an explicit allowed-keys list per section — candidate for a dataclass or pydantic model. Small hardening PR.
- **`.github/CODEOWNERS` path-scoped ownership.** Once Aviv & Lihi's GitHub handles are documented, add: Aviv & Lihi on `segmentation_validation/` and `src/backbones/probe_cache.py`; team-wide on `docs/` and `configs/`. Ron was added as a co-owner in Commit 8; Max remains sole reviewer for everything else until then.
- **numpy 1.26.4 + Python 3.13 (Windows) native crash.** PyPI serves a MinGW-w64 build of `numpy` 1.26.4 for cp313 that warns "CRASHES ARE TO BE EXPECTED" and does crash inside `scripts.apply_cut3r_compatibility_patch` (exit `0xC0000005`). Fix options: (a) pin `python = ">=3.11,<3.13"` and standardize on 3.12, or (b) bump `numpy` to 2.x (needs a wider dep-compat check with torch/pyarrow). Track after this PR merges.
- **Windows workspace inside OneDrive is hostile to venvs.** Documented in `docs/project/LOCAL_DEV_WINDOWS.md`. Long-term fix: enable Windows long-path support system-wide, or standardize on the Technion VM.
- **CI: `.github/workflows/*.yml` running `pytest` on every PR.** Not essential to the redesign; would make "58 tests pass" machine-verified for every reviewer.
