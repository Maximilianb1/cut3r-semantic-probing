---
description: Negotiate and resolve a single PR review comment thread — read code, take a position, agree or disagree, then act.
---

# Fix a PR review comment

You are helping the user respond to one review-comment thread on a pull
request. Read this whole prompt before doing anything. Then execute the steps
below in order.

## Ground rules (non-negotiable)

- Handle **one comment thread at a time**. If the user asks for batch mode,
  ask them to confirm and then loop the steps below per thread.
- **Never push commits or post GitHub comments unattended.** Hand control back
  to the user before writing anything to the remote.
- **Never invent test results.** If you claim a fix works, you must have run
  the relevant test in this session. Otherwise say "not yet verified".
- Respect every guardrail in [../../LLM_GUIDE.md](../../LLM_GUIDE.md) (no split
  leakage, no silent renames, no unverified benchmarks, and so on).
- The user has the final call at any step. If they say "just do X", stop
  negotiating and do X.

## Step 0 — Ingest the comment thread

- If you have direct GitHub access (an MCP GitHub server, the `gh` CLI, or the
  VS Code Pull Request extension), fetch the PR, the specific thread, the diff
  hunk it points at, and any code suggestion the reviewer attached. Report
  what you fetched.
- If you do not have GitHub access, ask the user for four things and wait for
  the answer:
    1. the reviewer's comment text (verbatim);
    2. the file path and line range the comment points at;
    3. any code suggestion the reviewer attached;
    4. the PR URL, so you can quote it in the resolution reply.

## Step 1 — Read the code

- Read the file on disk at the exact line range from step 0. Read at least
  30 lines around the target so you understand context.
- If the code depends on other symbols, read their definitions too. Do not
  guess.
- Do not modify anything yet.

## Step 2 — Read the comment carefully

- Restate the reviewer's point in your own words in one sentence. Confirm you
  understood it before continuing.
- If the comment references a standard, doc, or code area you have not
  loaded, load it now.

## Step 3 — Take a position (and stop)

- State clearly which of these applies:
    - **(a) Correct and worth fixing** — the reviewer is right, and the fix
      is worth the change.
    - **(b) Partially correct** — some of the point is right; the rest is
      not. Say which.
    - **(c) Wrong or not applicable** — the reviewer's premise does not hold
      in this codebase.
- Justify the position in **one paragraph** with concrete evidence:
    - a link to a repository file (`path/to/file.py#L42`),
    - an ADR reference (`docs/decisions/…`),
    - or an external doc / standard.
- **Stop and wait for the user's response.** Do not act yet.

## Step 4 — Negotiate

- If the user agrees with your position, go to step 5.
- If the user disagrees:
    - Ask what specifically they disagree with.
    - You may **challenge their reasoning** with new evidence — a docs link,
      a code reference, a benchmark. Be direct, not deferential.
    - You may equally **admit you were wrong** if their counter-argument is
      stronger. Say so plainly.
    - Loop until you reach agreement or either side declares the discussion
      a dead end. On dead end, log the disagreement in the eventual reply
      ("agree to disagree; the author chose to X because Y") and follow the
      **user's** decision.

## Step 5 — Act on the agreed outcome

Take one of the two actions below, based on what step 4 concluded:

### Outcome: fix the code

- Apply the change. Match the repository's existing style. Do not touch
  unrelated code.
- Run the tests that cover the changed code. Report the exact command you
  ran and its output ("`pytest tests/test_windows.py -x` → 4 passed").
- If tests fail or you cannot run them (missing env, external dependency),
  say so. Do not claim success.
- Draft a short "resolved by …" reply for the user to paste into the PR
  thread. Include the commit SHA once the user commits, or say "pending
  commit" if they have not yet.
- Return control. The user commits and posts.

### Outcome: defend the code (no fix)

- Draft a polite, evidence-based reply to the reviewer. Cite the same
  evidence you used in step 3. Assume the reviewer is technically strong
  and short on time.
- Do **not** change any code.
- Return control. The user posts the reply.

## Optional: batch mode

- Only if the user explicitly asks for it. Repeat steps 0–5 per thread,
  finishing each one fully (fix + reply drafted) before starting the next.
- Never batch step 3 across threads; each position needs its own review.
