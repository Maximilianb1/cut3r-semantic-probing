# Prompt files

Reusable prompts checked into the repository so any team member can invoke a
saved workflow the same way, in any coding-agent tool.

## Why a repository-level convention

Every coding-agent tool has its own auto-discovery convention, and they do not
overlap:

- **GitHub Copilot in VS Code** auto-discovers `.github/prompts/*.prompt.md`
  and adds them to the chat slash-command menu. See
  [VS Code Copilot prompt files](https://code.visualstudio.com/docs/copilot/copilot-customization#_prompt-files-experimental).
- **Claude Code**, **Codex CLI**, **Cursor**, and **Aider** do not scan
  `.github/prompts/` automatically, but their users can attach any Markdown
  file to a session on demand (`@filename` or an equivalent) and get the same
  behavior.

Putting reusable prompts under `.github/prompts/` gets Copilot's auto-discovery
for free and leaves them plainly readable for every other tool.

## What belongs here

- A concrete, repeatable workflow that a team member would otherwise re-type or
  re-explain each time. Example: "respond to a PR review comment".
- Tool-agnostic language. Do not hard-code Copilot-only tool names in the body
  (`vscode.chatEditFiles`, etc.). Describe the intent; let the tool decide how
  to execute.

## What does not belong here

- Project guardrails (data-leakage rules, naming conventions, ADR requirements)
  — those live in [../../LLM_GUIDE.md](../../LLM_GUIDE.md) so every tool loads
  them automatically.
- Copilot-specific auto-loaded rules — those live in
  [../copilot-instructions.md](../copilot-instructions.md).
- One-off requests. If you would only run it once, keep it in your local
  conversation.

## File format

Each file uses `.prompt.md` and starts with a minimal YAML frontmatter block
containing a `description` field, followed by the prompt body:

```markdown
---
description: One-line summary shown in menus.
---

# Prompt title

Intent, inputs, and steps as imperative bullet points.
```

Copilot users invoke a prompt by typing `/<filename-without-extension>` in the
Copilot chat panel. Other tools can `@` the file path or paste its contents.

## Adding a new prompt

1. Pick a short, descriptive lowercase filename (`fix_pr_comments`,
   `write_adr`, `review_test_coverage`).
2. Write imperative bullet steps. Keep the body tool-agnostic.
3. Mention any repository files the prompt should read first (guardrails,
   templates, previous session notes).
4. Open a small PR. Prompts are shared infrastructure — one review is enough.
