# ADR 0001: Repository governance and tracking

- Status: Accepted
- Date: 2026-07-17
- Owners: Project team
- Related issue/PR: Initial repository creation

## Context

Six teammates will work in parallel and may use different coding assistants. The project needs a lightweight shared record of scientific choices, implementation progress, experiments, and handoffs.

## Decision

- `main` represents reviewed, coherent project state.
- Work is organized by responsibility, not by teammate.
- Meaningful changes use issues, short-lived branches, pull requests, and one teammate review.
- ADRs record durable scientific and architectural choices.
- Session notes record handoffs and substantial work.
- Experiment records capture reproducibility and results.
- `PROJECT_STATUS.md` is the current project-level summary.
- `LLM_GUIDE.md` is the canonical instruction file for all assistants.

## Consequences

The team must update the relevant small record as part of completing work. Raw chats and minute-by-minute diaries are unnecessary. Reviewers can trace why a result or design exists without depending on one person's memory or one assistant's conversation.

## Validation

After the first milestone, ask whether a teammate can resume another member's task using only the issue, pull request, status file, and linked records. Simplify or extend the process based on that test.
