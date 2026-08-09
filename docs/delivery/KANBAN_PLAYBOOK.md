# Kanban Delivery Playbook

## Board

GitHub Project: `PathLens-GNN Delivery`.

Statuses: Backlog, Ready, In Progress, Awaiting Compute, In Review, Blocked, Done. Custom fields: Priority P0/P1/P2; Track; Estimate 1/2/3/5; Risk Low/Medium/High.

## WIP

- One human work item in progress.
- External training may wait in `Awaiting Compute` without blocking fixture-driven API/UI work.
- At most three dependency-cleared Ready items.
- Blocked items require a blocker comment and linked dependency.
- Done requires accepted checks and a merged PR.

## Milestones

M1 Validated Research Foundation; M2 Model and Artifact Freeze; M3 Inference Product Beta; M4 Portfolio Release. Dates remain unset until a hard delivery date exists.

## Issue map

Four epics contain the dependency-ordered child issues defined in `docs/decisions/PROJECT_CHARTER.md`: research foundation, path-aware model, inference product, and release. Each implementation Issue must include purpose, dependencies, acceptance checks, test evidence, priority, track, estimate, and risk.

## Branch and PR policy

Use `issue-<number>-<slug>` from `main`, stage explicit paths, open a draft PR, link the Issue, run CI, then squash merge. `planning` changes enter `main` through one planning PR.
