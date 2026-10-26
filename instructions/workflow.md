# Agent workflow

Read this file together with the [instruction index](instruction.md) and all other required preflight files.

## Scope and execution

- Work within the user's authorized task. Current development covers code and the initial simulation environment; manual keyboard control is the default application mode.
- Do not start automated training, rebuild environments, or update system dependencies without explicit user direction. Follow [training.md](training.md) for startup trials and further training authorization.
- Do not install system packages, global libraries, or project dependencies without prior approval.
- Never run destructive or state-reset commands without explicit, upfront user confirmation. This includes recursive deletion, hard resets, repository cleaning, and force pushes.
- Do not launch persistent daemons, watch processes, or unmonitored scripts without an explicit termination condition.
- Any subprocess environment backend must own its workers and clean them up on completion, interruption, and failed or partial initialization. Keep startup waits and shutdown joins bounded; terminate only workers created by that backend.

## Protected files and information

- Do not edit, overwrite, rename, or delete `AGENTS.md` or every file in `private/`.
- Treat every file in `private/` as confidential. Never stage, commit, export, echo to logs, or push those files or their contents to Git or external endpoints.
- Do not copy private content into public documentation, code, experiment snapshots, issue text, or test fixtures.
- If a task conflicts with private guidance, halt and await written user confirmation after explaining the discrepancy without exposing sensitive information.
- Preserve existing experiment artifacts. Do not use them as an automatic initial policy for the simulator or a fresh startup trial.

## Document maintenance

- Keep the human overview in `README.md` concise; put technical explanations and project-stage descriptions in `docs/`.
- Put agent-specific procedures, architecture constraints, and maintenance requirements in `instructions/`. These agent-maintained instruction files may be versioned in Git.
- Keep human guides descriptive and self-contained. Do not move agent authorization gates or private instructions into reader guides.
- Maintain links between the reader overview and the appropriate `docs/` pages, and between this instruction index and the agent procedures.
- Retain the active requirements when reorganizing documentation. A draft, code example, or documentation change does not authorize execution.

## Editing and verification

- Keep changes strictly necessary for the request. Avoid unrelated refactoring, mass formatting, and whitespace churn.
- Match local naming, code style, and design patterns.
- After each functional change, run relevant local tests, type checks, or linters. All existing tests must pass; diagnose regressions or report a concrete blocker.
- Use fake counters and environments for trainer budget/callback tests. Unit tests must not quietly launch real training or automatic policy evaluation.
- For documentation-only changes, verify local links, command/options against the existing CLI or code, and preservation of active constraints. Do not install a documentation toolchain solely for that check.
- Remove temporary debug files, logs, and test artifacts before handoff; retain requested deliverables and historical experiments.
- Inspect the workspace with non-destructive commands such as `git status`. Do not stage or discard unrelated user changes.
