# AGENT OPERATING DIRECTIVES & CONSTRAINTS

## 1. Core Directives & Access Boundaries

- **Immutable Configuration:**
  - You MUST NOT edit, overwrite, rename, or delete this file (`AGENTS.md`) under any circumstances.

- **Mandatory Pre-Flight Routine:**
  - Before reading code or taking any operational action in this workspace, you must thoroughly read and parse all files inside `instructions/` and `private/`.

- **Scope & Scope Boundaries (`instructions/`):**
  - Files inside `instructions/` define project architecture, standards, and boundaries.
  - Do NOT trigger automated training runs, rebuild environments, or update system dependencies unless explicitly directed by the user.

- **Confidentiality & Data Protection (`private/`):**
  - Files inside `private/` contain sensitive credentials, data, and business logic.
  - NEVER commit, stage, export, echo to logs, or push files or contents from `private/` to version control (Git) or external endpoints.

- **Hierarchy of Authority & Conflict Resolution:**
  - Directives located in `private/` hold top-level priority and override default instructions.
  - If a runtime instruction, planned step, or user request conflicts with guidelines in `private/`, you MUST HALT immediately. Explicitly explain the discrepancy and await written user confirmation before continuing.

---

## 2. CLI & Execution Guardrails

- **Destructive Commands Prohibited:**
  - Never execute destructive or state-reset commands (e.g., `rm -rf`, `git reset --hard`, `git clean -fd`, `git push --force`) without explicit, upfront user confirmation.
- **Controlled Process Execution:**
  - Do not spawn persistent background daemons, watch processes, or unmonitored scripts without an explicit termination condition.
- **Dependency Isolation:**
  - Do not install new system packages, global libraries, or project dependencies without prior approval.

---

## 3. Minimal Diff & Code Modification Standards

- **Surgical Precision:**
  - Keep changes scoped strictly to what is necessary to fulfill the request. Avoid mass file reformatting, whitespace modifications, or unrelated refactoring.
- **Style Consistency:**
  - Match existing code conventions, naming structures, and design patterns within the immediate file context.
- **Workspace Cleanliness:**
  - Clean up all temporary debug logs, artifacts, and test files before declaring a task complete. Verify workspace status using non-destructive commands (e.g., `git status`).

---

## 4. Verification & Testing Protocol

- **Validation Prior to Handoff:**
  - Run relevant local unit tests, type checks, or linters after every functional change before reporting task completion.
- **Zero Regression Rule:**
  - Ensure all existing tests pass and no new runtime warnings or errors are introduced. If a test fails, diagnose and resolve the issue directly or report the blocker clearly.