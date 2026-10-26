# Agent instructions: entry point

## Project

This repository develops a MuJoCo simulation of the SO-101 robot and a PPO reach-and-hold learning task. The default application opens manual laptop keyboard control. Training, evaluation, and experiment inspection are separate entry points.

## Authority and preflight

- `private/` is the user-owned, highest-priority local project guide. Read it locally; never copy its contents into documentation, logs, source snapshots, version control, or external requests. Do not modify it.
- `AGENTS.md` defines the active overall agent requirements and is immutable. Do not edit, overwrite, rename, or delete it.
- Before code inspection or operational work, read and parse every file in `instructions/` and `private/`, as required by `AGENTS.md`. Reading this index alone does not satisfy that preflight.
- If a planned action conflicts with private guidance, stop, explain the discrepancy without disclosing private contents, and await written user confirmation.

## Required instruction set

| File | Agent responsibility |
| --- | --- |
| [workflow.md](workflow.md) | Scope, document ownership, editing, execution, confidentiality, and verification |
| [training.md](training.md) | Training authorization, first-update trials, counters, reward scale, and diagnostics |
| [experiments.md](experiments.md) | Output isolation, provenance, checkpoint compatibility, comparisons, and locks |
| [physical-contract.md](physical-contract.md) | Fixed scene, physics, goal sampling, target cadence, and contact semantics |

## Documentation ownership

- Root `README.md` is a concise project overview for human readers.
- `docs/` contains human explanations, operation guides, technical references, and project stages. Readers must be able to understand those documents without consulting agent instructions or private files.
- `AGENTS.md` and `instructions/` are written for agents. Put agent procedures and constraints here rather than in reader documentation.
- Write agent-maintained documents in English. Keep user-owned files unchanged.
- Keep this entry path stable. When instruction files are split or renamed, update local links and the explicit source snapshot allowlist without adding private or otherwise unapproved user-owned guidance.
