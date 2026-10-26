# SO-101 agent operating guide

## Project

SO-101 Physical AI simulation is a Python/MuJoCo project for desktop keyboard control and PPO reach-and-hold experiments with a physical ball on a desk. Manual simulation is the default application mode. Training, evaluation, and saved-experiment inspection are separate commands.

## Read before work

1. Read this entry point and every file in `private/` and `instructions/` before code inspection or operational work.
2. Treat the user-owned `private/` as the highest-priority local project guide. Keep it unchanged and confidential.
3. Use [instructions/instruction.md](instructions/instruction.md) as the index to the workflow, training, experiment, and physical contracts. Read all linked instruction files; a task-specific subset does not replace the required preflight.
4. Keep `README.md` concise and human-facing. Put reader explanations in `docs/` and agent procedures in `instructions/`.

## Overall requirements

- Stay within the authorized task and preserve unrelated user changes.
- Do not modify protected user-owned files or disclose private content through logs, Git, snapshots, or external requests.
- If private guidance conflicts with an intended action, stop, explain the discrepancy safely, and await written user confirmation.
- Do not install dependencies, rebuild environments, or start training/evaluation outside the explicitly authorized scope. Authorized startup trials stop at the first successful optimizer step.
- Preserve pinned robot assets, the physical task contract, checkpoint compatibility, output locks, and historical run bytes.
- Do not add a scripted solver or teacher that completes the learning task for the policy.
- Keep changes narrow, verify functional changes with relevant tests, clean temporary artifacts, and inspect workspace status before handoff.
