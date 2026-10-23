## Current development and training scope

- Work on code and the initial simulation environment. Manual laptop keyboard control is the default application mode.
- A startup trial may run only until the first actual network update. There is no fixed 500-transition ceiling. The agent may choose the initial number of steps according to the task, algorithm, rollout length, environment count, and any necessary warmup. Use the smallest reasonable amount of initial data needed to reach that update; 500 steps is only a convenience default.
- Choose and explain the trial budget before a real trial, then record the requested and actual environment transitions, optimizer steps, and stopping reason. Count transitions across all environments, including warmup. An episode ending or a rollout completing is not itself a network update.
- For this project, the first network update means one successful optimizer step on trainable network parameters. Stop after that step. Do not continue with another minibatch, optimization epoch, rollout, or update. For the current PPO trial, collect one selected rollout, use the complete rollout as one batch, and run one optimization epoch. Count actual optimizer calls rather than infer them from completed rollouts or epochs.
- Code changes and unit tests do not automatically authorize a real training run. When a startup trial is needed within the authorized task, the agent may choose its step budget without asking again. Do not run long training, repeated trial searches, checkpoint selection, or automatic evaluation without a later explicit user request authorizing that work. A larger initial rollout is not permission to continue training after the first update.
- Startup output is a diagnostic of initialization, observations, actions, reward, and the first update. It must not claim that the task has been learned. If the run fails before an update, record the observed steps and zero updates accurately.
- Never perform the learning task on behalf of the policy. Do not add an IK solver, scripted task controller, teacher actions, or automatic task completion. Initial pose setup and reachable-goal sampling must not generate robot actions for the policy.
- Keep existing experiment artifacts as historical records. Do not load them automatically when starting the simulation or a fresh startup trial.

## Instruction file ownership and Git

- Do not modify `AGENTS.md`.
- Write agent-maintained guidance inside the `instructions` folder. These instruction files may be versioned in Git.
