## Current development and training scope

- Work on code and the initial simulation environment. Manual laptop keyboard control is the default application mode.
- The existing `so101_sim.train` entry point now defaults to continuous user-started training. Agents must explicitly pass `--trial` for startup diagnostics; do not invoke the unrestricted default while implementing or verifying code. `--timesteps` alone limits full training and does not select a startup trial. Keep `--full-training` as a compatibility alias, and keep continuous training, optional exact step limits, and trials in the existing trainer file.
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

## Output locations

- Use the managed default `runs/current` for startup trials, full training, and evaluation reports. Ordinary commands and examples must not require a new `--output` path for each run, viewer setting, or task change.
- The output manager archives the complete previous run under `runs/history` before a new training run. Preserve checkpoint bytes, metadata, and logs; do not automatically delete history, load archived policies, or migrate incompatible checkpoints.
- Default evaluation finds the current checkpoint automatically. Existing reports are retained under the current run's `evaluation_history` folder before replacement.
- Respect output writer locks. Do not rotate an active run or bypass its lock to start another writer. This output organization does not authorize training or evaluation beyond the scope above.

## Fixed physical workspace contract

- The authorized surface-ball upgrade uses a fixed 1.2 x 0.7 x 0.75 m desk, with a 0.03 m top and four collision-enabled legs. The desk is centered at world x/y = [0.4, 0]; the robot base is mounted at [0, 0, 0.75] m. Compose this workspace in `so101_sim/env.py`; keep the pinned upstream robot assets and their manifest hashes unchanged.
- The ball is a physical free body with radius 0.015 m and mass 0.02 kg. Explicit gravity is [0, 0, -9.81] m/s^2. Ball/support contacts use condim 6, friction [0.6, 0.005, 0.0005], and solref [0.002, 1]. Physics uses a 0.001 s timestep and up to 50 solver iterations; frame_skip 50 retains 20 Hz policy control. These are nominal, uncalibrated material parameters. Do not claim an exact real-material match.
- Reset places a stationary ball directly on the table or adjacent floor, with equal surface probabilities by default. `options={"ball_surface": "table"}` or `"floor"` selects a surface. Never restore random floating targets or pin/teleport the ball during simulation steps. Goal observations and distances follow its current simulated position.
- Table samples use the local curved patch specified by `task_config`; floor samples stay outside the desk footprint and legs. The table-mounted arm cannot reach the floor. Keep this limitation explicit; do not silently move the robot, add a solver, migrate controllers, or expand the learning task to compensate.
- Ball support and robot/ball contacts are permitted. Robot self-contact and contact with the table/floor still fail on penetration deeper than 4 mm. Keep the six robot joints/actuators separate from the ball's seven qpos and six qvel coordinates.
- Geometry, physics, surface placement, contact filtering, and live goal tracking are locked by physical contract tests. Task `SO101SurfaceBallReach-v2` and checkpoint environment signatures record the changed semantics. Historical checkpoints remain incompatible and must not be migrated automatically. Further object/dynamics changes need a user request that authorizes those specific changes.
