# Training and learning diagnostics

[Documentation](README.md) · [Project overview](../README.md) · [Environment](environment.md) · [Experiment records](experiments.md)

The trainer uses PPO with a small multilayer perceptron on the CPU. Manual keyboard control is a separate application: starting the simulator does not start training or load a policy. All commands below run from the repository root in PowerShell, using the prepared project environment described in [Getting started](getting-started.md).

## Choose a run mode

| Mode | Command | Result |
| --- | --- | --- |
| Continuous training | `.\.venv\Scripts\python -m so101_sim.train` | A fresh policy trains until Ctrl+C or a viewer interruption. |
| Bounded training | `.\.venv\Scripts\python -m so101_sim.train --timesteps 307200 --seed 42` | A fresh policy trains for the requested session transition count, then performs final validation and held-out evaluation. |
| Continued training | `.\.venv\Scripts\python -m so101_sim.train --resume latest --timesteps 200000 --seed 42` | A checked checkpoint restores policy and optimizer state for an additional 200,000 transitions in a new run. |
| Startup diagnostic | `.\.venv\Scripts\python -m so101_sim.train --trial --timesteps 500 --seed 42` | A fresh policy collects one batch and completes exactly one optimizer step, without performance evaluation. |

`--full-training` remains an alias for the default training mode. Without `--trial` or a step limit, training is continuous. Each invocation creates a separate output directory. Omitting `--resume` starts a fresh policy; resuming is an explicit choice.

A transition is one control step in one environment. Full training defaults to eight environments and 256 steps per environment per rollout, giving 2,048 transitions per rollout. The bounded example above contains 150 complete rollouts. If the requested limit is not divisible by eight, the default changes to one environment; an explicit `--envs` value must divide the limit. PPO updates only complete rollouts, so a final incomplete rollout is discarded. Validation and held-out episodes add simulation work outside the training transition limit.

The trial defaults to 500 transitions when `--timesteps` is omitted. It supports a task-specific budget of at least two transitions, including budgets above 500, but always uses one environment, the whole rollout as one batch, and one optimization epoch. An optimizer-call guard prevents a second update. An episode or rollout ending is not itself a network update. The trial checks initialization and the first update; it does not establish that the robot has learned the task.

## Continue a saved policy

To continue from the latest available full-training checkpoint:

```powershell
.\.venv\Scripts\python.exe -m so101_sim.train --resume latest --timesteps 200000 --seed 42
```

`--timesteps` is the number of **additional transitions in this session**. Cumulative transitions equal the selected checkpoint's starting count plus the observed session transitions. The exact cap is measured from the checkpoint's starting count, including when the final rollout is incomplete. Validation cadence is also measured in the new session's transitions.

`latest` resolves the existing latest-run pointer before the new output directory becomes the latest run. It uses the checkpoint selection described in [Experiment records](experiments.md), which prefers a best-validation checkpoint. To continue from a particular final or best policy, pass its path to `--resume` instead. Only supported full-training checkpoints can be resumed; `--trial --resume ...` is rejected before output creation.

Continuation restores policy weights, the Adam optimizer state, and cumulative training counters through `DiagnosticPPO`. The new session initializes environments and random generators using the selected seed. It does not restore the previous ball pose, unfinished episode, or partially collected rollout, so it is not an exact replay of an uninterrupted process.

The checkpoint's learning settings are inherited. An explicitly supplied learning rate, epoch count, KL threshold, exploration setting, or initial log standard deviation must match the saved setting. Environment count, backend, viewer, seed, evaluation settings, and additional step budget can be chosen for the new session. A different environment count or seed can change subsequent learning trajectories.

The requested reward scale must also match the checkpoint. The default remains `0.01`; a checkpoint trained with raw reward scale `1` requires an explicit matching `--reward-scale 1`. The trainer rejects a mismatch rather than silently changing the critic's target units or reinitializing its optimizer. Incompatible task signatures, assets, packages, checkpoint hashes, missing optimizer state, or missing continuation counters also cause a clear failure. Historical checkpoints are not migrated automatically.

The parent checkpoint remains unchanged. The new run records its parent path, SHA256, run identifier, and starting counters. [Experiment records](experiments.md) explains the session and cumulative fields. These commands describe how to launch continuation; no continued-learning performance is established by the feature itself.

## Observe the actual training run

```powershell
.\.venv\Scripts\python -m so101_sim.train --viewer --seed 42
```

`--viewer` and its alias `--render` display environment zero. The terminal shows total training transitions, the observed episode count, distance to the ball, optimizer steps, and the current target's age in simulated seconds. Messages identify rollout collection and network optimization. The displayed robot may remain still during optimization or validation.

The viewer runs at real time by default. Add `--viewer-speed 4` for four times simulation speed. Pacing changes wall-clock duration, while physics time, actions, and transition budgets remain the same. The viewer needs to be enabled when the run starts; it cannot attach to an existing headless run. Camera controls remain available. It displays independent model/data copies, so viewer physics controls do not edit training physics.

Ctrl+C or closing the viewer interrupts full training, saves `final_model.zip`, records the observed counts and interrupted status, and skips final held-out evaluation. Reaching a requested step limit instead completes the normal evaluation path. See [Experiment records](experiments.md) for checkpoint selection and report locations.

After final validation, the trainer saves `final_model.zip` and evaluates the selected checkpoint against the no-motion and random-action baselines sequentially. This phase can take substantial time: the default 100 episodes per policy means 300 episodes and up to 360,000 additional control steps. The `Monitor` and `DummyVecEnv` messages at this point accompany checkpoint loading. Loading and policy-start messages, in-flight progress about every ten seconds, and policy-completion messages identify the current work, completed episodes, control steps, and overall elapsed time. `training.json` records status `evaluating` and measured training counts before this phase; `evaluation.json` appears when the full report is complete.

## Training settings

| Option | Fresh-run default | Meaning |
| --- | --- | --- |
| `--timesteps` | Unlimited | Session training transitions; additional transitions when resuming, or the initial rollout budget in trial mode. |
| `--resume` | Unset | Explicit full-training checkpoint path or `latest`; restores policy and optimizer into a new run. |
| `--seed` | `42` | Training seed. |
| `--envs` | `8`, or `1` for trial | Number of environment copies. |
| `--reward-scale` | `0.01` | Positive finite reward multiplier applied only to training. |
| `--learning-rate` | `0.0003` | Positive finite PPO learning rate. |
| `--n-epochs` | `10` | Optimization epochs per full-training rollout; trials use one. |
| `--log-std-init` | `-2.0` | Finite initial policy log standard deviation. |
| `--target-kl` | Disabled | Optional positive KL threshold for early stopping within a PPO update; unavailable in trial mode. |
| `--use-sde` | Off | Use state-dependent exploration for a separate experiment. |
| `--sde-sample-freq` | `-1` | With gSDE, resample noise at rollout start only, or use a positive control-step interval. |
| `--vec-env` | `dummy` | Sequential environments, or opt-in `subproc` workers. |
| `--eval-frequency` | `100000` | Training transitions between periodic validation rounds. |
| `--eval-episodes` | `20` | Fixed-seed validation episodes per round. |
| `--heldout-episodes` | `100` | Episodes per policy in the final held-out report; smaller counts reduce runtime and statistical precision. |
| `--viewer-speed` | `1.0` | Positive display speed multiplier when using the viewer. |
| `--output` | New managed directory | Optional custom empty training directory. |

Additional PPO settings are recorded in each run's `training.json`: batch size 256, discount factor 0.99, GAE lambda 0.95, clip range 0.2, entropy coefficient 0, value coefficient 0.5, gradient norm limit 0.5, and two 64-unit hidden layers. The trainer uses one PyTorch thread and deterministic algorithms. Observations have manual scaling described in [Environment](environment.md); running observation/reward normalization is disabled.

## Reward scale and controlled comparisons

`RewardScale` multiplies the environment reward by 0.01 before PPO receives it. `Monitor` sits inside this wrapper, so episode logs, validation, and held-out reports retain the original reward units. The physical task, success threshold, action controller, and environment reward formula are unchanged by the multiplier.

Reward scaling is a controlled way to examine critic target magnitude while preserving raw task metrics. Approach behavior and critic fitting can improve before precise reaching and stable holding are reliable. Broad implementation progress is described in [Project stages](project-stages.md); run-specific reviews and measurements remain confidential local records under `reports/`.

Change one setting at a time. For example, compare a new run with `--learning-rate 0.0001`, then a separate run with `--target-kl 0.01`. Match task/environment signatures, budgets, seeds, and evaluation settings. Use `--reward-scale 1` to compare raw-reward training, with a matched validation cadence in both configurations. Fresh runs do not automatically load earlier policies.

## CPU parallelism

`DummyVecEnv` steps the environment copies sequentially in one process. To measure multiprocessing on the laptop, add either of these configurations to a full-training command:

```powershell
--vec-env subproc --envs 4
--vec-env subproc --envs 8
```

These are alternative flag sets, not standalone commands. Process communication has a cost, so throughput should be measured before choosing a backend. `subproc` supports headless full training; combining it with `--viewer` or `--trial` is rejected before output creation. Workers use the Windows-compatible spawn method, bounded initialization waits, and bounded shutdown cleanup. This backend does not change the physics timestep or controller.

## Read the diagnostics

`progress.csv` contains standard PPO metrics and additional diagnostics calculated from the existing rollout data. At most 256 existing observations are used for sampled critic checks; this does not add environment transitions or sample additional actions.

| Metric family | What it helps distinguish |
| --- | --- |
| `train/explained_variance` | Whether rollout value predictions explain variation in return targets. |
| `diagnostics/rollout_value_*`, `diagnostics/target_*` | Value/target scale and variation before the update. |
| `diagnostics/critic_value_*`, `diagnostics/critic_target_rmse` | Sampled value predictions and target error after the update. |
| `diagnostics/critic_tanh_*_saturation` | Fraction of sampled critic tanh outputs with absolute value above 0.999. |
| `diagnostics/first_gradient_norm_after_clip` | Gradient norm for the first optimizer call, measured after SB3 clips it. |
| `diagnostics/action_*` | Action clipping and conditional exploration standard deviation per action component. |
| `diagnostics/action_*_mean_outside_fraction` | Fraction of the bounded observation sample whose unsquashed actor mean is outside the action limits; helps distinguish actor saturation from exploration-driven clipping. |
| `rollout/mean_distance_m`, `rollout/minimum_distance_m`, `rollout/within_*mm_fraction` | Approach accuracy across the collected rollout. |
| `rollout/maximum_hold_steps` | Progress toward sustained success. |

PPO training samples actions, while validation uses deterministic predictions. Their success rates can differ because of both exploration noise and different state/seed distributions. Evaluate approach distance, velocity/holding behavior, timeout/collision causes, and repeated checkpoint results together.

`update_count` records completed PPO optimization phases containing at least one successful optimizer step. `optimizer_steps` records actual successful calls. `optimization_epochs` counts SB3 epochs entered and may include a partial epoch stopped by KL. These top-level counts describe the current session. Matching `cumulative_*` fields include the parent checkpoint's counts, and `starting_*` fields record that baseline. These counters remain distinct when early stopping or an interruption occurs.

In `progress.csv`, `time/total_timesteps`, `train/n_updates`, `train/rollout_update_count`, and `train/actual_optimizer_steps` retain the model's cumulative counters. Additional `time/session_timesteps` and `time/cumulative_timesteps` columns distinguish the two transition counts. In `evaluations.csv` and `best_validation.json`, `timesteps` describes the current session and `cumulative_timesteps` includes the selected parent.

`training.json` records `phase_seconds` accumulated within the current session; parent timing totals are not carried into the new session:

| Key | Scope |
| --- | --- |
| `rollout_including_callbacks` | Rollout collection, including periodic validation and viewer callbacks. |
| `optimization` | PPO optimization work. |
| `diagnostics` | Additional sampled diagnostics after optimization. |
| `validation` | Validation episode execution. |
| `heldout` | Final held-out evaluation. |

These durations overlap: periodic validation is inside rollout timing, while final validation occurs outside rollout collection. Adding or subtracting the entire validation duration does not produce a disjoint breakdown. Validation step/episode counts cover completed reports; an interrupted partial validation retains elapsed time without assigning missing counts.

SAC, `VecNormalize`, observation extensions, and reward shaping are possible future experiments rather than enabled options. The current implementation and diagnostic results do not establish that any of them improves this task.

