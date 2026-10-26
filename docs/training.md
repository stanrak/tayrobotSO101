# Training and learning diagnostics

[Documentation](README.md) · [Project overview](../README.md) · [Environment](environment.md) · [Experiment records](experiments.md)

The trainer uses PPO with a small multilayer perceptron on the CPU. Manual keyboard control is a separate application: starting the simulator does not start training or load a policy. All commands below run from the repository root in PowerShell, using the prepared project environment described in [Getting started](getting-started.md).

## Choose a run mode

| Mode | Command | Result |
| --- | --- | --- |
| Continuous training | `.\.venv\Scripts\python -m so101_sim.train` | A fresh policy trains until Ctrl+C or a viewer interruption. |
| Bounded training | `.\.venv\Scripts\python -m so101_sim.train --timesteps 307200 --seed 42` | Training stops at the requested total transition count, then performs final validation and held-out evaluation. |
| Startup diagnostic | `.\.venv\Scripts\python -m so101_sim.train --trial --timesteps 500 --seed 42` | A fresh policy collects one batch and completes exactly one optimizer step, without performance evaluation. |

`--full-training` remains an alias for the default training mode. Without `--trial` or a step limit, training is continuous. Every invocation starts a new policy in a separate output directory; it does not resume a historical checkpoint.

A transition is one control step in one environment. Full training defaults to eight environments and 256 steps per environment per rollout, giving 2,048 transitions per rollout. The bounded example above contains 150 complete rollouts. If the requested limit is not divisible by eight, the default changes to one environment; an explicit `--envs` value must divide the limit. PPO updates only complete rollouts, so a final incomplete rollout is discarded. Validation and held-out episodes add simulation work outside the training transition limit.

The trial defaults to 500 transitions when `--timesteps` is omitted. It supports a task-specific budget of at least two transitions, including budgets above 500, but always uses one environment, the whole rollout as one batch, and one optimization epoch. An optimizer-call guard prevents a second update. An episode or rollout ending is not itself a network update. The trial checks initialization and the first update; it does not establish that the robot has learned the task.

## Observe the actual training run

```powershell
.\.venv\Scripts\python -m so101_sim.train --viewer --seed 42
```

`--viewer` and its alias `--render` display environment zero. The terminal shows total training transitions, the observed episode count, distance to the ball, optimizer steps, and the current target's age in simulated seconds. Messages identify rollout collection and network optimization. The displayed robot may remain still during optimization or validation.

The viewer runs at real time by default. Add `--viewer-speed 4` for four times simulation speed. Pacing changes wall-clock duration, while physics time, actions, and transition budgets remain the same. The viewer needs to be enabled when the run starts; it cannot attach to an existing headless run. Camera controls remain available. It displays independent model/data copies, so viewer physics controls do not edit training physics.

Ctrl+C or closing the viewer interrupts full training, saves `final_model.zip`, records the observed counts and interrupted status, and skips final held-out evaluation. Reaching a requested step limit instead completes the normal evaluation path. See [Experiment records](experiments.md) for checkpoint selection and report locations.

## Training settings

| Option | Default | Meaning |
| --- | --- | --- |
| `--timesteps` | Unlimited | Total training transitions; in trial mode, the initial rollout budget. |
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
| `--heldout-episodes` | `100` | Episodes per policy in the final held-out report. |
| `--viewer-speed` | `1.0` | Positive display speed multiplier when using the viewer. |
| `--output` | New managed directory | Optional custom empty training directory. |

Additional PPO settings are recorded in each run's `training.json`: batch size 256, discount factor 0.99, GAE lambda 0.95, clip range 0.2, entropy coefficient 0, value coefficient 0.5, gradient norm limit 0.5, and two 64-unit hidden layers. The trainer uses one PyTorch thread and deterministic algorithms. Observations have manual scaling described in [Environment](environment.md); running observation/reward normalization is disabled.

## Reward scale and controlled comparisons

`RewardScale` multiplies the environment reward by 0.01 before PPO receives it. `Monitor` sits inside this wrapper, so episode logs, validation, and held-out reports retain the original reward units. The physical task, success threshold, action controller, and environment reward formula are unchanged by the multiplier.

The scale was introduced as a controlled response to a nearly constant critic observed in an earlier saved run. Improved task performance still needs to be measured. A short pilot can reveal better distance reduction or critic learning before any successful episode; it is not a learned-performance benchmark.

Change one setting at a time. For example, compare a new run with `--learning-rate 0.0001`, then a separate run with `--target-kl 0.01`. Match task/environment signatures, budgets, seeds, and evaluation settings. Use `--reward-scale 1 --eval-frequency 25000` to reproduce the previous reward scale and validation cadence. Fresh runs do not automatically load earlier policies.

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
| `rollout/mean_distance_m`, `rollout/minimum_distance_m`, `rollout/within_*mm_fraction` | Approach accuracy across the collected rollout. |
| `rollout/maximum_hold_steps` | Progress toward sustained success. |

PPO training samples actions, while validation uses deterministic predictions. Their success rates can differ because of both exploration noise and different state/seed distributions. Evaluate approach distance, velocity/holding behavior, timeout/collision causes, and repeated checkpoint results together.

`update_count` records completed PPO optimization phases containing at least one successful optimizer step. `optimizer_steps` records actual successful calls. `optimization_epochs` counts SB3 epochs entered and may include a partial epoch stopped by KL. These counters remain distinct when early stopping or an interruption occurs.

`training.json` records cumulative `phase_seconds`:

| Key | Scope |
| --- | --- |
| `rollout_including_callbacks` | Rollout collection, including periodic validation and viewer callbacks. |
| `optimization` | PPO optimization work. |
| `diagnostics` | Additional sampled diagnostics after optimization. |
| `validation` | Validation episode execution. |
| `heldout` | Final held-out evaluation. |

These durations overlap: periodic validation is inside rollout timing, while final validation occurs outside rollout collection. Adding or subtracting the entire validation duration does not produce a disjoint breakdown. Validation step/episode counts cover completed reports; an interrupted partial validation retains elapsed time without assigning missing counts.

SAC, `VecNormalize`, observation extensions, and reward shaping are possible future experiments rather than enabled options. The current implementation and diagnostic results do not establish that any of them improves this task.

