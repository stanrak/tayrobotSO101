# Training and learning diagnostics

Read the [instruction index](instruction.md), [workflow](workflow.md), and [physical contract](physical-contract.md) before working on training.

## Entry points and authorization

- Keep continuous training, optional exact total step limits, and startup trials in the existing `so101_sim.train` entry point.
- User-started training is continuous by default. Keep `--full-training` as a compatibility alias.
- Agents must explicitly pass `--trial` for authorized startup diagnostics. Do not invoke the unrestricted default while implementing or verifying code. `--timesteps` by itself limits full training; it does not select a trial.
- Code changes and unit tests do not automatically authorize a real training run. When a startup trial is needed within the authorized task, choose its initial budget without asking again.
- Long training, repeated trial searches, checkpoint selection, and automatic evaluation require a later explicit user request authorizing that work. A larger trial rollout is not permission for more updates.
- Never perform the learning task for the policy. Do not add an IK solver, scripted reach controller, teacher actions, or automatic task completion. Initial pose setup and reachable-target sampling must not generate robot actions.
- Keep historical artifacts unchanged; do not automatically load them when opening the simulation or starting a fresh trial.

## Explicit checkpoint continuation

- `--resume CHECKPOINT` or `--resume latest` is an explicit full-training choice. Keep omitted `--resume` as a fresh start, and reject `--trial` combined with resume before output creation.
- Resolve `latest` against the existing pointer before creating a new output or replacing that pointer. Preserve the selected checkpoint and all parent-run bytes.
- Validate checkpoint content hash, schema/algorithm, exact environment/assets/package signatures, full-training state, optimizer availability, and measured continuation counters. Do not infer absent cumulative optimizer/update counts or silently migrate unsupported historical checkpoints.
- Restore with `DiagnosticPPO.load` so diagnostic/counter methods remain active. Keep the Adam optimizer state and cumulative network counters; do not replace them with a freshly initialized policy or optimizer while claiming continuation.
- Inherit saved learning hyperparameters. Reject explicitly supplied mismatches for learning rate, epochs, KL, exploration settings, and initial log standard deviation. Reward scale must match the chosen checkpoint; the default remains `0.01`, so a raw-scale checkpoint needs explicit matching scale `1`. Do not silently rescale critic targets or reset optimizer state.
- New-session seed, environment count/backend, viewer, evaluation settings, and step budget may be selected separately. Reset environment trajectories and session timing; do not claim exact continuation of the parent's ball pose, unfinished episode, RNG stream, or partial rollout.
- Treat `--timesteps` as an additional/session budget. Base exact stopping and periodic validation on observed transitions since the selected checkpoint, not the absolute cumulative count. Retain the complete-rollout update rule and discard a final incomplete rollout.
- Keep `actual_timesteps`, `optimizer_steps`, `update_count`, and `optimization_epochs` as session observations. Record `starting_*` baselines and matching `cumulative_*` counts, plus parent checkpoint path/hash/run identity. Report interrupted or failed sessions using only observed work.
- Implementing or testing resume does not authorize launching a real resumed training run. Keep tests on fake collectors/counters or checkpoint state round-trips without actual learning.

## First-update trial contract

1. Choose and explain the initial transition budget before a real trial. Use the smallest reasonable amount of data needed for the algorithm's first update, accounting for rollout length, environment count, and any warmup.
2. There is no fixed 500-transition ceiling. The value 500 is a convenience default; a task-specific budget may be larger.
3. The first network update is one successful `optimizer.step` on trainable network parameters. An episode end, collected rollout, or completed epoch is not a substitute.
4. Stop after that optimizer step. Do not execute another minibatch, optimization epoch, rollout, or update.
5. The current PPO trial uses one environment, collects one selected rollout, trains on the entire rollout as one batch, and performs one epoch. Keep its actual optimizer-call guard. KL early stopping is unavailable in a trial because it could prevent the required first update.
6. Record requested and actual transitions, actual optimizer calls, and the stopping reason. Count transitions across all environments and warmup. If initialization fails or the run stops before an update, record the observed steps and zero updates accurately.
7. Do not add validation, baselines, held-out evaluation, or checkpoint selection to a trial. Startup output diagnoses initialization, observations, actions, reward, and the first update; it is not evidence that the task has been learned.

The trial budget, an environment's target-duration budget, and a full trainer's total transition limit are separate quantities. Preserve each independently.

## Learning settings and observations

- Preserve the physical task and existing controller unless a user request authorizes their specific changes.
- The current training-only reward multiplier defaults to `0.01`. Keep `Monitor` inside `RewardScale` so monitor episode returns remain in raw environment units. Validation and held-out evaluation use raw rewards.
- Record reward scale, backend, PPO learning rate/epoch/KL settings, and exploration settings in experiment configuration. Do not silently change multiple learning settings while presenting a result as a single-variable comparison.
- Existing CLI options expose separate learning-rate, epoch, KL, and gSDE experiments. The default learning rate and epoch count remain `0.0003` and `10`; validation defaults to every `100000` training transitions.
- `subproc` is an opt-in backend for headless full training. Keep its rejection with `--trial` or `--viewer` before output creation or environment initialization.
- Observation extensions, running normalization, reward shaping, and other algorithms are separate experiments. Do not claim they are enabled by the current fixed reward scale or migrate old checkpoints into them automatically.

## Counters and diagnostics

- Count successful optimizer calls directly, including partial or interrupted optimization phases. Count a completed PPO rollout update only when its optimization phase completes and has at least one successful optimizer call.
- Do not infer completed rollout updates from `_n_updates // n_epochs`: KL stopping can end an epoch early. `optimization_epochs` reports SB3 epochs entered, including a KL-stopped partial epoch; label that unit accurately.
- Diagnostics may inspect the existing rollout buffer and at most the configured bounded sample of observations. They must not collect additional transitions, sample policy actions, advance training RNG state, or become a teacher/controller.
- Keep actor/critic inference hooks and temporary optimizer wrappers reversible. Restore policy mode, hooks, and original optimizer calls on both success and failure.
- Label gradient norms as measured after SB3 clipping when measured at `optimizer.step`. Distinguish action clipping, conditional exploration standard deviation, critic value/target spread, and saturation from task success.
- Record rollout, optimization, diagnostic, validation, and held-out durations with their actual scope. Rollout collection includes periodic validation/viewer callbacks; final validation occurs outside collection. Do not add overlapping durations or subtract all validation time to claim pure simulation time.
- Validation step and episode counters cover completed reports; retain elapsed time for interrupted partial validation and clearly label uncounted partial reports.
- Use multiple training seeds and matching evaluation seeds/budgets for conclusions about a learning change. Separate measured results from hypotheses and proposed performance targets.
