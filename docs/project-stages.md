# Project stages

[Documentation guide](README.md) · [Project overview](../README.md)

The project develops a simulated robot task in stages so that control, physics, and learning quality can be examined separately. The status below distinguishes implemented behavior from improvements that still need experimental evidence.

| Stage | Status | Result |
| --- | --- | --- |
| Manual SO-101 simulation | Implemented | A laptop keyboard controls six robot joints in a MuJoCo viewer. |
| Physical workspace and reachable ball | Implemented | A table-mounted robot interacts with a free ball under gravity and contact dynamics. |
| Reach-and-hold learning task | Implemented | A three-joint policy approaches the moving ball and must maintain the required accuracy and stability. |
| Reproducible training and evaluation | Implemented | Separate run records preserve configuration, source snapshots, checkpoints, metrics, and measured evaluation results. |
| Critic diagnostics and controlled tuning | Implemented; approach improvement observed in one fresh run | Training-only reward scaling, phase timings, KL/gSDE options, and an optional subprocess backend support controlled experiments. |
| Explicit checkpoint continuation | Implemented; continued-learning performance unmeasured | A new session can restore a compatible policy and optimizer while preserving parent records and separate session/cumulative counts. |
| More reliable and efficient learning | Experimental | Compare configurations on matching tasks, budgets, and seeds before selecting a new baseline. |

## Manual control establishes the workspace

The application starts with manual keyboard control. This makes robot motion, camera behavior, contact, and target placement visible before training. The manual controller supports all six joints; it is separate from policy training. See [getting started](getting-started.md).

## The physical ball defines the current task

The current task is `SO101ReachableSurfaceBall-v3`. The ball starts on the desk within verified reach and remains a free physical body during stepping. Its current position is the target. Each target receives 1,200 control steps per environment before automatic resampling, and success requires a sustained accurate, slow reach.

Earlier task versions and checkpoints have different semantics. Their success rates are historical results, rather than direct baselines for the current environment. See [environment](environment.md) and [checkpoint compatibility](experiments.md).

## The saved baseline exposes a learning problem

The local run `20261026T031519273198Z_full_training_seed42_8d9cfc9f` provides a reference for the training review:

| Measurement | Saved result |
| --- | ---: |
| Training transitions | 2,577,392 |
| Elapsed time through interruption | About 7 hours 16 minutes |
| Best validation | 11/20 successes (55%) at 2,000,000 transitions |
| Last saved validation | 4/20 successes (20%) at 2,575,000 transitions |
| Held-out result for this run | Not measured |

The policy reduced its distance to the ball, but validation performance varied and fell after the best checkpoint. The saved critic explained variance stayed near zero. Validation also consumed 2,274,864 additional control steps, which were outside the training transition budget. Those counts describe interaction workload, not the share of wall-clock time spent in each phase.

The best validation rate came from 20 fixed seeds; its Wilson 95% interval is approximately 34.2–74.2%. Repeated checkpoint selection on that set does not establish held-out performance. The last validation measurement also does not evaluate the final saved checkpoint automatically. See [experiments](experiments.md) for interpreting the files.

These measurements belong to the saved configuration. Generated artifacts are excluded from version control and may be absent from another checkout.

## Current instrumentation supports controlled experiments

The current trainer applies a reward multiplier of `0.01` only to learning and validates every 100,000 training transitions by default. Episode logs and evaluation retain the original reward units. It also measures critic behavior, action clipping and exploration statistics, actual optimizer calls, and time spent in training phases.

The multiplier is a testable response to the observed critic behavior. Learning rate and epoch defaults remain unchanged so that the first comparison can isolate the reward-scale change.

## A fresh 300,000-transition run improves approach

The later local run `20261026T151230370817Z_full_training_seed42_cbf93550` completed 300,000 transitions with reward scale `0.01`. It started from a new policy (`checkpoint_loaded=false`); it did not inherit the earlier run's weights or optimizer.

| Training transitions | Validation success | Mean final validation distance |
| ---: | ---: | ---: |
| 100,000 | 0/20 | 110.70 mm |
| 200,000 | 0/20 | 43.88 mm |
| 300,000 | 0/20 | 40.31 mm |

At 200,000 transitions, the previous configuration's distance was 148.66 mm using the same environment signature, training seed, and validation seed set. The latest run's last logged critic explained variance was about 0.995, compared with approximately zero in the previous run. These measurements support improved approach and critic fitting in one training seed.

The final held-out report still recorded 0/100 successes and 38.26 mm mean final distance. None of those episodes reached within 15 mm or recorded a valid hold step. Learning within a run had improved, but the precision/holding task remained unsolved. These historical results do not measure the newly added continuation feature.

## Explicit continuation preserves learned state

The trainer now supports `--resume CHECKPOINT` and `--resume latest`. This restores the selected full-training policy and optimizer in a new run. Omitting the option still starts a fresh policy, and startup diagnostics remain fresh.

Continuation checks compatibility and records parent identity, starting counts, new-session work, and cumulative totals. Environments start new seeded trajectories rather than restoring a mid-episode simulator state. A 300,000-transition checkpoint followed by a requested 200,000-transition session can therefore record 500,000 cumulative transitions without overwriting the parent experiment. See [Training](training.md) and [Experiment records](experiments.md).

The implementation and unit tests establish state-restoration behavior. Any improvement in success rate, stability, or time to a learning target still requires a measured continued-training experiment.

## Next experiments

1. Continue a compatible checkpoint with a finite additional budget and evaluate precision and hold progress alongside critic diagnostics.
2. Measure rollout, optimizer, diagnostic, and validation costs before choosing a different execution backend.
3. Test lower learning rate, KL stopping, and gSDE in separate experiments with matching budgets and seeds.
4. Confirm promising settings with additional training seeds and a held-out evaluation after configuration selection.

Running normalization, additional observations such as ball velocity, reward-shaping changes, and SAC remain possible later experiments. They are not enabled by the current PPO options. The [training guide](training.md) gives commands and explains how to compare one change at a time.
