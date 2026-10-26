# Project stages

[Documentation guide](README.md) · [Project overview](../README.md)

The project develops a simulated robot task in stages so that control, physics, and learning quality can be examined separately. The status below distinguishes implemented behavior from improvements that still need experimental evidence.

| Stage | Status | Result |
| --- | --- | --- |
| Manual SO-101 simulation | Implemented | A laptop keyboard controls six robot joints in a MuJoCo viewer. |
| Physical workspace and reachable ball | Implemented | A table-mounted robot interacts with a free ball under gravity and contact dynamics. |
| Reach-and-hold learning task | Implemented | A three-joint policy approaches the moving ball and must maintain the required accuracy and stability. |
| Reproducible training and evaluation | Implemented | Separate run records preserve configuration, source snapshots, checkpoints, metrics, and measured evaluation results. |
| Critic diagnostics and controlled tuning | Implemented; learning benefit unmeasured | Training-only reward scaling, phase timings, KL/gSDE options, and an optional subprocess backend support controlled experiments. |
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

The multiplier is a testable response to the observed critic behavior. Improved success rate or throughput has not yet been demonstrated for this configuration. Learning rate and epoch defaults remain unchanged so that the first comparison can isolate the reward-scale change.

## Next experiments

1. Use a finite pilot to compare critic value spread, saturation, and the distance curve with the saved baseline.
2. Measure rollout, optimizer, diagnostic, and validation costs before choosing a different execution backend.
3. Test lower learning rate, KL stopping, and gSDE in separate experiments with matching budgets and seeds.
4. Confirm promising settings with additional training seeds and a held-out evaluation after configuration selection.

Running normalization, additional observations such as ball velocity, reward-shaping changes, and SAC remain possible later experiments. They are not enabled by the current PPO options. The [training guide](training.md) gives commands and explains how to compare one change at a time.
