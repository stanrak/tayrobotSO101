# Project stages

[Documentation guide](README.md) | [Project overview](../README.md)

The project develops a simulated robot task in stages so that control, physics, and learning quality can be examined separately. This page describes implementation progress and broad learning observations. Detailed run reviews and measurements remain confidential local records under `report/`.

| Stage | Status | Progress |
| --- | --- | --- |
| Manual simulation | Implemented | Laptop keyboard control exposes robot motion, camera behavior, contacts, and target placement. |
| Physical workspace and reachable ball | Implemented | A desk-mounted robot interacts with a free ball under gravity and contact dynamics. |
| Reach-and-hold learning task | Implemented | A policy approaches the moving ball and must maintain accurate, stable proximity. |
| Reproducible training and evaluation | Implemented | Separate local records preserve settings, provenance, checkpoints, and measurements. |
| Learning diagnostics and controlled tuning | Implemented | Reward scaling, critic/action diagnostics, phase timings, exploration options, and an optional subprocess backend support comparisons. |
| Explicit checkpoint continuation | Implemented | Compatibility checks restore policy and optimizer state into a separate run with parent lineage and separate session/cumulative counters. |
| Reliable task completion | In progress | Approach behavior has improved in exploratory work; precise approach and sustained holding still need improvement. |

## Manual control establishes the workspace

The application starts with manual keyboard control. This makes robot motion, camera behavior, contact, and target placement visible before training. The manual controller supports all robot joints and is separate from policy training. See [getting started](getting-started.md).

## The physical ball defines the current task

The ball starts on the desk within verified reach and remains a free physical body during stepping. Its current position is the target. Each target has a bounded control-step budget, and success requires a sustained accurate, slow reach.

Earlier task versions and checkpoints have different semantics. Their observations do not establish a baseline for the current environment. See [environment](environment.md) and [checkpoint compatibility](experiments.md).

## Training instrumentation supports controlled experiments

The trainer applies reward scaling only to learning; episode logs and evaluation retain original reward units. Diagnostics distinguish critic fitting, actor saturation, exploration spread, action clipping, approach accuracy, and valid holding. Actual optimizer counters and phase timings help explain the training process without substituting for task performance.

Exploratory work has shown useful approach behavior and critic learning. Reliable precision and sustained holding remain open learning goals. Detailed evidence, checkpoint comparisons, and run-specific conclusions belong in confidential local reviews rather than this documentation.

## Explicit continuation preserves learned state

The trainer supports `--resume CHECKPOINT` and `--resume latest`. This restores the selected full-training policy and optimizer in a new run. Omitting the option starts a fresh policy, and startup diagnostics remain fresh.

Continuation checks compatibility and records parent identity, starting counts, new-session work, and cumulative totals. Environments start new seeded trajectories rather than restoring a mid-episode simulator state. See [Training](training.md) and [Experiment records](experiments.md).

## Further learning work

Controlled comparisons can examine Gaussian and state-dependent PPO exploration with matching settings, task, budgets, and seeds. Approach accuracy, valid holding, actor means, exploration spread, and clipping should be interpreted together. Candidate improvements need confirmation across training seeds and unused evaluation seeds.

Running normalization, additional observations such as ball velocity, reward-shaping changes, and other algorithms remain possible later experiments. They are not enabled by the current PPO options. The [training guide](training.md) explains how to compare one change at a time.
