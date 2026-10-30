# Project stages

[Documentation guide](README.md) | [Project overview](../README.md)

The project develops a simulated robot task in stages so that control, physics, and learning quality can be examined separately. This page describes implementation progress and broad learning observations. Detailed run reviews and measurements remain confidential local records under `local-artifacts/reports/`.

| Stage | Status | Progress |
| --- | --- | --- |
| Manual simulation | Implemented | Laptop keyboard control exposes robot motion, camera behavior, contacts, and target placement. |
| Physical workspace and reachable ball | Implemented | A desk-mounted robot interacts with a free ball under gravity and contact dynamics. |
| Reach-and-hold task | Retained | Three-joint proximity task retained for isolated checks; it is not a physical grasp. |
| Stage-one grasp, lift and hold | Implemented; learning unverified | Six joint actions, physical two-jaw hold and tabletop clearance; see [Grasping](grasping.md). |
| Reproducible training and evaluation | Implemented | Separate local records preserve settings, provenance, checkpoints, and measurements. |
| Learning diagnostics and controlled tuning | Implemented | Reward scaling, critic/action diagnostics, phase timings, exploration options, and an optional subprocess backend support comparisons. |
| Explicit checkpoint continuation | Implemented | Compatibility checks restore policy and optimizer state into a separate run with parent lineage and separate session/cumulative counters. |
| Reliable task completion | In progress | Approach behavior has improved in exploratory work; precise approach and sustained holding still need improvement. |

## Manual control establishes the workspace

The application starts with manual keyboard control. This makes robot motion, camera behavior, contact, and target placement visible before training. The manual controller supports all robot joints and is separate from policy training. See [getting started](getting-started.md).

## The physical ball defines the current task

The ball starts on the desk within verified reach and remains a free physical body during stepping. Its current position is the target. Each target has a bounded control-step budget, and the default grasp task requires two-jaw contact, clearance above the tabletop and a stable hold. A target lasts five simulated minutes across retries. The reachability sampler certifies approach proximity; it does not certify a complete grasp trajectory.

Earlier task versions and checkpoints have different semantics. Their observations do not establish a baseline for the current environment. See [environment](environment.md) and [checkpoint compatibility](experiments.md).

## Training instrumentation supports controlled experiments

The trainer applies reward scaling only to learning; episode logs and evaluation retain original reward units. Diagnostics distinguish critic fitting, actor saturation, exploration spread, action clipping, approach accuracy, and valid holding. Actual optimizer counters and phase timings help explain the training process without substituting for task performance.

Exploratory work has shown useful approach behavior and critic learning. Reliable precision and sustained holding remain open learning goals. Detailed evidence, checkpoint comparisons, and run-specific conclusions belong in confidential local reviews rather than this documentation.

## Explicit continuation preserves learned state

The trainer supports `--resume CHECKPOINT` and `--resume latest`. This restores the selected full-training policy and optimizer in a new run. Omitting the option starts a fresh policy, and startup diagnostics remain fresh.

Continuation checks compatibility and records parent identity, starting counts, new-session work, and cumulative totals. Environments start new seeded trajectories rather than restoring a mid-episode simulator state. See [Training](training.md) and [Experiment records](experiments.md).

## Further learning work

Controlled comparisons can examine Gaussian and state-dependent PPO exploration with matching settings, task, budgets, and seeds. Approach accuracy, valid holding, actor means, exploration spread, and clipping should be interpreted together. Candidate improvements need confirmation across training seeds and unused evaluation seeds.

The grasp task adds ball velocity and physical contact observations with its own reward and success definition. Running normalization and other algorithms remain possible later experiments. The [training guide](training.md) explains how to compare one change at a time.
