# Documentation guide

[Project overview](../README.md)

These documents explain how to use the SO-101 simulation, how the learning task works, and how to interpret its results. Commands assume PowerShell opened in the repository root unless a page states otherwise.

## Choose a reading path

- **Use the simulator:** start with [getting started](getting-started.md), then read the [environment](environment.md) to understand the robot and physical ball.
- **Train a policy:** read the [environment](environment.md), [training](training.md), and [experiments](experiments.md) pages in that order.
- **Understand or develop the implementation:** read [development](development.md), then the technical page for the component you are changing.
- **Review project progress:** read [project stages](project-stages.md), then use the experiment guide to examine local measurements.

## Reference

| Document | Contents |
| --- | --- |
| [Getting started](getting-started.md) | Dependencies, Windows launchers, keyboard controls, troubleshooting, and model assets |
| [Simulation environment](environment.md) | Desk and ball dynamics, goal placement, action/observation spaces, reward, and success conditions |
| [Training](training.md) | PPO modes, command options, transition budgets, viewer behavior, reward scaling, and critic diagnostics |
| [Experiments](experiments.md) | Run folders, checkpoint compatibility, validation and held-out evaluation, comparison tables, and historical records |
| [Development](development.md) | Source layout, component responsibilities, local checks, and verification limits |
| [Project stages](project-stages.md) | Implemented capabilities, broad learning progress, and remaining work |

## Reading results correctly

Simulation time, training transitions, optimizer steps, and wall-clock time measure different things. A startup trial checks initialization and one optimizer step; it does not establish task performance. Validation selects a checkpoint, while held-out evaluation estimates performance on separate seeds. The [training](training.md) and [experiment](experiments.md) guides explain these distinctions.

Run data, detailed reviews, and measurements are confidential local artifacts. Detailed reviews belong under `reports/`; versioned documentation contains general technical descriptions, process, and broad achievements. A fresh checkout includes the implementation and documentation without local results.
