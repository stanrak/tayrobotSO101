# SO-101 Physical AI simulation

A MuJoCo simulation for exploring SO-101 robot control and reinforcement learning. The workspace contains a table-mounted arm and a physical ball. You can control the arm with a laptop keyboard or train a PPO policy to reach the ball and hold a stable position near it.

![SO-101 simulation workspace](docs/simulation.png)

## What you can do

- Move all six robot joints manually, pause the simulation, and choose a new reachable ball position.
- Train and observe a policy with a fixed step budget or continuous training.
- Inspect checkpoints, learning diagnostics, and reproducible experiment records.

The default application opens manual control. The learning task uses three arm joints and evaluates reaching accuracy, stability, and collision avoidance. The current work focuses on simulation and measuring learning quality.

## Start here

On Windows, launch the prepared workspace with:

```powershell
.\run_simulation.bat
```

See [getting started](docs/getting-started.md) for installation, keyboard controls, and troubleshooting.

## Documentation

The [documentation guide](docs/README.md) provides reading paths and the complete technical reference.

| Topic | Read |
| --- | --- |
| Setup and manual control | [Getting started](docs/getting-started.md) |
| Robot, ball, physics, and learning task | [Simulation environment](docs/environment.md) |
| Training commands and learning diagnostics | [Training](docs/training.md) |
| Evaluation, checkpoints, and comparisons | [Experiments](docs/experiments.md) |
| Code structure and verification | [Development](docs/development.md) |
| Implemented stages and next experiments | [Project stages](docs/project-stages.md) |

The robot assets come from the pinned [MuJoCo Menagerie SO-101 model](https://github.com/google-deepmind/mujoco_menagerie/tree/3abca2e302625cec9429c49806121ea2855cef81/robotstudio_so101) and retain their [Apache-2.0 license](assets/robotstudio_so101/LICENSE).
