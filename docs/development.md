# Architecture and verification

[Documentation](README.md) · [Project overview](../README.md) · [Environment](environment.md) · [Training](training.md) · [Experiments](experiments.md)

The project separates manual interaction, the physical learning environment, training instrumentation, and reproducible result storage. The default application opens the manually controlled simulator; training and policy evaluation have their own entry points.

## Source map

| Path | Responsibility |
| --- | --- |
| [`run_simulation.bat`](../run_simulation.bat) | Windows launcher, preferring the prepared project interpreter. |
| [`so101_sim/__main__.py`](../so101_sim/__main__.py) | Startup dependency checks and the default manual application. |
| [`so101_sim/teleop.py`](../so101_sim/teleop.py) | Laptop keyboard controls, pause/reset, camera, and viewer state. |
| [`so101_sim/env.py`](../so101_sim/env.py) | Physical workspace, reachable ball sampling, observations/actions/reward, success, and reset cadence. |
| [`so101_sim/train.py`](../so101_sim/train.py) | Continuous/bounded PPO training, single-update trials, fixed-seed validation, checkpoints, and CLI settings. |
| [`so101_sim/learning.py`](../so101_sim/learning.py) | Training-only reward scaling and PPO diagnostics from already collected data. |
| [`so101_sim/training_view.py`](../so101_sim/training_view.py) | Observation of actual training state through independent viewer model/data copies. |
| [`so101_sim/vector_env.py`](../so101_sim/vector_env.py) | Optional subprocess environment startup and bounded worker cleanup. |
| [`so101_sim/evaluate.py`](../so101_sim/evaluate.py) | Checked checkpoint loading, seeded comparisons, metrics, and policy playback. |
| [`so101_sim/outputs.py`](../so101_sim/outputs.py) | Unique output directories, writer locks, and latest-checkpoint discovery. |
| [`so101_sim/experiments.py`](../so101_sim/experiments.py) | Run provenance, source snapshots, finalization, and comparisons from saved records. |
| [`scripts/fetch_model.py`](../scripts/fetch_model.py) | Verification or restoration of the pinned upstream robot assets. |
| [`assets/model_manifest.json`](../assets/model_manifest.json) | Upstream revision and per-file asset SHA256 values. |
| `tests/` | Local unit and physical-contract checks. |

The robot assets keep their upstream geometry, joint/actuator limits, dynamics, and collisions. The desk and free ball are composed by the environment. Task/environment signatures describe the semantics associated with a saved policy; changing those semantics makes older checkpoints incompatible. See [Environment](environment.md) for the physical and learning interface.

## Verify an existing environment

Run from the repository root with the same interpreter used for the application:

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
.\.venv\Scripts\python scripts/fetch_model.py --verify
```

The asset command verifies the pinned files and hashes without fetching replacements. Running `scripts/fetch_model.py` without `--verify` is the separate restoration operation. [Getting started](getting-started.md) explains interpreter selection and dependencies.

The local test suite covers:

- Keyboard mappings, real MuJoCo movement, pause/resume/reset, and independent viewer state.
- Fixed geometry/physics, gravity/contact/friction, moving-ball observations, reachable placements, and target persistence.
- Success holding requirements, collision failure, exact target/attempt limits, and truncation behavior.
- Exact training transition caps, first-update stopping, actual optimizer counters, KL stopping, and interruption records.
- Raw versus scaled rewards, sampled critic diagnostics, checkpoint compatibility, and subprocess ownership/cleanup.
- Unique output records, locking, latest-run lookup, provenance allowlists, and comparison labels.

Training-budget tests use fake environments and optimizer counters rather than real learning runs. Critic tests can use synthetic tensors, and worker-lifecycle tests use fake pipes/processes. Physical environment tests do step MuJoCo. Passing these tests verifies specified behavior, not the task success rate of a trained policy or the speed of real multiprocessing.

CLI help can be inspected without starting simulation or training:

```powershell
.\.venv\Scripts\python -m so101_sim.train --help
.\.venv\Scripts\python -m so101_sim.evaluate --help
.\.venv\Scripts\python -m so101_sim.experiments --help
```

## Manual and performance verification

A visible application still needs a laptop check for window creation, focus, keyboard response, camera controls, and closing behavior. The keyboard guide is in [Getting started](getting-started.md).

Training effectiveness requires separate measured experiments. Unit tests do not establish that the reward scale, exploration method, or environment backend improves learning. Compare raw task metrics on matching signatures/seeds/budgets and use held-out evaluation after selecting a configuration. [Training](training.md) explains the diagnostic counters and overlapping timing scopes; [Experiments](experiments.md) explains how records support those comparisons.

