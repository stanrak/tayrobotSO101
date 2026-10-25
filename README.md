# SO-101 Physical AI simulation

## Start the simulation and use your laptop keyboard

Run this in PowerShell:

```powershell
cd D:\tayrobotSO101
python -m so101_sim
```

This opens a MuJoCo 3D window with the SO-101 at its initial pose. Click the 3D view, then use the laptop keys below to move the robot. The application starts in manual control mode.

Startup checks NumPy, MuJoCo, and Gymnasium before opening the window. If the current Python is missing one, it prints that interpreter's path and uses the prepared project `.venv` when available. Otherwise it prints an installation command for the exact Python that needs the packages.

On Windows you can also **double-click [run_simulation.bat](run_simulation.bat)** in File Explorer, or run:

```powershell
.\run_simulation.bat
```

The launcher uses `.venv\Scripts\python.exe` when present, otherwise `python` from PATH. The existing `python -m so101_sim.teleop` command still opens the same manual simulator.

| Increase / decrease | Robot joint |
| --- | --- |
| Q / A | Shoulder pan: turn the base |
| W / S | Shoulder lift |
| E / D | Elbow flex |
| R / F | Wrist flex |
| T / G | Wrist roll |
| Y / H | Gripper opening angle |
| Space | Pause / resume simulation |
| Backspace | Move joint targets smoothly back to the initial pose |
| N | Reset the initial pose and explicitly choose a new reachable ball position |
| Escape | Close the simulation |

Use plain keys without Ctrl or Alt; no numeric keypad is required. Press repeatedly: each press changes its joint target by 2 degrees. Holding a key does not repeat robot commands in the current MuJoCo viewer. Position commands are limited to 0.8 rad/s and the model's joint and actuator limits. Commands typed while paused take effect after resuming. MuJoCo also receives keyboard shortcuts, so some letters change rendering options. Tab toggles the viewer panels.

The green ball starts on the desk within the robot's verified reach. Gravity and contact forces can move it, including off the desk edge. Press N to explicitly choose another reachable position. Manual mode does not automatically resample targets. In manual mode the robot moves according to your keyboard commands. The launcher does not start PPO, load a trained policy, or complete the task automatically. It controls the simulated robot on this laptop.

For an automatically closing manual window:

```powershell
python -m so101_sim --seconds 10
```

![Initial SO-101 simulation](docs/simulation.png)

## Install dependencies if needed

Run from the repository root. The project was verified with Python 3.14.6 and the versions in `requirements-sim.txt`. If these simulation dependencies are already installed, the start command above is sufficient.

For a separate virtual environment, activation is optional:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-sim.txt
.\.venv\Scripts\python -m so101_sim
```

`requirements-sim.txt` installs NumPy, MuJoCo, and Gymnasium for keyboard control. Install `requirements.txt` into the same environment when reinforcement learning libraries are needed.

If `ModuleNotFoundError: No module named 'gymnasium'` appears in a terminal or IDE, install into the interpreter you will use to launch the simulation:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-sim.txt
.\.venv\Scripts\python -m so101_sim
```

If the window does not open, run the explicit `.venv` command in PowerShell to read the error. Use the same Python interpreter for dependency installation and application launch. The batch launcher also uses this project environment.

The robot uses the pinned [MuJoCo Menagerie SO101 model](https://github.com/google-deepmind/mujoco_menagerie/tree/3abca2e302625cec9429c49806121ea2855cef81/robotstudio_so101), with its original dynamics, meshes, collisions, joint and actuator limits, and [Apache-2.0 license](assets/robotstudio_so101/LICENSE). Its assets are included:

```powershell
python scripts/fetch_model.py --verify
```

`python scripts/fetch_model.py` restores missing or damaged assets from the pinned upstream commit. `assets/model_manifest.json` records all downloaded file hashes.

## Development scope and the first network update

Agent-run development checks are limited to code, the initial simulation environment, and a startup trial ending at the **first actual network update**. Use `--trial` explicitly for that diagnostic. The agent may choose the initial step count according to the task and training setup. There is no fixed 500-step maximum; 500 remains the trial default when `--timesteps` is omitted. Code changes do not authorize the agent to launch long training or automatic evaluation. User-started training runs continuously by default; manual keyboard control remains a separate entry point.

For example, a task needing 2048 initial transitions can use:

```powershell
.\.venv\Scripts\python -m so101_sim.train --trial --timesteps 2048 --seed 42
```

The agent must choose and explain a suitable initial budget before a real trial. The current PPO startup mode uses one environment, collects exactly the requested number of transitions (at least two), then performs one optimization epoch with the complete rollout as one batch. This produces exactly one optimizer step. An actual optimizer-call counter prevents a second update; a completed episode, rollout, or set of epochs is not used as a substitute for counting network updates.

The trial does not add validation episodes, comparison baselines, checkpoint selection, or an old policy. Its metadata records the selected step budget, actual transitions, actual optimizer steps, and stopping reason. This is a startup diagnostic, not evidence that the task has been learned. Each authorized trial creates a separate experiment under the managed `runs/` root and starts a fresh policy.

Agents must use `--trial` for authorized startup checks. Continuing beyond the first network update requires explicit user authorization to run training. Choosing more initial transitions with `--trial` does not authorize further updates. Neither the manual launcher nor code and documentation changes automatically start training.

Only the policy's sampled actions control the robot during a trial. There is no IK solver, scripted reach controller, or teacher that performs the task for the policy. The environment verifies reachable ball positions using forward kinematics on separate scratch simulation data. Sampling never generates robot actions, and its generating joint angles are withheld from observations, info, and controls.

## Separate experiment records for every run

Both startup trials and full training automatically create **`runs/experiments/<UTC timestamp>_<mode>_seed<seed>_<unique suffix>/`**. Omit `--output` in normal use: every invocation receives a different directory, including runs with identical settings or timestamps. Existing experiments stay at their original paths; starting another run never moves, deletes, or overwrites them.

Each experiment records the command, seed, timestamps, status, Python/platform/package versions, Git commit and dirty flag, source file SHA256 hashes, and an exact source/config snapshot. `training.json` retains the full hyperparameters, environment signature, requested/actual transitions, and actual optimizer steps. Checkpoints, per-environment episode logs, progress metrics, and any measured evaluation results stay together. Failed and interrupted runs are retained with their observed status; an abrupt process kill may leave the last recorded running status and partial logs.

`runs/latest.json` points to the most recently started managed training experiment; it is a convenience pointer, not the experiment data. Evaluation can still find the latest checkpoint automatically. Existing `runs/current/`, `runs/history/`, and other saved run folders are preserved without relocation. When no latest pointer exists, checkpoint lookup falls back to the legacy `runs/current/` directory. Starting a new run always initializes a fresh policy and keeps compatibility checks.

The main files are:

| Location | Contents |
| --- | --- |
| `runs/experiments/<run_id>/run.json` | Identity, provenance, source hashes, final status, and observed counts |
| `runs/experiments/<run_id>/source/` | Exact source/config snapshot used at run start |
| `runs/experiments/<run_id>/training.json` | Hyperparameters, environment, status, and actual network updates |
| `runs/experiments/<run_id>/progress.csv`, `monitor_*.csv` | Training metrics and episode records |
| `runs/experiments/<run_id>/*_model.zip` and matching metadata | Startup, best-validation, and final policies |
| `runs/experiments/<run_id>/evaluation.json` | Held-out report produced by a completed full training run, when measured |
| `<checkpoint directory>/evaluations/<evaluation_id>/` | Separate report and provenance for each standalone evaluation invocation |
| `runs/index.csv` | Rebuildable comparison table covering new and legacy runs |
| `runs/latest.json` | Pointer used by default checkpoint lookup |

To rebuild and view comparisons from saved records without running a policy:

```powershell
.\.venv\Scripts\python -m so101_sim.experiments
```

Open `runs/index.csv` in a spreadsheet to compare seeds, task/environment/source fingerprints, code revisions, hyperparameters, step budgets, optimizer counts, duration, success rate, return, and final distance. Each evaluation row identifies its checkpoint hash and evaluation seeds. Compare matching task/environment signatures, evaluation seeds, and budgets; validation, held-out, and standalone results are labeled separately. Empty metrics mean they were not measured. A startup trial has no task-performance score, and a short diagnostic must not be ranked as a learned policy. The index is derived and can be regenerated; raw experiment files are retained.

To evaluate and view the latest policy without specifying a checkpoint path:

```powershell
.\.venv\Scripts\python -m so101_sim.evaluate --render --episodes 5
```

The evaluator selects the latest full-training best policy, startup trial policy, or final policy. Every invocation writes a new `evaluations/<evaluation_id>/evaluation.json` next to the checkpoint, with its own `run.json` and source snapshot. Reports identify the checkpoint by absolute path and SHA256. Existing reports remain unchanged, including evaluations of historical checkpoints. Finish or stop the current training before evaluating it separately. The live training viewer remains the way to watch an active run.

Output locking prevents simultaneous training, standalone evaluation, or catalog rebuild writers. The existing `runs/.current.lock` name is retained so an older running process remains protected; the operating system releases its lock automatically, including after a crash. Do not delete it to bypass another running process. Training `--output` accepts a custom empty directory; an existing experiment is rejected. Evaluation `--output` accepts a new report file and also retains a canonical copy/provenance under the checkpoint's `evaluations/` directory; existing files are rejected. Custom training folders outside `runs/` are not in the default index; use `python -m so101_sim.experiments --root <parent>` to compare them.

Run results stay local: `.gitignore` excludes `runs/`, historical `checkpoints/` and `reports/`, and generated learning GIFs. Every new experiment also contains an ignore marker, protecting custom output folders from ordinary `git add`. Previously tracked result files are removed from the Git index without deleting the local copies. Static simulation illustrations remain documentation assets. Do not force-add experimental outputs or rewrite Git history as part of ordinary runs.

## Run training continuously or with a step limit

Run training without a graphics window:

```powershell
.\.venv\Scripts\python -m so101_sim.train
```

Training continues until Ctrl+C. There is no default total step limit. The existing `--full-training` flag is still accepted as an alias for this default mode. Every new invocation starts a fresh policy in its own experiment directory.

Add `--viewer` (also accepted as `--render`) to observe the robot during the actual training rollout. The window shows environment 0; the terminal reports total transitions across all environments, the observed episode count, distance to the target, and completed optimizer steps. It also announces when a rollout ends and network optimization begins.

Run continuous training with a live viewer:

```powershell
.\.venv\Scripts\python -m so101_sim.train --viewer --seed 42
```

To stop after a particular number of training transitions, add `--timesteps`:

```powershell
.\.venv\Scripts\python -m so101_sim.train --viewer --timesteps 300000 --seed 42
```

The limit counts total training transitions across all environments and stops at the exact requested count. Full training normally uses eight environments; if the limit is not divisible by eight, the default switches to one environment. An explicit `--envs` must divide the limit. PPO updates complete rollouts; a final incomplete rollout is discarded without another network update. Periodic validation and final held-out evaluation use separate episodes outside this training transition count.

For a startup diagnostic ending at the first optimizer step, use the explicit trial mode:

```powershell
.\.venv\Scripts\python -m so101_sim.train --trial --viewer --timesteps 500 --seed 42
```

Omit `--viewer` from any command to run without a graphics window. All modes create separate records under `runs/experiments/`. The viewer option must be supplied when starting the run; it does not attach to an already running headless process. The terminal displays `unlimited` as the target during continuous training.

By default the observed environment is paced at real time. `--viewer-speed 4` displays it at four times simulation speed. Display pacing slows wall-clock training but does not alter physics time, policy actions, or update budgets. With multiple environments, only environment 0 is displayed. During network optimization or validation the robot display may remain still while terminal messages show the training phase.

Mouse camera controls remain available. The viewer uses separate copies of the training model and data, so viewer physics controls cannot modify the robot used for learning. No keyboard task controller, scripted solver, or extra policy rollout is used. Close the viewer window or press Ctrl+C in the terminal to interrupt training. Full training saves `final_model.zip` inside that run's experiment directory with actual step and optimizer counts, records the interrupted status, and skips final evaluation. At a requested step limit, the viewer closes and full training completes its normal checkpoint evaluation.

## Desk, physical ball, and learning environment

The workspace uses a fixed **120 × 70 × 75 cm desk** (width × depth × floor-to-top height), a common desk size matching the published [IKEA TROTTEN dimensions](https://www.ikea.com.tw/en/products/office-workspace/office-desks-and-tables/trotten-spr-39621918). This is a dimension reference, not a detailed replica or a universal furniture standard. The simulated desk has a 30 mm thick top and four fixed collision-enabled legs. The robot is mounted on the top at world position `[0, 0, 0.75]` m; the desktop spans x = −0.2…1.0 m and y = −0.35…0.35 m. Upstream robot files, joint dynamics, actuator limits, and meshes remain unchanged.

The green ball has a 15 mm radius and 20 g mass. A free joint allows translation and rotation. Gravity is explicitly `[0, 0, -9.81]` m/s². Six-dimensional contacts provide normal support and sliding, torsional, and rolling friction, with coefficients `[0.6, 0.005, 0.0005]` and contact `solref=[0.002, 1]`. Physics uses 1 ms steps and up to 50 solver iterations to resolve small-ball impacts. MuJoCo calculates the forces; the application never pins the ball to a surface, teleports it during stepping, or applies an artificial support force. See the [MuJoCo contact reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom). These are nominal parameters for an unspecified ball and surface, not measurements calibrated to particular real materials. Aerodynamic drag is not modeled.

Default targets are sampled **only on the desk within the robot's reach**. A candidate is accepted only after forward kinematics finds a scratch pose inside the existing three-joint action limits whose end-effector is within 14 mm of the ball centre, with all robot, table, and actual ball contacts nonpenetrating to a 1 µm numerical tolerance. The sampler uses pan −0.35…0.35, lift −0.55…−0.45, and elbow 1.48…1.6 rad, plus a 6…10 mm outward offset from the end-effector's XY position to clear the fixed jaw. The actual robot still resets to its slightly jittered home pose, at least 40 mm from the ball. No generating angles are used as actions or exposed to the policy. Success still requires the policy to reach and hold the ball itself.

The stationary ball starts directly on the desk, with centre z = 0.765 m and zero initial velocity. Positions vary near x = 0.24…0.26 m and y = ±0.075 m; the FK and collision checks determine acceptance, rather than an arbitrary 3D box. Automatic target changes and manual N select a point at least 20 mm from the previous spawn. Explicit seeded resets reproduce their seed independently of previous positions. Physical ball motion after placement remains unrestricted.

Each learning target receives **1,200 control steps, equivalent to 60 simulated seconds at 20 Hz**, before automatic resampling. A single attempt also has a 1,200-step limit, increased from the previous 120 steps / 6 seconds. Success or collision can end an attempt early; the automatic unseeded reset then restores the **same initial ball position** and resets the robot, retaining the cumulative target step count. At the 1,200-step target boundary, the current attempt ends and the next automatic reset samples a new reachable target. This prevents repeated early failures from changing the goal every few frames. It provides more interaction time without claiming that learning is guaranteed in one target budget.

Exposure is counted independently for each environment, across PPO rollouts and network updates. Simulation time is derived from actual control steps, not wall-clock time: a real-time viewer shows about 60 seconds per target, higher viewer speeds shorten that wall-clock interval, and headless execution runs as fast as possible. Training status reports target age in simulated seconds and steps. This is an environment budget, separate from the trainer's total `--timesteps` budget and the explicit first-update trial limit.

For an external caller, seeded resets always start a fresh reproducible target; `new_target` explicitly overrides the dwell budget:

```python
env.reset(seed=42)  # Fresh, reproducible reachable table target.
env.reset()  # Retry the existing target until its 1,200-step budget expires.
env.reset(options={"new_target": True})  # Explicitly choose a new target now.
```

The table-mounted SO-101 cannot reach the floor 0.75 m below its base, so floor positions are excluded from default learning and manual targets. An explicit `env.reset(seed=42, options={"ball_surface": "floor"})` remains available only for physics diagnostics, such as support-force and fall tests. The existing three-joint reach controller is retained; a future learnable floor-reaching task requires a separately authorized mounting/task change.

The task remains reach and hold, with the ball's **current simulated centre** as the target. Success requires the `gripperframe` site to stay within 15 mm of that centre, with combined speed of the first three joints below 0.3 rad/s, for ten consecutive 20 Hz control steps (0.5 s).

The first three joints are policy actions; wrist and gripper targets stay at their initial values. Manual keyboard control supports all six joints. Physics runs at 1000 Hz with 50 physics steps per 20 Hz policy control step. The 15-value observation contains robot joint positions, velocities, current actuator commands, the current ball position, and Cartesian error. The ball adds seven position coordinates and six velocity coordinates to MuJoCo state without adding robot controls. Exact geometry, physics, reachability checks, target timing, action ranges, and reward coefficients are recorded by `SO101ReachEnv.task_config` under task `SO101ReachableSurfaceBall-v3`. Existing checkpoints are incompatible and are rejected by the existing compatibility checks; they are retained without migration.

Ball placements start at least 40 mm from the initial end-effector. Per-step reward is nonpositive and terminal success receives a bonus. Robot self-collisions and robot contacts with the table or floor deeper than 4 mm are checked at 20 Hz control boundaries and terminate the episode with a penalty. Ball contacts with the table, floor, or robot are allowed, so ordinary support and pushing do not end an episode. Observations, reward, and reported distance follow the moving ball. Viewer physics edits cannot disable gravity or replace the application-owned state.

## Verification and historical artifacts

```powershell
python -m unittest discover -s tests -v
python scripts/fetch_model.py --verify
```

Tests cover keyboard mappings and real MuJoCo movement through a simulated viewer, pause/reset behavior, reachable supported ball placement, target persistence through automatic resets, exact target step budgets, gravitational falls, friction, drops from the desk edge, moving-goal observations, task contracts, model and checkpoint integrity, continuous training interruption, exact step limits, and the first-update stopping rule. Unit tests for training use fake counters rather than real training rollouts. A visible GUI still needs a manual keyboard check on the laptop.

Earlier long-run checkpoints and reports already on this laptop remain under `checkpoints/first_reach/` and `reports/first_reach/` as local historical artifacts excluded from Git; a clean clone does not contain them. Starting the application or a startup trial does not replay or continue those experiments. Checkpoint compatibility checks remain in place; no model or checkpoint migration is performed by the launcher.

| Path | Purpose |
| --- | --- |
| `run_simulation.bat` | Double-click Windows simulation launcher |
| `so101_sim/__main__.py` | `python -m so101_sim` manual entry point |
| `so101_sim/teleop.py` | Laptop keyboard controls |
| `so101_sim/env.py` | Initial robot environment and task |
| `so101_sim/train.py` | Continuous training, optional step limit, and explicit first-update trial |
| `assets/robotstudio_so101/` | Unmodified upstream robot assets |
| `so101_sim/experiments.py` | Experiment provenance and saved-result comparisons |
| `runs/experiments/` | Separate training records and nested evaluation records |
| `runs/index.csv`, `runs/latest.json` | Comparison table and latest-run discovery |
| `runs/current/`, `runs/history/` | Preserved local legacy outputs |

Read `AGENTS.md` and every local file in `instructions` before repository work. Agent-maintained guidance belongs in English in `instructions/instruction.md`. The user-owned Vietnamese file `instructions/huongdan.md` must stay unchanged and excluded from Git.
