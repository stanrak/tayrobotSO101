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
| N | Reset the initial pose and sample a new target |
| Escape | Close the simulation |

Use plain keys without Ctrl or Alt; no numeric keypad is required. Press repeatedly: each press changes its joint target by 2 degrees. Holding a key does not repeat robot commands in the current MuJoCo viewer. Position commands are limited to 0.8 rad/s and the model's joint and actuator limits. Commands typed while paused take effect after resuming. MuJoCo also receives keyboard shortcuts, so some letters change rendering options. Tab toggles the viewer panels.

The green sphere marks a learning target. In manual mode the robot moves according to your keyboard commands. The launcher does not start PPO, load a trained policy, or complete the task automatically. It controls the simulated robot on this laptop.

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

Current work is limited to code, the initial simulation environment, and a startup trial ending at the **first actual network update**. The agent may choose the initial step count according to the task and training setup. There is no fixed 500-step maximum; 500 remains a convenience default when `--timesteps` is omitted. Long training and automatic evaluation require a later explicit user request. Manual keyboard control and startup training are separate entry points.

For example, a task needing 2048 initial transitions can use:

```powershell
python -m so101_sim.train --timesteps 2048 --output runs/startup_trial --seed 42
```

The agent must choose and explain a suitable initial budget before a real trial. The current PPO startup mode uses one environment, collects exactly the requested number of transitions (at least two), then performs one optimization epoch with the complete rollout as one batch. This produces exactly one optimizer step. An actual optimizer-call counter prevents a second update; a completed episode, rollout, or set of epochs is not used as a substitute for counting network updates.

The trial does not add validation episodes, comparison baselines, checkpoint selection, or an old policy. Its metadata records the selected step budget, actual transitions, actual optimizer steps, and stopping reason. This is a startup diagnostic, not evidence that the task has been learned. A new trial requires a fresh output directory and must remain within the authorized work.

Continuing beyond the first network update requires a later explicit user request and the `--full-training` mode. Choosing more initial transitions does not require that flag and does not authorize further updates. Neither the manual launcher nor code and documentation changes automatically start training.

Only the policy's sampled actions control the robot during a trial. There is no IK solver, scripted reach controller, or teacher that performs the task for the policy. The environment initializes the robot and samples reachable Cartesian targets using forward kinematics on separate scratch simulation data. The generating joint angles are withheld from the policy and never become control commands.

## Watch training as it happens

Add `--viewer` (also accepted as `--render`) to observe the robot during the actual training rollout. The window shows environment 0; the terminal reports total transitions across all environments, the observed episode count, distance to the target, and completed optimizer steps. It also announces when a rollout ends and network optimization begins.

Watch a startup trial that stops after the first network update:

```powershell
.\.venv\Scripts\python -m so101_sim.train --viewer --timesteps 500 --output runs/trial_view_01 --seed 42
```

For a user-started full training run with a live viewer:

```powershell
.\.venv\Scripts\python -m so101_sim.train --full-training --viewer --timesteps 300000 --output runs/train_view_01 --seed 42
```

Omit `--viewer` to train without a graphics window. Use a new output directory for each run. The viewer option must be supplied when starting the run; it does not attach to an already running headless process.

By default the observed environment is paced at real time. `--viewer-speed 4` displays it at four times simulation speed. Display pacing slows wall-clock training but does not alter physics time, policy actions, or update budgets. With multiple environments, only environment 0 is displayed. During network optimization or validation the robot display may remain still while terminal messages show the training phase.

Mouse camera controls remain available. The viewer uses separate copies of the training model and data, so viewer physics controls cannot modify the robot used for learning. No keyboard task controller, scripted solver, or extra policy rollout is used. Close the viewer window or press Ctrl+C in the terminal to interrupt training; its status is recorded and no final evaluation is started on interruption. The viewer closes after policy training; full training then completes its normal checkpoint evaluation.

## Initial learning environment

The first task is local reach and hold. Success requires the `gripperframe` site to stay within 15 mm of the target, with combined speed of the first three joints below 0.3 rad/s, for ten consecutive 20 Hz control steps (0.5 s). Each episode has a six second deadline.

The first three joints are policy actions; wrist and gripper targets stay at their initial values. Manual keyboard control supports all six joints. Physics runs at 200 Hz. The 15-value observation contains joint positions, velocities, current actuator commands, the Cartesian target, and Cartesian error. Exact reset settings, action ranges, and reward coefficients are recorded by `SO101ReachEnv.task_config`.

Targets start at least 40 mm from the initial end-effector and above 80 mm height, with penetration rejection at the configured 4 mm tolerance. Per-step reward is nonpositive and terminal success receives a bonus. Contacts deeper than the tolerance are checked at 20 Hz control boundaries and terminate the episode with a penalty.

## Verification and historical artifacts

```powershell
python -m unittest discover -s tests -v
python scripts/fetch_model.py --verify
```

Tests cover keyboard mappings and real MuJoCo movement through a simulated viewer, pause/reset behavior, task contracts, model and checkpoint integrity, and the first-update stopping rule. Unit tests for training use fake counters rather than real training rollouts. A visible GUI still needs a manual keyboard check on the laptop.

Earlier long-run checkpoints and reports remain under `checkpoints/first_reach/` and `reports/first_reach/` as historical artifacts. Starting the application or a startup trial does not replay or continue those experiments. Checkpoint compatibility checks remain in place; no model or checkpoint migration is performed by the launcher.

| Path | Purpose |
| --- | --- |
| `run_simulation.bat` | Double-click Windows simulation launcher |
| `so101_sim/__main__.py` | `python -m so101_sim` manual entry point |
| `so101_sim/teleop.py` | Laptop keyboard controls |
| `so101_sim/env.py` | Initial robot environment and task |
| `so101_sim/train.py` | Trial through the first network update and explicit full training mode |
| `assets/robotstudio_so101/` | Unmodified upstream robot assets |
| `runs/` | Ignored local diagnostic outputs |

Read `AGENTS.md` and every local file in `instructions` before repository work. Agent-maintained guidance belongs in English in `instructions/instruction.md`. The user-owned Vietnamese file `instructions/huongdan.md` must stay unchanged and excluded from Git.
