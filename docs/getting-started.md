# Getting started and manual control

[Project overview](../README.md) · [Documentation index](README.md)

The simulator opens an SO-101 arm mounted on a desk, with a physical ball on the tabletop. Manual control is the default: the robot moves when you press the laptop keys. Training and trained-policy playback have separate commands, described in [Training](training.md) and [Experiments and evaluation](experiments.md).

## Open the simulator

Run from the repository root in PowerShell:

```powershell
cd D:\tayrobotSO101
python -m so101_sim
```

If the project environment is already prepared, you can select its Python directly:

```powershell
.\.venv\Scripts\python.exe -m so101_sim
```

On Windows, double-click [run_simulation.bat](../run_simulation.bat), or run:

```powershell
.\run_simulation.bat
```

The batch launcher uses `.venv\Scripts\python.exe` when it exists, otherwise `python` from PATH. The module entry point checks NumPy, MuJoCo, and Gymnasium. If the current interpreter lacks a package and a project environment exists, it retries with the project Python; otherwise it prints an installation command for the interpreter that failed.

For a manual window that closes after ten seconds:

```powershell
.\.venv\Scripts\python.exe -m so101_sim --seconds 10
```

The existing `python -m so101_sim.teleop` entry point also opens manual control.

![Initial desk-mounted SO-101 simulation](simulation.png)

## Laptop keyboard controls

Click the 3D view before pressing keys. Plain keys are sufficient; Ctrl, Alt, and a numeric keypad are not needed.

| Increase / decrease | Joint or action |
| --- | --- |
| Q / A | Shoulder pan: turn the base |
| W / S | Shoulder lift |
| E / D | Elbow flex |
| R / F | Wrist flex |
| T / G | Wrist roll |
| Y / H | Gripper opening angle |
| Space | Pause or resume simulation |
| Backspace | Move joint targets smoothly toward the initial pose |
| N | Reset the robot to its initial pose and select a new reachable ball position |
| Escape | Close the window |

Each joint-key press changes the requested angle by two degrees. Repeated presses move farther; holding a key does not repeat commands in the current MuJoCo viewer. Position commands respect joint and actuator limits and change at no more than 0.8 rad/s. Commands entered while paused are applied after simulation resumes.

The viewer also has its own camera and rendering controls. Some letter keys can change rendering options as well as send robot commands; Tab toggles the viewer panels. Mouse camera controls remain available.

The application owns the simulation state, so native viewer resets, joint/control sliders, and mouse physics edits are ignored. Use the keyboard commands above to change the robot or choose another ball position.

The ball can move when touched and can fall off the desk. Manual mode keeps the current ball until you press N; it does not automatically choose another target. The launcher starts keyboard control without loading a policy or starting a learning run. See [The environment](environment.md) for the physical behavior and task definition.

## Prepare dependencies when needed

The recorded setup uses Python 3.14.6 and the versions in [requirements-sim.txt](../requirements-sim.txt). If the simulator already opens, no installation is needed.

To prepare a project environment yourself:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-sim.txt
.\.venv\Scripts\python.exe -m so101_sim
```

Activation is optional because these commands select the environment's interpreter explicitly. The simulation requirements contain NumPy, MuJoCo, and Gymnasium. The learning commands additionally use [requirements.txt](../requirements.txt):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

These are setup instructions, not steps performed automatically by the launcher.

## Troubleshooting startup

| Symptom | What to check |
| --- | --- |
| `ModuleNotFoundError`, for example `gymnasium` | Install `requirements-sim.txt` using the same Python that will launch the simulator. |
| Double-clicking opens no usable window | Run the explicit `.venv` command in PowerShell so the error remains visible. |
| The IDE and terminal behave differently | Compare their selected Python interpreters; the project launcher prefers `.venv` when present. |
| Key presses do not move the robot | Click the 3D view, check whether simulation is paused, and use repeated key presses. |
| A model integrity or missing-assets error appears | Verify the model files with the command below. |

The robot assets come from a pinned [MuJoCo Menagerie SO101 revision](https://github.com/google-deepmind/mujoco_menagerie/tree/3abca2e302625cec9429c49806121ea2855cef81/robotstudio_so101), with the upstream dynamics, meshes, collision geometry, limits, and [Apache-2.0 license](../assets/robotstudio_so101/LICENSE). The local [model manifest](../assets/model_manifest.json) records the expected hashes.

Verify the included assets without downloading replacements:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_model.py --verify
```

Running `scripts/fetch_model.py` without `--verify` restores missing or damaged assets from the pinned revision and can access the network. For source layout and local checks, continue to [Development](development.md).
