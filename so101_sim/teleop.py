"""Control the simulated SO-101 with keys in the MuJoCo viewer window."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from queue import Empty, SimpleQueue
import time

import mujoco
import mujoco.viewer
import numpy as np


JOINT_KEY_PAIRS = ("QA", "WS", "ED", "RF", "TG", "YH")
STEP_RADIANS = math.radians(2.0)
MAX_COMMAND_SPEED = 0.8


@dataclass(frozen=True)
class KeyCommand:
    """A keyboard event, independent of the viewer and physics state."""

    kind: str
    joint_index: int = -1
    direction: int = 0


def key_to_command(key: int) -> KeyCommand | None:
    """Map a GLFW key code to a command without changing simulation state.

    GLFW supplies uppercase ASCII codes for letter keys. Lowercase ASCII is
    accepted as well so callers do not need to depend on GLFW for testing.
    """
    if ord("a") <= key <= ord("z"):
        key -= ord("a") - ord("A")
    for index, pair in enumerate(JOINT_KEY_PAIRS):
        if key == ord(pair[0]):
            return KeyCommand("joint", index, 1)
        if key == ord(pair[1]):
            return KeyCommand("joint", index, -1)
    return {
        32: KeyCommand("pause"),
        259: KeyCommand("home"),  # GLFW_KEY_BACKSPACE
        ord("N"): KeyCommand("reset"),
        256: KeyCommand("quit"),  # GLFW_KEY_ESCAPE
    }.get(key)


def slew_limited_control(
    current: np.ndarray,
    target: np.ndarray,
    limits: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Limit commanded position changes to 0.8 radians per simulated second."""
    bounded_target = np.clip(target, limits[:, 0], limits[:, 1])
    max_change = MAX_COMMAND_SPEED * dt
    return np.clip(
        current + np.clip(bounded_target - current, -max_change, max_change),
        limits[:, 0],
        limits[:, 1],
    )


def _print_targets(names: tuple[str, ...], target: np.ndarray) -> None:
    positions = ", ".join(
        f"{name}={angle:.1f} deg"
        for name, angle in zip(names, np.rad2deg(target), strict=True)
    )
    print(f"Joint targets: {positions}", flush=True)


def _sync_keyboard_view(viewer, model, data, state: np.ndarray) -> None:
    """Update the viewer without importing its built-in physics shortcuts.

    MuJoCo's Backspace shortcut resets MjData before invoking key_callback.
    Its callback cannot consume a key, so preserve integration state around
    sync. Camera and rendering changes remain available; physics is owned by
    this keyboard loop, including while paused.
    """
    with viewer.lock():
        mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        viewer.sync()
        mujoco.mj_setState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        # Publish the restored state so a native reset never appears on screen.
        viewer.sync(state_only=True)


def run(*, seconds: float | None = None) -> None:
    """Run real-time keyboard control; input is local to the focused viewer."""
    from so101_sim.env import SO101ReachEnv

    commands: SimpleQueue[KeyCommand] = SimpleQueue()

    def on_key(key: int) -> None:
        # The GUI callback only enqueues events. All physics changes happen on
        # the main thread while holding the viewer lock.
        command = key_to_command(key)
        if command is not None:
            commands.put(command)

    env = SO101ReachEnv()
    try:
        env.reset(seed=0)
        names = tuple(env.joint_names)
        limits = np.column_stack(
            (
                np.maximum(env.joint_limits[:, 0], env.ctrl_limits[:, 0]),
                np.minimum(env.joint_limits[:, 1], env.ctrl_limits[:, 1]),
            )
        )
        target = np.clip(env.home.copy(), limits[:, 0], limits[:, 1])
        control = env.data.ctrl.copy()
        print("SO-101 keyboard control (simulation only).", flush=True)
        print("Focus the MuJoCo viewer, then press keys (+ / -):", flush=True)
        for pair, name in zip(JOINT_KEY_PAIRS, names, strict=True):
            print(f"  {pair[0]} / {pair[1]}: {name}", flush=True)
        print(
            "Press repeatedly: each press changes its target by 2 degrees.\n"
            "Holding a key does not repeat commands; command speed <= 0.8 rad/s.\n"
            "Backspace: home targets | N: new episode | Space: pause/resume | Esc: quit\n"
            "Joint commands typed while paused take effect after resuming.\n"
            "MuJoCo also receives these keys; some letters toggle rendering options,\n"
            "and Esc selects the free camera before exiting. Tab toggles its UI.\n"
            "Viewer reset, joint/control sliders, and mouse physics edits are ignored.\n"
            "Use plain keys without Ctrl/Alt; click the 3D view before controlling.",
            flush=True,
        )
        _print_targets(names, target)

        with mujoco.viewer.launch_passive(
            env.model,
            env.data,
            key_callback=on_key,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.17, 0.0, 0.17]
                viewer.cam.distance = 0.8
                viewer.cam.azimuth = 140.0
                viewer.cam.elevation = -25.0
            paused = False
            quit_requested = False
            start = time.monotonic()
            next_tick = start
            next_render = start
            viewer_state = np.empty(
                mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION)
            )
            while viewer.is_running() and not quit_requested:
                now = time.monotonic()
                if seconds is not None and now - start >= seconds:
                    break
                with viewer.lock():
                    while True:
                        try:
                            command = commands.get_nowait()
                        except Empty:
                            break
                        if command.kind == "quit":
                            quit_requested = True
                            break
                        if command.kind == "pause":
                            paused = not paused
                            print("Paused." if paused else "Resumed.", flush=True)
                        elif command.kind == "reset":
                            env.reset()
                            target = np.clip(env.home.copy(), limits[:, 0], limits[:, 1])
                            control = env.data.ctrl.copy()
                            print("New episode; robot reset to home.", flush=True)
                            _print_targets(names, target)
                        elif command.kind == "home":
                            target = np.clip(env.home.copy(), limits[:, 0], limits[:, 1])
                            _print_targets(names, target)
                        elif command.kind == "joint":
                            index = command.joint_index
                            target[index] = np.clip(
                                target[index] + command.direction * STEP_RADIANS,
                                limits[index, 0],
                                limits[index, 1],
                            )
                            _print_targets(names, target)
                    timestep = float(env.model.opt.timestep)
                    if not paused and not quit_requested:
                        control = slew_limited_control(control, target, limits, timestep)
                        env.data.ctrl[:] = control
                        mujoco.mj_step(env.model, env.data)
                if now >= next_render:
                    _sync_keyboard_view(viewer, env.model, env.data, viewer_state)
                    next_render = now + 1.0 / 60.0
                next_tick += timestep
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -0.1:
                    next_tick = time.monotonic()
    finally:
        env.close()


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be a finite positive number")
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seconds",
        type=_positive_seconds,
        help="close automatically after this many wall-clock seconds",
    )
    args = parser.parse_args()
    try:
        run(seconds=args.seconds)
    except KeyboardInterrupt:
        print("Keyboard control stopped.", flush=True)


if __name__ == "__main__":
    main()
