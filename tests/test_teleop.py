"""Headless keyboard checks, including the real MuJoCo physics loop."""

from contextlib import nullcontext, redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from so101_sim.env import SO101ReachEnv
from so101_sim.teleop import (
    JOINT_KEY_PAIRS,
    KeyCommand,
    _sync_keyboard_view,
    key_to_command,
    run,
    slew_limited_control,
)


class FakeViewer:
    """Deliver focused-window key events without opening a graphics window."""

    def __init__(self, model, data, callback=None, key_batches=(), native_reset=False, physics_edit=False):
        self.model = model
        self.data = data
        self.callback = callback
        self.key_batches = list(key_batches)
        self.native_reset = native_reset
        self.physics_edit = physics_edit
        self.initial_qpos = data.qpos.copy()
        self.initial_ctrl = data.ctrl.copy()
        self.times = []
        self.controls = []
        self.cam = SimpleNamespace(lookat=np.zeros(3))

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def lock(self):
        return nullcontext()

    def is_running(self):
        return True

    def sync(self, state_only=False):
        if state_only:
            return
        self.times.append(float(self.data.time))
        self.controls.append(self.data.ctrl.copy())
        if self.native_reset:
            mujoco.mj_resetData(self.model, self.data)
            self.native_reset = False
        if self.physics_edit:
            self.model.opt.gravity[:] = 0
            self.model.opt.timestep = 0.02
            self.physics_edit = False
        if self.key_batches:
            for key in self.key_batches.pop(0):
                self.callback(key)


class TeleopTests(unittest.TestCase):
    def test_all_joints_and_special_keys(self):
        for index, pair in enumerate(JOINT_KEY_PAIRS):
            for key, direction in zip(pair, (1, -1), strict=True):
                self.assertEqual(key_to_command(ord(key)), KeyCommand("joint", index, direction))
                self.assertEqual(key_to_command(ord(key.lower())), KeyCommand("joint", index, direction))
        for key, kind in ((32, "pause"), (259, "home"), (ord("N"), "reset"), (256, "quit")):
            self.assertEqual(key_to_command(key).kind, kind)
        self.assertIsNone(key_to_command(0))
        self.assertIsNone(key_to_command(ord("Z")))

    def test_slew_and_joint_limits(self):
        limits = np.tile([-0.2, 0.2], (6, 1))
        target = np.array([1.0, -1.0, 0.001, -0.001, 0.0, 0.0])
        current = np.zeros(6)
        next_control = slew_limited_control(current, target, limits, 0.005)
        np.testing.assert_allclose(next_control, [0.004, -0.004, 0.001, -0.001, 0.0, 0.0])
        np.testing.assert_array_equal(current, np.zeros(6))
        saturated = slew_limited_control(current, target, limits, 1.0)
        self.assertEqual(saturated[0], 0.2)
        self.assertEqual(saturated[1], -0.2)

    def _run_with_keys(self, batches, seconds=0.05):
        env = SO101ReachEnv()
        viewers = []

        def launch(model, data, *, key_callback, **_):
            viewer = FakeViewer(model, data, key_callback, batches)
            viewers.append(viewer)
            return viewer

        with (
            patch("so101_sim.env.SO101ReachEnv", return_value=env),
            patch("so101_sim.teleop.mujoco.viewer.launch_passive", side_effect=launch),
            redirect_stdout(io.StringIO()),
        ):
            run(seconds=seconds)
        return env, viewers[0]

    def test_headless_keys_move_real_physics(self):
        env, viewer = self._run_with_keys([[ord("Q"), ord("Y")]])
        self.addCleanup(env.close)
        self.assertGreater(env.data.ctrl[0], viewer.initial_ctrl[0])
        self.assertGreater(env.data.ctrl[5], viewer.initial_ctrl[5])
        self.assertGreater(np.linalg.norm(env.data.qpos[:6] - viewer.initial_qpos[:6]), 1e-6)
        self.assertTrue(np.all(env.data.ctrl >= env.ctrl_limits[:, 0]))
        self.assertTrue(np.all(env.data.ctrl <= env.ctrl_limits[:, 1]))
        self.assertTrue(np.all(env.data.qpos[:6] >= env.joint_limits[:, 0] - 1e-4))
        self.assertTrue(np.all(env.data.qpos[:6] <= env.joint_limits[:, 1] + 1e-4))
        self.assertGreater(env.data.time, 0.0)

    def test_pause_stops_physics_and_resume_applies_queued_targets(self):
        env, viewer = self._run_with_keys([[32, ord("Q"), ord("Y")], [], [32]], seconds=0.09)
        self.addCleanup(env.close)
        self.assertGreaterEqual(len(viewer.times), 4)
        self.assertEqual(viewer.times[0], viewer.times[1])
        self.assertEqual(viewer.times[1], viewer.times[2])
        self.assertGreater(viewer.times[3], viewer.times[2])
        np.testing.assert_array_equal(viewer.controls[0], viewer.controls[2])
        self.assertGreater(env.data.ctrl[5], viewer.initial_ctrl[5])

    def test_new_episode_changes_goal(self):
        baseline = SO101ReachEnv()
        self.addCleanup(baseline.close)
        baseline.reset(seed=0)
        first_goal = baseline.goal.copy()
        env, _ = self._run_with_keys([[ord("N")]])
        self.addCleanup(env.close)
        self.assertFalse(np.array_equal(first_goal, env.goal))

    def test_builtin_backspace_reset_cannot_overwrite_keyboard_physics(self):
        env = SO101ReachEnv()
        self.addCleanup(env.close)
        env.reset(seed=0)
        mujoco.mj_step(env.model, env.data)
        state_size = mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        expected = np.empty(state_size)
        actual = np.empty(state_size)
        mujoco.mj_getState(env.model, env.data, expected, mujoco.mjtState.mjSTATE_INTEGRATION)
        viewer = FakeViewer(env.model, env.data, native_reset=True, physics_edit=True)
        _sync_keyboard_view(viewer, env.model, env.data, np.empty(state_size))
        mujoco.mj_getState(env.model, env.data, actual, mujoco.mjtState.mjSTATE_INTEGRATION)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(env.model.opt.gravity, [0, 0, -9.81])
        self.assertEqual(env.model.opt.timestep, 0.001)


if __name__ == "__main__":
    unittest.main()
