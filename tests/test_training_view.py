"""Test observation of real model data without training or opening a window."""

from contextlib import nullcontext, redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from so101_sim.env import SO101ReachEnv
from so101_sim.training_view import TrainingViewer


class FakeViewer:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.cam = SimpleNamespace(lookat=np.zeros(3))
        self.closed = False

    def lock(self):
        return nullcontext()

    def is_running(self):
        return not self.closed

    def sync(self):
        # A GUI user deliberately edits the observer physics.
        self.data.qpos[:] = 1
        self.model.site_pos[:] = 2
        self.model.opt.gravity[:] = 0

    def close(self):
        self.closed = True


class TrainingViewerTests(unittest.TestCase):
    def setUp(self):
        self.env = SO101ReachEnv()
        self.env.reset(seed=42)
        self.addCleanup(self.env.close)
        self.callback = TrainingViewer(progress=lambda: {"optimizer_steps": 1})
        self.callback.init_callback(SimpleNamespace(
            get_env=lambda: SimpleNamespace(envs=[self.env]), _total_timesteps=500,
        ))
        self.callback.locals = {"dones": [False], "infos": [{"distance": .05}]}
        self.output = io.StringIO()
        self.capture = redirect_stdout(self.output)
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)
        self.patch = patch("mujoco.viewer.launch_passive", side_effect=lambda m, d, **kw: FakeViewer(m, d))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.callback._on_training_start()
        self.addCleanup(self.callback.close)

    def test_viewer_edits_cannot_edit_training_or_take_actions(self):
        qpos = self.env.data.qpos.copy()
        targets = self.env.model.site_pos.copy()
        gravity = self.env.model.opt.gravity.copy()
        with patch.object(self.env, "step", side_effect=AssertionError("Observer took an action")), patch("time.sleep"):
            self.assertTrue(self.callback._on_step())
            self.assertTrue(self.callback._on_step())
        np.testing.assert_array_equal(self.env.data.qpos, qpos)
        np.testing.assert_array_equal(self.env.model.site_pos, targets)
        np.testing.assert_array_equal(self.env.model.opt.gravity, gravity)
        self.assertEqual(self.env.data.time, 0)
        self.assertIn("network updates: 1", self.output.getvalue())

    def test_close_window_interrupts_training(self):
        self.callback.viewer.close()
        with self.assertRaises(KeyboardInterrupt):
            self.callback._on_step()

    def test_continuous_status_displays_an_unlimited_target(self):
        self.callback.model._total_timesteps = float("inf")
        self.callback.num_timesteps = 600_000
        with patch.object(self.env, "step", side_effect=AssertionError("Observer took an action")), patch("time.sleep"):
            self.assertTrue(self.callback._on_step())
        self.assertIn("Training steps: 600,000/unlimited", self.output.getvalue())

    def test_status_reports_target_age_without_advancing_training(self):
        self.callback.locals["infos"] = [{
            "distance": 0.05,
            "target_steps": 100,
            "target_min_steps": 1200,
            "target_elapsed_seconds": 5.0,
            "target_duration_seconds": 60.0,
        }]
        qpos = self.env.data.qpos.copy()
        with patch.object(self.env, "step", side_effect=AssertionError("Observer took an action")), patch("time.sleep"):
            self.assertTrue(self.callback._on_step())
        self.assertIn("target: 5.0/60.0 sim s (100/1200 steps)", self.output.getvalue())
        np.testing.assert_array_equal(self.env.data.qpos, qpos)
        self.assertEqual(self.env.data.time, 0)

    def test_end_closes_window_and_reports_actual_update_count(self):
        viewer = self.callback.viewer
        self.callback._on_rollout_end()
        self.callback._on_training_end()
        self.assertTrue(viewer.closed)
        self.assertIn("updating the network", self.output.getvalue())
        self.assertIn("Network updates completed: 1", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
