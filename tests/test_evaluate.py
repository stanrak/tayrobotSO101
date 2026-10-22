"""Checkpoint compatibility and reproducible playback checks without training."""

from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from stable_baselines3 import PPO
import torch

from so101_sim.env import SO101ReachEnv
from so101_sim.evaluate import (
    environment_signature,
    load_checked_policy,
    metadata_path,
    run_episodes,
    save_checkpoint,
)


class InterferingViewer:
    """Simulate a native reset and Physics-panel edits during a viewer sync."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.cam = SimpleNamespace(lookat=np.zeros(3))
        self.did_interfere = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def lock(self):
        return nullcontext()

    def is_running(self):
        return True

    def sync(self, state_only=False):
        if not state_only and not self.did_interfere:
            mujoco.mj_resetData(self.model, self.data)
            self.model.opt.gravity[:] = 0
            self.model.opt.timestep = 0.02
            mujoco.mj_forward(self.model, self.data)
            self.did_interfere = True


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env = SO101ReachEnv()
        cls.model = PPO(
            "MlpPolicy", cls.env, n_steps=256, batch_size=256,
            policy_kwargs={"net_arch": [64, 64]}, seed=42, device="cpu",
        )
        cls.signature = environment_signature(cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="so101-evaluation-test-")
        self.addCleanup(self.temporary.cleanup)
        self.checkpoint = Path(self.temporary.name) / "untrained.zip"
        save_checkpoint(self.model, self.checkpoint, self.signature, {"actual_timesteps": 0})

    def test_checkpoint_roundtrip_preserves_predictions(self):
        restored, metadata = load_checked_policy(self.checkpoint, self.env)
        observation, _ = self.env.reset(seed=123)
        original_action, _ = self.model.predict(observation, deterministic=True)
        restored_action, _ = restored.predict(observation, deterministic=True)
        np.testing.assert_array_equal(original_action, restored_action)
        self.assertEqual(metadata["training"]["actual_timesteps"], 0)

    def test_changed_task_metadata_is_rejected(self):
        path = metadata_path(self.checkpoint)
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["environment"]["task_config"]["hold_required"] = 999
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "environment mismatch: task_config"):
            load_checked_policy(self.checkpoint, self.env)

    def test_changed_package_metadata_is_rejected(self):
        path = metadata_path(self.checkpoint)
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["package_versions"]["mujoco"] = "incompatible"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "package versions differ"):
            load_checked_policy(self.checkpoint, self.env)

    def test_changed_checkpoint_contents_are_rejected(self):
        with self.checkpoint.open("ab") as handle:
            handle.write(b"modified")
        with self.assertRaisesRegex(ValueError, "metadata SHA256"):
            load_checked_policy(self.checkpoint, self.env)

    def test_viewer_physics_edits_cannot_change_episode_metrics(self):
        expected = run_episodes(self.env, [20_000], model=self.model)
        viewers = []

        def launch(model, data):
            viewer = InterferingViewer(model, data)
            viewers.append(viewer)
            return viewer

        with (
            patch("mujoco.viewer.launch_passive", side_effect=launch),
            patch("so101_sim.evaluate.time.sleep"),
        ):
            actual = run_episodes(self.env, [20_000], model=self.model, render=True)
        self.assertTrue(viewers[0].did_interfere)
        self.assertEqual(actual, expected)
        np.testing.assert_array_equal(self.env.model.opt.gravity, [0, 0, -9.81])
        self.assertEqual(self.env.model.opt.timestep, 0.005)

    def test_timeout_and_collision_are_separate_reported_causes(self):
        baseline = run_episodes(self.env, [20_000], policy="no_motion")
        self.assertEqual(baseline["summary"]["time_limit_episodes"], 1)
        self.assertEqual(baseline["summary"]["collision_episodes"], 0)
        self.assertEqual(baseline["episodes"][0]["termination_reason"], "time_limit")
        collision_env = SimpleNamespace(
            reset=lambda seed: (np.zeros(15), {"distance": 0.05}),
            neutral_action=lambda: np.zeros(3),
            step=lambda action: (
                np.zeros(15), -10.0, True, False,
                {"distance": 0.05, "hold_steps": 0, "is_success": False, "collision": True},
            ),
        )
        collision = run_episodes(collision_env, [20_000], policy="no_motion")
        self.assertEqual(collision["summary"]["time_limit_episodes"], 0)
        self.assertEqual(collision["summary"]["collision_episodes"], 1)
        self.assertEqual(collision["episodes"][0]["termination_reason"], "collision")


if __name__ == "__main__":
    unittest.main()
