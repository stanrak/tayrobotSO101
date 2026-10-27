"""Checkpoint compatibility and reproducible playback checks without training."""

from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
import re
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
    evaluate_checkpoint,
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


class ProgressEnvironment:
    """Finite fake episodes with a clock advanced only by control steps."""

    def __init__(self, horizon=4, clock=None, fail_on_step=None):
        self.max_steps = horizon
        self.clock = clock
        self.fail_on_step = fail_on_step
        self.action_space = SimpleNamespace(
            low=np.full(3, -1, dtype=np.float32), high=np.ones(3, dtype=np.float32),
        )
        self.seeds = []
        self.actions = []
        self.steps = 0
        self.closed = False

    def reset(self, *, seed):
        self.seeds.append(seed)
        self.steps = 0
        return np.array([seed, 0], dtype=np.float32), {"distance": 0.05}

    def neutral_action(self):
        return np.array([-0.25, 0.5, -0.75], dtype=np.float32)

    def step(self, action):
        if self.steps + 1 == self.fail_on_step:
            raise RuntimeError("fake control failure")
        self.steps += 1
        self.actions.append(np.asarray(action).copy())
        if self.clock is not None:
            self.clock.now += self.clock.seconds_per_step
        return (
            np.array([self.seeds[-1], self.steps], dtype=np.float32), -1.0,
            False, self.steps >= self.max_steps,
            {"distance": 0.05 - min(self.steps, 10) * 0.001, "hold_steps": 0,
             "is_success": False, "collision": False},
        )

    def close(self):
        self.closed = True


class ProgressModel:
    def __init__(self):
        self.observations = []

    def predict(self, observation, *, deterministic):
        self.observations.append((np.asarray(observation).copy(), deterministic))
        return np.array([0.25, -0.5, 0.75], dtype=np.float32), None


class EvaluationProgressTests(unittest.TestCase):
    def test_step_progress_preserves_episode_reports_actions_and_seeds(self):
        expected_env, actual_env = ProgressEnvironment(3), ProgressEnvironment(3)
        expected_model, actual_model = ProgressModel(), ProgressModel()
        expected = run_episodes(expected_env, [91, 92], model=expected_model)
        events = []
        actual = run_episodes(
            actual_env, [91, 92], model=actual_model,
            progress=lambda completed, steps: events.append((completed, steps)),
        )
        self.assertEqual(events, [
            (0, 0), (0, 1), (0, 2), (0, 3), (1, 3),
            (1, 4), (1, 5), (1, 6), (2, 6),
        ])
        self.assertEqual(actual, expected)
        self.assertEqual(actual_env.seeds, [91, 92])
        np.testing.assert_array_equal(actual_env.actions, expected_env.actions)
        for actual_prediction, expected_prediction in zip(actual_model.observations, expected_model.observations):
            np.testing.assert_array_equal(actual_prediction[0], expected_prediction[0])
            self.assertTrue(actual_prediction[1])

    def test_checkpoint_reports_heartbeat_and_all_policy_totals(self):
        clock = SimpleNamespace(now=0.0, seconds_per_step=0.05)
        environment = ProgressEnvironment(1200, clock=clock)
        model = ProgressModel()
        metadata = {"checkpoint_sha256": "fake-sha256", "environment": {"fake": True}}
        output = io.StringIO()
        with (
            patch("so101_sim.evaluate.SO101ReachEnv", return_value=environment),
            patch("so101_sim.evaluate.load_checked_policy", return_value=(model, metadata)),
            patch("so101_sim.evaluate.package_versions", return_value={"fake": "1"}),
            patch("so101_sim.evaluate.time.perf_counter", side_effect=lambda: clock.now),
            patch("so101_sim.evaluate.time.monotonic", side_effect=lambda: clock.now),
            redirect_stdout(output),
        ):
            report = evaluate_checkpoint(Path("fake.zip"), episodes=2, seed=91)
        lines = output.getvalue().splitlines()
        self.assertIn("Progress is reported every 30 seconds", output.getvalue())
        heartbeat_steps = []
        heartbeat_elapsed = []
        for line in lines:
            if "Evaluation trained: 0/2 episodes" in line:
                match = re.search(r"([\d,]+) control steps", line)
                if match is not None:
                    heartbeat_steps.append(int(match.group(1).replace(",", "")))
                    heartbeat_elapsed.append(float(re.search(r"elapsed ([\d.]+)s", line).group(1)))
        self.assertIn(0, heartbeat_steps)
        self.assertTrue(any(0 < steps < 1200 for steps in heartbeat_steps))
        self.assertAlmostEqual(heartbeat_elapsed[1], 30.0, delta=0.1)
        self.assertLess(len(lines), 30)
        for index, policy in enumerate(("trained", "no_motion", "random")):
            completed = [line for line in lines if f"Evaluation {policy}: 2/2 episodes" in line]
            self.assertTrue(completed)
            self.assertIn("2,400 control steps", completed[-1])
            self.assertIn(f"overall {(index + 1) * 2}/6 episodes", completed[-1])
            result = report["results"][policy]
            self.assertEqual([episode["seed"] for episode in result["episodes"]], [91, 92])
            self.assertEqual(result["summary"]["mean_episode_steps"], 1200)
            self.assertEqual(result["summary"]["mean_return"], -1200)
            self.assertEqual(result["summary"]["time_limit_episodes"], 2)
        self.assertEqual(environment.seeds, [91, 92] * 3)
        self.assertEqual(report["evaluation_seed_start"], 91)
        self.assertEqual(report["evaluation_seed_end"], 92)
        self.assertEqual(list(report["results"]), ["trained", "no_motion", "random"])
        self.assertEqual(len(model.observations), 2400)
        np.testing.assert_array_equal(environment.actions[:2400], np.tile([0.25, -0.5, 0.75], (2400, 1)))
        np.testing.assert_array_equal(environment.actions[2400:4800], np.tile([-0.25, 0.5, -0.75], (2400, 1)))
        for episode_index, seed in enumerate((91, 92)):
            generator = np.random.default_rng(seed + 1_000_000)
            expected = np.asarray([
                generator.uniform(environment.action_space.low, environment.action_space.high).astype(np.float32)
                for _ in range(1200)
            ])
            start = 4800 + episode_index * 1200
            np.testing.assert_array_equal(environment.actions[start:start + 1200], expected)
        self.assertTrue(environment.closed)

    def test_checkpoint_closes_environment_after_control_failure(self):
        environment = ProgressEnvironment(fail_on_step=2)
        metadata = {"checkpoint_sha256": "fake-sha256", "environment": {"fake": True}}
        with (
            patch("so101_sim.evaluate.SO101ReachEnv", return_value=environment),
            patch("so101_sim.evaluate.load_checked_policy", return_value=(ProgressModel(), metadata)),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "fake control failure"):
                evaluate_checkpoint(Path("fake.zip"), episodes=2, seed=91)
        self.assertTrue(environment.closed)
        self.assertEqual(environment.seeds, [91])
        self.assertEqual(len(environment.actions), 1)


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

    def test_older_task_policy_is_rejected_without_migration(self):
        # Build a local old-task fixture; generated historical checkpoints are
        # deliberately excluded from Git and absent in a clean checkout.
        historical = self.checkpoint
        metadata = metadata_path(historical)
        old_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        old_metadata["environment"]["task_config"]["task"] = "SO101LocalReach-v1"
        metadata.write_text(json.dumps(old_metadata), encoding="utf-8")
        original_checkpoint = historical.read_bytes()
        original_metadata = metadata.read_bytes()
        with patch("so101_sim.evaluate.PPO.load") as load:
            with self.assertRaisesRegex(ValueError, "Checkpoint environment mismatch:"):
                load_checked_policy(historical, self.env)
            load.assert_not_called()
        self.assertEqual(historical.read_bytes(), original_checkpoint)
        self.assertEqual(metadata.read_bytes(), original_metadata)

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
        self.assertEqual(self.env.model.opt.timestep, 0.001)

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
