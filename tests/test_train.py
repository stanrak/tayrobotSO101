"""Verify training modes and budgets with fake environments and a fake PPO."""

from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import csv
import io
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np

from so101_sim import train
from so101_sim import outputs


class CountingEnvironment(gym.Env):
    """Count transitions, including those made through DummyVecEnv/Monitor."""

    instances = []

    def __init__(self):
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.step_calls = 0
        self.episode_steps = 0
        self.closed = False
        self.instances.append(self)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_steps = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.step_calls += 1
        self.episode_steps += 1
        return (
            np.zeros(3, dtype=np.float32), -1.0, False, self.episode_steps >= 7,
            {"is_success": False, "distance": 0.1, "hold_steps": 0},
        )

    def close(self):
        self.closed = True


class CountingOptimizer:
    """Record actual calls without creating a tensor or changing any weights."""

    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


class RolloutRoundingPPO:
    """Mimic SB3's complete-rollout rounding; never create or train a policy."""

    instances = []
    interrupt_after_rollouts = None
    fast_forward_at_interrupt = None
    early_stop_epochs = None

    def __init__(self, policy, env, **kwargs):
        self.env = env
        self.kwargs = kwargs
        self.num_timesteps = 0
        self._total_timesteps = 0
        self._n_updates = 0
        self.callback_calls = 0
        self.learn_budgets = []
        self.completed_rollouts = 0
        self.rollout_update_count = 0
        self.diagnostic_optimizer_steps = 0
        self.rollout_seconds = 0.0
        self.optimization_seconds = 0.0
        self.diagnostics_seconds = 0.0
        self.collected_rewards = []
        self.policy = SimpleNamespace(optimizer=CountingOptimizer())
        self.instances.append(self)

    def set_logger(self, logger):
        self.logger = logger

    def get_env(self):
        return self.env

    def learn(self, total_timesteps, callback=None, log_interval=None, reset_num_timesteps=True):
        self.callback = callback
        self.learn_budgets.append(total_timesteps)
        self.reset_num_timesteps = reset_num_timesteps
        if not reset_num_timesteps:
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self.env.reset()
        if callback is not None:
            callback.init_callback(self)
            callback.on_training_start({}, {})
        # This intentionally rounds up exactly like OnPolicyAlgorithm.learn.
        # A default 256*8 rollout would expose an accidental 2048-step trial.
        while self.num_timesteps < total_timesteps:
            if callback is not None:
                callback.on_rollout_start()
            for _ in range(self.kwargs["n_steps"]):
                _, rewards, _, infos = self.env.step(np.zeros((self.env.num_envs, 3), dtype=np.float32))
                self.collected_rewards.extend(rewards.tolist())
                self.num_timesteps += self.env.num_envs
                if callback is not None:
                    self.callback_calls += 1
                    callback.update_locals({"infos": infos})
                    if not callback.on_step():
                        # Like SB3, discard the partial rollout when a callback
                        # stops collection; do not optimize incomplete data.
                        callback.on_training_end()
                        return self
            if callback is not None:
                callback.on_rollout_end()
            batch_count = math.ceil(self.kwargs["n_steps"] * self.env.num_envs / self.kwargs["batch_size"])
            epochs = self.kwargs["n_epochs"] if self.early_stop_epochs is None else self.early_stop_epochs
            for _ in range(epochs):
                for _ in range(batch_count):
                    self.policy.optimizer.step()
                    self.diagnostic_optimizer_steps += 1
                self._n_updates += 1
            self.completed_rollouts += 1
            self.rollout_update_count += 1
            if self.completed_rollouts == self.interrupt_after_rollouts:
                if self.fast_forward_at_interrupt is not None:
                    # Simulate a much later user interruption cheaply. No real
                    # policy, environment rollout or optimizer is involved.
                    self.num_timesteps = self.fast_forward_at_interrupt
                raise KeyboardInterrupt
        if callback is not None:
            callback.on_training_end()
        return self


class StartupTrialTests(unittest.TestCase):
    def setUp(self):
        CountingEnvironment.instances = []
        RolloutRoundingPPO.instances = []
        self.temporary = TemporaryDirectory(prefix="so101-startup-test-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "trial"
        output_root = patch.object(outputs, "RUNS_ROOT", Path(self.temporary.name) / "runs")
        output_root.start()
        self.addCleanup(output_root.stop)

    def test_training_output_receives_run_mode_and_seed(self):
        for options, mode, seed in (
            (["--trial", "--seed", "7"], "startup_trial", 7),
            ([], "full_training", 42),
            (["--full-training", "--seed", "11"], "full_training", 11),
        ):
            with (
                self.subTest(options=options),
                patch.object(train, "training_output") as destination,
                patch.object(train, "_run") as run,
            ):
                destination.return_value.__enter__.return_value = self.output
                train.main(options)
                destination.assert_called_once_with(None, mode=mode, seed=seed)
                self.assertEqual(run.call_args.args[1], self.output)

    def run_fake_trial(self, *options, managed=False):
        def inspect_checkpoint(model, *args):
            self.assertIs(model.policy.optimizer.step.__func__, CountingOptimizer.step)

        with ExitStack() as stack:
            stack.enter_context(patch.object(train, "SO101ReachEnv", CountingEnvironment))
            stack.enter_context(patch.object(train, "PPO", RolloutRoundingPPO))
            stack.enter_context(patch.object(train, "environment_signature", return_value={"fake": True}))
            stack.enter_context(patch.object(train, "configure", return_value=MagicMock()))
            checkpoint = stack.enter_context(patch.object(train, "save_checkpoint", side_effect=inspect_checkpoint))
            validation = stack.enter_context(patch.object(train, "FixedSeedEvaluation"))
            evaluate = stack.enter_context(patch.object(train, "evaluate_checkpoint"))
            episodes = stack.enter_context(patch.object(train, "run_episodes"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            output_options = [] if managed else ["--output", str(self.output)]
            train.main(["--trial", *output_options, *options])
        return checkpoint, validation, evaluate, episodes

    @contextmanager
    def fake_full_training(self):
        """Patch every source of policy training, evaluation and GUI work."""
        summary = {"successes": 0, "episodes": 1, "mean_final_distance_m": 0.1}
        with ExitStack() as stack:
            stack.enter_context(patch.object(train, "SO101ReachEnv", CountingEnvironment))
            stack.enter_context(patch.object(train, "PPO", RolloutRoundingPPO))
            stack.enter_context(patch.object(train, "environment_signature", return_value={"fake": True}))
            stack.enter_context(patch.object(train, "configure", return_value=MagicMock()))
            checkpoint = stack.enter_context(patch.object(train, "save_checkpoint"))
            validation = stack.enter_context(patch.object(train, "FixedSeedEvaluation"))
            validation.return_value.training_metadata.return_value = {"validation_marker": True}
            validation.return_value.validation_seconds = 0.0
            validation.return_value.validation_episodes = 0
            validation.return_value.validation_steps = 0
            evaluate = stack.enter_context(patch.object(
                train, "evaluate_checkpoint", return_value={"results": {"trained": {"summary": summary}}},
            ))
            episodes = stack.enter_context(patch.object(train, "run_episodes"))
            viewer = stack.enter_context(patch("so101_sim.training_view.TrainingViewer"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            yield SimpleNamespace(
                checkpoint=checkpoint, validation=validation, evaluate=evaluate,
                episodes=episodes, viewer=viewer,
            )

    def resume_record(self, **changes):
        training = {
            "mode": "full_training", **train.LEARNING_DEFAULTS, "reward_scale": 0.01,
            "n_steps": 256, "batch_size": 256, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_range": 0.2, "ent_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5,
            "policy_net_arch": [64, 64], "observation_normalization": False,
            "reward_normalization": False, **changes,
        }
        return SimpleNamespace(
            path=Path(self.temporary.name) / "parent" / "final_model.zip",
            metadata={"checkpoint_sha256": "parent-digest"}, training=training,
            counters={"timesteps": 300_000, "optimizer_steps": 11_680,
                      "update_count": 146, "optimization_epochs": 1460},
        )

    @contextmanager
    def fake_resume_training(self, record=None):
        record = record or self.resume_record()

        def restore(checkpoint, vector, *, seed):
            self.assertIs(checkpoint, record)
            model = RolloutRoundingPPO("MlpPolicy", vector, **record.training)
            model.num_timesteps = record.counters["timesteps"]
            model._n_updates = record.counters["optimization_epochs"]
            model.rollout_update_count = record.counters["update_count"]
            model.diagnostic_optimizer_steps = record.counters["optimizer_steps"]
            model.policy.optimizer.step_calls = record.counters["optimizer_steps"]
            return model

        with (
            self.fake_full_training() as fake,
            patch.object(train, "inspect_resume_checkpoint", return_value=record) as inspect,
            patch.object(train, "load_resume_policy", side_effect=restore) as load,
            patch.object(train, "PPO", side_effect=AssertionError("Resume initialized a fresh policy")) as fresh,
        ):
            # Include the actual progress callback in saved checkpoint metadata.
            fake.validation.return_value.training_metadata.side_effect = lambda: {
                **fake.validation.call_args.args[2],
                **fake.validation.call_args.kwargs["progress"](),
            }
            yield SimpleNamespace(record=record, fake=fake, inspect=inspect, load=load, fresh=fresh)

    def test_default_trial_has_exactly_500_total_transitions_and_one_update(self):
        checkpoint, validation, evaluate, episodes = self.run_fake_trial()
        self.assertEqual(sum(env.step_calls for env in CountingEnvironment.instances), 500)
        self.assertTrue(all(env.closed for env in CountingEnvironment.instances))
        model = RolloutRoundingPPO.instances[0]
        self.assertEqual(model.env.num_envs, 1)
        model.logger.close.assert_called_once()
        self.assertEqual(model.kwargs["n_steps"], 500)
        self.assertEqual(model.callback_calls, 0)
        self.assertEqual(model._n_updates, 1)
        self.assertEqual(model.kwargs["n_epochs"], 1)
        self.assertEqual(model.policy.optimizer.step_calls, 1)
        self.assertIs(model.policy.optimizer.step.__func__, CountingOptimizer.step)
        validation.assert_not_called()
        evaluate.assert_not_called()
        episodes.assert_not_called()
        checkpoint.assert_called_once()
        self.assertEqual(checkpoint.call_args.args[1].name, "trial_model.zip")
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "trial_completed")
        self.assertEqual(report["mode"], "startup_trial")
        self.assertEqual(report["actual_timesteps"], 500)
        self.assertEqual(report["update_count"], 1)
        self.assertEqual(report["optimization_epochs"], 1)
        self.assertEqual(report["gradient_update_count"], 1)
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["trial_stop_criterion"], "first_optimizer_step")
        self.assertEqual(report["trial_stop_reason"], "first_optimizer_step_completed")
        self.assertEqual(report["selected_rollout_budget"], 500)
        self.assertEqual(report["trial_status"], "first_network_update_completed")
        self.assertFalse(report["task_performance_evaluated"])
        self.assertFalse(report["config"]["checkpoint_loaded"])
        self.assertEqual(report["config"]["validation_episodes"], 0)
        self.assertEqual(report["config"]["heldout_episodes"], 0)
        self.assertFalse((self.output / "evaluation.json").exists())

    def test_smaller_trial_does_not_round_up_or_evaluate_at_frequency_one(self):
        _, validation, evaluate, episodes = self.run_fake_trial(
            "--timesteps", "23", "--eval-frequency", "1", "--envs", "1",
        )
        self.assertEqual(sum(env.step_calls for env in CountingEnvironment.instances), 23)
        self.assertEqual(RolloutRoundingPPO.instances[0].kwargs["batch_size"], 23)
        validation.assert_not_called()
        evaluate.assert_not_called()
        episodes.assert_not_called()

    def test_default_runs_have_unique_paths_and_keep_prior_results(self):
        with patch.object(outputs, "RUNS_ROOT", Path(self.temporary.name) / "managed"):
            self.run_fake_trial("--timesteps", "23", managed=True)
            first_output = outputs.current_output()
            first_report = (first_output / "training.json").read_bytes()
            self.run_fake_trial("--timesteps", "37", "--seed", "7", managed=True)
            second_output = outputs.current_output()
            self.assertNotEqual(first_output, second_output)
            self.assertEqual(first_output.parent, outputs.RUNS_ROOT / "experiments")
            self.assertEqual(second_output.parent, first_output.parent)
            current = json.loads((second_output / "training.json").read_text())
            self.assertEqual(current["actual_timesteps"], 37)
            self.assertEqual(current["config"]["seed"], 7)
            self.assertEqual((first_output / "training.json").read_bytes(), first_report)
            self.assertTrue((outputs.RUNS_ROOT / "latest.json").is_file())
            self.assertFalse((outputs.RUNS_ROOT / "history").exists())

    def test_two_step_minimum_budget_still_completes_one_network_update(self):
        self.run_fake_trial("--timesteps", "2")
        model = RolloutRoundingPPO.instances[0]
        self.assertEqual(sum(env.step_calls for env in CountingEnvironment.instances), 2)
        self.assertEqual(model.policy.optimizer.step_calls, 1)

    def test_task_specific_trial_can_exceed_500_steps_with_one_network_update(self):
        _, validation, evaluate, episodes = self.run_fake_trial("--timesteps", "2048")
        model = RolloutRoundingPPO.instances[0]
        self.assertEqual(sum(env.step_calls for env in CountingEnvironment.instances), 2048)
        self.assertEqual(model.kwargs["n_steps"], 2048)
        self.assertEqual(model.kwargs["batch_size"], 2048)
        self.assertEqual(model.kwargs["n_epochs"], 1)
        self.assertEqual(model.policy.optimizer.step_calls, 1)
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["selected_rollout_budget"], 2048)
        self.assertEqual(report["config"]["max_gradient_updates"], 1)
        self.assertEqual(report["gradient_update_count"], 1)
        self.assertEqual(report["optimizer_steps"], 1)
        validation.assert_not_called()
        evaluate.assert_not_called()
        episodes.assert_not_called()

    def test_second_optimizer_step_is_blocked_before_any_weight_update(self):
        learn_once = RolloutRoundingPPO.learn

        def attempt_second_update(model, *args, **kwargs):
            learn_once(model, *args, **kwargs)
            model.policy.optimizer.step()

        with patch.object(RolloutRoundingPPO, "learn", attempt_second_update):
            with self.assertRaisesRegex(RuntimeError, "second optimizer step"):
                self.run_fake_trial("--timesteps", "23")
        model = RolloutRoundingPPO.instances[0]
        self.assertEqual(model.policy.optimizer.step_calls, 1)
        self.assertIs(model.policy.optimizer.step.__func__, CountingOptimizer.step)
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertEqual(report["gradient_update_count"], 1)
        self.assertEqual(report["actual_timesteps"], 23)
        self.assertNotIn("checkpoint", report)
        self.assertTrue(all(env.closed for env in CountingEnvironment.instances))

    def test_trial_cannot_report_completion_without_actual_optimizer_step(self):
        def skip_update(model, total_timesteps, callback=None, log_interval=None):
            model.num_timesteps = total_timesteps
            model._n_updates = 1
            return model

        with patch.object(RolloutRoundingPPO, "learn", skip_update):
            with self.assertRaisesRegex(RuntimeError, "first optimizer step"):
                self.run_fake_trial()
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["gradient_update_count"], 0)

    def test_invalid_budgets_and_vectorization_are_rejected_before_any_work(self):
        for options in (
            ["--timesteps", "1"], ["--timesteps", "0"], ["--timesteps", "-1"],
            ["--envs", "8"], ["--envs", "0"],
        ):
            with (
                self.subTest(options=options),
                patch.object(train, "SO101ReachEnv") as environment,
                patch.object(train, "PPO") as policy,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as caught:
                    train.main(["--trial", "--output", str(self.output), *options])
                self.assertEqual(caught.exception.code, 2)
                self.assertFalse(self.output.exists())
                environment.assert_not_called()
                policy.assert_not_called()

    def test_interrupted_trial_records_only_observed_steps_and_no_update(self):
        def interrupt(model, total_timesteps, callback=None, log_interval=None):
            self.assertIsNone(callback)
            model.env.reset()
            for _ in range(3):
                model.env.step(np.zeros((model.env.num_envs, 3), dtype=np.float32))
                model.num_timesteps += model.env.num_envs
            raise KeyboardInterrupt

        with patch.object(RolloutRoundingPPO, "learn", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_fake_trial()
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["mode"], "startup_trial")
        self.assertEqual(report["trial_status"], "interrupted")
        self.assertEqual(report["actual_timesteps"], 3)
        self.assertEqual(report["update_count"], 0)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertEqual(report["gradient_update_count"], 0)
        self.assertFalse(report["task_performance_evaluated"])
        self.assertEqual(sum(env.step_calls for env in CountingEnvironment.instances), 3)
        self.assertTrue(all(env.closed for env in CountingEnvironment.instances))

    def test_viewer_trial_still_performs_only_one_network_update(self):
        with patch("so101_sim.training_view.TrainingViewer") as viewer:
            _, validation, evaluation, episodes = self.run_fake_trial("--viewer", "--timesteps", "23")
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["optimizer_steps"], 1)
        self.assertTrue(report["config"]["live_viewer"])
        viewer.return_value.close.assert_called_once()
        validation.assert_not_called()
        evaluation.assert_not_called()
        episodes.assert_not_called()

    def test_full_viewer_keeps_validation_checkpoint_metadata_and_cleanup(self):
        with self.fake_full_training() as fake:
            train.main(["--viewer", "--timesteps", "501", "--output", str(self.output)])
        model = RolloutRoundingPPO.instances[0]
        self.assertIn(fake.viewer.return_value, model.callback.callbacks)
        self.assertIn(fake.validation.return_value, model.callback.callbacks)
        self.assertIs(
            fake.checkpoint.call_args.args[3],
            fake.validation.return_value.training_metadata.return_value,
        )
        fake.validation.return_value.eval_env.close.assert_called_once()
        fake.viewer.return_value.close.assert_called_once()
        self.assertEqual(model.num_timesteps, 501)

    def test_legacy_full_training_alias_keeps_rollout_and_validation_defaults(self):
        with self.fake_full_training() as fake:
            train.main(["--full-training", "--timesteps", "2048", "--output", str(self.output)])
        model = RolloutRoundingPPO.instances[0]
        self.assertEqual(model.env.num_envs, 8)
        self.assertEqual(model.kwargs["n_steps"], 256)
        self.assertEqual(model.kwargs["batch_size"], 256)
        self.assertEqual(model.kwargs["n_epochs"], 10)
        self.assertEqual(model.policy.optimizer.step_calls, 80)
        self.assertEqual(sum(env.step_calls for env in CountingEnvironment.instances), 2048)
        fake.validation.assert_called_once()
        fake.evaluate.assert_called_once_with(self.output / "best_model.zip", episodes=100, seed=20_000)
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["mode"], "full_training")
        self.assertEqual(report["actual_timesteps"], 2048)
        self.assertEqual(report["optimizer_steps"], 80)
        self.assertIsNone(report["config"]["max_gradient_updates"])

    def test_default_training_is_unlimited_and_saves_when_user_interrupts(self):
        with (
            self.fake_full_training() as fake,
            patch.object(RolloutRoundingPPO, "interrupt_after_rollouts", 2),
            patch.object(RolloutRoundingPPO, "fast_forward_at_interrupt", 600_000),
        ):
            with self.assertRaises(KeyboardInterrupt):
                train.main(["--output", str(self.output)])
        model = RolloutRoundingPPO.instances[0]
        self.assertEqual(model.learn_budgets, [math.inf])
        self.assertEqual(model.completed_rollouts, 2)
        self.assertEqual(model.env.num_envs, 8)
        self.assertEqual(model.policy.optimizer.step_calls, 160)
        self.assertIs(model.policy.optimizer.step.__func__, CountingOptimizer.step)
        fake.checkpoint.assert_called_once()
        self.assertEqual(fake.checkpoint.call_args.args[1], self.output / "final_model.zip")
        metadata = fake.checkpoint.call_args.args[3]
        self.assertEqual(metadata["actual_timesteps"], 600_000)
        self.assertEqual(metadata["optimizer_steps"], 160)
        self.assertEqual(metadata["stop_reason"], "user_interrupt")
        fake.evaluate.assert_not_called()
        fake.episodes.assert_not_called()
        fake.viewer.assert_not_called()
        fake.validation.return_value.eval_env.close.assert_called_once()
        model.logger.close.assert_called_once()
        self.assertTrue(all(env.closed for env in CountingEnvironment.instances))
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["mode"], "full_training")
        self.assertEqual(report["actual_timesteps"], 600_000)
        self.assertEqual(report["update_count"], 2)
        self.assertEqual(report["optimizer_steps"], 160)
        self.assertEqual(report["stop_reason"], "user_interrupt")
        self.assertEqual(report["checkpoint"], "final_model.zip")
        self.assertIsNone(report["config"]["requested_timesteps"])
        self.assertIsNone(report["config"]["environment_step_limit"])
        self.assertFalse(report["task_performance_evaluated"])
        self.assertFalse((self.output / "evaluation.json").exists())

    def test_finite_default_training_stops_at_exact_cap_without_rollout_rounding(self):
        with self.fake_full_training() as fake:
            train.main(["--timesteps", "501", "--output", str(self.output)])
        model = RolloutRoundingPPO.instances[0]
        self.assertEqual(model.learn_budgets, [501])
        self.assertEqual(model.env.num_envs, 1)
        self.assertEqual(model.kwargs["n_steps"], 256)
        self.assertEqual(model.kwargs["n_epochs"], 10)
        self.assertEqual(model.num_timesteps, 501)
        self.assertEqual(sum(env.step_calls for env in CountingEnvironment.instances), 501)
        self.assertEqual(model.completed_rollouts, 1)
        self.assertEqual(model.policy.optimizer.step_calls, 10)
        fake.evaluate.assert_called_once()
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["actual_timesteps"], 501)
        self.assertEqual(report["update_count"], 1)
        self.assertEqual(report["config"]["environment_step_limit"], 501)

    def test_finite_cap_at_rollout_boundary_keeps_the_last_complete_update(self):
        with self.fake_full_training():
            train.main(["--timesteps", "512", "--envs", "1", "--output", str(self.output)])
        model = RolloutRoundingPPO.instances[0]
        self.assertEqual(model.num_timesteps, 512)
        self.assertEqual(model.completed_rollouts, 2)
        self.assertEqual(model.policy.optimizer.step_calls, 20)
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["update_count"], 2)
        self.assertEqual(report["optimizer_steps"], 20)

    def test_one_step_full_budget_is_allowed_without_optimizing_partial_data(self):
        with self.fake_full_training():
            train.main(["--timesteps", "1", "--output", str(self.output)])
        model = RolloutRoundingPPO.instances[0]
        self.assertEqual(model.num_timesteps, 1)
        self.assertEqual(model.completed_rollouts, 0)
        self.assertEqual(model.policy.optimizer.step_calls, 0)

    def test_invalid_full_budgets_and_conflicting_modes_fail_before_any_work(self):
        for options in (
            ["--timesteps", "0"], ["--timesteps", "-1"], ["--envs", "0"],
            ["--timesteps", "501", "--envs", "8"], ["--trial", "--full-training"],
        ):
            with (
                self.subTest(options=options),
                patch.object(train, "SO101ReachEnv") as environment,
                patch.object(train, "PPO") as policy,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as caught:
                    train.main(["--output", str(self.output), *options])
                self.assertEqual(caught.exception.code, 2)
                self.assertFalse(self.output.exists())
                environment.assert_not_called()
                policy.assert_not_called()

    def test_reward_scale_changes_training_rewards_but_monitor_keeps_raw_returns(self):
        self.run_fake_trial("--timesteps", "23")
        model = RolloutRoundingPPO.instances[0]
        np.testing.assert_allclose(model.collected_rewards, -0.01)
        monitor_path = self.output / "monitor_0.csv.monitor.csv"
        with monitor_path.open(encoding="utf-8") as handle:
            next(handle)
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(float(row["r"]) == -7.0 for row in rows))
        report = json.loads((self.output / "training.json").read_text())
        self.assertEqual(report["config"]["reward_scale"], 0.01)
        self.assertEqual(report["environment"], {"fake": True})

    def test_full_training_experiment_options_are_forwarded_and_recorded(self):
        with self.fake_full_training() as fake:
            train.main([
                "--timesteps", "256", "--envs", "1", "--output", str(self.output),
                "--reward-scale", "1", "--learning-rate", "0.0001", "--n-epochs", "5",
                "--target-kl", "0.02", "--use-sde", "--sde-sample-freq", "8",
            ])
        model = RolloutRoundingPPO.instances[0]
        self.assertEqual(model.kwargs["learning_rate"], 0.0001)
        self.assertEqual(model.kwargs["n_epochs"], 5)
        self.assertEqual(model.kwargs["target_kl"], 0.02)
        self.assertTrue(model.kwargs["use_sde"])
        self.assertEqual(model.kwargs["sde_sample_freq"], 8)
        np.testing.assert_array_equal(model.collected_rewards, -np.ones(256))
        self.assertEqual(fake.validation.call_args.kwargs["eval_frequency"], 100_000)
        report = json.loads((self.output / "training.json").read_text())
        self.assertEqual(report["config"]["reward_scale"], 1.0)
        self.assertEqual(report["config"]["target_kl"], 0.02)
        self.assertEqual(report["config"]["vec_env_backend"], "dummy")
        self.assertEqual(set(report["phase_seconds"]), {
            "rollout_including_callbacks", "optimization", "diagnostics", "validation", "heldout",
        })

    def test_full_training_records_and_announces_heldout_phase_before_evaluation(self):
        with self.fake_full_training() as fake, patch("builtins.print") as log:
            completed_report = fake.evaluate.return_value
            checkpoint_bytes = b"saved final checkpoint"
            fake.checkpoint.side_effect = lambda model, path, *args: path.write_bytes(checkpoint_bytes)

            def inspect_heldout_start(checkpoint, *, episodes, seed):
                fake.checkpoint.assert_called_once()
                self.assertEqual((self.output / "final_model.zip").read_bytes(), checkpoint_bytes)
                self.assertEqual(checkpoint, self.output / "best_model.zip")
                self.assertEqual((episodes, seed), (7, 20_000))
                report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "evaluating")
                self.assertEqual(report["config"]["heldout_episodes"], 7)
                self.assertEqual(report["environment"], {"fake": True})
                self.assertEqual(report["actual_timesteps"], 256)
                self.assertEqual(report["optimizer_steps"], 10)
                self.assertEqual(report["update_count"], 1)
                self.assertEqual(report["heldout_checkpoint"], "best_model.zip")
                self.assertFalse(report["task_performance_evaluated"])
                self.assertGreaterEqual(report["training_seconds"], 0)
                self.assertEqual(report["phase_seconds"]["heldout"], 0)
                self.assertNotIn("finished_utc", report)
                self.assertNotIn("heldout_summaries", report)
                self.assertFalse((self.output / "evaluation.json").exists())
                phase_logs = [call for call in log.call_args_list if call.kwargs.get("flush")]
                self.assertTrue(any("Training finished. Final checkpoint saved" in call.args[0]
                                    for call in phase_logs))
                self.assertTrue(any("3 policies x 7 episodes (21 total)" in call.args[0]
                                    and "Ctrl+C" in call.args[0]
                                    and "saved checkpoints are retained" in call.args[0]
                                    for call in phase_logs))
                return completed_report

            fake.evaluate.side_effect = inspect_heldout_start
            train.main([
                "--timesteps", "256", "--envs", "1", "--heldout-episodes", "7",
                "--output", str(self.output),
            ])
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "completed")
        self.assertIn("finished_utc", report)
        self.assertEqual(report["heldout_summaries"], {
            "trained": completed_report["results"]["trained"]["summary"],
        })

    def test_interrupted_heldout_keeps_checkpoint_and_observed_training_counters(self):
        with self.fake_full_training() as fake:
            checkpoint_bytes = b"saved checkpoint"
            fake.checkpoint.side_effect = lambda model, path, *args: path.write_bytes(checkpoint_bytes)
            fake.evaluate.side_effect = KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                train.main(["--timesteps", "256", "--envs", "1", "--output", str(self.output)])
        report = json.loads((self.output / "training.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["checkpoint"], "final_model.zip")
        self.assertEqual((self.output / "final_model.zip").read_bytes(), checkpoint_bytes)
        self.assertEqual(report["actual_timesteps"], 256)
        self.assertEqual(report["optimizer_steps"], 10)
        self.assertEqual(report["update_count"], 1)
        self.assertFalse(report["task_performance_evaluated"])
        self.assertNotIn("heldout_summaries", report)
        self.assertFalse((self.output / "evaluation.json").exists())
        self.assertTrue(all(env.closed for env in CountingEnvironment.instances))

    def test_kl_stopped_epoch_still_counts_an_actual_rollout_update(self):
        with self.fake_full_training(), patch.object(RolloutRoundingPPO, "early_stop_epochs", 1):
            train.main([
                "--timesteps", "2048", "--target-kl", "0.01", "--output", str(self.output),
            ])
        report = json.loads((self.output / "training.json").read_text())
        self.assertEqual(report["optimization_epochs"], 1)
        self.assertEqual(report["update_count"], 1)
        self.assertEqual(report["optimizer_steps"], 8)

    def test_subproc_backend_uses_spawn_and_closes_on_completion(self):
        vectors = []

        def local_vector(factories, *, start_method):
            self.assertEqual(start_method, "spawn")
            vector = train.DummyVecEnv(factories)
            vector.seed = MagicMock(wraps=vector.seed)
            vector.close = MagicMock(wraps=vector.close)
            vectors.append(vector)
            return vector

        with self.fake_full_training(), patch.object(train, "SubprocVecEnv", side_effect=local_vector):
            train.main([
                "--timesteps", "512", "--envs", "2", "--vec-env", "subproc",
                "--output", str(self.output),
            ])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors[0].num_envs, 2)
        vectors[0].seed.assert_called_once_with(42)
        vectors[0].close.assert_called_once()
        self.assertTrue(all(env.closed for env in CountingEnvironment.instances))

    def test_vector_is_closed_and_failure_recorded_if_seeding_fails(self):
        vector = MagicMock()
        vector.seed.side_effect = RuntimeError("seed failed")
        with self.fake_full_training(), patch.object(train, "DummyVecEnv", return_value=vector):
            with self.assertRaisesRegex(RuntimeError, "seed failed"):
                train.main(["--timesteps", "256", "--output", str(self.output)])
        vector.close.assert_called_once()
        report = json.loads((self.output / "training.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["optimizer_steps"], 0)

    def test_worker_cleanup_error_does_not_skip_logger_or_evaluation_cleanup(self):
        def local_vector(factories, *, start_method):
            vector = train.DummyVecEnv(factories)
            original_close = vector.close

            def fail_after_closing():
                original_close()
                raise RuntimeError("worker cleanup failed")

            vector.close = fail_after_closing
            return vector

        with self.fake_full_training() as fake, patch.object(train, "SubprocVecEnv", side_effect=local_vector):
            with self.assertRaisesRegex(RuntimeError, "worker cleanup failed"):
                train.main([
                    "--timesteps", "256", "--envs", "1", "--vec-env", "subproc",
                    "--output", str(self.output),
                ])
        RolloutRoundingPPO.instances[0].logger.close.assert_called_once()
        fake.validation.return_value.eval_env.close.assert_called_once()
        self.assertTrue(all(env.closed for env in CountingEnvironment.instances))

    def test_invalid_experiment_options_fail_before_creating_any_output(self):
        for options in (
            ["--reward-scale", "0"], ["--reward-scale", "nan"], ["--reward-scale", "inf"],
            ["--learning-rate", "-0.1"], ["--n-epochs", "0"], ["--target-kl", "0"],
            ["--sde-sample-freq", "0"], ["--sde-sample-freq", "8"],
            ["--viewer", "--vec-env", "subproc"], ["--trial", "--vec-env", "subproc"],
            ["--trial", "--target-kl", "0.01"],
        ):
            with (
                self.subTest(options=options),
                patch.object(train, "training_output") as destination,
                patch.object(train, "SO101ReachEnv") as environment,
                patch.object(train, "PPO") as policy,
                redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as caught:
                    train.main(["--output", str(self.output), *options])
                self.assertEqual(caught.exception.code, 2)
                destination.assert_not_called()
                environment.assert_not_called()
                policy.assert_not_called()

    def test_resume_adds_a_complete_rollout_to_an_unaligned_parent_count(self):
        with self.fake_resume_training() as resumed:
            train.main([
                "--resume", str(resumed.record.path), "--timesteps", "2048",
                "--output", str(self.output),
            ])
        model = RolloutRoundingPPO.instances[0]
        self.assertEqual(model.learn_budgets, [2048])
        self.assertFalse(model.reset_num_timesteps)
        self.assertEqual(model.num_timesteps, 302_048)
        self.assertEqual(model.policy.optimizer.step_calls, 11_760)
        self.assertEqual(sum(env.step_calls for env in CountingEnvironment.instances), 2048)
        resumed.load.assert_called_once()
        resumed.fresh.assert_not_called()
        report = json.loads((self.output / "training.json").read_text())
        self.assertTrue(report["config"]["checkpoint_loaded"])
        self.assertEqual(report["config"]["parent_checkpoint_sha256"], "parent-digest")
        self.assertEqual(report["config"]["starting_timesteps"], 300_000)
        self.assertEqual(report["actual_timesteps"], 2048)
        self.assertEqual(report["cumulative_timesteps"], 302_048)
        self.assertEqual(report["update_count"], 1)
        self.assertEqual(report["cumulative_update_count"], 147)
        self.assertEqual(report["optimizer_steps"], 80)
        self.assertEqual(report["cumulative_optimizer_steps"], 11_760)
        self.assertEqual(report["optimization_epochs"], 10)
        self.assertEqual(report["cumulative_optimization_epochs"], 1470)
        saved = resumed.fake.checkpoint.call_args.args[3]
        self.assertEqual(saved["actual_timesteps"], 2048)
        self.assertEqual(saved["cumulative_timesteps"], 302_048)

    def test_resume_stops_at_exact_session_cap_and_discards_partial_rollout(self):
        with self.fake_resume_training() as resumed:
            train.main([
                "--resume", str(resumed.record.path), "--timesteps", "2056",
                "--output", str(self.output),
            ])
        report = json.loads((self.output / "training.json").read_text())
        self.assertEqual(report["actual_timesteps"], 2056)
        self.assertEqual(report["cumulative_timesteps"], 302_056)
        self.assertEqual(report["optimizer_steps"], 80)
        self.assertEqual(report["update_count"], 1)
        self.assertEqual(sum(env.step_calls for env in CountingEnvironment.instances), 2056)

    def test_resume_inherits_saved_learning_settings_and_requires_explicit_reward_units(self):
        record = self.resume_record(learning_rate=0.0001, n_epochs=5, target_kl=0.01,
                                    use_sde=True, sde_sample_freq=8, log_std_init=-2.5,
                                    reward_scale=1.0)
        with self.fake_resume_training(record):
            train.main([
                "--resume", str(record.path), "--reward-scale", "1", "--timesteps", "2048",
                "--output", str(self.output),
            ])
        report = json.loads((self.output / "training.json").read_text())
        for key in train.LEARNING_DEFAULTS:
            self.assertEqual(report["config"][key], record.training[key])
        self.assertEqual(report["optimizer_steps"], 40)
        np.testing.assert_allclose(RolloutRoundingPPO.instances[0].collected_rewards, -1)

    def test_resume_rejects_changed_learning_or_reward_settings_before_output(self):
        record = self.resume_record(learning_rate=0.0001)
        for options in (["--learning-rate", "0.0003"], ["--reward-scale", "1"],
                        ["--n-epochs", "5"], ["--use-sde"], ["--target-kl", "0.01"]):
            with (
                self.subTest(options=options), self.fake_resume_training(record) as resumed,
                patch.object(train, "training_output") as destination,
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit),
            ):
                train.main(["--resume", str(record.path), *options])
            destination.assert_not_called()
            resumed.load.assert_not_called()

    def test_resume_latest_resolves_parent_before_replacing_latest_pointer(self):
        parent = Path(self.temporary.name) / "runs" / "experiments" / "parent"
        parent.mkdir(parents=True)
        checkpoint = parent / "best_model.zip"
        checkpoint.write_bytes(b"parent remains unchanged")
        pointer = parent.parent.parent / "latest.json"
        pointer.write_text(json.dumps({"schema_version": 1, "path": "experiments/parent"}))
        record = self.resume_record()
        record.path = checkpoint.resolve()
        with self.fake_resume_training(record) as resumed:
            train.main(["--resume", "latest", "--timesteps", "2048"])
        resumed.inspect.assert_called_once_with(checkpoint.resolve(), {"fake": True})
        output = outputs.current_output()
        self.assertNotEqual(output, parent.resolve())
        self.assertEqual(checkpoint.read_bytes(), b"parent remains unchanged")
        report = json.loads((output / "training.json").read_text())
        self.assertEqual(report["config"]["parent_checkpoint"], str(checkpoint.resolve()))

    def test_interrupted_resume_keeps_session_and_parent_counters(self):
        with self.fake_resume_training() as resumed, patch.object(RolloutRoundingPPO, "interrupt_after_rollouts", 1):
            with self.assertRaises(KeyboardInterrupt):
                train.main(["--resume", str(resumed.record.path), "--output", str(self.output)])
        report = json.loads((self.output / "training.json").read_text())
        self.assertEqual(report["status"], "interrupted")
        self.assertEqual(report["actual_timesteps"], 2048)
        self.assertEqual(report["cumulative_timesteps"], 302_048)
        self.assertEqual(report["optimizer_steps"], 80)
        self.assertEqual(report["cumulative_optimizer_steps"], 11_760)
        resumed.fake.evaluate.assert_not_called()
        metadata = resumed.fake.checkpoint.call_args.args[3]
        self.assertEqual(metadata["actual_timesteps"], 2048)
        self.assertEqual(metadata["cumulative_timesteps"], 302_048)

    def test_resume_load_failure_is_recorded_without_falling_back_to_fresh(self):
        with self.fake_resume_training() as resumed, redirect_stderr(io.StringIO()):
            resumed.load.side_effect = train.ResumeError("restore failed")
            with self.assertRaises(SystemExit):
                train.main([
                    "--resume", str(resumed.record.path), "--timesteps", "2048",
                    "--output", str(self.output),
                ])
        report = json.loads((self.output / "training.json").read_text())
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["config"]["checkpoint_loaded"])
        self.assertEqual(report["actual_timesteps"], 0)
        self.assertEqual(report["cumulative_timesteps"], 300_000)
        resumed.fresh.assert_not_called()
        resumed.fake.checkpoint.assert_not_called()
        self.assertTrue(all(env.closed for env in CountingEnvironment.instances))

    def test_trial_rejects_resume_before_inspection_or_output(self):
        with (
            patch.object(train, "inspect_resume_checkpoint") as inspect,
            patch.object(train, "training_output") as destination,
            patch.object(train, "SO101ReachEnv") as environment,
            redirect_stderr(io.StringIO()), self.assertRaises(SystemExit),
        ):
            train.main(["--trial", "--resume", "latest"])
        inspect.assert_not_called()
        destination.assert_not_called()
        environment.assert_not_called()


class ValidationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="so101-validation-metrics-")
        self.addCleanup(self.temporary.cleanup)
        with patch.object(train, "SO101ReachEnv", CountingEnvironment):
            self.callback = train.FixedSeedEvaluation(
                Path(self.temporary.name), {"raw_environment": True}, {"reward_scale": 0.01},
                eval_episodes=2, start_time=0,
                progress=lambda: {"optimizer_steps": 4, "update_count": 1},
            )
        self.addCleanup(self.callback.eval_env.close)
        self.logger = MagicMock()
        self.callback.init_callback(SimpleNamespace(logger=self.logger, get_env=lambda: None))
        self.callback.num_timesteps = 100_000

    def test_validation_uses_unscaled_env_and_counts_actual_steps_and_time(self):
        report = {
            "summary": {"episodes": 2, "successes": 1, "success_rate": 0.5,
                        "mean_final_distance_m": 0.02, "mean_return": -7.0},
            "episodes": [{"steps": 3}, {"steps": 7}],
        }
        with (
            patch.object(train, "run_episodes", return_value=report) as episodes,
            patch.object(train, "save_checkpoint") as checkpoint,
            patch("so101_sim.train.time.perf_counter", side_effect=[10, 12.5, 13, 14]),
            redirect_stdout(io.StringIO()),
        ):
            self.callback.evaluate()
        self.assertIsInstance(episodes.call_args.args[0], CountingEnvironment)
        self.assertEqual(self.callback.eval_env.step_calls, 0)
        self.assertEqual(self.callback.validation_seconds, 2.5)
        self.assertEqual(self.callback.validation_steps, 10)
        self.assertEqual(self.callback.validation_episodes, 2)
        metadata = checkpoint.call_args.args[3]
        self.assertEqual(metadata["optimizer_steps"], 4)
        self.assertEqual(metadata["reward_scale"], 0.01)
        with (Path(self.temporary.name) / "evaluations.csv").open(encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(float(row["mean_return"]), -7)
        self.assertEqual(float(row["evaluation_seconds"]), 2.5)
        self.assertEqual(int(row["validation_control_steps"]), 10)

    def test_failed_validation_retains_elapsed_time_without_inventing_counts(self):
        with (
            patch.object(train, "run_episodes", side_effect=RuntimeError("evaluation interrupted")),
            patch("so101_sim.train.time.perf_counter", side_effect=[2, 4]),
            self.assertRaisesRegex(RuntimeError, "evaluation interrupted"),
        ):
            self.callback.evaluate()
        self.assertEqual(self.callback.validation_seconds, 2)
        self.assertEqual(self.callback.validation_steps, 0)
        self.assertEqual(self.callback.validation_episodes, 0)

    def test_distance_and_hold_diagnostics_use_existing_rollout_info_only(self):
        self.callback._on_rollout_start()
        self.callback.locals = {"infos": [
            {"distance": 0.01, "hold_steps": 3},
            {"distance": 0.05, "hold_steps": 0},
            {"distance": 0.10, "hold_steps": 0},
        ]}
        self.callback.num_timesteps = 8
        self.callback._on_step()
        self.callback._on_rollout_end()
        metrics = {call.args[0]: call.args[1] for call in self.logger.record.call_args_list}
        self.assertEqual(metrics["rollout/maximum_hold_steps"], 3)
        self.assertEqual(metrics["rollout/within_15mm_fraction"], 1 / 3)
        self.assertEqual(metrics["rollout/within_80mm_fraction"], 2 / 3)
        self.assertEqual(self.callback.eval_env.step_calls, 0)

    def test_resumed_validation_waits_for_session_frequency_and_saves_session_counts(self):
        self.callback.starting_timesteps = 300_000
        self.callback.next_evaluation = 400_000
        self.callback.locals = {"infos": []}
        with patch.object(self.callback, "evaluate") as evaluate:
            self.callback.num_timesteps = 300_008
            self.callback._on_step()
            evaluate.assert_not_called()
            self.callback.num_timesteps = 400_000
            self.callback._on_step()
            evaluate.assert_called_once()
        metadata = self.callback.training_metadata()
        self.assertEqual(metadata["actual_timesteps"], 100_000)
        self.assertEqual(metadata["cumulative_timesteps"], 400_000)
        self.assertEqual(self.callback.next_evaluation, 500_000)


if __name__ == "__main__":
    unittest.main()
