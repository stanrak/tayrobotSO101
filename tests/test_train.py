"""Verify training modes and budgets with fake environments and a fake PPO."""

from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
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

    def __init__(self, policy, env, **kwargs):
        self.env = env
        self.kwargs = kwargs
        self.num_timesteps = 0
        self._total_timesteps = 0
        self._n_updates = 0
        self.callback_calls = 0
        self.learn_budgets = []
        self.completed_rollouts = 0
        self.policy = SimpleNamespace(optimizer=CountingOptimizer())
        self.instances.append(self)

    def set_logger(self, logger):
        self.logger = logger

    def get_env(self):
        return self.env

    def learn(self, total_timesteps, callback=None, log_interval=None):
        self.callback = callback
        self.learn_budgets.append(total_timesteps)
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
                _, _, _, infos = self.env.step(np.zeros((self.env.num_envs, 3), dtype=np.float32))
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
            for _ in range(self.kwargs["n_epochs"]):
                for _ in range(batch_count):
                    self.policy.optimizer.step()
                self._n_updates += 1
            self.completed_rollouts += 1
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


if __name__ == "__main__":
    unittest.main()
