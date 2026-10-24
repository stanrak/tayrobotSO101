"""Verify stable output paths, retained history, and exclusive writer access."""

from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from so101_sim import outputs
from so101_sim import evaluate


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="so101-outputs-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "runs"
        replacement = patch.object(outputs, "RUNS_ROOT", self.root)
        replacement.start()
        self.addCleanup(replacement.stop)

    def test_repeated_runs_share_one_path_and_preserve_complete_previous_run(self):
        with outputs.training_output() as first:
            (first / "trial_model.zip").write_bytes(b"original model")
            (first / "training.json").write_bytes(b"original metadata")
            (first / "extra").mkdir()
            (first / "extra" / "log.txt").write_bytes(b"nested log")
        with outputs.training_output() as second:
            self.assertEqual(first, second)
            self.assertEqual(list(second.iterdir()), [])
        archives = list((self.root / "history").iterdir())
        self.assertEqual(len(archives), 1)
        self.assertEqual((archives[0] / "trial_model.zip").read_bytes(), b"original model")
        self.assertEqual((archives[0] / "training.json").read_bytes(), b"original metadata")
        self.assertEqual((archives[0] / "extra/log.txt").read_bytes(), b"nested log")

    def test_active_training_prevents_rotation_or_evaluation(self):
        with outputs.training_output() as current:
            (current / "training.json").write_text("active", encoding="utf-8")
            with self.assertRaises(outputs.OutputError):
                with outputs.training_output():
                    self.fail("Second writer entered")
            with self.assertRaises(outputs.OutputError):
                with outputs.evaluation_output():
                    self.fail("Evaluation entered during training")
            self.assertEqual((current / "training.json").read_text(), "active")
            self.assertFalse((self.root / "history").exists())

    def test_exception_releases_lock_and_next_run_keeps_partial_output(self):
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with outputs.training_output() as current:
                (current / "training.json").write_text("partial", encoding="utf-8")
                raise RuntimeError("interrupted")
        with outputs.training_output():
            pass
        archive = next((self.root / "history").iterdir())
        self.assertEqual((archive / "training.json").read_text(), "partial")

    def test_custom_nonempty_directory_is_preserved(self):
        custom = self.root / "custom"
        custom.mkdir(parents=True)
        (custom / "notes.txt").write_text("keep", encoding="utf-8")
        with self.assertRaises(outputs.OutputError):
            with outputs.training_output(custom):
                self.fail("Nonempty custom output was accepted")
        self.assertEqual((custom / "notes.txt").read_text(), "keep")

    def test_default_checkpoint_selects_current_best_then_trial(self):
        with self.assertRaises(outputs.OutputError):
            outputs.current_checkpoint()
        current = outputs.current_output()
        current.mkdir(parents=True)
        (current / "trial_model.zip").write_bytes(b"trial")
        self.assertEqual(outputs.current_checkpoint().name, "trial_model.zip")
        (current / "best_model.zip").write_bytes(b"best")
        self.assertEqual(outputs.current_checkpoint().name, "best_model.zip")

    def test_repeated_evaluation_keeps_previous_report(self):
        with outputs.evaluation_output() as report:
            report.write_bytes(b"previous report")
        with outputs.evaluation_output() as report:
            outputs.archive_evaluation(report)
            report.write_bytes(b"new report")
        self.assertEqual(report, self.root / "current/evaluation.json")
        saved = list((report.parent / "evaluation_history").glob("*.json"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_bytes(), b"previous report")

    def test_archive_destination_outside_runs_is_rejected(self):
        with self.assertRaises(outputs.OutputError):
            outputs._inside(self.root / "../outside", self.root)

    def test_evaluation_cli_defaults_to_current_checkpoint_and_report(self):
        current = outputs.current_output()
        current.mkdir(parents=True)
        checkpoint = current / "trial_model.zip"
        checkpoint.write_bytes(b"model")
        report = {"checkpoint": str(checkpoint), "results": {"trained": {"summary": {
            "successes": 0, "episodes": 5, "success_rate": 0,
            "mean_final_distance_m": .05, "mean_return": -5,
        }}}}
        with (
            patch("sys.argv", ["evaluate", "--render", "--episodes", "5"]),
            patch.object(evaluate, "evaluate_checkpoint", return_value=report) as rollout,
            redirect_stdout(io.StringIO()),
        ):
            evaluate.main()
        rollout.assert_called_once_with(checkpoint, episodes=5, seed=20_000, render=True)
        saved = json.loads((current / "evaluation.json").read_text())
        self.assertEqual(saved["checkpoint"], "trial_model.zip")

    def test_evaluation_without_current_checkpoint_does_not_run_rollouts(self):
        with (
            patch("sys.argv", ["evaluate"]),
            patch.object(evaluate, "evaluate_checkpoint") as rollout,
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as caught:
                evaluate.main()
        self.assertEqual(caught.exception.code, 2)
        rollout.assert_not_called()
