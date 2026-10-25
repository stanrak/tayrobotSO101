"""Verify separate experiment records, retained bytes, and writer locks."""

from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import shutil
import subprocess
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

    @staticmethod
    def _training_record(output, status="trial_completed", *, mode="startup_trial", seed=42):
        record = {"status": status, "mode": mode, "actual_timesteps": 23,
                  "optimizer_steps": 1, "config": {"seed": seed, "requested_timesteps": 23},
                  "task_performance_evaluated": False}
        (output / "training.json").write_text(json.dumps(record), encoding="utf-8")

    def _checkpoint(self):
        with outputs.training_output(mode="startup_trial", seed=42) as output:
            self._training_record(output)
            checkpoint = output / "trial_model.zip"
            checkpoint.write_bytes(b"model fixture")
        return checkpoint

    @staticmethod
    def _snapshot(directory):
        return {path.relative_to(directory).as_posix(): path.read_bytes()
                for path in directory.rglob("*") if path.is_file()}

    @staticmethod
    def _evaluation_record(checkpoint):
        return {"checkpoint": str(checkpoint.resolve()), "results": {"trained": {"summary": {
            "successes": 0, "episodes": 5, "success_rate": 0,
            "mean_final_distance_m": .05, "mean_return": -5,
        }}}}

    def test_repeated_runs_get_separate_paths_and_keep_complete_previous_run(self):
        with outputs.training_output(mode="startup_trial", seed=42) as first:
            self._training_record(first)
            (first / "trial_model.zip").write_bytes(b"original model")
            (first / "extra").mkdir()
            (first / "extra/log.txt").write_bytes(b"nested log")
        original = self._snapshot(first)
        with outputs.training_output(mode="full_training", seed=7) as second:
            self.assertNotEqual(first, second)
            self._training_record(second, status="interrupted", mode="full_training", seed=7)
            self.assertEqual(outputs.current_output(), second)
        self.assertEqual(first.parent, self.root / "experiments")
        self.assertEqual(second.parent, first.parent)
        self.assertEqual(self._snapshot(first), original)
        self.assertTrue((first / "run.json").is_file())
        self.assertTrue((second / "run.json").is_file())
        self.assertTrue((self.root / "index.csv").is_file())
        self.assertFalse((self.root / "history").exists())

    def test_same_timestamp_runs_and_evaluations_still_get_unique_paths(self):
        with patch.object(outputs, "_stamp", return_value="20261025T000000000000Z"):
            first = self._checkpoint()
            first_snapshot = self._snapshot(first.parent)
            second = self._checkpoint()
            self.assertNotEqual(first.parent, second.parent)
            self.assertEqual(self._snapshot(first.parent), first_snapshot)
            reports = []
            for _ in range(2):
                with outputs.evaluation_output(checkpoint=second) as report:
                    report.write_text(json.dumps(self._evaluation_record(second)), encoding="utf-8")
                    reports.append(report)
            self.assertNotEqual(reports[0], reports[1])
            self.assertTrue(all(report.is_file() for report in reports))

    def test_active_training_prevents_new_training_or_evaluation(self):
        with outputs.training_output() as current:
            self._training_record(current, status="running")
            (current / "trial_model.zip").write_bytes(b"active model")
            original = (current / "training.json").read_bytes()
            pointer = (self.root / "latest.json").read_bytes()
            with self.assertRaises(outputs.OutputError):
                with outputs.training_output():
                    self.fail("Second writer entered")
            with self.assertRaises(outputs.OutputError):
                with outputs.evaluation_output():
                    self.fail("Evaluation entered during training")
            self.assertEqual((current / "training.json").read_bytes(), original)
            self.assertEqual((self.root / "latest.json").read_bytes(), pointer)
            self.assertEqual(list((self.root / "experiments").iterdir()), [current])

    def test_exception_releases_lock_and_preserves_failed_run(self):
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with outputs.training_output() as failed:
                self._training_record(failed, status="running")
                (failed / "partial.bin").write_bytes(b"keep partial output")
                raise RuntimeError("fixture failure")
        record = json.loads((failed / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        failed_snapshot = self._snapshot(failed)
        with outputs.training_output() as following:
            self._training_record(following)
        self.assertNotEqual(failed, following)
        self.assertEqual(self._snapshot(failed), failed_snapshot)

    def test_interrupted_run_is_retained_and_lock_is_released(self):
        with self.assertRaises(KeyboardInterrupt):
            with outputs.training_output() as interrupted:
                (interrupted / "partial.bin").write_bytes(b"interrupted output")
                raise KeyboardInterrupt
        record = json.loads((interrupted / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "interrupted")
        with outputs.training_output() as next_run:
            self._training_record(next_run)
        self.assertEqual((interrupted / "partial.bin").read_bytes(), b"interrupted output")
        self.assertNotEqual(interrupted, next_run)

    def test_custom_nonempty_directory_is_preserved(self):
        custom = self.root / "custom"
        custom.mkdir(parents=True)
        (custom / "notes.txt").write_text("keep", encoding="utf-8")
        with self.assertRaises(outputs.OutputError):
            with outputs.training_output(custom):
                self.fail("Nonempty custom output was accepted")
        self.assertEqual((custom / "notes.txt").read_text(), "keep")

    @unittest.skipUnless(shutil.which("git"), "Git is required to verify ignore rules")
    def test_custom_experiment_artifacts_are_ignored_by_git(self):
        # A disposable repository proves custom outputs are protected even
        # outside RUNS_ROOT and without the project's top-level ignore rules.
        repository = self.root.parent / "disposable_repository"
        repository.mkdir()
        subprocess.run(["git", "init", "--quiet", str(repository)],
                       check=True, capture_output=True)
        custom = repository / "custom_training_run"
        with outputs.training_output(custom, mode="startup_trial", seed=42) as output:
            self._training_record(output)
            (output / "trial_model.zip").write_bytes(b"local checkpoint fixture")
        artifacts = [custom / name for name in (
            "run.json", "training.json", "source/so101_sim/env.py", "trial_model.zip", ".gitignore",
        )]
        self.assertTrue(all(path.is_file() for path in artifacts))
        relative = [path.relative_to(repository).as_posix() for path in artifacts]
        ignored = subprocess.run(
            ["git", "-c", f"safe.directory={repository.as_posix()}",
             "check-ignore", "--verbose", "--", *relative],
            cwd=repository, check=True, capture_output=True, text=True,
        )
        lines = ignored.stdout.splitlines()
        self.assertEqual({line.split("\t", 1)[1] for line in lines}, set(relative))
        self.assertTrue(all(line.startswith("custom_training_run/.gitignore:2:*\t") for line in lines))

    def test_legacy_current_directory_is_not_moved_or_changed(self):
        legacy = self.root / "current"
        legacy.mkdir(parents=True)
        self._training_record(legacy)
        (legacy / "trial_model.zip").write_bytes(b"legacy model")
        original = self._snapshot(legacy)
        self.assertEqual(outputs.current_output(), legacy)
        self.assertEqual(outputs.current_checkpoint(), legacy / "trial_model.zip")
        with outputs.training_output(legacy) as new_run:
            self._training_record(new_run)
        self.assertNotEqual(new_run, legacy)
        self.assertEqual(outputs.current_output(), new_run)
        self.assertEqual(self._snapshot(legacy), original)
        self.assertFalse((self.root / "history").exists())

    def test_default_checkpoint_selects_latest_best_then_trial_then_final(self):
        with self.assertRaises(outputs.OutputError):
            outputs.current_checkpoint()
        checkpoint = self._checkpoint()
        current = checkpoint.parent
        self.assertEqual(outputs.current_checkpoint(), checkpoint)
        (current / "best_model.zip").write_bytes(b"best")
        self.assertEqual(outputs.current_checkpoint().name, "best_model.zip")
        (current / "final_model.zip").write_bytes(b"final")
        self.assertEqual(outputs.current_checkpoint().name, "best_model.zip")
        (current / "best_model.zip").unlink()
        (current / "trial_model.zip").unlink()
        self.assertEqual(outputs.current_checkpoint().name, "final_model.zip")

    def test_invalid_latest_pointer_is_rejected_instead_of_falling_back(self):
        self.root.mkdir(parents=True)
        (self.root / "not_directory").write_bytes(b"file")
        pointer = self.root / "latest.json"
        for content in ("not json", "[]", json.dumps({"schema_version": 1, "path": "../outside"}),
                        json.dumps({"schema_version": 1, "path": "experiments/missing"}),
                        json.dumps({"schema_version": 1, "path": "not_directory"}),
                        json.dumps({"schema_version": 1, "path": str(self.root.parent / "outside")})):
            with self.subTest(content=content):
                pointer.write_text(content, encoding="utf-8")
                with self.assertRaises(outputs.OutputError):
                    outputs.current_output()

    def test_repeated_evaluations_get_separate_provenance_and_reports(self):
        checkpoint = self._checkpoint()
        reports = []
        for index in range(2):
            with outputs.evaluation_output(checkpoint=checkpoint, seed=20_000 + index) as report:
                report.write_text(json.dumps(self._evaluation_record(checkpoint)), encoding="utf-8")
                reports.append(report)
            self.assertTrue((report.parent / "run.json").is_file())
        self.assertNotEqual(reports[0], reports[1])
        self.assertTrue(all(report.parent.parent == checkpoint.parent / "evaluations" for report in reports))
        self.assertEqual(reports[0].read_bytes(), reports[1].read_bytes())
        self.assertEqual(outputs.current_output(), checkpoint.parent)
        self.assertFalse((checkpoint.parent / "evaluation.json").exists())

    def test_existing_custom_evaluation_report_is_not_overwritten(self):
        checkpoint = self._checkpoint()
        existing = self.root / "my_report.json"
        existing.write_bytes(b"original report bytes")
        with self.assertRaises(outputs.OutputError):
            with outputs.evaluation_output(existing, checkpoint=checkpoint):
                self.fail("Existing evaluation report was accepted")
        self.assertEqual(existing.read_bytes(), b"original report bytes")

    def test_custom_evaluation_report_also_has_a_canonical_experiment_record(self):
        checkpoint = self._checkpoint()
        custom = self.root / "exports" / "report.json"
        contents = json.dumps(self._evaluation_record(checkpoint))
        with outputs.evaluation_output(custom, checkpoint=checkpoint) as report:
            self.assertEqual(report, custom.resolve())
            report.write_text(contents, encoding="utf-8")
        canonical = list((checkpoint.parent / "evaluations").glob("*/evaluation.json"))
        self.assertEqual(len(canonical), 1)
        self.assertEqual(canonical[0].read_bytes(), custom.read_bytes())
        self.assertTrue((canonical[0].parent / "run.json").is_file())

    def test_path_outside_runs_is_rejected(self):
        with self.assertRaises(outputs.OutputError):
            outputs._inside(self.root / "../outside", self.root)

    def test_evaluation_cli_defaults_to_latest_checkpoint_and_unique_report(self):
        checkpoint = self._checkpoint()
        report = self._evaluation_record(checkpoint)
        with (
            patch("sys.argv", ["evaluate", "--render", "--episodes", "5"]),
            patch.object(evaluate, "evaluate_checkpoint", return_value=report) as rollout,
            redirect_stdout(io.StringIO()),
        ):
            evaluate.main()
        rollout.assert_called_once_with(checkpoint, episodes=5, seed=20_000, render=True)
        reports = list((checkpoint.parent / "evaluations").glob("*/evaluation.json"))
        self.assertEqual(len(reports), 1)
        saved = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(saved["checkpoint"], str(checkpoint.resolve()))

    def test_evaluation_without_checkpoint_does_not_run_or_create_experiment(self):
        with (
            patch("sys.argv", ["evaluate"]),
            patch.object(evaluate, "evaluate_checkpoint") as rollout,
            redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as caught:
                evaluate.main()
        self.assertEqual(caught.exception.code, 2)
        rollout.assert_not_called()
        self.assertFalse((self.root / "experiments").exists())
        self.assertFalse((self.root / "current").exists())
