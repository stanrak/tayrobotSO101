"""Experiment provenance and comparisons use saved data, never policy rollouts."""

from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from so101_sim import experiments
from so101_sim import outputs


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="so101-experiments-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "runs"

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def training(self, directory, *, mode="full_training", seed=42, task="test-task", status="completed"):
        record = {
            "status": status, "mode": mode, "actual_timesteps": 600,
            "optimizer_steps": 2, "training_seconds": 4.5,
            "config": {"seed": seed, "requested_timesteps": 600, "num_envs": 1,
                       "learning_rate": 0.0003, "started_utc": "2026-10-25T01:00:00+00:00"},
            "environment": {"task_config": {"task": task}, "environment_source_sha256": "source"},
        }
        self.write_json(directory / "training.json", record)
        return record

    def report(self, *, success=0.5, seed=20000):
        return {
            "evaluation_seed_start": seed, "evaluation_seed_end": seed + 1,
            "checkpoint_sha256": "checkpoint-digest",
            "environment": {"task_config": {"task": "test-task"}},
            "results": {"trained": {
                "summary": {"episodes": 2, "success_rate": success, "mean_return": -2,
                            "mean_final_distance_m": 0.03},
                "episodes": [{"seed": seed}, {"seed": seed + 1}],
            }},
        }

    def test_provenance_snapshots_only_allowlisted_sources(self):
        repository = Path(self.temporary.name) / "repository"
        files = {
            "so101_sim/env.py": b"environment source\n",
            "so101_sim/train.py": b"trainer source\n",
            "so101_sim/learning.py": b"learning diagnostics source\n",
            "requirements.txt": b"numpy==test\n",
            "assets/model_manifest.json": b"{}\n",
            "instructions/instruction.md": b"project rules\n",
            "instructions/workflow.md": b"agent workflow\n",
            "instructions/training.md": b"training contract\n",
            "instructions/experiments.md": b"experiment contract\n",
            "instructions/physical-contract.md": b"physical contract\n",
            "instructions/huongdan.md": b"user-owned guidance\n",
            "private/huongdan.md": b"private fixture excluded from snapshots\n",
            ".env": b"SECRET=must-not-copy\n",
            "runs/older/training.json": b"old result\n",
        }
        for name, content in files.items():
            path = repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        directory = self.root / "experiments/run-a"
        with (
            patch.object(experiments, "ROOT", repository),
            patch.object(experiments, "_git_provenance", return_value={"commit": "abc", "dirty": True}),
            patch.object(experiments, "version", return_value="test"),
            patch.object(experiments.sys, "orig_argv", ["python", "-m", "so101_sim.train", "--seed", "42"]),
        ):
            record = experiments.create_run_record(directory, mode="startup_trial", seed=42)
        self.assertEqual(record["mode"], "startup_trial")
        self.assertEqual(record["seed"], 42)
        self.assertEqual(record["git"], {"commit": "abc", "dirty": True})
        self.assertEqual(record["command"][-2:], ["--seed", "42"])
        self.assertEqual(record["source_files_sha256"]["so101_sim/env.py"],
                         hashlib.sha256(files["so101_sim/env.py"]).hexdigest())
        self.assertEqual(record["source_files_sha256"]["so101_sim/learning.py"],
                         hashlib.sha256(files["so101_sim/learning.py"]).hexdigest())
        self.assertTrue({
            "instructions/instruction.md", "instructions/workflow.md",
            "instructions/training.md", "instructions/experiments.md",
            "instructions/physical-contract.md",
        }.issubset(record["source_files_sha256"]))
        for name in record["source_files_sha256"]:
            self.assertEqual((directory / "source" / name).read_bytes(), files[name])
        self.assertFalse((directory / "source/.env").exists())
        self.assertFalse((directory / "source/runs").exists())
        self.assertFalse((directory / "source/instructions/huongdan.md").exists())
        self.assertFalse((directory / "source/private").exists())
        self.assertNotIn("environ", record)
        with self.assertRaises(FileExistsError):
            experiments.create_run_record(directory, mode="full_training", seed=42)

    def test_finalization_preserves_raw_files_and_observed_counts(self):
        directory = self.root / "experiments/run-a"
        original = {"schema_version": 1, "run_id": "run-a", "status": "running",
                    "created_utc": "start", "source_sha256": "immutable", "command": ["python"]}
        self.write_json(directory / "run.json", original)
        training = self.training(directory, mode="startup_trial", status="trial_completed")
        before = (directory / "training.json").read_bytes()
        (directory / "model.zip").write_bytes(b"checkpoint")
        record = experiments.finish_run_record(directory, status="completed")
        self.assertEqual(record["status"], "trial_completed")
        self.assertEqual(record["actual_timesteps"], training["actual_timesteps"])
        self.assertEqual(record["optimizer_steps"], 2)
        self.assertEqual(record["command"], original["command"])
        self.assertEqual(record["source_sha256"], "immutable")
        self.assertEqual((directory / "training.json").read_bytes(), before)
        self.assertEqual((directory / "model.zip").read_bytes(), b"checkpoint")

    def test_context_failure_overrides_stale_training_status(self):
        for training_status in ("running", "completed"):
            directory = self.root / "experiments" / training_status
            self.write_json(directory / "run.json", {"status": "running"})
            self.training(directory, status=training_status)
            record = experiments.finish_run_record(directory, status="failed", error="initialization failed")
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["error"], "initialization failed")
            row = next(row for row in experiments.catalog_rows(self.root) if row["path"].endswith(training_status))
            self.assertEqual(row["status"], "failed")

    def test_catalog_keeps_legacy_and_new_runs_with_comparable_metrics(self):
        old = self.root / "history/legacy/older"
        recent = self.root / "experiments/recent"
        validation = self.root / "current"
        self.training(old, seed=1, task="old-task")
        self.training(recent)
        self.training(validation, status="interrupted")
        self.write_json(recent / "run.json", {"run_id": "recent", "git": {"commit": "abc", "dirty": False}})
        self.write_json(recent / "evaluation.json", self.report())
        self.write_json(validation / "best_validation.json", {
            "summary": {"episodes": 1, "success_rate": 1.0, "mean_return": 3.0},
            "episodes": [{"seed": 10000}],
        })
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        rows = experiments.refresh_catalog(self.root)
        self.assertEqual(len(rows), 3)
        by_path = {row["path"]: row for row in rows}
        self.assertEqual(by_path["experiments/recent"]["eval_kind"], "heldout")
        self.assertEqual(by_path["experiments/recent"]["success_rate"], 0.5)
        self.assertEqual(by_path["experiments/recent"]["eval_seed_start"], 20000)
        self.assertEqual(by_path["current"]["eval_kind"], "validation")
        self.assertEqual(by_path["current"]["eval_seed_start"], 10000)
        self.assertNotEqual(by_path["current"]["eval_seed_sha256"], by_path["experiments/recent"]["eval_seed_sha256"])
        self.assertNotEqual(by_path["history/legacy/older"]["environment_sha256"],
                            by_path["experiments/recent"]["environment_sha256"])
        self.assertNotIn("success_rate", by_path["history/legacy/older"])
        with (self.root / "index.csv").open(newline="", encoding="utf-8") as handle:
            indexed = list(csv.DictReader(handle))
        self.assertEqual(len(indexed), 3)
        self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))

    def test_startup_trial_never_claims_evaluated_task_performance(self):
        directory = self.root / "experiments/trial"
        self.training(directory, mode="startup_trial", status="trial_completed")
        self.write_json(directory / "evaluation.json", self.report(success=1.0))
        rows = experiments.refresh_catalog(self.root)
        self.assertNotIn("success_rate", rows[0])
        self.assertNotIn("eval_kind", rows[0])
        self.assertEqual(rows[0]["optimizer_steps"], 2)

    def test_catalog_compares_learning_settings_counts_and_measured_phase_times(self):
        legacy = self.root / "history/legacy"
        recent = self.root / "experiments/recent"
        self.training(legacy)
        training = self.training(recent)
        settings = {
            "reward_scale": 0.01, "vec_env_backend": "subproc", "target_kl": 0.015,
            "use_sde": False, "sde_sample_freq": -1,
        }
        training["config"].update(settings)
        training.update({
            "update_count": 1, "optimization_epochs": 3,
            "phase_seconds": {"rollout_including_callbacks": 2.0, "optimization": 0.5, "diagnostics": 0.1,
                              "validation": 0.0, "heldout": 1.0},
        })
        self.write_json(recent / "training.json", training)
        before = (recent / "training.json").read_bytes()
        rows = experiments.refresh_catalog(self.root)
        by_path = {row["path"]: row for row in rows}
        row = by_path["experiments/recent"]
        for key, value in settings.items():
            self.assertEqual(row[key], value)
        measured = {
            "update_count": 1, "optimization_epochs": 3,
            "rollout_seconds_including_callbacks": 2.0, "optimization_seconds": 0.5, "diagnostics_seconds": 0.1,
            "validation_seconds": 0.0, "heldout_seconds": 1.0,
        }
        for key, value in measured.items():
            self.assertEqual(row[key], value)
        for key in (*settings, *measured):
            self.assertEqual(by_path["history/legacy"][key], "")
        with (self.root / "index.csv").open(newline="", encoding="utf-8") as handle:
            indexed = {row["path"]: row for row in csv.DictReader(handle)}
        self.assertEqual(indexed["experiments/recent"]["reward_scale"], "0.01")
        self.assertEqual(indexed["experiments/recent"]["use_sde"], "False")
        self.assertEqual(indexed["experiments/recent"]["validation_seconds"], "0.0")
        self.assertEqual(indexed["history/legacy"]["validation_seconds"], "")
        self.assertEqual((recent / "training.json").read_bytes(), before)

    def test_finalization_retains_observed_phase_times_and_counts(self):
        directory = self.root / "experiments/timed"
        self.write_json(directory / "run.json", {"status": "running"})
        training = self.training(directory, status="interrupted")
        training.update({"update_count": 3, "optimization_epochs": 7,
                         "phase_seconds": {"rollout_including_callbacks": 2.0,
                                           "optimization": 0.5, "validation": 0.25,
                                           "heldout": 0.0}})
        self.write_json(directory / "training.json", training)
        record = experiments.finish_run_record(directory, status="interrupted")
        self.assertEqual(record["phase_seconds"], training["phase_seconds"])
        self.assertEqual(record["update_count"], 3)
        self.assertEqual(record["optimization_epochs"], 7)

    def test_resume_catalog_separates_session_totals_and_checkpoint_lineage(self):
        legacy = self.root / "history/legacy"
        fresh = self.root / "experiments/fresh"
        resumed = self.root / "experiments/resumed"
        self.training(legacy)
        original = self.training(fresh)
        original["config"].update({
            "checkpoint_loaded": False, "starting_timesteps": 0,
            "starting_optimizer_steps": 0, "starting_update_count": 0,
            "starting_optimization_epochs": 0,
        })
        original.update({
            "cumulative_timesteps": 600, "cumulative_optimizer_steps": 2,
            "cumulative_update_count": 1, "cumulative_optimization_epochs": 3,
        })
        self.write_json(fresh / "training.json", original)
        continuation = self.training(resumed)
        lineage = {
            "checkpoint_loaded": True,
            "parent_checkpoint": str((fresh / "best_model.zip").resolve()),
            "parent_checkpoint_sha256": "parent-checkpoint-digest", "parent_run_id": "fresh",
            "starting_timesteps": 600, "starting_optimizer_steps": 2,
            "starting_update_count": 1, "starting_optimization_epochs": 3,
        }
        totals = {
            "cumulative_timesteps": 1200, "cumulative_optimizer_steps": 4,
            "cumulative_update_count": 2, "cumulative_optimization_epochs": 6,
        }
        continuation["config"].update(lineage)
        continuation.update({"update_count": 1, "optimization_epochs": 3, **totals})
        self.write_json(resumed / "training.json", continuation)
        before = {path: path.read_bytes() for path in self.root.rglob("*.json")}
        rows = {row["path"]: row for row in experiments.refresh_catalog(self.root)}
        row = rows["experiments/resumed"]
        self.assertEqual(row["requested_timesteps"], 600)
        self.assertEqual(row["actual_timesteps"], 600)
        self.assertEqual(row["optimizer_steps"], 2)
        self.assertEqual(row["update_count"], 1)
        self.assertEqual(row["optimization_epochs"], 3)
        for key, value in {**lineage, **totals}.items():
            self.assertEqual(row[key], value)
        self.assertFalse(rows["experiments/fresh"]["checkpoint_loaded"])
        self.assertEqual(rows["experiments/fresh"]["starting_timesteps"], 0)
        self.assertEqual(rows["experiments/fresh"]["parent_checkpoint"], "")
        self.assertEqual(rows["experiments/fresh"]["cumulative_timesteps"], 600)
        for key in (*experiments.LINEAGE_FIELDS, *experiments.CUMULATIVE_FIELDS):
            self.assertEqual(rows["history/legacy"][key], "")
        with (self.root / "index.csv").open(newline="", encoding="utf-8") as handle:
            indexed = {row["path"]: row for row in csv.DictReader(handle)}
        self.assertEqual(indexed["experiments/resumed"]["actual_timesteps"], "600")
        self.assertEqual(indexed["experiments/resumed"]["cumulative_timesteps"], "1200")
        self.assertEqual(indexed["experiments/fresh"]["checkpoint_loaded"], "False")
        self.assertEqual(indexed["history/legacy"]["cumulative_timesteps"], "")
        self.assertTrue(all(path.read_bytes() == content for path, content in before.items()))

    def test_finalization_retains_resume_lineage_and_cumulative_counts(self):
        directory = self.root / "experiments/resumed"
        self.write_json(directory / "run.json", {"run_id": "resumed", "status": "running"})
        training = self.training(directory, status="interrupted")
        lineage = {
            "checkpoint_loaded": True, "parent_checkpoint": "parent/best_model.zip",
            "parent_checkpoint_sha256": "parent-digest", "parent_run_id": "parent",
            "starting_timesteps": 300000, "starting_optimizer_steps": 11680,
            "starting_update_count": 146, "starting_optimization_epochs": 1460,
        }
        totals = {
            "cumulative_timesteps": 300600, "cumulative_optimizer_steps": 11682,
            "cumulative_update_count": 147, "cumulative_optimization_epochs": 1463,
        }
        training["config"].update(lineage)
        training.update(totals)
        self.write_json(directory / "training.json", training)
        before = (directory / "training.json").read_bytes()
        record = experiments.finish_run_record(directory, status="interrupted")
        self.assertEqual(record["actual_timesteps"], 600)
        self.assertEqual(record["optimizer_steps"], 2)
        for key, value in {**lineage, **totals}.items():
            self.assertEqual(record[key], value)
        self.assertEqual((directory / "training.json").read_bytes(), before)
        # Finalized provenance still exposes measured continuity if the training
        # metadata is unavailable; no parent checkpoint is opened or inferred.
        self.write_json(directory / "training.json", {})
        row = experiments.catalog_rows(self.root)[0]
        for key, value in {**lineage, **totals}.items():
            self.assertEqual(row[key], value)

    def test_malformed_optional_phase_times_leave_metrics_unmeasured(self):
        directory = self.root / "experiments/malformed-phases"
        training = self.training(directory)
        training["phase_seconds"] = []
        self.write_json(directory / "training.json", training)
        row = experiments.catalog_rows(self.root)[0]
        self.assertEqual(row["rollout_seconds_including_callbacks"], "")
        self.assertEqual(row["optimization_seconds"], "")
        self.assertEqual(row["validation_seconds"], "")
        self.assertEqual(row["heldout_seconds"], "")

    def test_standalone_evaluation_is_a_distinct_row(self):
        directory = self.root / "experiments/parent/evaluations/eval-a"
        self.write_json(directory / "run.json", {"run_id": "eval-a", "mode": "evaluation", "seed": 20000,
                                                "status": "completed", "git": {"commit": "abc"}})
        self.write_json(directory / "evaluation.json", self.report())
        rows = experiments.refresh_catalog(self.root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "evaluation")
        self.assertEqual(rows[0]["eval_kind"], "standalone")
        self.assertEqual(rows[0]["task"], "test-task")
        self.assertEqual(rows[0]["evaluation_checkpoint_sha256"], "checkpoint-digest")
        self.assertEqual(rows[0]["evaluation_path"], "experiments/parent/evaluations/eval-a/evaluation.json")
        self.assertEqual(rows[0]["actual_timesteps"], "")

    def test_malformed_legacy_metadata_does_not_break_catalog(self):
        for name, contents in (("broken", "not JSON"), ("array", "[]"), ("nonfinite", '{"value":NaN}')):
            directory = self.root / "history" / name
            directory.mkdir(parents=True)
            (directory / "training.json").write_text(contents, encoding="utf-8")
        rows = experiments.refresh_catalog(self.root)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["metadata_error"] for row in rows))
        self.assertTrue(all(row["status"] == "unknown" for row in rows))
        self.assertTrue(all("success_rate" not in row for row in rows))

    def test_cli_uses_writer_lock_without_loading_training_dependencies(self):
        self.training(self.root / "experiments/run-a")
        with (
            patch.object(outputs, "RUNS_ROOT", self.root),
            patch("sys.argv", ["experiments"]),
            redirect_stdout(io.StringIO()) as printed,
        ):
            experiments.main()
        self.assertIn("run-a", printed.getvalue())
        self.assertIn("blank metrics were not measured", printed.getvalue())
        self.assertTrue((self.root / "index.csv").exists())


if __name__ == "__main__":
    unittest.main()
