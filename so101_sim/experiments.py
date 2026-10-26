"""Record reproducible experiments and compare saved runs without loading policies."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("numpy", "mujoco", "gymnasium", "stable-baselines3", "torch")
LINEAGE_FIELDS = (
    "checkpoint_loaded", "parent_checkpoint", "parent_checkpoint_sha256", "parent_run_id",
    "starting_timesteps", "starting_optimizer_steps", "starting_update_count",
    "starting_optimization_epochs",
)
CUMULATIVE_FIELDS = (
    "cumulative_timesteps", "cumulative_optimizer_steps", "cumulative_update_count",
    "cumulative_optimization_epochs",
)
FIELDS = (
    "run_id", "path", "created_utc", "status", "mode", "seed", "task",
    "environment_sha256", "source_sha256", "git_commit", "git_dirty",
    "requested_timesteps", "actual_timesteps", "optimizer_steps", "update_count",
    "optimization_epochs", *CUMULATIVE_FIELDS, *LINEAGE_FIELDS,
    "num_envs", "vec_env_backend", "reward_scale",
    "training_seconds", "learning_rate", "batch_size", "n_steps", "n_epochs",
    "gamma", "log_std_init", "target_kl", "use_sde", "sde_sample_freq",
    "rollout_seconds_including_callbacks", "optimization_seconds", "diagnostics_seconds", "validation_seconds",
    "heldout_seconds", "eval_kind", "eval_seed_start", "eval_seed_end",
    "eval_seed_sha256", "eval_episodes", "success_rate", "mean_return",
    "mean_final_distance_m", "evaluation_path", "evaluation_checkpoint_sha256",
    "metadata_error",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    content = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8", newline="")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        return {}, ""
    try:
        def reject_constant(value: str):
            raise ValueError(f"Non-finite JSON value: {value}")

        result = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
        if not isinstance(result, dict):
            raise ValueError("expected a JSON object")
        return result, ""
    except (OSError, ValueError) as exc:
        return {}, f"{path.name}: {type(exc).__name__}: {exc}"


def _git_provenance() -> dict[str, Any]:
    """Read this repository only; never change global Git ownership settings."""
    def read(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", *arguments],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=True, timeout=10,
        )
        return result.stdout.strip()

    try:
        return {"commit": read("rev-parse", "HEAD"),
                "dirty": bool(read("status", "--porcelain", "--untracked-files=normal"))}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"commit": None, "dirty": None, "error": f"{type(exc).__name__}: Git unavailable"}


def _source_files() -> list[Path]:
    # Deliberate allowlist: never copy credentials, environment files, run data,
    # user-owned guidance, virtual environments, or arbitrary repository files.
    files = [ROOT / name for name in (
        "AGENTS.md", ".gitignore", "requirements.txt", "requirements-sim.txt",
        "assets/model_manifest.json", "instructions/instruction.md",
        "instructions/workflow.md", "instructions/training.md",
        "instructions/experiments.md", "instructions/physical-contract.md",
    )]
    for folder in ("so101_sim", "scripts"):
        files.extend((ROOT / folder).glob("*.py"))
    return sorted((path for path in files if path.is_file() and path.resolve().is_relative_to(ROOT.resolve())),
                  key=lambda path: path.as_posix())


def create_run_record(output: Path, *, mode: str, seed: int | None) -> dict[str, Any]:
    """Snapshot code and provenance before an output manager starts a new run."""
    output = Path(output)
    if (output / "run.json").exists() or (output / "source").exists():
        raise FileExistsError(f"Experiment provenance already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Also protect explicitly requested output folders outside the managed runs
    # tree from an ordinary `git add`; the marker ignores itself and all data.
    marker = output / ".gitignore"
    try:
        with marker.open("x", encoding="utf-8") as handle:
            handle.write("# Generated experiment artifacts stay local.\n*\n")
    except FileExistsError:
        pass
    hashes = {}
    for path in _source_files():
        relative = path.relative_to(ROOT)
        content = path.read_bytes()
        hashes[relative.as_posix()] = hashlib.sha256(content).hexdigest()
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    record = {
        "schema_version": 1, "run_id": output.name, "created_utc": _now(),
        "status": "running", "mode": mode, "seed": seed,
        "command": list(getattr(sys, "orig_argv", sys.argv)),
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "package_versions": packages, "git": _git_provenance(),
        "source_snapshot": "source", "source_files_sha256": hashes,
        "source_sha256": _digest(hashes),
    }
    _atomic_json(output / "run.json", record)
    return record


def finish_run_record(output: Path, *, status: str, error: str | None = None) -> dict[str, Any]:
    """Finalize provenance without modifying checkpoint, training, or log bytes."""
    output = Path(output)
    record, problem = _read_json(output / "run.json")
    if problem or not record:
        raise ValueError(problem or f"Missing experiment provenance: {output}")
    training, _ = _read_json(output / "training.json")
    terminal = {"completed", "trial_completed", "failed", "interrupted"}
    exceptional = status in {"failed", "interrupted"}
    record["status"] = training.get("status") if not exceptional and training.get("status") in terminal else status
    record["finished_utc"] = (_now() if exceptional else training.get("finished_utc")) or _now()
    for key in ("actual_timesteps", "optimizer_steps", "update_count", "optimization_epochs",
                *CUMULATIVE_FIELDS, "training_seconds", "elapsed_seconds", "phase_seconds"):
        if key in training:
            record[key] = training[key]
    config = training.get("config", {})
    if isinstance(config, dict):
        for key in LINEAGE_FIELDS:
            if key in config:
                record[key] = config[key]
    if error is not None:
        record["error"] = str(error)
    elif "error" in training:
        record["error"] = training["error"]
    _atomic_json(output / "run.json", record)
    return record


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    results = report.get("results", {})
    if isinstance(results, dict):
        trained = results.get("trained", {})
        if isinstance(trained, dict) and isinstance(trained.get("summary"), dict):
            return trained["summary"]
    return report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}


def _evaluation_fields(report: dict[str, Any], *, kind: str, path: Path, root: Path) -> dict[str, Any]:
    summary = _summary(report)
    if not summary:
        return {}
    episodes = report.get("episodes", [])
    seeds = [row["seed"] for row in episodes
             if isinstance(row, dict) and isinstance(row.get("seed"), int)] if isinstance(episodes, list) else []
    if not seeds:
        results = report.get("results", {})
        trained = results.get("trained", {}) if isinstance(results, dict) else {}
        measured = trained.get("episodes", []) if isinstance(trained, dict) else []
        seeds = [row["seed"] for row in measured
                 if isinstance(row, dict) and isinstance(row.get("seed"), int)] if isinstance(measured, list) else []
    return {
        "eval_kind": kind,
        "eval_seed_start": report.get("evaluation_seed_start", min(seeds) if seeds else ""),
        "eval_seed_end": report.get("evaluation_seed_end", max(seeds) if seeds else ""),
        "eval_seed_sha256": _digest(seeds) if seeds else "",
        "eval_episodes": summary.get("episodes", ""),
        **{key: summary.get(key, "") for key in ("success_rate", "mean_return", "mean_final_distance_m")},
        "evaluation_path": path.relative_to(root).as_posix(),
        "evaluation_checkpoint_sha256": report.get("checkpoint_sha256", ""),
    }


def _run_row(directory: Path, root: Path) -> dict[str, Any]:
    record, record_error = _read_json(directory / "run.json")
    training, training_error = _read_json(directory / "training.json")
    config = training.get("config", {})
    config = config if isinstance(config, dict) else {}
    phases = training.get("phase_seconds", record.get("phase_seconds", {}))
    phases = phases if isinstance(phases, dict) else {}
    environment = training.get("environment", {})
    environment = environment if isinstance(environment, dict) else {}
    task = environment.get("task_config", {})
    task = task if isinstance(task, dict) else {}
    git = record.get("git", {})
    git = git if isinstance(git, dict) else {}
    mode = training.get("mode") or config.get("mode") or record.get("mode", "")
    if mode == "evaluation":
        report_path = directory / "evaluation.json"
        report, report_error = _read_json(report_path)
        environment = report.get("environment", {})
        environment = environment if isinstance(environment, dict) else {}
        task = environment.get("task_config", {})
        task = task if isinstance(task, dict) else {}
    else:
        report_error = ""
    row = {
        "run_id": record.get("run_id") or directory.relative_to(root).as_posix(),
        "path": directory.relative_to(root).as_posix(),
        "created_utc": record.get("created_utc") or config.get("started_utc", ""),
        "status": (record.get("status") if record.get("status") in {"completed", "trial_completed", "failed", "interrupted"}
                   else training.get("status") or record.get("status") or "unknown"),
        "mode": mode,
        "seed": config.get("seed", record.get("seed", "")), "task": task.get("task", ""),
        "environment_sha256": _digest(environment) if environment else "",
        "source_sha256": record.get("source_sha256", ""),
        "git_commit": git.get("commit") or "", "git_dirty": git.get("dirty", ""),
        "requested_timesteps": config.get("requested_timesteps", ""),
        "actual_timesteps": training.get("actual_timesteps", record.get("actual_timesteps", "")),
        "optimizer_steps": training.get("optimizer_steps", record.get("optimizer_steps", "")),
        "update_count": training.get("update_count", record.get("update_count", "")),
        "optimization_epochs": training.get("optimization_epochs", record.get("optimization_epochs", "")),
        **{key: training.get(key, record.get(key, "")) for key in CUMULATIVE_FIELDS},
        **{key: config.get(key, record.get(key, "")) for key in LINEAGE_FIELDS},
        "num_envs": config.get("num_envs", ""),
        "training_seconds": training.get("training_seconds", training.get("elapsed_seconds", "")),
        **{key: config.get(key, "") for key in (
            "learning_rate", "batch_size", "n_steps", "n_epochs", "gamma", "log_std_init",
            "reward_scale", "vec_env_backend", "target_kl", "use_sde", "sde_sample_freq",
        )},
        "rollout_seconds_including_callbacks": phases.get("rollout_including_callbacks", ""),
        **{f"{key}_seconds": phases.get(key, "") for key in ("optimization", "diagnostics", "validation", "heldout")},
        "metadata_error": "; ".join(error for error in (record_error, training_error, report_error) if error),
    }
    if mode == "evaluation":
        row.update(_evaluation_fields(report, kind="standalone", path=report_path, root=root))
        return row
    # A startup trial is explicitly not a task-performance measurement.
    if row["mode"] != "startup_trial":
        report_path = directory / "evaluation.json"
        report, report_error = _read_json(report_path)
        if _summary(report):
            row.update(_evaluation_fields(report, kind="heldout", path=report_path, root=root))
        else:
            validation_path = directory / "best_validation.json"
            validation, validation_error = _read_json(validation_path)
            row.update(_evaluation_fields(validation, kind="validation", path=validation_path, root=root))
            report_error = "; ".join(error for error in (report_error, validation_error) if error)
        if report_error:
            row["metadata_error"] = "; ".join(filter(None, (row["metadata_error"], report_error)))
    return row


def catalog_rows(root: Path) -> list[dict[str, Any]]:
    """Read legacy and new experiments, leaving every raw artifact untouched."""
    root = Path(root).resolve()
    if not root.exists():
        return []
    directories = {path.parent for name in ("training.json", "run.json") for path in root.rglob(name)
                   if "source" not in path.relative_to(root).parts and path.resolve().is_relative_to(root)}
    rows = [_run_row(directory, root) for directory in sorted(directories)]
    return sorted(rows, key=lambda row: (str(row.get("created_utc", "")), row["path"]))


def refresh_catalog(root: Path) -> list[dict[str, Any]]:
    """Rebuild the derived comparison index; the caller owns the writer lock."""
    root = Path(root)
    rows = catalog_rows(root)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(root / "index.csv", buffer.getvalue())
    return rows


def main() -> None:
    # Import only here, avoiding model dependencies and an output-manager cycle.
    from .outputs import RUNS_ROOT, OutputError, _writer_lock

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=RUNS_ROOT, help="Experiment root (default: runs).")
    args = parser.parse_args()
    try:
        with _writer_lock(args.root / ".current.lock"):
            rows = refresh_catalog(args.root)
    except OutputError as exc:
        parser.error(str(exc))
    columns = ("run_id", "mode", "seed", "status", "actual_timesteps", "eval_kind", "success_rate", "mean_final_distance_m")
    print("\t".join(columns))
    for row in rows:
        print("\t".join(str(row.get(column, "")) for column in columns))
    print(f"Comparison index: {args.root / 'index.csv'} ({len(rows)} experiments)")
    print("Compare matching task/environment signatures and evaluation seeds; blank metrics were not measured.")


if __name__ == "__main__":
    main()
