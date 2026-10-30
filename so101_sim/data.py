"""Cache one finalized managed experiment with the already-installed DVC CLI."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

from . import outputs


ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"completed", "trial_completed", "interrupted", "failed"}
RESERVED = {"private", "private-example", ".git", ".dvc", ".venv"}


def _regular_path(path: Path) -> None:
    """Reject links and Windows junctions before reading or walking a target."""
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
        raise outputs.OutputError("DVC snapshots cannot contain links or reparse points.")
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise outputs.OutputError("DVC snapshots require regular files and directories.")


def _finished_record(path: Path, *, modes: set[str]) -> None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("schema_version") != 1 or record.get("status") not in TERMINAL
                or record.get("mode") not in modes):
            raise ValueError("not a finalized supported experiment")
        finished = datetime.fromisoformat(record["finished_utc"])
        if finished.utcoffset() is None:
            raise ValueError("missing finish timezone")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise outputs.OutputError(
            "A valid finalized run.json is required; running or abruptly killed runs cannot be archived."
        ) from exc


def _target(requested: str) -> Path:
    # Inspect each unresolved component first: resolve() would hide a junction.
    if requested == "latest":
        for path in (outputs.RUNS_ROOT, outputs.RUNS_ROOT / "latest.json"):
            _regular_path(path)
        outputs.current_output()  # Validate the existing pointer contract.
        pointer = json.loads((outputs.RUNS_ROOT / "latest.json").read_text(encoding="utf-8"))
        path = outputs.RUNS_ROOT / pointer["path"]
    else:
        path = Path(requested)
    if not path.is_absolute():
        path = ROOT / path
    path = Path(os.path.abspath(path))
    root = ROOT.resolve()
    if not path.is_relative_to(root):
        raise outputs.OutputError("Select a managed run inside local-artifacts/runs/experiments.")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        _regular_path(current)
    resolved = path.resolve()
    managed = (outputs.RUNS_ROOT / "experiments").resolve()
    if resolved.parent != managed or not resolved.is_dir():
        raise outputs.OutputError("Only direct run directories in local-artifacts/runs/experiments are supported.")
    return resolved


def _validate_tree(run: Path) -> None:
    # os.walk does not follow links; inspect children before descending anyway.
    def unreadable(error: OSError) -> None:
        raise error

    for directory, folders, files in os.walk(run, followlinks=False, onerror=unreadable):
        for name in folders + files:
            if name.casefold() in RESERVED:
                raise outputs.OutputError("The run contains a reserved directory or file.")
            _regular_path(Path(directory) / name)
    _finished_record(run / "run.json", modes={"full_training", "startup_trial"})
    evaluations = run / "evaluations"
    if evaluations.exists():
        for evaluation in evaluations.iterdir():
            if not evaluation.is_dir():
                raise outputs.OutputError("Unexpected file in the evaluations directory.")
            _finished_record(evaluation / "run.json", modes={"evaluation"})


def add_run(requested: str = "latest") -> Path:
    """Keep an immutable per-run pointer, holding the existing writer lock."""
    executable = shutil.which("dvc")
    if executable is None:
        raise outputs.OutputError("DVC is not on PATH. Open a terminal where dvc --version works.")
    if not (ROOT / ".dvc" / "config").is_file():
        raise outputs.OutputError("The project DVC configuration is missing.")
    # Reject redirected parents and lock files before opening the writer lock.
    runs = Path(os.path.abspath(outputs.RUNS_ROOT))
    if not runs.is_relative_to(ROOT.resolve()):
        raise outputs.OutputError("The managed runs directory must remain inside this project.")
    current = ROOT.resolve()
    for part in runs.relative_to(current).parts:
        current = current / part
        _regular_path(current)
    lock = runs / ".current.lock"
    if lock.exists() or lock.is_symlink():
        _regular_path(lock)
    with outputs._writer_lock(lock):
        run = _target(requested)
        manifest = run.with_name(run.name + ".dvc")
        if manifest.exists() or manifest.is_symlink():
            raise outputs.OutputError(
                "This run already has a DVC snapshot. Use dvc status to inspect it; the snapshot will not be overwritten."
            )
        _validate_tree(run)
        environment = os.environ.copy()
        environment["DVC_NO_ANALYTICS"] = "1"
        relative = run.relative_to(ROOT).as_posix()
        try:
            # A renamed sidecar still owns its output. Do not let add update it.
            listed = subprocess.run(
                [executable, "list", ".", relative, "--dvc-only", "--json"],
                cwd=ROOT, env=environment, check=True, timeout=3600,
                capture_output=True, text=True,
            )
            tracked = json.loads(listed.stdout)
            if not isinstance(tracked, list):
                raise ValueError("Invalid DVC output listing")
            if tracked:
                raise outputs.OutputError("DVC already tracks this run; its existing snapshot will not be changed.")
            subprocess.run(
                [executable, "add", "--quiet", "--no-relink", relative],
                cwd=ROOT, env=environment, check=True, timeout=3600,
            )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise outputs.OutputError(
                "DVC could not finish the local snapshot. Inspect dvc status before retrying; retained run files are unchanged."
            ) from exc
        if not manifest.is_file():
            raise outputs.OutputError("DVC did not create the expected run pointer; the snapshot is not complete.")
        _regular_path(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Save one finalized run to the local DVC cache.")
    add.add_argument("run", nargs="?", default="latest", help="latest or local-artifacts/runs/experiments/<run_directory>.")
    args = parser.parse_args()
    try:
        manifest = add_run(args.run)
    except (outputs.OutputError, OSError) as exc:
        parser.error(str(exc))
    print(f"Local DVC snapshot: {manifest.relative_to(ROOT).as_posix()}")
    print("Keep the .dvc pointer and .dvc/cache together. No remote upload was performed.")


if __name__ == "__main__":
    main()
