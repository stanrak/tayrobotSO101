"""Separate experiment directories, latest-run discovery, and writer locks."""

from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
import os
import json
import re
import shutil
import sys
import tempfile
from uuid import uuid4


RUNS_ROOT = Path(__file__).resolve().parents[1] / "local-artifacts" / "runs"


class OutputError(RuntimeError):
    """An output destination cannot safely be used by this run."""


def current_output() -> Path:
    """Find the latest managed experiment, with read-only legacy fallback."""
    pointer = RUNS_ROOT / "latest.json"
    if not pointer.exists():
        return _inside(RUNS_ROOT / "current", RUNS_ROOT)
    try:
        record = json.loads(pointer.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("Latest-run pointer must be a JSON object")
        relative = Path(record["path"])
        if record.get("schema_version") != 1 or relative.is_absolute():
            raise ValueError("Unsupported or absolute latest-run pointer")
        output = _inside(RUNS_ROOT / relative, RUNS_ROOT)
        if not output.is_dir():
            raise ValueError("Latest experiment directory is missing")
        return output
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise OutputError(f"Invalid latest-run pointer: {pointer}: {exc}") from exc


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved == root.resolve() or not resolved.is_relative_to(root.resolve()):
        raise OutputError(f"Output path must remain inside {root}: {path}")
    return resolved


def _write_json(path: Path, value: dict) -> None:
    """Publish a small pointer atomically; experiment artifacts stay in place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _new_directory(parent: Path, mode: str, seed: int | None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", mode):
        raise OutputError("Experiment mode must contain only letters, numbers, '_' or '-'.")
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        name = f"{_stamp()}_{mode}_seed{seed if seed is not None else 'none'}_{uuid4().hex[:8]}"
        destination = parent / name
        try:
            destination.mkdir()
            return destination
        except FileExistsError:
            continue
    raise OutputError(f"Could not allocate a unique experiment directory under {parent}.")


@contextmanager
def _experiment_record(output: Path, *, mode: str, seed: int | None):
    from .experiments import create_run_record, finish_run_record, refresh_catalog

    create_run_record(output, mode=mode, seed=seed)
    status, error = "completed", None
    try:
        refresh_catalog(RUNS_ROOT)
        yield
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            finish_run_record(output, status=status, error=error)
        except Exception as exc:
            if error is None:
                raise
            print(f"Could not finalize experiment metadata at {output}: {exc}", file=sys.stderr)
        try:
            refresh_catalog(RUNS_ROOT)
        except Exception as exc:
            # This table is derived from retained raw records and can be rebuilt.
            print(f"Could not refresh comparison index; rebuild with python -m so101_sim.experiments: {exc}", file=sys.stderr)


@contextmanager
def _writer_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # OS locks are released even if Python crashes; the harmless lock file stays.
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise OutputError("This output is in use. Stop the existing training or evaluation before starting another writer.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def training_output(requested: Path | None = None, *, mode: str = "training", seed: int | None = None):
    # Keep the existing lock name so an older running trainer remains protected.
    managed = requested is None or requested.resolve() == (RUNS_ROOT / "current").resolve()
    with ExitStack() as stack:
        stack.enter_context(_writer_lock(RUNS_ROOT / ".current.lock"))
        if managed:
            parent = _inside(RUNS_ROOT / "experiments", RUNS_ROOT)
            output = _new_directory(parent, mode, seed)
        else:
            output = requested.resolve()
            stack.enter_context(_writer_lock(output.parent / f".{output.name}.lock"))
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise OutputError(f"Custom output must be an empty directory: {output}. Omit --output for a new experiment.")
            output.mkdir(parents=True, exist_ok=True)
        with _experiment_record(output, mode=mode, seed=seed):
            if managed:
                _write_json(RUNS_ROOT / "latest.json", {
                    "schema_version": 1, "run_id": output.name,
                    "path": output.relative_to(RUNS_ROOT.resolve()).as_posix(),
                })
            print(f"Experiment output: {output}", flush=True)
            yield output


def current_checkpoint() -> Path:
    for name in ("best_model.zip", "trial_model.zip", "final_model.zip"):
        checkpoint = current_output() / name
        if checkpoint.is_file():
            return checkpoint
    raise OutputError("No checkpoint is available in the latest experiment (or legacy local-artifacts/runs/current). Finish a startup trial or wait for a checkpoint.")


@contextmanager
def evaluation_output(requested: Path | None = None, *, checkpoint: Path | None = None, seed: int | None = None):
    with ExitStack() as stack:
        stack.enter_context(_writer_lock(RUNS_ROOT / ".current.lock"))
        checkpoint = (checkpoint or current_checkpoint()).resolve()
        if checkpoint.suffix != ".zip":
            checkpoint = checkpoint.with_suffix(".zip")
        if not checkpoint.is_file():
            raise OutputError(f"Checkpoint does not exist: {checkpoint}")
        stack.enter_context(_writer_lock(checkpoint.parent / ".evaluation.lock"))
        if requested is not None:
            output = requested.resolve()
            stack.enter_context(_writer_lock(output.parent / f".{output.name}.lock"))
            if output.exists():
                raise OutputError(f"Evaluation output already exists: {output}. Omit --output for a unique report.")
            output.parent.mkdir(parents=True, exist_ok=True)
        evaluations = checkpoint.parent / "evaluations"
        if checkpoint.is_relative_to(RUNS_ROOT.resolve()):
            evaluations = _inside(evaluations, RUNS_ROOT)
        record_dir = _new_directory(evaluations, "evaluation", seed)
        canonical = record_dir / "evaluation.json"
        if requested is None:
            output = canonical
        with _experiment_record(record_dir, mode="evaluation", seed=seed):
            print(f"Evaluation output: {output}", flush=True)
            try:
                yield output
            finally:
                # Preserve an explicit report path while retaining a canonical
                # per-evaluation record next to the evaluated checkpoint.
                if output != canonical and output.is_file():
                    shutil.copyfile(output, canonical)


def archive_evaluation(output: Path) -> None:
    """Compatibility guard: an existing report must never be overwritten."""
    if output.exists():
        raise OutputError(f"Evaluation report already exists: {output}. Use a new experiment report path.")
