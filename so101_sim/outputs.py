"""Stable current outputs with automatic history and exclusive writer access."""

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import os


RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"


class OutputError(RuntimeError):
    """An output destination cannot safely be used by this run."""


def current_output() -> Path:
    return RUNS_ROOT / "current"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if resolved == root.resolve() or not resolved.is_relative_to(root.resolve()):
        raise OutputError(f"Output path must remain inside {root}: {path}")
    return resolved


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
def training_output(requested: Path | None = None):
    managed = requested is None or requested.resolve() == current_output().resolve()
    output = current_output() if managed else requested.resolve()
    lock = RUNS_ROOT / ".current.lock" if managed else output.parent / f".{output.name}.lock"
    with _writer_lock(lock):
        if output.exists() and not output.is_dir():
            raise OutputError(f"Output must be a directory: {output}")
        if managed:
            _inside(output, RUNS_ROOT)
            if output.exists() and any(output.iterdir()):
                destination = _inside(RUNS_ROOT / "history" / _stamp(), RUNS_ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                output.rename(destination)
                print(f"Previous output archived: {destination}", flush=True)
        elif output.exists() and any(output.iterdir()):
            raise OutputError(f"Custom output is not empty: {output}. Omit --output to use managed runs/current.")
        output.mkdir(parents=True, exist_ok=True)
        print(f"Current output: {output}", flush=True)
        yield output


def current_checkpoint() -> Path:
    for name in ("best_model.zip", "trial_model.zip", "final_model.zip"):
        checkpoint = current_output() / name
        if checkpoint.is_file():
            return checkpoint
    raise OutputError("No checkpoint is available in runs/current yet. Finish a startup trial or wait for a training checkpoint.")


@contextmanager
def evaluation_output(requested: Path | None = None):
    output = requested.resolve() if requested is not None else current_output() / "evaluation.json"
    managed = output.resolve().is_relative_to(current_output().resolve())
    lock = RUNS_ROOT / ".current.lock" if managed else output.parent / f".{output.name}.lock"
    with _writer_lock(lock):
        if managed:
            _inside(output, RUNS_ROOT)
        output.parent.mkdir(parents=True, exist_ok=True)
        yield output


def archive_evaluation(output: Path) -> None:
    """Keep previous report bytes before publishing a new report at the same path."""
    if output.exists():
        if output.resolve().is_relative_to(current_output().resolve()):
            history = _inside(current_output() / "evaluation_history", RUNS_ROOT)
        else:
            history = output.parent / f"{output.stem}_history"
        history.mkdir(parents=True, exist_ok=True)
        output.rename(history / f"{_stamp()}.json")
