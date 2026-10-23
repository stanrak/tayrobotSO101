"""Start keyboard control with dependency checks for the selected Python."""

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULES = ("numpy", "mujoco", "gymnasium")


def _project_python() -> Path | None:
    environment = ROOT / ".venv"
    if Path(sys.prefix).resolve() == environment.resolve():
        return None
    executable = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return executable if executable.is_file() else None


def main() -> int:
    missing = [name for name in RUNTIME_MODULES if find_spec(name) is None]
    if missing:
        print(f"Missing simulation dependencies: {', '.join(missing)}", file=sys.stderr)
        print(f"Current Python: {sys.executable}", file=sys.stderr)
        project_python = _project_python()
        if project_python is not None:
            print(f"Starting with the project Python: {project_python}", file=sys.stderr, flush=True)
            return subprocess.run(
                [str(project_python), "-m", "so101_sim", *sys.argv[1:]],
                cwd=ROOT,
                check=False,
            ).returncode
        executable = str(sys.executable).replace("'", "''")
        requirements = str(ROOT / "requirements-sim.txt").replace("'", "''")
        print("Install the simulation packages into this Python from PowerShell:", file=sys.stderr)
        print(f"  & '{executable}' -m pip install -r '{requirements}'", file=sys.stderr)
        return 1
    import_module("so101_sim.teleop").main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
