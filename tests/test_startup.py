"""Check interpreter diagnostics and fallback without opening the viewer."""

from contextlib import redirect_stderr
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from so101_sim import __main__ as startup


class StartupTests(unittest.TestCase):
    def test_missing_gymnasium_reports_install_command_for_current_python(self):
        error = io.StringIO()
        interpreter = "C:/Python With Spaces/python.exe"
        with (
            patch.object(startup, "find_spec", side_effect=lambda name: None if name == "gymnasium" else object()),
            patch.object(startup, "_project_python", return_value=None),
            patch.object(startup.sys, "executable", interpreter),
            patch.object(startup, "import_module") as import_simulator,
            redirect_stderr(error),
        ):
            self.assertEqual(startup.main(), 1)
        self.assertIn("gymnasium", error.getvalue())
        self.assertIn(f"& '{interpreter}' -m pip install", error.getvalue())
        self.assertIn("requirements-sim.txt", error.getvalue())
        import_simulator.assert_not_called()

    def test_missing_dependencies_use_project_python_and_preserve_arguments_and_exit_code(self):
        interpreter = Path("project/.venv/Scripts/python.exe")
        with (
            patch.object(startup, "find_spec", return_value=None),
            patch.object(startup, "_project_python", return_value=interpreter),
            patch.object(startup.sys, "argv", ["so101_sim", "--seconds", "10"]),
            patch.object(startup.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as run,
            patch.object(startup, "import_module") as import_simulator,
            redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(startup.main(), 7)
        run.assert_called_once_with(
            [str(interpreter), "-m", "so101_sim", "--seconds", "10"],
            cwd=startup.ROOT, check=False,
        )
        import_simulator.assert_not_called()

    def test_project_environment_does_not_relaunch_itself(self):
        with patch.object(startup.sys, "prefix", str(startup.ROOT / ".venv")):
            self.assertIsNone(startup._project_python())

    def test_available_dependencies_open_manual_mode_without_subprocess(self):
        with (
            patch.object(startup, "find_spec", return_value=object()),
            patch.object(startup, "import_module") as import_simulator,
            patch.object(startup.subprocess, "run") as run,
        ):
            self.assertEqual(startup.main(), 0)
        import_simulator.assert_called_once_with("so101_sim.teleop")
        import_simulator.return_value.main.assert_called_once_with()
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
