@echo off
setlocal
cd /d "%~dp0"

set "simulation_python=python"
if exist ".venv\Scripts\python.exe" set "simulation_python=.venv\Scripts\python.exe"
"%simulation_python%" -m so101_sim %*
set "simulation_exit_code=%ERRORLEVEL%"

if not "%simulation_exit_code%"=="0" (
    echo.
    echo The simulation could not start. Read the error above.
    echo Install the simulation dependencies with this same Python:
    echo     "%simulation_python%" -m pip install -r requirements-sim.txt
    echo See README.md for setup and laptop keyboard controls.
    pause
)
exit /b %simulation_exit_code%
