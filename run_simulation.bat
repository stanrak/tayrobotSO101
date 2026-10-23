@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m so101_sim %*
) else (
    python -m so101_sim %*
)
set "simulation_exit_code=%ERRORLEVEL%"

if not "%simulation_exit_code%"=="0" (
    echo.
    echo The simulation could not start. Read the error above.
    echo Install the dependencies with your selected Python:
    echo     python -m pip install -r requirements.txt
    echo See README.md for setup and laptop keyboard controls.
    pause
)
exit /b %simulation_exit_code%
