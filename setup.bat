@echo off
setlocal

rem Finds a Python launcher, in order of preference, and stops with a
rem clear message if none is on PATH -- pip install fails with a much
rem more confusing error if this step is skipped.
where py >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON=py
) else (
    where python >nul 2>&1
    if %errorlevel% == 0 (
        set PYTHON=python
    ) else (
        echo Python was not found on PATH. Install Python 3.9+ and re-run this script.
        exit /b 1
    )
)

rem Reuse an existing venv untouched; only create one if it's missing.
if exist venv\Scripts\activate.bat (
    echo Using existing virtual environment in venv\
) else (
    echo Creating virtual environment in venv\...
    %PYTHON% -m venv venv
    if errorlevel 1 (
        echo Failed to create the virtual environment.
        exit /b 1
    )
)

call venv\Scripts\activate.bat

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. See the error above.
    exit /b 1
)

echo.
echo Setup complete. To run the pipeline:
echo   venv\Scripts\activate.bat
echo   python pipeline.py data_json output

endlocal