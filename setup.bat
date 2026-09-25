@echo off
setlocal

rem Find a Python launcher, in order of preference. If none is on PATH,
rem stop with a clear message. Without this check, pip install fails
rem later with a less clear error.
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

rem If venv already exists, reuse it. Otherwise, create it.
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
echo   python start.py data_json output

endlocal