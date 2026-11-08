@echo off
setlocal

cd /d "%~dp0"

set "PYTHON=python"
if exist "%~dp0ingestion\.venv\Scripts\python.exe" set "PYTHON=%~dp0ingestion\.venv\Scripts\python.exe"

if "%PYTHON%"=="python" (
    where python >nul 2>&1
    if errorlevel 1 goto :python_missing
) else if not exist "%PYTHON%" goto :python_missing

echo Starting Infochat API at http://127.0.0.1:8000
"%PYTHON%" -m uvicorn api:app --reload --host 127.0.0.1 --port 8000

if errorlevel 1 (
    echo.
    echo The API stopped with an error.
    pause
    exit /b 1
)

endlocal
exit /b 0

:python_missing
    echo Python was not found. Install Python and ensure it is available on PATH.
    pause
    exit /b 1
