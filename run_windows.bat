@echo off
echo ================================================
echo   MindSight - Mental Health AI System
echo ================================================
echo.

REM Check Python
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)

REM Create venv if it doesn't exist
IF NOT EXIST "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate venv
call venv\Scripts\activate

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt -q

REM Quick setup if model doesn't exist
IF NOT EXIST "models\bert_mental_health\config.json" (
    echo Setting up demo model...
    python scripts/quick_setup.py
)

REM Start server
echo.
echo Starting server at http://localhost:5000
echo Press Ctrl+C to stop.
echo.
python app.py
pause
