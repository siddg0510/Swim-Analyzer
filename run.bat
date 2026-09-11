@echo off
if "%1"=="web" (
    echo Starting Swim Analyzer Web App on http://localhost:8000 ...
    python -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
) else if "%1"=="--web" (
    echo Starting Swim Analyzer Web App on http://localhost:8000 ...
    python -m uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
) else if "%1"=="main.py" (
    python main.py %2 %3 %4
) else (
    python main.py %*
)
