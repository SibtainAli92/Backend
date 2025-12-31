@echo off
REM Startup script for FastAPI backend on Windows

echo Starting FastAPI Backend...
echo.

REM Load environment variables and start server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

pause
