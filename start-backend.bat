@echo off
title Pack Web - Backend
echo =========================================
echo    Pack Web - Backend Service
echo =========================================
echo.
echo Activating conda environment: pack
call conda activate pack
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment 'pack'
    echo Please create it first: conda create -n pack python=3.9
    pause
    exit /b 1
)
echo [OK] conda environment activated
echo.

cd /d "%~dp0web\backend"
echo Current directory: %cd%
echo.
echo Starting FastAPI backend (http://0.0.0.0:8000)
echo API docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo =========================================
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

if errorlevel 1 (
    echo.
    echo [ERROR] Backend startup failed
    echo Check dependencies: pip install -r requirements.txt
    pause
)
