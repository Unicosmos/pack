@echo off
title Pack Web - Frontend
echo =========================================
echo    Pack Web - Frontend Service
echo =========================================
echo.
echo Activating conda environment: pack
call conda activate pack
if errorlevel 1 (
    echo [WARN] conda env 'pack' not activated, but trying anyway
)
echo.

cd /d "%~dp0web\frontend"
echo Current directory: %cd%
echo.

if not exist "node_modules" (
    echo First run, installing dependencies...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed
        pause
        exit /b 1
    )
)

echo Starting Vite dev server (http://localhost:5173)
echo.
echo Press Ctrl+C to stop
echo =========================================
echo.

npm run dev

if errorlevel 1 (
    echo.
    echo [ERROR] Frontend startup failed
    echo Check Node.js installation: node --version
    pause
)
