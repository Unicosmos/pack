@echo off
title Pack Web - Start All
echo =========================================
echo    Pack Web - Start All Services
echo =========================================
echo.

cd /d "%~dp0"

echo [1/3] Starting backend...
start "Pack Backend" cmd /k "%~dp0start-backend.bat"

timeout /t 2 /nobreak >nul

echo [2/3] Starting frontend...
start "Pack Frontend" cmd /k "%~dp0start-frontend.bat"

timeout /t 3 /nobreak >nul

echo [3/3] Done!
echo.
echo =========================================
echo    Service URLs
echo =========================================
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Frontend: http://localhost:5173
echo =========================================
echo.
echo Tips:
echo   - Closing this window won't affect the services
echo   - Run stop-all.bat to stop all services
echo.
pause
