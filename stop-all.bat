@echo off
setlocal enabledelayedexpansion
title Pack Web - Stop All
echo =========================================
echo    Pack Web - Stop All Services
echo =========================================
echo.

echo Stopping backend (port 8000)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    taskkill /F /PID %%a >nul 2>&1
    if !errorlevel! equ 0 (
        echo Stopped backend process: %%a
    )
)

echo.
echo Stopping frontend (port 5173)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5173') do (
    taskkill /F /PID %%a >nul 2>&1
    if !errorlevel! equ 0 (
        echo Stopped frontend process: %%a
    )
)

echo.
echo Stopping node.exe...
taskkill /F /IM node.exe >nul 2>&1

echo.
echo =========================================
echo    All services stopped
echo =========================================
echo.
pause
