@echo off
setlocal enabledelayedexpansion
title Pack Web - Stop All
echo =========================================
echo    Pack Web - Stop All Services
echo =========================================
echo.

echo Stopping backend (port 8000)...
for /f "tokens=1-5" %%a in ('netstat -ano ^| findstr ":8000"') do (
    set "pid=%%e"
    if not "!pid!"=="" (
        taskkill /F /PID !pid! >nul 2>&1
        if !errorlevel! equ 0 (
            echo Stopped backend process: !pid!
        )
    )
)

echo.
echo Stopping frontend (port 5173)...
for /f "tokens=1-5" %%a in ('netstat -ano ^| findstr ":5173"') do (
    set "pid=%%e"
    if not "!pid!"=="" (
        taskkill /F /PID !pid! >nul 2>&1
        if !errorlevel! equ 0 (
            echo Stopped frontend process: !pid!
        )
    )
)

echo.
echo Stopping Python backend processes...
taskkill /F /IM python.exe >nul 2>&1
taskkill /F /IM pythonw.exe >nul 2>&1

echo.
echo Stopping frontend node processes...
taskkill /F /IM node.exe >nul 2>&1

echo.
echo =========================================
echo    All services stopped
echo =========================================
echo.
pause