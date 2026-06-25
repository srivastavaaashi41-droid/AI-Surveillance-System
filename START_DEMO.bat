@echo off
title Pinakin Infra - AI Surveillance System
color 0A
cls
echo ============================================================
echo    PINAKIN INFRA - AI SURVEILLANCE AND SECURITY SYSTEM
echo    Starting up, please wait...
echo ============================================================
echo.

REM Wait a moment then open the dashboard in the browser automatically
start "" cmd /c "timeout /t 8 >nul && start http://localhost:5000"

REM Start the main system (camera + AI + dashboard server)
python main.py

pause