@echo off
setlocal
title MoE4All Launch Wizard
powershell.exe -NoLogo -NoProfile -File "%~dp0scripts\infr-wizard.ps1" %*
set "MOE4ALL_WIZARD_EXIT=%ERRORLEVEL%"
echo.
if not "%MOE4ALL_WIZARD_EXIT%"=="0" echo MoE4All Wizard exited with code %MOE4ALL_WIZARD_EXIT%.
pause
exit /b %MOE4ALL_WIZARD_EXIT%
