@echo off
setlocal
title MoE4All Launch Wizard
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\infr-wizard.ps1" %*
set "MOE4ALL_WIZARD_EXIT=%ERRORLEVEL%"
if "%MOE4ALL_WIZARD_EXIT%"=="42" exit /b 0
if "%MOE4ALL_WIZARD_EXIT%"=="43" exit /b 0
echo.
if not "%MOE4ALL_WIZARD_EXIT%"=="0" echo MoE4All launch failed ^(exit code %MOE4ALL_WIZARD_EXIT%^). Review the messages above.
pause
exit /b %MOE4ALL_WIZARD_EXIT%
