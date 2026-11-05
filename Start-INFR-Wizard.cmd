@echo off
setlocal
title MoE4All Launch Wizard
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\infr-wizard.ps1" -ClassicWizard %*
set "MOE4ALL_WIZARD_EXIT=%ERRORLEVEL%"
if "%MOE4ALL_WIZARD_EXIT%"=="42" exit /b 0
if "%MOE4ALL_WIZARD_EXIT%"=="43" exit /b 0
if "%MOE4ALL_WIZARD_EXIT%"=="44" exit /b 0
if "%MOE4ALL_WIZARD_EXIT%"=="45" exit /b 0
if "%MOE4ALL_WIZARD_EXIT%"=="46" exit /b 1
echo.
if not "%MOE4ALL_WIZARD_EXIT%"=="0" echo MoE4All launch failed ^(exit code %MOE4ALL_WIZARD_EXIT%^). Review the messages above.
pause
exit /b %MOE4ALL_WIZARD_EXIT%
