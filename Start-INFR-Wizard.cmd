@echo off
setlocal
powershell.exe -NoLogo -NoProfile -File "%~dp0scripts\infr-wizard.ps1" %*
set "INFR_WIZARD_EXIT=%ERRORLEVEL%"
echo.
if not "%INFR_WIZARD_EXIT%"=="0" echo INFR Wizard exited with code %INFR_WIZARD_EXIT%.
pause
exit /b %INFR_WIZARD_EXIT%
