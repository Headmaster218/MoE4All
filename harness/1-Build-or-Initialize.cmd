@echo off
setlocal

set "SCRIPT=%~dp0scripts\Build-DevHarness.ps1"

where pwsh.exe >nul 2>&1
if errorlevel 1 goto windows_powershell

pwsh.exe -NoLogo -NoProfile -File "%SCRIPT%"
goto finished

:windows_powershell
powershell.exe -NoLogo -NoProfile -File "%SCRIPT%"

:finished
set "RESULT=%ERRORLEVEL%"
echo.
if not "%RESULT%"=="0" echo Build or initialization failed with exit code %RESULT%.
if "%RESULT%"=="0" echo Build and initialization completed successfully.
pause
exit /b %RESULT%
