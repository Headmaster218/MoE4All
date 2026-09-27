@echo off
setlocal

set "SCRIPT=%~dp0scripts\Run-DevHarness.ps1"

where pwsh.exe >nul 2>&1
if errorlevel 1 goto windows_powershell

pwsh.exe -NoLogo -NoProfile -File "%SCRIPT%"
goto finished

:windows_powershell
powershell.exe -NoLogo -NoProfile -File "%SCRIPT%"

:finished
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" (
    echo.
    echo Harness stopped with exit code %RESULT%.
    pause
)
exit /b %RESULT%
