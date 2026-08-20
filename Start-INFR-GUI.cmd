@echo off
setlocal
powershell.exe -NoLogo -NoProfile -File "%~dp0crates\infr-gui\Start-INFR-GUI.ps1" %*
set "INFR_GUI_EXIT=%ERRORLEVEL%"
if not "%INFR_GUI_EXIT%"=="0" (
  echo.
  echo INFR GUI exited with code %INFR_GUI_EXIT%.
  pause
)
exit /b %INFR_GUI_EXIT%
