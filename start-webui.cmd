@echo off
setlocal
title INFR Console
cd /d "%~dp0"

if not exist "%~dp0infr-console.exe" (
    echo infr-console.exe was not found next to this script.
    pause
    exit /b 1
)

set "CERT=%APPDATA%\infr-console\certs\console-cert.pem"
if not exist "%CERT%" (
    echo No HTTPS certificate found - generating a self-signed one for this machine...
    call "%~dp0make-cert.cmd"
)
echo.
echo Starting INFR Console - HTTPS is used automatically when a certificate exists.
echo Press Ctrl+C in this window to stop it.
echo.
"%~dp0infr-console.exe" %*
if errorlevel 1 pause
