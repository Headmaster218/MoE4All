@echo off
setlocal
title INFR Console
cd /d "%~dp0"

if not exist "%~dp0infr-console.exe" (
    echo infr-console.exe was not found next to this script.
    pause
    exit /b 1
)

rem Prefer the publicly trusted Tailscale certificate: browsers let you click past an untrusted
rem certificate, but third-party API clients fail the TLS handshake and report a network error.
set "CERT=%APPDATA%\infr-console\certs\console-cert.pem"
if exist "%~dp0use-tailscale-cert.cmd" (
    call "%~dp0use-tailscale-cert.cmd"
    echo.
)
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
