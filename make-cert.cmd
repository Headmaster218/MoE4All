@echo off
setlocal
rem Generate a self-signed certificate for HTTPS and store it under %APPDATA%\infr-console\certs.
rem Pass -Trust to also import it into the current user's Trusted Root store.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\make-cert.ps1" %*
if errorlevel 1 pause
