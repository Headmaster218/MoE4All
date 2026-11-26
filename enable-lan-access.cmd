@echo off
setlocal
rem Allow other devices on the LAN to reach INFR Console.
rem Windows Firewall blocks inbound connections by default, and a denied prompt leaves a
rem permanent block rule behind, so this also removes stale block rules for infr-console.exe.
rem Run it from the package folder; it asks for administrator rights itself.

net session >nul 2>&1
if not %errorlevel%==0 (
    echo Requesting administrator rights...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

set "EXE=%~dp0infr-console.exe"
if not exist "%EXE%" (
    echo infr-console.exe was not found next to this script.
    echo Put this file in the same folder as infr-console.exe and run it again.
    pause
    exit /b 1
)

echo Removing stale block rules for infr-console.exe ...
netsh advfirewall firewall delete rule name="infr-console.exe" >nul 2>&1
netsh advfirewall firewall delete rule name="INFR Console" >nul 2>&1
netsh advfirewall firewall delete rule name="INFR Console TCP" >nul 2>&1

echo Adding inbound allow rules ...
netsh advfirewall firewall add rule name="INFR Console" dir=in action=allow program="%EXE%" enable=yes profile=private,domain
netsh advfirewall firewall add rule name="INFR Console TCP" dir=in action=allow protocol=TCP localport=8090 profile=private,domain

echo.
echo Done. Start infr-console.exe again, then open http://YOUR-LAN-IP:8090 from another device.
echo If it still fails, check that the network profile is Private and that the router does not
echo isolate wireless clients.
pause
