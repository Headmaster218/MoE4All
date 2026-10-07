@echo off
setlocal EnableExtensions
title MoE4All Launch Wizard

rem ============================================================================
rem  Uncensor (refusal-direction projection)
rem ----------------------------------------------------------------------------
rem  Removes one direction per layer from a Qwen3.8 (qwen4exp) model's wide
rem  residual, which is what stops it declining prompts it can answer. Costs a
rem  few tenths of a percent per token. It is OFF unless a control-vector file
rem  is named (a .gguf whose general.architecture is `controlvector`), so no
rem  other model - and no session that ignores this - changes at all.
rem
rem  The four variables below are the switch:
rem    UNCENSOR         on | off | ask  ('ask' prompts at launch; the default)
rem    UNCENSOR_VECTOR  the control-vector .gguf. Empty = look for a
rem                     *cvec*.gguf / *uncensor*.gguf next to this script and
rem                     in .\models, then use the path remembered from the last
rem                     time it was chosen.
rem    UNCENSOR_FIRST   first layer projected, inclusive (empty = engine's 4)
rem    UNCENSOR_LAST    last layer projected, inclusive (empty = engine's 44)
rem
rem  For one run, without editing this file:
rem    Start-INFR-Wizard.cmd -uncensor off                   force OFF
rem    Start-INFR-Wizard.cmd -uncensor on                    force ON
rem    Start-INFR-Wizard.cmd -uncensor D:\models\cvec.gguf   ON, with that file
rem    Start-INFR-Wizard.cmd -no-uncensor                    same as '-uncensor off'
rem  A forced run does not overwrite the saved choice; everything except these
rem  arguments is handed to the wizard unchanged.
rem
rem  How it reaches the engine: `infr` reads INFR_UNCENSOR_VECTOR (plus
rem  INFR_UNCENSOR_FIRST_LAYER / INFR_UNCENSOR_LAST_LAYER) from its environment,
rem  and this script's environment is passed through to the wizard and from
rem  there to the engine untouched. Naming a file is the ONLY way the projection
rem  gets built in - there is no config-file key and no CLI flag for it,
rem  because the projection is allocated into the session at model load and
rem  cannot appear afterwards. OFF therefore means "the variable is not set";
rem  this engine also treats an empty value as off.
rem
rem  Startup decides whether the projection EXISTS. Each API request then
rem  decides whether it is ON: send "uncensor": false (or Strata's spelling,
rem  "experimental_speed_projection": false) to /v1/chat/completions or
rem  /v1/responses for one answer from the model as published. Omit the field
rem  and the session keeps running the way it is running now.
rem ============================================================================
set "UNCENSOR=ask"
set "UNCENSOR_VECTOR="
set "UNCENSOR_FIRST="
set "UNCENSOR_LAST="

set "STATE_DIR=%~dp0gui-data"
set "STATE=%STATE_DIR%\uncensor.txt"
set "LAST_ENABLED="
set "LAST_VECTOR="
if exist "%STATE%" for /f "usebackq tokens=1,* delims==" %%A in ("%STATE%") do (
    if /i "%%A"=="enabled" set "LAST_ENABLED=%%B"
    if /i "%%A"=="vector" set "LAST_VECTOR=%%B"
)

set "FORWARD="
:arg_loop
if "%~1"=="" goto arg_done
if /i "%~1"=="-no-uncensor" (set "UNCENSOR=off" & shift & goto arg_loop)
if /i "%~1"=="-uncensor" (shift & goto arg_value)
set "FORWARD=%FORWARD% "%~1""
shift
goto arg_loop

rem -uncensor takes an optional value: on, off, or the direction file to use.
:arg_value
if "%~1"=="" (set "UNCENSOR=on" & goto arg_loop)
if /i "%~1"=="on" (set "UNCENSOR=on" & shift & goto arg_loop)
if /i "%~1"=="off" (set "UNCENSOR=off" & shift & goto arg_loop)
rem A substring needs a plain variable; `%~1:~0,1%` is not a thing in cmd. An
rem argument that starts with `-` is the NEXT option, not this one's value.
set "ARG=%~1"
if "%ARG:~0,1%"=="-" (set "UNCENSOR=on" & goto arg_loop)
set "UNCENSOR_VECTOR=%ARG%"
set "UNCENSOR=on"
shift
goto arg_loop

:arg_done
if /i "%UNCENSOR%"=="on" (set "ENABLED=1" & goto resolve)
if /i "%UNCENSOR%"=="off" (set "ENABLED=0" & goto apply)

set "DEFAULT=N"
if /i "%LAST_ENABLED%"=="1" set "DEFAULT=Y"
>nul choice /c YN /n /m "Uncensor (refusal-direction projection)? [Y,N] " /t 30 /d %DEFAULT%
rem 1 = Y, 2 = N. Anything else (no console, Ctrl+C) keeps the saved choice: a
rem prompt that failed is not a reason to change what the last run did.
if errorlevel 3 (
    set "ENABLED=0"
    if /i "%LAST_ENABLED%"=="1" set "ENABLED=1"
    goto resolve
)
if errorlevel 2 (set "ENABLED=0" & goto apply)
set "PERSIST=1"
set "ENABLED=1"

:resolve
set "VECTOR=%UNCENSOR_VECTOR%"
if not "%VECTOR%"=="" goto have_vector
set "VECTOR=%LAST_VECTOR%"
if not "%VECTOR%"=="" goto have_vector
for %%F in ("%~dp0*cvec*.gguf" "%~dp0*uncensor*.gguf" "%~dp0models\*cvec*.gguf" "%~dp0models\*uncensor*.gguf") do if not defined VECTOR set "VECTOR=%%~fF"

:have_vector
if not "%VECTOR%"=="" if exist "%VECTOR%" goto apply
echo Uncensor: OFF  - no control-vector file.
echo     Put one next to this script as *cvec*.gguf, set UNCENSOR_VECTOR above,
echo     or pass -uncensor ^<PATH^>. The model starts without it.
set "ENABLED=0"

:apply
if not "%ENABLED%"=="1" (
    set "INFR_UNCENSOR_VECTOR="
    set "INFR_UNCENSOR_FIRST_LAYER="
    set "INFR_UNCENSOR_LAST_LAYER="
    echo Uncensor: OFF
    goto save
)
set "INFR_UNCENSOR_VECTOR=%VECTOR%"
if not "%UNCENSOR_FIRST%"=="" set "INFR_UNCENSOR_FIRST_LAYER=%UNCENSOR_FIRST%"
if not "%UNCENSOR_LAST%"=="" set "INFR_UNCENSOR_LAST_LAYER=%UNCENSOR_LAST%"
echo Uncensor: ON  - %VECTOR%
set "CUSTOM_RANGE=0"
if not "%UNCENSOR_FIRST%"=="" set "CUSTOM_RANGE=1"
if not "%UNCENSOR_LAST%"=="" set "CUSTOM_RANGE=1"
if "%CUSTOM_RANGE%"=="1" (
    echo     layers: %UNCENSOR_FIRST%..%UNCENSOR_LAST%  (an empty side = engine default^)
) else (
    echo     layers: engine default 4-44
)
echo     one answer without it: post "uncensor": false to /v1/chat/completions

rem Only an interactive choice is remembered, so one forced run cannot quietly
rem become the new default. The engine keeps its own state elsewhere.
:save
if not "%PERSIST%"=="1" goto launch
if not exist "%STATE_DIR%" mkdir "%STATE_DIR%" 2>nul
> "%STATE%" (
    echo enabled=%ENABLED%
    echo vector=%VECTOR%
) 2>nul

:launch
powershell.exe -NoLogo -NoProfile -File "%~dp0scripts\infr-wizard.ps1" %FORWARD%
set "MOE4ALL_WIZARD_EXIT=%ERRORLEVEL%"
if "%MOE4ALL_WIZARD_EXIT%"=="42" exit /b 0
echo.
if not "%MOE4ALL_WIZARD_EXIT%"=="0" echo MoE4All Wizard exited with code %MOE4ALL_WIZARD_EXIT%.
pause
exit /b %MOE4ALL_WIZARD_EXIT%
