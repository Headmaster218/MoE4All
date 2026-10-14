[CmdletBinding()]
param([string]$OutputDirectory)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $root 'artifacts\agent-release' }
$testRoot = Join-Path $OutputDirectory ('launcher-test-' + [guid]::NewGuid().ToString('N'))
$fixtureRoot = Join-Path $testRoot ('path with spaces ' + [char]0x4e2d + [char]0x6587)
$scriptRoot = Join-Path $fixtureRoot 'harness\scripts'
New-Item -ItemType Directory -Path $scriptRoot -Force | Out-Null
$launcher = Join-Path $fixtureRoot 'MoE4All.exe'
& (Join-Path $PSScriptRoot 'Build-AgentLauncher.ps1') -OutputPath $launcher -Console

function Write-FixtureScript([string]$name, [string]$content) {
    $path = Join-Path $scriptRoot $name
    [IO.File]::WriteAllText($path, $content, (New-Object Text.UTF8Encoding($true)))
    Set-Content -LiteralPath $path -Stream Zone.Identifier -Value "[ZoneTransfer]`r`nZoneId=3"
}

function Invoke-FixtureLauncher {
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $launcher
    $start.Arguments = '--no-pause'
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.StandardOutputEncoding = New-Object Text.UTF8Encoding($false)
    $start.StandardErrorEncoding = New-Object Text.UTF8Encoding($false)
    $start.EnvironmentVariables['PSModulePath'] = Join-Path $testRoot 'missing-modules'
    $child = New-Object Diagnostics.Process
    $child.StartInfo = $start
    try {
        if (-not $child.Start()) { throw 'Cannot start the fixture launcher.' }
        $stdout = $child.StandardOutput.ReadToEndAsync()
        $stderr = $child.StandardError.ReadToEndAsync()
        if (-not $child.WaitForExit(60000)) {
            $child.Kill()
            throw 'The fixture launcher exceeded its timeout.'
        }
        return [pscustomobject]@{
            ExitCode = $child.ExitCode
            Output = $stdout.GetAwaiter().GetResult() + $stderr.GetAwaiter().GetResult()
        }
    } finally {
        $child.Dispose()
    }
}

$missing = Invoke-FixtureLauncher
if ($missing.ExitCode -ne 3 -or $missing.Output -notmatch 'runtime is incomplete') {
    throw "Missing-runtime diagnostics failed: $($missing.Output)"
}

Write-FixtureScript 'Prepare-BundledRuntime.ps1' @'
param([string]$InstallRoot, [string]$StateRoot)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$null = Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256
New-Item -ItemType Directory -Path (Join-Path $StateRoot 'home\profiles\web\node_modules') -Force | Out-Null
Write-Host 'prepare-ok'
exit 0
'@
Write-FixtureScript 'Run-Agent.ps1' @'
param([string]$StateRoot, [switch]$Open)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
if ($Open) { throw 'The desktop EXE unexpectedly requested browser opening.' }
if ((Split-Path -Leaf $StateRoot) -ne 'data') { throw 'The EXE did not use the unified data directory.' }
$null = Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256
Write-Host 'run-ok'
exit 0
'@

$success = Invoke-FixtureLauncher
if ($success.ExitCode -ne 0 -or $success.Output -notmatch 'prepare-ok' -or $success.Output -notmatch 'run-ok') {
    throw "Downloaded-script/module-path launch failed: $($success.Output)"
}

$windowedLauncher = Join-Path $fixtureRoot 'MoE4All-windowed.exe'
& (Join-Path $PSScriptRoot 'Build-AgentLauncher.ps1') -OutputPath $windowedLauncher
$windowed = Start-Process -FilePath $windowedLauncher -ArgumentList '--no-pause' -WorkingDirectory $fixtureRoot -PassThru -Wait
if ($windowed.ExitCode -ne 0) {
    throw "Windowed launcher failed without a console handle: exit $($windowed.ExitCode)"
}

Write-FixtureScript 'Prepare-BundledRuntime.ps1' @'
param([string]$InstallRoot, [string]$StateRoot)
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
throw ('prepare-error-' + [char]0x4e2d + [char]0x6587)
'@
$failed = Invoke-FixtureLauncher
$expectedError = 'prepare-error-' + [char]0x4e2d + [char]0x6587
$logPath = Join-Path $fixtureRoot 'data\logs\launcher.log'
$logContent = Get-Content -LiteralPath $logPath -Raw -Encoding UTF8
if ($failed.ExitCode -eq 0 -or -not $failed.Output.Contains($expectedError) -or
    -not $logContent.Contains($expectedError) -or $failed.Output -notmatch 'Startup log:') {
    throw "Child-error logging/encoding failed: $($failed.Output)"
}
Write-Host 'Launcher tests passed: missing runtime, unified data root, no browser launch, windowed startup without a console, downloaded scripts, isolated module path, child error logging, UTF-8 output.'
Write-Host "Evidence: $testRoot"
