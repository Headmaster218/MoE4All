[CmdletBinding()]
param(
    [string]$WorkspaceRoot = 'D:\AISuperAssistant\DSH',
    [string]$HostAddress = '127.0.0.1',
    [int]$Port = 17373,
    [switch]$Open,
    [switch]$Inspect,
    [int]$InspectPort = 9229
)

$ErrorActionPreference = 'Stop'

$harnessRoot = Split-Path -Parent $PSScriptRoot
$dshRoot = Join-Path $harnessRoot 'dsh'
$devRoot = Join-Path $harnessRoot 'dev-state'
$devHome = Join-Path $devRoot 'home'
$node = Get-ChildItem -LiteralPath (Join-Path $harnessRoot 'runtime') -Filter node.exe -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match 'node-v[^\\]+-win-x64\\node\.exe$' } |
    Select-Object -First 1 -ExpandProperty FullName

if ($null -eq $node) { throw 'Isolated Node runtime is missing. Run Install-DevHarness.ps1 first.' }
if (-not (Test-Path -LiteralPath (Join-Path $dshRoot 'apps\cli\lib\bin.js'))) {
    throw 'The DSH source checkout has not been built. Run Install-DevHarness.ps1 first.'
}
if (-not (Test-Path -LiteralPath (Join-Path $devHome 'profiles\web\node_modules'))) {
    throw 'The isolated profile dependencies are missing. Run Install-DevHarness.ps1 first.'
}
if (-not (Test-Path -LiteralPath $WorkspaceRoot -PathType Container)) {
    throw "Workspace directory not found: $WorkspaceRoot"
}

$devResolved = [System.IO.Path]::GetFullPath($devHome)
$productionDefault = [System.IO.Path]::GetFullPath("$env:APPDATA\dsh-desktop\harness")
if ($devResolved.Equals($productionDefault, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Refusing to run with the production DSH_HOME.'
}

$oldHome = $env:DSH_HOME
$oldAgentsHome = $env:DSH_AGENTS_HOME
$oldPath = $env:PATH
try {
    $env:DSH_HOME = $devResolved
    $env:DSH_AGENTS_HOME = Join-Path $devRoot 'agents'
    $env:PATH = "$(Split-Path -Parent $node);$oldPath"

    $entry = Join-Path $dshRoot 'apps\cli\lib\bin.js'
    $nodeArgs = @()
    if ($Inspect) { $nodeArgs += "--inspect=$InspectPort" }
    $nodeArgs += @($entry, 'web', '--host', $HostAddress, '--port', [string]$Port)
    if (-not $Open) { $nodeArgs += '--no-open' }

    Write-Host "DSH_HOME:  $devResolved"
    Write-Host "Workspace: $WorkspaceRoot"
    Write-Host "Web UI:    http://${HostAddress}:$Port"

    Push-Location $WorkspaceRoot
    try {
        & $node @nodeArgs
        exit $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    $env:PATH = $oldPath
    if ($null -eq $oldHome) { Remove-Item Env:DSH_HOME -ErrorAction SilentlyContinue } else { $env:DSH_HOME = $oldHome }
    if ($null -eq $oldAgentsHome) { Remove-Item Env:DSH_AGENTS_HOME -ErrorAction SilentlyContinue } else { $env:DSH_AGENTS_HOME = $oldAgentsHome }
}
