[CmdletBinding()]
param([string]$OutputDirectory)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $repo 'artifacts\agent-update-tests' }
$testRoot = Join-Path ([IO.Path]::GetFullPath($OutputDirectory)) ('rolling-' + [guid]::NewGuid().ToString('N'))
$install = Join-Path $testRoot 'MoE4All'
$utf8 = New-Object Text.UTF8Encoding($false)

function Write-Text([string]$Path, [string]$Value) {
    New-Item -ItemType Directory -Path (Split-Path -Parent $Path) -Force | Out-Null
    [IO.File]::WriteAllText($Path, $Value, $utf8)
}

function Write-Manifest([string]$Root, [string]$Version) {
    $manifest = [ordered]@{
        schema = 2
        product = 'moe4all-agent'
        packageKind = 'integrated'
        version = $Version
        sourceCommit = ('a' * 40)
        dshCommit = ('b' * 40)
        engineVersion = 'v0.10.0'
    }
    Write-Text (Join-Path $Root 'harness\distribution\agent-release.json') (($manifest | ConvertTo-Json) + "`n")
}

function Write-App([string]$Root, [string]$Version) {
    Write-Text (Join-Path $Root 'MoE4All.exe') $Version
    Write-Text (Join-Path $Root 'MoE4All.exe.config') '<configuration />'
    Write-Text (Join-Path $Root 'node.exe') 'node'
    Write-Text (Join-Path $Root 'harness\bootstrap\node.exe') 'bootstrap-node'
    Write-Text (Join-Path $Root 'Microsoft.Web.WebView2.Core.dll') 'core'
    Write-Text (Join-Path $Root 'Microsoft.Web.WebView2.WinForms.dll') 'forms'
    Write-Text (Join-Path $Root 'WebView2Loader.dll') 'loader'
    Write-Text (Join-Path $Root 'Microsoft.Web.WebView2.LICENSE.txt') 'license'
    Write-Manifest $Root $Version
    New-Item -ItemType Directory -Path (Join-Path $Root 'harness\scripts') -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'AgentUpdateVersion.ps1') -Destination (Join-Path $Root 'harness\scripts\AgentUpdateVersion.ps1') -Force
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Check-AgentUpdate.ps1') -Destination (Join-Path $Root 'harness\scripts\Check-AgentUpdate.ps1') -Force
}

function Write-Archive([string]$Version) {
    $name = "MoE4All-Agent-Windows-x64-$Version"
    $source = Join-Path $testRoot $name
    Write-App $source $Version
    $archive = Join-Path $testRoot "$name.zip"
    Compress-Archive -LiteralPath $source -DestinationPath $archive
    $hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Text "$archive.sha256" "$hash  $name.zip`n"
    Remove-Item -LiteralPath $source -Recurse -Force
    return $archive
}

New-Item -ItemType Directory -Path $install -Force | Out-Null
Write-App $install 'agent-v0.10.0-preview.1'
foreach ($directory in @('data', 'models', 'cache')) {
    Write-Text (Join-Path $install "$directory\sentinel.txt") "$directory-stays"
}
$preview2 = Write-Archive 'agent-v0.10.0-preview.2'
$preview3 = Write-Archive 'agent-v0.10.0-preview.3'

$oldGuard = $env:MOE4ALL_AGENT_UPDATE_TEST
try {
    $env:MOE4ALL_AGENT_UPDATE_TEST = '1'
    & (Join-Path $install 'harness\scripts\Check-AgentUpdate.ps1') -InstallRoot $install -ParentPid $PID -TestArchive $preview2 -TestRoot $testRoot -TestInstallOnly
    & (Join-Path $install 'harness\scripts\Check-AgentUpdate.ps1') -InstallRoot $install -ParentPid $PID -TestArchive $preview3 -TestRoot $testRoot -TestInstallOnly
} finally {
    $env:MOE4ALL_AGENT_UPDATE_TEST = $oldGuard
}

$current = Get-Content -LiteralPath (Join-Path $install 'harness\distribution\agent-release.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$previous = Get-Content -LiteralPath (Join-Path $install 'updates\previous\harness\distribution\agent-release.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if ($current.version -ne 'agent-v0.10.0-preview.3' -or $previous.version -ne 'agent-v0.10.0-preview.2') {
    throw "Rolling versions are wrong: current=$($current.version), previous=$($previous.version)"
}
foreach ($directory in @('data', 'models', 'cache')) {
    $sentinel = Join-Path $install "$directory\sentinel.txt"
    if ((Get-Content -LiteralPath $sentinel -Raw) -ne "$directory-stays") { throw "$directory changed during update." }
    if (Test-Path -LiteralPath (Join-Path $install "updates\previous\$directory")) { throw "$directory leaked into the previous application backup." }
}
$unexpected = @(Get-ChildItem -LiteralPath (Join-Path $install 'updates') -Force | Where-Object Name -ne 'previous')
if ($unexpected.Count -ne 0) { throw "Updater staging was not cleaned: $($unexpected.Name -join ', ')" }

Write-Host 'Agent updater tests passed: application-only swap, persistent data, one rolling previous version, staging cleanup.'
Write-Host "Evidence: $testRoot"
