[CmdletBinding()]
param(
    [string]$StateRoot,
    [string]$WorkspaceRoot,
    [string]$HostAddress = '127.0.0.1',
    [int]$Port = 17373,
    [string]$CliPath,
    [string]$DesktopBridge,
    [string]$DesktopToken,
    [switch]$AllowSharedWorkspace,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$harnessRoot = Split-Path -Parent $PSScriptRoot
$installRoot = Split-Path -Parent $harnessRoot
if (-not $StateRoot) { $StateRoot = Join-Path $installRoot 'data' }
$state = [System.IO.Path]::GetFullPath($StateRoot)
if (-not $WorkspaceRoot) { $WorkspaceRoot = Join-Path $state 'workspace' }
$workspace = [System.IO.Path]::GetFullPath($WorkspaceRoot)
$defaultWorkspace = [System.IO.Path]::GetFullPath((Join-Path $state 'workspace'))
if (-not $AllowSharedWorkspace -and
    -not $workspace.Equals($defaultWorkspace, [System.StringComparison]::OrdinalIgnoreCase) -and
    -not $workspace.StartsWith($defaultWorkspace + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'A workspace outside MoE4All state requires -AllowSharedWorkspace.'
}
$sourceHomes = @(
    [System.IO.Path]::GetFullPath((Join-Path $env:APPDATA 'dsh-desktop\harness')),
    [System.IO.Path]::GetFullPath((Join-Path $env:USERPROFILE '.dsh'))
)
if ($env:DSH_HOME) {
    $inheritedHome = [System.IO.Path]::GetFullPath($env:DSH_HOME)
    $agentHomePath = [System.IO.Path]::GetFullPath((Join-Path $state 'home'))
    if (-not $inheritedHome.Equals($agentHomePath, [System.StringComparison]::OrdinalIgnoreCase)) {
        $sourceHomes += $inheritedHome
    }
}
foreach ($path in @($state, $workspace)) {
    foreach ($sourceHome in $sourceHomes) {
        if ($path.Equals($sourceHome, [System.StringComparison]::OrdinalIgnoreCase) -or
            $path.StartsWith($sourceHome + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "MoE4All cannot run inside a DSH home: $path"
        }
    }
    $cursor = $path
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            $entry = Get-Item -LiteralPath $cursor -Force
            if (($entry.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "MoE4All path cannot traverse a junction or symlink: $cursor"
            }
        }
        $parent = Split-Path -Parent $cursor
        if (-not $parent -or $parent -eq $cursor) { break }
        $cursor = $parent
    }
}
$agentHome = Join-Path $state 'home'
$profile = Join-Path $agentHome 'profiles\web'
if (-not (Test-Path -LiteralPath (Join-Path $profile 'node_modules') -PathType Container)) {
    throw 'MoE4All profile is not built. Run Build-Agent.ps1 first.'
}
New-Item -ItemType Directory -Path $workspace -Force | Out-Null
$node = Join-Path $harnessRoot 'bootstrap\node.exe'
if (-not (Test-Path -LiteralPath $node -PathType Leaf)) {
    $node = Get-ChildItem -LiteralPath (Join-Path $harnessRoot 'runtime') -Directory -Filter 'node-v*-win-x64' |
        ForEach-Object { Join-Path $_.FullName 'node.exe' } |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
}
if (-not $node) { throw 'The bundled MoE4All Node runtime is missing.' }
if (-not $CliPath) { $CliPath = Join-Path $harnessRoot 'runtime\dsh\lib\bin.js' }
$cliEntry = [System.IO.Path]::GetFullPath($CliPath)
if (-not (Test-Path -LiteralPath $cliEntry -PathType Leaf)) { throw "MoE4All CLI entry not found: $cliEntry" }
$cacheRoot = Join-Path $installRoot 'cache'
$modelsRoot = Join-Path $installRoot 'models'
$cacheDirectories = @(
    $cacheRoot,
    (Join-Path $cacheRoot 'temp'),
    (Join-Path $cacheRoot 'node'),
    (Join-Path $cacheRoot 'python'),
    (Join-Path $cacheRoot 'huggingface'),
    (Join-Path $cacheRoot 'torch'),
    (Join-Path $cacheRoot 'kv-sessions'),
    $modelsRoot
)
New-Item -ItemType Directory -Path $cacheDirectories -Force | Out-Null

$managedEnvironment = [ordered]@{
    DSH_HOME = $agentHome
    DSH_AGENTS_HOME = (Join-Path $state 'agents')
    MOE4ALL_AGENT = '1'
    MOE4ALL_ROOT = $installRoot
    MOE4ALL_DATA_DIR = $state
    MOE4ALL_MODELS_DIR = $modelsRoot
    MOE4ALL_CACHE_DIR = $cacheRoot
    MOE4ALL_KV_CACHE_DIR = (Join-Path $cacheRoot 'kv-sessions')
    XDG_CACHE_HOME = $cacheRoot
    HF_HOME = (Join-Path $cacheRoot 'huggingface')
    HUGGINGFACE_HUB_CACHE = (Join-Path $cacheRoot 'huggingface\hub')
    TRANSFORMERS_CACHE = (Join-Path $cacheRoot 'huggingface\transformers')
    TORCH_HOME = (Join-Path $cacheRoot 'torch')
    NPM_CONFIG_CACHE = (Join-Path $cacheRoot 'node\npm')
    UV_CACHE_DIR = (Join-Path $cacheRoot 'python\uv')
    PIP_CACHE_DIR = (Join-Path $cacheRoot 'python\pip')
    TEMP = (Join-Path $cacheRoot 'temp')
    TMP = (Join-Path $cacheRoot 'temp')
    MOE4ALL_DESKTOP_BRIDGE = $DesktopBridge
    MOE4ALL_DESKTOP_TOKEN = $DesktopToken
    PATH = "$(Split-Path -Parent $node);$env:PATH"
}
$oldEnvironment = @{}
try {
    foreach ($environmentEntry in $managedEnvironment.GetEnumerator()) {
        $oldEnvironment[$environmentEntry.Key] = [Environment]::GetEnvironmentVariable($environmentEntry.Key, 'Process')
        [Environment]::SetEnvironmentVariable($environmentEntry.Key, $environmentEntry.Value, 'Process')
    }
    Write-Host "MoE4All Agent: http://${HostAddress}:$Port"
    Write-Host "DSH_HOME: $agentHome"
    Write-Host "Workspace: $workspace"
    Write-Host "Models: $modelsRoot"
    Write-Host "Cache: $cacheRoot"
    Push-Location $workspace
    try {
        $nodeArgs = @($cliEntry, 'web', '--host', $HostAddress, '--port', [string]$Port)
        if (-not $Open) { $nodeArgs += '--no-open' }
        & $node @nodeArgs
        if ($LASTEXITCODE -ne 0) { throw "MoE4All Agent exited with code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} finally {
    foreach ($environmentEntry in $managedEnvironment.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($environmentEntry.Key, $oldEnvironment[$environmentEntry.Key], 'Process')
    }
}
