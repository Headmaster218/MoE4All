[CmdletBinding()]
param(
    [string]$StateRoot,
    [string]$WorkspaceRoot,
    [string]$HostAddress = '127.0.0.1',
    [int]$Port = 17373,
    [switch]$AllowSharedWorkspace,
    [switch]$Open
)

$ErrorActionPreference = 'Stop'
$harnessRoot = Split-Path -Parent $PSScriptRoot
if (-not $StateRoot) { $StateRoot = Join-Path $harnessRoot 'agent-state' }
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
if (-not (Test-Path -LiteralPath $workspace -PathType Container)) {
    throw "Workspace directory not found: $workspace"
}
$node = Get-ChildItem -LiteralPath (Join-Path $harnessRoot 'runtime') -Filter node.exe -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match 'node-v[^\\]+-win-x64\\node\.exe$' } |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $node) { throw 'MoE4All Node runtime is missing. Run Build-Agent.ps1 first.' }
$entry = Join-Path $harnessRoot 'dsh\apps\cli\lib\bin.js'
$oldHome = $env:DSH_HOME
$oldAgentsHome = $env:DSH_AGENTS_HOME
$oldPath = $env:PATH
$oldProduct = $env:MOE4ALL_AGENT
try {
    $env:DSH_HOME = $agentHome
    $env:DSH_AGENTS_HOME = Join-Path $state 'agents'
    $env:MOE4ALL_AGENT = '1'
    $env:PATH = "$(Split-Path -Parent $node);$oldPath"
    Write-Host "MoE4All Agent: http://${HostAddress}:$Port"
    Write-Host "DSH_HOME: $agentHome"
    Write-Host "Workspace: $workspace"
    Push-Location $workspace
    try {
        $nodeArgs = @($entry, 'web', '--host', $HostAddress, '--port', [string]$Port)
        if (-not $Open) { $nodeArgs += '--no-open' }
        & $node @nodeArgs
        if ($LASTEXITCODE -ne 0) { throw "MoE4All Agent exited with code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} finally {
    $env:PATH = $oldPath
    if ($null -eq $oldHome) { Remove-Item Env:DSH_HOME -ErrorAction SilentlyContinue } else { $env:DSH_HOME = $oldHome }
    if ($null -eq $oldAgentsHome) { Remove-Item Env:DSH_AGENTS_HOME -ErrorAction SilentlyContinue } else { $env:DSH_AGENTS_HOME = $oldAgentsHome }
    if ($null -eq $oldProduct) { Remove-Item Env:MOE4ALL_AGENT -ErrorAction SilentlyContinue } else { $env:MOE4ALL_AGENT = $oldProduct }
}
