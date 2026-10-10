[CmdletBinding()]
param(
    [string]$StateRoot,
    [string]$EngineEndpoint = ''
)

$ErrorActionPreference = 'Stop'
$harnessRoot = Split-Path -Parent $PSScriptRoot
if (-not $StateRoot) { $StateRoot = Join-Path $harnessRoot 'agent-state' }
$state = [System.IO.Path]::GetFullPath($StateRoot)
& (Join-Path $PSScriptRoot 'Initialize-Agent.ps1') -StateRoot $state -EngineEndpoint $EngineEndpoint
$oldProfile = $env:DSH_CLIENT_BUILD_PROFILE
$oldTitle = $env:DSH_CLIENT_TITLE
try {
    $env:DSH_CLIENT_BUILD_PROFILE = 'moe4all'
    $env:DSH_CLIENT_TITLE = 'MoE4All Agent'
    & (Join-Path $PSScriptRoot 'Build-DevHarness.ps1') -SkipInitialize -ProfileDir (Join-Path $state 'home\profiles\web')
} finally {
    if ($null -eq $oldProfile) { Remove-Item Env:DSH_CLIENT_BUILD_PROFILE -ErrorAction SilentlyContinue } else { $env:DSH_CLIENT_BUILD_PROFILE = $oldProfile }
    if ($null -eq $oldTitle) { Remove-Item Env:DSH_CLIENT_TITLE -ErrorAction SilentlyContinue } else { $env:DSH_CLIENT_TITLE = $oldTitle }
}
