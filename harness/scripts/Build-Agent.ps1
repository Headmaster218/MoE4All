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
$oldCommit = $env:DSH_CLIENT_COMMIT_HASH
try {
    $env:DSH_CLIENT_BUILD_PROFILE = 'moe4all'
    $env:DSH_CLIENT_TITLE = 'MoE4All Agent'
    $releaseManifest = Join-Path $harnessRoot 'distribution\agent-release.json'
    if (Test-Path -LiteralPath $releaseManifest -PathType Leaf) {
        $release = Get-Content -LiteralPath $releaseManifest -Raw -Encoding UTF8 | ConvertFrom-Json
        if ([string]$release.dshCommit -notmatch '^[0-9a-f]{40}$') { throw 'Invalid DSH commit in Agent release manifest.' }
        $env:DSH_CLIENT_COMMIT_HASH = [string]$release.dshCommit
    }
    & (Join-Path $PSScriptRoot 'Build-DevHarness.ps1') -SkipInitialize -ProfileDir (Join-Path $state 'home\profiles\web')
} finally {
    if ($null -eq $oldProfile) { Remove-Item Env:DSH_CLIENT_BUILD_PROFILE -ErrorAction SilentlyContinue } else { $env:DSH_CLIENT_BUILD_PROFILE = $oldProfile }
    if ($null -eq $oldTitle) { Remove-Item Env:DSH_CLIENT_TITLE -ErrorAction SilentlyContinue } else { $env:DSH_CLIENT_TITLE = $oldTitle }
    if ($null -eq $oldCommit) { Remove-Item Env:DSH_CLIENT_COMMIT_HASH -ErrorAction SilentlyContinue } else { $env:DSH_CLIENT_COMMIT_HASH = $oldCommit }
}
