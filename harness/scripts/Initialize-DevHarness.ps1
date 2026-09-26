[CmdletBinding()]
param(
    [string]$ProductionHome = "$env:APPDATA\dsh-desktop\harness",
    [string]$WorkspaceRoot = 'D:\AISuperAssistant\DSH',
    [switch]$Refresh
)

$ErrorActionPreference = 'Stop'

$harnessRoot = Split-Path -Parent $PSScriptRoot
$devRoot = Join-Path $harnessRoot 'dev-state'
$devHome = Join-Path $devRoot 'home'
$devProfile = Join-Path $devHome 'profiles\web'
$productionProfile = Join-Path $ProductionHome 'profiles\web'

$productionResolved = [System.IO.Path]::GetFullPath($ProductionHome)
$devResolved = [System.IO.Path]::GetFullPath($devHome)
if ($productionResolved.Equals($devResolved, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'The development DSH_HOME must not equal the production DSH_HOME.'
}
if (-not (Test-Path -LiteralPath $productionProfile)) {
    throw "Production web profile not found: $productionProfile"
}
if (-not (Test-Path -LiteralPath $WorkspaceRoot -PathType Container)) {
    throw "Workspace directory not found: $WorkspaceRoot"
}

New-Item -ItemType Directory -Path $devProfile -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $devHome 'storages') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $devRoot 'agents') -Force | Out-Null

function Copy-ConfigFile {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) { return }
    if ((Test-Path -LiteralPath $Destination) -and -not $Refresh) { return }
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

Copy-ConfigFile (Join-Path $ProductionHome 'settings.yaml') (Join-Path $devHome 'settings.yaml')
Copy-ConfigFile (Join-Path $ProductionHome '.credentials.yaml') (Join-Path $devHome '.credentials.yaml')
Copy-ConfigFile (Join-Path $ProductionHome '.env') (Join-Path $devHome '.env')
Copy-ConfigFile (Join-Path $ProductionHome 'storages\workspace.json') (Join-Path $devHome 'storages\workspace.json')
Copy-ConfigFile (Join-Path $productionProfile 'cordis.yml') (Join-Path $devProfile 'cordis.yml')

$productionStubs = Join-Path $productionProfile 'stubs'
$devStubs = Join-Path $devProfile 'stubs'
if ((Test-Path -LiteralPath $productionStubs -PathType Container) -and ($Refresh -or -not (Test-Path -LiteralPath $devStubs))) {
    New-Item -ItemType Directory -Path $devStubs -Force | Out-Null
    Get-ChildItem -LiteralPath $productionStubs -Force | Copy-Item -Destination $devStubs -Recurse -Force
}

$patchSource = Join-Path $productionProfile 'cordis.patch.yml'
$patchDestination = Join-Path $devProfile 'cordis.patch.yml'
if ($Refresh -or -not (Test-Path -LiteralPath $patchDestination)) {
    $patch = [System.IO.File]::ReadAllText($patchSource)
    $patch = [regex]::Replace($patch, '(?m)^- id: dsh-mneme\s*$', '- id: dsh-mneme-moe4all')
    $patch = [regex]::Replace($patch, '(?m)^- id: remote-web-ui\s*$', '- id: remote-web-ui-moe4all')
    [System.IO.File]::WriteAllText($patchDestination, $patch, [System.Text.UTF8Encoding]::new($false))
}

$dependencies = [ordered]@{
    'dsh-market-moe4all' = 'file:../../../../runtime/plugin-packs/dsh-market-moe4all.tgz'
    'dsh-web-search-brave-moe4all' = 'file:../../../../runtime/plugin-packs/dsh-web-search-brave-moe4all.tgz'
    'dsh-mneme-moe4all' = 'file:../../../../runtime/plugin-packs/dsh-mneme-moe4all.tgz'
    'dsh-easyrewrite-moe4all' = 'file:../../../../runtime/plugin-packs/dsh-easyrewrite-moe4all.tgz'
    'dsh-remote-web-ui-moe4all' = 'file:../../../../runtime/plugin-packs/dsh-remote-web-ui-moe4all.tgz'
    'dsh-plugin-cron-moe4all' = 'file:../../../../runtime/plugin-packs/dsh-plugin-cron-moe4all.tgz'
    'dsh-llm-moe4all' = 'file:../../../../runtime/plugin-packs/dsh-llm-moe4all.tgz'
    'dsh-plugin-tts-moe4all' = 'file:../../../../runtime/plugin-packs/dsh-plugin-tts-moe4all.tgz'
}
$bundles = @(
    '@deepseek-ai/dsh-base',
    '@deepseek-ai/dsh-web-app',
    'dsh-market-moe4all',
    'dsh-web-search-brave-moe4all',
    'dsh-mneme-moe4all',
    'dsh-easyrewrite-moe4all',
    'dsh-remote-web-ui-moe4all',
    'dsh-plugin-cron-moe4all',
    'dsh-llm-moe4all',
    'dsh-plugin-tts-moe4all'
)
$manifest = [ordered]@{
    name = 'moe4all-profile-web-dev'
    private = $true
    dependencies = $dependencies
    dsh = [ordered]@{ profile = [ordered]@{ bundles = $bundles } }
}
$manifestJson = $manifest | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText((Join-Path $devProfile 'package.json'), "$manifestJson`n", [System.Text.UTF8Encoding]::new($false))

$workspaceYaml = @'
packages:
  - .

overrides:
  '@huggingface/transformers': link:./stubs/hf-transformers-stub

nodeLinker: hoisted
autoInstallPeers: false

allowBuilds:
  cloudflared: false
'@
[System.IO.File]::WriteAllText((Join-Path $devProfile 'pnpm-workspace.yaml'), $workspaceYaml, [System.Text.UTF8Encoding]::new($false))

$npmrc = @'
package-import-method=clone-or-copy
child-concurrency=4
side-effects-cache=false
store-dir=../../../../runtime/pnpm-store
ignore-scripts=true
'@
[System.IO.File]::WriteAllText((Join-Path $devProfile '.npmrc'), $npmrc, [System.Text.UTF8Encoding]::new($false))

$origin = [ordered]@{
    initializedAt = [DateTime]::UtcNow.ToString('o')
    productionHome = $productionResolved
    workspaceRoot = [System.IO.Path]::GetFullPath($WorkspaceRoot)
    copied = @('settings.yaml', '.credentials.yaml', '.env', 'profiles/web/cordis.yml', 'profiles/web/cordis.patch.yml', 'profiles/web/stubs', 'storages/workspace.json')
    deliberatelyExcluded = @('sessions', 'attachments', 'storages/session_projcache.json', 'profiles/web/node_modules', 'remote-web-ui-devices.json')
}
$originJson = $origin | ConvertTo-Json -Depth 5
[System.IO.File]::WriteAllText((Join-Path $devRoot 'origin.json'), "$originJson`n", [System.Text.UTF8Encoding]::new($false))

Write-Host "Isolated DSH_HOME: $devHome"
Write-Host "Local profile:      $devProfile"
Write-Host "Shared workspace:   $WorkspaceRoot"
