[CmdletBinding()]
param(
    [string]$NodeVersion = '24.9.0',
    [string]$Registry = 'https://registry.npmmirror.com',
    [string]$ProductionHome = "$env:APPDATA\dsh-desktop\harness",
    [string]$WorkspaceRoot = 'D:\AISuperAssistant\DSH',
    [switch]$RefreshConfig
)

$ErrorActionPreference = 'Stop'

$harnessRoot = Split-Path -Parent $PSScriptRoot
$dshRoot = Join-Path $harnessRoot 'dsh'
$runtimeRoot = Join-Path $harnessRoot 'runtime'
$nodeFolder = "node-v$NodeVersion-win-x64"
$nodeRoot = Join-Path $runtimeRoot $nodeFolder
$node = Join-Path $nodeRoot 'node.exe'
$npmCli = Join-Path $nodeRoot 'node_modules\npm\bin\npm-cli.js'
$pnpmRoot = Join-Path $runtimeRoot 'pnpm'
$pnpmCli = Join-Path $pnpmRoot 'node_modules\pnpm\bin\pnpm.cjs'
$pnpmBin = Join-Path $pnpmRoot 'node_modules\.bin'
$pnpmStore = Join-Path $runtimeRoot 'pnpm-store'
$pluginPackRoot = Join-Path $runtimeRoot 'plugin-packs'
$devProfile = Join-Path $harnessRoot 'dev-state\home\profiles\web'

$initializeParams = @{
    ProductionHome = $ProductionHome
    WorkspaceRoot = $WorkspaceRoot
}
if ($RefreshConfig) { $initializeParams.Refresh = $true }
& (Join-Path $PSScriptRoot '_Initialize-DevHarness.ps1') @initializeParams

if (-not (Test-Path -LiteralPath $node)) {
    New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null
    $archive = Join-Path $runtimeRoot "$nodeFolder.zip"
    if (-not (Test-Path -LiteralPath $archive)) {
        $uri = "https://nodejs.org/dist/v$NodeVersion/$nodeFolder.zip"
        Write-Host "Downloading $uri"
        Invoke-WebRequest -Uri $uri -OutFile $archive
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $runtimeRoot -Force
}
if (-not (Test-Path -LiteralPath $node)) { throw "Node runtime not found after extraction: $node" }
if (-not (Test-Path -LiteralPath $npmCli)) { throw "npm CLI not found in Node runtime: $npmCli" }

if (-not (Test-Path -LiteralPath $pnpmCli)) {
    New-Item -ItemType Directory -Path $pnpmRoot -Force | Out-Null
    & $node $npmCli install --prefix $pnpmRoot --registry $Registry --no-audit --no-fund 'pnpm@11.7.0'
    if ($LASTEXITCODE -ne 0) { throw 'Installing the isolated pnpm runtime failed.' }
}

$oldCi = $env:CI
$oldPath = $env:PATH
$oldRegistry = $env:NPM_CONFIG_REGISTRY
try {
    $env:CI = 'true'
    $env:PATH = "$pnpmBin;$nodeRoot;$oldPath"
    $env:NPM_CONFIG_REGISTRY = $Registry

    Push-Location $dshRoot
    try {
        & $node $pnpmCli install --frozen-lockfile --store-dir $pnpmStore --fetch-timeout 300000 --fetch-retries 5
        if ($LASTEXITCODE -ne 0) { throw 'Installing the DSH source workspace failed.' }

        & $node $pnpmCli "--config.store-dir=$pnpmStore" run build
        if ($LASTEXITCODE -ne 0) { throw 'Building the DSH source workspace failed.' }
    } finally {
        Pop-Location
    }

    New-Item -ItemType Directory -Path $pluginPackRoot -Force | Out-Null
    $pluginNames = @(
        'dsh-market-moe4all',
        'dsh-web-search-brave-moe4all',
        'dsh-mneme-moe4all',
        'dsh-easyrewrite-moe4all',
        'dsh-remote-web-ui-moe4all',
        'dsh-plugin-cron-moe4all',
        'dsh-llm-moe4all',
        'dsh-plugin-tts-moe4all'
    )
    foreach ($pluginName in $pluginNames) {
        $pluginPath = Join-Path $harnessRoot "plugins\$pluginName"
        if (-not (Test-Path -LiteralPath (Join-Path $pluginPath 'package.json'))) {
            throw "Plugin source is missing a package manifest: $pluginPath"
        }

        $packOutput = & $node $npmCli pack $pluginPath --pack-destination $pluginPackRoot --ignore-scripts --json
        if ($LASTEXITCODE -ne 0) { throw "Packing $pluginName failed." }
        $pack = ($packOutput -join [Environment]::NewLine) | ConvertFrom-Json
        $generatedPack = Join-Path $pluginPackRoot $pack.filename
        $stablePack = Join-Path $pluginPackRoot "$pluginName.tgz"
        if (-not (Test-Path -LiteralPath $generatedPack -PathType Leaf)) {
            throw "npm did not create the expected archive for ${pluginName}: $generatedPack"
        }
        if (-not $generatedPack.Equals($stablePack, [System.StringComparison]::OrdinalIgnoreCase)) {
            Copy-Item -LiteralPath $generatedPack -Destination $stablePack -Force
        }
    }

    Push-Location $devProfile
    try {
        & $node $pnpmCli install --no-frozen-lockfile --store-dir $pnpmStore --fetch-timeout 300000 --fetch-retries 5
        if ($LASTEXITCODE -ne 0) { throw 'Installing the isolated web profile failed.' }
    } finally {
        Pop-Location
    }
} finally {
    $env:PATH = $oldPath
    if ($null -eq $oldCi) { Remove-Item Env:CI -ErrorAction SilentlyContinue } else { $env:CI = $oldCi }
    if ($null -eq $oldRegistry) { Remove-Item Env:NPM_CONFIG_REGISTRY -ErrorAction SilentlyContinue } else { $env:NPM_CONFIG_REGISTRY = $oldRegistry }
}

Write-Host "Node: $(& $node --version)"
Write-Host "pnpm: $(& $node $pnpmCli --version)"
Write-Host 'MoE4All Harness development dependencies are ready.'
