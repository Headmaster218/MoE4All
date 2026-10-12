[CmdletBinding()]
param(
    [string]$InstallRoot,
    [string]$StateRoot
)

$ErrorActionPreference = 'Stop'
$harnessRoot = Split-Path -Parent $PSScriptRoot
if (-not $InstallRoot) { $InstallRoot = Split-Path -Parent $harnessRoot }
if (-not $StateRoot) { $StateRoot = Join-Path $harnessRoot 'agent-state' }
$root = [IO.Path]::GetFullPath($InstallRoot)
$state = [IO.Path]::GetFullPath($StateRoot)

function Resolve-ContainedPath([string]$base, [string]$relative) {
    if ([IO.Path]::IsPathRooted($relative) -or $relative -match '(^|[\\/])\.\.([\\/]|$)') {
        throw "Unsafe bundled relative path: $relative"
    }
    $resolvedBase = [IO.Path]::GetFullPath($base).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $resolved = [IO.Path]::GetFullPath((Join-Path $resolvedBase $relative))
    if (-not $resolved.StartsWith($resolvedBase + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Bundled path escapes its root: $relative"
    }
    return $resolved
}

$dshRoot = Join-Path $harnessRoot 'runtime\dsh'
$linksPath = Join-Path $harnessRoot 'runtime\dsh-links.json'
$linksMarker = Join-Path $dshRoot '.moe4all-links-ready'
if (-not (Test-Path -LiteralPath $linksMarker -PathType Leaf)) {
    if (-not (Test-Path -LiteralPath $linksPath -PathType Leaf)) { throw 'The bundled DSH link manifest is missing.' }
    $manifest = Get-Content -LiteralPath $linksPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($manifest.schema -ne 1 -or $null -eq $manifest.links) { throw 'The bundled DSH link manifest is invalid.' }
    foreach ($link in @($manifest.links)) {
        $path = Resolve-ContainedPath $dshRoot ([string]$link.path)
        $target = Resolve-ContainedPath $dshRoot ([string]$link.target)
        if (-not (Test-Path -LiteralPath $target -PathType Container)) {
            throw "Bundled DSH junction target is missing: $($link.target)"
        }
        if (Test-Path -LiteralPath $path) {
            $entry = Get-Item -LiteralPath $path -Force
            if (($entry.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
                throw "Bundled DSH junction path is occupied: $($link.path)"
            }
            continue
        }
        $parent = Split-Path -Parent $path
        if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
        }
        New-Item -ItemType Junction -Path $path -Target $target | Out-Null
    }
    [IO.File]::WriteAllText($linksMarker, "schema=1`n", (New-Object Text.UTF8Encoding($false)))
}

$seed = Join-Path $harnessRoot 'distribution\prebuilt-state'
$profileModules = Join-Path $state 'home\profiles\web\node_modules'
if (-not (Test-Path -LiteralPath $profileModules -PathType Container)) {
    if (-not (Test-Path -LiteralPath $seed -PathType Container)) { throw 'The bundled Agent state seed is missing.' }
    New-Item -ItemType Directory -Path $state -Force | Out-Null
    & robocopy $seed $state /E /XJ /XC /XN /XO /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Preparing the bundled Agent profile failed (robocopy $LASTEXITCODE)." }
    if (-not (Test-Path -LiteralPath $profileModules -PathType Container)) {
        throw 'The bundled Agent profile is incomplete.'
    }
}

$runtimePnpm = Join-Path $dshRoot 'node_modules\.pnpm'
$runtimeScope = Join-Path $dshRoot 'node_modules\@deepseek-ai'
$profileScope = Join-Path $profileModules '@deepseek-ai'
if (-not (Test-Path -LiteralPath $runtimePnpm -PathType Container)) {
    throw 'The bundled DSH package store is missing.'
}
New-Item -ItemType Directory -Path $runtimeScope -Force | Out-Null
New-Item -ItemType Directory -Path $profileScope -Force | Out-Null
$runtimePackages = @{}

function Add-RuntimePackage([string]$path) {
    if (($null -eq $path) -or -not (Test-Path -LiteralPath $path -PathType Container)) { return }
    $packageManifest = Join-Path $path 'package.json'
    if (-not (Test-Path -LiteralPath $packageManifest -PathType Leaf)) { return }
    try {
        $packageName = [string](Get-Content -LiteralPath $packageManifest -Raw -Encoding UTF8 | ConvertFrom-Json).name
    } catch {
        return
    }
    if ($packageName -notmatch '^@deepseek-ai/[0-9A-Za-z._-]+$' -or $runtimePackages.ContainsKey($packageName)) { return }
    $runtimePackages[$packageName] = [IO.Path]::GetFullPath($path)
}

foreach ($entry in Get-ChildItem -LiteralPath $runtimePnpm -Directory) {
    $scope = Join-Path $entry.FullName 'node_modules\@deepseek-ai'
    if (-not (Test-Path -LiteralPath $scope -PathType Container)) { continue }
    foreach ($package in Get-ChildItem -LiteralPath $scope -Directory -Force) {
        if (($package.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { continue }
        Add-RuntimePackage $package.FullName
    }
}
$materializedRoot = Join-Path $dshRoot '.moe4all-workspace'
if (Test-Path -LiteralPath $materializedRoot -PathType Container) {
    foreach ($package in Get-ChildItem -LiteralPath $materializedRoot -Directory -Force) {
        Add-RuntimePackage $package.FullName
    }
}
$requiredRuntimePackages = @(
    '@deepseek-ai/cosmokit',
    '@deepseek-ai/dsh-base',
    '@deepseek-ai/dsh-scope',
    '@deepseek-ai/dsh-timeout',
    '@deepseek-ai/dsh-web-app',
    '@deepseek-ai/schemastery'
)
foreach ($packageName in $requiredRuntimePackages) {
    if (-not $runtimePackages.ContainsKey($packageName)) {
        throw "The bundled DSH package store is incomplete: $packageName is missing."
    }
}

function Ensure-PackageJunction([string]$link, [string]$target) {
    $existing = Get-Item -LiteralPath $link -Force -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        if (($existing.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) { return }
        $existingTargetValue = [string]@($existing.Target)[0]
        if ([IO.Path]::IsPathRooted($existingTargetValue)) {
            $existingTarget = [IO.Path]::GetFullPath($existingTargetValue)
        } else {
            $existingTarget = [IO.Path]::GetFullPath((Join-Path $existing.Parent.FullName $existingTargetValue))
        }
        if ($existingTarget.Equals($target, [StringComparison]::OrdinalIgnoreCase)) { return }
        [IO.Directory]::Delete($link, $false)
    }
    New-Item -ItemType Junction -Path $link -Target $target | Out-Null
}

foreach ($packageName in @($runtimePackages.Keys | Sort-Object)) {
    $shortName = $packageName.Substring('@deepseek-ai/'.Length)
    $target = [string]$runtimePackages[$packageName]
    Ensure-PackageJunction (Join-Path $runtimeScope $shortName) $target
    Ensure-PackageJunction (Join-Path $profileScope $shortName) $target
}

$profilePatch = Join-Path $state 'home\profiles\web\cordis.patch.yml'
if (Test-Path -LiteralPath $profilePatch -PathType Leaf) {
    $patch = Get-Content -LiteralPath $profilePatch -Raw -Encoding UTF8
    $legacyEndpoint = '    endpoint: http://127.0.0.1:1234/v1'
    if ($patch.Contains($legacyEndpoint)) {
        $patch = $patch.Replace($legacyEndpoint, '    endpoint: http://127.0.0.1:8080/v1')
    }
    $mnemePattern = '(?ms)^- id: dsh-mneme-moe4all\s*\r?\n(?:(?!^- id: ).)*?(?=^- id: |\z)'
    $mnemeMatch = [regex]::Match($patch, $mnemePattern)
    if ($mnemeMatch.Success -and $mnemeMatch.Value -notmatch '(?m)^\s{4}vectorDefaults:\s*$') {
        $configLine = [regex]::Match($mnemeMatch.Value, '(?m)^  config:\s*\r?\n')
        if ($configLine.Success) {
            $lineEnding = "`n"
            if ($patch.Contains("`r`n")) { $lineEnding = "`r`n" }
            $defaults = @(
                '    vectorDefaults:'
                '      enabled: true'
                '      baseUrl: http://127.0.0.1:8080/v1'
                '      apiKey: moe4all-local'
                '      model: Qwen3-Embedding-0.6B-Q8_0'
                ''
            ) -join $lineEnding
            $insertAt = $mnemeMatch.Index + $configLine.Index + $configLine.Length
            $patch = $patch.Insert($insertAt, $defaults)
        }
    }
    [IO.File]::WriteAllText($profilePatch, $patch, (New-Object Text.UTF8Encoding($false)))
}

$engineManifestPath = Join-Path $harnessRoot 'runtime\engine\bundled-engine.json'
if (Test-Path -LiteralPath $engineManifestPath -PathType Leaf) {
    $engine = Get-Content -LiteralPath $engineManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($engine.schema -ne 1 -or [string]$engine.tag -notmatch '^v[0-9]+\.[0-9]+\.[0-9]+$' -or
        [string]$engine.sha256 -notmatch '^[a-f0-9]{64}$') {
        throw 'The bundled Engine manifest is invalid.'
    }
    $source = Resolve-ContainedPath (Join-Path $harnessRoot 'runtime\engine') ([string]$engine.executable)
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw 'The bundled Engine executable is missing.' }
    $managedRoot = Join-Path $state 'home\moe4all-engine'
    $metadataPath = Join-Path $managedRoot 'installed.json'
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        $releaseRoot = Join-Path $managedRoot ("releases\" + [string]$engine.tag)
        $destination = Join-Path $releaseRoot 'infr.exe'
        New-Item -ItemType Directory -Path $releaseRoot -Force | Out-Null
        $actual = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne [string]$engine.sha256) { throw 'The bundled Engine SHA256 does not match its manifest.' }
        Copy-Item -LiteralPath $source -Destination $destination -Force
        $installed = [ordered]@{
            tag = [string]$engine.tag
            name = [string]$engine.name
            executable = $destination
            workingDirectory = $releaseRoot
            installedAt = (Get-Date).ToUniversalTime().ToString('o')
            sourceUrl = 'bundled:moe4all-agent'
        }
        New-Item -ItemType Directory -Path $managedRoot -Force | Out-Null
        [IO.File]::WriteAllText($metadataPath, (($installed | ConvertTo-Json) + "`n"), (New-Object Text.UTF8Encoding($false)))
    }
}

Write-Host 'Bundled MoE4All runtime is ready.'
exit 0
