[CmdletBinding()]
param(
    [string]$BinaryPath = 'target\release\infr.exe',
    [string]$OutputDirectory = 'dist',
    [string]$Version = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

$binaryFullPath = Resolve-RepoPath $BinaryPath
if (-not (Test-Path -LiteralPath $binaryFullPath -PathType Leaf)) {
    throw "Release executable not found: $binaryFullPath"
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    $metadataJson = & cargo metadata --no-deps --format-version 1
    if ($LASTEXITCODE -ne 0) {
        throw "cargo metadata failed with exit code $LASTEXITCODE"
    }
    $metadata = $metadataJson | ConvertFrom-Json
    $cliPackage = $metadata.packages | Where-Object { $_.name -eq 'infr-cli' } | Select-Object -First 1
    if ($null -eq $cliPackage) {
        throw 'Could not find the infr-cli package in cargo metadata.'
    }
    $Version = [string]$cliPackage.version
}

if ($Version -notmatch '^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$') {
    throw "Invalid release version: $Version"
}

$outputFullPath = Resolve-RepoPath $OutputDirectory
New-Item -ItemType Directory -Path $outputFullPath -Force | Out-Null

$packageName = "infr-windows-x86_64-$Version"
$stagingPath = Join-Path $outputFullPath $packageName
$outputPrefix = $outputFullPath.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $stagingPath.StartsWith($outputPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use a staging path outside the output directory: $stagingPath"
}
if (Test-Path -LiteralPath $stagingPath) {
    Remove-Item -LiteralPath $stagingPath -Recurse -Force
}

New-Item -ItemType Directory -Path (Join-Path $stagingPath 'scripts') -Force | Out-Null
Copy-Item -LiteralPath $binaryFullPath -Destination (Join-Path $stagingPath 'infr.exe')

$rootFiles = @(
    'Start-INFR-Wizard.cmd'
    'GETTING_STARTED.md'
    'README.md'
    'README_EN.md'
    'CHANGELOG.md'
    'infr.example.toml'
    'LICENSE'
    'LICENSE-MIT'
    'NOTICE'
)
foreach ($file in $rootFiles) {
    $sourcePath = Join-Path $repoRoot $file
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Required release file is missing: $sourcePath"
    }
    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $stagingPath $file)
}
Copy-Item -LiteralPath (Join-Path $repoRoot 'scripts\infr-wizard.ps1') -Destination (Join-Path $stagingPath 'scripts\infr-wizard.ps1')

$archivePath = Join-Path $outputFullPath "$packageName.zip"
$checksumPath = "$archivePath.sha256"
Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $checksumPath -Force -ErrorAction SilentlyContinue

Compress-Archive -LiteralPath $stagingPath -DestinationPath $archivePath -CompressionLevel Optimal
$hash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$checksumLine = "$hash  $([System.IO.Path]::GetFileName($archivePath))`n"
[System.IO.File]::WriteAllText($checksumPath, $checksumLine, [System.Text.UTF8Encoding]::new($false))
Remove-Item -LiteralPath $stagingPath -Recurse -Force

Write-Host "Created $archivePath"
Write-Host "Created $checksumPath"
