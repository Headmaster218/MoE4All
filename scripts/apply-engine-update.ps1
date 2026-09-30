[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InstallRoot,
    [Parameter(Mandatory = $true)][string]$StagedPackageRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedVersion,
    [int]$ParentProcessId = 0,
    [string]$CleanupRoot = '',
    [switch]$DeleteSelf,
    [switch]$NoRelaunch
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$InstallRoot = [System.IO.Path]::GetFullPath($InstallRoot).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
$StagedPackageRoot = [System.IO.Path]::GetFullPath($StagedPackageRoot).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
$commonPath = Join-Path $StagedPackageRoot 'scripts\moe4all-update-common.ps1'
if (-not (Test-Path -LiteralPath $commonPath -PathType Leaf)) {
    throw "Update package is missing $commonPath"
}
. $commonPath

function Start-UpdatedWizard {
    param([Parameter(Mandatory = $true)][string]$Root)

    if ($NoRelaunch) { return }
    $launcher = Join-Path $Root 'Start-INFR-Wizard.cmd'
    if (Test-Path -LiteralPath $launcher -PathType Leaf) {
        Start-Process -FilePath $launcher -ArgumentList '-SkipUpdateCheck' -WorkingDirectory $Root | Out-Null
    }
}

function Write-UpdateResult {
    param(
        [Parameter(Mandatory = $true)][ValidateSet('success', 'error')][string]$Status,
        [Parameter(Mandatory = $true)][string]$Message
    )

    $updateRoot = Join-Path $InstallRoot '.update'
    New-Item -ItemType Directory -Path $updateRoot -Force | Out-Null
    $result = [ordered]@{
        status = $Status
        message = $Message
        version = $ExpectedVersion
        at = [DateTimeOffset]::Now.ToString('o')
    }
    $json = $result | ConvertTo-Json
    [System.IO.File]::WriteAllText((Join-Path $updateRoot 'last-update-result.json'), $json, [System.Text.UTF8Encoding]::new($false))
}

if ($ParentProcessId -gt 0) {
    try {
        $parent = Get-Process -Id $ParentProcessId -ErrorAction Stop
        [void]$parent.WaitForExit(30000)
    } catch {
        # The parent may already have exited before the helper starts.
    }
}
Start-Sleep -Milliseconds 1200

$manifestPath = Join-Path $StagedPackageRoot 'install-manifest.json'
$manifest = Read-Moe4AllInstallManifest -ManifestPath $manifestPath -PackageRoot $StagedPackageRoot -VerifyFiles
if ([string]$manifest.version -ne $ExpectedVersion) {
    throw "Staged version $($manifest.version) does not match expected version $ExpectedVersion"
}

$oldManifestPath = Join-Path $InstallRoot 'install-manifest.json'
$oldManifest = $null
if (Test-Path -LiteralPath $oldManifestPath -PathType Leaf) {
    try {
        $oldManifest = Read-Moe4AllInstallManifest -ManifestPath $oldManifestPath -PackageRoot $InstallRoot
    } catch {
        Write-Warning "The previous install manifest is invalid; obsolete product files will be preserved. $($_.Exception.Message)"
    }
}

$timestamp = [DateTime]::Now.ToString('yyyyMMdd-HHmmss')
$oldVersion = if ($null -eq $oldManifest) { 'unknown' } else { [string]$oldManifest.version }
$backupRoot = Join-Path $InstallRoot ".update\backups\$oldVersion-$timestamp"
New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null

$newEntries = @{}
foreach ($entry in @($manifest.files)) { $newEntries[[string]$entry.path] = $entry }
$oldEntries = @{}
if ($null -ne $oldManifest) {
    foreach ($entry in @($oldManifest.files)) { $oldEntries[[string]$entry.path] = $entry }
}
$allPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($path in $newEntries.Keys) { [void]$allPaths.Add($path) }
foreach ($path in $oldEntries.Keys) { [void]$allPaths.Add($path) }

$originalPaths = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
foreach ($relative in $allPaths) {
    $target = Resolve-Moe4AllManagedPath -Root $InstallRoot -RelativePath $relative
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        [void]$originalPaths.Add($relative)
        $backup = Resolve-Moe4AllManagedPath -Root $backupRoot -RelativePath $relative
        New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($backup)) -Force | Out-Null
        Copy-Item -LiteralPath $target -Destination $backup -Force
    }
}
$hadManifest = Test-Path -LiteralPath $oldManifestPath -PathType Leaf
if ($hadManifest) {
    Copy-Item -LiteralPath $oldManifestPath -Destination (Join-Path $backupRoot 'install-manifest.json') -Force
}

try {
    foreach ($relative in $newEntries.Keys) {
        $source = Resolve-Moe4AllManagedPath -Root $StagedPackageRoot -RelativePath $relative
        $target = Resolve-Moe4AllManagedPath -Root $InstallRoot -RelativePath $relative
        $directory = [System.IO.Path]::GetDirectoryName($target)
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        $temporary = "$target.moe4all-new-$([guid]::NewGuid().ToString('N'))"
        Copy-Item -LiteralPath $source -Destination $temporary -Force
        Move-Item -LiteralPath $temporary -Destination $target -Force
    }
    foreach ($relative in $oldEntries.Keys) {
        if ($newEntries.ContainsKey($relative)) { continue }
        $target = Resolve-Moe4AllManagedPath -Root $InstallRoot -RelativePath $relative
        Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
    }

    $temporaryManifest = "$oldManifestPath.moe4all-new-$([guid]::NewGuid().ToString('N'))"
    Copy-Item -LiteralPath $manifestPath -Destination $temporaryManifest -Force
    Move-Item -LiteralPath $temporaryManifest -Destination $oldManifestPath -Force
    [void](Read-Moe4AllInstallManifest -ManifestPath $oldManifestPath -PackageRoot $InstallRoot -VerifyFiles)

    $binaryPath = Join-Path $InstallRoot 'infr.exe'
    $versionOutput = (& $binaryPath --version 2>$null | Out-String).Trim()
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -or $versionOutput -notmatch [regex]::Escape("infr $ExpectedVersion")) {
        throw "Updated infr.exe failed version validation: $versionOutput"
    }

    Write-UpdateResult -Status success -Message "MoE4All Engine was updated to v$ExpectedVersion."
    if (-not [string]::IsNullOrWhiteSpace($CleanupRoot)) {
        $cleanup = [System.IO.Path]::GetFullPath($CleanupRoot).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
        $updateRoot = [System.IO.Path]::GetFullPath((Join-Path $InstallRoot '.update')).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
        $updatePrefix = $updateRoot + [System.IO.Path]::DirectorySeparatorChar
        if ($cleanup.StartsWith($updatePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $cleanup -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    Write-Host "MoE4All Engine updated to v$ExpectedVersion." -ForegroundColor Green
    Start-UpdatedWizard -Root $InstallRoot
} catch {
    $failure = $_
    Write-Warning "Update failed; restoring the previous installation. $($failure.Exception.Message)"
    foreach ($relative in $allPaths) {
        $target = Resolve-Moe4AllManagedPath -Root $InstallRoot -RelativePath $relative
        $backup = Resolve-Moe4AllManagedPath -Root $backupRoot -RelativePath $relative
        if ($originalPaths.Contains($relative) -and (Test-Path -LiteralPath $backup -PathType Leaf)) {
            New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($target)) -Force | Out-Null
            Copy-Item -LiteralPath $backup -Destination $target -Force
        } else {
            Remove-Item -LiteralPath $target -Force -ErrorAction SilentlyContinue
        }
    }
    $backupManifest = Join-Path $backupRoot 'install-manifest.json'
    if ($hadManifest -and (Test-Path -LiteralPath $backupManifest -PathType Leaf)) {
        Copy-Item -LiteralPath $backupManifest -Destination $oldManifestPath -Force
    } else {
        Remove-Item -LiteralPath $oldManifestPath -Force -ErrorAction SilentlyContinue
    }
    Write-UpdateResult -Status error -Message "Engine update failed and was rolled back: $($failure.Exception.Message)"
    Start-UpdatedWizard -Root $InstallRoot
    throw
} finally {
    if ($DeleteSelf -and -not [string]::IsNullOrWhiteSpace($PSCommandPath)) {
        Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue
    }
}
