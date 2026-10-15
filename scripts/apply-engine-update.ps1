[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InstallRoot,
    [Parameter(Mandatory = $true)][string]$StagedPackageRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedVersion,
    [int]$ParentProcessId = 0,
    [int]$LauncherProcessId = 0,
    [string]$CleanupRoot = '',
    [switch]$DeleteSelf,
    [switch]$NoRelaunch,
    [switch]$RelaunchProbeOnly,
    [switch]$NoFailureRelaunch
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

function Stop-StartedWizardTree {
    param([Parameter(Mandatory = $true)][System.Diagnostics.Process]$Process)

    try {
        if ($Process.HasExited) { return }
        & taskkill.exe /PID $Process.Id /T /F *> $null
    } catch {
        try { Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
}

function Start-UpdatedWizard {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [string]$ExpectedVersion = '',
        [switch]$SkipUpdateCheck,
        [switch]$RequireReadinessProbe
    )

    if ($NoRelaunch) { return $false }
    $launcher = Join-Path $Root 'Start-INFR-Wizard.cmd'
    if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
        throw "The updated launcher is missing: $launcher"
    }

    $launcherArguments = [System.Collections.Generic.List[string]]::new()
    if ($SkipUpdateCheck) { $launcherArguments.Add('-SkipUpdateCheck') }
    $probePath = ''
    if ($RequireReadinessProbe) {
        $probePath = Join-Path $Root ('.update\relaunch-' + [guid]::NewGuid().ToString('N') + '.json')
        $launcherArguments.Add('-PostUpdateProbePath')
        $launcherArguments.Add('"' + $probePath.Replace('"', '""') + '"')
        if ($ExpectedVersion) {
            $launcherArguments.Add('-ExpectedPostUpdateVersion')
            $launcherArguments.Add('"' + $ExpectedVersion.Replace('"', '""') + '"')
        }
        if ($RelaunchProbeOnly) { $launcherArguments.Add('-ExitAfterPostUpdateProbe') }
    }

    $quotedLauncher = '"' + $launcher.Replace('"', '""') + '"'
    $commandLine = '/d /s /c "' + $quotedLauncher + ' ' + ($launcherArguments -join ' ') + '"'
    $process = Start-Process -FilePath $env:ComSpec -ArgumentList $commandLine -WorkingDirectory $Root -WindowStyle Normal -PassThru
    if (-not $RequireReadinessProbe) { return $true }

    $deadline = [DateTime]::UtcNow.AddSeconds(20)
    while ([DateTime]::UtcNow -lt $deadline) {
        if (Test-Path -LiteralPath $probePath -PathType Leaf) { break }
        if ($process.HasExited) {
            throw "The updated launcher exited before reporting readiness (exit code $($process.ExitCode))."
        }
        Start-Sleep -Milliseconds 100
    }
    if (-not (Test-Path -LiteralPath $probePath -PathType Leaf)) {
        Stop-StartedWizardTree -Process $process
        throw 'The updated launcher did not report readiness within 20 seconds.'
    }

    try {
        $probe = Get-Content -LiteralPath $probePath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($ExpectedVersion -and [string]$probe.version -ne $ExpectedVersion) {
            throw "The updated launcher reported v$($probe.version); expected v$ExpectedVersion."
        }
        if ($RelaunchProbeOnly -and -not $process.WaitForExit(10000)) {
            throw 'The probe-only updated launcher did not exit after reporting readiness.'
        }
    } catch {
        Stop-StartedWizardTree -Process $process
        throw
    } finally {
        Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
    }
    return $true
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

if ($LauncherProcessId -le 0 -and $ParentProcessId -gt 0) {
    try {
        $wizardProcess = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $ParentProcessId" -ErrorAction Stop
        $launcherProcess = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $([int]$wizardProcess.ParentProcessId)" -ErrorAction Stop
        $launcherCommandLine = [string]$launcherProcess.CommandLine
        if ([string]$launcherProcess.Name -ieq 'cmd.exe' -and $launcherCommandLine -match '(?i)(?:^|\s)/c(?:\s|$)' -and $launcherCommandLine -match '(?i)Start-INFR-Wizard\.cmd') {
            $LauncherProcessId = [int]$launcherProcess.ProcessId
        }
    } catch {
        $LauncherProcessId = 0
    }
}

if ($ParentProcessId -gt 0) {
    try {
        $parent = Get-Process -Id $ParentProcessId -ErrorAction Stop
        if (-not $parent.WaitForExit(30000)) {
            throw 'The previous wizard did not exit within 30 seconds.'
        }
    } catch {
        if (Get-Process -Id $ParentProcessId -ErrorAction SilentlyContinue) { throw }
        # The parent may already have exited before the helper starts.
    }
}
if ($LauncherProcessId -gt 0) {
    try {
        $launcherProcess = Get-Process -Id $LauncherProcessId -ErrorAction Stop
        if (-not $launcherProcess.WaitForExit(30000)) {
            throw 'The previous CMD launcher did not exit within 30 seconds.'
        }
    } catch {
        if (Get-Process -Id $LauncherProcessId -ErrorAction SilentlyContinue) { throw }
        # The transient cmd.exe launcher may already have exited.
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
    $relaunched = Start-UpdatedWizard -Root $InstallRoot -ExpectedVersion $ExpectedVersion -SkipUpdateCheck -RequireReadinessProbe
    if (-not [string]::IsNullOrWhiteSpace($CleanupRoot)) {
        $cleanup = [System.IO.Path]::GetFullPath($CleanupRoot).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
        $updateRoot = [System.IO.Path]::GetFullPath((Join-Path $InstallRoot '.update')).TrimEnd([System.IO.Path]::DirectorySeparatorChar)
        $updatePrefix = $updateRoot + [System.IO.Path]::DirectorySeparatorChar
        if ($cleanup.StartsWith($updatePrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $cleanup -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    if ($relaunched -and (Test-Path -LiteralPath $backupRoot)) {
        Remove-Item -LiteralPath $backupRoot -Recurse -Force
    }
    Write-Host "MoE4All Engine updated to v$ExpectedVersion." -ForegroundColor Green
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
    if (-not $NoFailureRelaunch) {
        try {
            [void](Start-UpdatedWizard -Root $InstallRoot)
        } catch {
            Write-Warning "The previous installation was restored, but its launcher could not be reopened. $($_.Exception.Message)"
        }
    }
    throw $failure
} finally {
    if ($DeleteSelf -and -not [string]::IsNullOrWhiteSpace($PSCommandPath)) {
        Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue
    }
}
