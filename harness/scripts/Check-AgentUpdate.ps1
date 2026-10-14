[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InstallRoot,
    [Parameter(Mandatory = $true)][int]$ParentPid,
    [ValidateSet('Legacy', 'Check', 'Install')][string]$Action = 'Legacy',
    [string]$FeedUri = 'https://api.github.com/repos/Headmaster218/MoE4All/releases?per_page=30',
    [string]$TestArchive,
    [string]$TestRoot,
    [switch]$TestInstallOnly
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($InstallRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
$manifestPath = Join-Path $root 'harness\distribution\agent-release.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { return }
$installed = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($installed.product -ne 'moe4all-agent') { return }
. (Join-Path $PSScriptRoot 'AgentUpdateVersion.ps1')

function Write-UpdateStatus([string]$Kind, [long]$Completed, [long]$Total, [string]$Message) {
    $safeMessage = ([string]$Message).Replace("`r", ' ').Replace("`n", ' ').Replace("`t", ' ')
    Write-Output "@@MOE4ALL_UPDATE`t$Kind`t$Completed`t$Total`t$safeMessage"
    [Console]::Out.Flush()
}

function Receive-UpdateFile([Uri]$Uri, [string]$Destination, [long]$ExpectedBytes) {
    $request = [Net.HttpWebRequest]::Create($Uri)
    $request.UserAgent = 'MoE4All-Agent-Updater'
    $request.Timeout = 900000
    $request.ReadWriteTimeout = 900000
    $response = $request.GetResponse()
    $inputStream = $null
    $outputStream = $null
    try {
        $total = if ($response.ContentLength -gt 0) { [long]$response.ContentLength } else { $ExpectedBytes }
        $inputStream = $response.GetResponseStream()
        $outputStream = New-Object IO.FileStream($Destination, ([IO.FileMode]::Create), ([IO.FileAccess]::Write), ([IO.FileShare]::None), 1048576, ([IO.FileOptions]::SequentialScan))
        $buffer = New-Object byte[] 1048576
        $completed = [long]0
        $clock = [Diagnostics.Stopwatch]::StartNew()
        Write-UpdateStatus 'download' 0 $total 'Downloading update'
        while (($read = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
            $outputStream.Write($buffer, 0, $read)
            $completed += $read
            if ($clock.ElapsedMilliseconds -ge 200 -or ($total -gt 0 -and $completed -ge $total)) {
                Write-UpdateStatus 'download' $completed $total 'Downloading update'
                $clock.Restart()
            }
        }
        $outputStream.Flush()
        Write-UpdateStatus 'download' $completed $total 'Download complete'
    } finally {
        if ($outputStream) { $outputStream.Dispose() }
        if ($inputStream) { $inputStream.Dispose() }
        $response.Dispose()
    }
}

$testMode = [bool]$TestArchive
if ($testMode) {
    if ($env:MOE4ALL_AGENT_UPDATE_TEST -ne '1' -or -not $TestRoot) { throw 'Local updater testing is not enabled.' }
    $testBase = [IO.Path]::GetFullPath($TestRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $testZip = [IO.Path]::GetFullPath($TestArchive)
    if (-not $root.StartsWith($testBase + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
        -not $testZip.StartsWith($testBase + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Local updater test paths must stay inside TestRoot.'
    }
} elseif ($TestInstallOnly) {
    throw 'TestInstallOnly is available only for a guarded local updater test.'
}

$updatesRoot = [IO.Path]::GetFullPath((Join-Path $root 'updates')).TrimEnd([IO.Path]::DirectorySeparatorChar)
New-Item -ItemType Directory -Path $updatesRoot -Force | Out-Null

function Assert-UpdatePath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($updatesRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Updater path escaped the product update directory: $full"
    }
    return $full
}

$managedNames = @(
    'MoE4All.exe',
    'MoE4All.exe.config',
    'node.exe',
    'Microsoft.Web.WebView2.Core.dll',
    'Microsoft.Web.WebView2.WinForms.dll',
    'WebView2Loader.dll',
    'Microsoft.Web.WebView2.LICENSE.txt',
    'harness'
)

function Move-ManagedItems([string]$Source, [string]$Destination) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    foreach ($name in $managedNames) {
        $item = Join-Path $Source $name
        if (Test-Path -LiteralPath $item) {
            Move-Item -LiteralPath $item -Destination (Join-Path $Destination $name) -ErrorAction Stop
        }
    }
}

if (-not $TestInstallOnly -and -not $testMode) {
    # The checker is detached from the launcher and waits until first-run setup finishes.
    for ($attempt = 0; $attempt -lt 600; $attempt++) {
        if (-not (Get-Process -Id $ParentPid -ErrorAction SilentlyContinue)) { return }
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:17373/' -TimeoutSec 3
            if ($response.StatusCode -eq 200) { break }
        } catch {}
        Start-Sleep -Seconds 2
    }
    if ($attempt -ge 600) { return }
}

$best = $null
$zipAsset = $null
$sumAsset = $null
if ($Action -ne 'Legacy') { Write-UpdateStatus 'checking' 0 0 'Checking for updates' }
if ($testMode) {
    $zipName = [IO.Path]::GetFileName($testZip)
    if ($zipName -notmatch '^MoE4All-Agent-Windows-x64-(agent-v[0-9]+\.[0-9]+\.[0-9]+(?:-preview\.[0-9]+)?)\.zip$') {
        throw 'Test archive name is not an Agent release asset.'
    }
    $tag = $Matches[1]
    if (-not (Test-AgentVersionNewer $tag ([string]$installed.version))) { throw 'Test archive is not newer.' }
    $best = @{ tag_name = $tag }
    $zipAsset = @{ name = $zipName }
} else {
    try {
        $releases = Invoke-RestMethod -Uri $FeedUri -Headers @{ 'User-Agent' = 'MoE4All-Agent-Updater' } -TimeoutSec 15
    } catch {
        if ($Action -ne 'Legacy') {
            Write-UpdateStatus 'error' 0 0 ("Could not check for updates: " + $_.Exception.Message)
        }
        return
    }
    foreach ($release in @($releases)) {
        $tag = [string]$release.tag_name
        if ($release.draft -eq $true -or -not (Test-AgentVersionNewer $tag ([string]$installed.version))) { continue }
        if ($null -ne $best -and -not (Test-AgentVersionNewer $tag ([string]$best.tag_name))) { continue }
        $zipName = "MoE4All-Agent-Windows-x64-$tag.zip"
        $zip = @($release.assets | Where-Object name -eq $zipName | Select-Object -First 1)[0]
        $sum = @($release.assets | Where-Object name -eq "$zipName.sha256" | Select-Object -First 1)[0]
        if ($null -eq $zip -or $null -eq $sum) { continue }
        try {
            $zipUri = [Uri]$zip.browser_download_url
            $sumUri = [Uri]$sum.browser_download_url
        } catch { continue }
        $prefix = "/Headmaster218/MoE4All/releases/download/$tag/"
        if ($zipUri.Scheme -ne 'https' -or $sumUri.Scheme -ne 'https' -or
            $zipUri.Host -ne 'github.com' -or $sumUri.Host -ne 'github.com' -or
            -not $zipUri.AbsolutePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase) -or
            -not $sumUri.AbsolutePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { continue }
        $best, $zipAsset, $sumAsset = $release, $zip, $sum
    }
}
if ($null -eq $best) {
    if ($Action -ne 'Legacy') { Write-UpdateStatus 'none' 0 0 ([string]$installed.version) }
    return
}

if ($Action -eq 'Check') {
    $availableBytes = if ($zipAsset.size) { [long]$zipAsset.size } elseif ($testMode) { (Get-Item -LiteralPath $testZip).Length } else { 0 }
    Write-UpdateStatus 'available' 0 $availableBytes ([string]$best.tag_name)
    return
}

if (-not $testMode -and $Action -eq 'Legacy') {
    Add-Type -AssemblyName System.Windows.Forms
    $message = "MoE4All Agent $($best.tag_name) is available.`n`nThe update replaces the application only. Models, conversations and caches stay in this directory. One previous application version is kept for rollback.`n`nUpdate now?"
    $choice = [Windows.Forms.MessageBox]::Show($message, 'MoE4All Agent update', 'YesNo', 'Question')
    if ($choice -ne [Windows.Forms.DialogResult]::Yes) { return }
}

$work = Assert-UpdatePath (Join-Path $updatesRoot ('staging-' + [guid]::NewGuid().ToString('N')))
$download = Join-Path $work $zipAsset.name
$checksum = "$download.sha256"
$expanded = Join-Path $work 'expanded'
$newRoot = Join-Path $expanded ([IO.Path]::GetFileNameWithoutExtension($zipAsset.name))
$backup = Assert-UpdatePath (Join-Path $updatesRoot 'previous')
$failed = Assert-UpdatePath (Join-Path $updatesRoot ('failed-' + [guid]::NewGuid().ToString('N')))
$oldMoved = $false
$newMoved = $false
$launched = $null
$phase = 'download'
try {
    New-Item -ItemType Directory -Path $work, $expanded -Force | Out-Null
    if ($testMode) {
        Copy-Item -LiteralPath $testZip -Destination $download
        Copy-Item -LiteralPath "$testZip.sha256" -Destination $checksum
        $testBytes = (Get-Item -LiteralPath $download).Length
        Write-UpdateStatus 'download' $testBytes $testBytes 'Download complete'
    } else {
        $expectedBytes = if ($zipAsset.size) { [long]$zipAsset.size } else { 0 }
        Receive-UpdateFile ([Uri]$zipAsset.browser_download_url) $download $expectedBytes
        Invoke-WebRequest -UseBasicParsing -Uri $sumAsset.browser_download_url -Headers @{ 'User-Agent' = 'MoE4All-Agent-Updater' } -OutFile $checksum -TimeoutSec 30
    }
    Write-UpdateStatus 'verify' 0 0 'Verifying downloaded package'
    $sumText = Get-Content -LiteralPath $checksum -Raw -Encoding UTF8
    $pattern = '(?im)^([a-f0-9]{64})\s+\*?' + [regex]::Escape($zipAsset.name) + '\s*$'
    if ($sumText -notmatch $pattern) { throw 'Release checksum file does not name the expected Agent archive.' }
    $actualHash = (Get-FileHash -LiteralPath $download -Algorithm SHA256).Hash
    if ($actualHash -ne $Matches[1]) { throw 'Agent archive SHA256 mismatch.' }

    $phase = 'extract'
    Write-UpdateStatus 'extract' 0 0 'Preparing the new application'
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $reader = [IO.Compression.ZipFile]::OpenRead($download)
    try {
        $expectedTop = [IO.Path]::GetFileNameWithoutExtension($zipAsset.name)
        foreach ($entry in $reader.Entries) {
            $parts = $entry.FullName -split '[\\/]'
            if ($parts.Count -lt 2 -or $parts[0] -ne $expectedTop -or
                $parts -contains '..' -or $entry.FullName.Contains(':')) {
                throw "Unexpected path in Agent archive: $($entry.FullName)"
            }
        }
    } finally { $reader.Dispose() }
    Expand-Archive -LiteralPath $download -DestinationPath $expanded
    $nextManifest = Join-Path $newRoot 'harness\distribution\agent-release.json'
    foreach ($required in @('MoE4All.exe', 'MoE4All.exe.config', 'harness', 'harness\bootstrap\node.exe', 'Microsoft.Web.WebView2.Core.dll', 'Microsoft.Web.WebView2.WinForms.dll', 'WebView2Loader.dll')) {
        if (-not (Test-Path -LiteralPath (Join-Path $newRoot $required))) { throw "The Agent archive is missing $required." }
    }
    if (-not (Test-Path -LiteralPath $nextManifest -PathType Leaf)) { throw 'The Agent archive manifest is missing.' }
    $next = Get-Content -LiteralPath $nextManifest -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($next.product -ne 'moe4all-agent' -or $next.version -ne $best.tag_name -or
        [string]$next.sourceCommit -notmatch '^[0-9a-f]{40}$') { throw 'Agent archive manifest does not match the release.' }

    if (-not $TestInstallOnly) {
        $phase = 'stop old Agent'
        Write-UpdateStatus 'install' 0 0 'Installing update and restarting'
        $process = Get-Process -Id $ParentPid -ErrorAction Stop
        $expectedExe = Join-Path $root 'MoE4All.exe'
        if (-not $process.Path.Equals($expectedExe, [StringComparison]::OrdinalIgnoreCase)) {
            throw 'The running launcher no longer matches this installation.'
        }
        Stop-Process -Id $ParentPid -Force
        Wait-Process -Id $ParentPid -Timeout 30 -ErrorAction SilentlyContinue
    }

    Set-Location -LiteralPath $updatesRoot
    $phase = 'rotate previous application'
    if (Test-Path -LiteralPath $backup) {
        Assert-UpdatePath $backup | Out-Null
        Remove-Item -LiteralPath $backup -Recurse -Force
    }
    New-Item -ItemType Directory -Path $backup -Force | Out-Null
    Move-ManagedItems $root $backup
    $oldMoved = $true

    $phase = 'activate new application'
    Move-ManagedItems $newRoot $root
    $newMoved = $true

    if ($TestInstallOnly) { return }

    $phase = 'start new Agent'
    $launched = Start-Process -FilePath (Join-Path $root 'MoE4All.exe') -WorkingDirectory $root -WindowStyle Normal -PassThru
    for ($attempt = 0; $attempt -lt 600; $attempt++) {
        if ($launched.HasExited) { throw "Updated Agent exited with code $($launched.ExitCode)." }
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:17373/' -TimeoutSec 3
            if ($response.StatusCode -eq 200) { return }
        } catch {}
        Start-Sleep -Seconds 2
    }
    throw 'Updated Agent did not become healthy within 20 minutes.'
} catch {
    $detail = "${phase}: $($_.Exception.Message)"
    if ($oldMoved) {
        try {
            if ($launched -and -not $launched.HasExited) { Stop-Process -Id $launched.Id -Force }
            if ($newMoved) { Move-ManagedItems $root $failed }
            Move-ManagedItems $backup $root
            if (Test-Path -LiteralPath $backup) { Remove-Item -LiteralPath $backup -Recurse -Force }
            if (Test-Path -LiteralPath $failed) { Remove-Item -LiteralPath $failed -Recurse -Force }
            if (-not $TestInstallOnly) {
                Start-Process -FilePath (Join-Path $root 'MoE4All.exe') -WorkingDirectory $root -WindowStyle Normal | Out-Null
            }
        } catch { $detail += "`nAutomatic rollback also failed: $($_.Exception.Message)" }
    }
    if ($testMode) { throw "Agent update test failed: $detail" }
    if ($Action -eq 'Legacy') {
        [Windows.Forms.MessageBox]::Show("Agent update failed: $detail", 'MoE4All Agent', 'OK', 'Error') | Out-Null
    } else {
        Write-UpdateStatus 'error' 0 0 ("Update failed: " + $detail)
    }
} finally {
    if (Test-Path -LiteralPath $work) {
        Assert-UpdatePath $work | Out-Null
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
    }
}
