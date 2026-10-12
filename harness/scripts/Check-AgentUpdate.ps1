[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$InstallRoot,
    [Parameter(Mandatory = $true)][int]$ParentPid,
    [string]$FeedUri = 'https://api.github.com/repos/Headmaster218/MoE4All/releases?per_page=30',
    [string]$TestArchive,
    [string]$TestRoot
)

$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($InstallRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
$parent = [IO.Directory]::GetParent($root).FullName
$manifestPath = Join-Path $root 'harness\distribution\agent-release.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) { return }
$installed = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($installed.product -ne 'moe4all-agent') { return }
. (Join-Path $PSScriptRoot 'AgentUpdateVersion.ps1')
$testMode = [bool]$TestArchive
if ($testMode) {
    if ($env:MOE4ALL_AGENT_UPDATE_TEST -ne '1' -or -not $TestRoot) { throw 'Local updater testing is not enabled.' }
    $testBase = [IO.Path]::GetFullPath($TestRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $testZip = [IO.Path]::GetFullPath($TestArchive)
    if (-not $root.StartsWith($testBase + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
        -not $testZip.StartsWith($testBase + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Local updater test paths must stay inside TestRoot.'
    }
}

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

$best = $null
$zipAsset = $null
$sumAsset = $null
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
    } catch { return }
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
if ($null -eq $best) { return }

if (-not $testMode) {
    Add-Type -AssemblyName System.Windows.Forms
    $message = "MoE4All Agent $($best.tag_name) is available.`n`nUpdating will verify the full package, stop this Agent, preserve your data and installed Engine versions, and restart it.`n`nUpdate now?"
    $choice = [Windows.Forms.MessageBox]::Show($message, 'MoE4All Agent update', 'YesNo', 'Question')
    if ($choice -ne [Windows.Forms.DialogResult]::Yes) { return }
}

$work = Join-Path $parent ('.moe4all-update-' + [guid]::NewGuid().ToString('N'))
$download = Join-Path $work $zipAsset.name
$checksum = "$download.sha256"
$expanded = Join-Path $work 'expanded'
$newRoot = Join-Path $expanded ([IO.Path]::GetFileNameWithoutExtension($zipAsset.name))
$backup = Join-Path $parent ('.moe4all-previous-' + [guid]::NewGuid().ToString('N'))
$movedOld = $false
$movedNew = $false
$phase = 'download'
try {
    New-Item -ItemType Directory -Path $work, $expanded -Force | Out-Null
    if ($testMode) {
        Copy-Item -LiteralPath $testZip -Destination $download
        Copy-Item -LiteralPath "$testZip.sha256" -Destination $checksum
    } else {
        Invoke-WebRequest -UseBasicParsing -Uri $zipAsset.browser_download_url -Headers @{ 'User-Agent' = 'MoE4All-Agent-Updater' } -OutFile $download -TimeoutSec 900
        Invoke-WebRequest -UseBasicParsing -Uri $sumAsset.browser_download_url -Headers @{ 'User-Agent' = 'MoE4All-Agent-Updater' } -OutFile $checksum -TimeoutSec 30
    }
    $sumText = Get-Content -LiteralPath $checksum -Raw -Encoding UTF8
    $pattern = '(?im)^([a-f0-9]{64})\s+\*?' + [regex]::Escape($zipAsset.name) + '\s*$'
    if ($sumText -notmatch $pattern) { throw 'Release checksum file does not name the expected Agent archive.' }
    $actualHash = (Get-FileHash -LiteralPath $download -Algorithm SHA256).Hash
    if ($actualHash -ne $Matches[1]) { throw 'Agent archive SHA256 mismatch.' }
    $phase = 'extract'
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
    if (-not (Test-Path -LiteralPath (Join-Path $newRoot 'MoE4All.exe') -PathType Leaf) -or
        -not (Test-Path -LiteralPath $nextManifest -PathType Leaf)) { throw 'The Agent archive is incomplete.' }
    $next = Get-Content -LiteralPath $nextManifest -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($next.product -ne 'moe4all-agent' -or $next.version -ne $best.tag_name -or
        [string]$next.sourceCommit -notmatch '^[0-9a-f]{40}$') { throw 'Agent archive manifest does not match the release.' }

    $phase = 'stop old Agent'
    $process = Get-Process -Id $ParentPid -ErrorAction Stop
    $expectedExe = Join-Path $root 'MoE4All.exe'
    if (-not $process.Path.Equals($expectedExe, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'The running launcher no longer matches this installation.'
    }
    Stop-Process -Id $ParentPid -Force
    Wait-Process -Id $ParentPid -Timeout 30 -ErrorAction SilentlyContinue
    Set-Location -LiteralPath $parent
    $phase = 'move old Agent'
    for ($retry = 0; $retry -lt 30; $retry++) {
        try {
            Move-Item -LiteralPath $root -Destination $backup -ErrorAction Stop
            break
        } catch [IO.IOException] {
            if ($retry -eq 29) { throw }
            Start-Sleep -Seconds 1
        }
    }
    $movedOld = $true
    $phase = 'activate new Agent'
    Move-Item -LiteralPath $newRoot -Destination $root -ErrorAction Stop
    $movedNew = $true
    $phase = 'copy user state'
    $oldState = Join-Path $backup 'harness\agent-state'
    if (Test-Path -LiteralPath $oldState -PathType Container) {
        $newState = Join-Path $root 'harness\agent-state'
        & robocopy $oldState $newState /E /XJ /XD node_modules /R:1 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "Copying Agent state failed (robocopy $LASTEXITCODE)." }
    }
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
    if ($movedOld) {
        try {
            if ($movedNew) {
                if ($launched -and -not $launched.HasExited) { Stop-Process -Id $launched.Id -Force }
                Move-Item -LiteralPath $root -Destination (Join-Path $work 'failed-install')
            }
            Move-Item -LiteralPath $backup -Destination $root
            Start-Process -FilePath (Join-Path $root 'MoE4All.exe') -WorkingDirectory $root -WindowStyle Normal | Out-Null
        } catch { $detail += "`nAutomatic rollback also failed: $($_.Exception.Message)" }
    }
    if ($testMode) { throw "Agent update test failed: $detail" }
    [Windows.Forms.MessageBox]::Show("Agent update failed: $detail", 'MoE4All Agent', 'OK', 'Error') | Out-Null
}
