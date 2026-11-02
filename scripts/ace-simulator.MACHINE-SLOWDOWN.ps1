<#
.SYNOPSIS
    DANGER - DELIBERATELY SLOWS THE MACHINE DOWN. Opt-in only. Nothing in MoE4All calls this.

.DESCRIPTION
    An affectionate reproduction of the Tencent ACE anti-cheat feeling: it walks the whole disk
    looking for model files, SHA-256s every one of them across all logical processors at once,
    then establishes - without sending HuggingFace a single request - whether the hub is reachable,
    pauses for exactly as long as an anti-cheat verification takes when it has nothing to say, and
    reports the verdict as FAILED.

    The no-requests rule is not shyness, it is the point: a joke script has no business generating
    model-info traffic against a public service, so phase 3 opens one TCP connection to host:port,
    learns what it can learn from that, and closes it. No HTTP verb, no path, no TLS handshake,
    no User-Agent, no retry, no bulk query. The hashes stay local and the answer stays "FAILED",
    which is also the most honest verdict a script that never asked can produce.

    This is a joke, and jokes still cost the person running them real seconds. It is therefore
    quadruple-fenced:

      * NOT wired into Start-INFR-Wizard.cmd, infr-wizard.ps1, the engine, the release scripts or
        CI. Nothing runs it automatically. If you want it in a startup path, do not file a bug
        about it - that is the one version of this that would hurt other people.
      * it refuses to do anything at all unless you pass -IAccept.
      * and then, even with -IAccept given, it prints the warning and asks you to type `y` on the
        console before starting. The default answer is No, Enter means No, a wrong key means No,
        and -PromptTimeoutSec seconds of silence means No.
      * -DryRun shows the whole plan (which drives, which extensions, how many files) without
        hashing, probing or asking anything, and -LimitFiles / -Root narrow the blast radius
        further.

    The confirmation is asked fresh on every single start. There is no "don't ask again", no
    remembered answer, no environment variable, no config key and no profile entry involved - the
    script reads nothing about you and writes nothing about you. Same for the timeout:
    `-PromptTimeoutSec 1` is not a new default for next time, because there is no next-time state
    at all.

    Read-only by construction: it opens files for reading, and never writes, moves or deletes a
    single byte outside its own console output.

.PARAMETER Root
    Where to look. Default: every volume the operating system reports as mounted - fixed,
    removable and network alike, whatever the machine has, counted at start-up rather than
    hardcoded. Pass one or more paths to stay in a corner.

.PARAMETER Extension
    File extensions treated as "models the engine can load". Default: .gguf and .mmproj.

.PARAMETER Degree
    Worker threads for the hash phase. Default: one per logical processor, i.e. the good part.

.PARAMETER HubEndpoint
    Which host to sniff for reachability. Honours INFR_HF_ENDPOINT, same knob the engine uses, so
    a mirror works. Only the host and port are used - no request is ever sent to it.

.PARAMETER HangSec
    The ACE-flavoured pause: how long phase 3 stares at the ceiling before delivering its verdict.
    Default 30; 0 reports immediately. Applies whether or not the hub answered, because the
    discomfort is the feature.

.PARAMETER ConnectTimeoutSec
    How long the TCP reachability probe waits. Default 5. A probe failure is not an error here,
    it is just the offline branch.

.PARAMETER TolerateOffline
    Do not die when the endpoint is unreachable; fall through to the pause and the verdict.
    Equivalent to INFR_HF_OFFLINE=1, which is the knob the engine family already uses. Without
    either, an unreachable hub exits 1 on purpose - that is the original punchline, not a bug.

.PARAMETER DryRun
    Enumerate, report the plan, hash nothing, probe nothing.

.PARAMETER LimitFiles
    Cap the number of files hashed (0 = no cap). Handy while having a laugh.

.PARAMETER PromptTimeoutSec
    How long the startup confirmation waits for an answer before the answer becomes No.
    Default 300; 0 disables the timeout and waits forever. Never persisted, never inherited:
    every start re-asks with the default unless the flag is typed again.

.PARAMETER IAccept
    Necessary but not sufficient. Lets the script reach the interactive confirmation at all;
    it does not answer it.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\ace-simulator.MACHINE-SLOWDOWN.ps1 -DryRun

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\ace-simulator.MACHINE-SLOWDOWN.ps1 -IAccept -Root F:\models

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\ace-simulator.MACHINE-SLOWDOWN.ps1 -IAccept -PromptTimeoutSec 0

.EXAMPLE
    $env:INFR_HF_OFFLINE = '1'; powershell -ExecutionPolicy Bypass -File scripts\ace-simulator.MACHINE-SLOWDOWN.ps1 -IAccept

.NOTES
    Windows PowerShell 5.1 compatible on purpose (no ForEach-Object -Parallel), matching the other
    scripts in this folder.

    Exit codes: 0 clean, 1 hub unreachable and -TolerateOffline / INFR_HF_OFFLINE not given,
    2 refused at a fence, 3 the hub was reachable and verification still FAILED - as it must, since
    nothing was ever asked.
#>
[CmdletBinding()]
param(
    [string[]]$Root = @(),
    [string[]]$Extension = @('.gguf', '.mmproj'),
    [int]$Degree = 0,
    [string]$HubEndpoint = '',
    [switch]$TolerateOffline,
    [switch]$DryRun,
    [switch]$IAccept,
    [int]$LimitFiles = 0,
    [int]$PromptTimeoutSec = 300,
    [int]$HangSec = 30,
    [int]$ConnectTimeoutSec = 5
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# PS 5.1 otherwise writes native output through the active OEM codepage.
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)

if (-not $HubEndpoint) {
    $HubEndpoint = if ($env:INFR_HF_ENDPOINT) { $env:INFR_HF_ENDPOINT } else { 'https://huggingface.co' }
}
if ($Degree -le 0) { $Degree = [Environment]::ProcessorCount }

# ── which drives: ask the operating system, then take all of them ────────────
# No drive list to maintain and no count to edit when someone moves the model pack onto a second
# card. [IO.DriveInfo]::GetDrives() is the OS answer to "what is mounted right now", so the script
# covers a 2-volume laptop and a 12-volume workstation with the same line, and it reports the KIND
# of each volume - which is what lets a CD-ROM, a RAM disk or a card reader that lost its media be
# filtered on a fact instead of on a guessed letter. `-Root` still overrides: pointing this at one
# folder is the difference between a joke and a bad afternoon.
$mounted = [Collections.Generic.List[object]]::new()
foreach ($drive in [IO.DriveInfo]::GetDrives()) {
    try {
        if (-not $drive.IsReady) { continue }
        $isScannable = $drive.DriveType -in @(
            [IO.DriveType]::Fixed, [IO.DriveType]::Removable, [IO.DriveType]::Network)
        if ($isScannable) { $mounted.Add($drive) }
    } catch {
        # A volume that cannot even answer `IsReady` is not one we will walk. Say so, keep going.
        Write-Warning "a mounted drive refused to be described: $($_.Exception.Message)"
    }
}
$autoDetected = -not $Root -or $Root.Count -eq 0
if ($autoDetected) {
    $Root = @($mounted | ForEach-Object { $_.Name })
}

# The offline switch has two spellings and neither is persisted anywhere.
$offlineAllowed = [bool]$TolerateOffline -or ($env:INFR_HF_OFFLINE -and $env:INFR_HF_OFFLINE -ne '0')

# Ask on the console, with a deadline, and mean No on every ambiguity. Deliberately no
# `Read-Host`: a piped / `-File`-redirected stdin is not a human typing `y`, and redirected input
# must not be able to start this. `Enter` and the timeout are both answers-in-themselves, so the
# caller only has to compare against 'y'.
function Read-Confirmation {
    param(
        [string]$Question,
        [int]$TimeoutSec
    )
    if (-not [Console]::IsInputRedirected) {
        Write-Host -NoNewline $Question
    }
    $clock = [Diagnostics.Stopwatch]::StartNew()
    while ($true) {
        if (-not [Console]::IsInputRedirected) {
            if ([Console]::KeyAvailable) {
                $key = [Console]::ReadKey($true).Key
                if ($key -eq [ConsoleKey]::Y) { Write-Host 'y'; return 'y' }
                if ($key -eq [ConsoleKey]::N) { Write-Host 'n'; return 'n' }
                if ($key -eq [ConsoleKey]::Enter) { Write-Host ''; return 'n' }
                continue  # any other key is noise, not an answer: let the deadline keep running.
            }
        }
        if ($TimeoutSec -gt 0 -and $clock.Elapsed.TotalSeconds -ge $TimeoutSec) { return '' }
        Start-Sleep -Milliseconds 100
    }
}

Write-Host ''
Write-Host '  ace-simulator.MACHINE-SLOWDOWN.ps1  -  DANGER, opt-in, read-only' -ForegroundColor DarkYellow
if ($autoDetected) {
    Write-Host "  drives:  $($Root.Count) of $($mounted.Count) mounted volume(s), picked automatically" -ForegroundColor Yellow
} else {
    Write-Host "  drives:  $($Root.Count) named by -Root (this machine has $($mounted.Count) mounted)" -ForegroundColor Yellow
}
$capacityBytes = [int64]0
$freeBytes = [int64]0
foreach ($drive in $mounted) {
    $size = 0
    $avail = 0
    try { $size = $drive.TotalSize; $avail = $drive.TotalFreeSpace } catch { }
    $capacityBytes += $size
    $freeBytes += $avail
    $state = if ($Root -contains $drive.Name) { 'SCAN' } else { 'skip' }
    Write-Host ("           [{0}] {1,-5} {2,-10} {3,9:N0} GiB ({4,8:N0} free)" -f `
        $state, $drive.Name, $drive.DriveType, ($size / 1GB), ($avail / 1GB)) -ForegroundColor DarkGray
}
Write-Host ("           {0:N2} TiB of volumes mounted, {1:N2} TiB of it free" -f `
    ($capacityBytes / 1TB), ($freeBytes / 1TB)) -ForegroundColor DarkGray
Write-Host "  roots:   $($Root -join ', ')"
Write-Host "  types:   $($Extension -join ', ')"
Write-Host "  hash fs: $Degree workers (one per logical processor)"
Write-Host "  hub:     $HubEndpoint (reachability only - this script sends it no requests)"

if ($Root.Count -eq 0) {
    Write-Host ''
    Write-Host '  Nothing to scan: no mounted volume matched, and -Root named none. Exiting clean.' -ForegroundColor Yellow
    exit 0
}

if ($DryRun) {
    Write-Host '  -DryRun: nothing will be hashed and nothing will be probed.' -ForegroundColor DarkGray
    exit 0
}
if (-not $IAccept) {
    Write-Host ''
    Write-Host '  Refusing to start: this pins every core and reads every disk until it is done.' -ForegroundColor Red
    Write-Host '  Re-run with -IAccept when you really mean it (or -DryRun to see the plan).' -ForegroundColor Red
    exit 2
}

Write-Host ''
Write-Host '  WARNING - this is the machine-slowdown one.' -ForegroundColor Red
Write-Host "  It will recursively read every file under $($Root.Count) volume(s): $($Root -join ', ')" -ForegroundColor Red
Write-Host "  ...and hash them on all $Degree logical processors at once." -ForegroundColor Red
Write-Host '  Disk and CPU go to 100% for as long as it takes. Nothing is written, but nothing is' -ForegroundColor Red
Write-Host '  fast either: your other work - a running engine, a training job, a desktop - stalls.' -ForegroundColor Red
Write-Host '  -Root / -LimitFiles shrink that; Ctrl+C works; the answer here is not remembered.' -ForegroundColor DarkGray

$answer = Read-Confirmation -Question '  Continue? [y/N] ' -TimeoutSec $PromptTimeoutSec
if ($answer -ne 'y') {
    Write-Host ''
    if ($answer -eq '') {
        if ($PromptTimeoutSec -gt 0) {
            Write-Host "  No answer in $PromptTimeoutSec s - default is No, so nothing started." -ForegroundColor Yellow
        } else {
            Write-Host '  No answer - default is No, so nothing started.' -ForegroundColor Yellow
        }
    } else {
        Write-Host "  Answered '$answer' - only 'y' starts it, so nothing started." -ForegroundColor Yellow
    }
    Write-Host '  (The answer is never saved: next start asks this again, defaulting to No.)' -ForegroundColor DarkGray
    exit 2
}

# ── phase 1: find the models ─────────────────────────────────────────────────
Write-Host "`n[1/3] scanning the disk..." -ForegroundColor Cyan
$found = [Collections.Generic.List[object]]::new()
foreach ($r in $Root) {
    try {
        Get-ChildItem -LiteralPath $r -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
            if ($_.Attributes -band [IO.FileAttributes]::ReparsePoint) { return }
            if ($Extension -contains $_.Extension.ToLower()) { $found.Add($_) }
        }
    } catch {
        Write-Warning "skipped $r : $($_.Exception.Message)"
    }
}
if ($LimitFiles -gt 0 -and $found.Count -gt $LimitFiles) {
    $found = [Collections.Generic.List[object]]($found | Select-Object -First $LimitFiles)
}
$totalBytes = [int64]0
foreach ($f in $found) { $totalBytes += $f.Length }
Write-Host ("      {0} files, {1:N2} GiB" -f $found.Count, ($totalBytes / 1GB))
if ($found.Count -eq 0) { Write-Host '      nothing to hash; exiting clean.'; exit 0 }

# ── phase 2: the ACE feeling, i.e. hash all of it, on everything ─────────────
# A RunspacePool, not [Threading.Tasks.Parallel]::ForEach: under Windows PowerShell 5.1 the
# scriptblock-to-Action[] conversion behind that call throws inside the body (AggregateException),
# while the pool is the idiomatic in-process parallel here and stays PS 5.1-compatible on purpose
# (ForEach-Object -Parallel would need PowerShell 7).
Write-Host "`n[2/3] hashing in parallel (this is the part that stutters)..." -ForegroundColor Cyan
$startedAt = Get-Date
$hashOne = {
    param($path, $size)
    $record = [pscustomobject]@{ Path = $path; Size = $size; Sha256 = ''; Error = '' }
    try {
        $stream = [IO.File]::OpenRead($path)
        try {
            $digest = [Security.Cryptography.SHA256]::Create().ComputeHash($stream)
            $record.Sha256 = [BitConverter]::ToString($digest).Replace('-', '').ToLowerInvariant()
        } finally { $stream.Dispose() }
    } catch {
        $record.Error = $_.Exception.Message
    }
    $record
}
$pool = [RunspaceFactory]::CreateRunspacePool(1, $Degree)
$pool.Open()
$inFlight = foreach ($file in $found) {
    $powershell = [PowerShell]::Create().AddScript($hashOne).AddArgument($file.FullName).AddArgument($file.Length)
    $powershell.RunspacePool = $pool
    [pscustomobject]@{ Shell = $powershell; Handle = $powershell.BeginInvoke() }
}
$records = @(foreach ($job in $inFlight) {
    $result = $job.Shell.EndInvoke($job.Handle)
    $job.Shell.Dispose()
    $result
})
$pool.Close()
$pool.Dispose()
$elapsed = ((Get-Date) - $startedAt).TotalSeconds
$hashed = @($records | Where-Object { $_.Sha256 })
$gib = (($hashed | Measure-Object Size -Sum).Sum) / 1GB
Write-Host ("      {0}/{1} files, {2:N2} GiB in {3:N1}s = {4:N2} GiB/s" -f `
    $hashed.Count, $records.Count, $gib, $elapsed, ($gib / [Math]::Max($elapsed, 0.001)))
foreach ($bad in @($records | Where-Object { $_.Error })) {
    Write-Warning "unreadable: $($bad.Path) ($($bad.Error))"
}

# ── phase 3: is the hub there? ask it nothing, then deliver the verdict ──────
Write-Host "`n[3/3] probing $HubEndpoint for reachability..." -ForegroundColor Cyan
# Reachability by SYN and nothing more. The previous revision of this script called
# `/api/models/<repo>?blobs=true` for every HF cache directory it found, which - however polite -
# is real inbound traffic against a public service generated by a joke. This version sends zero
# requests: it resolves nothing itself, opens one TCP socket to host:port, reads nothing, closes.
# If that comes up the hub exists; if it does not, the hub is offline as far as we can tell.
$hostName = ([Uri]$HubEndpoint).Host
$port = ([Uri]$HubEndpoint).Port
if ($port -le 0 -or $port -gt 65535) { $port = 443 }
$offlineReason = ''
if ($env:INFR_HF_OFFLINE -and $env:INFR_HF_OFFLINE -ne '0') {
    $offlineReason = 'INFR_HF_OFFLINE is set'
} elseif (-not $hostName) {
    $offlineReason = "$HubEndpoint has no host part"
} else {
    $socket = [Net.Sockets.TcpClient]::new()
    try {
        $connected = $socket.ConnectAsync($hostName, $port).Wait($ConnectTimeoutSec * 1000)
        if (-not $connected -or -not $socket.Connected) {
            $offlineReason = "no TCP answer from ${hostName}:$port in $ConnectTimeoutSec s"
        }
    } catch {
        $offlineReason = $_.Exception.Message
    } finally {
        $socket.Close()
    }
}
if ($offlineReason) {
    Write-Host "      unreachable: $offlineReason" -ForegroundColor Yellow
    if (-not $offlineAllowed) {
        Write-Host ''
        Write-Host '  Not one hash can be confirmed, so this is where it dies. That is the feature.' -ForegroundColor Red
        Write-Host '  (Escape hatch: -TolerateOffline, or INFR_HF_OFFLINE=1 to stay and enjoy the pause.)' -ForegroundColor DarkGray
        exit 1
    }
    Write-Host '      staying on the offline branch, as asked - the pause still applies.' -ForegroundColor DarkGray
} else {
    Write-Host "      ${hostName}:$port is connectable (that is all this script learns, and all it asks)" -ForegroundColor DarkGray
}

# The anti-cheat beat: the "verification" that no longer happens takes exactly as long as one that
# does, because the wait - not the answer - is the experience being reproduced.
if ($HangSec -gt 0) {
    Write-Host "      contacting the hub for the authoritative verdict... (${HangSec}s)" -ForegroundColor DarkGray
    Start-Sleep -Seconds $HangSec
}

# And the verdict can only ever be one thing. There is no record to compare against, because
# comparing would mean asking, and asking is exactly what phase 3 refuses to do.
$verdict = if ($offlineReason) { 'NOT CHECKED (offline)' } else { 'CHECK FAILED' }
foreach ($record in $hashed) {
    Write-Host ("      {0,-12} {1}" -f $verdict, (Split-Path -Leaf $record.Path)) -ForegroundColor Yellow
}

Write-Host ("`ndone: verdict={0} checked=0 files={1} gib={2:N2} hash-seconds={3:N1}" -f `
    $verdict, $hashed.Count, $gib, $elapsed) -ForegroundColor Green
Write-Host '  Nothing was sent to HuggingFace, so nothing was confirmed. Same feeling, no traffic.' -ForegroundColor DarkGray
if (-not $offlineReason) {
    Write-Host '  Reachable and still failed: the hub is innocent, the script is the anti-cheat.' -ForegroundColor DarkGray
    exit 3
}
exit 0
