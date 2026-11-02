<#
.SYNOPSIS
    DANGER - DELIBERATELY SLOWS THE MACHINE DOWN. Opt-in only. Nothing in MoE4All calls this.

.DESCRIPTION
    An affectionate reproduction of the Tencent ACE anti-cheat feeling: it walks the whole disk
    looking for model files, SHA-256s every one of them across all logical processors at once,
    then asks HuggingFace whether those hashes are the official ones - and if HuggingFace cannot
    be reached, it gives up loudly with a non-zero exit code instead of carrying on offline.

    This is a joke, and jokes still cost the person running them real seconds. It is therefore
    triple-fenced:

      * NOT wired into Start-INFR-Wizard.cmd, infr-wizard.ps1, the engine, the release scripts or
        CI. Nothing runs it automatically. If you want it in a startup path, do not file a bug
        about it - that is the one version of this that would hurt other people.
      * it refuses to do anything at all unless you pass -IAccept.
      * -DryRun shows the whole plan (which drives, which extensions, how many files) without
        hashing or dialling out, and -LimitFiles / -Root narrow the blast radius further.

    Read-only by construction: it opens files for reading, and never writes, moves or deletes a
    single byte outside its own console output.

.PARAMETER Root
    Where to look. Default: every fixed local drive. Pass one or more paths to stay in a corner.

.PARAMETER Extension
    File extensions treated as "models the engine can load". Default: .gguf and .mmproj.

.PARAMETER Degree
    Worker threads for the hash phase. Default: one per logical processor, i.e. the good part.

.PARAMETER HubEndpoint
    Where to verify hashes. Honours INFR_HF_ENDPOINT, same knob the engine uses, so a mirror works.

.PARAMETER TolerateOffline
    Warn instead of exiting non-zero when the endpoint cannot be reached. The unflavoured script
    exits 1 on purpose; that is the punchline, not a bug.

.PARAMETER DryRun
    Enumerate, report the plan, hash nothing, call nothing.

.PARAMETER LimitFiles
    Cap the number of files hashed (0 = no cap). Handy while having a laugh.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\ace-simulator.MACHINE-SLOWDOWN.ps1 -DryRun

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\ace-simulator.MACHINE-SLOWDOWN.ps1 -IAccept -Root F:\models

.NOTES
    Windows PowerShell 5.1 compatible on purpose (no ForEach-Object -Parallel), matching the other
    scripts in this folder.
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
    [int]$LimitFiles = 0
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
# PS 5.1 otherwise writes native output through the active OEM codepage.
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)

if (-not $HubEndpoint) {
    $HubEndpoint = if ($env:INFR_HF_ENDPOINT) { $env:INFR_HF_ENDPOINT } else { 'https://huggingface.co' }
}
if ($Degree -le 0) { $Degree = [Environment]::ProcessorCount }
if (-not $Root -or $Root.Count -eq 0) {
    $Root = Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Free -ne $null -and $_.Name.Length -eq 1 } |
        ForEach-Object { "$($_.Root)" }
}

Write-Host ''
Write-Host '  ace-simulator.MACHINE-SLOWDOWN.ps1  -  DANGER, opt-in, read-only' -ForegroundColor DarkYellow
Write-Host "  drives:  $($Root -join ', ')"
Write-Host "  types:   $($Extension -join ', ')"
Write-Host "  hash fs: $Degree workers (one per logical processor)"
Write-Host "  hub:     $HubEndpoint"

if ($DryRun) {
    Write-Host '  -DryRun: nothing will be hashed and no request will be sent.' -ForegroundColor DarkGray
    exit 0
}
if (-not $IAccept) {
    Write-Host ''
    Write-Host '  Refusing to start: this pins every core and reads every disk until it is done.' -ForegroundColor Red
    Write-Host '  Re-run with -IAccept when you really mean it (or -DryRun to see the plan).' -ForegroundColor Red
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

# ── phase 3: ask HuggingFace, and fall over if it does not answer ───────────
Write-Host "`n[3/3] verifying those hashes against $HubEndpoint ..." -ForegroundColor Cyan
$headers = @{ Accept = 'application/json'; 'User-Agent' = 'moe4all-ace-simulator/1.0' }
# Reachability first, and reachability ONLY. An HTTP status (401 on an anonymous search, 404 on an
# unknown repo, 429 on a burst) means the host ANSWERED, which is not the failure this script is
# written to die on - treating a status code as "offline" would make the punchline fire every single
# time and turn the verification phase into a lie. A DNS failure, a refused connection or a timeout
# is the "连不上" case, and that one is fatal here.
$offline = $null
try {
    $probe = Invoke-WebRequest -Uri "$HubEndpoint" -Headers $headers -TimeoutSec 20 -UseBasicParsing
    Write-Host "      reachable (HTTP $($probe.StatusCode))" -ForegroundColor DarkGray
} catch {
    if ($_.Exception.Response) {
        Write-Host "      reachable (HTTP $([int]$_.Exception.Response.StatusCode))" -ForegroundColor DarkGray
    } else {
        $offline = $_.Exception.Message
    }
}
if ($offline) {
    Write-Host ''
    Write-Host "  HuggingFace is not reachable: $offline" -ForegroundColor Red
    if ($TolerateOffline) {
        Write-Host '  -TolerateOffline given, continuing without verification.' -ForegroundColor DarkGray
    } else {
        Write-Host '  Not one hash can be confirmed, so this is where it dies. That is the feature.' -ForegroundColor Red
        Write-Host '  (Escape hatch: -TolerateOffline, or point INFR_HF_ENDPOINT at a mirror.)' -ForegroundColor DarkGray
        exit 1
    }
}

# Local layout tells us which repositories to ask about: the HF cache directory naming
# `models--<owner>--<name>` is exactly `<owner>/<name>`, which is what the model-info endpoint
# takes. Anything else has no official record to check against, and is reported as such rather
# than guessed at - a wrong "MISMATCH" on a re-packaged quant would be worse than no answer.
$match = 0; $mismatch = 0; $noRecord = 0
foreach ($directory in ($hashed | ForEach-Object { Split-Path -Parent $_.Path } | Select-Object -Unique)) {
    $leaf = Split-Path -Leaf $directory
    $recordsHere = @($hashed | Where-Object { (Split-Path -Parent $_.Path) -eq $directory })
    if ($leaf -notmatch '^models--([^-]+)--(.+)$') {
        $noRecord += $recordsHere.Count
        Write-Host "      NO RECORD  $leaf (not an HF cache directory, so no owner/name to ask about)" -ForegroundColor DarkGray
        continue
    }
    $repo = "$($Matches[1])/$($Matches[2])"
    try {
        $detail = Invoke-RestMethod -Uri "$HubEndpoint/api/models/$([Uri]::EscapeDataString($repo))?blobs=true" `
            -Headers $headers -TimeoutSec 30
        $oids = @{}
        foreach ($sibling in $detail.siblings) {
            if ($sibling.sha) { $oids[$sibling.sha.ToLowerInvariant()] = $sibling.rfilename }
        }
        foreach ($record in $recordsHere) {
            if ($oids.ContainsKey($record.Sha256)) {
                $match++
                Write-Host ("      MATCH     {0}  ({1})" -f (Split-Path -Leaf $record.Path), $oids[$record.Sha256]) -ForegroundColor DarkGreen
            } else {
                $mismatch++
                Write-Host ("      MISMATCH  {0}" -f (Split-Path -Leaf $record.Path)) -ForegroundColor Yellow
            }
        }
        Write-Host "      checked against $repo" -ForegroundColor DarkGray
    } catch {
        $noRecord += $recordsHere.Count
        $why = if ($_.Exception.Response) { "HTTP $([int]$_.Exception.Response.StatusCode)" } else { $_.Exception.Message }
        Write-Warning "no record for ${repo}: $why"
    }
}

Write-Host ("`ndone: match={0} mismatch={1} no-record={2} unreadable={3} files={4} gib={5:N2} seconds={6:N1}" -f `
    $match, $mismatch, $noRecord, @($records | Where-Object { $_.Error }).Count, $hashed.Count, `
    (($hashed | Measure-Object Size -Sum).Sum / 1GB), $elapsed) -ForegroundColor Green
