[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [int]$TargetPid,

    [Parameter(Mandatory = $true)]
    [long]$TargetStartTicks,

    [Parameter(Mandatory = $true)]
    [string]$Profile,

    [Parameter(Mandatory = $true)]
    [string]$Samples,

    [Parameter(Mandatory = $true)]
    [string]$Summary,

    [Parameter(Mandatory = $true)]
    [string]$Violation,

    [Parameter(Mandatory = $true)]
    [string]$StopFile,

    [Parameter(Mandatory = $true)]
    [string]$ReadyFile,

    [int]$IntervalMs = 500,
    [int]$SlowIntervalMs = 2000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function ConvertTo-ByteCount {
    param([Parameter(Mandatory = $true)]$Value)

    if ($Value -is [byte] -or $Value -is [int16] -or $Value -is [int32] -or
        $Value -is [int64] -or $Value -is [uint16] -or $Value -is [uint32] -or
        $Value -is [uint64]) {
        return [uint64]$Value
    }
    $text = ([string]$Value).Trim()
    if ($text -notmatch '^(?<number>[0-9]+(?:\.[0-9]+)?)\s*(?<unit>[kmgt]?)(?:i?b)?$') {
        throw "Invalid byte count: $text"
    }
    $number = [double]::Parse(
        $Matches.number,
        [Globalization.CultureInfo]::InvariantCulture
    )
    $power = switch ($Matches.unit.ToLowerInvariant()) {
        '' { 0 }
        'k' { 1 }
        'm' { 2 }
        'g' { 3 }
        't' { 4 }
    }
    return [uint64]($number * [math]::Pow(1024, $power))
}

function Get-TargetProcess {
    try {
        $process = Get-Process -Id $TargetPid -ErrorAction Stop
        if ($process.StartTime.ToUniversalTime().Ticks -ne $TargetStartTicks) {
            return $null
        }
        return $process
    }
    catch {
        return $null
    }
}

function Get-GpuMemory {
    $pattern = "(^|_)pid_$([regex]::Escape([string]$TargetPid))(_|$)"
    $cimParams = @{
        ClassName = 'Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory'
        ErrorAction = 'Stop'
    }
    $rows = Get-CimInstance @cimParams | Where-Object { $_.Name -match $pattern }
    if (-not $rows) {
        return $null
    }
    $dedicated = [uint64]0
    $shared = [uint64]0
    foreach ($row in $rows) {
        $dedicated += [uint64]$row.DedicatedUsage
        $shared += [uint64]$row.SharedUsage
    }
    return [pscustomobject]@{
        Dedicated = $dedicated
        Shared = $shared
    }
}

$profileData = Get-Content -LiteralPath $Profile -Raw | ConvertFrom-Json
$vramLimit = (ConvertTo-ByteCount $profileData.vram_total) -
    (ConvertTo-ByteCount $profileData.vram_used)
$ramLimit = (ConvertTo-ByteCount $profileData.ram_total) -
    (ConvertTo-ByteCount $profileData.ram_used)

$utf8 = [Text.UTF8Encoding]::new($false)
$sampleStream = [IO.FileStream]::new(
    $Samples,
    [IO.FileMode]::Create,
    [IO.FileAccess]::Write,
    [IO.FileShare]::Read
)
$writer = [IO.StreamWriter]::new($sampleStream, $utf8)
$peakWorkingSet = [uint64]0
$peakPrivate = [uint64]0
$peakDedicated = [uint64]0
$peakShared = [uint64]0
$gpuCounterSeen = $false
$sampleCount = 0
$lastSlow = [DateTime]::MinValue
$gpu = $null
$systemAvailable = $null
$pageFaultsPerSec = $null
$violationData = $null

try {
    [IO.File]::WriteAllText($ReadyFile, "ready`n", $utf8)
    while (-not (Test-Path -LiteralPath $StopFile)) {
        $process = Get-TargetProcess
        if ($null -eq $process) {
            break
        }
        $process.Refresh()
        $now = [DateTime]::UtcNow
        if (($now - $lastSlow).TotalMilliseconds -ge $SlowIntervalMs) {
            try {
                $gpu = Get-GpuMemory
                if ($null -ne $gpu) {
                    $gpuCounterSeen = $true
                }
            }
            catch {
                $gpu = $null
            }
            try {
                $os = Get-CimInstance -ClassName 'Win32_OperatingSystem' -ErrorAction Stop
                $systemAvailable = [uint64]$os.FreePhysicalMemory * 1024
            }
            catch {
                $systemAvailable = $null
            }
            try {
                $perfParams = @{
                    ClassName = 'Win32_PerfFormattedData_PerfProc_Process'
                    Filter = "IDProcess = $TargetPid"
                    ErrorAction = 'Stop'
                }
                $perf = Get-CimInstance @perfParams | Select-Object -First 1
                $pageFaultsPerSec = if ($null -eq $perf) {
                    $null
                }
                else {
                    [uint64]$perf.PageFaultsPersec
                }
            }
            catch {
                $pageFaultsPerSec = $null
            }
            $lastSlow = $now
        }

        $workingSet = [uint64]$process.WorkingSet64
        $privateBytes = [uint64]$process.PrivateMemorySize64
        $dedicated = if ($null -eq $gpu) { $null } else { [uint64]$gpu.Dedicated }
        $shared = if ($null -eq $gpu) { $null } else { [uint64]$gpu.Shared }
        $peakWorkingSet = [math]::Max($peakWorkingSet, $workingSet)
        $peakPrivate = [math]::Max($peakPrivate, $privateBytes)
        if ($null -ne $dedicated) {
            $peakDedicated = [math]::Max($peakDedicated, $dedicated)
        }
        if ($null -ne $shared) {
            $peakShared = [math]::Max($peakShared, $shared)
        }

        $sample = [ordered]@{
            timestamp_utc = $now.ToString('o')
            working_set_bytes = $workingSet
            private_bytes = $privateBytes
            gpu_dedicated_bytes = $dedicated
            gpu_shared_bytes = $shared
            system_available_bytes = $systemAvailable
            page_faults_per_sec = $pageFaultsPerSec
        }
        $writer.WriteLine(($sample | ConvertTo-Json -Compress))
        $writer.Flush()
        $sampleCount++

        $reason = $null
        if ($workingSet -gt $ramLimit) {
            $reason = "working set exceeded simulated available RAM"
        }
        elseif ($null -ne $dedicated -and $dedicated -gt $vramLimit) {
            $reason = "dedicated GPU memory exceeded simulated available VRAM"
        }
        if ($null -ne $reason) {
            $violationData = [ordered]@{
                reason = $reason
                timestamp_utc = $now.ToString('o')
                working_set_bytes = $workingSet
                ram_limit_bytes = $ramLimit
                gpu_dedicated_bytes = $dedicated
                vram_limit_bytes = $vramLimit
            }
            [IO.File]::WriteAllText(
                $Violation,
                ($violationData | ConvertTo-Json -Depth 5),
                $utf8
            )
            $verified = Get-TargetProcess
            if ($null -ne $verified) {
                Stop-Process -Id $TargetPid -Force -ErrorAction SilentlyContinue
            }
            break
        }
        Start-Sleep -Milliseconds $IntervalMs
    }
}
finally {
    $writer.Dispose()
    $summaryData = [ordered]@{
        samples = $sampleCount
        gpu_counter_available = $gpuCounterSeen
        peak_working_set_bytes = $peakWorkingSet
        peak_private_bytes = $peakPrivate
        peak_gpu_dedicated_bytes = $peakDedicated
        peak_gpu_shared_bytes = $peakShared
        ram_limit_bytes = $ramLimit
        vram_limit_bytes = $vramLimit
        violated = $null -ne $violationData
    }
    [IO.File]::WriteAllText(
        $Summary,
        ($summaryData | ConvertTo-Json -Depth 5),
        $utf8
    )
}
