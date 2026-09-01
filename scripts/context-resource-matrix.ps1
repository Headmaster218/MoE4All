[CmdletBinding()]
param(
    [string]$Manifest = 'tests/context-resource/matrix.local.json',
    [string[]]$CaseId,
    [switch]$List,
    [switch]$Force,
    [switch]$SkipCliSmoke
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$script:Utf8 = [Text.UTF8Encoding]::new($false)

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string]$Path)

    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $script:RepoRoot $Path))
}

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

function Format-Gib {
    param([Parameter(Mandatory = $true)][uint64]$Bytes)
    return ($Bytes / 1GB).ToString('0.00', [Globalization.CultureInfo]::InvariantCulture)
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )
    $parent = Split-Path -Parent $Path
    if ($parent) {
        [IO.Directory]::CreateDirectory($parent) | Out-Null
    }
    $json = $Value | ConvertTo-Json -Depth 20
    [IO.File]::WriteAllText($Path, $json, $script:Utf8)
}

function ConvertTo-WindowsCommandLineArgument {
    param([AllowEmptyString()][Parameter(Mandatory = $true)][string]$Argument)

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }
    $builder = [Text.StringBuilder]::new()
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * ($backslashes * 2 + 1)))
            [void]$builder.Append('"')
        }
        else {
            if ($backslashes -gt 0) {
                [void]$builder.Append(('\' * $backslashes))
            }
            [void]$builder.Append($character)
        }
        $backslashes = 0
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append(('\' * ($backslashes * 2)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function New-CleanStartInfo {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $FilePath
    $info.WorkingDirectory = $script:RepoRoot
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $quoted = foreach ($argument in $Arguments) {
        ConvertTo-WindowsCommandLineArgument -Argument $argument
    }
    $info.Arguments = $quoted -join ' '
    foreach ($key in @($info.EnvironmentVariables.Keys)) {
        if ($key.StartsWith('INFR_', [StringComparison]::OrdinalIgnoreCase)) {
            $info.EnvironmentVariables.Remove($key)
        }
    }
    $info.EnvironmentVariables['RUST_LOG'] = 'info'
    return $info
}

function Start-LoggedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$StdoutPath,
        [Parameter(Mandatory = $true)][string]$StderrPath,
        [switch]$RedirectInput
    )
    $info = New-CleanStartInfo -FilePath $FilePath -Arguments $Arguments
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.RedirectStandardInput = $RedirectInput.IsPresent
    $info.StandardOutputEncoding = $script:Utf8
    $info.StandardErrorEncoding = $script:Utf8
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $info
    if (-not $process.Start()) {
        throw "Failed to start $FilePath"
    }
    $stdout = [IO.FileStream]::new(
        $StdoutPath,
        [IO.FileMode]::Create,
        [IO.FileAccess]::Write,
        [IO.FileShare]::ReadWrite
    )
    $stderr = [IO.FileStream]::new(
        $StderrPath,
        [IO.FileMode]::Create,
        [IO.FileAccess]::Write,
        [IO.FileShare]::ReadWrite
    )
    return [pscustomobject]@{
        Process = $process
        Stdout = $stdout
        Stderr = $stderr
        StdoutTask = $process.StandardOutput.BaseStream.CopyToAsync($stdout)
        StderrTask = $process.StandardError.BaseStream.CopyToAsync($stderr)
    }
}

function Complete-LoggedProcess {
    param([Parameter(Mandatory = $true)]$Handle)

    $Handle.Process.WaitForExit()
    $null = $Handle.StdoutTask.GetAwaiter().GetResult()
    $null = $Handle.StderrTask.GetAwaiter().GetResult()
    $Handle.Stdout.Dispose()
    $Handle.Stderr.Dispose()
    return $Handle.Process.ExitCode
}

function Stop-ServerProcess {
    param(
        [Parameter(Mandatory = $true)]$Handle,
        [Parameter(Mandatory = $true)][string]$ShutdownFile,
        [int]$TimeoutSeconds = 120
    )
    if (-not $Handle.Process.HasExited) {
        [IO.File]::WriteAllText($ShutdownFile, "stop`n", $script:Utf8)
        if (-not $Handle.Process.WaitForExit($TimeoutSeconds * 1000)) {
            $Handle.Process.Kill()
        }
    }
    return Complete-LoggedProcess -Handle $Handle
}

function Invoke-CapturedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $info = New-CleanStartInfo -FilePath $FilePath -Arguments $Arguments
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.StandardOutputEncoding = $script:Utf8
    $info.StandardErrorEncoding = $script:Utf8
    $process = [Diagnostics.Process]::Start($info)
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    if ($process.ExitCode -ne 0) {
        throw "$FilePath failed with exit code $($process.ExitCode): $stderr"
    }
    return [pscustomobject]@{
        Stdout = $stdout
        Stderr = $stderr
    }
}

function Get-FreeTcpPort {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $listener.Start()
    try {
        return ([Net.IPEndPoint]$listener.LocalEndpoint).Port
    }
    finally {
        $listener.Stop()
    }
}

function Start-ResourceMonitor {
    param(
        [Parameter(Mandatory = $true)][Diagnostics.Process]$Target,
        [Parameter(Mandatory = $true)][string]$ProfilePath,
        [Parameter(Mandatory = $true)][string]$CaseDir
    )
    $monitorScript = Join-Path $PSScriptRoot 'context-resource-monitor.ps1'
    $pwsh = (Get-Process -Id $PID).Path
    $paths = [ordered]@{
        Samples = Join-Path $CaseDir 'resource-samples.jsonl'
        Summary = Join-Path $CaseDir 'resource-summary.json'
        Violation = Join-Path $CaseDir 'resource-violation.json'
        Stop = Join-Path $CaseDir 'monitor.stop'
        Ready = Join-Path $CaseDir 'monitor.ready'
    }
    foreach ($path in $paths.Values) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }
    $arguments = @(
        '-NoLogo'
        '-NoProfile'
        '-NonInteractive'
        '-File'
        $monitorScript
        '-TargetPid'
        [string]$Target.Id
        '-TargetStartTicks'
        [string]$Target.StartTime.ToUniversalTime().Ticks
        '-Profile'
        $ProfilePath
        '-Samples'
        $paths.Samples
        '-Summary'
        $paths.Summary
        '-Violation'
        $paths.Violation
        '-StopFile'
        $paths.Stop
        '-ReadyFile'
        $paths.Ready
    )
    $info = New-CleanStartInfo -FilePath $pwsh -Arguments $arguments
    $monitor = [Diagnostics.Process]::Start($info)
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while (-not (Test-Path -LiteralPath $paths.Ready)) {
        if ($monitor.HasExited) {
            throw "Resource monitor exited before becoming ready"
        }
        if ([DateTime]::UtcNow -ge $deadline) {
            $monitor.Kill()
            throw "Resource monitor did not become ready within 30 seconds"
        }
        Start-Sleep -Milliseconds 100
    }
    return [pscustomobject]@{
        Process = $monitor
        Paths = [pscustomobject]$paths
    }
}

function Stop-ResourceMonitor {
    param([Parameter(Mandatory = $true)]$Monitor)
    if (-not $Monitor.Process.HasExited) {
        [IO.File]::WriteAllText($Monitor.Paths.Stop, "stop`n", $script:Utf8)
        if (-not $Monitor.Process.WaitForExit(15000)) {
            $Monitor.Process.Kill()
        }
    }
    $Monitor.Process.WaitForExit()
    return $Monitor.Process.ExitCode
}

function Wait-ServerReady {
    param(
        [Parameter(Mandatory = $true)][string]$BaseUri,
        [Parameter(Mandatory = $true)][Diagnostics.Process]$Process,
        [Parameter(Mandatory = $true)][string]$ViolationPath,
        [int]$TimeoutSeconds = 1800
    )
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        if ($Process.HasExited) {
            throw "infr serve exited during startup with code $($Process.ExitCode)"
        }
        if (Test-Path -LiteralPath $ViolationPath) {
            throw "resource limit was exceeded during startup"
        }
        try {
            $response = Invoke-WebRequest -Uri "$BaseUri/health" -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                return
            }
        }
        catch {
            Start-Sleep -Milliseconds 500
        }
    }
    throw "infr serve did not become healthy within $TimeoutSeconds seconds"
}

function Invoke-PromptPlanner {
    param(
        [Parameter(Mandatory = $true)][string]$InfrExe,
        [Parameter(Mandatory = $true)][string]$ModelPath,
        [Parameter(Mandatory = $true)]$Messages,
        [Parameter(Mandatory = $true)][int]$TargetTokens,
        [Parameter(Mandatory = $true)][string]$CaseDir,
        [Parameter(Mandatory = $true)][int]$Turn
    )
    $templatePath = Join-Path $CaseDir "turn-$Turn-template.json"
    $plannedPath = Join-Path $CaseDir "turn-$Turn-planned.json"
    Write-JsonFile -Path $templatePath -Value @($Messages)
    $arguments = @(
        '__test-plan-prompt'
        $ModelPath
        '--messages'
        $templatePath
        '--target'
        [string]$TargetTokens
        '--output'
        $plannedPath
        '--set'
        'sampling.no_think=true'
    )
    $captured = Invoke-CapturedProcess -FilePath $InfrExe -Arguments $arguments
    [IO.File]::WriteAllText(
        (Join-Path $CaseDir "turn-$Turn-planner.stderr.log"),
        $captured.Stderr,
        $script:Utf8
    )
    return Read-JsonFile -Path $plannedPath
}

function Invoke-ChatCompletion {
    param(
        [Parameter(Mandatory = $true)][string]$BaseUri,
        [Parameter(Mandatory = $true)][string]$ModelId,
        [Parameter(Mandatory = $true)]$Messages,
        [int]$MaxTokens = 32
    )
    $body = [ordered]@{
        model = $ModelId
        messages = @($Messages)
        stream = $false
        temperature = 0.0
        top_p = 1.0
        seed = 937162211
        max_tokens = $MaxTokens
    }
    $json = $body | ConvertTo-Json -Depth 20 -Compress
    $request = @{
        Uri = "$BaseUri/v1/chat/completions"
        Method = 'Post'
        ContentType = 'application/json; charset=utf-8'
        Body = $script:Utf8.GetBytes($json)
        TimeoutSec = 3600
    }
    return Invoke-RestMethod @request
}

function Get-KvGrowthEvents {
    param([Parameter(Mandatory = $true)][string]$LogPath)
    if (-not (Test-Path -LiteralPath $LogPath)) {
        return @()
    }
    $events = @()
    foreach ($line in Get-Content -LiteralPath $LogPath) {
        if ($line -match 'expanded dynamic KV cache.*requested_tokens=(\d+).*committed_tokens=(\d+).*segments=(\d+)') {
            $events += [pscustomobject]@{
                requested_tokens = [int64]$Matches[1]
                committed_tokens = [int64]$Matches[2]
                segments = [int]$Matches[3]
            }
        }
    }
    return $events
}

function Assert-KvGrowthPattern {
    param([Parameter(Mandatory = $true)]$Events)
    $decode32 = @($Events | Where-Object {
        $_.committed_tokens -eq 65536 -and
        $_.requested_tokens -gt 32768 -and
        $_.requested_tokens -le 32800
    })
    $prefill64 = @($Events | Where-Object {
        $_.committed_tokens -eq 98304 -and
        $_.requested_tokens -gt 65536
    })
    $decode96 = @($Events | Where-Object {
        $_.committed_tokens -eq 131072 -and
        $_.requested_tokens -gt 98304 -and
        $_.requested_tokens -le 98336
    })
    if ($decode32.Count -eq 0) {
        throw 'No decode-triggered KV growth was observed at 32K'
    }
    if ($prefill64.Count -eq 0) {
        throw 'No prefill-triggered KV growth was observed beyond 64K'
    }
    if ($decode96.Count -eq 0) {
        throw 'No decode-triggered KV growth was observed at 96K'
    }
}

function Test-LogForFatalError {
    param([Parameter(Mandatory = $true)][string]$LogPath)
    if (-not (Test-Path -LiteralPath $LogPath)) {
        return $null
    }
    $pattern = 'panicked at|VK_ERROR_DEVICE_LOST|device lost|out of memory|budget exceeded|allocation has failed'
    $match = Select-String -LiteralPath $LogPath -Pattern $pattern -CaseSensitive:$false |
        Select-Object -First 1
    if ($null -eq $match) {
        return $null
    }
    return $match.Line.Trim()
}

function New-ServerArguments {
    param(
        [Parameter(Mandatory = $true)]$ManifestData,
        [Parameter(Mandatory = $true)]$Case,
        [Parameter(Mandatory = $true)][string]$Address,
        [Parameter(Mandatory = $true)][string]$ShutdownFile
    )
    $arguments = [Collections.Generic.List[string]]::new()
    foreach ($value in @(
        'serve'
        $Case.ModelPath
        '--addr'
        $Address
        '--parallel'
        '1'
        '--ctx'
        [string]$ManifestData.context
        '--dev'
        [string]$ManifestData.device
        '--test-resource-profile'
        $Case.ProfilePath
    )) {
        $arguments.Add([string]$value)
    }
    $sets = [Collections.Generic.List[string]]::new()
    foreach ($set in @(
        'kv.type_k=q8_0'
        'kv.type_v=q8_0'
        'sampling.no_think=true'
        'sampling.ignore_eos=true'
        'serve.stats_interval_secs=0'
        "serve.shutdown_file=$ShutdownFile"
        'paging.stats=true'
        'kernels.vulkan.no_vram_guard=false'
    )) {
        $sets.Add($set)
    }
    if ($Case.Mode -eq 'manual') {
        $sets.Add("device.vram_budget=$($Case.ManualVramBudget)")
        $sets.Add("device.ram_budget=$($Case.ManualRamBudget)")
    }
    foreach ($set in $sets) {
        $arguments.Add('--set')
        $arguments.Add($set)
    }
    return $arguments.ToArray()
}

function Invoke-ApiCase {
    param(
        [Parameter(Mandatory = $true)]$ManifestData,
        [Parameter(Mandatory = $true)]$Case,
        [Parameter(Mandatory = $true)][string]$OutputRoot
    )
    $caseDir = Join-Path $OutputRoot $Case.Id
    [IO.Directory]::CreateDirectory($caseDir) | Out-Null
    $resultPath = Join-Path $caseDir 'result.json'
    if (-not $Force -and (Test-Path -LiteralPath $resultPath)) {
        $existing = Read-JsonFile -Path $resultPath
        if ($existing.status -eq 'pass') {
            Write-Host "SKIP $($Case.Id) (already passed)"
            return $existing
        }
    }

    Write-Host "RUN  $($Case.Id)"
    $stdoutPath = Join-Path $caseDir 'server.stdout.log'
    $stderrPath = Join-Path $caseDir 'server.stderr.log'
    $shutdownFile = Join-Path $caseDir 'server.shutdown'
    foreach ($path in @($shutdownFile, (Join-Path $caseDir 'monitor.stop'))) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
        }
    }
    $port = Get-FreeTcpPort
    $address = "127.0.0.1:$port"
    $baseUri = "http://$address"
    $serverArgumentParams = @{
        ManifestData = $ManifestData
        Case = $Case
        Address = $address
        ShutdownFile = $shutdownFile
    }
    $arguments = New-ServerArguments @serverArgumentParams
    $server = $null
    $monitor = $null
    $turnResults = @()
    $caseError = $null
    $exitCode = $null
    $monitorExitCode = $null
    try {
        $serverParams = @{
            FilePath = $Case.InfrExe
            Arguments = $arguments
            StdoutPath = $stdoutPath
            StderrPath = $stderrPath
        }
        $server = Start-LoggedProcess @serverParams
        $monitorParams = @{
            Target = $server.Process
            ProfilePath = $Case.ProfilePath
            CaseDir = $caseDir
        }
        $monitor = Start-ResourceMonitor @monitorParams
        $readyParams = @{
            BaseUri = $baseUri
            Process = $server.Process
            ViolationPath = $monitor.Paths.Violation
            TimeoutSeconds = [int]$ManifestData.startup_timeout_seconds
        }
        Wait-ServerReady @readyParams

        $models = Invoke-RestMethod -Uri "$baseUri/v1/models" -TimeoutSec 30
        $modelId = [string]$models.data[0].id
        $history = @(
            [pscustomobject]@{
                role = 'system'
                content = 'This is a deterministic memory-layout test. Ignore repeated context filler and follow the final instruction in each user turn.'
            }
        )
        $phases = @(
            [pscustomobject]@{
                Target = 32768
                Instruction = 'After the context payload, produce a numbered list of at least one hundred short integer identities. Do not stop early.'
            }
            [pscustomobject]@{
                Target = 65792
                Instruction = 'After the additional payload, produce a numbered list of at least one hundred short arithmetic facts. Do not stop early.'
            }
            [pscustomobject]@{
                Target = 98304
                Instruction = 'After the final payload, produce a numbered list of at least one hundred short algebraic equalities. Do not stop early.'
            }
        )
        $previousPrompt = 0
        for ($i = 0; $i -lt $phases.Count; $i++) {
            $turn = $i + 1
            $user = [pscustomobject]@{
                role = 'user'
                content = "Synthetic context payload:{{INFR_FILLER}}`n$($phases[$i].Instruction)"
            }
            $template = @($history) + @($user)
            $plannerParams = @{
                InfrExe = $Case.InfrExe
                ModelPath = $Case.ModelPath
                Messages = $template
                TargetTokens = $phases[$i].Target
                CaseDir = $caseDir
                Turn = $turn
            }
            $planned = Invoke-PromptPlanner @plannerParams
            $chatParams = @{
                BaseUri = $baseUri
                ModelId = $modelId
                Messages = $planned.messages
                MaxTokens = 32
            }
            $response = Invoke-ChatCompletion @chatParams
            $responseParams = @{
                Path = Join-Path $caseDir "turn-$turn-response.json"
                Value = $response
            }
            Write-JsonFile @responseParams

            $usage = $response.usage
            $choice = $response.choices[0]
            $plannedTokens = [int]$planned.prompt_tokens
            if ([int]$usage.prompt_tokens -ne $plannedTokens) {
                throw "turn $turn tokenizer mismatch: planned $plannedTokens, API reported $($usage.prompt_tokens)"
            }
            if ([int]$usage.completion_tokens -ne 32 -or $choice.finish_reason -ne 'length') {
                throw "turn $turn did not decode exactly 32 tokens"
            }
            if ([string]::IsNullOrWhiteSpace([string]$choice.message.content)) {
                throw "turn $turn returned no assistant content"
            }
            $cached = [int]$usage.prompt_tokens_details.cached_tokens
            if ($turn -gt 1 -and $cached -lt ($previousPrompt - 128)) {
                throw "turn $turn reused only $cached cached tokens; expected the prior conversation prefix"
            }
            $turnResults += [pscustomobject]@{
                turn = $turn
                target_prompt_tokens = [int]$phases[$i].Target
                prompt_tokens = [int]$usage.prompt_tokens
                cached_prompt_tokens = $cached
                completion_tokens = [int]$usage.completion_tokens
                finish_reason = [string]$choice.finish_reason
            }
            $history = @($planned.messages) + @(
                [pscustomobject]@{
                    role = 'assistant'
                    content = [string]$choice.message.content
                }
            )
            $previousPrompt = [int]$usage.prompt_tokens
            if (Test-Path -LiteralPath $monitor.Paths.Violation) {
                throw "resource limit was exceeded during turn $turn"
            }
        }
    }
    catch {
        $caseError = $_.Exception.Message
    }
    finally {
        if ($null -ne $server) {
            try {
                $exitCode = Stop-ServerProcess -Handle $server -ShutdownFile $shutdownFile
            }
            catch {
                if ($null -eq $caseError) {
                    $caseError = "server cleanup failed: $($_.Exception.Message)"
                }
            }
        }
        if ($null -ne $monitor) {
            $monitorExitCode = Stop-ResourceMonitor -Monitor $monitor
        }
    }

    $resource = if ($null -ne $monitor -and (Test-Path -LiteralPath $monitor.Paths.Summary)) {
        Read-JsonFile -Path $monitor.Paths.Summary
    }
    else {
        $null
    }
    if ($null -ne $monitorExitCode -and $monitorExitCode -ne 0 -and $null -eq $caseError) {
        $caseError = "resource monitor exited with code $monitorExitCode"
    }
    if ($null -ne $monitor -and $null -eq $resource -and $null -eq $caseError) {
        $caseError = 'resource monitor produced no summary'
    }
    if ($null -ne $monitor -and (Test-Path -LiteralPath $monitor.Paths.Violation)) {
        $violation = Read-JsonFile -Path $monitor.Paths.Violation
        $caseError = [string]$violation.reason
    }
    if ($null -ne $resource -and $ManifestData.require_gpu_counter -and
        -not [bool]$resource.gpu_counter_available -and $null -eq $caseError) {
        $caseError = 'Windows per-process GPU memory counter was unavailable'
    }
    # The shutdown-file watcher uses the same graceful SIGTERM semantics as a supervisor. infr
    # drains and releases Vulkan first, then deliberately exits with the conventional 128+15.
    if ($null -ne $exitCode -and $exitCode -notin @(0, 143) -and $null -eq $caseError) {
        $caseError = "infr serve exited with code $exitCode"
    }
    $fatal = Test-LogForFatalError -LogPath $stderrPath
    if ($null -ne $fatal -and $null -eq $caseError) {
        $caseError = $fatal
    }
    $growth = @(Get-KvGrowthEvents -LogPath $stderrPath)
    if ($null -eq $caseError) {
        try {
            Assert-KvGrowthPattern -Events $growth
            $last = $turnResults[-1]
            if (($last.prompt_tokens + $last.completion_tokens) -le 98304) {
                throw 'final context depth did not exceed 96K'
            }
        }
        catch {
            $caseError = $_.Exception.Message
        }
    }

    $result = [ordered]@{
        id = $Case.Id
        status = if ($null -eq $caseError) { 'pass' } else { 'fail' }
        error = $caseError
        model = $Case.ModelId
        model_path = $Case.ModelPath
        profile = $Case.ProfileId
        mode = $Case.Mode
        context = [string]$ManifestData.context
        kv_type = 'q8_0/q8_0'
        exit_code = $exitCode
        monitor_exit_code = $monitorExitCode
        turns = $turnResults
        kv_growth = $growth
        resources = $resource
    }
    Write-JsonFile -Path $resultPath -Value $result
    if ($result.status -eq 'pass') {
        Write-Host "PASS $($Case.Id)"
    }
    else {
        Write-Warning "FAIL $($Case.Id): $caseError"
    }
    return [pscustomobject]$result
}

function Invoke-CliSmoke {
    param(
        [Parameter(Mandatory = $true)]$ManifestData,
        [Parameter(Mandatory = $true)]$Case,
        [Parameter(Mandatory = $true)][string]$OutputRoot
    )
    $caseDir = Join-Path $OutputRoot $Case.Id
    [IO.Directory]::CreateDirectory($caseDir) | Out-Null
    $resultPath = Join-Path $caseDir 'result.json'
    if (-not $Force -and (Test-Path -LiteralPath $resultPath)) {
        $existing = Read-JsonFile -Path $resultPath
        if ($existing.status -eq 'pass') {
            Write-Host "SKIP $($Case.Id) (already passed)"
            return $existing
        }
    }

    Write-Host "RUN  $($Case.Id)"
    $stdoutPath = Join-Path $caseDir 'cli.stdout.log'
    $stderrPath = Join-Path $caseDir 'cli.stderr.log'
    $arguments = @(
        'run'
        $Case.ModelPath
        '--ctx'
        [string]$ManifestData.context
        '--dev'
        [string]$ManifestData.device
        '--max-new'
        '32'
        '--no-think'
        '--temp'
        '0'
        '--seed'
        '937162211'
        '--test-resource-profile'
        $Case.ProfilePath
        '--set'
        'kv.type_k=q8_0'
        '--set'
        'kv.type_v=q8_0'
        '--set'
        'kernels.vulkan.no_vram_guard=false'
    )
    $handle = $null
    $monitor = $null
    $caseError = $null
    $exitCode = $null
    $monitorExitCode = $null
    try {
        $processParams = @{
            FilePath = $Case.InfrExe
            Arguments = $arguments
            StdoutPath = $stdoutPath
            StderrPath = $stderrPath
            RedirectInput = $true
        }
        $handle = Start-LoggedProcess @processParams
        $monitorParams = @{
            Target = $handle.Process
            ProfilePath = $Case.ProfilePath
            CaseDir = $caseDir
        }
        $monitor = Start-ResourceMonitor @monitorParams
        foreach ($line in @(
            'Introduce yourself in two short sentences.'
            'Now summarize your previous answer in one sentence.'
            'Finally, list three prime numbers and explain why each is prime.'
            'quit'
        )) {
            $handle.Process.StandardInput.WriteLine($line)
        }
        $handle.Process.StandardInput.Close()
        $timeout = ([int]$ManifestData.startup_timeout_seconds + 900) * 1000
        if (-not $handle.Process.WaitForExit($timeout)) {
            $handle.Process.Kill()
            $caseError = 'CLI smoke timed out'
        }
        $exitCode = Complete-LoggedProcess -Handle $handle
    }
    catch {
        $caseError = $_.Exception.Message
        if ($null -ne $handle -and -not $handle.Process.HasExited) {
            $handle.Process.Kill()
            try {
                $exitCode = Complete-LoggedProcess -Handle $handle
            }
            catch {}
        }
    }
    finally {
        if ($null -ne $monitor) {
            $monitorExitCode = Stop-ResourceMonitor -Monitor $monitor
        }
    }
    $resource = if ($null -ne $monitor -and (Test-Path -LiteralPath $monitor.Paths.Summary)) {
        Read-JsonFile -Path $monitor.Paths.Summary
    }
    else {
        $null
    }
    if ($null -ne $monitorExitCode -and $monitorExitCode -ne 0 -and $null -eq $caseError) {
        $caseError = "resource monitor exited with code $monitorExitCode"
    }
    if ($null -ne $monitor -and $null -eq $resource -and $null -eq $caseError) {
        $caseError = 'resource monitor produced no summary'
    }
    if ($null -ne $monitor -and (Test-Path -LiteralPath $monitor.Paths.Violation)) {
        $violation = Read-JsonFile -Path $monitor.Paths.Violation
        $caseError = [string]$violation.reason
    }
    if ($null -ne $resource -and $ManifestData.require_gpu_counter -and
        -not [bool]$resource.gpu_counter_available -and $null -eq $caseError) {
        $caseError = 'Windows per-process GPU memory counter was unavailable'
    }
    if ($null -ne $exitCode -and $exitCode -ne 0 -and $null -eq $caseError) {
        $caseError = "infr run exited with code $exitCode"
    }
    $fatal = Test-LogForFatalError -LogPath $stderrPath
    if ($null -ne $fatal -and $null -eq $caseError) {
        $caseError = $fatal
    }
    $turnLines = 0
    foreach ($path in @($stdoutPath, $stderrPath)) {
        if (Test-Path -LiteralPath $path) {
            $turnLines += @(Select-String -LiteralPath $path -Pattern '\[prefill .*decode').Count
        }
    }
    if ($turnLines -lt 3 -and $null -eq $caseError) {
        $caseError = "CLI produced only $turnLines completed-turn timing lines"
    }
    $result = [ordered]@{
        id = $Case.Id
        status = if ($null -eq $caseError) { 'pass' } else { 'fail' }
        error = $caseError
        model = $Case.ModelId
        model_path = $Case.ModelPath
        profile = $Case.ProfileId
        mode = 'auto-cli'
        context = [string]$ManifestData.context
        kv_type = 'q8_0/q8_0'
        exit_code = $exitCode
        monitor_exit_code = $monitorExitCode
        completed_turns = $turnLines
        resources = $resource
    }
    Write-JsonFile -Path $resultPath -Value $result
    if ($result.status -eq 'pass') {
        Write-Host "PASS $($Case.Id)"
    }
    else {
        Write-Warning "FAIL $($Case.Id): $caseError"
    }
    return [pscustomobject]$result
}

function Get-TestCases {
    param(
        [Parameter(Mandatory = $true)]$ManifestData,
        [Parameter(Mandatory = $true)][string]$InfrExe
    )
    $cases = @()
    foreach ($profile in $ManifestData.profiles) {
        $profilePath = Resolve-RepoPath ([string]$profile.path)
        foreach ($model in $ManifestData.models) {
            $modelPath = Resolve-RepoPath ([string]$model.path)
            foreach ($mode in @('auto', 'manual')) {
                $cases += [pscustomobject]@{
                    Id = "$($profile.id)-$mode-$($model.id)"
                    ProfileId = [string]$profile.id
                    ProfilePath = $profilePath
                    ManualVramBudget = [string]$profile.manual_vram_budget
                    ManualRamBudget = [string]$profile.manual_ram_budget
                    ModelId = [string]$model.id
                    ModelPath = $modelPath
                    Mode = $mode
                    InfrExe = $InfrExe
                }
            }
        }
    }
    return $cases
}

function Assert-Manifest {
    param(
        [Parameter(Mandatory = $true)]$ManifestData,
        [Parameter(Mandatory = $true)][string]$InfrExe,
        [Parameter(Mandatory = $true)]$Cases
    )
    if ($PSVersionTable.PSVersion.Major -lt 5) {
        throw 'Windows PowerShell 5.1 or PowerShell 7 is required'
    }
    if ($env:OS -ne 'Windows_NT') {
        throw 'The resource matrix monitor currently requires Windows'
    }
    if (-not (Test-Path -LiteralPath $InfrExe -PathType Leaf)) {
        throw "infr executable not found: $InfrExe"
    }
    foreach ($case in $Cases) {
        if (-not (Test-Path -LiteralPath $case.ModelPath -PathType Leaf)) {
            throw "model not found for $($case.Id): $($case.ModelPath)"
        }
        if (-not (Test-Path -LiteralPath $case.ProfilePath -PathType Leaf)) {
            throw "profile not found for $($case.Id): $($case.ProfilePath)"
        }
        $profile = Read-JsonFile -Path $case.ProfilePath
        $null = ConvertTo-ByteCount $profile.vram_total
        $null = ConvertTo-ByteCount $profile.vram_used
        $null = ConvertTo-ByteCount $profile.ram_total
        $null = ConvertTo-ByteCount $profile.ram_used
    }
    $os = Get-CimInstance -ClassName 'Win32_OperatingSystem'
    $realRam = [uint64]$os.TotalVisibleMemorySize * 1024
    $largestProfile = [uint64]0
    foreach ($profileSpec in $ManifestData.profiles) {
        $profile = Read-JsonFile -Path (Resolve-RepoPath ([string]$profileSpec.path))
        $total = ConvertTo-ByteCount $profile.ram_total
        if ($total -gt $largestProfile) {
            $largestProfile = $total
        }
    }
    if ($largestProfile -gt ($realRam + 1GB)) {
        throw "largest RAM profile is $(Format-Gib $largestProfile) GiB, but this host reports $(Format-Gib $realRam) GiB"
    }
}

function Write-MatrixReport {
    param(
        [Parameter(Mandatory = $true)]$Results,
        [Parameter(Mandatory = $true)][string]$OutputRoot
    )
    $lines = [Collections.Generic.List[string]]::new()
    $lines.Add('# Long-context resource matrix')
    $lines.Add('')
    $lines.Add("Generated: $([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss zzz'))")
    $lines.Add('')
    $lines.Add('| Case | Model | Profile | Mode | Final depth | Peak private RAM GiB | Peak total WS GiB | Peak VRAM GiB | Result |')
    $lines.Add('|---|---|---|---|---:|---:|---:|---:|---|')
    foreach ($result in $Results) {
        $turnProperty = $result.PSObject.Properties['turns']
        $finalDepth = if ($null -ne $turnProperty -and @($turnProperty.Value).Count -gt 0) {
            $last = @($turnProperty.Value)[-1]
            [int64]$last.prompt_tokens + [int64]$last.completion_tokens
        }
        else {
            '-'
        }
        $peakRam = if ($null -ne $result.resources) {
            $privateWs = $result.resources.PSObject.Properties['peak_private_working_set_bytes']
            if ($null -ne $privateWs) {
                Format-Gib ([uint64]$privateWs.Value)
            }
            else {
                Format-Gib ([uint64]$result.resources.peak_working_set_bytes)
            }
        }
        else {
            '-'
        }
        $peakTotalWs = if ($null -ne $result.resources) {
            Format-Gib ([uint64]$result.resources.peak_working_set_bytes)
        }
        else {
            '-'
        }
        $peakVram = if ($null -ne $result.resources -and
            [bool]$result.resources.gpu_counter_available) {
            Format-Gib ([uint64]$result.resources.peak_gpu_dedicated_bytes)
        }
        else {
            '-'
        }
        $lines.Add("| $($result.id) | $($result.model) | $($result.profile) | $($result.mode) | $finalDepth | $peakRam | $peakTotalWs | $peakVram | $($result.status) |")
    }
    $lines.Add('')
    $lines.Add('Each case keeps its detailed requests, responses, logs, KV growth events, and resource samples in its own directory.')
    [IO.File]::WriteAllLines((Join-Path $OutputRoot 'report.md'), $lines, $script:Utf8)
}

$manifestPath = Resolve-RepoPath $Manifest
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Manifest not found: $manifestPath"
}
$manifestData = Read-JsonFile -Path $manifestPath
$infrExe = Resolve-RepoPath ([string]$manifestData.infr)
$outputRoot = Resolve-RepoPath ([string]$manifestData.output)
[IO.Directory]::CreateDirectory($outputRoot) | Out-Null
$cases = @(Get-TestCases -ManifestData $manifestData -InfrExe $infrExe)
$cliModel = @($manifestData.models | Where-Object { $_.id -eq $manifestData.cli_smoke.model })[0]
$cliProfile = @($manifestData.profiles | Where-Object { $_.id -eq $manifestData.cli_smoke.profile })[0]
$cliCase = [pscustomobject]@{
    Id = 'cli-smoke-qwen35-auto'
    ProfileId = [string]$cliProfile.id
    ProfilePath = Resolve-RepoPath ([string]$cliProfile.path)
    ModelId = [string]$cliModel.id
    ModelPath = Resolve-RepoPath ([string]$cliModel.path)
    InfrExe = $infrExe
}
$allIds = @($cases.Id) + @($cliCase.Id)
if ($CaseId) {
    foreach ($requested in $CaseId) {
        if ($allIds -notcontains $requested) {
            throw "Unknown case id: $requested"
        }
    }
    $cases = @($cases | Where-Object { $CaseId -contains $_.Id })
}

Assert-Manifest -ManifestData $manifestData -InfrExe $infrExe -Cases $cases
if ($List) {
    foreach ($case in $cases) {
        Write-Output "$($case.Id) | $($case.ModelPath) | $($case.ProfilePath)"
    }
    if (-not $SkipCliSmoke -and (-not $CaseId -or $CaseId -contains $cliCase.Id)) {
        Write-Output "$($cliCase.Id) | $($cliCase.ModelPath) | $($cliCase.ProfilePath)"
    }
    return
}

$results = @()
foreach ($case in $cases) {
    $results += Invoke-ApiCase -ManifestData $manifestData -Case $case -OutputRoot $outputRoot
}
if (-not $SkipCliSmoke -and (-not $CaseId -or $CaseId -contains $cliCase.Id)) {
    $cliParams = @{
        ManifestData = $manifestData
        Case = $cliCase
        OutputRoot = $outputRoot
    }
    $results += Invoke-CliSmoke @cliParams
}
Write-MatrixReport -Results $results -OutputRoot $outputRoot
$failed = @($results | Where-Object { $_.status -ne 'pass' })
if ($failed.Count -gt 0) {
    throw "$($failed.Count) context resource case(s) failed; see $(Join-Path $outputRoot 'report.md')"
}
