[CmdletBinding()]
param([ValidateSet('', 'dry', 'decline', 'reuse')][string]$ChildFixture = '')

$ErrorActionPreference = 'Stop'
$wizardPath = Join-Path $PSScriptRoot 'infr-wizard.ps1'
$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($wizardPath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
foreach ($definition in $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    . ([scriptblock]::Create($definition.Extent.Text))
}
$realStartModelReasoningProbe = ${function:Start-ModelReasoningProbe}
$realInvokeRecommendedModelDownload = ${function:Invoke-RecommendedModelDownload}
$source = Get-Content -LiteralPath $wizardPath -Raw -Encoding UTF8
$start = $source.IndexOf('$reuseSavedSettings = $false')
$save = $ast.EndBlock.Statements | Where-Object { $_.Extent.Text.StartsWith('$state = [ordered]') } | Select-Object -First 1
if ($start -lt 0 -or $null -eq $save) { throw 'Missing wizard flow' }
$flow = [scriptblock]::Create($source.Substring($start, $save.Extent.EndOffset - $start))
$choice = $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Read-Choice' }, $true) | Select-Object -First 1
. ([scriptblock]::Create($choice.Extent.Text.Replace('function Read-Choice {', 'function Read-TestChoice {')))

function Read-Choice {
    param([string]$Label, [object[]]$Options, [string]$DefaultValue)
    [void]$script:Events.Add($Label)
    Read-TestChoice @PSBoundParameters
}
function Read-Host {
    param([string]$Prompt, [switch]$AsSecureString)
    [void]$script:Events.Add($Prompt)
    if ($script:Answers.Count -eq 0) { throw "Unexpected prompt: $Prompt`n$($script:Events -join "`n")" }
    $answer = $script:Answers.Dequeue()
    if ($AsSecureString) { return ConvertTo-SecureString $answer -AsPlainText -Force }
    return $answer
}
function Write-Host {
    param($Object, $ForegroundColor)
    [void]$script:Output.Add([string]$Object)
    if ($ChildFixture) { Microsoft.PowerShell.Utility\Write-Host $Object }
}
function Get-VulkanDeviceOptions { return $script:Devices }
function Start-VulkanDeviceProbe {}
function Start-ModelReasoningProbe { param([string]$ModelPath) }
function Get-CimInstance { return [pscustomobject]@{ Name = 'Fixture CPU' } }
function Get-CpuMissTopology { return $script:CpuTopology }

function Assert {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}
function Write-GgufString {
    param([System.IO.BinaryWriter]$Writer, [string]$Value)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($Value)
    $Writer.Write([uint64]$bytes.Length)
    $Writer.Write($bytes)
}
function New-ModelFixture {
    param([string]$Path, [string]$Architecture, [string]$Template)
    $writer = [System.IO.BinaryWriter]::new([System.IO.File]::Create($Path))
    try {
        $writer.Write([uint32]0x46554747)
        $writer.Write([uint32]3)
        $writer.Write([uint64]0)
        $writer.Write([uint64]3)
        Write-GgufString $writer 'general.architecture'
        $writer.Write([uint32]8)
        Write-GgufString $writer $Architecture
        Write-GgufString $writer 'tokenizer.ggml.tokens'
        $writer.Write([uint32]9)
        $writer.Write([uint32]8)
        $writer.Write([uint64]2)
        Write-GgufString $writer 'token one'
        Write-GgufString $writer 'token two'
        Write-GgufString $writer 'tokenizer.chat_template'
        $writer.Write([uint32]8)
        Write-GgufString $writer $Template
    } finally { $writer.Dispose() }
}
function Test-Flow {
    param([string]$Name, $Saved, [string[]]$Inputs, [string]$InitialPath = '', [scriptblock]$Check)
    $script:Saved = $Saved
    $script:Answers = [System.Collections.Generic.Queue[string]]::new()
    foreach ($value in $Inputs) { $script:Answers.Enqueue($value) }
    $script:Events = [System.Collections.Generic.List[string]]::new()
    $script:Output = [System.Collections.Generic.List[string]]::new()
    $repoRoot = $testRoot
    $guiStatePath = Join-Path $testRoot 'no-gui-state.json'
    $infrPath = Join-Path $testRoot 'never-run.exe'
    $InitialModelPath = $InitialPath
    . $flow
    Assert ($script:Answers.Count -eq 0) "$Name left unused answers"
    . $Check
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host "PASS $Name"
}
function Test-UpdateAction {
    param([string]$Name, [string[]]$Inputs, [string]$Expected)
    $script:Answers = [System.Collections.Generic.Queue[string]]::new()
    foreach ($value in $Inputs) { $script:Answers.Enqueue($value) }
    $script:Events = [System.Collections.Generic.List[string]]::new()
    $script:Output = [System.Collections.Generic.List[string]]::new()
    $result = Read-EngineUpdateAction -Release ([pscustomobject]@{ Tag = 'release-9.9.9' })
    Assert ($result -eq $Expected -and $script:Answers.Count -eq 0) "$Name update action mismatch"
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host "PASS $Name"
}

$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('moe4all-wizard-test-' + [guid]::NewGuid().ToString('N'))
$script:Passed = 0
try {
    New-Item -ItemType Directory -Path $testRoot | Out-Null
    $model = Join-Path $testRoot 'model with spaces.gguf'
    $model35 = Join-Path $testRoot '35b.gguf'
    $aux = Join-Path $testRoot 'auxiliary model.gguf'
    New-ModelFixture $model 'qwen4exp' '{{ enable_thinking }} {{ reasoning_effort }}'
    New-ModelFixture $model35 'qwen35moe' '{{ enable_thinking }}'
    New-ModelFixture $aux 'qwen3' '{{ messages }}'
    $script:Devices = @([pscustomobject]@{ Key = '1'; Value = 'Vulkan0'; Label = 'Vulkan0: Fixture GPU [24 GiB]'; IsDefault = $true })
    $script:CpuTopology = [pscustomobject]@{ Physical = 6; Preferred = 6; Detected = $true }

    if ($ChildFixture) {
        $script:Saved = $null
        $script:Answers = [System.Collections.Generic.Queue[string]]::new()
        $script:Events = [System.Collections.Generic.List[string]]::new()
        $script:Output = [System.Collections.Generic.List[string]]::new()
        $repoRoot = $testRoot
        $dataDir = Join-Path $testRoot 'gui-data'
        $statePath = Join-Path $dataDir 'wizard-state.json'
        $guiStatePath = Join-Path $testRoot 'no-gui-state.json'
        $infrPath = 'Invoke-FixtureEngine'
        $InitialModelPath = ''
        $DryRun = ($ChildFixture -eq 'dry')
        function Invoke-FixtureEngine {
            Microsoft.PowerShell.Utility\Write-Host 'FIXTURE_ENGINE_STARTED'
            $global:LASTEXITCODE = 0
        }
        if ($ChildFixture -eq 'reuse') {
            $script:Saved = [pscustomobject]@{ launch_mode = 'chat'; setup_mode = 'conservative'; model = $model; device = 'Vulkan0' }
            $script:Answers.Enqueue('')
        } else {
            foreach ($value in @('', $model, '', '', '', '', '', '', '')) { $script:Answers.Enqueue($value) }
            if ($ChildFixture -eq 'decline') { $script:Answers.Enqueue('n') }
        }
        . ([scriptblock]::Create($source.Substring($start)))
        throw 'The complete wizard should exit normally'
    }

    Assert ($source.IndexOf('Show-UpdateStatus -CurrentVersion $productVersion') -lt $start) 'Update must precede saved launch'
    Test-UpdateAction 'update skipped by Enter' @('') 'skip'
    Test-UpdateAction 'explicit update still available' @('2') 'update'
    Test-UpdateAction 'invalid update input reprompts' @('bad', '') 'skip'
    function Show-EngineReleaseNotes { param($Release) [void]$script:Events.Add('release notes') }
    Test-UpdateAction 'release notes then skip' @('1', '') 'skip'

    Test-Flow 'fresh chat defaults' -Inputs @('', $model, '', '', '', '', '', '', '') -Check {
        Assert ($nativeArgs[0] -eq 'run') 'Expected chat'
        Assert ('65536' -in $nativeArgs -and 'kv.type_k=q8_0' -in $nativeArgs) 'Fresh defaults changed'
        Assert (-not ('--think' -in $nativeArgs) -and -not ('--reasoning-effort' -in $nativeArgs)) 'Default reasoning should be inherited'
        Assert (@($script:Events | Where-Object { $_ -match 'Compute device' }).Count -eq 1) 'One GPU still needs manual selection'
        Assert (($script:Output -join "`n") -notmatch 'CPU: Fixture CPU') 'CPU must not be a compute-device option'
    }
    Test-Flow 'homogeneous CPU miss defaults to physical minus two' -Inputs @('', $model, '', '', '', '', '', 'y', '', '', '') -Check {
        Assert ('kernels.vulkan.cpu_miss_threads=4' -in $nativeArgs) 'Six physical cores should default to four compute cores'
        Assert ('kernels.vulkan.cpu_miss_max=1' -in $nativeArgs) 'CPU miss count default changed'
    }
    $script:CpuTopology = [pscustomobject]@{ Physical = 24; Preferred = 8; Detected = $true }
    Test-Flow 'hybrid CPU miss defaults to total physical minus two' -Inputs @('', $model, '', '', '', '', '', 'y', '', '', '') -Check {
        Assert ('kernels.vulkan.cpu_miss_threads=22' -in $nativeArgs) 'Hybrid default should include efficiency cores while reserving two performance cores'
    }
    Test-Flow 'hybrid CPU miss allows selecting all physical cores' -Inputs @('', $model, '', '', '', '', '', 'y', '', '24', '') -Check {
        Assert ('kernels.vulkan.cpu_miss_threads=24' -in $nativeArgs) 'Explicit all-core selection must be honored'
    }
    $cpuSaved = [pscustomobject]@{ launch_mode = 'chat'; setup_mode = 'conservative'; model = $model; device = 'Vulkan0'; cpu_miss_enabled = $true; cpu_miss_cores = '8'; cpu_miss_max = '2' }
    Test-Flow 'saved CPU miss count survives reconfiguration' -Saved $cpuSaved -Inputs @('n', '', '', '', '', '', '', '', '', '', '', '') -Check {
        Assert ('kernels.vulkan.cpu_miss_threads=8' -in $nativeArgs) 'Saved custom CPU core count lost'
        Assert ('kernels.vulkan.cpu_miss_max=2' -in $nativeArgs) 'Saved CPU miss limit lost'
    }
    $script:CpuTopology = [pscustomobject]@{ Physical = 6; Preferred = 6; Detected = $true }
    Test-Flow 'aggressive API three slots and local cache' -Inputs @('2', $model, '', '', '', '2', '', '', '', '', '3', '100k', 'y', '', '', '') -Check {
        Assert ($nativeArgs[0] -eq 'serve' -and 'device.auto_profile=aggressive' -in $nativeArgs) 'Expected aggressive API'
        Assert ($state.server_parallel -eq '3' -and $state.context -eq '100k') 'Parallel/context prompts shifted'
        Assert ("kv.session_cache_dir=$testRoot\kv-sessions" -in $nativeArgs) 'Cache must default to the install directory'
        Assert ('serve.api_key=' -in $nativeArgs) 'Explicit no-auth override lost'
    }
    Test-Flow 'fresh API enables local cache by default' -Inputs @('2', $model, '', '', '', '', '', '', '', '', '', '', '', '', '', '') -Check {
        Assert ($state.server_session_cache) 'Fresh API installs should enable SSD KV caching'
        Assert ("kv.session_cache_dir=$testRoot\kv-sessions" -in $nativeArgs) 'Default SSD KV cache path was not emitted'
    }
    Test-Flow 'manual hardware and sampling' -Inputs @('', $model, '', '3', '3072', '4', '', '', 'n', '3', '256', 'n', 'paging.expert_prefetch=false', 'n', '2048', 'y', '0.7', '20', '0.95', '9', 'n', '20k') -Check {
        foreach ($value in @('--ubatch', '3072', '--threads', '4', 'device.submit_dispatches=256', 'paging.expert_prefetch=false', '--temp', '0.7', '--seed', '9')) {
            Assert ($value -in $nativeArgs) "Missing manual setting $value"
        }
        Assert ($state.context -eq '20k') 'Manual context should be asked once at the common step'
    }
    Test-Flow 'vision embedding before device and concurrency' -Inputs @('2', $model, 'y', $aux, 'y', $aux, '17', '', '', '', '', '', '', '2', '20k', 'y', '', '', '') -Check {
        foreach ($value in @('spec.mtp=false', '--mmproj', '--embedding-model')) {
            Assert ($value -in $nativeArgs) "Missing auxiliary setting $value"
        }
        Assert ($state.embedding_idle_timeout -eq '17') 'Auxiliary settings changed'
        $order = @('Enable image understanding', 'Also serve the Embedding', 'Compute device', 'Configuration', 'Configure default reasoning', 'experimental CPU', 'Concurrent slots', 'Context window', 'Cache idle-session', 'Listen address')
        $previous = -1
        foreach ($pattern in $order) {
            $position = -1
            for ($i = 0; $i -lt $script:Events.Count; $i++) { if ($script:Events[$i] -match $pattern) { $position = $i; break } }
            Assert ($position -gt $previous) "Prompt out of order: $pattern"
            $previous = $position
        }
    }
    Test-Flow 'FlashNext native reasoning effort' -Inputs @('', $model, '', '', 'y', 'y', '4', '', '', '', '') -Check {
        Assert ('--think' -in $nativeArgs -and 'xhigh' -in $nativeArgs) 'FlashNext reasoning effort was not emitted'
    }
    Test-Flow '35B prompt reasoning effort' -Inputs @('', $model35, '', '', 'y', 'y', '5', '', '', '', '') -Check {
        Assert ('--think' -in $nativeArgs -and '--reasoning-effort' -in $nativeArgs -and 'max' -in $nativeArgs) '35B reasoning effort was not emitted'
    }
    Test-Flow 'default reasoning disabled' -Inputs @('', $model, '', '', 'y', 'n', '', '', '', '') -Check {
        Assert ('--no-think' -in $nativeArgs -and -not ('--reasoning-effort' -in $nativeArgs)) 'Disable reasoning failed'
    }
    Test-Flow 'benchmark skips API and generation defaults' -Inputs @('3', $model, '', '', '', '90k', '3', '1024', '64', '3', '2048', '2', 'y') -Check {
        Assert ($nativeArgs[0] -eq 'bench' -and '--pg' -in $nativeArgs -and '--json' -in $nativeArgs) 'Benchmark flags lost'
        Assert (-not ('--max-new' -in $nativeArgs) -and -not ('spec.mtp=false' -in $nativeArgs)) 'Benchmark received chat settings'
    }
    $chatSettings = [pscustomobject]@{ launch_mode = 'chat'; setup_mode = 'quick'; model = $model; device = 'Vulkan0'; context = '150k'; max_new = '8192'; think_mode = 'think'; reasoning_effort = 'medium'; last_command = 'NEVER EXECUTE THIS' }
    Test-Flow 'saved launch uses structured settings without reconfiguration' -Saved $chatSettings -Inputs @('') -Check {
        Assert ($reuseSavedSettings -and $script:Events.Count -eq 1) 'Saved launch asked configuration questions'
        Assert ('8192' -in $nativeArgs -and '150k' -in $nativeArgs -and 'medium' -in $nativeArgs) 'Saved generation settings lost'
        Assert ('device.auto_profile=conservative' -in $nativeArgs) 'Legacy quick profile must migrate'
        Assert (-not ($commandText -match 'NEVER EXECUTE')) 'Saved command text must not execute'
    }
    Test-Flow 'decline saved launch keeps defaults' -Saved $chatSettings -Inputs @('n', '', '', '', '', '', '', 'y', 'y', '', '', '', '', '') -Check {
        Assert (-not $reuseSavedSettings -and $state.reasoning_effort -eq 'medium') 'Saved defaults were discarded'
    }
    $legacy = [pscustomobject]@{ launch_mode = 'chat'; model = $model; device = 'Vulkan0'; configure_memory = $true; dram_cache = '48g'; ubatch = '3072'; submit_mode = 'fixed'; submit_cap = '256'; max_new = '' }
    Test-Flow 'legacy manual settings survive direct launch' -Saved $legacy -Inputs @('') -Check {
        Assert ($setupMode -eq 'manual' -and 'device.ram_budget=48g' -in $nativeArgs -and '--ubatch' -in $nativeArgs) 'Legacy hardware settings lost'
        Assert (-not ('--max-new' -in $nativeArgs)) 'Saved blank max-new must keep the engine default'
    }
    $authenticated = [pscustomobject]@{ launch_mode = 'server'; setup_mode = 'aggressive'; model = $model; device = 'Vulkan0'; server_auth = $true; server_addr = '127.0.0.1:8080' }
    Test-Flow 'saved authenticated launch asks only for unsaved secret' -Saved $authenticated -Inputs @('', 'fixture-secret-not-a-real-key') -Check {
        Assert ($script:Events.Count -eq 2 -and $serverApiKey -eq 'fixture-secret-not-a-real-key') 'Saved authentication prompt changed'
        Assert (-not (($state | ConvertTo-Json) -match 'fixture-secret')) 'Secret leaked into saved settings'
    }
    Test-Flow 'explicit launcher model overrides previous model' -Saved $chatSettings -InitialPath $model35 -Inputs @('', '', '', '', 'y', 'y', '', '', '', '', '') -Check {
        Assert (-not $reuseSavedSettings -and $state.model -eq $model35) 'Launcher model must win'
        Assert ('--reasoning-effort' -in $nativeArgs -and 'medium' -in $nativeArgs) 'A reasoning effort supported by both models should survive the switch'
    }
    $missingSettings = [pscustomobject]@{ launch_mode = 'chat'; setup_mode = 'conservative'; model = (Join-Path $testRoot 'missing.gguf') }
    Test-Flow 'missing saved model goes through model selection' -Saved $missingSettings -Inputs @('', $model, '', '', '', '', '', '', '') -Check {
        Assert (-not $reuseSavedSettings -and $state.model -eq $model) 'Missing model should not launch directly'
        Assert (@($script:Events | Where-Object { $_ -match 'Start with the previous settings' }).Count -eq 0) 'Missing model should not offer direct launch'
    }
    Test-Flow 'single slot with vision keeps SSD cache' -Inputs @('2', $model, 'y', $aux, 'n', '', '', '', '', '', '', '1', '100k', 'y', '', '', '') -Check {
        Assert ($state.server_parallel -eq '1' -and $state.server_session_cache) 'Vision serving should keep SSD KV caching available'
    }
    $script:Saved = $null
    $script:ReasoningProbeModel = ''
    $script:ReasoningProbeLength = -1
    $script:ReasoningProbeMtimeTicks = -1
    $script:ReasoningProbeEfforts = @()
    $script:ModelReasoningProbe = $null
    & $realStartModelReasoningProbe -ModelPath $model
    $asyncEfforts = @(Get-ModelReasoningEfforts -ModelPath $model)
    Assert (($asyncEfforts -join ',') -eq 'low,medium,xhigh') 'Background reasoning probe returned the wrong capabilities'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS background reasoning capability probe'

    $cachedIdentity = Get-ModelProbeIdentity -ModelPath $model35
    $script:Saved = [pscustomobject]@{
        reasoning_probe_model = $cachedIdentity.Path
        reasoning_probe_length = $cachedIdentity.Length
        reasoning_probe_mtime_ticks = $cachedIdentity.LastWriteTicks
        reasoning_probe_efforts = @('low', 'medium', 'high', 'max')
    }
    $script:ReasoningProbeModel = ''
    $script:ReasoningProbeLength = -1
    $script:ReasoningProbeMtimeTicks = -1
    $script:ReasoningProbeEfforts = @()
    $cachedEfforts = @(Get-ModelReasoningEfforts -ModelPath $model35)
    Assert (($cachedEfforts -join ',') -eq 'low,medium,high,max') 'Persisted reasoning capability cache was not reused'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS persisted reasoning capability cache'
    $script:Devices = @(
        [pscustomobject]@{ Key = '1'; Value = 'Vulkan0'; Label = 'Vulkan0: small GPU'; IsDefault = $false },
        [pscustomobject]@{ Key = '2'; Value = 'Vulkan1'; Label = 'Vulkan1: large GPU'; IsDefault = $true },
        [pscustomobject]@{ Key = '3'; Value = 'Vulkan2'; Label = 'Vulkan2: third GPU'; IsDefault = $false }
    )
    $script:Answers = [System.Collections.Generic.Queue[string]]::new()
    $script:Answers.Enqueue('')
    $script:Output.Clear()
    $selected = Read-ComputeDevice -Default 'Vulkan2'
    Assert ($selected -eq 'Vulkan2') 'Saved device should be the default selection'
    foreach ($name in @('Vulkan0', 'Vulkan1', 'Vulkan2')) {
        Assert (($script:Output -join "`n") -match $name) "Device list omitted $name"
    }
    Assert (($script:Output -join "`n") -notmatch 'CPU') 'CPU must not be listed as a device'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS only GPUs listed, saved device selected'
    $script:Answers.Enqueue('')
    $selected = Read-ComputeDevice -Default 'cpu'
    Assert ($selected -eq 'Vulkan1') 'A saved CPU selection must fall back to the default GPU'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS saved CPU selection falls back to GPU'
    $script:Devices = @()
    $script:Events.Clear()
    $deviceError = ''
    try { Read-ComputeDevice | Out-Null } catch { $deviceError = $_.Exception.Message }
    Assert ($deviceError -match 'No Vulkan GPU was found') 'No GPU must produce a clear error'
    Assert ($script:Events.Count -eq 0) 'No GPU must not offer a CPU fallback or ask for a device'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS no GPU stops device selection without CPU fallback'

    $catalog = @(Get-RecommendedModelCatalog)
    $flashBundle = @($catalog | Where-Object { $_.Id -eq 'flashnext-427' })[0]
    $embeddingBundle = @($catalog | Where-Object { $_.Id -eq 'qwen3-embedding-06b-q8' })[0]
    Assert ($catalog.Count -eq 4 -and -not ($catalog.Kind -contains 'mtp')) 'Recommended catalog should contain only main, vision and embedding downloads'
    Assert (@($flashBundle.Files).Count -eq 33 -and $flashBundle.Primary -match '00001-of-00033') 'FlashNext shard catalog is incomplete'
    Assert ($embeddingBundle.Files[0].Url -match 'Qwen3-Embedding-0\.6B-Q8_0\.gguf') 'Embedding direct download URL is missing'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS recommended download catalog'

    function Invoke-RecommendedModelDownload {
        param([string]$Id)
        [void]$script:RequestedDownloadIds.Add($Id)
        return $aux
    }
    $script:Saved = $null
    $repoRoot = $testRoot
    $guiStatePath = Join-Path $testRoot 'no-gui-state.json'
    $script:Answers = [System.Collections.Generic.Queue[string]]::new()
    foreach ($answer in @('D', '1,3,4', 'y', $aux)) { $script:Answers.Enqueue($answer) }
    $script:Output.Clear()
    $script:RequestedDownloadIds = [System.Collections.Generic.List[string]]::new()
    $selected = Select-ModelPath
    Assert ($selected -eq $aux) 'Model selection should return to the local list after downloading'
    Assert (($script:RequestedDownloadIds -join ',') -eq 'qwen35-balanced,flashnext-vision,qwen3-embedding-06b-q8') 'Multi-select download did not preserve the requested items'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS model/component multi-select download returns to local selection'

    $script:Answers = [System.Collections.Generic.Queue[string]]::new()
    $script:Answers.Enqueue('D')
    $script:RequestedDownloadIds = [System.Collections.Generic.List[string]]::new()
    $selected = Select-VisionProjectorPath
    Assert ($selected -eq $aux -and ($script:RequestedDownloadIds -join ',') -eq 'flashnext-vision') 'Vision automatic download selection failed'
    $script:Answers.Enqueue('D')
    $script:RequestedDownloadIds = [System.Collections.Generic.List[string]]::new()
    $selected = Select-EmbeddingModelPath
    Assert ($selected -eq $aux -and ($script:RequestedDownloadIds -join ',') -eq 'qwen3-embedding-06b-q8') 'Embedding automatic download selection failed'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS Vision and Embedding automatic download selection'
    Set-Item -Path function:Invoke-RecommendedModelDownload -Value $realInvokeRecommendedModelDownload

    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $probe.Start()
    $downloadPort = ([System.Net.IPEndPoint]$probe.LocalEndpoint).Port
    $probe.Stop()
    $payload = [byte[]]::new(262144)
    for ($i = 0; $i -lt $payload.Length; $i++) { $payload[$i] = [byte]($i % 251) }
    $downloadJob = Start-Job -ArgumentList $downloadPort, $payload -ScriptBlock {
        param([int]$Port, [byte[]]$Payload)
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        Write-Output 'READY'
        try {
            for ($requestNumber = 1; $requestNumber -le 2; $requestNumber++) {
                $client = $listener.AcceptTcpClient()
                try {
                    $stream = $client.GetStream()
                    $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::ASCII, $false, 1024, $true)
                    $rangeStart = 0
                    while ($true) {
                        $line = $reader.ReadLine()
                        if ([string]::IsNullOrEmpty($line)) { break }
                        if ($line -match '^Range:\s*bytes=(\d+)-') { $rangeStart = [int]$Matches[1] }
                    }
                    if ($requestNumber -eq 1) {
                        $header = "HTTP/1.1 200 OK`r`nContent-Length: $($Payload.Length)`r`nConnection: close`r`n`r`n"
                        $bytes = [System.Text.Encoding]::ASCII.GetBytes($header)
                        $stream.Write($bytes, 0, $bytes.Length)
                        $half = [int]($Payload.Length / 2)
                        $stream.Write($Payload, 0, $half)
                        $stream.Flush()
                    } else {
                        $remaining = $Payload.Length - $rangeStart
                        $header = "HTTP/1.1 206 Partial Content`r`nContent-Length: $remaining`r`nContent-Range: bytes $rangeStart-$($Payload.Length - 1)/$($Payload.Length)`r`nConnection: close`r`n`r`n"
                        $bytes = [System.Text.Encoding]::ASCII.GetBytes($header)
                        $stream.Write($bytes, 0, $bytes.Length)
                        $stream.Write($Payload, $rangeStart, $remaining)
                        $stream.Flush()
                    }
                } finally {
                    $client.Close()
                }
            }
        } finally {
            $listener.Stop()
        }
    }
    try {
        $ready = $false
        for ($i = 0; $i -lt 50 -and -not $ready; $i++) {
            $ready = @((Receive-Job -Job $downloadJob -ErrorAction SilentlyContinue)) -contains 'READY'
            if (-not $ready) { Start-Sleep -Milliseconds 100 }
        }
        Assert $ready 'Local download fixture did not start'
        $downloadTarget = Join-Path $testRoot 'download retry\fixture.gguf'
        Invoke-ModelFileDownload -Url "http://127.0.0.1:$downloadPort/model.gguf" -Destination $downloadTarget -Label 'fixture' -MaximumAttempts 3
        $actualPayload = [System.IO.File]::ReadAllBytes($downloadTarget)
        Assert ($actualPayload.Length -eq $payload.Length) 'Retried download length changed'
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $actualHash = [System.Convert]::ToBase64String($sha.ComputeHash($actualPayload))
            $expectedHash = [System.Convert]::ToBase64String($sha.ComputeHash($payload))
        } finally {
            $sha.Dispose()
        }
        Assert ($actualHash -eq $expectedHash) 'Retried download content changed'
        Assert (-not (Test-Path -LiteralPath "$downloadTarget.part")) 'Successful download left a .part file'
        $script:Passed++
        Microsoft.PowerShell.Utility\Write-Host 'PASS interrupted download resumes from .part'
    } finally {
        if ($downloadJob.State -eq 'Running') { Stop-Job -Job $downloadJob }
        Remove-Job -Job $downloadJob -Force
    }

    foreach ($kind in @('vision', 'embedding')) {
        $script:Answers = [System.Collections.Generic.Queue[string]]::new()
        $script:Answers.Enqueue('R')
        $script:Answers.Enqueue($aux)
        $script:Output.Clear()
        $selected = switch ($kind) {
            'vision' { Select-VisionProjectorPath -Default $aux }
            'embedding' { Select-EmbeddingModelPath -Default $aux }
        }
        Assert ($selected -eq $aux -and $script:Answers.Count -eq 0) "$kind recommendation must return to path selection"
        Assert (($script:Output -join "`n") -match 'huggingface.co') "$kind recommendation link was omitted"
        $script:Passed++
        Microsoft.PowerShell.Utility\Write-Host "PASS $kind recommendation then path selection"
    }
    foreach ($scenario in @('dry', 'decline', 'reuse')) {
        $result = & powershell.exe -NoLogo -NoProfile -File $PSCommandPath -ChildFixture $scenario 2>&1 | Out-String
        $childExit = $LASTEXITCODE
        Assert ($childExit -eq 0) "Complete wizard $scenario failed: $result"
        if ($scenario -eq 'reuse') {
            Assert ($result -match 'FIXTURE_ENGINE_STARTED') 'Direct launch did not reach the fixture engine'
        } else {
            Assert ($result -notmatch 'FIXTURE_ENGINE_STARTED') "$scenario must not start the engine"
        }
        if ($scenario -eq 'dry') { Assert ($result -match 'DryRun: command was not started') 'Dry run did not complete' }
        $script:Passed++
        Microsoft.PowerShell.Utility\Write-Host "PASS complete wizard $scenario"
    }
    Microsoft.PowerShell.Utility\Write-Host "Launch wizard tests passed: $script:Passed; no engine, GPU, update or user state was touched."
} finally {
    if ($ChildFixture -eq 'decline') {
        Assert (-not (Test-Path -LiteralPath $statePath)) 'Declining launch must not save settings'
    } elseif ($ChildFixture) {
        Assert (Test-Path -LiteralPath $statePath -PathType Leaf) 'Accepted launch or DryRun must save settings'
    }
    $resolvedRoot = [System.IO.Path]::GetFullPath($testRoot)
    $tempPrefix = [System.IO.Path]::GetFullPath([System.IO.Path]::GetTempPath()).TrimEnd('\') + '\'
    if ($resolvedRoot.StartsWith($tempPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedRoot) -like 'moe4all-wizard-test-*' -and (Test-Path -LiteralPath $resolvedRoot)) {
        Remove-Item -LiteralPath $resolvedRoot -Recurse -Force
    }
}
