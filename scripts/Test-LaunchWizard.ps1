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
$realOfferBenchmarkUpload = ${function:Offer-BenchmarkUpload}
$realInvokeNativeCommandResult = ${function:Invoke-NativeCommandResult}
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
function Start-BenchmarkEstimateProbe {
    $script:BenchmarkEstimateMinutes = @{}
    foreach ($suiteName in @('quick', 'standard', 'full')) {
        foreach ($parallelCount in 1..4) {
            $base = switch ($suiteName) { 'quick' { 1 } 'standard' { 2 } default { 10 } }
            $script:BenchmarkEstimateMinutes["${suiteName}:$parallelCount"] = $base + $parallelCount - 1
        }
    }
}
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

$workRoot = if ($env:MOE4ALL_WORK_ROOT) { $env:MOE4ALL_WORK_ROOT } else { throw 'MOE4ALL_WORK_ROOT is required for wizard tests' }
$testParent = if ($env:MOE4ALL_WIZARD_TEST_TMP) { $env:MOE4ALL_WIZARD_TEST_TMP } else { Join-Path (Join-Path $workRoot 'tmp') 'moe4all' }
$testRoot = Join-Path $testParent ('wizard-test-' + [guid]::NewGuid().ToString('N'))
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
        Assert ('65536' -in $nativeArgs -and $state.kv_preset -eq 'auto') 'Fresh defaults changed'
        Assert (-not (@($nativeArgs | Where-Object { $_ -match '^kv.type_[kv]=' }).Count)) 'Automatic defaults must defer KV format to the engine'
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
    Test-Flow 'benchmark skips API and generation defaults' -Inputs @('3', $model, '', '', '', '4', '3', '1024', '64', '3', '2048', '2') -Check {
        Assert ($nativeArgs[0] -eq 'bench' -and '--pg' -in $nativeArgs -and '--json' -in $nativeArgs) 'Benchmark flags lost'
        Assert (-not ('--max-new' -in $nativeArgs) -and -not ('spec.mtp=false' -in $nativeArgs) -and -not ('--ctx' -in $nativeArgs)) 'Benchmark received chat or context settings'
        Assert (($script:Output -join "`n") -match 'Current plan') 'Manual benchmark did not show the incremental plan'
    }
    Test-Flow 'manual mode changes restore positive defaults' -Saved ([pscustomobject]@{ setup_mode = 'conservative'; prompt_tokens = '0'; gen_tokens = '0'; depth_tokens = '0' }) -Inputs @('3', $model, '', '1', '', '4', '3', '', '', '2', '', '1') -Check {
        Assert ('6144,512' -in $nativeArgs -and '-d' -in $nativeArgs -and '6144' -in $nativeArgs) 'Zero defaults survived switching manual modes'
    }
    Test-Flow 'manual positive defaults stay unchanged' -Saved ([pscustomobject]@{ setup_mode = 'conservative'; prompt_tokens = '1024'; gen_tokens = '64'; depth_tokens = '2048' }) -Inputs @('3', $model, '', '1', '', '4', '3', '', '', '2', '', '1') -Check {
        Assert ('1024,64' -in $nativeArgs -and '-d' -in $nativeArgs -and '2048' -in $nativeArgs) 'Valid manual defaults were overwritten'
    }
    Test-Flow 'automatic standard benchmark' -Inputs @('3', $model, '', '', '', '2', '3') -Check {
        $joinedOutput = $script:Output -join "`n"
        Assert ($nativeArgs[0] -eq 'bench' -and '--suite' -in $nativeArgs -and 'standard' -in $nativeArgs) 'Automatic suite flags lost'
        Assert ('--parallel' -in $nativeArgs -and '3' -in $nativeArgs -and '--result-dir' -in $nativeArgs) 'Automatic concurrency or result directory lost'
        Assert (-not ('--synthetic-depth' -in $nativeArgs) -and -not ('-p' -in $nativeArgs) -and -not ('--ctx' -in $nativeArgs)) 'Automatic suite must use its own real context plan'
        Assert ($joinedOutput.Contains('6K Prefill + 1024 Decode. x1, one run')) 'Fixed-token bilingual benchmark copy is missing'
        Assert ($joinedOutput.Contains('Shared KV, different questions.')) 'Shared context benchmark copy is missing'
        Assert (-not $joinedOutput.Contains('Estimated about')) 'Benchmark menu must not display an estimate'
        Assert (-not $state.benchmark_upload -and $state.benchmark_upload_has_github -and $state.benchmark_upload_gh_auth -and $state.benchmark_upload_fallback -eq 'folder') 'Benchmark upload preference defaults changed'
    }
    $chatSettings = [pscustomobject]@{ launch_mode = 'chat'; setup_mode = 'quick'; model = $model; device = 'Vulkan0'; context = '150k'; max_new = '8192'; think_mode = 'think'; reasoning_effort = 'medium'; last_command = 'NEVER EXECUTE THIS' }
    foreach ($profile in @('conservative', 'aggressive')) {
        foreach ($mode in @('chat', 'server', 'benchmark')) {
            $oldAutomatic = [pscustomobject]@{ launch_mode = $mode; setup_mode = $profile; model = $model; device = 'Vulkan0'; kv_preset = 'q8'; kv_type_k = 'q8_0'; kv_type_v = 'q8_0'; server_addr = '127.0.0.1:8080' }
            Test-Flow "saved automatic $profile $mode migrates stale Q8" -Saved $oldAutomatic -Inputs @('') -Check {
                Assert ($reuseSavedSettings) 'Saved automatic launch should still reuse settings'
                Assert (-not (@($nativeArgs | Where-Object { $_ -match '^kv.type_[kv]=' }).Count)) 'Stale Q8 must not override engine capability selection'
                Assert ($state.kv_preset -eq 'auto' -and -not $state.kv_type_k -and -not $state.kv_type_v) 'Migrated automatic state must stay automatic on the next launch'
            }
        }
    }
    foreach ($preset in @('q8', 'f16', 'custom', 'auto')) {
        $k = if ($preset -eq 'custom') { 'q4_0' } elseif ($preset -eq 'f16') { 'f16' } else { 'q8_0' }
        $v = if ($preset -eq 'custom') { 'f16' } else { $k }
        $manualKv = [pscustomobject]@{ launch_mode = 'chat'; setup_mode = 'manual'; model = $model; device = 'Vulkan0'; kv_preset = $preset; kv_type_k = $k; kv_type_v = $v }
        Test-Flow "manual $preset KV choice is preserved" -Saved $manualKv -Inputs @('') -Check {
            if ($preset -eq 'auto') {
                Assert (-not (@($nativeArgs | Where-Object { $_ -match '^kv.type_[kv]=' }).Count)) 'Manual engine-default must not pass stale values'
            } else {
                Assert ("kv.type_k=$k" -in $nativeArgs -and "kv.type_v=$v" -in $nativeArgs) 'Explicit manual KV choice was changed'
            }
        }
    }
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
        Assert (@($state.model_history).Count -eq 2 -and $state.model_history[0] -eq $model35 -and $state.model_history[1] -eq $model) 'Model history must retain every valid selected model'
    }
    $script:Saved = [pscustomobject]@{ model = $model; model_history = @($model35, $model) }
    $repoRoot = $testRoot
    $guiStatePath = Join-Path $testRoot 'no-gui-state.json'
    $script:Answers = [System.Collections.Generic.Queue[string]]::new()
    $script:Answers.Enqueue('2')
    $selectedHistoryModel = Select-ModelPath
    Assert ($selectedHistoryModel -eq $model35) 'Saved model history was not offered for selection'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS saved model history remains selectable'
    $resultDir = Join-Path $testRoot 'benchmark-database\v1\2026-10'
    New-Item -ItemType Directory -Path $resultDir -Force | Out-Null
    $dataDir = Join-Path $testRoot 'gui-data'
    $infrPath = Join-Path $testRoot 'never-run.exe'
    $localOnly = Write-ManualBenchmarkResult -RawJson '[{"avg_ts":123.4,"reps_ts":[123.4],"benchmark_metadata":{"engine_commit":"abcdef1234","hardware":{"cpu":{"name":"Native CPU"}},"model":{"name":"Native.gguf","quantization":"Q4_K"},"configuration":{"actual_ubatch":512,"kv_type_k":"q8_0","backend":"vulkan"}}}]' -ModelPath $model35 -Kind mixed -PromptTokens 512 -GenTokens 128 -DepthMode synthetic -DepthTokens 6144 -Repetitions 1
    Assert ($null -eq $localOnly) 'Mixed speed must not enter standardized upload'
    $localResult = @(Get-ChildItem -LiteralPath (Join-Path $dataDir 'benchmark-results') -Filter '*.json')
    Assert ($localResult.Count -eq 1) 'Mixed benchmark raw result was not retained'
    $localPayload = Get-Content -LiteralPath $localResult[0].FullName -Raw | ConvertFrom-Json
    Assert ($localPayload.configuration.benchmark_kind -eq 'mixed' -and $localPayload.contexts[0].manual_result.raw.reps_ts[0] -eq 123.4) 'Mixed result lost its parameters or raw samples'
    Assert ($localPayload.hardware.cpu.name -eq 'Native CPU' -and $localPayload.model.quantization -eq 'Q4_K' -and $localPayload.configuration.actual_ubatch -eq 512 -and $localPayload.engine_commit -eq 'abcdef1234') 'Manual export discarded native engine metadata'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS mixed benchmark stays local with its raw samples'
    $fixtureSubmissionPath = Join-Path $resultDir 'manual-fixture.m4b'
    $resultValue = [ordered]@{
        schema_version = 1; id = 'manual-fixture'; created_at_utc = '2026-10-18T00:00:00Z'; engine_version = '1.0.3'; profile = 'manual'
        model = [ordered]@{ name = 'fixture.gguf' }
        hardware = [ordered]@{ os = 'windows'; ram = [ordered]@{ total_bytes = 1024 }; gpu = [ordered]@{ name = 'Fixture GPU' } }
        contexts = @([ordered]@{ requested_context_tokens = 0; prefill = $null; decode = @(); manual_result = [ordered]@{ average_tokens_per_second = 1.0 }; failures = @() })
    }
    function Read-MoE4AllSubmissionPayload { param([string]$Path) return [System.IO.File]::ReadAllText($Path) | ConvertFrom-Json }
    [System.IO.File]::WriteAllText($fixtureSubmissionPath, ($resultValue | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
    Assert ((Test-BenchmarkResultForUpload -Path $fixtureSubmissionPath).id -eq 'manual-fixture') 'Valid manual benchmark submission was rejected'
    $browserPlan = Get-BenchmarkBrowserUploadPlan -Path $fixtureSubmissionPath
    Assert ($browserPlan.Url -match '/new/main/benchmark-database/v1/2026-10\?') 'Browser upload does not open the target month directory'
    $decodedBrowserUrl = [Uri]::UnescapeDataString($browserPlan.Url)
    Assert ($browserPlan.GuidancePrefilled -and $browserPlan.Url -match 'filename=manual-fixture.m4b' -and $browserPlan.Url -match '&value=' -and $browserPlan.Url -match '&message=') 'Browser upload did not prefill its guidance'
    Assert ($decodedBrowserUrl -match 'Ctrl\+A' -and $decodedBrowserUrl -match 'Ctrl\+V' -and $decodedBrowserUrl -match 'Create pull request') 'Browser upload guidance is incomplete'
    Assert ($decodedBrowserUrl -notmatch 'Fixture CPU') 'Browser upload must not put benchmark JSON in the URL'
    $errorBrowserPath = Join-Path $testRoot 'error-fixture.m4e'
    [System.IO.File]::WriteAllText($errorBrowserPath, (@{ schema_version = 1; id = 'error-fixture'; created_at_utc = '2026-10-18T00:00:00Z'; description = 'Synthetic error description' } | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
    $errorPlan = Get-SubmissionBrowserUploadPlan -Path $errorBrowserPath -Kind error -ClipboardReady $false
    $errorUrl = [Uri]::UnescapeDataString($errorPlan.Url)
    Assert ($errorUrl -match '/new/main/error-database/v1/2026-10\?' -and $errorUrl -match 'error: add error-fixture') 'Error reports must use their own GitHub database and commit title'
    Assert ($errorUrl -match 'Clipboard copy failed' -and $errorUrl -notmatch 'Synthetic error description') 'Missing clipboard must be explained without embedding report content'
    $resultValue['padding'] = 'x' * 7000
    [System.IO.File]::WriteAllText($fixtureSubmissionPath, ($resultValue | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
    $largeBrowserPlan = Get-BenchmarkBrowserUploadPlan -Path $fixtureSubmissionPath
    Assert ($largeBrowserPlan.GuidancePrefilled -and $largeBrowserPlan.Url.Length -lt 6000 -and $largeBrowserPlan.Url -match '&value=') 'Large browser upload must keep the short guidance URL'
    $resultValue.Remove('padding')
    [System.IO.File]::WriteAllText($fixtureSubmissionPath, ($resultValue | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
    Assert ($source -notmatch 'gh auth login|Invoke-GhBrowserLogin') 'Upload must not require CLI authorization before using the GitHub browser form'
    Assert ($source.Contains('$branch = "community/$Kind/$fileHash"')) 'Report branches must be deterministic and separated by kind'
    Assert ($source.Contains("Create the branch and report PR now?' -Default `$true")) 'Final automated PR confirmation should accept Enter'
    Assert ($source -notmatch 'Get-WindowsBenchmarkHardwareDetails|Add-BenchmarkHardwareDetails') 'Hardware collection must be owned by infr, not repeated by the wizard'
    $savedApiBase = [Environment]::GetEnvironmentVariable('MOE4ALL_SUBMISSION_API_BASE', 'Process')
    $savedSiteOrigin = [Environment]::GetEnvironmentVariable('MOE4ALL_SUBMISSION_SITE_ORIGIN', 'Process')
    try {
        [Environment]::SetEnvironmentVariable('MOE4ALL_SUBMISSION_API_BASE', $null, 'Process')
        [Environment]::SetEnvironmentVariable('MOE4ALL_SUBMISSION_SITE_ORIGIN', $null, 'Process')
        $formalApi = Get-Moe4AllSubmissionApiConfig
        Assert ($formalApi.ApiBase.AbsoluteUri -eq 'https://api.moe4all.com/v1' -and $formalApi.SiteOrigin -eq 'https://moe4all.com' -and -not $formalApi.IsLocalDevelopment) 'Formal submission API defaults changed'
        [Environment]::SetEnvironmentVariable('MOE4ALL_SUBMISSION_API_BASE', 'http://127.0.0.1:18765/v1', 'Process')
        $localApi = Get-Moe4AllSubmissionApiConfig
        Assert ($localApi.IsLocalDevelopment -and $localApi.SiteOrigin -eq 'http://127.0.0.1:18765') 'Explicit loopback submission API was rejected'
        [Environment]::SetEnvironmentVariable('MOE4ALL_SUBMISSION_API_BASE', 'http://example.invalid/v1', 'Process')
        $cleartextRejected = $false
        try { [void](Get-Moe4AllSubmissionApiConfig) } catch { $cleartextRejected = $true }
        Assert $cleartextRejected 'Non-loopback clear-text submission API must be rejected'
    } finally {
        [Environment]::SetEnvironmentVariable('MOE4ALL_SUBMISSION_API_BASE', $savedApiBase, 'Process')
        [Environment]::SetEnvironmentVariable('MOE4ALL_SUBMISSION_SITE_ORIGIN', $savedSiteOrigin, 'Process')
    }
    Assert ((Get-UnicodeCodePointCount -Value "a$([char]::ConvertFromUtf32(0x1F600))b") -eq 3) 'Unicode description length must count surrogate pairs once'
    Assert ($source -match '\$bytes\.Length -gt 131072') 'Website submission client must enforce the 128 KiB envelope limit'
    $receipt = [pscustomobject]@{
        receipt_id = 'rc_fixture'
        receipt_token = 'private-fixture-token'
        kind = 'error'
        report_id = 'fixture-report'
        file_sha256 = ('a' * 64)
        submitted_file_sha256 = ('a' * 64)
        payload_sha256 = ('b' * 64)
        status = 'forwarding'
        duplicate = $false
        received_at = '2026-10-19T00:00:00Z'
        updated_at = '2026-10-19T00:00:01Z'
        public_url = $null
    }
    $receiptPath = Save-Moe4AllSubmissionReceipt -Receipt $receipt -ApiBase ([Uri]'https://api.moe4all.com/v1')
    $savedReceipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    Assert ($savedReceipt.status -eq 'forwarding' -and $savedReceipt.updated_at -eq $receipt.updated_at) 'Forwarding receipt was not saved with its lifecycle state'
    $receipt.status = 'pending'
    $invalidReceiptRejected = $false
    try { [void](Save-Moe4AllSubmissionReceipt -Receipt $receipt -ApiBase ([Uri]'https://api.moe4all.com/v1')) } catch { $invalidReceiptRejected = $true }
    Assert $invalidReceiptRejected 'Unknown legacy receipt status must be rejected'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS website submission endpoint security policy'
    $savedStatePath = $statePath
    $savedStateObject = $script:state
    $savedUploadDefault = $script:BenchmarkUploadDefault
    $savedHasGithubDefault = $script:BenchmarkUploadHasGithubDefault
    $savedGhAuthDefault = $script:BenchmarkUploadGhAuthDefault
    $savedFallbackDefault = $script:BenchmarkUploadFallbackDefault
    $savedBrowserAction = $script:SubmissionBrowserActionDefault
    $savedFailureAction = $script:SubmissionFailureActionDefault
    $savedPendingRetry = $script:SubmissionPendingRetryDefault
    try {
        $statePath = Join-Path $testRoot 'upload-preferences.json'
        $script:state = [ordered]@{ unrelated = 'preserved' }
        $script:BenchmarkUploadDefault = $true
        $script:BenchmarkUploadHasGithubDefault = $false
        $script:BenchmarkUploadGhAuthDefault = $false
        $script:BenchmarkUploadFallbackDefault = 'clipboard'
        $script:SubmissionBrowserActionDefault = 'cloudflare'
        $script:SubmissionFailureActionDefault = 'later'
        $script:SubmissionPendingRetryDefault = $false
        Save-BenchmarkUploadPreferences
        $savedUploadPreferences = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
        Assert ($savedUploadPreferences.unrelated -eq 'preserved' -and $savedUploadPreferences.benchmark_upload -and -not $savedUploadPreferences.benchmark_upload_has_github -and -not $savedUploadPreferences.benchmark_upload_gh_auth -and $savedUploadPreferences.benchmark_upload_fallback -eq 'clipboard') 'Benchmark upload preferences were not persisted safely'
        Assert ($savedUploadPreferences.submission_browser_action -eq 'cloudflare' -and $savedUploadPreferences.submission_failure_action -eq 'later') 'Submission menu preferences were not saved'
        Assert ($savedUploadPreferences.submission_pending_retry -eq $false) 'Pending retry preference was not saved'
        $script:Answers.Enqueue('')
        Assert ((Read-SubmissionNextAction -Browser) -eq 'cloudflare') 'Enter must reuse the saved browser action'
    } finally {
        $statePath = $savedStatePath
        $script:state = $savedStateObject
        $script:BenchmarkUploadDefault = $savedUploadDefault
        $script:BenchmarkUploadHasGithubDefault = $savedHasGithubDefault
        $script:BenchmarkUploadGhAuthDefault = $savedGhAuthDefault
        $script:BenchmarkUploadFallbackDefault = $savedFallbackDefault
        $script:SubmissionBrowserActionDefault = $savedBrowserAction
        $script:SubmissionFailureActionDefault = $savedFailureAction
        $script:SubmissionPendingRetryDefault = $savedPendingRetry
    }
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS benchmark browser prefill and engine-owned hardware collection'
    $resultValue.model.local_path = 'C:\private\model.gguf'
    [System.IO.File]::WriteAllText($fixtureSubmissionPath, ($resultValue | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
    $privacyRejected = $false
    try { [void](Test-BenchmarkResultForUpload -Path $fixtureSubmissionPath) } catch { $privacyRejected = $true }
    Assert $privacyRejected 'Benchmark upload must reject local paths'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS benchmark upload privacy gate'
    $resultValue.model.Remove('local_path')
    [System.IO.File]::WriteAllText($fixtureSubmissionPath, ($resultValue | ConvertTo-Json -Depth 8), [System.Text.UTF8Encoding]::new($false))
    $pendingSubmissionDir = Join-Path $dataDir 'private\pending-submissions'
    $queuedPath = Save-PendingSubmission -Path $fixtureSubmissionPath -Kind benchmark
    Assert ((Test-Path -LiteralPath $queuedPath -PathType Leaf) -and (Get-FileHash $queuedPath).Hash -eq (Get-FileHash $fixtureSubmissionPath).Hash) 'Pending submission was not preserved byte-for-byte'
    Remove-PendingSubmission -Path $queuedPath
    Assert (-not (Test-Path -LiteralPath $queuedPath)) 'Successful pending submission cleanup failed'

    $realGhSubmissionUpload = ${function:Invoke-GhSubmissionUpload}
    $realWebsiteSubmissionUpload = ${function:Invoke-WebsiteSubmissionUpload}
    $realTestGhAuthenticated = ${function:Test-GhAuthenticated}
    $realOpenSubmissionBrowserUpload = ${function:Open-SubmissionBrowserUpload}
    function gh { }
    function Test-GhAuthenticated { return $script:GhSignedIn }
    function Invoke-GhSubmissionUpload {
        param($Path, $Kind, [switch]$Retry)
        $script:GitAttempts++
        if ($script:CancelGit) { $script:SubmissionUploadCancelled = $true; return $false }
        return $script:GitResults.Dequeue()
    }
    function Invoke-WebsiteSubmissionUpload {
        $script:CloudflareAttempts++
        return $script:WebResults.Dequeue()
    }
    function Open-SubmissionBrowserUpload { $script:BrowserAttempts++ }
    function Reset-UploadFixture {
        param([bool[]]$GitResults = @(), [bool[]]$WebResults = @(), [string[]]$Answers = @(), [bool]$SignedIn = $true)
        $script:GitAttempts = 0; $script:CloudflareAttempts = 0; $script:BrowserAttempts = 0
        $script:CancelGit = $false; $script:GhSignedIn = $SignedIn
        $script:GitResults = [System.Collections.Generic.Queue[bool]]::new()
        foreach ($value in $GitResults) { $script:GitResults.Enqueue($value) }
        $script:WebResults = [System.Collections.Generic.Queue[bool]]::new()
        foreach ($value in $WebResults) { $script:WebResults.Enqueue($value) }
        $script:Answers = [System.Collections.Generic.Queue[string]]::new()
        foreach ($value in $Answers) { $script:Answers.Enqueue($value) }
    }
    Reset-UploadFixture -GitResults @($true) -Answers @('y')
    Assert (Offer-SubmissionUpload -Path $fixtureSubmissionPath -Kind benchmark -ConsentAlreadyGranted) 'CLI first-attempt success failed'
    Assert ($script:GitAttempts -eq 1 -and $script:BrowserAttempts -eq 0 -and $script:CloudflareAttempts -eq 0) 'CLI success must finish immediately'
    Reset-UploadFixture -GitResults @($false, $true) -Answers @('y')
    Assert (Offer-SubmissionUpload -Path $fixtureSubmissionPath -Kind benchmark -ConsentAlreadyGranted) 'CLI second-attempt success failed'
    Assert ($script:GitAttempts -eq 2 -and $script:BrowserAttempts -eq 0) 'CLI must retry once before opening the browser'
    Reset-UploadFixture -GitResults @($false, $false) -Answers @('y', '4')
    Assert (-not (Offer-SubmissionUpload -Path $fixtureSubmissionPath -Kind benchmark -ConsentAlreadyGranted)) 'Send-later choice must retain the report'
    Assert ($script:GitAttempts -eq 2 -and $script:BrowserAttempts -eq 1 -and $script:CloudflareAttempts -eq 0) 'CLI failures must open GitHub, not auto-switch to Cloudflare'
    Reset-UploadFixture -GitResults @($false, $false) -WebResults @($true) -Answers @('y', '3')
    Assert (Offer-SubmissionUpload -Path $fixtureSubmissionPath -Kind benchmark -ConsentAlreadyGranted) 'Explicit Cloudflare fallback failed'
    Assert ($script:GitAttempts -eq 2 -and $script:BrowserAttempts -eq 1 -and $script:CloudflareAttempts -eq 1) 'Explicit fallback order changed'
    Reset-UploadFixture -SignedIn $false -Answers @('y', '2', '1')
    Assert (Offer-SubmissionUpload -Path $fixtureSubmissionPath -Kind benchmark -ConsentAlreadyGranted) 'Browser reopen then user confirmation failed'
    Assert ($script:GitAttempts -eq 0 -and $script:BrowserAttempts -eq 2 -and $script:CloudflareAttempts -eq 0) 'Unsigned CLI must not upload, authorize or choose a relay'
    Reset-UploadFixture -WebResults @($true) -Answers @('n')
    Assert (Offer-SubmissionUpload -Path $fixtureSubmissionPath -Kind benchmark -ConsentAlreadyGranted) 'No-account Cloudflare route failed'
    Assert ($script:GitAttempts -eq 0 -and $script:BrowserAttempts -eq 0 -and $script:CloudflareAttempts -eq 1) 'No-account route must directly use Cloudflare'
    Reset-UploadFixture -WebResults @($false, $true) -Answers @('n', '1')
    Assert (Offer-SubmissionUpload -Path $fixtureSubmissionPath -Kind benchmark -ConsentAlreadyGranted) 'Manual Cloudflare retry failed'
    Assert ($script:CloudflareAttempts -eq 2) 'Cloudflare retry must be user-selected'
    Reset-UploadFixture -SignedIn $false -WebResults @($false) -Answers @('n', '2', 'y', '1')
    Assert (Offer-SubmissionUpload -Path $fixtureSubmissionPath -Kind benchmark -ConsentAlreadyGranted) 'Return to channel selection failed'
    Assert ($script:CloudflareAttempts -eq 1 -and $script:BrowserAttempts -eq 1) 'Channel selection did not follow the user'
    Reset-UploadFixture -WebResults @($false) -Answers @('n', '3')
    Assert (-not (Offer-SubmissionUpload -Path $fixtureSubmissionPath -Kind benchmark -ConsentAlreadyGranted)) 'Failed relay must not claim success'
    Assert (@(Get-PendingSubmissions).Count -eq 1 -and (Test-Path -LiteralPath $fixtureSubmissionPath)) 'Send-later lost the original or queue'
    Reset-UploadFixture -Answers @('y')
    $script:CancelGit = $true
    Assert (-not (Offer-SubmissionUpload -Path $fixtureSubmissionPath -Kind benchmark -ConsentAlreadyGranted)) 'Cancelled CLI submission must remain pending'
    Assert ($script:GitAttempts -eq 1 -and $script:BrowserAttempts -eq 0 -and $script:CloudflareAttempts -eq 0) 'User cancellation must stop retries and fallbacks'
    Reset-UploadFixture -Answers @('')
    $script:SubmissionPendingRetryDefault = $false
    $script:PendingSubmissionPrompted = $false
    Offer-PendingSubmissions
    Assert (@(Get-PendingSubmissions).Count -eq 1 -and $script:GitAttempts -eq 0) 'Enter must preserve the previous decision to retry later'
    Reset-UploadFixture -GitResults @($true) -Answers @('y', 'y')
    $script:PendingSubmissionPrompted = $false
    Offer-PendingSubmissions
    Assert (@(Get-PendingSubmissions).Count -eq 0 -and (Test-Path -LiteralPath $fixtureSubmissionPath)) 'Pending retry must remove only the queue copy'
    Assert ($source -notmatch 'SubmissionConnectivity|SubmissionConnectivityProbe|Get-SubmissionConnectivitySnapshot') 'Submission connectivity preflight must be removed'
    $savedProxyValue = [Environment]::GetEnvironmentVariable('HTTPS_PROXY', 'Process')
    try {
        [Environment]::SetEnvironmentVariable('HTTPS_PROXY', 'http://127.0.0.1:9', 'Process')
        $originalNoProxy = $env:NO_PROXY
        $sawDirect = Invoke-WithSubmissionGithubRoute -Action { return ($env:HTTPS_PROXY -eq 'http://127.0.0.1:9' -and $env:NO_PROXY -eq $originalNoProxy) }
        Assert $sawDirect 'GitHub transport changed explicit proxy settings'
        Assert ($env:HTTPS_PROXY -eq 'http://127.0.0.1:9') 'GitHub route selection leaked proxy changes into the parent process'
        $savedNoProxy = $env:NO_PROXY
        try {
            $env:NO_PROXY = '.fixture.invalid:443,localhost'
            Assert ((Get-SubmissionProxy -ProbeUri ([Uri]'https://api.fixture.invalid')).IsBypassed([Uri]'https://api.fixture.invalid')) 'NO_PROXY domain/port bypass was ignored'
            Assert (-not (Get-SubmissionProxy -ProbeUri ([Uri]'https://api.fixture.invalid:444')).IsBypassed([Uri]'https://api.fixture.invalid:444')) 'NO_PROXY port bypass matched another port'
            $env:NO_PROXY = '*'
            Assert ((Get-SubmissionProxy -ProbeUri ([Uri]'https://api.github.com')).IsBypassed([Uri]'https://api.github.com')) 'NO_PROXY wildcard was ignored'
        } finally { $env:NO_PROXY = $savedNoProxy }
    } finally {
        [Environment]::SetEnvironmentVariable('HTTPS_PROXY', $savedProxyValue, 'Process')
    }
    Set-Item Function:Invoke-GhSubmissionUpload $realGhSubmissionUpload
    Set-Item Function:Invoke-WebsiteSubmissionUpload $realWebsiteSubmissionUpload
    Set-Item Function:Test-GhAuthenticated $realTestGhAuthenticated
    Set-Item Function:Open-SubmissionBrowserUpload $realOpenSubmissionBrowserUpload
    Remove-Item Function:gh
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS asynchronous route priority and pending submission lifecycle'
    $nativeFailure = Invoke-NativeCommandResult -Executable (Join-Path $PSHOME 'powershell.exe') -Arguments @(
        '-NoLogo', '-NoProfile', '-NonInteractive', '-Command',
        '[Console]::Error.WriteLine(''not logged in''); exit 1'
    )
    Assert ($nativeFailure.ExitCode -eq 1) 'Expected native stderr probe failure was not returned as an exit code'
    Assert (($nativeFailure.Output -join "`n") -match 'not logged in') 'Native stderr was not captured for diagnostics'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS native stderr does not terminate wizard'
    function Offer-BenchmarkUpload { throw 'fixture upload failure' }
    $uploadFailureEscaped = $false
    try { Invoke-BenchmarkUploadSafely -Path $fixtureSubmissionPath } catch { $uploadFailureEscaped = $true }
    Set-Item -Path Function:Offer-BenchmarkUpload -Value $realOfferBenchmarkUpload
    Assert (-not $uploadFailureEscaped) 'Upload helper failure escaped and changed the wizard result'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS upload failure cannot override benchmark success'
    $script:Answers = [System.Collections.Generic.Queue[string]]::new()
    foreach ($answer in @('first line', '.', '3', 'second line', '.', '')) { $script:Answers.Enqueue($answer) }
    $script:Events = [System.Collections.Generic.List[string]]::new()
    $script:Output = [System.Collections.Generic.List[string]]::new()
    $description = Read-ProblemDescription
    Assert ($description -eq "first line`nsecond line" -and $script:Answers.Count -eq 0) 'Problem description review/append flow changed'
    foreach ($answer in @('.', '', 'required description', '.', '')) { $script:Answers.Enqueue($answer) }
    $requiredDescription = Read-ProblemDescription
    Assert ($requiredDescription -eq 'required description' -and $script:Answers.Count -eq 0) 'Empty error descriptions must be rejected before upload'
    Assert ($source -match "Value = 'report'" -and $source -match 'INFR_DIAGNOSTIC_DIR') 'Runtime/fatal report entry is not wired into the launcher'
    $script:Passed++
    Microsoft.PowerShell.Utility\Write-Host 'PASS error report description review and launcher wiring'
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
    if ($ChildFixture -in @('decline', 'dry')) {
        Assert (-not (Test-Path -LiteralPath $statePath)) 'Declining launch or DryRun must not save settings'
    } elseif ($ChildFixture) {
        Assert (Test-Path -LiteralPath $statePath -PathType Leaf) 'Accepted launch must save settings'
    }
    $resolvedRoot = [System.IO.Path]::GetFullPath($testRoot)
    $tempPrefix = [System.IO.Path]::GetFullPath((Join-Path (Join-Path $workRoot 'tmp') 'moe4all')).TrimEnd('\') + '\'
    if ($resolvedRoot.StartsWith($tempPrefix, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Split-Path -Leaf $resolvedRoot) -like 'wizard-test-*' -and (Test-Path -LiteralPath $resolvedRoot)) {
        Remove-Item -LiteralPath $resolvedRoot -Recurse -Force
    }
}
