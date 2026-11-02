[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$WorkDir)

$ErrorActionPreference = 'Stop'
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
if (-not $WorkDir.StartsWith('D:\AIinfr\infr\local-work\tmp\moe4all\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Use local-work/tmp/moe4all/<task>/windows.'
}
[void][IO.Directory]::CreateDirectory($WorkDir)
$testRoot = Join-Path $WorkDir ('modules-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($testRoot)
$source = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'infr-wizard.ps1'))
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseInput($source, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
# Import functions and evaluate only configuration/argv/state assembly. Never run the entry point.
foreach ($fn in $ast.FindAll({ param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    . ([scriptblock]::Create($fn.Extent.Text))
}
$start = $source.IndexOf('$reuseSavedSettings = $false')
$endMarker = 'if (-not $llmEnabled) { Restore-DisabledLlmState -State $state }'
$end = $source.IndexOf($endMarker)
if ($start -lt 0 -or $end -lt $start) { throw 'Missing wizard configuration flow' }
$flow = [scriptblock]::Create($source.Substring($start, $end + $endMarker.Length - $start))
$script:checks = 0
function Check($Condition, [string]$Message) {
    if (-not $Condition) { throw $Message }
    $script:checks++
}
function Read-Host {
    param($Prompt, [switch]$AsSecureString)
    [void]$script:prompts.Add([string]$Prompt)
    if (-not $script:answers.Count) { throw "Unexpected prompt: $Prompt" }
    $answer = $script:answers.Dequeue()
    if ($AsSecureString) { return ConvertTo-SecureString $answer -AsPlainText -Force }
    return $answer
}
function Write-Host { param($Object, $ForegroundColor) [void]$script:output.Add([string]$Object) }
function Invoke-BenchmarkStartupGuide { param($Interactive, $JsonAutomation) return $false }
function Test-BenchmarkGuideInteractive { return $false }
function Start-VulkanDeviceProbe { $script:deviceProbes++ }
function Get-VulkanDeviceOptions {
    return @([pscustomobject]@{ Key = '1'; Value = 'Vulkan1'; Label = 'Fixture device'; IsDefault = $true })
}
function Start-ModelReasoningProbe { param($ModelPath) $script:reasoningProbes++ }
function Select-ModelPath {
    param($InitialPath)
    Check ($script:prompts.Count -ge 4) 'Model selection preceded module switches'
    $script:llmSelections++
    return $llmFile
}
function Find-VisionProjectorPath { param($ModelPath) $script:visionProbes++; return '' }
function Invoke-RecommendedModelDownload {
    param($Id)
    Check ($Id -eq 'sensevoice-small-q8') 'Unexpected download'
    $script:downloads++
    return $audioFile
}
function Invoke-NativeCommandResult { throw 'Native execution is forbidden in module tests' }
function Get-CpuMissTopology { throw 'LLM CPU topology must not be probed by these fixtures' }

$llmFile = Join-Path $testRoot "language model's.gguf"
$embeddingFile = Join-Path $testRoot "embedding model's.gguf"
$audioFile = Join-Path $testRoot "audio model's.gguf"
foreach ($file in @($llmFile, $embeddingFile, $audioFile)) { [IO.File]::WriteAllText($file, 'fixture') }

function Test-Modules {
    param([int]$Mask, $Saved, [switch]$Reuse, [switch]$RejectEmptyFirst, [switch]$DownloadAudio, [switch]$Authenticate)
    $script:Saved = $Saved
    $script:answers = [Collections.Generic.Queue[string]]::new()
    $script:prompts = [Collections.Generic.List[string]]::new()
    $script:output = [Collections.Generic.List[string]]::new()
    $script:deviceProbes = 0; $script:reasoningProbes = 0; $script:llmSelections = 0; $script:visionProbes = 0; $script:downloads = 0
    $llm = ($Mask -band 1) -ne 0; $embedding = ($Mask -band 2) -ne 0; $audio = ($Mask -band 4) -ne 0
    $repoRoot = $testRoot; $dataDir = Join-Path $testRoot 'state with spaces'
    $guiStatePath = Join-Path $testRoot 'no-gui-state.json'; $statePath = Join-Path $dataDir 'wizard-state.json'
    $infrPath = Join-Path $testRoot 'never-execute.exe'; $InitialModelPath = ''; $DryRun = $true
    if ($null -ne $Saved -and (Test-SavedWizardModels)) { $script:answers.Enqueue($(if ($Reuse) { '' } else { 'n' })) }
    if (-not $Reuse) {
        $script:answers.Enqueue('2')
        if ($RejectEmptyFirst) { foreach ($value in @('n', 'n', 'n')) { $script:answers.Enqueue($value) } }
        foreach ($enabled in @($llm, $embedding, $audio)) { $script:answers.Enqueue($(if ($enabled) { 'y' } else { 'n' })) }
        if ($llm) { $script:answers.Enqueue('n') } # Vision off.
        if ($embedding) { $script:answers.Enqueue($embeddingFile); $script:answers.Enqueue('17') }
        if ($audio) {
            $script:answers.Enqueue($(if ($DownloadAudio) { 'D' } else { $audioFile }))
            foreach ($value in @('', '', '')) { $script:answers.Enqueue($value) } # threads, idle, upload dir
        }
        if ($llm -or $embedding) { $script:answers.Enqueue('') } # Shared device.
        if ($llm) {
            # Conservative setup, reasoning, max-new, sampling, CPU miss, parallel, context, SSD cache.
            foreach ($value in @('1', 'n', '', 'n', 'n', '', '', 'n')) { $script:answers.Enqueue($value) }
        }
        $script:answers.Enqueue('') # Listen address.
        $script:answers.Enqueue($(if ($Authenticate) { 'y' } else { 'n' }))
    }
    if ($Authenticate) { $script:answers.Enqueue('fixture-secret') }
    . $flow
    Check ($script:answers.Count -eq 0) "Unused answers for mask $Mask"
    Check ($nativeArgs[0] -eq 'serve') 'Expected shared serve command'
    Check ($state.server_llm_enabled -is [bool] -and $state.server_llm_enabled -eq $llm) 'LLM bool state'
    Check ($state.server_embedding -is [bool] -and $state.server_embedding -eq $embedding) 'Embedding bool state'
    Check ($state.server_transcription -is [bool] -and $state.server_transcription -eq $audio) 'Transcription bool state'
    Check (($nativeArgs.Contains('--embedding-model')) -eq $embedding) 'Embedding argv gate'
    Check (($nativeArgs.Contains('--transcription-model')) -eq $audio) 'Transcription argv gate'
    Check (($nativeArgs.Contains($llmFile)) -eq $llm) 'Positional LLM gate'
    Check (($nativeArgs.Contains('--parallel')) -eq $llm) 'LLM concurrency gate'
    Check (($nativeArgs.Contains('spec.mtp=false')) -eq $llm) 'LLM MTP gate'
    if ($embedding) {
        Check ($nativeArgs[$nativeArgs.IndexOf('--embedding-model') + 1] -eq $embeddingFile) 'Embedding path argv boundary'
        Check ($nativeArgs[$nativeArgs.IndexOf('--embedding-idle-timeout') + 1] -eq '17') 'Embedding idle argv'
    }
    if ($audio) {
        Check ($nativeArgs[$nativeArgs.IndexOf('--transcription-dev') + 1] -eq 'cpu') 'Transcription is always CPU'
        Check ($nativeArgs[$nativeArgs.IndexOf('--transcription-threads') + 1] -eq '4') 'Transcription defaults to four threads'
        Check ($nativeArgs[$nativeArgs.IndexOf('--transcription-model') + 1] -eq $audioFile) 'Audio path argv boundary'
    }
    if (-not $llm) {
        $allowed = @('serve', '--dev', '--addr', '--set', '--embedding-model', '--embedding-idle-timeout',
            '--transcription-model', '--transcription-dev', '--transcription-threads', '--transcription-idle-timeout', '--transcription-upload-dir')
        foreach ($argument in $nativeArgs) {
            if ($argument.StartsWith('--')) { Check ($argument -in $allowed) "Unexpected non-LLM argument: $argument" }
            if ($argument.Contains('=')) { Check ($argument -eq 'serve.api_key=') "Unexpected non-LLM override: $argument" }
        }
        Check ($script:llmSelections -eq 0 -and $script:reasoningProbes -eq 0 -and $script:visionProbes -eq 0) 'Disabled LLM must not select or probe'
        Check (-not (@($script:prompts | Where-Object { $_ -match 'Configuration|Context window|Ubatch|KV cache|Configure default reasoning|Concurrent slots|CPU expert|image understanding' }).Count)) 'Disabled LLM must not ask LLM prompts'
        if ($null -ne $Saved) {
            foreach ($property in $Saved.PSObject.Properties) {
                if ($property.Name -in @('launch_mode', 'server_llm_enabled', 'server_embedding', 'embedding_model', 'embedding_idle_timeout', 'server_transcription', 'server_addr', 'server_auth', 'last_command', 'device')) { continue }
                Check (($state[$property.Name] | ConvertTo-Json -Depth 8 -Compress) -ceq ($property.Value | ConvertTo-Json -Depth 8 -Compress)) "Preserved LLM setting changed: $($property.Name)"
            }
        }
    }
    $expectedProbes = if (-not $Reuse -and ($llm -or $embedding)) { 1 } else { 0 }
    Check ($script:deviceProbes -eq $expectedProbes) 'Device discovery gate'
    if (-not ($llm -or $embedding)) {
        Check ($nativeArgs[$nativeArgs.IndexOf('--dev') + 1] -eq 'cpu') 'Audio-only command must select CPU'
        Check ($state.device -eq [string](Get-SavedValue 'device' '')) 'Audio-only must preserve selected user device'
    }
    Check ($script:downloads -eq [int]$DownloadAudio.IsPresent) 'Download requires explicit D'
    if (-not $Reuse) {
        $switchPrompts = @($script:prompts | Where-Object { $_ -match 'Enable the .* API\?' })
        $expectedCount = if ($RejectEmptyFirst) { 6 } else { 3 }
        Check ($switchPrompts.Count -eq $expectedCount) 'Exactly three module switches before model prompts'
        $ordered = @('Enable the language model API?', 'Enable the Embedding API?', 'Enable the local transcription API?')
        for ($index = 0; $index -lt $expectedCount; $index++) {
            Check ($switchPrompts[$index].Contains($ordered[$index % 3])) 'Module switch order'
        }
        if ($embedding) {
            Check (($script:output -join "`n").Contains('The Embedding model and backend load on the first request; the entire instance unloads after the idle timeout.')) 'Embedding lazy initialization explanation'
        }
        if ($audio) {
            Check (($script:output -join "`n").Contains('Transcription uses the CPU with 4 threads by default; the loaded model uses about 300 MiB RAM.')) 'CPU/RAM explanation'
            Check (($script:output -join "`n").Contains('no extra GPU weights or external runtime are needed.')) 'No external runtime or GPU weights explanation'
            Check (($script:output -join "`n").Contains('WAV/PCM')) 'WAV explanation'
        }
    }
    if ($Authenticate) {
        Check ($serverApiKey -eq 'fixture-secret' -and -not $commandText.Contains('fixture-secret') -and -not ($state | ConvertTo-Json).Contains('fixture-secret')) 'API key remains hidden and unsaved'
    }
    if ($Reuse) { Check ($reuseSavedSettings -and $script:prompts.Count -eq 1) 'Auxiliary-only saved launch reuse' }
    # Save and reload only test state, then verify bool and preserved model-history types.
    $jsonPath = Join-Path $testRoot "mask-$Mask.json"
    [IO.File]::WriteAllText($jsonPath, ($state | ConvertTo-Json -Depth 8))
    $restored = [IO.File]::ReadAllText($jsonPath) | ConvertFrom-Json
    Check ($restored.server_llm_enabled -is [bool] -and $restored.server_llm_enabled -eq $llm) 'JSON LLM boolean round trip'
}

foreach ($mask in 1..7) { Test-Modules -Mask $mask }
Test-Modules -Mask 4 -RejectEmptyFirst -DownloadAudio
Test-Modules -Mask 6 -Authenticate
$savedLlm = [pscustomobject]@{
    launch_mode = 'server'; setup_mode = 'advanced'; model = (Join-Path $testRoot 'missing old LLM.gguf')
    model_history = @('missing-a.gguf', 'missing-b.gguf'); device = 'Vulkan7'; context = '32000'; ubatch = '128'
    threads = '9'; config_path = 'missing-old.toml'; kv_preset = 'custom'; kv_type_k = 'q8_0'; kv_type_v = 'f16'
    configure_memory = $true; vram_budget = '22GiB'; vram_reserve = '1GiB'; expert_cache = '12GiB'; ram_budget = '50GiB'
    host_dma = $true; pager_ring = '512MiB'; pager_ring_slots = '4'; kv_overflow = $true; kv_overflow_vram_mb = '512'; kv_overflow_reserve_mb = '256'
    submit_mode = 'fixed'; submit_cap = '99'; configure_diagnostics = $true; pager_stats = $true; pager_profile = $true; stage_profile = $true; vram_profile = $true
    think_mode = 'think'; configure_thinking = $true; max_new = '777'; configure_sampling = $true; reasoning_effort = 'high'
    reasoning_probe_model = 'old.gguf'; reasoning_probe_length = 123; reasoning_probe_mtime_ticks = 456; reasoning_probe_efforts = @('low', 'high')
    temperature = '0.1'; top_k = '2'; top_p = '0.5'; seed = '123'; server_parallel = '4'; server_session_cache = $true
    session_cache_dir = 'old-kv'; session_idle_secs = '55'; session_cache_max = '12GiB'; session_cache_ttl_hours = '22'
    server_vision = $true; vision_projector = 'old-projector.gguf'; cpu_miss_enabled = $true; cpu_miss_max = '3'; cpu_miss_cores = '7'
    custom_sets = 'spec.mtp=true;paging.stats=true'; server_embedding = $true; embedding_model = $embeddingFile; embedding_idle_timeout = '17'
    mtp_enabled = $true; mtp_model = 'old-mtp.gguf'; dram_cache = '48GiB'; auto_profile = 'aggressive'; future_llm_setting = 'keep-me'
}
foreach ($mask in @(2, 4, 6)) { Test-Modules -Mask $mask -Saved $savedLlm }
# These states deliberately have no usable saved LLM. Only enabled auxiliary models matter.
foreach ($mask in @(2, 4, 6)) {
    $saved = [pscustomobject]@{ launch_mode = 'server'; server_llm_enabled = $false; device = 'Vulkan7'; setup_mode = 'aggressive'
        server_embedding = (($mask -band 2) -ne 0); embedding_model = $embeddingFile; embedding_idle_timeout = '17'
        server_transcription = (($mask -band 4) -ne 0); transcription_model = $audioFile; model = 'missing-old.gguf' }
    Test-Modules -Mask $mask -Saved $saved -Reuse
}
# Missing bool in legacy settings must continue to enable the LLM by default.
Test-Modules -Mask 1 -Saved ([pscustomobject]@{ launch_mode = 'server'; model = $llmFile; device = 'Vulkan1' }) -Reuse
# Saved string false must not become true through a PowerShell bool cast.
Test-Modules -Mask 4 -Saved ([pscustomobject]@{ launch_mode = 'server'; server_llm_enabled = 'false'; server_embedding = 'false'; server_transcription = $true; transcription_model = $audioFile }) -Reuse
$script:Saved = [pscustomobject]@{ launch_mode = 'server'; server_llm_enabled = $false; server_embedding = $true; embedding_model = 'missing-embedding.gguf'; model = $llmFile }
Check (-not (Test-SavedWizardModels)) 'A disabled usable LLM must not hide a missing enabled embedding model'
$script:Saved = $null
$script:answers = [Collections.Generic.Queue[string]]::new()
foreach ($answer in @('', '', '')) { $script:answers.Enqueue($answer) }
$defaults = Read-ServerModules -LlmEnabled $true -EmbeddingEnabled $false -TranscriptionEnabled $false
Check ($defaults.Llm -and -not $defaults.Embedding -and -not $defaults.Transcription) 'Legacy/default Enter responses keep LLM enabled'
Write-Output "PASS: $script:checks Windows module checks; no engine, GPU, network download or production state used."
