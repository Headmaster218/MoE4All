[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$WorkDir)

$ErrorActionPreference = 'Stop'
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
$workRoot = if ($env:MOE4ALL_WORK_ROOT) { $env:MOE4ALL_WORK_ROOT } else { 'D:\AIinfr\infr\local-work' }
$allowedRoot = [IO.Path]::GetFullPath((Join-Path $workRoot 'tmp\moe4all')).TrimEnd('\') + '\'
if (-not $WorkDir.StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Use local-work/tmp/moe4all/<task>/tts.' }
$ancestor = $WorkDir
while ($ancestor) {
    if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Test outputs cannot use reparse points.' }
    $ancestor = [IO.Path]::GetDirectoryName($ancestor)
}
[void][IO.Directory]::CreateDirectory($WorkDir)
$fixture = Join-Path $WorkDir ('speech-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($fixture)
$source = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'infr-wizard.ps1'))
$tokens = $null; $parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseInput($source, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
# Import definitions only: no wizard entry point, engine, network or user state.
foreach ($fn in $ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    . ([scriptblock]::Create($fn.Extent.Text))
}
. (Join-Path $PSScriptRoot 'infr-dashboard-state.ps1')
# Import just path resolution and draft persistence; no UI initialization.
$dashboardTokens = $null; $dashboardErrors = $null
$dashboardAst = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'infr-dashboard.ps1'), [ref]$dashboardTokens, [ref]$dashboardErrors)
if ($dashboardErrors.Count) { throw ($dashboardErrors | Out-String) }
foreach ($fn in $dashboardAst.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -in @('Resolve-DashboardLaunchPaths', 'Resolve-DashboardModelFile', 'Save-DashboardDraft', 'Invoke-DashboardHelper', 'Test-WizardPromptCancellation', 'Update-DashboardModelMetadata') }, $true)) {
    . ([scriptblock]::Create($fn.Extent.Text))
}
$script:checks = 0
function Check([bool]$Condition, [string]$Message) { if (-not $Condition) { throw $Message }; $script:checks++ }
function Json($Value) { ConvertTo-Json -InputObject $Value -Depth 20 -Compress }
function Read-Host {
    param($Prompt, [switch]$AsSecureString)
    [void]$script:prompts.Add([string]$Prompt)
    if (-not $script:answers.Count) { throw "Unexpected prompt: $Prompt" }
    $answer = $script:answers.Dequeue()
    if ($AsSecureString) { return ConvertTo-SecureString $answer -AsPlainText -Force }
    return $answer
}
function Write-Host { param($Object, $ForegroundColor) }
function Write-Warning { param($Message) }
function Invoke-BenchmarkStartupGuide { param($Interactive, $JsonAutomation) return $false }
function Test-BenchmarkGuideInteractive { return $false }
function Start-VulkanDeviceProbe { throw 'GPU/device probe is forbidden in speech tests.' }
function Start-ModelReasoningProbe { param($ModelPath) throw 'LLM probe is forbidden in speech tests.' }
function Get-ModelReasoningEfforts { param($ModelPath) throw 'LLM metadata probe is forbidden in speech tests.' }
function Invoke-NativeCommandResult { throw 'Engine execution is forbidden in speech tests.' }
function Invoke-RecommendedModelDownload { param($Id) throw 'Downloads are forbidden in speech tests.' }
function Select-RecommendedDownloads { param($Kind) Check ($Kind -eq 'speech') 'Speech setup cannot route to unrelated models'; $script:guideRequests++; return @() }
function Clear-Host { }

$speechFile = Join-Path $fixture "speech voice's [test].gguf"
$llmFile = Join-Path $fixture 'chat.gguf'
$embeddingFile = Join-Path $fixture 'embedding.gguf'
$audioFile = Join-Path $fixture 'transcription.gguf'
$projectorFile = Join-Path $fixture 'mmproj.gguf'
$wrongFile = Join-Path $fixture 'wrong.bin'
foreach ($file in @($speechFile, $llmFile, $embeddingFile, $audioFile, $projectorFile, $wrongFile)) { [IO.File]::WriteAllText($file, 'fixture, not model weights') }
$assets = Join-Path $fixture "voice assets' [test]"
[void][IO.Directory]::CreateDirectory($assets)
foreach ($name in @('lexicon.txt', 'tokens.txt', 'config.json')) { [IO.File]::WriteAllText((Join-Path $assets $name), 'fixture') }
function New-SpeechState($Changes = @{}) {
    $values = @{ launch_mode = 'server'; server_llm_enabled = $false; server_speech = $true; speech_model = $speechFile; speech_lexicon = $assets }
    foreach ($key in $Changes.Keys) { $values[$key] = $Changes[$key] }
    Get-DashboardInitialState -Saved $values
}
function Assert-Errors($State, [string[]]$Expected = @()) {
    $actual = @(@(Test-DashboardState -State $State).field | Sort-Object -Unique)
    $wanted = @($Expected | Sort-Object -Unique)
    Check (($actual -join '/') -ceq ($wanted -join '/')) ("Expected errors [$($wanted -join ',')] but got [$($actual -join ',')]")
}
function New-SpeechSettings($Changes = @{}) {
    $values = @{ Enabled = $true; Model = $speechFile; Lexicon = $assets; Threads = '4'; IdleTimeout = '300'; MaxOutputSeconds = '30' }
    foreach ($key in $Changes.Keys) { $values[$key] = $Changes[$key] }
    [pscustomobject]$values
}

$fresh = Get-DashboardInitialState
Check ($fresh.server_speech -is [bool] -and -not $fresh.server_speech) 'TTS is opt-in for legacy state'
Check ($fresh.speech_threads -eq '4' -and $fresh.speech_idle_timeout -eq '300' -and $fresh.speech_max_output_seconds -eq '30') 'Defaults match ServeOptions'
foreach ($value in @('false', '0', 'off')) { Check (-not (Get-DashboardInitialState -Saved @{server_speech = $value}).server_speech) 'Saved false is not truthy' }
$speechState = New-SpeechState
$metadataState = New-SpeechState @{model=$llmFile;reasoning_effort='high'}
$metadataBefore = Json $metadataState
Update-DashboardModelMetadata $metadataState
Check ((Json $metadataState) -ceq $metadataBefore) 'Speech-only dashboard never probes or changes a saved inactive LLM'
$before = Json $speechState
$schema = Get-DashboardSchema -State $speechState
Check ((Json $speechState) -ceq $before) 'Schema does not mutate input'
$group = @($schema.groups | Where-Object { $_.id -eq 'speech' })
Check ($group.Count -eq 1 -and $group[0].summary_key -eq 'server_speech' -and -not $group[0].open) 'TTS has a folded independent module group'
$fields = @($schema.fields | Where-Object { $_.group -eq 'speech' })
Check (($fields.key -join '/') -eq 'server_speech/speech_model/speech_lexicon/speech_threads/speech_idle_timeout/speech_max_output_seconds') 'Exact native UI field contract'
foreach ($field in $fields) {
    Check ($field.when._api -eq 'true') ('API-only field: ' + $field.key)
    if ($field.key -ne 'server_speech') { Check ($field.when.server_speech -eq 'true') ('TTS-enabled field: ' + $field.key) }
}
Check (($fields.help -join ' ') -match 'TTS-enabled build') 'Explain the optional engine feature'
Check (($fields.help -join ' ') -match 'CPU') 'Explain actual CPU implementation'
Check (@($schema.catalog | Where-Object { $_.Kind -eq 'speech' -and $_.Downloadable -ne $false }).Count -eq 0) 'Do not invent a speech download bundle'
Assert-Errors $speechState
Assert-Errors (New-SpeechState @{model='missing-unused';device="`nbad";setup_mode='unused';context='unused';server_parallel='unused';server_session_cache='unused'})
Assert-Errors (New-SpeechState @{server_speech='invalid'}) @('server_speech','server_llm_enabled')
Assert-Errors (New-SpeechState @{speech_model=''}) @('speech_model')
Assert-Errors (New-SpeechState @{speech_model=$wrongFile}) @('speech_model')
Assert-Errors (New-SpeechState @{speech_model=$assets}) @('speech_model')
Assert-Errors (New-SpeechState @{speech_model="bad`0.gguf"}) @('speech_model')
Assert-Errors (New-SpeechState @{speech_lexicon=$speechFile}) @('speech_lexicon')
Assert-Errors (New-SpeechState @{speech_lexicon=''}) @('speech_lexicon')
foreach ($change in @(@{Model=''}, @{Model=$wrongFile}, @{Model=$assets}, @{Model="bad`0.gguf"}, @{Lexicon=$speechFile}, @{Lexicon="bad`nassets"})) {
    $rejected=$false; $arguments=[Collections.Generic.List[string]]::new()
    try { Add-SpeechArguments -Arguments $arguments -Settings (New-SpeechSettings $change) } catch { $rejected=$true }
    Check ($rejected -and $arguments.Count -eq 0) 'Invalid speech paths never append partial argv'
}
foreach ($target in @('speech_model', 'speech_lexicon')) {
    $script:Saved=$null
    $helperState=New-SpeechState
    $helperBefore=Json $helperState
    $script:guideRequests=0; $script:answers=[Collections.Generic.Queue[string]]::new();$script:answers.Enqueue('');$script:prompts=[Collections.Generic.List[string]]::new()
    Invoke-DashboardHelper -Action 'download' -Target $target -State $helperState
    Check ($script:guideRequests -eq 1) 'Targeted TTS setup opens only the speech guide'
    Check ((Json $helperState) -ceq $helperBefore -and $null -eq $script:Saved) 'Speech guide preserves settings'
}
foreach ($missing in @('lexicon.txt', 'tokens.txt', 'config.json')) {
    $incomplete = Join-Path $fixture ('missing-' + $missing)
    [void][IO.Directory]::CreateDirectory($incomplete)
    foreach ($name in @('lexicon.txt', 'tokens.txt', 'config.json')) { if ($name -ne $missing) { [IO.File]::WriteAllText((Join-Path $incomplete $name), 'fixture') } }
    Assert-Errors (New-SpeechState @{speech_lexicon=$incomplete}) @('speech_lexicon')
    $rejected = $false
    try { Add-SpeechArguments -Arguments ([Collections.Generic.List[string]]::new()) -Settings (New-SpeechSettings @{Lexicon=$incomplete}) } catch { $rejected=$true }
    Check $rejected ('Argv rejects missing ' + $missing)
}
foreach ($entry in @(
    @{field='speech_threads';setting='Threads';values=@('0','257','-1','1.5','four')},
    @{field='speech_idle_timeout';setting='IdleTimeout';values=@('-1','1.5','18446744073709551616','none')},
    @{field='speech_max_output_seconds';setting='MaxOutputSeconds';values=@('0','0.99','120.1','NaN','Infinity','1e100','1,5','')}
)) {
    foreach ($value in $entry.values) {
        $change = @{}; $change[$entry.field] = $value
        Assert-Errors (New-SpeechState $change) @($entry.field)
        $settingsChange = @{}; $settingsChange[$entry.setting] = $value
        $arguments = [Collections.Generic.List[string]]::new(); [void]$arguments.Add('serve')
        $rejected=$false
        try { Add-SpeechArguments -Arguments $arguments -Settings (New-SpeechSettings $settingsChange) } catch { $rejected=$true }
        Check ($rejected -and $arguments.Count -eq 1) ('Invalid limits cannot partially append argv: ' + $entry.field + '=' + $value)
    }
}
foreach ($seconds in @('1','30','30.5','120')) {
    Assert-Errors (New-SpeechState @{speech_threads='256';speech_idle_timeout='18446744073709551615';speech_max_output_seconds=$seconds})
    $arguments = [Collections.Generic.List[string]]::new()
    Add-SpeechArguments -Arguments $arguments -Settings (New-SpeechSettings @{Threads='256';IdleTimeout='18446744073709551615';MaxOutputSeconds=$seconds})
    Check ($arguments.Count -eq 10 -and $arguments[1] -eq $speechFile -and $arguments[3] -eq $assets) 'Paths with spaces, apostrophes and brackets remain single arguments'
    Check ($arguments[9] -eq $seconds) 'Duration is serialized invariantly'
}
$cultureBefore = [Threading.Thread]::CurrentThread.CurrentCulture
try {
    [Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::GetCultureInfo('de-DE')
    Assert-Errors (New-SpeechState @{speech_idle_timeout='0';speech_max_output_seconds='1.5'})
    $arguments=[Collections.Generic.List[string]]::new()
    Add-SpeechArguments -Arguments $arguments -Settings (New-SpeechSettings @{IdleTimeout='0';MaxOutputSeconds='1.5'})
    Check ($arguments[9] -eq '1.5' -and $arguments[7] -eq '0') 'Locale cannot turn a decimal duration into a comma'
} finally { [Threading.Thread]::CurrentThread.CurrentCulture = $cultureBefore }
foreach ($mode in @('chat','benchmark')) {
    Assert-Errors (New-SpeechState @{launch_mode=$mode;model=$llmFile;speech_model='missing';speech_lexicon='missing';speech_threads='bad';speech_idle_timeout='bad';speech_max_output_seconds='bad'})
}
Assert-Errors (New-SpeechState @{server_speech=$false;server_embedding=$true;embedding_model=$embeddingFile;speech_model='missing';speech_lexicon='missing';speech_threads='bad'})
$arguments=[Collections.Generic.List[string]]::new()
Add-SpeechArguments -Arguments $arguments -Settings (New-SpeechSettings @{Enabled=$false;Model='missing';Lexicon='missing';Threads='invalid'})
Check ($arguments.Count -eq 0) 'Disabled speech contributes no argv'
$repoRoot=$fixture
$relativeState=New-SpeechState @{speech_model=[IO.Path]::GetFileName($speechFile);speech_lexicon=[IO.Path]::GetFileName($assets);model='missing-unused'}
Resolve-DashboardLaunchPaths $relativeState
Check ($relativeState.speech_model -eq $speechFile -and $relativeState.speech_lexicon -eq $assets -and $relativeState.model -eq 'missing-unused') 'Resolve active speech paths without touching inactive LLM'
$disabledState=New-SpeechState @{server_speech=$false;speech_model='missing-unused';speech_lexicon='missing-unused'}
Resolve-DashboardLaunchPaths $disabledState
Check ($disabledState.speech_model -eq 'missing-unused' -and $disabledState.speech_lexicon -eq 'missing-unused') 'Disabled speech paths remain untouched'
$dataDir=Join-Path $fixture 'draft-state'; $statePath=Join-Path $dataDir 'wizard-state.json'
Save-DashboardDraft $speechState
$restored=Get-DashboardInitialState -Saved ([IO.File]::ReadAllText($statePath) | ConvertFrom-Json)
Check ($restored.server_speech -is [bool] -and $restored.server_speech -and $restored.speech_model -eq $speechFile -and $restored.speech_lexicon -eq $assets) 'Draft round trip retains TTS state'

# Exercise the actual shared configuration/argv/save block, ending before persistence or execution.
$start=$source.IndexOf('$reuseSavedSettings = $false')
$endMarker='if (-not $llmEnabled) { Restore-DisabledLlmState -State $state }'
$end=$source.IndexOf($endMarker)
if ($start -lt 0 -or $end -le $start) { throw 'Cannot locate bounded configuration block.' }
$flow=[scriptblock]::Create($source.Substring($start,$end+$endMarker.Length-$start))
function Test-SpeechFlow([int]$Mask, [switch]$Interactive) {
    $llm=($Mask -band 1) -ne 0; $embedding=($Mask -band 2) -ne 0; $audio=($Mask -band 4) -ne 0
    $script:Saved=Get-DashboardInitialState -Saved @{
        launch_mode='server';server_llm_enabled=$llm;server_embedding=$embedding;server_transcription=$audio
        server_speech=$true;speech_model=$speechFile;speech_lexicon=$assets;speech_threads='7';speech_idle_timeout='0';speech_max_output_seconds='12.5'
        model=$llmFile;device='Vulkan7';embedding_model=$embeddingFile;transcription_model=$audioFile;server_vision=$llm;vision_projector=$projectorFile
    }
    $script:Saved|Add-Member -NotePropertyName future_llm_setting -NotePropertyValue 'preserve-me'
    $script:answers=[Collections.Generic.Queue[string]]::new(); $script:prompts=[Collections.Generic.List[string]]::new()
    $script:DashboardAccepted=$false; $script:ModelReasoningProbe=$null
    $script:BenchmarkUploadDefault=$false; $script:BenchmarkUploadHasGithubDefault=$true; $script:BenchmarkUploadGhAuthDefault=$true
    $script:BenchmarkUploadFallbackDefault='folder'; $script:SubmissionBrowserActionDefault='later'; $script:SubmissionFailureActionDefault='later'
    $script:SubmissionPendingRetryDefault=$true; $script:ErrorUploadDefault=$false
    $repoRoot=$fixture; $dataDir=Join-Path $fixture 'flow-state'; $guiStatePath=Join-Path $fixture 'no-gui.json'
    $infrPath=Join-Path $fixture 'never-execute.exe'; $InitialModelPath=''; $DryRun=$true
    if ($Interactive) {
        Check ($Mask -eq 0) 'Interactive fixture is independent speech'
        foreach ($answer in @('n','2','n','n','n','y','','','','','','','n')) { $script:answers.Enqueue($answer) }
    } else { $script:answers.Enqueue('') }
    . $flow
    Check ($script:answers.Count -eq 0) ('No unused answers for mask ' + $Mask)
    Check ($nativeArgs[0] -eq 'serve') 'Speech uses shared serve'
    Check ($nativeArgs[$nativeArgs.IndexOf('--speech-model')+1] -eq $speechFile) 'Full flow speech model argument'
    Check ($nativeArgs[$nativeArgs.IndexOf('--speech-lexicon')+1] -eq $assets) 'Full flow assets argument'
    Check ($nativeArgs[$nativeArgs.IndexOf('--speech-threads')+1] -eq '7' -and $nativeArgs[$nativeArgs.IndexOf('--speech-max-output-seconds')+1] -eq '12.5') 'Full flow retains configured limits'
    Check ($nativeArgs.Contains('--embedding-model') -eq $embedding -and $nativeArgs.Contains('--transcription-model') -eq $audio) 'Speech composes with auxiliary modules'
    Check ($nativeArgs.Contains($llmFile) -eq $llm -and $nativeArgs.Contains('--mmproj') -eq $llm -and $nativeArgs.Contains('--parallel') -eq $llm) 'LLM, vision and concurrency gated independently'
    if (-not $llm) { Check ($state.future_llm_setting -eq 'preserve-me') 'Preserve disabled LLM state' }
    if (-not ($llm -or $embedding)) { Check ($nativeArgs[$nativeArgs.IndexOf('--dev')+1] -eq 'cpu' -and $state.device -eq 'Vulkan7') 'Pure CPU speech does not overwrite saved GPU selection' }
    Check ($state.server_speech -is [bool] -and $state.server_speech -and $state.speech_lexicon -eq $assets -and $state.speech_max_output_seconds -eq '12.5') 'State assembly persists speech'
    $viewState=Get-DashboardInitialState -Saved ([pscustomobject]$state)
    $viewSchema=Get-DashboardSchema -State $viewState
    Check ($viewState.server_speech -and @($viewSchema.fields | Where-Object {$_.key -eq 'server_speech'}).Count -eq 1) 'Current dashboard includes the enabled CPU speech module'
    Check ($viewState.speech_model -eq $speechFile -and @($viewSchema.fields | Where-Object {$_.key -eq 'speech_model'}).Count -eq 1) 'Current dashboard retains the speech model'
    $plan=New-WizardExecutionPlan
    $plan=Json $plan | ConvertFrom-Json
    Check ($plan.Variables.speech.Enabled -is [bool] -and $plan.Variables.speech.Enabled) 'Execution plan round trip includes speech flag'
    Check (@($plan.Variables.speech.PSObject.Properties).Count -eq 1) 'Execution metadata contains only Enabled'
    Check (($plan.Variables.nativeArgs -join [char]0) -ceq ($nativeArgs -join [char]0)) 'Execution plan preserves speech argv boundaries'
    Assert-Errors ([pscustomobject]$state)
}
foreach ($mask in 0..7) { Test-SpeechFlow $mask }
Test-SpeechFlow 0 -Interactive
$script:Saved=New-SpeechState @{server_speech='false'}
Check (-not (Get-SpeechSettings).Enabled) 'Legacy saved string false stays disabled'
$script:Saved=New-SpeechState
Check (Test-SavedWizardModels) 'Speech-only saved models can be reused without an LLM'
$script:Saved.speech_lexicon=$speechFile
Check (-not (Test-SavedWizardModels)) 'Saved reuse rejects assets that are not a directory'
$script:answers=[Collections.Generic.Queue[string]]::new(); $script:prompts=[Collections.Generic.List[string]]::new()
foreach ($answer in @('n','n','n','n','n','n','n','y')) { $script:answers.Enqueue($answer) }
$selected=Read-ServerModules -LlmEnabled $false -EmbeddingEnabled $false -TranscriptionEnabled $false -SpeechEnabled $false
Check ($selected.Speech -and -not $selected.Llm -and -not $selected.Embedding -and -not $selected.Transcription -and $script:prompts.Count -eq 8) 'All-off selection retries and permits speech alone'
Write-Output ("PASS: {0} TTS wizard checks; no engine, GPU, downloads, services or production state used." -f $script:checks)
