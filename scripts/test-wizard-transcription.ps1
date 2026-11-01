param([Parameter(Mandatory = $true)][string]$WorkDir)
$ErrorActionPreference = 'Stop'
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
if (-not $WorkDir.StartsWith('D:\AIinfr\infr\local-work\tmp\moe4all\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Use local-work/tmp/moe4all/<task>/windows.' }
[void][IO.Directory]::CreateDirectory($WorkDir)
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$dataDir = Join-Path $WorkDir 'state with spaces'
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'infr-wizard.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
# Import functions only: never evaluate the wizard entry point, probes or launcher.
$names = @('Get-SavedValue', 'Get-RecommendedModelCatalog', 'Show-RecommendedModelDownloads', 'ConvertTo-FullPath',
    'Read-TextValue', 'Read-IntegerValue', 'Read-YesNo', 'Select-TranscriptionModelPath', 'Get-TranscriptionSettings',
    'Read-TranscriptionSettings', 'Add-TranscriptionArguments', 'Add-SetArgument', 'Format-PowerShellArgument')
foreach ($fn in $ast.FindAll({ param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    if ($fn.Name -in $names) { . ([scriptblock]::Create($fn.Extent.Text)) }
}
$checks = 0
function Check($Condition, [string]$Message) { if (-not $Condition) { throw $Message }; $script:checks++ }
function Reject([scriptblock]$Action) { $failed = $false; try { & $Action } catch { $failed = $true }; Check $failed 'Expected rejection' }
$script:Saved = $null
$transcription = Get-TranscriptionSettings
Check (-not $transcription.Enabled -and $transcription.Threads -eq '4' -and $transcription.IdleTimeout -eq '300') 'Defaults'
foreach ($savedBool in @($false, 'false', $null)) {
    $script:Saved = [pscustomobject]@{ server_transcription = $savedBool }
    Check (-not (Get-TranscriptionSettings).Enabled) 'False/null must not enable transcription'
}
$audio = @(Get-RecommendedModelCatalog | Where-Object Kind -eq audio)
Check ($audio.Count -eq 1 -and $audio[0].Primary -eq 'SenseVoice-Small-Q8_0\sense-voice-small-q8_0.gguf') 'Audio catalog isolation'
Check ($audio[0].Files.Count -eq 1 -and $audio[0].Files[0].Url -eq 'https://huggingface.co/lovemefan/sense-voice-gguf/resolve/main/sense-voice-small-q8_0.gguf?download=true') 'Official single GGUF download'
$modelDir = Join-Path $WorkDir "audio space's"
[void][IO.Directory]::CreateDirectory($modelDir)
$modelFile = Join-Path $modelDir 'voice.gguf'; [IO.File]::WriteAllText($modelFile, 'fixture')
$script:answers = [Collections.Generic.Queue[string]]::new()
function Read-Host { param($Prompt) if (-not $script:answers.Count) { throw 'Unexpected prompt' }; return $script:answers.Dequeue() }
$script:downloads = 0
function Invoke-RecommendedModelDownload { param($Id) Check ($Id -eq 'sensevoice-small-q8') 'Wrong download'; $script:downloads++; return $modelFile }
$script:Saved = $null; $transcription = Get-TranscriptionSettings
$script:answers.Enqueue(''); Read-TranscriptionSettings $transcription
Check (-not $transcription.Enabled -and $script:answers.Count -eq 0 -and $script:downloads -eq 0) 'Disabled flow'
foreach ($answer in @('y', 'R', ('"' + $modelDir + '"'), '0', '4', '-1', '0', (Join-Path $WorkDir "upload space's"))) { $script:answers.Enqueue($answer) }
Read-TranscriptionSettings $transcription
Check ($transcription.Enabled -and $transcription.Model -eq $modelFile -and $transcription.Threads -eq '4' -and $transcription.IdleTimeout -eq '0' -and $script:answers.Count -eq 0) 'Interactive input and integer retry'
Check ($script:downloads -eq 0) 'R/local path must not download'
$script:answers.Enqueue('D'); Check ((Select-TranscriptionModelPath) -eq $modelFile -and $script:downloads -eq 1) 'Explicit D only'
# Evaluate only the existing command-assembly server branch, with all execution mocked out.
$serverBlock = @($ast.EndBlock.Statements | Where-Object { $_ -is [Management.Automation.Language.IfStatementAst] -and $_.Extent.Text.Contains('Add-TranscriptionArguments -Arguments') })
Check ($serverBlock.Count -eq 1) 'Server command integration'
$serverAuth = $true; $serverSessionCache = $false; $serverVision = $false; $serverEmbedding = $false; $serverAddr = '127.0.0.1:8080'; $serverParallel = '1'
foreach ($launchMode in @('chat', 'benchmark', 'server')) {
    $nativeArgs = [Collections.Generic.List[string]]::new()
    . ([scriptblock]::Create($serverBlock[0].Extent.Text))
    Check (($nativeArgs.Contains('--transcription-model')) -eq ($launchMode -eq 'server')) 'Serve-only flags'
}
Check ($nativeArgs[$nativeArgs.IndexOf('--transcription-model') + 1] -eq $modelFile -and $nativeArgs[$nativeArgs.IndexOf('--transcription-dev') + 1] -eq 'cpu') 'CPU and model argv boundaries'
Check ($nativeArgs[$nativeArgs.IndexOf('--transcription-upload-dir') + 1] -eq $transcription.UploadDir) 'Upload argv boundaries'
$quoted = Format-PowerShellArgument $modelFile
Check ((& ([scriptblock]::Create($quoted))) -eq $modelFile) 'PowerShell quoting round trip'
$transcription.Enabled = $false; $nativeArgs = [Collections.Generic.List[string]]::new()
. ([scriptblock]::Create($serverBlock[0].Extent.Text))
Check (-not (($nativeArgs -join ' ') -match 'transcription')) 'Disabled command unchanged'
$transcription.Enabled = $true
foreach ($invalid in @('0', '-1', 'bad', '257', '1000000000')) {
    $transcription.Threads = $invalid; Reject { Add-TranscriptionArguments $nativeArgs $transcription }
}
$transcription.Threads = '4'; $transcription.IdleTimeout = '-1'; Reject { Add-TranscriptionArguments $nativeArgs $transcription }; $transcription.IdleTimeout = '300'
$transcription.Model = 'missing.gguf'; Reject { Add-TranscriptionArguments $nativeArgs $transcription }; $transcription.Model = $modelFile
# Round-trip just the real state fields, without executing unrelated wizard state code.
$stateNode = $ast.Find({ param($n) $n -is [Management.Automation.Language.AssignmentStatementAst] -and $n.Left.Extent.Text -eq '$state' -and $n.Right.Extent.Text.Contains('transcription_threads') }, $true)
$table = $stateNode.Find({ param($n) $n -is [Management.Automation.Language.HashtableAst] }, $true)
$savedFields = @{}
foreach ($pair in $table.KeyValuePairs) { if ($pair.Item1.Value -match '^(server_transcription|transcription_)') { $savedFields[$pair.Item1.Value] = & ([scriptblock]::Create($pair.Item2.Extent.Text)) } }
$script:Saved = $savedFields | ConvertTo-Json | ConvertFrom-Json
$restored = Get-TranscriptionSettings
Check ($savedFields.Count -eq 5 -and $restored.Enabled -and $restored.Model -eq $modelFile -and $restored.Threads -eq '4' -and $restored.IdleTimeout -eq '300' -and $restored.UploadDir -eq $transcription.UploadDir) 'JSON history round trip'
$promptNode = $ast.Find({ param($n) $n -is [Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq 'Read-TranscriptionSettings' }, $true)
$guarded = $false
for ($parent = $promptNode.Parent; $null -ne $parent; $parent = $parent.Parent) {
    if ($parent -is [Management.Automation.Language.IfStatementAst] -and $parent.Clauses[0].Item1.Extent.Text -eq "`$launchMode -eq 'server'") { $guarded = $true; break }
}
Check $guarded 'New prompt is guarded by server mode'
# Static macOS review: shared behavior and bilingual strings must stay aligned.
# Execute test-wizard-transcription.sh with zsh on a host with zsh for interpreter coverage.
$linux = [IO.File]::ReadAllText((Join-Path $repoRoot 'Start-INFR-Wizard-Linux.sh')).Replace("`r`n", "`n")
$mac = [IO.File]::ReadAllText((Join-Path $repoRoot 'Start-INFR-Wizard-macOS.sh')).Replace("`r`n", "`n")
foreach ($name in @('configure_transcription', 'add_transcription_arguments', 'show_transcription_api')) {
    $pattern = '(?ms)^' + $name + '\(\) \{\n.*?^\}'
    $linuxFunction = [regex]::Match($linux, $pattern).Value
    $macFunction = [regex]::Match($mac, $pattern).Value.Replace('${REPLY:l}', '${REPLY,,}')
    Check ($linuxFunction -and $linuxFunction -eq $macFunction) ('Shell behavior parity: ' + $name)
}
$settingsFunction = $ast.Find({ param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq 'Read-TranscriptionSettings' }, $true)
foreach ($literal in $settingsFunction.FindAll({ param($n) $n -is [Management.Automation.Language.StringConstantExpressionAst] -and $n.Value.Contains(' / ') }, $true)) {
    Check ($linux.Contains($literal.Value) -and $mac.Contains($literal.Value)) 'Three-platform bilingual text parity'
}
foreach ($shell in @($linux, $mac)) {
    Check ($shell.Contains('[server_transcription]=false [transcription_model]='''' [transcription_threads]=4 [transcription_idle_timeout]=300')) 'Shell defaults'
    Check ($shell.Contains($audio[0].Files[0].Url) -and $shell.Contains($audio[0].Primary.Replace('\', '/'))) 'Shell official catalog'
    Check ($shell.Contains('add_transcription_arguments || return 1') -and $shell.Contains('        configure_transcription')) 'Shell integration'
}
Write-Output "PASS: $checks Windows transcription and cross-platform static checks (functions/AST only)"
