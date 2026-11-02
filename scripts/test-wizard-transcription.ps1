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
Read-TranscriptionSettings $transcription
Check (-not $transcription.Enabled -and $script:answers.Count -eq 0 -and $script:downloads -eq 0) 'Disabled flow'
$transcription.Enabled = $true
foreach ($answer in @('R', ('"' + $modelDir + '"'), '0', '4', '-1', '0', (Join-Path $WorkDir "upload space's"))) { $script:answers.Enqueue($answer) }
Read-TranscriptionSettings $transcription
Check ($transcription.Enabled -and $transcription.Model -eq $modelFile -and $transcription.Threads -eq '4' -and $transcription.IdleTimeout -eq '0' -and $script:answers.Count -eq 0) 'Interactive input and integer retry'
Check ($script:downloads -eq 0) 'R/local path must not download'
$script:answers.Enqueue('D'); Check ((Select-TranscriptionModelPath) -eq $modelFile -and $script:downloads -eq 1) 'Explicit D only'
# Evaluate only the existing command-assembly server branch, with all execution mocked out.
$serverBlock = @($ast.EndBlock.Statements | Where-Object { $_ -is [Management.Automation.Language.IfStatementAst] -and $_.Extent.Text.Contains('Add-TranscriptionArguments -Arguments') })
Check ($serverBlock.Count -eq 1) 'Server command integration'
$llmEnabled = $false; $serverAuth = $true; $serverSessionCache = $false; $serverVision = $false; $serverEmbedding = $false; $serverAddr = '127.0.0.1:8080'; $serverParallel = '1'
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
# Static cross-platform contracts only. Interpreter/backend coverage belongs to the shell tests.
# Normalize only the two intentional Bash/zsh syntax differences in these shared functions.
function Get-ShellFunction([string]$Source, [string]$Name) {
    $match = [regex]::Match($Source, ('(?ms)^' + [regex]::Escape($Name) + '\(\) \{\n.*?^\}'))
    Check $match.Success ("Missing shell function: $Name")
    return $match.Value
}
$linux = [IO.File]::ReadAllText((Join-Path $repoRoot 'Start-INFR-Wizard-Linux.sh')).Replace("`r`n", "`n")
$mac = [IO.File]::ReadAllText((Join-Path $repoRoot 'Start-INFR-Wizard-macOS.sh')).Replace("`r`n", "`n")
foreach ($name in @('configure_api_modules', 'configure_embedding', 'configure_transcription', 'add_transcription_arguments', 'show_transcription_api')) {
    $linuxFunction = Get-ShellFunction $linux $name
    $macFunction = (Get-ShellFunction $mac $name).Replace('${REPLY:l}', '${REPLY,,}')
    $macFunction = [regex]::Replace($macFunction, '(?m)^(\s*)ask_integer(?=\s)', '${1}integer')
    Check ($linuxFunction -ceq $macFunction) ("Shell shared behavior parity: $name")
}

# Match the complete Chinese/English literal, not just its English half. Pin counts to
# avoid silently losing coverage when strings move out of the expected functions.
foreach ($contract in @(
    @{ Function = 'Read-ServerModules'; Count = 4 }
    @{ Function = 'Read-TranscriptionSettings'; Count = 6 }
)) {
    $functionName = $contract.Function
    $definition = $ast.Find({ param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $functionName }, $true)
    Check ($null -ne $definition) ("Missing Windows function: $functionName")
    $literals = @($definition.FindAll({ param($n) $n -is [Management.Automation.Language.StringConstantExpressionAst] -and $n.Value.Contains('/ ') }, $true))
    Check ($literals.Count -eq $contract.Count) ("Public bilingual contract count: $functionName")
    foreach ($literal in $literals) {
        Check ($linux.Contains($literal.Value)) ("Linux bilingual parity: $($literal.Value)")
        Check ($mac.Contains($literal.Value)) ("macOS bilingual parity: $($literal.Value)")
    }
}
foreach ($phrase in @('The Embedding model and backend load on the first request;', 'LLM and Embedding share the selected device;')) {
    $literals = @($ast.FindAll({ param($n) $n -is [Management.Automation.Language.StringConstantExpressionAst] -and $n.Value.Contains($phrase) }, $true))
    Check ($literals.Count -eq 1) ("Public module explanation exists once: $phrase")
    Check ($linux.Contains($literals[0].Value) -and $mac.Contains($literals[0].Value)) ("Three-platform module explanation parity: $phrase")
}

# Verify the defaults in their actual declarations, including old-state LLM compatibility.
$expectedDefaults = [ordered]@{
    server_llm_enabled = $true; server_embedding = $false; server_transcription = $false
    embedding_model = ''; embedding_idle_timeout = '300'; transcription_model = ''
    transcription_threads = '4'; transcription_idle_timeout = '300'
}
$shells = [ordered]@{ Linux = $linux; macOS = $mac }
foreach ($key in $expectedDefaults.Keys) {
    $expected = $expectedDefaults[$key]
    $defaults = @($ast.FindAll({ param($n) $n -is [Management.Automation.Language.CommandAst] -and $n.GetCommandName() -eq 'Get-SavedValue' -and $n.CommandElements.Count -eq 3 -and $n.CommandElements[1].Value -eq $key }, $true))
    Check ($defaults.Count -gt 0) ("Windows saved default exists: $key")
    foreach ($default in $defaults) {
        $value = & ([scriptblock]::Create($default.CommandElements[2].Extent.Text))
        Check ($value -is $expected.GetType() -and $value -ceq $expected) ("Windows saved default: $key")
    }
    foreach ($platform in $shells.Keys) {
        $declaration = Get-ShellFunction $shells[$platform] 'defaults'
        $assignments = [regex]::Matches($declaration, ('\[' + [regex]::Escape($key) + '\]=(''''|[^\s)]+)'))
        Check ($assignments.Count -eq 1) ("$platform default declared once: $key")
        $value = $assignments[0].Groups[1].Value.Trim("'")
        Check ($value -ceq ([string]$expected).ToLowerInvariant()) ("$platform default: $key")
    }
}

# Check each common auxiliary bundle in its own catalog cases, so a matching URL
# elsewhere in a script cannot mask an incorrect ID, kind, path or file count.
$catalog = @(Get-RecommendedModelCatalog)
foreach ($spec in @(
    @{ Id = 'sensevoice-small-q8'; Kind = 'audio'; Primary = 'SenseVoice-Small-Q8_0/sense-voice-small-q8_0.gguf'
        Info = 'https://huggingface.co/lovemefan/sense-voice-gguf/blob/main/sense-voice-small-q8_0.gguf'
        Url = 'https://huggingface.co/lovemefan/sense-voice-gguf/resolve/main/sense-voice-small-q8_0.gguf?download=true' }
    @{ Id = 'qwen3-embedding-06b-q8'; Kind = 'embedding'; Primary = 'Qwen3-Embedding-0.6B-Q8_0/Qwen3-Embedding-0.6B-Q8_0.gguf'
        Info = 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/blob/main/Qwen3-Embedding-0.6B-Q8_0.gguf'
        Url = 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true' }
)) {
    $bundles = @($catalog | Where-Object Id -eq $spec.Id)
    Check ($bundles.Count -eq 1) ("Windows auxiliary catalog ID: $($spec.Id)")
    $bundle = $bundles[0]
    Check ($bundle.Kind -ceq $spec.Kind -and $bundle.Primary.Replace('\', '/') -ceq $spec.Primary -and $bundle.InfoUrl -ceq $spec.Info) ("Windows auxiliary catalog metadata: $($spec.Id)")
    Check ($bundle.Files.Count -eq 1 -and $bundle.Files[0].Url -ceq $spec.Url -and $bundle.Files[0].RelativePath.Replace('\', '/') -ceq $spec.Primary) ("Windows official single-GGUF download: $($spec.Id)")
    foreach ($platform in $shells.Keys) {
        $shell = $shells[$platform]
        Check ([regex]::IsMatch($shell, ('(?m)^CATALOG_IDS=\([^\n)]*\b' + [regex]::Escape($spec.Id) + '\b'))) ("$platform selectable catalog ID: $($spec.Id)")
        $casePattern = '(?ms)^\s*' + [regex]::Escape($spec.Id) + '\)\n(?<body>.*?)\s*;;'
        $metadata = [regex]::Match((Get-ShellFunction $shell 'catalog_load'), $casePattern)
        $file = [regex]::Match((Get-ShellFunction $shell 'catalog_file'), $casePattern)
        Check ($metadata.Success -and $file.Success) ("$platform catalog cases: $($spec.Id)")
        $metadataBody = $metadata.Groups['body'].Value
        Check ($metadataBody.Contains('CATALOG_KIND=' + $spec.Kind + ';') -and $metadataBody.Contains("CATALOG_PRIMARY='$($spec.Primary)'") -and $metadataBody.Contains("CATALOG_INFO='$($spec.Info)'") -and $metadataBody.Contains('CATALOG_COUNT=1')) ("$platform auxiliary catalog metadata: $($spec.Id)")
        Check ($file.Groups['body'].Value.Contains("DOWNLOAD_URL='$($spec.Url)'") -and $file.Groups['body'].Value.Contains('DOWNLOAD_RELATIVE=$CATALOG_PRIMARY')) ("$platform official download case: $($spec.Id)")
    }
}
foreach ($platform in $shells.Keys) {
    $shell = $shells[$platform]
    $build = Get-ShellFunction $shell 'build_command'
    $configure = Get-ShellFunction $shell 'configure'
    Check ($build.Contains('add_transcription_arguments || return 1')) ("$platform transcription argv integration")
    foreach ($name in @('configure_api_modules', 'configure_embedding', 'configure_transcription')) {
        Check ([regex]::IsMatch($configure, ('(?m)(?:^|;|\bthen)\s+' + $name + '(?=\s*(?:;|$))'))) ("$platform interactive integration: $name")
    }
}
Write-Output "PASS: $checks Windows transcription and cross-platform static checks (functions/AST only)"
