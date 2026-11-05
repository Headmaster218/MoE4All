[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [string]$WizardPath = ''
)

# Pure classic-CLI fixtures: no window assemblies, engine, model or network.
$ErrorActionPreference = 'Stop'
if (-not $WizardPath) { $WizardPath = Join-Path $PSScriptRoot 'infr-wizard.ps1' }
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
if (-not $WorkDir.StartsWith('D:\AIinfr\infr\local-work\tmp\moe4all\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Use local-work/tmp/moe4all/<task>.' }
$ancestor = $WorkDir
while ($ancestor) {
    if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Fixture paths cannot traverse reparse points.' }
    $ancestor = [IO.Path]::GetDirectoryName($ancestor)
}
[void][IO.Directory]::CreateDirectory($WorkDir)
$repoRoot = Join-Path $WorkDir ([guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($repoRoot)
$tokens = $null; $errors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($WizardPath, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
foreach ($fn in $ast.FindAll({param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst]}, $true)) { . ([scriptblock]::Create($fn.Extent.Text)) }
$script:checks = 0
function Check([bool]$Value, [string]$Message) { if (-not $Value) { throw $Message }; $script:checks++ }
function ArgsFor($Settings) { $list = [Collections.Generic.List[string]]::new(); Add-SpeechArguments -Arguments $list -Settings $Settings; return ,$list }
$script:Saved = $null
$defaults = Get-SpeechSettings
Check (-not $defaults.Enabled -and $defaults.Threads -eq '4' -and $defaults.IdleTimeout -eq '300' -and $defaults.MaxOutputSeconds -eq '30') 'Default speech settings changed'
Check ((ArgsFor $defaults).Count -eq 0) 'Disabled speech must not emit arguments'
$model = Join-Path $repoRoot "native voice's.gguf"; $assets = Join-Path $repoRoot "voice assets' [test]"
[void][IO.Directory]::CreateDirectory($assets)
[IO.File]::WriteAllText($model, 'not model weights')
foreach ($file in @('lexicon.txt', 'tokens.txt', 'config.json')) { [IO.File]::WriteAllText((Join-Path $assets $file), 'fixture') }
$script:Saved = [pscustomobject]@{server_speech=$true;speech_model=$model;speech_lexicon=$assets;speech_threads='7';speech_idle_timeout='0';speech_max_output_seconds='30.5'}
$settings = Get-SpeechSettings; $argv = ArgsFor $settings
$expected = @('--speech-model',$model,'--speech-lexicon',$assets,'--speech-threads','7','--speech-idle-timeout','0','--speech-max-output-seconds','30.5')
Check (($argv -join "`n") -ceq ($expected -join "`n")) 'Speech argv lost paths, settings or zero timeout'
foreach ($case in @(@('Threads','0'),@('Threads','257'),@('IdleTimeout','-1'),@('IdleTimeout','0.1'),@('MaxOutputSeconds','NaN'),@('MaxOutputSeconds','0'),@('MaxOutputSeconds','121'))) {
    $invalid = Get-SpeechSettings; $invalid.($case[0]) = $case[1]; $rejected = $false
    try { $null = ArgsFor $invalid } catch { $rejected = $true }
    Check $rejected ('Invalid speech setting accepted: ' + $case[0] + '=' + $case[1])
}
$missing = Get-SpeechSettings; $missing.Lexicon = Join-Path $repoRoot 'missing assets'; $rejected = $false
try { $null = ArgsFor $missing } catch { $rejected = $true }
Check $rejected 'Missing speech assets accepted'
$script:answers = [Collections.Generic.Queue[string]]::new()
foreach ($answer in @('n','n','n','y')) { $script:answers.Enqueue($answer) }
function Write-Host { param($Object, $ForegroundColor) }
function Read-Host { param($Prompt) if (-not $script:answers.Count) { throw 'Unexpected extra module prompt' }; $script:answers.Dequeue() }
$modules = Read-ServerModules -LlmEnabled $false -EmbeddingEnabled $false -TranscriptionEnabled $false -SpeechEnabled $true
Check ($modules.Speech -and -not $modules.Llm -and -not $modules.Embedding -and -not $modules.Transcription -and $script:answers.Count -eq 0) 'Speech-only API selection failed'
$script:Saved | Add-Member -NotePropertyName launch_mode -NotePropertyValue 'server'
$script:Saved | Add-Member -NotePropertyName server_llm_enabled -NotePropertyValue $false
$script:Saved | Add-Member -NotePropertyName server_embedding -NotePropertyValue $false
$script:Saved | Add-Member -NotePropertyName server_transcription -NotePropertyValue $false
Check (Test-SavedWizardModels) 'Usable speech-only saved settings rejected'
Microsoft.PowerShell.Utility\Write-Host "Speech classic CLI: $script:checks checks passed; no GUI, engine or GPU."
