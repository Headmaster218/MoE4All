[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$WorkDir)

$ErrorActionPreference = 'Stop'
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
$workRoot = if ($env:MOE4ALL_WORK_ROOT) { $env:MOE4ALL_WORK_ROOT } else { 'D:\AIinfr\infr\local-work' }
$allowedRoot = [IO.Path]::GetFullPath((Join-Path $workRoot 'tmp\moe4all')).TrimEnd('\') + '\'
if (-not $WorkDir.StartsWith($allowedRoot, [StringComparison]::OrdinalIgnoreCase)) { throw 'Use local-work/tmp/moe4all/<task>/model-catalog.' }
$ancestor = $WorkDir
while ($ancestor) {
    if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Test output paths cannot use reparse points.' }
    $ancestor = [IO.Path]::GetDirectoryName($ancestor)
}
[void][IO.Directory]::CreateDirectory($WorkDir)
$repoRoot = Join-Path $WorkDir ('catalog-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($repoRoot)
$tokens = $null; $parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'infr-wizard.ps1'), [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
$names = @('Get-RecommendedModelCatalog', 'Show-RecommendedModelDownloads', 'Select-RecommendedDownloads', 'Format-DownloadBytes')
foreach ($name in $names) {
    $definition = @($ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true))
    if ($definition.Count -ne 1) { throw ('Expected one definition: ' + $name) }
    . ([scriptblock]::Create($definition[0].Extent.Text))
}
# Only these three pure/catalog functions are imported. HTTP and the real
# downloader/launcher entry point are never imported or invoked by this test.
$script:productVersion='1.3.0'
$script:checks = 0
function Check([bool]$Condition, [string]$Message) { if (-not $Condition) { throw $Message }; $script:checks++ }
function Write-Host { param($Object, $ForegroundColor) [void]$script:output.Add([string]$Object) }
function Read-Host {
    param($Prompt)
    $script:promptCount++
    if (-not $script:answers.Count) { throw ('Unexpected prompt: ' + $Prompt) }
    $script:answers.Dequeue()
}
function Read-YesNo {
    param($Label, $Default)
    $script:consentCount++
    return $script:consent
}
function Invoke-RecommendedModelDownload {
    param($Id)
    $bundle = @(Get-RecommendedModelCatalog | Where-Object { $_.Id -eq $Id })
    Check ($bundle.Count -eq 1) 'Fake downloader receives a known catalog ID'
    Check (-not $bundle[0].PSObject.Properties['Downloadable'] -or $bundle[0].Downloadable) 'Unavailable TTS can never reach the downloader'
    Check (@($bundle[0].Files).Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($bundle[0].Primary)) 'Only complete downloadable bundles reach fake HTTP'
    [void]$script:requests.Add($Id)
    return (Join-Path (Join-Path $repoRoot 'models') $bundle[0].Primary)
}
function Invoke-WebRequest { throw 'Real HTTP is forbidden in catalog tests.' }
function Invoke-RestMethod { throw 'Real HTTP is forbidden in catalog tests.' }
function Get-ModelDownloadDirectory { Join-Path $repoRoot 'models' }
function Open-ModelSetupGuide { param($Url) Check ($Url -eq 'https://github.com/Headmaster218/MoE4All/releases/tag/release-1.3.0') 'Guide uses current engine version'; $script:guideCount++ }
function Reset-Fixture([string[]]$Answers = @(), [bool]$Consent = $true) {
    $script:output = [Collections.Generic.List[string]]::new()
    $script:requests = [Collections.Generic.List[string]]::new()
    $script:answers = [Collections.Generic.Queue[string]]::new()
    foreach ($answer in $Answers) { $script:answers.Enqueue($answer) }
    $script:promptCount = 0; $script:guideCount = 0; $script:consentCount = 0; $script:consent = $Consent
}

Reset-Fixture
$catalog = @(Get-RecommendedModelCatalog)
Check ($catalog.Count -eq 7) 'Six downloads including the 35B projector plus the unavailable TTS entry'
Check (@($catalog.Id | Sort-Object -Unique).Count -eq $catalog.Count) 'Catalog IDs are unique'
$available = @($catalog | Where-Object { -not $_.PSObject.Properties['Downloadable'] -or $_.Downloadable })
Check ($available.Count -eq 6) 'Existing downloads and the 35B projector remain selectable'
foreach ($bundle in $available) {
    Check ($bundle.Kind -in @('llm', 'vision', 'embedding', 'audio')) 'No unverified speech weights are offered'
    Check (-not [IO.Path]::IsPathRooted($bundle.Primary) -and $bundle.Primary -notmatch '(^|[\\/])\.\.([\\/]|$)') 'Primary path stays package relative'
    Check ($bundle.Primary -in @($bundle.Files.RelativePath)) 'Primary is actually supplied by the bundle'
    foreach ($file in $bundle.Files) {
        $url = $null
        Check ([Uri]::TryCreate($file.Url, [UriKind]::Absolute, [ref]$url) -and $url.Scheme -eq 'https') 'Download source uses HTTPS'
        Check ([IO.Path]::GetExtension($file.RelativePath) -ieq '.gguf') 'Only existing native GGUF runtime downloads'
        Check ($file.Bytes -gt 0) 'Every downloadable file displays its actual byte size'
        Check (-not [IO.Path]::IsPathRooted($file.RelativePath) -and $file.RelativePath -notmatch '(^|[\\/])\.\.([\\/]|$)') 'File path stays package relative'
    }
}
$flash = @($catalog | Where-Object { $_.Id -eq 'flashnext-427' })[0]
Check ($flash.Files.Count -eq 3 -and ($flash.Files | Measure-Object Bytes -Sum).Sum -eq 94525391424) 'Use upstream unchanged weights re-split into three files'
Check (@($flash.Files | Where-Object { $_.Url -notmatch '/resolve/f5b0b790eebb89033c48091c26abf74c1b385940/Qwen3.8[^/]+of-00003.gguf' }).Count -eq 0) 'Deleted upstream shards are replaced by verified pinned download URLs'
$audio = @($catalog | Where-Object { $_.Kind -eq 'audio' })
Check ($audio.Count -eq 1 -and $audio[0].Id -eq 'sensevoice-small-q8') 'Existing native ASR bundle remains available without duplication'
Check ($audio[0].Files[0].Url -eq 'https://huggingface.co/lovemefan/sense-voice-gguf/resolve/main/sense-voice-small-q8_0.gguf?download=true') 'ASR keeps the verified SenseVoice source'
$speech = @($catalog | Where-Object { $_.Kind -eq 'speech' })
Check ($speech.Count -eq 1 -and $speech[0].Id -eq 'melotts-chinese-native-f32') 'Stable TTS descriptor for the downloads UI'
Check ($speech[0].Downloadable -is [bool] -and -not $speech[0].Downloadable) 'Unavailable is an explicit boolean'
Check (@($speech[0].Files).Count -eq 0 -and [string]::IsNullOrEmpty($speech[0].Primary)) 'No incompatible or invented download URL'
Check ($speech[0].UnavailableReason -match 'release notes' -and $speech[0].InstructionsUrl -eq 'https://github.com/Headmaster218/MoE4All/releases/tag/release-1.3.0') 'Distinguish verified native package from pending public hosting'
Check ($speech[0].LocalPackageReady -is [bool] -and $speech[0].LocalPackageReady -and $speech[0].PackageScript -eq 'scripts/package-native-speech.ps1') 'Catalog points to reusable native packaging'
Check ($speech[0].ModelSha256 -eq '6d0d6fffe43cf6dea169f584c07bbb1a43756f37958ba9f9b78e27e8f9b32cc2' -and $speech[0].ModelBytes -eq 164882176) 'Catalog identifies the previously successful native artifact'
Check (($speech[0].RuntimeAssets -join '/') -eq 'melotts-chinese-f32.gguf/lexicon.txt/tokens.txt/config.json/frontend.json/custom-lexicon.txt') 'Preserve the complete successful voice frontend'
Check (($speech[0].RequiredAssets -join '/') -eq 'lexicon.txt/tokens.txt/config.json') 'Matching resource requirements are explicit'
Check ($speech[0].ConversionScript -eq 'scripts/convert_melotts_chinese.py') 'Points to the established offline conversion workflow'
Check ($speech[0].SourceUrls[0] -match '/af5d207a364ea4208c6f589c89f57f88414bdd16$' -and $speech[0].SourceUrls[1] -match '/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6$') 'Conversion origins are revision pinned'
$script:productVersion='1.4.0-beta.1'
Check ((@(Get-RecommendedModelCatalog | Where-Object Kind -eq speech)[0]).InstructionsUrl -eq 'https://github.com/Headmaster218/MoE4All/releases/tag/release-1.4.0-beta.1') 'Speech instructions follow the installed prerelease version'
$script:productVersion='1.3.0'

Reset-Fixture
Show-RecommendedModelDownloads -Kind speech
$text = $script:output -join "`n"
Check ($text -match 'MeloTTS' -and $text -match 'release notes') 'Speech recommendation explains pending public hosting'
Check ($text -notmatch 'Choose D' -and $text -notmatch 'SenseVoice') 'Speech-only view cannot suggest unrelated downloads'
Check ($script:promptCount -eq 0 -and $script:requests.Count -eq 0) 'Recommendation is read-only'
Reset-Fixture
Show-RecommendedModelDownloads -Kind audio
Check (($script:output -join "`n") -match 'SenseVoice' -and ($script:output -join "`n") -notmatch 'MeloTTS') 'ASR-only view remains scoped'
Reset-Fixture
Show-RecommendedModelDownloads -Kind all
Check (($script:output -join "`n") -match 'SenseVoice' -and ($script:output -join "`n") -match 'release notes') 'All recommendations expose ASR and honest TTS status'

Reset-Fixture
$result = @(Select-RecommendedDownloads -Kind speech)
Check ($result.Count -eq 0 -and $script:promptCount -eq 0 -and $script:consentCount -eq 0 -and $script:requests.Count -eq 0) 'Speech guide never downloads incompatible files'
Check ($script:guideCount -eq 1) 'Speech guide is opened exactly once'
Check (($script:output -join "`n") -match 'release notes') 'Unavailable speech returns a visible explanation'
Reset-Fixture -Answers @('A')
$result = @(Select-RecommendedDownloads -Kind all)
Check ($result.Count -eq 6 -and ($script:requests -join '/') -eq ($available.Id -join '/')) 'All selects only the six real downloadable bundles'
Check ($script:consentCount -eq 1) 'All requires one explicit confirmation'
Reset-Fixture -Answers @('1')
$result = @(Select-RecommendedDownloads -Kind audio)
Check ($result.Count -eq 1 -and $script:requests[0] -eq 'sensevoice-small-q8') 'ASR-specific selection uses fake download only'
Reset-Fixture -Answers @('1') -Consent $false
$result = @(Select-RecommendedDownloads -Kind audio)
Check ($result.Count -eq 0 -and $script:requests.Count -eq 0) 'Declining confirmation makes no HTTP requests'
Reset-Fixture -Answers @('7', 'B')
$result = @(Select-RecommendedDownloads -Kind all)
Check ($result.Count -eq 0 -and $script:requests.Count -eq 0 -and $script:consentCount -eq 0 -and $script:promptCount -eq 2) 'Unavailable TTS descriptor has no selectable numeric index'
Reset-Fixture -Answers @('5,5,3')
$result = @(Select-RecommendedDownloads -Kind all)
Check ($result.Count -eq 2 -and ($script:requests -join '/') -eq 'sensevoice-small-q8/flashnext-vision') 'Existing numeric ordering and selection deduplication remain stable'
Reset-Fixture -Answers @('B')
$result = @(Select-RecommendedDownloads -Kind embedding)
Check ($result.Count -eq 0 -and $script:requests.Count -eq 0) 'Back never downloads'

# Exercise the real path helper separately from the menu's fake downloader.
foreach ($name in @('Get-SavedValue','ConvertTo-FullPath','Save-ModelDownloadDirectory','Get-ModelDownloadDirectory','Invoke-FileDownloadWithProgress')) {
    $definition = @($ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true))
    . ([scriptblock]::Create($definition[0].Extent.Text))
}
$script:Saved = $null; $script:ModelDownloadDirectoryChosen = $false
Check ((Get-ModelDownloadDirectory) -eq (Join-Path $repoRoot 'models')) 'Fresh installations default to their own models folder'
$script:directoryPrompts = 0
function Read-TextValue { param($Label,$Default,[switch]$Required) $script:directoryPrompts++; return $script:chosenDirectory }
$script:chosenDirectory = '"D:\Models with spaces"'
$chosen = Get-ModelDownloadDirectory -Choose
Check ($chosen -eq 'D:\Models with spaces' -and $script:Saved.download_dir -eq $chosen) 'Pasted quoted custom folder is normalized and retained'
Check ((Get-ModelDownloadDirectory -Choose) -eq $chosen -and $script:directoryPrompts -eq 1) 'A multi-model download batch asks for its directory once'
$script:ModelDownloadDirectoryChosen = $false
$script:chosenDirectory = 'relative models'
Check ((Get-ModelDownloadDirectory -Choose) -eq (Join-Path $repoRoot 'relative models')) 'Relative input resolves beside this installation'
$blocked = Join-Path $repoRoot 'not-a-folder'
[IO.File]::WriteAllText($blocked,'keep')
$script:ModelDownloadDirectoryChosen = $false; $script:chosenDirectory = $blocked
$rejected = $false; try { Get-ModelDownloadDirectory -Choose | Out-Null } catch { $rejected = $true }
Check ($rejected -and [IO.File]::ReadAllText($blocked) -eq 'keep') 'A file cannot become the download directory or be overwritten'
$sizeRejected = $false
try { Invoke-FileDownloadWithProgress -Url 'https://example.invalid/not-contacted' -Destination $blocked -Label 'fixture' -ExpectedBytes 999 } catch { $sizeRejected = $_.Exception.Message -match 'Existing file size differs' }
Check ($sizeRejected -and [IO.File]::ReadAllText($blocked) -eq 'keep') 'A wrong-sized installed file is preserved and rejected before HTTP'
$script:statePath = Join-Path $repoRoot 'wizard-state.json'
[IO.File]::WriteAllText($script:statePath,'{"model_history":["existing.gguf"],"future_setting":"keep"}')
$script:chosenDirectory = 'D:\Persistent Models'; $script:ModelDownloadDirectoryChosen = $false
Get-ModelDownloadDirectory -Choose | Out-Null
$retained = [IO.File]::ReadAllText($script:statePath) | ConvertFrom-Json
Check ($retained.download_dir -eq $script:chosenDirectory -and $retained.future_setting -eq 'keep' -and $retained.model_history[0] -eq 'existing.gguf') 'Download-only exits retain the selected folder without replacing other settings'

Write-Output ("PASS: {0} offline model catalog checks; no HTTP, model downloads, engine, GPU or production state." -f $script:checks)
