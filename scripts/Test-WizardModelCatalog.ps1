[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$WorkDir)

$ErrorActionPreference = 'Stop'
$WorkDir = [IO.Path]::GetFullPath($WorkDir)
if (-not $WorkDir.StartsWith('D:\AIinfr\infr\local-work\tmp\moe4all\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Use local-work/tmp/moe4all/<task>/model-catalog.' }
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
$names = @('Get-RecommendedModelCatalog', 'Show-RecommendedModelDownloads', 'Select-RecommendedDownloads')
foreach ($name in $names) {
    $definition = @($ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name }, $true))
    if ($definition.Count -ne 1) { throw ('Expected one definition: ' + $name) }
    . ([scriptblock]::Create($definition[0].Extent.Text))
}
# Only these three pure/catalog functions are imported. HTTP and the real
# downloader/launcher entry point are never imported or invoked by this test.
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
function Reset-Fixture([string[]]$Answers = @(), [bool]$Consent = $true) {
    $script:output = [Collections.Generic.List[string]]::new()
    $script:requests = [Collections.Generic.List[string]]::new()
    $script:answers = [Collections.Generic.Queue[string]]::new()
    foreach ($answer in $Answers) { $script:answers.Enqueue($answer) }
    $script:promptCount = 0; $script:consentCount = 0; $script:consent = $Consent
}

Reset-Fixture
$catalog = @(Get-RecommendedModelCatalog)
Check ($catalog.Count -eq 6) 'Five existing downloads plus one honest unavailable TTS entry'
Check (@($catalog.Id | Sort-Object -Unique).Count -eq $catalog.Count) 'Catalog IDs are unique'
$available = @($catalog | Where-Object { -not $_.PSObject.Properties['Downloadable'] -or $_.Downloadable })
Check ($available.Count -eq 5) 'Existing downloads remain selectable'
foreach ($bundle in $available) {
    Check ($bundle.Kind -in @('llm', 'vision', 'embedding', 'audio')) 'No unverified speech weights are offered'
    Check (-not [IO.Path]::IsPathRooted($bundle.Primary) -and $bundle.Primary -notmatch '(^|[\\/])\.\.([\\/]|$)') 'Primary path stays package relative'
    Check ($bundle.Primary -in @($bundle.Files.RelativePath)) 'Primary is actually supplied by the bundle'
    foreach ($file in $bundle.Files) {
        $url = $null
        Check ([Uri]::TryCreate($file.Url, [UriKind]::Absolute, [ref]$url) -and $url.Scheme -eq 'https') 'Download source uses HTTPS'
        Check ([IO.Path]::GetExtension($file.RelativePath) -ieq '.gguf') 'Only existing native GGUF runtime downloads'
        Check (-not [IO.Path]::IsPathRooted($file.RelativePath) -and $file.RelativePath -notmatch '(^|[\\/])\.\.([\\/]|$)') 'File path stays package relative'
    }
}
$audio = @($catalog | Where-Object { $_.Kind -eq 'audio' })
Check ($audio.Count -eq 1 -and $audio[0].Id -eq 'sensevoice-small-q8') 'Existing native ASR bundle remains available without duplication'
Check ($audio[0].Files[0].Url -eq 'https://huggingface.co/lovemefan/sense-voice-gguf/resolve/main/sense-voice-small-q8_0.gguf?download=true') 'ASR keeps the verified SenseVoice source'
$speech = @($catalog | Where-Object { $_.Kind -eq 'speech' })
Check ($speech.Count -eq 1 -and $speech[0].Id -eq 'melotts-chinese-native-f32') 'Stable TTS descriptor for the downloads UI'
Check ($speech[0].Downloadable -is [bool] -and -not $speech[0].Downloadable) 'Unavailable is an explicit boolean'
Check (@($speech[0].Files).Count -eq 0 -and [string]::IsNullOrEmpty($speech[0].Primary)) 'No incompatible or invented download URL'
Check ($speech[0].UnavailableReason -match 'public download hosting is pending' -and $speech[0].UnavailableReason -match 'End users do not need conversion') 'Distinguish verified native package from pending public hosting'
Check ($speech[0].LocalPackageReady -is [bool] -and $speech[0].LocalPackageReady -and $speech[0].PackageScript -eq 'scripts/package-native-speech.ps1') 'Catalog points to reusable native packaging'
Check ($speech[0].ModelSha256 -eq '6d0d6fffe43cf6dea169f584c07bbb1a43756f37958ba9f9b78e27e8f9b32cc2' -and $speech[0].ModelBytes -eq 164882176) 'Catalog identifies the previously successful native artifact'
Check (($speech[0].RuntimeAssets -join '/') -eq 'melotts-chinese-f32.gguf/lexicon.txt/tokens.txt/config.json/frontend.json/custom-lexicon.txt') 'Preserve the complete successful voice frontend'
Check (($speech[0].RequiredAssets -join '/') -eq 'lexicon.txt/tokens.txt/config.json') 'Matching resource requirements are explicit'
Check ($speech[0].ConversionScript -eq 'scripts/convert_melotts_chinese.py') 'Points to the established offline conversion workflow'
Check ($speech[0].SourceUrls[0] -match '/af5d207a364ea4208c6f589c89f57f88414bdd16$' -and $speech[0].SourceUrls[1] -match '/a0d5c6a264c0ef92d70d8661d8cc502d79627cd6$') 'Conversion origins are revision pinned'

Reset-Fixture
Show-RecommendedModelDownloads -Kind speech
$text = $script:output -join "`n"
Check ($text -match 'MeloTTS' -and $text -match 'public download hosting is pending') 'Speech recommendation explains pending public hosting'
Check ($text -notmatch 'Choose D' -and $text -notmatch 'SenseVoice') 'Speech-only view cannot suggest unrelated downloads'
Check ($script:promptCount -eq 0 -and $script:requests.Count -eq 0) 'Recommendation is read-only'
Reset-Fixture
Show-RecommendedModelDownloads -Kind audio
Check (($script:output -join "`n") -match 'SenseVoice' -and ($script:output -join "`n") -notmatch 'MeloTTS') 'ASR-only view remains scoped'
Reset-Fixture
Show-RecommendedModelDownloads -Kind all
Check (($script:output -join "`n") -match 'SenseVoice' -and ($script:output -join "`n") -match 'public download hosting is pending') 'All recommendations expose ASR and honest TTS status'

Reset-Fixture
$result = @(Select-RecommendedDownloads -Kind speech)
Check ($result.Count -eq 0 -and $script:promptCount -eq 0 -and $script:consentCount -eq 0 -and $script:requests.Count -eq 0) 'Unavailable speech cannot ask consent or call HTTP'
Check (($script:output -join "`n") -match 'public download hosting is pending') 'Unavailable speech returns a visible explanation'
Reset-Fixture -Answers @('A')
$result = @(Select-RecommendedDownloads -Kind all)
Check ($result.Count -eq 5 -and ($script:requests -join '/') -eq ($available.Id -join '/')) 'All selects only the five real downloadable bundles'
Check ($script:consentCount -eq 1) 'All requires one explicit confirmation'
Reset-Fixture -Answers @('1')
$result = @(Select-RecommendedDownloads -Kind audio)
Check ($result.Count -eq 1 -and $script:requests[0] -eq 'sensevoice-small-q8') 'ASR-specific selection uses fake download only'
Reset-Fixture -Answers @('1') -Consent $false
$result = @(Select-RecommendedDownloads -Kind audio)
Check ($result.Count -eq 0 -and $script:requests.Count -eq 0) 'Declining confirmation makes no HTTP requests'
Reset-Fixture -Answers @('6', 'B')
$result = @(Select-RecommendedDownloads -Kind all)
Check ($result.Count -eq 0 -and $script:requests.Count -eq 0 -and $script:consentCount -eq 0 -and $script:promptCount -eq 2) 'Unavailable sixth descriptor has no selectable numeric index'
Reset-Fixture -Answers @('5,5,3')
$result = @(Select-RecommendedDownloads -Kind all)
Check ($result.Count -eq 2 -and ($script:requests -join '/') -eq 'sensevoice-small-q8/flashnext-vision') 'Existing numeric ordering and selection deduplication remain stable'
Reset-Fixture -Answers @('B')
$result = @(Select-RecommendedDownloads -Kind embedding)
Check ($result.Count -eq 0 -and $script:requests.Count -eq 0) 'Back never downloads'

Write-Output ("PASS: {0} offline model catalog checks; no HTTP, model downloads, engine, GPU or production state." -f $script:checks)
