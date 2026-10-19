[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Infr,
    [Parameter(Mandatory = $true)][string]$Directory,
    [Parameter(Mandatory = $true)][string]$BenchmarkFile,
    [Parameter(Mandatory = $true)][string]$ErrorFile,
    [Parameter(Mandatory = $true)][string]$ApiBase
)
$ErrorActionPreference = 'Stop'
Import-Module Microsoft.PowerShell.Utility -Global -Force
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$uri = [Uri]$ApiBase
if (-not $uri.IsLoopback -or $uri.AbsolutePath -ne '/v1') { throw 'Integration tests require an explicit loopback API' }
$root = [System.IO.Path]::GetFullPath($env:MOE4ALL_WORK_ROOT).TrimEnd('\') + '\runtime\'
if (-not [System.IO.Path]::GetFullPath($Directory).StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { throw 'Test state must stay under local-work/runtime' }
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'infr-wizard.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Wizard parse failed' }
foreach ($node in $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) { . ([scriptblock]::Create($node.Extent.Text)) }
$native = ${function:Invoke-NativeCommandResult}
$repoRoot = $Directory; $dataDir = Join-Path $Directory 'config'; $infrPath = $Infr; $productVersion = 'fixture'
$env:MOE4ALL_SUBMISSION_API_BASE = $ApiBase
$env:MOE4ALL_SUBMISSION_SITE_ORIGIN = $uri.GetLeftPart([UriPartial]::Authority)
$script:UseGit = $true; $script:Calls = [System.Collections.Generic.List[string]]::new(); $script:ReceiptUrls = [System.Collections.Generic.List[string]]::new()
$script:state = [ordered]@{}; $script:BenchmarkUploadHasGithubDefault = $true
function Assert { param([bool]$Value, [string]$Message) if (-not $Value) { throw $Message } }
function Read-YesNo { param([string]$Label, [bool]$Default) if ($Label -like '*Do you have a GitHub*') { return $script:UseGit }; return $true }
function Save-BenchmarkUploadPreferences { }
function Get-Command { param([string]$Name, $ErrorAction) if ($Name -eq 'gh') { return [pscustomobject]@{ Source = 'fixture-gh' } }; Microsoft.PowerShell.Core\Get-Command -Name $Name -ErrorAction $ErrorAction }
function Invoke-NativeCommandResult {
    param([string]$Executable, [string[]]$Arguments)
    if ($Executable -ne 'fixture-gh') { return & $native -Executable $Executable -Arguments $Arguments }
    $line = $Arguments -join ' '; [void]$script:Calls.Add($line)
    $output = '{}'
    if ($line -like 'api user*') { $output = 'fixture' }
    if ($line -like '*git/ref/heads/main*') { $output = 'a' * 40 }
    if ($line -like 'pr create*') { $output = 'https://github.com/Headmaster218/MoE4All/pull/7' }
    return [pscustomobject]@{ ExitCode = 0; Output = @($output) }
}
function Start-Process {
    param([string]$FilePath)
    $url = [Uri]$FilePath
    if ($url.AbsolutePath -eq '/community/upload.html') {
        $values = @{}
        foreach ($part in $url.Fragment.TrimStart('#').Split('&')) { $pair = $part.Split('=', 2); $values[$pair[0]] = [Uri]::UnescapeDataString($pair[1]) }
        $confirmation = Invoke-Moe4AllSubmissionApi -Method GET -Uri ([Uri]($ApiBase + '/upload-sessions/' + $values.session + '/confirmation')) -Token $values.authorize
        Assert (-not $confirmation.Value.summary_verified -and @($confirmation.Value.summary).Count -gt 0) 'Confirmation must receive an unverified readable summary'
        [void](Invoke-Moe4AllSubmissionApi -Method POST -Uri ([Uri]($ApiBase + '/upload-sessions/' + $values.session + '/authorize')) -Token $values.authorize -JsonBody @{ consent = $confirmation.Value.consent; turnstile_token = 'local-fixture' })
    } elseif ($url.AbsolutePath -eq '/community/index.html') { [void]$script:ReceiptUrls.Add($url.AbsoluteUri) }
    else { throw 'Unexpected browser destination' }
}
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
foreach ($entry in @(@('benchmark', $BenchmarkFile), @('error', $ErrorFile))) {
    $kind = $entry[0]; $inputFile = $entry[1]
    $parent = Join-Path $repoRoot ($kind + '-database\v1\2026-10'); New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $path = Join-Path $parent ([IO.Path]::GetFileName($inputFile)); Copy-Item -LiteralPath $inputFile -Destination $path -Force
    if ($kind -eq 'benchmark') {
        $details = [pscustomobject]@{
            MemoryGeneration = 'DDR5'; MemoryModules = @()
            GpuLinks = @([pscustomobject]@{ name = 'Fixture GPU'; current_generation = 4; current_width = 8; max_generation = 5; max_width = 16; current_theoretical_one_way_gb_per_second = 15.752; max_theoretical_one_way_gb_per_second = 63.008; source = 'windows_pnp' })
            ModelStorage = [pscustomobject]@{ model = 'Fixture SSD'; media_type = 'SSD'; bus_type = 'NVMe' }
        }
        Add-BenchmarkHardwareDetails -Path $path -Details $details
    }
    $script:UseGit = $true
    Offer-SubmissionUpload -Path $path -Kind $kind -ConsentAlreadyGranted
    Assert (($script:Calls -join "`n") -like "*$kind-database/v1/2026-10/*") 'Direct PR must use the correct report database'
    Assert (($script:Calls -join "`n") -like "*community/$kind/*") 'Direct PR branch must be separated by kind'
    $script:UseGit = $false
    Offer-SubmissionUpload -Path $path -Kind $kind -ConsentAlreadyGranted
    $text = [IO.File]::ReadAllText($path).Replace("`r`n", "`n").Replace("`n", "`r`n")
    [IO.File]::WriteAllText($path, $text, [Text.UTF8Encoding]::new($false))
    Offer-SubmissionUpload -Path $path -Kind $kind -ConsentAlreadyGranted
}
Assert ($script:ReceiptUrls.Count -eq 4) 'Every HTTP receipt must open a browser handoff, including duplicates'
$receiptDir = Join-Path $dataDir 'private\submission-receipts'
$files = @(Get-ChildItem -LiteralPath $receiptDir -Filter '*.private-receipt.json')
Assert ($files.Count -eq 2) 'Duplicate uploads must reuse their receipt'
foreach ($file in $files) {
    $receipt = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
    Assert ($receipt.file_sha256 -ne $receipt.submitted_file_sha256) 'CRLF retry must retain original and submitted hashes separately'
}
$private = Join-Path $dataDir 'private\handoff-urls.json'
[IO.File]::WriteAllText($private, (ConvertTo-Json -InputObject @($script:ReceiptUrls) -Compress), [Text.UTF8Encoding]::new($false))
Write-Host 'Submission integration passed: both PR routes, both HTTP uploads, duplicate retries and browser receipt handoffs. GitHub is a fixture; HTTP uses the local server.'
