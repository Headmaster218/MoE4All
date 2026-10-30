[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$TaskRoot)

$ErrorActionPreference = 'Stop'
if (-not [IO.Path]::IsPathRooted($TaskRoot)) { throw 'TaskRoot must be an absolute path' }
$taskPath = [IO.Path]::GetFullPath($TaskRoot)
if ($env:MOE4ALL_WORK_ROOT) {
    $allowed = [IO.Path]::GetFullPath((Join-Path (Join-Path $env:MOE4ALL_WORK_ROOT 'tmp') 'moe4all')).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    if (-not $taskPath.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) { throw 'TaskRoot must be under MOE4ALL_WORK_ROOT/tmp/moe4all' }
}
$env:TEMP = $taskPath; $env:TMP = $taskPath; $env:TMPDIR = $taskPath
$testRoot = Join-Path $taskPath ('guide-ps-' + [guid]::NewGuid().ToString('N'))
[void][IO.Directory]::CreateDirectory($testRoot)
$wizard = Join-Path $PSScriptRoot 'infr-wizard.ps1'
$tokens = $null; $parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($wizard, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw ($parseErrors | Out-String) }
foreach ($definition in $ast.FindAll({ param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    . ([scriptblock]::Create($definition.Extent.Text))
}
$realChoice = ${function:Read-Choice}
$realNextAction = ${function:Read-SubmissionNextAction}
$realSaveUploadPreferences = ${function:Save-BenchmarkUploadPreferences}
$realSaveErrorPreference = ${function:Save-ErrorUploadPreference}
function Assert($Condition, [string]$Message) { if (-not $Condition) { throw $Message } }
function Get-BenchmarkGuideNow { return $script:Now }
function Read-Choice {
    param($Label, $DefaultValue, $Options)
    $script:Prompts++
    Assert ($DefaultValue -eq 'later') 'The guide must default to later'
    Assert ($Options.Count -eq 2 -and $Options[0].Label.EndsWith(' / Try it!') -and $Options[1].Label.EndsWith(' / Maybe later~')) 'Wrong guide options'
    if ($script:Cancel) { throw 'fixture cancellation' }
    return $script:Answer
}
function Reset-Case([string]$Name) {
    $script:dataDir = Join-Path $testRoot $Name
    $script:statePath = Join-Path $dataDir 'wizard-state.json'
    $script:DryRun = $false; $script:Now = [long]1800000000
    $script:Prompts = 0; $script:Answer = 'later'; $script:Cancel = $false
    $script:BenchmarkGuideActive = $false
    $script:guidePath = Join-Path $dataDir 'private\benchmark-first-submission.json'
}
function Write-Receipt($Receipt) {
    $directory = Join-Path $dataDir 'private\submission-receipts'
    [void][IO.Directory]::CreateDirectory($directory)
    [IO.File]::WriteAllText((Join-Path $directory 'fixture.private-receipt.json'), ($Receipt | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
}

Reset-Case 'transitions'
Assert (-not (Invoke-BenchmarkStartupGuide -Interactive $true)) 'Later must not enter a benchmark'
Assert ($script:Prompts -eq 1) 'First startup must invite'
Assert ((Get-BenchmarkGuideState).next_remind_at -eq ($script:Now + 172800)) 'Later must defer exactly 48 hours'
$first = [IO.File]::ReadAllText($guidePath)
$script:Now += 172799
Assert (-not (Invoke-BenchmarkStartupGuide -Interactive $true)) 'Cooldown must hold until the boundary'
Assert ($script:Prompts -eq 1 -and [IO.File]::ReadAllText($guidePath) -eq $first) 'Cooldown startup must not prompt or write'
$script:Now++; $script:Answer = 'try'
Assert (Invoke-BenchmarkStartupGuide -Interactive $true) 'Exact 48-hour boundary must allow Try'
Assert ((Get-BenchmarkGuideState).next_remind_at -eq ($script:Now + 172800)) 'Try must defer even before a result exists'
$script:Now += 172800; $script:Cancel = $true
try { [void](Invoke-BenchmarkStartupGuide -Interactive $true) } catch { Assert ($_.Exception.Message -eq 'fixture cancellation') 'Unexpected failure' }
Assert ((Get-BenchmarkGuideState).next_remind_at -eq ($script:Now + 172800)) 'Cancellation must retain the cooldown'
$script:Cancel = $false
Set-BenchmarkSubmissionSucceeded error
Assert (-not (Get-BenchmarkGuideState).submitted) 'Error uploads must not count'
Set-BenchmarkSubmissionSucceeded benchmark
Save-BenchmarkGuideState -NextRemindAt ($script:Now + 999999)
$script:Now += 9999999
$before = $script:Prompts
Assert (-not (Invoke-BenchmarkStartupGuide -Interactive $true)) 'Success must suppress permanently'
Assert ($script:Prompts -eq $before) 'Success must disable repeats'
Assert (([IO.File]::ReadAllText($guidePath) | ConvertFrom-Json).submitted -eq $true) 'A later reminder write must not erase success'
Assert ([IO.File]::ReadAllText($guidePath) -notmatch 'token|receipt_id') 'Guide state must not contain credentials or receipt data'

foreach ($mode in @('headless', 'dry', 'json')) {
    Reset-Case $mode
    $script:DryRun = ($mode -eq 'dry')
    Assert (-not (Invoke-BenchmarkStartupGuide -Interactive ($mode -ne 'headless') -JsonAutomation ($mode -eq 'json'))) "$mode entered the guide"
    Assert ($script:Prompts -eq 0 -and -not (Test-Path -LiteralPath $dataDir)) "$mode prompted or wrote state"
}

$receipt = [pscustomobject]@{
    api_version = 1; kind = 'benchmark'; receipt_id = 'rc_fixture'; report_id = 'report-fixture'
    file_sha256 = ('a' * 64); payload_sha256 = ('b' * 64)
    status = 'forwarding'; received_at = '2026-10-25T00:00:00Z'; receipt_token = 'never-log-this-fixture-secret'
}
foreach ($status in @('forwarding', 'pending_review', 'published', 'rejected', 'delivery_failed')) {
    Reset-Case ("backfill-" + $status)
    $receipt.status = $status; Write-Receipt $receipt
    Assert (-not (Invoke-BenchmarkStartupGuide -Interactive $true)) 'Backfill must suppress the invitation'
    Assert ((Get-BenchmarkGuideState).submitted -and $script:Prompts -eq 0) 'Body-received status must count regardless of review/delivery outcome'
}
foreach ($invalid in @('error', 'authorization', 'hash', 'received')) {
    Reset-Case ("invalid-" + $invalid)
    $copy = $receipt | ConvertTo-Json | ConvertFrom-Json
    switch ($invalid) {
        error { $copy.kind = 'error' }
        authorization { $copy.status = 'ready' }
        hash { $copy.payload_sha256 = '' }
        received { $copy.received_at = '' }
    }
    Write-Receipt $copy
    [void](Invoke-BenchmarkStartupGuide -Interactive $true)
    Assert (-not (Get-BenchmarkGuideState).submitted -and $script:Prompts -eq 1) "$invalid is not historical benchmark proof"
}
Reset-Case 'browser-preference'
$script:SubmissionBrowserActionDefault = 'sent'
[void](Invoke-BenchmarkStartupGuide -Interactive $true)
Assert ($script:Prompts -eq 1) 'Saved browser sent preference must not backfill'
function Read-Host { return '' }
Set-Item Function:Read-Choice $realChoice
function Save-BenchmarkUploadPreferences { }
Assert ((Read-SubmissionNextAction -Browser) -eq 'later') 'Enter must not implicitly confirm a new browser upload'

# Exercise the real upload dispatcher with local mocks, including explicit consent.
function Test-SubmissionForUpload { }
function Save-PendingSubmission { return (Join-Path $dataDir 'pending.m4b') }
function Remove-PendingSubmission { }
function Read-BenchmarkUploadConsent { return $script:Consent }
function Read-YesNo { return $true }
function Invoke-WithSubmissionGithubRoute { return $false }
function Open-SubmissionBrowserUpload { }
function Read-SubmissionNextAction { return $script:BrowserAnswer }
foreach ($kind in @('benchmark', 'error')) {
    Reset-Case ("browser-" + $kind)
    $script:Consent = $false; $script:BrowserAnswer = 'sent'
    if ($kind -eq 'benchmark') {
        Assert (-not (Offer-SubmissionUpload -Path fixture.m4b -Kind $kind)) 'Try must not imply upload consent'
        Assert (-not (Test-Path -LiteralPath $guidePath)) 'Declined consent must not count'
    }
    [void](Offer-SubmissionUpload -Path fixture.m4b -Kind $kind -ConsentAlreadyGranted)
    Assert ((Get-BenchmarkGuideState).submitted -eq ($kind -eq 'benchmark')) 'Only explicit benchmark browser confirmation counts'
}

# Exercise valid/invalid PR responses through the real GitHub implementation.
function gh { }
function Test-GhAuthenticated { return $true }
function Test-SubmissionForUpload { return [pscustomobject]@{ id = 'fixture'; created_at_utc = '2026-10-25T00:00:00Z' } }
function Invoke-NativeCommandResult {
    param($Executable, [string[]]$Arguments)
    $output = @()
    if ($Arguments[0] -eq 'api' -and $Arguments[1] -eq 'user') { $output = @('fixture-user') }
    elseif ($Arguments -contains '.object.sha') { $output = @(('a' * 40)) }
    elseif ($Arguments[0] -eq 'pr') { $output = @($script:PrResponse) }
    return [pscustomobject]@{ ExitCode = 0; Output = $output }
}
foreach ($kind in @('benchmark', 'error')) {
    foreach ($valid in @($false, $true)) {
        Reset-Case ("github-" + $kind + '-' + $valid)
        [void][IO.Directory]::CreateDirectory($dataDir)
        $reportPath = Join-Path $dataDir 'fixture.m4b'; [IO.File]::WriteAllText($reportPath, 'fixture')
        $script:PrResponse = if ($valid) { 'https://github.com/Headmaster218/MoE4All/pull/123' } else { 'not-a-pr' }
        $uploaded = Invoke-GhSubmissionUpload -Path $reportPath -Kind $kind
        Assert ($uploaded -eq $valid) 'GitHub must accept only a validated PR URL'
        Assert ((Get-BenchmarkGuideState).submitted -eq ($valid -and $kind -eq 'benchmark')) 'Only valid benchmark PRs must count'
    }
}

# Exercise the real website client with mocked session and body responses.
function Read-MoE4AllSubmissionPayload { return [pscustomobject]@{ id = 'fixture'; description = 'fixture' } }
function Get-Moe4AllSubmissionApiConfig { return [pscustomobject]@{ ApiBase = [Uri]'https://api.moe4all.com/v1'; SiteOrigin = 'https://moe4all.com' } }
function Start-Moe4AllSubmissionVerification { return $true }
function Invoke-Moe4AllSubmissionApi {
    param($Method, $Uri, $Token, $JsonBody, $FileBytes)
    if ($Uri.AbsolutePath.EndsWith('/report')) {
        if (-not $script:BodyOkay) { throw 'fixture body upload failed' }
        $value = [pscustomobject]@{
            api_version = 1; kind = $script:ReportKind; report_id = 'fixture'; receipt_id = 'rc_fixture'
            receipt_token = 'private-fixture'; file_sha256 = $script:ReportHash; payload_sha256 = ('b' * 64)
            status = 'forwarding'; received_at = '2026-10-25T00:00:00Z'
        }
    } elseif ($Method -eq 'GET') {
        $value = [pscustomobject]@{ api_version = 1; state = 'ready'; kind = $script:ReportKind; report_id = 'fixture'; file_sha256 = $script:ReportHash; size_bytes = $script:ReportSize }
    } else {
        $value = [pscustomobject]@{
            api_version = 1; state = 'awaiting_verification'; session_id = 'us_fixture'; session_token = 'private-fixture'
            verification_url = 'https://moe4all.com/community/upload.html#fixture'; max_upload_bytes = 131072
            expires_at = [DateTimeOffset]::UtcNow.AddHours(1).ToString('o'); poll_after_seconds = 3
        }
    }
    return [pscustomobject]@{ Value = $value }
}
foreach ($kind in @('benchmark', 'error')) {
    Reset-Case ("cloudflare-" + $kind)
    [void][IO.Directory]::CreateDirectory($dataDir)
    $extension = if ($kind -eq 'benchmark') { '.m4b' } else { '.m4e' }
    $reportPath = Join-Path $dataDir ('fixture' + $extension)
    [IO.File]::WriteAllText($reportPath, ("MOE4ALL-SUBMISSION/1`nkind: $kind`nid: fixture`n"))
    $script:ReportKind = $kind; $script:ReportHash = (Get-FileHash -LiteralPath $reportPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $script:ReportSize = ([IO.File]::ReadAllBytes($reportPath)).Length
    $script:BodyOkay = $false
    try { [void](Invoke-WebsiteSubmissionUpload -Path $reportPath -Kind $kind); throw 'Body failure was ignored' }
    catch { Assert ($_.Exception.Message -eq 'fixture body upload failed') 'Unexpected website failure' }
    Assert (-not (Test-Path -LiteralPath $guidePath)) 'Authorization without body receipt must not count'
    $script:BodyOkay = $true
    Assert (Invoke-WebsiteSubmissionUpload -Path $reportPath -Kind $kind) 'Valid body receipt should complete'
    Assert ((Get-BenchmarkGuideState).submitted -eq ($kind -eq 'benchmark')) 'Only a benchmark body receipt counts'
}

# Extract only the startup/configuration flow, using the existing wizard-test mocks.
$source = [IO.File]::ReadAllText($wizard)
$save = $ast.EndBlock.Statements | Where-Object { $_.Extent.Text.StartsWith('$state = [ordered]') } | Select-Object -First 1
$start = $source.IndexOf('$reuseSavedSettings = $false')
$flow = [scriptblock]::Create($source.Substring($start, $save.Extent.EndOffset - $start))
Reset-Case 'guided-flow'
$model = Join-Path $testRoot 'model.gguf'; [IO.File]::WriteAllText($model, '')
$script:Saved = [pscustomobject]@{ launch_mode = 'server'; model = $model; bench_kind = 'full'; bench_parallel = '4'; server_parallel = '3' }
function Test-BenchmarkGuideInteractive { return $true }
function Invoke-BenchmarkStartupGuide { Save-BenchmarkGuideState -NextRemindAt ($script:Now + 172800); return $true }
function Start-VulkanDeviceProbe { }
function Select-ModelPath { return $model }
function Read-ComputeDevice { return 'Vulkan0' }
$script:GuidedProfile = 'conservative'
function Read-Choice {
    param($Label, $Options, $DefaultValue)
    Assert ($Label -notmatch 'Benchmark plan|What would you like') 'Guided flow must use the quick plan'
    if ($Label -match 'Configuration') { return $script:GuidedProfile }
    return $DefaultValue
}
function Read-YesNo { return $false }
function Read-IntegerValue { param($Label, $Default); return $Default }
function Get-CpuMissTopology { return [pscustomobject]@{ Physical = 6; Preferred = 6; Detected = $true } }
function Start-BenchmarkEstimateProbe { }
function Get-UpdatedModelHistory { return @($model) }
function Format-NativeArgument { param($Value); return $Value }
function Invoke-NativeCommandResult { throw 'No native process is permitted in guide flow tests' }
$repoRoot = $testRoot; $guiStatePath = Join-Path $testRoot 'no-gui.json'
$infrPath = Join-Path $testRoot 'never-run.exe'; $InitialModelPath = ''
. $flow
Assert ($nativeArgs[0] -eq 'bench' -and 'quick' -in $nativeArgs -and '--parallel' -in $nativeArgs) 'Try must build the quick benchmark command'
Assert ('device.auto_profile=conservative' -in $nativeArgs -and 'device.auto_profile=aggressive' -notin $nativeArgs) 'Guided quick benchmark must preserve a conservative selection'
$script:GuidedProfile = 'aggressive'
. $flow
Assert ('device.auto_profile=aggressive' -in $nativeArgs -and 'device.auto_profile=conservative' -notin $nativeArgs) 'Guided quick benchmark must preserve an aggressive selection'
Assert ($script:Saved.launch_mode -eq 'server' -and $script:Saved.bench_kind -eq 'full' -and $script:Saved.bench_parallel -eq '4') 'Daily defaults changed'
Assert (-not (Test-Path -LiteralPath $statePath)) 'Guided configuration must not save daily defaults'
[void][IO.Directory]::CreateDirectory($dataDir)
$daily = $script:Saved | ConvertTo-Json
[IO.File]::WriteAllText($statePath, $daily, [Text.UTF8Encoding]::new($false))
$saveGuard = $ast.EndBlock.Statements | Where-Object { $_.Extent.Text.StartsWith('if (-not $DryRun -and -not $script:BenchmarkGuideActive)') } | Select-Object -First 1
Assert ($null -ne $saveGuard) 'Full settings save must be guarded during the guide'
. ([scriptblock]::Create($saveGuard.Extent.Text))
Set-Item Function:Save-BenchmarkUploadPreferences $realSaveUploadPreferences
Set-Item Function:Save-ErrorUploadPreference $realSaveErrorPreference
Save-BenchmarkUploadPreferences
Save-ErrorUploadPreference
$persisted = [IO.File]::ReadAllText($statePath) | ConvertFrom-Json
Assert ($persisted.launch_mode -eq 'server' -and $persisted.bench_kind -eq 'full' -and $persisted.bench_parallel -eq '4' -and $persisted.server_parallel -eq '3') 'Guided upload preference rewrites must preserve daily configuration'
$script:BenchmarkUploadDefault = $true; $script:SubmissionBrowserActionDefault = 'sent'
Save-BenchmarkUploadPreferences
$persisted = [IO.File]::ReadAllText($statePath) | ConvertFrom-Json
Assert ($persisted.benchmark_upload -and $persisted.submission_browser_action -eq 'sent') 'Guided upload defaults must be remembered'
Assert ($persisted.launch_mode -eq 'server' -and $persisted.bench_kind -eq 'full') 'Saving guided upload defaults must not save the temporary quick plan'
Write-Output 'PASS benchmark startup guide: clock boundaries, cancellation/failure cooldown, permanent success, consent, backfill, browser confirmation, automation guards and quick flow'
