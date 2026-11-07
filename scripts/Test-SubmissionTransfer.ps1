[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ApiBase,
    [Parameter(Mandatory = $true)][string]$Scenario,
    [Parameter(Mandatory = $true)][string]$TaskRoot,
    [Parameter(Mandatory = $true)][string]$BenchmarkFile,
    [Parameter(Mandatory = $true)][string]$Infr,
    [string]$WizardPath = '',
    [switch]$Baseline
)

# Run through Test-SubmissionTransfer.cjs: a real loopback HTTP peer, no cloud writes.
$ErrorActionPreference = 'Stop'
if (-not $WizardPath) { $WizardPath = Join-Path $PSScriptRoot 'infr-wizard.ps1' }
Import-Module Microsoft.PowerShell.Utility -Global -Force
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
if ($PSVersionTable.PSVersion.Major -ne 5) { throw 'Run this fixture with Windows PowerShell 5.1' }
$origin = [Uri]$ApiBase
if ($origin.Scheme -ne 'http' -or -not $origin.IsLoopback -or $origin.AbsolutePath -ne '/v1') { throw 'Only explicit loopback HTTP is allowed' }
$allowed = 'D:\AIinfr\infr\local-work\tmp\moe4all\'
$taskPath = [IO.Path]::GetFullPath($TaskRoot)
if (-not $taskPath.StartsWith($allowed, [StringComparison]::OrdinalIgnoreCase)) { throw 'TaskRoot must be in local-work/tmp/moe4all' }
$env:TEMP = $taskPath; $env:TMP = $taskPath; $env:TMPDIR = $taskPath
$env:MOE4ALL_SUBMISSION_API_BASE = $ApiBase
$env:MOE4ALL_SUBMISSION_SITE_ORIGIN = $origin.GetLeftPart([UriPartial]::Authority)
$env:NO_PROXY = '127.0.0.1,localhost'
$tokens = $null; $parseErrors = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($WizardPath, [ref]$tokens, [ref]$parseErrors)
if ($parseErrors.Count) { throw 'Wizard parse failed' }
foreach ($node in $ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst]}, $true)) {
    . ([scriptblock]::Create($node.Extent.Text))
}
$script:dataDir = Join-Path $taskPath $Scenario
$script:infrPath = $Infr; $script:productVersion = 'fixture'
$script:NativePromptsEnabled = $false; $script:DryRun = $false
$script:Messages = [Collections.Generic.List[string]]::new()
$script:BrowserCount = 0
$script:NextActionCount = 0
$timer = [Diagnostics.Stopwatch]::StartNew()
function Assert([bool]$Value, [string]$Message) { if (-not $Value) { throw $Message } }
function Write-Host {
    param($Object, $ForegroundColor, [switch]$NoNewline)
    [void]$script:Messages.Add([string]$Object)
}
function Write-Warning { param([string]$Message) [void]$script:Messages.Add($Message) }
function Start-Moe4AllSubmissionVerification {
    param([Uri]$VerificationUri)
    Assert ($VerificationUri.IsLoopback) 'Unexpected browser origin'
    $script:BrowserCount++
    return $true
}
$before = (Get-FileHash -LiteralPath $BenchmarkFile -Algorithm SHA256).Hash
$failure = $null; $result = $null
$offerFlow = $Scenario.StartsWith('offer-')
$fullFlow = $Scenario.StartsWith('flow-') -or $offerFlow
if ($fullFlow -and $Scenario -in @('flow-timeout', 'offer-later', 'offer-retry')) {
    # Exercise the complete real upload path with short test-only deadlines.
    $realApi = ${function:Invoke-Moe4AllSubmissionApi}
    function Invoke-Moe4AllSubmissionApi {
        param($Method, $Uri, $Token = '', $JsonBody = $null, [byte[]]$FileBytes = $null)
        $retries = if ($offerFlow) { 0 } else { 1 }
        & $realApi -Method $Method -Uri $Uri -Token $Token -JsonBody $JsonBody -FileBytes $FileBytes -TimeoutSeconds 1 -MaximumRetries $retries
    }
}
if ($offerFlow) {
    $script:repoRoot = $dataDir
    $directory = Join-Path $repoRoot 'benchmark-database'
    [void][IO.Directory]::CreateDirectory($directory)
    $copy = Join-Path $directory ([IO.Path]::GetFileName($BenchmarkFile))
    Copy-Item -LiteralPath $BenchmarkFile -Destination $copy
    function Read-YesNo { param($Label, $Default) return $false }
    function Save-BenchmarkUploadPreferences { }
    function Read-SubmissionNextAction {
        $script:NextActionCount++
        if ($Scenario -eq 'offer-retry' -and $script:NextActionCount -eq 1) { return 'retry' }
        return 'later'
    }
}
if ($Scenario -eq 'cancel') {
    function Assert-WizardOperationActive { if ($timer.ElapsedMilliseconds -gt 400) { throw [OperationCanceledException]::new('fixture cancellation') } }
    function Test-WizardOperationCancelled {
        param([Exception]$Exception)
        while ($null -ne $Exception) { if ($Exception.Message -eq 'fixture cancellation') { return $true }; $Exception = $Exception.InnerException }
        return $false
    }
}
try {
    if ($Scenario -eq 'offline-deadline') {
        Add-Type -AssemblyName System.Net.Http
        $script:never = [Threading.Tasks.TaskCompletionSource[System.Net.Http.HttpResponseMessage]]::new()
        $fakeClient = [pscustomobject]@{}
        $fakeClient | Add-Member -MemberType ScriptMethod -Name SendAsync -Value { param($Request, $Completion, $Token) return $script:never.Task }
        $result = Invoke-Moe4AllSubmissionTransfer -Client $fakeClient -Request ([pscustomobject]@{}) -TimeoutSeconds 1
    } elseif ($offerFlow) {
        $result = Offer-SubmissionUpload -Path $copy -Kind benchmark -ConsentAlreadyGranted
    } elseif ($fullFlow) {
        $result = Invoke-WebsiteSubmissionUpload -Path $BenchmarkFile -Kind benchmark
    } else {
        $options = @{ Method = 'POST'; Uri = [Uri]($ApiBase + '/test'); Token = 'fixture-session-secret'; FileBytes = [IO.File]::ReadAllBytes($BenchmarkFile) }
        if (-not $Baseline) {
            $options.TimeoutSeconds = 1
            $options.MaximumRetries = 0
            if ($Scenario -in @('retry-429', 'retry-503', 'retry-date')) { $options.MaximumRetries = 1 }
            if ($Scenario -eq 'exhausted') { $options.MaximumRetries = 3 }
            if ($Scenario -eq 'progress') { $options.TimeoutSeconds = 9 }
        }
        $result = Invoke-Moe4AllSubmissionApi @options
    }
} catch { $failure = $_.Exception }
$timer.Stop()
$messages = $script:Messages -join "`n"
$safeMessages = @($script:Messages | Where-Object { $_ -notmatch '^(Receipt ID:|Private receipt:)' })
[IO.Directory]::CreateDirectory($dataDir) | Out-Null
[IO.File]::WriteAllText((Join-Path $dataDir 'ui.log'), ($safeMessages -join "`n"), [Text.UTF8Encoding]::new($false))
$receiptDir = Join-Path $dataDir 'private\submission-receipts'
$receipts = @(Get-ChildItem -LiteralPath $receiptDir -File -Filter '*.private-receipt.json' -ErrorAction SilentlyContinue)
$guide = Get-BenchmarkGuideState
Assert ($before -eq (Get-FileHash -LiteralPath $BenchmarkFile -Algorithm SHA256).Hash) 'Original bytes changed'
$successful = $Scenario -in @('raw', 'retry-429', 'retry-503', 'retry-date', 'progress', 'flow-success', 'flow-retry', 'flow-lost', 'flow-pending', 'flow-published', 'offer-retry')
if ($successful) {
    Assert ($null -eq $failure) ('Unexpected request failure in ' + $Scenario)
    if ($fullFlow) {
        Assert ($result -eq $true -and $receipts.Count -eq 1 -and $guide.submitted) 'Valid receipt must be saved before marking success'
        $saved = Get-Content -LiteralPath $receipts[0].FullName -Raw -Encoding UTF8 | ConvertFrom-Json
        Assert ($saved.receipt_token -eq 'fixture-receipt-secret' -and $saved.file_sha256 -eq $before.ToLowerInvariant()) 'Saved receipt identity changed'
        $expectedBrowsers = if ($Scenario -eq 'offer-retry') { 2 } else { 1 }
        Assert ($script:BrowserCount -eq $expectedBrowsers) 'Automatic retries must reuse confirmation; a user retry creates a new session'
        Assert ($messages -match 'Received, status:') 'Missing successful receipt message'
        if ($Scenario -notin @('flow-pending', 'flow-published')) {
            Assert ($saved.status -eq 'forwarding' -and $messages -match 'Received, status: forwarding') 'Forwarding must be displayed as forwarding'
        }
        if ($Scenario -eq 'flow-lost') { Assert $saved.duplicate 'Lost response must recover the idempotent receipt' }
    } else { Assert ($result.StatusCode -in @(200, 202)) 'Missing successful API result' }
} else {
    if ($Scenario -eq 'offer-later') { Assert ($result -eq $false) 'Send later must return false' }
    else { Assert ($null -ne $failure) ('Expected failure in ' + $Scenario) }
    Assert ($receipts.Count -eq 0 -and -not $guide.submitted -and $messages -notmatch 'Received, status:') 'Failure must not save or announce success'
    if ($Scenario -in @('timeout', 'slow-body', 'flow-timeout')) {
        Assert ($failure.Data['ApiCode'] -eq 'request_timeout') 'Timeout must have a safe actionable code'
        Assert ($timer.Elapsed.TotalSeconds -lt 12) 'Request deadline was not enforced'
    }
    if ($Scenario -eq 'cancel') { Assert (Test-WizardOperationCancelled $failure) 'Cancellation was swallowed or retried' }
    if ($Scenario -eq 'offline-deadline') {
        Assert ($failure.InnerException -is [TimeoutException] -or $failure -is [TimeoutException]) 'Independent deadline must stop a task that ignores HttpClient cancellation'
        Assert ($timer.Elapsed.TotalSeconds -lt 3) 'Independent deadline did not terminate promptly'
    }
    if ($Scenario -in @('invalid-json', 'empty-json')) { Assert ($failure.Data['ApiCode'] -eq 'invalid_response') 'Invalid response must stop without replay' }
}
if ($offerFlow) {
    $pending = @(Get-PendingSubmissions)
    Assert ($script:NextActionCount -eq 1) 'Failed upload must return to the user action prompt'
    if ($Scenario -eq 'offer-later') {
        Assert ($pending.Count -eq 1 -and $messages -match 'Submission saved for a later retry') 'Failure must preserve the pending report'
        Assert ((Get-FileHash -LiteralPath $pending[0].FullName).Hash -eq $before) 'Pending report bytes changed'
    } else { Assert ($pending.Count -eq 0) 'Verified success must remove the pending copy' }
}
if ($Scenario -in @('retry-429', 'retry-503', 'retry-date', 'exhausted', 'flow-retry', 'flow-lost', 'flow-timeout')) {
    Assert ($messages -match 'Retrying in') 'Retries must be visible'
}
if ($Scenario -eq 'progress') { Assert ($messages -match 'Waiting for submission service: attempt 1/1') 'Missing bounded waiting progress' }
Assert ($messages -notmatch 'fixture-session-secret|fixture-receipt-secret|fixture-browser-secret') 'Credential entered ordinary output'
[pscustomobject]@{ scenario = $Scenario; ps_version = $PSVersionTable.PSVersion.ToString(); elapsed_ms = $timer.ElapsedMilliseconds; failed_as_expected = (-not $successful); receipt_saved = ($receipts.Count -eq 1); success_reminder = $guide.submitted; browser_count = $script:BrowserCount; failure_code = $(if ($failure) { [string]$failure.Data['ApiCode'] } else { '' }); original_preserved = $true } | ConvertTo-Json -Compress | Write-Output
