[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$TaskRoot)
$ErrorActionPreference='Stop'
$TaskRoot=[IO.Path]::GetFullPath($TaskRoot)
$allowed=[IO.Path]::GetFullPath((Join-Path $env:MOE4ALL_WORK_ROOT 'tmp\moe4all')).TrimEnd('\')+'\'
if(-not $TaskRoot.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Use the shared task tmp root'}
[void][IO.Directory]::CreateDirectory($TaskRoot)
$tokens=$null;$errors=$null
$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'infr-wizard.ps1'),[ref]$tokens,[ref]$errors)
if($errors.Count){throw ($errors|Out-String)}
foreach($fn in $ast.FindAll({param($n)$n-is[Management.Automation.Language.FunctionDefinitionAst]},$true)){. ([scriptblock]::Create($fn.Extent.Text))}
. (Join-Path $PSScriptRoot 'moe4all-update-common.ps1')
$script:checks=0
function Check([bool]$ok,[string]$text){$script:checks++;if(-not $ok){throw $text}}
function Get-BenchmarkGuideNow{return $script:now}
function Write-Host{param($Object,$ForegroundColor)}
function Write-Warning{param($Message)}
function Read-Choice{param($Label,$DefaultValue,$Options)$script:prompts++;return 'later'}
function Reset-Fixture{
    $script:dataDir=Join-Path $TaskRoot ([guid]::NewGuid().ToString('N'))
    $script:Saved=$null;$script:DryRun=$false;$script:SkipUpdateCheck=$false;$script:productVersion='1.3.0'
    $script:now=[long]1800000000;$script:prompts=0;$script:EngineUpdateProbe=$null
}
Reset-Fixture
Check (-not (Invoke-BenchmarkStartupGuide -Interactive $true)) 'Later must keep the normal task'
$first=Get-BenchmarkGuideState
Check ($script:prompts-eq 1 -and $first.last_prompt_version-eq'1.3.0' -and $first.next_remind_at-eq($now+172800)) 'First-use reminder must use 48 hours'
$script:now+=20
Invoke-BenchmarkStartupGuide -Interactive $true|Out-Null
Check ($script:prompts-eq 1) 'Same-version cooldown'
$script:productVersion='1.4.0'
Invoke-BenchmarkStartupGuide -Interactive $true|Out-Null
$upgraded=Get-BenchmarkGuideState
Check ($script:prompts-eq 2 -and $upgraded.reminder_reason-eq'new_version' -and $upgraded.next_remind_at-eq($now+604800)) 'A new version bypasses the old cooldown and then uses 7 days'
$script:now+=604799
Invoke-BenchmarkStartupGuide -Interactive $true|Out-Null
Check ($script:prompts-eq 2) 'New-version cooldown holds'
$script:now++
Invoke-BenchmarkStartupGuide -Interactive $true|Out-Null
Check ($script:prompts-eq 3 -and (Get-BenchmarkGuideState).next_remind_at-eq($now+604800)) 'Subsequent version reminders retain 7 days'
Set-BenchmarkSubmissionSucceeded benchmark -ReportVersion '1.3.0'
Check ((Get-BenchmarkGuideState).last_submitted_version-eq'1.3.0') 'Old pending report retains its own version'
Set-BenchmarkSubmissionSucceeded benchmark -ReportVersion '1.4.0'
$script:now+=9999999
Invoke-BenchmarkStartupGuide -Interactive $true|Out-Null
Check ($script:prompts-eq 3) 'Current-version success suppresses its reminder'
Set-BenchmarkSubmissionSucceeded benchmark -ReportVersion '1.3.0'
Check ((Get-BenchmarkGuideState).last_submitted_version-eq'1.4.0') 'Old pending uploads cannot erase current-version success'
$script:productVersion='1.5.0'
Invoke-BenchmarkStartupGuide -Interactive $true|Out-Null
Check ($script:prompts-eq 4) 'Success on an old version cannot suppress a new version'
Reset-Fixture
$directory=Join-Path $dataDir 'private';[void][IO.Directory]::CreateDirectory($directory)
[IO.File]::WriteAllText((Join-Path $directory 'benchmark-first-submission.json'),' {"schema_version":1,"submitted":true,"next_remind_at":9999999999}')
Invoke-BenchmarkStartupGuide -Interactive $true|Out-Null
Check ($script:prompts-eq 1 -and (Get-BenchmarkGuideState).reminder_reason-eq'new_version') 'Legacy permanent suppression migrates to a new-version invitation'
Reset-Fixture
Set-BenchmarkSubmissionSucceeded error
Check (-not (Get-BenchmarkGuideState).submitted) 'Error uploads do not count'
foreach($mode in @('headless','dry','json')){
    Reset-Fixture;$script:DryRun=$mode-eq'dry'
    Invoke-BenchmarkStartupGuide -Interactive ($mode-ne'headless') -JsonAutomation ($mode-eq'json')|Out-Null
    Check ($script:prompts-eq 0 -and -not(Test-Path -LiteralPath $dataDir)) 'Noninteractive paths do not prompt or persist'
}
Reset-Fixture
$script:action='snooze';$script:updatePrompts=0;$script:updates=0;$script:offline=$false
$script:release=[pscustomobject]@{Tag='release-1.4.0';Version='1.4.0';SemVer=(ConvertTo-Moe4AllSemVer '1.4.0');Notes='fixture';Url='https://github.com/Headmaster218/MoE4All/releases/tag/release-1.4.0'}
function Get-LatestEngineRelease{param($CurrentVersion)if($script:offline){throw 'fixture offline'};return $script:release}
function Read-EngineUpdateAction{param($Release,[switch]$Window,$CurrentVersion)$script:updatePrompts++;$script:lastWindow=[bool]$Window;return $script:action}
function Invoke-EngineSelfUpdate{param($Release)$script:updates++}
Show-UpdateStatus '1.3.0' -Window
Check ($script:updatePrompts-eq 1 -and $script:lastWindow -and (Get-EngineUpdateReminder).next_remind_at-eq($now+259200)) 'Native update prompt and 3-day snooze'
$script:now+=259199
Show-UpdateStatus '1.3.0'
Check ($script:updatePrompts-eq 1) 'Same-release update snooze'
$script:now++
Show-UpdateStatus '1.3.0'
Check ($script:updatePrompts-eq 2) 'Update snooze expires at the boundary'
$script:release.Tag='release-1.5.0';$script:release.Version='1.5.0';$script:release.SemVer=ConvertTo-Moe4AllSemVer '1.5.0'
Show-UpdateStatus '1.3.0'
Check ($script:updatePrompts-eq 3) 'A newer release bypasses the previous release snooze'
$script:action='skip';Show-UpdateStatus '1.3.0' -Force
Check ((Get-EngineUpdateReminder).next_remind_at-eq 0) 'Next-time update choice clears snooze'
Show-UpdateStatus '1.3.0'
Check ($script:updatePrompts-eq 5) 'Next-time asks on the following startup'
$script:action='update';Show-UpdateStatus '1.3.0'
Check ($script:updates-eq 1) 'Only Update invokes the installer'
$script:offline=$true;Show-UpdateStatus '1.3.0'
Check ($script:updatePrompts-eq 6) 'Offline update checks do not block startup'
$script:SkipUpdateCheck=$true;Show-UpdateStatus '1.3.0'
Check ($script:updatePrompts-eq 6) 'Skip-update remains respected'
Reset-Fixture
$script:updateCommonPath=Join-Path $PSScriptRoot 'moe4all-update-common.ps1'
function Get-LatestEngineRelease {
    param($CurrentVersion)
    [pscustomobject]@{Tag='release-1.4.0';Version='1.4.0';SemVer=(ConvertTo-Moe4AllSemVer '1.4.0');Notes='Background fixture';Url='https://github.com/Headmaster218/MoE4All/releases/tag/release-1.4.0'}
}
$script:action='skip';$script:updatePrompts=0
Start-EngineUpdateProbe -CurrentVersion '1.3.0'
Check ($null-ne $script:EngineUpdateProbe) 'The background probe starts without showing a prompt'
Check ($script:EngineUpdateProbe.Async.AsyncWaitHandle.WaitOne(5000)) 'The background probe can independently resolve release metadata'
Show-UpdateStatus '1.3.0' -Window
Check ($script:updatePrompts-eq 1 -and $null-eq $script:EngineUpdateProbe) 'The native path consumes the background result and disposes the probe'
Write-Output ('PASS: '+$script:checks+' startup reminder/update policy checks; fixtures only, no network or GPU.')
