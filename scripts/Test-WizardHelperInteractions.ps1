[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
if(-not $env:MOE4ALL_WORK_ROOT -or -not $env:MOE4ALL_WIZARD_TEST_TMP){throw 'Task work and tmp paths required.'}
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'));$dataDir=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'unused';$script:IsPackagedInstall=$false
$parseErrors=$null;$tokens=$null;$ast=[Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'infr-wizard.ps1'),[ref]$tokens,[ref]$parseErrors)
if($parseErrors.Count){throw ($parseErrors|Out-String)}
foreach($definition in $ast.FindAll({param($n)$n -is [Management.Automation.Language.FunctionDefinitionAst]},$true)){. ([scriptblock]::Create($definition.Extent.Text))}
. (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
. (Join-Path $PSScriptRoot 'infr-dashboard-state.ps1')
Initialize-DashboardRuntime
$checks=0
function Check([bool]$Condition,[string]$Message){if(-not $Condition){throw $Message};$script:checks++}
$private=[Reflection.BindingFlags]'NonPublic,Static'
try{
 [MoE4All.Launcher.NativePrompt]::BeginHelper('Isolated helper fixture');$script:NativePromptsEnabled=$true
 Write-Warning 'fixture warning visible inside helper'
 Write-Progress -Id 2 -Activity 'Fixture download' -Status '1 MiB / 2 MiB' -PercentComplete 50
 $snapshot=[MoE4All.Launcher.NativePrompt].GetMethod('Snapshot',$private).Invoke($null,@())
 Check ($snapshot -match 'fixture warning') 'Warnings must appear in the native helper'
 $progress=[MoE4All.Launcher.NativePrompt].GetField('progress',$private).GetValue($null)
 Check ($progress -match '1 MiB / 2 MiB') 'Progress must appear without growing the log'
 $helper=[MoE4All.Launcher.NativePrompt].GetField('helper',$private).GetValue($null)
 $close=[Delegate]::CreateDelegate([Windows.Forms.MethodInvoker],$helper,'Close');$helper.Invoke($close)
 $cancelled=$false;try{Assert-WizardOperationActive}catch{$cancelled=Test-WizardOperationCancelled $_.Exception}
 Check $cancelled 'Closing the helper must cancel work'
 $cancelled=$false;try{$null=Invoke-NativeCommandResult -Executable 'never-run-this-fixture.exe' -Arguments @()}catch{$cancelled=Test-WizardOperationCancelled $_.Exception}
 Check $cancelled 'Cancellation must prevent external command execution'
 $cancelled=$false;try{$null=Invoke-Moe4AllSubmissionApi -Method GET -Uri ([Uri]'https://unused.invalid/')}catch{$cancelled=Test-WizardOperationCancelled $_.Exception}
 Check $cancelled 'Cancellation must prevent requests and retries'
}finally{$script:NativePromptsEnabled=$false;[MoE4All.Launcher.NativePrompt]::EndHelper()}

$script:PendingSubmissionPrompted=$true;$script:pendingReads=0;$script:pendingPrompts=0
function Get-PendingSubmissions {$script:pendingReads++;return @([pscustomobject]@{Extension='.m4b';FullName='fixture.m4b'})}
function Read-YesNo {param($Label,$Default)$script:pendingPrompts++;return $false}
function Save-BenchmarkUploadPreferences {}
Offer-PendingSubmissions
Check ($script:pendingReads -eq 0) 'Automatic pending notice remains once per launch'
Offer-PendingSubmissions -ExplicitAction;Offer-PendingSubmissions -ExplicitAction
Check ($script:pendingReads -eq 2 -and $script:pendingPrompts -eq 2) 'Explicit pending actions remain usable after declining'

function Select-RecommendedDownloads {param($Kind)$script:downloadKind=$Kind;return 'fixture/audio.gguf'}
function Read-Host {param($Prompt)$script:returnPrompts++;return ''}
$state=Get-DashboardInitialState -Saved @{model='keep-this-model.gguf'};$script:returnPrompts=0
Invoke-DashboardHelper -Action download -Target transcription_model -State $state
Check ($script:downloadKind -eq 'audio' -and $state.transcription_model -eq 'fixture/audio.gguf') 'Component download uses its own catalog and fills its own field'
Check ($state.model -eq 'keep-this-model.gguf') 'Downloading a component does not replace the LLM'
function Select-RecommendedDownloads {param($Kind)throw [MoE4All.Launcher.NativePromptCanceledException]::new()}
$script:returnPrompts=0;Invoke-DashboardHelper -Action download -Target model -State $state
Check ($script:returnPrompts -eq 0 -and -not $script:NativePromptsEnabled) 'Helper cancellation returns quietly without another prompt or fallback'

$source=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'device-probe-fixture.cs';$fixture=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'DeviceProbeFixture.exe'
[IO.File]::WriteAllText($source,'using System;using System.Threading;class Fixture{static void Main(){if(Environment.GetEnvironmentVariable("MOE4ALL_PROBE_FIXTURE_MODE")=="stall"){Thread.Sleep(12000);return;}Console.Error.Write(new string(''x'',262144));Console.Out.WriteLine("Vulkan0: Fixture GPU [discrete, 2.0 GiB device-local]  <- default");}}',[Text.UTF8Encoding]::new($true))
$compiler=Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
& $compiler /nologo /target:exe ('/out:'+$fixture) $source
if($LASTEXITCODE -ne 0){throw 'Probe fixture compile failed'}
$infrPath=$fixture;$script:VulkanDeviceProbe=$null;$script:VulkanDeviceProbeError=$null
$watch=[Diagnostics.Stopwatch]::StartNew();$devices=@(Get-VulkanDeviceOptions)
Check ($devices.Count -eq 1 -and $devices[0].Value -eq 'Vulkan0' -and $watch.Elapsed.TotalSeconds -lt 6) 'Device probe drains full stderr concurrently'
try{
 $env:MOE4ALL_PROBE_FIXTURE_MODE='stall';$script:VulkanDeviceProbe=$null;$script:VulkanDeviceProbeError=$null
 $watch.Restart();$devices=@(Get-VulkanDeviceOptions)
 Check ($devices.Count -eq 0 -and $watch.Elapsed.TotalSeconds -ge 5 -and $watch.Elapsed.TotalSeconds -lt 9) 'Stalled probe returns control with a bounded wait'
}finally{Remove-Item Env:MOE4ALL_PROBE_FIXTURE_MODE -ErrorAction SilentlyContinue}
Write-Output ('Helper interactions: '+$checks+' checks passed; own fixture processes only, no GPU/network/uploads.')
