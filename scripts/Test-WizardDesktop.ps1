[CmdletBinding()]
param([string]$EvidenceDirectory = '')
$ErrorActionPreference='Stop'
if (-not $env:MOE4ALL_WORK_ROOT -or -not $env:MOE4ALL_WIZARD_TEST_TMP) { throw 'Set the project work root and task tmp directory first.' }
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$dataDir=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'unused-user-state'
$script:IsPackagedInstall=$false
. (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
. (Join-Path $PSScriptRoot 'infr-dashboard-state.ps1')
Initialize-DashboardRuntime
$checks=0
function Check([bool]$Value,[string]$Name){if(-not $Value){throw $Name};$script:checks++}
function Json($Value){ConvertTo-Json -InputObject $Value -Depth 25 -Compress}
function MakeUi($Overrides){
 $saved=@{wizard_language='zh';launch_mode='chat';model='models\example.gguf';device='Vulkan1';setup_mode='aggressive';context='30k';server_parallel='2'}
 foreach($key in $Overrides.Keys){$saved[$key]=$Overrides[$key]}
 $s=Get-DashboardInitialState -Saved $saved
 $schema=Get-DashboardSchema -State $s -DeviceOptions @([pscustomobject]@{value='Vulkan1';label='AMD Radeon RX 7900 XTX'})
 return [MoE4All.Launcher.NativeWizard]::new((Json $s),(Json $schema),'1.2.0',$false,'')
}
function Find($Ui,[string]$Name){@($Ui.Controls.Find($Name,$true))|Select-Object -First 1}
$private=[Reflection.BindingFlags]'Instance,NonPublic'
function Call($Ui,[string]$Method,[object[]]$Values=@()){$Ui.GetType().GetMethod($Method,$private).Invoke($Ui,$Values)}
function Capture($Ui,[string]$Name){
 if(-not $EvidenceDirectory){return}
 $resolved=[IO.Path]::GetFullPath($EvidenceDirectory)
 $allowed=[IO.Path]::GetFullPath((Join-Path $env:MOE4ALL_WORK_ROOT 'evidence')).TrimEnd('\')+'\'
 if(-not $resolved.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Evidence must be under local-work/evidence.'}
 [void][IO.Directory]::CreateDirectory($resolved)
 $Ui.ShowInTaskbar=$false;$Ui.StartPosition=[Windows.Forms.FormStartPosition]::Manual;$Ui.Location=[Drawing.Point]::new(-5000,-5000)
 $Ui.Show();[Windows.Forms.Application]::DoEvents();$Ui.PerformLayout();$bitmap=[Drawing.Bitmap]::new($Ui.Width,$Ui.Height)
 $layout=@(foreach($controlName in @('CaptionBar','TopBar','Appearance','Language','FooterBar','MoreSettings','Tools','PrimaryAction')){foreach($control in $Ui.Controls.Find($controlName,$true)){[pscustomobject]@{name=$controlName;visible=$control.Visible;bounds=$control.Bounds.ToString();text=$control.Text}}})
 [IO.File]::WriteAllText((Join-Path $resolved ($Name+'-layout.json')),($layout|ConvertTo-Json -Depth 4),[Text.UTF8Encoding]::new($false))
 try{$Ui.DrawToBitmap($bitmap,[Drawing.Rectangle]::new(0,0,$Ui.Width,$Ui.Height));$bitmap.Save((Join-Path $resolved ($Name+'.png')),[Drawing.Imaging.ImageFormat]::Png)}finally{$bitmap.Dispose();$Ui.Hide()}
}
$ui=MakeUi @{}
try{
 Check ($null -ne (Find $ui 'Task_chat')) 'Home offers chat'
 Check ($null -ne (Find $ui 'Task_server')) 'Home offers application service'
 Check ($null -ne (Find $ui 'Task_benchmark')) 'Home offers speed testing'
 Check ($null -eq (Find $ui 'context')) 'Home does not expose technical parameters'
 Capture $ui 'native-home'
 [MoE4All.Launcher.WizardTheme]::SetPreference('light');$ui.RefreshTheme();Capture $ui 'native-home-light'
 Check ([MoE4All.Launcher.WizardTheme]::Current.Background.ToArgb() -eq [Drawing.ColorTranslator]::FromHtml('#fafbfc').ToArgb()) 'Light theme uses the Agent palette'
 [MoE4All.Launcher.WizardTheme]::SetPreference('dark');$ui.RefreshTheme();Capture $ui 'native-home-dark'
 Check ([MoE4All.Launcher.WizardTheme]::Current.Background.ToArgb() -eq [Drawing.ColorTranslator]::FromHtml('#1e201e').ToArgb()) 'Dark theme uses the Agent forest palette'
 Check ((Find $ui 'CaptionClose').Parent -eq $ui.CaptionBar) 'Window uses its own caption controls'
 [MoE4All.Launcher.WizardTheme]::SetPreference('system');$ui.RefreshTheme()
 $ui.SelectTask('benchmark')
 Check ($ui.StateValue('launch_mode') -eq 'benchmark') 'Explicit task choice switches task'
 Check ($ui.StateValue('context') -eq '30k') 'Task switch retains configuration'
 $choice=Find $ui 'bench_kind';$choice.SelectedIndex=1
 Check ($ui.StateValue('bench_parallel') -eq '2') 'Standard plan keeps existing two-lane policy'
 $choice.SelectedIndex=2
 Check ($ui.StateValue('bench_parallel') -eq '4') 'Full plan keeps four-lane policy'
 $null=Call $ui 'Rebuild'
 Capture $ui 'native-benchmark'
 $ui.SelectTask('server')
 $model=Find $ui 'model';$model.Text='models\中文 model.gguf'
 Check ($ui.StateValue('model') -eq 'models\中文 model.gguf') 'Typed Chinese and spaced paths persist immediately'
 Capture $ui 'native-service'
 foreach($mode in @('dark','light')) {
  [MoE4All.Launcher.WizardTheme]::SetPreference($mode);$ui.RefreshTheme()
  foreach($key in @('model','device','setup_mode')) {
   $selector=Find $ui $key
   Check ($selector -is [MoE4All.Launcher.WizardComboBox]) ('Shared themed selector: '+$key)
   $bitmap=[Drawing.Bitmap]::new($selector.Width,$selector.Height)
   try {
    $selector.DrawToBitmap($bitmap,[Drawing.Rectangle]::new(0,0,$bitmap.Width,$bitmap.Height))
    $arrow=$bitmap.GetPixel($bitmap.Width-7,[int]($bitmap.Height/2))
    $surface=[MoE4All.Launcher.WizardTheme]::Current.Surface
    Check ([Math]::Abs($arrow.R-$surface.R)+[Math]::Abs($arrow.G-$surface.G)+[Math]::Abs($arrow.B-$surface.B) -lt 30) ('Dropdown arrow surface follows '+$mode+': '+$key)
   }finally{$bitmap.Dispose()}
  }
  Capture $ui ('native-service-'+$mode)
 }
 [MoE4All.Launcher.WizardTheme]::SetPreference('system');$ui.RefreshTheme()
 $device=Find $ui 'device';$frame=Find $ui 'TopBar';$toggleWatch=[Diagnostics.Stopwatch]::StartNew()
 foreach($i in 1..20){$switch=Find $ui 'server_embedding';$switch.Checked=-not $switch.Checked}
 $toggleWatch.Stop()
 Check ([object]::ReferenceEquals($model,(Find $ui 'model')) -and -not $model.IsDisposed) 'Toggling services preserves model input and focus'
 Check ([object]::ReferenceEquals($device,(Find $ui 'device')) -and [object]::ReferenceEquals($frame,(Find $ui 'TopBar'))) 'Toggling services never rebuilds the page or chrome'
 if($EvidenceDirectory){[IO.File]::WriteAllText((Join-Path $EvidenceDirectory 'ui-toggle-latency.json'),(@{transitions=20;elapsed_ms=$toggleWatch.Elapsed.TotalMilliseconds;ms_per_transition=($toggleWatch.Elapsed.TotalMilliseconds/20);retained_controls=$true}|ConvertTo-Json),[Text.UTF8Encoding]::new($false))}
 (Find $ui 'server_llm_enabled').Checked=$false
 (Find $ui 'server_transcription').Checked=$true
 Check (-not $ui.IsFieldVisible('model')) 'Audio-only service does not require LLM model'
 Check (-not $ui.IsFieldVisible('device')) 'Audio-only service hides unused GPU settings'
 Check ($null -ne (Find $ui 'transcription_model')) 'Audio-only service exposes audio model'
 $r=$ui.Result('save')|ConvertFrom-Json
 Check ($r.state.launch_mode -eq 'server' -and -not $r.state.server_llm_enabled -and $r.state.server_transcription) 'Independent service state survives handoff'
 Capture $ui 'native-transcription'
}finally{$ui.Dispose()}
$ui=MakeUi @{server_api_key='PRIVATE_NATIVE_TEST';last_command='PRIVATE_COMMAND_TEST';server_auth=$true;launch_mode='server'}
try{
 # Ephemeral fields are injected by the bridge after state normalization.
 $stateField=$ui.GetType().GetField('state',$private);$values=$stateField.GetValue($ui);$values['server_api_key']='PRIVATE_NATIVE_TEST';$values['last_command']='PRIVATE_COMMAND_TEST'
 $r=$ui.Result('save')|ConvertFrom-Json
 Check ($r.state.PSObject.Properties.Name -notcontains 'server_api_key') 'API key never enters exportable state'
 Check ($r.state.PSObject.Properties.Name -notcontains 'last_command') 'Last command never enters exportable state'
 Check ($r.secrets.server_api_key -eq 'PRIVATE_NATIVE_TEST') 'API key stays in ephemeral channel'
 $before=$ui.StateValue('launch_mode')
 # Exercise real settings dialog controls using an in-process timer, no desktop input injection.
 $script:settingsChecks=@{}
 $timer=[Windows.Forms.Timer]::new();$timer.Interval=100
 $timer.add_Tick({
  $form=$ui.GetType().GetField('settings',$private).GetValue($ui)
  if($null -eq $form){return}
  $timer.Stop()
  $tabs=@($form.Controls.Find('SettingsContent',$true))[0]
  $script:settingsChecks.tabs=$tabs.TabPages.Count
  $script:settingsChecks.keys=@($tabs.Controls.Find('server_api_key',$true)).Count
  $field=@($tabs.Controls.Find('session_idle_secs',$true))[0];$field.Text='240';$script:retiredEditor=$field
  $script:settingsChecks.masked=@($tabs.Controls.Find('server_api_key',$true))[0].UseSystemPasswordChar
  $secret=@($tabs.Controls.Find('server_api_key',$true))[0];$auth=@($tabs.Controls.Find('server_auth',$true))[0]
  $auth.Checked=$false;$auth.Checked=$true
  $script:settingsChecks.reused=[object]::ReferenceEquals($secret,@($tabs.Controls.Find('server_api_key',$true))[0]) -and -not $secret.IsDisposed
  if($EvidenceDirectory){$bitmap=[Drawing.Bitmap]::new($form.Width,$form.Height);try{$form.DrawToBitmap($bitmap,[Drawing.Rectangle]::new(0,0,$form.Width,$form.Height));$bitmap.Save((Join-Path $EvidenceDirectory 'native-settings.png'),[Drawing.Imaging.ImageFormat]::Png)}finally{$bitmap.Dispose()}}
  if($EvidenceDirectory){
   foreach($theme in @('dark','light')) {
    [MoE4All.Launcher.WizardTheme]::SetPreference($theme);$form.RefreshTheme()
    foreach($tab in $tabs.TabPages){
     $tabs.SelectedTab=$tab;[Windows.Forms.Application]::DoEvents()
     $bitmap=[Drawing.Bitmap]::new($form.Width,$form.Height)
     try{$form.DrawToBitmap($bitmap,[Drawing.Rectangle]::new(0,0,$form.Width,$form.Height));$bitmap.Save((Join-Path $EvidenceDirectory ('native-settings-'+$theme+'-'+$tab.Name+'.png')),[Drawing.Imaging.ImageFormat]::Png)}finally{$bitmap.Dispose()}
    }
   }
   [MoE4All.Launcher.WizardTheme]::SetPreference('system')
  }
  $form.Close() # Cancel changes.
 })
 $timer.Start();try{$ui.OpenSettings('api','')}finally{$timer.Dispose()}
 Check ($settingsChecks.tabs -eq 1) 'Each page action opens only its own settings'
 Check ($settingsChecks.keys -eq 1 -and $settingsChecks.masked) 'Access key uses masked native input'
 Check $settingsChecks.reused 'Dependent advanced settings update without replacing controls or losing input'
 Check ($ui.StateValue('server_parallel') -eq '2') 'Cancel restores modal settings changes'
 Check ($ui.StateValue('launch_mode') -eq $before) 'Browsing settings never changes task'
 $tooltip=$ui.GetType().GetField('tips',$private).GetValue($ui)
 Check ($tooltip.GetToolTip($script:retiredEditor) -eq '') 'Disposed settings controls no longer remain registered in tooltips'
 $timer=[Windows.Forms.Timer]::new();$timer.Interval=100
 $timer.add_Tick({
  $form=$ui.GetType().GetField('settings',$private).GetValue($ui)
  if($null -eq $form){return};$timer.Stop()
  $field=@($form.Controls.Find('session_idle_secs',$true))[0];$field.Text='240'
  $form.AcceptButton.PerformClick()
 })
 $timer.Start();try{$ui.OpenSettings('api','')}finally{$timer.Dispose()}
 Check ($ui.StateValue('session_idle_secs') -eq '240') 'Done commits modal settings'
}finally{$ui.Dispose()}
$modelFixture=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP ('native-model-'+[guid]::NewGuid().ToString('N')+'.gguf')
[IO.File]::WriteAllBytes($modelFixture,[byte[]]@())
$ui=MakeUi @{model=$modelFixture;reasoning_probe_model='old-model.gguf'}
try{
 $ui.OpenSettings('runtime','reasoning_effort')
 $response=$ui.Result($ui.GetType().GetField('action',$private).GetValue($ui))|ConvertFrom-Json
 Check ($response.action -eq 'model' -and $response.settings_page -eq 'runtime' -and $response.settings_field -eq 'reasoning_effort') 'Typed model paths refresh metadata before opening reasoning choices and preserve the intended destination'
}finally{$ui.Dispose()}
foreach($size in @(@(820,650),@(1000,770),@(1280,900))){
 $ui=MakeUi @{}
 try{
  $ui.Size=[Drawing.Size]::new($size[0],$size[1]);$ui.PerformLayout()
  $top=Find $ui 'TopBar';$footer=Find $ui 'FooterBar';$body=Find $ui 'ContentArea'
  Check ($body.Top -ge $top.Bottom -and $body.Bottom -le $footer.Top) 'Scrolling content reserves header and footer'
  foreach($name in @('Task_chat','Task_server','Task_benchmark','PrimaryAction','Tools')){
   $control=Find $ui $name
   Check ($control.Width -gt 70 -and $control.Height -gt 20) ($name+' remains usable at '+$size[0])
   Check ($control.Left -ge 0 -and $control.Right -le $control.Parent.ClientSize.Width) ($name+' fits horizontally at '+$size[0])
  }
  Capture $ui ('native-home-'+$size[0])
 }finally{$ui.Dispose()}
}
# Root windows must reopen after save/helper actions, without an old modal quit closing them.
$state=Get-DashboardInitialState -Saved @{wizard_language='zh';model='fixture.gguf'}
$schema=Get-DashboardSchema -State $state
$script:rootStep=0;$timer=[Windows.Forms.Timer]::new();$timer.Interval=100
$timer.add_Tick({
 $window=@([Windows.Forms.Application]::OpenForms|Where-Object{$_ -is [MoE4All.Launcher.NativeWizard]})|Select-Object -First 1
 if(-not $window){return}
 $next=if($script:rootStep -eq 0){'save'}else{'cancel'}
 $script:rootStep++
 $null=$window.GetType().GetMethod('Finish',$private).Invoke($window,[object[]]@($next))
})
$timer.Start()
try{
 $first=[MoE4All.Launcher.NativeWizard]::Run((Json $state),(Json $schema),'test',$false,'')|ConvertFrom-Json
 $second=[MoE4All.Launcher.NativeWizard]::Run((Json $state),(Json $schema),'test',$false,'')|ConvertFrom-Json
 Check ($first.action -eq 'save' -and $second.action -eq 'cancel' -and $script:rootStep -eq 2) 'Save returns and the next root window stays active until explicitly closed'
}finally{$timer.Stop();$timer.Dispose()}
# Repeated create/switch/dispose checks native resource lifecycle without inference.
$process=[Diagnostics.Process]::GetCurrentProcess()
foreach($round in 1..2){
 for($i=0;$i -lt 12;$i++){$ui=MakeUi @{};try{$ui.SelectTask('server');$ui.SelectTask('benchmark');$null=$ui.Result('save')}finally{$ui.Dispose()}}
 [Windows.Forms.Application]::DoEvents();[GC]::Collect();[GC]::WaitForPendingFinalizers();$process.Refresh()
 if($round -eq 1){$startHandles=$process.HandleCount}
}
$handleGrowth=$process.HandleCount-$startHandles
Check ($handleGrowth -lt 40) ('Repeated window rebuild/disposal has bounded handle growth; delta='+$handleGrowth)
Write-Output "Native launcher: $checks checks passed; own in-process forms only, no engine/GPU/network/user-state access."
