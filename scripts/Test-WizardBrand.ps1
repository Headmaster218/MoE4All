[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$LauncherPath,[Parameter(Mandatory=$true)][string]$EvidenceDirectory)
$ErrorActionPreference='Stop'
if(-not $env:MOE4ALL_WORK_ROOT -or -not $env:MOE4ALL_WIZARD_TEST_TMP){throw 'Set the project work root and task tmp first.'}
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'));$dataDir=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'unused';$script:IsPackagedInstall=$false
. (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
. (Join-Path $PSScriptRoot 'infr-dashboard-state.ps1')
Initialize-DashboardRuntime
[void][IO.Directory]::CreateDirectory($EvidenceDirectory)
$checks=0
function Check([bool]$Ok,[string]$Message){if(-not $Ok){throw $Message};$script:checks++}
$ico=Join-Path $PSScriptRoot 'moe4all.ico'
Check ((Get-Item -LiteralPath $ico).Length -lt 16KB) 'The shared icon should stay below 16 KiB'
$stream=[IO.File]::OpenRead($ico);$reader=[IO.BinaryReader]::new($stream)
try{
    Check ($reader.ReadUInt16()-eq0-and$reader.ReadUInt16()-eq1-and$reader.ReadUInt16()-eq6) 'The ICO must contain six resolutions'
    foreach($size in @(16,24,32,48,64,256)){$encoded=if($size-eq256){0}else{$size};Check ($reader.ReadByte()-eq$encoded-and$reader.ReadByte()-eq$encoded) 'The ICO resolution directory is incomplete';[void]$reader.ReadBytes(14)}
}finally{$reader.Dispose();$stream.Dispose()}
# .NET Framework's Icon selector ignores 0/256 directory dimensions. Explorer
# uses the 256px PNG; window/taskbar handles use the explicit smaller frames.
foreach($size in @(16,24,32,48,64)){$icon=[MoE4All.Launcher.WizardBrand]::CreateIcon($size);try{Check ($icon.Width-eq$size-and$icon.Height-eq$size) 'The requested window icon resolution is missing'}finally{$icon.Dispose()}}
$shell=[Drawing.Icon]::ExtractAssociatedIcon($LauncherPath)
try{$image=$shell.ToBitmap();try{$image.Save((Join-Path $EvidenceDirectory 'exe-icon.png'))}finally{$image.Dispose()};Check ($shell.Handle-ne[IntPtr]::Zero) 'EXE shell icon is unavailable'}finally{$shell.Dispose()}
foreach($theme in @('dark','light')){
    $state=Get-DashboardInitialState -Saved @{wizard_theme=$theme;wizard_language='en';launch_mode='chat'}
    $schema=Get-DashboardSchema -State $state
    $ui=[MoE4All.Launcher.NativeWizard]::new(($state|ConvertTo-Json -Depth 25 -Compress),($schema|ConvertTo-Json -Depth 25 -Compress),'1.3.0',$false,'')
    try{
        $ui.Opacity=0;$ui.ShowInTaskbar=$false;$ui.Show();[Windows.Forms.Application]::DoEvents()
        Check ($ui.ShowIcon-and$null-ne$ui.Icon-and$ui.Icon.Handle-ne[IntPtr]::Zero) 'The main window has no application icon'
        $link=@($ui.Controls.Find('WebsiteLink',$true))[0]
        Check ($link-is[Windows.Forms.LinkLabel]-and$link.Visible-and$link.AccessibleDescription-eq'https://moe4all.com/') 'The home website link is missing or points elsewhere'
        Check ($link.Bottom-le$link.Parent.ClientSize.Height) 'The website link is clipped'
        $image=[Drawing.Bitmap]::new($ui.Width,$ui.Height);try{$ui.DrawToBitmap($image,$ui.ClientRectangle);$image.Save((Join-Path $EvidenceDirectory ('home-brand-'+$theme+'.png')))}finally{$image.Dispose()}
        $child=[MoE4All.Launcher.WizardWindow]::new();try{Check ($child.ShowIcon-and$child.Icon.Handle-ne[IntPtr]::Zero) 'Child windows do not inherit the brand icon'}finally{$child.Dispose()}
    }finally{$ui.Dispose()}
}
Write-Output ('PASS: '+$checks+' launcher branding checks; no browser, engine, GPU or user-state changes.')
