[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$EvidenceDirectory)
$ErrorActionPreference='Stop'
if(-not $env:MOE4ALL_WORK_ROOT -or -not $env:MOE4ALL_WIZARD_TEST_TMP){throw 'Use explicit shared task roots'}
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'));$dataDir=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'unused';$script:IsPackagedInstall=$false
. (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
Initialize-DashboardRuntime
$code=@'
using System;using System.Drawing;using System.IO;using System.Reflection;using System.Windows.Forms;using MoE4All.Launcher;
public static class StartupDialogChecks {
 static int checks;
 static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
 static Control Find(Control form,string name){var values=form.Controls.Find(name,true);return values.Length>0?values[0]:null;}
 static Form Make(bool update,bool english,string reason){var type=typeof(NativeStartupPrompts).Assembly.GetType("MoE4All.Launcher.StartupNoticeDialog");return (Form)Activator.CreateInstance(type,BindingFlags.Instance|BindingFlags.NonPublic,null,new object[]{update,english,"1.3.0","1.4.0",update?"Example release notes\n\nA small fixture update.":reason,"https://github.com/Headmaster218/MoE4All/releases/tag/release-1.4.0"},null);}
 public static int Run(string evidence){checks=0;foreach(string theme in new[]{"dark","light"})foreach(bool en in new[]{false,true}){
   WizardTheme.SetPreference(theme);
   foreach(string action in new[]{"UpdateNow","UpdateSkip","UpdateSnooze"})using(var form=Make(true,en,"")){
    form.Opacity=0;form.ShowInTaskbar=false;form.Show();Application.DoEvents();
    Check(Find(form,"ReleaseNoteLink") is LinkLabel,"Release note must be a text link");
    Check(Find(form,"StartupReleaseNotes") is RichTextBox,"Release notes are not visible");
    Check(((Control)form.AcceptButton).Name=="UpdateSkip","Enter must not install an update");
    Check(Find(form,"NativePromptInput")==null,"Update uses a console-style input");
    foreach(string name in new[]{"UpdateNow","UpdateSkip","UpdateSnooze"}){var button=Find(form,name);Check(button.Visible&&button.Width>0&&button.Right<=button.Parent.ClientSize.Width,"Update action is clipped: "+name);}
    if(action=="UpdateNow"&&en&&evidence.Length>0)using(var bitmap=new Bitmap(form.Width,form.Height)){form.DrawToBitmap(bitmap,new Rectangle(0,0,bitmap.Width,bitmap.Height));bitmap.Save(Path.Combine(evidence,"update-dialog-"+theme+".png"));}
    ((Button)Find(form,action)).PerformClick();Application.DoEvents();string value=(string)form.GetType().GetField("Result",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(form);
    Check(value==(action=="UpdateNow"?"update":action=="UpdateSnooze"?"snooze":"skip"),"Wrong native update action");
   }
   foreach(string reason in new[]{"first_use","new_version"})foreach(string action in new[]{"BenchmarkTry","BenchmarkLater"})using(var form=Make(false,en,reason)){
    form.Opacity=0;form.ShowInTaskbar=false;form.Show();Application.DoEvents();Check(Find(form,"BenchmarkInvitation")!=null,"Benchmark invitation missing");
    Check(((Control)form.AcceptButton).Name=="BenchmarkLater","Benchmark defaults to later");Check(Find(form,"NativePromptInput")==null,"Benchmark uses console-style input");
    Check(Find(form,"BenchmarkInvitation").Text.Contains(en?(reason=="new_version"?"version":"device"):(reason=="new_version"?"版本":"设备")),"First-use and upgrade wording are not distinct");
    ((Button)Find(form,action)).PerformClick();Application.DoEvents();string value=(string)form.GetType().GetField("Result",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(form);Check(value==(action=="BenchmarkTry"?"try":"later"),"Wrong benchmark choice");
   }
  }return checks;
 }
}
'@
$source=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'startup-dialog-checks.cs';[IO.File]::WriteAllText($source,$code,[Text.UTF8Encoding]::new($true))
Add-Type -Path $source -ReferencedAssemblies @([MoE4All.Launcher.NativeStartupPrompts].Assembly.Location,'System.Windows.Forms.dll','System.Drawing.dll')
[void][IO.Directory]::CreateDirectory($EvidenceDirectory)
Write-Output ('PASS: '+[StartupDialogChecks]::Run($EvidenceDirectory)+' native startup dialog checks; no downloads, browsers, uploads or GPU work.')
