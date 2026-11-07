[CmdletBinding()]
param()
$ErrorActionPreference='Stop'
if(-not $env:MOE4ALL_WORK_ROOT -or -not $env:MOE4ALL_WIZARD_TEST_TMP){throw 'Set project work and tmp paths first.'}
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'));$dataDir=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'unused';$script:IsPackagedInstall=$false
. (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
Initialize-DashboardRuntime
$code=@'
using System;using System.Reflection;using System.Windows.Forms;using MoE4All.Launcher;
public static class UiPromptHarness{
 public static string Case(bool secret,bool cancel,bool helper){
  Timer timer=null;bool maskVerified=false;string failure="";Form helperWindow=null;
  Action install=delegate{
   timer=new Timer{Interval=80};timer.Tick+=delegate{
    Form dialog=null;foreach(Form f in Application.OpenForms)if(f.Controls.Find("NativePromptInput",true).Length>0){dialog=f;break;}
    if(dialog==null)return;timer.Stop();var input=(TextBox)dialog.Controls.Find("NativePromptInput",true)[0];maskVerified=input.UseSystemPasswordChar==secret;input.Text=secret?"PRIVATE_FIXTURE_VALUE":"n";
    if(cancel)dialog.CancelButton.PerformClick();else dialog.AcceptButton.PerformClick();
   };timer.Start();
  };
  try{
   if(helper){NativePrompt.BeginHelper("Helper fixture");NativePrompt.Append("Choice A or B\n");helperWindow=(Form)typeof(NativePrompt).GetField("helper",BindingFlags.NonPublic|BindingFlags.Static).GetValue(null);helperWindow.Invoke((MethodInvoker)delegate{install();});}else install();
   try{string value=NativePrompt.Read("Fixture question",secret);if(cancel)failure="Cancellation did not stop the prompt";else if(value!=(secret?"PRIVATE_FIXTURE_VALUE":"n"))failure="Prompt lost input";}
   catch(NativePromptCanceledException){if(!cancel)failure="Unexpected cancellation";}
   if(!maskVerified)failure="Secret masking mismatch";
  }finally{if(timer!=null){if(helper&&helperWindow!=null&&!helperWindow.IsDisposed)helperWindow.Invoke((MethodInvoker)delegate{timer.Dispose();});else timer.Dispose();}if(helper)NativePrompt.EndHelper();}
  return failure;
 }
}
'@
$source=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'ui-prompt-harness.cs';[IO.File]::WriteAllText($source,$code,[Text.UTF8Encoding]::new($true))
Add-Type -Path $source -ReferencedAssemblies @([MoE4All.Launcher.NativePrompt].Assembly.Location,'System.Windows.Forms.dll','System.Drawing.dll')
foreach($case in @(@($false,$false,$false),@($true,$false,$false),@($false,$true,$false),@($false,$false,$true),@($true,$true,$true))){$failure=[UiPromptHarness]::Case($case[0],$case[1],$case[2]);if($failure){throw $failure}}
Write-Output 'Native prompts: 5 input, secure-input, cancel and helper-thread cases passed; no external actions.'
