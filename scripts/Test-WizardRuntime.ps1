[CmdletBinding()]
param([string]$EvidenceDirectory='')
$ErrorActionPreference='Stop'
if(-not $env:MOE4ALL_WORK_ROOT -or -not $env:MOE4ALL_WIZARD_TEST_TMP){throw 'Set task work and tmp paths first.'}
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'));$dataDir=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'unused';$script:IsPackagedInstall=$false
. (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
Initialize-DashboardRuntime
$compiler=Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$fixtureSource=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'ui-runtime-fixture.cs'
$fixture=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'MoE4AllUiRuntimeFixture.exe'
$code=@'
using System;using System.Text;using System.Threading;using System.Diagnostics;
class UiRuntimeFixture{
 static int Main(string[] a){Console.InputEncoding=new UTF8Encoding(false);Console.OutputEncoding=new UTF8Encoding(false);Console.CancelKeyPress+=delegate(object s,ConsoleCancelEventArgs e){e.Cancel=true;Console.WriteLine("CTRL_C_CAUGHT");Console.Out.Flush();Environment.Exit(0);};
  string mode=a.Length>0?a[0]:"chat";
  if(mode=="args"){for(int i=1;i<a.Length;i++)Console.WriteLine("ARG:"+Convert.ToBase64String(Encoding.UTF8.GetBytes(a[i])));return 0;}
  if(mode=="exit17"){Console.Error.WriteLine("EXPECTED_FAILURE_17");return 17;}
  if(mode=="burst"){for(int i=0;i<18000;i++)Console.WriteLine("bounded log row "+i+" abcdefghijklmnopqrstuvwxyz");return 0;}
  if(mode=="spawn"){var child=Process.Start(new ProcessStartInfo(Process.GetCurrentProcess().MainModule.FileName,"hold"){UseShellExecute=false});Console.WriteLine("CHILD_PID:"+child.Id);return 0;}
  Console.WriteLine("READY_"+mode.ToUpperInvariant());Console.Out.Flush();
  if(mode=="hold"){while(true)Thread.Sleep(30);}
  string line;while((line=Console.ReadLine())!=null){if(line==":q")return 0;Console.WriteLine("USER:"+line);Console.Out.Flush();}return 0;
 }
}
'@
[IO.File]::WriteAllText($fixtureSource,$code,[Text.UTF8Encoding]::new($true))
& $compiler /nologo /optimize+ /target:exe ('/out:'+$fixture) $fixtureSource
if($LASTEXITCODE -ne 0){throw 'Fixture compile failed'}
$harnessSource=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'ui-runtime-harness.cs'
$harness=@'
using System;using System.IO;using System.Drawing;using System.Reflection;using System.Windows.Forms;using MoE4All.Launcher;
public static class UiRuntimeHarness{
 public static string Case(string exe,string mode,string evidence){
  string failure="";int ticks=0,step=0;bool closeCancelled=false;string output="";
  string[] argv=mode=="args"?new[]{mode,"","space value","quote\"value","trailing\\","\u4e16\u754c"}:new[]{mode};
  using(var ui=new RuntimeWindow(exe,argv,Path.GetDirectoryName(exe),"chat","UI test fixture","127.0.0.1:1234",false))using(var timer=new Timer{Interval=80}){
   ui.ShowInTaskbar=false;ui.StartPosition=FormStartPosition.Manual;ui.Location=new Point(-5000,-5000);
   timer.Tick+=delegate{
    try{
     ++ticks;Control[] fields=ui.Controls.Find("RuntimeOutput",true);if(fields.Length==0)return;output=fields[0].Text;
     Form confirm=null;foreach(Form open in Application.OpenForms)if(open.Name=="RuntimeConfirm"){confirm=open;break;}
     if(confirm!=null){string name=mode=="hold"&&step==1?"RuntimeConfirmCancel":"RuntimeConfirmAccept";if(name=="RuntimeConfirmCancel"){step=2;closeCancelled=true;}((Button)confirm.Controls.Find(name,true)[0]).PerformClick();return;}
     if(ticks>250){failure="Timeout: "+mode;var session=ui.GetType().GetField("session",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(ui);session.GetType().GetMethod("Force",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(session,null);}
     if(ui.TaskCompleted&&ui.Controls.Find("RuntimeDone",true)[0].Enabled){
      int expected=mode=="exit17"?17:0;if(ui.ExitCode!=expected&&failure.Length==0)failure="Wrong exit "+ui.ExitCode+" for "+mode;
      if(mode=="chat"&&!output.Contains("USER:hello \u4e16\u754c"))failure="Missing UTF-8 streamed reply";
      if(mode=="args"){foreach(string value in new[]{"space value","quote\"value","trailing\\","\u4e16\u754c"})if(!output.Contains("ARG:"+Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(value))))failure="Windows argv quoting changed input";}
      if(mode=="spawn"&&!output.Contains("CTRL_C_CAUGHT"))failure="Owned child tree did not stop gracefully";
      if(mode=="hold"&&(!closeCancelled||!output.Contains("CTRL_C_CAUGHT")))failure="Graceful cancel/stop was not verified";
      if(output.Length>220000)failure="Transcript is unbounded";
      if(evidence.Length>0){using(var b=new Bitmap(ui.Width,ui.Height)){ui.DrawToBitmap(b,new Rectangle(0,0,b.Width,b.Height));b.Save(Path.Combine(evidence,"runtime-"+mode+".png"));}File.WriteAllText(Path.Combine(evidence,"runtime-"+mode+".txt"),output);}
      timer.Stop();((Button)ui.Controls.Find("RuntimeDone",true)[0]).PerformClick();return;
     }
     if(mode=="chat"&&step==0&&output.Contains("READY_CHAT")){ui.Controls.Find("RuntimeInput",true)[0].Text="hello \u4e16\u754c";((Button)ui.Controls.Find("RuntimeSend",true)[0]).PerformClick();step=1;}
     else if(mode=="chat"&&step==1&&output.Contains("USER:hello ")){ui.Controls.Find("RuntimeInput",true)[0].Text=":q";((Button)ui.Controls.Find("RuntimeSend",true)[0]).PerformClick();step=2;}
     if(mode=="hold"&&step==0&&output.Contains("READY_HOLD")){step=1;ui.Close();}
     else if(mode=="hold"&&step==2){step=3;((Button)ui.Controls.Find("RuntimeStop",true)[0]).PerformClick();}
     if(mode=="spawn"&&step==0&&output.Contains("READY_HOLD")){step=3;((Button)ui.Controls.Find("RuntimeStop",true)[0]).PerformClick();}
    }catch(Exception ex){failure=ex.ToString();var session=ui.GetType().GetField("session",BindingFlags.Instance|BindingFlags.NonPublic).GetValue(ui);session.GetType().GetMethod("Force",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(session,null);}
   };
   timer.Start();Application.Run(ui);
  }
  return failure;
 }
}
'@
[IO.File]::WriteAllText($harnessSource,$harness,[Text.UTF8Encoding]::new($true))
Add-Type -Path $harnessSource -ReferencedAssemblies @([MoE4All.Launcher.NativeRuntime].Assembly.Location,'System.Windows.Forms.dll','System.Drawing.dll')
if($EvidenceDirectory){$full=[IO.Path]::GetFullPath($EvidenceDirectory);$allowed=[IO.Path]::GetFullPath((Join-Path $env:MOE4ALL_WORK_ROOT 'evidence')).TrimEnd('\')+'\';if(-not $full.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Invalid evidence directory'};[void][IO.Directory]::CreateDirectory($full)}
foreach($mode in @('args','chat','exit17','hold','spawn','burst')){
 $failure=[UiRuntimeHarness]::Case($fixture,$mode,$EvidenceDirectory)
 if($failure){throw $failure}
 Write-Output ('PASS native runtime '+$mode)
}
Write-Output 'Native runtime passed: own simulated process only; no GPU, models, network or production services.'
