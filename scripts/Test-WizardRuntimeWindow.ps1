[CmdletBinding()]
param([string]$TaskId='20261105-175000-live-benchmark-f19c')
$ErrorActionPreference='Stop'
Get-Content -LiteralPath 'D:\AIinfr\disk_usage_rule.md' | Out-Null
if(Test-Path -LiteralPath 'D:\cpuandram.lock'){throw 'Diagnostic lock exists; defer window tests.'}
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$workRoot='D:\AIinfr\infr\local-work'
$task=$TaskId
$run='window-'+$PID+'-'+[guid]::NewGuid().ToString('N').Substring(0,8)
function CheckedPath([string]$Path,[string]$Root){
    $full=[IO.Path]::GetFullPath($Path);$allowed=[IO.Path]::GetFullPath($Root).TrimEnd('\')+'\'
    if(-not $full.StartsWith($allowed,[StringComparison]::OrdinalIgnoreCase)){throw 'Output outside task root'}
    $p=$full
    while($p){if(Test-Path -LiteralPath $p){if(((Get-Item -LiteralPath $p -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)-ne 0){throw 'Output traverses a reparse point'}};$p=[IO.Path]::GetDirectoryName($p)}
    $full
}
function Identity([string]$Path){
    $s=[Security.Cryptography.SHA256]::Create();$f=[IO.File]::OpenRead($Path)
    try{$hash=[BitConverter]::ToString($s.ComputeHash($f)).Replace('-','').ToLowerInvariant()}finally{$f.Dispose();$s.Dispose()}
    [ordered]@{path=$Path;sha256=$hash}
}
$tmpRoot=Join-Path $workRoot ('tmp\moe4all\'+$task)
$evidenceRoot=Join-Path $workRoot ('evidence\moe4all\'+$task)
$tmp=CheckedPath (Join-Path $tmpRoot $run) $tmpRoot
$evidence=CheckedPath (Join-Path $evidenceRoot $run) $evidenceRoot
$drive=New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($workRoot))
if($drive.AvailableFreeSpace-lt 64MB){throw 'At least 64 MiB free disk space is required.'}
[void][IO.Directory]::CreateDirectory($tmp);[void][IO.Directory]::CreateDirectory($evidence)
$saved=@{}
foreach($name in @('TEMP','TMP','TMPDIR','MOE4ALL_WORK_ROOT','MOE4ALL_WIZARD_TEST_TMP','MOE4ALL_LAUNCHER_CACHE','INFR_API_KEY')){$saved[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
$manifest=[ordered]@{task=$task;status='running';purpose='CPU-only runtime window layout and close lifecycle';repository=$repoRoot;startedUtc=[DateTime]::UtcNow.ToString('o');temp=$tmp;evidence=$evidence;processId=$PID;command='powershell.exe -NoLogo -NoProfile -NonInteractive -File scripts/Test-WizardRuntimeWindow.ps1';scope='Own lightweight child fixtures, invisible forms, loopback model-list stub; no GPU/models/upload'}
Write-Output ('Output: '+$evidence);Write-Output ('Temp/compiler output: '+$tmp+'; KV/GPU: unused')
try{
    $env:TEMP=$tmp;$env:TMP=$tmp;$env:TMPDIR=$tmp;$env:MOE4ALL_WORK_ROOT=$workRoot;$env:MOE4ALL_WIZARD_TEST_TMP=$tmp
    $env:MOE4ALL_LAUNCHER_CACHE=Join-Path $tmp 'launcher';$env:INFR_API_KEY=$null
    $dataDir=Join-Path $tmp 'unused';$script:IsPackagedInstall=$false
    . (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
    Initialize-DashboardRuntime
    $assembly=[MoE4All.Launcher.RuntimeWindow].Assembly
    if(-not $assembly.Location.StartsWith($env:MOE4ALL_LAUNCHER_CACHE+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Use a fresh powershell.exe process.'}
    $binding=Join-Path $env:MOE4ALL_LAUNCHER_CACHE ($assembly.GetName().Name+'.dll')
    if($binding-ne $assembly.Location){Copy-Item -LiteralPath $assembly.Location -Destination $binding}
    $source=Join-Path $tmp 'RuntimeWindowHarness.cs';$exe=Join-Path $env:MOE4ALL_LAUNCHER_CACHE 'RuntimeWindowHarness.exe'
    $code=@'
using System;using System.IO;using System.Drawing;using System.Collections.Generic;using System.Diagnostics;using System.Net;using System.Net.Sockets;using System.Reflection;using System.Runtime.InteropServices;using System.Text;using System.Threading;using System.Web.Script.Serialization;using System.Windows.Forms;using MoE4All.Launcher;
internal static class RuntimeWindowHarness {
 const BindingFlags All=BindingFlags.Instance|BindingFlags.Public|BindingFlags.NonPublic;
 static string evidence,exe;static int checks,failed;static readonly List<object> results=new List<object>();static readonly JavaScriptSerializer Json=new JavaScriptSerializer();
 [DllImport("user32.dll",EntryPoint="SendMessageW")]static extern IntPtr SendMessage(IntPtr h,int m,IntPtr w,IntPtr l);
 static object Field(object o,string n){return o.GetType().GetField(n,All).GetValue(o);}
 static object Call(object o,string n,params object[] a){try{return o.GetType().GetMethod(n,All).Invoke(o,a);}catch(TargetInvocationException e){throw e.InnerException??e;}}
 static Control Find(Control c,string name){var found=c.Controls.Find(name,true);if(found.Length!=1)throw new Exception("Missing/ambiguous "+name);return found[0];}
 static void Check(bool ok,string name){checks++;if(!ok)throw new Exception(name);}
 static void Pump(int milliseconds){var t=Stopwatch.StartNew();while(t.ElapsedMilliseconds<milliseconds){Application.DoEvents();Thread.Sleep(10);}}
 static void Until(Func<bool> done,string name){var t=Stopwatch.StartNew();while(!done()){if(t.ElapsedMilliseconds>10000)throw new TimeoutException(name);Pump(10);}}
 static void Case(string name,Action action){var t=Stopwatch.StartNew();string error="";try{action();Console.WriteLine("PASS "+name);}catch(Exception e){failed++;error=e.ToString();Console.WriteLine("FAIL "+name+": "+e.Message);}results.Add(new{test=name,passed=error.Length==0,error=error,elapsedMs=t.ElapsedMilliseconds});File.WriteAllText(Path.Combine(evidence,"results.json"),Json.Serialize(results),new UTF8Encoding(true));}
 // Only supplies /v1/models to activate the real chat workspace and its buttons.
 sealed class Models:IDisposable {
  readonly TcpListener listener=new TcpListener(IPAddress.Loopback,0);readonly Thread worker;volatile bool stopped;internal string Address;
  internal Models(){listener.Start();Address="http://127.0.0.1:"+((IPEndPoint)listener.LocalEndpoint).Port+"/v1";worker=new Thread(delegate(){while(!stopped)try{using(var c=listener.AcceptTcpClient())using(var s=c.GetStream()){s.ReadTimeout=2000;var h=new StringBuilder();int b;while((b=s.ReadByte())>=0){h.Append((char)b);if(h.Length>8192||h.ToString().EndsWith("\r\n\r\n"))break;}string body="{\"data\":[{\"id\":\"window-fixture\"}]}";byte[] response=Encoding.ASCII.GetBytes("HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: "+body.Length+"\r\n\r\n"+body);s.Write(response,0,response.Length);}}catch{if(stopped)return;}}){IsBackground=true};worker.Start();}
  public void Dispose(){stopped=true;listener.Stop();worker.Join(2500);}
 }
 sealed class Host:IDisposable {
  internal readonly RuntimeWindow Window;internal readonly object Session;internal bool Closed;internal int Dialogs,Unexpected,Exit;internal string Output="";internal readonly Queue<bool> Answers=new Queue<bool>();
  readonly System.Windows.Forms.Timer timer=new System.Windows.Forms.Timer{Interval=25};readonly HashSet<IntPtr> seen=new HashSet<IntPtr>();
  internal Host(string mode,bool english,string fixture,string endpoint){
   Window=new RuntimeWindow(exe,new[]{"--fixture",fixture},Path.GetDirectoryName(exe),mode,"synthetic-window",endpoint,english){Opacity=0,ShowInTaskbar=false,StartPosition=FormStartPosition.CenterScreen};Session=Field(Window,"session");
   Window.FormClosed+=delegate{Output=Find(Window,"RuntimeOutput").Text;Exit=Window.ExitCode;Closed=true;};
   timer.Tick+=delegate{Form prompt=null;foreach(Form f in Application.OpenForms)if(f.Name=="RuntimeConfirm"&&f.Owner==Window){prompt=f;break;}if(prompt==null||!seen.Add(prompt.Handle))return;Dialogs++;bool yes=false;if(Answers.Count==0)Unexpected++;else yes=Answers.Dequeue();try{((Button)Find(prompt,yes?"RuntimeConfirmAccept":"RuntimeConfirmCancel")).PerformClick();}catch{prompt.DialogResult=DialogResult.Cancel;}};
   timer.Start();Window.Show();Pump(30);
  }
  internal object State(){return Call(Session,"ReadState");}
  internal void Ready(){Until(delegate{return !Closed&&(bool)Field(State(),"Started")&&Find(Window,"RuntimeOutput").Text.Contains("READY_");},"fixture ready");Check((bool)Field(State(),"Pty"),"fixture did not use ConPTY/Ctrl+C");}
  internal void CaptionClose(){var close=Find(Window,"CaptionClose");close.GetType().GetMethod("OnClick",All).Invoke(close,new object[]{EventArgs.Empty});}
  public void Dispose(){timer.Stop();if(!Closed&&!Window.TaskCompleted){Call(Session,"Force");Until(delegate{return Window.TaskCompleted;},"owned fixture cleanup");}if(!Window.IsDisposed)Window.Dispose();timer.Dispose();}
 }
 static void Buttons(Control root,List<Control> found){foreach(Control c in root.Controls){if(c is ButtonBase&&c.Visible)found.Add(c);Buttons(c,found);}}
 static void Layout(Host h,string label){
  var w=h.Window;w.PerformLayout();Pump(25);var buttons=new List<Control>();Buttons(Find(w,"RuntimeContent"),buttons);var data=new List<object>();
  foreach(var b in buttons)data.Add(new{name=b.Name,text=b.Text,bounds=w.RectangleToClient(b.RectangleToScreen(b.ClientRectangle)),parent=b.Parent.Name});
  File.WriteAllText(Path.Combine(evidence,label+"-bounds.json"),Json.Serialize(data),new UTF8Encoding(true));
  Check(buttons.Count>=3,"too few visible actions: "+label);
  for(int i=0;i<buttons.Count;i++){
   var b=buttons[i];Rectangle screen=b.RectangleToScreen(b.ClientRectangle);Check(b.Width>0&&b.Height>0,"empty button "+b.Name);
   for(Control p=b.Parent;p!=null;p=p.Parent)Check(p.RectangleToScreen(p.ClientRectangle).Contains(screen),"button clipped by "+p.Name+": "+b.Name+" / "+label);
   Size text=TextRenderer.MeasureText(b.Text,b.Font,new Size(10000,1000),TextFormatFlags.SingleLine|TextFormatFlags.NoPadding);
   Check(text.Width+8<=b.ClientSize.Width&&text.Height<=b.ClientSize.Height,"button text clipped: "+b.Name+" / "+label);
   for(int j=i+1;j<buttons.Count;j++)Check(!screen.IntersectsWith(buttons[j].RectangleToScreen(buttons[j].ClientRectangle)),"buttons overlap: "+b.Name+" / "+buttons[j].Name);
  }
  var actions=(TableLayoutPanel)Find(w,"RuntimeActions");Check(actions.RowCount==2,"actions are not two rows");Check(Find(w,"RuntimeDisplay").Height>0,"content has no height");
 }
 static void ResizeEdges(RuntimeWindow w){
  Point[] points={new Point(2,w.ClientSize.Height/2),new Point(w.ClientSize.Width-2,w.ClientSize.Height/2),new Point(w.ClientSize.Width/2,2),new Point(w.ClientSize.Width/2,w.ClientSize.Height-2)};int[] expected={10,11,12,15};
  Check(w.ContentHost.Left>=6&&w.ClientSize.Width-w.ContentHost.Right>=6&&w.ClientSize.Height-w.ContentHost.Bottom>=6,"resize band covered by content");
  for(int i=0;i<points.Length;i++){Check(w.GetChildAtPoint(points[i])==null,"child HWND covers resize edge");Point p=w.PointToScreen(points[i]);int packed=(p.X&65535)|((p.Y&65535)<<16);Check(SendMessage(w.Handle,0x0084,IntPtr.Zero,new IntPtr(packed)).ToInt32()==expected[i],"wrong native resize hit "+i);}
  // Use queued message coordinates, deliberately different from Cursor.Position.
  // A real mouse drag can finish moving before its mouse-down is dispatched.
  Rectangle before=w.Bounds;Point down=w.PointToScreen(points[1]);int downPacked=(down.X&65535)|((down.Y&65535)<<16);
  try{SendMessage(w.Handle,0x00A1,new IntPtr(11),new IntPtr(downPacked));Point move=new Point(points[1].X+35,points[1].Y);int packed=(move.X&65535)|((move.Y&65535)<<16);SendMessage(w.Handle,0x0200,new IntPtr(1),new IntPtr(packed));Check(w.Width==before.Width+35,"actual resize ignored queued message coordinates");}
  finally{SendMessage(w.Handle,0x0202,IntPtr.Zero,IntPtr.Zero);w.Bounds=before;}
 }
 static void LiveBenchmark(){
  using(var view=new BenchmarkResultsView(true)){
   Check(!view.HasRows&&view.ExportTsv().StartsWith("Context\tWorkload\tFirst"),"empty table/header");
   Check(!view.ObserveLine("Prefill warmup: 256 evaluated tokens (not scored)."),"warmup was scored");
   Check(view.ObserveLine("Prefill 6144 once, copy the shared KV"),"pending prefill missing");
   Check(view.ObserveLine("Prefill 6144: 6143 evaluated tokens in 2.000s = 3071.5 tok/s. Context slots ready in 2.1s."),"real prefill line not parsed");
   Check(!view.ObserveLine("Prefill 6144 once"),"pending replaced score");
   Check(view.ObserveLine("Decode 6144 x2 rep 1/3..."),"pending decode missing");
   string score="Decode 6144 x2 rep 1/3: first 120.1, middle 121.2, last 122.3, overall 121.2 tok/s";
   Check(view.ObserveLine(score)&&!view.ObserveLine(score),"complete/deduplicate score");
   Check(view.ExportTsv().Contains("6K\tDecode x2 (1/3)\t120.1\t121.2\t122.3\t121.2"),"TSV lost sections");
   Check(view.ObserveLine("Decode 6144 x2 rep 2/3 failed; continuing: expected fixture failure"),"failed result lost");
   view.LoadRows(new object[]{new Dictionary<string,object>{{"context","6K"},{"workload","Decode x2"},{"overall","121.0"}}});
   Check(view.ExportTsv().Contains("Failed: expected fixture failure")&&view.ExportTsv().Contains("121.0"),"final summary discarded failures");
   Check(!view.ObserveLine(score),"raw line changed authoritative final summary");
  }
  using(var h=new Host("benchmark",false,"bench","")){
   h.Ready();Until(delegate{return ((BenchmarkResultsView)Find(h.Window,"BenchmarkResults")).ExportTsv().Contains("3333.3");},"real ConPTY benchmark stdout parsed live");var transcript=Field(h.Session,"Transcript");Type decoderType=transcript.GetType().GetNestedType("Decoder",BindingFlags.NonPublic);object decoder=Activator.CreateInstance(decoderType,All,null,new object[]{transcript,null},null);
   Call(decoder,"Feed","Prefill 6144: 1527.7 tok/s\n"+new string('z',18000)+"\nDecode 6144 x1 rep 1/1: first 120.1, middle 121.2, last 122.3, overall 121.2 tok/s\n");
   Until(delegate{return ((BenchmarkResultsView)Find(h.Window,"BenchmarkResults")).ExportTsv().Contains("121.2");},"live table before final report");
   Check(!h.Window.TaskCompleted&&Find(h.Window,"BenchmarkResults").Visible,"table is not visible during testing");
   var table=(BenchmarkResultsView)Find(h.Window,"BenchmarkResults");Check(table.ExportTsv().Contains("1527.7"),"completed result lost beyond visible log tail");
   ((Button)Find(h.Window,"BenchmarkCopyResults")).PerformClick();Check(Clipboard.GetText(TextDataFormat.UnicodeText)==table.ExportTsv(),"copy button did not copy table TSV");
   ((Button)Find(h.Window,"RuntimeShowLogs")).PerformClick();Check(!table.Visible,"logs did not replace table");((Button)Find(h.Window,"RuntimeShowLogs")).PerformClick();Check(table.Visible&&table.ExportTsv().Contains("1527.7"),"log toggle lost live rows");
   ((Button)Find(h.Window,"RuntimeStop")).PerformClick();Until(delegate{return h.Window.TaskCompleted;},"live benchmark stop");Check(table.Visible&&table.HasRows,"stopping lost partial results");
   h.CaptionClose();Until(delegate{return h.Closed;},"live benchmark close");
  }
 }
 static void Matrix(){using(var models=new Models())foreach(string mode in new[]{"chat","server","benchmark"})foreach(bool english in new[]{false,true}){
  string id=mode+(english?"-en":"-zh");using(var h=new Host(mode,english,"hold",models.Address)){
   h.Ready();if(mode=="chat")Until(delegate{return ((ChatClient)Field(h.Window,"chat")).Read().Ready;},"chat model stub readiness");Pump(150);
   string expectedStop=english?(mode=="chat"?"Stop model":mode=="server"?"Stop service":"Stop benchmark"):(mode=="chat"?"\u505c\u6b62\u6a21\u578b":mode=="server"?"\u505c\u6b62\u670d\u52a1":"\u505c\u6b62\u6d4b\u8bd5");Check(Find(h.Window,"RuntimeStop").Text==expectedStop,"ordinary stop caption/ellipsis wrong: "+id);
   Rectangle area=Screen.FromControl(h.Window).WorkingArea;Check(area.Contains(h.Window.Bounds),"default window exceeds working area: "+id);Layout(h,id+"-default");ResizeEdges(h.Window);
   var toggle=(Button)Find(h.Window,"RuntimeShowLogs");bool canToggle=toggle.Visible&&toggle.Enabled;
   if(canToggle){toggle.PerformClick();string back=english?(mode=="chat"?"Back to chat":mode=="server"?"Back to status":"Back to results"):(mode=="chat"?"\u8fd4\u56de\u5bf9\u8bdd":mode=="server"?"\u8fd4\u56de\u72b6\u6001":"\u8fd4\u56de\u7ed3\u679c");Check(toggle.Text==back,"logs return caption wrong: "+id);Layout(h,id+"-logs-default");}
   else Check(mode=="server","workspace unexpectedly has no log toggle: "+id);
   h.Window.StartPosition=FormStartPosition.Manual;h.Window.Location=new Point(-8000,-8000);h.Window.Size=h.Window.MinimumSize;Layout(h,id+"-minimum");Size small=Find(h.Window,"RuntimeDisplay").Size;
   h.Window.Size=new Size(Math.Min(area.Width-24,h.Window.MinimumSize.Width+180),Math.Min(area.Height-24,h.Window.MinimumSize.Height+100));Layout(h,id+"-grown");Size grown=Find(h.Window,"RuntimeDisplay").Size;
   Check(grown.Width>small.Width&&grown.Height>small.Height,"content did not stretch: "+id);
   if(canToggle){toggle.PerformClick();Check(toggle.Text==(english?"Show logs":"\u67e5\u770b\u65e5\u5fd7"),"return to workspace did not restore log caption: "+id);Layout(h,id+"-workspace-return");}
   int dialogs=h.Dialogs;((Button)Find(h.Window,"RuntimeStop")).PerformClick();Until(delegate{return h.Window.TaskCompleted;},"ordinary stop completes");Pump(120);
   Check(h.Dialogs==dialogs&&h.Unexpected==0,"ordinary stop unexpectedly asked confirmation");Check(!h.Closed&&h.Window.Visible,"ordinary stop closed result window");Check(h.Window.ExitCode==0,"fixture did not stop gracefully");
   foreach(string name in new[]{"RuntimeStop","RuntimeInput","ChatNewConversation","ChatCancelReply"})foreach(Control c in h.Window.Controls.Find(name,true))Check(!c.Visible,"completed window retains obsolete control "+name+" / "+id);
   Check(Find(h.Window,"RuntimeOutput").Text.Contains("CTRL_C_CAUGHT"),"TaskCompleted exposed before final stop log was rendered: "+id);Layout(h,id+"-completed");
   h.CaptionClose();Until(delegate{return h.Closed;},"completed caption X closes directly");Check(h.Dialogs==dialogs,"completed X asked confirmation");
   File.WriteAllText(Path.Combine(evidence,id+"-stop.log"),h.Output,new UTF8Encoding(true));
  }
 }}
 static void RunningClose(){using(var models=new Models())using(var h=new Host("chat",false,"hold",models.Address)){
  h.Ready();h.Window.StartPosition=FormStartPosition.Manual;h.Window.Location=new Point(-8000,-8000);
  h.Answers.Enqueue(false);h.CaptionClose();Pump(100);Check(h.Dialogs==1&&!h.Closed&&!(bool)Field(h.State(),"Stopping"),"cancel X did not preserve running window");
  h.Answers.Enqueue(true);((Button)Find(h.Window,"RuntimeDone")).PerformClick();Until(delegate{return h.Closed;},"confirmed footer close automatically closes after stop");
  Check(h.Dialogs==2&&h.Unexpected==0&&h.Answers.Count==0,"close asked more than once per request");Check(h.Exit==0&&h.Output.Contains("CTRL_C_CAUGHT"),"confirmed close did not stop gracefully");
 }}
 static void NaturalCompletion(){using(var h=new Host("benchmark",true,"exit","")){
  Until(delegate{return h.Window.TaskCompleted;},"natural completion");Check(Find(h.Window,"RuntimeOutput").Text.Contains("FINAL_FIXTURE_OUTPUT"),"TaskCompleted exposed before final exit log was rendered");Check(!h.Closed,"natural completion skipped result review");h.CaptionClose();Until(delegate{return h.Closed;},"natural completion X");Check(h.Dialogs==0&&h.Exit==0,"completed X did not close immediately without dialog");
 }}
 static void ForceRequiresConsent(){using(var h=new Host("server",true,"stubborn","")){
  h.Ready();h.Window.StartPosition=FormStartPosition.Manual;h.Window.Location=new Point(-8000,-8000);((Button)Find(h.Window,"RuntimeStop")).PerformClick();Until(delegate{return (bool)Field(h.State(),"Stopping");},"stopping state");Pump(5700);
  Check(!h.Closed&&!h.Window.TaskCompleted&&h.Dialogs==0,"stubborn task was automatically killed or prompted");
  h.Answers.Enqueue(false);((Button)Find(h.Window,"RuntimeStop")).PerformClick();Check(!h.Window.TaskCompleted&&h.Dialogs==1,"cancel force did not retain task");
  h.Answers.Enqueue(true);((Button)Find(h.Window,"RuntimeStop")).PerformClick();Until(delegate{return h.Window.TaskCompleted;},"confirmed force exits owned fixture");Check(!h.Closed&&h.Dialogs==2&&h.Unexpected==0,"force stop did not retain result window");h.Window.Close();Until(delegate{return h.Closed;},"close forced-complete window");
 }}
 static void AgeStop(Host h){var state=Field(h.Session,"state");state.GetType().GetField("StopTime",All).SetValue(state,DateTime.UtcNow.AddSeconds(-6));Call(h.Window,"TickRuntime");}
 static void StubbornClose(){using(var models=new Models())foreach(string mode in new[]{"chat","server","benchmark"})foreach(bool english in new[]{false,true})using(var h=new Host(mode,english,"stubborn",models.Address)){
  string id=mode+(english?"-en":"-zh");h.Ready();h.Window.StartPosition=FormStartPosition.Manual;h.Window.Location=new Point(-8000,-8000);
  h.Answers.Enqueue(true);h.CaptionClose();Pump(100);Check(!h.Closed&&(bool)Field(h.State(),"Stopping"),"closing must first request graceful stop: "+id);
  h.CaptionClose();Check(h.Dialogs==1,"repeated X forced too early: "+id);Check(!Find(h.Window,"RuntimeDone").Enabled,"close grace period disappeared: "+id);
  AgeStop(h);Check(!h.Window.TaskCompleted&&!h.Closed,"stubborn child was killed automatically: "+id);
  Check(Find(h.Window,"RuntimeDone").Enabled,"closing disabled its escape route: "+id);Check(Find(h.Window,"RuntimeDone").Text==(english?"Force close…":"\u5f3a\u505c\u5e76\u5173\u95ed…"),"force-close action unclear: "+id);Layout(h,id+"-force-close");
  h.Answers.Enqueue(false);h.CaptionClose();Check(!h.Closed&&!h.Window.TaskCompleted&&h.Dialogs==2,"cancel force close killed task: "+id);
  h.Answers.Enqueue(true);((Button)Find(h.Window,"RuntimeDone")).PerformClick();Until(delegate{return h.Closed;},"force close stubborn task: "+id);Check(h.Dialogs==3&&h.Unexpected==0,"force close confirmation count wrong: "+id);
 }}
 static void StoppedThenClose(){using(var h=new Host("benchmark",true,"stubborn","")){
  h.Ready();h.Window.StartPosition=FormStartPosition.Manual;h.Window.Location=new Point(-8000,-8000);((Button)Find(h.Window,"RuntimeStop")).PerformClick();Until(delegate{return (bool)Field(h.State(),"Stopping");},"stubborn stop requested");AgeStop(h);
  h.Answers.Enqueue(true);h.CaptionClose();Until(delegate{return h.Closed;},"X after prior stop must offer force immediately");Check(h.Dialogs==1&&h.Unexpected==0,"X after prior stop required another click");
 }}
 static int Fixture(string mode){Console.OutputEncoding=new UTF8Encoding(false);Console.CancelKeyPress+=delegate(object s,ConsoleCancelEventArgs e){e.Cancel=true;Console.WriteLine("CTRL_C_CAUGHT");Console.Out.Flush();if(mode!="stubborn")Environment.Exit(0);};if(mode=="bench"){Console.WriteLine("Prefill 30720: 30719 evaluated tokens in 9.216s = 3333.3 tok/s. Context slots ready in 9.3s.");Console.WriteLine("Decode 30720 x4 rep 1/1: first 200.1, middle 201.2, last 202.3, overall 201.2 tok/s");}Console.WriteLine("READY_"+mode);Console.Out.Flush();if(mode=="exit"){Thread.Sleep(200);Console.WriteLine("FINAL_FIXTURE_OUTPUT");Console.Out.Flush();return 0;}var t=Stopwatch.StartNew();while(t.ElapsedMilliseconds<60000)Thread.Sleep(20);return 99;}
 [STAThread]static int Main(string[] args){if(args.Length>0&&args[0]=="--fixture")return Fixture(args[1]);evidence=args[0];exe=Assembly.GetExecutingAssembly().Location;Application.EnableVisualStyles();Case("Six layouts: default/minimum/resize + stop retains + completed X",Matrix);Case("Running X cancel + footer confirm closes automatically",RunningClose);Case("Natural completion X closes immediately",NaturalCompletion);Case("No automatic force; force requires confirmation",ForceRequiresConsent);Case("Live benchmark rows, failure retention and clipboard",LiveBenchmark);Case("Stubborn close stays escapable in all workspaces and languages",StubbornClose);Case("Closing an already-stopping task offers force immediately",StoppedThenClose);Console.WriteLine("Runtime window: "+checks+" checks; "+(7-failed)+"/7 groups passed; own CPU fixtures only.");return failed==0?0:1;}
}
'@
    [IO.File]::WriteAllText($source,$code,(New-Object Text.UTF8Encoding $true))
    $compiler=Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    if(-not(Test-Path -LiteralPath $compiler)){$compiler=Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe'}
    $compilerArgs=@('/nologo','/langversion:5','/target:exe','/optimize+','/utf8output','/r:System.dll','/r:System.Core.dll','/r:System.Windows.Forms.dll','/r:System.Drawing.dll','/r:System.Web.Extensions.dll','/r:Accessibility.dll',('/r:'+$assembly.Location),('/out:'+$exe),$source)
    $manifest.inputs=@((Identity $source),(Identity $PSCommandPath),(Identity $assembly.Location),(Identity $compiler));$manifest.compilerArguments=$compilerArgs
    $manifest.commit=(& git -C $repoRoot rev-parse HEAD|Out-String).Trim();$manifest.dirty=@(& git -C $repoRoot status --porcelain)
    $copies=Join-Path $evidence 'sources';[void][IO.Directory]::CreateDirectory($copies)
    foreach($name in @('wizard-console.cs','wizard-theme.cs','wizard-runtime.cs','wizard-desktop.cs','wizard-workspaces.cs','wizard-markdown.cs','wizard-prompts.cs','wizard-downloads.cs','infr-dashboard.ps1','Test-WizardRuntimeWindow.ps1')){$p=Join-Path $PSScriptRoot $name;$manifest.inputs+=Identity $p;Copy-Item -LiteralPath $p -Destination (Join-Path $copies $name)}
    Copy-Item -LiteralPath $source -Destination (Join-Path $evidence 'RuntimeWindowHarness.cs')
    $manifest|ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path $evidence 'manifest.json') -Encoding UTF8
    $log=& $compiler @compilerArgs|Out-String;$compileExit=$LASTEXITCODE;$log|Set-Content -LiteralPath (Join-Path $evidence 'compile.log') -Encoding UTF8
    if($compileExit-ne 0){throw ('Window harness compilation failed: '+$log)}
    $manifest.executable=Identity $exe
    $ErrorActionPreference='Continue'
    try{& $exe $evidence 1> (Join-Path $evidence 'tests.log') 2> (Join-Path $evidence 'tests.err.log');$testExit=$LASTEXITCODE}finally{$ErrorActionPreference='Stop'}
    Get-Content -LiteralPath (Join-Path $evidence 'tests.log')
    if($testExit-ne 0){throw ('Runtime window regression failed: '+$evidence)}
    $manifest.status='passed'
}catch{$manifest.status='failed';$manifest.error=$_.Exception.Message;throw}
finally{$manifest.finishedUtc=[DateTime]::UtcNow.ToString('o');$manifest|ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path $evidence 'manifest.json') -Encoding UTF8;foreach($name in $saved.Keys){[Environment]::SetEnvironmentVariable($name,$saved[$name],'Process')}}
