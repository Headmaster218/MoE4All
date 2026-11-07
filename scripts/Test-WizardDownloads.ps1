[CmdletBinding()]
param([string]$EvidenceDirectory='')
$ErrorActionPreference='Stop'
if(-not $env:MOE4ALL_WORK_ROOT -or -not $env:MOE4ALL_WIZARD_TEST_TMP){throw 'Set project work and tmp paths first.'}
$repoRoot=[IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'));$dataDir=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'unused';$script:IsPackagedInstall=$false
. (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
. (Join-Path $PSScriptRoot 'infr-dashboard-state.ps1')
Initialize-DashboardRuntime
$code=@'
using System;using System.Collections.Generic;using System.Drawing;using System.IO;using System.Net;using System.Net.Sockets;using System.Reflection;using System.Text;using System.Threading;using System.Windows.Forms;using MoE4All.Launcher;
public static class DownloadChecks {
 static int checks;static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
 static Control Find(Control c,string name){var found=c.Controls.Find(name,true);return found.Length==0?null:found[0];}
 static void Until(Func<bool> ready,string message){var clock=System.Diagnostics.Stopwatch.StartNew();while(!ready()){Application.DoEvents();Thread.Sleep(10);if(clock.ElapsedMilliseconds>8000)throw new Exception(message);}Application.DoEvents();}
 public static int TransferControls(string root){checks=0;Directory.CreateDirectory(root);
  Application.ThreadException+=delegate(object sender,ThreadExceptionEventArgs e){Console.Error.WriteLine("Transfer fixture UI error: "+e.Exception);};
  byte[] bytes=new byte[1048576];for(int i=0;i<bytes.Length;i++)bytes[i]=(byte)(i%251);
  var listener=new TcpListener(IPAddress.Loopback,0);listener.Start();int port=((IPEndPoint)listener.LocalEndpoint).Port;bool resumed=false;Exception serverError=null;
  var server=new Thread(delegate(){try{for(int request=0;request<3;request++)using(var client=listener.AcceptTcpClient())using(var stream=client.GetStream()){
   client.ReceiveTimeout=5000;var reader=new StreamReader(stream,Encoding.ASCII,false,1024,true);string line;int offset=0;
   while(!String.IsNullOrEmpty(line=reader.ReadLine()))if(line.StartsWith("Range: bytes=")){offset=Int32.Parse(line.Substring(13).Split('-')[0]);resumed=offset>0;}
   var header=Encoding.ASCII.GetBytes("HTTP/1.1 "+(offset>0?"206 Partial Content":"200 OK")+"\r\nContent-Length: "+(bytes.Length-offset)+"\r\n"+(offset>0?"Content-Range: bytes "+offset+"-"+(bytes.Length-1)+"/"+bytes.Length+"\r\n":"")+"Connection: close\r\n\r\n");
   try{stream.Write(header,0,header.Length);for(int at=offset;at<bytes.Length;at+=8192){stream.Write(bytes,at,Math.Min(8192,bytes.Length-at));stream.Flush();Thread.Sleep(4);}}catch(IOException){}
  }}catch(Exception e){serverError=e;}});server.IsBackground=true;server.Start();
  var items=new List<ModelBundle>();foreach(string id in new[]{"resume","cancel"}){var item=new ModelBundle{Id=id,Name=id,Kind="llm",Primary=id+".gguf"};item.Files.Add(new ModelFile{Url="http://127.0.0.1:"+port+"/"+id,RelativePath=item.Primary,Bytes=bytes.Length});items.Add(item);}
  try{using(var window=new WizardWindow{Font=new Font("Microsoft YaHei UI",10),ClientSize=new Size(820,550),ShowInTaskbar=false,Opacity=0}){
   var view=new ModelDownloadsView(items,root,"llm",true,false,delegate{}){Dock=DockStyle.Fill};window.ContentHost.Controls.Add(view);window.Show();Application.DoEvents();
   var pause=(Button)Find(view,"PauseDownload");var cancel=(Button)Find(view,"CancelDownload");var footer=Find(view,"DownloadFooter");
   Check(pause!=null&&cancel!=null&&footer!=null,"Transfer controls were not constructed");
   Check(footer.Bottom<=view.ClientSize.Height&&pause.Visible&&cancel.Visible,"Transfer actions are outside the viewport");
   Check(Find(view,"Get_resume")!=null,"Resume fixture is missing from the catalog");
   ((Button)Find(view,"Get_resume")).PerformClick();
   Until(delegate{return File.Exists(Path.Combine(root,"resume.gguf.part"))&&new FileInfo(Path.Combine(root,"resume.gguf.part")).Length>0;},"Transfer did not start");
   pause.PerformClick();Until(delegate{return pause.Enabled&&pause.Text=="Resume";},"Pause did not settle promptly");
   long retained=new FileInfo(Path.Combine(root,"resume.gguf.part")).Length;Thread.Sleep(100);Application.DoEvents();
   Check(view.Busy&&new FileInfo(Path.Combine(root,"resume.gguf.part")).Length==retained,"Paused transfer keeps writing or releases its active bundle");
   Check(retained>0&&retained<bytes.Length,"Pause did not retain a partial transfer");
   pause.PerformClick();Until(delegate{return !view.Busy&&File.Exists(Path.Combine(root,"resume.gguf"));},"Resume did not finish");
   Check(resumed,"Resume did not request the saved byte range");Check(Convert.ToBase64String(File.ReadAllBytes(Path.Combine(root,"resume.gguf")))==Convert.ToBase64String(bytes),"Resume changed model content");
   Check(Find(view,"Get_cancel")!=null,"Cancel fixture disappeared after download completion: "+Find(view,"DownloadStatus").Text+"; cards="+((WizardScrollView)Find(view,"CatalogList")).Controls.Count);
   ((Button)Find(view,"Get_cancel")).PerformClick();Until(delegate{return File.Exists(Path.Combine(root,"cancel.gguf.part"))&&new FileInfo(Path.Combine(root,"cancel.gguf.part")).Length>0;},"Cancel fixture did not start");
   cancel.PerformClick();Until(delegate{return !view.Busy&&Find(view,"DownloadStatus").Text.StartsWith("Cancelled.");},"Cancel did not release the page");
   Check(File.Exists(Path.Combine(root,"cancel.gguf.part"))&&!File.Exists(Path.Combine(root,"cancel.gguf")),"Cancel lost the partial file or installed an incomplete model");
   Check(Find(view,"ChooseDownloadDirectory").Enabled&&!pause.Enabled&&!cancel.Enabled,"Idle controls were not restored");
   Check(server.Join(5000),"Transfer control fixture did not stop");if(serverError!=null)throw serverError;
  }}finally{listener.Stop();}return checks;
 }
 public static int Network(string root){checks=0;Directory.CreateDirectory(root);
  foreach(string bad in new[]{"../outside.gguf","C:\\outside.gguf","sub/../../outside.gguf"}){bool denied=false;try{ModelDownload.PathUnder(root,bad);}catch(IOException){denied=true;}Check(denied,"Unsafe model path accepted");}
  var listener=new TcpListener(IPAddress.Loopback,0);listener.Start();int port=((IPEndPoint)listener.LocalEndpoint).Port;Exception serverError=null;bool range=false;byte[] payload=new byte[262144];for(int i=0;i<payload.Length;i++)payload[i]=(byte)(i%251);
  var server=new Thread(delegate(){try{for(int n=0;n<2;n++){using(var client=listener.AcceptTcpClient())using(var stream=client.GetStream()){
   client.ReceiveTimeout=5000;var reader=new StreamReader(stream,Encoding.ASCII,false,1024,true);string line;int start=0;while(!String.IsNullOrEmpty(line=reader.ReadLine()))if(line.StartsWith("Range: bytes=")){start=Int32.Parse(line.Substring(13).Split('-')[0]);range=start==payload.Length/2;}
   int length=payload.Length-start;byte[] header=Encoding.ASCII.GetBytes("HTTP/1.1 "+(n==0?"200 OK":"206 Partial Content")+"\r\nContent-Type: application/octet-stream\r\nContent-Length: "+length+"\r\n"+(n==0?"":"Content-Range: bytes "+start+"-"+(payload.Length-1)+"/"+payload.Length+"\r\n")+"Connection: close\r\n\r\n");stream.Write(header,0,header.Length);stream.Write(payload,start,n==0?payload.Length/2:length);stream.Flush();
  }}}catch(Exception e){serverError=e;}});server.IsBackground=true;server.Start();
  var bundle=new ModelBundle{Id="fixture",Kind="llm",Name="fixture",Primary="fixture/model.gguf"};bundle.Files.Add(new ModelFile{Url="http://127.0.0.1:"+port+"/model.gguf",RelativePath=bundle.Primary,Bytes=payload.Length});
  try{int notifications=0;ModelDownload.Fetch(bundle,root,delegate{return false;},delegate(ModelDownloadProgress p){notifications++;},delegate{});Check(server.Join(8000),"Loopback server did not stop");if(serverError!=null)throw serverError;Check(range,"Retry did not resume from the retained partial file");Check(ModelDownload.Installed(bundle,root),"Completed bundle not detected");Check(!File.Exists(ModelDownload.PathUnder(root,bundle.Primary)+".part"),"Completed download left a partial file");Check(Convert.ToBase64String(payload)==Convert.ToBase64String(File.ReadAllBytes(ModelDownload.PathUnder(root,bundle.Primary))),"Download content changed");Check(notifications>0,"Download does not report progress");
   ModelDownload.Fetch(bundle,root,delegate{return false;},delegate{},delegate{throw new Exception("Installed file was downloaded again");});Check(true,"Existing model retained");
   bool cancelled=false;try{ModelDownload.Fetch(bundle,root,delegate{return true;},delegate{},delegate{});}catch(OperationCanceledException){cancelled=true;}Check(cancelled,"Cancellation before work ignored");
  }finally{listener.Stop();}
  var cancelListener=new TcpListener(IPAddress.Loopback,0);cancelListener.Start();int cancelPort=((IPEndPoint)cancelListener.LocalEndpoint).Port;
  var cancelServer=new Thread(delegate(){try{using(var client=cancelListener.AcceptTcpClient())using(var stream=client.GetStream()){client.ReceiveTimeout=5000;var reader=new StreamReader(stream,Encoding.ASCII,false,1024,true);while(!String.IsNullOrEmpty(reader.ReadLine())){}byte[] header=Encoding.ASCII.GetBytes("HTTP/1.1 200 OK\r\nContent-Length: "+payload.Length+"\r\nConnection: close\r\n\r\n");stream.Write(header,0,header.Length);stream.Write(payload,0,payload.Length);stream.Flush();}}catch(IOException){}catch(SocketException){}});cancelServer.IsBackground=true;cancelServer.Start();
  string cancelRoot=Path.Combine(root,"cancel");var cancelBundle=new ModelBundle{Id="cancel",Primary="model.gguf"};cancelBundle.Files.Add(new ModelFile{Url="http://127.0.0.1:"+cancelPort+"/model",RelativePath="model.gguf",Bytes=payload.Length});bool stop=false,stopped=false;
  try{try{ModelDownload.Fetch(cancelBundle,cancelRoot,delegate{return stop;},delegate{stop=true;},delegate{});}catch(OperationCanceledException){stopped=true;}Check(stopped,"Active transfer did not cancel");Check(!File.Exists(Path.Combine(cancelRoot,"model.gguf"))&&File.Exists(Path.Combine(cancelRoot,"model.gguf.part")),"Cancellation lost the resume file or installed an incomplete model");Check(new FileInfo(Path.Combine(cancelRoot,"model.gguf.part")).Length>0,"Cancelled download did not retain transferred bytes");Check(cancelServer.Join(5000),"Cancel fixture did not finish");}finally{cancelListener.Stop();}
  string existing=Path.Combine(root,"existing");Directory.CreateDirectory(Path.Combine(existing,"fixture"));File.WriteAllText(Path.Combine(existing,"fixture/model.gguf"),"keep");bool refused=false;try{ModelDownload.Fetch(bundle,existing,delegate{return false;},delegate{},delegate{});}catch(IOException){refused=true;}Check(refused&&File.ReadAllText(Path.Combine(existing,"fixture/model.gguf"))=="keep","Existing model was overwritten");
  return checks;
 }
 public static int Ui(string state,string schema,string root,string evidence){checks=0;
  foreach(string theme in new[]{"dark","light"}){WizardTheme.SetPreference(theme);using(var ui=new NativeWizard(state,schema,"test",false,"")){
   ui.Opacity=0;ui.ShowInTaskbar=false;ui.Show();ui.SelectTask("server");Application.DoEvents();IntPtr handle=ui.Handle;
   foreach(string common in new[]{"context","think_mode","reasoning_effort","server_parallel"})Check(Find(ui,common)!=null&&Find(ui,common).Visible,"Common setting is not on configuration page: "+common);
   Check(Find(ui,"Quick_context")==null&&Find(ui,"Quick_think_mode")==null,"Duplicate common-setting shortcuts remain");
   ((Button)Find(ui,"Download_model")).PerformClick();Application.DoEvents();Check(ui.Handle==handle&&!ui.IsDisposed,"Download navigation replaced the main window");
   ((Button)Find(ui,"Catalog_all")).PerformClick();Application.DoEvents();foreach(string id in new[]{"llm","vision","embedding","audio","speech"})Check(Find(ui,"Bundle_"+id)!=null,"Missing catalog category: "+id);
   var downloads=(ModelDownloadsView)Find(ui,"ModelDownloads");Check(Find(ui,"ChooseDownloadDirectory")!=null,"Download folder chooser missing");
   var chooser=Find(ui,"ChooseDownloadDirectory");Check(chooser.Visible&&chooser.Right<=chooser.Parent.ClientSize.Width&&chooser.Left>=0,"Download folder chooser is outside the visible header");
   var footer=Find(ui,"DownloadFooter");var outer=(WizardScrollView)Find(ui,"ContentArea");
   Check(outer.ScrollOffset==0,"Catalog opened with the outer page scrolled");
   Check(downloads.Bottom<=downloads.Parent.ClientSize.Height&&footer.Bottom<=downloads.Height,"Catalog transfer controls need another page scroll");
   Check(Find(ui,"DownloadDirectory").Right<=chooser.Left,"Long download path covers the folder chooser");
   Check(Find(ui,"Size_llm").Text.Contains("KiB"),"Installed models lost their size label");
   string alternate=Path.Combine(root,"custom folder");downloads.SetDirectory(alternate);Application.DoEvents();
   Check(ui.StateValue("download_dir")==alternate,"Custom download directory was not retained in wizard state");
   Check(!Directory.Exists(alternate),"Choosing a folder creates data before a download");
   Check(((Button)Find(ui,"Get_llm")).Text.Contains("下载")||((Button)Find(ui,"Get_llm")).Text.Contains("Download"),"Installed detection ignores custom directory");
   bool invalid=false;try{downloads.SetDirectory("relative");}catch(IOException){invalid=true;}Check(invalid,"Relative native download folder accepted");
   downloads.SetDirectory(root);Application.DoEvents();
   var speech=(Button)Find(ui,"Get_speech");Check(speech.Text.Contains("说明")||speech.Text.Contains("guide"),"TTS does not show the release-note setup route");Check(Find(ui,"Size_speech").Text.Contains("script")||Find(ui,"Size_speech").Text.Contains("脚本"),"TTS conversion script route is missing");Check(!Find(ui,"Source_speech").Visible,"TTS repeats the same source action twice");Check(Find(ui,"Bundle_llm").Controls[0].Font.Size==ui.Font.Size,"Catalog heading uses a smaller default system font");
   if(evidence.Length>0){ui.Refresh();using(var image=new Bitmap(ui.Width,ui.Height)){ui.DrawToBitmap(image,new Rectangle(0,0,image.Width,image.Height));image.Save(Path.Combine(evidence,"downloads-"+theme+".png"));}}
   ((Button)Find(ui,"Get_llm")).PerformClick();Application.DoEvents();Check(ui.Handle==handle&&Find(ui,"ModelDownloads")==null,"Use model did not return in the same window: handle="+handle+"/"+ui.Handle+" page="+(Find(ui,"ModelDownloads")!=null)+" button="+Find(ui,"Get_llm"));Check(ui.StateValue("model")==ModelDownload.PathUnder(root,"fixture/model.gguf"),"Use model did not fill the path");Check(ui.StateValue("context")=="30k","Download navigation lost edited configuration");
   for(int i=0;i<8;i++){((Button)Find(ui,"Download_transcription_model")).PerformClick();Application.DoEvents();Check(ui.Handle==handle,"Repeated catalog navigation recreates the window");((Button)Find(ui,"DownloadsBack")).PerformClick();Application.DoEvents();}
   ((Button)Find(ui,"Download_model")).PerformClick();Application.DoEvents();((ModelDownloadsView)Find(ui,"ModelDownloads")).SetDirectory(alternate);
   ((Button)Find(ui,"DownloadsBack")).PerformClick();Application.DoEvents();((Button)Find(ui,"Download_model")).PerformClick();Application.DoEvents();
   Check(((ModelDownloadsView)Find(ui,"ModelDownloads")).DownloadDirectory==alternate,"Changing pages loses the chosen download directory");
  }}return checks;
 }
}
'@
$source=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP 'download-checks.cs'
[IO.File]::WriteAllText($source,$code,[Text.UTF8Encoding]::new($true))
Add-Type -Path $source -ReferencedAssemblies @([MoE4All.Launcher.NativeWizard].Assembly.Location,'System.Windows.Forms.dll','System.Drawing.dll')
$root=Join-Path $env:MOE4ALL_WIZARD_TEST_TMP ('download-fixture-'+[guid]::NewGuid().ToString('N'))
$network=[DownloadChecks]::Network($root)
$controls=[DownloadChecks]::TransferControls((Join-Path $root 'controls'))
$state=Get-DashboardInitialState -Saved @{wizard_language='zh';launch_mode='server';server_transcription=$true;model='fixture.gguf';context='30k'}
$schema=Get-DashboardSchema -State $state
$catalog=@(foreach($kind in @('llm','vision','embedding','audio','speech')){
 $item=[pscustomobject]@{Id=$kind;Kind=$kind;Name=('Fixture '+$kind);Primary='fixture/model.gguf';InfoUrl='https://example.com/model';Description='Offline fixture';UnavailableReason='Read this version release notes';InstructionsUrl='';Files=@()}
 if($kind -ne 'speech'){$item.Files=@([pscustomobject]@{Url='http://127.0.0.1/model';RelativePath='fixture/model.gguf';Bytes=262144})}
 else{$item.InstructionsUrl='https://github.com/Headmaster218/MoE4All/releases/tag/release-1.3.0'}
 $item
})
$schema|Add-Member -NotePropertyName download_catalog -NotePropertyValue $catalog
$schema|Add-Member -NotePropertyName download_root -NotePropertyValue $root
$ui=[DownloadChecks]::Ui(($state|ConvertTo-Json -Depth 20 -Compress),($schema|ConvertTo-Json -Depth 20 -Compress),$root,$EvidenceDirectory)
Write-Output ('Model downloads: '+$network+' transfer/path, '+$controls+' pause/resume/cancel and '+$ui+' same-window UI checks passed. Loopback fixtures only; no real models, GPU or external downloads.')
