// Native model catalog and resumable downloads. Uses only inbox .NET Framework.
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Threading;
using System.Windows.Forms;

namespace MoE4All.Launcher {
    public sealed class ModelBundle {
        public string Id,Kind,Name,InfoUrl,Primary,Description,UnavailableReason,InstructionsUrl;
        public long PackageBytes;
        public readonly List<ModelFile> Files=new List<ModelFile>();
        public bool Available { get { return Files.Count>0; } }
    }
    public sealed class ModelFile { public string Url,RelativePath; public long Bytes; }
    public sealed class ModelDownloadProgress { public long Bytes,Total;public int File,Count; public string Name;public double BytesPerSecond; }
    public static class ModelDownload {
        public static string PathUnder(string root,string relative){
            if(String.IsNullOrWhiteSpace(root)||String.IsNullOrWhiteSpace(relative)||Path.IsPathRooted(relative))throw new IOException("Invalid model path");
            string prefix=Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar)+Path.DirectorySeparatorChar;
            string path=Path.GetFullPath(Path.Combine(prefix,relative));
            if(!path.StartsWith(prefix,StringComparison.OrdinalIgnoreCase))throw new IOException("Model path escapes download directory");
            for(string p=Path.GetDirectoryName(path);p!=null&&p.StartsWith(prefix.TrimEnd(Path.DirectorySeparatorChar),StringComparison.OrdinalIgnoreCase);p=Path.GetDirectoryName(p))
                if(Directory.Exists(p)&&(File.GetAttributes(p)&FileAttributes.ReparsePoint)!=0)throw new IOException("Download directories cannot be symbolic links");
            return path;
        }
        public static bool Installed(ModelBundle bundle,string root){try{if(!bundle.Available)return false;foreach(var f in bundle.Files){string p=PathUnder(root,f.RelativePath);if(!File.Exists(p)||new FileInfo(p).Length==0||(f.Bytes>0&&new FileInfo(p).Length!=f.Bytes))return false;}return File.Exists(PathUnder(root,bundle.Primary));}catch{return false;}}
        public static void Fetch(ModelBundle bundle,string root,Func<bool> cancelled,Action<ModelDownloadProgress> progress,Action<HttpWebRequest> requestChanged){
            if(!bundle.Available)throw new IOException(bundle.UnavailableReason);
            ServicePointManager.SecurityProtocol|=SecurityProtocolType.Tls12;
            for(int index=0;index<bundle.Files.Count;index++){
                var file=bundle.Files[index];string path=PathUnder(root,file.RelativePath);if(cancelled())throw new OperationCanceledException();
                if(File.Exists(path)){if(new FileInfo(path).Length==0||(file.Bytes>0&&new FileInfo(path).Length!=file.Bytes))throw new IOException("Existing model has a different size; choose another download directory: "+path);continue;}
                Directory.CreateDirectory(Path.GetDirectoryName(path));string part=path+".part";if(File.Exists(part)&&(File.GetAttributes(part)&FileAttributes.ReparsePoint)!=0)throw new IOException("Partial download cannot be a symbolic link");
                if(file.Bytes>0&&File.Exists(part)&&new FileInfo(part).Length==file.Bytes){if(cancelled())throw new OperationCanceledException();File.Move(part,path);continue;}
                for(int attempt=0;;attempt++){
                    try{
                        if(cancelled())throw new OperationCanceledException();
                        long offset=File.Exists(part)?new FileInfo(part).Length:0;
                        var request=(HttpWebRequest)WebRequest.Create(file.Url);request.Timeout=15000;request.ReadWriteTimeout=60000;request.AllowAutoRedirect=true;if(new Uri(file.Url).IsLoopback)request.Proxy=null;request.UserAgent="MoE4All-Wizard/1.2";if(offset>0)request.AddRange(offset);requestChanged(request);
                        using(var response=(HttpWebResponse)request.GetResponse()){
                            if(offset>0&&response.StatusCode==HttpStatusCode.PartialContent){string range=response.Headers["Content-Range"]??"";if(!range.StartsWith("bytes "+offset+"-",StringComparison.Ordinal))throw new IOException("Invalid resume response");}
                            else if(response.StatusCode==HttpStatusCode.OK)offset=0;
                            else if(response.StatusCode!=HttpStatusCode.PartialContent)throw new IOException("Unexpected HTTP status "+response.StatusCode);
                            else if(!(response.Headers["Content-Range"]??"").StartsWith("bytes 0-",StringComparison.Ordinal))throw new IOException("Invalid partial response");
                            long total=response.ContentLength>=0?offset+response.ContentLength:file.Bytes;long bytes=offset;
                            var watch=Stopwatch.StartNew();long reported=-200;byte[] buffer=new byte[131072];
                            using(var input=response.GetResponseStream())using(var output=new FileStream(part,offset>0?FileMode.Append:FileMode.Create,FileAccess.Write,FileShare.Read)){
                                int n;while((n=input.Read(buffer,0,buffer.Length))>0){if(cancelled())throw new OperationCanceledException();output.Write(buffer,0,n);bytes+=n;if(watch.ElapsedMilliseconds-reported>=120){progress(new ModelDownloadProgress{Bytes=bytes,Total=total,File=index+1,Count=bundle.Files.Count,Name=Path.GetFileName(path),BytesPerSecond=(bytes-offset)/Math.Max(.001,watch.Elapsed.TotalSeconds)});reported=watch.ElapsedMilliseconds;}}
                                output.Flush();
                            }
                            if(cancelled())throw new OperationCanceledException();
                            if(bytes==0||(total>0&&bytes!=total)||(file.Bytes>0&&bytes!=file.Bytes))throw new IOException("Incomplete model download");
                        }
                        requestChanged(null);File.Move(part,path);break;
                    }catch{requestChanged(null);if(cancelled())throw new OperationCanceledException();if(attempt>=2)throw;for(int n=0;n<10*(attempt+1);n++){if(cancelled())throw new OperationCanceledException();Thread.Sleep(100);}}
                }
            }
        }
    }
    public sealed class ModelDownloadsView : WizardPanel {
        readonly bool english,preview;
        readonly List<ModelBundle> bundles;
        string root;
        readonly Label location;
        readonly SoftButton chooseDirectory;
        public event Action<string> DirectoryChanged;
        public string DownloadDirectory { get { return root; } }
        readonly Action<ModelBundle,string> use;
        readonly WizardScrollView list;
        readonly Label status;
        readonly SoftButton cancellation,pause;
        readonly List<SoftButton> filters=new List<SoftButton>();
        readonly Dictionary<string,SoftButton> actions=new Dictionary<string,SoftButton>();
        readonly BackgroundWorker worker=new BackgroundWorker{WorkerReportsProgress=true,WorkerSupportsCancellation=true};
        volatile HttpWebRequest request;
        string category;
        volatile bool cancelling,pausing,transferring;
        bool paused,closing;
        ModelBundle active;
        public bool Busy { get { return transferring||paused; } }
        public event EventHandler BusyChanged;
        string T(string zh,string en){return english?en:zh;}
        public ModelDownloadsView(List<ModelBundle> catalog,string directory,string initialCategory,bool en,bool demo,Action<ModelBundle,string> apply){
            bundles=catalog;root=directory;category=initialCategory;english=en;preview=demo;use=apply;Name="ModelDownloads";Width=860;Height=620;
            var heading=new WizardPanel{Dock=DockStyle.Top,Height=106,Width=Width};
            location=new Label{Name="DownloadDirectory",Text=T("保存到：","Save to: ")+root,Left=6,Top=8,Height=26,AutoEllipsis=true,Width=Width-175,Anchor=AnchorStyles.Left|AnchorStyles.Right|AnchorStyles.Top};heading.Controls.Add(location);
            chooseDirectory=new SoftButton{Name="ChooseDownloadDirectory",Text=T("更改目录…","Change folder…"),Width=150,Height=34,Left=Width-156,Top=0,Anchor=AnchorStyles.Top|AnchorStyles.Right};
            chooseDirectory.Click+=delegate{if(Busy)return;using(var dialog=new FolderBrowserDialog()){dialog.Description=T("选择模型下载目录","Choose the model download folder");dialog.SelectedPath=root;dialog.ShowNewFolderButton=true;if(dialog.ShowDialog(FindForm())==DialogResult.OK)SetDirectory(dialog.SelectedPath);}};heading.Controls.Add(chooseDirectory);
            var categories=new WizardFlowPanel{Left=0,Top=40,Height=58,Width=860,Anchor=AnchorStyles.Left|AnchorStyles.Right|AnchorStyles.Top,WrapContents=true};heading.Controls.Add(categories);
            string[] values={"all","llm","vision","embedding","audio","speech"};string[] names={T("全部","All"),T("语言模型","LLM"),T("视觉","Vision"),"Embedding",T("语音识别","Transcription"),T("语音合成","Speech")};
            for(int i=0;i<values.Length;i++){string selected=values[i];var button=new SoftButton{Name="Catalog_"+selected,Text=names[i],Width=i>=4?130:110,Height=36,Margin=new Padding(0,0,8,0),Primary=category==selected};button.Click+=delegate{category=selected;BuildList();};filters.Add(button);categories.Controls.Add(button);}
            var footer=new WizardPanel{Name="DownloadFooter",Dock=DockStyle.Bottom,Height=76};
            var transferActions=new WizardFlowPanel{Name="DownloadActions",Dock=DockStyle.Right,Width=276,Padding=new Padding(0,12,0,0),WrapContents=false};
            status=new Label{Name="DownloadStatus",Dock=DockStyle.Fill,Padding=new Padding(8,10,8,5),AutoEllipsis=true,Text=T("下载完成后点“使用”。暂停或取消均保留断点。","Choose Use after downloading. Pause or cancel keeps partial files.")};footer.Controls.Add(status);footer.Controls.Add(transferActions);
            pause=new SoftButton{Name="PauseDownload",Enabled=false,Text=T("暂停","Pause"),Width=126,Height=38,Margin=new Padding(0,0,8,0)};pause.Click+=delegate{if(paused)Resume();else Pause();};transferActions.Controls.Add(pause);
            cancellation=new SoftButton{Name="CancelDownload",Enabled=false,Text=T("取消","Cancel"),Width=126,Height=38,Margin=Padding.Empty};cancellation.Click+=delegate{Cancel();};transferActions.Controls.Add(cancellation);
            list=new WizardScrollView{Name="CatalogList",Dock=DockStyle.Fill};Controls.Add(list);Controls.Add(heading);Controls.Add(footer);
            worker.DoWork+=delegate(object sender,DoWorkEventArgs e){var bundle=(ModelBundle)e.Argument;ModelDownload.Fetch(bundle,root,delegate{return cancelling||pausing;},delegate(ModelDownloadProgress p){worker.ReportProgress(0,p);},delegate(HttpWebRequest r){request=r;if((cancelling||pausing)&&r!=null)r.Abort();});e.Result=bundle;};
            worker.ProgressChanged+=delegate(object sender,ProgressChangedEventArgs e){PostToUi(delegate{if(!transferring||cancelling||pausing)return;var p=(ModelDownloadProgress)e.UserState;status.Text=p.File+"/"+p.Count+"  "+p.Name+Environment.NewLine+FormatSize(p.Bytes)+(p.Total>0?" / "+FormatSize(p.Total):"")+"   "+FormatSize((long)p.BytesPerSecond)+"/s";});};
            worker.RunWorkerCompleted+=delegate(object sender,RunWorkerCompletedEventArgs e){PostToUi(delegate{CompleteDownload(e);});};
            BuildList();
        }
        void PostToUi(Action action){if(IsDisposed||closing)return;if(InvokeRequired){try{BeginInvoke((MethodInvoker)delegate{if(!IsDisposed&&!closing)action();});}catch(InvalidOperationException){}return;}action();}
        void CompleteDownload(RunWorkerCompletedEventArgs e){
            request=null;transferring=false;paused=pausing&&!cancelling&&e.Error!=null;
            string text=cancelling?T("已取消，断点已保留；可再次点击下载。","Cancelled. Partial files are retained; choose Download to resume."):paused?T("已暂停，断点已保留。点击“继续”恢复下载。","Paused. Choose Resume to continue from the retained partial file."):e.Error!=null?T("下载失败：","Download failed: ")+e.Error.GetBaseException().Message:T("下载完成。点击“使用”即可返回配置。","Download complete. Choose Use to return to configuration.");
            cancelling=false;pausing=false;if(!paused){active=null;BuildList(true);}UpdateTransferActions();status.Text=text;if(BusyChanged!=null)BusyChanged(this,EventArgs.Empty);
        }
        void UpdateTransferActions(){pause.Text=paused?T("继续","Resume"):T("暂停","Pause");pause.Enabled=Busy&&!cancelling&&!pausing;cancellation.Enabled=Busy&&!cancelling;chooseDirectory.Enabled=!Busy;}
        void StartDownload(ModelBundle bundle){active=bundle;paused=false;cancelling=false;pausing=false;transferring=true;worker.RunWorkerAsync(bundle);UpdateTransferActions();pause.Focus();status.Text=T("正在连接：","Connecting: ")+bundle.Name;foreach(var a in actions.Values)a.Enabled=false;if(BusyChanged!=null)BusyChanged(this,EventArgs.Empty);}
        public void Pause(){if(!worker.IsBusy||cancelling||pausing)return;pausing=true;UpdateTransferActions();status.Text=T("正在暂停…","Pausing…");var r=request;if(r!=null)r.Abort();}
        public void Resume(){if(!paused||active==null||worker.IsBusy)return;StartDownload(active);}
        public void Cancel(){
            if(!Busy)return;
            if(paused){paused=false;active=null;BuildList(true);status.Text=T("已取消，断点已保留；可再次点击下载。","Cancelled. Partial files are retained; choose Download to resume.");if(BusyChanged!=null)BusyChanged(this,EventArgs.Empty);return;}
            cancelling=true;pausing=false;UpdateTransferActions();worker.CancelAsync();var r=request;if(r!=null)r.Abort();status.Text=T("正在取消…","Cancelling…");
        }
        public void SetDirectory(string directory){
            if(Busy)throw new InvalidOperationException("Cannot change the download folder during a transfer");
            if(String.IsNullOrWhiteSpace(directory)||!Path.IsPathRooted(directory))throw new IOException("Choose an absolute download folder");
            string selected=Path.GetFullPath(directory);ModelDownload.PathUnder(selected,".download-path-check");
            if(File.Exists(selected))throw new IOException("The download folder is a file");
            root=selected;location.Text=T("保存到：","Save to: ")+root;BuildList();if(DirectoryChanged!=null)DirectoryChanged(root);
        }
        static string FormatSize(long bytes){return bytes>=1073741824?((double)bytes/1073741824).ToString("0.00")+" GiB":bytes>=1048576?((double)bytes/1048576).ToString("0.0")+" MiB":bytes>=1024?((double)bytes/1024).ToString("0.0")+" KiB":bytes+" B";}
        void BuildList(bool preserveScroll=false){UpdateTransferActions();int scroll=preserveScroll?list.ScrollOffset:0;int dpi=WizardDpi.Dpi(this);
            foreach(var button in filters){button.Primary=button.Name=="Catalog_"+category;button.Invalidate();}
            list.SuspendLayout();try{while(list.Controls.Count>0){var c=list.Controls[0];list.Controls.RemoveAt(0);c.Dispose();}actions.Clear();int y=4;
                foreach(var bundle in bundles){if(category!="all"&&category!=bundle.Kind)continue;var selected=bundle;
                    var card=new SurfacePanel{Name="Bundle_"+bundle.Id,Left=4,Top=y,Width=Math.Max(280,(int)Math.Round(list.ViewportWidth*96.0/dpi)-10),Height=146,Tag="WizardTheme.Surface",Anchor=AnchorStyles.Top|AnchorStyles.Left|AnchorStyles.Right};
                    var title=new Label{Text=bundle.Name,Left=18,Top=15,Height=30,Width=card.Width-185,AutoEllipsis=true,Anchor=AnchorStyles.Top|AnchorStyles.Left|AnchorStyles.Right};card.Controls.Add(title);
                    bool installed=ModelDownload.Installed(bundle,root);string description=!bundle.Available?bundle.UnavailableReason:String.IsNullOrEmpty(bundle.Description)?bundle.Kind:bundle.Description;
                    card.Controls.Add(new Label{Text=description,Left=18,Top=50,Height=42,Width=card.Width-185,AutoEllipsis=true,Anchor=AnchorStyles.Top|AnchorStyles.Left|AnchorStyles.Right});
                    bool guide=!String.IsNullOrEmpty(bundle.InstructionsUrl);
                    long bytes=bundle.PackageBytes;if(bytes<=0)foreach(var f in bundle.Files)bytes+=f.Bytes;var detail=new Label{Name="Size_"+bundle.Id,Text=(bytes>0?FormatSize(bytes)+" · ":"")+(installed?T("已下载 · 可直接使用","Installed · ready to use"):guide?T("发行说明 · 配套转换脚本","Release notes · conversion script"):!bundle.Available?T("支持导入本地兼容包","Import a matching local package"):bundle.Files.Count+T(" 个文件"," file(s)")),Left=18,Top=103,Height=26,Width=card.Width-185,AutoEllipsis=true,Anchor=AnchorStyles.Top|AnchorStyles.Left|AnchorStyles.Right};card.Controls.Add(detail);
                    var action=new SoftButton{Name="Get_"+bundle.Id,Text=installed?T("使用","Use"):bundle.Available?T("下载","Download"):guide?T("安装说明","Setup guide"):T("查看来源","View source"),Left=card.Width-145,Top=21,Width=126,Height=40,Primary=installed,Enabled=!Busy,Anchor=AnchorStyles.Right|AnchorStyles.Top};
                    action.Click+=delegate{if(installed){use(selected,root);return;}if(!selected.Available){OpenSource(String.IsNullOrEmpty(selected.InstructionsUrl)?selected.InfoUrl:selected.InstructionsUrl);return;}if(preview){status.Text=T("演示模式不下载模型。","Preview does not download models.");return;}StartDownload(selected);};card.Controls.Add(action);actions[bundle.Id]=action;
                    var source=new SoftButton{Name="Source_"+bundle.Id,Visible=bundle.Available,Text=T("模型来源 ↗","Source ↗"),Left=card.Width-145,Top=78,Width=126,Height=34,Quiet=true,Anchor=AnchorStyles.Right|AnchorStyles.Top};source.Click+=delegate{OpenSource(selected.InfoUrl);};card.Controls.Add(source);if(dpi!=96)card.Font=Font;WizardDpi.AdaptTree(card,this);list.Controls.Add(card);y+=158;
                }
                if(list.Controls.Count==0)list.Controls.Add(new Label{Text=T("此分类暂无可下载的模型。","No models in this category yet."),Top=25,Left=16,Width=500,Height=40});
                list.AutoScrollMinSize=new Size(0,(int)Math.Round((y+8)*dpi/96.0));WizardTheme.Apply(this);
            }finally{list.ResumeLayout(true);list.AutoScrollPosition=new Point(0,scroll);}
        }
        void OpenSource(string url){try{Uri uri;if(!Uri.TryCreate(url,UriKind.Absolute,out uri)||(uri.Scheme!="https"&&uri.Scheme!="http"))throw new IOException("Invalid model source");Process.Start(new ProcessStartInfo(url){UseShellExecute=true});}catch(Exception e){status.Text=e.Message;}}
        protected override void Dispose(bool disposing){if(disposing){closing=true;cancelling=true;paused=false;var r=request;if(r!=null)r.Abort();worker.Dispose();}base.Dispose(disposing);}
    }
}
