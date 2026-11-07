// Window-backed prompts for existing PowerShell helpers. No credentials are logged or persisted.
using System;
using System.Drawing;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Diagnostics;
using System.Windows.Forms;

namespace MoE4All.Launcher {
    internal sealed class StartupNoticeDialog : WizardWindow {
        internal string Result="skip";
        readonly Font headingFont;
        internal StartupNoticeDialog(bool update,bool english,string current,string version,string notes,string url) {
            Name=update?"EngineUpdateDialog":"BenchmarkWelcomeDialog";ShowWindowButtons=false;
            StartPosition=FormStartPosition.CenterScreen;ShowInTaskbar=true;Font=new Font("Microsoft YaHei UI",10F);
            Text=update?(english?"MoE4All · Update available":"MoE4All · 发现新版本"):(english?"MoE4All · Benchmark":"MoE4All · 性能测试");
            ClientSize=new Size(780,update?560:280);MinimumSize=new Size(580,update?340:240);
            headingFont=new Font(Font.FontFamily,16F,FontStyle.Bold);
            var layout=new TableLayoutPanel{Name="StartupLayout",Dock=DockStyle.Fill,ColumnCount=1,RowCount=3,Padding=new Padding(24,20,24,16)};
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,100));layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));layout.RowStyles.Add(new RowStyle(SizeType.Percent,100));layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            string heading=update?(english?"Version "+version+" is available":"发现新版本 "+version):(notes=="new_version"?(english?"Updated to v"+version:"已更新到 v"+version):(english?"Welcome to MoE4All":"欢迎使用 MoE4All"));
            layout.Controls.Add(new Label{Name="StartupHeading",Text=heading,AutoSize=true,Dock=DockStyle.Fill,Font=headingFont,Padding=new Padding(0,0,0,16)},0,0);
            var body=new WizardPanel{Dock=DockStyle.Fill};
            if(update){
                var link=new LinkLabel{Name="ReleaseNoteLink",Text=english?"Read the full Release Note ↗":"查看完整 Release Note ↗",Dock=DockStyle.Bottom,Height=38,TextAlign=ContentAlignment.MiddleLeft,LinkBehavior=LinkBehavior.AlwaysUnderline};
                link.LinkClicked+=delegate{try{Uri target;if(!Uri.TryCreate(url,UriKind.Absolute,out target)||target.Scheme!="https"||target.Host!="github.com"||!target.AbsolutePath.StartsWith("/Headmaster218/MoE4All/releases/",StringComparison.Ordinal))return;Process.Start(new ProcessStartInfo(target.AbsoluteUri){UseShellExecute=true});}catch{link.Text=english?"Open in a browser: "+url:"请在浏览器打开："+url;}};
                var content=new WizardRichTextBox{Name="StartupReleaseNotes",ReadOnly=true,BorderStyle=BorderStyle.None,WordWrap=true,DetectUrls=false,Dock=DockStyle.Fill,ScrollBars=RichTextBoxScrollBars.Vertical,Font=Font,Text=(english?"Current version: ":"当前版本：")+current+Environment.NewLine+Environment.NewLine+(notes??"")};
                body.Controls.Add(content);body.Controls.Add(link);
                Shown+=delegate{var p=WizardTheme.Current;link.LinkColor=p.Accent;link.ActiveLinkColor=p.Accent;link.VisitedLinkColor=p.Accent;};
            }else{
                string question=notes=="new_version"?(english?"Run another quick Benchmark and see how this version performs?":"要不要再测一次快速 Benchmark，看看这个版本的表现？"):(english?"Try a quick Benchmark and see how your device performs?":"要不要先跑一次快速 Benchmark，看看这台设备的表现？");
                string extra=english?"You can share the result on the leaderboard. Uploading is always a separate choice.":"结果可以分享至排行榜；上传前会另行征求你的同意。";
                body.Controls.Add(new Label{Name="BenchmarkInvitation",Text=question+Environment.NewLine+Environment.NewLine+extra,Dock=DockStyle.Fill,Padding=new Padding(0,2,0,10)});
                Result="later";
            }
            layout.Controls.Add(body,0,1);
            var footer=new FlowLayoutPanel{Name="StartupActions",Dock=DockStyle.Fill,AutoSize=true,FlowDirection=FlowDirection.RightToLeft,WrapContents=true,Padding=new Padding(0,12,0,0)};
            if(update){
                footer.Controls.Add(Choice("UpdateNow",english?"Update":"更新","update",true));
                var skip=Choice("UpdateSkip",english?"Next time":"下次更新","skip",false);footer.Controls.Add(skip);
                footer.Controls.Add(Choice("UpdateSnooze",english?"Remind me in 3 days":"三天后再提醒","snooze",false));AcceptButton=skip;CancelButton=skip;
            }else{
                footer.Controls.Add(Choice("BenchmarkTry",english?"Try it":"试试看","try",true));
                var later=Choice("BenchmarkLater",english?"Maybe later":"稍后再说","later",false);footer.Controls.Add(later);AcceptButton=later;CancelButton=later;
            }
            layout.Controls.Add(footer,0,2);ContentHost.Controls.Add(layout);RefreshTheme();
        }
        SoftButton Choice(string name,string text,string result,bool primary){var button=new SoftButton{Name=name,Text=text,Primary=primary,AutoSize=false,Size=new Size(Math.Max(132,TextRenderer.MeasureText(text,Font).Width+38),42),Margin=new Padding(8,4,0,4)};button.Click+=delegate{Result=result;DialogResult=DialogResult.OK;Close();};return button;}
        protected override void Dispose(bool disposing){if(disposing){headingFont.Dispose();Font.Dispose();}base.Dispose(disposing);}
    }
    public static class NativeStartupPrompts {
        public static string Benchmark(string version,bool newVersion,bool english){Application.EnableVisualStyles();using(var dialog=new StartupNoticeDialog(false,english,"",version,newVersion?"new_version":"first_use","")){dialog.ShowDialog();return dialog.Result;}}
        public static string Update(string current,string target,string notes,string url,bool english){Application.EnableVisualStyles();using(var dialog=new StartupNoticeDialog(true,english,current,target,notes,url)){dialog.ShowDialog();return dialog.Result;}}
    }
    public sealed class NativePromptCanceledException : OperationCanceledException {
        public NativePromptCanceledException():base("操作已取消 / Operation cancelled"){}
    }
    internal sealed class RuntimeSubmissionDialog : WizardWindow {
        readonly bool english;
        readonly string inputType;
        readonly Font questionFont;
        readonly TextBox input;
        readonly List<SoftButton> choices=new List<SoftButton>();
        readonly WizardScrollView choiceList;
        internal string Reply;
        internal bool Cancelled=true;
        string T(string zh,string en){return english?en:zh;}
        string Local(string text){int i=(text??"").IndexOf("/ ",StringComparison.Ordinal);return i<0?(text??""):english?text.Substring(i+2).TrimStart():text.Substring(0,i).TrimEnd();}
        string LocalLines(string text){var lines=(text??"").Split('\n');for(int i=0;i<lines.Length;i++)lines[i]=Local(lines[i]);return String.Join("\n",lines);}
        static string S(Dictionary<string,object> p,string key,string fallback=""){object value;return p.TryGetValue(key,out value)&&value!=null?Convert.ToString(value):fallback;}
        void Complete(string value){Reply=value;Cancelled=false;DialogResult=DialogResult.OK;Close();}
        SoftButton ActionButton(string name,string text,bool primary,Action click){var b=new SoftButton{Name=name,Text=text,Primary=primary,AutoSize=false,Size=new Size(Math.Max(124,TextRenderer.MeasureText(text,Font).Width+40),42),Margin=new Padding(8,8,0,8)};b.Click+=delegate{click();};return b;}
        internal RuntimeSubmissionDialog(Dictionary<string,object> packet,bool english){
            this.english=english;inputType=S(packet,"input_type","text");
            Name="RuntimeSubmissionDialog";ShowWindowButtons=false;Text=T("Benchmark · 提交结果","Benchmark · Submit results");
            if(S(packet,"title").Length>0)Text=Local(S(packet,"title"));
            Font=new Font("Microsoft YaHei UI",10F);ClientSize=new Size(740,500);MinimumSize=new Size(580,220);
            StartPosition=FormStartPosition.CenterParent;ShowInTaskbar=false;
            questionFont=new Font(Font.FontFamily,14F,FontStyle.Bold);
            var layout=new TableLayoutPanel{Dock=DockStyle.Fill,ColumnCount=1,RowCount=3,Padding=new Padding(24,18,24,12)};
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,100));layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));layout.RowStyles.Add(new RowStyle(SizeType.Percent,100));layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            string question=Local(S(packet,"prompt"));question=System.Text.RegularExpressions.Regex.Replace(question,@"\s*\[[Yy]/[Nn]\]\s*$","");
            if(question.Length==0)question=T("准备好后继续","Continue when ready");
            var title=new Label{Name="SubmissionQuestion",Text=question,Font=questionFont,AutoSize=true,Dock=DockStyle.Fill,Padding=new Padding(0,0,0,18)};
            layout.Controls.Add(title,0,0);
            var footer=new FlowLayoutPanel{Dock=DockStyle.Fill,AutoSize=true,FlowDirection=FlowDirection.RightToLeft,WrapContents=false,Margin=Padding.Empty};
            var body=new WizardPanel{Dock=DockStyle.Fill,Margin=Padding.Empty};
            if(inputType=="choice"){
                choiceList=new WizardScrollView{Dock=DockStyle.Fill,Name="SubmissionChoices"};body.Controls.Add(choiceList);
                var items=packet.ContainsKey("options")?packet["options"] as IEnumerable:null;
                if(items!=null)foreach(object item in items){var option=item as Dictionary<string,object>;if(option==null)continue;
                    string value=S(option,"value",S(option,"key"));string label=Local(S(option,"label"));string detail=Local(S(option,"detail"));
                    if(detail.Length>0)label+="\n"+detail;
                    var b=ActionButton("SubmissionChoice_"+S(option,"key"),label,value==S(packet,"default"),delegate{Complete(value);});
                    if(value==S(packet,"default"))AcceptButton=b;
                    b.TextAlign=ContentAlignment.MiddleLeft;choices.Add(b);choiceList.Controls.Add(b);
                }
                if(choices.Count==0)throw new InvalidOperationException("A choice dialog requires options.");
                choiceList.SizeChanged+=delegate{LayoutChoices();};
            }else{
                var explanation=new WizardRichTextBox{Name="SubmissionContext",Dock=DockStyle.Fill,ReadOnly=true,BorderStyle=BorderStyle.None,WordWrap=true,ScrollBars=RichTextBoxScrollBars.None,Text=LocalLines(S(packet,"context")),Font=Font};body.Controls.Add(explanation);
                if(inputType=="text"){
                    input=new WizardTextBox{Name="SubmissionInput",Text=S(packet,"default"),UseSystemPasswordChar=S(packet,"secret")=="True"||S(packet,"secret")=="true",Font=Font};
                    var frame=new WizardInputFrame(input){Dock=DockStyle.Bottom,Height=46,Margin=new Padding(0,12,0,0)};body.Controls.Add(frame);frame.BringToFront();
                }
            }
            layout.Controls.Add(body,0,1);layout.Controls.Add(footer,0,2);ContentHost.Controls.Add(layout);
            if(inputType=="confirm"){
                var no=ActionButton("SubmissionNo",Local(S(packet,"no_label",T("否","No"))),false,delegate{Complete("n");});
                var yes=ActionButton("SubmissionYes",Local(S(packet,"yes_label",T("是","Yes"))),true,delegate{Complete("y");});
                if(no.Text.Length==0)no.Text=T("否","No");if(yes.Text.Length==0)yes.Text=T("是","Yes");
                footer.Controls.Add(yes);footer.Controls.Add(no);CancelButton=no;AcceptButton=S(packet,"default")=="y"?yes:no;
            }else{
                var cancel=ActionButton("SubmissionCancel",T("取消提交","Cancel submission"),false,delegate{Close();});footer.Controls.Add(cancel);CancelButton=cancel;
                if(inputType!="choice"){
                    var next=ActionButton("SubmissionContinue",T("继续","Continue"),true,delegate{
                        string value=input==null?"":input.Text;
                        if(value.IndexOfAny(new[]{'\r','\n'})>=0)return;
                        Complete(value);
                    });footer.Controls.Add(next);next.BringToFront();AcceptButton=next;
                }
            }
            if(inputType!="choice"){
                string context=LocalLines(S(packet,"context"));int width=ClientSize.Width-48;
                int questionHeight=TextRenderer.MeasureText(question,questionFont,new Size(width,1000),TextFormatFlags.WordBreak).Height+18;
                int contextHeight=context.Length==0?0:Math.Min(230,TextRenderer.MeasureText(context,Font,new Size(width,2000),TextFormatFlags.WordBreak).Height+20);
                int bodyHeight=contextHeight+(input==null?0:58);
                layout.RowStyles[1].SizeType=SizeType.Absolute;layout.RowStyles[1].Height=bodyHeight;body.Visible=bodyHeight>0;
                ClientSize=new Size(740,Math.Max(200,questionHeight+bodyHeight+115));
            }
            Shown+=delegate{if(choiceList!=null)LayoutChoices();if(input!=null){input.SelectAll();input.Focus();}else if(AcceptButton!=null)((Control)AcceptButton).Focus();};
            RefreshTheme();
        }
        void LayoutChoices(){int y=0,w=Math.Max(120,choiceList.ViewportWidth-12);foreach(var b in choices){int height=Math.Max(58,TextRenderer.MeasureText(b.Text,Font,new Size(Math.Max(80,w-36),1000),TextFormatFlags.WordBreak).Height+28);b.SetBounds(0,y,w,height);y+=height+10;}choiceList.AutoScrollMinSize=new Size(0,y);}
        internal void Dismiss(){Cancelled=true;DialogResult=DialogResult.Cancel;Close();}
        protected override void Dispose(bool disposing){base.Dispose(disposing);if(disposing){questionFont.Dispose();Font.Dispose();}}
    }

    public static class NativePrompt {
        static readonly object gate=new object();
        static readonly StringBuilder transcript=new StringBuilder();
        static HelperWindow helper;
        static Thread thread;
        static bool cancelled;
        static string progress="";
        public static void CheckCancellation(){lock(gate)if(cancelled)throw new NativePromptCanceledException();}
        public static void Progress(string text){lock(gate){if(cancelled)throw new NativePromptCanceledException();progress=text??"";}}
        public static void Append(string text){lock(gate){if(cancelled)throw new NativePromptCanceledException();transcript.Append(text);if(transcript.Length>65536)transcript.Remove(0,transcript.Length-49152);}}
        static string Snapshot(){lock(gate)return transcript.ToString();}
        public static void BeginHelper(string title){
            EndHelper();lock(gate){cancelled=false;progress="";transcript.Clear();}
            var ready=new ManualResetEvent(false);Exception failure=null;
            thread=new Thread(delegate(){try{using(var window=new HelperWindow(title)){lock(gate)helper=window;window.Shown+=delegate{ready.Set();};Application.Run(window);}}catch(Exception ex){failure=ex;ready.Set();}finally{lock(gate)helper=null;}});
            thread.IsBackground=true;thread.SetApartmentState(ApartmentState.STA);thread.Start();ready.WaitOne();ready.Dispose();if(failure!=null)throw failure;
        }
        public static void EndHelper(){
            HelperWindow window;Thread current;lock(gate){window=helper;current=thread;thread=null;}
            if(window!=null&&!window.IsDisposed&&window.IsHandleCreated){try{window.BeginInvoke((MethodInvoker)delegate{window.AllowClose=true;window.Close();});}catch(InvalidOperationException){}}
            if(current!=null&&current!=Thread.CurrentThread)current.Join(3000);
            lock(gate){cancelled=false;}
        }
        public static string Read(string prompt,bool secret){
            HelperWindow window;lock(gate){if(cancelled)throw new NativePromptCanceledException();window=helper;}
            if(window!=null&&!window.IsDisposed){
                string value=null;Exception failure=null;var completed=new ManualResetEvent(false);
                window.BeginInvoke((MethodInvoker)delegate{try{value=ShowInput(window,prompt,secret);}catch(Exception ex){failure=ex;}finally{completed.Set();}});
                completed.WaitOne();completed.Dispose();if(failure!=null)throw failure;return value;
            }
            return ShowInput(null,prompt,secret);
        }
        static string ShowInput(IWin32Window owner,string prompt,bool secret){
            using(var dialog=new WizardWindow{ShowWindowButtons=false,Text=secret?"访问密钥 / Access key":"请确认 / Confirm",Size=new Size(800,600),MinimumSize=new Size(650,450),StartPosition=owner==null?FormStartPosition.CenterScreen:FormStartPosition.CenterParent,ShowInTaskbar=owner==null,MinimizeBox=false,MaximizeBox=false}){
                dialog.Font=new Font("Microsoft YaHei UI",10);
                var host=dialog.ContentHost;
                var footer=new WizardPanel{Dock=DockStyle.Bottom,Height=156,Width=host.Width};
                var label=new Label{Text=prompt??"",Left=22,Top=12,Width=host.Width-44,Height=42,AutoEllipsis=true,Anchor=AnchorStyles.Top|AnchorStyles.Left|AnchorStyles.Right};
                var input=new WizardTextBox{Name="NativePromptInput",UseSystemPasswordChar=secret};
                var inputFrame=new WizardInputFrame(input){Left=22,Top=56,Width=host.Width-44,Height=38,Anchor=AnchorStyles.Top|AnchorStyles.Left|AnchorStyles.Right};
                string submitted="";
                var cancel=new SoftButton{Text="取消 / Cancel",Left=22,Top=106,Width=126,Height=38};cancel.Click+=delegate{dialog.DialogResult=DialogResult.Cancel;if(owner==null)dialog.Close();};
                var submit=new SoftButton{Text="确认 / Continue",Primary=true,Left=host.Width-178,Top=106,Width=156,Height=38,Anchor=AnchorStyles.Top|AnchorStyles.Right};submit.Click+=delegate{submitted=input.Text;input.Clear();dialog.DialogResult=DialogResult.OK;if(owner==null)dialog.Close();};
                footer.Controls.AddRange(new Control[]{label,inputFrame,cancel,submit});
                var log=new WizardRichTextBox{Name="NativePromptContext",Dock=DockStyle.Fill,ReadOnly=true,BorderStyle=BorderStyle.None,Font=new Font("Microsoft YaHei UI",10),Text=Snapshot(),ScrollBars=RichTextBoxScrollBars.None};
                host.Controls.Add(log);host.Controls.Add(footer);dialog.AcceptButton=submit;dialog.CancelButton=cancel;dialog.RefreshTheme();dialog.Shown+=delegate{log.SelectionStart=log.TextLength;log.ScrollToCaret();input.Focus();};
                DialogResult result;if(owner==null){Application.Run(dialog);result=dialog.DialogResult;}else result=dialog.ShowDialog(owner);string value=submitted;
                log.Font.Dispose();dialog.Font.Dispose();if(result!=DialogResult.OK)throw new NativePromptCanceledException();return value;
            }
        }
        public static void ShowError(string message){
            using(var dialog=new WizardWindow{ShowWindowButtons=false,Text="无法继续 / Unable to continue",Size=new Size(820,560),MinimumSize=new Size(640,420),StartPosition=FormStartPosition.CenterScreen}){
                var text=new WizardRichTextBox{Dock=DockStyle.Fill,ReadOnly=true,BorderStyle=BorderStyle.None,Text=(message??"")+"\n\n"+Snapshot(),Font=new Font("Microsoft YaHei UI",10)};
                var footer=new WizardPanel{Dock=DockStyle.Bottom,Height=72,Width=dialog.ContentHost.Width};var close=new SoftButton{Text="关闭 / Close",Primary=true,Left=footer.Width-162,Top=16,Width=140,Height=40,Anchor=AnchorStyles.Right|AnchorStyles.Top};close.Click+=delegate{dialog.Close();};footer.Controls.Add(close);dialog.ContentHost.Controls.Add(text);dialog.ContentHost.Controls.Add(footer);dialog.RefreshTheme();Application.Run(dialog);text.Font.Dispose();
            }
        }
        sealed class HelperWindow:WizardWindow {
            readonly RichTextBox output;
            readonly System.Windows.Forms.Timer timer;
            readonly Label status;
            string displayed="";
            internal bool AllowClose;
            internal HelperWindow(string title){
                ShowWindowButtons=false;Text=title;Font=new Font("Microsoft YaHei UI",10);Size=new Size(920,700);MinimumSize=new Size(700,480);StartPosition=FormStartPosition.CenterScreen;
                status=new Label{Dock=DockStyle.Top,Height=56,Padding=new Padding(22,16,22,0),Text="正在处理… / Working…"};
                output=new WizardRichTextBox{Dock=DockStyle.Fill,ReadOnly=true,BorderStyle=BorderStyle.None,Font=new Font("Microsoft YaHei UI",10),ScrollBars=RichTextBoxScrollBars.None};
                var footer=new WizardPanel{Dock=DockStyle.Bottom,Height=64};var cancel=new SoftButton{Name="HelperCancel",Text="取消 / Cancel",Width=124,Height=38,Left=22,Top=12};cancel.Click+=delegate{Close();};footer.Controls.Add(cancel);ContentHost.Controls.Add(output);ContentHost.Controls.Add(status);ContentHost.Controls.Add(footer);RefreshTheme();
                timer=new System.Windows.Forms.Timer{Interval=120};timer.Tick+=delegate{string current=Snapshot();lock(gate){if(!cancelled)status.Text=progress.Length==0?"正在处理… / Working…":progress;}if(current!=displayed){displayed=current;output.Text=current;output.SelectionStart=output.TextLength;output.ScrollToCaret();}};timer.Start();
                FormClosing+=delegate(object sender,FormClosingEventArgs e){if(AllowClose)return;e.Cancel=true;lock(gate)cancelled=true;status.Text="正在取消，完成当前操作后返回… / Cancelling after the current operation…";};
            }
            protected override void Dispose(bool disposing){if(disposing){timer.Stop();timer.Dispose();output.Font.Dispose();Font.Dispose();}base.Dispose(disposing);}
        }
    }
}
