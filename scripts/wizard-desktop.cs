// Small inbox-.NET launcher. PowerShell owns configuration, validation and engine argv.
// ModelDownloadsView keeps catalog and transfer interaction inside this window.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Web.Script.Serialization;
using System.Windows.Forms;

namespace MoE4All.Launcher {
    public sealed class SoftButton : Button {
        int P(int value){return WizardDpi.Pixels(this,value);}
        public Color Accent = Color.FromArgb(21,108,93);
        public bool Primary, Quiet;
        bool hover,pressed;
        public SoftButton() { FlatStyle=FlatStyle.Flat;FlatAppearance.BorderSize=0;Cursor=Cursors.Hand;UseVisualStyleBackColor=false;DoubleBuffered=true;SetStyle(ControlStyles.UserPaint|ControlStyles.AllPaintingInWmPaint|ControlStyles.OptimizedDoubleBuffer,true); }
        protected override void OnMouseEnter(EventArgs e){hover=true;Invalidate();base.OnMouseEnter(e);}
        protected override void OnMouseLeave(EventArgs e){hover=false;Invalidate();base.OnMouseLeave(e);}
        protected override void OnMouseDown(MouseEventArgs e){if(e.Button==MouseButtons.Left)pressed=true;Invalidate();base.OnMouseDown(e);}
        protected override void OnMouseUp(MouseEventArgs e){pressed=false;Invalidate();base.OnMouseUp(e);}
        protected override void OnMouseCaptureChanged(EventArgs e){if(!Capture)pressed=false;Invalidate();base.OnMouseCaptureChanged(e);}
        protected override void OnKeyDown(KeyEventArgs e){if(e.KeyCode==Keys.Space)pressed=true;Invalidate();base.OnKeyDown(e);}
        protected override void OnKeyUp(KeyEventArgs e){pressed=false;Invalidate();base.OnKeyUp(e);}
        internal static GraphicsPath Round(Rectangle r,int radius){var p=new GraphicsPath();int d=radius*2;p.AddArc(r.Left,r.Top,d,d,180,90);p.AddArc(r.Right-d,r.Top,d,d,270,90);p.AddArc(r.Right-d,r.Bottom-d,d,d,0,90);p.AddArc(r.Left,r.Bottom-d,d,d,90,90);p.CloseFigure();return p;}
        protected override void OnPaint(PaintEventArgs e){
            e.Graphics.Clear(Parent==null?BackColor:Parent.BackColor);e.Graphics.SmoothingMode=SmoothingMode.AntiAlias;
            var theme=WizardTheme.Current;
            Color bg=Primary?(hover?theme.PrimaryHover:theme.Primary):Quiet?(hover?theme.Hover:BackColor):(hover?theme.Hover:theme.Surface);
            if(pressed)bg=Primary?Color.FromArgb((theme.Primary.R*4+theme.Background.R)/5,(theme.Primary.G*4+theme.Background.G)/5,(theme.Primary.B*4+theme.Background.B)/5):theme.Selected;
            using(var p=Round(new Rectangle(P(1),P(1),Width-P(3),Height-P(3)),P(9))){using(var b=new SolidBrush(bg))e.Graphics.FillPath(b,p);if(!Primary&&!Quiet)using(var pen=new Pen(theme.Outline))e.Graphics.DrawPath(pen,p);}
            TextRenderer.DrawText(e.Graphics,Text,Font,ClientRectangle,Enabled?(Primary?(theme.Dark?theme.Text:theme.OnAccent):theme.Text):theme.Muted,TextFormatFlags.HorizontalCenter|TextFormatFlags.VerticalCenter|TextFormatFlags.EndEllipsis);
            if(Focused)using(var path=Round(new Rectangle(P(4),P(4),Width-P(9),Height-P(9)),P(7)))using(var pen=new Pen(Primary?(theme.Dark?theme.Text:theme.OnAccent):theme.Accent))e.Graphics.DrawPath(pen,path);
        }
    }
    public sealed class TaskTile : Button {
        int P(int value){return WizardDpi.Pixels(this,value);}
        public string TitleText, DetailText;
        public int IconKind;
        public bool Selected;
        bool hover;
        public TaskTile(){FlatStyle=FlatStyle.Flat;FlatAppearance.BorderSize=0;Cursor=Cursors.Hand;DoubleBuffered=true;UseVisualStyleBackColor=false;}
        protected override void OnMouseEnter(EventArgs e){hover=true;Invalidate();base.OnMouseEnter(e);}
        protected override void OnMouseLeave(EventArgs e){hover=false;Invalidate();base.OnMouseLeave(e);}
        protected override void OnPaint(PaintEventArgs e){
            e.Graphics.Clear(Parent.BackColor);e.Graphics.SmoothingMode=SmoothingMode.AntiAlias;
            var theme=WizardTheme.Current;Color ink=theme.Accent;Color bg=hover||Selected?theme.Selected:theme.Surface;
            using(var p=SoftButton.Round(new Rectangle(P(1),P(1),Width-P(3),Height-P(3)),P(14))){using(var b=new SolidBrush(bg))e.Graphics.FillPath(b,p);using(var pen=new Pen(Selected||hover?theme.Accent:theme.Outline,P(Focused?2:1)))e.Graphics.DrawPath(pen,p);}
            GraphicsState iconState=e.Graphics.Save();float scale=WizardDpi.Dpi(this)/96F;e.Graphics.ScaleTransform(scale,scale);
            using(var pen=new Pen(ink,2)){
                if(IconKind==0){e.Graphics.DrawRectangle(pen,23,23,29,20);e.Graphics.DrawLines(pen,new[]{new Point(29,43),new Point(29,49),new Point(37,43)});e.Graphics.DrawLine(pen,30,30,45,30);e.Graphics.DrawLine(pen,30,36,40,36);}
                else if(IconKind==1){e.Graphics.DrawRectangle(pen,24,22,29,11);e.Graphics.DrawRectangle(pen,24,38,29,11);e.Graphics.DrawLine(pen,29,27,32,27);e.Graphics.DrawLine(pen,29,43,32,43);}
                else {e.Graphics.DrawLine(pen,24,47,24,23);e.Graphics.DrawLine(pen,24,47,54,47);e.Graphics.DrawLines(pen,new[]{new Point(28,40),new Point(35,34),new Point(41,37),new Point(52,24)});}
            }
            e.Graphics.Restore(iconState);
            using(var title=WizardDpi.PaintFont(Font,this,2,FontStyle.Bold))TextRenderer.DrawText(e.Graphics,TitleText,title,new Rectangle(P(23),P(70),Width-P(44),P(30)),theme.Text,TextFormatFlags.Left|TextFormatFlags.EndEllipsis);
            TextRenderer.DrawText(e.Graphics,DetailText,Font,new Rectangle(P(23),P(110),Width-P(44),P(48)),theme.Muted,TextFormatFlags.Left|TextFormatFlags.WordBreak);
            TextRenderer.DrawText(e.Graphics,"→",Font,new Rectangle(Width-P(44),P(25),P(25),P(30)),ink,TextFormatFlags.HorizontalCenter);
        }
    }
    public sealed class SurfacePanel : Panel {
        int P(int value){return WizardDpi.Pixels(this,value);}
        public SurfacePanel(){DoubleBuffered=true;ResizeRedraw=true;Padding=new Padding(1);}
        protected override void OnPaint(PaintEventArgs e){base.OnPaint(e);e.Graphics.SmoothingMode=SmoothingMode.AntiAlias;using(var path=SoftButton.Round(new Rectangle(0,0,Math.Max(P(20),Width-P(1)),Math.Max(P(20),Height-P(1))),P(12)))using(var pen=new Pen(Color.FromArgb(70,WizardTheme.Current.Border))){e.Graphics.DrawPath(pen,path);}}
    }
    public sealed class ToggleSwitch : CheckBox {
        int P(int value){return WizardDpi.Pixels(this,value);}
        public ToggleSwitch(){AutoSize=false;Size=new Size(58,30);SetStyle(ControlStyles.UserPaint|ControlStyles.AllPaintingInWmPaint|ControlStyles.OptimizedDoubleBuffer,true);Cursor=Cursors.Hand;}
        protected override void OnPaint(PaintEventArgs e){e.Graphics.Clear(Parent.BackColor);e.Graphics.SmoothingMode=SmoothingMode.AntiAlias;var t=WizardTheme.Current;using(var path=SoftButton.Round(new Rectangle(P(1),P(3),P(49),P(24)),P(12)))using(var brush=new SolidBrush(Checked?t.Primary:t.Hover)){e.Graphics.FillPath(brush,path);}using(var brush=new SolidBrush(Checked?(t.Dark?t.Text:t.OnAccent):t.Muted)){e.Graphics.FillEllipse(brush,P(Checked?27:5),P(7),P(16),P(16));}if(Focused)ControlPaint.DrawFocusRectangle(e.Graphics,ClientRectangle,t.Accent,Parent.BackColor);}
    }
    public sealed class CapabilityChip : CheckBox {
        int P(int value){return WizardDpi.Pixels(this,value);}
        public CapabilityChip(){AutoSize=false;Height=32;Cursor=Cursors.Hand;SetStyle(ControlStyles.UserPaint|ControlStyles.AllPaintingInWmPaint|ControlStyles.OptimizedDoubleBuffer,true);}
        protected override void OnPaint(PaintEventArgs e){e.Graphics.Clear(Parent.BackColor);e.Graphics.SmoothingMode=SmoothingMode.AntiAlias;var t=WizardTheme.Current;using(var path=SoftButton.Round(new Rectangle(P(1),P(1),Width-P(3),Height-P(3)),P(9))){using(var brush=new SolidBrush(Checked?t.Selected:t.Surface)){e.Graphics.FillPath(brush,path);}using(var pen=new Pen(Checked?t.Accent:t.Outline)){e.Graphics.DrawPath(pen,path);}}TextRenderer.DrawText(e.Graphics,(Checked?"✓  ":"+  ")+Text,Font,ClientRectangle,t.Text,TextFormatFlags.HorizontalCenter|TextFormatFlags.VerticalCenter|TextFormatFlags.EndEllipsis);if(Focused)ControlPaint.DrawFocusRectangle(e.Graphics,new Rectangle(P(5),P(5),Width-P(10),Height-P(10)));}
    }
    public sealed class SettingsPages : Panel {
        public readonly List<WizardScrollView> TabPages=new List<WizardScrollView>();
        WizardScrollView selected;
        public WizardScrollView SelectedTab {get{return selected;}set{if(selected==value)return;if(selected!=null)selected.Visible=false;selected=value;if(value!=null){WizardDpi.AdaptTree(value,this);if(!Controls.Contains(value))Controls.Add(value);value.Dock=DockStyle.Fill;value.Visible=true;value.BringToFront();}}}
        public SettingsPages(){DoubleBuffered=true;}
    }
    public sealed class NativeWizard : WizardWindow {
        static NativeWizard(){Application.EnableVisualStyles();}
        static readonly JavaScriptSerializer Json=new JavaScriptSerializer{MaxJsonLength=1024*1024};
        [DllImport("kernel32.dll")] static extern IntPtr GetConsoleWindow();
        [DllImport("user32.dll")] static extern bool ShowWindow(IntPtr hwnd,int command);
        [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr hwnd);
        static bool hiddenConsole;
        static Point? lastPosition;
        readonly Dictionary<string,object> state,schema;
        readonly List<Dictionary<string,object>> fields,pages;
        readonly Font titleFont,sectionFont,smallFont;
        readonly ToolTip tips=new ToolTip();
        readonly bool demo;
        readonly string version;
        string screen,action="cancel",focusKey="",message,reopenSettingsPage="",reopenSettingsField="";
        bool dirty,finishing,english,building,syncing;
        Form settings;
        WizardScrollView body;Panel content,topBar,bottomBar;
        SoftButton primary;
        Label noticeLabel;
        static Color Paper{get{return WizardTheme.Current.Background;}}static Color Ink{get{return WizardTheme.Current.Text;}}static Color Muted{get{return WizardTheme.Current.Muted;}}static Color Accent{get{return WizardTheme.Current.Accent;}}
        static Dictionary<string,object> Dict(object o){return o as Dictionary<string,object>??new Dictionary<string,object>();}
        static List<Dictionary<string,object>> Items(object o){var result=new List<Dictionary<string,object>>();if(o is IEnumerable&&!(o is string))foreach(object x in (IEnumerable)o)if(x is Dictionary<string,object>)result.Add((Dictionary<string,object>)x);return result;}
        static string S(object o){return o is bool?((bool)o?"true":"false"):o==null?"":Convert.ToString(o,CultureInfo.InvariantCulture);}
        static string Get(Dictionary<string,object> d,string key,string fallback){return d.ContainsKey(key)?S(d[key]):fallback;}
        string V(string key){return Get(state,key,"");}
        bool Flag(string key){return V(key).Equals("true",StringComparison.OrdinalIgnoreCase);}
        bool Llm{get{return V("launch_mode")!="server"||Flag("server_llm_enabled");}}
        string T(string zh,string en){return english?en:zh;}
        string Local(string label){int i=label.IndexOf(" / ",StringComparison.Ordinal);return i<0?label:english?label.Substring(i+3):label.Substring(0,i);}
        public NativeWizard(string stateJson,string schemaJson,string appVersion,bool preview,string initialMessage){
            state=Dict(Json.DeserializeObject(stateJson));schema=Dict(Json.DeserializeObject(schemaJson));fields=Items(schema.ContainsKey("fields")?schema["fields"]:null);pages=Items(schema.ContainsKey("pages")?schema["pages"]:null);
            version=appVersion;demo=preview;dirty=Get(schema,"resume_dirty","false")=="true";message=initialMessage??"";
            string savedTheme=V("wizard_theme");WizardTheme.SetPreference(savedTheme=="light"||savedTheme=="dark"?savedTheme:"system");
            english=V("wizard_language")=="en"||(V("wizard_language")==""&&CultureInfo.CurrentUICulture.TwoLetterISOLanguageName!="zh");
            screen=Get(schema,"resume_page","home")=="home"?"home":"setup";focusKey=Get(schema,"resume_field","");
            Font=new Font("Microsoft YaHei UI",10F);titleFont=new Font(Font.FontFamily,24,FontStyle.Bold);sectionFont=new Font(Font.FontFamily,11,FontStyle.Bold);smallFont=new Font(Font.FontFamily,9);
            tips.OwnerDraw=true;tips.UseAnimation=false;tips.UseFading=false;tips.InitialDelay=700;tips.ReshowDelay=350;tips.AutoPopDelay=8000;
            tips.Popup+=delegate(object sender,PopupEventArgs e){using(var tipFont=WizardDpi.PaintFont(smallFont,this,0)){var p=WizardTheme.Current;tips.BackColor=p.SurfaceAlt;tips.ForeColor=p.Text;string text=tips.GetToolTip(e.AssociatedControl);if(text.Length>1024)text=text.Substring(0,1024)+"…";Size size=TextRenderer.MeasureText(text,tipFont,new Size(LayoutPixels(420),LayoutPixels(600)),TextFormatFlags.WordBreak|TextFormatFlags.NoPrefix);e.ToolTipSize=new Size(Math.Min(LayoutPixels(444),size.Width+LayoutPixels(20)),Math.Min(LayoutPixels(620),size.Height+LayoutPixels(16)));}};
            tips.Draw+=delegate(object sender,DrawToolTipEventArgs e){using(var tipFont=WizardDpi.PaintFont(smallFont,this,0)){var p=WizardTheme.Current;using(var brush=new SolidBrush(p.SurfaceAlt))e.Graphics.FillRectangle(brush,e.Bounds);using(var pen=new Pen(p.Outline))e.Graphics.DrawRectangle(pen,0,0,Math.Max(1,e.Bounds.Width-1),Math.Max(1,e.Bounds.Height-1));string text=e.ToolTipText;if(text.Length>1024)text=text.Substring(0,1024)+"…";TextRenderer.DrawText(e.Graphics,text,tipFont,Rectangle.Inflate(e.Bounds,-LayoutPixels(10),-LayoutPixels(8)),p.Text,TextFormatFlags.WordBreak|TextFormatFlags.NoPrefix);}};
            Text="MoE4All · "+(demo?"Preview":"v"+version);BackColor=Paper;ForeColor=Ink;ClientSize=new Size(1060,820);MinimumSize=new Size(820,680);StartPosition=FormStartPosition.CenterScreen;KeyPreview=true;
            if(lastPosition.HasValue){StartPosition=FormStartPosition.Manual;Location=lastPosition.Value;}
            FormClosing+=ClosingWizard;
            KeyDown+=delegate(object sender,KeyEventArgs e){if(e.KeyCode==Keys.Escape){if(screen=="downloads"){BackFromDownloads();}else if(screen=="setup"){screen="home";Rebuild();}else Close();e.Handled=true;}if(e.KeyCode==Keys.F6){Finish("save");e.Handled=true;}if(e.KeyCode==Keys.F5){StartRun();e.Handled=true;}};
            Rebuild();
            Shown+=delegate{
                if(Get(schema,"native_settings_page","").Length>0)BeginInvoke((MethodInvoker)delegate{OpenSection(Get(schema,"native_settings_page","runtime"),Get(schema,"native_settings_field",""),Get(schema,"native_settings_groups",""));});
                else if(!String.IsNullOrEmpty(initialMessage)&&focusKey.Length>0)BeginInvoke((MethodInvoker)delegate{var found=Controls.Find(focusKey,true);if(found.Length>0&&found[0].Visible){found[0].Focus();var input=found[0] as TextBox;if(input!=null)input.SelectAll();}else OpenSettings(FieldPage(focusKey),focusKey);});
            };
        }
        string FieldPage(string key){var f=fields.Find(delegate(Dictionary<string,object> item){return Get(item,"key","")==key&&Get(item,"page","")!="home";});return f==null?"runtime":Get(f,"page","runtime");}
        Dictionary<string,object> Field(string key){return fields.Find(delegate(Dictionary<string,object> f){return Get(f,"key","")==key;});}
        string Derived(string key){if(key=="_llm")return S(Llm);if(key=="_api")return S(V("launch_mode")=="server");if(key=="_bench")return S(V("launch_mode")=="benchmark");if(key=="_manual")return S(V("setup_mode")=="manual");if(key=="_device")return S(Llm||Flag("server_embedding"));if(key=="_audio")return S(Flag("server_transcription"));return V(key);}
        public bool IsFieldVisible(string key){var f=Field(key);return f!=null&&VisibleField(f);}
        bool VisibleField(Dictionary<string,object> f){foreach(var pair in Dict(f.ContainsKey("when")?f["when"]:null)){string actual=Derived(pair.Key);if(pair.Value is IEnumerable&&!(pair.Value is string)){bool found=false;foreach(object x in (IEnumerable)pair.Value)if(S(x).Equals(actual,StringComparison.OrdinalIgnoreCase))found=true;if(!found)return false;}else if(!S(pair.Value).Equals(actual,StringComparison.OrdinalIgnoreCase))return false;}return true;}
        public string StateValue(string key){return V(key);}
        public void SelectTask(string mode){if(mode!="chat"&&mode!="server"&&mode!="benchmark")throw new ArgumentException("Unknown task");Change("launch_mode",mode);screen="setup";message="";Rebuild();}
        void Change(string key,object value){if(V(key)!=S(value)){state[key]=value;dirty=true;if(key=="bench_kind"&&S(value)!="manual")state["bench_parallel"]=S(value)=="quick"?"1":S(value)=="standard"?"2":"4";if(key=="think_mode"||key=="reasoning_effort")state["configure_thinking"]=V("think_mode")!="default"||V("reasoning_effort")!="default";}}
        Label LabelAt(string text,int x,int y,int w,int h,Font font,Color color){return new Label{Text=text,Left=x,Top=y,Width=w,Height=h,Font=font,ForeColor=color,BackColor=Color.Transparent,AutoEllipsis=true};}
        SoftButton ButtonAt(string text,int x,int y,int w,int h,bool main,Action click){var b=new SoftButton{Text=text,Left=x,Top=y,Width=w,Height=h,Primary=main,ForeColor=Ink,BackColor=Paper,Font=Font};b.Click+=delegate{click();};return b;}
        void ReleaseTips(Control control){tips.SetToolTip(control,null);foreach(Control child in control.Controls)ReleaseTips(child);}
        void DisposeControls(Control parent){while(parent.Controls.Count>0){Control child=parent.Controls[0];ReleaseTips(child);parent.Controls.RemoveAt(0);child.Dispose();}}
        void Rebuild(){
            using(LogicalLayout(ContentHost)){
            if(building||IsDisposed)return;building=true;SuspendLayout();try{
                bool fresh=body==null;
                if(fresh){
                var top=new WizardPanel{Name="TopBar",Dock=DockStyle.Top,Width=ContentHost.ClientSize.Width,Height=48,BackColor=Paper};topBar=top;
                var brand=LabelAt("",32,12,360,28,smallFont,Muted);brand.Name="PageBreadcrumb";top.Controls.Add(brand);
                var appearance=ButtonAt(T("外观","Appearance"),top.Width-234,7,94,34,false,ShowAppearance);appearance.Name="Appearance";appearance.Quiet=true;appearance.Anchor=AnchorStyles.Top|AnchorStyles.Right;top.Controls.Add(appearance);
                var language=ButtonAt(english?"中文":"English",top.Width-124,7,86,34,false,delegate{english=!english;Change("wizard_language",english?"en":"zh");RefreshLanguage();Rebuild();});language.Name="Language";language.Quiet=true;language.Anchor=AnchorStyles.Top|AnchorStyles.Right;top.Controls.Add(language);
                var bottom=new WizardPanel{Name="FooterBar",Dock=DockStyle.Bottom,Width=ContentHost.ClientSize.Width,Height=88,BackColor=Paper};bottomBar=bottom;
                var tools=ButtonAt(T("报告与工具","Reports & tools"),30,23,145,40,false,ShowTools);tools.Name="Tools";tools.Quiet=true;bottom.Controls.Add(tools);
                primary=ButtonAt("",bottom.Width-226,20,192,46,true,StartRun);primary.Name="PrimaryAction";primary.Anchor=AnchorStyles.Top|AnchorStyles.Right;bottom.Controls.Add(primary);AcceptButton=primary;
                body=new WizardScrollView{Name="ContentArea",Dock=DockStyle.Fill,Size=new Size(ContentHost.ClientSize.Width,ContentHost.ClientSize.Height-136),AutoScroll=true,BackColor=Paper};content=new WizardPanel{Width=860,Height=600,BackColor=Paper};body.Controls.Add(content);body.Resize+=delegate{FitContentWidth();};
                ContentHost.Controls.Add(body);ContentHost.Controls.Add(top);ContentHost.Controls.Add(bottom);
                }
                DisposeControls(content);topBar.Controls["Language"].Enabled=true;modelRows.Clear();body.AutoScrollPosition=Point.Empty;
                if(screen=="home")BuildHome();else if(screen=="downloads")BuildDownloads();else BuildSetup();
                string notice=String.IsNullOrEmpty(message)?Local(Get(schema,"notice","")):message;
                noticeLabel=LabelAt(notice,0,content.Height+4,content.Width,32,smallFont,Color.FromArgb(172,81,36));noticeLabel.Name="Notice";noticeLabel.Visible=notice.Length>0;noticeLabel.Anchor=AnchorStyles.Left|AnchorStyles.Right|AnchorStyles.Top;content.Controls.Add(noticeLabel);if(notice.Length>0)content.Height+=44;
                UpdateShell();content.Width=Math.Max(700,Math.Min(900,body.ClientSize.Width-64));content.Left=Math.Max(24,(body.ClientSize.Width-content.Width)/2);RelayoutWidth(content);RefreshTheme();
            }finally{ResumeLayout(true);building=false;}
            }
            FitContentWidth();
        }
        protected override void OnDpiLayoutChanged(EventArgs e){base.OnDpiLayoutChanged(e);FitContentWidth();}
        void FitContentWidth(){if(DpiScaling||building||content==null||body==null)return;content.Width=Math.Max(LayoutPixels(700),Math.Min(LayoutPixels(900),body.ClientSize.Width-LayoutPixels(64)));content.Left=Math.Max(LayoutPixels(24),(body.ClientSize.Width-content.Width)/2);RelayoutWidth(content);
            // The catalog owns its one scroll area. Keep its transfer controls in
            // the visible viewport instead of scrolling a second, oversized page.
            if(screen=="downloads"&&downloadsView!=null&&!downloadsView.IsDisposed){
                int notice=noticeLabel!=null&&noticeLabel.Visible?LayoutPixels(40):0;
                downloadsView.Height=Math.Max(LayoutPixels(220),body.ClientSize.Height-downloadsView.Top-LayoutPixels(8)-notice);
                content.Height=downloadsView.Bottom+LayoutPixels(8)+notice;
                if(noticeLabel!=null)noticeLabel.Top=downloadsView.Bottom+LayoutPixels(8);
                body.AutoScrollPosition=Point.Empty;
            }
        }
        void UpdateShell(){primary.Visible=screen!="downloads";primary.Text=screen=="home"?T(HasSelection()?"按上次设置启动":"开始使用",HasSelection()?"Use last settings":"Get started"):StartText();var label=topBar.Controls["PageBreadcrumb"];label.Text=screen=="home"?T("本地模型 · 由你掌控","Local models · Your control"):screen=="downloads"?T("模型下载","Model downloads"):ModeName();}
        void RefreshLanguage(){topBar.Controls["Language"].Text=english?"中文":"English";topBar.Controls["Appearance"].Text=T("外观","Appearance");bottomBar.Controls["Tools"].Text=T("报告与工具","Reports & tools");}
        ContextMenuStrip toolsMenu,appearanceMenu;
        ContextMenuStrip ContextMenu(ref ContextMenuStrip menu){if(menu==null)menu=new ContextMenuStrip{ShowImageMargin=false};while(menu.Items.Count>0){var item=menu.Items[0];menu.Items.RemoveAt(0);item.Dispose();}menu.Font=Font;return menu;}
        void ShowTools(){var menu=ContextMenu(ref toolsMenu);var reports=new ToolStripMenuItem(T("报告反馈与待发记录","Reports & pending submissions"));reports.Click+=delegate{OpenSettings("reports","");};menu.Items.Add(reports);var config=new ToolStripMenuItem(T("保存、导入导出与更新","Save, import, export & updates"));config.Click+=delegate{OpenSettings("maintenance","");};menu.Items.Add(config);WizardTheme.Apply(menu);menu.Show(bottomBar.Controls["Tools"],new Point(0,-menu.PreferredSize.Height));}
        void ShowAppearance(){var menu=ContextMenu(ref appearanceMenu);string[] values={"system","light","dark"};foreach(string value in values){string mode=value;var item=new ToolStripMenuItem(value=="system"?T("跟随系统","Follow system"):value=="light"?T("浅色","Light"):T("深色","Dark"));item.Checked=V("wizard_theme")==value||(V("wizard_theme")==""&&value=="system");item.Click+=delegate{Change("wizard_theme",mode);WizardTheme.SetPreference(mode);RefreshTheme();};menu.Items.Add(item);}WizardTheme.Apply(menu);menu.Show(topBar.Controls["Appearance"],new Point(0,34));}
        void RelayoutWidth(Control parent){foreach(Control c in parent.Controls){if(c.Name=="TaskCards"){int width=(c.Width-LayoutPixels(24))/3;for(int i=0;i<c.Controls.Count;i++){c.Controls[i].Left=i*(width+LayoutPixels(12));c.Controls[i].Width=width;}}}}
        bool HasSelection(){return Llm?V("model").Length>0:V("launch_mode")=="server"&&((Flag("server_embedding")&&V("embedding_model").Length>0)||(Flag("server_transcription")&&V("transcription_model").Length>0)||(Flag("server_speech")&&V("speech_model").Length>0&&V("speech_lexicon").Length>0));}
        string BaseName(string path){return path.Replace('\\','/').Substring(path.Replace('\\','/').LastIndexOf('/')+1);}
        string ModeName(){return V("launch_mode")=="server"?T("API 服务 · 连接应用","API · Connect apps"):V("launch_mode")=="benchmark"?T("Benchmark · 性能测试","Benchmark"):T("本机聊天","Chat locally");}
        string StartText(){return V("launch_mode")=="server"?T("启动服务","Start service"):V("launch_mode")=="benchmark"?T("开始测速","Run benchmark"):T("开始聊天","Start chatting");}
        void BuildHome(){
            int w=content.Width;content.Controls.Add(LabelAt(T("今天想做什么？","What would you like to do?"),0,20,w,52,titleFont,Ink));
            content.Controls.Add(LabelAt(T("选一件事，剩下的交给 MoE4All。","Choose a task. MoE4All takes care of the setup."),2,83,w,32,Font,Muted));
            var cards=new WizardPanel{Name="TaskCards",Left=0,Top=139,Width=w,Height=186,Anchor=AnchorStyles.Left|AnchorStyles.Top|AnchorStyles.Right};
            string[] modes={"chat","server","benchmark"};string[] titles={T("本机聊天","Chat with a model"),T("API · 连接应用","API · Connect apps"),T("Benchmark · 测速","Benchmark")};string[] details={T("选一个模型，直接开始对话。","Pick a model and start a conversation."),T("OpenAI 兼容 API，供 DSH 等应用连接。","OpenAI-compatible API for DSH and other apps."),T("测量 Prefill、Decode 与并发速度。","Measure Prefill, Decode and concurrency.")};
            for(int i=0;i<3;i++){string mode=modes[i];var card=new TaskTile{Name="Task_"+mode,TitleText=titles[i],DetailText=details[i],IconKind=i,Selected=V("launch_mode")==mode,Font=Font,Left=i*290,Top=0,Width=278,Height=180,AccessibleName=titles[i]};card.Click+=delegate{SelectTask(mode);};cards.Controls.Add(card);}content.Controls.Add(cards);
            if(HasSelection()){
                var recent=new WizardPanel{Left=0,Top=353,Width=w,Height=92,BackColor=Color.FromArgb(236,240,232),Anchor=AnchorStyles.Left|AnchorStyles.Top|AnchorStyles.Right};
                recent.Controls.Add(LabelAt(T("上次的设置","Last used"),20,14,130,24,smallFont,Muted));
                string name=Llm?BaseName(V("model")):T("独立模块服务","Standalone modules");recent.Controls.Add(LabelAt(ModeName()+"  ·  "+name,20,43,w-160,28,Font,Ink));
                var edit=ButtonAt(T("调整一下","Edit"),w-116,28,96,36,false,delegate{screen="setup";Rebuild();});edit.Anchor=AnchorStyles.Top|AnchorStyles.Right;edit.Quiet=true;recent.Controls.Add(edit);content.Controls.Add(recent);
            }else content.Controls.Add(LabelAt(T("还没有模型？选好用途后，可以直接下载。","No model yet? Choose a task to browse recommended downloads."),2,365,w,40,smallFont,Muted));
            content.Height=HasSelection()?450:420;
        }
        Panel Card(int top,int height,string title){var p=new SurfacePanel{Left=0,Top=top,Width=content.Width,Height=height,BackColor=WizardTheme.Current.Surface,Tag="WizardTheme.Surface",Anchor=AnchorStyles.Left|AnchorStyles.Right|AnchorStyles.Top};if(title.Length>0)p.Controls.Add(LabelAt(title,20,14,p.Width-40,30,sectionFont,Ink));content.Controls.Add(p);return p;}
        void CardAction(Panel card,string name,string title,Action action){var button=ButtonAt(title,card.Width-186,9,166,34,false,action);button.Name=name;button.Quiet=true;button.Anchor=AnchorStyles.Top|AnchorStyles.Right;foreach(Control c in card.Controls)if(c is Label&&c.Top==14)c.Width=Math.Max(80,card.Width-228);card.Controls.Add(button);button.BringToFront();}
        string requestedGroups="",settingsGroupFilter="",settingsPageId="",downloadTarget="";
        void OpenSection(string page,string field,string groups){requestedGroups=groups;try{OpenSettings(page,field);}finally{requestedGroups="";}}
        Panel capabilitiesSection,modelsSection,runtimeSection,testSection,connectionSection;
        ModelDownloadsView downloadsView;
        FlowLayoutPanel quickActions;
        readonly Dictionary<string,Panel> modelRows=new Dictionary<string,Panel>();
        void OpenDownloads(string target){downloadTarget=target;screen="downloads";Rebuild();}
        void BackFromDownloads(){if(downloadsView!=null&&downloadsView.Busy)return;screen="setup";Rebuild();}
        void BuildDownloads(){
            var back=ButtonAt(T("← 返回配置","← Back to configuration"),0,0,210,36,false,BackFromDownloads);back.Name="DownloadsBack";back.Quiet=true;content.Controls.Add(back);
            content.Controls.Add(LabelAt(T("模型下载","Model downloads"),0,46,content.Width,44,titleFont,Ink));
            var catalog=new List<ModelBundle>();foreach(var item in Items(schema.ContainsKey("download_catalog")?schema["download_catalog"]:null)){
                var bundle=new ModelBundle{Id=Get(item,"Id",""),Kind=Get(item,"Kind",""),Name=Local(Get(item,"Name","")),InfoUrl=Get(item,"InfoUrl",""),InstructionsUrl=Get(item,"InstructionsUrl",""),Primary=Get(item,"Primary",""),Description=Local(Get(item,"Description","")),UnavailableReason=Local(Get(item,"UnavailableReason",T("尚无兼容的直接下载包。","No compatible direct-download package yet.")))};
                long packageBytes;Int64.TryParse(Get(item,"PackageBytes","0"),out packageBytes);bundle.PackageBytes=packageBytes;
                if(bundle.Description.Length==0)bundle.Description=bundle.Kind=="llm"?T("聊天与 API · GGUF","Chat & API · GGUF"):bundle.Kind=="vision"?T("配套图片理解文件 · mmproj GGUF","Matching vision projector · mmproj GGUF"):bundle.Kind=="embedding"?T("文本向量服务 · 按需载入","Text embeddings · on-demand loading"):bundle.Kind=="audio"?T("语音转文字 · CPU · 按需载入","Speech to text · CPU · on-demand loading"):T("文字转语音 · CPU · 原生 Melo","Text to speech · CPU · native Melo");foreach(var f in Items(item.ContainsKey("Files")?item["Files"]:null)){long bytes;Int64.TryParse(Get(f,"Bytes","0"),out bytes);bundle.Files.Add(new ModelFile{Url=Get(f,"Url",""),RelativePath=Get(f,"RelativePath",""),Bytes=bytes});}catalog.Add(bundle);
            }
            string category=downloadTarget=="model"?"llm":downloadTarget=="vision_projector"?"vision":downloadTarget=="embedding_model"?"embedding":downloadTarget=="transcription_model"?"audio":downloadTarget.StartsWith("speech_",StringComparison.Ordinal)?"speech":"all";
            downloadsView=new ModelDownloadsView(catalog,Path.IsPathRooted(V("download_dir"))?V("download_dir"):Get(schema,"download_root",Path.Combine(AppDomain.CurrentDomain.BaseDirectory,"models")),category,english,demo,delegate(ModelBundle bundle,string root){
                string path=ModelDownload.PathUnder(root,bundle.Primary);if(bundle.Kind=="llm"){Change("model",path);var history=new List<string>();history.Add(path);if(state.ContainsKey("model_history"))foreach(object old in (IEnumerable)state["model_history"])if(!history.Contains(S(old)))history.Add(S(old));state["model_history"]=history.ToArray();}
                else if(bundle.Kind=="vision")Change("vision_projector",path);else if(bundle.Kind=="embedding")Change("embedding_model",path);else if(bundle.Kind=="audio")Change("transcription_model",path);else if(bundle.Kind=="speech"){Change("speech_model",path);Change("speech_lexicon",Path.GetDirectoryName(path));}
                BackFromDownloads();
            }){Font=Font,Left=0,Top=100,Width=content.Width,Height=Math.Max(490,body.ClientSize.Height-115),Anchor=AnchorStyles.Top|AnchorStyles.Left|AnchorStyles.Right};
            downloadsView.DirectoryChanged+=delegate(string directory){Change("download_dir",directory);};
            downloadsView.BusyChanged+=delegate{back.Enabled=!downloadsView.Busy;topBar.Controls["Language"].Enabled=!downloadsView.Busy;};content.Controls.Add(downloadsView);content.Height=downloadsView.Bottom+8;
        }
        void BuildSetup(){
            int w=content.Width;var back=ButtonAt(T("← 换个用途","← Change task"),0,0,138,34,false,delegate{screen="home";message="";Rebuild();});back.Quiet=true;content.Controls.Add(back);
            string title=V("launch_mode")=="server"?T("API 服务","API service"):V("launch_mode")=="benchmark"?T("Benchmark 性能测试","Benchmark"):T("本机聊天","Chat locally");
            content.Controls.Add(LabelAt(title,0,36,w,44,titleFont,Ink));
            int y=88;
            if(V("launch_mode")=="server"){
                var modules=Card(y,84,T("你想提供哪些能力？","Which capabilities do you need?"));capabilitiesSection=modules;string[] keys={"server_llm_enabled","server_vision","server_embedding","server_transcription","server_speech"};string[] labels={T("聊天","Chat"),T("识图","Vision"),T("知识库向量","Embedding"),T("语音转文字 · CPU","Transcription · CPU"),T("文字转语音 · CPU","Speech · CPU")};int px=20;
                for(int i=0;i<keys.Length;i++){string key=keys[i];var cb=new CapabilityChip{Name=key,Text=labels[i],Checked=Flag(key),Width=i==2?170:i==3?205:i==4?195:114,Left=px,Top=44,Font=Font};px+=cb.Width+10;cb.CheckedChanged+=delegate{if(syncing)return;Change(key,cb.Checked);RefreshSetupState();};modules.Controls.Add(cb);}
            }
            modelsSection=Card(y,102,T("使用哪个模型？","Choose your models"));foreach(string key in new[]{"model","vision_projector","embedding_model","transcription_model","speech_model","speech_lexicon"})AddModelRow(modelsSection,key,44);
            CardAction(modelsSection,"ModelOptions",T("模块高级设置 ›","Module settings ›"),delegate{OpenSettings("models","");});
            runtimeSection=Card(y,132,T("运行配置","Run configuration"));AddCoreEditor(runtimeSection,"device",20,48,runtimeSection.Width-40);AddCoreEditor(runtimeSection,"setup_mode",20,96,runtimeSection.Width-40);
            if(V("launch_mode")!="benchmark"){AddCoreEditor(runtimeSection,"context",20,144,runtimeSection.Width-40);AddCoreEditor(runtimeSection,"think_mode",20,192,runtimeSection.Width-40);AddCoreEditor(runtimeSection,"reasoning_effort",20,240,runtimeSection.Width-40);}
            CardAction(runtimeSection,"RuntimeOptions",T("资源高级设置 ›","Resource settings ›"),delegate{OpenSection("runtime","","basic,memory,diagnostics");});
            if(V("launch_mode")=="benchmark"){
                var plan=Card(y,188,T("测多少？","Choose a test"));testSection=plan;AddCoreEditor(plan,"bench_kind",20,52,plan.Width-40);AddCoreEditor(plan,"bench_parallel",20,108,plan.Width-40);
                var custom=ButtonAt(T("设置自定义测试…","Configure a custom test…"),20,108,310,38,false,delegate{OpenSettings("benchmark","");});custom.Name="CustomTest";plan.Controls.Add(custom);
                var summary=LabelAt(PlanSummary(),20,157,plan.Width-40,25,smallFont,Muted);summary.Name="PlanSummaryText";plan.Controls.Add(summary);
            }else if(V("launch_mode")=="server"){
                var connection=Card(y,90,T("API 连接","API connection"));connectionSection=connection;AddCoreEditor(connection,"server_addr",20,48,connection.Width-40);AddCoreEditor(connection,"server_parallel",20,96,connection.Width-40);
                CardAction(connection,"ApiOptions",T("鉴权与会话缓存 ›","Auth & session cache ›"),delegate{OpenSettings("api","");});
            }
            var options=new WizardFlowPanel{Left=0,Top=y,Width=w,Height=46,WrapContents=false,Anchor=AnchorStyles.Left|AnchorStyles.Right|AnchorStyles.Top};quickActions=options;
            if(V("launch_mode")!="benchmark"){var b=ButtonAt(T("生成高级设置 ›","Generation settings ›"),0,0,210,36,false,delegate{OpenSection("runtime","","generation,sampling");});b.Name="GenerationOptions";b.Quiet=true;options.Controls.Add(b);}
            content.Controls.Add(options);y+=40;
            RefreshSetupState();
        }
        bool ActiveModel(string key){return key=="model"?Llm:V("launch_mode")=="server"&&(key=="vision_projector"?Llm&&Flag("server_vision"):key=="embedding_model"?Flag("server_embedding"):key=="transcription_model"?Flag("server_transcription"):Flag("server_speech"));}
        void RefreshSetupState(){
            if(screen!="setup"||modelsSection==null||modelsSection.IsDisposed)return;
            using(LogicalLayout(ContentHost)){
            int scroll=-body.AutoScrollPosition.Y;content.SuspendLayout();try{
                syncing=true;
                int y=88;
                if(capabilitiesSection!=null&&!capabilitiesSection.IsDisposed){int x=20,cy=44;foreach(Control c in capabilitiesSection.Controls){var cb=c as CheckBox;if(cb==null)continue;cb.Checked=Flag(cb.Name);bool visible=cb.Name!="server_vision"||Llm;cb.Visible=visible;if(visible){if(x+cb.Width>capabilitiesSection.Width-20){x=20;cy+=42;}cb.Left=x;cb.Top=cy;x+=cb.Width+10;}}capabilitiesSection.Height=cy+42;capabilitiesSection.Top=y;y+=capabilitiesSection.Height+10;}
                int rowY=44,count=0;foreach(string key in new[]{"model","vision_projector","embedding_model","transcription_model","speech_model","speech_lexicon"}){var row=modelRows[key];bool visible=ActiveModel(key);row.Visible=visible;if(visible){row.Top=rowY;rowY+=58;count++;}var editor=row.Controls[key] as WizardComboBox;if(editor!=null&&editor.Text!=V(key))editor.Text=V(key);}
                modelsSection.Visible=count>0;modelsSection.Height=44+count*58;if(count>0){modelsSection.Top=y;y+=modelsSection.Height+10;}
                bool device=Llm||(V("launch_mode")=="server"&&Flag("server_embedding"));runtimeSection.Visible=device;runtimeSection.Height=Llm?(V("launch_mode")=="benchmark"?142:286):90;SetCoreVisible(runtimeSection,"setup_mode",Llm);SetCoreVisible(runtimeSection,"context",Llm);SetCoreVisible(runtimeSection,"think_mode",Llm);SetCoreVisible(runtimeSection,"reasoning_effort",Llm&&V("think_mode")!="no-think");
                modelsSection.Controls["ModelOptions"].Visible=HasAdvancedFields("models");runtimeSection.Controls["RuntimeOptions"].Visible=HasAdvancedFields("runtime");
                if(device){runtimeSection.Top=y;y+=runtimeSection.Height+10;}SyncEditors(runtimeSection);
                if(testSection!=null&&!testSection.IsDisposed){testSection.Top=y;bool manual=V("bench_kind")=="manual";testSection.Controls["bench_parallel"].Visible=!manual;testSection.Controls["Label_bench_parallel"].Visible=!manual;testSection.Controls["CustomTest"].Visible=manual;testSection.Controls["PlanSummaryText"].Text=PlanSummary();SyncEditors(testSection);y+=testSection.Height+14;}
                if(connectionSection!=null&&!connectionSection.IsDisposed){connectionSection.Height=Llm?142:90;SetCoreVisible(connectionSection,"server_parallel",Llm);connectionSection.Top=y;y+=connectionSection.Height+10;SyncEditors(connectionSection);}
                quickActions.Top=y;quickActions.Visible=Llm;content.Height=y+(Llm&&quickActions.Controls.Count>0?54:12);
                if(noticeLabel!=null&&!noticeLabel.IsDisposed){noticeLabel.Top=content.Height+4;if(noticeLabel.Visible)content.Height+=44;}
                if(primary!=null)UpdateShell();
            }finally{syncing=false;content.ResumeLayout(true);body.AutoScrollPosition=new Point(0,scroll);}
            }
        }
        void SyncEditors(Control parent){foreach(Control c in parent.Controls){var cb=c as WizardComboBox;if(cb!=null&&cb.DropDownStyle==ComboBoxStyle.DropDownList){for(int i=0;i<cb.Items.Count;i++){var choice=cb.Items[i] as Choice;if(choice!=null&&choice.Value==V(cb.Name)&&cb.SelectedIndex!=i){cb.SelectedIndex=i;break;}}}else if(c is TextBox&&Field(c.Name)!=null){if(c.Text!=V(c.Name))c.Text=V(c.Name);}else if(c is CheckBox&&Field(c.Name)!=null)((CheckBox)c).Checked=Flag(c.Name);else if(c is WizardInputFrame)SyncEditors(c);}}
        string QuickText(string key){if(key=="context")return T("上下文长度：","Context: ")+(V(key)==""?T("自动","Automatic"):V(key))+"  ›";if(key=="ubatch")return T("性能细项…","Performance options…");return T("思考：","Reasoning: ")+(V(key)=="default"?T("跟随模型","Model default"):V(key)=="no-think"?T("关闭","Off"):T("开启","On"))+"  ›";}
        string PlanSummary(){string rows=V("bench_kind")=="quick"?"6K":V("bench_kind")=="standard"?"6K + 30K":"6K + 30K + 150K";return V("bench_kind")=="manual"?T("按自定义配置测试，结果自动保存。","Your custom test; results are saved automatically."):rows+T(" 上下文 · 512 预热 + 1024 正式生成 / 路"," context · 512 warmup + 1024 measured tokens / lane");}
        string Friendly(string key,string fallback){switch(key){case "device":return T("计算设备","Device");case "setup_mode":return T("资源策略","Resource profile");case "think_mode":return T("思考模式","Reasoning mode");case "reasoning_effort":return T("思考强度","Reasoning effort");case "bench_kind":return T("测试方案","Test plan");case "bench_parallel":return T("并发路数","Concurrency");case "server_addr":return T("监听地址","Listen address");case "context":return T("上下文长度","Context length");case "server_parallel":return T("并发槽位","Concurrent slots");case "server_auth":return T("启用 API Key","Require an API key");case "server_api_key":return T("本次 API Key","API key for this run");default:return Local(fallback);}}
        string OptionName(string key,Dictionary<string,object> option){string v=Get(option,"value","");if(key=="setup_mode")return v=="conservative"?T("自动保守 · 留出余量","Automatic conservative"):v=="aggressive"?T("自动激进 · 优先性能","Automatic aggressive"):T("手动配置","Manual");if(key=="bench_kind")return v=="quick"?T("快速 · 6K，一次","Quick · 6K, once"):v=="standard"?T("标准 · 6K 和 30K，各一次","Standard · 6K & 30K, once each"):v=="full"?T("完整 · 6K、30K、150K，各一次","Full · 6K, 30K & 150K, once each"):T("自定义","Custom");if(key=="device"&&v=="")return T("自动选择（推荐）","Automatic (recommended)");return Local(Get(option,"label",v));}
        void SetCoreVisible(Control parent,string key,bool visible){var editor=parent.Controls.Find(key,true);if(editor.Length>0){Control surface=editor[0].Parent is WizardInputFrame?editor[0].Parent:editor[0];surface.Visible=visible;}var label=parent.Controls["Label_"+key];if(label!=null)label.Visible=visible;}
        bool MainField(string key){return key=="device"||key=="setup_mode"||key=="context"||key=="think_mode"||key=="reasoning_effort"||key=="server_addr"||key=="server_parallel"||key=="model"||key=="server_llm_enabled"||key=="server_vision"||key=="vision_projector"||key=="server_embedding"||key=="embedding_model"||key=="server_transcription"||key=="transcription_model"||key=="server_speech"||key=="speech_model"||key=="speech_lexicon"||key=="action.download";}
        bool HasAdvancedFields(string page){return fields.Exists(delegate(Dictionary<string,object> f){return Get(f,"page","")==page&&!MainField(Get(f,"key",""))&&VisibleField(f);});}
        Control FieldSurface(Control editor){var input=editor as TextBox;return input==null?editor:new WizardInputFrame(input);}
        void AddCoreEditor(Panel parent,string key,int x,int y,int width){var field=Field(key);if(field==null)return;var label=LabelAt(Friendly(key,Get(field,"label",key)),x,y,width>600?160:150,34,Font,Muted);label.Name="Label_"+key;parent.Controls.Add(label);var editor=Editor(field,false);var surface=FieldSurface(editor);surface.Left=x+170;surface.Top=y-4;surface.Width=width-170;surface.Anchor=AnchorStyles.Left|AnchorStyles.Right|AnchorStyles.Top;parent.Controls.Add(surface);}
        void AddModelRow(Panel parent,string key,int y){
            var row=new WizardPanel{Name="ModelRow_"+key,Left=20,Top=y,Width=parent.Width-40,Height=58,BackColor=parent.BackColor,Tag="WizardTheme.Surface",Anchor=AnchorStyles.Left|AnchorStyles.Right|AnchorStyles.Top};modelRows[key]=row;parent.Controls.Add(row);
            string label=key=="model"?T("聊天模型","Chat model"):key=="vision_projector"?T("配套视觉文件","Vision file"):key=="embedding_model"?T("向量模型","Embedding model"):key=="transcription_model"?T("语音识别模型","Transcription model"):key=="speech_model"?T("语音合成模型","Speech model"):T("合成资源目录","Speech assets");row.Controls.Add(LabelAt(label,0,0,150,30,Font,Muted));
            var input=new WizardComboBox{Name=key,Left=164,Top=0,Width=row.Width-338,Height=32,DropDownStyle=ComboBoxStyle.DropDown,Font=Font,Anchor=AnchorStyles.Left|AnchorStyles.Right|AnchorStyles.Top,MaxDropDownItems=8};
            if(key=="model"&&state.ContainsKey("model_history")&&state["model_history"] is IEnumerable&&!(state["model_history"] is string))foreach(object item in (IEnumerable)state["model_history"])if(S(item).Length>0)input.Items.Add(S(item));input.Text=V(key);tips.SetToolTip(input,V(key));input.TextChanged+=delegate{if(syncing)return;Change(key,input.Text.Trim().Trim('"'));tips.SetToolTip(input,input.Text);};row.Controls.Add(input);
            var browse=ButtonAt(T(key=="speech_lexicon"?"选目录":"选文件",key=="speech_lexicon"?"Folder":"Browse"),row.Width-164,0,80,36,false,delegate{if(Browse(key))input.Text=V(key);});browse.Anchor=AnchorStyles.Right|AnchorStyles.Top;row.Controls.Add(browse);
            var download=ButtonAt(T("下载","Get model"),row.Width-78,0,76,36,false,delegate{OpenDownloads(key);});download.Name="Download_"+key;download.Quiet=true;download.Anchor=AnchorStyles.Right|AnchorStyles.Top;row.Controls.Add(download);
            row.Controls.Add(LabelAt(key=="speech_lexicon"?T("包含 lexicon.txt、tokens.txt 和 config.json 的资源目录。","Folder containing lexicon.txt, tokens.txt and config.json."):key=="speech_model"?T("Melo GGUF · CPU，首次请求时载入，需要语音合成版引擎。","Melo GGUF · CPU, loads on demand; requires a TTS-enabled engine."):key=="transcription_model"?T("语音转录使用 CPU，首次请求时载入。","Transcription uses the CPU and loads on demand."):T("可选择文件，也可粘贴路径。","Browse a file or paste its path."),164,34,row.Width-174,24,smallFont,Muted));
        }
        sealed class Choice {public string Value,Label;public override string ToString(){return Label;}}
        Control Editor(Dictionary<string,object> f,bool advanced){
            string key=Get(f,"key",""),kind=Get(f,"kind","text");Control result;
            if(kind=="toggle") {var box=new ToggleSwitch{Checked=Flag(key)};box.CheckedChanged+=delegate{if(syncing)return;Change(key,box.Checked);if(advanced)QueueSettingsRefresh();};result=box;}
            else if(kind=="choice"){
                var combo=new WizardComboBox{DropDownStyle=ComboBoxStyle.DropDownList,DrawMode=DrawMode.OwnerDrawFixed,FlatStyle=FlatStyle.Flat,Height=32,ItemHeight=26,IntegralHeight=false,DropDownHeight=240};
                foreach(var option in Items(f.ContainsKey("options")?f["options"]:null))combo.Items.Add(new Choice{Value=Get(option,"value",""),Label=OptionName(key,option)});
                for(int i=0;i<combo.Items.Count;i++)if(((Choice)combo.Items[i]).Value==V(key))combo.SelectedIndex=i;
                combo.SelectedIndexChanged+=delegate{if(syncing||combo.SelectedItem==null)return;Change(key,((Choice)combo.SelectedItem).Value);if(advanced)QueueSettingsRefresh();else if(key=="bench_kind"||key=="setup_mode"||key=="think_mode")RefreshSetupState();};result=combo;
            }else if(kind=="action") {var button=ButtonAt(Local(Get(f,"label",key)),0,0,250,36,false,delegate{HandleAction(Get(f,"action",""));});result=button;}
            else if(kind=="path"){
                var panel=new WizardPanel{Height=38,Width=382};var text=new WizardTextBox{Text=V(key),Width=292};text.Name="PathText";text.TextChanged+=delegate{if(!syncing)Change(key,text.Text);};var frame=new WizardInputFrame(text){Left=0,Top=0,Width=292,Anchor=AnchorStyles.Top|AnchorStyles.Left|AnchorStyles.Right};var browse=ButtonAt(T("浏览","Browse"),302,0,80,38,false,delegate{if(Browse(key)){text.Text=V(key);}});browse.Name="Browse_"+key;browse.Anchor=AnchorStyles.Top|AnchorStyles.Right;panel.Controls.Add(frame);panel.Controls.Add(browse);result=panel;
            }else {var text=new WizardTextBox{Text=V(key),UseSystemPasswordChar=kind=="secret",BorderStyle=BorderStyle.FixedSingle,Height=32};text.TextChanged+=delegate{if(!syncing)Change(key,text.Text);};result=text;}
            result.Name=key;result.Font=Font;result.ForeColor=Ink;result.AccessibleName=Friendly(key,Get(f,"label",key));tips.SetToolTip(result,Local(Get(f,"help","")));return result;
        }
        bool Browse(string key){if(demo){SetNotice(T("演示不会选择或下载本机文件。","Preview does not select or download local files."));return false;}try{
            if(key=="speech_lexicon"||key.EndsWith("_dir",StringComparison.Ordinal)){using(var dialog=new FolderBrowserDialog()){dialog.Description=T("选择保存位置","Choose a folder");if(Directory.Exists(V(key)))dialog.SelectedPath=V(key);if(dialog.ShowDialog(settings??this)!=DialogResult.OK)return false;Change(key,dialog.SelectedPath);}}
            else using(var dialog=new OpenFileDialog()){dialog.Filter=key=="config_path"?"TOML (*.toml)|*.toml|All files (*.*)|*.*":"GGUF (*.gguf)|*.gguf|All files (*.*)|*.*";dialog.Title=T("选择模型文件","Choose a model file");if(File.Exists(V(key)))dialog.FileName=V(key);if(dialog.ShowDialog(settings??this)!=DialogResult.OK)return false;Change(key,dialog.FileName);}return true;
        }catch(Exception ex){SetNotice(ex.Message);return false;}}
        void SetNotice(string value){message=value;if(noticeLabel!=null&&!noticeLabel.IsDisposed){noticeLabel.Text=value;noticeLabel.Visible=value.Length>0;if(value.Length>0)content.Height=Math.Max(content.Height,noticeLabel.Bottom+8);}}
        void StartRun(){
            if(screen=="downloads")return;
            if(screen=="home"&&!HasSelection()){screen="setup";Rebuild();return;}
            if(demo){SetNotice(T("这是演示，实际使用请打开 Start-INFR-Wizard.cmd。","Preview only. Open Start-INFR-Wizard.cmd to run models."));return;}
            if(primary!=null)primary.Enabled=false;Finish("start");
        }
        void HandleAction(string value){if(value=="cancel"){if(settings!=null)settings.Close();Close();return;}if(value=="start"){StartRun();return;}if(value=="download"){OpenDownloads("");return;}if(value.StartsWith("task.",StringComparison.Ordinal)){SelectTask(value.Substring(5));QueueSettingsRefresh();return;}if(value.StartsWith("page.",StringComparison.Ordinal)){OpenSettings(value.Substring(5),"");return;}Finish(value);}
        void QueueSettingsRefresh(){if(settings!=null&&!settings.IsDisposed)FillSettings();}
        SettingsPages tabs;
        sealed class SettingsRow{public Dictionary<string,object> Field;public Panel Row;public Control Editor;}
        sealed class SettingsGroup{public WizardScrollView Page;public SurfacePanel Card;public bool Expanded;public SoftButton Toggle;public readonly List<SettingsRow> Rows=new List<SettingsRow>();}
        readonly List<SettingsGroup> settingsGroups=new List<SettingsGroup>();
        readonly List<WizardScrollView> settingsPages=new List<WizardScrollView>();
        bool openingSettings;
        public void OpenSettings(string pageId,string fieldKey){
            if(settings!=null)return;
            settingsPageId=pageId;settingsGroupFilter=requestedGroups;
            bool reasoning=pageId=="runtime"&&(fieldKey=="think_mode"||fieldKey=="reasoning_effort"||settingsGroupFilter.Contains("generation"));
            if(reasoning&&V("launch_mode")!="benchmark"&&Llm&&!String.Equals(V("model"),V("reasoning_probe_model"),StringComparison.OrdinalIgnoreCase)&&(File.Exists(V("model"))||Directory.Exists(V("model")))){reopenSettingsPage=pageId;reopenSettingsField=fieldKey;screen="setup";Finish("model");return;}
            string before=Json.Serialize(state);bool priorDirty=dirty;string oldModel=V("model");bool accepted=false;
            string title=pageId=="models"?T("模块高级设置","Module settings"):pageId=="runtime"?(settingsGroupFilter.Contains("generation")?T("生成高级设置","Generation settings"):T("资源高级设置","Resource settings")):pageId=="api"?T("鉴权与会话缓存","Auth & session cache"):pageId=="benchmark"?T("自定义测速","Custom benchmark"):pageId=="reports"?T("报告反馈","Reports"):T("配置与更新","Configuration & updates");
            using(var dialog=new WizardWindow{ShowWindowButtons=false,Text=title,Font=Font,BackColor=Paper,ForeColor=Ink,StartPosition=FormStartPosition.CenterParent,Size=new Size(900,740),MinimumSize=new Size(760,600),ShowInTaskbar=false,KeyPreview=true}){
                settings=dialog;openingSettings=true;settingsGroups.Clear();settingsPages.Clear();
                tabs=new SettingsPages{Name="SettingsContent",Dock=DockStyle.Fill};var host=dialog.ContentHost;var buttons=new WizardPanel{Dock=DockStyle.Bottom,Width=host.ClientSize.Width,Height=76,BackColor=Paper};
                var cancel=ButtonAt(T("取消修改","Cancel"),24,18,120,40,false,delegate{dialog.Close();});buttons.Controls.Add(cancel);var done=ButtonAt(T("完成","Done"),host.ClientSize.Width-150,18,118,40,true,delegate{accepted=true;dialog.Close();});done.Anchor=AnchorStyles.Top|AnchorStyles.Right;buttons.Controls.Add(done);host.Controls.Add(tabs);host.Controls.Add(buttons);dialog.AcceptButton=done;dialog.CancelButton=cancel;
                if(fieldKey.Length>0&&message.Length>0){var error=new WizardPanel{Dock=DockStyle.Top,Height=50,BackColor=WizardTheme.Current.SurfaceAlt};var label=LabelAt(message,20,12,host.ClientSize.Width-40,30,smallFont,Color.FromArgb(172,81,36));label.Anchor=AnchorStyles.Top|AnchorStyles.Left|AnchorStyles.Right;error.Controls.Add(label);tips.SetToolTip(label,message);host.Controls.Add(error);}
                FillSettings();foreach(WizardScrollView tab in tabs.TabPages)if(tab.Name==pageId)tabs.SelectedTab=tab;
                if(fieldKey.Length>0){foreach(var group in settingsGroups)foreach(var row in group.Rows)if(Get(row.Field,"key","")==fieldKey)group.Expanded=true;FillSettings();}
                if(fieldKey.Length>0)dialog.Shown+=delegate{Control[] found=tabs.Controls.Find(fieldKey,true);if(found.Length>0){found[0].Focus();var input=found[0] as TextBox;if(input!=null)input.SelectAll();}};
                openingSettings=false;dialog.RefreshTheme();
                try{dialog.ShowDialog(this);}finally{ReleaseTips(dialog);foreach(var page in settingsPages)if(!tabs.TabPages.Contains(page)){ReleaseTips(page);page.Dispose();}settings=null;tabs=null;settingsGroups.Clear();settingsPages.Clear();}
            }
            if(finishing)return;
            if(!accepted){state.Clear();foreach(var pair in Dict(Json.DeserializeObject(before)))state[pair.Key]=pair.Value;dirty=priorDirty;}
            if(accepted)message="";
            if(accepted&&oldModel!=V("model")){screen="setup";Finish("model");return;}
            if(screen=="setup")RefreshSetupState();else Rebuild();
        }
        void FillSettings(){
            if(settings==null)return;using(((WizardWindow)settings).LogicalLayout(((WizardWindow)settings).ContentHost)){
            foreach(var detached in settingsPages)if(!tabs.Controls.Contains(detached))WizardDpi.AdaptTree(detached,tabs);
            if(tabs==null)return;bool initial=settingsPages.Count==0;string active=tabs.SelectedTab==null?"":tabs.SelectedTab.Name;tabs.SuspendLayout();bool priorSync=syncing;syncing=true;
            try{
                if(initial){
                    foreach(var page in pages){string id=Get(page,"id","");if(id=="home"||id!=settingsPageId)continue;
                        var tab=new WizardScrollView{Name=id,BackColor=Paper,AutoScroll=true,Size=new Size(Math.Max(680,tabs.ClientSize.Width),Math.Max(400,tabs.ClientSize.Height))};settingsPages.Add(tab);
                        string groupId="";SettingsGroup group=null;
                        foreach(var field in fields){if(MainField(Get(field,"key",""))||Get(field,"page","")!=id||Get(field,"action","")=="start"||Get(field,"action","").StartsWith("task.",StringComparison.Ordinal))continue;if(settingsGroupFilter.Length>0&&Array.IndexOf(settingsGroupFilter.Split(','),Get(field,"group",""))<0)continue;
                            string next=Get(field,"group","");if(group==null||next!=groupId){groupId=next;var meta=Items(schema.ContainsKey("groups")?schema["groups"]:null).Find(delegate(Dictionary<string,object> item){return Get(item,"id","")==next&&Get(item,"page","")==id;});
                                group=new SettingsGroup{Page=tab,Expanded=meta==null||Get(meta,"open","true")=="true",Card=new SurfacePanel{Left=16,Width=tab.ViewportWidth-32,Height=48,BackColor=WizardTheme.Current.Surface,Tag="WizardTheme.Surface",Anchor=AnchorStyles.Left|AnchorStyles.Right|AnchorStyles.Top}};
                                group.Card.Controls.Add(LabelAt(meta==null?next:Local(Get(meta,"label",next)),18,14,group.Card.Width-118,28,sectionFont,Ink));
                                group.Card.Name="Group_"+groupId;SettingsGroup captured=group;group.Toggle=ButtonAt("",group.Card.Width-90,8,76,34,false,delegate{captured.Expanded=!captured.Expanded;FillSettings();});group.Toggle.Name="Expand_"+groupId;group.Toggle.Quiet=true;group.Toggle.Anchor=AnchorStyles.Top|AnchorStyles.Right;group.Card.Controls.Add(group.Toggle);settingsGroups.Add(group);tab.Controls.Add(group.Card);
                            }
                            string key=Get(field,"key","");bool actionField=Get(field,"kind","")=="action",toggleField=Get(field,"kind","")=="toggle";var row=new WizardPanel{Left=18,Width=group.Card.Width-36,Height=actionField?48:toggleField?48:80,BackColor=WizardTheme.Current.Surface,Tag="WizardTheme.Surface",Anchor=AnchorStyles.Left|AnchorStyles.Right|AnchorStyles.Top};
                            if(!actionField){var label=LabelAt(Friendly(key,Get(field,"label",key)),0,toggleField?8:0,toggleField?row.Width-86:row.Width,26,toggleField?Font:smallFont,toggleField?Ink:Muted);row.Controls.Add(label);}
                            var editor=Editor(field,true);var surface=FieldSurface(editor);surface.Left=toggleField?row.Width-62:0;surface.Top=actionField?2:toggleField?6:29;surface.Width=toggleField?58:actionField?Math.Min(360,row.Width):row.Width;surface.Anchor=toggleField?AnchorStyles.Top|AnchorStyles.Right:actionField?AnchorStyles.Top|AnchorStyles.Left:AnchorStyles.Top|AnchorStyles.Left|AnchorStyles.Right;row.Controls.Add(surface);surface.BringToFront();group.Card.Controls.Add(row);group.Rows.Add(new SettingsRow{Field=field,Row=row,Editor=editor});
                        }
                    }
                }
                foreach(var tab in settingsPages){int scroll=-tab.AutoScrollPosition.Y,y=14;bool any=false;tab.SuspendLayout();tab.AutoScrollPosition=Point.Empty;
                    foreach(var group in settingsGroups){if(group.Page!=tab)continue;int rowY=54;bool used=false;foreach(var row in group.Rows){bool relevant=VisibleField(row.Field);used|=relevant;row.Row.Visible=relevant&&group.Expanded;if(relevant&&group.Expanded){row.Row.Top=rowY;rowY+=row.Row.Height;}SyncFieldEditor(row.Editor,Get(row.Field,"key",""));}
                        group.Card.Visible=used;group.Card.Height=group.Expanded?rowY+6:52;group.Toggle.Text=group.Expanded?T("收起","Less"):T("展开","More");if(used){any=true;group.Card.Top=y;y+=group.Card.Height+14;}
                    }
                    if(any&&!tabs.TabPages.Contains(tab))tabs.TabPages.Add(tab);else if(!any&&tabs.TabPages.Contains(tab))tabs.TabPages.Remove(tab);tab.AutoScrollMinSize=new Size(0,y);tab.ResumeLayout(true);tab.AutoScrollPosition=new Point(0,Math.Min(scroll,Math.Max(0,y-tab.ClientSize.Height)));
                }
                foreach(WizardScrollView tab in tabs.TabPages)if(tab.Name==active||tabs.SelectedTab==null)tabs.SelectedTab=tab;
                if(initial)WizardTheme.Apply(tabs);
            }finally{syncing=priorSync;tabs.ResumeLayout(true);}
            }
        }
        void SyncFieldEditor(Control editor,string key){var combo=editor as WizardComboBox;if(combo!=null){for(int i=0;i<combo.Items.Count;i++){var option=combo.Items[i] as Choice;if(option!=null&&option.Value==V(key)&&combo.SelectedIndex!=i){combo.SelectedIndex=i;break;}}return;}var toggle=editor as CheckBox;if(toggle!=null){toggle.Checked=Flag(key);return;}var input=editor as TextBox;if(input!=null){if(input.Text!=V(key))input.Text=V(key);return;}var paths=editor.Controls.Find("PathText",true);var path=paths.Length>0?paths[0] as TextBox:null;if(path!=null&&path.Text!=V(key))path.Text=V(key);}
        void ClosingWizard(object sender,FormClosingEventArgs e){
            lastPosition=Location;if(finishing)return;
            if(!dirty){action="cancel";finishing=true;return;}
            using(var prompt=new WizardWindow{ShowWindowButtons=false,Text=T("保留这次修改吗？","Keep your changes?"),Font=Font,Size=new Size(540,260),MaximizeBox=false,MinimizeBox=false,StartPosition=FormStartPosition.CenterParent,ShowInTaskbar=false,BackColor=Paper}){
                prompt.ContentHost.Controls.Add(LabelAt(T("下次可以接着用这些设置。","You can use these settings next time."),24,24,480,44,sectionFont,Ink));
                var save=ButtonAt(T("保存并退出","Save & exit"),22,108,145,42,true,delegate{prompt.DialogResult=DialogResult.Yes;});var discard=ButtonAt(T("不保存","Discard"),175,108,130,42,false,delegate{prompt.DialogResult=DialogResult.No;});var keep=ButtonAt(T("继续编辑","Keep editing"),315,108,180,42,false,delegate{prompt.DialogResult=DialogResult.Cancel;});prompt.ContentHost.Controls.AddRange(new Control[]{save,discard,keep});prompt.CancelButton=keep;prompt.AcceptButton=keep;prompt.RefreshTheme();var result=prompt.ShowDialog(this);
                if(result==DialogResult.Cancel){e.Cancel=true;return;}action=result==DialogResult.Yes?"save_exit":"cancel";finishing=true;
            }
        }
        void Finish(string value){if(finishing)return;action=value;finishing=true;if(settings!=null)settings.Close();Close();}
        public string Result(string value){var exported=Dict(Json.DeserializeObject(Json.Serialize(state)));var secrets=new Dictionary<string,object>();foreach(var f in fields)if(Get(f,"kind","")=="secret"){string key=Get(f,"key","");if(exported.ContainsKey(key)){secrets[key]=exported[key];exported.Remove(key);}}exported.Remove("last_command");exported.Remove("api_key");return Json.Serialize(new Dictionary<string,object>{{"action",value},{"state",exported},{"secrets",secrets},{"page",screen=="home"?"home":"setup"},{"focus",""},{"groups",new string[0]},{"dirty",dirty},{"settings_page",reopenSettingsPage},{"settings_field",reopenSettingsField},{"settings_groups",settingsGroupFilter},{"download_target",downloadTarget}});}
        public static void RestoreConsole(){if(hiddenConsole){ShowWindow(GetConsoleWindow(),5);hiddenConsole=false;}}
        public static string Run(string stateJson,string schemaJson,string version,bool preview,string initialMessage){
            IntPtr console=GetConsoleWindow();if(console!=IntPtr.Zero&&IsWindowVisible(console)){ShowWindow(console,0);hiddenConsole=true;}
            try{using(var ui=new NativeWizard(stateJson,schemaJson,version,preview,initialMessage)){Application.Run(ui);string result=ui.Result(ui.action);if(ui.action=="cancel"||ui.action=="save_exit")RestoreConsole();return result;}}
            catch{RestoreConsole();throw;}
        }
        protected override void Dispose(bool disposing){if(disposing){if(toolsMenu!=null)toolsMenu.Dispose();if(appearanceMenu!=null)appearanceMenu.Dispose();tips.Dispose();base.Dispose(true);titleFont.Dispose();sectionFont.Dispose();smallFont.Dispose();Font.Dispose();return;}base.Dispose(false);}
    }
}
