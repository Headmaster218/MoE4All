// Standalone .NET Framework 4 WinForms theme/chrome. No wizard/schema dependencies.
// Derive NativeWizard from WizardWindow and rebuild only ContentHost.Controls.
// Call RefreshTheme after rebuilding; never clear the window's Controls collection.
// Optional string Tags: WizardTheme.Muted, .Text, .Accent, .Primary,
// .Background, .Surface, .SurfaceAlt. Existing business Tags are never modified.
using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security;
using System.Windows.Forms;
using Microsoft.Win32;

namespace MoE4All.Launcher
{
    public static class WizardBrand
    {
        public const string Website = "https://moe4all.com/";
        [DllImport("shell32.dll", CharSet=CharSet.Unicode)] static extern int SetCurrentProcessExplicitAppUserModelID(string id);
        internal static void Initialize(){try{SetCurrentProcessExplicitAppUserModelID("MoE4All.Engine");}catch(EntryPointNotFoundException){}}
        public static Icon CreateIcon(int pixels){
            using(var stream=typeof(WizardBrand).Assembly.GetManifestResourceStream("MoE4All.Icon")){
                if(stream==null)throw new InvalidOperationException("The launcher icon resource is missing. Extract the complete package.");
                using(var icon=new Icon(stream,new Size(pixels,pixels)))return (Icon)icon.Clone();
            }
        }
    }

    // PowerShell is the GUI host. A bootstrap manifest cannot set its UI thread's
    // awareness. Call before constructing any controls (including ToolTips).
    public static class WizardDpi
    {
        [DllImport("user32.dll")] static extern IntPtr SetThreadDpiAwarenessContext(IntPtr context);
        [DllImport("user32.dll")] static extern bool SetProcessDpiAwarenessContext(IntPtr context);
        [DllImport("shcore.dll")] static extern int SetProcessDpiAwareness(int awareness);
        [DllImport("user32.dll")] static extern bool SetProcessDPIAware();
        [DllImport("user32.dll")] static extern uint GetDpiForWindow(IntPtr hwnd);
        public static void Initialize()
        {
            try {
                // A thread override also works when powershell.exe has already
                // selected a process awareness. Reapply on every calling UI thread.
                if (SetThreadDpiAwarenessContext(new IntPtr(-4)) != IntPtr.Zero) return;
                if (SetThreadDpiAwarenessContext(new IntPtr(-3)) != IntPtr.Zero) return;
            } catch (EntryPointNotFoundException) { }
            try { if (SetProcessDpiAwarenessContext(new IntPtr(-4))) return; } catch (EntryPointNotFoundException) { }
            try { if (SetProcessDpiAwareness(2) == 0) return; } catch (DllNotFoundException) { } catch (EntryPointNotFoundException) { }
            try { SetProcessDPIAware(); } catch (EntryPointNotFoundException) { }
        }
        internal static int WindowDpi(Control control)
        {
            try { uint value = GetDpiForWindow(control.Handle); if (value > 0) return (int)value; } catch (EntryPointNotFoundException) { }
            using (Graphics g = control.CreateGraphics()) return Math.Max(96, (int)Math.Round(g.DpiX));
        }
        // Preserve design point sizes across monitor changes. Pixel-unit native
        // fonts avoid the .NET Framework host caching an HFONT at system DPI.
        // Each control owns its replacement; caller-owned design fonts stay alive.
        sealed class FontDesign { internal string Family; internal float Points; internal FontStyle Style; }
        sealed class OwnedFont { internal Font Value; }
        sealed class TreeDpi { internal int Value=96; }
        static readonly ConditionalWeakTable<Font, FontDesign> designs = new ConditionalWeakTable<Font, FontDesign>();
        static readonly ConditionalWeakTable<Control, OwnedFont> owned = new ConditionalWeakTable<Control, OwnedFont>();
        static readonly ConditionalWeakTable<Control, TreeDpi> treeDpis = new ConditionalWeakTable<Control, TreeDpi>();
        internal static void AdaptTree(Control child,Control parent){int from=treeDpis.GetOrCreateValue(child).Value,to=Dpi(parent);if(from!=to)ScaleTree(child,from,to);}
        static FontDesign Design(Font font) { return designs.GetValue(font, delegate(Font f) { return new FontDesign { Family=f.FontFamily.Name, Points=f.SizeInPoints, Style=f.Style }; }); }
        public static Font PaintFont(Font font, Control control, float extraPoints, FontStyle? style = null)
        {
            FontDesign d=Design(font); return new Font(d.Family,(d.Points+extraPoints)*Dpi(control)/72F,style??d.Style,GraphicsUnit.Pixel);
        }
        public static int Dpi(Control control) { var window=control.FindForm() as WizardWindow;return window==null?96:window.LayoutDpi; }
        public static int Pixels(Control control,int value) { return (int)Math.Round(value*Dpi(control)/96.0); }
        internal static void ScaleTree(Control root,int from,int to)
        {
            var controls=new System.Collections.Generic.List<Control>();
            var fonts=new System.Collections.Generic.List<FontDesign>();
            Collect(root,controls,fonts);
            try {
                if(from!=to)root.Scale(new SizeF(to/(float)from,to/(float)from));
                for(int i=0;i<controls.Count;i++){
                    Control c=controls[i];FontDesign d=fonts[i];
                    treeDpis.GetOrCreateValue(c).Value=to;
                    Font next=new Font(d.Family,d.Points*to/72F,d.Style,GraphicsUnit.Pixel);designs.Add(next,d);
                    OwnedFont holder=owned.GetValue(c,delegate(Control key){var value=new OwnedFont();key.Disposed+=delegate{if(value.Value!=null)value.Value.Dispose();};return value;});
                    Font previous=holder.Value;holder.Value=next;c.Font=next;if(previous!=null)previous.Dispose();
                }
            } finally { for(int i=controls.Count-1;i>=0;i--)controls[i].ResumeLayout(true); }
            root.Invalidate(true);
        }
        static void Collect(Control root,System.Collections.Generic.List<Control> controls,System.Collections.Generic.List<FontDesign> fonts)
        {
            root.SuspendLayout();controls.Add(root);fonts.Add(Design(root.Font));
            foreach(Control child in root.Controls)Collect(child,controls,fonts);
        }
    }

    // Fully managed selector surface: no system combo border or arrow can flash white.
    // Editable paths keep an inner native EDIT for IME, paste and text selection.
    public sealed class WizardComboBox : UserControl
    {
        int P(int value){return WizardDpi.Pixels(this,value);}
        readonly System.Collections.Generic.List<object> items=new System.Collections.Generic.List<object>();
        readonly TextBox editor;
        ContextMenuStrip choices;
        ComboBoxStyle style=ComboBoxStyle.DropDown;
        int selected=-1;
        bool hover,syncing;
        public event EventHandler SelectedIndexChanged;
        public event EventHandler SelectionChangeCommitted;
        public System.Collections.Generic.List<object> Items{get{return items;}}
        public DrawMode DrawMode{get;set;}
        public FlatStyle FlatStyle{get;set;}
        public int ItemHeight{get;set;}
        public int MaxDropDownItems{get;set;}
        public int DropDownHeight{get;set;}
        public bool IntegralHeight{get;set;}
        public ComboBoxStyle DropDownStyle{get{return style;}set{style=value;editor.Visible=value!=ComboBoxStyle.DropDownList;Invalidate();}}
        public int SelectedIndex{get{return selected;}set{if(value< -1||value>=items.Count)throw new ArgumentOutOfRangeException("value");if(value==selected)return;selected=value;if(value>=0)Text=GetItemText(items[value]);else if(style==ComboBoxStyle.DropDownList)Text="";if(SelectedIndexChanged!=null)SelectedIndexChanged(this,EventArgs.Empty);Invalidate();}}
        public object SelectedItem{get{return selected<0?null:items[selected];}set{SelectedIndex=items.IndexOf(value);}}
        public override string Text{get{return base.Text;}set{value=value??"";if(base.Text==value)return;base.Text=value;if(editor!=null&&editor.Text!=value){syncing=true;try{editor.Text=value;}finally{syncing=false;}}Invalidate();}}
        public string GetItemText(object value){return value==null?"":Convert.ToString(value);}
        public WizardComboBox(){SetStyle(ControlStyles.UserPaint|ControlStyles.AllPaintingInWmPaint|ControlStyles.OptimizedDoubleBuffer|ControlStyles.ResizeRedraw,true);AutoScaleMode=AutoScaleMode.None;Height=38;MinimumSize=new Size(80,32);TabStop=true;AccessibleRole=AccessibleRole.ComboBox;editor=new WizardTextBox{BorderStyle=BorderStyle.None,AutoSize=false,TabStop=true,BackColor=WizardTheme.Current.Surface,ForeColor=WizardTheme.Current.Text};Controls.Add(editor);editor.TextChanged+=delegate{if(!syncing)Text=editor.Text;};editor.Enter+=delegate{WizardScrolling.Reveal(this);Invalidate();};editor.Leave+=delegate{Invalidate();};}
        protected override void OnLayout(LayoutEventArgs e){base.OnLayout(e);if(editor!=null){int h=Math.Max(WizardDpi.Pixels(this,20),editor.PreferredHeight);editor.SetBounds(WizardDpi.Pixels(this,11),Math.Max(WizardDpi.Pixels(this,4),(Height-h)/2),Math.Max(WizardDpi.Pixels(this,20),Width-WizardDpi.Pixels(this,45)),h);}}
        protected override void OnFontChanged(EventArgs e){base.OnFontChanged(e);if(editor!=null){editor.Font=Font;PerformLayout();}}
        protected override void OnBackColorChanged(EventArgs e){base.OnBackColorChanged(e);if(editor!=null)editor.BackColor=WizardTheme.Current.Surface;}
        protected override void OnForeColorChanged(EventArgs e){base.OnForeColorChanged(e);if(editor!=null)editor.ForeColor=ForeColor;}
        protected override void OnMouseEnter(EventArgs e){hover=true;Invalidate();base.OnMouseEnter(e);}
        protected override void OnMouseLeave(EventArgs e){hover=false;Invalidate();base.OnMouseLeave(e);}
        protected override void OnEnter(EventArgs e){base.OnEnter(e);WizardScrolling.Reveal(this);Invalidate();}
        protected override void OnLeave(EventArgs e){base.OnLeave(e);Invalidate();}
        protected override void OnMouseDown(MouseEventArgs e){base.OnMouseDown(e);if(e.Button==MouseButtons.Left&&(style==ComboBoxStyle.DropDownList||e.X>=Width-WizardDpi.Pixels(this,32))){if(choices==null||!choices.Visible)OpenChoices();else CloseChoices();}}
        protected override bool ProcessCmdKey(ref Message m,Keys key){if(key==Keys.F4||key==(Keys.Alt|Keys.Down)||(key==Keys.Space&&style==ComboBoxStyle.DropDownList)){OpenChoices();return true;}if(style==ComboBoxStyle.DropDownList&&(key==Keys.Down||key==Keys.Up)){if(items.Count>0)SelectedIndex=Math.Max(0,Math.Min(items.Count-1,selected+(key==Keys.Down?1:-1)));return true;}return base.ProcessCmdKey(ref m,key);}
        protected override void OnMouseWheel(MouseEventArgs e){var page=WizardScrolling.Nearest(this);if(page!=null)page.ScrollBy(-e.Delta/120*64);}
        protected override void WndProc(ref Message m){if(m.Msg==0x020a){int delta=unchecked((short)((m.WParam.ToInt64()>>16)&0xffff));var page=WizardScrolling.Nearest(this);if(page!=null)page.ScrollBy(-delta/120*64);m.Result=IntPtr.Zero;return;}base.WndProc(ref m);}
        public void OpenChoices(){if(!Enabled||items.Count==0||!IsHandleCreated)return;CloseChoices();Focus();if(choices==null){choices=new ContextMenuStrip{ShowImageMargin=false,ShowCheckMargin=true,Padding=new Padding(5)};choices.Closed+=delegate{Invalidate();};}while(choices.Items.Count>0){var old=choices.Items[0];choices.Items.RemoveAt(0);old.Dispose();}choices.MaximumSize=new Size(Math.Max(P(220),Width),P(420));choices.Font=Font;for(int i=0;i<items.Count;i++){int index=i;var item=new ToolStripMenuItem(GetItemText(items[i])){Name="Choice_"+i,Checked=i==selected,AutoSize=false,Width=Math.Max(P(180),Width-P(12)),Height=P(36),TextAlign=ContentAlignment.MiddleLeft};item.Click+=delegate{SelectedIndex=index;if(SelectionChangeCommitted!=null)SelectionChangeCommitted(this,EventArgs.Empty);};choices.Items.Add(item);}WizardTheme.Apply(choices);choices.Show(this,new Point(0,Height+P(3)));Invalidate();}
        public void CloseChoices(){if(choices!=null&&!choices.IsDisposed)choices.Close();Invalidate();}
        protected override void OnPaintBackground(PaintEventArgs e){using(var brush=new SolidBrush(Parent==null?WizardTheme.Current.Background:Parent.BackColor))e.Graphics.FillRectangle(brush,ClientRectangle);}
        protected override void OnPaint(PaintEventArgs e){var p=WizardTheme.Current;e.Graphics.SmoothingMode=SmoothingMode.AntiAlias;bool focused=ContainsFocus||(choices!=null&&choices.Visible);using(var path=SoftButton.Round(new Rectangle(P(1),P(1),Math.Max(P(18),Width-P(3)),Math.Max(P(18),Height-P(3))),P(8))){using(var brush=new SolidBrush(hover?p.Hover:p.Surface))e.Graphics.FillPath(brush,path);using(var pen=new Pen(focused?p.Accent:p.Outline))e.Graphics.DrawPath(pen,path);}if(style==ComboBoxStyle.DropDownList)TextRenderer.DrawText(e.Graphics,Text,Font,new Rectangle(P(11),P(1),Math.Max(1,Width-P(45)),Height-P(2)),Enabled?p.Text:p.Muted,TextFormatFlags.Left|TextFormatFlags.VerticalCenter|TextFormatFlags.EndEllipsis);if(items.Count>0){int x=Width-P(18),y=Height/2;using(var pen=new Pen(p.Muted,1.5f*WizardDpi.Dpi(this)/96F)){e.Graphics.DrawLine(pen,x-P(4),y-P(2),x,y+P(2));e.Graphics.DrawLine(pen,x,y+P(2),x+P(4),y-P(2));}}}
        protected override void Dispose(bool disposing){if(disposing&&choices!=null){choices.Dispose();choices=null;}base.Dispose(disposing);}
    }
    public class WizardTextBox : TextBox
    {
        public WizardTextBox(){BorderStyle=BorderStyle.None;}
        protected override CreateParams CreateParams{get{var p=base.CreateParams;p.Style&=~0x00b00000;p.ExStyle&=~0x300;return p;}}
        protected override void WndProc(ref Message m){if(m.Msg==0x020a&&!Multiline){var page=WizardScrolling.Nearest(this);if(page!=null){int delta=unchecked((short)((m.WParam.ToInt64()>>16)&0xffff));page.ScrollBy(-delta/120*64);m.Result=IntPtr.Zero;return;}}base.WndProc(ref m);}
    }
    // A proper padded field, rather than painting over a system EDIT border.
    // The inner native EDIT retains its normal IME, selection and password behavior.
    public sealed class WizardInputFrame : Panel
    {
        int P(int value){return WizardDpi.Pixels(this,value);}
        readonly TextBox input;
        public WizardInputFrame(TextBox editor){input=editor;Name="Frame_"+editor.Name;Height=38;Width=Math.Max(120,editor.Width);DoubleBuffered=true;ResizeRedraw=true;Tag="WizardTheme.Surface";editor.BorderStyle=BorderStyle.None;editor.Dock=DockStyle.None;editor.AutoSize=false;Controls.Add(editor);editor.Enter+=delegate{Invalidate();};editor.Leave+=delegate{Invalidate();};editor.FontChanged+=delegate{LayoutInput();};SizeChanged+=delegate{LayoutInput();};LayoutInput();}
        void LayoutInput(){int h=input.Multiline?Math.Max(WizardDpi.Pixels(this,18),Height-WizardDpi.Pixels(this,16)):Math.Max(WizardDpi.Pixels(this,20),input.PreferredHeight);input.SetBounds(WizardDpi.Pixels(this,11),input.Multiline?WizardDpi.Pixels(this,8):Math.Max(WizardDpi.Pixels(this,4),(Height-h)/2),Math.Max(WizardDpi.Pixels(this,16),Width-WizardDpi.Pixels(this,22)),h);input.Anchor=AnchorStyles.Left|AnchorStyles.Right|AnchorStyles.Top|(input.Multiline?AnchorStyles.Bottom:0);}
        protected override void OnPaint(PaintEventArgs e){var p=WizardTheme.Current;using(var brush=new SolidBrush(Parent==null?p.Background:Parent.BackColor))e.Graphics.FillRectangle(brush,ClientRectangle);e.Graphics.SmoothingMode=SmoothingMode.AntiAlias;using(var path=SoftButton.Round(new Rectangle(P(1),P(1),Math.Max(P(18),Width-P(3)),Math.Max(P(18),Height-P(3))),P(8))){using(var brush=new SolidBrush(p.Surface))e.Graphics.FillPath(brush,path);using(var pen=new Pen(input.Focused?p.Accent:p.Outline))e.Graphics.DrawPath(pen,path);}}
    }

    public sealed class WizardMenuColors : ProfessionalColorTable
    {
        public override Color ToolStripDropDownBackground { get { return WizardTheme.Current.Surface; } }
        public override Color ImageMarginGradientBegin { get { return WizardTheme.Current.Surface; } }
        public override Color ImageMarginGradientMiddle { get { return WizardTheme.Current.Surface; } }
        public override Color ImageMarginGradientEnd { get { return WizardTheme.Current.Surface; } }
        public override Color MenuBorder { get { return WizardTheme.Current.Border; } }
        public override Color MenuItemBorder { get { return WizardTheme.Current.Border; } }
        public override Color MenuItemSelected { get { return WizardTheme.Current.Selected; } }
        public override Color CheckBackground { get { return WizardTheme.Current.Selected; } }
        public override Color CheckSelectedBackground { get { return WizardTheme.Current.Selected; } }
    }

    public sealed class WizardCheckBox : CheckBox
    {
        int P(int value){return WizardDpi.Pixels(this,value);}
        public WizardCheckBox(){SetStyle(ControlStyles.UserPaint|ControlStyles.AllPaintingInWmPaint|ControlStyles.OptimizedDoubleBuffer,true);UseVisualStyleBackColor=false;}
        protected override void OnPaint(PaintEventArgs e){var p=WizardTheme.Current;using(var brush=new SolidBrush(Parent==null?p.Background:Parent.BackColor))e.Graphics.FillRectangle(brush,ClientRectangle);int size=P(14),y=(Height-size)/2;using(var brush=new SolidBrush(Checked?p.Primary:p.Surface))e.Graphics.FillRectangle(brush,P(1),y,size,size);using(var pen=new Pen(Focused?p.Accent:p.Border))e.Graphics.DrawRectangle(pen,P(1),y,size,size);if(Checked){using(var pen=new Pen(p.Dark?p.Text:p.OnAccent,1.8f*WizardDpi.Dpi(this)/96F)){e.Graphics.DrawLine(pen,P(4),y+P(7),P(7),y+P(10));e.Graphics.DrawLine(pen,P(7),y+P(10),P(12),y+P(4));}}TextRenderer.DrawText(e.Graphics,Text,Font,new Rectangle(P(23),0,Math.Max(1,Width-P(23)),Height),Enabled?p.Text:p.Muted,TextFormatFlags.Left|TextFormatFlags.VerticalCenter|TextFormatFlags.EndEllipsis);}
    }

    // Pages scroll a managed canvas. Never create a native WS_VSCROLL/WS_HSCROLL
    // area and then paint over it: Windows can repaint that area independently.
    public class WizardPanel : Panel
    {
        public WizardPanel(){DoubleBuffered=true;ResizeRedraw=true;SetStyle(ControlStyles.AllPaintingInWmPaint|ControlStyles.OptimizedDoubleBuffer,true);}
    }
    public sealed class WizardFlowPanel : FlowLayoutPanel
    {
        public WizardFlowPanel(){DoubleBuffered=true;SetStyle(ControlStyles.AllPaintingInWmPaint|ControlStyles.OptimizedDoubleBuffer,true);}
    }
    public class WizardScrollView : Panel
    {
        readonly Panel viewport;
        readonly Panel canvas;
        readonly WizardScrollTrack track;
        int offset,extent;
        Size minimum;
        bool layoutBusy;
        public WizardScrollView(){DoubleBuffered=true;base.AutoScroll=false;viewport=new WizardPanel{Dock=DockStyle.Fill};canvas=new WizardPanel{Name="ScrollCanvas"};track=new WizardScrollTrack{Dock=DockStyle.Right,Width=14};base.Controls.Add(viewport);base.Controls.Add(track);viewport.Controls.Add(canvas);track.ValueChanged+=delegate{SetOffset(track.Value);};canvas.ControlAdded+=delegate(object s,ControlEventArgs e){WizardScrolling.Attach(e.Control);Reflow();};canvas.ControlRemoved+=delegate{Reflow();};viewport.Resize+=delegate{Reflow();};}
        public new Control.ControlCollection Controls{get{return canvas==null?base.Controls:canvas.Controls;}}
        public new bool AutoScroll{get{return true;}set{base.AutoScroll=false;}}
        public new Point AutoScrollPosition{get{return new Point(0,-offset);}set{SetOffset(Math.Abs(value.Y));}}
        public new Size AutoScrollMinSize{get{return minimum;}set{minimum=value;Reflow();}}
        public int ViewportWidth{get{return viewport==null?ClientSize.Width:viewport.ClientSize.Width;}}
        public int ScrollOffset{get{return offset;}}
        public void ScrollBy(int pixels){Reflow();SetOffset(offset+pixels);}
        public new void ScrollControlIntoView(Control control){if(control==null||!control.Visible)return;Point p=canvas.PointToClient(control.PointToScreen(Point.Empty));if(p.Y<offset+8)SetOffset(p.Y-8);else if(p.Y+control.Height>offset+viewport.Height-8)SetOffset(p.Y+control.Height-viewport.Height+8);}
        void SetOffset(int value){int next=Math.Max(0,Math.Min(Math.Max(0,extent-viewport.Height),value));if(next==offset)return;offset=next;canvas.Top=-offset;track.SetMetrics(extent,viewport.Height,offset);}
        void Reflow(){if(layoutBusy||canvas==null||viewport==null)return;layoutBusy=true;try{int size=minimum.Height;foreach(Control child in canvas.Controls)if(child.Visible||!Visible)size=Math.Max(size,child.Bottom);extent=Math.Max(viewport.Height,size);offset=Math.Max(0,Math.Min(offset,Math.Max(0,extent-viewport.Height)));canvas.SetBounds(0,-offset,Math.Max(1,viewport.Width),Math.Max(1,extent));track.SetMetrics(extent,viewport.Height,offset);viewport.BackColor=BackColor;canvas.BackColor=BackColor;}finally{layoutBusy=false;}}
        protected override void OnLayout(LayoutEventArgs e){base.OnLayout(e);Reflow();}
        protected override void ScaleControl(SizeF factor,BoundsSpecified specified){minimum=new Size((int)Math.Round(minimum.Width*factor.Width),(int)Math.Round(minimum.Height*factor.Height));offset=(int)Math.Round(offset*factor.Height);base.ScaleControl(factor,specified);Reflow();}
        protected override void OnBackColorChanged(EventArgs e){base.OnBackColorChanged(e);if(viewport!=null){viewport.BackColor=BackColor;canvas.BackColor=BackColor;}}
        protected override void OnMouseWheel(MouseEventArgs e){ScrollBy(-e.Delta/120*64);}
    }

    public sealed class WizardScrollTrack : Control
    {
        int P(int value){return WizardDpi.Pixels(this,value);}
        int total,view,value,dragStart,startValue;
        bool dragging,hover;
        public event EventHandler ValueChanged;
        public int Value{get{return value;}}
        public bool Horizontal{get;set;}
        public WizardScrollTrack(){SetStyle(ControlStyles.UserPaint|ControlStyles.AllPaintingInWmPaint|ControlStyles.OptimizedDoubleBuffer|ControlStyles.ResizeRedraw,true);TabStop=false;AccessibleRole=AccessibleRole.ScrollBar;}
        public void SetMetrics(int size,int visible,int position){int next=Math.Max(0,Math.Min(Math.Max(0,size-visible),position));if(total==size&&view==visible&&value==next)return;total=size;view=visible;value=next;Invalidate();}
        Rectangle Thumb(){int length=Math.Max(1,(Horizontal?Width:Height)-P(8));int len=Math.Min(length,Math.Max(P(28),(int)(length*(double)view/Math.Max(1,total))));int pos=P(4)+(int)((length-len)*(double)value/Math.Max(1,total-view));return Horizontal?new Rectangle(pos,Math.Max(P(2),(Height-P(5))/2),len,P(5)):new Rectangle(Math.Max(P(2),(Width-P(5))/2),pos,P(5),len);}
        void MoveTo(int next){next=Math.Max(0,Math.Min(Math.Max(0,total-view),next));if(next==value)return;value=next;Invalidate();if(ValueChanged!=null)ValueChanged(this,EventArgs.Empty);}
        protected override void OnPaint(PaintEventArgs e){var p=WizardTheme.Current;e.Graphics.Clear(Parent is TextBoxBase?p.Surface:p.Background);if(total<=view)return;e.Graphics.SmoothingMode=SmoothingMode.AntiAlias;var r=Thumb();using(var path=SoftButton.Round(r,P(2)))using(var brush=new SolidBrush(hover||dragging?p.Muted:p.Outline))e.Graphics.FillPath(brush,path);}
        protected override void OnMouseEnter(EventArgs e){hover=true;Invalidate();base.OnMouseEnter(e);}
        protected override void OnMouseLeave(EventArgs e){hover=false;Invalidate();base.OnMouseLeave(e);}
        protected override void OnMouseDown(MouseEventArgs e){if(e.Button!=MouseButtons.Left||total<=view)return;var thumb=Thumb();int axis=Horizontal?e.X:e.Y;if(thumb.Contains(e.Location)){dragging=true;dragStart=axis;startValue=value;Capture=true;}else MoveTo(value+(axis<(Horizontal?thumb.Left:thumb.Top)?-view:view));}
        protected override void OnMouseMove(MouseEventArgs e){if(dragging)MoveTo(startValue+(int)(((Horizontal?e.X:e.Y)-dragStart)*(double)(total-view)/Math.Max(1,(Horizontal?Width:Height)-P(8)-(Horizontal?Thumb().Width:Thumb().Height))));}
        protected override void OnMouseUp(MouseEventArgs e){dragging=false;Capture=false;Invalidate();}
        protected override void OnMouseCaptureChanged(EventArgs e){if(!Capture)dragging=false;base.OnMouseCaptureChanged(e);}
    }

    public class WizardRichTextBox : RichTextBox
    {
        readonly WizardScrollTrack vertical,horizontal;
        bool updating,latestPinned,latestQueued;
        long userScrollRevision,latestRevision;
        int longest,rectWidth=-1,rectHeight=-1,wheelRemainder;
        public event EventHandler UserScrolled;
        [StructLayout(LayoutKind.Sequential)] struct TextRect{public int L,T,R,B;}
        [DllImport("user32.dll")] static extern IntPtr SendMessage(IntPtr h,int message,IntPtr w,IntPtr l);
        [DllImport("user32.dll",EntryPoint="SendMessageW")] static extern IntPtr SetRect(IntPtr h,int message,IntPtr w,ref TextRect r);
        public new RichTextBoxScrollBars ScrollBars{get{return RichTextBoxScrollBars.None;}set{base.ScrollBars=RichTextBoxScrollBars.None;}}
        public WizardRichTextBox(){base.ScrollBars=RichTextBoxScrollBars.None;BorderStyle=BorderStyle.None;vertical=new WizardScrollTrack{Width=12};horizontal=new WizardScrollTrack{Horizontal=true,Height=12};Controls.Add(vertical);Controls.Add(horizontal);vertical.ValueChanged+=delegate{if(!IsHandleCreated||updating)return;int first=SendMessage(Handle,0x00ce,IntPtr.Zero,IntPtr.Zero).ToInt32();ScrollLines(vertical.Value-first);};horizontal.ValueChanged+=delegate{if(!IsHandleCreated||updating)return;var point=new TextPoint();ScrollPoint(Handle,0x04dd,IntPtr.Zero,ref point);point.X=horizontal.Value*Math.Max(1,Font.Height/2);ScrollPoint(Handle,0x04de,IntPtr.Zero,ref point);lastHorizontal=horizontal.Value;NotifyUserScroll();};}
                [StructLayout(LayoutKind.Sequential)] struct TextPoint{public int X,Y;}
        [DllImport("user32.dll",EntryPoint="SendMessageW")] static extern IntPtr ScrollPoint(IntPtr h,int m,IntPtr w,ref TextPoint p);
        int lastHorizontal;
        public int FirstVisibleLine{get{return IsHandleCreated?SendMessage(Handle,0x00ce,IntPtr.Zero,IntPtr.Zero).ToInt32():0;}}
        public void ScrollLines(int lines){if(!IsHandleCreated)return;SendMessage(Handle,0x00b6,IntPtr.Zero,new IntPtr(lines));UpdateTracks(false);NotifyUserScroll();}
        void ScrollLatestNow(){UpdateTracks(false);Select(TextLength,0);ScrollToCaret();SendMessage(Handle,0x0115,new IntPtr(7),IntPtr.Zero);UpdateTracks(false);}
        public void ScrollToLatest(){
            latestPinned=true;if(!IsHandleCreated||IsDisposed)return;
            ScrollLatestNow();latestRevision=userScrollRevision;
            if(latestQueued)return;latestQueued=true;
            BeginInvoke((MethodInvoker)delegate{
                latestQueued=false;
                if(IsDisposed||!IsHandleCreated||!latestPinned||latestRevision!=userScrollRevision||SelectionLength>0)return;
                // RichEdit completes wrapping and formatting after Rtf/Text changes.
                // Pin again after layout, rather than one line above the new bottom.
                ScrollLatestNow();
            });
        }
        void NotifyUserScroll(){userScrollRevision++;latestPinned=IsAtLatest;if(UserScrolled!=null)UserScrolled(this,EventArgs.Empty);}
        internal void Wheel(int delta){wheelRemainder+=delta;int ticks=wheelRemainder/120;wheelRemainder%=120;if(ticks!=0)ScrollLines(-ticks*Math.Max(1,SystemInformation.MouseWheelScrollLines<0?Math.Max(1,Height/Math.Max(1,Font.Height)-1):SystemInformation.MouseWheelScrollLines));}
        protected override void WndProc(ref Message m){if(m.Msg==0x020a){Wheel(unchecked((short)((m.WParam.ToInt64()>>16)&0xffff)));m.Result=IntPtr.Zero;return;}base.WndProc(ref m);}
        public bool IsAtLatest{get{if(!IsHandleCreated)return true;int first=SendMessage(Handle,0x00ce,IntPtr.Zero,IntPtr.Zero).ToInt32();int last=GetCharIndexFromPosition(new Point(Math.Max(1,Width-17),Math.Max(1,Height-17)));return last>=TextLength-2||first+Math.Max(1,(Height-16)/Math.Max(1,Font.Height))>GetLineFromCharIndex(TextLength);}}
        public void RefreshScrollIndicators(){UpdateTracks(false);}
        protected override CreateParams CreateParams{get{var p=base.CreateParams;p.Style=(p.Style&~0x00300000)|0x02000000;p.ExStyle&=~0x200;return p;}}
        protected override void OnHandleCreated(EventArgs e){base.OnHandleCreated(e);UpdateTracks(true);if(latestPinned)ScrollToLatest();}
        protected override void OnTextChanged(EventArgs e){base.OnTextChanged(e);UpdateTracks(true);}
        protected override void OnResize(EventArgs e){base.OnResize(e);UpdateTracks(false);if(latestPinned&&SelectionLength==0)ScrollToLatest();}
        protected override void OnSelectionChanged(EventArgs e){base.OnSelectionChanged(e);UpdateTracks(false);}
        protected override void OnMouseWheel(MouseEventArgs e){Wheel(e.Delta);}
        void UpdateTracks(bool measure){if(updating||vertical==null||horizontal==null||!IsHandleCreated||IsDisposed)return;updating=true;try{vertical.SetBounds(Math.Max(0,Width-WizardDpi.Pixels(this,12)),0,WizardDpi.Pixels(this,12),Math.Max(1,Height-WizardDpi.Pixels(this,12)));horizontal.SetBounds(0,Math.Max(0,Height-WizardDpi.Pixels(this,12)),Math.Max(1,Width-WizardDpi.Pixels(this,12)),WizardDpi.Pixels(this,12));var rect=new TextRect{L=WizardDpi.Pixels(this,4),T=WizardDpi.Pixels(this,4),R=Math.Max(5,Width-WizardDpi.Pixels(this,16)),B=Math.Max(5,Height-WizardDpi.Pixels(this,16))};if(rectWidth!=Width||rectHeight!=Height){rectWidth=Width;rectHeight=Height;SetRect(Handle,0x00b3,IntPtr.Zero,ref rect);}int lines=GetLineFromCharIndex(TextLength)+1,visible=Math.Max(1,(Height-WizardDpi.Pixels(this,16))/Math.Max(1,Font.Height));int first=SendMessage(Handle,0x00ce,IntPtr.Zero,IntPtr.Zero).ToInt32();vertical.SetMetrics(lines,visible,first);if(measure){int row=0;longest=0;foreach(char c in Text){if(c=='\n'){longest=Math.Max(longest,row);row=0;}else row+=c=='\t'?4:c>255?2:1;}longest=Math.Max(longest,row);}int columns=Math.Max(1,(Width-WizardDpi.Pixels(this,20))/Math.Max(1,TextRenderer.MeasureText("0000000000",Font,new Size(1000,100),TextFormatFlags.NoPadding).Width/10));horizontal.SetMetrics(WordWrap?columns:longest,columns,lastHorizontal);}finally{updating=false;}}
    }

    internal sealed class WizardWheelRouter : IMessageFilter
    {
        readonly Form owner;
        internal WizardWheelRouter(Form form){owner=form;}
        public bool PreFilterMessage(ref Message m){if(m.Msg!=0x020a)return false;Control target=Control.FromChildHandle(m.HWnd);if(target==null||target.FindForm()!=owner)return false;Control current=target;while(current!=null){var rich=current as WizardRichTextBox;if(rich!=null){rich.Wheel(unchecked((short)((m.WParam.ToInt64()>>16)&0xffff)));return true;}var page=current as WizardScrollView;if(page!=null){int delta=unchecked((short)((m.WParam.ToInt64()>>16)&0xffff));page.ScrollBy(-delta/120*64);return true;}if(current is RichTextBox||(current is TextBox&&((TextBox)current).Multiline))return false;current=current.Parent;}return target is ComboBox;}
    }

    internal static class WizardScrolling
    {
        static readonly ConditionalWeakTable<Control,object> attached=new ConditionalWeakTable<Control,object>();
        internal static void Attach(Control control){attached.GetValue(control,delegate(Control c){c.Enter+=delegate{Reveal(c);};c.SizeChanged+=delegate{var page=Nearest(c);if(page!=null)page.PerformLayout();};c.ControlAdded+=delegate(object s,ControlEventArgs e){Attach(e.Control);};return new object();});foreach(Control child in control.Controls)Attach(child);}
        internal static WizardScrollView Nearest(Control control){for(Control c=control.Parent;c!=null;c=c.Parent)if(c is WizardScrollView)return (WizardScrollView)c;return null;}
        internal static void Reveal(Control control){var page=Nearest(control);if(page!=null)page.ScrollControlIntoView(control);}
    }

    public sealed class WizardPalette
    {
        public bool Dark;
        public Color Background, Surface, SurfaceAlt, Text, Muted, Accent, OnAccent;
        public Color Border, Hover, Selected, Primary, PrimaryHover, GradientStart, GradientEnd;
        public Color Outline { get { return Color.FromArgb((Surface.R*2+Border.R)/3,(Surface.G*2+Border.G)/3,(Surface.B*2+Border.B)/3); } }
    }

    public static class WizardTheme
    {
        private static readonly object Gate = new object();
        private static readonly ConditionalWeakTable<Control, Roles> ControlRoles =
            new ConditionalWeakTable<Control, Roles>();
private static string preference = "system";
        private static WizardPalette current = MakePalette(SystemDark);

        public static WizardPalette Current { get { lock (Gate) return current; } }
        public static event EventHandler Changed;

        // Reads only the per-user application theme; never changes Windows settings.
        public static bool SystemDark
        {
            get
            {
                try
                {
                    using (RegistryKey key = Registry.CurrentUser.OpenSubKey(
                        @"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"))
                    {
                        object value = key == null ? null : key.GetValue("AppsUseLightTheme");
                        return value is int && (int)value == 0;
                    }
                }
                catch (SecurityException) { return false; }
                catch (UnauthorizedAccessException) { return false; }
                catch (System.IO.IOException) { return false; }
            }
        }

        public static void SetPreference(string mode)
        {
            if (mode == null) throw new ArgumentNullException("mode");
            mode = mode.Trim().ToLowerInvariant();
            if (mode != "system" && mode != "light" && mode != "dark")
                throw new ArgumentException("Expected system, light or dark.", "mode");
            bool changed;
            lock (Gate)
            {
                preference = mode;
                changed = UpdatePalette();
            }
            RaiseChanged(changed);
        }

        internal static void RefreshSystemPreference()
        {
            bool changed;
            lock (Gate) changed = UpdatePalette();
            RaiseChanged(changed);
        }

        private static bool UpdatePalette()
        {
            bool dark = preference == "dark" || (preference == "system" && SystemDark);
            if (current.Dark == dark) return false;
            current = MakePalette(dark);
            return true;
        }

        private static void RaiseChanged(bool changed)
        {
            EventHandler handler = Changed;
            if (changed && handler != null) handler(null, EventArgs.Empty);
        }

        private static Color Hex(int rgb) { return Color.FromArgb(255, Color.FromArgb(rgb)); }

        private static WizardPalette MakePalette(bool dark)
        {
            return new WizardPalette {
                Dark = dark,
                Background = Hex(dark ? 0x1e201e : 0xfafbfc),
                Surface = Hex(dark ? 0x252a26 : 0xffffff),
                SurfaceAlt = Hex(dark ? 0x2c322d : 0xf1f3f5),
                Text = Hex(dark ? 0xf3f6f3 : 0x171c21),
                Muted = Hex(dark ? 0xcbd6ce : 0x434e5a),
                Accent = Hex(dark ? 0xb8d2bf : 0x344b5e),
                OnAccent = Hex(dark ? 0x243b2b : 0xffffff),
                Border = Hex(dark ? 0x829487 : 0x7a8692),
                Hover = Hex(dark ? 0x303932 : 0xe9edf1),
                Selected = Hex(dark ? 0x343e36 : 0xdde5ec),
                Primary = Hex(dark ? 0x49644f : 0x344b5e),
                PrimaryHover = Hex(dark ? 0x55725c : 0x435d73),
                GradientStart = Hex(dark ? 0x252a26 : 0xffffff),
                GradientEnd = Hex(dark ? 0x343e36 : 0xdde5ec)
            };
        }

        private sealed class Roles
        {
            internal string Foreground;
            internal string Background;
        }

        private static bool Same(Color a, Color b) { return a.ToArgb() == b.ToArgb(); }

        private static Roles Remember(Control control)
        {
            Color fg = control.ForeColor;
            Color bg = control.BackColor;
            Roles roles = new Roles();
            // Remember semantics before the first recoloring, independently of Tag.
            if (Same(fg, Hex(0x58616b)) || Same(fg, Hex(0xbdc9bf)) ||
                Same(fg, Hex(0x6f7e73)) || Same(fg, Hex(0x6a796f)) ||
                Same(fg, SystemColors.GrayText) || Same(fg, Color.Gray) || Same(fg, Color.DimGray))
                roles.Foreground = "Muted";
            else if (Same(fg, Hex(0x344b5e)) || Same(fg, Hex(0xb8d2bf)) || Same(fg, Hex(0x156c5d)))
                roles.Foreground = "Accent";
            else if (control is Label && !Same(fg, SystemColors.ControlText) &&
                !Same(fg, SystemColors.WindowText) && !Same(fg, Hex(0x24282d)) &&
                !Same(fg, Hex(0xe5ece5)) && !Same(fg, Hex(0x23342b)))
                roles.Foreground = "Custom"; // Preserve warning/error/status label colors.
            else roles.Foreground = "Text";

            if (bg.A == 0) roles.Background = "Transparent";
            else if (Same(bg, Color.White) || Same(bg, Hex(0x252a26))) roles.Background = "Surface";
            else if (Same(bg, Hex(0xf1f3f5)) || Same(bg, Hex(0x2c322d))) roles.Background = "SurfaceAlt";
            else roles.Background = "Background";
            return roles;
        }

        public static void Apply(Control root)
        {
            if (root == null) throw new ArgumentNullException("root");
            if (root.IsDisposed || root.Disposing) return;
            if (root.InvokeRequired)
            {
                try { root.BeginInvoke((MethodInvoker)delegate { Apply(root); }); }
                catch (InvalidOperationException) { /* Handle destroyed during dispatch. */ }
                return;
            }
            // Before handle creation the caller must already be on the owning UI thread.
            ApplyCore(root, Current);
        }

        private static void ApplyCore(Control control, WizardPalette p)
        {
            if (control.IsDisposed || control.Disposing) return;
            Roles roles = ControlRoles.GetValue(control, Remember);
            string tag = control.Tag as string;
            string role = tag != null && tag.StartsWith("WizardTheme.", StringComparison.Ordinal)
                ? tag.Substring("WizardTheme.".Length) : "";
            string foreground = role == "Muted" || role == "Accent" || role == "Text"
                ? role : roles.Foreground;
            if (foreground != "Custom")
                control.ForeColor = foreground == "Muted" ? p.Muted : foreground == "Accent" ? p.Accent : p.Text;

            string background = role == "Background" || role == "Surface" || role == "SurfaceAlt"
                ? role : roles.Background;
            Color fill = background == "Surface" ? p.Surface :
                background == "SurfaceAlt" ? p.SurfaceAlt : p.Background;
            if (control is TextBoxBase || control is ListBox || control is ListView ||
                control is TreeView || control is WizardComboBox || control is UpDownBase)
                fill = p.Surface;
            if (control is Form) fill = p.Background;
            if (background != "Transparent") control.BackColor = fill;


            ToolStrip strip=control as ToolStrip;
            if(strip!=null){strip.Renderer=new ToolStripProfessionalRenderer(new WizardMenuColors());strip.BackColor=p.Surface;foreach(ToolStripItem item in strip.Items){item.BackColor=p.Surface;item.ForeColor=p.Text;}}

            LinkLabel link = control as LinkLabel;
            if (link != null)
            {
                link.LinkColor = p.Accent;
                link.ActiveLinkColor = p.Accent;
                link.VisitedLinkColor = p.Muted;
                link.DisabledLinkColor = p.Muted;
            }
            Button button = control as Button;
            if (button != null)
            {
                Form form = button.FindForm();
                bool primary = role == "Primary" || (form != null && form.AcceptButton == button);
                button.UseVisualStyleBackColor = false;
                button.FlatStyle = FlatStyle.Flat;
                button.FlatAppearance.BorderColor = p.Border;
                button.FlatAppearance.MouseOverBackColor = primary ? p.PrimaryHover : p.Hover;
                button.FlatAppearance.MouseDownBackColor = primary ? p.PrimaryHover : p.Selected;
                button.BackColor = primary ? p.Primary : p.Surface;
                // Agent's dark OnAccent is for the pale accent, not the dark primary fill.
                button.ForeColor = primary ? (p.Dark ? p.Text : p.OnAccent) : p.Text;
            }
            CheckBox check = control as CheckBox;
            if (check != null) check.UseVisualStyleBackColor = false;
            RadioButton radio = control as RadioButton;
            if (radio != null) radio.UseVisualStyleBackColor = false;
            TabPage tab = control as TabPage;
            if (tab != null) tab.UseVisualStyleBackColor = false;
            DataGridView grid = control as DataGridView;
            if (grid != null)
            {
                grid.EnableHeadersVisualStyles = false;
                grid.BackgroundColor = p.Background;
                grid.GridColor = p.Border;
                GridStyle(grid.DefaultCellStyle, p.Surface, p);
                GridStyle(grid.ColumnHeadersDefaultCellStyle, p.SurfaceAlt, p);
                GridStyle(grid.RowHeadersDefaultCellStyle, p.SurfaceAlt, p);
            }
            // Native scrollbars, tab headers and system popup menus remain OS rendered.
            foreach (Control child in control.Controls) ApplyCore(child, p);
            control.Invalidate();
        }

        private static void GridStyle(DataGridViewCellStyle style, Color fill, WizardPalette p)
        {
            style.BackColor = fill;
            style.ForeColor = p.Text;
            style.SelectionBackColor = p.Selected;
            style.SelectionForeColor = p.Text;
        }
    }

    public class WizardWindow : Form
    {
        // The chrome lives for the window's lifetime; rebuilding pages touches only ContentHost.
        private readonly Panel contentHost;
        private readonly WizardCaption caption;
        private bool subscribed;
        private bool initialized;
        private int dpi = 96;
        private int contentDpi = 96, logicalDepth;
        private bool dpiScaling;
        private int resizeHit;
        private Point resizeOrigin;
        private Rectangle resizeBounds;
        private bool windowButtons=true;
        private bool resizable=true;
        private WizardWheelRouter wheelRouter;
        private Icon brandIcon;
        private int brandIconSize;

        public Panel ContentHost { get { return contentHost; } }
        public Control CaptionBar { get { return caption; } }
        public bool ShowWindowButtons { get { return windowButtons; } set { windowButtons=value;if(!value){MinimizeBox=false;MaximizeBox=false;resizable=false;}if(caption!=null)caption.RefreshButtons();PerformLayout(); } }
        internal int ChromeDpi { get { return dpi; } }
        internal int LayoutDpi { get { return logicalDepth>0?96:contentDpi; } }
        protected bool DpiScaling { get { return dpiScaling; } }
        protected int LayoutPixels(int value) { return (int)Math.Round(value*LayoutDpi/96.0); }
        static WizardWindow() { WizardDpi.Initialize(); WizardBrand.Initialize(); }

        public WizardWindow()
        {
            ShowIcon=true;UpdateBrandIcon(96);
            AutoScaleDimensions = new SizeF(96F, 96F);
            // Explicit scaling also works in the inbox PowerShell .NET host,
            // whose WinForms app.config does not opt into dynamic DPI resizing.
            AutoScaleMode = AutoScaleMode.None;
            FormBorderStyle = FormBorderStyle.None;
            SetStyle(ControlStyles.ResizeRedraw | ControlStyles.OptimizedDoubleBuffer, true);
            contentHost = new WizardPanel { Name = "ContentHost", TabIndex = 0 };
            caption = new WizardCaption(this) { Name = "CaptionBar", TabIndex = 1 };
            Controls.Add(contentHost);
            Controls.Add(caption);
            initialized = true;
            RefreshTheme();
        }

        protected override CreateParams CreateParams
        {
            get
            {
                CreateParams cp = base.CreateParams;
                // Keep native resizing, system commands, taskbar and restore-drag behavior.
                cp.Style &= ~(Native.WS_CAPTION | Native.WS_THICKFRAME);
                cp.ExStyle &= ~0x300;
                if (ControlBox) cp.Style |= Native.WS_SYSMENU;
                if (MinimizeBox) cp.Style |= Native.WS_MINIMIZEBOX;
                if (MaximizeBox) cp.Style |= Native.WS_MAXIMIZEBOX;
                return cp;
            }
        }

        protected override void OnHandleCreated(EventArgs e)
        {
            base.OnHandleCreated(e);
            if(wheelRouter==null){wheelRouter=new WizardWheelRouter(this);Application.AddMessageFilter(wheelRouter);}
            dpi = WizardDpi.WindowDpi(this);
            if (!subscribed)
            {
                WizardTheme.Changed += ThemeChanged;
                SystemEvents.UserPreferenceChanged += UserPreferenceChanged;
                subscribed = true;
            }
            WizardTheme.RefreshSystemPreference();
            RefreshTheme();
        }

        private void Unsubscribe()
        {
            if (!subscribed) return;
            WizardTheme.Changed -= ThemeChanged;
            SystemEvents.UserPreferenceChanged -= UserPreferenceChanged;
            subscribed = false;
        }

        protected override void OnHandleDestroyed(EventArgs e)
        {
            if(wheelRouter!=null){Application.RemoveMessageFilter(wheelRouter);wheelRouter=null;}
            Unsubscribe();
            base.OnHandleDestroyed(e);
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) Unsubscribe();
            base.Dispose(disposing);
            if(disposing&&brandIcon!=null){brandIcon.Dispose();brandIcon=null;}
        }

        private void ThemeChanged(object sender, EventArgs e) { Dispatch(RefreshTheme); }

        private void UserPreferenceChanged(object sender, UserPreferenceChangedEventArgs e)
        {
            // SystemEvents runs off the UI thread. No controls/handles touched there.
            Dispatch(delegate {
                WizardTheme.RefreshSystemPreference();
                RefreshTheme();
            });
        }

        private void Dispatch(MethodInvoker action)
        {
            if (IsDisposed || Disposing || !IsHandleCreated) return;
            try
            {
                if (InvokeRequired) BeginInvoke((MethodInvoker)delegate {
                    if (!IsDisposed && !Disposing && IsHandleCreated) action();
                });
                else action();
            }
            catch (InvalidOperationException) { /* Closing or recreating the handle. */ }
        }

        public void RefreshTheme()
        {
            if (!initialized || IsDisposed || Disposing) return;
            if (InvokeRequired) { Dispatch(RefreshTheme); return; }
            WizardTheme.Apply(this);
            caption.RefreshButtons();
            PerformLayout();
            Invalidate(true);
        }

        internal int Pixels(int logical) { return (int)Math.Round(logical * dpi / 96.0); }

        protected override void OnLoad(EventArgs e)
        {
            ApplyWindowDpi(dpi, null);
            base.OnLoad(e);
        }
        private void ApplyWindowDpi(int next, Rectangle? suggested)
        {
            next=Math.Max(96,next);int previous=contentDpi;dpi=next;contentDpi=next;
            UpdateBrandIcon(next);
            dpiScaling=true;SuspendLayout();Point location=Location;
            try { WizardDpi.ScaleTree(this,previous,next);if(suggested.HasValue&&WindowState==FormWindowState.Normal)Bounds=suggested.Value;else Location=location; }
            finally { dpiScaling=false;ResumeLayout(true); }
            OnDpiLayoutChanged(EventArgs.Empty);PerformLayout();Invalidate(true);
        }
        protected virtual void OnDpiLayoutChanged(EventArgs e) { }
        private void UpdateBrandIcon(int dpi){int size=Math.Max(32,(int)Math.Round(32*dpi/96.0));if(size==brandIconSize)return;var next=WizardBrand.CreateIcon(size);var previous=brandIcon;brandIcon=next;brandIconSize=size;Icon=next;if(previous!=null)previous.Dispose();}
        internal IDisposable LogicalLayout(Control root) { return new LogicalScope(this,root); }
        // Existing page builders author 96-DPI lengths. Normalize only while
        // rebuilding, then scale once, including newly created controls/fonts.
        sealed class LogicalScope : IDisposable
        {
            WizardWindow window;readonly Control root;readonly bool outer;
            internal LogicalScope(WizardWindow owner,Control control){window=owner;root=control;outer=owner.logicalDepth++==0;if(outer&&owner.contentDpi!=96){owner.dpiScaling=true;try{WizardDpi.ScaleTree(root,owner.contentDpi,96);}finally{owner.dpiScaling=false;}}}
            public void Dispose(){if(window==null)return;WizardWindow owner=window;window=null;owner.logicalDepth--;try{if(outer&&owner.contentDpi!=96){owner.dpiScaling=true;WizardDpi.ScaleTree(root,96,owner.contentDpi);}}finally{owner.dpiScaling=false;if(outer)owner.OnDpiLayoutChanged(EventArgs.Empty);}}
        }

        protected override void OnShown(EventArgs e)
        {
            base.OnShown(e);
            if(WindowState!=FormWindowState.Normal||StartPosition==FormStartPosition.Manual)return;
            Rectangle area=Screen.FromControl(this).WorkingArea;
            int margin=Pixels(12),width=Math.Max(1,area.Width-margin*2),height=Math.Max(1,area.Height-margin*2);
            MinimumSize=new Size(Math.Min(MinimumSize.Width,width),Math.Min(MinimumSize.Height,height));
            Size=new Size(Math.Min(Width,width),Math.Min(Height,height));
            Location=new Point(Math.Max(area.Left+margin,Math.Min(Left,area.Right-margin-Width)),
                Math.Max(area.Top+margin,Math.Min(Top,area.Bottom-margin-Height)));
        }

        protected override void OnLayout(LayoutEventArgs e)
        {
            base.OnLayout(e);
            if (!initialized) return;
            // Keep the complete resize hit band outside child HWNDs. Previously
            // a one-pixel border exposed only a fraction of the six-pixel grip.
            int edge = WindowState == FormWindowState.Maximized ? 0 : Pixels(resizable?6:1);
            int width = Math.Max(0, ClientSize.Width - edge * 2);
            int height = Math.Min(Pixels(42), Math.Max(0, ClientSize.Height - edge * 2));
            caption.SetBounds(edge, edge, width, height);
            contentHost.SetBounds(edge, edge + height, width,
                Math.Max(0, ClientSize.Height - height - edge * 2));
            caption.RefreshButtons();
        }

        protected override void OnTextChanged(EventArgs e)
        {
            base.OnTextChanged(e);
            if (caption != null) { caption.AccessibleName = Text; caption.Invalidate(); }
        }

        protected override void OnStyleChanged(EventArgs e)
        {
            base.OnStyleChanged(e);
            if (initialized) { caption.RefreshButtons(); PerformLayout(); }
        }

        protected override void OnResize(EventArgs e)
        {
            base.OnResize(e);
            if (initialized) { caption.RefreshButtons(); Invalidate(); }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            if (WindowState != FormWindowState.Maximized && Width > 1 && Height > 1)
                using (Pen pen = new Pen(WizardTheme.Current.Outline))
                    e.Graphics.DrawRectangle(pen, 0, 0, ClientSize.Width - 1, ClientSize.Height - 1);
        }

        internal void CaptionCommand(int command)
        {
            if (!ControlBox) return;
            if (command == Native.SC_MINIMIZE && !MinimizeBox) return;
            if ((command == Native.SC_MAXIMIZE || command == Native.SC_RESTORE) && !MaximizeBox) return;
            // Borderless Forms have no standard non-client caption to dispatch to.
            // Use their managed lifecycle so close validation and taskbar restore work.
            if(command==Native.SC_MINIMIZE)WindowState=FormWindowState.Minimized;
            else if(command==Native.SC_MAXIMIZE)WindowState=FormWindowState.Maximized;
            else if(command==Native.SC_RESTORE)WindowState=FormWindowState.Normal;
            else if(command==Native.SC_CLOSE)Close();
        }

        internal void ToggleMaximize()
        {
            CaptionCommand(WindowState == FormWindowState.Maximized ? Native.SC_RESTORE : Native.SC_MAXIMIZE);
        }

        internal void DragCaption()
        {
            Native.ReleaseCapture();
            Native.SendMessage(Handle, Native.WM_NCLBUTTONDOWN, new IntPtr(Native.HTCAPTION), IntPtr.Zero);
        }

        internal void ShowSystemMenu(Point screen)
        {
            if (!ControlBox) return;
            IntPtr menu = Native.GetSystemMenu(Handle, false);
            if (menu == IntPtr.Zero) return;
            bool normal = WindowState == FormWindowState.Normal;
            Native.EnableMenuItem(menu, Native.SC_RESTORE, normal ? 1U : 0U);
            Native.EnableMenuItem(menu, Native.SC_MOVE, normal ? 0U : 1U);
            Native.EnableMenuItem(menu, Native.SC_SIZE, normal ? 0U : 1U);
            Native.EnableMenuItem(menu, Native.SC_MINIMIZE, MinimizeBox ? 0U : 1U);
            Native.EnableMenuItem(menu, Native.SC_MAXIMIZE, MaximizeBox && normal ? 0U : 1U);
            int command = Native.TrackPopupMenuEx(menu, 0x0100 | 0x0002, screen.X, screen.Y, Handle, IntPtr.Zero);
            if (command != 0) Native.SendMessage(Handle, Native.WM_SYSCOMMAND, new IntPtr(command), IntPtr.Zero);
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (keyData == (Keys.Alt | Keys.Space))
            {
                ShowSystemMenu(PointToScreen(new Point(Pixels(8), Pixels(42))));
                return true;
            }
            if (keyData == (Keys.Alt | Keys.F4) && ControlBox)
            {
                CaptionCommand(Native.SC_CLOSE);
                return true;
            }
            // Standard Form processing retains Tab, Shift+Tab, Accept/CancelButton.
            return base.ProcessCmdKey(ref msg, keyData);
        }

        protected override void WndProc(ref Message m)
        {
            if(m.Msg==Native.WM_SETCURSOR){int hit=resizeHit!=0?resizeHit:unchecked((short)(m.LParam.ToInt64()&0xffff));Cursor cursor=ResizeCursor(hit);if(cursor!=null&&resizable&&WindowState==FormWindowState.Normal){Native.SetCursor(cursor.Handle);m.Result=new IntPtr(1);return;}}
            if(m.Msg==0x00a1&&resizable&&m.WParam.ToInt32()>=10&&m.WParam.ToInt32()<=17&&WindowState==FormWindowState.Normal){resizeHit=m.WParam.ToInt32();long down=m.LParam.ToInt64();resizeOrigin=new Point(unchecked((short)(down&0xffff)),unchecked((short)((down>>16)&0xffff)));resizeBounds=Bounds;Capture=true;m.Result=IntPtr.Zero;return;}
            if(m.Msg==0x0200&&resizeHit!=0){long packedMove=m.LParam.ToInt64();Point move=PointToScreen(new Point(unchecked((short)(packedMove&0xffff)),unchecked((short)((packedMove>>16)&0xffff))));Point delta=new Point(move.X-resizeOrigin.X,move.Y-resizeOrigin.Y);Rectangle r=resizeBounds;bool left=resizeHit==10||resizeHit==13||resizeHit==16,right=resizeHit==11||resizeHit==14||resizeHit==17,top=resizeHit==12||resizeHit==13||resizeHit==14,bottom=resizeHit==15||resizeHit==16||resizeHit==17;int minW=Math.Max(100,MinimumSize.Width),minH=Math.Max(100,MinimumSize.Height);if(left){r.X=Math.Min(resizeBounds.Right-minW,resizeBounds.Left+delta.X);r.Width=resizeBounds.Right-r.X;}if(right)r.Width=Math.Max(minW,resizeBounds.Width+delta.X);if(top){r.Y=Math.Min(resizeBounds.Bottom-minH,resizeBounds.Top+delta.Y);r.Height=resizeBounds.Bottom-r.Y;}if(bottom)r.Height=Math.Max(minH,resizeBounds.Height+delta.Y);Bounds=r;m.Result=IntPtr.Zero;return;}
            if(m.Msg==0x0202&&resizeHit!=0){resizeHit=0;Capture=false;m.Result=IntPtr.Zero;return;}
            if(m.Msg==0x0215)resizeHit=0;
            if (m.Msg == Native.WM_NCCALCSIZE)
            {
                m.Result = IntPtr.Zero; // All native frame space is painted by this window.
                return;
            }
            if (m.Msg == Native.WM_GETMINMAXINFO)
            {
                base.WndProc(ref m);
                Native.MINMAXINFO info = (Native.MINMAXINFO)Marshal.PtrToStructure(m.LParam, typeof(Native.MINMAXINFO));
                Native.MONITORINFO monitor = new Native.MONITORINFO();
                monitor.Size = Marshal.SizeOf(typeof(Native.MONITORINFO));
                if (Native.GetMonitorInfo(Native.MonitorFromWindow(Handle, 2), ref monitor))
                {
                    info.MaxPosition.X = monitor.Work.Left - monitor.Monitor.Left;
                    info.MaxPosition.Y = monitor.Work.Top - monitor.Monitor.Top;
                    info.MaxSize.X = monitor.Work.Right - monitor.Work.Left;
                    info.MaxSize.Y = monitor.Work.Bottom - monitor.Work.Top;
                }
                if (!MinimumSize.IsEmpty)
                {
                    info.MinTrackSize.X = Math.Max(info.MinTrackSize.X, MinimumSize.Width);
                    info.MinTrackSize.Y = Math.Max(info.MinTrackSize.Y, MinimumSize.Height);
                }
                if (MaximumSize.Width > 0) info.MaxTrackSize.X = MaximumSize.Width;
                if (MaximumSize.Height > 0) info.MaxTrackSize.Y = MaximumSize.Height;
                Marshal.StructureToPtr(info, m.LParam, false);
                m.Result = IntPtr.Zero;
                return;
            }
            if (m.Msg == Native.WM_NCHITTEST)
            {
                base.WndProc(ref m);
                if (!initialized || !resizable || WindowState != FormWindowState.Normal) return;
                long packed = m.LParam.ToInt64();
                Point pt = PointToClient(new Point(unchecked((short)(packed & 0xffff)),
                    unchecked((short)((packed >> 16) & 0xffff))));
                int grip = Pixels(6);
                bool left = pt.X >= 0 && pt.X < grip;
                bool right = pt.X < ClientSize.Width && pt.X >= ClientSize.Width - grip;
                bool top = pt.Y >= 0 && pt.Y < grip;
                bool bottom = pt.Y < ClientSize.Height && pt.Y >= ClientSize.Height - grip;
                // Give caption buttons their full hit area, including the top-right corner.
                if (caption.ButtonAt(PointToScreen(pt))) { m.Result = new IntPtr(Native.HTCLIENT); return; }
                int hit = top && left ? 13 : top && right ? 14 : bottom && left ? 16 : bottom && right ? 17 :
                    left ? 10 : right ? 11 : top ? 12 : bottom ? 15 : Native.HTCLIENT;
                m.Result = new IntPtr(hit);
                return;
            }
            if (m.Msg == Native.WM_NCLBUTTONDBLCLK && m.WParam.ToInt32() == Native.HTCAPTION)
            {
                ToggleMaximize();
                m.Result = IntPtr.Zero;
                return;
            }
            if (m.Msg == Native.WM_DPICHANGED)
            {
                Native.RECT rect = (Native.RECT)Marshal.PtrToStructure(m.LParam, typeof(Native.RECT));
                ApplyWindowDpi((int)(m.WParam.ToInt64() & 0xffff),Rectangle.FromLTRB(rect.Left,rect.Top,rect.Right,rect.Bottom));
                m.Result=IntPtr.Zero;
                return;
            }
            base.WndProc(ref m);
        }
        static Cursor ResizeCursor(int hit){switch(hit){case 10:case 11:return Cursors.SizeWE;case 12:case 15:return Cursors.SizeNS;case 13:case 17:return Cursors.SizeNWSE;case 14:case 16:return Cursors.SizeNESW;default:return null;}}
    }

    internal sealed class WizardCaption : Control
    {
        private readonly WizardWindow window;
        private readonly CaptionButton minimize, maximize, close;
        private string lastState="";

        internal WizardCaption(WizardWindow owner)
        {
            window = owner;
            SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint |
                ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw |
                ControlStyles.StandardDoubleClick | ControlStyles.ContainerControl, true);
            TabStop = false;
            AccessibleRole = AccessibleRole.TitleBar;
            minimize = new CaptionButton(window, 0) { Name = "CaptionMinimize", TabIndex = 0 };
            maximize = new CaptionButton(window, 1) { Name = "CaptionMaximize", TabIndex = 1 };
            close = new CaptionButton(window, 2) { Name = "CaptionClose", TabIndex = 2 };
            Controls.Add(minimize);
            Controls.Add(maximize);
            Controls.Add(close);
        }

        internal void RefreshButtons()
        {
            if (minimize == null) return;
            minimize.Visible = window.ShowWindowButtons && window.ControlBox && window.MinimizeBox;
            maximize.Visible = window.ShowWindowButtons && window.ControlBox && window.MaximizeBox;
            close.Visible = window.ShowWindowButtons && window.ControlBox;
            minimize.AccessibleName = "Minimize";
            maximize.AccessibleName = window.WindowState == FormWindowState.Maximized ? "Restore" : "Maximize";
            close.AccessibleName = "Close";
            minimize.AccessibleDescription = "Minimize window";
            maximize.AccessibleDescription = maximize.AccessibleName + " window";
            close.AccessibleDescription = "Close window (Alt+F4)";
            LayoutButtons();
            string state=window.WindowState+"/"+window.ShowWindowButtons+"/"+window.ControlBox+"/"+window.MinimizeBox+"/"+window.MaximizeBox;
            if(state!=lastState){lastState=state;Invalidate(true);}
        }

        private void LayoutButtons()
        {
            int x = Width;
            int width = window.Pixels(46);
            CaptionButton[] buttons = { close, maximize, minimize };
            foreach (CaptionButton button in buttons)
            {
                // Visible includes the parent's visibility, so do not use it while
                // laying out a not-yet-shown form (all buttons would overlap at 0,0).
                if (!window.ShowWindowButtons || !window.ControlBox || (button == minimize && !window.MinimizeBox) ||
                    (button == maximize && !window.MaximizeBox)) continue;
                x -= width;
                button.SetBounds(x, 0, width, Height);
            }
        }

        internal bool ButtonAt(Point screen)
        {
            foreach (Control child in Controls)
                if (child.Visible && child.RectangleToScreen(child.ClientRectangle).Contains(screen)) return true;
            return false;
        }

        protected override void OnLayout(LayoutEventArgs e) { base.OnLayout(e); LayoutButtons(); }

        protected override void OnPaintBackground(PaintEventArgs e)
        {
            if (Width <= 0 || Height <= 0) return;
            WizardPalette p = WizardTheme.Current;
            using (LinearGradientBrush brush = new LinearGradientBrush(ClientRectangle,
                p.GradientStart, p.GradientEnd, LinearGradientMode.Horizontal))
                e.Graphics.FillRectangle(brush, ClientRectangle);
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            int right = Width;
            foreach (Control button in Controls) if (button.Visible) right = Math.Min(right, button.Left);
            int left=window.Pixels(16);
            if(window.ShowIcon&&window.Icon!=null){int size=window.Pixels(20);e.Graphics.DrawIcon(window.Icon,new Rectangle(left,(Height-size)/2,size,size));left+=size+window.Pixels(10);}
            Rectangle title = new Rectangle(left, 0, Math.Max(0, right - left - window.Pixels(12)), Height);
            TextRenderer.DrawText(e.Graphics, window.Text, Font, title, WizardTheme.Current.Text,
                TextFormatFlags.Left | TextFormatFlags.VerticalCenter | TextFormatFlags.EndEllipsis |
                TextFormatFlags.SingleLine | TextFormatFlags.NoPrefix);
            using (Pen pen = new Pen(WizardTheme.Current.Border))
                e.Graphics.DrawLine(pen, 0, Height - 1, Width, Height - 1);
        }

        protected override void OnMouseDown(MouseEventArgs e)
        {
            base.OnMouseDown(e);
            if (e.Button == MouseButtons.Left && e.Clicks == 1) window.DragCaption();
        }

        protected override void OnMouseDoubleClick(MouseEventArgs e)
        {
            base.OnMouseDoubleClick(e);
            if (e.Button == MouseButtons.Left) window.ToggleMaximize();
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            base.OnMouseUp(e);
            if (e.Button == MouseButtons.Right) window.ShowSystemMenu(PointToScreen(e.Location));
        }
    }

    // A real Button retains MSAA default actions, focus, Tab navigation and Space/Enter.
    // Only the visual is custom; no dependency on SoftButton or TaskTile.
    internal sealed class CaptionButton : Control, IButtonControl
    {
        private readonly WizardWindow window;
        private readonly int kind;
        private bool hover, pressed;

        internal CaptionButton(WizardWindow owner, int buttonKind)
        {
            window = owner;
            kind = buttonKind;
            SetStyle(ControlStyles.UserPaint | ControlStyles.AllPaintingInWmPaint |
                ControlStyles.OptimizedDoubleBuffer, true);
            BackColor = WizardTheme.Current.GradientEnd;
            SetStyle(ControlStyles.StandardClick|ControlStyles.StandardDoubleClick,false);
            AccessibleRole = AccessibleRole.PushButton;
            TabStop = true;
        }

        public DialogResult DialogResult { get;set; }
        public void NotifyDefault(bool value) { }
        public void PerformClick(){if(Enabled&&Visible)OnClick(EventArgs.Empty);}
        protected override AccessibleObject CreateAccessibilityInstance(){return new CaptionAccessibility(this);}
        sealed class CaptionAccessibility:ControlAccessibleObject{
            readonly CaptionButton button;
            internal CaptionAccessibility(CaptionButton owner):base(owner){button=owner;}
            public override AccessibleRole Role{get{return AccessibleRole.PushButton;}}
            public override string DefaultAction{get{return "Press";}}
            public override void DoDefaultAction(){button.PerformClick();}
        }

        protected override void OnClick(EventArgs e)
        {
            base.OnClick(e);
            if (IsDisposed || window.IsDisposed) return;
            if (kind == 0) window.CaptionCommand(Native.SC_MINIMIZE);
            else if (kind == 1) window.ToggleMaximize();
            else window.CaptionCommand(Native.SC_CLOSE);
        }

        protected override void OnMouseEnter(EventArgs e) { hover = true; base.OnMouseEnter(e); Invalidate(); }
        protected override void OnMouseLeave(EventArgs e) { hover = false; base.OnMouseLeave(e); Invalidate(); }
        protected override void OnMouseDown(MouseEventArgs e)
        {
            if (e.Button == MouseButtons.Left) { pressed = true;Capture=true;Focus(); }
            base.OnMouseDown(e); Invalidate();
        }
        protected override void OnMouseUp(MouseEventArgs e) { bool click=pressed&&e.Button==MouseButtons.Left&&ClientRectangle.Contains(e.Location);pressed=false;Capture=false;base.OnMouseUp(e);Invalidate();if(click)PerformClick(); }
        protected override void OnMouseCaptureChanged(EventArgs e)
        {
            if (!Capture) pressed = false;
            base.OnMouseCaptureChanged(e); Invalidate();
        }
        protected override void OnKeyDown(KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Space) pressed = true;
            if(e.KeyCode==Keys.Enter){PerformClick();e.Handled=true;}
            base.OnKeyDown(e); Invalidate();
        }
        protected override void OnKeyUp(KeyEventArgs e) { bool click=pressed&&e.KeyCode==Keys.Space;pressed=false;base.OnKeyUp(e);Invalidate();if(click)PerformClick(); }
        protected override void OnGotFocus(EventArgs e) { base.OnGotFocus(e); Invalidate(); }
        protected override void OnLostFocus(EventArgs e) { pressed = false; base.OnLostFocus(e); Invalidate(); }

        protected override void OnPaintBackground(PaintEventArgs e)
        {
            var p=WizardTheme.Current;
            var full=new Rectangle(-Left,0,Math.Max(1,Parent==null?Width:Parent.Width),Math.Max(1,Height));
            using(var brush=new LinearGradientBrush(full,p.GradientStart,p.GradientEnd,LinearGradientMode.Horizontal))e.Graphics.FillRectangle(brush,ClientRectangle);
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            OnPaintBackground(e);
            WizardPalette p = WizardTheme.Current;
            Color ink = Enabled ? p.Text : p.Muted;
            if (Enabled && (hover || pressed))
            {
                Color fill = pressed ? p.Selected : p.Hover;
                if (kind == 2) { fill = pressed ? Color.FromArgb(183, 31, 43) : Color.FromArgb(196, 43, 55); ink = Color.White; }
                using (SolidBrush brush = new SolidBrush(fill)) e.Graphics.FillRectangle(brush, ClientRectangle);
            }
            float scale = window.ChromeDpi / 96F;
            float size = 10F * scale;
            float x = (Width - size) / 2F;
            float y = (Height - size) / 2F;
            e.Graphics.SmoothingMode = SmoothingMode.None;
            using (Pen pen = new Pen(ink, Math.Max(1F, scale)))
            {
                if (kind == 0) e.Graphics.DrawLine(pen, x, y + size / 2F, x + size, y + size / 2F);
                else if (kind == 2)
                {
                    e.Graphics.DrawLine(pen, x, y, x + size, y + size);
                    e.Graphics.DrawLine(pen, x + size, y, x, y + size);
                }
                else if (window.WindowState == FormWindowState.Maximized)
                {
                    float offset = 2F * scale;
                    e.Graphics.DrawLines(pen, new PointF[] {
                        new PointF(x + offset, y + offset), new PointF(x + offset, y),
                        new PointF(x + size, y), new PointF(x + size, y + size - offset),
                        new PointF(x + size - offset, y + size - offset) });
                    e.Graphics.DrawRectangle(pen, x, y + offset, size - offset, size - offset);
                }
                else e.Graphics.DrawRectangle(pen, x, y, size, size);
            }
            if (Focused && ShowFocusCues)
            {
                Rectangle focus = ClientRectangle;
                focus.Inflate(-window.Pixels(5), -window.Pixels(5));
                if (focus.Width > 0 && focus.Height > 0)
                    ControlPaint.DrawFocusRectangle(e.Graphics, focus, ink, p.SurfaceAlt);
            }
        }
    }

    internal static class Native
    {
        internal const int WS_CAPTION = 0x00c00000, WS_THICKFRAME = 0x00040000,
            WS_SYSMENU = 0x00080000, WS_MINIMIZEBOX = 0x00020000, WS_MAXIMIZEBOX = 0x00010000;
        internal const int WM_NCCALCSIZE = 0x0083, WM_NCHITTEST = 0x0084, WM_GETMINMAXINFO = 0x0024,
            WM_NCLBUTTONDOWN = 0x00a1, WM_NCLBUTTONDBLCLK = 0x00a3, WM_SYSCOMMAND = 0x0112,
            WM_DPICHANGED = 0x02e0, WM_SETCURSOR = 0x0020;
        internal const int HTCLIENT = 1, HTCAPTION = 2;
        internal const int SC_SIZE = 0xf000, SC_MOVE = 0xf010, SC_MINIMIZE = 0xf020,
            SC_MAXIMIZE = 0xf030, SC_CLOSE = 0xf060, SC_RESTORE = 0xf120;
        [DllImport("user32.dll")] internal static extern IntPtr SetCursor(IntPtr cursor);

        [StructLayout(LayoutKind.Sequential)]
        internal struct POINT { internal int X, Y; }
        [StructLayout(LayoutKind.Sequential)]
        internal struct RECT { internal int Left, Top, Right, Bottom; }
        [StructLayout(LayoutKind.Sequential)]
        internal struct MINMAXINFO
        {
            internal POINT Reserved, MaxSize, MaxPosition, MinTrackSize, MaxTrackSize;
        }
        [StructLayout(LayoutKind.Sequential)]
        internal struct MONITORINFO
        {
            internal int Size;
            internal RECT Monitor, Work;
            internal int Flags;
        }

        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool ReleaseCapture();
        [DllImport("user32.dll", CharSet = CharSet.Auto)]
        internal static extern IntPtr SendMessage(IntPtr hwnd, int message, IntPtr wParam, IntPtr lParam);
        [DllImport("user32.dll")]
        internal static extern IntPtr MonitorFromWindow(IntPtr hwnd, int flags);
        [DllImport("user32.dll", CharSet = CharSet.Auto)]
        [return: MarshalAs(UnmanagedType.Bool)]
        internal static extern bool GetMonitorInfo(IntPtr monitor, ref MONITORINFO info);
        [DllImport("user32.dll")]
        internal static extern IntPtr GetSystemMenu(IntPtr hwnd, [MarshalAs(UnmanagedType.Bool)] bool revert);
        [DllImport("user32.dll")]
        internal static extern int TrackPopupMenuEx(IntPtr menu, int flags, int x, int y, IntPtr hwnd, IntPtr parameters);
        [DllImport("user32.dll")]
        internal static extern uint EnableMenuItem(IntPtr menu, int item, uint enable);
    }
}
