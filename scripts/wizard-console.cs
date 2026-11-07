// MoE4All native terminal dashboard. .NET Framework 4.x / C# 5, Windows inbox APIs only.
// No browser, web server, npm, font downloads or inference logic.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Web.Script.Serialization;
using System.Windows.Forms;

namespace MoE4All.Launcher {
    public sealed class Cell {
        public string Text = " "; public string Fg = "cbd5e1"; public string Bg = "0b121b";
        public bool Tail;
    }
    public sealed class Canvas {
        public readonly int Width, Height; public readonly Cell[,] Cells;
        public Canvas(int width, int height) {
            Width = Math.Max(1, width); Height = Math.Max(1, height); Cells = new Cell[Height, Width];
            for (int y = 0; y < Height; y++) for (int x = 0; x < Width; x++) Cells[y,x] = new Cell();
        }
        public static int ElementWidth(string element) {
            if (String.IsNullOrEmpty(element)) return 0;
            UnicodeCategory cat = CharUnicodeInfo.GetUnicodeCategory(element, 0);
            if (cat == UnicodeCategory.NonSpacingMark || cat == UnicodeCategory.EnclosingMark || cat == UnicodeCategory.Format) return 0;
            if (Char.IsSurrogate(element[0]) && (element.Length < 2 || !Char.IsSurrogatePair(element,0))) return 1;
            int c = Char.ConvertToUtf32(element, 0);
            return (c >= 0x1100 && (c <= 0x115f || c == 0x2329 || c == 0x232a || (c >= 0x2e80 && c <= 0xa4cf) ||
                (c >= 0xac00 && c <= 0xd7a3) || (c >= 0xf900 && c <= 0xfaff) || (c >= 0xfe10 && c <= 0xfe6f) ||
                (c >= 0xff01 && c <= 0xff60) || (c >= 0xffe0 && c <= 0xffe6) || c >= 0x1f300)) ? 2 : 1;
        }
        public static string Safe(string text) {
            var b = new StringBuilder(); foreach (char c in text ?? "") b.Append(Char.IsControl(c) ? ' ' : c); return b.ToString();
        }
        public static string Clip(string text, int columns) {
            if (columns <= 0) return "";
            text = Safe(text); var e = StringInfo.GetTextElementEnumerator(text); var b = new StringBuilder(); int used = 0;
            while (e.MoveNext()) { string v = e.GetTextElement(); int n = ElementWidth(v); if (used + n > columns) break; b.Append(v); used += n; }
            return b.ToString();
        }
        public static int Columns(string text) { int n=0;var e=StringInfo.GetTextElementEnumerator(Safe(text));while(e.MoveNext())n+=ElementWidth(e.GetTextElement());return n; }
        public void Fill(int x, int y, int width, int height, string bg) {
            for (int row = Math.Max(0,y); row < Math.Min(Height,y+height); row++) for (int col = Math.Max(0,x); col < Math.Min(Width,x+width); col++) {
                Cells[row,col].Text = " "; Cells[row,col].Tail = false; Cells[row,col].Bg = bg;
            }
        }
        public void Put(int x, int y, string text, string fg, string bg, int max) {
            if (y < 0 || y >= Height) return;
            int end = Math.Min(Width, x + Math.Max(0,max));
            var e = StringInfo.GetTextElementEnumerator(Safe(text));
            while (e.MoveNext()) {
                string v = e.GetTextElement(); int n = ElementWidth(v);
                if (n == 0) continue; if (x < 0 || x+n > end) break;
                Cells[y,x].Text = v; Cells[y,x].Fg = fg; Cells[y,x].Bg = bg; Cells[y,x].Tail = false;
                if (n == 2) { Cells[y,x+1].Text = ""; Cells[y,x+1].Fg = fg; Cells[y,x+1].Bg = bg; Cells[y,x+1].Tail = true; }
                x += n;
            }
        }
        public void Text(int x, int y, string text, string fg, int max) { Put(x,y,text,fg,"0b121b",max); }
        public string Plain() {
            var b = new StringBuilder(); for (int y=0;y<Height;y++) { for (int x=0;x<Width;x++) if (!Cells[y,x].Tail) b.Append(Cells[y,x].Text); b.Append('\n'); } return b.ToString();
        }
        static string Color(string h) { return Convert.ToInt32(h.Substring(0,2),16)+";"+Convert.ToInt32(h.Substring(2,2),16)+";"+Convert.ToInt32(h.Substring(4,2),16); }
        public string Ansi(bool color) {
            var b = new StringBuilder("\x1b[H"); string fg = "", bg = "";
            for (int y=0;y<Height;y++) {
                b.Append("\x1b[").Append(y+1).Append(";1H");
                for (int x=0;x<Width;x++) { Cell c = Cells[y,x]; if (c.Tail) continue;
                    if (color && (c.Fg != fg || c.Bg != bg)) { b.Append("\x1b[38;2;").Append(Color(c.Fg)).Append(";48;2;").Append(Color(c.Bg)).Append('m'); fg=c.Fg;bg=c.Bg; }
                    b.Append(c.Text);
                }
            }
            b.Append("\x1b[0m"); return b.ToString();
        }
    }
    public sealed class Dashboard {
        static readonly JavaScriptSerializer Json = new JavaScriptSerializer { MaxJsonLength = 1024 * 1024 };
        [DllImport("kernel32.dll")] static extern IntPtr GetStdHandle(int id);
        [DllImport("kernel32.dll")] static extern bool GetConsoleMode(IntPtr handle, out uint mode);
        [DllImport("kernel32.dll")] static extern bool SetConsoleMode(IntPtr handle, uint mode);
        readonly Dictionary<string,object> state, schema;
        readonly List<Dictionary<string,object>> pages, fields, groups;
        readonly HashSet<string> expanded = new HashSet<string>(StringComparer.Ordinal);
        int page, selected, offset, navSelected; bool nav, dirty, demo; string notice = "", version, query = "";
        int width = 128, height = 40;
        static Dictionary<string,object> Dict(object o) { return o as Dictionary<string,object> ?? new Dictionary<string,object>(); }
        static List<Dictionary<string,object>> Items(object o) {
            var result = new List<Dictionary<string,object>>(); var list = o as IEnumerable;
            if (list != null && !(o is string)) foreach (object item in list) if (item is Dictionary<string,object>) result.Add((Dictionary<string,object>)item);
            return result;
        }
        static string S(object o) { if (o is bool) return (bool)o ? "true" : "false"; return o == null ? "" : Convert.ToString(o, CultureInfo.InvariantCulture); }
        static string Get(Dictionary<string,object> d, string key, string fallback) { return d.ContainsKey(key) ? S(d[key]) : fallback; }
        static string Primary(string label) { int split=label.IndexOf(" / ",StringComparison.Ordinal);return split<0?label:label.Substring(0,split); }
        string Value(string key) { return Get(state,key,""); }
        bool Flag(string key) { return Value(key).Equals("true", StringComparison.OrdinalIgnoreCase); }
        bool Llm { get { return Value("launch_mode") != "server" || Flag("server_llm_enabled"); } }
        string Derived(string key) {
            if (key=="_llm") return S(Llm);
            if (key=="_api") return S(Value("launch_mode")=="server");
            if (key=="_bench") return S(Value("launch_mode")=="benchmark");
            if (key=="_manual") return S(Value("setup_mode")=="manual");
            if (key=="_device") return S(Llm || Flag("server_embedding"));
            if (key=="_audio") return S(Flag("server_transcription"));
            return Value(key);
        }
        public Dashboard(string stateJson, string schemaJson, string appVersion, bool preview) {
            state = Dict(Json.DeserializeObject(stateJson)); schema = Dict(Json.DeserializeObject(schemaJson));
            pages = Items(schema.ContainsKey("pages") ? schema["pages"] : null); fields = Items(schema.ContainsKey("fields") ? schema["fields"] : null);groups=Items(schema.ContainsKey("groups")?schema["groups"]:null);
            version = appVersion; demo=preview;
            notice=Get(schema,"notice","");
            dirty=Get(schema,"resume_dirty","false")=="true";
            foreach(var g in groups)if(Get(g,"open","false")=="true")expanded.Add(Get(g,"page","")+"/"+Get(g,"id",""));
            if(schema.ContainsKey("resume_groups")){expanded.Clear();var list=schema["resume_groups"] as IEnumerable;if(list!=null)foreach(object item in list)expanded.Add(S(item));}
            if (schema.ContainsKey("resume_page")) for(int i=0;i<pages.Count;i++) if(Get(pages[i],"id","")==S(schema["resume_page"]))page=i;
            navSelected=page;
            FocusFirst();
            if(schema.ContainsKey("resume_field")){
                string key=S(schema["resume_field"]);var target=Current().Find(delegate(Dictionary<string,object> f){return Get(f,"key","")==key;});
                if(target!=null&&Get(target,"group","").Length>0)expanded.Add(Get(target,"page","")+"/"+Get(target,"group",""));
                var current=Rows();int found=current.FindIndex(delegate(Dictionary<string,object> f){return Get(f,"key","")==key;});if(found>=0)selected=found;
            }
        }
        bool Visible(Dictionary<string,object> f) {
            if (Get(f,"page","") != Get(pages[page],"id","home")) return false;
            var when = Dict(f.ContainsKey("when")?f["when"]:null);
            foreach(var pair in when) {
                string actual=Derived(pair.Key); var many=pair.Value as IEnumerable;
                if (many!=null && !(pair.Value is string)) { bool ok=false;foreach(object item in many) if(S(item).Equals(actual,StringComparison.OrdinalIgnoreCase))ok=true;if(!ok)return false; }
                else if (!S(pair.Value).Equals(actual,StringComparison.OrdinalIgnoreCase)) return false;
            }
            return query.Length==0 || (Get(f,"label","")+Get(f,"key","")).IndexOf(query,StringComparison.OrdinalIgnoreCase)>=0;
        }
        List<Dictionary<string,object>> Current() { return fields.FindAll(Visible); }
        List<Dictionary<string,object>> Rows() {
            var result=new List<Dictionary<string,object>>();var seen=new HashSet<string>();
            foreach(var f in Current()) {
                if(PageId=="home"){result.Add(f);continue;}
                string id=Get(f,"group","");if(id.Length==0){result.Add(f);continue;}
                string full=Get(f,"page","")+"/"+id;
                if(seen.Add(full)) {
                    var g=groups.Find(delegate(Dictionary<string,object> item){return Get(item,"id","")==id&&Get(item,"page","")==Get(f,"page","");});
                    var header=new Dictionary<string,object>{{"kind","group"},{"key","@group:"+full},{"group_id",full},{"label",g==null?id:Get(g,"label",id)},{"help","Enter 展开或收起"}};
                    if(g!=null&&g.ContainsKey("summary_key"))header["summary_key"]=g["summary_key"];
                    result.Add(header);
                }
                if(query.Length>0||expanded.Contains(full))result.Add(f);
            }
            return result;
        }
        static int RowHeight(Dictionary<string,object> row) {return Get(row,"kind","")=="group"?1:2;}
        int FitEnd(List<Dictionary<string,object>> rows,int start,int budget) {int used=0,end=start;while(end<rows.Count&&used+RowHeight(rows[end])<=budget){used+=RowHeight(rows[end]);end++;}return end;}
        void FocusFirst() {
            var rows=Rows();int index=rows.FindIndex(delegate(Dictionary<string,object> f){return Get(f,"kind","")!="group";});
            if(PageId=="models"){int model=rows.FindIndex(delegate(Dictionary<string,object> f){return Get(f,"key","")=="model";});if(model>=0)index=model;}
            selected=Math.Max(0,index);offset=0;
        }
        void SetPage(int index) {
            page=Math.Max(0,Math.Min(index,pages.Count-1));navSelected=page;query="";notice="";FocusFirst();
        }
        string PageId {get{return Get(pages[page],"id","home");}}
        string LaunchCaption {get{if(PageId=="api"&&Value("launch_mode")!="server")return "切换为 API 服务";if(PageId=="benchmark"&&Value("launch_mode")!="benchmark")return "切换为性能测试";return Value("launch_mode")=="server"?"启动 API 服务":Value("launch_mode")=="benchmark"?"开始测试":"开始对话";}}
        bool PrepareLaunch() {
            if(PageId=="api"&&Value("launch_mode")!="server"){Change("launch_mode","server");FocusFirst();notice="已切换为 API 服务";return false;}
            if(PageId=="benchmark"&&Value("launch_mode")!="benchmark"){Change("launch_mode","benchmark");FocusFirst();notice="已切换为性能测试";return false;}
            return true;
        }
        string Display(Dictionary<string,object> f) {
            string key=Get(f,"key",""); string kind=Get(f,"kind","text"); string v=Value(key);
            if(kind=="group"){
                string summary=Get(f,"summary_key","");if(summary.Length==0)return "";
                if(summary=="server_llm_enabled")return Llm?"已开启":"未开启";
                if(summary=="setup_mode")return Value(summary)=="manual"?"手动":Value(summary)=="aggressive"?"激进":"保守";
                if(summary=="think_mode")return Value(summary)=="default"?"模型默认":Value(summary)=="no-think"?"关闭":"开启";
                if(summary=="bench_kind")return PlanName();
                if(state.ContainsKey(summary)&&state[summary] is bool)return Flag(summary)?"已开启":"未开启";
                return Canvas.Clip(Value(summary),10);
            }
            if (kind=="action") {
                string action=Get(f,"action","");
                if(action=="start")return "F5 · "+LaunchCaption+"  →";
                if(action=="page.models")return ModuleNames()+"  →";
                if(action=="page.benchmark")return PlanName()+" · "+PlanDepths()+" · "+LaneLabel()+"  →";
                return Get(f,"value","打开  →");
            }
            if (kind=="secret") return v.Length>0 ? "已填写 · 不保存 / Not saved" : "未设置 / Not set";
            if (kind=="toggle") return Flag(key)?"● 开启 / On":"○ 关闭 / Off";
            foreach(var option in Items(f.ContainsKey("options")?f["options"]:null)) if(Get(option,"value","")==v)return Get(option,"label",v);
            if(Get(f,"kind","")=="path"&&key=="model")return v.Length>0?ModelName:"选择模型…";
            return v.Length>0?v:"自动";
        }
        string ModuleNames(){var list=new List<string>();if(Llm)list.Add("语言模型");if(Value("launch_mode")=="server"){if(Llm&&Flag("server_vision"))list.Add("视觉");if(Flag("server_embedding"))list.Add("向量");if(Flag("server_transcription"))list.Add("转录");}return list.Count>0?String.Join(" · ",list.ToArray()):"选择要提供的服务";}
        string PlanName(){return Value("bench_kind")=="standard"?"标准":Value("bench_kind")=="full"?"完整":Value("bench_kind")=="manual"?"手动":"快速";}
        string PlanDepths(){return Value("bench_kind")=="standard"?"6K + 30K":Value("bench_kind")=="full"?"6K + 30K + 150K":Value("bench_kind")=="manual"?"自定义":"6K";}
        string LaneLabel(){return Value("bench_parallel")=="1"?"单路":"x1–x"+Value("bench_parallel");}
        string TaskName { get { return Value("launch_mode")=="chat"?"终端对话":Value("launch_mode")=="benchmark"?"性能测试":"API 服务"; } }
        string ModelName { get { string name=Value("model").Replace('\\','/');int i=name.LastIndexOf('/');return i>=0?name.Substring(i+1):name; } }
        void Resize() { try { width=Math.Max(45,Console.WindowWidth-1);height=Math.Max(16,Console.WindowHeight-1); } catch {width=128;height=40;} }
        static string[] Wrap(string text, int cols, int count) {
            var lines=new List<string>(); string rest=Canvas.Safe(text);cols=Math.Max(1,cols);
            while(rest.Length>0 && lines.Count<count) {string part=Canvas.Clip(rest,cols);if(part.Length==0)break;lines.Add(part);rest=rest.Substring(part.Length);}
            return lines.ToArray();
        }
        public Canvas Draw(int w, int h) {
            width=Math.Max(45,w);height=Math.Max(16,h);var c=new Canvas(width,height);
            int sidebar=width>=100?23:18;bool right=width>=118;int end=right?width-32:width-3;int x=sidebar+3;int bodyWidth=end-x-1;
            var current=Rows();if(current.Count==0)selected=0;else selected=Math.Max(0,Math.Min(selected,current.Count-1));
            c.Fill(0,0,width,3,"142131");c.Put(3,1,"m4 / MoE4All","69e5d2","142131",25);
            c.Put(29,1,"LOCAL AI WORKSPACE","7f96ab","142131",30);
            c.Put(Math.Max(45,width-27),1,(demo?"UI PREVIEW":"v"+version)+"  ·  "+(dirty?"未保存":"配置"),"c4d4e0","142131",26);
            c.Fill(0,3,sidebar,height-6,"101b29");c.Put(3,5,"工作空间","668198","101b29",sidebar-4);
            for(int i=0;i<pages.Count;i++) {
                bool compact=height<34;int y=compact?7+i*2:8+i*3;if(y>=(compact?height-3:height-6))break;bool focus=i==(nav?navSelected:page);
                string bg=focus?(nav?"20515a":"173c45"):"101b29";c.Fill(1,compact?y:y-1,sidebar-2,compact?1:3,bg);
                c.Put(2,y,(focus?"▌ ":"  ")+Primary(Get(pages[i],"label",Get(pages[i],"id",""))),focus?"84f5da":"a6b8ca",bg,sidebar-3);
                if(!compact)c.Put(4,y+1,Get(pages[i],"en",Get(pages[i],"id","")),focus?"56a69c":"536b82",bg,sidebar-5);
            }
            if(height>=34)c.Put(3,height-6,"moe4all.com","506b7f","101b29",sidebar-4);
            c.Fill(0,height-3,width,3,"142131");
            string focusedKind=current.Count>0?Get(current[selected],"kind",""):"";
            string footer=width>=118?(focusedKind=="group"?"Enter 展开/收起   Tab 导航   F5 "+LaunchCaption+"   F6 保存   Esc 返回":focusedKind=="path"?"Enter 选择文件   F2 模型历史   F4 浏览   F5 "+LaunchCaption+"   F6 保存   Esc 返回":"↑↓ 选择   Enter 编辑   Tab 导航   F5 "+LaunchCaption+"   F6 保存   Esc 返回"):"↑↓ Enter编辑 Tab导航 F5启动 F6保存 Esc返回";
            c.Put(3,height-2,footer,"b2c9d8","142131",width-6);
            string id=Get(pages[page],"id","home");
            c.Text(x,5,(page+1).ToString("00")+" / "+Get(pages[page],"en",id).ToUpperInvariant(),"528e9d",bodyWidth);
            c.Text(x,7,id=="home"?"启动工作台":Primary(Get(pages[page],"label",id)),"f0f7fa",bodyWidth);
            c.Text(x,9,Get(pages[page],"description",""),"71899e",bodyWidth);
            int budget=Math.Max(2,height-18);
            if(selected<offset)offset=selected;
            while(offset<selected&&FitEnd(current,offset,budget)<=selected)offset++;
            int visibleEnd=FitEnd(current,offset,budget);int rowY=11;
            for(int n=offset;n<visibleEnd;n++) {
                var f=current[n];bool focus=!nav&&n==selected;string kind=Get(f,"kind","");bool action=kind=="action";
                if(kind=="group") {
                    string full=Get(f,"group_id","");bool open=query.Length>0||expanded.Contains(full);string bg=focus?"183548":"101f2c";
                    c.Fill(x-1,rowY,bodyWidth+1,1,bg);c.Put(x,rowY,(open?"▾ ":"▸ ")+Get(f,"label",""),focus?"8ae8d5":"8aa8ba",bg,bodyWidth-13);
                    c.Put(end-11,rowY,Display(f),"587d8c",bg,10);rowY++;continue;
                }
                bool primary=action&&Get(f,"action","")=="start";
                string background=focus?(action?"4cbfa7":"183548"):primary?"102d32":"0b121b";
                c.Fill(x-1,rowY,bodyWidth+1,2,background);
                string label=action&&Get(f,"action","")=="start"?LaunchCaption:Get(f,"label",Get(f,"key",""));
                c.Put(x,rowY,(focus?"› ":"  ")+label,focus&&action?"062b29":focus?"e6f4fc":"b5c4d3",background,bodyWidth-2);
                c.Put(x+2,rowY+1,Display(f),focus&&action?"123c35":focus?"77e2d3":"6e8da6",background,bodyWidth-4);rowY+=2;
            }
            if(visibleEnd<current.Count||offset>0)c.Text(end-14,7,(selected+1)+" / "+current.Count,"677f94",14);
            if(PageId=="benchmark"&&Value("launch_mode")=="benchmark"&&visibleEnd==current.Count)DrawPlan(c,x,rowY+1,bodyWidth,height-8);
            c.Text(x,height-7,new String('─',Math.Max(0,bodyWidth)),"20374b",bodyWidth);
            string help=notice.Length>0?notice:current.Count>0?Get(current[selected],"help",Get(current[selected],"key","")):"此页没有适用项目。";
            string[] notes=Wrap(help,bodyWidth,2);for(int i=0;i<notes.Length;i++)c.Text(x,height-6+i,notes[i],notice.Length>0?"e4bc74":"6e8ba0",bodyWidth);
            if(right) {
                int rx=width-29; c.Fill(rx-2,4,31,height-8,"0e1925");
                c.Put(rx,6,"当前启动方案","bdd7e5","0e1925",26);
                c.Put(rx,8,"● 尚未启动","d5ac63","0e1925",26);
                string[] mn=Wrap(Llm?(ModelName.Length>0?ModelName:"请选择模型"):"独立模块服务",25,3);
                for(int i=0;i<mn.Length;i++)c.Put(rx,11+i,mn[i],"87cfc9","0e1925",25);
                string profile=Value("setup_mode")=="aggressive"?"自动激进":Value("setup_mode")=="conservative"?"自动保守":"手动配置";
                string device=Value("device").Length>0?Value("device"):"自动";
                var summaryRows=new List<KeyValuePair<string,string>>();
                summaryRows.Add(new KeyValuePair<string,string>("用途",TaskName));
                summaryRows.Add(new KeyValuePair<string,string>("设备",Llm||Flag("server_embedding")?device:"CPU"));
                if(Llm)summaryRows.Add(new KeyValuePair<string,string>("资源策略",profile));
                if(Value("launch_mode")=="benchmark") {
                    summaryRows.Add(new KeyValuePair<string,string>("测试方案",PlanName()));
                    summaryRows.Add(new KeyValuePair<string,string>("上下文",PlanDepths()));
                    summaryRows.Add(new KeyValuePair<string,string>("并发",Value("bench_kind")=="manual"?"单路":LaneLabel()));
                } else if(Llm) {
                    summaryRows.Add(new KeyValuePair<string,string>("上下文",Value("context").Length>0?Value("context"):"自动"));
                    if(Value("launch_mode")=="server")summaryRows.Add(new KeyValuePair<string,string>("并发",Value("server_parallel")));
                    summaryRows.Add(new KeyValuePair<string,string>("KV",Value("setup_mode")!="manual"||Value("kv_preset")=="auto"?"自动":Value("kv_preset")));
                } else summaryRows.Add(new KeyValuePair<string,string>("加载方式","按需加载"));
                for(int i=0;i<summaryRows.Count;i++){c.Put(rx,16+i*2,summaryRows[i].Key,"5c7a91","0e1925",12);c.Put(rx+12,16+i*2,summaryRows[i].Value,"bacddc","0e1925",15);}
                int moduleY=18+summaryRows.Count*2;
                if(moduleY+2<height-7) {
                    c.Put(rx,moduleY,"已启用","5c7a91","0e1925",25);
                    string[] modules=Wrap(ModuleNames(),26,2);for(int i=0;i<modules.Length;i++)c.Put(rx,moduleY+2+i,modules[i],"85bcaa","0e1925",26);
                }
                c.Put(rx,height-6,"F5  "+LaunchCaption+"  →","6ce0c3","0e1925",26);
                c.Put(rx,height-5,"预算仍由引擎计算","496c80","0e1925",26);
            }
            return c;
        }
        void DrawPlan(Canvas c,int x,int y,int columns,int bottom) {
            if(y+3>=bottom)return;
            c.Text(x,y,"将运行 / Plan","8fb2bd",columns);
            if(Value("bench_kind")=="manual") {
                c.Text(x,y+2,Value("manual_bench_kind")+" · Prefill "+Value("prompt_tokens")+" · Decode "+Value("gen_tokens"),"8b9dab",columns);
                if(y+4<bottom)c.Text(x,y+4,"重复 "+Value("reps")+" 次 · 结果自动保存","607e91",columns);return;
            }
            string[] depths=Value("bench_kind")=="full"?new[]{"6K","30K","150K"}:Value("bench_kind")=="standard"?new[]{"6K","30K"}:new[]{"6K"};
            string reps="1";int yy=y+2;
            c.Text(x,yy,"上下文          并发          重复","607e91",columns);yy++;
            foreach(string depth in depths){if(yy>=bottom)break;c.Text(x,yy,depth.PadRight(16)+LaneLabel().PadRight(14)+reps+" 次","9bb9c6",columns);yy++;}
            if(yy+1<bottom)c.Text(x,yy+1,"512 预热 + 1024 正式 / 路 · 三段统计","607e91",columns);
        }
        void Paint() { Resize(); Console.Write(Draw(width,height).Ansi(Environment.GetEnvironmentVariable("NO_COLOR")==null)); }
        Canvas Modal(string title,string help,int rows) {
            var c=Draw(width,height);int mw=Math.Min(width-8,88),left=(width-mw)/2,top=Math.Max(3,(height-rows)/2);
            c.Fill(left,top,mw,rows,"1b2d40"); c.Put(left+3,top+1,title,"dff9fa","1b2d40",mw-6);
            c.Put(left+3,top+3,help,"8da8bc","1b2d40",mw-6);return c;
        }
        void Popup(string title,string text) {
            while(true){Resize();int rows=Math.Min(height-6,12),mw=Math.Min(width-8,88),left=(width-mw)/2,top=(height-rows)/2;var c=Modal(title,text,rows);
                string[] lines=Wrap(text,mw-6,rows-6);for(int i=0;i<lines.Length;i++)c.Put(left+3,top+3+i,lines[i],"9ab9cc","1b2d40",mw-6);
                c.Put(left+3,top+rows-2,"Enter / Esc  返回","6ddfc6","1b2d40",mw-6);Console.Write(c.Ansi(Environment.GetEnvironmentVariable("NO_COLOR")==null));
                var k=Console.ReadKey(true);if(k.Key==ConsoleKey.Enter||k.Key==ConsoleKey.Escape)return;
            }
        }
        int Choose(string title,List<Dictionary<string,object>> choices,int initial) {
            if(choices.Count==0)return -1;int focus=Math.Max(0,Math.Min(initial,choices.Count-1));
            while(true){Resize();int count=Math.Max(1,Math.Min(choices.Count,height-15));int rows=count+9;int mw=Math.Min(width-8,88),left=(width-mw)/2,top=Math.Max(3,(height-rows)/2);
                var c=Modal(title,"↑↓ 选择，Enter 确认；Esc 取消",rows);int start=Math.Max(0,Math.Min(focus-count+1,choices.Count-count));
                for(int i=0;i<count;i++){bool sel=start+i==focus;string bg=sel?"2a5663":"1b2d40";c.Fill(left+2,top+5+i,mw-4,1,bg);c.Put(left+3,top+5+i,(sel?"› ":"  ")+Get(choices[start+i],"label",Get(choices[start+i],"value","")),sel?"9ff3dc":"91aabe",bg,mw-6);}
                string detail=Get(choices[focus],"type","")=="history"?Get(choices[focus],"value",""):Canvas.Columns(Get(choices[focus],"label",""))>mw-8?Get(choices[focus],"label",""):"";
                string[] lines=Wrap(detail,mw-6,2);for(int i=0;i<lines.Length;i++)c.Put(left+3,top+rows-3+i,lines[i],"63859b","1b2d40",mw-6);
                Console.Write(c.Ansi(Environment.GetEnvironmentVariable("NO_COLOR")==null));var key=Console.ReadKey(true);
                if(key.Key==ConsoleKey.Escape)return -1;if(key.Key==ConsoleKey.Enter)return focus;
                if(key.Key==ConsoleKey.UpArrow)focus=(focus-1+choices.Count)%choices.Count;if(key.Key==ConsoleKey.DownArrow)focus=(focus+1)%choices.Count;
                if(key.Key==ConsoleKey.Home)focus=0;if(key.Key==ConsoleKey.End)focus=choices.Count-1;
                if(key.KeyChar>='1'&&key.KeyChar<='9'&&key.KeyChar-'1'<choices.Count)focus=key.KeyChar-'1';
            }
        }
        string Edit(string title,string help,string value,bool secret) {
            string input=value;int cursor=input.Length;bool all=true;
            while(true) {
                Resize();int rows=11,mw=Math.Min(width-8,88),left=(width-mw)/2,top=(height-rows)/2;var c=Modal(title,help,rows);
                string shown=secret?new String('•',input.Length):input;
                string prefix=shown.Substring(0,Math.Min(cursor,shown.Length));int displayLimit=mw-9;
                while(prefix.Length>0 && Canvas.Clip(prefix,displayLimit).Length<prefix.Length)prefix=prefix.Substring(1);
                int start=Math.Max(0,cursor-prefix.Length);string tail=Canvas.Clip(shown.Substring(Math.Min(start,shown.Length)),displayLimit);
                c.Fill(left+3,top+5,mw-6,2,all?"264959":"0b1927");c.Put(left+4,top+5,tail,"e8f8fa",all?"264959":"0b1927",mw-8);
                int caret=left+4+Math.Min(displayLimit,Canvas.Columns(prefix));c.Put(caret,top+6,"▔","7cf2d5",all?"264959":"0b1927",1);
                c.Put(left+3,top+8,"Enter 保存 · Esc 取消 · Ctrl+A 全选 · Ctrl+V 粘贴","7aa0b7","1b2d40",mw-6);Console.Write(c.Ansi(Environment.GetEnvironmentVariable("NO_COLOR")==null));
                var k=Console.ReadKey(true);bool ctrl=(k.Modifiers&ConsoleModifiers.Control)!=0;
                if(k.Key==ConsoleKey.Escape)return null;if(k.Key==ConsoleKey.Enter)return input;
                if(ctrl&&k.Key==ConsoleKey.A){all=true;continue;}
                if(ctrl&&k.Key==ConsoleKey.C&&all&&!secret){try{Clipboard.SetText(input);}catch{}continue;}
                if(ctrl&&k.Key==ConsoleKey.V){try {string clip=Canvas.Safe(Clipboard.GetText());if(all){input="";cursor=0;}if(input.Length+clip.Length<=16384){input=input.Insert(cursor,clip);cursor+=clip.Length;}}catch{}all=false;continue;}
                if(k.Key==ConsoleKey.Home){cursor=0;all=false;continue;}if(k.Key==ConsoleKey.End){cursor=input.Length;all=false;continue;}
                if(k.Key==ConsoleKey.LeftArrow){if(all){cursor=0;all=false;continue;}cursor=Math.Max(0,cursor-1);if(cursor>0&&Char.IsLowSurrogate(input[cursor])&&Char.IsHighSurrogate(input[cursor-1]))cursor--;continue;}if(k.Key==ConsoleKey.RightArrow){if(all){cursor=input.Length;all=false;continue;}if(cursor<input.Length){cursor+=cursor+1<input.Length&&Char.IsSurrogatePair(input,cursor)?2:1;}continue;}
                if(k.Key==ConsoleKey.Backspace||k.Key==ConsoleKey.Delete){if(all){input="";cursor=0;}else if(k.Key==ConsoleKey.Backspace&&cursor>0){int len=cursor>1&&Char.IsLowSurrogate(input[cursor-1])&&Char.IsHighSurrogate(input[cursor-2])?2:1;input=input.Remove(cursor-len,len);cursor-=len;}else if(k.Key==ConsoleKey.Delete&&cursor<input.Length){int len=cursor+1<input.Length&&Char.IsHighSurrogate(input[cursor])&&Char.IsLowSurrogate(input[cursor+1])?2:1;input=input.Remove(cursor,len);}all=false;continue;}
                if(!Char.IsControl(k.KeyChar)&&input.Length<16384){if(all){input="";cursor=0;}input=input.Insert(cursor,k.KeyChar.ToString());cursor++;all=false;}
            }
        }
        void Change(string key,object value) {
            bool changed=Value(key)!=S(value);
            state[key]=value;if(changed){dirty=true;notice="";}
            if(changed&&key=="bench_kind"&&(S(value)=="quick"||S(value)=="standard"||S(value)=="full"))state["bench_parallel"]=S(value)=="quick"?"1":S(value)=="standard"?"2":"4";
            if(key=="think_mode"||key=="reasoning_effort")state["configure_thinking"]=Value("think_mode")!="default"||Value("reasoning_effort")!="default";
        }
        string Activate(Dictionary<string,object> f) {
            string key=Get(f,"key","");string kind=Get(f,"kind","text");
            if(kind=="group"){string id=Get(f,"group_id","");if(!expanded.Remove(id))expanded.Add(id);return null;}
            if(kind=="action"){
                string action=Get(f,"action",key.Replace("action.",""));
                if(action.StartsWith("task.",StringComparison.Ordinal)){Change("launch_mode",action.Substring(5));FocusFirst();notice="用途已切换，设置已保留";return null;}
                if(action.StartsWith("page.",StringComparison.Ordinal)){string id=action.Substring(5);int index=pages.FindIndex(delegate(Dictionary<string,object> p){return Get(p,"id","")==id;});if(index>=0)SetPage(index);return null;}
                return action;
            }
            if(kind=="toggle"){Change(key,!Flag(key));return null;}
            if(kind=="choice") {var opts=Items(f.ContainsKey("options")?f["options"]:null);int index=0;for(int i=0;i<opts.Count;i++)if(Get(opts[i],"value","")==Value(key))index=i;int picked=Choose(Get(f,"label",key),opts,index);if(picked>=0)Change(key,opts[picked].ContainsKey("value")?opts[picked]["value"]:"");return null;}
            if(kind=="path")return PickPath(f);
            string input=Edit(Get(f,"label",key),Get(f,"help","输入内容；留空沿用引擎默认。"),Value(key),kind=="secret");if(input!=null)Change(key,input);return null;
        }
        bool BrowsePath(Dictionary<string,object> f) {
            string key=Get(f,"key","model");
            try {
                if(key.EndsWith("_dir",StringComparison.Ordinal)){using(var folder=new FolderBrowserDialog()){folder.Description=Primary(Get(f,"label","选择目录"));if(Directory.Exists(Value(key)))folder.SelectedPath=Value(key);if(folder.ShowDialog()!=DialogResult.OK)return false;Change(key,folder.SelectedPath);}}
                else using(var dialog=new OpenFileDialog()){dialog.Filter=key=="config_path"?"TOML (*.toml)|*.toml|All files (*.*)|*.*":"GGUF models (*.gguf)|*.gguf|All files (*.*)|*.*";dialog.Title=Primary(Get(f,"label","选择文件"));if(File.Exists(Value(key)))dialog.FileName=Value(key);if(dialog.ShowDialog()!=DialogResult.OK)return false;Change(key,dialog.FileName);}
                return true;
            }catch(Exception ex){Popup("无法打开文件选择器",ex.Message);return false;}
        }
        string PickPath(Dictionary<string,object> f) {
            string key=Get(f,"key","model");var choices=new List<Dictionary<string,object>>();
            if(key=="model"&&state.ContainsKey("model_history")&&state["model_history"] is IEnumerable&&!(state["model_history"] is string))foreach(object item in (IEnumerable)state["model_history"]){string path=S(item);if(path.Length>0)choices.Add(new Dictionary<string,object>{{"label",BaseName(path)},{"value",path},{"type","history"}});}
            choices.Add(new Dictionary<string,object>{{"label",key.EndsWith("_dir",StringComparison.Ordinal)?"浏览文件夹… / Browse folder":"浏览文件… / Browse file"},{"type","browse"}});
            choices.Add(new Dictionary<string,object>{{"label","输入或粘贴路径… / Paste path"},{"type","edit"}});
            if(key=="model"||key=="vision_projector"||key=="embedding_model"||key=="transcription_model")choices.Add(new Dictionary<string,object>{{"label","下载推荐模型… / Download models"},{"type","download"}});
            int picked=Choose(Get(f,"label","选择路径"),choices,0);if(picked<0)return null;
            string type=Get(choices[picked],"type","");
            if(type=="history")Change(key,choices[picked]["value"]);
            if(type=="browse")BrowsePath(f);
            if(type=="edit"){string input=Edit(Get(f,"label",key),Get(f,"help",""),Value(key),false);if(input!=null)Change(key,input);}
            return type=="download"?"download":null;
        }
        static string BaseName(string path){string normalized=(path??"").Replace('\\','/');int split=normalized.LastIndexOf('/');return split>=0?normalized.Substring(split+1):normalized;}
        string ExitRequest() {
            if(!dirty)return "cancel";
            int result=Choose("退出前保留这些修改？",new List<Dictionary<string,object>>{new Dictionary<string,object>{{"label","保存并退出 / Save and exit"}},new Dictionary<string,object>{{"label","放弃更改 / Discard"}},new Dictionary<string,object>{{"label","继续编辑 / Keep editing"}}},2);
            return result==0?"save_exit":result==1?"cancel":null;
        }
        string Result(string action) {
            var exported=new Dictionary<string,object>(state);var secrets=new Dictionary<string,object>();
            foreach(var f in fields)if(Get(f,"kind","")=="secret"){string key=Get(f,"key","");if(exported.ContainsKey(key)){secrets[key]=exported[key];exported.Remove(key);}}
            exported.Remove("last_command");exported.Remove("api_key");
            var current=Rows();string focus=current.Count>0&&selected<current.Count?Get(current[Math.Max(0,selected)],"key",""):"";
            return Json.Serialize(new Dictionary<string,object>{{"action",action},{"state",exported},{"secrets",secrets},{"page",Get(pages[page],"id","home")},{"focus",focus},{"groups",new List<string>(expanded)},{"dirty",dirty}});
        }
        public static string Run(string stateJson,string schemaJson,string version,bool preview,string initialMessage) {
            var ui=new Dashboard(stateJson,schemaJson,version,preview);return ui.Loop(initialMessage);
        }
        public static string Snapshot(string stateJson,string schemaJson,string pageId,int width,int height,bool ansi) {
            var ui=new Dashboard(stateJson,schemaJson,"1.2.0",true);for(int i=0;i<ui.pages.Count;i++)if(Get(ui.pages[i],"id","")==pageId)ui.page=i;var frame=ui.Draw(width,height);return ansi?frame.Ansi(true):frame.Plain();
        }
        string Loop(string initialMessage) {
            IntPtr output=GetStdHandle(-11);uint mode;bool restore=GetConsoleMode(output,out mode);bool priorCursor=Console.CursorVisible;bool priorControl=Console.TreatControlCAsInput;ConsoleColor fg=Console.ForegroundColor,bg=Console.BackgroundColor;
            if(!restore||!SetConsoleMode(output,mode|4))throw new InvalidOperationException("This terminal does not support VT output. Use -ClassicWizard.");
            try {
                Console.TreatControlCAsInput=true;Console.Write("\x1b[?1049h\x1b[?25l\x1b[2J");
                if(!String.IsNullOrEmpty(initialMessage))Popup("请检查配置 / Review",initialMessage);
                while(true){Paint();var current=Rows();while(!Console.KeyAvailable){System.Threading.Thread.Sleep(50);int priorW=width,priorH=height;Resize();if(width!=priorW||height!=priorH)Paint();}var k=Console.ReadKey(true);
                    bool ctrl=(k.Modifiers&ConsoleModifiers.Control)!=0;
                    if(k.Key==ConsoleKey.F10||(ctrl&&k.Key==ConsoleKey.C)){string exit=ExitRequest();if(exit!=null)return Result(exit);continue;}
                    if(k.Key==ConsoleKey.Escape){if(query.Length>0){query="";selected=offset=0;}else if(nav){string exit=ExitRequest();if(exit!=null)return Result(exit);}else{nav=true;navSelected=page;}continue;}
                    if(k.KeyChar>='1'&&k.KeyChar<='7'&&k.KeyChar-'1'<pages.Count){SetPage(k.KeyChar-'1');nav=false;continue;}
                    if(k.Key==ConsoleKey.Tab){nav=!nav;if(nav)navSelected=page;continue;}
                    if(k.Key==ConsoleKey.F6||(ctrl&&k.Key==ConsoleKey.S))return Result("save");
                    if(k.Key==ConsoleKey.F5){if(!PrepareLaunch())continue;if(demo){Popup("预览模式","预览不启动引擎；实际使用请打开 Start-INFR-Wizard.cmd。");continue;}return Result("start");}
                    if(k.Key==ConsoleKey.F3){string q=Edit("查找当前页设置","输入名称，留空显示全部",query,false);if(q!=null){query=q;FocusFirst();}continue;}
                    if(k.Key==ConsoleKey.F2&&current.Count>0&&Get(current[selected],"key","")=="model") {
                        var choices=new List<Dictionary<string,object>>();var history=state.ContainsKey("model_history")?state["model_history"] as IEnumerable:null;
                        if(history!=null&&!(history is string))foreach(object item in history){string val=S(item);if(val.Length>0)choices.Add(new Dictionary<string,object>{{"label",val},{"value",val}});}
                        if(choices.Count==0)Popup("模型历史","尚无已保存模型；Enter 输入路径，F4 浏览 GGUF 文件。");else{int chosen=Choose("已添加的模型 / Model history",choices,0);if(chosen>=0){Change("model",choices[chosen]["value"]);return Result("model");}}continue;
                    }
                    if(k.Key==ConsoleKey.F4&&current.Count>0&&Get(current[selected],"kind","")=="path"){
                        if(BrowsePath(current[selected])&&Get(current[selected],"key","")=="model")return Result("model");continue;
                    }
                    if(nav){if(k.Key==ConsoleKey.UpArrow)navSelected=(navSelected-1+pages.Count)%pages.Count;if(k.Key==ConsoleKey.DownArrow)navSelected=(navSelected+1)%pages.Count;if(k.Key==ConsoleKey.Enter||k.Key==ConsoleKey.RightArrow){SetPage(navSelected);nav=false;}continue;}
                    if(k.Key==ConsoleKey.LeftArrow){nav=true;navSelected=page;continue;}
                    if(k.Key==ConsoleKey.UpArrow)selected=Math.Max(0,selected-1);if(k.Key==ConsoleKey.DownArrow)selected=Math.Min(current.Count-1,selected+1);
                    if(k.Key==ConsoleKey.PageUp)selected=Math.Max(0,selected-Math.Max(1,(height-20)/2));if(k.Key==ConsoleKey.PageDown)selected=Math.Min(current.Count-1,selected+Math.Max(1,(height-20)/2));
                    if(k.Key==ConsoleKey.Home)selected=0;if(k.Key==ConsoleKey.End)selected=Math.Max(0,current.Count-1);
                    if((k.Key==ConsoleKey.Enter||k.Key==ConsoleKey.Spacebar)&&current.Count>0){string beforeModel=Value("model");string action=Activate(current[selected]);if(action!=null){if(action=="start"&&!PrepareLaunch())continue;if(action=="start"&&demo)Popup("预览模式","预览不启动引擎。");else return Result(action);}if(beforeModel!=Value("model"))return Result("model");}
                }
            } finally {
                Console.Write("\x1b[0m\x1b[?25h\x1b[?1049l");Console.ForegroundColor=fg;Console.BackgroundColor=bg;
                try{Console.CursorVisible=priorCursor;Console.TreatControlCAsInput=priorControl;}catch{}if(restore)SetConsoleMode(output,mode);
            }
        }
    }
}
