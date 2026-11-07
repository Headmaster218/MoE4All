// Task-specific native runtime components; no browser or third-party runtime.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Net;
using System.Text;
using System.Globalization;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;

namespace MoE4All.Launcher {
    public sealed class ChatClient : IDisposable {
        readonly object gate=new object(); readonly string endpoint,key;
        readonly JavaScriptSerializer json=new JavaScriptSerializer{MaxJsonLength=2097152};
        readonly List<Dictionary<string,object>> history=new List<Dictionary<string,object>>();
        volatile bool disposed,cancelled;HttpWebRequest request;Thread worker;
        string model="",fault="";readonly StringBuilder answer=new StringBuilder(),reasoning=new StringBuilder();bool ready,busy;long revision;
        double prefillTps,decodeTps;long generatedTokens,promptTokens,cachedTokens,contextTokens;bool hasTimings;
        public sealed class State{public bool Ready,Busy,HasTimings;public string Model,Answer,Reasoning,Fault;public long Revision,GeneratedTokens,PromptTokens,CachedTokens,ContextTokens;public double PrefillTokensPerSecond,DecodeTokensPerSecond;}
        public ChatClient(string address,string apiKey){Uri uri;if(!Uri.TryCreate(address,UriKind.Absolute,out uri)||!uri.IsLoopback||uri.Scheme!="http")throw new ArgumentException("Local chat requires a loopback HTTP endpoint");endpoint=address.TrimEnd('/');key=apiKey??"";worker=new Thread(Prepare){IsBackground=true,Name="MoE4All local chat readiness"};worker.Start();}
        HttpWebRequest Create(string path){var r=(HttpWebRequest)WebRequest.Create(endpoint+path);r.Proxy=null;r.Timeout=5000;r.ReadWriteTimeout=30000;if(key.Length>0)r.Headers["Authorization"]="Bearer "+key;lock(gate){if(disposed||(busy&&cancelled)){r.Abort();throw new OperationCanceledException();}request=r;}return r;}
        static Dictionary<string,object> Dict(object o){return o as Dictionary<string,object>??new Dictionary<string,object>();}
        void Prepare(){while(!disposed){try{var r=Create("/models");using(var response=r.GetResponse())using(var reader=new StreamReader(response.GetResponseStream(),Encoding.UTF8)){var data=Dict(json.DeserializeObject(reader.ReadToEnd()));var models=data.ContainsKey("data")?data["data"] as object[]:null;if(models==null||models.Length==0)throw new IOException("No chat model is available");var entry=Dict(models[0]);string id=Convert.ToString(entry["id"]);lock(gate){model=id;ready=true;fault="";revision++;}return;}}catch(Exception ex){if(disposed)return;var web=ex as WebException;var response=web==null?null:web.Response;if(response!=null){response.Dispose();lock(gate){fault=ex.Message;revision++;}return;}Thread.Sleep(400);}}}
        public State Read(){lock(gate)return new State{Ready=ready,Busy=busy,Model=model,Answer=answer.ToString(),Reasoning=reasoning.ToString(),Fault=fault,Revision=revision,HasTimings=hasTimings,GeneratedTokens=generatedTokens,PromptTokens=promptTokens,CachedTokens=cachedTokens,ContextTokens=contextTokens,PrefillTokensPerSecond=prefillTps,DecodeTokensPerSecond=decodeTps};}
        static double Number(Dictionary<string,object> data,string key){object value;double n;return data.TryGetValue(key,out value)&&Double.TryParse(Convert.ToString(value,CultureInfo.InvariantCulture),NumberStyles.Float,CultureInfo.InvariantCulture,out n)&&n>=0&&!Double.IsInfinity(n)&&!Double.IsNaN(n)?n:0;}
        void ReadMetrics(Dictionary<string,object> packet){
            var timing=Dict(packet.ContainsKey("timings")?packet["timings"]:null);var usage=Dict(packet.ContainsKey("usage")?packet["usage"]:null);
            lock(gate){
                if(timing.Count>0){hasTimings=true;prefillTps=Number(timing,"prompt_per_second");decodeTps=Number(timing,"predicted_per_second");generatedTokens=(long)Number(timing,"predicted_n");promptTokens=(long)Number(timing,"prompt_n");cachedTokens=(long)Number(timing,"cached_n");contextTokens=(long)Number(timing,"context_n");revision++;}
                else if(usage.Count>0){generatedTokens=(long)Number(usage,"completion_tokens");promptTokens=(long)Number(usage,"prompt_tokens");revision++;}
            }
        }
        public bool Ask(string text){if(String.IsNullOrWhiteSpace(text))return false;lock(gate){if(!ready||busy||disposed)return false;busy=true;cancelled=false;answer.Clear();reasoning.Clear();fault="";hasTimings=false;prefillTps=decodeTps=0;generatedTokens=promptTokens=cachedTokens=contextTokens=0;revision++;history.Add(new Dictionary<string,object>{{"role","user"},{"content",text}});worker=new Thread(Generate){IsBackground=true,Name="MoE4All streamed conversation"};worker.Start();return true;}}
        void Generate(){string error="";bool finished=false;try{
            string body;lock(gate)body=json.Serialize(new Dictionary<string,object>{{"model",model},{"messages",history.ToArray()},{"stream",true},{"stream_options",new Dictionary<string,object>{{"include_usage",true}}}});
            byte[] bytes=Encoding.UTF8.GetBytes(body);var r=Create("/chat/completions");r.Method="POST";r.ContentType="application/json";r.ContentLength=bytes.Length;r.Timeout=30000;r.ReadWriteTimeout=600000;
            using(var stream=r.GetRequestStream())stream.Write(bytes,0,bytes.Length);
            using(var response=r.GetResponse())using(var reader=new StreamReader(response.GetResponseStream(),Encoding.UTF8)){
                string line;while(!disposed&&!cancelled&&(line=reader.ReadLine())!=null){if(!line.StartsWith("data:",StringComparison.Ordinal))continue;string payload=line.Substring(5).Trim();if(payload=="[DONE]"){finished=true;break;}if(payload.Length==0)continue;var packet=Dict(json.DeserializeObject(payload));if(packet.ContainsKey("error"))throw new IOException(Convert.ToString(Dict(packet["error"])["message"]));ReadMetrics(packet);var choices=packet.ContainsKey("choices")?packet["choices"] as object[]:null;if(choices==null)continue;foreach(var choice in choices){var value=Dict(choice);var delta=Dict(value.ContainsKey("delta")?value["delta"]:null);lock(gate){if(delta.ContainsKey("content"))answer.Append(Convert.ToString(delta["content"]));if(delta.ContainsKey("reasoning_content"))reasoning.Append(Convert.ToString(delta["reasoning_content"]));revision++;}if(value.ContainsKey("finish_reason")&&value["finish_reason"]!=null)finished=true;}}
            }
            if(!finished&&!cancelled&&!disposed)throw new IOException("The response ended before the completion marker");
        }catch(Exception e){if(!cancelled&&!disposed)error=e.Message;}finally{lock(gate){if(error.Length>0||cancelled){if(history.Count>0&&Convert.ToString(history[history.Count-1]["role"])=="user")history.RemoveAt(history.Count-1);}else if(!disposed)history.Add(new Dictionary<string,object>{{"role","assistant"},{"content",answer.ToString()}});fault=error;busy=false;request=null;revision++;}}}
        public bool Reset(){lock(gate){if(busy)return false;history.Clear();answer.Clear();reasoning.Clear();fault="";hasTimings=false;prefillTps=decodeTps=0;generatedTokens=promptTokens=cachedTokens=contextTokens=0;revision++;return true;}}
        public void Cancel(){cancelled=true;HttpWebRequest r;lock(gate)r=request;if(r!=null)r.Abort();}
        public void Dispose(){disposed=true;Cancel();if(worker!=null&&worker!=Thread.CurrentThread)worker.Join(2000);}
    }

    public sealed class BenchmarkResultsView : WizardPanel {
        sealed class ResultGrid : DataGridView {
            internal Action<int> Wheel;
            internal ResultGrid(){DoubleBuffered=true;}
            protected override void OnMouseWheel(MouseEventArgs e){if(Wheel!=null)Wheel(e.Delta);}
        }
        readonly ResultGrid grid;
        readonly WizardScrollTrack track;
        Font headerFont;
        bool scrolling;
        readonly List<string[]> rows=new List<string[]>();
        readonly Dictionary<string,int> liveRows=new Dictionary<string,int>();
        readonly HashSet<string> finishedRows=new HashSet<string>();
        readonly bool english;bool final;
        static readonly Regex prefill=new Regex(@"^Prefill (\d+):(?:.*?=\s*)?\s*([0-9]+(?:\.[0-9]+)?) tok/s",RegexOptions.CultureInvariant);
        static readonly Regex decode=new Regex(@"^Decode (\d+) x(\d+) rep (\d+)/(\d+): first ([0-9.]+), middle ([0-9.]+), last ([0-9.]+), overall ([0-9.]+) tok/s",RegexOptions.CultureInvariant);
        static readonly Regex preparing=new Regex(@"^Prefill (\d+) once",RegexOptions.CultureInvariant);
        static readonly Regex starting=new Regex(@"^Decode (\d+) x(\d+) rep (\d+)/(\d+)\.\.\.$",RegexOptions.CultureInvariant);
        static readonly Regex failure=new Regex(@"^Decode (\d+) x(\d+) rep (\d+)/(\d+) failed; continuing: (.*)$",RegexOptions.CultureInvariant);
        public BenchmarkResultsView():this(true){}
        public BenchmarkResultsView(bool en){
            english=en;Name="BenchmarkResults";Padding=new Padding(1);TabStop=true;
            grid=new ResultGrid{Name="BenchmarkResultGrid",Dock=DockStyle.Fill,ReadOnly=true,AllowUserToAddRows=false,AllowUserToDeleteRows=false,AllowUserToResizeRows=false,AllowUserToOrderColumns=false,RowHeadersVisible=false,MultiSelect=true,SelectionMode=DataGridViewSelectionMode.CellSelect,AutoSizeColumnsMode=DataGridViewAutoSizeColumnsMode.Fill,ScrollBars=ScrollBars.None,BorderStyle=BorderStyle.None,CellBorderStyle=DataGridViewCellBorderStyle.Single,ColumnHeadersHeightSizeMode=DataGridViewColumnHeadersHeightSizeMode.DisableResizing,ClipboardCopyMode=DataGridViewClipboardCopyMode.EnableAlwaysIncludeHeaderText};
            int[] weights={14,22,16,16,16,16};string[] headers=Headers();
            for(int i=0;i<headers.Length;i++)grid.Columns.Add(new DataGridViewTextBoxColumn{Name=headers[i],HeaderText=headers[i],FillWeight=weights[i],MinimumWidth=i==1?110:68,SortMode=DataGridViewColumnSortMode.NotSortable});
            grid.DefaultCellStyle.Padding=new Padding(8,2,8,2);
            for(int i=0;i<grid.Columns.Count;i++){var align=i<2?DataGridViewContentAlignment.MiddleLeft:DataGridViewContentAlignment.MiddleRight;grid.Columns[i].DefaultCellStyle.Alignment=align;grid.Columns[i].HeaderCell.Style.Alignment=align;}
            track=new WizardScrollTrack{Name="BenchmarkResultScroll",Dock=DockStyle.Right,Width=14};Controls.Add(grid);Controls.Add(track);
            track.ValueChanged+=delegate{SetFirst(track.Value);};grid.Scroll+=delegate{RefreshScroll();};grid.SizeChanged+=delegate{RefreshScroll();};
            grid.Wheel=delegate(int delta){if(delta!=0)SetFirst(FirstRow-(delta/120)*3);};
            WizardTheme.Changed+=ThemeChanged;ApplyGridTheme();Render();
        }
        string[] Headers(){return new[]{"Context","Workload","First","Middle","Last","Overall"};}
        static string Context(string raw){double n;if(!Double.TryParse(raw,NumberStyles.None,CultureInfo.InvariantCulture,out n))return raw;return n>=1024?(n/1024).ToString("0.#",CultureInfo.InvariantCulture)+"K":raw;}
        static string Work(string n,string rep,string total){return "Decode x"+n+(total=="1"?"":" ("+rep+"/"+total+")");}
        bool Put(string key,string[] values,bool complete){
            int index;if(liveRows.TryGetValue(key,out index)){if(!complete||finishedRows.Contains(key))return false;rows[index]=values;}
            else{liveRows[key]=rows.Count;rows.Add(values);}
            if(complete)finishedRows.Add(key);Render();return true;
        }
        public bool ObserveLine(string text){
            if(final)return false;string line=(text??"").Trim();Match m=prefill.Match(line);
            if(m.Success)return Put(m.Groups[1].Value+"|p",new[]{Context(m.Groups[1].Value),"Prefill","-","-","-",m.Groups[2].Value},true);
            m=decode.Match(line);
            if(m.Success)return Put(m.Groups[1].Value+"|d|"+m.Groups[2].Value+"|"+m.Groups[3].Value,new[]{Context(m.Groups[1].Value),Work(m.Groups[2].Value,m.Groups[3].Value,m.Groups[4].Value),m.Groups[5].Value,m.Groups[6].Value,m.Groups[7].Value,m.Groups[8].Value},true);
            m=failure.Match(line);
            if(m.Success)return Put(m.Groups[1].Value+"|d|"+m.Groups[2].Value+"|"+m.Groups[3].Value,new[]{Context(m.Groups[1].Value),Work(m.Groups[2].Value,m.Groups[3].Value,m.Groups[4].Value),"-","-","-",(english?"Failed: ":"失败：")+m.Groups[5].Value},true);
            m=preparing.Match(line);
            if(m.Success)return Put(m.Groups[1].Value+"|p",new[]{Context(m.Groups[1].Value),"Prefill","-","-","-","…"},false);
            m=starting.Match(line);
            if(m.Success)return Put(m.Groups[1].Value+"|d|"+m.Groups[2].Value+"|"+m.Groups[3].Value,new[]{Context(m.Groups[1].Value),Work(m.Groups[2].Value,m.Groups[3].Value,m.Groups[4].Value),"…","…","…","…"},false);
            return false;
        }
        public string ExportTsv(){var output=new StringBuilder();output.AppendLine(String.Join("\t",Headers()));foreach(var row in rows){var clean=new string[row.Length];for(int i=0;i<row.Length;i++)clean[i]=(row[i]??"").Replace('\t',' ').Replace('\r',' ').Replace('\n',' ');output.AppendLine(String.Join("\t",clean));}return output.ToString();}
        public string SelectedTsv(){if(grid.SelectedCells.Count>0){var data=grid.GetClipboardContent();if(data!=null){string selected=data.GetData(DataFormats.UnicodeText) as string;if(!String.IsNullOrEmpty(selected))return selected;}}return ExportTsv();}
        public bool HasRows{get{return rows.Count>0;}}
        public int ViewportWidth{get{return grid.ClientSize.Width;}}
        int FirstRow{get{return grid.Rows.Count==0||grid.FirstDisplayedScrollingRowIndex<0?0:grid.FirstDisplayedScrollingRowIndex;}}
        int VisibleRows{get{return Math.Max(1,(grid.ClientSize.Height-grid.ColumnHeadersHeight)/Math.Max(1,grid.RowTemplate.Height));}}
        void SetFirst(int value){if(grid.Rows.Count==0||!grid.IsHandleCreated)return;int next=Math.Max(0,Math.Min(Math.Max(0,grid.Rows.Count-VisibleRows),value));if(next!=FirstRow)grid.FirstDisplayedScrollingRowIndex=next;RefreshScroll();}
        void RefreshScroll(){if(scrolling||grid==null||track==null)return;scrolling=true;try{track.SetMetrics(grid.Rows.Count,VisibleRows,FirstRow);}finally{scrolling=false;}}
        public void ScrollToLatest(){SetFirst(Math.Max(0,grid.Rows.Count-VisibleRows));}
        void ThemeChanged(object sender,EventArgs e){if(IsDisposed)return;if(IsHandleCreated&&InvokeRequired){BeginInvoke((Action)ApplyGridTheme);return;}ApplyGridTheme();}
        void ApplyGridTheme(){if(grid==null)return;var p=WizardTheme.Current;BackColor=p.Border;grid.EnableHeadersVisualStyles=false;grid.BackgroundColor=p.Surface;grid.GridColor=p.Outline;grid.DefaultCellStyle.BackColor=p.Surface;grid.DefaultCellStyle.ForeColor=p.Text;grid.DefaultCellStyle.SelectionBackColor=p.Selected;grid.DefaultCellStyle.SelectionForeColor=p.Text;grid.ColumnHeadersDefaultCellStyle.BackColor=p.SurfaceAlt;grid.ColumnHeadersDefaultCellStyle.ForeColor=p.Text;grid.ColumnHeadersDefaultCellStyle.SelectionBackColor=p.SurfaceAlt;grid.ColumnHeadersDefaultCellStyle.SelectionForeColor=p.Text;ApplyGridFont();ColorRows();grid.Invalidate();}
        void ApplyGridFont(){if(grid==null)return;grid.Font=Font;if(headerFont!=null)headerFont.Dispose();headerFont=new Font(Font,FontStyle.Bold);grid.ColumnHeadersDefaultCellStyle.Font=headerFont;grid.RowTemplate.Height=Math.Max(28,Font.Height+12);grid.ColumnHeadersHeight=Math.Max(32,Font.Height+16);foreach(DataGridViewRow row in grid.Rows)row.Height=grid.RowTemplate.Height;RefreshScroll();}
        void ColorRows(){if(grid==null)return;var p=WizardTheme.Current;bool alternate=false;string previous=null;for(int i=0;i<rows.Count&&i<grid.Rows.Count;i++){if(previous!=null&&rows[i][0]!=previous)alternate=!alternate;previous=rows[i][0];grid.Rows[i].DefaultCellStyle.BackColor=alternate?p.SurfaceAlt:p.Surface;}}
        protected override void OnFontChanged(EventArgs e){base.OnFontChanged(e);ApplyGridFont();}
        protected override void Dispose(bool disposing){if(disposing){WizardTheme.Changed-=ThemeChanged;if(headerFont!=null){headerFont.Dispose();headerFont=null;}}base.Dispose(disposing);}
        public void LoadRows(object source){var failures=rows.FindAll(delegate(string[] row){return row[5].StartsWith("Failed: ",StringComparison.Ordinal)||row[5].StartsWith("失败：",StringComparison.Ordinal);});rows.Clear();liveRows.Clear();finishedRows.Clear();final=true;var list=source as IEnumerable;if(list!=null)foreach(object item in list){var row=item as Dictionary<string,object>;if(row==null)continue;var values=new string[6];string[] keys={"context","workload","first","middle","last","overall"};for(int i=0;i<keys.Length;i++)values[i]=row.ContainsKey(keys[i])?Convert.ToString(row[keys[i]]):"-";rows.Add(values);}foreach(var failureRow in failures)if(!rows.Exists(delegate(string[] row){return String.Join("\t",row)==String.Join("\t",failureRow);}))rows.Add(failureRow);Render();}
        void Render(){bool follow=FirstRow+VisibleRows>=grid.Rows.Count-1;bool empty=grid.Rows.Count==0;grid.SuspendLayout();try{while(grid.Rows.Count>rows.Count)grid.Rows.RemoveAt(grid.Rows.Count-1);while(grid.Rows.Count<rows.Count)grid.Rows.Add();for(int i=0;i<rows.Count;i++)for(int j=0;j<6;j++){var cell=grid.Rows[i].Cells[j];if(Convert.ToString(cell.Value)!=rows[i][j])cell.Value=rows[i][j];cell.ToolTipText=rows[i][j];}ColorRows();if(empty){grid.ClearSelection();grid.CurrentCell=null;}RefreshScroll();if(follow)ScrollToLatest();}finally{grid.ResumeLayout();}}
    }
}
