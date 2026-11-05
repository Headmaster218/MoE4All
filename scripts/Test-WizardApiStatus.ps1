[CmdletBinding()]
param(
    [ValidatePattern('^[0-9]{8}-[0-9]{6}-[A-Za-z0-9-]+$')][string]$TaskId = '20261105-223000-benchmark-submit-dialog-b8d4'
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($PSVersionTable.PSEdition -ne 'Desktop') { throw 'Use powershell.exe (Windows PowerShell 5.1) for the inbox .NET module tests.' }
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$workRoot = 'D:\AIinfr\infr\local-work'
$runId = [DateTime]::Now.ToString('yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0,8)
$tmp = Join-Path $workRoot ('tmp\moe4all\' + $TaskId + '\api-status\' + $runId)
$evidence = Join-Path $workRoot ('evidence\moe4all\' + $TaskId + '\api-status\' + $runId)
foreach ($directory in @($tmp, $evidence)) {
    $full = [IO.Path]::GetFullPath($directory)
    if (-not $full.StartsWith($workRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Test output escaped the shared work root.' }
    $cursor = $full
    while ($cursor) {
        if ((Test-Path -LiteralPath $cursor) -and ((Get-Item -LiteralPath $cursor).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Test output traverses a directory link.' }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
}
$drive = [IO.DriveInfo]::new([IO.Path]::GetPathRoot($workRoot))
if ($drive.AvailableFreeSpace -lt 64MB) { throw 'At least 64 MiB free space is required for this small fixture.' }
foreach ($directory in @($tmp, $evidence)) { [void][IO.Directory]::CreateDirectory($directory) }
Write-Output ('TEMP/TMP/compiler: ' + $tmp)
Write-Output ('Evidence: ' + $evidence)
Write-Output 'CPU-only in-process WinForms module; no service, engine, GPU, model or clipboard modification.'
$oldEnvironment = @{}
foreach ($name in @('TEMP','TMP','TMPDIR')) { $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name,'Process') }
$manifest = [ordered]@{ task_id=$TaskId; module='api-status'; status='running'; started_utc=[DateTime]::UtcNow.ToString('o'); repo=$repoRoot; commit=(& git -C $repoRoot rev-parse HEAD); dirty=$true; tmp=$tmp; evidence=$evidence; command='powershell.exe -NoLogo -NoProfile -NonInteractive -File scripts/Test-WizardApiStatus.ps1'; network='none'; engine='unused'; gpu='unused'; user_clipboard='unchanged'; source_files=@() }
try {
    $env:TEMP=$tmp; $env:TMP=$tmp; $env:TMPDIR=$tmp
    $sources = @()
    # Use the real widget dependencies (SoftButton lives in wizard-desktop.cs),
    # but exercise only ApiStatusView; never initialize a runtime or service.
    foreach ($name in @('wizard-console.cs','wizard-theme.cs','wizard-desktop.cs','wizard-prompts.cs','wizard-runtime.cs','wizard-downloads.cs','wizard-markdown.cs','wizard-workspaces.cs','wizard-api-status.cs')) {
        $source = Join-Path $PSScriptRoot $name
        $snapshot = Join-Path $tmp $name
        Copy-Item -LiteralPath $source -Destination $snapshot
        $sources += $snapshot
        $manifest.source_files += [ordered]@{ path=$source; sha256=(Get-FileHash -LiteralPath $snapshot -Algorithm SHA256).Hash.ToLowerInvariant() }
    }
    $harness = @'
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Text;
using System.Windows.Forms;
using MoE4All.Launcher;
static class ApiStatusTests {
    sealed class TestHost : Form {
        protected override bool ShowWithoutActivation { get { return true; } }
        protected override CreateParams CreateParams { get { var p=base.CreateParams; p.ExStyle|=0x08000000; return p; } }
    }
    const string Border = "+---------------+---------------------------------+-----------+---------------+-----------------------+";
    static int checks;
    static void Check(bool ok, string label) { checks++; if (!ok) throw new Exception(label); }
    static string Row(string slot,string work,string state,string speed,string context) {
        return String.Format(CultureInfo.InvariantCulture,"| {0,-13} | {1,-31} | {2,-9} | {3,13} | {4,21} |\n",slot,work,state,speed,context);
    }
    static string Frame(string rows,string total) {
        return "INFR live resources\n"+Border+"\n"+Row("Slot","Work","State","Speed (now)","Context")+Border+"\n"+rows+Border+"\n"+total+"\n";
    }
    static string Total(string decode) { return "Total: prefill 1200.0 tok/s | decode "+decode+" tok/s | active 2 | queued 3 | done 12 | failed 0"; }
    static string Active() { return Frame(Row("KV 1/2","#17 Model | A \u4e2d\u6587","Decode","37.5 tok/s","50,057 / 160,000")+Row("KV 2/2","#19 model","Prefill","1200.0 tok/s","30,000 / 160,000")+Row("Embedding","#18 3 input(s)","Embedding","1.4s","-"),Total("99.9")); }
    static DataGridView Grid(ApiStatusView view) { return (DataGridView)view.Controls.Find("ApiStatusGrid",true)[0]; }
    [STAThread] static int Main(string[] argv) {
        try {
            Application.EnableVisualStyles();
            using (var view=new ApiStatusView(true)) {
                Check(!view.HasFrame && !view.HasRows && view.ActiveDecodeTokensPerSecond==null,"empty view must not invent speed");
                Check(!view.ObserveTranscript("serve stats decode_tps=999.0\nrequest done decode_tps=888.0\n"),"ordinary aggregate/request logs are not slot frames");
                Check(!view.ObserveTranscript(Total("777.0")+"\n"),"orphan Total rejected");
                string first=Active();
                Check(view.ObserveTranscript(first),"complete active frame accepted");
                Check(view.HasFrame && view.RowCount==3 && view.HasRows,"two KV rows plus embedding");
                Check(view.ActiveDecodeTokensPerSecond==37.5,"actual per-slot Decode sum, not Total or divided aggregate");
                Check(view.TotalText==Total("99.9"),"Total kept independently and verbatim");
                var grid=Grid(view);
                Check(Convert.ToString(grid.Rows[0].Cells[1].Value)=="#17 Model | A \u4e2d\u6587","pipe and Unicode in Work preserved");
                Check(Convert.ToString(grid.Rows[0].Cells[4].Value)=="50,057 / 160,000","grouped context preserved");
                Check(Convert.ToString(grid.Rows[2].Cells[3].Value)=="1.4s","embedding elapsed seconds not token rate");
                Check(view.ExportTsv().Contains("Slot\tWork\tState\tSpeed (now)\tContext") && view.ExportTsv().Contains(Total("99.9")),"copy export includes headers and Total");
                grid.ClearSelection(); grid.Rows[0].Cells[1].Selected=true;
                Check(view.SelectedTsv().Contains("#17 Model | A"),"selected-cell export without touching clipboard");
                Check(!view.ObserveTranscript(first),"identical transcript rejected");
                Check(!view.ObserveTranscript("log\n"+first+first+"unrelated log\n"),"repeated frame rejected");
                Check(grid.Rows[0].Cells[1].Selected,"duplicate does not discard selection");
                string idle=Frame(Row("KV 1/1","model","Idle","-","-")+Row("Embedding","embed-model","Idle","-","-"),Total("987.0"));
                for (int split=0;split<idle.Length;split++) {
                    Check(!view.ObserveTranscript(idle.Substring(0,split)),"partial frame accepted at split "+split);
                    Check(view.ActiveDecodeTokensPerSecond==37.5 && view.TotalText==Total("99.9"),"partial read overwrote last complete frame");
                }
                Check(!view.ObserveTranscript(first+idle.Substring(0,idle.Length-1)),"unterminated Total must not be mistaken for complete failed count");
                Check(view.ObserveTranscript(first+idle),"latest complete frame wins");
                Check(view.ActiveDecodeTokensPerSecond==0 && view.RowCount==2,"idle frame clears active speed despite nonzero Total interval");
                Check(!view.ObserveTranscript("cropped middle of table\n") && view.HasFrame,"transcript trim retains last complete frame");
                Check(!view.ObserveTranscript(null),"null snapshot harmless");
                Console.WriteLine("PASS atomic framing, every split boundary, duplicates, trim, verbatim metrics and copy");
                for (int slots=1;slots<=8;slots++) {
                    var rows=new StringBuilder(); double expected=0;
                    for (int i=1;i<=slots;i++) {
                        string state=i%3==0?"Starting":i%3==1?"Decode":"Prefill";
                        string speed=state=="Starting"?"-":(i*10).ToString("0.0",CultureInfo.InvariantCulture)+" tok/s";
                        if(state=="Decode")expected+=i*10;
                        rows.Append(Row("KV "+i+"/"+slots,"model",state,speed,state=="Starting"?"-":"1,024 / 163,840"));
                    }
                    rows.Append(Row("Embed 1/2","#22 2 input(s)","Embedding","0.2s","-"));
                    rows.Append(Row("Embed 2/2","embedding","Idle","-","-"));
                    Check(view.ObserveTranscript(Frame(rows.ToString(),Total("800.0"))),"accept capacity "+slots);
                    Check(view.RowCount==slots+2 && view.ActiveDecodeTokensPerSecond==expected,"preserve capacity and Decode-only sum "+slots);
                }
                string good=view.ExportTsv();
                string[] malformed={
                    Frame(Row("KV 1/2","model","Decode","1.0 tok/s","-"),Total("1.0")),
                    Frame(Row("KV 1/2","model","Decode","1.0 tok/s","-")+Row("KV 1/2","model","Decode","1.0 tok/s","-"),Total("1.0")),
                    Frame(Row("KV 2/2","model","Decode","1.0 tok/s","-"),Total("1.0")),
                    Frame(Row("KV 1/9","model","Decode","1.0 tok/s","-"),Total("1.0")),
                    Frame(Row("KV 1/1","model","Starting","1.0 tok/s","-"),Total("1.0")),
                    Frame(Row("KV 1/1","model","Decode","NaN tok/s","-"),Total("1.0")),
                    Frame(Row("KV 1/1","model","Decode","-1.0 tok/s","-"),Total("1.0")),
                    Frame(Row("KV 1/1","model","Decode","1,5 tok/s","-"),Total("1.0")),
                    Frame(Row("KV 1/1","model","Decode","1.0 tok/s","1,,000 / 160,000"),Total("1.0")),
                    Frame(Row("Embedding","embed","Embedding","30.0 tok/s","-"),Total("1.0")),
                    first.Replace("Speed (now)","Speed (average)"),first.Replace("failed 0","failed"),first.Replace("decode 99.9","decode NaN"),
                    "INFR live resources\n"+Row("KV 1/1","model","Decode","1.0 tok/s","-")+Total("1.0")+"\n"
                };
                foreach(string bad in malformed) { Check(!view.ObserveTranscript(bad),"malformed frame accepted"); Check(view.ExportTsv()==good,"malformed frame replaced visible values"); }
                var previous=System.Threading.Thread.CurrentThread.CurrentCulture;
                try { System.Threading.Thread.CurrentThread.CurrentCulture=new CultureInfo("fr-FR"); Check(view.ObserveTranscript(first.Replace("37.5 tok/s","38.5 tok/s").Replace("\n","\r\n")),"invariant decimal parser and CRLF"); Check(view.ActiveDecodeTokensPerSecond==38.5,"decimal separator independent of UI culture"); }
                finally { System.Threading.Thread.CurrentThread.CurrentCulture=previous; }
                Check(view.ObserveTranscript(Frame("",Total("444.0"))) && view.HasFrame && !view.HasRows && view.ActiveDecodeTokensPerSecond==0,"zero-slot/legacy total is not a fabricated slot speed");
                Console.WriteLine("PASS capacities 1-8, embedding lanes, malformed/incomplete groups, locale and zero-slot legacy frames");
            }
            using(var form=new TestHost {Size=new Size(1000,440),ShowInTaskbar=false,StartPosition=FormStartPosition.Manual,Location=new Point(-20000,-20000)})
            using(var view=new ApiStatusView(false) {Dock=DockStyle.Fill,Font=new Font("Microsoft YaHei UI",10)}) {
                var eight=new StringBuilder();
                for(int i=1;i<=8;i++)eight.Append(Row("KV "+i+"/8","model",i==1?"Decode":i==2?"Prefill":"Idle",i==1?"37.5 tok/s":i==2?"1200.0 tok/s":"-",i<=2?"50,057 / 160,000":"-"));
                eight.Append(Row("Embed 1/2","#18 3 input(s)","Embedding","1.4s","-"));
                eight.Append(Row("Embed 2/2","embed-model","Idle","-","-"));
                form.Controls.Add(view); view.ObserveTranscript(Frame(eight.ToString(),Total("99.9"))); form.Show(); Application.DoEvents();
                foreach(string theme in new[]{"light","dark"}) {
                    WizardTheme.SetPreference(theme); form.PerformLayout(); view.PerformLayout();
                    var grid=Grid(view); grid.CreateControl();
                    Check(grid.DefaultCellStyle.ForeColor==WizardTheme.Current.Text,"theme text "+theme);
                    Check(grid.DefaultCellStyle.SelectionBackColor==WizardTheme.Current.Selected,"theme selection "+theme);
                    Check(grid.ColumnHeadersDefaultCellStyle.BackColor==WizardTheme.Current.SurfaceAlt,"theme header "+theme);
                    Check(Convert.ToString(grid.Rows[0].Cells[3].Value)=="37.5 tok/s","theme must preserve data");
                    using(var bitmap=new Bitmap(form.Width,form.Height)) { form.DrawToBitmap(bitmap,new Rectangle(0,0,bitmap.Width,bitmap.Height)); bitmap.Save(Path.Combine(argv[0],"api-status-"+theme+".png")); }
                    form.Width=430; form.PerformLayout(); view.PerformLayout(); grid.PerformLayout();
                    Check(grid.Columns.GetColumnsWidth(DataGridViewElementStates.Visible)>grid.ClientSize.Width,"narrow viewport needs horizontal scroll");
                    var track=(WizardScrollTrack)view.Controls.Find("ApiStatusHorizontalScroll",true)[0]; Check(track.Horizontal,"themed horizontal scroll available");
                    form.Width=1000;
                }
                form.Height=230; form.PerformLayout(); view.PerformLayout(); Grid(view).PerformLayout();
                IntPtr handle=Grid(view).Handle;
                Grid(view).FirstDisplayedScrollingRowIndex=0; view.ScrollToLatest();
                Check(Grid(view).FirstDisplayedScrollingRowIndex>0 && Grid(view).Rows[view.RowCount-1].Displayed,"ScrollToLatest reveals embedding rows after 8 KV slots");
                view.Font=new Font("Microsoft YaHei UI",14);
                Check(Grid(view).Rows[0].Height>=view.Font.Height+10,"font growth updates compact row metrics");
                Console.WriteLine("PASS detached light/dark render, narrow viewport, scalable fonts and theme data retention");
            }
            Console.WriteLine("PASS "+checks+" CPU module assertions; offscreen nonactivating test host only; no clipboard writes, services, GPU or network");
            return 0;
        } catch(Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
'@
    $harnessPath = Join-Path $tmp 'ApiStatusTests.cs'
    [IO.File]::WriteAllText($harnessPath,$harness,[Text.UTF8Encoding]::new($true))
    $exe = Join-Path $tmp 'ApiStatusTests.exe'
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    if (-not (Test-Path -LiteralPath $compiler)) { $compiler=Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe' }
    $nativeArgs = @('/nologo','/optimize+','/target:exe','/utf8output','/r:System.Windows.Forms.dll','/r:System.Drawing.dll','/r:System.Web.Extensions.dll',('/out:'+$exe)) + $sources + @($harnessPath)
    $previousPreference = $ErrorActionPreference; $ErrorActionPreference='Continue'
    try { $compileOutput = & $compiler @nativeArgs 2>&1 | Out-String; $compileExit = $LASTEXITCODE }
    finally { $ErrorActionPreference=$previousPreference }
    [IO.File]::WriteAllText((Join-Path $evidence 'compile.log'),$compileOutput,[Text.UTF8Encoding]::new($false))
    if ($compileExit -ne 0) { throw $compileOutput }
    $manifest.fixture_sha256=(Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
    $previousPreference = $ErrorActionPreference; $ErrorActionPreference='Continue'
    try { $testOutput = & $exe $evidence 2>&1 | Out-String; $testExit = $LASTEXITCODE }
    finally { $ErrorActionPreference=$previousPreference }
    [IO.File]::WriteAllText((Join-Path $evidence 'tests.log'),$testOutput,[Text.UTF8Encoding]::new($false))
    if ($testExit -ne 0) { throw $testOutput }
    $manifest.status='passed'; $manifest.results=$testOutput.Trim(); Write-Output $testOutput
} catch {
    $manifest.status='failed'; $manifest.error=$_.Exception.Message; throw
} finally {
    $manifest.finished_utc=[DateTime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText((Join-Path $evidence 'manifest.json'),($manifest|ConvertTo-Json -Depth 7),[Text.UTF8Encoding]::new($false))
    foreach ($name in $oldEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name,$oldEnvironment[$name],'Process') }
}
