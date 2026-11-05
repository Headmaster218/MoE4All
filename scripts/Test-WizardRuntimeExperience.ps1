[CmdletBinding()]
param(
    [string]$FixturePath = 'D:\AIinfr\infr\local-work\tmp\moe4all\20261105-031000-runtime-experience-c81e\MoE4AllUiRuntimeFixture.exe',
    [string]$TaskId = '20261105-031000-runtime-experience-c81e'
)

# Simulation only. Uses real launcher classes, reflection for internal decoder/log,
# in-process WinForms and an ephemeral TCP HTTP stub bound only to 127.0.0.1.
# Run in a fresh Windows PowerShell 5.1 process. Starts one synthetic PowerShell
# child through the real ConPTY/runtime path; never starts the model/fake EXE.
$ErrorActionPreference = 'Stop'
Get-Content -LiteralPath 'D:\AIinfr\disk_usage_rule.md' | Out-Null
if (Test-Path -LiteralPath 'D:\cpuandram.lock') { throw 'CPU/RAM diagnostic lock exists; defer this test.' }
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$workRoot = 'D:\AIinfr\infr\local-work'
$runId = 'experience-' + $PID + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)

function Assert-TaskPath([string]$Path, [string]$AllowedRoot) {
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $allowed = [IO.Path]::GetFullPath($AllowedRoot).TrimEnd('\')
    if (-not $full.StartsWith($allowed + '\', [StringComparison]::OrdinalIgnoreCase)) { throw ('Invalid task path: ' + $full) }
    $cursor = $full
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            if (((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw ('Task path traverses a reparse point: ' + $cursor)
            }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $full
}
function Get-TestIdentity([string]$Path) {
    $sha = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try { $value = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $sha.Dispose() }
    return [ordered]@{ path = $Path; sha256 = $value }
}

$tmpRoot = Join-Path $workRoot ('tmp\moe4all\' + $taskId)
$tmpDirectory = Assert-TaskPath (Join-Path $tmpRoot $runId) $tmpRoot
$evidenceRoot = Join-Path $workRoot ('evidence\moe4all\' + $taskId)
$evidenceDirectory = Assert-TaskPath (Join-Path $evidenceRoot ('runtime-experience-tests\' + $runId)) $evidenceRoot
$FixturePath = Assert-TaskPath $FixturePath $tmpRoot
if (-not (Test-Path -LiteralPath $FixturePath -PathType Leaf)) { throw ('Existing simulated fixture not found: ' + $FixturePath) }
$drive = New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($workRoot))
if ($drive.AvailableFreeSpace -lt 96MB) { throw 'At least 96 MiB free disk space is required.' }
[void][IO.Directory]::CreateDirectory($tmpDirectory)
[void][IO.Directory]::CreateDirectory($evidenceDirectory)
$cache = Join-Path $tmpDirectory 'launcher'
Write-Output ('Output: ' + $evidenceDirectory)
Write-Output ('Temp and launcher compilation: ' + $tmpDirectory + '; KV/GPU/models/upload: unused')
$oldEnvironment = @{}
foreach ($name in @('TEMP', 'TMP', 'TMPDIR', 'MOE4ALL_WORK_ROOT', 'MOE4ALL_WIZARD_TEST_TMP', 'MOE4ALL_LAUNCHER_CACHE', 'INFR_API_KEY')) {
    $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$manifest = [ordered]@{
    task = $taskId; owner = 'Test-WizardRuntimeExperience.ps1'; status = 'running'
    purpose = 'Simulated runtime scrolling, control bridge, local SSE chat and task layouts'
    startedUtc = [DateTime]::UtcNow.ToString('o'); repository = $repoRoot
    command = 'powershell.exe -NoLogo -NoProfile -NonInteractive -File scripts/Test-WizardRuntimeExperience.ps1'
    temp = $tmpDirectory; evidence = $evidenceDirectory
    processId = $PID; processStartedUtc = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
    network = 'Ephemeral 127.0.0.1 TCP HTTP stub only; no external service or upload'
    fixture = Get-TestIdentity $FixturePath
}

try {
    $env:TEMP = $tmpDirectory; $env:TMP = $tmpDirectory; $env:TMPDIR = $tmpDirectory
    $env:MOE4ALL_WORK_ROOT = $workRoot; $env:MOE4ALL_WIZARD_TEST_TMP = $tmpDirectory
    $env:MOE4ALL_LAUNCHER_CACHE = $cache
    $env:INFR_API_KEY = $null # Do not send a real inherited credential to the test stub.
    $dataDir = Join-Path $tmpDirectory 'unused-data'; $script:IsPackagedInstall = $false
    . (Join-Path $PSScriptRoot 'infr-dashboard.ps1')
    Initialize-DashboardRuntime
    $assembly = [MoE4All.Launcher.RuntimeWindow].Assembly.Location
    if (-not $assembly.StartsWith($cache + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Launcher was already loaded from another run; use a fresh powershell.exe process.'
    }
    # The launcher compiles under a unique temporary assembly name, then renames
    # the DLL to its source hash. A separate executable needs the CLR identity name.
    $bindingAssembly = Join-Path $cache ([MoE4All.Launcher.RuntimeWindow].Assembly.GetName().Name + '.dll')
    if ($bindingAssembly -ne $assembly) { Copy-Item -LiteralPath $assembly -Destination $bindingAssembly }
    $source = Join-Path $tmpDirectory 'RuntimeExperienceHarness.cs'
    $exe = Join-Path $cache 'RuntimeExperienceHarness.exe'
    $childSource = Join-Path $tmpDirectory 'NamedPipeChild.ps1'
    $childCode = @'
param([string]$BridgePath,[switch]$MultilineCase,[string]$DialogFlow="",[string]$WizardPath="")
$ErrorActionPreference='Stop'
$env:MOE4ALL_DESKTOP_IO='1'
. $BridgePath
if(-not $env:MOE4ALL_DESKTOP_CONTROL_PIPE){throw 'Child did not receive a private control pipe'}
if($DialogFlow){
    $tokens=$null;$errors=$null
    $ast=[Management.Automation.Language.Parser]::ParseFile($WizardPath,[ref]$tokens,[ref]$errors)
    if($errors.Count){throw 'Wizard parse failed'}
    $names=@('Read-YesNo','Read-Choice','Read-TextValue','Read-BenchmarkUploadConsent','Read-SubmissionNextAction','Offer-BenchmarkLeaderboard')
    foreach($definition in $ast.FindAll({param($node) $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -in $names},$true)){. ([ScriptBlock]::Create($definition.Extent.Text))}
    function Save-BenchmarkUploadPreferences {}
    function Open-BenchmarkLeaderboard {[Console]::Out.WriteLine('LEADERBOARD_OPEN_REQUESTED');[Console]::Out.Flush()}
    $DesktopExecutionPlanPath='fixture';$launchMode='benchmark';$script:BenchmarkUploadDefault=$false;$script:SubmissionBrowserActionDefault='later';$script:SubmissionFailureActionDefault='later'
    Write-WizardRuntimeEvent @{kind='result';rows=@(@{context='6K';workload='Decode x1';first='150.1';middle='151.2';last='152.3';overall='151.2'})}
    try{
        if(Read-BenchmarkUploadConsent){
            $hasGit=Read-YesNo -Label '你有 GitHub 账号吗？/ Do you have a GitHub account?' -Default $true
            if($hasGit){$next=Read-SubmissionNextAction -Browser;[Console]::Out.WriteLine('WEB_ACTION='+$next);if($next -eq 'cloudflare'){$next=Read-SubmissionNextAction;[Console]::Out.WriteLine('RELAY_ACTION='+$next)}}
        }
    }catch [OperationCanceledException]{[Console]::Out.WriteLine('SUBMISSION_CANCELLED_REPORT_RETAINED')}
    Offer-BenchmarkLeaderboard
    [Console]::Out.WriteLine('DIALOG_FLOW_DONE');[Console]::Out.Flush();exit 0
}
if($MultilineCase){
    $first=Read-Host 'Path fixture'
    [Console]::Out.WriteLine('FIRST_PROMPT_REPLIED')
    $second=Read-Host 'Second consent fixture [y/N]'
    if($second.Trim().ToLowerInvariant() -ne 'n'){
        [Console]::Out.WriteLine('BUFFERED_SECOND_REPLY_REJECTED_BY_FIXTURE');exit 19
    }
    [Console]::Out.WriteLine('SECOND_PROMPT_EXPLICIT_NO');[Console]::Out.Flush();exit 0
}
[Console]::Out.WriteLine('PIPE_ACTIVE:'+$env:MOE4ALL_DESKTOP_CONTROL_PIPE)
$rows=@(for($i=0;$i -lt 24;$i++){
    @{context=([string]($i+1)+'K');workload=(('中文'*45)+' row'+$i);first='12.3';middle='45.6';last='56.7';overall='40.1'}
})
$packet=@{kind='result';rows=$rows}
$encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((ConvertTo-Json -InputObject $packet -Depth 8 -Compress)))
if($encoded.Length -le 120){throw 'Fixture must exceed the ConPTY screen width'}
Write-WizardRuntimeEvent $packet
# Trigger real ConPTY scrolling/rewrites while structured packets use the pipe.
for($i=0;$i -lt 65;$i++){[Console]::Out.WriteLine(('screen-row-'+$i+' '+('x'*145)))}
[Console]::Out.Flush()
$answer=Read-Host 'Upload simulated fixture [y/N]'
if($answer.Trim().ToLowerInvariant() -ne 'n'){[Console]::Out.WriteLine('WRONG_REPLY');exit 19}
[Console]::Out.WriteLine('NO_SELECTED_NO_UPLOAD')
[Console]::Out.Flush()
exit 0
'@
    [IO.File]::WriteAllText($childSource, $childCode, (New-Object Text.UTF8Encoding $true))
    $bridgeSource = Join-Path $PSScriptRoot 'wizard-runtime-bridge.ps1'
    $code = @'
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Pipes;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using MoE4All.Launcher;

internal static class RuntimeExperienceHarness
{
    const BindingFlags All = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    static readonly JavaScriptSerializer Json = new JavaScriptSerializer();
    static readonly List<Dictionary<string, object>> Results = new List<Dictionary<string, object>>();
    static readonly Assembly Runtime = typeof(RuntimeWindow).Assembly;
    static string evidence;
    static int checks, failures;
    [DllImport("user32.dll", EntryPoint = "SendMessageW")]
    static extern IntPtr SendMessage(IntPtr hwnd, int message, IntPtr wparam, IntPtr lparam);
    [DllImport("user32.dll")]
    static extern bool IsWindowVisible(IntPtr hwnd);
    [StructLayout(LayoutKind.Sequential)] struct ScrollPoint { public int X, Y; }
    [DllImport("user32.dll", EntryPoint = "SendMessageW")]
    static extern IntPtr SendPoint(IntPtr hwnd, int message, IntPtr unused, ref ScrollPoint point);

    static void Check(bool condition, string message)
    { checks++; if (!condition) throw new Exception(message); }
    static object Field(object target, string name)
    { var f = target.GetType().GetField(name, All); if (f == null) throw new MissingFieldException(target.GetType().FullName, name); return f.GetValue(target); }
    static object Call(object target, string name, params object[] args)
    {
        var method = target.GetType().GetMethod(name, All);
        if (method == null) throw new MissingMethodException(target.GetType().FullName, name);
        try { return method.Invoke(target, args); }
        catch (TargetInvocationException e) { throw e.InnerException ?? e; }
    }
    static void Case(string name, Action action)
    {
        var watch = Stopwatch.StartNew();
        var result = new Dictionary<string, object> { { "case", name } };
        try { action(); result["status"] = "passed"; Console.WriteLine("PASS " + name); }
        catch (Exception e) { failures++; result["status"] = "failed"; result["error"] = e.ToString(); Console.WriteLine("FAIL " + name + ": " + e.Message); }
        result["elapsedMs"] = watch.ElapsedMilliseconds; Results.Add(result);
        File.WriteAllText(Path.Combine(evidence, "results.json"), Json.Serialize(Results), new UTF8Encoding(true));
    }
    static void Until(Func<bool> condition, string name)
    {
        var watch = Stopwatch.StartNew();
        while (!condition()) { if (watch.ElapsedMilliseconds > 8000) throw new TimeoutException(name); Application.DoEvents(); Thread.Sleep(10); }
    }
    sealed class Transcript
    {
        internal readonly object Value, Decoder;
        internal Transcript()
        {
            Type type = Runtime.GetType("MoE4All.Launcher.RuntimeTranscript", true);
            Value = Activator.CreateInstance(type, true);
            Decoder = Activator.CreateInstance(type.GetNestedType("Decoder", BindingFlags.NonPublic), All, null, new object[] { Value, null }, null);
        }
        internal void Feed(string text) { Call(Decoder, "Feed", text); }
        internal string Text { get { return (string)Field(Call(Value, "Read", -1L), "Text"); } }
        internal string[] Drain() { return (string[])Call(Value, "DrainControls"); }
    }
    static string Frame(string json)
    { return "__MOE4ALL_UI__:" + Convert.ToBase64String(Encoding.UTF8.GetBytes(json)) + ":__END_UI__"; }
    static string Decode(string value) { return Encoding.UTF8.GetString(Convert.FromBase64String(value)); }

    static void ControlChannelCloseStress(){
        Type type=Runtime.GetType("MoE4All.Launcher.RuntimeControlChannel",true);
        for(int i=0;i<80;i++){
            var transcript=new Transcript();var channel=(IDisposable)Activator.CreateInstance(type,All,null,new object[]{transcript.Value},null);
            if(i%2==0)Thread.Sleep(2);channel.Dispose();Check(!((Thread)Field(channel,"worker")).IsAlive,"unconnected private pipe worker remained after close");
        }
        var settle=Stopwatch.StartNew();while(settle.ElapsedMilliseconds<150){Application.DoEvents();Thread.Sleep(5);}
    }    static void ControlReadBoundaries()
    {
        string packet = Json.Serialize(new Dictionary<string, object> { { "kind", "prompt" }, { "prompt", "\u7ee7\u7eed [Y/n]" }, { "context", "**Markdown**\n\u4e2d\u6587" } });
        string frame = Frame(packet);
        foreach (int split in new[] { 1, 8, 14, 19, frame.Length - 10, frame.Length - 2, frame.Length })
        {
            var t = new Transcript(); t.Feed("before\n");
            t.Feed(frame.Substring(0, split)); t.Feed(frame.Substring(split)); t.Feed("\r\n");
            string[] controls = t.Drain();
            Check(controls.Length == 1 && Decode(controls[0]) == packet, "cross-read frame lost at split " + split);
            Check(t.Text == "before\n", "completed control frame leaked at split " + split);
            Check(t.Drain().Length == 0, "control delivered twice");
            t.Feed("after\n"); Check(t.Text == "before\nafter\n", "normal log after control lost");
        }
        var single = new Transcript(); foreach (char c in frame + "\n") single.Feed(c.ToString());
        Check(single.Drain().Length == 1 && single.Text == "", "one-character reads lost a frame");
    }
    static void ControlSoftWrap()
    {
        string packet = Json.Serialize(new Dictionary<string, object> { { "kind", "prompt" }, { "prompt", "\u786e\u8ba4" }, { "context", new string('\u4e2d', 300) } });
        string frame = Frame(packet);
        foreach (bool cursorMove in new[] { false, true })
        {
            var t = new Transcript(); t.Feed("before\r\n"); int row = 3;
            for (int p = 0; p < frame.Length; p += 120)
            {
                int count = Math.Min(120, frame.Length - p);
                // Simulate ConPTY read boundaries independently of its 120-column wrap.
                int half = count / 2; t.Feed(frame.Substring(p, half)); t.Feed(frame.Substring(p + half, count - half));
                if (p + count < frame.Length) t.Feed(cursorMove ? "\x1b[" + (row++) + ";1H" : "\r\n");
            }
            t.Feed("\r\nafter\n"); var controls = t.Drain();
            Check(controls.Length == 1 && Decode(controls[0]) == packet, "soft-wrapped payload lost: cursorMove=" + cursorMove);
            Check(t.Text == "before\nafter\n", "soft-wrapped control appeared in log");
        }
    }
    static void InflightControlIsolation()
    {
        string frame = Frame("{\"kind\":\"prompt\",\"prompt\":\"fixture\"}");
        var t = new Transcript(); t.Feed("before\n"); t.Feed(frame.Substring(0, 24));
        Check(t.Text == "before\n", "Read() exposes an unfinished control prefix/base64 in the transcript: " + t.Text);
        t.Feed(frame.Substring(24)); t.Feed("\n"); Check(t.Drain().Length == 1, "pending frame did not drain");
    }
    static void RuntimeLogScrolling()
    {
        var transcript = new Transcript();
        using (var form = new Form { ClientSize = new Size(620, 260), ShowInTaskbar = false, StartPosition = FormStartPosition.Manual, Location = new Point(-8000, -8000) })
        using (var log = (WizardRichTextBox)Activator.CreateInstance(Runtime.GetType("MoE4All.Launcher.RuntimeLog", true), true))
        {
            form.Controls.Add(log); form.Show(); Application.DoEvents();
            var rows = new StringBuilder(); for (int i = 0; i < 240; i++) rows.Append("row ").Append(i).Append(" \u4e2d\u6587\n");
            transcript.Feed(rows.ToString()); Call(log, "RefreshFrom", transcript.Value); Application.DoEvents();
            Check(log.IsAtLatest && log.FirstVisibleLine > 100, "RuntimeLog did not follow initial output to bottom");
            SendMessage(log.Handle, 0x020a, new IntPtr(720 << 16), IntPtr.Zero); // actual WM_MOUSEWHEEL
            Application.DoEvents(); int first = log.FirstVisibleLine;
            Check(!log.IsAtLatest, "wheel up did not pause follow");
            var expected = new ScrollPoint(); SendPoint(log.Handle, 0x04dd, IntPtr.Zero, ref expected);
            for (int i = 0; i < 3; i++) { transcript.Feed("new output " + i + "\n"); Call(log, "RefreshFrom", transcript.Value); }
            Application.DoEvents(); var actual = new ScrollPoint(); SendPoint(log.Handle, 0x04dd, IntPtr.Zero, ref actual);
            Check(log.FirstVisibleLine == first && actual.Y == expected.Y, "new output moved a user-scrolled viewport");
            Call(log, "Latest"); Check(log.IsAtLatest, "Latest did not reach bottom");
            transcript.Feed("after Latest\n"); Call(log, "RefreshFrom", transcript.Value); Application.DoEvents();
            Check(log.IsAtLatest && log.Text.EndsWith("after Latest\n", StringComparison.Ordinal), "Latest did not restore ongoing follow");
            log.Visible = false;
            for (int i = 0; i < 3; i++)
            {
                transcript.Feed("hidden output " + i + "\n"); Call(log, "RefreshFrom", transcript.Value); Application.DoEvents();
                Check(!log.Visible && !IsWindowVisible(log.Handle), "hidden RuntimeLog regained native WS_VISIBLE during refresh");
            }
            File.WriteAllText(Path.Combine(evidence, "runtime-log.txt"), log.Text, new UTF8Encoding(true));
            form.Hide();
        }
    }

    static double Luminance(Color c){double[] x={c.R/255.0,c.G/255.0,c.B/255.0};for(int i=0;i<3;i++)x[i]=x[i]<=0.04045?x[i]/12.92:Math.Pow((x[i]+0.055)/1.055,2.4);return 0.2126*x[0]+0.7152*x[1]+0.0722*x[2];}
    static double Contrast(Color a,Color b){double x=Luminance(a),y=Luminance(b);return (Math.Max(x,y)+0.05)/(Math.Min(x,y)+0.05);}
    static void MarkdownFollowAndContrast(){
        foreach(string theme in new[]{"dark","light"}){
            WizardTheme.SetPreference(theme);var palette=WizardTheme.Current;Check(Contrast(palette.Text,palette.Surface)>=10,"body text contrast too low");Check(Contrast(palette.Muted,palette.SurfaceAlt)>=7,"status contrast too low");
            using(var form=new Form{ClientSize=new Size(650,340),ShowInTaskbar=false,StartPosition=FormStartPosition.Manual,Location=new Point(-8000,-8000)})using(var view=new MarkdownView{Dock=DockStyle.Fill}){
                bool follow=true;view.UserScrolled+=delegate{follow=view.IsAtLatest;};form.Controls.Add(view);form.Show();var content=new StringBuilder();
                for(int i=0;i<60;i++)content.Append("### Heading "+i+"\n\nLong wrapped text and **bold text** with Chinese \u4e2d\u6587\u6d4b\u8bd5.\n\n");
                view.SetMarkdown(content.ToString());view.ScrollToLatest();Application.DoEvents();
                foreach(int width in new[]{650,420,920,540}){
                    form.ClientSize=new Size(width,340);Application.DoEvents();Check(view.IsAtLatest,"resize lost pinned bottom: "+width);
                    for(int i=0;i<6;i++){content.Append("\n\nNew streamed paragraph **"+i+"** \u4e2d\u6587.\n");view.SetMarkdown(content.ToString());if(follow&&view.SelectionLength==0)view.ScrollToLatest();Application.DoEvents();Check(view.IsAtLatest,"stream update stopped following");}
                }
                view.ScrollToLatest();SendMessage(view.Handle,0x020a,new IntPtr(720<<16),IntPtr.Zero);Application.DoEvents();int first=view.FirstVisibleLine;Check(!follow&&!view.IsAtLatest,"manual scroll did not pause deferred follow");
                content.Append("\n\nA later paragraph must not steal the viewport.");view.SetMarkdown(content.ToString());if(follow)view.ScrollToLatest();Application.DoEvents();Check(view.FirstVisibleLine==first,"new Markdown pulled reading viewport down");
                follow=true;view.ScrollToLatest();Application.DoEvents();Check(view.IsAtLatest,"Latest did not restore Markdown follow");form.Hide();
            }
        }
    }
    // Minimal HTTP/1.1 fixture. No URL reservation, proxy, DNS or remote endpoint.
    sealed class LocalStub : IDisposable
    {
        readonly TcpListener listener = new TcpListener(IPAddress.Loopback, 0);
        readonly Thread worker;
        readonly object gate = new object();
        readonly List<Dictionary<string, string>> requests = new List<Dictionary<string, string>>();
        internal readonly ManualResetEvent ReleaseCancel = new ManualResetEvent(false);
        volatile bool stopping;
        internal string Fault = "";
        internal readonly string Address;
        readonly bool unauthorizedModels, holdPostHeaders;
        int posts;
        internal LocalStub() : this(false, false) { }
        internal LocalStub(bool unauthorizedModels, bool holdPostHeaders)
        {
            this.unauthorizedModels = unauthorizedModels; this.holdPostHeaders = holdPostHeaders;
            listener.Start(); Address = "http://127.0.0.1:" + ((IPEndPoint)listener.LocalEndpoint).Port + "/v1";
            worker = new Thread(Serve) { IsBackground = true, Name = "Runtime experience loopback fixture" }; worker.Start();
        }
        internal Dictionary<string, string>[] Requests { get { lock (gate) return requests.ToArray(); } }
        static void Write(NetworkStream stream, string text)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            // Deliberately split UTF-8 codepoints across transport writes.
            for (int p = 0; p < bytes.Length; p += 7) stream.Write(bytes, p, Math.Min(7, bytes.Length - p));
            stream.Flush();
        }
        static string Event(string key, string value)
        { return "data: " + new JavaScriptSerializer().Serialize(new { choices = new[] { new { delta = new Dictionary<string, object> { { key, value } }, finish_reason = (string)null } } }) + "\n\n"; }
        static void Headers(NetworkStream stream, string type, int status)
        { Write(stream, "HTTP/1.1 " + status + (status == 200 ? " OK" : " Fixture Error") + "\r\nContent-Type: " + type + "; charset=utf-8\r\nConnection: close\r\n\r\n"); }
        void Serve()
        {
            while (!stopping)
            {
                try
                {
                    using (TcpClient client = listener.AcceptTcpClient())
                    using (NetworkStream stream = client.GetStream())
                    {
                        client.NoDelay = true; stream.ReadTimeout = 5000; stream.WriteTimeout = 5000;
                        var bytes = new List<byte>(); int b;
                        while ((b = stream.ReadByte()) >= 0)
                        {
                            bytes.Add((byte)b); int n = bytes.Count;
                            if (n >= 4 && bytes[n - 4] == 13 && bytes[n - 3] == 10 && bytes[n - 2] == 13 && bytes[n - 1] == 10) break;
                            if (n > 32768) throw new IOException("Fixture request headers too large");
                        }
                        if (bytes.Count == 0) continue;
                        string header = Encoding.ASCII.GetString(bytes.ToArray()); string[] lines = header.Split(new[] { "\r\n" }, StringSplitOptions.None);
                        string[] first = lines[0].Split(' '); int length = 0;
                        foreach (string line in lines)
                        {
                            if (line.StartsWith("Content-Length:", StringComparison.OrdinalIgnoreCase)) length = Int32.Parse(line.Substring(15).Trim());
                            if (line.Equals("Expect: 100-continue", StringComparison.OrdinalIgnoreCase)) Write(stream, "HTTP/1.1 100 Continue\r\n\r\n");
                        }
                        if (length < 0 || length > 262144) throw new IOException("Fixture request body too large");
                        byte[] body = new byte[length]; int read = 0;
                        while (read < length) { int count = stream.Read(body, read, length - read); if (count == 0) throw new EndOfStreamException(); read += count; }
                        lock (gate) requests.Add(new Dictionary<string, string> { { "method", first[0] }, { "path", first[1] }, { "body", Encoding.UTF8.GetString(body) } });
                        if (first[0] == "GET" && first[1] == "/v1/models")
                        {
                            Headers(stream, "application/json", unauthorizedModels ? 401 : 200);
                            Write(stream, unauthorizedModels ? "{\"error\":{\"message\":\"fixture unauthorized\"}}" : "{\"data\":[{\"id\":\"fixture-\u4e2d\u6587\"}]}");
                        }
                        else if (first[0] == "POST" && first[1] == "/v1/chat/completions")
                        {
                            posts++;
                            if (holdPostHeaders) ReleaseCancel.WaitOne(5000);
                            if (posts == 4) { Headers(stream, "application/json", 500); Write(stream, "{\"error\":{\"message\":\"fixture failure\"}}"); continue; }
                            Headers(stream, "text/event-stream", 200);
                            Write(stream, ": keepalive\n\n" + Event("role", "assistant"));
                            if (posts == 3)
                            { Write(stream, Event("content", "cancel-partial")); ReleaseCancel.WaitOne(5000); continue; }
                            Write(stream, Event("reasoning_content", posts == 1 ? "\u5148\u601d\u8003\n" : "\u7b2c\u4e8c\u8f6e\u601d\u8003"));
                            Write(stream, Event("content", posts == 1 ? "\u4f60\u597d\n\n## \u6807\u9898\n" : "```csharp\n    "));
                            Write(stream, Event("content", posts == 1 ? "- **\u7b2c\u4e00\u9879**\n- \u7b2c\u4e8c\u9879" : "Console.WriteLine(\"\u4e2d\u6587\");\n```"));
                            Write(stream, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"completion_tokens\":12},\"timings\":{\"prompt_n\":64,\"prompt_per_second\":2400.5,\"predicted_n\":12,\"predicted_per_second\":151.25,\"cached_n\":128,\"context_n\":204}}\n\ndata: [DONE]\n\n");
                        }
                        else { Headers(stream, "application/json", 404); Write(stream, "{}"); }
                    }
                }
                catch (Exception e) { if (!stopping) Fault = e.ToString(); }
            }
        }
        public void Dispose() { stopping = true; ReleaseCancel.Set(); listener.Stop(); worker.Join(6000); ReleaseCancel.Dispose(); }
    }
    static Dictionary<string, object>[] History(ChatClient client)
    { return ((List<Dictionary<string, object>>)Field(client, "history")).ToArray(); }
    static void CancelBeforeCreate()
    {
        using (var stub = new LocalStub(false, true))
        using (var client = new ChatClient(stub.Address, ""))
        {
            Until(delegate { return client.Read().Ready; }, "early-cancel readiness");
            object gate = Field(client, "gate");
            // Deterministically hold the worker before history serialization and
            // Create, reproducing Cancel immediately after Ask on the UI thread.
            Monitor.Enter(gate);
            try
            {
                Check(client.Ask("CANCEL_BEFORE_CREATE"), "early-cancel ask rejected");
                client.Cancel();
            }
            finally { Monitor.Exit(gate); }
            try
            {
                Until(delegate { return !client.Read().Busy; }, "cancel before Create did not leave Busy");
                foreach (var request in stub.Requests)
                    Check(request["method"] != "POST", "cancel before request registration still sent a POST");
                Check(History(client).Length == 0 && client.Read().Answer == "", "early cancellation polluted committed history or answer");
                Check(stub.Fault == "", "early-cancel stub failed: " + stub.Fault);
                File.WriteAllText(Path.Combine(evidence, "early-cancel-http.json"), Json.Serialize(stub.Requests), new UTF8Encoding(true));
            }
            finally { stub.ReleaseCancel.Set(); }
        }
    }
    static void ReadinessUnauthorized()
    {
        using (var stub = new LocalStub(true, false))
        using (var client = new ChatClient(stub.Address, ""))
        {
            Until(delegate { return client.Read().Fault.Length > 0; }, "401 readiness remained silent instead of exposing Fault");
            var state = client.Read();
            Check(!state.Ready && !state.Busy && state.Fault.Length > 0, "401 marked chat ready or hid its error");
            Check(!client.Ask("MUST_NOT_SEND") && History(client).Length == 0, "401 readiness accepted a chat turn");
            Check(stub.Requests.Length > 0 && stub.Requests[0]["path"] == "/v1/models", "401 readiness did not probe models");
            foreach (var request in stub.Requests) Check(request["method"] == "GET", "401 readiness unexpectedly issued POST");
            Check(stub.Fault == "", "401 readiness stub failed: " + stub.Fault);
            File.WriteAllText(Path.Combine(evidence, "readiness-401.json"), Json.Serialize(state), new UTF8Encoding(true));
        }
    }
    static void ChatHttp()
    {
        using (var stub = new LocalStub())
        using (var client = new ChatClient(stub.Address, ""))
        {
            Until(delegate { return client.Read().Ready; }, "GET /v1/models readiness");
            Check(client.Read().Model == "fixture-\u4e2d\u6587", "model ID changed");
            Check(stub.Requests.Length == 1 && stub.Requests[0]["method"] == "GET" && stub.Requests[0]["path"] == "/v1/models", "wrong model readiness request");
            string question1 = "\u4f60\u597d\n\u8bf7\u7528 **Markdown** \u56de\u7b54";
            string answer1 = "\u4f60\u597d\n\n## \u6807\u9898\n- **\u7b2c\u4e00\u9879**\n- \u7b2c\u4e8c\u9879";
            Check(client.Ask(question1), "first ask rejected"); Until(delegate { return !client.Read().Busy; }, "first SSE completion");
            Check(client.Read().Answer == answer1 && client.Read().Reasoning == "\u5148\u601d\u8003\n" && client.Read().Fault == "", "first SSE Unicode/multiline/Markdown/reasoning changed");
            Check(client.Read().HasTimings&&client.Read().GeneratedTokens==12&&client.Read().DecodeTokensPerSecond==151.25&&client.Read().PrefillTokensPerSecond==2400.5&&client.Read().CachedTokens==128,"authoritative SSE metrics lost or counted from chunks");
            Check(History(client).Length == 2, "first round history incomplete");
            string question2 = "\u7b2c\u4e8c\u8f6e\n\u8bf7\u7ee7\u7eed";
            Check(client.Ask(question2), "second ask rejected"); Until(delegate { return !client.Read().Busy; }, "second SSE completion");
            string answer2 = "```csharp\n    Console.WriteLine(\"\u4e2d\u6587\");\n```";
            Check(client.Read().Answer == answer2 && client.Read().Reasoning == "\u7b2c\u4e8c\u8f6e\u601d\u8003", "second SSE/code indentation changed");
            var posts = new List<Dictionary<string, string>>(); foreach (var request in stub.Requests) if (request["method"] == "POST") posts.Add(request);
            Check(posts.Count == 2 && posts[0]["path"] == "/v1/chat/completions", "wrong chat path/count");
            var request2 = (Dictionary<string, object>)Json.DeserializeObject(posts[1]["body"]);
            var messages = (object[])request2["messages"];
            Check(messages.Length == 3 && (bool)request2["stream"] && Convert.ToString(request2["model"]) == "fixture-\u4e2d\u6587", "second request lacks stream/model/full history");
            string[] roles = { "user", "assistant", "user" }; string[] content = { question1, answer1, question2 };
            for (int i = 0; i < messages.Length; i++) { var m = (Dictionary<string, object>)messages[i]; Check(Convert.ToString(m["role"]) == roles[i] && Convert.ToString(m["content"]) == content[i], "history content/order changed at " + i); }
            Check(History(client).Length == 4 && Convert.ToString(History(client)[3]["content"]) == answer2, "second assistant missing from history");
            string committed = Json.Serialize(History(client));
            Check(client.Ask("CANCELLED_QUESTION"), "cancel ask rejected");
            Until(delegate { return client.Read().Answer == "cancel-partial"; }, "partial response before cancel");
            Check(!client.Ask("REJECTED_BUSY") && !client.Reset(), "busy request/reset unexpectedly accepted"); Check(!client.Read().HasTimings&&client.Read().GeneratedTokens==0,"new reply retained previous metrics");
            client.Cancel(); stub.ReleaseCancel.Set(); Until(delegate { return !client.Read().Busy; }, "cancel completion");
            Check(Json.Serialize(History(client)) == committed, "cancelled question/partial answer polluted history");
            Check(client.Ask("FAILED_QUESTION"), "error ask rejected"); Until(delegate { return !client.Read().Busy; }, "HTTP error completion");
            Check(client.Read().Fault.Length > 0 && Json.Serialize(History(client)) == committed, "HTTP failure polluted history or lost error");
            foreach (var request in stub.Requests)
                if (request["body"].Contains("FAILED_QUESTION")) Check(!request["body"].Contains("CANCELLED_QUESTION") && !request["body"].Contains("cancel-partial"), "cancelled turn leaked into next wire request");
            Check(stub.Fault == "", "local HTTP fixture fault: " + stub.Fault);
            File.WriteAllText(Path.Combine(evidence, "http-requests.json"), Json.Serialize(stub.Requests), new UTF8Encoding(true));
            File.WriteAllText(Path.Combine(evidence, "chat-history.json"), committed, new UTF8Encoding(true));
            Check(client.Reset() && History(client).Length == 0, "new conversation did not clear history");
        }
    }
    static Dictionary<string, object> ResultRow()
    {
        return new Dictionary<string, object> { { "context", "32768" }, { "workload", "\u4e2d\u6587 long context" }, { "first", "12.3" }, { "middle", "45.6" }, { "last", "56.7" }, { "overall", "40.1" } };
    }
    static void Labels(Control root, List<string> labels)
    { foreach (Control child in root.Controls) { if (child is Label) labels.Add(child.Text); Labels(child, labels); } }
    static void BenchmarkFields()
    {
        using (var view = new BenchmarkResultsView { Width = 900, Height = 350 })
        {
            view.LoadRows(new object[] { ResultRow(), new Dictionary<string, object> { { "context", "8192" } } });
            var grid=(DataGridView)Find(view,"BenchmarkResultGrid");
            var labels = new List<string>();foreach(DataGridViewColumn column in grid.Columns)labels.Add(column.HeaderText);foreach(DataGridViewRow row in grid.Rows)foreach(DataGridViewCell cell in row.Cells)labels.Add(Convert.ToString(cell.Value));
            foreach (string text in new[] { "Context", "Workload", "First", "Middle", "Last", "Overall", "32768", "\u4e2d\u6587 long context", "12.3", "45.6", "56.7", "40.1", "8192", "-" })
                Check(labels.Contains(text), "benchmark field missing: " + text);
            view.LoadRows(new object[] { ResultRow() });
            Check(grid.Rows.Count==1&&!view.ExportTsv().Contains("8192"), "benchmark reload retained stale rows");
            Check(grid.ReadOnly&&grid.Columns.Count==6&&!grid.RowHeadersVisible&&grid.ScrollBars==ScrollBars.None,"benchmark is not a compact read-only grid");
            grid.ClearSelection();grid.Rows[0].Cells[2].Selected=true;Check(view.SelectedTsv().Contains("12.3"),"selected cells cannot be copied");grid.ClearSelection();
            using (var form = new Form { ClientSize = new Size(1000, 350), ShowInTaskbar = false, StartPosition = FormStartPosition.Manual, Location = new Point(-8000, -8000) })
            {
                view.Dock = DockStyle.Fill; form.Controls.Add(view); form.Show();
                foreach (int width in new[] { 1000, 760, 640, 1200 })
                {
                    form.ClientSize = new Size(width, 350); form.PerformLayout(); view.PerformLayout(); Application.DoEvents();
                    Check(grid.Columns.GetColumnsWidth(DataGridViewElementStates.Visible)<=grid.ClientSize.Width,"benchmark columns escaped viewport after resize "+width);
                    Check(grid.ColumnHeadersVisible&&grid.Rows[0].Height<=40,"benchmark header or row density regressed");
                }
                form.Hide();
            }
        }
    }
    static bool LocalVisible(Control control)
    {
        // Visible on an unshown Form is false for every child. Inspect the local
        // visibility bit through the actual ancestor chain, excluding the root.
        var state = typeof(Control).GetMethod("GetState", BindingFlags.Instance | BindingFlags.NonPublic);
        for (Control c = control; c != null && !(c is Form); c = c.Parent)
            if (!(bool)state.Invoke(c, new object[] { 2 })) return false;
        return true;
    }
    static Control Find(Control root, string name)
    { Control[] items = root.Controls.Find(name, true); if (items.Length != 1) throw new Exception("Expected one " + name); return items[0]; }
    static void Layouts(string fixture)
    {
        using (var stub = new LocalStub())
        {
            foreach (string mode in new[] { "chat", "server", "benchmark" })
            using (var window = new RuntimeWindow(fixture, new[] { "args" }, Path.GetDirectoryName(fixture), mode, "fixture-model.gguf", stub.Address, true))
            {
                // Never Show(): that would launch a process via the production Shown handler.
                IntPtr unused = window.Handle; window.PerformLayout();
                var entry = Find(window, "RuntimeInput"); var layout = (TableLayoutPanel)Find(window, "RuntimeContent");
                Check(LocalVisible(entry) == (mode == "chat"), "default RuntimeInput visibility wrong for " + mode);
                Check((mode == "chat" ? layout.RowStyles[5].Height > 0 : layout.RowStyles[5].Height == 0), "input row not collapsed for " + mode);
                Check(LocalVisible(Find(window, "RuntimeEndpoint")) == (mode == "server"), "endpoint row visibility wrong for " + mode);
                if (mode == "chat")
                {
                    var client = (ChatClient)Field(window, "chat");
                    Until(delegate { return client.Read().Ready; }, "chat workspace readiness"); Call(window, "RefreshChat");
                    Check(LocalVisible(Find(window, "ChatConversation")) && !LocalVisible(Find(window, "RuntimeOutput")), "chat workspace not activated");
                    Check(!LocalVisible(Find(window, "RuntimeRawKeys")), "HTTP chat exposes raw terminal keys");
                    Check(client.Ask("UI metrics fixture"),"UI metric ask rejected");Until(delegate{return !client.Read().Busy;},"UI metric completion");Call(window,"RefreshChat");Check(((Label)Field(window,"retention")).Text.Contains("Decode 151."),"chat window omitted current reply speed");
                }
                else if (mode == "benchmark")
                {
                    object session = Field(window, "session"); object transcript = Field(session, "Transcript");
                    Type decoder = Runtime.GetType("MoE4All.Launcher.RuntimeTranscript+Decoder", true);
                    object parser = Activator.CreateInstance(decoder, All, null, new object[] { transcript, null }, null);
                    string packet = Json.Serialize(new Dictionary<string, object> { { "kind", "result" }, { "rows", new[] { ResultRow() } } });
                    Call(parser, "Feed", Frame(packet) + "\n"); Call(window, "ReadControls");
                    Check(LocalVisible(Find(window, "BenchmarkResults")) && !LocalVisible(Find(window, "RuntimeOutput")), "result bridge did not switch benchmark page");
                    Check(((BenchmarkResultsView)Find(window,"BenchmarkResults")).ExportTsv().Contains("56.7"),"result bridge lost fields");
                    Check(!LocalVisible(entry), "result packet revealed terminal input");
                }
                else {
                    Check(Field(window,"chat")==null&&Field(window,"results")==null,"server uses the wrong task workspace");
                    object session=Field(window,"session"),transcript=Field(session,"Transcript");Type decoder=Runtime.GetType("MoE4All.Launcher.RuntimeTranscript+Decoder",true);object parser=Activator.CreateInstance(decoder,All,null,new object[]{transcript,null},null);
                    string border="+---------------+---------------------------------+-----------+---------------+-----------------------+";
                    string header=String.Format("| {0,-13} | {1,-31} | {2,-9} | {3,13} | {4,21} |\n","Slot","Work","State","Speed (now)","Context");
                    string row=String.Format("| {0,-13} | {1,-31} | {2,-9} | {3,13} | {4,21} |\n","KV 1/1","#1 fixture","Decode","151.2 tok/s","6,144 / 163,840");
                    Call(parser,"Feed","INFR live resources\n"+border+"\n"+header+border+"\n"+row+border+"\nTotal: prefill 0.0 tok/s | decode 151.2 tok/s | active 1 | queued 0 | done 0 | failed 0\n");Call(window,"TickRuntime");
                    var api=(ApiStatusView)Field(window,"apiStatus");Check(api.HasFrame&&api.RowCount==1&&LocalVisible(api)&&!LocalVisible(Find(window,"RuntimeOutput")),"API runtime did not activate per-slot status");Check(api.ExportTsv().Contains("151.2 tok/s"),"API runtime dropped per-slot speed");
                    Call(window,"ShowWorkspace",true);Check(LocalVisible(Find(window,"RuntimeOutput"))&&!LocalVisible(api),"API logs toggle lost state");Call(window,"ShowWorkspace",false);Check(LocalVisible(api)&&api.RowCount==1,"API state disappeared after log toggle");
                }
                Check(!(bool)Field(Call(Field(window, "session"), "ReadState"), "Started"), "layout test unexpectedly started a child process");
            }
            Check(stub.Fault == "", "layout HTTP stub failed: " + stub.Fault);
        }
    }
    sealed class DialogAnswers:IDisposable {
        readonly System.Windows.Forms.Timer timer=new System.Windows.Forms.Timer{Interval=25};readonly HashSet<IntPtr> seen=new HashSet<IntPtr>();readonly RuntimeWindow owner;readonly Action<Form,int> answer;internal int Count;internal string Error="";
        internal DialogAnswers(RuntimeWindow owner,Action<Form,int> answer){this.owner=owner;this.answer=answer;timer.Tick+=delegate{Form d=null;foreach(Form f in Application.OpenForms)if(f.Name=="RuntimeSubmissionDialog"&&f.Owner==owner){d=f;break;}if(d==null||!seen.Add(d.Handle))return;Count++;try{answer(d,Count);}catch(Exception e){Error=e.ToString();d.Close();}};timer.Start();}
        public void Dispose(){timer.Stop();timer.Dispose();}
    }
    static void NamedPipeProcess(string child,string bridge){
        string oldPipe=Environment.GetEnvironmentVariable("MOE4ALL_DESKTOP_CONTROL_PIPE");Environment.SetEnvironmentVariable("MOE4ALL_DESKTOP_CONTROL_PIPE","unchanged-parent-fixture");
        string powershell=Path.Combine(Environment.SystemDirectory,@"WindowsPowerShell\v1.0\powershell.exe");
        try{using(var window=new RuntimeWindow(powershell,new[]{"-NoLogo","-NoProfile","-STA","-File",child,"-BridgePath",bridge},Path.GetDirectoryName(child),"benchmark","synthetic-pipe-fixture","",true)){
            object session=Field(window,"session");string name="";window.ShowInTaskbar=false;window.StartPosition=FormStartPosition.Manual;window.Location=new Point(-8000,-8000);
            using(var dialogs=new DialogAnswers(window,delegate(Form dialog,int number){
                Check(number==1,"duplicate upload dialog");object state=Call(session,"ReadState");Check((bool)Field(state,"Started")&&(bool)Field(state,"Pty"),"private pipe did not use ConPTY");
                object channel=Field(session,"control");name=Convert.ToString(Field(channel,"Name"));Check(name!="unchanged-parent-fixture"&&Environment.GetEnvironmentVariable("MOE4ALL_DESKTOP_CONTROL_PIPE")=="unchanged-parent-fixture","pipe environment leaked");
                var pipe=(NamedPipeServerStream)Field(channel,"pipe");var acl=pipe.GetAccessControl();Check(acl.AreAccessRulesProtected,"pipe ACL inherited broader rules");var sid=WindowsIdentity.GetCurrent().User;int allows=0;
                foreach(PipeAccessRule rule in acl.GetAccessRules(true,true,typeof(SecurityIdentifier)))if(rule.AccessControlType==AccessControlType.Allow){allows++;Check(rule.IdentityReference.Equals(sid),"pipe grants another user");}Check(allows>0,"missing current user ACL");
                var rows=(List<string[]>)Field(Field(window,"results"),"rows");Check(rows.Count==24,"long result lost rows");var workload=new StringBuilder();for(int i=0;i<45;i++)workload.Append("\u4e2d\u6587");
                for(int i=0;i<rows.Count;i++)Check(rows[i][0]==(i+1)+"K"&&rows[i][1]==workload+" row"+i&&rows[i][5]=="40.1","Chinese row corrupted");
                Check(Find(window,"BenchmarkResults").Visible&&!Find(window,"RuntimeInput").Visible&&!Find(window,"RuntimePrompt").Visible,"inline command prompt over result table");
                Check(dialog.Controls.Find("SubmissionInput",true).Length==0,"yes/no dialog exposes a text editor");
                using(var image=new Bitmap(dialog.Width,dialog.Height)){dialog.DrawToBitmap(image,new Rectangle(0,0,image.Width,image.Height));image.Save(Path.Combine(evidence,"namedpipe-result-prompt.png"));}
                ((Button)Find(dialog,"SubmissionNo")).PerformClick();
            }))try{
                window.Show();Until(delegate{return window.TaskCompleted;},"PowerShell exits after modal No");Call(window,"TickRuntime");string log=Find(window,"RuntimeOutput").Text;
                Check(dialogs.Error.Length==0,dialogs.Error);Check(dialogs.Count==1,"missing native consent dialog");Check(window.ExitCode==0&&log.Contains("NO_SELECTED_NO_UPLOAD"),"No did not reach PowerShell");
                Check(log.Contains("PIPE_ACTIVE:"+name)&&!log.Contains("__MOE4ALL_UI__:")&&!log.Contains("__END_UI__"),"protocol leaked into logs");Check(!Find(window,"RuntimeInput").Visible,"input remained after modal reply");
                File.WriteAllText(Path.Combine(evidence,"namedpipe-process.log"),log,new UTF8Encoding(true));
            }finally{if(!window.TaskCompleted){Call(session,"Force");Until(delegate{return window.TaskCompleted;},"fixture cleanup");}window.Hide();}
        }}finally{Environment.SetEnvironmentVariable("MOE4ALL_DESKTOP_CONTROL_PIPE",oldPipe);}
    }
    static void PromptMultilineProcess(string child,string bridge){
        string powershell=Path.Combine(Environment.SystemDirectory,@"WindowsPowerShell\v1.0\powershell.exe");
        using(var window=new RuntimeWindow(powershell,new[]{"-NoLogo","-NoProfile","-STA","-File",child,"-BridgePath",bridge,"-MultilineCase"},Path.GetDirectoryName(child),"benchmark","synthetic-prompt-fixture","",true)){
            object session=Field(window,"session");window.ShowInTaskbar=false;window.StartPosition=FormStartPosition.Manual;window.Location=new Point(-8000,-8000);
            using(var dialogs=new DialogAnswers(window,delegate(Form dialog,int number){
                if(number==1){var input=(TextBox)Find(dialog,"SubmissionInput");Check(!input.Multiline,"prompt editor is multiline");input.Multiline=true;input.Text="safe-path\r\ny";((Button)Find(dialog,"SubmissionContinue")).PerformClick();Check(!dialog.IsDisposed&&dialog.Visible,"multiline pre-answered next question");input.Multiline=false;input.Text="safe-path";((Button)Find(dialog,"SubmissionContinue")).PerformClick();}
                else{Check(number==2,"extra prompt from pasted newline");Check(Find(dialog,"SubmissionQuestion").Text.Contains("Second consent"),"second question changed");Check(dialog.Controls.Find("SubmissionInput",true).Length==0,"consent has input editor");((Button)Find(dialog,"SubmissionNo")).PerformClick();}
            }))try{
                window.Show();Until(delegate{return window.TaskCompleted;},"two explicit modal replies");Call(window,"TickRuntime");string log=Find(window,"RuntimeOutput").Text;
                Check(dialogs.Error.Length==0,dialogs.Error);Check(dialogs.Count==2,"prompts combined or omitted");Check(window.ExitCode==0&&log.Contains("SECOND_PROMPT_EXPLICIT_NO")&&!log.Contains("BUFFERED_SECOND_REPLY_REJECTED_BY_FIXTURE"),"buffered text answered consent");
                File.WriteAllText(Path.Combine(evidence,"prompt-multiline-process.log"),log,new UTF8Encoding(true));
            }finally{if(!window.TaskCompleted){Call(session,"Force");Until(delegate{return window.TaskCompleted;},"multiline cleanup");}window.Hide();}
        }
    }
    static void SubmissionFlow(string child,string bridge,string wizard){
        string powershell=Path.Combine(Environment.SystemDirectory,@"WindowsPowerShell\v1.0\powershell.exe");
        foreach(bool english in new[]{false,true})foreach(string flow in new[]{"no","github","cloudflare","cancel"})using(var window=new RuntimeWindow(powershell,new[]{"-NoLogo","-NoProfile","-STA","-File",child,"-BridgePath",bridge,"-DialogFlow",flow,"-WizardPath",wizard},Path.GetDirectoryName(child),"benchmark","submission flow fixture","",english)){
            object session=Field(window,"session");window.ShowInTaskbar=false;window.StartPosition=FormStartPosition.Manual;window.Location=new Point(-8000,-8000);int choice=0;bool sawLeaderboard=false;
            using(var dialogs=new DialogAnswers(window,delegate(Form dialog,int number){
                string question=Find(dialog,"SubmissionQuestion").Text;
                Check(!Find(window,"RuntimeInput").Visible&&!Find(window,"RuntimePrompt").Visible,"command-line editor exposed in modal flow");
                Check(Find(window,"BenchmarkResults").Visible,"result table hidden while asking consent");
                bool leaderboard=question.IndexOf("leaderboard",StringComparison.OrdinalIgnoreCase)>=0||question.Contains("\u6392\u884c\u699c");
                if(leaderboard){sawLeaderboard=true;Check(dialog.Text.Contains(english?"Leaderboard":"\u6392\u884c\u699c"),"leaderboard uses upload title");((Button)Find(dialog,flow=="github"?"SubmissionYes":"SubmissionNo")).PerformClick();return;}
                if(number==1){Check(dialog.Controls.Find("SubmissionInput",true).Length==0,"consent has a text entry");Check(Find(dialog,"SubmissionYes").Text==(english?"Submit results":"\u63d0\u4ea4\u7ed3\u679c"),"consent lacks action label");using(var image=new Bitmap(dialog.Width,dialog.Height)){dialog.DrawToBitmap(image,new Rectangle(0,0,image.Width,image.Height));image.Save(Path.Combine(evidence,"submit-consent-"+(english?"en":"zh")+".png"));}((Button)Find(dialog,flow=="no"?"SubmissionNo":"SubmissionYes")).PerformClick();return;}
                if(number==2){((Button)Find(dialog,"SubmissionYes")).PerformClick();return;}
                choice++;Check(dialog.Controls.Find("SubmissionInput",true).Length==0,"choice requires numeric input");
                if(flow=="cancel"){((Button)Find(dialog,"SubmissionCancel")).PerformClick();return;}
                string key=flow=="github"?"1":"3";Check(dialog.Controls.Find("SubmissionChoice_"+key,true).Length==1,"route option missing");
                using(var image=new Bitmap(dialog.Width,dialog.Height)){dialog.DrawToBitmap(image,new Rectangle(0,0,image.Width,image.Height));image.Save(Path.Combine(evidence,"submit-choices-"+flow+"-"+(english?"en":"zh")+"-"+choice+".png"));}
                ((Button)Find(dialog,"SubmissionChoice_"+key)).PerformClick();
            }))try{
                window.Show();Until(delegate{return window.TaskCompleted;},"native submission flow "+flow);Call(window,"TickRuntime");string log=Find(window,"RuntimeOutput").Text;
                Check(dialogs.Error.Length==0,dialogs.Error);Check(window.ExitCode==0&&log.Contains("DIALOG_FLOW_DONE"),"submission changed benchmark exit code: "+log);Check(sawLeaderboard,"no final leaderboard invitation");
                Check(log.Contains("LEADERBOARD_OPEN_REQUESTED")== (flow=="github"),"leaderboard browser decision ignored");
                if(flow=="github")Check(log.Contains("WEB_ACTION=sent"),"web success choice lost");if(flow=="cloudflare")Check(log.Contains("WEB_ACTION=cloudflare")&&log.Contains("RELAY_ACTION=later"),"fallback choice lost");if(flow=="cancel")Check(log.Contains("SUBMISSION_CANCELLED_REPORT_RETAINED"),"cancel did not retain report");
                Check(dialogs.Count==(flow=="no"?2:flow=="cloudflare"?5:4),"duplicate or missing dialog");File.WriteAllText(Path.Combine(evidence,"submission-"+flow+"-"+(english?"en":"zh")+".log"),log,new UTF8Encoding(true));
            }finally{if(!window.TaskCompleted){Call(session,"Force");Until(delegate{return window.TaskCompleted;},"submission fixture cleanup");}window.Hide();}
        }
    }
    [STAThread]
    static int Main(string[] args)
    {
        evidence = args[1]; Application.EnableVisualStyles();
        Case("RuntimeLog wheel / preserve viewport / Latest follow", RuntimeLogScrolling);
        Case("Streaming Markdown follows after layout and text contrast", MarkdownFollowAndContrast);
        Case("Control frames across reads", ControlReadBoundaries);
        Case("Private control-pipe close before client connect", ControlChannelCloseStress);
        Case("ConPTY soft wraps and cursor moves", ControlSoftWrap);
        Case("In-flight control frame isolation", InflightControlIsolation);
        Case("Local models + two SSE rounds + cancel/error history", ChatHttp);
        Case("Cancellation before Create issues no POST", CancelBeforeCreate);
        Case("401 readiness exposes Fault and rejects Ask", ReadinessUnauthorized);
        Case("Benchmark result fields and reload", BenchmarkFields);
        Case("Chat / server / benchmark layouts and result bridge", delegate { Layouts(args[0]); });
        Case("Real ConPTY PowerShell + private pipe + long Chinese result + UI No", delegate { NamedPipeProcess(args[2], args[3]); });
        Case("Real Read-Host multiline cannot pre-answer second prompt", delegate { PromptMultilineProcess(args[2], args[3]); });
        Case("Native submission consent, choices, cancel and leaderboard in both languages", delegate { SubmissionFlow(args[2], args[3], args[4]); });
        Console.WriteLine("Runtime experience: " + checks + " checks; " + (Results.Count - failures) + "/" + Results.Count + " groups passed; simulated only.");
        return failures == 0 ? 0 : 1;
    }
}
'@
    [IO.File]::WriteAllText($source, $code, (New-Object Text.UTF8Encoding $true))
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    if (-not (Test-Path -LiteralPath $compiler)) { $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe' }
    $compilerArgs = @('/nologo', '/langversion:5', '/target:exe', '/optimize+', '/utf8output',
        '/r:System.dll', '/r:System.Core.dll', '/r:System.Windows.Forms.dll', '/r:System.Drawing.dll',
        '/r:System.Web.Extensions.dll', '/r:Accessibility.dll', ('/r:' + $assembly), ('/out:' + $exe), $source)
    $manifest.compiler = $compiler; $manifest.compilerArguments = $compilerArgs
    $manifest.commit = (& git -C $repoRoot rev-parse HEAD | Out-String).Trim()
    $manifest.dirty = @(& git -C $repoRoot status --porcelain)
    $manifest.inputs = @((Get-TestIdentity $source), (Get-TestIdentity $childSource), (Get-TestIdentity $PSCommandPath), (Get-TestIdentity $assembly), (Get-TestIdentity $compiler))
    $sourceEvidence = Join-Path $evidenceDirectory 'sources'; [void][IO.Directory]::CreateDirectory($sourceEvidence)
    foreach ($name in @('wizard-console.cs', 'wizard-theme.cs', 'wizard-desktop.cs', 'wizard-prompts.cs', 'wizard-runtime.cs', 'wizard-downloads.cs', 'wizard-markdown.cs', 'wizard-workspaces.cs', 'wizard-runtime-bridge.ps1', 'infr-dashboard.ps1', 'Test-WizardRuntimeExperience.ps1')) {
        $path = Join-Path $PSScriptRoot $name
        $manifest.inputs += Get-TestIdentity $path
        Copy-Item -LiteralPath $path -Destination (Join-Path $sourceEvidence $name)
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $evidenceDirectory 'RuntimeExperienceHarness.cs')
    Copy-Item -LiteralPath $childSource -Destination (Join-Path $evidenceDirectory 'NamedPipeChild.ps1')
    $manifest | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $evidenceDirectory 'manifest.json') -Encoding UTF8
    $compileLog = & $compiler @compilerArgs | Out-String; $compileExit = $LASTEXITCODE
    $compileLog | Set-Content -LiteralPath (Join-Path $evidenceDirectory 'compile.log') -Encoding UTF8
    if ($compileExit -ne 0) { throw ('Test harness compilation failed: ' + $compileLog) }
    $manifest.executable = Get-TestIdentity $exe
    $ErrorActionPreference = 'Continue'
    try {
        & $exe $FixturePath $evidenceDirectory $childSource $bridgeSource (Join-Path $PSScriptRoot 'infr-wizard.ps1') 1> (Join-Path $evidenceDirectory 'tests.log') 2> (Join-Path $evidenceDirectory 'tests.err.log')
        $testExit = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = 'Stop' }
    Get-Content -LiteralPath (Join-Path $evidenceDirectory 'tests.log')
    if ($testExit -ne 0) { throw ('Runtime experience checks failed; inspect results.json in ' + $evidenceDirectory) }
    $manifest.status = 'passed'
}
catch { $manifest.status = 'failed'; $manifest.error = $_.Exception.Message; throw }
finally {
    $manifest.finishedUtc = [DateTime]::UtcNow.ToString('o')
    $manifest | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $evidenceDirectory 'manifest.json') -Encoding UTF8
    foreach ($name in $oldEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], 'Process') }
}
