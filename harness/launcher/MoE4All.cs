using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

internal static class MoE4All
{
    private const int KillOnJobClose = 0x2000;
    private const int ExtendedLimitInformation = 9;
    private const int WebPort = 17373;
    private static readonly object LogLock = new object();
    private static StreamWriter log;
    private static string logPath;
    private static AgentWindow window;

    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters
    {
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
        public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimitInformation
    {
        public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
        public int LimitFlags;
        public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
        public int ActiveProcessLimit;
        public UIntPtr Affinity;
        public int PriorityClass, SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimits
    {
        public BasicLimitInformation BasicLimitInformation;
        public IoCounters IoInfo;
        public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimits limits, uint length);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);

    private static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }

    private static ProcessStartInfo PowerShellStart(string root)
    {
        string powershell = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System),
            "WindowsPowerShell", "v1.0", "powershell.exe");
        ProcessStartInfo start = new ProcessStartInfo(powershell);
        start.WorkingDirectory = root;
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        start.EnvironmentVariables.Remove("PSModulePath");
        return start;
    }

    private static void WriteMessage(string message, bool error)
    {
        lock (LogLock)
        {
            if (error) Console.Error.WriteLine(message);
            else Console.WriteLine(message);
            if (log != null) log.WriteLine(message);
        }
        AgentWindow active = window;
        if (active != null) active.AppendLog(message, error);
    }

    private static void WriteMessage(string message)
    {
        WriteMessage(message, false);
    }

    private static int RunScript(IntPtr job, string root, string script, string arguments)
    {
        ProcessStartInfo start = PowerShellStart(root);
        start.Arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " + Quote(script) + arguments;
        start.RedirectStandardOutput = true;
        start.RedirectStandardError = true;
        start.StandardOutputEncoding = new UTF8Encoding(false);
        start.StandardErrorEncoding = new UTF8Encoding(false);
        using (Process child = new Process())
        {
            child.StartInfo = start;
            child.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null) WriteMessage(e.Data); };
            child.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { if (e.Data != null) WriteMessage(e.Data, true); };
            if (!child.Start()) return 4;
            if (!AssignProcessToJobObject(job, child.Handle))
            {
                child.Kill();
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            child.BeginOutputReadLine();
            child.BeginErrorReadLine();
            child.WaitForExit();
            child.WaitForExit();
            return child.ExitCode;
        }
    }

    private static IntPtr CreateChildJob()
    {
        IntPtr job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
        ExtendedLimits limits = new ExtendedLimits();
        limits.BasicLimitInformation.LimitFlags = KillOnJobClose;
        if (!SetInformationJobObject(job, ExtendedLimitInformation, ref limits, (uint)Marshal.SizeOf(typeof(ExtendedLimits))))
        {
            CloseHandle(job);
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }
        return job;
    }

    private static void PrepareProductDirectories(string root)
    {
        string data = Path.Combine(root, "data");
        string legacy = Path.Combine(root, "harness", "agent-state");
        if (!Directory.Exists(data) && Directory.Exists(legacy))
        {
            Directory.Move(legacy, data);
            WriteMessage("Migrated legacy Agent state to " + data);
        }

        string models = Path.Combine(root, "models");
        string legacyModels = Path.Combine(data, "home", "moe4all-models");
        if (!Directory.Exists(models) && Directory.Exists(legacyModels))
        {
            Directory.Move(legacyModels, models);
            WriteMessage("Migrated the managed model library to " + models);
        }

        Directory.CreateDirectory(data);
        Directory.CreateDirectory(models);
        Directory.CreateDirectory(Path.Combine(root, "cache"));
        Directory.CreateDirectory(Path.Combine(root, "updates"));
        Directory.CreateDirectory(Path.Combine(data, "logs"));
    }

    private static void OpenUrl(string url)
    {
        try
        {
            ProcessStartInfo start = new ProcessStartInfo(url);
            start.UseShellExecute = true;
            Process.Start(start);
        }
        catch (Exception error)
        {
            WriteMessage("Could not open " + url + ": " + error.Message, true);
        }
    }

    private static void StartUpdateChecker(string root)
    {
        string manifest = Path.Combine(root, "harness", "distribution", "agent-release.json");
        string script = Path.Combine(root, "harness", "scripts", "Check-AgentUpdate.ps1");
        if (!File.Exists(manifest) || !File.Exists(script)) return;
        ProcessStartInfo start = PowerShellStart(root);
        start.Arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " + Quote(script)
            + " -InstallRoot " + Quote(root) + " -ParentPid " + Process.GetCurrentProcess().Id;
        start.WorkingDirectory = root;
        start.WindowStyle = ProcessWindowStyle.Hidden;
        try
        {
            using (Process checker = Process.Start(start)) { }
        }
        catch (Exception error)
        {
            WriteMessage("Agent update check could not start: " + error.Message, true);
        }
    }

    private static void OpenLog(string root)
    {
        string path = Path.Combine(root, "data", "logs", "launcher.log");
        if (File.Exists(path)) OpenUrl(path);
    }

    [STAThread]
    private static int Main(string[] arguments)
    {
        try
        {
            Console.OutputEncoding = new UTF8Encoding(false);
        }
        catch (IOException)
        {
            // The normal WinExe launcher intentionally has no console handle.
        }
        bool headless = Array.IndexOf(arguments, "--headless") >= 0 || Array.IndexOf(arguments, "--no-pause") >= 0;
        string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
        try
        {
            PrepareProductDirectories(root);
            logPath = Path.Combine(root, "data", "logs", "launcher.log");
            log = new StreamWriter(new FileStream(logPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite), new UTF8Encoding(false));
            log.AutoFlush = true;
            WriteMessage("=== MoE4All Agent launch " + DateTime.UtcNow.ToString("o") + " ===");
            WriteMessage("Install: " + root);

            if (headless)
            {
                int result = RunHeadless(root);
                if (result != 0)
                {
                    WriteMessage("MoE4All Agent exited with code " + result + ".", true);
                    WriteMessage("Startup log: " + logPath, true);
                }
                return result;
            }

            bool first;
            using (Mutex mutex = new Mutex(true, "Local\\MoE4All-Agent", out first))
            {
                if (!first)
                {
                    MessageBox.Show("MoE4All Agent is already running in this Windows session.", "MoE4All Agent",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                    return 2;
                }
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                window = new AgentWindow(root);
                Application.Run(window);
                window = null;
            }
            return 0;
        }
        catch (Exception error)
        {
            WriteMessage("MoE4All Agent could not start: " + error, true);
            if (!headless)
            {
                MessageBox.Show("MoE4All Agent could not start.\r\n\r\n" + error.Message + "\r\n\r\nLog: " + logPath,
                    "MoE4All Agent", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            return 1;
        }
        finally
        {
            if (log != null) log.Dispose();
        }
    }

    private static int RunHeadless(string root)
    {
        bool first;
        using (Mutex mutex = new Mutex(true, "Local\\MoE4All-Agent-Headless", out first))
        {
            if (!first) return 2;
            string state = Path.Combine(root, "data");
            string profile = Path.Combine(state, "home", "profiles", "web", "node_modules");
            string run = Path.Combine(root, "harness", "scripts", "Run-Agent.ps1");
            string prepare = Path.Combine(root, "harness", "scripts", "Prepare-BundledRuntime.ps1");
            if (!File.Exists(run) || !File.Exists(prepare))
            {
                WriteMessage("The bundled Agent runtime is incomplete. Install the full MoE4All package, not MoE4All.exe alone.", true);
                return 3;
            }
            IntPtr job = CreateChildJob();
            try
            {
                int prepared = RunScript(job, root, prepare, " -InstallRoot " + Quote(root) + " -StateRoot " + Quote(state));
                if (prepared != 0 || !Directory.Exists(profile)) return prepared != 0 ? prepared : 3;
                return RunScript(job, root, run, " -StateRoot " + Quote(state));
            }
            finally
            {
                CloseHandle(job);
            }
        }
    }

    private sealed class AgentWindow : Form
    {
        private readonly string root;
        private readonly Label status;
        private readonly TextBox output;
        private readonly Panel startup;
        private readonly Button browserButton;
        private readonly Button runtimeButton;
        private readonly WebView2 webView;
        private IntPtr job = IntPtr.Zero;
        private DesktopBridge bridge;
        private volatile bool closing;
        private volatile bool ready;

        public AgentWindow(string installRoot)
        {
            root = installRoot;
            Text = "MoE4All Agent";
            StartPosition = FormStartPosition.CenterScreen;
            MinimumSize = new Size(900, 620);
            Size = new Size(1280, 820);
            BackColor = Color.FromArgb(31, 32, 35);
            ForeColor = Color.FromArgb(232, 234, 237);
            Font = new Font("Segoe UI", 9F);

            Panel toolbar = new Panel();
            toolbar.Dock = DockStyle.Top;
            toolbar.Height = 44;
            toolbar.Padding = new Padding(14, 8, 10, 7);
            toolbar.BackColor = Color.FromArgb(42, 44, 48);

            Label title = new Label();
            title.Text = "MoE4All";
            title.Font = new Font("Segoe UI Semibold", 11F);
            title.AutoSize = true;
            title.Location = new Point(14, 12);
            toolbar.Controls.Add(title);

            status = new Label();
            status.Text = "Starting";
            status.ForeColor = Color.FromArgb(176, 179, 184);
            status.AutoSize = true;
            status.Location = new Point(98, 14);
            toolbar.Controls.Add(status);

            browserButton = CreateButton("Open in browser");
            browserButton.Dock = DockStyle.Right;
            browserButton.Width = 118;
            browserButton.Click += delegate { OpenUrl("http://127.0.0.1:" + WebPort + "/"); };
            toolbar.Controls.Add(browserButton);

            webView = new WebView2();
            webView.Dock = DockStyle.Fill;
            webView.Visible = false;
            webView.BackColor = BackColor;

            startup = new Panel();
            startup.Dock = DockStyle.Fill;
            startup.Padding = new Padding(54);
            startup.BackColor = BackColor;

            Label heading = new Label();
            heading.Text = "Preparing MoE4All Agent";
            heading.Font = new Font("Segoe UI Semibold", 22F);
            heading.AutoSize = true;
            heading.Location = new Point(50, 58);
            startup.Controls.Add(heading);

            Label explanation = new Label();
            explanation.Text = "Runtime, plugins and your private workspace stay inside this MoE4All directory.";
            explanation.ForeColor = Color.FromArgb(176, 179, 184);
            explanation.AutoSize = true;
            explanation.Location = new Point(54, 106);
            startup.Controls.Add(explanation);

            output = new TextBox();
            output.Multiline = true;
            output.ReadOnly = true;
            output.ScrollBars = ScrollBars.Vertical;
            output.BorderStyle = BorderStyle.FixedSingle;
            output.BackColor = Color.FromArgb(37, 38, 41);
            output.ForeColor = Color.FromArgb(196, 199, 204);
            output.Font = new Font("Consolas", 9F);
            output.Location = new Point(54, 152);
            output.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
            output.Size = new Size(Width - 128, Height - 300);
            startup.Controls.Add(output);

            runtimeButton = CreateButton("Install WebView2 Runtime");
            runtimeButton.Visible = false;
            runtimeButton.Location = new Point(54, Height - 130);
            runtimeButton.Anchor = AnchorStyles.Left | AnchorStyles.Bottom;
            runtimeButton.Width = 180;
            runtimeButton.Click += delegate { OpenUrl("https://developer.microsoft.com/microsoft-edge/webview2/"); };
            startup.Controls.Add(runtimeButton);

            Button logButton = CreateButton("Open startup log");
            logButton.Location = new Point(246, Height - 130);
            logButton.Anchor = AnchorStyles.Left | AnchorStyles.Bottom;
            logButton.Width = 138;
            logButton.Click += delegate { OpenLog(root); };
            startup.Controls.Add(logButton);

            Controls.Add(webView);
            Controls.Add(startup);
            Controls.Add(toolbar);
            Shown += delegate { StartAgent(); };
            FormClosing += OnClosing;
        }

        private static Button CreateButton(string text)
        {
            Button button = new Button();
            button.Text = text;
            button.Height = 29;
            button.FlatStyle = FlatStyle.Flat;
            button.FlatAppearance.BorderColor = Color.FromArgb(82, 85, 91);
            button.BackColor = Color.FromArgb(50, 52, 56);
            button.ForeColor = Color.FromArgb(232, 234, 237);
            button.UseVisualStyleBackColor = false;
            return button;
        }

        public void AppendLog(string message, bool isError)
        {
            if (IsDisposed) return;
            if (InvokeRequired)
            {
                try { BeginInvoke(new Action<string, bool>(AppendLog), message, isError); }
                catch (InvalidOperationException) { }
                return;
            }
            status.Text = isError ? "Needs attention" : (ready ? "Ready" : "Starting");
            if (isError) status.ForeColor = Color.FromArgb(235, 112, 112);
            output.AppendText(message + Environment.NewLine);
        }

        private void StartAgent()
        {
            Thread worker = new Thread(RunAgent);
            worker.IsBackground = true;
            worker.Name = "MoE4All Agent launcher";
            worker.Start();
        }

        private void RunAgent()
        {
            string state = Path.Combine(root, "data");
            string profile = Path.Combine(state, "home", "profiles", "web", "node_modules");
            string script = Path.Combine(root, "harness", "scripts", "Run-Agent.ps1");
            string prepareScript = Path.Combine(root, "harness", "scripts", "Prepare-BundledRuntime.ps1");
            if (!File.Exists(script) || !File.Exists(prepareScript))
            {
                Fail("The bundled Agent runtime is incomplete. Install the full MoE4All package, not MoE4All.exe alone.");
                return;
            }

            try
            {
                job = CreateChildJob();
                bridge = new DesktopBridge(this, Path.Combine(root, "models"));
                bridge.Start();
                if (!Directory.Exists(profile)) WriteMessage("Preparing the bundled MoE4All Agent runtime.");
                int prepared = RunScript(job, root, prepareScript,
                    " -InstallRoot " + Quote(root) + " -StateRoot " + Quote(state));
                if (prepared != 0 || !Directory.Exists(profile))
                {
                    Fail("Preparing the bundled Agent runtime failed with code " + prepared + ".");
                    return;
                }

                WriteMessage("MoE4All Agent is starting.");
                Thread readiness = new Thread(WaitForReady);
                readiness.IsBackground = true;
                readiness.Start();
                int result = RunScript(job, root, script,
                    " -StateRoot " + Quote(state)
                    + " -DesktopBridge " + Quote(bridge.BaseUrl)
                    + " -DesktopToken " + Quote(bridge.Token));
                if (!closing) Fail("MoE4All Agent exited with code " + result + ".");
            }
            catch (Exception error)
            {
                if (!closing) Fail("MoE4All Agent could not start: " + error.Message);
                WriteMessage(error.ToString(), true);
            }
        }

        private void WaitForReady()
        {
            for (int attempt = 0; attempt < 600 && !closing; attempt++)
            {
                try
                {
                    HttpWebRequest request = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:" + WebPort + "/");
                    request.Timeout = 1000;
                    request.ReadWriteTimeout = 1000;
                    using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                    {
                        if ((int)response.StatusCode >= 200 && (int)response.StatusCode < 400)
                        {
                            BeginInvoke(new Action(ActivateWebView));
                            StartUpdateChecker(root);
                            return;
                        }
                    }
                }
                catch { }
                Thread.Sleep(500);
            }
            if (!closing) Fail("The Agent web service did not become ready. See the startup log for details.");
        }

        private async void ActivateWebView()
        {
            try
            {
                string userData = Path.Combine(root, "cache", "webview2");
                Directory.CreateDirectory(userData);
                CoreWebView2Environment environment = await CoreWebView2Environment.CreateAsync(null, userData);
                await webView.EnsureCoreWebView2Async(environment);
                webView.CoreWebView2.Settings.AreDevToolsEnabled = false;
                webView.CoreWebView2.Settings.AreDefaultContextMenusEnabled = true;
                webView.CoreWebView2.NewWindowRequested += delegate(object sender, CoreWebView2NewWindowRequestedEventArgs e)
                {
                    e.Handled = true;
                    OpenUrl(e.Uri);
                };
                webView.CoreWebView2.NavigationStarting += delegate(object sender, CoreWebView2NavigationStartingEventArgs e)
                {
                    Uri uri;
                    if (Uri.TryCreate(e.Uri, UriKind.Absolute, out uri) && !uri.IsLoopback)
                    {
                        e.Cancel = true;
                        OpenUrl(e.Uri);
                    }
                };
                webView.Source = new Uri("http://127.0.0.1:" + WebPort + "/");
                ready = true;
                status.Text = "Ready";
                status.ForeColor = Color.FromArgb(131, 201, 157);
                startup.Visible = false;
                webView.Visible = true;
                webView.BringToFront();
            }
            catch (WebView2RuntimeNotFoundException error)
            {
                runtimeButton.Visible = true;
                Fail("Microsoft WebView2 Runtime is not installed. Install it or use Open in browser. " + error.Message);
            }
            catch (Exception error)
            {
                Fail("The desktop view could not start. You can continue with Open in browser. " + error.Message);
            }
        }

        private void Fail(string message)
        {
            WriteMessage(message, true);
            if (IsDisposed) return;
            if (InvokeRequired)
            {
                try { BeginInvoke(new Action<string>(FailUi), message); }
                catch (InvalidOperationException) { }
            }
            else FailUi(message);
        }

        private void FailUi(string message)
        {
            ready = false;
            status.Text = "Needs attention";
            status.ForeColor = Color.FromArgb(235, 112, 112);
            startup.Visible = true;
            startup.BringToFront();
            browserButton.Enabled = true;
        }

        private void OnClosing(object sender, FormClosingEventArgs e)
        {
            closing = true;
            if (bridge != null) bridge.Dispose();
            IntPtr current = job;
            job = IntPtr.Zero;
            if (current != IntPtr.Zero) CloseHandle(current);
        }

        public string PickDirectory()
        {
            if (InvokeRequired) return (string)Invoke(new Func<string>(PickDirectory));
            using (FolderBrowserDialog dialog = new FolderBrowserDialog())
            {
                dialog.Description = "Select a MoE4All workspace directory";
                dialog.ShowNewFolderButton = true;
                return dialog.ShowDialog(this) == DialogResult.OK ? dialog.SelectedPath : null;
            }
        }

        public string PickModel(string initialDirectory)
        {
            if (InvokeRequired) return (string)Invoke(new Func<string, string>(PickModel), initialDirectory);
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Title = "Select a MoE4All GGUF model";
                dialog.Filter = "GGUF model (*.gguf)|*.gguf|All files (*.*)|*.*";
                dialog.Multiselect = false;
                dialog.CheckFileExists = true;
                dialog.InitialDirectory = Directory.Exists(initialDirectory) ? initialDirectory : root;
                return dialog.ShowDialog(this) == DialogResult.OK ? dialog.FileName : null;
            }
        }
    }

    private sealed class DesktopBridge : IDisposable
    {
        private readonly AgentWindow owner;
        private readonly string modelDirectory;
        private readonly TcpListener listener;
        private Thread thread;
        private volatile bool stopping;
        public readonly string Token;
        public string BaseUrl { get; private set; }

        public DesktopBridge(AgentWindow windowOwner, string models)
        {
            owner = windowOwner;
            modelDirectory = models;
            listener = new TcpListener(IPAddress.Loopback, 0);
            byte[] secret = new byte[32];
            using (RandomNumberGenerator random = RandomNumberGenerator.Create()) random.GetBytes(secret);
            Token = Convert.ToBase64String(secret);
        }

        public void Start()
        {
            listener.Start();
            IPEndPoint endpoint = (IPEndPoint)listener.LocalEndpoint;
            BaseUrl = "http://127.0.0.1:" + endpoint.Port;
            thread = new Thread(Listen);
            thread.IsBackground = true;
            thread.Name = "MoE4All desktop picker bridge";
            thread.Start();
        }

        private void Listen()
        {
            while (!stopping)
            {
                try
                {
                    TcpClient client = listener.AcceptTcpClient();
                    ThreadPool.QueueUserWorkItem(delegate { Handle(client); });
                }
                catch (SocketException) { if (!stopping) WriteMessage("Desktop bridge stopped unexpectedly.", true); }
                catch (ObjectDisposedException) { }
            }
        }

        private void Handle(TcpClient client)
        {
            using (client)
            using (NetworkStream stream = client.GetStream())
            {
                stream.ReadTimeout = 5000;
                string headers = ReadHeaders(stream);
                string[] lines = headers.Split(new string[] { "\r\n" }, StringSplitOptions.None);
                string[] request = lines.Length == 0 ? new string[0] : lines[0].Split(' ');
                string method = request.Length > 0 ? request[0] : "";
                string path = request.Length > 1 ? request[1] : "";
                string supplied = "";
                for (int index = 1; index < lines.Length; index++)
                {
                    int separator = lines[index].IndexOf(':');
                    if (separator > 0 && lines[index].Substring(0, separator).Trim().Equals("X-MoE4All-Token", StringComparison.OrdinalIgnoreCase))
                        supplied = lines[index].Substring(separator + 1).Trim();
                }
                if (method != "POST" || supplied != Token)
                {
                    Respond(stream, 403, "{\"error\":\"forbidden\"}");
                    return;
                }
                string selected;
                if (path == "/pick-directory") selected = owner.PickDirectory();
                else if (path == "/pick-model") selected = owner.PickModel(modelDirectory);
                else
                {
                    Respond(stream, 404, "{\"error\":\"not-found\"}");
                    return;
                }
                Respond(stream, 200, "{\"path\":" + (selected == null ? "null" : "\"" + JsonEscape(selected) + "\"") + "}");
            }
        }

        private static string ReadHeaders(NetworkStream stream)
        {
            MemoryStream buffer = new MemoryStream();
            int matched = 0;
            while (buffer.Length < 16384)
            {
                int value = stream.ReadByte();
                if (value < 0) break;
                buffer.WriteByte((byte)value);
                if ((matched == 0 && value == 13) || (matched == 1 && value == 10)
                    || (matched == 2 && value == 13) || (matched == 3 && value == 10))
                {
                    matched++;
                    if (matched == 4) break;
                }
                else matched = value == 13 ? 1 : 0;
            }
            return Encoding.ASCII.GetString(buffer.ToArray());
        }

        private static string JsonEscape(string value)
        {
            StringBuilder result = new StringBuilder();
            foreach (char character in value)
            {
                if (character == '\\') result.Append("\\\\");
                else if (character == '\"') result.Append("\\\"");
                else if (character == '\r') result.Append("\\r");
                else if (character == '\n') result.Append("\\n");
                else if (character < 32) result.Append("\\u" + ((int)character).ToString("x4"));
                else result.Append(character);
            }
            return result.ToString();
        }

        private static void Respond(NetworkStream stream, int status, string body)
        {
            byte[] payload = new UTF8Encoding(false).GetBytes(body);
            string reason = status == 200 ? "OK" : (status == 403 ? "Forbidden" : "Not Found");
            string headers = "HTTP/1.1 " + status + " " + reason + "\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: "
                + payload.Length + "\r\nConnection: close\r\n\r\n";
            byte[] prefix = Encoding.ASCII.GetBytes(headers);
            stream.Write(prefix, 0, prefix.Length);
            stream.Write(payload, 0, payload.Length);
        }

        public void Dispose()
        {
            stopping = true;
            try { listener.Stop(); }
            catch { }
        }
    }
}
