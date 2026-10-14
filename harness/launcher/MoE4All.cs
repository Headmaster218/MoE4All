using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

internal static class MoE4All
{
    private const int KillOnJobClose = 0x2000;
    private const int ExtendedLimitInformation = 9;
    private const int WebPort = 17373;
    private const string ProgressPrefix = "@@MOE4ALL_PROGRESS\t";
    private const string UpdatePrefix = "@@MOE4ALL_UPDATE\t";
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

    private static Assembly ResolveBundledManagedAssembly(object sender, ResolveEventArgs arguments)
    {
        string name = new AssemblyName(arguments.Name).Name;
        if (name != "Microsoft.Web.WebView2.Core" && name != "Microsoft.Web.WebView2.WinForms") return null;
        string filename = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, name + ".dll");
        if (!File.Exists(filename)) return null;
        return Assembly.Load(File.ReadAllBytes(filename));
    }

    private static void RepairLauncherConfiguration(string root)
    {
        string active = Path.Combine(root, "MoE4All.exe.config");
        if (File.Exists(active)) return;
        string bundled = Path.Combine(root, "harness", "launcher", "MoE4All.exe.config");
        if (File.Exists(bundled)) File.Copy(bundled, active, false);
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

    private static bool TryHandleProgress(string message)
    {
        if (!message.StartsWith(ProgressPrefix, StringComparison.Ordinal)) return false;
        string[] parts = message.Split(new char[] { '\t' }, 4);
        long completed;
        long total;
        if (parts.Length != 4 || !long.TryParse(parts[1], out completed) || !long.TryParse(parts[2], out total)) return true;
        AgentWindow active = window;
        if (active != null) active.UpdatePreparationProgress(completed, total, parts[3]);
        return true;
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
            child.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e)
            {
                if (e.Data != null && !TryHandleProgress(e.Data)) WriteMessage(e.Data);
            };
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

    private static void OpenLog(string root)
    {
        string path = Path.Combine(root, "data", "logs", "launcher.log");
        if (File.Exists(path)) OpenUrl(path);
    }

    [STAThread]
    private static int Main(string[] arguments)
    {
        AppContext.SetSwitch("Switch.System.IO.UseLegacyPathHandling", false);
        AppContext.SetSwitch("Switch.System.IO.BlockLongPaths", false);
        AppDomain.CurrentDomain.AssemblyResolve += ResolveBundledManagedAssembly;
        try
        {
            Console.OutputEncoding = new UTF8Encoding(false);
        }
        catch (IOException)
        {
            // The normal WinExe launcher intentionally has no console handle.
        }
        bool headless = Array.IndexOf(arguments, "--headless") >= 0 || Array.IndexOf(arguments, "--no-pause") >= 0;
        bool startupUiTest = Array.IndexOf(arguments, "--startup-ui-test") >= 0;
        string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
        if (startupUiTest) return RunStartupUiTest(root);
        try
        {
            RepairLauncherConfiguration(root);
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

    private static int RunStartupUiTest(string root)
    {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        using (AgentWindow testWindow = new AgentWindow(root, false))
        {
            testWindow.ShowInTaskbar = false;
            testWindow.Opacity = 0;
            testWindow.Show();
            Application.DoEvents();
            Size[] sizes = { new Size(760, 520), new Size(1024, 640), new Size(1280, 760) };
            foreach (Size size in sizes)
            {
                string error;
                if (!testWindow.ValidateStartupLayoutForTest(size, out error))
                {
                    Console.Error.WriteLine("Startup UI layout failed at " + size.Width + "x" + size.Height + ": " + error);
                    return 5;
                }
            }
        }
        Console.WriteLine("Startup UI layout tests passed at 760x520, 1024x640 and 1280x760.");
        return 0;
    }

    private sealed class StartupProgressIndicator : Control
    {
        private readonly System.Windows.Forms.Timer animation;
        private int progress;
        private int animationOffset;
        private bool indeterminate = true;
        private bool failed;

        public StartupProgressIndicator()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer
                | ControlStyles.ResizeRedraw | ControlStyles.UserPaint, true);
            Height = 12;
            TabStop = false;
            animation = new System.Windows.Forms.Timer();
            animation.Interval = 28;
            animation.Tick += delegate
            {
                if (!indeterminate || Width <= 0) return;
                animationOffset = (animationOffset + Math.Max(4, Width / 100)) % Math.Max(1, Width + Math.Max(48, Width / 4));
                Invalidate();
            };
            animation.Start();
        }

        public void SetProgress(int value)
        {
            progress = Math.Max(0, Math.Min(1000, value));
            indeterminate = false;
            failed = false;
            Invalidate();
        }

        public void SetIndeterminate()
        {
            indeterminate = true;
            failed = false;
            animationOffset = 0;
            Invalidate();
        }

        public void SetFailed()
        {
            indeterminate = false;
            failed = true;
            Invalidate();
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);
            if (ClientSize.Width < 2 || ClientSize.Height < 2) return;

            e.Graphics.SmoothingMode = SmoothingMode.AntiAlias;
            int barHeight = Math.Min(8, Math.Max(4, ClientSize.Height - 2));
            Rectangle track = new Rectangle(0, (ClientSize.Height - barHeight) / 2, ClientSize.Width - 1, barHeight);
            using (GraphicsPath trackPath = RoundedRectangle(track, barHeight / 2))
            using (SolidBrush trackBrush = new SolidBrush(Color.FromArgb(57, 60, 65)))
            {
                e.Graphics.FillPath(trackBrush, trackPath);
                e.Graphics.SetClip(trackPath);
                Color fillColor = failed ? Color.FromArgb(199, 93, 93) : Color.FromArgb(126, 181, 151);
                using (SolidBrush fillBrush = new SolidBrush(fillColor))
                {
                    if (indeterminate)
                    {
                        int segmentWidth = Math.Max(48, track.Width / 4);
                        Rectangle segment = new Rectangle(animationOffset - segmentWidth, track.Y, segmentWidth, track.Height);
                        using (GraphicsPath segmentPath = RoundedRectangle(segment, barHeight / 2))
                            e.Graphics.FillPath(fillBrush, segmentPath);
                    }
                    else if (progress > 0 || failed)
                    {
                        int fillWidth = failed ? track.Width : Math.Max(1, track.Width * progress / 1000);
                        Rectangle fill = new Rectangle(track.X, track.Y, fillWidth, track.Height);
                        using (GraphicsPath fillPath = RoundedRectangle(fill, barHeight / 2))
                            e.Graphics.FillPath(fillBrush, fillPath);
                    }
                }
                e.Graphics.ResetClip();
            }
        }

        private static GraphicsPath RoundedRectangle(Rectangle bounds, int radius)
        {
            GraphicsPath path = new GraphicsPath();
            int diameter = Math.Max(1, Math.Min(radius * 2, Math.Min(bounds.Width, bounds.Height)));
            if (diameter <= 1)
            {
                path.AddRectangle(bounds);
                return path;
            }
            Rectangle arc = new Rectangle(bounds.X, bounds.Y, diameter, diameter);
            path.AddArc(arc, 180, 90);
            arc.X = bounds.Right - diameter;
            path.AddArc(arc, 270, 90);
            arc.Y = bounds.Bottom - diameter;
            path.AddArc(arc, 0, 90);
            arc.X = bounds.X;
            path.AddArc(arc, 90, 90);
            path.CloseFigure();
            return path;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) animation.Dispose();
            base.Dispose(disposing);
        }
    }

    private sealed class AgentWindow : Form
    {
        private readonly string root;
        private readonly bool firstLaunch;
        private readonly Label status;
        private readonly Label startupHeading;
        private readonly Label startupExplanation;
        private readonly Label progressText;
        private readonly Label progressDetail;
        private readonly Label progressPercent;
        private readonly StartupProgressIndicator progressBar;
        private readonly TextBox output;
        private readonly Panel startup;
        private readonly TableLayoutPanel startupLayout;
        private readonly Panel detailsPanel;
        private readonly FlowLayoutPanel startupActions;
        private readonly Button detailsButton;
        private readonly Button logButton;
        private readonly Button browserButton;
        private readonly Button updateButton;
        private readonly Button runtimeButton;
        private readonly WebView2 webView;
        private readonly string installedVersion;
        private IntPtr job = IntPtr.Zero;
        private DesktopBridge bridge;
        private UpdateDialog updateDialog;
        private volatile bool closing;
        private volatile bool ready;
        private bool preparingProgress;
        private bool updateBusy;
        private bool updateInstalling;
        private bool updateStatusFailed;
        private bool revealUpdateResult;
        private string availableUpdate;
        private long availableUpdateBytes;

        public AgentWindow(string installRoot, bool startAgentOnShown = true)
        {
            root = installRoot;
            firstLaunch = !Directory.Exists(Path.Combine(root, "data", "home", "profiles", "web", "node_modules"));
            installedVersion = ReadInstalledVersion(root);
            Text = "MoE4All Agent";
            StartPosition = FormStartPosition.CenterScreen;
            AutoScaleMode = AutoScaleMode.Dpi;
            Rectangle workingArea = Screen.FromPoint(Cursor.Position).WorkingArea;
            int initialWidth = Math.Max(680, Math.Min(1180, workingArea.Width - 48));
            int initialHeight = Math.Max(500, Math.Min(760, workingArea.Height - 48));
            MinimumSize = new Size(Math.Min(760, initialWidth), Math.Min(520, initialHeight));
            Size = new Size(initialWidth, initialHeight);
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

            updateButton = CreateButton("Check updates");
            updateButton.Dock = DockStyle.Right;
            updateButton.Width = 112;
            updateButton.Enabled = false;
            updateButton.Click += delegate { ShowUpdates(); };
            toolbar.Controls.Add(updateButton);

            webView = new WebView2();
            webView.Dock = DockStyle.Fill;
            webView.Visible = false;
            webView.BackColor = BackColor;

            startup = new Panel();
            startup.Dock = DockStyle.Fill;
            startup.BackColor = BackColor;
            startup.AutoScroll = true;

            startupLayout = new TableLayoutPanel();
            startupLayout.Dock = DockStyle.Fill;
            startupLayout.Padding = new Padding(52, 38, 52, 28);
            startupLayout.BackColor = BackColor;
            startupLayout.ColumnCount = 1;
            startupLayout.RowCount = 10;
            startupLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
            startupLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 25F));
            startupLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 56F));
            startupLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 66F));
            startupLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 12F));
            startupLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 32F));
            startupLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 16F));
            startupLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 38F));
            startupLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
            startupLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 22F));
            startupLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 42F));
            startup.Controls.Add(startupLayout);

            Label eyebrow = new Label();
            eyebrow.Text = firstLaunch ? "MOE4ALL AGENT  /  FIRST LAUNCH" : "MOE4ALL AGENT";
            eyebrow.Dock = DockStyle.Fill;
            eyebrow.Font = new Font("Segoe UI Semibold", 9F);
            eyebrow.ForeColor = Color.FromArgb(132, 184, 155);
            eyebrow.TextAlign = ContentAlignment.MiddleLeft;
            startupLayout.Controls.Add(eyebrow, 0, 0);

            startupHeading = new Label();
            startupHeading.Text = firstLaunch ? "专家正在登机" : "正在唤醒你的 Agent";
            startupHeading.Dock = DockStyle.Fill;
            startupHeading.Font = new Font("Microsoft YaHei UI", 24F, FontStyle.Bold);
            startupHeading.ForeColor = Color.FromArgb(239, 240, 242);
            startupHeading.TextAlign = ContentAlignment.MiddleLeft;
            startupLayout.Controls.Add(startupHeading, 0, 1);

            startupExplanation = new Label();
            startupExplanation.Text = firstLaunch
                ? "这是第一次启动。MoE4All 正在准备 Agent 运行时、内置插件和独立工作区。只需一次，完成后会自动进入。"
                : "正在连接本地运行时、内置插件和工作区，很快就好。";
            startupExplanation.Dock = DockStyle.Fill;
            startupExplanation.Font = new Font("Microsoft YaHei UI", 10F);
            startupExplanation.ForeColor = Color.FromArgb(174, 178, 184);
            startupExplanation.TextAlign = ContentAlignment.TopLeft;
            startupLayout.Controls.Add(startupExplanation, 0, 2);

            TableLayoutPanel progressHeader = new TableLayoutPanel();
            progressHeader.Dock = DockStyle.Fill;
            progressHeader.ColumnCount = 2;
            progressHeader.RowCount = 1;
            progressHeader.Margin = Padding.Empty;
            progressHeader.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
            progressHeader.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 64F));
            progressHeader.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
            startupLayout.Controls.Add(progressHeader, 0, 4);

            progressText = new Label();
            progressText.Text = firstLaunch ? "正在核验登机清单" : "正在检查本地运行时";
            progressText.Dock = DockStyle.Fill;
            progressText.Font = new Font("Microsoft YaHei UI", 10F, FontStyle.Bold);
            progressText.ForeColor = Color.FromArgb(214, 216, 219);
            progressText.TextAlign = ContentAlignment.MiddleLeft;
            progressHeader.Controls.Add(progressText, 0, 0);

            progressPercent = new Label();
            progressPercent.Text = "准备中";
            progressPercent.Dock = DockStyle.Fill;
            progressPercent.ForeColor = Color.FromArgb(142, 147, 154);
            progressPercent.TextAlign = ContentAlignment.MiddleRight;
            progressHeader.Controls.Add(progressPercent, 1, 0);

            progressBar = new StartupProgressIndicator();
            progressBar.Dock = DockStyle.Fill;
            progressBar.Margin = new Padding(0, 2, 0, 2);
            startupLayout.Controls.Add(progressBar, 0, 5);

            progressDetail = new Label();
            progressDetail.Text = firstLaunch
                ? "运行时、插件与工作区将在本机完成准备。"
                : "正在复用已经准备好的本地资源。";
            progressDetail.Dock = DockStyle.Fill;
            progressDetail.Font = new Font("Microsoft YaHei UI", 9F);
            progressDetail.ForeColor = Color.FromArgb(139, 144, 151);
            progressDetail.TextAlign = ContentAlignment.TopLeft;
            progressDetail.AutoEllipsis = true;
            startupLayout.Controls.Add(progressDetail, 0, 6);

            detailsPanel = new Panel();
            detailsPanel.Dock = DockStyle.Fill;
            detailsPanel.Margin = new Padding(0, 8, 0, 4);
            detailsPanel.Visible = false;
            startupLayout.Controls.Add(detailsPanel, 0, 7);

            TableLayoutPanel detailsLayout = new TableLayoutPanel();
            detailsLayout.Dock = DockStyle.Fill;
            detailsLayout.ColumnCount = 1;
            detailsLayout.RowCount = 2;
            detailsLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100F));
            detailsLayout.RowStyles.Add(new RowStyle(SizeType.Absolute, 24F));
            detailsLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100F));
            detailsPanel.Controls.Add(detailsLayout);

            Label detailsHeading = new Label();
            detailsHeading.Text = "启动详情";
            detailsHeading.Dock = DockStyle.Fill;
            detailsHeading.ForeColor = Color.FromArgb(154, 158, 164);
            detailsHeading.TextAlign = ContentAlignment.MiddleLeft;
            detailsLayout.Controls.Add(detailsHeading, 0, 0);

            output = new TextBox();
            output.Multiline = true;
            output.ReadOnly = true;
            output.ScrollBars = ScrollBars.Vertical;
            output.BorderStyle = BorderStyle.FixedSingle;
            output.BackColor = Color.FromArgb(37, 38, 41);
            output.ForeColor = Color.FromArgb(196, 199, 204);
            output.Font = new Font("Consolas", 9F);
            output.Dock = DockStyle.Fill;
            output.Margin = Padding.Empty;
            detailsLayout.Controls.Add(output, 0, 1);

            Label firstLaunchNote = new Label();
            firstLaunchNote.Text = firstLaunch
                ? "专家抵达后，后续启动会直接复用这套本地环境。"
                : "本地环境已就位，正在等待 Agent 响应。";
            firstLaunchNote.Dock = DockStyle.Fill;
            firstLaunchNote.Font = new Font("Microsoft YaHei UI", 8.5F);
            firstLaunchNote.ForeColor = Color.FromArgb(122, 127, 134);
            firstLaunchNote.TextAlign = ContentAlignment.MiddleLeft;
            startupLayout.Controls.Add(firstLaunchNote, 0, 8);

            startupActions = new FlowLayoutPanel();
            startupActions.Dock = DockStyle.Fill;
            startupActions.Margin = Padding.Empty;
            startupActions.Padding = new Padding(0, 6, 0, 0);
            startupActions.FlowDirection = FlowDirection.LeftToRight;
            startupActions.WrapContents = false;
            startupLayout.Controls.Add(startupActions, 0, 9);

            detailsButton = CreateButton("查看启动详情");
            detailsButton.Width = 122;
            detailsButton.Margin = new Padding(0, 0, 10, 0);
            detailsButton.Click += delegate { SetDetailsVisible(!detailsPanel.Visible); };
            startupActions.Controls.Add(detailsButton);

            runtimeButton = CreateButton("Install WebView2 Runtime");
            runtimeButton.Visible = false;
            runtimeButton.Width = 180;
            runtimeButton.Margin = new Padding(0, 0, 10, 0);
            runtimeButton.Click += delegate { OpenUrl("https://developer.microsoft.com/microsoft-edge/webview2/"); };
            startupActions.Controls.Add(runtimeButton);

            logButton = CreateButton("打开启动日志");
            logButton.Width = 138;
            logButton.Margin = Padding.Empty;
            logButton.Click += delegate { OpenLog(root); };
            startupActions.Controls.Add(logButton);

            Controls.Add(webView);
            Controls.Add(startup);
            Controls.Add(toolbar);
            Load += delegate { FitToWorkingArea(); };
            if (startAgentOnShown) Shown += delegate { StartAgent(); };
            FormClosing += OnClosing;
        }

        private static string ReadInstalledVersion(string installRoot)
        {
            string manifest = Path.Combine(installRoot, "harness", "distribution", "agent-release.json");
            if (!File.Exists(manifest)) return "development build";
            try
            {
                Match match = Regex.Match(File.ReadAllText(manifest), "\\\"version\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
                return match.Success ? match.Groups[1].Value : "development build";
            }
            catch
            {
                return "development build";
            }
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

        private void SetDetailsVisible(bool visible)
        {
            detailsPanel.Visible = visible;
            detailsButton.Text = visible ? "收起启动详情" : "查看启动详情";
            startupLayout.PerformLayout();
        }

        private void FitToWorkingArea()
        {
            Rectangle workingArea = Screen.FromControl(this).WorkingArea;
            int maxWidth = Math.Max(640, workingArea.Width - 32);
            int maxHeight = Math.Max(460, workingArea.Height - 32);
            MinimumSize = new Size(Math.Min(MinimumSize.Width, maxWidth), Math.Min(MinimumSize.Height, maxHeight));
            int width = Math.Min(Width, maxWidth);
            int height = Math.Min(Height, maxHeight);
            SetBounds(workingArea.Left + (workingArea.Width - width) / 2,
                workingArea.Top + (workingArea.Height - height) / 2, width, height);
        }

        private string PreparationStage(int percent)
        {
            if (!firstLaunch)
            {
                if (percent < 35) return "正在检查本地运行时";
                if (percent < 85) return "正在连接插件与工作区";
                return "Agent 即将醒来";
            }
            if (percent < 10) return "正在核验登机清单";
            if (percent < 38) return "运行时正在装舱";
            if (percent < 78) return "专家正在飞往本地工作区";
            if (percent < 96) return "即将抵达，正在整理最后一排座位";
            return "专家即将落地";
        }

        private static string PreparationDetail(string message)
        {
            if (string.IsNullOrWhiteSpace(message)) return "正在准备本地运行时。";
            if (message.IndexOf("verifying bundled runtime", StringComparison.OrdinalIgnoreCase) >= 0)
                return "正在核验整合包中的运行时文件。";
            if (message.IndexOf("unpacking bundled runtime", StringComparison.OrdinalIgnoreCase) >= 0)
                return "正在解压运行时、内置插件与工作区模板。";
            if (message.IndexOf("bundled runtime ready", StringComparison.OrdinalIgnoreCase) >= 0)
                return "本地运行时已经准备完成。";
            return message;
        }

        public bool ValidateStartupLayoutForTest(Size clientSize, out string error)
        {
            ClientSize = clientSize;
            CreateControl();
            PerformLayout();
            startup.PerformLayout();
            startupLayout.PerformLayout();

            Control[] primaryControls = { startupHeading, startupExplanation, progressText, progressBar, startupActions };
            foreach (Control control in primaryControls)
            {
                if (!ControlFitsInside(control, startup, out error)) return false;
            }

            SetDetailsVisible(true);
            PerformLayout();
            startup.PerformLayout();
            startupLayout.PerformLayout();
            if (!ControlFitsInside(output, startup, out error)) return false;
            SetDetailsVisible(false);
            error = null;
            return true;
        }

        private static bool ControlFitsInside(Control control, Control ancestor, out string error)
        {
            int x = 0;
            int y = 0;
            Control current = control;
            while (current != null && current != ancestor)
            {
                x += current.Left;
                y += current.Top;
                current = current.Parent;
            }
            if (current != ancestor)
            {
                error = control.Name + " is not inside the startup surface.";
                return false;
            }

            Rectangle bounds = new Rectangle(x, y, control.Width, control.Height);
            Rectangle client = ancestor.ClientRectangle;
            if (bounds.Width <= 0 || bounds.Height <= 0 || bounds.Left < 0 || bounds.Top < 0
                || bounds.Right > client.Right || bounds.Bottom > client.Bottom)
            {
                error = control.GetType().Name + " bounds " + bounds + " exceed startup client " + client + ".";
                return false;
            }
            error = null;
            return true;
        }

        private void ShowUpdates()
        {
            EnsureUpdateDialog();
            ShowUpdateDialog();
            if (updateBusy) return;
            if (!string.IsNullOrEmpty(availableUpdate))
            {
                updateDialog.ShowAvailable(availableUpdate, availableUpdateBytes);
                return;
            }
            StartUpdateAction("Check", true);
        }

        private void EnsureUpdateDialog()
        {
            if (updateDialog != null && !updateDialog.IsDisposed) return;
            updateDialog = new UpdateDialog(installedVersion,
                delegate { StartUpdateAction("Check", true); },
                delegate { StartUpdateAction("Install", true); },
                delegate
                {
                    if (!string.IsNullOrEmpty(availableUpdate))
                    {
                        OpenUrl("https://github.com/Headmaster218/MoE4All/releases/tag/" + Uri.EscapeDataString(availableUpdate));
                    }
                });
        }

        private void ShowUpdateDialog()
        {
            if (!updateDialog.Visible) updateDialog.Show(this);
            else
            {
                updateDialog.BringToFront();
                updateDialog.Activate();
            }
        }

        private void StartUpdateAction(string action, bool revealResult)
        {
            if (closing || updateBusy || !ready) return;
            string manifest = Path.Combine(root, "harness", "distribution", "agent-release.json");
            string script = Path.Combine(root, "harness", "scripts", "Check-AgentUpdate.ps1");
            if (!File.Exists(manifest) || !File.Exists(script))
            {
                EnsureUpdateDialog();
                ShowUpdateDialog();
                updateDialog.ShowError("The bundled updater is unavailable. Reinstall the complete MoE4All Agent package.");
                return;
            }

            updateBusy = true;
            updateInstalling = action == "Install";
            updateStatusFailed = false;
            revealUpdateResult = revealResult;
            updateButton.Enabled = false;
            updateButton.Text = updateInstalling ? "Updating..." : "Checking...";
            if (revealResult || updateInstalling)
            {
                EnsureUpdateDialog();
                ShowUpdateDialog();
                if (updateInstalling) updateDialog.ShowStage("download", "Starting update");
                else updateDialog.ShowChecking();
            }

            Thread worker = new Thread(new ThreadStart(delegate { RunUpdateAction(action); }));
            worker.IsBackground = true;
            worker.Name = "MoE4All Agent updater";
            worker.Start();
        }

        private void RunUpdateAction(string action)
        {
            StringBuilder errors = new StringBuilder();
            int exitCode = -1;
            try
            {
                string script = Path.Combine(root, "harness", "scripts", "Check-AgentUpdate.ps1");
                ProcessStartInfo start = PowerShellStart(root);
                start.Arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " + Quote(script)
                    + " -InstallRoot " + Quote(root)
                    + " -ParentPid " + Process.GetCurrentProcess().Id
                    + " -Action " + action;
                start.RedirectStandardOutput = true;
                start.RedirectStandardError = true;
                start.StandardOutputEncoding = new UTF8Encoding(false);
                start.StandardErrorEncoding = new UTF8Encoding(false);
                using (Process updater = new Process())
                {
                    updater.StartInfo = start;
                    updater.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e)
                    {
                        if (e.Data != null && !TryHandleUpdateStatus(e.Data)) WriteMessage(e.Data);
                    };
                    updater.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e)
                    {
                        if (e.Data == null) return;
                        lock (errors) errors.AppendLine(e.Data);
                        WriteMessage(e.Data, true);
                    };
                    if (!updater.Start()) throw new InvalidOperationException("The updater process could not start.");
                    updater.BeginOutputReadLine();
                    updater.BeginErrorReadLine();
                    updater.WaitForExit();
                    updater.WaitForExit();
                    exitCode = updater.ExitCode;
                }
            }
            catch (Exception error)
            {
                lock (errors) errors.AppendLine(error.Message);
            }

            string detail;
            lock (errors) detail = errors.ToString().Trim();
            if (!closing)
            {
                try { BeginInvoke(new Action<string, int, string>(FinishUpdateAction), action, exitCode, detail); }
                catch (InvalidOperationException) { }
            }
        }

        private bool TryHandleUpdateStatus(string message)
        {
            if (!message.StartsWith(UpdatePrefix, StringComparison.Ordinal)) return false;
            string[] parts = message.Split(new char[] { '\t' }, 5);
            long completed;
            long total;
            if (parts.Length != 5 || !long.TryParse(parts[2], out completed) || !long.TryParse(parts[3], out total)) return true;
            if (!closing)
            {
                try
                {
                    BeginInvoke(new Action<string, long, long, string>(ApplyUpdateStatus),
                        parts[1], completed, total, parts[4]);
                }
                catch (InvalidOperationException) { }
            }
            return true;
        }

        private void ApplyUpdateStatus(string kind, long completed, long total, string message)
        {
            if (IsDisposed) return;
            if (kind == "checking")
            {
                if (updateInstalling && updateDialog != null) updateDialog.ShowStage("download", "Checking release information");
                else if (revealUpdateResult && updateDialog != null) updateDialog.ShowChecking();
                return;
            }
            if (kind == "none")
            {
                availableUpdate = null;
                availableUpdateBytes = 0;
                updateButton.Text = "Check updates";
                updateButton.BackColor = Color.FromArgb(50, 52, 56);
                updateButton.FlatAppearance.BorderColor = Color.FromArgb(82, 85, 91);
                if (revealUpdateResult)
                {
                    EnsureUpdateDialog();
                    ShowUpdateDialog();
                    updateDialog.ShowCurrent();
                }
                return;
            }
            if (kind == "available")
            {
                availableUpdate = message;
                availableUpdateBytes = total;
                updateButton.Text = "Update available";
                updateButton.BackColor = Color.FromArgb(62, 91, 76);
                updateButton.FlatAppearance.BorderColor = Color.FromArgb(93, 132, 111);
                EnsureUpdateDialog();
                ShowUpdateDialog();
                updateDialog.ShowAvailable(availableUpdate, availableUpdateBytes);
                return;
            }
            if (kind == "error")
            {
                updateStatusFailed = true;
                if (revealUpdateResult || updateInstalling)
                {
                    EnsureUpdateDialog();
                    ShowUpdateDialog();
                    updateDialog.ShowError(message);
                }
                return;
            }

            EnsureUpdateDialog();
            ShowUpdateDialog();
            if (kind == "download") updateDialog.ShowDownload(completed, total, message);
            else updateDialog.ShowStage(kind, message);
        }

        private void FinishUpdateAction(string action, int exitCode, string error)
        {
            updateBusy = false;
            updateButton.Enabled = ready;
            if (updateInstalling) updateInstalling = false;

            if (updateStatusFailed)
            {
                updateButton.Text = string.IsNullOrEmpty(availableUpdate) ? "Check updates" : "Update available";
                return;
            }
            if (exitCode != 0)
            {
                updateButton.Text = string.IsNullOrEmpty(availableUpdate) ? "Check updates" : "Update available";
                if (revealUpdateResult || action == "Install")
                {
                    EnsureUpdateDialog();
                    ShowUpdateDialog();
                    updateDialog.ShowError(string.IsNullOrEmpty(error) ? "The updater exited before completing." : error);
                }
                return;
            }
            if (action == "Check" && !string.IsNullOrEmpty(availableUpdate) && updateDialog != null)
            {
                updateDialog.ShowAvailable(availableUpdate, availableUpdateBytes);
            }
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
            if (isError) status.Text = "Needs attention";
            else if (ready) status.Text = "Ready";
            else if (!preparingProgress) status.Text = "Starting";
            if (isError) status.ForeColor = Color.FromArgb(235, 112, 112);
            output.AppendText(message + Environment.NewLine);
        }

        public void UpdatePreparationProgress(long completed, long total, string message)
        {
            if (IsDisposed) return;
            if (InvokeRequired)
            {
                try { BeginInvoke(new Action<long, long, string>(UpdatePreparationProgress), completed, total, message); }
                catch (InvalidOperationException) { }
                return;
            }
            preparingProgress = true;
            int value = total <= 0 ? 0 : (int)Math.Max(0, Math.Min(1000, completed * 1000 / total));
            int percent = value / 10;
            progressBar.SetProgress(value);
            progressText.Text = PreparationStage(percent);
            progressDetail.Text = PreparationDetail(message);
            progressPercent.Text = percent + "%";
            status.Text = "Preparing " + percent + "%";
            status.ForeColor = Color.FromArgb(176, 179, 184);
        }

        private void CompletePreparation()
        {
            if (IsDisposed) return;
            if (InvokeRequired)
            {
                try { BeginInvoke(new Action(CompletePreparation)); }
                catch (InvalidOperationException) { }
                return;
            }
            preparingProgress = false;
            progressText.Text = firstLaunch ? "专家已落地，正在唤醒 Agent" : "本地环境就绪，正在唤醒 Agent";
            progressDetail.Text = "正在启动桌面界面与本地服务。";
            progressPercent.Text = "启动中";
            progressBar.SetIndeterminate();
            status.Text = "Starting";
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

                CompletePreparation();
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
                            BeginInvoke(new Action(OnAgentReady));
                            return;
                        }
                    }
                }
                catch { }
                Thread.Sleep(500);
            }
            if (!closing) Fail("The Agent web service did not become ready. See the startup log for details.");
        }

        private void OnAgentReady()
        {
            ready = true;
            status.Text = "Ready";
            status.ForeColor = Color.FromArgb(131, 201, 157);
            updateButton.Enabled = true;
            ActivateWebView();
            StartUpdateAction("Check", false);
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
            preparingProgress = false;
            status.Text = "Needs attention";
            status.ForeColor = Color.FromArgb(235, 112, 112);
            startupHeading.Text = "启动遇到问题";
            startupExplanation.Text = "MoE4All 没有继续运行。具体原因已展开在下方，修正后重新启动即可。";
            progressBar.SetFailed();
            progressText.Text = "需要处理后再继续";
            progressDetail.Text = message;
            progressPercent.Text = "已停止";
            SetDetailsVisible(true);
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

        public string PickEngine()
        {
            if (InvokeRequired) return (string)Invoke(new Func<string>(PickEngine));
            using (OpenFileDialog dialog = new OpenFileDialog())
            {
                dialog.Title = "Select a MoE4All engine package or infr.exe";
                dialog.Filter = "MoE4All engine (*.zip;infr.exe)|*.zip;infr.exe|ZIP archive (*.zip)|*.zip|Engine executable (infr.exe)|infr.exe|All files (*.*)|*.*";
                dialog.Multiselect = false;
                dialog.CheckFileExists = true;
                string downloads = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads");
                dialog.InitialDirectory = Directory.Exists(downloads) ? downloads : root;
                return dialog.ShowDialog(this) == DialogResult.OK ? dialog.FileName : null;
            }
        }
    }

    private sealed class UpdateDialog : Form
    {
        private readonly string installedVersion;
        private readonly Action checkRequested;
        private readonly Action installRequested;
        private readonly Action releaseRequested;
        private readonly Label statusText;
        private readonly Label detailText;
        private readonly Label progressText;
        private readonly ProgressBar progressBar;
        private readonly LinkLabel releaseLink;
        private readonly Button primaryButton;
        private readonly Button secondaryButton;
        private bool hasUpdate;
        private bool installing;

        public UpdateDialog(string currentVersion, Action checkAction, Action installAction, Action releaseAction)
        {
            installedVersion = currentVersion;
            checkRequested = checkAction;
            installRequested = installAction;
            releaseRequested = releaseAction;

            Text = "MoE4All Agent updates";
            StartPosition = FormStartPosition.CenterParent;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;
            ShowInTaskbar = false;
            ClientSize = new Size(560, 330);
            BackColor = Color.FromArgb(31, 32, 35);
            ForeColor = Color.FromArgb(232, 234, 237);
            Font = new Font("Segoe UI", 9F);
            AutoScaleMode = AutoScaleMode.Dpi;

            Label heading = new Label();
            heading.Text = "Agent updates";
            heading.Font = new Font("Segoe UI Semibold", 17F);
            heading.AutoSize = true;
            heading.Location = new Point(28, 24);
            Controls.Add(heading);

            Label version = new Label();
            version.Text = "Installed  " + installedVersion;
            version.ForeColor = Color.FromArgb(166, 170, 176);
            version.AutoSize = true;
            version.Location = new Point(30, 62);
            Controls.Add(version);

            Panel separator = new Panel();
            separator.BackColor = Color.FromArgb(61, 63, 68);
            separator.Location = new Point(28, 91);
            separator.Size = new Size(504, 1);
            Controls.Add(separator);

            statusText = new Label();
            statusText.Text = "Ready to check";
            statusText.Font = new Font("Segoe UI Semibold", 12F);
            statusText.AutoEllipsis = true;
            statusText.Location = new Point(28, 112);
            statusText.Size = new Size(504, 28);
            Controls.Add(statusText);

            detailText = new Label();
            detailText.Text = "Check GitHub for a newer MoE4All Agent package.";
            detailText.ForeColor = Color.FromArgb(180, 184, 190);
            detailText.Location = new Point(28, 145);
            detailText.Size = new Size(504, 44);
            Controls.Add(detailText);

            progressBar = new ProgressBar();
            progressBar.Location = new Point(28, 204);
            progressBar.Size = new Size(504, 14);
            progressBar.Minimum = 0;
            progressBar.Maximum = 1000;
            progressBar.Style = ProgressBarStyle.Continuous;
            Controls.Add(progressBar);

            progressText = new Label();
            progressText.ForeColor = Color.FromArgb(153, 158, 165);
            progressText.AutoEllipsis = true;
            progressText.Location = new Point(28, 225);
            progressText.Size = new Size(330, 20);
            Controls.Add(progressText);

            releaseLink = new LinkLabel();
            releaseLink.Text = "View release page";
            releaseLink.LinkColor = Color.FromArgb(145, 190, 164);
            releaseLink.ActiveLinkColor = Color.FromArgb(177, 216, 194);
            releaseLink.VisitedLinkColor = releaseLink.LinkColor;
            releaseLink.AutoSize = true;
            releaseLink.Location = new Point(30, 247);
            releaseLink.Visible = false;
            releaseLink.LinkClicked += delegate { releaseRequested(); };
            Controls.Add(releaseLink);

            primaryButton = MakeButton("Check for updates", true);
            primaryButton.Location = new Point(346, 279);
            primaryButton.Size = new Size(186, 31);
            primaryButton.Click += delegate
            {
                if (hasUpdate) installRequested();
                else checkRequested();
            };
            Controls.Add(primaryButton);

            secondaryButton = MakeButton("Close", false);
            secondaryButton.Location = new Point(242, 279);
            secondaryButton.Size = new Size(94, 31);
            secondaryButton.Click += delegate { Hide(); };
            Controls.Add(secondaryButton);

            FormClosing += delegate(object sender, FormClosingEventArgs e)
            {
                if (installing)
                {
                    e.Cancel = true;
                    return;
                }
                if (e.CloseReason == CloseReason.UserClosing)
                {
                    e.Cancel = true;
                    Hide();
                }
            };
        }

        private static Button MakeButton(string text, bool primary)
        {
            Button button = new Button();
            button.Text = text;
            button.Height = 31;
            button.FlatStyle = FlatStyle.Flat;
            button.UseVisualStyleBackColor = false;
            button.ForeColor = Color.FromArgb(239, 241, 243);
            button.BackColor = primary ? Color.FromArgb(62, 91, 76) : Color.FromArgb(48, 50, 54);
            button.FlatAppearance.BorderColor = primary ? Color.FromArgb(93, 132, 111) : Color.FromArgb(79, 82, 88);
            return button;
        }

        private void SetProgress(long completed, long total)
        {
            progressBar.MarqueeAnimationSpeed = 0;
            progressBar.Style = ProgressBarStyle.Continuous;
            int value = total <= 0 ? 0 : (int)Math.Max(0, Math.Min(1000, completed * 1000 / total));
            progressBar.Value = value;
            progressText.Text = total > 0
                ? FormatBytes(completed) + " / " + FormatBytes(total) + "  " + (value / 10) + "%"
                : FormatBytes(completed);
        }

        private void SetIndeterminate(string caption)
        {
            progressBar.Value = 0;
            progressBar.Style = ProgressBarStyle.Marquee;
            progressBar.MarqueeAnimationSpeed = 24;
            progressText.Text = caption;
        }

        private static string FormatBytes(long value)
        {
            if (value >= 1024L * 1024L * 1024L) return (value / (1024d * 1024d * 1024d)).ToString("0.0") + " GB";
            if (value >= 1024L * 1024L) return (value / (1024d * 1024d)).ToString("0.0") + " MB";
            if (value >= 1024L) return (value / 1024d).ToString("0.0") + " KB";
            return value + " B";
        }

        private void SetButtons(bool canAct, string primaryText, string secondaryText)
        {
            primaryButton.Text = primaryText;
            primaryButton.Enabled = canAct;
            secondaryButton.Text = secondaryText;
            secondaryButton.Enabled = !installing;
        }

        public void ShowChecking()
        {
            hasUpdate = false;
            installing = false;
            statusText.Text = "Checking for updates";
            detailText.Text = "Contacting the official MoE4All release feed.";
            detailText.ForeColor = Color.FromArgb(180, 184, 190);
            releaseLink.Visible = false;
            SetIndeterminate("Checking release information");
            SetButtons(false, "Checking...", "Close");
        }

        public void ShowCurrent()
        {
            hasUpdate = false;
            installing = false;
            statusText.Text = "You're up to date";
            detailText.Text = installedVersion + " is the newest available Agent version.";
            detailText.ForeColor = Color.FromArgb(180, 184, 190);
            releaseLink.Visible = false;
            SetProgress(1, 1);
            progressText.Text = "No update is required";
            SetButtons(true, "Check again", "Close");
        }

        public void ShowAvailable(string version, long bytes)
        {
            hasUpdate = true;
            installing = false;
            statusText.Text = version + " is available";
            detailText.Text = "Download and replace application files only. Models, conversations and caches stay in place; one previous version is kept for rollback.";
            detailText.ForeColor = Color.FromArgb(180, 184, 190);
            releaseLink.Visible = true;
            SetProgress(0, bytes);
            progressText.Text = bytes > 0 ? "Package size  " + FormatBytes(bytes) : "Ready to download";
            SetButtons(true, "Download and install", "Not now");
        }

        public void ShowDownload(long completed, long total, string message)
        {
            hasUpdate = false;
            installing = true;
            statusText.Text = "Downloading update";
            detailText.Text = "MoE4All Agent remains usable while the package downloads.";
            detailText.ForeColor = Color.FromArgb(180, 184, 190);
            releaseLink.Visible = false;
            SetProgress(completed, total);
            if (total <= 0) progressText.Text = message;
            SetButtons(false, "Updating...", "Please wait");
        }

        public void ShowStage(string stage, string message)
        {
            hasUpdate = false;
            installing = stage != "checking";
            if (stage == "verify")
            {
                statusText.Text = "Verifying download";
                detailText.Text = "Checking the package SHA-256 before any application files are changed.";
            }
            else if (stage == "extract")
            {
                statusText.Text = "Preparing update";
                detailText.Text = "Unpacking and validating the new application beside the current version.";
            }
            else if (stage == "install")
            {
                statusText.Text = "Installing and restarting";
                detailText.Text = "The Agent window will close briefly, switch versions, and reopen automatically.";
            }
            else
            {
                statusText.Text = "Starting update";
                detailText.Text = "Preparing a secure application-only update.";
            }
            detailText.ForeColor = Color.FromArgb(180, 184, 190);
            releaseLink.Visible = false;
            SetIndeterminate(message);
            SetButtons(false, "Updating...", "Please wait");
        }

        public void ShowError(string message)
        {
            hasUpdate = false;
            installing = false;
            statusText.Text = "Update needs attention";
            detailText.Text = message;
            detailText.ForeColor = Color.FromArgb(235, 137, 137);
            releaseLink.Visible = false;
            progressBar.MarqueeAnimationSpeed = 0;
            progressBar.Style = ProgressBarStyle.Continuous;
            progressBar.Value = 0;
            progressText.Text = "No application files were replaced";
            SetButtons(true, "Try again", "Close");
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
                    ThreadPool.QueueUserWorkItem(delegate { HandleSafely(client); });
                }
                catch (SocketException) { if (!stopping) WriteMessage("Desktop bridge stopped unexpectedly.", true); }
                catch (ObjectDisposedException) { }
            }
        }

        private void HandleSafely(TcpClient client)
        {
            try
            {
                Handle(client);
            }
            catch (Exception error)
            {
                if (!(error is IOException) && !(error is SocketException) && !(error is ObjectDisposedException) && !stopping)
                    WriteMessage("Desktop bridge request failed: " + error.Message, true);
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
                else if (path == "/pick-engine") selected = owner.PickEngine();
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
