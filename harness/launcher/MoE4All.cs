using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

internal static class MoE4All
{
    private const int KillOnJobClose = 0x2000;
    private const int ExtendedLimitInformation = 9;
    private static readonly object LogLock = new object();
    private static StreamWriter log;
    private static string logPath;

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
        var start = new ProcessStartInfo(powershell);
        start.WorkingDirectory = root;
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        // Windows PowerShell must rebuild its module path instead of inheriting PowerShell 7's modules.
        start.EnvironmentVariables.Remove("PSModulePath");
        return start;
    }

    private static void WriteMessage(string message, bool error = false)
    {
        lock (LogLock)
        {
            if (error) Console.Error.WriteLine(message);
            else Console.WriteLine(message);
            if (log != null) log.WriteLine(message);
        }
    }

    private static int RunScript(IntPtr job, string root, string script, string arguments)
    {
        var start = PowerShellStart(root);
        // The release's own scripts can carry Mark-of-the-Web after browser download and extraction.
        start.Arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " + Quote(script) + arguments;
        start.RedirectStandardOutput = true;
        start.RedirectStandardError = true;
        start.StandardOutputEncoding = new UTF8Encoding(false);
        start.StandardErrorEncoding = new UTF8Encoding(false);
        using (var child = new Process())
        {
            child.StartInfo = start;
            child.OutputDataReceived += (sender, e) => { if (e.Data != null) WriteMessage(e.Data); };
            child.ErrorDataReceived += (sender, e) => { if (e.Data != null) WriteMessage(e.Data, true); };
            if (!child.Start()) return 4;
            if (!AssignProcessToJobObject(job, child.Handle))
            {
                child.Kill();
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
            child.BeginOutputReadLine();
            child.BeginErrorReadLine();
            child.WaitForExit();
            return child.ExitCode;
        }
    }

    private static void StartUpdateChecker(string root)
    {
        string manifest = Path.Combine(root, "harness", "distribution", "agent-release.json");
        string script = Path.Combine(root, "harness", "scripts", "Check-AgentUpdate.ps1");
        if (!File.Exists(manifest) || !File.Exists(script)) return;
        var start = PowerShellStart(root);
        start.Arguments = "-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File " + Quote(script)
            + " -InstallRoot " + Quote(root) + " -ParentPid " + Process.GetCurrentProcess().Id;
        start.WorkingDirectory = Directory.GetParent(root).FullName;
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        start.WindowStyle = ProcessWindowStyle.Hidden;
        try
        {
            using (var checker = Process.Start(start)) { }
        }
        catch (Exception error)
        {
            WriteMessage("Agent update check could not start: " + error.Message, true);
        }
    }

    private static int Main(string[] arguments)
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
        int result;
        try
        {
            logPath = Path.Combine(root, "harness", "agent-state", "logs", "launcher.log");
            Directory.CreateDirectory(Path.GetDirectoryName(logPath));
            log = new StreamWriter(new FileStream(logPath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite),
                new UTF8Encoding(false));
            log.AutoFlush = true;
            WriteMessage("=== MoE4All Agent launch " + DateTime.UtcNow.ToString("o") + " ===");
            WriteMessage("Install: " + root);
            result = RunAgent(root);
        }
        catch (Exception error)
        {
            WriteMessage("MoE4All Agent could not start: " + error, true);
            result = 1;
        }
        if (result != 0)
        {
            WriteMessage("MoE4All Agent exited with code " + result + ".", true);
            if (log != null) WriteMessage("Startup log: " + logPath, true);
            if (Array.IndexOf(arguments, "--no-pause") < 0 && !Console.IsInputRedirected)
            {
                WriteMessage("Press Enter to close this window.");
                Console.ReadLine();
            }
        }
        if (log != null) log.Dispose();
        return result;
    }

    private static int RunAgent(string root)
    {
        bool first;
        using (var mutex = new Mutex(true, "Local\\MoE4All-Agent", out first))
        {
            if (!first)
            {
                WriteMessage("MoE4All is already running in this session.", true);
                return 2;
            }

            string state = Path.Combine(root, "harness", "agent-state");
            string profile = Path.Combine(state, "home", "profiles", "web", "node_modules");
            string script = Path.Combine(root, "harness", "scripts", "Run-Agent.ps1");
            string prepareScript = Path.Combine(root, "harness", "scripts", "Prepare-BundledRuntime.ps1");
            if (!File.Exists(script) || !File.Exists(prepareScript))
            {
                WriteMessage("The bundled Agent runtime is incomplete. Install the full MoE4All package, not MoE4All.exe alone.", true);
                return 3;
            }

            IntPtr job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
            try
            {
                var limits = new ExtendedLimits();
                limits.BasicLimitInformation.LimitFlags = KillOnJobClose;
                if (!SetInformationJobObject(job, ExtendedLimitInformation, ref limits, (uint)Marshal.SizeOf(typeof(ExtendedLimits))))
                    throw new Win32Exception(Marshal.GetLastWin32Error());

                if (!Directory.Exists(profile))
                    WriteMessage("Preparing the bundled MoE4All Agent runtime.");
                int prepared = RunScript(job, root, prepareScript,
                    " -InstallRoot " + Quote(root) + " -StateRoot " + Quote(state));
                if (prepared != 0 || !Directory.Exists(profile)) return prepared != 0 ? prepared : 3;
                WriteMessage("MoE4All Agent is starting. Close this window to stop it.");
                StartUpdateChecker(root);
                return RunScript(job, root, script, " -StateRoot " + Quote(state) + " -Open");
            }
            finally
            {
                CloseHandle(job);
            }
        }
    }
}
