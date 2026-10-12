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

    private static int RunScript(IntPtr job, string root, string script, string arguments)
    {
        var start = new ProcessStartInfo("powershell.exe");
        start.Arguments = "-NoLogo -NoProfile -NonInteractive -File " + Quote(script) + arguments;
        start.WorkingDirectory = root;
        start.UseShellExecute = false;
        start.CreateNoWindow = true;
        start.RedirectStandardOutput = true;
        start.RedirectStandardError = true;
        using (var child = new Process())
        {
            child.StartInfo = start;
            child.OutputDataReceived += (sender, e) => { if (e.Data != null) Console.WriteLine(e.Data); };
            child.ErrorDataReceived += (sender, e) => { if (e.Data != null) Console.Error.WriteLine(e.Data); };
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
        var start = new ProcessStartInfo("powershell.exe");
        start.Arguments = "-NoLogo -NoProfile -NonInteractive -File " + Quote(script)
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
            Console.Error.WriteLine("Agent update check could not start: " + error.Message);
        }
    }

    private static int Main()
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        bool first;
        using (var mutex = new Mutex(true, "Local\\MoE4All-Agent", out first))
        {
            if (!first)
            {
                Console.Error.WriteLine("MoE4All is already running in this session.");
                return 2;
            }

            string root = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
            string state = Path.Combine(root, "harness", "agent-state");
            string profile = Path.Combine(state, "home", "profiles", "web", "node_modules");
            string script = Path.Combine(root, "harness", "scripts", "Run-Agent.ps1");
            string prepareScript = Path.Combine(root, "harness", "scripts", "Prepare-BundledRuntime.ps1");
            if (!File.Exists(script) || !File.Exists(prepareScript))
            {
                Console.Error.WriteLine("The bundled Agent runtime is incomplete. Install the full MoE4All package, not MoE4All.exe alone.");
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
                    Console.WriteLine("Preparing the bundled MoE4All Agent runtime.");
                int prepared = RunScript(job, root, prepareScript,
                    " -InstallRoot " + Quote(root) + " -StateRoot " + Quote(state));
                if (prepared != 0 || !Directory.Exists(profile)) return prepared != 0 ? prepared : 3;
                Console.WriteLine("MoE4All Agent is starting. Close this window to stop it.");
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
