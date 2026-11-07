// Small Windows GUI bootstrap; uses only the inbox .NET Framework and PowerShell.
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
using System.Windows.Forms;

internal static class WizardEntry {
    [DllImport("user32.dll")] static extern IntPtr SetThreadDpiAwarenessContext(IntPtr context);
    [DllImport("shcore.dll")] static extern int SetProcessDpiAwareness(int awareness);
    [DllImport("user32.dll")] static extern bool SetProcessDPIAware();
    static void InitializeDpi() {
        // This applies only to bootstrap error dialogs. The PowerShell UI host
        // independently initializes its own thread before constructing controls.
        try { if (SetThreadDpiAwarenessContext(new IntPtr(-4)) != IntPtr.Zero) return;
              if (SetThreadDpiAwarenessContext(new IntPtr(-3)) != IntPtr.Zero) return; }
        catch (EntryPointNotFoundException) { }
        try { if (SetProcessDpiAwareness(2) == 0) return; }
        catch (DllNotFoundException) { } catch (EntryPointNotFoundException) { }
        SetProcessDPIAware();
    }
    static string Quote(string value) {
        if (value == null) value = "";
        var b = new StringBuilder("\""); int slashes = 0;
        foreach (char c in value) {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') b.Append('\\', slashes * 2 + 1);
            else b.Append('\\', slashes);
            b.Append(c); slashes = 0;
        }
        return b.Append('\\', slashes * 2).Append('"').ToString();
    }
    static void Append(StringBuilder buffer, string line) {
        if (line == null) return;
        lock (buffer) { buffer.AppendLine(line); if (buffer.Length > 16384) buffer.Remove(0, buffer.Length - 16384); }
    }
    [STAThread] static int Main(string[] args) {
        InitializeDpi();
        try {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string script = Path.Combine(root, "scripts", "infr-wizard.ps1");
            if (!File.Exists(script)) throw new FileNotFoundException("请完整解压安装包。Extract the entire package before starting.", script);
            string system = Environment.GetFolderPath(Environment.SpecialFolder.System);
            if (Environment.Is64BitOperatingSystem && !Environment.Is64BitProcess)
                system = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Windows), "Sysnative");
            var argv = new StringBuilder("-NoLogo -NoProfile -STA -ExecutionPolicy Bypass -File ").Append(Quote(script)).Append(" -WindowWizard");
            foreach (string arg in args) argv.Append(' ').Append(Quote(arg));
            var info = new ProcessStartInfo(Path.Combine(system, "WindowsPowerShell", "v1.0", "powershell.exe"), argv.ToString()) {
                WorkingDirectory = root, UseShellExecute = false, CreateNoWindow = true,
                RedirectStandardOutput = true, RedirectStandardError = true
            };
            var output = new StringBuilder(); var errors = new StringBuilder();
            using (var child = new Process { StartInfo = info }) {
                child.OutputDataReceived += delegate(object sender, DataReceivedEventArgs e) { Append(output, e.Data); };
                child.ErrorDataReceived += delegate(object sender, DataReceivedEventArgs e) { Append(errors, e.Data); };
                child.Start(); child.BeginOutputReadLine(); child.BeginErrorReadLine(); child.WaitForExit();
                int code = child.ExitCode;
                // The wizard has already presented native completion/error screens for these codes.
                if (code == 46) return 1;
                if (code == 0 || code == 42 || code == 43 || code == 44 || code == 45) return 0;
                string detail = errors.Length > 0 ? errors.ToString() : output.ToString();
                if (detail.Length > 6000) detail = detail.Substring(detail.Length - 6000);
                throw new InvalidOperationException("启动失败 / Launch failed (exit " + code + ").\n\n" + detail);
            }
        } catch (Exception error) {
            MessageBox.Show(error.Message, "MoE4All", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
