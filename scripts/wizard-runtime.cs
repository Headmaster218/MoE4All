// Inbox .NET Framework / C# 5. Compile with wizard-theme.cs and wizard-desktop.cs.
// This hosts the WHOLE supplied process (including script prompts and uploads).
// No shell wrapper, external terminal, package, persistent log or engine policy.
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.IO.Pipes;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace MoE4All.Launcher {
    public static class RuntimeArgv {
        // CommandLineToArgvW / Microsoft CRT rules. Always quote, including empty argv.
        // These are process arguments, NOT PowerShell source or cmd.exe escaping.
        public static string Quote(string value) {
            if (value == null) throw new ArgumentNullException("value");
            if (value.IndexOf('\0') >= 0) throw new ArgumentException("An argument contains NUL.", "value");
            var b = new StringBuilder("\""); int slashes = 0;
            foreach (char c in value) {
                if (c == '\\') { slashes++; continue; }
                if (c == '"') { b.Append('\\', slashes * 2 + 1); b.Append(c); }
                else { b.Append('\\', slashes); b.Append(c); }
                slashes = 0;
            }
            b.Append('\\', slashes * 2); return b.Append('"').ToString();
        }
        public static string CommandLine(string executable, string[] arguments) {
            if (String.IsNullOrWhiteSpace(executable)) throw new ArgumentException("Executable is required.", "executable");
            var b = new StringBuilder(Quote(executable));
            foreach (string argument in arguments ?? new string[0]) b.Append(' ').Append(Quote(argument));
            if (b.Length >= 32767) throw new ArgumentException("Windows command line exceeds 32766 characters.");
            return b.ToString();
        }
    }

    public static class NativeRuntime {
        // Also works when called by a PowerShell MTA runspace. The caller remains blocked
        // until the user closes the window. Use a top-level message loop: sequential
        // ShowDialog roots on WinPS 5.1 can inherit a stale modal quit and close at once.
        public static int Run(string executable, string[] arguments, string workingDirectory,
                              string mode, string model, string endpoint, bool english) {
            string[] argv = arguments == null ? new string[0] : (string[])arguments.Clone();
            int result = -1; Exception failure = null;
            ThreadStart show = delegate {
                try {
                    Application.EnableVisualStyles();
                    using (var window = new RuntimeWindow(executable, argv, workingDirectory, mode, model, endpoint, english)) {
                        Application.Run(window); result = window.ExitCode;
                    }
                } catch (Exception ex) { failure = ex; }
            };
            if (Thread.CurrentThread.GetApartmentState() == ApartmentState.STA) show();
            else { var thread = new Thread(show); thread.SetApartmentState(ApartmentState.STA); thread.Start(); thread.Join(); }
            if (failure != null) throw new InvalidOperationException("Native runtime window failed.", failure);
            return result;
        }
    }

    // Bounded transcript. CR rewrites the current line; escape state survives reads.
    // stdout/stderr have separate escape decoders in pipe mode. All mutations are locked.
    internal sealed class RuntimeTranscript {
        internal const int Limit = 196608, LineLimit = 8192;const string ControlPrefix="__MOE4ALL_UI__:",ControlSuffix=":__END_UI__";
        readonly object gate = new object();
        readonly StringBuilder history = new StringBuilder(), line = new StringBuilder();
        readonly Queue<string> controls=new Queue<string>();string controlFrame="";
        readonly Queue<string> benchmarkLines=new Queue<string>();
        int cursor,row=1; long revision, removed;
        internal sealed class Snapshot {
            internal string Text; internal long Revision, Removed;
        }
        void Trim() {
            if (history.Length <= Limit) return;
            int n = history.Length - Limit + 16384;
            while (n < history.Length && history[n - 1] != '\n') n++;
            history.Remove(0, n); removed += n;
        }
        void NewLine() {
            string value=line.ToString();
            if(value.StartsWith("__MOE4ALL_UI__:",StringComparison.Ordinal)||controlFrame.Length>0){controlFrame+=value.Trim();if(controlFrame.EndsWith(":__END_UI__",StringComparison.Ordinal)){if(controls.Count<32)controls.Enqueue(controlFrame.Substring(ControlPrefix.Length,controlFrame.Length-ControlPrefix.Length-ControlSuffix.Length));controlFrame="";}else if(controlFrame.Length>65536)controlFrame="";}
            else {history.Append(value).Append('\n');if(value.StartsWith("Prefill ",StringComparison.Ordinal)||value.StartsWith("Decode ",StringComparison.Ordinal)||value.StartsWith("Preparing ",StringComparison.Ordinal)){if(benchmarkLines.Count>=1024)benchmarkLines.Dequeue();benchmarkLines.Enqueue(value);}}
            line.Length=0;cursor=0;row=Math.Min(LineLimit,row+1);Trim();
        }
        internal string[] DrainControls(){lock(gate){var items=controls.ToArray();controls.Clear();return items;}}
        internal string[] DrainBenchmarkLines(){lock(gate){var items=benchmarkLines.ToArray();benchmarkLines.Clear();return items;}}
        internal void Control(string encoded){lock(gate){if(controls.Count>=128)throw new IOException("Runtime UI inbox is full");controls.Enqueue(encoded);}}
        internal void Note(string text) {
            lock (gate) {
                if (line.Length != 0) NewLine();
                history.Append("[MoE4All] ").Append(text).Append('\n'); Trim(); revision++;
            }
        }
        internal Snapshot Read(long previous) {
            lock (gate) {
                if (previous == revision) return null;
                string partial=line.ToString();
                // A control frame may arrive over several reads (or terminal wraps).
                // Never flash its base64 payload in the user's transcript.
                bool privateFrame=controlFrame.Length>0 || partial.StartsWith(ControlPrefix,StringComparison.Ordinal)
                    || (partial.Length>0 && ControlPrefix.StartsWith(partial,StringComparison.Ordinal));
                return new Snapshot { Text = history.ToString() + (privateFrame?"":partial), Revision = revision, Removed = removed };
            }
        }
        internal sealed class Decoder {
            readonly RuntimeTranscript owner; readonly Action<string> reply;
            readonly StringBuilder escape = new StringBuilder();
            int state; // 0 text, 1 ESC, 2 CSI, 3 OSC/string, 4 string ESC
            internal Decoder(RuntimeTranscript owner, Action<string> reply) { this.owner = owner; this.reply = reply; }
            internal void Feed(string text) {
                // Terminal replies are queued, never synchronously written on a reader.
                lock (owner.gate) {
                    foreach (char c in text) {
                        if (state == 3) { if (c == '\a') state = 0; else if (c == '\x1b') state = 4; continue; }
                        if (state == 4) { state = c == '\\' ? 0 : 3; continue; }
                        if (state == 1) {
                            if (c == '[') { state = 2; escape.Length = 0; }
                            else if (c == ']' || c == 'P' || c == '^' || c == '_') state = 3;
                            else if (c >= ' ' && c <= '/') state = 5; // e.g. ESC ( B
                            else state = 0;
                            continue;
                        }
                        if (state == 5) { state = 0; continue; }
                        if (state == 2) {
                            if (c >= '@' && c <= '~') { Csi(c, escape.ToString()); state = 0; }
                            else if (c == '\x1b') { state = 1; escape.Length = 0; }
                            else if (escape.Length < 96) escape.Append(c);
                            else { state = 0; escape.Length = 0; }
                            continue;
                        }
                        if (c == '\x1b') { state = 1; continue; }
                        if (c == '\r') { owner.cursor = 0; continue; }
                        if (c == '\n') { owner.NewLine(); continue; }
                        if (c == '\b') { owner.cursor = Math.Max(0, owner.cursor - 1); continue; }
                        if (c == '\t') { int n = 8 - owner.cursor % 8; while (n-- > 0) Put(' '); continue; }
                        if (!Char.IsControl(c)) Put(c);
                    }
                    owner.revision++;
                }
            }
            void Put(char c) {
                if ((owner.cursor>=LineLimit&&(owner.line.Length<ControlPrefix.Length||owner.line.ToString(0,ControlPrefix.Length)!=ControlPrefix))||owner.cursor>=65536) {
                    // Even a process producing no newlines has bounded memory and cost.
                    owner.NewLine();
                }
                while (owner.line.Length < owner.cursor) owner.line.Append(' ');
                if (owner.cursor < owner.line.Length) owner.line[owner.cursor] = c;
                else owner.line.Append(c);
                owner.cursor++;
            }
            static int Number(string value, int fallback) {
                int n; return Int32.TryParse(value, out n) ? Math.Max(0, Math.Min(LineLimit, n)) : fallback;
            }
            void Csi(char command, string parameters) {
                string[] parts = parameters.Split(';'); int n = Number(parts[0], 1);
                if (command == 'n' && parameters == "6" && reply != null) reply("\x1b["+owner.row+";" + (owner.cursor + 1) + "R");
                else if (command == 'n' && parameters == "5" && reply != null) reply("\x1b[0n");
                else if (command == 'c' && reply != null && (parameters == "" || parameters == "0")) reply("\x1b[?1;2c");
                else if (command == 'G' || command == '`') owner.cursor = Math.Max(0, n - 1);
                else if (command == 'H' || command == 'f') {int next=Math.Max(1,n);if(next!=owner.row&&owner.line.Length>0)owner.NewLine();owner.row=next;owner.cursor = Math.Max(0, Number(parts.Length > 1 ? parts[1] : "1", 1) - 1);}
                else if(command=='A'||command=='B'||command=='E'||command=='F'){int next=Math.Max(1,Math.Min(LineLimit,owner.row+((command=='A'||command=='F')?-Math.Max(1,n):Math.Max(1,n))));if(owner.line.Length>0)owner.NewLine();owner.row=next;if(command=='E'||command=='F')owner.cursor=0;}
                else if (command == 'C') owner.cursor = Math.Min(LineLimit, owner.cursor + Math.Max(1, n));
                else if (command == 'D') owner.cursor = Math.Max(0, owner.cursor - Math.Max(1, n));
                else if (command == 'K') {
                    int kind = Number(parts[0], 0);
                    if (kind == 0 && owner.cursor < owner.line.Length) owner.line.Length = owner.cursor;
                    else if (kind == 1) for (int i = 0; i <= owner.cursor && i < owner.line.Length; i++) owner.line[i] = ' ';
                    else if (kind == 2) owner.line.Length = 0;
                }
                else if(command=='X'){int end=Math.Min(owner.line.Length,owner.cursor+Math.Max(1,n));for(int i=owner.cursor;i<end;i++)owner.line[i]=' ';}
                else if(command=='J'){int kind=Number(parts[0],0);if(kind==0&&owner.cursor<owner.line.Length)owner.line.Length=owner.cursor;else if(kind==2||kind==3){if(owner.line.Length>0)owner.NewLine();owner.row=1;owner.cursor=0;}}
                // SGR, cursor visibility, full-screen clears/moves, alternate-screen
                // switches and OSC hyperlinks are stripped; history is never erased.
            }
        }
    }

    internal static class RuntimeWin32 {
        internal const uint Suspended = 4, Extended = 0x00080000, NoWindow = 0x08000000;
        internal const uint WaitObject = 0, WaitTimeout = 258, WaitFailed = 0xffffffff;
        internal const uint Cancelled = 1223;
        [StructLayout(LayoutKind.Sequential)] internal struct Coord {
            internal short X, Y; internal Coord(short x, short y) { X = x; Y = y; }
        }
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)] internal struct StartupInfo {
            internal int cb; internal IntPtr reserved, desktop, title;
            internal uint x, y, xSize, ySize, xChars, yChars, fill, flags;
            internal short show, reservedSize; internal IntPtr reserved2, input, output, error;
        }
        [StructLayout(LayoutKind.Sequential)] internal struct StartupInfoEx { internal StartupInfo Info; internal IntPtr Attributes; }
        [StructLayout(LayoutKind.Sequential)] internal struct ProcessInfo { internal IntPtr Process, Thread; internal uint Pid, Tid; }
        [StructLayout(LayoutKind.Sequential)] internal struct BasicLimit {
            internal long ProcessTime, JobTime; internal uint Flags;
            internal UIntPtr MinimumWorkingSet, MaximumWorkingSet; internal uint ActiveProcessLimit;
            internal UIntPtr Affinity; internal uint Priority, Scheduling;
        }
        [StructLayout(LayoutKind.Sequential)] internal struct IoCounters { internal ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes; }
        [StructLayout(LayoutKind.Sequential)] internal struct ExtendedLimit {
            internal BasicLimit Basic; internal IoCounters Io;
            internal UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
        }
        [StructLayout(LayoutKind.Sequential)] internal struct Accounting {
            internal long UserTime, KernelTime, PeriodUserTime, PeriodKernelTime;
            internal uint PageFaults, TotalProcesses, ActiveProcesses, TerminatedProcesses;
        }
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool CloseHandle(IntPtr handle);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool CreatePipe(out IntPtr read, out IntPtr write, IntPtr attributes, uint size);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool SetHandleInformation(IntPtr handle, uint mask, uint flags);
        [DllImport("kernel32.dll")] internal static extern int CreatePseudoConsole(Coord size, IntPtr input, IntPtr output, uint flags, out IntPtr console);
        [DllImport("kernel32.dll")] internal static extern void ClosePseudoConsole(IntPtr console);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool InitializeProcThreadAttributeList(IntPtr list, int count, uint flags, ref IntPtr size);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool UpdateProcThreadAttribute(IntPtr list, uint flags, IntPtr attribute, IntPtr value, IntPtr size, IntPtr previous, IntPtr returned);
        [DllImport("kernel32.dll")] internal static extern void DeleteProcThreadAttributeList(IntPtr list);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] internal static extern bool CreateProcessW(string application, StringBuilder commandLine, IntPtr processSecurity, IntPtr threadSecurity, bool inherit, uint flags, IntPtr environment, string directory, ref StartupInfoEx startup, out ProcessInfo process);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] internal static extern IntPtr CreateJobObjectW(IntPtr security, string name);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool SetInformationJobObject(IntPtr job, int kind, ref ExtendedLimit limit, uint size);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool QueryInformationJobObject(IntPtr job, int kind, out Accounting info, uint size, IntPtr returned);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool TerminateJobObject(IntPtr job, uint code);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool TerminateProcess(IntPtr process, uint code);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern uint ResumeThread(IntPtr thread);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool GetExitCodeProcess(IntPtr process, out uint code);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool ReadFile(IntPtr file, byte[] bytes, int size, out int read, IntPtr overlapped);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool WriteFile(IntPtr file, byte[] bytes, int size, out int written, IntPtr overlapped);
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern IntPtr OpenThread(uint access, bool inherit, uint id);
        [DllImport("kernel32.dll")] internal static extern uint GetCurrentThreadId();
        [DllImport("kernel32.dll", SetLastError = true)] internal static extern bool CancelSynchronousIo(IntPtr thread);
        internal static Win32Exception Error(string operation) { return new Win32Exception(Marshal.GetLastWin32Error(), operation); }
        internal static void Check(bool result, string operation) { if (!result) throw Error(operation); }
        internal static void Close(ref IntPtr handle) {
            IntPtr old = handle; handle = IntPtr.Zero;
            if (old != IntPtr.Zero && old != new IntPtr(-1)) CloseHandle(old);
        }
    }

    // A single synchronous I/O operation per dedicated background thread, as ConPTY
    // requires. CancelSynchronousIo + join precedes closing the pipe (no handle reuse
    // race, no blocked UI, and cancellation is retried across the check/read race).
    internal sealed class RuntimeIo : IDisposable {
        readonly IntPtr pipe; readonly bool writer; readonly Action<string> fault;
        readonly RuntimeTranscript.Decoder decoder;
        readonly object gate = new object(); readonly Queue<string> queue = new Queue<string>();
        readonly AutoResetEvent wake = new AutoResetEvent(false);
        readonly Thread thread; IntPtr nativeThread; volatile bool stopping;
        int queued; bool interrupt; internal volatile bool Ended;
        internal RuntimeIo(IntPtr pipe, bool writer, RuntimeTranscript.Decoder decoder, Action<string> fault) {
            this.pipe = pipe; this.writer = writer; this.decoder = decoder; this.fault = fault;
            thread = new Thread(Work); thread.IsBackground = true; thread.Name = writer ? "MoE4All stdin" : "MoE4All output";
            thread.Start();
        }
        internal bool Enqueue(string text) {
            lock (gate) {
                if (stopping || Ended || text.Length > 65536 - queued) return false;
                queue.Enqueue(text); queued += text.Length; wake.Set(); return true;
            }
        }
        internal void Interrupt() {
            lock (gate) { queue.Clear(); queued = 0; interrupt = true; wake.Set(); }
            IntPtr nt = Interlocked.CompareExchange(ref nativeThread, IntPtr.Zero, IntPtr.Zero);
            if (nt != IntPtr.Zero) RuntimeWin32.CancelSynchronousIo(nt);
        }
        void Work() {
            try {
                Interlocked.Exchange(ref nativeThread, RuntimeWin32.OpenThread(1, false, RuntimeWin32.GetCurrentThreadId()));
                if (nativeThread == IntPtr.Zero) throw RuntimeWin32.Error("OpenThread (I/O cancellation)");
                if (writer) WriteLoop(); else ReadLoop();
            } catch (Exception ex) { Ended = true; if (!stopping) fault(ex.ToString()); }
            finally { Ended = true; }
        }
        void ReadLoop() {
            var utf8 = new UTF8Encoding(false, false).GetDecoder(); byte[] bytes = new byte[8192];
            // A previous read can end in the middle of a UTF-8 scalar; its completion
            // plus a full ASCII chunk can need more chars than the new byte count.
            char[] chars = new char[Encoding.UTF8.GetMaxCharCount(bytes.Length)];
            while (!stopping) {
                int n;
                if (!RuntimeWin32.ReadFile(pipe, bytes, bytes.Length, out n, IntPtr.Zero)) {
                    int error = Marshal.GetLastWin32Error();
                    if (error == 109 || error == 232 || (error == 995 && stopping)) break;
                    throw new Win32Exception(error, "ReadFile");
                }
                if (n == 0) break;
                int count = utf8.GetChars(bytes, 0, n, chars, 0, false);
                if (count != 0) decoder.Feed(new string(chars, 0, count));
            }
            int tail = utf8.GetChars(new byte[0], 0, 0, chars, 0, true);
            if (tail != 0) decoder.Feed(new string(chars, 0, tail));
        }
        void WriteLoop() {
            var encoder = new UTF8Encoding(false, false).GetEncoder();
            while (!stopping) {
                string text = null;
                lock (gate) {
                    if (interrupt) { interrupt = false; encoder.Reset(); text = "\x03"; }
                    else if (queue.Count > 0) { text = queue.Dequeue(); queued -= text.Length; }
                }
                if (text == null) { wake.WaitOne(100); continue; }
                char[] chars = text.ToCharArray(); byte[] bytes = new byte[Encoding.UTF8.GetMaxByteCount(chars.Length)];
                int count = encoder.GetBytes(chars, 0, chars.Length, bytes, 0, false), offset = 0;
                while (offset < count && !stopping) {
                    lock (gate) { if (interrupt) break; }
                    // WriteFile has no array offset; bounded chunks avoid quadratic copies
                    // for a pasted multiline message, and allow interrupt between writes.
                    int size = Math.Min(4096, count - offset); byte[] chunk = new byte[size];
                    Buffer.BlockCopy(bytes, offset, chunk, 0, size); int written;
                    if (!RuntimeWin32.WriteFile(pipe, chunk, size, out written, IntPtr.Zero)) {
                        int error = Marshal.GetLastWin32Error();
                        if (error == 995) {
                            if (stopping) return;
                            // Interrupt can cancel the ETX write itself if it races with
                            // the writer taking the urgent item. Retry that byte; discard
                            // an interrupted ordinary message and take the queued ETX.
                            if (text == "\x03") continue;
                            break;
                        }
                        throw new Win32Exception(error, "WriteFile (stdin)");
                    }
                    if (written == 0) throw new IOException("WriteFile (stdin) made no progress.");
                    offset += written;
                }
            }
        }
        internal bool Join(int milliseconds) { return thread.Join(milliseconds); }
        public void Dispose() {
            stopping = true; wake.Set();
            while (!thread.Join(25)) {
                IntPtr nt = Interlocked.CompareExchange(ref nativeThread, IntPtr.Zero, IntPtr.Zero);
                if (nt != IntPtr.Zero) RuntimeWin32.CancelSynchronousIo(nt);
            }
            RuntimeWin32.Close(ref nativeThread); wake.Dispose();
        }
    }

    // Structured UI messages must not traverse ConPTY: its cursor rewrites and
    // screen scrolling can corrupt a long machine-readable line. Restrict this
    // local pipe to the current Windows user; keep logs and Ctrl+C on ConPTY.
    internal sealed class RuntimeControlChannel : IDisposable {
        internal readonly string Name="moe4all-ui-"+Guid.NewGuid().ToString("N");
        readonly NamedPipeServerStream pipe;readonly Thread worker;volatile bool disposed;
        internal RuntimeControlChannel(RuntimeTranscript transcript){
            var security=new PipeSecurity();security.SetAccessRuleProtection(true,false);
            security.AddAccessRule(new PipeAccessRule(WindowsIdentity.GetCurrent().User,PipeAccessRights.ReadWrite,AccessControlType.Allow));
            pipe=new NamedPipeServerStream(Name,PipeDirection.In,1,PipeTransmissionMode.Byte,PipeOptions.Asynchronous,4096,4096,security);
            worker=new Thread(delegate(){try{
                // Wait on the dedicated worker. Do not dispose an APM result's
                // AsyncWaitHandle: the overlapped completion callback still signals
                // it after pipe.Dispose cancels an unconnected accept operation.
                pipe.WaitForConnection();
                using(var reader=new StreamReader(pipe,new UTF8Encoding(false,true))){var line=new StringBuilder();int c;while(!disposed&&(c=reader.Read())>=0){if(c=='\n'){if(line.Length>0)transcript.Control(line.ToString().TrimEnd('\r'));line.Length=0;}else{if(line.Length>=65536)throw new IOException("Runtime UI frame is too large");line.Append((char)c);}}}
            }catch(Exception ex){if(!disposed)transcript.Note("Runtime UI channel: "+ex.Message);}}){IsBackground=true,Name="MoE4All structured UI messages"};worker.Start();
        }
        public void Dispose(){disposed=true;pipe.Dispose();worker.Join(2000);}
    }

    internal sealed class RuntimeSession : IDisposable {
        readonly string executable, directory; readonly string[] arguments; readonly bool english,preferPty;
        readonly object gate = new object();
        internal readonly RuntimeTranscript Transcript = new RuntimeTranscript();
        Thread worker; RuntimeIo input, output, errorOutput;
        RuntimeControlChannel control;
        IntPtr job, process, pty, inputWrite, outputRead, errorRead;
        int wantStop, wantForce, wantEof; bool resumed, rootExited, killed;
        string lastMonitorError = "";
        internal sealed class State {
            internal string Status, Fault; internal bool Started, Pty, Completed, InputOpen, RootExited, Stopping;
            internal int ExitCode = -1; internal DateTime StopTime;
        }
        readonly State state = new State();
        string T(string zh, string en) { return english ? en : zh; }
        internal RuntimeSession(string executable, string[] arguments, string directory, bool english, bool useTerminal=true) {
            this.executable = executable; this.arguments = arguments == null ? new string[0] : (string[])arguments.Clone();
            this.directory = directory; this.english = english;preferPty=useTerminal;
            state.Status = T("正在准备运行…", "Preparing to run…");
        }
        internal State ReadState() {
            lock (gate) return new State { Status = state.Status, Fault = state.Fault, Started = state.Started,
                Pty = state.Pty, Completed = state.Completed, InputOpen = state.InputOpen, RootExited = state.RootExited,
                Stopping = state.Stopping, StopTime = state.StopTime, ExitCode = state.ExitCode };
        }
        void Status(string value) { lock (gate) state.Status = value; }
        void Note(string value) { Transcript.Note(value); }
        void Fault(string value) { lock (gate) state.Fault = value; Note(value); }
        internal void Start() {
            worker = new Thread(Work); worker.IsBackground = true; worker.Name = "MoE4All process lifetime"; worker.Start();
        }
        internal bool Send(string text, bool line) {
            lock (gate) {
                if (!state.InputOpen || input == null) return false;
                if (line) {
                    text = text.Replace("\r\n", "\n").Replace('\r', '\n');
                    text = text.Replace("\n", state.Pty ? "\r" : "\r\n") + (state.Pty ? "\r" : "\r\n");
                }
                return input.Enqueue(text);
            }
        }
        internal void Stop() {
            lock (gate) { state.Stopping = true; state.StopTime = DateTime.UtcNow; }
            Interlocked.Exchange(ref wantStop, 1);
        }
        internal void Force() { Interlocked.Exchange(ref wantForce, 1); }
        internal void Eof() { Interlocked.Exchange(ref wantEof, 1); }
        void TerminalReply(string text) {
            bool failed;
            lock (gate) failed = input != null && !input.Enqueue(text);
            if (failed) Fault(T("终端响应未能写入输入队列。", "Could not queue a terminal response."));
        }
        static void Pipe(out IntPtr read, out IntPtr write) {
            RuntimeWin32.Check(RuntimeWin32.CreatePipe(out read, out write, IntPtr.Zero, 0), "CreatePipe");
        }
        void CreateJob() {
            job = RuntimeWin32.CreateJobObjectW(IntPtr.Zero, null);
            if (job == IntPtr.Zero) throw RuntimeWin32.Error("CreateJobObject");
            var limit = new RuntimeWin32.ExtendedLimit(); limit.Basic.Flags = 0x00002000; // KILL_ON_JOB_CLOSE; no breakaway
            RuntimeWin32.Check(RuntimeWin32.SetInformationJobObject(job, 9, ref limit, (uint)Marshal.SizeOf(typeof(RuntimeWin32.ExtendedLimit))), "SetInformationJobObject");
        }
        bool TryPty(IntPtr inputRead, IntPtr outputWrite) {
            try {
                int hr = RuntimeWin32.CreatePseudoConsole(new RuntimeWin32.Coord(120, 40), inputRead, outputWrite, 0, out pty);
                if (hr >= 0) return true;
                Note(T("ConPTY 创建失败，改用 UTF-8 管道：", "ConPTY unavailable; using UTF-8 pipes: ") + "HRESULT 0x" + hr.ToString("X8"));
            } catch (EntryPointNotFoundException ex) { Note(T("系统不提供 ConPTY：", "ConPTY is not provided by this OS: ") + ex.Message); }
            catch (DllNotFoundException ex) { Note(T("系统不提供 ConPTY：", "ConPTY is not provided by this OS: ") + ex.Message); }
            return false;
        }
        void Launch() {
            CheckStartupCancellation();
            // Resolve only the executable, never parse it as a command. lpApplicationName
            // is non-null, removing the unquoted-path executable ambiguity.
            string application = ResolveExecutable(executable);
            string command = RuntimeArgv.CommandLine(application, arguments);
            string cwd = String.IsNullOrEmpty(directory) ? Environment.CurrentDirectory : Path.GetFullPath(directory);
            if (!Directory.Exists(cwd)) throw new DirectoryNotFoundException(cwd);
            if (cwd.IndexOf('\0') >= 0) throw new ArgumentException("Working directory contains NUL.");
            CreateJob();
            IntPtr inRead = IntPtr.Zero, outWrite = IntPtr.Zero, errWrite = IntPtr.Zero;
            IntPtr list = IntPtr.Zero, inherited = IntPtr.Zero, environment = IntPtr.Zero; bool initialized = false;
            RuntimeWin32.ProcessInfo pi = new RuntimeWin32.ProcessInfo();
            try {
                Pipe(out inRead, out inputWrite); Pipe(out outputRead, out outWrite);
                bool usePty = preferPty && TryPty(inRead, outWrite);
                lock (gate) state.Pty = usePty;
                var startup = new RuntimeWin32.StartupInfoEx();
                startup.Info.cb = Marshal.SizeOf(typeof(RuntimeWin32.StartupInfoEx));
                IntPtr size = IntPtr.Zero;
                RuntimeWin32.InitializeProcThreadAttributeList(IntPtr.Zero, 1, 0, ref size);
                if (size == IntPtr.Zero) throw RuntimeWin32.Error("InitializeProcThreadAttributeList (size)");
                list = Marshal.AllocHGlobal(size);
                RuntimeWin32.Check(RuntimeWin32.InitializeProcThreadAttributeList(list, 1, 0, ref size), "InitializeProcThreadAttributeList"); initialized = true;
                startup.Attributes = list;
                if (usePty) {
                    RuntimeWin32.Check(RuntimeWin32.UpdateProcThreadAttribute(list, 0, new IntPtr(0x00020016), pty, new IntPtr(IntPtr.Size), IntPtr.Zero, IntPtr.Zero), "UpdateProcThreadAttribute (ConPTY)");
                    // A console parent can have redirected stdio. Explicit null handles
                    // prevent CreateProcess from duplicating those handles over the PTY.
                    // microsoft/terminal discussion #15814 documents this Windows behavior.
                    startup.Info.flags = 0x00000100;
                } else {
                    Pipe(out errorRead, out errWrite);
                    RuntimeWin32.Check(RuntimeWin32.SetHandleInformation(inRead, 1, 1), "SetHandleInformation (stdin)");
                    RuntimeWin32.Check(RuntimeWin32.SetHandleInformation(outWrite, 1, 1), "SetHandleInformation (stdout)");
                    RuntimeWin32.Check(RuntimeWin32.SetHandleInformation(errWrite, 1, 1), "SetHandleInformation (stderr)");
                    inherited = Marshal.AllocHGlobal(IntPtr.Size * 3);
                    Marshal.WriteIntPtr(inherited, 0, inRead); Marshal.WriteIntPtr(inherited, IntPtr.Size, outWrite); Marshal.WriteIntPtr(inherited, IntPtr.Size * 2, errWrite);
                    RuntimeWin32.Check(RuntimeWin32.UpdateProcThreadAttribute(list, 0, new IntPtr(0x00020002), inherited, new IntPtr(IntPtr.Size * 3), IntPtr.Zero, IntPtr.Zero), "UpdateProcThreadAttribute (handle allowlist)");
                    startup.Info.flags = 0x00000100 | 1; startup.Info.show = 0; // STARTF_USESTDHANDLES / SW_HIDE
                    startup.Info.input = inRead; startup.Info.output = outWrite; startup.Info.error = errWrite;
                }
                // A ConPTY attachment provides its own invisible console. NO_WINDOW is
                // only for pipe fallback: it must not suppress the pseudo-console.
                control=new RuntimeControlChannel(Transcript);
                var variables=new SortedDictionary<string,string>(StringComparer.OrdinalIgnoreCase);
                foreach(System.Collections.DictionaryEntry item in Environment.GetEnvironmentVariables())variables[(string)item.Key]=(string)item.Value;
                variables["MOE4ALL_DESKTOP_CONTROL_PIPE"]=control.Name;
                var block=new StringBuilder();foreach(var item in variables)block.Append(item.Key).Append('=').Append(item.Value).Append('\0');block.Append('\0');
                environment=Marshal.StringToHGlobalUni(block.ToString());
                uint flags = RuntimeWin32.Suspended | RuntimeWin32.Extended | 0x00000400u | (usePty ? 0u : RuntimeWin32.NoWindow);
                CheckStartupCancellation();
                RuntimeWin32.Check(RuntimeWin32.CreateProcessW(application, new StringBuilder(command), IntPtr.Zero, IntPtr.Zero,
                    !usePty, flags, environment, cwd, ref startup, out pi), "CreateProcessW");
                process = pi.Process; pi.Process = IntPtr.Zero;
                RuntimeWin32.Check(RuntimeWin32.AssignProcessToJobObject(job, process), "AssignProcessToJobObject (refusing uncontained launch)");
                // Readers start before ResumeThread, so a fast writer cannot deadlock
                // startup or fill a pipe before the UI starts its refresh timer.
                input = new RuntimeIo(inputWrite, true, null, delegate(string e) { InputFault(e); });
                output = new RuntimeIo(outputRead, false, new RuntimeTranscript.Decoder(Transcript, usePty ? (Action<string>)TerminalReply : null), delegate(string e) { Fault("stdout/ConPTY: " + e); });
                if (!usePty) errorOutput = new RuntimeIo(errorRead, false, new RuntimeTranscript.Decoder(Transcript, null), delegate(string e) { Fault("stderr: " + e); });
                CheckStartupCancellation();
                if (RuntimeWin32.ResumeThread(pi.Thread) == UInt32.MaxValue) throw RuntimeWin32.Error("ResumeThread");
                resumed = true;
                lock (gate) { state.Started = true; state.InputOpen = !input.Ended; }
                Note(usePty ? T("ConPTY 已连接：UTF-8 输入输出，支持 Ctrl+C。", "ConPTY connected: UTF-8 input/output, Ctrl+C supported.") :
                    T("管道模式：支持文本/多行输入；不支持 TTY 按键或 Ctrl+C。停止先关闭 stdin (EOF)，必要时确认强停。stdout/stderr 独立读取，跨流顺序无法保证。", "Pipe mode: text/multiline input; no TTY keys or Ctrl+C. Stop first closes stdin (EOF); force requires confirmation. stdout/stderr are read independently; cross-stream ordering is not guaranteed."));
                Status(T("运行中", "Running"));
            } finally {
                if(environment!=IntPtr.Zero)Marshal.FreeHGlobal(environment);
                RuntimeWin32.Close(ref pi.Thread); RuntimeWin32.Close(ref pi.Process);
                if (initialized) RuntimeWin32.DeleteProcThreadAttributeList(list);
                if (list != IntPtr.Zero) Marshal.FreeHGlobal(list);
                if (inherited != IntPtr.Zero) Marshal.FreeHGlobal(inherited);
                RuntimeWin32.Close(ref inRead); RuntimeWin32.Close(ref outWrite); RuntimeWin32.Close(ref errWrite);
            }
        }
        void CheckStartupCancellation() {
            if (Interlocked.CompareExchange(ref wantStop, 0, 0) != 0 || Interlocked.CompareExchange(ref wantForce, 0, 0) != 0)
                throw new OperationCanceledException(T("启动已由用户取消。", "Startup cancelled by the user."));
        }
        static string ResolveExecutable(string value) {
            if (String.IsNullOrWhiteSpace(value) || value.IndexOf('\0') >= 0) throw new ArgumentException("Invalid executable path.");
            if (Path.IsPathRooted(value) || value.IndexOf(Path.DirectorySeparatorChar) >= 0 || value.IndexOf(Path.AltDirectorySeparatorChar) >= 0) {
                string path = Path.GetFullPath(value); if (!File.Exists(path)) throw new FileNotFoundException("Executable not found.", path); return path;
            }
            // Resolve a bare powershell.exe through the current process PATH without
            // calling cmd or relying on CreateProcess's ambiguous executable search.
            var folders = new List<string>(); folders.Add(Environment.SystemDirectory); folders.Add(Environment.CurrentDirectory);
            folders.AddRange((Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator));
            foreach (string folder in folders) {
                if (String.IsNullOrWhiteSpace(folder)) continue;
                string path;
                try { path = Path.GetFullPath(Path.Combine(folder.Trim().Trim('"'), value)); }
                catch (ArgumentException) { continue; } catch (NotSupportedException) { continue; }
                if (File.Exists(path)) return path;
                if (!Path.HasExtension(value) && File.Exists(path + ".exe")) return path + ".exe";
            }
            throw new FileNotFoundException("Executable not found on PATH.", value);
        }
        void InputFault(string error) { lock (gate) state.InputOpen = false; Fault("stdin: " + error); }
        void CloseInput() {
            RuntimeIo io;
            lock (gate) { state.InputOpen = false; io = input; input = null; }
            if (io != null) io.Dispose();
            RuntimeWin32.Close(ref inputWrite);
        }
        uint ActiveProcesses() {
            RuntimeWin32.Accounting info;
            RuntimeWin32.Check(RuntimeWin32.QueryInformationJobObject(job, 1, out info, (uint)Marshal.SizeOf(typeof(RuntimeWin32.Accounting)), IntPtr.Zero), "QueryInformationJobObject");
            return info.ActiveProcesses;
        }
        void Monitor() {
            for (;;) {
                try {
                    if (Interlocked.Exchange(ref wantForce, 0) != 0) {
                        RuntimeWin32.Check(RuntimeWin32.TerminateJobObject(job, RuntimeWin32.Cancelled), "TerminateJobObject");
                        killed = true; Status(T("已确认强停，等待子树退出…", "Force stop confirmed; waiting for the process tree…"));
                        Note(T("已向本窗口拥有的 Job 子树发送强制终止。", "Forced termination sent to this window's Job process tree."));
                    }
                    if (Interlocked.Exchange(ref wantStop, 0) != 0) {
                        lock (gate) { state.Stopping = true; state.StopTime = DateTime.UtcNow; }
                        if (state.Pty) {
                            if (input == null || input.Ended) Fault(T("stdin 已关闭，无法发送 Ctrl+C；可以确认强停。", "stdin is closed; Ctrl+C cannot be sent. Force stop is available."));
                            else { input.Interrupt(); Note(T("已发送 Ctrl+C；排队尚未发送的输入已取消，等待程序自行退出。", "Ctrl+C sent; queued unsent input cancelled. Waiting for the program to exit.")); }
                        } else { CloseInput(); Note(T("管道模式已关闭 stdin (EOF)，等待程序自行退出。", "Pipe stdin closed (EOF); waiting for the program to exit.")); }
                        Status(T("正在停止…（不会自动强停）", "Stopping… (force stop is never automatic)"));
                    }
                    if (Interlocked.Exchange(ref wantEof, 0) != 0) {
                        if (state.Pty) {
                            if (!Send("\x1a\r", false)) Fault(T("无法发送 Ctrl+Z/Enter。", "Could not send Ctrl+Z/Enter."));
                            else Note(T("已发送 Ctrl+Z/Enter（由目标程序决定是否视为 EOF）。", "Ctrl+Z/Enter sent (the target decides whether this means EOF)."));
                        } else { CloseInput(); Note(T("stdin 已关闭 (EOF)。", "stdin closed (EOF).")); }
                    }
                    if (!rootExited) {
                        uint wait = RuntimeWin32.WaitForSingleObject(process, 0);
                        if (wait == RuntimeWin32.WaitFailed) throw RuntimeWin32.Error("WaitForSingleObject");
                        if (wait == RuntimeWin32.WaitObject) {
                            uint code; RuntimeWin32.Check(RuntimeWin32.GetExitCodeProcess(process, out code), "GetExitCodeProcess");
                            rootExited = true;
                            lock (gate) { state.RootExited = true; state.ExitCode = unchecked((int)code); }
                            Note(T("入口进程退出，原始退出码：", "Entry process exited; original exit code: ") + unchecked((int)code) + " (0x" + code.ToString("X8") + ")");
                        }
                    }
                    uint active = ActiveProcesses();
                    if (rootExited && active == 0) return;
                    if (rootExited && !killed) Status(T("入口进程已退出，仍有本任务子进程运行；可继续交互或停止。", "Entry process exited; task descendants remain. Continue interacting or stop them."));
                    lastMonitorError = "";
                } catch (Exception ex) {
                    // A monitoring error must not silently dispose the Job and kill a
                    // running task. Keep the window/ownership alive and allow force stop.
                    string message = ex.ToString();
                    if (lastMonitorError != message) { lastMonitorError = message; Fault(message); }
                    Status(T("进程监控出错；任务仍受本窗口管理，可确认强停。", "Process monitoring failed; the task remains owned here. Force stop requires confirmation."));
                }
                Thread.Sleep(50);
            }
        }
        void CloseConsole() {
            if (pty == IntPtr.Zero) return;
            if (output == null) RuntimeWin32.Close(ref outputRead);
            IntPtr console = pty; pty = IntPtr.Zero;
            // On Windows before 24H2 ClosePseudoConsole can block while producing its
            // final output. Never run it on the reader or UI thread. Keep draining.
            Exception failure = null;
            var close = new Thread(delegate() { try { RuntimeWin32.ClosePseudoConsole(console); } catch (Exception ex) { failure = ex; } });
            close.IsBackground = true; close.Start();
            if (!close.Join(5000)) {
                Fault(T("ConPTY 尾部排空超时；关闭输出管道以解除控制台关闭等待，尾部输出可能不完整。", "ConPTY drain timed out; closing its output pipe to unblock shutdown. Final output may be incomplete."));
                if (output != null) { output.Dispose(); output = null; }
                RuntimeWin32.Close(ref outputRead); close.Join();
            }
            if (failure != null) Fault(failure.ToString());
        }
        void Drain(ref RuntimeIo reader, string name) {
            if (reader == null) return;
            if (!reader.Join(3000)) Fault(name + T(" 排空超时；已取消读取，尾部输出可能不完整。", " drain timed out; read cancelled. Final output may be incomplete."));
            reader.Dispose(); reader = null;
        }
        void Work() {
            try {
                Launch(); Monitor(); Status(T("进程已退出，正在收取剩余输出…", "Process exited; draining remaining output…"));
            } catch (Exception ex) {
                if (ex is OperationCanceledException) Note(ex.Message);
                else Fault(T("启动/运行失败：", "Launch/runtime failure: ") + ex.ToString());
                // Before ResumeThread, no user code or grandchildren can have run.
                // After resume the only main-loop failures are caught by Monitor.
                if (process != IntPtr.Zero && !resumed) {
                    if (!RuntimeWin32.TerminateProcess(process, RuntimeWin32.Cancelled)) Fault(RuntimeWin32.Error("TerminateProcess (suspended startup rollback)").ToString());
                    if (RuntimeWin32.WaitForSingleObject(process, UInt32.MaxValue) == RuntimeWin32.WaitFailed)
                        Fault(RuntimeWin32.Error("WaitForSingleObject (startup rollback)").ToString());
                    uint code;
                    if (RuntimeWin32.GetExitCodeProcess(process, out code)) { lock (gate) state.ExitCode = unchecked((int)code); }
                    else Fault(RuntimeWin32.Error("GetExitCodeProcess (startup rollback)").ToString());
                }
                if (resumed) Monitor();
            } finally {
                // KILL_ON_JOB_CLOSE is the final crash/exception ownership boundary.
                // Ordinary completion reaches here only after Job active count is zero.
                CloseInput(); CloseConsole(); Drain(ref output, "stdout/ConPTY"); Drain(ref errorOutput, "stderr");
                RuntimeWin32.Close(ref outputRead); RuntimeWin32.Close(ref errorRead);
                RuntimeWin32.Close(ref process); RuntimeWin32.Close(ref job);
                lock (gate) {
                    state.Completed = true; state.InputOpen = false;
                    state.Status = state.Fault != null ? T("任务结束（存在错误，请查看日志）", "Task ended with errors; review the log") :
                        killed || state.Stopping ? T("任务已停止", "Task stopped") : state.ExitCode == 0 ? T("任务已完成", "Task completed") : T("任务失败", "Task failed");
                }
                Note(T("结果保留在此窗口；查看后可直接关闭窗口。", "Results remain here. Close the window when you have finished reviewing them."));
            }
        }
        public void Dispose() {
            // Only abnormal external disposal reaches this while running. Normal close
            // is vetoed until completion. Do not leave owned processes or I/O behind.
            if (worker != null && worker.IsAlive) { Force(); worker.Join(); }
            if(control!=null){control.Dispose();control=null;}
        }
    }

    internal sealed class RuntimeLog : WizardRichTextBox {
        [StructLayout(LayoutKind.Sequential)] struct PointNative { internal int X, Y; }
        [StructLayout(LayoutKind.Sequential)] struct ScrollInfo { internal int Size; internal uint Mask; internal int Min, Max; internal uint Page; internal int Pos, Track; }
        [DllImport("user32.dll")] static extern IntPtr SendMessage(IntPtr handle, int message, IntPtr wparam, IntPtr lparam);
        [DllImport("user32.dll", EntryPoint = "SendMessageW")] static extern IntPtr SendPoint(IntPtr handle, int message, IntPtr wparam, ref PointNative point);
        [DllImport("user32.dll")] static extern bool GetScrollInfo(IntPtr handle, int bar, ref ScrollInfo info);
        string rendered = ""; long revision = -1, removed;bool followLatest=true;internal string RenderedText{get{return rendered;}}
        internal RuntimeLog() {
            ReadOnly = true; DetectUrls = false; WordWrap = false; HideSelection = false;
            ScrollBars = RichTextBoxScrollBars.None; BorderStyle = BorderStyle.None; Dock = DockStyle.Fill;
            MaxLength = RuntimeTranscript.Limit + RuntimeTranscript.LineLimit + 32768;UserScrolled+=delegate{followLatest=IsAtLatest;};
        }
        bool AtBottom() {
            return IsAtLatest;
        }
        internal void RefreshFrom(RuntimeTranscript transcript) {
            RuntimeTranscript.Snapshot snapshot = transcript.Read(revision); if (snapshot == null) return;
            revision = snapshot.Revision;
            string next = snapshot.Text; // RichEdit uses LF character indexes.
            if (rendered == next) { removed = snapshot.Removed; return; }
            bool follow = followLatest && SelectionLength == 0;
            PointNative scroll = new PointNative(); SendPoint(Handle, 0x04dd, IntPtr.Zero, ref scroll); // EM_GETSCROLLPOS
            int start = SelectionStart, length = SelectionLength;
            long dropped = Math.Max(0, snapshot.Removed - removed);
            int shifted = (int)Math.Min(rendered.Length, dropped);
            int oldLine = GetLineFromCharIndex(Math.Min(shifted, TextLength));
            int keepLine = GetLineFromCharIndex(GetCharIndexFromPosition(new Point(1, 1)));
            int common = 0, max = Math.Min(rendered.Length, next.Length);
            while (common < max && rendered[common] == next[common]) common++;
            // WM_SETREDRAW(TRUE) adds WS_VISIBLE. Applying it to a hidden log
            // would make it reappear over the chat/results workspace.
            bool redraw=Visible;
            if(redraw)SendMessage(Handle, 0x000b, IntPtr.Zero, IntPtr.Zero);
            try {
                Select(common, TextLength - common); SelectedText = next.Substring(common);
                if (follow) { Select(TextLength, 0); }
                else {
                    int selectionStart = Math.Max(0, Math.Min(TextLength, start - shifted));
                    int selectionEnd = Math.Max(selectionStart, Math.Min(TextLength, start + length - shifted));
                    Select(selectionStart, selectionEnd - selectionStart);
                    if (shifted == 0) SendPoint(Handle, 0x04de, IntPtr.Zero, ref scroll); // EM_SETSCROLLPOS
                    else {
                        int current = GetLineFromCharIndex(GetCharIndexFromPosition(new Point(1, 1)));
                        SendMessage(Handle, 0x00b6, IntPtr.Zero, new IntPtr(Math.Max(0, keepLine - oldLine) - current)); // EM_LINESCROLL
                    }
                }
            } finally { if(redraw){SendMessage(Handle, 0x000b, new IntPtr(1), IntPtr.Zero); Invalidate();} }
            rendered = next; removed = snapshot.Removed;if(follow)ScrollToLatest(); RefreshScrollIndicators();
        }
        internal void Latest() {followLatest=true;ScrollToLatest();}
    }

    internal sealed class RuntimeEntry : WizardTextBox {
        internal bool Raw;
        protected override bool IsInputKey(Keys keyData) {
            Keys key = keyData & Keys.KeyCode;
            if (Raw && (key == Keys.Tab || key == Keys.Escape)) return true;
            return base.IsInputKey(keyData);
        }
    }

    public sealed class RuntimeWindow : WizardWindow {
        readonly RuntimeSession session; readonly bool english; readonly string address,mode;
        ChatClient chat;MarkdownView conversation;BenchmarkResultsView results;Panel display,benchmarkProgress;readonly Label benchmarkStage=new Label(),benchmarkSpeed=new Label();string benchmarkParsed="";
        readonly SoftButton showLogs=new SoftButton(),newChat=new SoftButton(),cancelReply=new SoftButton(),copyResults=new SoftButton();
        readonly WizardRichTextBox promptView=new WizardRichTextBox{ReadOnly=true,Dock=DockStyle.Fill,Visible=false,WordWrap=true};TableLayoutPanel inputRow;Label retention;
        string chatHistory="",chatQuestion="",activePrompt="";bool promptSecret,promptYesNo,chatWasBusy,chatActivated,focusChatInput,viewLogs=true,hasResults;bool followConversation=true;long chatRevision=-1;
        readonly Queue<Dictionary<string,object>> submissionPrompts=new Queue<Dictionary<string,object>>();
        RuntimeSubmissionDialog submissionDialog;bool submissionScheduled;
        ApiStatusView apiStatus;double? chatLiveSpeed;bool apiStatusActivated;
        readonly System.Web.Script.Serialization.JavaScriptSerializer runtimeJson=new System.Web.Script.Serialization.JavaScriptSerializer{MaxJsonLength=131072};
        readonly System.Windows.Forms.Timer timer = new System.Windows.Forms.Timer();
        readonly RuntimeLog log = new RuntimeLog();
        readonly RuntimeEntry entry = new RuntimeEntry(); readonly TextBox endpoint = new WizardTextBox();
        readonly Label heading = new Label(), metadata = new Label(), status = new Label(), detail = new Label();
        readonly SoftButton stop = new SoftButton(), done = new SoftButton(), send = new SoftButton(), eof = new SoftButton(), copy = new SoftButton(), latest = new SoftButton();
        readonly CheckBox raw = new WizardCheckBox(); readonly TableLayoutPanel layout = new TableLayoutPanel();
        readonly List<Font> fonts = new List<Font>();
        bool finishing, prompting, disposed,closeWhenStopped,inputWanted=true,completionPresented; object palette;
        readonly TableLayoutPanel footer=new TableLayoutPanel(),mainActions=new TableLayoutPanel();
        readonly WizardFlowPanel workspaceActions=new WizardFlowPanel();
        Label inputHint;
        string T(string zh, string en) { return english ? en : zh; }
        public int ExitCode { get { return session.ReadState().ExitCode; } }
        // Completion is visible only after the final drained transcript is painted.
        public bool TaskCompleted { get { return completionPresented; } }
        Font OwnedFont(string family, float size, FontStyle style) { var f = new Font(family, size, style); fonts.Add(f); return f; }
        public RuntimeWindow(string executable, string[] arguments, string workingDirectory, string mode, string model, string endpointAddress, bool english) {
            this.english = english;this.mode=mode; address = mode == "server" ? (endpointAddress ?? "") : "";
            session = new RuntimeSession(executable, arguments, workingDirectory, english);
            Name = "RuntimeWindow"; layout.Name = "RuntimeContent";
            log.Name = "RuntimeOutput"; entry.Name = "RuntimeInput";
            endpoint.Name = "RuntimeEndpoint"; copy.Name = "RuntimeCopyEndpoint";
            heading.Name = "RuntimeHeading"; metadata.Name = "RuntimeMetadata";
            status.Name = "RuntimeStatus"; detail.Name = "RuntimeTransport";
            send.Name = "RuntimeSend"; stop.Name = "RuntimeStop"; done.Name = "RuntimeDone";
            eof.Name = "RuntimeEndInput"; latest.Name = "RuntimeLatest"; raw.Name = "RuntimeRawKeys";
            Text = "MoE4All · " + T("运行", "Runtime"); ClientSize = new Size(1000, 720); MinimumSize = new Size(800, 560);
            StartPosition = FormStartPosition.CenterScreen;
            Font = OwnedFont("Microsoft YaHei UI", 10, FontStyle.Regular); KeyPreview = true;
            layout.Dock = DockStyle.Fill; layout.Padding = new Padding(22, 14, 22, 14); layout.ColumnCount = 1; layout.RowCount = 7;
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            foreach (int h in new int[] { 66, 42, 70 }) layout.RowStyles.Add(new RowStyle(SizeType.Absolute, h));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            foreach (int h in new int[] { 24, 100, 96 }) layout.RowStyles.Add(new RowStyle(SizeType.Absolute, h));
            ContentHost.Controls.Add(layout);
            var title = new WizardPanel { Dock = DockStyle.Fill };
            heading.Text = mode=="chat"?T("正在加载模型…","Loading model…"):mode=="server"?T("API 服务","API service"):T("Benchmark 性能测试","Benchmark"); heading.Font = OwnedFont(Font.FontFamily.Name, 20, FontStyle.Bold);
            heading.SetBounds(0, 0, 940, 36); heading.Anchor = AnchorStyles.Left | AnchorStyles.Top | AnchorStyles.Right;
            string modeTitle=mode=="server"?T("本地服务","Local service"):mode=="benchmark"?T("性能测试","Benchmark"):T("本机聊天","Local chat");
            metadata.Text = modeTitle + (String.IsNullOrEmpty(model) ? "" : "  ·  " + Path.GetFileName(model)); metadata.AutoEllipsis = true;
            metadata.SetBounds(2, 40, 940, 24); metadata.Anchor = heading.Anchor;
            title.Controls.Add(heading); title.Controls.Add(metadata); layout.Controls.Add(title, 0, 0);
            var endpointRow = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 2, RowCount = 1, Margin = Padding.Empty };
            endpointRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100)); endpointRow.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 132));
            endpoint.ReadOnly = true; endpoint.Text = address.Length == 0 ? T("本任务未提供服务地址", "No service address supplied") : address;
            var endpointFrame=new WizardInputFrame(endpoint){Dock=DockStyle.Fill,Margin=new Padding(0,6,8,0)};
            Configure(copy, T("复制地址", "Copy address"), false); copy.Enabled = address.Length != 0; copy.Click += CopyAddress;
            endpointRow.Controls.Add(endpointFrame, 0, 0); endpointRow.Controls.Add(copy, 1, 0); layout.Controls.Add(endpointRow, 0, 1);
            if(address.Length==0){endpointRow.Visible=false;layout.RowStyles[1].Height=0;}
            var stateRow = new WizardPanel { Dock = DockStyle.Fill };
            status.SetBounds(0, 4, 940, 27); status.Anchor = heading.Anchor; status.AutoEllipsis = true;
            detail.SetBounds(0, 33, 940, 34); detail.Anchor = heading.Anchor; detail.AutoEllipsis = true;
            stateRow.Controls.Add(status); stateRow.Controls.Add(detail); layout.Controls.Add(stateRow, 0, 2);
            log.Font = OwnedFont("Consolas", 11, FontStyle.Regular); log.Margin = new Padding(0, 3, 0, 3);display=new WizardPanel{Name="RuntimeDisplay",Dock=DockStyle.Fill};display.Controls.Add(log);layout.Controls.Add(display, 0, 3);
            retention = new Label { Dock = DockStyle.Fill, Text = T("消息与确认", "Messages & replies"), AutoEllipsis = true };
            var promptHost=new WizardPanel{Dock=DockStyle.Fill};promptHost.Controls.Add(retention);promptView.Name="RuntimePrompt";promptHost.Controls.Add(promptView);layout.Controls.Add(promptHost,0,4);
            inputRow = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, RowCount = 2, Margin = Padding.Empty };
            inputRow.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            inputRow.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 108)); inputRow.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute, 136));
            inputRow.RowStyles.Add(new RowStyle(SizeType.Absolute, 82)); inputRow.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
            entry.Multiline = true; entry.AcceptsReturn = true; entry.ScrollBars = ScrollBars.None; entry.Dock = DockStyle.Fill; entry.MaxLength = 32768;
            entry.Margin = new Padding(0, 2, 8, 2); entry.KeyDown += InputKeyDown; entry.KeyPress += InputKeyPress;
            Configure(send, T("发送", "Send"), true); send.Click += delegate {if(promptYesNo&&activePrompt.Length>0&&entry.Text.Length==0)entry.Text="y";SendEntry(); };
            Configure(eof, T("结束输入", "End input"), false); eof.Click += EndInput;
            var entryFrame=new WizardInputFrame(entry){Dock=DockStyle.Fill,Margin=new Padding(0,2,8,2)};
            inputRow.Controls.Add(entryFrame, 0, 0); inputRow.Controls.Add(send, 1, 0); inputRow.Controls.Add(eof, 2, 0);
            raw.Text = T("直接发送按键", "Send keys directly"); raw.AutoSize = true;
            raw.CheckedChanged += delegate { entry.Raw = raw.Checked; entry.Focus(); }; raw.Margin = new Padding(0, 6, 4, 0);
            inputRow.Controls.Add(raw, 0, 1);
            var hint = new Label { Text = T("Enter 发送 · 每行单独提交", "Enter sends · each line is submitted separately"), AutoSize = true, Margin = new Padding(0, 6, 0, 0) };
            inputHint=hint;hint.Name="RuntimeInputHint";
            inputRow.Controls.Add(hint, 1, 1); inputRow.SetColumnSpan(hint, 2); layout.Controls.Add(inputRow, 0, 5);
            footer.Name="RuntimeActions";footer.Dock=DockStyle.Fill;footer.ColumnCount=1;footer.RowCount=2;footer.Margin=Padding.Empty;
            footer.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,100));
            footer.RowStyles.Add(new RowStyle(SizeType.Absolute,44));footer.RowStyles.Add(new RowStyle(SizeType.Percent,100));
            mainActions.Dock=DockStyle.Fill;mainActions.ColumnCount=3;mainActions.RowCount=1;mainActions.Margin=Padding.Empty;
            mainActions.ColumnStyles.Add(new ColumnStyle(SizeType.Percent,100));mainActions.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,170));mainActions.ColumnStyles.Add(new ColumnStyle(SizeType.Absolute,132));
            Configure(latest, T("回到底部", "Scroll to latest"), false); latest.Quiet = true; latest.Click += delegate { log.Latest(); };
            Configure(stop, T("停止…", "Stop…"), false); stop.Enabled = false; stop.Click += delegate { ConfirmStop(); };
            Configure(done, T("关闭窗口", "Close window"), true); done.Enabled = true;
            done.Click += delegate { Close(); };
            mainActions.Controls.Add(stop,1,0);mainActions.Controls.Add(done,2,0);footer.Controls.Add(mainActions,0,1);layout.Controls.Add(footer,0,6);
            workspaceActions.Dock=DockStyle.Fill;workspaceActions.WrapContents=false;workspaceActions.Margin=Padding.Empty;
            latest.Dock=DockStyle.None;latest.Size=new Size(150,38);workspaceActions.Controls.Add(latest);
            showLogs.Name="RuntimeShowLogs";showLogs.Text=T("查看日志","Show logs");showLogs.Size=new Size(110,38);showLogs.Quiet=true;showLogs.Click+=delegate{ShowWorkspace(viewLogs?false:true);};workspaceActions.Controls.Add(showLogs);footer.Controls.Add(workspaceActions,0,0);

            if(mode=="benchmark"){
                benchmarkProgress=new WizardPanel{Name="BenchmarkProgress",Dock=DockStyle.Fill};
                var progressHeader=new WizardPanel{Dock=DockStyle.Top,Height=76,Padding=new Padding(12)};
                benchmarkStage.Name="BenchmarkStage";benchmarkStage.Text=T("正在加载模型…","Loading model…");benchmarkStage.Font=OwnedFont(Font.FontFamily.Name,12,FontStyle.Bold);benchmarkStage.SetBounds(12,8,700,28);
                benchmarkSpeed.Name="BenchmarkLastSpeed";benchmarkSpeed.Text=T("完成一项即填入；速度均为聚合 tok/s。","Results fill as completed; all speeds are aggregate tok/s.");benchmarkSpeed.SetBounds(12,40,700,26);
                progressHeader.Controls.Add(benchmarkStage);progressHeader.Controls.Add(benchmarkSpeed);
                progressHeader.SizeChanged+=delegate{benchmarkStage.Width=benchmarkSpeed.Width=Math.Max(1,progressHeader.ClientSize.Width-Pixels(24));};
                results=new BenchmarkResultsView(english){Dock=DockStyle.Fill};benchmarkProgress.Controls.Add(results);benchmarkProgress.Controls.Add(progressHeader);display.Controls.Add(benchmarkProgress);
                log.Visible=false;viewLogs=false;retention.Text=T("测试结果会持续保留；结束后选择是否上传。","Results stay visible. Choose whether to upload when finished.");SetInputVisible(false);showLogs.Visible=true;raw.Visible=false;
                copyResults.Name="BenchmarkCopyResults";copyResults.Text=T("复制表格","Copy table");copyResults.Size=new Size(110,38);copyResults.Quiet=true;copyResults.Click+=delegate{CopyResults();};workspaceActions.Controls.Add(copyResults);
            }            else if(mode=="server"){
                apiStatus=new ApiStatusView(english){Dock=DockStyle.Fill,Visible=false};display.Controls.Add(apiStatus);
                SetInputVisible(false);showLogs.Visible=true;retention.Text=T("通过上面的 API 地址连接应用。","Connect applications to the API address above.");stop.Text=T("停止服务…","Stop service…");
            }
            else if(!String.IsNullOrEmpty(endpointAddress)&&endpointAddress.StartsWith("http://",StringComparison.OrdinalIgnoreCase)){
                apiStatus=new ApiStatusView(english){Visible=false};display.Controls.Add(apiStatus);chat=new ChatClient(endpointAddress,Environment.GetEnvironmentVariable("INFR_API_KEY"));conversation=new MarkdownView{Name="ChatConversation",Dock=DockStyle.Fill,Visible=false};conversation.Font=Font;conversation.UserScrolled+=delegate{followConversation=conversation.IsAtLatest;};display.Controls.Add(conversation);hint.Text=T("Enter 发送 · Shift+Enter 换行","Enter sends · Shift+Enter adds a line");inputRow.SetColumnSpan(hint,3);inputRow.SetColumn(hint,0);raw.Visible=eof.Visible=false;retention.Text=T("加载完成后即可对话 · Enter 发送，Shift+Enter 换行。","Chat after loading · Enter sends; Shift+Enter adds a line.");
                newChat.Name="ChatNewConversation";newChat.Text=T("新对话","New chat");newChat.Size=new Size(90,38);newChat.Quiet=true;newChat.Click+=delegate{if(chat.Reset()){chatHistory="";chatQuestion="";followConversation=true;conversation.SetMarkdown(T("可以开始新的对话。","Start a new conversation."));}};workspaceActions.Controls.Add(newChat);
                cancelReply.Name="ChatCancelReply";cancelReply.Text=T("停止生成","Stop reply");cancelReply.Size=new Size(100,38);cancelReply.Quiet=true;cancelReply.Click+=delegate{chat.Cancel();};workspaceActions.Controls.Add(cancelReply);
            }
            latest.Click+=delegate{if(conversation!=null&&conversation.Visible){followConversation=true;conversation.ScrollToLatest();}if(results!=null&&results.Visible)results.ScrollToLatest();if(apiStatus!=null&&apiStatus.Visible)apiStatus.ScrollToLatest();};
            FormClosing += ClosingRuntime; KeyDown += WindowKeyDown;
            timer.Interval = 100; timer.Tick += delegate { TickRuntime(); };
            ApplyTheme(true);
            layout.SizeChanged+=delegate{FitRuntimeLayout();};
            Shown += delegate { ApplyTheme(true);FitRuntimeLayout();timer.Start(); session.Start(); entry.Focus(); };
        }
        void Configure(SoftButton button, string text, bool primary) { button.Text = text; button.Primary = primary; button.Dock = DockStyle.Fill; button.Margin = new Padding(4, 3, 0, 3); }
        void SetInputVisible(bool value){inputWanted=value;inputRow.Visible=value;FitRuntimeLayout();}
        void FitRuntimeLayout(){
            if(DpiScaling||layout.RowStyles.Count<7||inputHint==null)return;
            int h=Font.Height,buttonHeight=Math.Max(Pixels(36),h+Pixels(18)),hintHeight=h+Pixels(8);
            layout.SuspendLayout();try{
                heading.Height=heading.Font.Height+Pixels(6);metadata.Top=heading.Bottom+Pixels(2);metadata.Height=metadata.Font.Height+Pixels(4);
                heading.Width=metadata.Width=Math.Max(1,heading.Parent.ClientSize.Width-Pixels(4));
                status.Width=detail.Width=Math.Max(1,status.Parent.ClientSize.Width-Pixels(4));
                layout.RowStyles[0].Height=metadata.Bottom+Pixels(4);
                bool compact=layout.ClientSize.Height<Pixels(600);
                detail.Visible=!compact;layout.RowStyles[2].Height=compact?h+Pixels(10):h*2+Pixels(16);
                status.Top=Pixels(2);status.Height=h+Pixels(5);detail.Top=h+Pixels(8);detail.Height=h+Pixels(6);
                layout.RowStyles[4].Height=promptView.Visible?(activePrompt.Length>0?Math.Max(Pixels(72),h*3+Pixels(12)):h*2+Pixels(12)):h+Pixels(6);
                int editorHeight=Math.Max(Pixels(52),h*2+Pixels(12));
                inputRow.RowStyles[0].Height=editorHeight;inputRow.RowStyles[1].Height=hintHeight;
                layout.RowStyles[5].Height=inputWanted?editorHeight+hintHeight+Pixels(6):0;
                foreach(Control button in workspaceActions.Controls){button.Height=buttonHeight;button.Width=Math.Max(Pixels(button==latest?132:90),TextRenderer.MeasureText(button.Text,button.Font).Width+Pixels(28));}
                footer.RowStyles[0].Height=buttonHeight+Pixels(6);layout.RowStyles[6].Height=(buttonHeight+Pixels(6))*2;
                mainActions.ColumnStyles[1].Width=Math.Max(Pixels(154),TextRenderer.MeasureText(stop.Text,stop.Font).Width+Pixels(28));
                mainActions.ColumnStyles[2].Width=Math.Max(Pixels(132),TextRenderer.MeasureText(done.Text,done.Font).Width+Pixels(28));
            }finally{layout.ResumeLayout();}
        }
        protected override void OnDpiLayoutChanged(EventArgs e){base.OnDpiLayoutChanged(e);FitRuntimeLayout();}
        void ShowWorkspace(bool logs){viewLogs=logs;if(benchmarkProgress!=null)benchmarkProgress.Visible=!logs;log.Visible=logs;if(conversation!=null)conversation.Visible=!logs;if(results!=null)results.Visible=true;if(apiStatus!=null)apiStatus.Visible=mode=="server"&&!logs;showLogs.Text=logs?T("返回结果","Back to results"):T("查看日志","Show logs");if(chat!=null&&logs)showLogs.Text=T("返回对话","Back to chat");if(mode=="server"&&logs)showLogs.Text=T("返回状态","Back to status");if(logs)log.Latest();FitRuntimeLayout();}
        void CopyResults(bool selection=false){try{if(results==null||!results.HasRows)return;Clipboard.SetText(selection?results.SelectedTsv():results.ExportTsv(),TextDataFormat.UnicodeText);copyResults.Text=T("已复制","Copied");FitRuntimeLayout();}catch(Exception ex){ReportUi(T("复制失败：","Copy failed: ")+ex.Message);}}
        void ReadControls(){foreach(string data in session.Transcript.DrainControls()){try{var packet=runtimeJson.DeserializeObject(Encoding.UTF8.GetString(Convert.FromBase64String(data))) as Dictionary<string,object>;if(packet==null)continue;string kind=Convert.ToString(packet["kind"]);
                if(kind=="prompt"&&mode=="benchmark"&&packet.ContainsKey("input_type")){submissionPrompts.Enqueue(packet);continue;}
                if(kind=="prompt"){activePrompt=Convert.ToString(packet["prompt"]);promptSecret=packet.ContainsKey("secret")&&(bool)packet["secret"];retention.Text=activePrompt;detail.Text=T("请在下面确认或输入。","Confirm or enter your reply below.");entry.Multiline=false;entry.UseSystemPasswordChar=promptSecret;promptYesNo=(packet.ContainsKey("input_type")&&Convert.ToString(packet["input_type"])=="confirm")||activePrompt.IndexOf("[Y/n]",StringComparison.OrdinalIgnoreCase)>=0||activePrompt.IndexOf("[y/n]",StringComparison.OrdinalIgnoreCase)>=0;promptView.Text=activePrompt+"\n\n"+(packet.ContainsKey("context")?Convert.ToString(packet["context"]):"");promptView.Visible=true;retention.Visible=false;layout.RowStyles[4].Height=132;promptView.Select(0,0);BeginInvoke((Action)delegate{promptView.Select(0,0);promptView.ScrollToCaret();});send.Text=promptYesNo?T("是 / Yes","Yes"):T("确认","Confirm");eof.Text=T("否 / No","No");eof.Visible=promptYesNo;SetInputVisible(true);raw.Visible=false;entry.Focus();}
                else if(kind=="notice"&&activePrompt.Length==0){promptView.Text=Convert.ToString(packet["message"]);promptView.Visible=true;retention.Visible=false;layout.RowStyles[4].Height=72;promptView.ScrollToLatest();}
                else if(kind=="result"&&results!=null){results.LoadRows(packet["rows"]);hasResults=true;showLogs.Visible=true;ShowWorkspace(false);retention.Text=T("速度均为聚合 tok/s；报告已保存在本地，是否上传由你决定。","Results are saved locally. You choose whether to upload.");}
            }catch(Exception ex){ReportUi(T("界面消息无法读取：","Could not read UI message: ")+ex.Message);}}}
        void RefreshApiStatus(){
            if(apiStatus==null)return;
            bool changed=apiStatus.ObserveTranscript(log.RenderedText);
            if(mode=="server"){
                showLogs.Enabled=apiStatus.HasFrame;
                if(apiStatus.HasFrame&&!apiStatusActivated){apiStatusActivated=true;ShowWorkspace(false);}
            }else if(chat!=null&&changed&&chat.Read().Busy){chatLiveSpeed=apiStatus.ActiveDecodeTokensPerSecond;}
        }        void ScheduleSubmissionPrompt(){
            if(submissionScheduled||submissionDialog!=null||submissionPrompts.Count==0||disposed||!Visible||!IsHandleCreated)return;
            submissionScheduled=true;BeginInvoke((MethodInvoker)ShowSubmissionPrompt);
        }
        void ShowSubmissionPrompt(){
            submissionScheduled=false;
            if(disposed||submissionDialog!=null||submissionPrompts.Count==0)return;
            var state=session.ReadState();if(state.Completed||state.Stopping){submissionPrompts.Clear();return;}
            var packet=submissionPrompts.Dequeue();activePrompt=Convert.ToString(packet["prompt"]);
            SetInputVisible(false);promptView.Visible=false;retention.Visible=true;
            try{
                using(var dialog=new RuntimeSubmissionDialog(packet,english)){
                    submissionDialog=dialog;dialog.ShowDialog(this);
                    state=session.ReadState();if(state.Completed||state.Stopping)return;
                    string reply=dialog.Cancelled?(Convert.ToString(packet["input_type"])=="confirm"?"n":"__MOE4ALL_UI_CANCEL_"+Convert.ToString(packet["id"])+"__"):dialog.Reply;
                    if(!session.Send(reply??"",true))ReportUi(T("输入未发送；本地报告仍保留。","Reply was not sent; the local report is retained."));
                }
            }catch(Exception ex){ReportUi(T("提交窗口无法继续：","Submission dialog could not continue: ")+ex.Message);session.Stop();}
            finally{submissionDialog=null;activePrompt="";retention.Text=T("结果保留在此窗口。","Results remain in this window.");}
        }
        void RefreshChat(){if(chat==null)return;var state=chat.Read();if(state.Ready&&!chatActivated){chatActivated=true;focusChatInput=true;retention.Text=T("Enter 发送 · Shift+Enter 换行。","Enter sends · Shift+Enter adds a line.");heading.Text=T("本机聊天","Local chat");ShowWorkspace(false);conversation.SetMarkdown(T("模型已就绪，可以开始对话。","The model is ready. Start a conversation."));chatRevision=state.Revision;}
            if(state.Revision!=chatRevision&&chatQuestion.Length>0){bool follow=followConversation&&conversation.SelectionLength==0;string text=chatHistory+"\n\n### "+T("你","You")+"\n\n"+chatQuestion+"\n\n### MoE4All\n\n"+(state.Reasoning.Length>0?"> "+T("思考过程","Reasoning")+"\n\n"+state.Reasoning+"\n\n---\n\n":"")+state.Answer;if(state.Fault.Length>0)text+="\n\n> "+T("请求未完成：","Request incomplete: ")+state.Fault;conversation.SetMarkdown(text);if(follow)conversation.ScrollToLatest();chatRevision=state.Revision;if(!state.Busy&&chatWasBusy){chatHistory=text;chatQuestion="";focusChatInput=true;}}
            if(state.HasTimings){retention.Text=T("本轮 · Prefill ","This reply · Prefill ")+state.PrefillTokensPerSecond.ToString("F1",System.Globalization.CultureInfo.InvariantCulture)+" tok/s · Decode "+state.DecodeTokensPerSecond.ToString("F1",System.Globalization.CultureInfo.InvariantCulture)+" tok/s · "+state.GeneratedTokens.ToString("N0",System.Globalization.CultureInfo.InvariantCulture)+" tokens";}
            else if(state.Busy){retention.Text=chatLiveSpeed.HasValue&&chatLiveSpeed.Value>0?T("本轮实时 Decode ","Live Decode ")+chatLiveSpeed.Value.ToString("F1",System.Globalization.CultureInfo.InvariantCulture)+" tok/s":T("本轮正在生成…","Generating this reply…");}else if(state.Ready){retention.Text=T("Enter 发送 · Shift+Enter 换行。","Enter sends · Shift+Enter adds a line.");}
            showLogs.Enabled=state.Ready;chatWasBusy=state.Busy;cancelReply.Enabled=state.Busy;newChat.Enabled=state.Ready&&!state.Busy;heading.Text=!state.Ready?T("正在加载模型…","Loading model…"):state.Busy?T("正在生成…","Generating…"):T("本机聊天","Local chat");
        }
        void PaintControls(Control parent, Color surface, Color text) {
            foreach (Control child in parent.Controls) {
                child.BackColor = surface; child.ForeColor = text; PaintControls(child, surface, text);
            }
        }
        void ApplyTheme(bool force) {
            // The base chrome subscribes to theme/system changes. Sampling the immutable
            // current palette handles ordering without another cross-thread UI event.
            var p = WizardTheme.Current;
            if (!force && Object.ReferenceEquals(palette, p)) return; palette = p;
            BackColor = p.Background; ForeColor = p.Text; PaintControls(layout, p.Background, p.Text);
            log.BackColor = p.Surface; log.ForeColor = p.Text;
            entry.BackColor = p.Surface; entry.ForeColor = p.Text;
            endpoint.BackColor = p.SurfaceAlt; endpoint.ForeColor = p.Text;
            metadata.ForeColor = p.Muted; detail.ForeColor = p.Muted;
            foreach (SoftButton b in new SoftButton[] { stop, done, send, eof, copy, latest }) { b.BackColor = p.Background; b.ForeColor = p.Text; b.Invalidate(); }
            Invalidate(true);
        }
        void RefreshBenchmark(){
            if(mode!="benchmark")return;
            foreach(string line in session.Transcript.DrainBenchmarkLines())if(!hasResults&&results.ObserveLine(line))copyResults.Text=T("复制表格","Copy table");
            copyResults.Enabled=results.HasRows;
            if(log.RenderedText==benchmarkParsed)return;benchmarkParsed=log.RenderedText;
            string text=benchmarkParsed.Length>5000?benchmarkParsed.Substring(benchmarkParsed.Length-5000):benchmarkParsed;
            foreach(string rawLine in text.Split('\n')){
                string line=rawLine.Trim();
                if(line.StartsWith("Preparing ",StringComparison.Ordinal))benchmarkStage.Text=T("正在准备 ","Preparing ")+line.Substring(10);
                else if(line.StartsWith("Prefill warmup",StringComparison.OrdinalIgnoreCase))benchmarkStage.Text=T("Prefill 预热","Prefill warmup");
                else if(line.StartsWith("Prefill ",StringComparison.Ordinal)||line.StartsWith("Decode ",StringComparison.Ordinal)){
                    var stage=System.Text.RegularExpressions.Regex.Match(line,@"^(Prefill|Decode) (\d+)(?: x(\d+))?");
                    if(!stage.Success)continue;
                    benchmarkStage.Text=stage.Groups[1].Value+" · "+stage.Groups[2].Value+" tokens"+(stage.Groups[3].Success?" · x"+stage.Groups[3].Value:"");
                    var speed=System.Text.RegularExpressions.Regex.Match(line,@"([0-9]+(?:\.[0-9]+)?) tok/s");
                    if(speed.Success)benchmarkSpeed.Text=T("上项完成：","Last completed: ")+benchmarkStage.Text+" · "+speed.Value;
                }
            }
            if(hasResults){benchmarkStage.Text=T("测试完成","Benchmark completed");heading.Text=T("Benchmark 结果","Benchmark results");}
        }
        void TickRuntime() {
            if (disposed) return;
            ApplyTheme(false); log.RefreshFrom(session.Transcript);ReadControls();RefreshApiStatus();RefreshChat();RefreshBenchmark();
            RuntimeSession.State s = session.ReadState();
            if(submissionDialog!=null&&(s.Completed||s.Stopping))submissionDialog.Dismiss();
            completionPresented=s.Completed;
            status.Text = s.Status + ((s.RootExited || s.Completed)&&!s.Stopping ? T("  ·  退出码 ", "  ·  Exit code ") + s.ExitCode : "");
            if(activePrompt.Length==0)detail.Text = s.Fault != null ? T("发生错误，详情见日志。", "An error occurred; details are in the log.") : !s.Started ? T("正在创建受控进程…", "Creating the owned process…") :
                mode=="server"?T("各槽位速度与上下文会自动更新。","Slot speeds and context update automatically."):
                s.Pty ? T("后续消息、确认与上传都在此窗口继续。", "Messages, confirmations and uploads continue in this window.") :
                T("当前系统使用兼容模式，无法正常停止时会再次询问。", "Compatibility mode: you will be asked again if a force stop is needed.");
            if(chat!=null){var connection=chat.Read();if(!connection.Ready&&connection.Fault.Length>0){heading.Text=T("聊天连接失败","Chat connection failed");detail.Text=connection.Fault;}} bool canInput = s.Started && !s.Completed && !s.Stopping && s.InputOpen;if(chat!=null&&activePrompt.Length==0){var c=chat.Read();canInput=c.Ready&&!c.Busy&&!s.Completed&&!s.Stopping;}
            entry.Enabled = send.Enabled = eof.Enabled = canInput;if(focusChatInput&&canInput){entry.Focus();focusChatInput=false;} raw.Enabled = canInput && s.Pty;
            if(s.Stopping){newChat.Enabled=false;cancelReply.Enabled=false;} stop.Enabled = !s.Completed&&(!s.Stopping||(DateTime.UtcNow-s.StopTime).TotalSeconds>=5);
            string oldStopCaption=stop.Text,oldDoneCaption=done.Text;
            stop.Text = s.Stopping ? ((DateTime.UtcNow-s.StopTime).TotalSeconds>=5?T("强制停止…","Force stop…"):T("正在停止…","Stopping…")) : mode=="server"?T("停止服务","Stop service"):mode=="chat"?T("停止模型","Stop model"):T("停止测试","Stop benchmark");
            bool forceAvailable = s.Stopping && !s.Completed && (DateTime.UtcNow-s.StopTime).TotalSeconds>=5;
            done.Enabled = !closeWhenStopped || forceAvailable;
            done.Text=closeWhenStopped?(forceAvailable?T("强停并关闭…","Force close…"):T("正在关闭…","Closing…")):T("关闭窗口","Close window");
            if(oldStopCaption!=stop.Text||oldDoneCaption!=done.Text)FitRuntimeLayout();
            if(s.Completed){stop.Visible=false;newChat.Visible=false;cancelReply.Visible=false;if(inputWanted)SetInputVisible(false);retention.Text=mode=="chat"?T("对话记录已保留，可查看或复制。","The conversation remains available to review and copy."):mode=="server"?T("服务已停止，日志保留在此窗口。","Service stopped. Logs remain in this window."):T("测试已结束，结果保留在本地。","The test has ended. Results are saved locally.");}
            if (s.Completed) heading.Text = mode=="server"?T("服务已停止","Service stopped"):mode=="benchmark"?T("Benchmark 结果","Benchmark results"):T("对话已结束","Chat ended");
            if(closeWhenStopped){status.Text=forceAvailable?T("任务未响应停止；可选择强停并关闭。","The task has not stopped. Force close is available."):T("正在停止任务，结束后自动关闭。","Stopping the task; the window will close automatically.");if(s.Completed){finishing=true;Close();}}
            ScheduleSubmissionPrompt();
        }
        void ReportUi(string message) { session.Transcript.Note(message); }
        void SendText(string text, bool line) {
            if (!session.Send(text, line)) ReportUi(T("输入未发送：进程输入已关闭或队列已满，请稍后重试。", "Input was not sent: stdin is closed or the queue is full. Try again shortly."));
        }
        void SendEntry() {
            if (!session.ReadState().InputOpen) return;
            string text = entry.Text;
            if(chat!=null&&activePrompt.Length==0){if(chat.Ask(text)){chatQuestion=text;chatWasBusy=true;chatLiveSpeed=null;followConversation=true;entry.Clear();ShowWorkspace(false);}return;}
            if(activePrompt.Length>0){if(text.IndexOfAny(new[]{'\r','\n'})>=0){ReportUi(T("请一次只输入一行；内容已保留。","Enter one line at a time; your text is retained."));return;}if(session.Send(text,true)){activePrompt="";entry.Clear();entry.UseSystemPasswordChar=false;entry.Multiline=true;if(mode!="chat")SetInputVisible(false);retention.Text=T("正在处理…","Working…");promptView.Visible=false;retention.Visible=true;layout.RowStyles[4].Height=24;promptYesNo=false;send.Text=T("发送","Send");eof.Text=T("结束输入","End input");eof.Visible=mode=="chat"&&chat==null;}return;}
            if (session.Send(text, !raw.Checked)) entry.Clear();
            else ReportUi(T("输入未发送：输入已关闭或队列已满。文本已保留。", "Input not sent: stdin is closed or the queue is full. Your text has been kept."));
            entry.Focus();
        }
        void CopyAddress(object sender, EventArgs e) {
            try { Clipboard.SetText(address); copy.Text = T("已复制", "Copied"); }
            catch (Exception ex) { ReportUi(T("复制地址失败：", "Copy address failed: ") + ex.Message); }
        }
        void InputKeyDown(object sender, KeyEventArgs e) {
            if (e.Control && e.KeyCode == Keys.C) {
                if (entry.SelectionLength == 0) { e.SuppressKeyPress = true; ConfirmStop(); } return;
            }
            if (!raw.Checked) {
                if (e.KeyCode == Keys.Enter && !e.Shift) { e.SuppressKeyPress = true; SendEntry(); }
                return;
            }
            if (e.Control && e.KeyCode == Keys.V) {
                e.SuppressKeyPress = true;
                try { SendText(Clipboard.GetText().Replace("\r\n", "\n").Replace('\r', '\n').Replace('\n', '\r'), false); }
                catch (Exception ex) { ReportUi(T("粘贴失败：", "Paste failed: ") + ex.Message); }
                return;
            }
            string sequence = null;
            switch (e.KeyCode) {
                case Keys.Up: sequence = "\x1b[A"; break; case Keys.Down: sequence = "\x1b[B"; break;
                case Keys.Right: sequence = "\x1b[C"; break; case Keys.Left: sequence = "\x1b[D"; break;
                case Keys.Home: sequence = "\x1b[H"; break; case Keys.End: sequence = "\x1b[F"; break;
                case Keys.Insert: sequence = "\x1b[2~"; break; case Keys.Delete: sequence = "\x1b[3~"; break;
                case Keys.PageUp: sequence = "\x1b[5~"; break; case Keys.PageDown: sequence = "\x1b[6~"; break;
                case Keys.F1: sequence = "\x1bOP"; break; case Keys.F2: sequence = "\x1bOQ"; break;
                case Keys.F3: sequence = "\x1bOR"; break; case Keys.F4: if (!e.Alt) sequence = "\x1bOS"; break;
                case Keys.F5: sequence = "\x1b[15~"; break; case Keys.F6: sequence = "\x1b[17~"; break;
                case Keys.F7: sequence = "\x1b[18~"; break; case Keys.F8: sequence = "\x1b[19~"; break;
                case Keys.F9: sequence = "\x1b[20~"; break; case Keys.F10: sequence = "\x1b[21~"; break;
                case Keys.F11: sequence = "\x1b[23~"; break; case Keys.F12: sequence = "\x1b[24~"; break;
                case Keys.Escape: sequence = "\x1b"; break;
                case Keys.Tab: sequence = e.Shift ? "\x1b[Z" : "\t"; break;
            }
            if (sequence != null) { e.SuppressKeyPress = true; SendText(sequence, false); }
        }
        void InputKeyPress(object sender, KeyPressEventArgs e) {
            if (!raw.Checked) return; e.Handled = true;
            if (e.KeyChar == '\x03') return; // Ctrl+C always goes through confirmation.
            SendText(e.KeyChar == '\b' ? "\x7f" : e.KeyChar.ToString(), false);
        }
        void WindowKeyDown(object sender, KeyEventArgs e) {
            if(e.Control&&e.KeyCode==Keys.C&&apiStatus!=null&&apiStatus.Visible&&apiStatus.ContainsFocus){e.SuppressKeyPress=true;try{Clipboard.SetText(apiStatus.SelectedTsv());}catch(Exception ex){ReportUi(ex.Message);}return;}
            if(e.Control&&e.KeyCode==Keys.C&&results!=null&&results.Visible&&(results.ContainsFocus||copyResults.Focused)){e.SuppressKeyPress=true;CopyResults(results.ContainsFocus);return;}
            if (e.Control && e.KeyCode == Keys.C && !entry.Focused && !(log.Focused && log.SelectionLength > 0) && !(endpoint.Focused && endpoint.SelectionLength > 0) && !(conversation!=null&&conversation.Focused&&conversation.SelectionLength>0) && !(promptView.Focused&&promptView.SelectionLength>0)) {
                e.SuppressKeyPress = true; ConfirmStop();
            }
        }
        bool Confirm(string title, string question) {
            // Named WinForms controls let a caller's in-process timer inspect/accept a
            // confirmation in fake-executable tests without desktop input injection.
            using (var prompt = new WizardWindow { ShowWindowButtons = false }) {
                prompt.Name = "RuntimeConfirm"; prompt.Text = title; prompt.Font = Font;
                prompt.ClientSize = new Size(560, 270); prompt.MinimumSize = new Size(480, 250);
                prompt.StartPosition = FormStartPosition.CenterParent; prompt.ShowInTaskbar = false;
                prompt.MinimizeBox = false; prompt.MaximizeBox = false;
                var message = new Label { Name = "RuntimeConfirmQuestion", Text = question, Dock = DockStyle.Fill, Padding = new Padding(22, 20, 22, 10) };
                var buttons = new WizardFlowPanel { Dock = DockStyle.Bottom, Height = 62, FlowDirection = FlowDirection.RightToLeft, Padding = new Padding(12, 7, 12, 8) };
                var cancel = new SoftButton { Name = "RuntimeConfirmCancel", Text = T("取消", "Cancel"), Size = new Size(124, 40), DialogResult = DialogResult.Cancel };
                var accept = new SoftButton { Name = "RuntimeConfirmAccept", Text = T("确认", "Confirm"), Size = new Size(124, 40), Primary = true, DialogResult = DialogResult.OK };
                buttons.Controls.Add(cancel); buttons.Controls.Add(accept);
                prompt.ContentHost.Controls.Add(message); prompt.ContentHost.Controls.Add(buttons);
                prompt.AcceptButton = cancel; prompt.CancelButton = cancel;
                prompt.Shown += delegate { cancel.Focus(); };
                return prompt.ShowDialog(this) == DialogResult.OK;
            }
        }
        void EndInput(object sender, EventArgs e) {if(activePrompt.Length>0&&promptYesNo){entry.Text="n";SendEntry();return;}
            RuntimeSession.State s = session.ReadState(); if (!s.InputOpen || prompting) return;
            prompting = true;
            try {
                string message = s.Pty ? T("发送 Ctrl+Z/Enter？目标程序决定是否将其视为输入结束。", "Send Ctrl+Z/Enter? The target decides whether this means end of input.") :
                    T("关闭本任务的 stdin (EOF)？关闭后不能再发送输入。", "Close this task's stdin (EOF)? Further input will be unavailable.");
                if (Confirm(T("结束输入", "End input"), message)) session.Eof();
            } finally { prompting = false; }
        }
        void ConfirmStop() {
            RuntimeSession.State s = session.ReadState(); if (s.Completed || prompting) return;
            if(!s.Stopping){if(chat!=null)chat.Cancel();session.Stop();return;}
            if((DateTime.UtcNow-s.StopTime).TotalSeconds<5)return;
            prompting = true;
            try {
                if (!Confirm(T("强制停止", "Force stop"),T("任务尚未退出。强制停止本窗口启动的任务？未完成的工作将中断。","The task has not exited. Force stop this window's task? Unfinished work will be interrupted."))) return;
                if (session.ReadState().Completed) return;
                session.Force();
            } finally { prompting = false; }
        }
        void ClosingRuntime(object sender, FormClosingEventArgs e) {
            if (finishing||session.ReadState().Completed) return;
            e.Cancel = true;
            if(prompting)return;
            if(closeWhenStopped){ConfirmStop();return;}
            var s=session.ReadState();
            if(!s.Stopping){
                prompting=true;
                try{if(!Confirm(T("关闭窗口","Close window"),T("停止当前任务并关闭窗口？","Stop the current task and close this window?")))return;}
                finally{prompting=false;}
            }
            closeWhenStopped=true;
            if(chat!=null)chat.Cancel();
            if(!s.Stopping)session.Stop();
            else ConfirmStop();
        }
        protected override void Dispose(bool disposing) {
            if (disposing && !disposed) {
                disposed = true; timer.Stop(); timer.Dispose();if(chat!=null)chat.Dispose(); session.Dispose();
                base.Dispose(true); foreach (Font font in fonts) font.Dispose(); return;
            }
            base.Dispose(disposing);
        }
    }
}
