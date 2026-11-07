[CmdletBinding()]
param(
    [string]$EvidenceDirectory = 'D:\AIinfr\infr\local-work\evidence\moe4all\20261105-031000-runtime-experience-c81e\markdown-tests'
)

# Standalone CPU/WinForms tests; does not load the runtime, model, GPU or network.
# Run in a fresh powershell.exe process. All generated files stay in local-work.
$ErrorActionPreference = 'Stop'
$workRoot = 'D:\AIinfr\infr\local-work'
$taskId = '20261105-031000-runtime-experience-c81e'
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Get-Content -LiteralPath 'D:\AIinfr\disk_usage_rule.md' | Out-Null

function Assert-OutputPath([string]$Path, [string]$Root) {
    $full = [IO.Path]::GetFullPath($Path).TrimEnd('\')
    $allowed = [IO.Path]::GetFullPath($Root).TrimEnd('\')
    if ($full -ne $allowed -and -not $full.StartsWith($allowed + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw ('Output path outside allowed task directory: ' + $full)
    }
    $check = $full
    while ($check) {
        if (Test-Path -LiteralPath $check) {
            $item = Get-Item -LiteralPath $check -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw ('Output path traverses a reparse point: ' + $check)
            }
        }
        $parent = [IO.Path]::GetDirectoryName($check)
        if ($parent -eq $check) { break }
        $check = $parent
    }
    return $full
}

$allowedEvidence = Join-Path $workRoot ('evidence\moe4all\' + $taskId)
$EvidenceDirectory = Assert-OutputPath $EvidenceDirectory $allowedEvidence
$tmpRoot = Join-Path $workRoot ('tmp\moe4all\' + $taskId)
$runId = 'markdown-' + $PID + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
$tmpDirectory = Assert-OutputPath (Join-Path $tmpRoot $runId) $tmpRoot
$evidenceRun = Assert-OutputPath (Join-Path $EvidenceDirectory $runId) $allowedEvidence
$drive = New-Object IO.DriveInfo ([IO.Path]::GetPathRoot($workRoot))
if ($drive.AvailableFreeSpace -lt 64MB) { throw 'At least 64 MiB free space is required.' }
[void][IO.Directory]::CreateDirectory($tmpDirectory)
[void][IO.Directory]::CreateDirectory($evidenceRun)
Write-Output ('Output: ' + $evidenceRun)
Write-Output ('Temp: ' + $tmpDirectory + '; build cache/KV/profile: not used')

$oldEnvironment = @{}
foreach ($key in @('TEMP', 'TMP', 'TMPDIR', 'MOE4ALL_WORK_ROOT')) {
    $oldEnvironment[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
}
$started = [DateTime]::UtcNow.ToString('o')
$manifest = [ordered]@{
    task = $taskId
    purpose = 'Standalone Markdown parser, RTF safety and WinForms integration tests'
    startedUtc = $started
    status = 'running'
    repository = $repoRoot
    owner = 'Test-WizardMarkdown.ps1'
    temp = $tmpDirectory
    evidence = $evidenceRun
    command = 'powershell.exe -NoLogo -NoProfile -NonInteractive -File scripts/Test-WizardMarkdown.ps1'
    scope = 'CPU only; no runtime, services, models, network or GPU'
    processId = $PID
    processStartedUtc = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
}

try {
    $env:TEMP = $tmpDirectory
    $env:TMP = $tmpDirectory
    $env:TMPDIR = $tmpDirectory
    $env:MOE4ALL_WORK_ROOT = $workRoot
    $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
    if (-not (Test-Path -LiteralPath $compiler)) {
        $compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe'
    }
    if (-not (Test-Path -LiteralPath $compiler)) { throw '.NET Framework C# compiler is unavailable.' }
    $source = Join-Path $tmpDirectory 'MarkdownHarness.cs'
    $exe = Join-Path $tmpDirectory 'MarkdownHarness.exe'
    $harness = @'
using System;
using System.Drawing;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;
using MoE4All.Launcher;

internal static class MarkdownHarness
{
    private static int passed;
    [StructLayout(LayoutKind.Sequential)]
    private struct ScrollPoint { public int X, Y; }
    [DllImport("user32.dll", EntryPoint = "SendMessageW")]
    private static extern IntPtr ScrollMessage(IntPtr window, int message, IntPtr unused, ref ScrollPoint point);
    private static void Check(bool condition, string name)
    {
        if (!condition) throw new Exception("FAIL " + name);
        passed++; Console.WriteLine("PASS " + name);
    }
    private static bool Has(string input, string text, MarkdownStyle style, int heading)
    {
        foreach (MarkdownRun run in MarkdownView.Parse(input).Runs)
            if (run.Text == text && (run.Style & style) == style && run.HeadingLevel == heading) return true;
        return false;
    }
    private static void Equal(string actual, string expected, string name)
    { Check(actual == expected, name + " [" + actual.Replace("\n", "\\n") + "]"); }

    [STAThread]
    private static int Main()
    {
        try
        {
            Equal(MarkdownView.ToPlainText(null), "", "null");
            Equal(MarkdownView.ToPlainText("a\r\nb\rc"), "a\nb\nc", "newline normalization");
            Equal(MarkdownView.ToPlainText("\u4e2d\u6587 \ud83d\ude80"), "\u4e2d\u6587 \ud83d\ude80", "Chinese and surrogate pair");
            Equal(MarkdownView.ToPlainText("a\0b"), "a\ufffdb", "NUL remains visible");
            Check(Has("# Title", "Title", MarkdownStyle.Bold, 1), "heading 1");
            Check(Has("###### Six", "Six", MarkdownStyle.Bold, 6), "heading 6");
            Equal(MarkdownView.ToPlainText("#identifier"), "#identifier", "heading requires separator");
            Check(Has("**bold** and *italic*", "bold", MarkdownStyle.Bold, 0), "bold");
            Check(Has("**bold** and *italic*", "italic", MarkdownStyle.Italic, 0), "italic");
            Check(Has("***both***", "both", MarkdownStyle.Bold | MarkdownStyle.Italic, 0), "combined emphasis");
            Check(Has("**bold *inside* text**", "inside", MarkdownStyle.Bold | MarkdownStyle.Italic, 0), "nested emphasis");
            Check(Has("__bold__ _italic_", "bold", MarkdownStyle.Bold, 0), "underscore emphasis");
            Equal(MarkdownView.ToPlainText("snake_case_name"), "snake_case_name", "identifier underscores");
            Check(Has("`**literal**`", "**literal**", MarkdownStyle.Code, 0), "inline code is literal");
            Check(Has("``a`b``", "a`b", MarkdownStyle.Code, 0), "multiple backticks");
            Equal(MarkdownView.ToPlainText("\\*literal\\*"), "*literal*", "escaped delimiters");
            Equal(MarkdownView.ToPlainText("a **unfinished"), "a **unfinished", "partial emphasis");
            Equal(MarkdownView.ToPlainText("[partial](https://x"), "[partial](https://x", "partial link");
            Equal(MarkdownView.ToPlainText("before\n```csharp\n    x\n\t y\n```\nafter"), "before\n    x\n\t y\nafter", "fenced code indentation");
            Equal(MarkdownView.ToPlainText("~~~\na\n  b"), "a\n  b", "unfinished fence keeps code");
            Equal(MarkdownView.ToPlainText("````\n```\nx\n````"), "```\nx\n", "short fence is code");
            Equal(MarkdownView.ToPlainText("- first\n  + second\n12. third"), "\u2022 first\n  \u2022 second\n12. third", "lists and indentation");
            Equal(MarkdownView.ToPlainText("> quote\n> **bold**"), "\u2502 quote\n\u2502 bold", "quote text");
            Check(Has("> **bold**", "bold", MarkdownStyle.Quote | MarkdownStyle.Bold, 0), "quote emphasis");
            Equal(MarkdownView.ToPlainText("[\u4e2d\u6587](javascript:alert(1))"), "\u4e2d\u6587", "untrusted link is label only");
            Check(Has("[label](file:///C:/x)", "label", MarkdownStyle.Link, 0), "link style without navigation");
            string table = "| Name | Value |\n| :--- | ---: |\n| \u4e2d\u6587 | **42** |";
            Equal(MarkdownView.ToPlainText(table), "Name | Value\n\u4e2d\u6587 | 42", "table Chinese and emphasis");
            Equal(MarkdownView.ToPlainText("| A | B |\n| --- | --- |\n| x\\|y | `a|b` |"), "A   | B\nx|y | a|b", "table literal pipes");
            Equal(MarkdownView.ToPlainText("A | B\n--- | ---\nx | y\n\nafter"), "A | B\nx | y\n\nafter", "table boundary");
            Equal(MarkdownView.ToPlainText("A | B\n-- | ---"), "A | B\n-- | ---", "invalid table is text");
            string html = "<script>alert(1)</script><img src=https://invalid.example/pixel>";
            Equal(MarkdownView.ToPlainText(html), html, "HTML is inert text");
            string payload = "{\\rtf1\\ansi{\\object\\objdata 0105}} \\field \\pict \\bin99";
            string safeRtf = MarkdownView.ToRtf(payload);
            Check(safeRtf.Contains("\\{\\\\rtf1") && safeRtf.Contains("\\\\object"), "RTF controls and braces escaped");
            Check(!MarkdownView.ToRtf("[x](javascript:alert(1))").Contains("javascript"), "link target omitted from RTF");
            string cultureRtf;
            var oldCulture = System.Threading.Thread.CurrentThread.CurrentCulture;
            try
            {
                System.Threading.Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("fr-FR");
                cultureRtf = MarkdownView.ToRtf("\u4e2d", WizardTheme.Current, "evil;{\\field}", 10.5f);
            }
            finally { System.Threading.Thread.CurrentThread.CurrentCulture = oldCulture; }
            Check(cultureRtf.Contains("\\fs21") && cultureRtf.Contains("\\u20013?"), "culture independent numbers and Unicode");
            Check(!cultureRtf.Contains("field") && cultureRtf.Contains("Segoe UI;"), "unsafe font name falls back");
            bool rejected = false;
            try { MarkdownView.ToRtf("x", WizardTheme.Current, "x", Single.NaN); }
            catch (ArgumentOutOfRangeException) { rejected = true; }
            Check(rejected, "invalid font size rejected");

            // Actual native RichEdit decoding, formatting, theme change and updates.
            Application.EnableVisualStyles();
            using (var form = new Form { Width = 640, Height = 300, ShowInTaskbar = false })
            using (var view = new MarkdownView { Dock = DockStyle.Fill })
            {
                form.Controls.Add(view);
                IntPtr handle = view.Handle;
                Check(view is WizardRichTextBox && view.ReadOnly && !view.DetectUrls && !view.AllowDrop, "reusable read-only control");
                view.SetMarkdown(payload);
                Equal(view.Text.Replace("\r\n", "\n"), payload, "RTF injection native round trip");
                view.SetMarkdown(html);
                Equal(view.Text, html, "native HTML round trip");
                string unicode = "\u4e2d\u6587 \ud83d\ude80 \\ { }";
                view.SetMarkdown(unicode);
                Equal(view.Text, unicode, "native Unicode and braces");
                view.SetMarkdown("```\n    a\n\tb\n```");
                Equal(view.Text.Replace("\r\n", "\n"), "    a\n\tb\n", "native code whitespace");
                view.SetMarkdown("**bold** *italic* `code` [label](https://invalid.example)");
                view.Select(0, 4); Check(view.SelectionFont != null && view.SelectionFont.Bold, "native bold font");
                view.Select(5, 6); Check(view.SelectionFont != null && view.SelectionFont.Italic, "native italic font");
                view.Select(12, 4); Check(view.SelectionBackColor.ToArgb() == WizardTheme.Current.SurfaceAlt.ToArgb(), "code theme background");
                view.Select(17, 5); Check(view.SelectionColor.ToArgb() == WizardTheme.Current.Accent.ToArgb(), "link theme color");
                foreach (string mode in new[] { "dark", "light" })
                {
                    WizardTheme.SetPreference(mode);
                    WizardTheme.Apply(form);
                    Check(view.BackColor.ToArgb() == WizardTheme.Current.Surface.ToArgb(), "theme background " + mode);
                    view.Select(17, 5);
                    Check(view.SelectionColor.ToArgb() == WizardTheme.Current.Accent.ToArgb(), "theme link " + mode);
                    view.Select(12, 4);
                    Check(view.SelectionBackColor.ToArgb() == WizardTheme.Current.SurfaceAlt.ToArgb(), "theme code " + mode);
                }
                string stream = "## \u4e2d\u6587\n**bold** [label](https://invalid.example)\n```\n  code\n```";
                for (int i = 0; i <= stream.Length; i++) view.SetMarkdown(stream.Substring(0, i));
                Equal(view.Text.Replace("\r\n", "\n"), MarkdownView.ToPlainText(stream), "all streaming prefixes converge");
                string firstRtf = view.Rtf;
                view.SetMarkdown(stream);
                Check(view.Rtf == firstRtf, "identical snapshot is a no-op");
                view.SetMarkdown(null); Equal(view.Text, "", "clear snapshot");
                var longText = new StringBuilder();
                for (int row = 0; row < 180; row++) longText.Append("row ").Append(row).Append(' ').Append('x', 120).Append('\n');
                view.WordWrap = false;
                view.SetMarkdown(longText.ToString());
                view.Select(1200, 7);
                var expectedScroll = new ScrollPoint { X = 80, Y = 350 };
                ScrollMessage(view.Handle, 0x04de, IntPtr.Zero, ref expectedScroll);
                ScrollMessage(view.Handle, 0x04dd, IntPtr.Zero, ref expectedScroll);
                Check(expectedScroll.Y > 0, "scroll fixture leaves top");
                view.SetMarkdown(longText.ToString() + "tail\n");
                var actualScroll = new ScrollPoint();
                ScrollMessage(view.Handle, 0x04dd, IntPtr.Zero, ref actualScroll);
                Check(actualScroll.X == expectedScroll.X && actualScroll.Y == expectedScroll.Y, "stream update preserves both scroll axes");
                Check(view.SelectionStart == 1200 && view.SelectionLength == 7, "stream update preserves selection");
                WizardTheme.SetPreference("dark");
                ScrollMessage(view.Handle, 0x04dd, IntPtr.Zero, ref actualScroll);
                Check(actualScroll.X == expectedScroll.X && actualScroll.Y == expectedScroll.Y, "theme update preserves viewport");
                Check(view.SelectionStart == 1200 && view.SelectionLength == 7, "theme update preserves selection");
                int userScrollEvents = 0;
                view.UserScrolled += delegate { userScrollEvents++; };
                view.ScrollToLatest();
                Check(view.IsAtLatest, "inherited ScrollToLatest after render");
                int bottomLine = view.FirstVisibleLine;
                view.ScrollLines(-5);
                Check(view.FirstVisibleLine < bottomLine && userScrollEvents == 1, "inherited user scroll contract");
                int userLine = view.FirstVisibleLine;
                view.SetMarkdown(longText.ToString() + "tail\nmore\n");
                Check(view.FirstVisibleLine == userLine && userScrollEvents == 1, "refresh preserves user scroll without a user event");
                view.SetMarkdown("short");
                Check(view.SelectionStart <= view.TextLength, "shorter snapshot clamps selection");
            }
            WizardTheme.SetPreference("system");
            // Deterministic malformed data checks parser termination and native safety.
            var random = new Random(7319);
            const string alphabet = "ab\u4e2d\u6587*_`[](){}\\|>~\t\n";
            using (var box = new RichTextBox { DetectUrls = false })
            {
                for (int sample = 0; sample < 160; sample++)
                {
                    var text = new StringBuilder();
                    for (int j = 0; j < 160; j++) text.Append(alphabet[random.Next(alphabet.Length)]);
                    string input = text.ToString();
                    box.Rtf = MarkdownView.ToRtf(input);
                    Equal(box.Text.Replace("\r\n", "\n"), MarkdownView.ToPlainText(input), "malformed round trip " + sample);
                }
            }
            string hostile = new string('[', 20000) + new string('`', 20000);
            Equal(MarkdownView.ToPlainText(hostile), hostile, "long unmatched delimiter runs");
            Console.WriteLine("Markdown: " + passed + " checks passed; .NET Framework / C# 5; no network or GPU.");
            return 0;
        }
        catch (Exception ex) { Console.Error.WriteLine(ex.ToString()); return 1; }
    }
}
'@
    [IO.File]::WriteAllText($source, $harness, (New-Object Text.UTF8Encoding $true))
    $themeSource = Join-Path $PSScriptRoot 'wizard-theme.cs'
    $markdownSource = Join-Path $PSScriptRoot 'wizard-markdown.cs'
    # Theme references the desktop's drawing helper. Compile the real SoftButton
    # class alone, avoiding unrelated runtime/services and avoiding a theme stub.
    $desktopSource = Join-Path $PSScriptRoot 'wizard-desktop.cs'
    $desktop = [IO.File]::ReadAllText($desktopSource)
    $buttonStart = $desktop.IndexOf('public sealed class SoftButton', [StringComparison]::Ordinal)
    $buttonEnd = $desktop.IndexOf('public sealed class TaskTile', [StringComparison]::Ordinal)
    if ($buttonStart -lt 0 -or $buttonEnd -le $buttonStart) { throw 'Cannot locate desktop SoftButton for theme test support.' }
    $supportSource = Join-Path $tmpDirectory 'DesktopButton.cs'
    $support = 'using System; using System.Drawing; using System.Drawing.Drawing2D; using System.Windows.Forms; namespace MoE4All.Launcher {' + $desktop.Substring($buttonStart, $buttonEnd - $buttonStart) + '}'
    [IO.File]::WriteAllText($supportSource, $support, (New-Object Text.UTF8Encoding $true))
    $compilerArgs = @(
        '/nologo', '/langversion:5', '/target:exe', '/optimize+', '/noconfig',
        '/reference:System.dll', '/reference:System.Core.dll',
        '/reference:System.Drawing.dll', '/reference:System.Windows.Forms.dll', '/reference:Accessibility.dll',
        ('/out:' + $exe), $themeSource, $markdownSource, $supportSource, $source
    )
    $manifest.compiler = $compiler
    $manifest.compilerArguments = $compilerArgs
    $manifest.commit = (& git -C $repoRoot rev-parse HEAD | Out-String).Trim()
    $manifest.dirty = @(& git -C $repoRoot status --porcelain)
    $identities = @()
    foreach ($path in @($themeSource, $markdownSource, $desktopSource, $PSCommandPath, $source, $supportSource, $compiler)) {
        $sha = [Security.Cryptography.SHA256]::Create()
        $stream = [IO.File]::OpenRead($path)
        try { $hash = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
        finally { $stream.Dispose(); $sha.Dispose() }
        $identities += [ordered]@{ path = $path; sha256 = $hash }
    }
    $manifest.sources = $identities
    Copy-Item -LiteralPath $source -Destination (Join-Path $evidenceRun 'MarkdownHarness.cs')
    Copy-Item -LiteralPath $markdownSource -Destination (Join-Path $evidenceRun 'wizard-markdown.cs')
    Copy-Item -LiteralPath $themeSource -Destination (Join-Path $evidenceRun 'wizard-theme.cs')
    Copy-Item -LiteralPath $supportSource -Destination (Join-Path $evidenceRun 'DesktopButton.cs')
    Copy-Item -LiteralPath $PSCommandPath -Destination (Join-Path $evidenceRun 'Test-WizardMarkdown.ps1')
    $manifest | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $evidenceRun 'manifest.json') -Encoding UTF8
    $compileLog = & $compiler @compilerArgs | Out-String
    $compileExit = $LASTEXITCODE
    $compileLog | Set-Content -LiteralPath (Join-Path $evidenceRun 'compile.log') -Encoding UTF8
    if ($compileExit -ne 0) { throw ('C# 5 compilation failed: ' + $compileLog) }
    $sha = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($exe)
    try { $exeHash = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $sha.Dispose() }
    $manifest.executable = [ordered]@{ path = $exe; sha256 = $exeHash }
    $manifest.runtimeVersion = [Environment]::Version.ToString()
    $manifest.toolchain = (Get-Item -LiteralPath $compiler).VersionInfo.FileVersion
    # Redirect native output to files so failure details survive a nonzero exit.
    $ErrorActionPreference = 'Continue'
    try {
        & $exe 1> (Join-Path $evidenceRun 'tests.log') 2> (Join-Path $evidenceRun 'tests.err.log')
        $testExit = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = 'Stop' }
    if ($testExit -ne 0) { throw ('Markdown tests failed; see ' + $evidenceRun) }
    Get-Content -LiteralPath (Join-Path $evidenceRun 'tests.log') -Tail 1
    $manifest.status = 'passed'
}
catch {
    $manifest.status = 'failed'
    $manifest.error = $_.Exception.Message
    throw
}
finally {
    $manifest.finishedUtc = [DateTime]::UtcNow.ToString('o')
    $manifest | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $evidenceRun 'manifest.json') -Encoding UTF8
    foreach ($key in $oldEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($key, $oldEnvironment[$key], 'Process')
    }
}
