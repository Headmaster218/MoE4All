// .NET Framework 4 / C# 5. Compile together with wizard-theme.cs.
// UI thread only. The caller throttles streaming snapshots and uses the inherited
// WizardRichTextBox scrolling contract. No HTML, URL activation or external I/O.
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Drawing;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

namespace MoE4All.Launcher
{
    [Flags]
    public enum MarkdownStyle
    {
        None = 0, Bold = 1, Italic = 2, Code = 4, Quote = 8, Link = 16, Table = 32
    }

    public sealed class MarkdownRun
    {
        public string Text { get; private set; }
        public MarkdownStyle Style { get; private set; }
        public int HeadingLevel { get; private set; }
        internal MarkdownRun(string text, MarkdownStyle style, int heading)
        { Text = text; Style = style; HeadingLevel = heading; }
    }

    public sealed class MarkdownDocument
    {
        public ReadOnlyCollection<MarkdownRun> Runs { get; private set; }
        public string Text { get; private set; }
        internal MarkdownDocument(List<MarkdownRun> runs)
        {
            Runs = runs.AsReadOnly();
            var text = new StringBuilder();
            foreach (MarkdownRun run in runs) text.Append(run.Text);
            Text = text.ToString();
        }
    }

    public sealed class MarkdownView : WizardRichTextBox
    {
        // Bound speculative searches on malformed/untrusted streaming input.
        // Longer unmatched inline constructs fall back to literal text.
        private const int InlineLookahead = 8192;
        private string markdown = "";
        private bool rendering, listening;

        [StructLayout(LayoutKind.Sequential)]
        private struct ScrollPoint { public int X, Y; }
        [DllImport("user32.dll", EntryPoint = "SendMessageW")]
        private static extern IntPtr ScrollMessage(IntPtr window, int message, IntPtr unused, ref ScrollPoint point);

        public MarkdownView()
        {
            ReadOnly = true;
            DetectUrls = false;
            AllowDrop = false;
            WordWrap = true;
            BorderStyle = BorderStyle.None;
            BackColor = WizardTheme.Current.Surface;
            ForeColor = WizardTheme.Current.Text;
            AccessibleRole = AccessibleRole.StaticText;
        }

        // Full snapshot, not an append operation. Unfinished delimiters stay literal.
        public void SetMarkdown(string value)
        {
            if (IsDisposed) throw new ObjectDisposedException("MarkdownView");
            value = value ?? "";
            if (markdown == value && TextLength != 0) return;
            markdown = value;
            RenderMarkdown();
        }

        private void RenderMarkdown()
        {
            if (rendering || IsDisposed) return;
            rendering = true;
            bool hadHandle = IsHandleCreated;
            int selection = SelectionStart, selectionLength = SelectionLength;
            var scroll = new ScrollPoint();
            if (hadHandle) ScrollMessage(Handle, 0x04dd, IntPtr.Zero, ref scroll); // EM_GETSCROLLPOS
            try
            {
                WizardPalette palette = WizardTheme.Current;
                BackColor = palette.Surface;
                ForeColor = palette.Text;
                DetectUrls = false;
                Rtf = ToRtf(markdown, palette, Font.Name, Font.SizeInPoints);
                ReadOnly = true;
            }
            finally
            {
                // Preserve the viewport even when RichEdit resets it during Rtf=.
                // Following latest remains the host's inherited scrolling policy.
                if (!IsDisposed)
                {
                    Select(Math.Min(selection, TextLength), Math.Min(selectionLength, Math.Max(0, TextLength - selection)));
                    if (hadHandle && IsHandleCreated) ScrollMessage(Handle, 0x04de, IntPtr.Zero, ref scroll); // EM_SETSCROLLPOS
                    RefreshScrollIndicators();
                }
                rendering = false;
            }
        }

        protected override void OnHandleCreated(EventArgs e)
        {
            base.OnHandleCreated(e);
            if (!listening) { WizardTheme.Changed += ThemeChanged; listening = true; }
            RenderMarkdown();
        }

        protected override void OnHandleDestroyed(EventArgs e)
        {
            StopListening();
            base.OnHandleDestroyed(e);
        }

        protected override void OnFontChanged(EventArgs e)
        {
            base.OnFontChanged(e);
            if (IsHandleCreated) RenderMarkdown();
        }

        // Even a host that subscribes to LinkClicked cannot accidentally navigate.
        protected override void OnLinkClicked(LinkClickedEventArgs e) { }

        private void ThemeChanged(object sender, EventArgs e)
        {
            if (IsDisposed || !IsHandleCreated) return;
            if (InvokeRequired)
            {
                try { BeginInvoke((MethodInvoker)RenderMarkdown); }
                catch (InvalidOperationException) { /* Handle destroyed during dispatch. */ }
            }
            else RenderMarkdown();
        }

        private void StopListening()
        {
            if (listening) { WizardTheme.Changed -= ThemeChanged; listening = false; }
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing) StopListening();
            base.Dispose(disposing);
        }

        public static MarkdownDocument Parse(string value)
        {
            var runs = new List<MarkdownRun>();
            string[] lines = Normalize(value).Split('\n');
            char fence = '\0';
            int fenceLength = 0;
            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i];
                string trimmed = line.TrimStart(' ');
                int indent = line.Length - trimmed.Length;
                int markerLength = MarkerLength(trimmed);
                if (fence != '\0')
                {
                    if (indent <= 3 && markerLength >= fenceLength && trimmed[0] == fence &&
                        trimmed.Substring(markerLength).Trim().Length == 0)
                    { fence = '\0'; continue; }
                    Add(runs, line + (i < lines.Length - 1 ? "\n" : ""), MarkdownStyle.Code, 0);
                    continue;
                }
                if (indent <= 3 && markerLength >= 3)
                { fence = trimmed[0]; fenceLength = markerLength; continue; }

                // A header followed by a pipe separator starts a simple table.
                if (i + 1 < lines.Length && line.IndexOf('|') >= 0 && IsTableSeparator(lines[i + 1]))
                {
                    var rows = new List<string[]>();
                    rows.Add(TableCells(line));
                    int end = i + 2;
                    while (end < lines.Length && lines[end].Trim().Length > 0 && lines[end].IndexOf('|') >= 0)
                    { rows.Add(TableCells(lines[end])); end++; }
                    int columns = 0;
                    foreach (string[] row in rows) columns = Math.Max(columns, row.Length);
                    int[] widths = new int[columns];
                    foreach (string[] row in rows)
                        for (int c = 0; c < row.Length; c++)
                        {
                            row[c] = ToPlainInline(row[c]);
                            widths[c] = Math.Max(widths[c], Math.Min(48, DisplayWidth(row[c])));
                        }
                    for (int r = 0; r < rows.Count; r++)
                    {
                        var text = new StringBuilder();
                        for (int c = 0; c < columns; c++)
                        {
                            string cell = c < rows[r].Length ? rows[r][c] : "";
                            if (c > 0) text.Append(" | ");
                            text.Append(cell);
                            if (c < columns - 1) text.Append(' ', Math.Max(0, widths[c] - DisplayWidth(cell)));
                        }
                        if (r < rows.Count - 1 || end < lines.Length) text.Append('\n');
                        Add(runs, text.ToString(), MarkdownStyle.Table | (r == 0 ? MarkdownStyle.Bold : MarkdownStyle.None), 0);
                    }
                    i = end - 1;
                    continue;
                }

                MarkdownStyle style = MarkdownStyle.None;
                int heading = 0;
                string body = line;
                if (indent <= 3 && trimmed.StartsWith(">", StringComparison.Ordinal))
                {
                    style |= MarkdownStyle.Quote;
                    int pos = 0;
                    while (pos < trimmed.Length && trimmed[pos] == '>')
                    { pos++; if (pos < trimmed.Length && trimmed[pos] == ' ') pos++; }
                    body = "\u2502 " + trimmed.Substring(pos);
                }
                else if (indent <= 3)
                {
                    while (heading < trimmed.Length && heading < 6 && trimmed[heading] == '#') heading++;
                    if (heading > 0 && (heading == trimmed.Length || trimmed[heading] == ' '))
                    { body = trimmed.Substring(heading).TrimStart(' '); style |= MarkdownStyle.Bold; }
                    else heading = 0;
                }
                if (heading == 0 && style == MarkdownStyle.None)
                {
                    int prefix = ListPrefix(trimmed);
                    if (prefix > 0)
                    {
                        bool numbered = Char.IsDigit(trimmed[0]);
                        body = line.Substring(0, indent) + (numbered ? trimmed.Substring(0, prefix).TrimEnd() : "\u2022") + " " + trimmed.Substring(prefix);
                    }
                }
                Inline(runs, body, style, heading, 0);
                if (i < lines.Length - 1) Add(runs, "\n", style, heading);
            }
            return new MarkdownDocument(runs);
        }

        public static string ToPlainText(string value) { return Parse(value).Text; }

        public static string ToRtf(string value)
        {
            return ToRtf(value, WizardTheme.Current, SystemFonts.MessageBoxFont.Name, SystemFonts.MessageBoxFont.SizeInPoints);
        }

        // The only RTF entry point: all input, including font names, is escaped.
        // No field/hyperlink/picture/object directives are generated.
        public static string ToRtf(string value, WizardPalette palette, string fontFamily, float fontSize)
        {
            if (palette == null) throw new ArgumentNullException("palette");
            if (Single.IsNaN(fontSize) || Single.IsInfinity(fontSize) || fontSize < 1 || fontSize > 200)
                throw new ArgumentOutOfRangeException("fontSize");
            var rtf = new StringBuilder("{\\rtf1\\ansi\\ansicpg1252\\deff0\\uc1");
            rtf.Append("{\\fonttbl{\\f0\\fnil\\fcharset0 ");
            // RichEdit does not reliably accept Unicode escapes inside fonttbl.
            // Use an ASCII UI family and let Windows supply Chinese fallback.
            string family = String.IsNullOrEmpty(fontFamily) ? "Segoe UI" : fontFamily;
            foreach (char ch in family)
                if (!(ch >= 'a' && ch <= 'z') && !(ch >= 'A' && ch <= 'Z') &&
                    !(ch >= '0' && ch <= '9') && ch != ' ' && ch != '-' && ch != '_')
                { family = "Segoe UI"; break; }
            Escape(rtf, family);
            rtf.Append(";}{\\f1\\fmodern Consolas;}}");
            rtf.Append("{\\colortbl ;");
            ColorEntry(rtf, palette.Text);
            ColorEntry(rtf, palette.Muted);
            ColorEntry(rtf, palette.Accent);
            ColorEntry(rtf, palette.SurfaceAlt);
            rtf.Append("}\\viewkind4\\pard\\cf1 ");
            foreach (MarkdownRun run in Parse(value).Runs)
            {
                bool code = (run.Style & (MarkdownStyle.Code | MarkdownStyle.Table)) != 0;
                float scale = run.HeadingLevel == 0 ? 1f : 1f + (7 - run.HeadingLevel) * 0.08f;
                int halfPoints = (int)Math.Round(fontSize * scale * 2);
                rtf.Append('{').Append(code ? "\\f1" : "\\f0").Append("\\fs").Append(halfPoints);
                if ((run.Style & MarkdownStyle.Bold) != 0) rtf.Append("\\b");
                if ((run.Style & MarkdownStyle.Italic) != 0) rtf.Append("\\i");
                if ((run.Style & MarkdownStyle.Quote) != 0) rtf.Append("\\cf2");
                if ((run.Style & MarkdownStyle.Link) != 0) rtf.Append("\\cf3\\ul");
                if (code) rtf.Append("\\highlight4");
                rtf.Append(' ');
                Escape(rtf, run.Text);
                rtf.Append('}');
            }
            // RichEdit's terminal paragraph is structural; without it a final
            // explicit line break can be consumed as that paragraph terminator.
            return rtf.Append("\\par}").ToString();
        }

        private static void ColorEntry(StringBuilder text, Color color)
        {
            text.Append("\\red").Append(color.R).Append("\\green").Append(color.G).Append("\\blue").Append(color.B).Append(';');
        }

        private static void Escape(StringBuilder output, string text)
        {
            foreach (char ch in text)
            {
                if (ch == '\\' || ch == '{' || ch == '}') output.Append('\\').Append(ch);
                else if (ch == '\n') output.Append("\\line ");
                else if (ch == '\t') output.Append("\\tab ");
                else if (ch >= 32 && ch < 127) output.Append(ch);
                else output.Append("\\u").Append(((short)ch).ToString(CultureInfo.InvariantCulture)).Append('?');
            }
        }

        private static string Normalize(string value)
        {
            // Native RichEdit treats NUL as a terminator. Make it visible instead.
            return (value ?? "").Replace("\r\n", "\n").Replace('\r', '\n').Replace('\0', '\ufffd');
        }

        private static void Add(List<MarkdownRun> runs, string text, MarkdownStyle style, int heading)
        {
            if (text.Length > 0) runs.Add(new MarkdownRun(text, style, heading));
        }

        private static int MarkerLength(string text)
        {
            if (text.Length == 0 || (text[0] != '`' && text[0] != '~')) return 0;
            int i = 1;
            while (i < text.Length && text[i] == text[0]) i++;
            return i;
        }

        private static int ListPrefix(string text)
        {
            if (text.Length >= 2 && (text[0] == '-' || text[0] == '*' || text[0] == '+') && text[1] == ' ')
            { int p = 2; while (p < text.Length && text[p] == ' ') p++; return p; }
            int i = 0;
            while (i < text.Length && i < 9 && text[i] >= '0' && text[i] <= '9') i++;
            if (i > 0 && i + 1 < text.Length && (text[i] == '.' || text[i] == ')') && text[i + 1] == ' ')
            { i += 2; while (i < text.Length && text[i] == ' ') i++; return i; }
            return 0;
        }

        private static bool IsTableSeparator(string line)
        {
            if (line.IndexOf('|') < 0) return false;
            foreach (string cell in TableCells(line))
            {
                string value = cell.Trim(':');
                if (value.Length < 3) return false;
                foreach (char c in value) if (c != '-') return false;
            }
            return true;
        }

        private static string[] TableCells(string line)
        {
            // Pipes inside backtick spans and escaped pipes remain in their cell.
            string value = line.Trim();
            if (value.StartsWith("|", StringComparison.Ordinal)) value = value.Substring(1);
            if (value.EndsWith("|", StringComparison.Ordinal) && !value.EndsWith("\\|", StringComparison.Ordinal)) value = value.Substring(0, value.Length - 1);
            var cells = new List<string>();
            var cell = new StringBuilder();
            int ticks = 0;
            for (int i = 0; i < value.Length; i++)
            {
                char c = value[i];
                if (c == '\\' && i + 1 < value.Length)
                { cell.Append(c).Append(value[++i]); continue; }
                if (c == '`')
                {
                    int count = 1;
                    while (i + count < value.Length && value[i + count] == '`') count++;
                    if (ticks == 0) ticks = count; else if (ticks == count) ticks = 0;
                    cell.Append('`', count); i += count - 1; continue;
                }
                if (c == '|' && ticks == 0) { cells.Add(cell.ToString().Trim()); cell.Length = 0; }
                else cell.Append(c);
            }
            cells.Add(cell.ToString().Trim());
            return cells.ToArray();
        }

        private static int DisplayWidth(string text)
        {
            int width = 0;
            foreach (char c in text) width += c >= 0x2e80 ? 2 : 1;
            return width;
        }

        private static string ToPlainInline(string text)
        {
            var runs = new List<MarkdownRun>();
            Inline(runs, text, MarkdownStyle.None, 0, 0);
            var output = new StringBuilder();
            foreach (MarkdownRun run in runs) output.Append(run.Text);
            return output.ToString();
        }

        private static bool Escapable(char c) { return "\\`*_{}[]()#+-.!>|~".IndexOf(c) >= 0; }

        private static int FindClosing(string text, string token, int start)
        {
            int found = start;
            int end = Math.Min(text.Length, start + InlineLookahead);
            while (found <= end - token.Length &&
                (found = text.IndexOf(token, found, end - found, StringComparison.Ordinal)) >= 0)
            {
                int slashes = 0;
                for (int p = found - 1; p >= 0 && text[p] == '\\'; p--) slashes++;
                if (slashes % 2 == 0) return found;
                found += token.Length;
            }
            return -1;
        }

        private static int LinkEnd(string text, int start)
        {
            int depth = 1;
            int end = Math.Min(text.Length, start + InlineLookahead);
            for (int p = start; p < end; p++)
            {
                if (text[p] == '\\') { p++; continue; }
                if (text[p] == '(') depth++;
                else if (text[p] == ')' && --depth == 0) return p;
            }
            return -1;
        }

        private static void Inline(List<MarkdownRun> runs, string text, MarkdownStyle style, int heading, int depth)
        {
            if (depth >= 12) { Add(runs, text, style, heading); return; }
            var plain = new StringBuilder();
            for (int i = 0; i < text.Length; )
            {
                char c = text[i];
                if (c == '\\' && i + 1 < text.Length && Escapable(text[i + 1]))
                { plain.Append(text[i + 1]); i += 2; continue; }
                string token = null;
                int close = -1, after = -1;
                MarkdownStyle inner = style;
                if (c == '`')
                {
                    int count = 1;
                    while (i + count < text.Length && text[i + count] == '`') count++;
                    token = new string('`', count);
                    close = FindClosing(text, token, i + count);
                    while (close >= 0 && ((close > 0 && text[close - 1] == '`') ||
                        (close + count < text.Length && text[close + count] == '`')))
                        close = FindClosing(text, token, close + count);
                    inner |= MarkdownStyle.Code;
                }
                else if (c == '[')
                {
                    int labelEnd = FindClosing(text, "](", i + 1);
                    if (labelEnd >= 0)
                    {
                        int end = LinkEnd(text, labelEnd + 2);
                        if (end >= 0) { token = "["; close = labelEnd; after = end + 1; inner |= MarkdownStyle.Link; }
                    }
                }
                else if (c == '*' || c == '_')
                {
                    int count = 1;
                    while (count < 3 && i + count < text.Length && text[i + count] == c) count++;
                    // Underscores within identifiers are literal.
                    if (!(c == '_' && i > 0 && Char.IsLetterOrDigit(text[i - 1])))
                    {
                        token = new string(c, count);
                        close = FindClosing(text, token, i + count);
                        if (close == i + count || i + count >= text.Length || Char.IsWhiteSpace(text[i + count]) ||
                            (close >= 0 && Char.IsWhiteSpace(text[close - 1]))) close = -1;
                        if (count >= 2) inner |= MarkdownStyle.Bold;
                        if (count != 2) inner |= MarkdownStyle.Italic;
                    }
                }
                if (token != null && close >= 0)
                {
                    Add(runs, plain.ToString(), style, heading); plain.Length = 0;
                    string content = text.Substring(i + token.Length, close - i - token.Length);
                    if ((inner & MarkdownStyle.Code) != 0) Add(runs, content, inner, heading);
                    else Inline(runs, content, inner, heading, depth + 1);
                    i = after >= 0 ? after : close + token.Length;
                }
                else if (c == '`' && token != null)
                { plain.Append(token); i += token.Length; }
                else { plain.Append(c); i++; }
            }
            Add(runs, plain.ToString(), style, heading);
        }
    }
}
