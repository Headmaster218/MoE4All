// Inbox .NET Framework / C# 5. Consumes RuntimeLog.RenderedText on the UI thread.
// Compile with the existing native launcher widgets/theme. No engine calls or timers.
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using System.Windows.Forms;

namespace MoE4All.Launcher {
    public sealed class ApiStatusView : WizardPanel {
        const string Marker = "INFR live resources";
        const string Border = "+---------------+---------------------------------+-----------+---------------+-----------------------+";
        const string Number = @"[0-9]+(?:\.[0-9]+)?";
        static readonly Regex Header = new Regex(@"^\|\s*Slot\s*\|\s*Work\s*\|\s*State\s*\|\s*Speed \(now\)\s*\|\s*Context\s*\|$", RegexOptions.CultureInvariant);
        // Work can itself contain '|'; anchor the known state/speed/context tail.
        static readonly Regex Row = new Regex(@"^\|\s*(KV [0-9]+/[0-9]+|Embed [0-9]+/[0-9]+|Embedding)\s*\| (.*?) \|\s*(Starting|Prefill|Decode|Idle|Embedding)\s*\|\s*(-|" + Number + @" tok/s|" + Number + @"s)\s*\|\s*(-|[0-9,]+ / [0-9,]+)\s*\|$", RegexOptions.CultureInvariant);
        static readonly Regex Slot = new Regex(@"^(KV|Embed) ([0-9]+)/([0-9]+)$", RegexOptions.CultureInvariant);
        static readonly Regex Total = new Regex(@"^Total: prefill (" + Number + @") tok/s \| decode (" + Number + @") tok/s \| active ([0-9]+) \| queued ([0-9]+) \| done ([0-9]+) \| failed ([0-9]+)$", RegexOptions.CultureInvariant);
        static readonly Regex Context = new Regex(@"^(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+) / (?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)$", RegexOptions.CultureInvariant);

        sealed class ActivityGrid : DataGridView {
            internal Action<int> Wheel;
            internal ActivityGrid() { DoubleBuffered = true; }
            protected override void OnMouseWheel(MouseEventArgs e) { if (Wheel != null) Wheel(e.Delta); }
        }
        sealed class Frame {
            internal readonly List<string[]> Rows = new List<string[]>();
            internal string Summary, Identity;
            internal double Decode;
        }
        readonly ActivityGrid grid;
        readonly WizardScrollTrack vertical, horizontal;
        readonly Label heading, summary;
        readonly SoftButton copy;
        readonly bool english;
        Font headerFont;
        Frame frame;
        string observed;
        bool scrolling;

        public ApiStatusView() : this(true) { }
        public ApiStatusView(bool en) {
            english = en; Name = "ApiStatus"; Padding = new Padding(1); TabStop = true;
            var layout = new TableLayoutPanel { Dock = DockStyle.Fill, RowCount = 3, ColumnCount = 1, Margin = Padding.Empty, Padding = Padding.Empty };
            layout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 36));
            layout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            layout.RowStyles.Add(new RowStyle(SizeType.Absolute, 48));
            var title = new WizardPanel { Dock = DockStyle.Fill, Padding = new Padding(8, 2, 4, 2), Margin = Padding.Empty };
            heading = new Label { Name = "ApiStatusHeading", Dock = DockStyle.Fill, TextAlign = ContentAlignment.MiddleLeft, Text = T("槽位状态与速度", "Slot status and speed"), AutoEllipsis = true };
            copy = new SoftButton { Name = "ApiStatusCopy", Dock = DockStyle.Right, Width = 112, Text = T("复制表格", "Copy table"), Quiet = true, Enabled = false };
            copy.Click += delegate { CopyAll(); };
            title.Controls.Add(heading); title.Controls.Add(copy); layout.Controls.Add(title, 0, 0);
            grid = new ActivityGrid { Name = "ApiStatusGrid", Dock = DockStyle.Fill, ReadOnly = true, AllowUserToAddRows = false, AllowUserToDeleteRows = false, AllowUserToResizeRows = false, AllowUserToOrderColumns = false, RowHeadersVisible = false, MultiSelect = true, SelectionMode = DataGridViewSelectionMode.CellSelect, AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill, ScrollBars = ScrollBars.None, BorderStyle = BorderStyle.None, CellBorderStyle = DataGridViewCellBorderStyle.Single, ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.DisableResizing, ClipboardCopyMode = DataGridViewClipboardCopyMode.EnableAlwaysIncludeHeaderText };
            string[] headers = Headers(); int[] weights = { 15, 31, 13, 18, 23 }, widths = { 80, 130, 74, 106, 150 };
            for (int i = 0; i < headers.Length; i++) {
                grid.Columns.Add(new DataGridViewTextBoxColumn { Name = headers[i], HeaderText = headers[i], FillWeight = weights[i], MinimumWidth = widths[i], SortMode = DataGridViewColumnSortMode.NotSortable });
                var align = i < 3 ? DataGridViewContentAlignment.MiddleLeft : DataGridViewContentAlignment.MiddleRight;
                grid.Columns[i].DefaultCellStyle.Alignment = align; grid.Columns[i].HeaderCell.Style.Alignment = align;
            }
            grid.DefaultCellStyle.Padding = new Padding(6, 2, 6, 2);
            var body = new WizardPanel { Dock = DockStyle.Fill, Margin = Padding.Empty };
            vertical = new WizardScrollTrack { Name = "ApiStatusVerticalScroll", Dock = DockStyle.Right, Width = 14 };
            horizontal = new WizardScrollTrack { Name = "ApiStatusHorizontalScroll", Dock = DockStyle.Bottom, Height = 14, Horizontal = true };
            body.Controls.Add(grid); body.Controls.Add(vertical); body.Controls.Add(horizontal); layout.Controls.Add(body, 0, 1);
            summary = new Label { Name = "ApiStatusTotal", Dock = DockStyle.Fill, Padding = new Padding(8, 4, 8, 4), Margin = Padding.Empty, Text = T("等待完整服务活动表…", "Waiting for a complete service activity frame…"), TextAlign = ContentAlignment.MiddleLeft };
            layout.Controls.Add(summary, 0, 2); Controls.Add(layout);
            vertical.ValueChanged += delegate { SetFirstRow(vertical.Value); };
            horizontal.ValueChanged += delegate { if (!scrolling) { grid.HorizontalScrollingOffset = horizontal.Value; RefreshScroll(); } };
            grid.Scroll += delegate { RefreshScroll(); }; grid.SizeChanged += delegate { RefreshScroll(); };
            grid.Wheel = delegate(int delta) { if (delta != 0) SetFirstRow(FirstRow - delta / 120 * 3); };
            grid.KeyDown += delegate(object sender, KeyEventArgs e) {
                if (e.Control && e.KeyCode == Keys.C) { e.SuppressKeyPress = true; CopySelection(); }
            };
            WizardTheme.Changed += ThemeChanged; ApplyGridTheme();
        }

        string T(string zh, string en) { return english ? en : zh; }
        static string[] Headers() { return new[] { "Slot", "Work", "State", "Speed (now)", "Context" }; }
        public bool HasFrame { get { return frame != null; } }
        public bool HasRows { get { return frame != null && frame.Rows.Count > 0; } }
        public int RowCount { get { return frame == null ? 0 : frame.Rows.Count; } }
        public string TotalText { get { return frame == null ? "" : frame.Summary; } }
        // Snapshot rate only: sum the actual Decode rows, never a request average
        // or Total/slot-count estimate. An idle complete frame yields zero.
        public double? ActiveDecodeTokensPerSecond { get { return frame == null ? (double?)null : frame.Decode; } }

        /// <summary>Accept a cumulative, ANSI-free RuntimeLog.RenderedText snapshot on the UI thread.
        /// Only newline-terminated complete frames replace the last good frame.
        /// The engine's draw_terminal_frame appends this newline, even though render_activity_table does not.
        /// Unchanged frames return false and do not rebuild cells or disturb selection.</summary>
        public bool ObserveTranscript(string text) {
            if (IsDisposed || text == null || String.Equals(observed, text, StringComparison.Ordinal)) return false;
            observed = text;
            string[] lines = text.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');
            for (int i = lines.Length - 1; i >= 0; i--) {
                if (lines[i].Trim() != Marker) continue;
                Frame candidate;
                if (!TryFrame(lines, i, out candidate)) continue;
                if (frame != null && candidate.Identity == frame.Identity) return false;
                frame = candidate; Render(); return true;
            }
            return false;
        }

        static bool Rate(string value, out double rate) {
            return Double.TryParse(value, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out rate) && !Double.IsInfinity(rate) && !Double.IsNaN(rate) && rate >= 0;
        }
        static bool TryFrame(string[] lines, int start, out Frame result) {
            result = null;
            if (start + 5 >= lines.Length || lines[start + 1].Trim() != Border || !Header.IsMatch(lines[start + 2].Trim()) || lines[start + 3].Trim() != Border) return false;
            var next = new Frame(); int index = start + 4, remaining = 0, capacity = 0; string kind = "";
            // Multiple hosted models may each own 1-8 KV/embedding slots. Bound
            // malformed input while keeping the per-model ordinal validation.
            while (index < lines.Length && lines[index].Trim() != Border) {
                if (next.Rows.Count >= 64 || lines[index].Length > 2048) return false;
                Match m = Row.Match(lines[index].Trim()); if (!m.Success) return false;
                var row = new string[5]; for (int c = 0; c < 5; c++) row[c] = m.Groups[c + 1].Value.Trim().Replace('\t', ' ');
                Match slot = Slot.Match(row[0]); int ordinal = 1, count = 1; string rowKind = "Embed";
                if (slot.Success) {
                    rowKind = slot.Groups[1].Value;
                    if (!Int32.TryParse(slot.Groups[2].Value, out ordinal) || !Int32.TryParse(slot.Groups[3].Value, out count) || count < 1 || count > 8 || ordinal < 1 || ordinal > count) return false;
                }
                if (remaining == 0) { if (ordinal != 1) return false; capacity = count; kind = rowKind; remaining = count; }
                if (rowKind != kind || count != capacity || ordinal != capacity - remaining + 1) return false;
                remaining--;
                bool idle = row[2] == "Idle", starting = row[2] == "Starting", embed = rowKind == "Embed";
                double rate = 0;
                if (idle || starting) {
                    if (row[3] != "-" || row[4] != "-" || (embed && starting)) return false;
                } else if (embed) {
                    if (row[2] != "Embedding" || !row[3].EndsWith("s", StringComparison.Ordinal) || !Rate(row[3].Substring(0, row[3].Length - 1), out rate) || row[4] != "-") return false;
                } else {
                    if ((row[2] != "Decode" && row[2] != "Prefill") || !row[3].EndsWith(" tok/s", StringComparison.Ordinal) || !Rate(row[3].Substring(0, row[3].Length - 6), out rate)) return false;
                    if (row[4] != "-" && !Context.IsMatch(row[4])) return false;
                    if (row[2] == "Decode") next.Decode += rate;
                }
                next.Rows.Add(row); index++;
            }
            if (remaining != 0 || index + 1 >= lines.Length) return false;
            string total = lines[index + 1].Trim();
            // Require LF after Total. Without it, a read ending at "failed 1"
            // could be the first half of "failed 12" and is not a complete frame.
            Match totals = Total.Match(total); double prefillRate, decodeRate;
            if (index + 1 == lines.Length - 1 || !totals.Success || !Rate(totals.Groups[1].Value, out prefillRate) || !Rate(totals.Groups[2].Value, out decodeRate) || Double.IsInfinity(next.Decode)) return false;
            next.Summary = total;
            var identity = new StringBuilder(); foreach (var row in next.Rows) identity.AppendLine(String.Join("\t", row)); identity.Append(total);
            next.Identity = identity.ToString(); result = next; return true;
        }

        public string ExportTsv() {
            var output = new StringBuilder(); output.AppendLine(String.Join("\t", Headers()));
            if (frame != null) { foreach (var row in frame.Rows) output.AppendLine(String.Join("\t", row)); output.AppendLine(frame.Summary); }
            return output.ToString();
        }
        public string SelectedTsv() {
            if (grid.SelectedCells.Count > 0) { var data = grid.GetClipboardContent(); if (data != null) { string text = data.GetData(DataFormats.UnicodeText) as string; if (!String.IsNullOrEmpty(text)) return text; } }
            return ExportTsv();
        }
        public void ScrollToLatest() { SetFirstRow(Math.Max(0, grid.Rows.Count - VisibleRows)); }
        void CopyAll() { CopyText(ExportTsv()); }
        void CopySelection() { CopyText(SelectedTsv()); }
        void CopyText(string text) { if (frame == null) return; try { Clipboard.SetText(text, TextDataFormat.UnicodeText); copy.Text = T("已复制", "Copied"); } catch (System.Runtime.InteropServices.ExternalException) { copy.Text = T("重试复制", "Retry copy"); } }
        void Render() {
            grid.SuspendLayout();
            try {
                bool empty = grid.Rows.Count == 0;
                while (grid.Rows.Count > frame.Rows.Count) grid.Rows.RemoveAt(grid.Rows.Count - 1);
                while (grid.Rows.Count < frame.Rows.Count) grid.Rows.Add();
                for (int i = 0; i < frame.Rows.Count; i++) for (int c = 0; c < 5; c++) {
                    var cell = grid.Rows[i].Cells[c]; string value = frame.Rows[i][c];
                    if (!String.Equals(Convert.ToString(cell.Value), value, StringComparison.Ordinal)) cell.Value = value;
                    cell.ToolTipText = value;
                }
                if (empty) { grid.ClearSelection(); grid.CurrentCell = null; }
                summary.Text = frame.Summary; copy.Enabled = true; copy.Text = T("复制表格", "Copy table"); ColorRows(); RefreshScroll();
            } finally { grid.ResumeLayout(); }
        }
        int FirstRow { get { return grid.Rows.Count == 0 || grid.FirstDisplayedScrollingRowIndex < 0 ? 0 : grid.FirstDisplayedScrollingRowIndex; } }
        int VisibleRows { get { return Math.Max(1, (grid.ClientSize.Height - grid.ColumnHeadersHeight) / Math.Max(1, grid.RowTemplate.Height)); } }
        void SetFirstRow(int value) { if (grid.Rows.Count == 0 || !grid.IsHandleCreated) return; grid.FirstDisplayedScrollingRowIndex = Math.Max(0, Math.Min(Math.Max(0, grid.Rows.Count - VisibleRows), value)); RefreshScroll(); }
        void RefreshScroll() {
            if (scrolling || grid == null || vertical == null || horizontal == null) return;
            scrolling = true;
            try { vertical.SetMetrics(grid.Rows.Count, VisibleRows, FirstRow); horizontal.SetMetrics(grid.Columns.GetColumnsWidth(DataGridViewElementStates.Visible), grid.ClientSize.Width, grid.HorizontalScrollingOffset); }
            finally { scrolling = false; }
        }
        void ColorRows() { if (grid == null) return; var p = WizardTheme.Current; for (int i = 0; i < grid.Rows.Count; i++) grid.Rows[i].DefaultCellStyle.BackColor = i % 2 == 0 ? p.Surface : p.SurfaceAlt; }
        void ApplyGridTheme() {
            if (grid == null) return; WizardTheme.Apply(this); var p = WizardTheme.Current;
            BackColor = p.Border; grid.EnableHeadersVisualStyles = false; grid.BackgroundColor = p.Surface; grid.GridColor = p.Outline;
            grid.DefaultCellStyle.BackColor = p.Surface; grid.DefaultCellStyle.ForeColor = p.Text; grid.DefaultCellStyle.SelectionBackColor = p.Selected; grid.DefaultCellStyle.SelectionForeColor = p.Text;
            grid.ColumnHeadersDefaultCellStyle.BackColor = p.SurfaceAlt; grid.ColumnHeadersDefaultCellStyle.ForeColor = p.Text;
            grid.ColumnHeadersDefaultCellStyle.SelectionBackColor = p.SurfaceAlt; grid.ColumnHeadersDefaultCellStyle.SelectionForeColor = p.Text;
            ApplyGridFont(); ColorRows(); grid.Invalidate();
        }
        void ApplyGridFont() {
            if (grid == null) return; grid.Font = Font; if (headerFont != null) headerFont.Dispose(); headerFont = new Font(Font, FontStyle.Bold);
            grid.ColumnHeadersDefaultCellStyle.Font = headerFont; grid.RowTemplate.Height = Math.Max(26, Font.Height + 10); grid.ColumnHeadersHeight = Math.Max(30, Font.Height + 14);
            foreach (DataGridViewRow row in grid.Rows) row.Height = grid.RowTemplate.Height;
            RefreshScroll();
        }
        void ThemeChanged(object sender, EventArgs e) { if (IsDisposed) return; if (IsHandleCreated && InvokeRequired) { BeginInvoke((Action)ApplyGridTheme); return; } ApplyGridTheme(); }
        protected override void OnFontChanged(EventArgs e) { base.OnFontChanged(e); ApplyGridFont(); }
        protected override void Dispose(bool disposing) { if (disposing) { WizardTheme.Changed -= ThemeChanged; if (headerFont != null) { headerFont.Dispose(); headerFont = null; } } base.Dispose(disposing); }
    }
}
