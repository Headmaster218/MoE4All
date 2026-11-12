"use strict";

// ── Small helpers ──────────────────────────────────────────────────────────
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
// SVG children must be born in the SVG namespace: an HTML-namespace <path> inside <svg> is
// invisible, and assigning .className on an SVG node throws in strict mode.
const SVG_TAGS = new Set(["svg", "g", "line", "path", "text", "circle", "rect", "polyline"]);
const el = (tag, attrs = {}, ...kids) => {
  const node = SVG_TAGS.has(tag)
    ? document.createElementNS("http://www.w3.org/2000/svg", tag)
    : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.setAttribute("class", v);
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null) node.append(kid);
  return node;
};
const num = (v, digits = 0) =>
  v == null || Number.isNaN(v) ? "—" : Number(v).toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits });
const rate = (v) => (v == null || !isFinite(v) || v <= 0 ? "—" : num(v, 1));
const pct = (v) => (v == null || !isFinite(v) ? "—" : `${(v * 100).toFixed(1)}%`);
const bytes = (v) => {
  if (v == null) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let n = Number(v), i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i >= 3 ? 2 : 0)} ${units[i]}`;
};
const clock = (ms) => new Date(ms).toLocaleTimeString("zh-CN", { hour12: false });
const dur = (ms) => (ms == null ? "—" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`);

const PHASE_LABEL = {
  idle: "空闲", prefill: "填充中", thinking: "思考中", tool_call: "工具调用",
  output: "输出中", done: "完成", error: "错误", cancelled: "已取消",
  Starting: "启动中", Prefill: "填充中", Decode: "解码中", Embedding: "嵌入中",
};
const phaseBadge = (phase, extra) =>
  el("span", { class: `badge ${phase || "idle"}`, text: (PHASE_LABEL[phase] || phase || "空闲") + (extra ? ` · ${extra}` : "") });

function toast(msg, kind = "") {
  const host = $("#toast-host");
  host.replaceChildren(el("div", { class: `toast ${kind}`, text: msg }));
  setTimeout(() => host.replaceChildren(), 4200);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: opts.body ? { "content-type": "application/json" } : {},
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const text = await res.text();
  let json = null;
  try { json = text ? JSON.parse(text) : null; } catch { /* non-JSON error body */ }
  if (!res.ok) throw new Error((json && (json.error || json.message)) || text || `HTTP ${res.status}`);
  return json;
}

// ── Theme ──────────────────────────────────────────────────────────────────
const THEME_KEY = "infr.console.theme";
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  localStorage.setItem(THEME_KEY, t);
}
applyTheme(localStorage.getItem(THEME_KEY) ||
  (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));
$("#theme-toggle").addEventListener("click", () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

// ── Views ──────────────────────────────────────────────────────────────────
function showView(name) {
  $$(".tab[data-view]").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.view === name)));
  $$("[data-pane]").forEach((p) => { p.hidden = p.dataset.pane !== name; });
  if (name === "launch") ensureSchema();
}
$$(".tab[data-view]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));
$$("[data-goto]").forEach((b) => b.addEventListener("click", () => showView(b.dataset.goto)));

// ── Live state ─────────────────────────────────────────────────────────────
let S = null;                 // latest /ui/state payload
let selectedRound = null;     // id shown in the sidebar
let sbTab = "frames";
let sbFollow = true;
const liveFrames = new Map(); // round id -> [{seq, data}] for the open sidebar
const roundRequests = new Map(); // round id -> request body (stripped from the 1 Hz state payload)

function connect() {
  const es = new EventSource("ui/events");
  es.addEventListener("state", (e) => { S = JSON.parse(e.data); render(); });
  es.addEventListener("frame", (e) => {
    const f = JSON.parse(e.data);
    let list = liveFrames.get(f.round);
    if (!list) { list = []; liveFrames.set(f.round, list); }
    list.push(f);
    if (list.length > 4000) list.splice(0, list.length - 4000);
    if (f.round === selectedRound && sbTab === "frames") appendFrame(f);
  });
  es.onerror = () => { setPill("bad", "控制台连接断开"); };
}

function setPill(kind, text) {
  const pill = $("#engine-pill");
  pill.replaceChildren(el("i", { class: `dot ${kind}` }), el("span", { text }));
}

function render() {
  if (!S) return;
  renderPill();
  renderUncensor();
  renderTiles();
  renderChart();
  renderLiveRounds();
  renderSlots();
  renderRecent();
  renderHistory();
  renderKeys();
  renderLogs();
  if (selectedRound != null) renderSidebarHead();
}

function renderPill() {
  const e = S.engine;
  if (!e.running && !e.reachable) return setPill("", "引擎未运行");
  if (!e.reachable) return setPill("busy", "引擎加载中…");
  const t = (e.stats && e.stats.totals) || {};
  if (t.active > 0) {
    const speeds = (e.stats.slots || []).map((s) => s.request && s.request.tokens_per_second).filter((v) => v > 0);
    const top = speeds.length ? Math.max(...speeds) : 0;
    return setPill("busy", `生成中 · ${rate(top)} t/s`);
  }
  if (t.queued > 0) return setPill("busy", `${t.queued} 排队`);
  setPill("live", "空闲");
}

// ── Tiles ──────────────────────────────────────────────────────────────────
function tile(label, value, sub, absent = false) {
  return el("div", { class: `tile${absent ? " absent" : ""}` },
    el("div", { class: "label", text: label }),
    el("div", { class: "value", text: value }),
    el("div", { class: "sub", text: sub || "" }));
}

function renderTiles() {
  const host = $("#tiles");
  const stats = S.engine.stats;
  const res = stats && (stats.models || []).map((m) => m.resources).find((r) => r);
  if (!res) {
    host.replaceChildren(tile("资源占用", "—", S.engine.reachable ? "引擎未上报" : "引擎不可达", true));
    $("#res-note").textContent = "";
    return;
  }
  const out = [];
  if (res.vram) {
    const v = res.vram;
    out.push(tile("显存占用", bytes(v.tracked_bytes), `/ ${bytes(v.total)}${v.uma ? " · UMA" : ""}`));
    out.push(tile("显存可用", bytes(v.available), v.live_budget ? "驱动实时预算" : "非实时（整堆）", !v.live_budget));
  }
  out.push(tile("进程内存", bytes(res.rss_bytes), "RSS 工作集", res.rss_bytes == null));

  const ex = res.experts || {};
  if (ex.paged) {
    out.push(tile("显存专家", `${num(ex.vram_resident)}`, `/ ${num(ex.vram_slots)} 槽`));
    if (ex.full_host_store) {
      out.push(tile("内存专家", num(ex.host_store_experts), `全量常驻 ${bytes(ex.host_store_bytes)}`));
    } else {
      out.push(tile("内存专家", num(ex.ram_resident), `/ ${num(ex.ram_slots)} 槽`));
    }
    const total = ex.hits + ex.misses;
    out.push(tile("专家命中率", total ? pct(ex.hits / total) : "—", `${num(ex.hits)} 命中 / ${num(ex.misses)} 未命中`));
  } else {
    out.push(tile("显存专家", "全驻留", "无分页会话"));
  }

  const kv = res.kv_pool;
  if (kv) {
    const vram = (kv.tiers || []).find((t) => t.tier === "vram");
    const ram = (kv.tiers || []).find((t) => t.tier === "ram");
    const freeTok = vram ? vram.free_frames * kv.page_tokens : null;
    out.push(tile("KV 池剩余", freeTok == null ? "—" : `${num(freeTok)} tok`,
      vram ? `${num(vram.free_frames)}/${num(vram.capacity_frames)} 帧 · ${kv.page_tokens} tok/帧` : ""));
    if (ram) out.push(tile("KV 内存层", `${num(ram.held_frames)}`, `/ ${num(ram.capacity_frames)} 帧 · ${bytes(ram.held_frames * kv.frame_bytes)}`));
  } else {
    out.push(tile("KV 池剩余", "—", "kv.paged 未开启", true));
  }
  host.replaceChildren(...out);
  $("#res-note").textContent = ex.gpu_owned ? "专家计数为 GPU 权威，主机侧为上次交接状态" : "";
}

// ── Speed chart ────────────────────────────────────────────────────────────
// Two independent y scales: prefill runs at thousands of t/s while decode runs at tens, so a
// shared axis would flatten decode to a line on the floor. Each axis is labelled with its own max.
function renderChart() {
  const svg = $("#speed-chart");
  const s = S.series || { t: [], decode: [], prefill: [] };
  const W = 600, H = 150, padL = 40, padR = 46, padT = 10, padB = 18;
  const n = s.t.length;
  svg.replaceChildren();
  if (n < 2) {
    svg.append(el("text", { x: W / 2 - 40, y: H / 2 }, "等待采样…"));
    $("#chart-window").textContent = "";
    return;
  }
  const maxD = Math.max(1, ...s.decode);
  const maxP = Math.max(1, ...s.prefill);
  const x = (i) => padL + (i / (n - 1)) * (W - padL - padR);
  const y = (v, max) => H - padB - (v / max) * (H - padT - padB);

  for (let g = 0; g <= 4; g++) {
    const gy = padT + (g / 4) * (H - padT - padB);
    svg.append(el("line", { class: "grid", x1: padL, y1: gy, x2: W - padR, y2: gy }));
  }
  const path = (vals, max) => vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v, max).toFixed(1)}`).join("");
  for (const [vals, max, cls] of [[s.prefill, maxP, "prefill"], [s.decode, maxD, "decode"]]) {
    const d = path(vals, max);
    svg.append(el("path", { class: `area series-${cls}`, d: `${d}L${x(n - 1).toFixed(1)} ${H - padB}L${padL} ${H - padB}Z`, fill: `var(--${cls})` }));
    svg.append(el("path", { class: `line series-${cls}`, d }));
  }
  for (let g = 0; g <= 4; g++) {
    const gy = padT + (g / 4) * (H - padT - padB) + 3;
    const frac = 1 - g / 4;
    svg.append(el("text", { x: 2, y: gy, fill: "var(--decode)" }, num(maxD * frac, maxD < 20 ? 1 : 0)));
    svg.append(el("text", { x: W - padR + 4, y: gy, fill: "var(--prefill)" }, num(maxP * frac, 0)));
  }
  $("#chart-window").textContent = `${Math.round((s.t[n - 1] - s.t[0]) / 1000)} s · 左 decode / 右 prefill`;
}

// ── Conversation windows ───────────────────────────────────────────────────
// The clickable list is the console's OWN rounds: only they carry the captured SSE frames the
// sidebar renders. Engine slots are a separate, non-clickable occupancy table — a slot is reused
// across turns, so it cannot identify a conversation.
function renderLiveRounds() {
  const body = $("#live-rounds");
  const all = S.rounds || [];
  const live = all.filter((r) => r.status === "streaming");
  const rows = (live.length ? live : all.slice(0, 4));
  if (!rows.length) {
    body.replaceChildren(el("tr", {}, el("td", { colspan: 8, class: "empty", text: "还没有请求经过控制台" })));
    return;
  }
  body.replaceChildren(...rows.map((r) => {
    const tr = el("tr", { class: `clickable${r.id === selectedRound ? " selected" : ""}` },
      el("td", { text: clock(r.started_at_ms) }),
      el("td", { text: r.key_label }),
      el("td", {}, phaseBadge(r.status === "streaming" ? r.phase : r.status)),
      el("td", { text: r.tools.length ? r.tools.join(", ") : "—" }),
      el("td", { class: "num", text: rate(r.status === "streaming" ? r.live_tps : r.decode_tps) }),
      el("td", { class: "num", text: num(r.completion_tokens) }),
      el("td", { class: "num", text: pct(reuseRatio(r)) }),
      el("td", { class: "num", text: num(r.ttft_ms) }));
    tr.addEventListener("click", () => openRound(r.id));
    return tr;
  }));
}

function renderSlots() {
  const body = $("#slots");
  const stats = S.engine.stats;
  const slots = (stats && stats.slots) || [];
  const totals = (stats && stats.totals) || {};
  $("#slot-note").textContent = stats ? `${totals.busy_slots || 0}/${totals.total_slots || 0} 占用 · ${totals.queued || 0} 排队` : "";
  if (!slots.length) { body.replaceChildren(el("tr", {}, el("td", { colspan: 7, class: "empty", text: "引擎未上报槽位" }))); return; }
  body.replaceChildren(...slots.map((slot) => {
    const req = slot.request;
    const ctx = req && req.context_limit ? req.context_tokens / req.context_limit : null;
    return el("tr", {},
      el("td", { text: `${slot.kind === "chat" ? "" : "emb "}${slot.ordinal}/${slot.capacity}` }),
      el("td", {}, phaseBadge(req ? req.phase : "idle")),
      el("td", { class: "num", text: req ? rate(req.tokens_per_second) : "—" }),
      el("td", { class: "num", text: req ? num(req.prompt_tokens) : "—" }),
      el("td", { class: "num", text: req && req.prompt_tokens ? pct(req.cached_prompt_tokens / req.prompt_tokens) : "—" }),
      el("td", { class: "num", text: req ? num(req.completion_tokens) : "—" }),
      el("td", {}, ctx == null ? "—" : el("div", { class: "bar", title: `${num(req.context_tokens)} / ${num(req.context_limit)}` }, el("i", { style: `width:${Math.min(100, ctx * 100).toFixed(1)}%` }))));
  }));
}

// ── Round tables ───────────────────────────────────────────────────────────
function reuseRatio(r) {
  return r.prompt_tokens ? r.cached_prompt_tokens / r.prompt_tokens : null;
}

function roundCells(r) {
  return [
    el("td", { class: "num", text: num(r.prompt_tokens) }),
    el("td", { class: "num", text: num(r.cached_prompt_tokens) }),
    el("td", { class: "num", text: num(r.prompt_tokens - r.cached_prompt_tokens) }),
    el("td", { class: "num", text: pct(reuseRatio(r)) }),
    el("td", { class: "num", text: r.reuse_est ? pct(r.reuse_est.ratio) : "—", title: r.reuse_est ? `代理侧前缀估计：${r.reuse_est.messages} 条消息 / ${num(r.reuse_est.chars)} 字符` : "" }),
    el("td", { class: "num", text: num(r.completion_tokens) }),
    el("td", { class: "num", text: num(r.ttft_ms) }),
    el("td", { class: "num", text: rate(r.prefill_tps) }),
    el("td", { class: "num", text: rate(r.decode_tps) }),
  ];
}

function renderRecent() {
  const body = $("#recent");
  const rows = (S.rounds || []).slice(0, 8);
  if (!rows.length) { body.replaceChildren(el("tr", {}, el("td", { colspan: 10, class: "empty", text: "还没有请求经过控制台" }))); return; }
  body.replaceChildren(...rows.map((r) => {
    const tr = el("tr", { class: "clickable" },
      el("td", { text: clock(r.started_at_ms) }),
      el("td", { text: r.key_label }),
      el("td", {}, phaseBadge(r.status === "streaming" ? r.phase : r.status)),
      el("td", {}, r.uncensor ? el("span", { class: "badge on", text: "开" }) : el("span", { class: "frame-seq", text: "关" })),
      ...roundCells(r).filter((_, i) => [0, 1, 5, 6, 7, 8].includes(i)));
    tr.addEventListener("click", () => openRound(r.id));
    return tr;
  }));
}

function renderHistory() {
  const filter = $("#history-key").value;
  const body = $("#history");
  const rows = (S.rounds || []).filter((r) => !filter || r.key_id === filter);
  const sel = $("#history-key");
  const want = ["", ...(S.keys || []).map((k) => k.id)];
  if (sel.options.length !== want.length) {
    sel.replaceChildren(el("option", { value: "", text: "全部 Key" }),
      ...(S.keys || []).map((k) => el("option", { value: k.id, text: k.label })));
    sel.value = filter;
  }
  if (!rows.length) { body.replaceChildren(el("tr", {}, el("td", { colspan: 16, class: "empty", text: "无记录" }))); return; }
  body.replaceChildren(...rows.map((r) => {
    const tr = el("tr", { class: `clickable${r.id === selectedRound ? " selected" : ""}` },
      el("td", { class: "num", text: r.id }),
      el("td", { text: clock(r.started_at_ms) }),
      el("td", { text: r.key_label }),
      el("td", {}, phaseBadge(r.status === "streaming" ? r.phase : r.status)),
      el("td", { text: r.tools.length ? r.tools.join(", ") : "—" }),
      el("td", {}, r.uncensor ? el("span", { class: "badge on", text: "开" }) : el("span", { class: "frame-seq", text: "关" })),
      ...roundCells(r),
      el("td", { class: "num", text: dur(r.duration_ms) }));
    tr.addEventListener("click", () => openRound(r.id));
    return tr;
  }));
}
$("#history-clear").addEventListener("click", async () => {
  try { await api("ui/rounds", { method: "DELETE" }); liveFrames.clear(); roundRequests.clear(); toast("历史已清空", "ok"); }
  catch (e) { toast(e.message, "bad"); }
});

// ── Keys ───────────────────────────────────────────────────────────────────
function renderKeys() {
  const body = $("#keys");
  const keys = S.keys || [];
  if (!keys.length) { body.replaceChildren(el("tr", {}, el("td", { colspan: 12, class: "empty", text: "还没有 Key，先添加一个" }))); return; }
  body.replaceChildren(...keys.map((k) => el("tr", {},
    el("td", { text: k.label }),
    el("td", {}, el("code", { class: "frame-seq", text: k.token })),
    el("td", { class: "num", text: num(k.requests) }),
    el("td", { class: "num", text: num(k.prompt_tokens) }),
    el("td", { class: "num", text: num(k.completion_tokens) }),
    el("td", { class: "num", text: num(k.cached_tokens) }),
    el("td", { class: "num", text: num(k.prompt_tokens - k.cached_tokens) }),
    el("td", { class: "num", text: k.prompt_tokens ? pct(k.cached_tokens / k.prompt_tokens) : "—" }),
    el("td", { class: "num", text: num(k.ttft_avg_ms) }),
    el("td", { class: "num", text: rate(k.prefill_avg_tps) }),
    el("td", { class: "num", text: rate(k.decode_avg_tps) }),
    el("td", {}, el("button", {
      class: "icon-btn", text: "删除",
      onclick: async () => {
        if (!confirm(`删除 Key「${k.label}」？它的计数一并清除。`)) return;
        try { await api(`ui/keys/${k.id}`, { method: "DELETE" }); toast("已删除", "ok"); }
        catch (e) { toast(e.message, "bad"); }
      },
    })))));
}
$("#new-key-add").addEventListener("click", async () => {
  const label = $("#new-key-label").value.trim();
  if (!label) return toast("先填名称", "bad");
  try {
    const k = await api("ui/keys", { method: "POST", body: { label, token: $("#new-key-token").value.trim() || null } });
    $("#new-key-label").value = ""; $("#new-key-token").value = "";
    toast(`已添加，令牌：${k.token}`, "ok");
  } catch (e) { toast(e.message, "bad"); }
});

// ── Logs ───────────────────────────────────────────────────────────────────
function renderLogs() {
  $("#engine-log").textContent = (S.engine.log_tail || []).join("\n") || "（无输出）";
  $("#log-note").textContent = S.engine.pid ? `pid ${S.engine.pid}` : "";
  $("#console-log").textContent = (S.console_log || []).join("\n");
}

// ── Sidebar ────────────────────────────────────────────────────────────────
function openRound(id) {
  selectedRound = id;
  $("#sidebar").classList.add("open");
  if (!liveFrames.has(id) || !roundRequests.has(id)) loadFrames(id);
  renderSidebar();
  renderHistory();
}
async function loadFrames(id) {
  try {
    const detail = await api(`ui/rounds/${id}`);
    // A live round's SSE capture is ahead of the server snapshot; only fill frames when absent.
    if (!liveFrames.has(id)) liveFrames.set(id, detail.frames || []);
    roundRequests.set(id, detail.round && detail.round.request);
    if (selectedRound === id) renderSidebar();
  } catch (e) { toast(e.message, "bad"); }
}
$("#sb-close").addEventListener("click", () => {
  $("#sidebar").classList.remove("open");
  selectedRound = null;
  renderHistory();
});
$("#sb-follow").addEventListener("click", (e) => {
  sbFollow = !sbFollow;
  e.currentTarget.setAttribute("aria-pressed", String(sbFollow));
});
$$("#sb-tabs .tab").forEach((b) => b.addEventListener("click", () => {
  sbTab = b.dataset.sb;
  $$("#sb-tabs .tab").forEach((o) => o.setAttribute("aria-selected", String(o === b)));
  renderSidebar();
}));

function currentRound() { return (S && S.rounds || []).find((r) => r.id === selectedRound); }

function renderSidebarHead() {
  const r = currentRound();
  if (!r) return;
  $("#sb-title").textContent = `#${r.id} · ${r.key_label}`;
  $("#sb-phase").replaceWith(Object.assign(phaseBadge(r.status === "streaming" ? r.phase : r.status,
    r.tools.length ? r.tools.join(",") : null), { id: "sb-phase" }));
}

function highlight(json) {
  const pre = el("pre", { class: "json" });
  const text = typeof json === "string" ? json : JSON.stringify(json, null, 2);
  const re = /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) pre.append(text.slice(last, m.index));
    const cls = m[1] ? (m[2] ? "k" : "s") : m[3] ? "b" : "n";
    pre.append(el("span", { class: cls, text: m[0] }));
    last = m.index + m[0].length;
  }
  pre.append(text.slice(last));
  return pre;
}

function frameNode(f) {
  let payload;
  try { payload = JSON.parse(f.data); } catch { payload = f.data; }
  return el("div", {}, el("div", { class: "frame-seq", text: `#${f.seq}` }), highlight(payload));
}
function appendFrame(f) {
  const body = $("#sb-body");
  body.append(frameNode(f));
  while (body.childElementCount > 4000) body.firstElementChild.remove();
  if (sbFollow) body.scrollTop = body.scrollHeight;
}

function renderSidebar() {
  renderSidebarHead();
  const body = $("#sb-body");
  const r = currentRound();
  body.replaceChildren();
  if (!r) { body.append(el("div", { class: "empty", text: "该轮次已不在保留窗口内" })); return; }
  const frames = liveFrames.get(r.id) || [];
  if (sbTab === "frames") {
    if (!frames.length) body.append(el("div", { class: "empty", text: "等待流式帧…" }));
    else frames.forEach((f) => body.append(frameNode(f)));
  } else if (sbTab === "request") {
    body.append(highlight(roundRequests.get(r.id) || { note: "请求体未保留" }));
  } else {
    const parts = assemble(frames);
    if (sbTab === "tools") {
      if (!parts.tools.length) body.append(el("div", { class: "empty", text: "本轮没有工具调用" }));
      parts.tools.forEach((t) => body.append(highlight(t)));
    } else {
      const text = sbTab === "output" ? parts.content : parts.reasoning;
      body.append(text ? el("div", { class: `prose${sbTab === "thinking" ? " thinking" : ""}`, text })
        : el("div", { class: "empty", text: sbTab === "output" ? "无输出文本" : "无思考文本" }));
    }
  }
  if (sbFollow) body.scrollTop = body.scrollHeight;
}

// Fold the SSE deltas back into what the client saw. Tool arguments arrive one fragment at a
// time, so they are concatenated per (index) rather than replaced.
function assemble(frames) {
  let content = "", reasoning = "";
  const tools = new Map();
  for (const f of frames) {
    let chunk;
    try { chunk = JSON.parse(f.data); } catch { continue; }
    for (const choice of chunk.choices || []) {
      const d = choice.delta || {};
      if (typeof d.content === "string") content += d.content;
      if (typeof d.reasoning_content === "string") reasoning += d.reasoning_content;
      for (const tc of d.tool_calls || []) {
        const i = tc.index ?? 0;
        const slot = tools.get(i) || { name: "", arguments: "" };
        if (tc.function && tc.function.name) slot.name += tc.function.name;
        if (tc.function && tc.function.arguments) slot.arguments += tc.function.arguments;
        tools.set(i, slot);
      }
    }
  }
  return { content, reasoning, tools: Array.from(tools.values()) };
}

// ── Uncensor hot toggle ────────────────────────────────────────────────────
function renderUncensor() {
  const u = S.uncensor || {};
  const btn = $("#uncensor-toggle");
  btn.textContent = `去审查：${u.default_on ? "开" : "关"}`;
  btn.style.color = u.default_on ? "var(--warn)" : "";
  btn.style.borderColor = u.default_on ? "var(--warn)" : "";
  btn.title = u.vector_loaded
    ? "逐请求注入 uncensor 投影（不改引擎启动参数）"
    : "引擎未加载去审查向量（uncensor.vector 未配置），开关不会生效";
}

$("#uncensor-toggle").addEventListener("click", async () => {
  const u = S.uncensor || {};
  if (u.default_on) {
    try { await api("ui/uncensor", { method: "POST", body: { on: false } }); toast("去审查已关闭", "ok"); }
    catch (e) { toast(e.message, "bad"); }
    return;
  }
  if (!u.vector_loaded) {
    return toast("引擎未加载去审查向量，请先在启动参数里设置 uncensor.vector", "bad");
  }
  riskModal(async () => {
    await api("ui/uncensor", { method: "POST", body: { on: true, consent: true } });
    toast("去审查已开启，后续请求将注入投影", "ok");
  });
});

// One consent gate for both ways the projection can be turned on: the per-request hot toggle and a
// launch that names a direction vector. `onConfirm` runs only after the box is ticked.
function riskModal(onConfirm) {
  const host = $("#modal-host");
  const agree = el("input", { type: "checkbox", id: "risk-agree" });
  const confirmBtn = el("button", { class: "btn danger", text: "同意并开启", disabled: "disabled" });
  agree.addEventListener("change", () => confirmBtn.toggleAttribute("disabled", !agree.checked));
  confirmBtn.addEventListener("click", async () => {
    confirmBtn.disabled = true;
    try {
      await onConfirm();
      host.replaceChildren();
    } catch (e) {
      confirmBtn.disabled = false;
      toast(e.message, "bad");
    }
  });
  host.replaceChildren(el("div", { class: "modal-backdrop" },
    el("div", { class: "modal" },
      el("h3", { text: "开启去审查投影的风险" }),
      el("div", { class: "risk" },
        "去审查会在推理时移除模型的对齐/拒答方向，输出将不再受原厂安全约束。",
        el("ul", {},
          el("li", { text: "可能产出违法、危险或严重不实的内容，后果由部署者承担。" }),
          el("li", { text: "模型能力可能下降，尤其在指令跟随与工具调用上。" }),
          el("li", { text: "不得用于面向公众或未成年人的服务。" }),
          el("li", { text: "本开关逐请求生效，可随时关闭；已生成的内容不会撤回。" }))),
      el("label", { class: "check", for: "risk-agree" }, agree, el("span", { text: "我已理解上述风险，并确认本部署仅用于我自有、合规的用途。" })),
      el("div", { class: "actions" },
        el("button", { class: "btn ghost", text: "取消", onclick: () => host.replaceChildren() }),
        confirmBtn))));
}

// ── Launch parameters ──────────────────────────────────────────────────────
let SCHEMA = null;           // {schema:[{path,ty,default,doc}], script_knobs:[…], sections:{…}}
let FORM = {};               // path -> string value (only what the user touched)
let PRESET = "conservative";

async function ensureSchema() {
  if (SCHEMA) return;
  try {
    SCHEMA = await api("ui/schema");
    FORM = {};
    renderLaunch();
  } catch (e) { $("#launch-note").textContent = `读取配置表失败：${e.message}`; }
}

function sectionOf(path) { return path.includes(".") ? path.split(".")[0] : "其它"; }
const SECTION_LABEL = {
  device: "设备与显存", kv: "KV 缓存与分页", paging: "专家分页", serve: "服务与鉴权",
  kernels: "内核选择", spec: "投机解码 / MTP", prof: "诊断与性能剖析", vision: "视觉",
  uncensor: "去审查", audio: "音频", embedding: "嵌入", model: "模型", debug: "调试",
  hub: "模型下载", session: "会话", logging: "日志",
};

function fieldNode(f) {
  const cur = FORM[f.path] !== undefined ? FORM[f.path] : f.default;
  const dirty = FORM[f.path] !== undefined && FORM[f.path] !== f.default;
  const wrap = el("div", { class: `field${dirty ? " dirty" : ""}`, "data-path": f.path });
  const id = `knob-${f.path.replace(/[.:]/g, "-")}`;
  let input;
  if (f.ty === "bool") {
    input = el("input", { type: "checkbox", id });
    input.checked = cur === "true";
    wrap.append(el("label", { class: "check", for: id }, input, el("span", { text: f.path })));
  } else {
    // `none` is how the engine spells an unset Option; an empty box says the same thing without
    // looking like a value the operator typed.
    input = el("input", { type: "text", id, value: cur === "none" ? "" : cur, placeholder: f.default });
    wrap.append(el("label", { for: id, text: `${f.path}  ·  ${f.ty}` }), input);
  }
  input.addEventListener("change", () => {
    const v = f.ty === "bool" ? String(input.checked) : input.value.trim();
    if (v === f.default || (v === "" && f.default === "none")) delete FORM[f.path];
    else FORM[f.path] = v;
    wrap.classList.toggle("dirty", FORM[f.path] !== undefined);
    renderCmdline();
  });
  if (f.doc) wrap.append(el("div", { class: "doc", text: f.doc }));
  return wrap;
}

function groupNode(title, fields, open) {
  const d = el("details", { class: "group", ...(open ? { open: "open" } : {}) },
    el("summary", { text: `${title} · ${fields.length}` }),
    el("div", { class: "group-body" }, el("div", { class: "knob-grid" }, fields.map(fieldNode))));
  return d;
}

function renderLaunch() {
  const scriptSet = new Set(SCHEMA.script_knobs || []);
  const manualSet = new Set(SCHEMA.manual_knobs || []);
  const all = SCHEMA.schema || [];
  // In 全手动 the memory/diagnostic knobs MOVE into the next-level menu rather than being shown
  // twice; in the two preset modes they stay in the common list, as the wizard does.
  const manual = PRESET === "manual";
  // The preset picker IS the auto_profile control; a second text box for it would let the two
  // disagree about what the command line says.
  const isProfile = (f) => f.path === "device.auto_profile";
  const basic = all.filter((f) => !isProfile(f) && scriptSet.has(f.path) && !(manual && manualSet.has(f.path)));
  const rest = all.filter((f) => !isProfile(f) && !scriptSet.has(f.path) && !manualSet.has(f.path));

  $("#knobs-basic").replaceChildren(...basic.map(fieldNode));

  const bySection = new Map();
  for (const f of rest) {
    const s = sectionOf(f.path);
    if (!bySection.has(s)) bySection.set(s, []);
    bySection.get(s).push(f);
  }
  const groups = Array.from(bySection.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([s, fs]) => groupNode(SECTION_LABEL[s] || s, fs, false));
  $("#knobs-advanced").replaceChildren(...groups);

  $("#knobs-manual").replaceChildren(groupNode("显存 / 内存 / 分页 / 诊断",
    all.filter((f) => manualSet.has(f.path)), true));

  $$("#presets .preset").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.dataset.preset === PRESET));
  });
  $("#manual-card").hidden = !manual;
  renderCmdline();
}

$("#knob-search").addEventListener("input", (e) => {
  const q = e.target.value.trim().toLowerCase();
  $$("#knobs-advanced .field, #knobs-basic .field, #knobs-manual .field").forEach((f) => {
    f.hidden = q !== "" && !f.dataset.path.toLowerCase().includes(q);
  });
  $$("#knobs-advanced details.group, #knobs-manual details.group").forEach((g) => {
    if (!q) { g.hidden = false; return; }
    const visible = $$(".field", g).some((f) => !f.hidden);
    g.hidden = !visible;
    g.open = visible;
  });
});

$$("#presets .preset").forEach((b) => b.addEventListener("click", () => {
  PRESET = b.dataset.preset;
  if (SCHEMA) renderLaunch();
}));

// Mirrors launch.rs::build_args, so the previewed command is the one that runs.
function renderCmdline() {
  if (!SCHEMA) return;
  const args = [SCHEMA.binary || "infr.exe", "serve"];
  for (const f of SCHEMA.schema || []) {
    if (!f.path.startsWith("cli:")) continue;
    const v = FORM[f.path];
    if (v === undefined || v === "" || v === f.default) continue;
    if (f.flag) args.push(f.flag, v); else args.push(v);
  }
  if (PRESET !== "manual") args.push("--set", `device.auto_profile=${PRESET}`);
  for (const [k, v] of Object.entries(FORM)) {
    if (k.startsWith("cli:") || k === "serve.shutdown_file") continue;
    if (k === "device.auto_profile" && PRESET !== "manual") continue;
    args.push("--set", `${k}=${v}`);
  }
  args.push("--set", `serve.shutdown_file=${SCHEMA.shutdown_file || "<控制台目录>/engine.shutdown"}`);
  $("#cmdline").textContent = args.map((a) => (/\s/.test(a) ? `"${a}"` : a)).join(" ");
}

$("#launch-reset").addEventListener("click", () => { FORM = {}; renderLaunch(); toast("已恢复默认", "ok"); });
$("#launch-copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("#cmdline").textContent); toast("已复制", "ok"); }
  catch { toast("浏览器拒绝了剪贴板访问", "bad"); }
});
$("#launch-start").addEventListener("click", () => {
  const vector = FORM["uncensor.vector"];
  if (vector) riskModal(() => doLaunch(true));
  else doLaunch(false);
});

async function doLaunch(consent) {
  const btn = $("#launch-start");
  btn.disabled = true;
  try {
    const r = await api("ui/launch", { method: "POST", body: { preset: PRESET, set: FORM, consent } });
    toast(`引擎已启动，pid ${r.pid}`, "ok");
  } catch (e) { toast(e.message, "bad"); throw e; }
  finally { btn.disabled = false; }
}
$("#launch-stop").addEventListener("click", async () => {
  if (!confirm("停止引擎？进行中的请求会被中断。")) return;
  try { await api("ui/stop", { method: "POST" }); toast("已请求停止", "ok"); }
  catch (e) { toast(e.message, "bad"); }
});

// ── Boot ───────────────────────────────────────────────────────────────────
connect();
api("ui/state").then((s) => { S = s; render(); }).catch((e) => setPill("bad", e.message));