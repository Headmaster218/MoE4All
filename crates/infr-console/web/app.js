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

// Display cap for the raw frame list; storage keeps 4000 so the assembled reply stays complete.
const FRAME_LIMITS = [20, 50, 100, 200, 500, 0]; // 0 = 全部
const FRAME_LIMIT_KEY = "infr.console.frameLimit";
let frameLimit = Number(localStorage.getItem(FRAME_LIMIT_KEY) || 50);
if (!FRAME_LIMITS.includes(frameLimit)) frameLimit = 50;

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
  renderContext();
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

// ── Responses context lifecycle ──────────────────────────────────────────────
// The engine's ResponseStore is the authority for creation / chain reuse / LRU eviction / disk
// mirror; the console cannot infer those from traffic alone. `clock` and `last_used` are monotonic
// operation ticks, NOT wall time — only `created` (Unix seconds) maps to a real timestamp.
const RESP_STATUS = {
  completed: ["完成", "done"], in_progress: ["生成中", "output"], queued: ["排队", "idle"],
  incomplete: ["未完成", "idle"], failed: ["失败", "error"], cancelled: ["已取消", "cancelled"],
};
function renderContext() {
  const rs = S.engine.stats && S.engine.stats.response_store;
  const tiles = $("#ctx-tiles"), note = $("#ctx-note");
  const entBody = $("#ctx-entries"), convBody = $("#ctx-conversations");
  const emptyRow = (body, cols, text) =>
    body.replaceChildren(el("tr", {}, el("td", { colspan: cols, class: "empty", text })));
  if (!rs) {
    note.textContent = S.engine.reachable ? "引擎未上报" : "引擎不可达";
    tiles.replaceChildren(tile("上下文存储", "—", "等待引擎上报", true));
    emptyRow(entBody, 10, "暂无数据"); emptyRow(convBody, 3, "暂无数据");
    return;
  }
  if (!rs.enabled) {
    note.textContent = "未启用";
    tiles.replaceChildren(tile("上下文存储", "关闭", "serve.response_store_entries = 0", true));
    emptyRow(entBody, 10, "存储未启用：请求为无状态，previous_response_id / conversation 会被拒绝（400）。在启动参数里把 serve.response_store_entries 设为大于 0 即可开启。");
    emptyRow(convBody, 3, "—");
    return;
  }
  const c = rs.counters || {};
  note.textContent = rs.mirror ? `落盘镜像 · ${rs.mirror}` : "仅内存（重启即失）";
  tiles.replaceChildren(
    tile("存储占用", `${num(rs.len)} / ${num(rs.capacity)}`, rs.len >= rs.capacity ? "已满 · 下次写入触发驱逐" : "LRU 容量"),
    tile("会话游标", num(rs.conversation_count), `${num(rs.item_count)} 个条目引用`),
    tile("累计创建", num(c.created), `已驱逐 ${num(c.evicted)}`),
    tile("链式复用", num(c.chain_hits), `失配 ${num(c.chain_misses)} · previous_response_id / conversation`),
    tile("逻辑时钟", num(rs.clock), rs.recency_dirty ? "recency 待落盘" : "recency 已同步"));

  const entries = rs.entries || [];
  if (!entries.length) emptyRow(entBody, 10, "还没有响应被存储");
  else entBody.replaceChildren(...entries.map((e) => {
    const nextEvict = e.lru_rank === rs.len;
    const st = RESP_STATUS[e.status] || [e.status || "—", "idle"];
    const tr = el("tr", { class: nextEvict ? "selected" : "" },
      el("td", { class: "num", text: num(e.lru_rank) }),
      el("td", {}, el("code", { class: "frame-seq", text: e.id })),
      el("td", { text: e.model || "—" }),
      el("td", {}, el("span", { class: `badge ${st[1]}`, text: st[0] })),
      el("td", { class: "num", text: num(e.turns) }),
      el("td", { class: "num", text: num(e.input_items) }),
      el("td", { class: "num", text: num(e.output_items) }),
      el("td", { text: e.created ? clock(e.created * 1000) : "—" }),
      el("td", { class: "num", text: num(rs.clock - e.last_used) }),
      el("td", { text: (e.conversations || []).join(", ") || "—" }));
    if (nextEvict) tr.title = "最久未使用：容量满时下一个被驱逐";
    return tr;
  }));

  const convs = rs.conversations || [];
  if (!convs.length) emptyRow(convBody, 3, "还没有命名会话");
  else convBody.replaceChildren(...convs.map((v) => el("tr", {},
    el("td", { text: v.name }),
    el("td", {}, el("code", { class: "frame-seq", text: v.cursor })),
    el("td", {}, v.alive
      ? el("span", { class: "badge output", text: "存活" })
      : el("span", { class: "badge error", text: "已驱逐 · 游标悬空" })))));
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

function framesBar(total) {
  const sel = el("select", { class: "icon-btn", title: "流式 JSON 列表只保留最近 N 条" },
    ...FRAME_LIMITS.map((n) => el("option", { value: n, text: n === 0 ? "全部" : `${n} 条` })));
  sel.value = String(frameLimit);
  sel.addEventListener("change", () => {
    frameLimit = Number(sel.value);
    localStorage.setItem(FRAME_LIMIT_KEY, String(frameLimit));
    renderSidebar();
  });
  return el("div", { class: "frames-bar" },
    el("span", { text: "保留最近" }), sel,
    el("span", { class: "frame-seq frame-count", text: `共 ${total} 帧` }));
}

// The assembled reply, fully expanded by default; the summary line collapses it on demand.
function replyPanel(frames) {
  const parts = assemble(frames);
  return el("details", { class: "reply", open: "open" },
    el("summary", {}, el("span", { text: "回复内容" }), el("span", { class: "frame-seq", text: "点击折叠 / 展开" })),
    el("div", { class: "prose reply-text", text: parts.content }));
}

function frameNode(f) {
  let payload;
  try { payload = JSON.parse(f.data); } catch { payload = f.data; }
  return el("div", {}, el("div", { class: "frame-seq", text: `#${f.seq}` }), highlight(payload));
}
function appendFrame(f) {
  const body = $("#sb-body");
  const prose = $(".reply-text", body);
  if (prose) {
    try {
      const chunk = JSON.parse(f.data);
      for (const choice of chunk.choices || []) {
        const d = choice.delta || {};
        if (typeof d.content === "string" && d.content) prose.append(d.content);
      }
    } catch { /* non-JSON frame */ }
  }
  const list = $(".frame-list", body);
  if (list) {
    const first = list.firstElementChild;
    if (first && first.classList.contains("empty")) list.replaceChildren();
    list.append(frameNode(f));
    const cap = frameLimit > 0 ? frameLimit : 4000;
    while (list.childElementCount > cap) list.firstElementChild.remove();
  }
  const count = $(".frame-count", body);
  if (count) count.textContent = `共 ${(liveFrames.get(f.round) || []).length} 帧`;
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
    body.append(framesBar(frames.length));
    body.append(replyPanel(frames));
    const list = el("div", { class: "frame-list" });
    const tail = frameLimit > 0 ? frames.slice(-frameLimit) : frames;
    if (!tail.length) list.append(el("div", { class: "empty", text: "等待流式帧…" }));
    else tail.forEach((f) => list.append(frameNode(f)));
    body.append(list);
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
let SEL_CAT = null;          // 二级菜单当前选中的分类 id（tier:key）
let KNOB_QUERY = "";         // 参数搜索词
let SETTINGS_CATS = [];      // 当前可见分类（含过滤后的字段），供详情 pane 查找

async function ensureSchema() {
  if (SCHEMA) return;
  try {
    SCHEMA = await api("ui/schema");
    // Restore the operator's saved form (preset + touched knobs) so a reload or a console restart
    // picks up where they left off. Keys no longer in the live schema are dropped, so a stale
    // settings.json from an older engine cannot hard-error the next launch.
    let saved = null;
    try { saved = await api("ui/settings"); } catch { /* nothing saved yet */ }
    FORM = {};
    if (saved && saved.set) {
      const known = new Set((SCHEMA.schema || []).map((f) => f.path));
      for (const [k, v] of Object.entries(saved.set)) if (known.has(k)) FORM[k] = v;
    }
    if (saved && ["conservative", "aggressive", "manual"].includes(saved.preset)) PRESET = saved.preset;
    renderLaunch();
    refreshVectorStatus();
  } catch (e) { $("#launch-note").textContent = `读取配置表失败：${e.message}`; }
}

function sectionOf(path) { return path.includes(".") ? path.split(".")[0] : "其它"; }
const SECTION_LABEL = {
  device: "设备与显存", kv: "KV 缓存与分页", paging: "专家分页", serve: "服务与鉴权",
  kernels: "内核选择", spec: "投机解码 / MTP", prof: "诊断与性能剖析", vision: "视觉",
  uncensor: "去审查", audio: "音频", embedding: "嵌入", model: "模型", debug: "调试",
  hub: "模型下载", session: "会话", logging: "日志",
};

// path -> [中文名, 中文说明]。说明留空时回退到 schema 自带的 doc（cli:* 已是中文）。
const KNOB_ZH = {
  "device.dev": ["设备选择", "原始设备串，如 vulkan:0 / cpu。CLI 与 Vulkan 各有一套解析，填错会在启动时报错。"],
  "device.ctx": ["上下文长度", "模型可用的上下文 token 数，支持 32k / 128k 等写法。留空按模型默认。"],
  "device.auto_profile": ["自动调档策略", "未显式指定的值按哪档策略自动选；显式填写的值始终优先。由上方预设按钮控制。"],
  "device.vram_budget": ["显存总预算", "本后端的显存总上限，含权重、KV、运行时 scratch 与分页 arena。百分比按显存总量折算。"],
  "device.ram_budget": ["内存总预算", "进程常驻内存目标：规划器扣除当前工作集，余量给权重缓存；0 关闭主机缓存。百分比按物理内存折算。"],
  "device.mem_guard_headroom": ["提交内存熔断水位", "系统剩余提交内存低于水位时进程以 87 退出。默认关：系统托管页面文件会让水位漂移，误杀健康启动。"],
  "device.mem_guard_ram_headroom": ["物理内存熔断水位", "可用物理内存低于此值即熔断，默认 4GiB。物理内存不像提交上限那样漂移，故默认启用；0 关闭。"],
  "device.mem_guard_poll_ms": ["熔断检查间隔", "熔断器轮询间隔（毫秒），上传闸口另有内联检查兜底。"],
  "device.vram_reserve": ["显存额外保留", "在分配器安全余量之外再多留的显存。百分比按显存总量折算。"],
  "device.ubatch": ["prefill 微批行数", "prefill 每个微批的 token 行数。留空走档位 / 核显自适应链。"],
  "device.ubatch_specified": ["ubatch 显式指定标记", "内部标记：用户是否显式给过 ubatch（哪怕值不可用）。一般不要手改。"],
  "device.ubatch_parallel": ["并发 prefill 批行数", "多路并发（serve -np N）共享 GPU 时每路的 prefill 批行数。"],
  "device.ubatch_parallel_specified": ["并发批显式指定标记", "内部标记：并发批行数是否显式配置；未配置时继承单请求的批行数。一般不要手改。"],
  "device.submit_dispatches": ["提交拆分上限", "单次提交最多拆成多少个 dispatch，0 不拆。留空用实测默认。"],
  "device.subgroup_pref": ["子组宽度偏好", "指定 16 或 32；留空按厂商推导。"],
  "device.threads": ["CPU 工作线程数", "rayon 工作线程数（CLI -t），会发布为 RAYON_NUM_THREADS。"],
  "device.decode_prepare_ahead": ["decode 计划预备", "当前步执行时在 CPU worker 上预备下一 token 的不可变计划。不执行未确认 token。"],
  "device.decode_command_ahead": ["decode 命令预备", "当前前向期间预备一条未提交的 decode 命令前缀；不支持的后端自动回退。"],
  "device.decode_command_early_submit": ["decode 提前提交", "后缀装配前先提交已确认的命令前缀，重叠 CPU 工作。"],
  "device.decode_command_resident": ["decode 驻留延伸", "把命令前探延伸到首个驻留专家提交；需同时开启命令预备与提前提交。"],
  "device.gpu_lock": ["GPU 启动锁", "同卡已有另一个 infr 持锁时拒绝启动，避免两个引擎互相挤显存。诊断性护栏，不是预约。"],
  "sampling.temp": ["温度", "0 = 贪心解码，结果确定；越大越随机。"],
  "sampling.top_k": ["Top-K", "每步只从概率最高的 K 个 token 里采样。"],
  "sampling.top_p": ["Top-P", "核采样：累计概率到 P 为止的候选集。"],
  "sampling.seed": ["随机种子", "固定种子可复现采样；留空各处用各自默认（时钟 / 42）。"],
  "sampling.max_new": ["最大生成长度", "单次请求最多生成的 token 数。"],
  "sampling.ignore_eos": ["忽略结束符", "遇到 EOS 也继续生成，直到长度上限。"],
  "sampling.no_think": ["关闭思考", "抑制模型的思考块输出。"],
  "sampling.reasoning_effort": ["思考力度", "模型原生的推理力度档位；留空用模板默认。"],
  "sampling.preserve_thinking": ["保留历史思考", "多轮中是否保留之前的思考内容；留空按模板策略。"],
  "kv.type_k": ["K 缓存精度", "K cache 数据类型（如 q8_0）。不满足后端 / 对齐门控时静默回退 f16。"],
  "kv.type_k_specified": ["K 精度显式指定标记", "内部标记：是否显式给过 K 精度。一般不要手改。"],
  "kv.type_v": ["V 缓存精度", "V cache 数据类型。"],
  "kv.type_v_specified": ["V 精度显式指定标记", "内部标记：是否显式给过 V 精度。一般不要手改。"],
  "kv.force_q8": ["强制 Q8 KV", "旧版别名：K / V 两侧都用 q8。"],
  "kv.slots": ["前缀缓存槽数", "VRAM 里驻留的会话前缀缓存槽位数。"],
  "kv.session_cache_dir": ["会话冷存目录", "空闲会话 KV 落盘目录。留空不落盘；设置后支持多会话驱逐与重启恢复。"],
  "kv.session_idle_secs": ["会话空闲秒数", "空闲会话多久后可被冷存。"],
  "kv.session_cache_max": ["会话冷存上限", "冷存目录最多保留的字节数，不接受百分比。"],
  "kv.session_cache_ttl_hours": ["冷存保留小时", "超过此时长的冷会话在维护时清除。"],
  "kv.ram_cache_max": ["内存前缀缓存上限", "主机内存 radix 前缀缓存的字节上限（VRAM 槽 + 内存快照池，不落盘）。设置后优先于会话冷存目录。"],
  "kv.ring": ["SWA 环形缓存", "滑窗注意力的环形 KV 缓存。"],
  "kv.dynamic": ["KV 动态增长", "支持的 Qwen KV 按 32K token 增量惰性提交；不支持分段寻址的保持平铺分配。"],
  "kv.paged": ["分页 KV 树", "在 VRAM 帧走廊上单独跑分页 radix 前缀树；冷链直接驱逐不下沉。paged_cache_max 仍可单独开树。"],
  "kv.paged_elastic": ["弹性帧走廊", "VRAM 帧走廊按需伸缩：空闲整段归还专家 arena，上限仍是固定走廊的最坏值。需分页树。"],
  "kv.paged_yield_under_pressure": ["专家压力让位", "专家缓存缺显存时把冷会话尾部下沉到内存层并归还走廊段；整段复用的会话不让。需弹性走廊 + 内存层。"],
  "kv.paged_page_tokens": ["页大小（token）", "每个物理 KV 页的 token 数，须为 2 的幂且整除各层 QSA 压缩比。32768 复刻旧 32K 增长步；小页（如 64 / 256）是帧共享与分层的前提。"],
  "kv.paged_cache_max": ["内存帧池上限", "分页 KV 的主机内存帧池字节上限：被驱逐的前缀留在内存而不是丢弃。0 / 留空关闭内存层；配合小页才有意义。"],
  "kv.disk_cache_max": ["磁盘帧层上限", "内存池耗尽后最冷全内存链下沉到稀疏文件的字节上限；命中经内存读回。需 paged_cache_max。"],
  "kv.disk_cache_path": ["磁盘帧层文件", "磁盘层 backing 文件路径；留空用系统临时目录。进程级记账，不跨重启恢复。"],
  "kv.inline_decode": ["KV 内联解码", "KV 内联解码路径（实验）。"],
  "kv.coopmat_bda": ["KV coopmat BDA", "KV 路径使用 coopmat + BDA 寻址（实验）。"],
  "kv.overflow": ["KV 溢出到内存", "KV 溢出到主机内存。注意：空串和 0 都是关。"],
  "kv.overflow_vram_mb": ["KV 溢出显存阈值", "KV 溢出使用的显存阈值（MiB）。"],
  "kv.overflow_reserve_mb": ["KV 溢出保留", "KV 溢出保留显存（MiB）；留空 = max(12% 显存, 2GiB)。"],
  "paging.cache": ["专家缓存预算", "分页专家缓存的显存预算。"],
  "paging.ring": ["上传 staging 环", "专家上传 staging 环大小；留空按预算比例策略。"],
  "paging.ring_slots": ["上传环分区数", "上传环里独立围栏的分区数。2 为历史双缓冲；更大可让 CPU staging 更超前。"],
  "paging.moe_layer_stream": ["MoE 整层流式", "显存装得下的完整专家层常驻，其余 prefill 按整层流式；decode 复用同一 arena 走 LRU。"],
  "paging.prefill_upload_async": ["prefill 异步上传", "GPU 算当前层时主机 worker 上传后续流式层。"],
  "paging.prefill_cross_chunk": ["跨 chunk 保温", "连续 prefill chunk 之间保持流式 MoE 管线热：末尾层顺带上传下一 chunk 的开头层。"],
  "paging.expert_prefetch": ["专家预取", "用当前层输入预测下一层路由，提前把有用专家块收进冷 LRU 边。实验性，默认关。"],
  "paging.moe_size_cache_bias": ["专家尺寸缓存偏置", "decode arena 在不同尺寸专家张量间的权重偏置，正值偏大张量。留空仅对均衡双尺寸布局自动 +2；显式 0 关闭自动。"],
  "paging.stats": ["分页统计", "打印分页器统计。"],
  "paging.trace": ["分页访问轨迹", "decode 分页访问 CSV 输出路径；内存缓冲、生成后落盘，不在关键路径上加 IO。"],
  "paging.dram": ["主机专家缓存（旧）", "旧版主机缓存覆盖项；新配置请用 device.ram_budget。仅在 ram_budget 未设时生效，0 关闭。"],
  "paging.dram_bypass": ["直读磁盘进显存", "分页块从磁盘直读 GPU、不经主机缓存——统一内存设备的天然形态，独显上也可显式启用。"],
  "paging.host_dma": ["主机内存 DMA", "用 VK_EXT_external_memory_host 把 DRAM arena 别名给 Vulkan 直读 / 拷贝；不支持的设备透明回退。"],
  "paging.dedicated_transfer": ["专用传输队列", "支持时用纯传输 Vulkan 队列做分页上传，与 host_dma 独立。"],
  "paging.layer_major": ["层主序 prefill", "三态：true 强制层主序（权重只扫一遍），false / 留空用 chunk 主序。诊断模式。"],
  "kernels.vulkan.gemv.no_rm": ["GEMV 强制 RM=1", "强制 GEMV 走 RM=1 路径。"],
  "kernels.vulkan.gemv.rm": ["GEMV RM 因子", "GEMV 的 RM 行因子。"],
  "kernels.vulkan.gemv.rm_maxout": ["GEMV RM 输出上限", "RM 路径适用的最大输出行数。"],
  "kernels.vulkan.gemv.rm_minout": ["GEMV RM 输出下限", "RM 路径适用的最小输出行数。"],
  "kernels.vulkan.gemv.no_sg": ["关闭 GEMV 子组", "禁用 GEMV 子组路径。"],
  "kernels.vulkan.gemv.no_id_sg": ["关闭 IQ 子组 GEMV", "禁用 IQ 量化的子组 id-GEMV。"],
  "kernels.vulkan.gemv.sg_minout": ["子组 GEMV 输出下限", "子组路径最小输出行数。"],
  "kernels.vulkan.gemv.sg_maxout": ["子组 GEMV 输出上限", "子组路径最大输出行数。"],
  "kernels.vulkan.gemv.sg_nr": ["子组 GEMV NR", "子组 GEMV 每工作组输出行数因子。"],
  "kernels.vulkan.gemv.id_grid_nr": ["IQ id-GEMV 行/组", "64 线程分页 IQ id-GEMV 每工作组处理的输出行数。"],
  "kernels.vulkan.gemv.id_grid_buffer_nr": ["IQ 码本 GEMV 行/组", "不可变码本 IQ2_S / IQ3_S / IQ3_XXS 每工作组输出行数；留空按实测设备策略，1/2/4/8 显式覆盖。"],
  "kernels.vulkan.gemv.id_grid_buffer": ["IQ 不可变码本", "IQ2_S / IQ3_S / IQ3_XXS 不可变码本路径；留空仅启用实测过的 Windows/RDNA3 decode 形状。"],
  "kernels.vulkan.gemv.id_sg_q51": ["Q5_1 子组 id-GEMV", "把 Q5_1 专家 down 纳入子组 id-GEMV 的 A/B 门；默认关，等模型级 A/B。"],
  "kernels.vulkan.gemv.q8_wave32": ["Q8 Wave32 GEMV", "位精确的 Wave32 大 K Q8_0 decode GEMV；留空仅实测形状，显式 true/false 覆盖。"],
  "kernels.vulkan.gemv.variant": ["GEMV 变体", "由其它开关推导的 GEMV 变体选择（reg 等），不建议手填。"],
  "kernels.vulkan.coopmat": ["协作矩阵", "Vulkan cooperative matrix 总开关（与 f16 相与）。"],
  "kernels.vulkan.coopmat_8x8": ["8x8 协作矩阵", "8x8x16 coopmat 形状（Intel Arc XMX 选用）。"],
  "kernels.vulkan.bf16_coopmat": ["BF16 协作矩阵", "BF16 coopmat 路径。"],
  "kernels.vulkan.f8_coopmat": ["F8 协作矩阵", "F8 coopmat 路径。"],
  "kernels.vulkan.f8_prepack": ["F8 预打包", "F8 权重预打包。"],
  "kernels.vulkan.i8_coopmat": ["I8 协作矩阵", "I8 coopmat 路径。"],
  "kernels.vulkan.i8_row_scale": ["I8 行缩放", "I8 行级缩放路径。"],
  "kernels.vulkan.f16": ["F16 计算", "f16 计算总开关（关闭会连带关 coopmat）。"],
  "kernels.vulkan.i8_dot": ["I8 点积", "int8 点积指令路径。"],
  "kernels.vulkan.gemm_warp": ["GEMM warp 瓦片", "warp 级 GEMM 瓦片路径。"],
  "kernels.vulkan.narrow_f32_gemv": ["窄投影 F32 GEMV", "实测 Q8_0/BF16 窄投影每 chunk 展开一次后走批量 F32 GEMV（合格 Navi31）。"],
  "kernels.vulkan.gemm_wide_tile": ["GEMM 宽瓦片", "GEMM 宽瓦片实验路径。"],
  "kernels.vulkan.gemm_direct_a": ["GEMM 直读 A", "把 store_f16 融合进 A_GLOBAL warp GEMM：直读 f32 激活、staging 时转 f16（选用）。"],
  "kernels.vulkan.small_bm": ["小 BM 瓦片", "小 BM 瓦片路径。"],
  "kernels.vulkan.bm16": ["BM16 瓦片", "BM16 瓦片路径。"],
  "kernels.vulkan.mmq": ["MMQ 内核", "MMQ 量化矩阵乘内核。"],
  "kernels.vulkan.qwen_hc_down_prefetch": ["HC down 预取内核", "Qwen3.8 Q8_0 单行形状用实测 PF4 HC-down 内核（Windows RDNA3）。"],
  "kernels.vulkan.mmq_fallback": ["MMQ 回退", "MMQ 不满足形状时的回退路径。"],
  "kernels.vulkan.mmv": ["MMV 内核", "MMV 向量乘内核层。"],
  "kernels.vulkan.mmv_decode": ["decode 用 MMV", "decode 也走 MMV（需 mmv 开）。"],
  "kernels.vulkan.mmv_m4": ["MMV M4", "MMV M4 打包路径。"],
  "kernels.vulkan.mmv_o4": ["MMV O4", "MMV O4 打包路径。"],
  "kernels.vulkan.mmv_mw": ["MMV MW", "MMV MW 路径，三态：留空按厂商默认，true/false 强制。"],
  "kernels.vulkan.mmv_mw_warps": ["MMV MW warp 数", "MMV MW 路径的 warp 数。"],
  "kernels.vulkan.mmv_fuse_quant": ["MMV 融合量化", "把每 Linear 的 quant_q8 融合进 decode GEMV（仅 Q4_K/Q6_K，选用）。"],
  "kernels.vulkan.mrow": ["MROW 内核", "多行（mrow）内核层。"],
  "kernels.vulkan.mrow16": ["MROW16", "mrow16 路径。"],
  "kernels.vulkan.f32_mrow": ["F32 MROW", "F32 多行内核。"],
  "kernels.vulkan.f32_v4": ["F32 V4", "F32 vec4 路径。"],
  "kernels.vulkan.moe_small_m": ["MoE 小 M 阈值", "MoE 小批行数阈值（0..=64，越界会被钳制）。"],
  "kernels.vulkan.moe_topk_sg": ["MoE 路由子组 top-k", "GPU MoE 路由 top-k 用 Wave32 两级归约；标量共享内存树留作 A/B 回退。"],
  "kernels.vulkan.qsa_score_decode8": ["QSA decode 八子组", "QSA 单行 decode 打分填满 8 个 wave32 子组。"],
  "kernels.vulkan.qsa_prepare_fused": ["QSA prepare 融合", "实验：拼接 indexer Q/K 投影并融合 raw-K 写缓存与 query norm/RoPE。未过模型 A/B，默认关。"],
  "kernels.vulkan.decode_down_overlap": ["down 提升重叠", "down 投影提升重叠；留空仅实测 Windows/RDNA3 深 2/3 行 decode。"],
  "kernels.vulkan.decode_sparse_lut": ["稀疏路由 LUT", "稀疏路由查找表；留空仅实测 Qwen3.8 标量 / 并行 decode。"],
  "kernels.vulkan.decode_early_submit": ["FFN 提前提交", "更早提交 FFN；留空仅实测无预取的 Qwen3.8 decode。"],
  "kernels.vulkan.decode_gpu_resident": ["GPU 驻留 decode", "分页 decode：全命中 MoE 层继续在 GPU 跑，只在首个 miss 处暂停；不支持自动回退。"],
  "kernels.vulkan.decode_gpu_resident_layers": ["驻留层数上限", "驻留投机与 miss 恢复的层数上限（仅普通 decode）。"],
  "kernels.vulkan.decode_gpu_resident_submit_layers": ["驻留滚动提交深度", "驻留滚动提交深度；0 用图内模型调优值。"],
  "kernels.vulkan.decode_gpu_resident_pure_mapped_layers": ["纯映射合并层数", "实验：只合并认证的 GPU 自有映射段；0 关，4 开。"],
  "kernels.vulkan.decode_gpu_resident_inflight": ["驻留在途深度", "滚动退休深度（1..2）；GPU 自有主机敏感段允许多一段投机。"],
  "kernels.vulkan.decode_gpu_resident_mapped_metadata": ["驻留元数据映射", "驻留谓词与 LUT 带放在一致性映射显存（设备支持时）。"],
  "kernels.vulkan.decode_gpu_resident_mapped_journal": ["驻留日志映射", "路由日志放一致性映射显存，带回读回退。"],
  "kernels.vulkan.decode_gpu_resident_fence_wait": ["驻留独立围栏", "驻留块等到自己的围栏而非整队（A/B）。"],
  "kernels.vulkan.decode_gpu_resident_fast_lut": ["驻留快速 LUT", "缓存不可变物理地址 + 密集槽保护。"],
  "kernels.vulkan.decode_gpu_resident_many_row": ["驻留多行融合", "普通 3..=8 行路由选择与 GPU 自有准入融合；false 保留分离录制供诊断。"],
  "kernels.vulkan.decode_gpu_resident_hit_first": ["命中优先续跑", "miss 通知开始主机恢复时，驻留命中继续跑。"],
  "kernels.vulkan.decode_gpu_resident_copy": ["GPU miss 拷贝", "从导入内存做 GPU miss 拷贝；留空仅普通 1..=8 行 decode，true 放开实验多行 / Verify。"],
  "kernels.vulkan.decode_gpu_resident_copy_groups": ["miss 拷贝工作组", "GPU 拷贝工作组数；0 选 8，显式值钳到 1..=8。仅 RX 7900 XTX 实测。"],
  "kernels.vulkan.decode_gpu_resident_copy_overlap": ["miss 拷贝重叠", "驻留命中投影与有界拷贝并发；需串行拷贝资格。"],
  "kernels.vulkan.decode_gpu_resident_role_pipeline": ["角色拷贝管线", "把自有 GPU 角色拷贝拆到 miss Gate/Up 两侧（默认关）。"],
  "kernels.vulkan.decode_gpu_resident_splitk_hit": ["split-K 命中试验", "独立 rows4 IQ2S/IQ4NL 的 H-only chunk32 FFN 试验。"],
  "kernels.vulkan.decode_gpu_resident_splitk_iq3_hit": ["split-K IQ3 扩展", "把 splitk-hit 扩到合格 rows4 IQ3S/IQ4NL；需上一项开。"],
  "kernels.vulkan.decode_gpu_resident_host_ready": ["主机就绪即提交", "认证纯 CPU 上传完成期间提交受保护的普通 decode 恢复。"],
  "kernels.vulkan.decode_gpu_resident_upload_worker": ["上传 worker 复用", "复用有界 CPU 上传助手，让旧 SKIP/ACK 观察与上传重叠。"],
  "kernels.vulkan.decode_gpu_resident_waiter": ["独立围栏等待线程", "专用主机线程等驻留围栏（驱动调度 A/B）。"],
  "kernels.vulkan.decode_gpu_resident_immutable_predicates": ["不可变谓词", "每检查层给不可变谓词字，避免谓词读写屏障。"],
  "kernels.vulkan.decode_gpu_resident_notice_thread": ["通知独立线程", "独立主机生产者线程收 GPU 通知并准备 miss 上传。"],
  "kernels.vulkan.decode_gpu_resident_mapped_serial_max_kib": ["映射小批串行拷贝", "通知生产者上串行拷贝小映射 miss 批；留空按上下文策略，0 关，KiB 显式覆盖（上限 8192）。"],
  "kernels.vulkan.decode_gpu_resident_early_recovery": ["提前 miss 恢复", "退休旧驻留块之前先排 miss 恢复（滚动上限 1）。"],
  "kernels.vulkan.decode_gpu_resident_full_graph": ["驻留全图模式", "普通 MoE decode 全图；留空对无主机互锁模型启用，FlashNext PLE 按实测上下文策略。"],
  "kernels.vulkan.decode_gpu_resident_indirect_predicates": ["间接派发谓词", "A/B：GPU 产出间接派发尺寸替代条件渲染命令。"],
  "kernels.vulkan.cpu_miss_early_submit": ["CPU miss 提前提交", "CPU miss 结果立即提交 / 合并，不等下一路由（A/B）。"],
  "kernels.vulkan.cpu_miss_input_tap": ["CPU miss 输入融合", "单行 F32 输入回读融合进路由，省一次拷贝。"],
  "kernels.vulkan.cpu_miss_direct_result": ["CPU miss 直写结果", "CPU 输出直写不相交映射 miss 槽，免 staging 拷贝。"],
  "kernels.vulkan.cpu_miss_host_result": ["CPU miss 主机结果", "实验：单 miss 合并直读小主机结果，免 scatter 拷贝。"],
  "kernels.vulkan.cpu_miss_threads": ["CPU miss 线程数", "实验性标量 decode 卸载（不含 PLE 层），需 AVX2+FMA；0 或不支持则留 GPU。"],
  "kernels.vulkan.cpu_miss_max": ["CPU miss 上限", "缺失集合分配数不超过此值才整体卸载（1..3，非法值关闭卸载）。"],
  "kernels.vulkan.cpu_miss_iq3": ["CPU miss 含 IQ3", "IQ3_S 默认仅 GPU；显式打开才纳入 CPU 对比。"],
  "kernels.vulkan.cpu_miss_spin": ["CPU miss 自旋", "私有 miss 池有界空转自旋次数。"],
  "kernels.vulkan.cpu_miss_poll": ["CPU miss 轮询", "decode 突发内协调者有界轮询；0 保持通道睡眠。"],
  "kernels.vulkan.cpu_miss_idle_park": ["CPU miss 空闲停泊", "发布计算 / 提升后停泊私有助手。"],
  "kernels.vulkan.cpu_miss_token_park": ["CPU miss 逐 token 停泊", "每完成一次前向就结束私有 CPU 突发。"],
  "kernels.vulkan.cpu_miss_push": ["CPU miss 推送提升", "提交结果后用私有 CPU 池提升 miss 权重，而非 DMA。"],
  "kernels.vulkan.cpu_miss_validate": ["CPU miss 校验", "诊断：保留 GPU miss 计算并与 CPU 原始输出对比。"],
  "kernels.vulkan.cpu_miss_split_acc": ["CPU miss 分裂累加", "实验：独立 FMA 链，改变归约顺序，不改激活精度。"],
  "kernels.vulkan.cpu_miss_grouped_dot": ["CPU miss 分组点积", "实验：按缩放分组 F32 点积，改变舍入。"],
  "kernels.vulkan.cpu_miss_flush_denormals": ["CPU miss 冲零", "CPU miss 任务内匹配 GPU 非规格化数冲零，逐线程恢复 MXCSR。"],
  "kernels.vulkan.qsa_topk_parallel": ["QSA 并行 top-k", "深单行 QSA decode 用多工作组精确 radix 选择。"],
  "kernels.vulkan.qsa_topk_scan": ["QSA 精确前缀计数", "精确 QSA 计数前缀；留空仅实测 Windows RDNA3 并行 top-512 路径。"],
  "kernels.vulkan.canvas_chunk_n": ["canvas 块行数", "canvas 路径每块行数（至少 1）。"],
  "kernels.vulkan.flash_splits": ["flash 拆分数", "flash attention 拆分 1..=8；留空自动。"],
  "kernels.vulkan.flash_bm32": ["flash BM32", "flash BM=32 形状（环境变量按字面 32 比较）。"],
  "kernels.vulkan.flash_min_rows": ["flash 最小行数", "走 flash 内核的最小行数。"],
  "kernels.vulkan.flash_stage": ["flash staging", "flash staging 路径。"],
  "kernels.vulkan.flash_dequant": ["flash 反量化", "flash 内反量化路径。"],
  "kernels.vulkan.flash_warp": ["flash warp", "flash warp 内核。"],
  "kernels.vulkan.amdvlk_hd256_flash": ["hd256 flash（amdvlk）", "AMD  proprietary 驱动 hd256 prefill 用融合 flash warp 内核；LLPC 编译全展开 IR 会崩，已改发 NO_UNROLL 孪生版。仍崩可关，回退 coopmat 三段式。"],
  "kernels.vulkan.nc_fa": ["NC flash attention", "非因果 flash attention 路径。"],
  "kernels.vulkan.qk_warp": ["QK warp", "QK warp 内核。"],
  "kernels.vulkan.pv_warp": ["PV warp", "PV warp 内核。"],
  "kernels.vulkan.pv_splits": ["PV 拆分数", "PV 拆分 1..=8；留空自动。"],
  "kernels.vulkan.no_attn_hd_spec": ["关闭注意力 HD 特化", "关闭按 head-dim 的注意力特化（存在即生效的负向开关）。"],
  "kernels.vulkan.attn_decode": ["decode 注意力特化", "decode 专用 split-K pass-1 特化；关闭则用通用 attn_partial，输出位相同（A/B）。"],
  "kernels.vulkan.q8_decode_d8": ["Q8 hd256 D8 内核", "Q8 hd256 decode 用 8 lane 聚簇内核；关闭回退全 wave QK 归约。"],
  "kernels.vulkan.q8_decode_ls128": ["Q8 decode LS128", "保留 64 线程 D8 内核作回退；默认 Q8 hd256 用 4 wave 暴露更多 key 并行。"],
  "kernels.vulkan.q8_decode_ls256": ["Q8 decode LS256", "Q8 hd256 优先 8 wave；关闭按 ls128 回退 LS128 → LS64。"],
  "kernels.vulkan.q8_decode_chunk1024": ["Q8 decode 1024 块", "允许 Q8 hd256 decode 用 1024-key 块；f16 / prefill / 其它 head-dim 仍 512。"],
  "kernels.vulkan.q8_decode_replay_chunk1024": ["Q8 录制 1024 块", "录制一次 decode 带从 512 扩到 1024 key；35B 端到端实测回退，默认关。"],
  "kernels.vulkan.q8_qk_f16": ["Q8 QK f16 积", "Q8 hd256 Q·K 用打包 fp16 积；softmax / PV 仍 fp32。"],
  "kernels.vulkan.q8_qk_i8": ["Q8 QK i8 点积", "实测设备把单 token query 按 Q8 块量化，与缓存 K 码做 int8 点积；未实测驱动留 fp16。"],
  "kernels.vulkan.q8_kv_write_fusion": ["Q8 KV 写融合", "录制 decode 中从 QkNormRope 的 LDS 直接量化旋转 K 行写入 Q8 缓存。"],
  "kernels.vulkan.q8_kv_write_segmented": ["Q8 KV 分段写", "有界静态分段 Q8 K norm/RoPE + 写。"],
  "kernels.vulkan.q8_pv_f16": ["Q8 PV f16 反量化", "Q8 V 反量化用打包 fp16 再转 fp32；加权 / 累加仍 fp32。"],
  "kernels.vulkan.q8_decode_gqa2": ["Q8 GQA2 复用", "hd256 decode 相邻 2 个 GQA 头复用一次 Q8 K/V 读。"],
  "kernels.vulkan.q8_decode_gqa4": ["Q8 GQA4 复用", "占用允许时 4 个相邻 GQA 头复用一次读。"],
  "kernels.vulkan.q8_decode_segmented": ["Q8 分段 KV", "实测 hd256 Q8 decode 内核复用到分段 KV 地址表。"],
  "kernels.vulkan.iq4_pair_lut": ["IQ4 对码本", "IQ4_NL 专家 MMQ 的对码本 decode（实测 Windows/RDNA3）；其它形状走标量查表。"],
  "kernels.vulkan.kquant_packed_decode": ["K 量化打包 decode", "实测 BM32 专家 MMQ 形状的 Q3_K/Q6_K 打包字节 decode。"],
  "kernels.vulkan.q8_prefill_bm128": ["prefill BM128 瓦片", "实测 RX 7900 XTX 的 128x128 单缓冲 prefill warp 瓦片；m≥1024 整 128 行 tile 且宽 k/n 才启用，回退形状保持 BM64。"],
  "kernels.vulkan.qsa_topk_ordered": ["QSA 时序 top-k", "实测设备上按时间序精确收集 QSA Top-K。"],
  "kernels.vulkan.flash_pv_reuse": ["flash PV 复用", "实测 hd256 prefill 寄存器 O 内核复用概率片段。"],
  "kernels.vulkan.q8_linear_pair": ["Q8 线性对", "独立原生 Q8 标量投影对；其它行数保持拆分 Linear。"],
  "kernels.vulkan.hc_up_mix": ["HC up 混合", "scratch 视图不相交时，精确单行 Q8 HC up 投影 + 有序混合。"],
  "kernels.vulkan.ple_gate_norm_lds": ["PLE gate+norm LDS", "实测 HC4/2560 PLE gate+norm LDS 对（合格 Windows/RDNA3 形状）；不支持保持两步。"],
  "kernels.vulkan.silu_quant_q8": ["SwiGLU 量化融合", "一次实测 dispatch 保持精确 F32 SwiGLU → 平面 Q8 边界。"],
  "kernels.vulkan.static_scratch_memo": ["scratch 分析缓存", "同一不可变编译计划内缓存纯图 scratch 分析。"],
  "kernels.vulkan.static_decode_tail_overlap": ["decode 尾部重叠", "最后提交执行时退休更早的分页 decode 提交；结果 / 回调 / 换阶段前仍完整排空。"],
  "kernels.vulkan.qsa_prefill_gqa": ["QSA prefill GQA 复用", "QSA prefill Q8 K/V 复用模式：2 寄存器两头；12 走 Qwen3.8 LDS 全 GQA。"],
  "kernels.vulkan.qsa_prefill_tile": ["QSA prefill 瓦片", "LDS GQA12 路径的 K/V 瓦片：16 逐 key 在线 softmax；32 FlashAttention 式瓦片合并。"],
  "kernels.vulkan.qsa_prefill_coopmat": ["QSA prefill QK coopmat", "Flash32 QSA prefill 的 QK 用 F16 协作矩阵（设备暴露可信 16x16x16 时）。"],
  "kernels.vulkan.qsa_prefill_coopmat_pv": ["QSA prefill PV coopmat", "同路径 PV 也用协作矩阵；独立开关留 QK-only 作 A/B 回退。"],
  "kernels.vulkan.qsa_verify_split": ["QSA verify 拆分", "小目标 Verify QSA 队列用逐行因果 split-K；大 prefill 批保持瓦片内核。"],
  "kernels.vulkan.verify_shared_iq2_nr8": ["Verify IQ2 共享 NR8", "精确形状因果 Verify IQ2_S 共享槽融合 + 分段 NR8；默认关待端到端验证。"],
  "kernels.vulkan.verify_iq4nl_down_nr4": ["Verify IQ4NL down NR4", "因果 Verify IQ4_NL down 一行输入 / 块缩放复用 4 相邻输出行，实测每 dispatch 快 15%，位相同；默认关待验证。"],
  "kernels.vulkan.verify_pair": ["Verify 行对复用", "相邻因果 Verify 行复用原生专家权重解码；不合格形状保持基线。"],
  "kernels.vulkan.qsa_score_tile": ["QSA 打分瓦片", "Qwen3.8 prefill indexer 打分 8-query F32 瓦片宽度，支持 32 / 64。"],
  "kernels.vulkan.q8_decode_combine_sg": ["Q8 合并子组归约", "Q8 hd256 decode 合并 pass 用 Wave32 max/sum 归约。"],
  "kernels.vulkan.mrows_attn": ["多行注意力", "三态：false 无条件关；true 跳过行数 / kv_len 启发；留空由启发决定。"],
  "kernels.vulkan.mla_sg": ["MLA 子组单趟", "MLA 子组单趟内核；关闭回退标量两趟。注意求和顺序不同，非位相同。"],
  "kernels.vulkan.dn_chunk_scan": ["DeltaNet 块扫描", "名字是正向但设置即禁用块扫描（历史极性，冻结不改）。"],
  "kernels.vulkan.dn_chunk": ["DeltaNet 分块", "DeltaNet 分块路径。"],
  "kernels.vulkan.dn_split": ["DeltaNet 拆分", "DeltaNet 拆分路径。"],
  "kernels.vulkan.delta_strided": ["DeltaNet 跨步直读", "单 token DeltaNet decode 直读交错 q/k/v；已过稠密 / 跨步输出与状态 parity。"],
  "kernels.vulkan.delta_fused_norm": ["DeltaNet 融合 norm", "实测设备融合单 token 交错 DeltaNet 与原位门控 norm；多 token / 独立状态行保持拆分。"],
  "kernels.vulkan.push_desc": ["推送描述符", "push descriptor 路径。"],
  "kernels.vulkan.pipeline_cache_disk": ["管线磁盘缓存", "编译管线缓存到磁盘，下次启动免重编。"],
  "kernels.vulkan.no_vram_guard": ["关闭显存护栏", "关闭显存护栏（负向开关）。"],
  "kernels.vulkan.no_moe_sm_pool": ["关闭 MoE 共享内存池", "关闭 MoE 共享内存池（负向开关）。"],
  "kernels.vulkan.no_replay": ["关闭录制回放", "关闭 seam 录制回放（负向开关）。"],
  "kernels.vulkan.gpu_pos": ["GPU 位置编码", "位置编码在 GPU 计算。"],
  "kernels.vulkan.fuse_add": ["加法融合", "共享融合 pass 的 linear_add。"],
  "kernels.vulkan.bda_chunk_elems": ["BDA 块元素上限", "每个 BDA 寻址单元的元素上限；留空用内置最大值，小于 2 视为未设。"],
  "kernels.vulkan.bda_chunk_bytes": ["BDA 块字节上限", "每个 BDA 寻址单元的字节上限。"],
  "kernels.metal.f16_native": ["Metal 原生 F16", "Metal 原生 f16 路径。"],
  "kernels.metal.f32_native": ["Metal 原生 F32", "Metal 原生 f32 路径。"],
  "kernels.metal.bf16_native": ["Metal 原生 BF16", "Metal 原生 bf16 路径。"],
  "kernels.metal.f16_cmm": ["Metal F16 CMM", "Metal f16 协作矩阵乘。"],
  "kernels.metal.bf16_cmm": ["Metal BF16 CMM", "Metal bf16 协作矩阵乘。"],
  "kernels.metal.f32_cmm": ["Metal F32 CMM", "Metal f32 协作矩阵乘。"],
  "kernels.metal.f16_rt": ["Metal F16 RT", "Metal f16 RT 内核。"],
  "kernels.metal.bf16_rt": ["Metal BF16 RT", "Metal bf16 RT 内核。"],
  "kernels.metal.f32_rt": ["Metal F32 RT", "Metal f32 RT 内核。"],
  "kernels.metal.kquant_native": ["Metal 原生 K 量化", "Metal 原生 K 量化内核。"],
  "kernels.metal.q5k_rt": ["Metal Q5K RT", "Metal Q5_K RT 内核。"],
  "kernels.metal.rmsnorm_vec4": ["Metal RMSNorm vec4", "Metal RMSNorm vec4 路径。"],
  "kernels.metal.conv1d_par": ["Metal conv1d 并行", "Metal conv1d 并行路径。"],
  "kernels.metal.dn_gate_prep": ["Metal DeltaNet gate 预备", "Metal DeltaNet gate 预备路径。"],
  "kernels.metal.dn_norm_prep": ["Metal DeltaNet norm 预备", "Metal DeltaNet norm 预备路径。"],
  "kernels.metal.lmhead_mrv_uncapped": ["Metal lmhead 解除上限", "解除 lmhead mrv out_f 上限（存在即生效）。"],
  "kernels.metal.deltanet": ["Metal DeltaNet", "Metal DeltaNet 总开关。"],
  "kernels.metal.moe": ["Metal MoE", "Metal MoE 总开关。"],
  "kernels.metal.pipeline_cache": ["Metal 管线缓存", "编译管线持久化为 MTLBinaryArchive；只缓存后端，前端编译仍每次跑。无环境变量，用 --set 关。"],
  "kernels.cpu.spin": ["CPU 自旋上限", "自旋池空闲自旋上限。"],
  "kernels.cpu.spinpool": ["CPU 自旋池", "关闭则任务走 rayon。"],
  "kernels.cpu.repack_mb": ["CPU 重打包缓存", "重打包缓存字节预算（MiB）。"],
  "kernels.cpu.reference": ["CPU 参考内核", "位参考内核路径，无环境变量，由调用方选择。"],
  "kernels.qkv_fuse": ["QKV 融合", "图形状级 QKV 融合（infr-llama，非 Vulkan 专属）。"],
  "kernels.gated_rmsnorm": ["门控 RMSNorm 融合", "门控 RMSNorm 融合（与设备能力相与）。"],
  "kernels.delta_ab_fuse": ["Delta AB 融合", "Delta AB 融合（与 linear_pair_f32 能力相与）。"],
  "kernels.qwen_hc_down_inject": ["HC down 注入", "Qwen HC down 注入（Vulkan 专属能力）。"],
  "kernels.ple_single_parallel": ["PLE 单行并行", "单行也走持久 gather 池。"],
  "spec.mtp": ["MTP 投机解码", "开启 Qwen3.8 MTP 草稿头投机解码。"],
  "spec.lookup": ["prompt lookup 投机", "精确 prompt 后缀查表提案，无需草稿权重；超出 Verify 能力的请求走普通 decode。"],
  "spec.lookup_max_draft": ["lookup 最大草稿", "最大提案后缀长度（1..=3，不含已选待定 token）。"],
  "spec.lookup_min_match": ["lookup 最小匹配", "触发历史续写提案的最短精确后缀匹配。"],
  "spec.lookup_adaptive_width": ["lookup 自适应宽度", "按近期接受率收窄 lookup 草稿；false 保持配置上限。"],
  "spec.mtp_ckpt": ["MTP 检查点", "MTP 检查点路径。"],
  "spec.mtp_reprime": ["MTP 重新引导", "MTP reprime（与 mtp_ckpt 相与）。"],
  "spec.mtp_draft_chain": ["MTP 草稿链", "MTP 草稿链式推进。"],
  "spec.mtp_tail_kv_only": ["MTP 尾步仅 KV", "最后草稿步只算所需 K/V 写；false 保留全步参考路径供对比。"],
  "spec.mtp_vocab": ["MTP 草稿词表", "可选排序 u32 token 表，草稿用紧凑输出投影；目标模型始终全词表验证。"],
  "spec.mtp_ple_overlap": ["MTP PLE 重叠", "PLE 行 gather 时跑 VERIFY 层 0。"],
  "spec.draft": ["草稿模型路径", "独立草稿模型 GGUF 路径。"],
  "spec.k": ["草稿长度上限", "每轮草稿 token 数上限。"],
  "spec.mtp_context": ["MTP 头上下文", "MTP 草稿头注意力视野上限；目标模型保持全上下文。"],
  "spec.mtp_max_context": ["MTP 自动启用上限", "目标上下文超过此值自动关 MTP（验证成本超过省下的 decode）；0 关闭该自动 cutoff。"],
  "spec.mtp_min_alpha": ["MTP 接受率熔断", "滚动接受率低于此值，下一请求起 MTP 让位普通 decode；0 关熔断。"],
  "spec.mtp_adaptive_width": ["MTP 自适应宽度", "按实测每提交行成本选 VERIFY 宽度，周期性重探。"],
  "spec.mtp_slot_min_accept": ["MTP 槽接受下限", "自适应 VERIFY 裁剪的草稿前缀概率下限（非目标接受率）。"],
  "spec.mtp_concurrent": ["MTP 并发探测", "两条活跃 Qwen3.8 车道探测批式 MTP VERIFY；默认关（普通批式更快）。"],
  "spec.debug": ["投机调试日志", "投机解码调试输出。"],
  "spec.decode_chain": ["decode 链长", "每次提交的 decode 迭代数。"],
  "spec.gpu_draft_prob": ["GPU 草稿概率", "草稿概率在 GPU 算。"],
  "spec.gpu_mtp_accept": ["GPU MTP 接受", "MTP 接受判定在 GPU。"],
  "spec.gpu_argmax": ["GPU argmax", "argmax 在 GPU。"],
  "spec.gpu_sample": ["GPU 采样", "采样在 GPU。"],
  "spec.gpu_embed": ["GPU 嵌入", "token 嵌入在 GPU。"],
  "uncensor.vector": ["去审查向量", "llama.cpp 兼容 controlvector GGUF，每层一个 f32 方向。文件不可用会拒绝启动而不是静默跳过。"],
  "uncensor.first_layer": ["投影起始层", "含端起投影层；更低层在理解题意，投影会损害理解。"],
  "uncensor.last_layer": ["投影结束层", "含端止投影层（钳到模型深度）；更高层已定型，太晚移除方向会让模型啰嗦。"],
  "multi.pipeline": ["管线并行设备", "层切分设备列表，至少 2 卡。"],
  "multi.tensor_parallel": ["张量并行设备", "张量并行设备列表，至少 2 卡。"],
  "multi.expert_parallel": ["专家并行设备", "专家并行设备列表，至少 1 卡。"],
  "multi.pipeline_p2p": ["管线 P2P", "管线并行用 P2P 传输（关闭走主机）。"],
  "multi.tp_p2p": ["张量 P2P", "张量并行用 P2P 传输。"],
  "multi.ep_p2p": ["专家 P2P", "专家并行用 P2P 传输。"],
  "prof.ops": ["逐算子剖析", "全后端逐算子计时（vulkan / metal / cpu 的总开关）。"],
  "prof.op_shapes": ["按形状细分", "逐算子标签按形状细分（Vulkan 专属，与 ops 相与）。"],
  "prof.stages": ["阶段计时", "主机侧阶段计时：prompt/decode 吞吐、decode 装配 / 执行、prefill 构建 / 编译 / 执行、MTP verify 等。"],
  "prof.vram": ["显存占用日志", "权重加载后记录实时显存占用。"],
  "prof.diffusion_trace": ["扩散逐步轨迹", "扩散模型每步调度 / 熵轨迹（采样轨迹，非计时）。"],
  "prof.pager_profile": ["分页器计时", "进程退出时汇总分页 / 缓存 / 上传 / 提交计时；独立于 ops（时间戳剖析会改变分页环行为）。"],
  "prof.timeline_path": ["时间线输出", "有界异步 decode 轨迹输出路径；独立于阻塞式逐算子剖析。"],
  "prof.timeline_cpu_only": ["时间线仅 CPU", "只抓主机 span，不用 Vulkan 时间戳。"],
  "prof.timeline_gpu_only": ["时间线仅 GPU", "只抓 GPU 时间戳与路由元数据。"],
  "prof.timeline_raw_kernels": ["时间线原始内核名", "保留每个内核名而非类别标签。"],
  "prof.timeline_raw_unknown": ["时间线未知内核名", "保留未知类别内核的名字。"],
  "prof.timeline_device_select": ["时间线设备过滤", "逗号分隔设备标签子串，仅诊断；留空抓全部。"],
  "prof.timeline_device_layer": ["时间线层过滤", "仅抓指定 GPU 层（诊断）。"],
  "prof.timeline_host_select": ["时间线主机过滤", "逗号分隔主机 span 子串；留空抓全部。"],
  "prof.cpu_miss_fixture_dir": ["CPU miss 夹具目录", "一次性 CPU-miss 权重 / 输入夹具，本地内核诊断用。"],
  "prof.timeline_skip_steps": ["时间线跳过步数", "首个捕获窗口前丢弃的 decode 步（避开时钟爬坡 / 冷驻留）。"],
  "prof.timeline_steps": ["时间线窗口步数", "每窗口连续捕获的 decode 步数，不得超过 stride。"],
  "prof.timeline_stride": ["时间线采样周期", "相邻窗口起始间隔的 decode 步数。"],
  "prof.timeline_windows": ["时间线窗口数", "捕获多少窗口后停止；0 关闭轨迹。"],
  "prof.timeline_lanes": ["时间线车道数", "0 抓任意并发；否则恰好抓这么多 decode 车道。"],
  "prof.out": ["剖析报告输出", "退出报告额外写 JSON 到此路径（stderr 仍有）。"],
  "prof.metal_device_time": ["Metal 设备计时", "Metal 逐算子设备时间获取方式。"],
  "prof.metal_debug": ["Metal 剖析调试", "Metal 剖析器额外调试输出。"],
  "serve.api_key": ["引擎 API Key", "引擎 bearer 令牌；空串 = 不鉴权。控制台用它向上游转发。"],
  "serve.max_tokens_cap": ["单请求 token 上限", "单请求最大 token 数硬上限，须为正整数。"],
  "serve.request_timeout_secs": ["请求超时秒数", "单请求墙钟 deadline；0 无超时（默认）。超时会腰斩正常长回答，了解模型速度后再设。"],
  "serve.stats_interval_secs": ["吞吐日志间隔", "serve 吞吐日志行间隔秒数；0 关闭。仅活动期输出，空闲不刷屏。"],
  "serve.shutdown_file": ["停机触发文件", "此文件出现即走优雅停机（Windows 无 POSIX 信号的监管方式）。控制台自动设置，勿手改。"],
  "serve.embedding_runner": ["嵌入 runner", "可选 llama-server 可执行文件，供托管嵌入 worker 使用；留空自动发现。"],
  "serve.response_store_entries": ["Responses 状态条数", "/v1/responses 服务端状态 LRU 上限（previous_response_id / conversation 链）。0 保持无状态（这些字段 400）。开启后链式请求重放历史，前缀自然命中分页树。"],
  "serve.response_store_path": ["Responses 状态落盘", "状态镜像文件，重启后链不断。留空仅内存（隐私默认）；文件明文存全部 prompt/response。entries=0 时忽略。"],
  "vision.image_url_allow_private": ["允许内网图片 URL", "允许 image_url 指向非公网主机（loopback / RFC1918 / 链路本地 / CGNAT）。默认关：无限制抓取等于把引擎当内网端口扫描器。本地 NAS / 摄像头等合法目标再显式开。"],
  "hub.endpoint": ["HF 端点", "HuggingFace 兼容 API / 下载源；镜像（如 hf-mirror.com）URL 布局相同。"],
  "hub.pull_jobs": ["下载并发连接", "单模型下载并发连接数：分片按文件、单文件按字节范围并行。0/1 严格单连接（代理 / 计费链路用）。"],
  "debug.bda_chunk": ["调试 BDA 分块", "BDA 分块调试输出。"],
  "debug.coopmat": ["调试协作矩阵", "coopmat 调试输出。"],
  "debug.wide_dispatch": ["调试宽派发", "宽派发调试输出。"],
  "debug.chat": ["调试聊天模板", "聊天模板调试输出。"],
  "debug.state_trace": ["状态选择轨迹", "每请求一行紧凑状态选择日志，诊断 recurrent/QSA 续接损坏，不逐 token。"],
  "debug.moe_counts": ["MoE 计数调试", "MoE 专家计数调试。"],
  "debug.moe_counts_dump": ["MoE 计数转储", "MoE 专家计数转储。"],
  "debug.qsa_index_dump": ["QSA 索引转储", "每个 frontier decode 步向 qsa-index-dump.bin 追加所选 QSA 块索引；单流专用。"],
  "debug.poison_uninit": ["未初始化投毒", "未初始化内存投毒（诊断）。"],
  "debug.paged_frontier_cut": ["frontier 强制切块", "强制 paged frontier 切 prefill chunk 而非块内导出 recurrent 状态；导出路径的 A/B 对照。"],
  "debug.no_barrier": ["关闭屏障", "关闭屏障（诊断）。"],
  "debug.full_barrier": ["全屏障", "全量屏障（诊断）。"],
  "cli:model": ["聊天模型", ""],
  "cli:config": ["TOML 配置文件", ""],
  "cli:addr": ["引擎监听地址", ""],
  "cli:parallel": ["并发槽数", ""],
  "cli:mmproj": ["视觉投影模型", ""],
  "cli:embedding-model": ["嵌入模型", ""],
  "cli:embedding-idle-timeout": ["嵌入空闲卸载", ""],
  "cli:embedding-runner": ["嵌入 runner", ""],
  "cli:transcription-model": ["转写模型", ""],
  "cli:transcription-dev": ["转写后端", ""],
  "cli:transcription-threads": ["转写线程数", ""],
  "cli:transcription-idle-timeout": ["转写空闲卸载", ""],
  "cli:transcription-upload-dir": ["转写上传目录", ""],
  "cli:speech-model": ["语音合成模型", ""],
  "cli:speech-lexicon": ["语音资产目录", ""],
  "cli:speech-threads": ["语音线程数", ""],
  "cli:speech-idle-timeout": ["语音空闲卸载", ""],
  "cli:speech-max-output-seconds": ["语音长度上限", ""],
};

function tyLabel(ty) {
  if (ty === "bool") return "开关";
  if (ty === "Option<bool>") return "开关 · 三态";
  if (ty.includes("SizeSpec")) return "大小";
  if (ty.includes("PathBuf")) return "路径";
  if (ty.includes("Vec<usize>")) return "设备列表";
  if (ty.includes("String")) return "文本";
  if (/^(Option<)?(u32|u64|usize|f32)/.test(ty)) return "数值";
  return "枚举";
}

function fieldSearchText(f) {
  const zh = KNOB_ZH[f.path];
  const name = zh ? zh[0] : f.path;
  const desc = (zh && zh[1]) || f.doc || "";
  return `${f.path} ${name} ${desc}`.toLowerCase();
}

function fieldNode(f) {
  const cur = FORM[f.path] !== undefined ? FORM[f.path] : f.default;
  const dirty = FORM[f.path] !== undefined && FORM[f.path] !== f.default;
  const zh = KNOB_ZH[f.path];
  const name = zh ? zh[0] : f.path;
  const desc = (zh && zh[1]) || f.doc || "";
  const wrap = el("div", {
    class: `field${dirty ? " dirty" : ""}`,
    "data-path": f.path,
    "data-search": fieldSearchText(f),
  });
  const id = `knob-${f.path.replace(/[.:]/g, "-")}`;
  const pathLine = el("div", { class: "knob-path" },
    el("code", { text: f.path }), el("span", { text: tyLabel(f.ty) }));
  let input;
  if (f.ty === "bool") {
    input = el("input", { type: "checkbox", id });
    input.checked = cur === "true";
    wrap.append(el("label", { class: "check", for: id }, input, el("span", { text: name })), pathLine);
  } else {
    // `none` is how the engine spells an unset Option; an empty box says the same thing without
    // looking like a value the operator typed.
    input = el("input", { type: "text", id, value: cur === "none" ? "" : cur, placeholder: f.default === "none" ? "未设置" : f.default });
    wrap.append(el("label", { for: id, text: name }), pathLine, input);
  }
  input.addEventListener("change", () => {
    const v = f.ty === "bool" ? String(input.checked) : input.value.trim();
    if (v === f.default || (v === "" && f.default === "none")) delete FORM[f.path];
    else FORM[f.path] = v;
    wrap.classList.toggle("dirty", FORM[f.path] !== undefined);
    renderCmdline();
  });
  if (desc) wrap.append(el("div", { class: "doc", text: desc }));
  return wrap;
}

// 把字段按用途分组为分类数据（二级菜单的一级项）；未列入的归「其它」。
function categoriesFrom(fields, groups) {
  const cats = groups
    .map(([title, paths]) => ({ key: title, title, fields: pickFields(fields, paths) }))
    .filter((c) => c.fields.length);
  const rest = leftoverFields(fields, groups);
  if (rest.length) cats.push({ key: "其它", title: "其它", fields: rest });
  return cats;
}
// 高级选项按 advGroup 的家族/配置节聚合，顺序按使用相关度。
function advCategories(fields) {
  const byGroup = new Map();
  for (const f of fields) {
    const [key, title, order] = advGroup(f.path);
    if (!byGroup.has(key)) byGroup.set(key, { key, title, order, fields: [] });
    byGroup.get(key).fields.push(f);
  }
  return Array.from(byGroup.values()).sort((a, b) => a.order - b.order);
}

// 常用区按用途分组（组内顺序即展示顺序）；未列入的自动归入「其它」。
const BASIC_GROUPS = [
  ["模型与文件", ["cli:model", "cli:config", "cli:mmproj"]],
  ["服务与并发", ["cli:addr", "cli:parallel", "serve.api_key"]],
  ["显存与上下文", ["device.dev", "device.ctx", "device.ubatch", "device.threads", "device.vram_budget", "device.vram_reserve", "device.ram_budget", "device.submit_dispatches"]],
  ["KV 缓存", ["kv.paged", "kv.paged_page_tokens", "kv.paged_cache_max", "kv.paged_elastic", "kv.paged_yield_under_pressure", "kv.ram_cache_max", "kv.type_k", "kv.type_v", "kv.overflow", "kv.overflow_vram_mb", "kv.overflow_reserve_mb", "kv.session_cache_dir", "kv.session_idle_secs", "kv.session_cache_max", "kv.session_cache_ttl_hours"]],
  ["专家分页", ["paging.cache", "paging.host_dma", "paging.ring", "paging.ring_slots", "paging.stats"]],
  ["采样", ["sampling.temp", "sampling.top_k", "sampling.top_p", "sampling.seed", "sampling.max_new", "sampling.no_think", "sampling.reasoning_effort"]],
  ["模型功能", ["spec.mtp", "vision.image_url_allow_private", "uncensor.vector"]],
  ["CPU 卸载（实验）", ["kernels.vulkan.cpu_miss_threads", "kernels.vulkan.cpu_miss_max", "kernels.vulkan.cpu_miss_push", "kernels.vulkan.cpu_miss_host_result", "kernels.vulkan.cpu_miss_token_park", "kernels.vulkan.cpu_miss_spin"]],
  ["诊断", ["prof.stages", "prof.vram", "prof.pager_profile"]],
  ["嵌入与语音", ["cli:embedding-model", "cli:embedding-idle-timeout", "cli:embedding-runner", "cli:transcription-model", "cli:transcription-dev", "cli:transcription-threads", "cli:transcription-idle-timeout", "cli:transcription-upload-dir", "cli:speech-model", "cli:speech-lexicon", "cli:speech-threads", "cli:speech-idle-timeout", "cli:speech-max-output-seconds"]],
];
const MANUAL_GROUPS = [
  ["显存与内存", ["device.vram_budget", "device.vram_reserve", "device.ram_budget", "device.mem_guard_headroom", "device.mem_guard_ram_headroom", "device.mem_guard_poll_ms"]],
  ["批与执行", ["device.ubatch", "device.ubatch_parallel", "device.submit_dispatches", "device.subgroup_pref", "device.threads"]],
  ["专家分页", ["paging.cache", "paging.host_dma", "paging.ring", "paging.ring_slots", "paging.stats"]],
  ["KV", ["kv.paged_page_tokens", "kv.paged_cache_max", "kv.ram_cache_max", "kv.overflow", "kv.overflow_vram_mb", "kv.overflow_reserve_mb"]],
  ["诊断", ["prof.stages", "prof.vram", "prof.pager_profile"]],
];

function pickFields(fields, paths) {
  const byPath = new Map(fields.map((f) => [f.path, f]));
  return paths.map((p) => byPath.get(p)).filter(Boolean);
}
function leftoverFields(fields, groups) {
  const named = new Set(groups.flatMap(([, paths]) => paths));
  return fields.filter((f) => !named.has(f.path));
}
function selectCategory(id) {
  SEL_CAT = id;
  $$("#settings-nav .settings-cat").forEach((b) => b.classList.toggle("selected", b.dataset.cat === id));
  renderDetail();
}

function renderDetail() {
  const detail = $("#settings-detail");
  const cat = SETTINGS_CATS.find((c) => c.id === SEL_CAT);
  if (!cat) {
    detail.replaceChildren(el("div", { class: "settings-empty", text: KNOB_QUERY ? "没有匹配的配置项" : "从左侧选择分类查看参数" }));
    return;
  }
  detail.replaceChildren(
    el("div", { class: "settings-detail-head" },
      el("h3", { text: cat.title }),
      el("span", { class: "frame-seq", text: `${cat.fields.length} 项 · 每项下方为实际配置名与中文说明` })),
    el("div", { class: "knob-grid" }, cat.fields.map(fieldNode)));
}

// 高级选项：内核按家族细分，其余按配置节，顺序按使用相关度而非字母。
const ADV_SECTION_ORDER = { device: 10, kv: 11, paging: 12, serve: 13, sampling: 14, spec: 15, uncensor: 16, multi: 17, prof: 60, hub: 61, debug: 62 };
function advGroup(path) {
  if (path.startsWith("kernels.vulkan.gemv.")) return ["vk-gemv", "内核 · Vulkan GEMV", 40];
  if (path.startsWith("kernels.vulkan.decode_gpu_resident")) return ["vk-resident", "内核 · GPU 驻留 decode", 41];
  if (path.startsWith("kernels.vulkan.cpu_miss")) return ["vk-cpumiss", "内核 · CPU miss 卸载", 42];
  if (/^kernels\.vulkan\.q8_(decode|qk|kv|pv)/.test(path)) return ["vk-q8", "内核 · Q8 注意力", 43];
  if (path.startsWith("kernels.vulkan.qsa_")) return ["vk-qsa", "内核 · QSA", 44];
  if (path.startsWith("kernels.vulkan.verify_")) return ["vk-verify", "内核 · Verify", 45];
  if (/^kernels\.vulkan\.(flash_|attn_|nc_fa|qk_warp|pv_|no_attn|amdvlk|mrows_attn|mla_sg)/.test(path)) return ["vk-attn", "内核 · Flash / 注意力", 46];
  if (/^kernels\.vulkan\.(dn_|delta_)/.test(path)) return ["vk-delta", "内核 · DeltaNet", 47];
  if (/^kernels\.vulkan\.moe_/.test(path)) return ["vk-moe", "内核 · MoE 路由", 48];
  if (/^kernels\.vulkan\.(coopmat|bf16_coopmat|f8_|i8_|f16$|gemm_|narrow_f32|small_bm|bm16|mmq|mmv|mrow|f32_|qwen_hc_down_prefetch|iq4_pair_lut|kquant_packed_decode|q8_prefill_bm128|q8_linear_pair|silu_quant_q8|hc_up_mix|ple_gate_norm_lds)/.test(path)) return ["vk-gemm", "内核 · GEMM / 量化 / 融合", 49];
  if (path.startsWith("kernels.vulkan.")) return ["vk-other", "内核 · Vulkan 其它", 50];
  if (path.startsWith("kernels.metal.")) return ["metal", "内核 · Metal（macOS）", 51];
  if (path.startsWith("kernels.cpu.")) return ["cpu", "内核 · CPU", 52];
  if (path.startsWith("kernels.")) return ["graph", "内核 · 图融合", 53];
  const sec = sectionOf(path);
  return [sec, SECTION_LABEL[sec] || sec, ADV_SECTION_ORDER[sec] ?? 55];
}

function renderLaunch() {
  const scriptSet = new Set(SCHEMA.script_knobs || []);
  const manualSet = new Set(SCHEMA.manual_knobs || []);
  const all = SCHEMA.schema || [];
  // 全手动时显存/诊断项移入「全手动」分类块，不在常用里重复；两档预设下留在常用，与向导一致。
  const manual = PRESET === "manual";
  // 预设按钮即 auto_profile 控件，不再单列文本框，避免两者对命令行的说法不一致。
  const isProfile = (f) => f.path === "device.auto_profile";
  const basic = all.filter((f) => !isProfile(f) && scriptSet.has(f.path) && !(manual && manualSet.has(f.path)));
  const rest = all.filter((f) => !isProfile(f) && !scriptSet.has(f.path) && !manualSet.has(f.path));
  const manualFields = all.filter((f) => manualSet.has(f.path));

  const q = KNOB_QUERY;
  const build = (tier, label, cats) => ({
    label,
    cats: cats
      .map((c) => ({ id: `${tier}:${c.key}`, title: c.title, fields: c.fields.filter((f) => !q || fieldSearchText(f).includes(q)) }))
      .filter((c) => c.fields.length),
  });
  const blocks = [
    build("basic", "常用参数", categoriesFrom(basic, BASIC_GROUPS)),
    ...(manual ? [build("manual", "全手动", categoriesFrom(manualFields, MANUAL_GROUPS))] : []),
    build("adv", "高级选项", advCategories(rest)),
  ];
  SETTINGS_CATS = blocks.flatMap((b) => b.cats);
  if (!SETTINGS_CATS.some((c) => c.id === SEL_CAT)) SEL_CAT = SETTINGS_CATS.length ? SETTINGS_CATS[0].id : null;

  const navNodes = [];
  for (const b of blocks) {
    if (!b.cats.length) continue;
    navNodes.push(el("div", { class: "settings-nav-block", text: b.label }));
    for (const c of b.cats) {
      const dirty = c.fields.filter((f) => FORM[f.path] !== undefined).length;
      const btn = el("button", { type: "button", class: `settings-cat${c.id === SEL_CAT ? " selected" : ""}`, "data-cat": c.id },
        el("span", { class: "settings-cat-title", text: c.title }),
        dirty ? el("span", { class: "settings-cat-dirty", title: `${dirty} 项已改`, text: "●" }) : null,
        el("span", { class: "settings-cat-count", text: String(c.fields.length) }));
      btn.addEventListener("click", () => selectCategory(c.id));
      navNodes.push(btn);
    }
  }
  $("#settings-nav").replaceChildren(...navNodes);
  renderDetail();

  $$("#presets .preset").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.preset === PRESET)));
  renderCmdline();
}

$("#knob-search").addEventListener("input", (e) => {
  KNOB_QUERY = e.target.value.trim().toLowerCase();
  if (SCHEMA) renderLaunch();
});

$$("#presets .preset").forEach((b) => b.addEventListener("click", () => {
  PRESET = b.dataset.preset;
  if (SCHEMA) renderLaunch();
}));

// Mirrors launch.rs::build_args (FULL DUMP), so the previewed command is the one that runs: every
// schema field is written explicitly in schema order, untouched Options as `none`, the engine-derived
// `*_specified` markers and the console-owned shutdown file excluded. The engine API key is hidden in
// the preview only — the real value is still injected at launch, never printed or copied.
function renderCmdline() {
  if (!SCHEMA) return;
  const fields = SCHEMA.schema || [];
  const eff = (f) => (FORM[f.path] !== undefined ? FORM[f.path] : (f.default !== undefined ? f.default : "none"));
  const args = [SCHEMA.binary || "infr.exe", "serve"];
  let model = null;
  for (const f of fields) {
    if (!f.path.startsWith("cli:")) continue;
    const v = eff(f);
    if (v === "") continue;
    if (f.flag) args.push(f.flag, v); else model = v;
  }
  if (model !== null) args.push(model);
  if (PRESET !== "manual") args.push("--set", `device.auto_profile=${PRESET}`);
  for (const f of fields) {
    const p = f.path;
    if (p.startsWith("cli:") || p === "serve.shutdown_file" || p.endsWith("_specified")) continue;
    if (p === "device.auto_profile" && PRESET !== "manual") continue;
    let v = eff(f);
    if (p === "serve.api_key" && v !== "" && v !== "none") v = "<已设置·预览隐藏>";
    args.push("--set", `${p}=${v}`);
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

// 推荐配置：本机（RX 7900 XTX 24GiB 独显 + 95GiB RAM）实测最优的可移植部分。模型 / 视觉 / 向量的
// 具体路径因人而异，不写死在共享前端里——套用后操作者再补 cli:model / cli:mmproj / uncensor.vector
// 与 serve.response_store_path 并保存。其余未列出的项仍按 schema 默认全量写出（见 renderCmdline）。
const RECOMMENDED = {
  "device.ctx": "262144",
  "kv.type_k": "q8_0",
  "kv.type_v": "q8_0",
  "sampling.temp": "1",
  "sampling.top_k": "20",
  "sampling.top_p": "0.95",
  "cli:parallel": "1",
  "spec.mtp": "false",
  "serve.response_store_entries": "256",
  "vision.image_url_allow_private": "false",
  // 分页树：page≥256 解码零税（本机实测），弹性走廊 + 专家压力让位，8GiB 内存帧池。
  "kv.paged": "true",
  "kv.paged_page_tokens": "256",
  "kv.paged_cache_max": "8GiB",
  "kv.paged_elastic": "true",
  "kv.paged_yield_under_pressure": "true",
};

function applyRecommended() {
  const byPath = new Map((SCHEMA.schema || []).map((f) => [f.path, f]));
  let n = 0;
  for (const [k, v] of Object.entries(RECOMMENDED)) {
    const f = byPath.get(k);
    if (!f) continue;
    // Only genuinely-non-default values enter FORM; the rest are still emitted at default by the
    // full dump, so the launch is identical either way and the form stays honest about what changed.
    if (v === f.default || (v === "" && f.default === "none")) delete FORM[k];
    else { FORM[k] = v; n++; }
  }
  PRESET = "conservative";
  renderLaunch();
  toast(`已套用推荐配置：${n} 项显式覆盖，其余按默认全量写出。请再补模型 / 视觉 / 向量路径后保存。`, "ok");
}

$("#launch-recommend").addEventListener("click", () => { if (SCHEMA) applyRecommended(); });
$("#launch-save").addEventListener("click", async () => {
  try {
    await api("ui/settings", { method: "POST", body: { preset: PRESET, set: FORM } });
    toast("设置已保存，刷新页面或重启控制台会自动恢复", "ok");
  } catch (e) { toast(e.message, "bad"); }
});

// ── 去审查向量下载 ───────────────────────────────────────────────────────────
// The URLs live in the backend (vector.rs), pinned to the wizard's real HF repo + mirror. The browser
// only asks for status / triggers the download; it never constructs or substitutes a download URL.
async function refreshVectorStatus() {
  const pathInput = $("#vector-path");
  const node = $("#vector-status");
  if (!pathInput || !node) return;
  const custom = pathInput.value.trim();
  try {
    const st = await api("ui/vector/status" + (custom ? `?path=${encodeURIComponent(custom)}` : ""));
    if (st.default_path) pathInput.placeholder = st.default_path;
    if (st.valid) { node.textContent = `✓ 已就绪，sha256 校验通过：${st.path}`; node.style.color = "var(--ok, #3fb950)"; }
    else if (st.exists) { node.textContent = `✗ 文件存在但校验不符（不是规范方向）：${st.path}`; node.style.color = "var(--bad, #f85149)"; }
    else { node.textContent = `未下载。默认路径：${st.default_path}`; node.style.color = ""; }
  } catch (e) { node.textContent = `状态读取失败：${e.message}`; node.style.color = ""; }
}

$("#vector-check").addEventListener("click", refreshVectorStatus);
$("#vector-path").addEventListener("change", refreshVectorStatus);
$("#vector-download").addEventListener("click", () => {
  const custom = $("#vector-path").value.trim();
  riskModal(async () => {
    const btn = $("#vector-download");
    btn.disabled = true;
    $("#vector-status").textContent = "下载中…（HuggingFace 优先，连不上自动转 hf-mirror）";
    try {
      const r = await api("ui/vector/download", { method: "POST", body: { path: custom || null, consent: true } });
      // Point the launch form at the downloaded file so 启动引擎 picks it up (consent asked again there).
      FORM["uncensor.vector"] = r.path;
      renderLaunch();
      toast("去审查向量已下载并校验通过", "ok");
    } finally {
      btn.disabled = false;
      refreshVectorStatus();
    }
  });
});

// ── Boot ───────────────────────────────────────────────────────────────────
connect();
api("ui/state").then((s) => { S = s; render(); }).catch((e) => setPill("bad", e.message));