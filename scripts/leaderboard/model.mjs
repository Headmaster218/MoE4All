// Compact index v1 projection shared by contract with the website reader.
export const INDEX_SCHEMA = 'moe4all-leaderboard/1';
export const INDEX_LIMITS = Object.freeze({ compressedBytes: 16777216, decodedBytes: 67108864, reports: 50000, contexts: 150000, measurements: 300000, cells: 8000000 });
// Column order is the v1 wire contract; incompatible changes require a new schema.
export const FIELDS = Object.freeze({
  reports: 'id date engine engine_commit profile model architecture quant model_bytes gpu cpu threads os system_arch backend ram_bytes ram_generation ram_modules ram_speeds vram_bytes unified_memory integrated pcie_generation pcie_width pcie_max_generation pcie_max_width pcie_gbps pcie_max_gbps pcie_source storage_model storage_type storage_bus storage_probe storage_seq storage_random storage_iops storage_cache storage_physical_fraction auto_profile mtp kv_k kv_v configured_ubatch actual_ubatch configured_threads ram_budget vram_budget vram_reserve actual_ram_budget actual_vram_budget host_dma submit_cap warmup_tokens measured_tokens prompt_tokens depth_tokens requested_rounds depth_mode failure_count failure_stages demo'.split(' '),
  contexts: 'report context prompt_min prompt_max start_min start_max end_min end_max prefill prefill_tokens evaluated_tokens prefill_seconds setup_slots failure_count failure_stages'.split(' '),
  measurements: 'context kind slots rate first middle last rounds status failure_count failure_stages'.split(' ')
});
const numericReports = new Set('model_bytes threads ram_bytes ram_modules vram_bytes pcie_generation pcie_width pcie_max_generation pcie_max_width pcie_gbps pcie_max_gbps storage_seq storage_random storage_iops storage_physical_fraction configured_ubatch actual_ubatch configured_threads actual_ram_budget actual_vram_budget submit_cap warmup_tokens measured_tokens prompt_tokens depth_tokens requested_rounds failure_count'.split(' '));
const booleans = new Set(['unified_memory', 'integrated', 'mtp', 'host_dma', 'demo']);
const listFields = new Set(['ram_speeds', 'failure_stages']);
const fail = () => { throw new Error('排行榜数据格式无效或超出读取上限'); };
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const number = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= Number.MAX_SAFE_INTEGER;
const numeric = value => number(value) ? value : null;
const integer = value => number(value) && Number.isInteger(value) ? value : null;
const text = value => typeof value === 'string' && value.length <= 1024 ? value : null;
const bool = value => typeof value === 'boolean' ? value : null;
const range = values => { const usable = Array.isArray(values) ? values.filter(value => integer(value) !== null) : []; return usable.length ? [Math.min(...usable), Math.max(...usable)] : [null, null]; };
const stages = failures => [...new Set((failures || []).map(f => text(f.stage)).filter(Boolean))].sort();
const budget = value => typeof value === 'string' ? text(value) : number(value) ? String(value) : null;
export function strategy(value) { const s = typeof value === 'string' ? value.trim().toLowerCase() : ''; return ['aggressive', 'conservative', 'manual'].includes(s) ? s : null; }
function depth(configuration) {
  if (configuration.depth_mode === 'none') return 'none';
  if (configuration.synthetic_depth === true || configuration.depth_mode === 'synthetic') return 'synthetic';
  if (configuration.real_context === true || configuration.depth_mode === 'real') return 'real';
  return null;
}

export function projectBenchmark(payload, { demo = false } = {}) {
  const h = payload.hardware || {}, gpu = h.gpu || {}, ram = h.ram || {}, c = payload.configuration || {}, disk = h.model_storage || {}, probe = disk.read_probe || {};
  const modules = Array.isArray(ram.modules) ? ram.modules : [];
  const report = {
    id: payload.id, date: payload.created_at_utc, engine: payload.engine_version, engine_commit: text(payload.engine_commit), profile: payload.profile,
    model: payload.model.name, architecture: text(payload.model.architecture), quant: text(payload.model.quantization ?? payload.model.quant), model_bytes: integer(payload.model.storage_bytes),
    gpu: text(gpu.name), cpu: text(h.cpu?.name), threads: integer(h.cpu?.logical_threads), os: text(h.os), system_arch: text(h.architecture), backend: text(c.backend),
    ram_bytes: integer(ram.total_bytes), ram_generation: text(ram.memory_generation), ram_modules: integer(ram.module_count),
    ram_speeds: [...new Set(modules.map(m => integer(m.configured_speed_mt_s)).filter(value => value !== null))].sort((a, b) => a - b),
    vram_bytes: integer(gpu.vram_total_bytes), unified_memory: bool(gpu.unified_memory), integrated: bool(gpu.integrated),
    pcie_generation: integer(gpu.pcie_generation), pcie_width: integer(gpu.pcie_width), pcie_max_generation: integer(gpu.pcie_max_generation), pcie_max_width: integer(gpu.pcie_max_width),
    pcie_gbps: numeric(gpu.pcie_theoretical_one_way_gb_per_second), pcie_max_gbps: numeric(gpu.pcie_max_theoretical_one_way_gb_per_second), pcie_source: text(gpu.pcie_source),
    storage_model: text(disk.model), storage_type: text(disk.media_type), storage_bus: text(disk.bus_type), storage_probe: text(probe.mode),
    storage_seq: numeric(probe.sequential?.mib_per_second), storage_random: numeric(probe.random_64k?.mib_per_second), storage_iops: numeric(probe.random_64k?.iops), storage_cache: text(probe.cache_disposition), storage_physical_fraction: numeric(probe.physical_read_fraction),
    auto_profile: strategy(c.auto_profile), mtp: bool(c.mtp), kv_k: text(c.kv_type_k), kv_v: text(c.kv_type_v),
    configured_ubatch: integer(c.configured_ubatch ?? c.ubatch), actual_ubatch: integer(c.actual_ubatch), configured_threads: integer(c.configured_threads),
    ram_budget: budget(c.configured_ram_budget ?? c.ram_budget), vram_budget: budget(c.configured_vram_budget ?? c.vram_budget), vram_reserve: budget(c.configured_vram_reserve ?? c.vram_reserve),
    actual_ram_budget: integer(c.actual_ram_budget_bytes), actual_vram_budget: integer(c.actual_vram_budget_bytes), host_dma: bool(c.host_dma), submit_cap: integer(c.submit_dispatches),
    warmup_tokens: integer(c.decode_warmup_tokens_per_rep), measured_tokens: integer(c.decode_tokens_per_rep ?? c.decode_tokens), prompt_tokens: integer(c.prompt_tokens), depth_tokens: integer(c.depth_tokens), requested_rounds: integer(c.decode_repetitions ?? c.repetitions), depth_mode: depth(c),
    failure_count: (payload.failures || []).length, failure_stages: stages(payload.failures), demo
  };
  const contexts = [], measurements = [];
  for (const context of payload.contexts) {
    const [prompt_min, prompt_max] = range(context.actual_prompt_tokens), [start_min, start_max] = range(context.decode_measurement_start_tokens), [end_min, end_max] = range(context.decode_measurement_end_tokens);
    const failures = context.failures || [], prefill = context.prefill;
    const index = contexts.length;
    contexts.push({ report: 0, context: context.requested_context_tokens, prompt_min, prompt_max, start_min, start_max, end_min, end_max,
      prefill: numeric(prefill?.tokens_per_second), prefill_tokens: integer(prefill?.actual_prompt_tokens), evaluated_tokens: integer(prefill?.evaluated_tokens), prefill_seconds: numeric(prefill?.seconds), setup_slots: integer(prefill?.setup_slots), failure_count: failures.length, failure_stages: stages(failures) });
    if (payload.profile === 'manual') {
      if (!['decode', 'prefill'].includes(context.manual_result?.kind)) fail();
      measurements.push({ context: index, kind: context.manual_result.kind === 'prefill' ? 'manual_prefill' : 'manual_decode', slots: integer(context.manual_result.parallel ?? c.parallel), rate: context.manual_result.average_tokens_per_second,
        first: null, middle: null, last: null, rounds: integer(c.repetitions), status: failures.length ? 'partial' : 'ok', failure_count: failures.length, failure_stages: stages(failures) });
      continue;
    }
    const seen = new Set();
    for (const d of context.decode || []) {
      seen.add(d.parallel);
      const failed = failures.filter(f => f.parallel === d.parallel), sections = d.section_mean_tokens_per_second;
      measurements.push({ context: index, kind: 'decode', slots: d.parallel, rate: d.overall_tokens_per_second, first: sections[0], middle: sections[1], last: sections[2], rounds: d.repetitions.length,
        status: failed.length || (report.requested_rounds !== null && d.repetitions.length < report.requested_rounds) ? 'partial' : 'ok', failure_count: failed.length, failure_stages: stages(failed) });
    }
    for (const parallel of [...new Set(failures.filter(f => Number.isInteger(f.parallel) && !seen.has(f.parallel)).map(f => f.parallel))]) {
      const failed = failures.filter(f => f.parallel === parallel);
      measurements.push({ context: index, kind: 'decode', slots: parallel, rate: null, first: null, middle: null, last: null, rounds: 0, status: 'failed', failure_count: failed.length, failure_stages: stages(failed) });
    }
    if (!seen.size && !measurements.some(m => m.context === index) && failures.length && prefill == null) measurements.push({ context: index, kind: 'decode', slots: null, rate: null, first: null, middle: null, last: null, rounds: 0, status: 'failed', failure_count: failures.length, failure_stages: stages(failures) });
  }
  return { report, contexts, measurements };
}

export function assembleIndex(entries, { mode = 'production', source = null } = {}) {
  const index = { schema: INDEX_SCHEMA, mode, source, reports: [], contexts: [], measurements: [] }, ids = new Set();
  const sorted = [...entries].sort((a, b) => a.payload.id < b.payload.id ? -1 : a.payload.id > b.payload.id ? 1 : 0);
  for (const entry of sorted) {
    if (ids.has(entry.payload.id)) throw new Error('重复的 Benchmark 报告 ID');
    ids.add(entry.payload.id);
    const { report, contexts, measurements } = projectBenchmark(entry.payload, { demo: entry.demo === true });
    const reportIndex = index.reports.length, contextOffset = index.contexts.length;
    index.reports.push(report);
    index.contexts.push(...contexts.map(c => ({ ...c, report: reportIndex })));
    index.measurements.push(...measurements.map(m => ({ ...m, context: m.context + contextOffset })));
  }
  validateIndex(index);
  return index;
}

function validValue(table, field, value) {
  if (listFields.has(field)) return Array.isArray(value) && value.length <= 64 && value.every(v => field === 'ram_speeds' ? number(v) : typeof v === 'string' && v.length <= 128);
  if (value === null) return true;
  if (booleans.has(field)) return typeof value === 'boolean';
  const isNumber = table === 'reports' ? numericReports.has(field) : table === 'contexts' ? field !== 'failure_stages' : !['kind', 'status', 'failure_stages'].includes(field);
  return isNumber ? number(value) : typeof value === 'string' && value.length <= 1024;
}
export function validSource(source) { return object(source) && /^[\w.-]+\/[\w.-]+$/.test(source.repository) && /^[a-f0-9]{40}$/.test(source.commit) && source.path_version === 1; }
export function validateManifest(m, { repository = 'Headmaster218/MoE4All', preview = false } = {}) {
  if (!object(m) || m.schema !== INDEX_SCHEMA || !/^[a-f0-9]{64}$/.test(m.version) || m.decoded_sha256 !== m.version || !['production', 'preview'].includes(m.mode) || (m.mode === 'preview' && !preview)) fail();
  if (m.source !== null && (!validSource(m.source) || m.source.repository !== repository)) fail();
  if (m.package_commit !== undefined && (typeof m.package_commit !== 'string' || !/^[a-f0-9]{40}$/.test(m.package_commit))) fail();
  if (!Number.isSafeInteger(m.decoded_bytes) || m.decoded_bytes < 1 || m.decoded_bytes > INDEX_LIMITS.decodedBytes || !object(m.counts)) fail();
  if (Object.keys(m.counts).sort().join(',') !== 'contexts,measurements,reports') fail();
  for (const key of ['reports', 'contexts', 'measurements']) if (!Number.isSafeInteger(m.counts[key]) || m.counts[key] < 0 || m.counts[key] > INDEX_LIMITS[key]) fail();
  if (!Number.isFinite(Date.parse(m.generated_at)) || (m.mode === 'production' && m.counts.reports && !validSource(m.source)) || !object(m.packages)) fail();
  for (const format of ['brotli', 'gzip']) {
    const p = m.packages[format], suffix = format === 'brotli' ? 'br' : 'gz';
    if (!object(p) || p.url !== `index-${m.version}.${suffix}.bin` || !Number.isSafeInteger(p.bytes) || p.bytes < 1 || p.bytes > INDEX_LIMITS.compressedBytes || !/^[a-f0-9]{64}$/.test(p.sha256)) fail();
  }
  return m;
}
export function validateIndex(index) {
  if (!object(index) || index.schema !== INDEX_SCHEMA || !['production', 'preview'].includes(index.mode)) fail();
  if (index.mode === 'production' && index.reports?.length && !validSource(index.source)) fail();
  if (index.source !== null && !validSource(index.source)) fail();
  let cells = 0;
  for (const [name, fields] of Object.entries(FIELDS)) {
    const rows = index[name];
    if (!Array.isArray(rows) || rows.length > INDEX_LIMITS[name]) fail();
    cells += rows.length * fields.length;
    if (cells > INDEX_LIMITS.cells) fail();
    for (const row of rows) {
      if (!object(row) || Object.keys(row).length !== fields.length || fields.some(field => !Object.hasOwn(row, field) || !validValue(name, field, row[field]))) fail();
    }
  }
  const ids = new Set();
  for (const r of index.reports) {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$/.test(r.id) || !/^\d{4}-\d{2}-\d{2}T.*(?:Z|\+00:00)$/.test(r.date) || !Number.isFinite(Date.parse(r.date)) || new Date(r.date).toISOString().slice(0, 10) !== r.date.slice(0, 10) || !r.model || !r.engine || ids.has(r.id) || !['quick', 'standard', 'full', 'manual'].includes(r.profile) || typeof r.demo !== 'boolean' || !Number.isSafeInteger(r.failure_count)) fail();
    if (index.mode === 'production' && r.demo || ![null, 'aggressive', 'conservative', 'manual'].includes(r.auto_profile)) fail();
    ids.add(r.id);
  }
  const contextKeys = new Set();
  for (const c of index.contexts) {
    if (!Number.isInteger(c.report) || !index.reports[c.report] || !Number.isInteger(c.context)) fail();
    const key = `${c.report}:${c.context}`; if (contextKeys.has(key)) fail(); contextKeys.add(key);
    for (const kind of ['prompt', 'start', 'end']) if ((c[kind + '_min'] === null) !== (c[kind + '_max'] === null) || c[kind + '_min'] > c[kind + '_max']) fail();
  }
  const caseKeys = new Set();
  for (const m of index.measurements) {
    if (!Number.isInteger(m.context) || !index.contexts[m.context] || !['decode', 'manual_decode', 'manual_prefill'].includes(m.kind) || !['ok', 'partial', 'failed'].includes(m.status) || (m.slots !== null && (!Number.isInteger(m.slots) || m.slots < 1 || m.slots > 64))) fail();
    if (m.status === 'failed' && m.rate !== null || m.status !== 'failed' && m.rate === null) fail();
    const key = `${m.context}:${m.kind}:${m.slots}`; if (caseKeys.has(key)) fail(); caseKeys.add(key);
  }
  return index;
}

function encodeTable(rows, fields) {
  if (!rows.length) return { n: 0, c: [] };
  return { n: rows.length, c: fields.map(field => {
    const values = rows.map(row => row[field]), dictionary = [], positions = new Map();
    const refs = values.map(value => { const key = JSON.stringify(value); if (!positions.has(key)) { positions.set(key, dictionary.length); dictionary.push(value); } return positions.get(key); });
    if (dictionary.length <= 1) return { k: dictionary[0] ?? null };
    const plain = { v: values }, packed = { d: dictionary, v: refs };
    return JSON.stringify(packed).length < JSON.stringify(plain).length ? packed : plain;
  }) };
}
export function encodeIndex(index) {
  validateIndex(index);
  return { schema: INDEX_SCHEMA, mode: index.mode, source: index.source, ...Object.fromEntries(Object.entries(FIELDS).map(([name, fields]) => [name, encodeTable(index[name], fields)])) };
}
export function decodeIndex(encoded) {
  if (!object(encoded) || encoded.schema !== INDEX_SCHEMA) fail();
  const index = { schema: encoded.schema, mode: encoded.mode, source: encoded.source };
  let cells = 0;
  for (const [name, fields] of Object.entries(FIELDS)) {
    const table = encoded[name];
    if (!object(table) || !Number.isInteger(table.n) || table.n < 0 || table.n > INDEX_LIMITS[name] || !Array.isArray(table.c) || table.c.length !== (table.n ? fields.length : 0)) fail();
    cells += table.n * fields.length; if (cells > INDEX_LIMITS.cells) fail();
    if (!table.n) { index[name] = []; continue; }
    const rows = Array.from({ length: table.n }, () => ({}));
    for (const [i, field] of fields.entries()) {
      const column = table.c[i];
      if (!object(column) || Object.keys(column).some(key => !['k', 'd', 'v'].includes(key))) fail();
      const constant = Object.hasOwn(column, 'k'), dictionary = Object.hasOwn(column, 'd');
      if (constant ? Object.keys(column).length !== 1 : !Array.isArray(column.v) || column.v.length !== table.n || (dictionary && (!Array.isArray(column.d) || column.d.length > table.n))) fail();
      for (let r = 0; r < table.n; r++) {
        if (dictionary && (!Number.isInteger(column.v[r]) || column.v[r] < 0 || column.v[r] >= column.d.length)) fail();
        const value = constant ? column.k : dictionary ? column.d[column.v[r]] : column.v[r];
        if (!validValue(name, field, value)) fail(); rows[r][field] = value;
      }
    }
    index[name] = rows;
  }
  return validateIndex(index);
}

export function provenanceLinks(index, row) {
  if (index.mode !== 'production' || row.demo || !validSource(index.source)) return null;
  const r = index.reports.find(report => report.id === row.report_id);
  if (!r) return null;
  const { repository, commit } = index.source, path = `benchmark-database/v1/${r.date.slice(0, 7)}/${encodeURIComponent(r.id)}.m4b`;
  return { view: `https://github.com/${repository}/blob/${commit}/${path}`, download: `https://raw.githubusercontent.com/${repository}/${commit}/${path}` };
}

export function indexRows(index, metric = 'decode') {
  const rows = [];
  const base = (c, i) => {
    const r = index.reports[c.report];
    return { ...r, report_id: r.id, id: `${r.id}:${c.context}`, context: c.context, prompt_min: c.prompt_min, prompt_max: c.prompt_max, start_min: c.start_min, start_max: c.start_max, end_min: c.end_min, end_max: c.end_max,
      gpu: r.gpu || '未提供 GPU 型号', quant: r.quant, ubatch: r.actual_ubatch, prefill: c.prefill, prefill_tokens: c.prefill_tokens, evaluated_tokens: c.evaluated_tokens, setup_slots: c.setup_slots, prefill_seconds: c.prefill_seconds, context_ref: i };
  };
  if (metric === 'prefill') {
    index.contexts.forEach((c, i) => {
      if (c.prefill !== null) rows.push({ ...base(c, i), id: base(c, i).id + ':prefill', kind: 'prefill', slots: null, decode: null, sections: null, rounds: 1, status: 'ok', failure_count: c.failure_count, failure_stages: c.failure_stages });
      else if (c.failure_stages.some(stage => /prefill/i.test(stage))) rows.push({ ...base(c, i), id: base(c, i).id + ':prefill', kind: 'prefill', slots: null, decode: null, sections: null, rounds: 0, status: 'failed', failure_count: c.failure_count, failure_stages: c.failure_stages });
    });
  }
  for (const m of index.measurements) {
    if ((metric === 'prefill') !== (m.kind === 'manual_prefill')) continue;
    const c = index.contexts[m.context], b = base(c, m.context);
    rows.push({ ...b, id: `${b.id}:${m.kind}:${m.slots ?? 'unknown'}`, kind: m.kind, slots: m.slots, decode: metric === 'decode' ? m.rate : null, prefill: metric === 'prefill' ? m.rate : c.prefill,
      sections: m.first !== null && m.middle !== null && m.last !== null ? [m.first, m.middle, m.last] : null, rounds: m.rounds, status: m.status, failure_count: m.failure_count, failure_stages: m.failure_stages });
  }
  return rows;
}
