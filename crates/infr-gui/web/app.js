const $ = (id) => document.getElementById(id);
let key = localStorage.getItem('infrGuiKey') || '';
let data = null;
let selectedModel = '';
let formDirty = false;

async function api(path, options = {}) {
  const headers = {...(options.headers || {}), 'Authorization': `Bearer ${key}`};
  if (options.body) headers['Content-Type'] = 'application/json';
  const response = await fetch(path, {...options, headers});
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

function bytes(n) {
  if (n === null || n === undefined) return '—';
  const units = ['B','KiB','MiB','GiB','TiB']; let i = 0; let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i < 2 ? 1 : 2)} ${units[i]}`;
}

function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function msg(text, error = false) { $('message').textContent = text; $('message').className = error ? 'error' : 'subtle'; }

async function refresh(full = false) {
  try {
    const next = await api(full || !data ? '/api/bootstrap' : '/api/status');
    data = full || !data ? next : {...data, ...next};
    $('login').classList.add('hidden');
    $('login-error').textContent = '';
    renderRuntime();
    renderDownload();
    if (full) {
      renderDirectories(); renderModels(); renderEmbeddingModels(); renderProfiles(); renderDevices(); renderSchema();
    }
  } catch (e) {
    if (String(e.message).includes('management key')) {
      $('login').classList.remove('hidden');
      $('login-error').textContent = e.message;
    }
    else msg(e.message, true);
    return false;
  }
  return true;
}

function renderRuntime() {
  const r = data.runtime;
  $('phase').textContent = ({stopped:'已停止',loading:'正在加载',ready:'运行中',stopping:'正在停止',failed:'运行失败'})[r.phase] || r.phase;
  $('phase-dot').className = `dot ${r.phase}`;
  $('active-model').textContent = r.model_path || '尚未载入模型';
  $('prefill').textContent = r.prefill_tps == null ? '—' : r.prefill_tps.toFixed(1);
  $('decode').textContent = r.decode_tps == null ? '—' : r.decode_tps.toFixed(1);
  $('pid').textContent = r.pid ? `PID ${r.pid} · ${r.service_addr}` : '';
  renderRuntimeMemory(r.memory || {});
  $('logs').textContent = r.logs.length ? r.logs.join('\n') : '等待 worker…';
  $('logs').scrollTop = $('logs').scrollHeight;
}

function hostMode(mode) {
  return ({full:'Full RAM',bounded:'Bounded RAM/SSD',bypass:'SSD 直读',disabled:'已关闭',unknown:'启动时确定'})[mode] || mode || '—';
}

function renderRuntimeMemory(memory) {
  const items = [];
  if (memory.expert_cache_target_bytes != null) items.push(['Expert target', bytes(memory.expert_cache_target_bytes)]);
  if (memory.elastic_pool_bytes != null) items.push(['Elastic pool', bytes(memory.elastic_pool_bytes)]);
  if (memory.unified_arena_bytes != null) items.push(['Unified arena', bytes(memory.unified_arena_bytes)]);
  if (memory.host_mode) {
    const coverage = memory.host_cache_bytes != null && memory.expert_payload_bytes
      ? ` · ${(memory.host_cache_bytes / memory.expert_payload_bytes * 100).toFixed(1)}%`
      : '';
    items.push(['Host tier', `${hostMode(memory.host_mode)} · ${bytes(memory.host_cache_bytes)}${coverage}`]);
  }
  if (memory.host_dma_imported_bytes != null) {
    const total = memory.host_dma_total_bytes == null ? '' : ` / ${bytes(memory.host_dma_total_bytes)}`;
    const arenas = memory.host_dma_arenas ? ` · ${memory.host_dma_arenas} arenas` : '';
    items.push(['Host DMA', `${bytes(memory.host_dma_imported_bytes)}${total}${arenas}`]);
  }
  if (memory.kv_layout || memory.context_tokens != null) {
    items.push(['KV / Context', `${esc(memory.kv_layout || '—')} · ${memory.context_tokens ?? '—'} tokens`]);
  }
  $('runtime-memory').classList.toggle('hidden', !items.length);
  $('runtime-memory').innerHTML = items.map(([name,value]) => `<div><span>${esc(name)}</span><b>${value}</b></div>`).join('');
}

function renderDownload() {
  const d = data.download;
  let text = ({idle:'空闲',downloading:'下载中',completed:'下载完成',failed:'下载失败'})[d.phase] || d.phase;
  if (d.model_ref) text += ` · ${d.model_ref}`;
  if (d.downloaded_path) text += `\n${d.downloaded_path}`;
  if (d.last_error) text += `\n${d.last_error}`;
  $('download-state').textContent = text;
  $('download-log-wrap').classList.toggle('hidden', !d.logs.length);
  $('download-logs').textContent = d.logs.join('\n');
  $('download-logs').scrollTop = $('download-logs').scrollHeight;
  $('download').disabled = d.phase === 'downloading';
}

function renderDirectories() {
  $('directories').innerHTML = data.saved.directories.map(p => `<span class="chip">${esc(p)}<button data-remove-dir="${esc(p)}">×</button></span>`).join('');
  document.querySelectorAll('[data-remove-dir]').forEach(b => b.onclick = () => post('/api/directories/remove', {path:b.dataset.removeDir}, true));
}

function renderModels() {
  const q = $('model-filter').value.toLowerCase();
  const favorites = new Set(data.saved.favorites);
  const recent = new Set(data.saved.recent);
  const recentOrder = new Map(data.saved.recent.map((path, index) => [path, index]));
  const models = data.catalog
    .filter(m => `${m.name} ${m.path} ${m.architecture||''}`.toLowerCase().includes(q))
    .sort((a, b) => {
      const favoriteDelta = Number(favorites.has(b.path)) - Number(favorites.has(a.path));
      if (favoriteDelta) return favoriteDelta;
      const recentA = recentOrder.has(a.path) ? recentOrder.get(a.path) : Number.MAX_SAFE_INTEGER;
      const recentB = recentOrder.has(b.path) ? recentOrder.get(b.path) : Number.MAX_SAFE_INTEGER;
      return recentA - recentB || a.name.localeCompare(b.name, 'zh-CN');
    });
  $('models').classList.toggle('empty', !models.length);
  $('models').innerHTML = models.length ? models.map(m => `
    <div class="model ${selectedModel===m.path?'selected':''}" data-model="${esc(m.path)}">
      <div class="model-title"><span>${esc(m.name)}</span><button class="star" data-star="${esc(m.path)}">${favorites.has(m.path)?'★':'☆'}</button></div>
      <div class="model-meta">${esc(m.architecture||'未知架构')} · ${bytes(m.size_bytes)} · ${m.is_moe?'MoE':'Dense'}${m.tasks.includes('embedding')?' · Embedding':''}${m.modalities.includes('image')?' · Vision':''}${recent.has(m.path)?' · 最近使用':''}</div>
      <div class="model-meta" title="${esc(m.path)}">${esc(m.path)}</div>
      ${m.error?`<div class="error">${esc(m.error)}</div>`:''}
    </div>`).join('') : '没有匹配的模型';
  document.querySelectorAll('[data-model]').forEach(el => el.onclick = (ev) => {
    if (ev.target.dataset.star) return;
    selectedModel = el.dataset.model; $('model-path').value = selectedModel; formDirty = true; renderModels();
    const model = data.catalog.find(m => m.path === selectedModel);
    if (model?.tasks?.length === 1 && model.tasks[0] === 'embedding') $('task').value = 'embedding';
    syncTaskFields();
  });
  document.querySelectorAll('[data-star]').forEach(b => b.onclick = async (ev) => { ev.stopPropagation(); await post('/api/favorites/toggle',{path:b.dataset.star},true); });
}

function renderProfiles() {
  const current = $('profile-select').value;
  $('profile-select').innerHTML = '<option value="">新配置</option>' + data.saved.profiles.map(p => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join('');
  if (data.saved.profiles.some(p => p.id === current)) $('profile-select').value = current;
}

function renderEmbeddingModels() {
  $('embedding-models').innerHTML = data.catalog
    .filter(model => model.tasks.includes('embedding'))
    .map(model => `<option value="${esc(model.path)}">${esc(model.name)}</option>`)
    .join('');
}

function renderDevices() {
  const current = $('backend').value;
  const base = ['<option value="cpu">CPU</option>','<option value="metal">Metal</option>'];
  const gpu = data.devices.map(d => `<option value="${d.id}">${esc(d.id)} · ${esc(d.name)} · ${bytes(d.vram_bytes)}</option>`);
  $('backend').innerHTML = gpu.concat(base).join('');
  $('backend').value = data.devices.some(d=>d.id===current) ? current : (data.devices.find(d=>d.default)?.id || 'cpu');
}

function renderSchema() {
  $('config-paths').innerHTML = data.config_schema.map(f => `<option value="${esc(f.path)}">${esc(f.default_value)}</option>`).join('');
}

function syncTaskFields() {
  const embeddingTask = $('task').value === 'embedding';
  const attachedEmbedding = $('embedding-model').value.trim() !== '';
  $('embedding-model-wrap').classList.toggle('hidden', embeddingTask);
  $('embedding-runner-wrap').classList.toggle('hidden', !embeddingTask && !attachedEmbedding);
  $('pager-controls').classList.toggle('hidden', embeddingTask);
  $('ram-budget').disabled = embeddingTask || $('dram-bypass').checked;
}

function readProfile() {
  const extra = {};
  $('advanced').value.split(/\r?\n/).map(s=>s.trim()).filter(Boolean).forEach(line => {
    const at = line.indexOf('='); if (at < 1) throw new Error(`高级参数缺少 =：${line}`);
    extra[line.slice(0,at).trim()] = line.slice(at+1).trim();
  });
  return {
    id: $('profile-id').value || `profile-${Date.now()}`,
    name: $('profile-name').value.trim() || 'Default', model_path:$('model-path').value.trim(), task:$('task').value,
    embedding_model_path:$('embedding-model').value.trim(),
    embedding_runner:$('embedding-runner').value.trim(),
    backend:$('backend').value, context:$('context').value.trim(), ubatch:$('ubatch').value?Number($('ubatch').value):null,
    kv_type_k:$('kv-k').value, kv_type_v:$('kv-v').value, vram_budget:$('vram-budget').value.trim(),
    vram_reserve:$('vram-reserve').value.trim(), ram_budget:$('ram-budget').value.trim(), expert_cache:$('expert-cache').value.trim(),
    host_dma:$('host-dma').checked, dram_bypass:$('dram-bypass').checked, pager_stats:$('pager-stats').checked,
    pager_trace:$('pager-trace').value.trim(),
    parallel:Number($('parallel').value), service_addr:$('service-addr').value.trim(), service_api_key:$('service-key').value,
    max_tokens_cap:Number($('max-tokens').value), extra
  };
}

function fillProfile(p) {
  $('profile-id').value=p.id||''; $('profile-name').value=p.name||'Default'; $('model-path').value=p.model_path||''; selectedModel=p.model_path||'';
  $('task').value=p.task||'chat'; $('embedding-model').value=p.embedding_model_path||''; $('embedding-runner').value=p.embedding_runner||''; $('backend').value=p.backend||'Vulkan0'; $('context').value=p.context||''; $('ubatch').value=p.ubatch||'';
  $('kv-k').value=p.kv_type_k||'auto'; $('kv-v').value=p.kv_type_v||'auto'; $('vram-budget').value=p.vram_budget||'';
  $('vram-reserve').value=p.vram_reserve||''; $('ram-budget').value=p.ram_budget||''; $('expert-cache').value=p.expert_cache||'';
  $('host-dma').checked=p.host_dma!==false; $('dram-bypass').checked=Boolean(p.dram_bypass); $('pager-stats').checked=Boolean(p.pager_stats); $('pager-trace').value=p.pager_trace||'';
  $('parallel').value=p.parallel||1; $('service-addr').value=p.service_addr||'0.0.0.0:8080'; $('service-key').value=p.service_api_key||'';
  $('max-tokens').value=p.max_tokens_cap||131072; $('advanced').value=Object.entries(p.extra||{}).map(([k,v])=>`${k}=${v}`).join('\n');
  formDirty=false; syncTaskFields(); renderModels();
}

async function estimateProfile(profile) {
  const e = await api('/api/estimate',{method:'POST',body:JSON.stringify(profile)});
  const cards = [
    ['GGUF 文件',e.model_bytes],['固定 VRAM 权重',e.fixed_vram_bytes],['Expert payload',e.expert_payload_bytes],
    ['KV / 持久状态',e.kv_bytes],['运行时弹性峰值',e.runtime_reserve_bytes],['Packing margin',e.weight_packing_margin_bytes],
    ['驱动 / Post-load 预留',(e.load_driver_reserve_bytes || 0) + (e.post_load_reserve_bytes || 0)],
    ['有效 VRAM 预算',e.effective_vram_budget_bytes],['Expert target',e.estimated_cache_room_bytes],['Elastic pool',e.elastic_pool_bytes],
    ['联合 Embedding 权重',e.embedding_model_bytes]
  ].filter(([,value]) => value !== null && value !== undefined);
  const coverage = e.host_cache_coverage == null ? '' : ` · ${(e.host_cache_coverage * 100).toFixed(1)}%`;
  if (e.host_cache_mode) cards.push([`Host tier · ${hostMode(e.host_cache_mode)}${coverage}`,e.effective_ram_cache_bytes]);
  const ramState = e.fits_ram_budget === false ? ' · SSD 参与运行时 miss' : '';
  $('memory-estimate').innerHTML = cards.map(([n,v])=>`<div class="memory-card"><b>${bytes(v)}</b><span>${esc(n)}</span></div>`).join('') + `<div class="memory-card"><b>${e.fits_minimum===false?'可能不足':'可规划'}</b><span>${esc(e.architecture||'未知')} · ${e.confidence}${ramState}</span></div>`;
  msg(e.notes.join(' '));
  return e;
}

async function estimate() {
  try { return await estimateProfile(readProfile()); }
  catch(e) { msg(e.message,true); throw e; }
}

async function post(path, body, full=false) {
  try { const r=await api(path,{method:'POST',body:JSON.stringify(body)}); msg(r.message||'完成'); await refresh(full); return r; }
  catch(e){ msg(e.message,true); throw e; }
}

$('login-form').onsubmit = async e => { e.preventDefault(); key=$('admin-key').value; localStorage.setItem('infrGuiKey',key); await refresh(true); };
$('add-directory').onclick = () => post('/api/directories/add',{path:$('directory').value},true);
$('rescan').onclick = () => post('/api/models/rescan',{},true);
$('model-filter').oninput = renderModels;
$('download-source').onchange = () => $('download-custom').classList.toggle('hidden',$('download-source').value!=='custom');
$('download').onclick = () => { const source=$('download-source').value; post('/api/downloads/start',{model_ref:$('download-ref').value,endpoint:source==='custom'?$('download-custom').value:source,jobs:Number($('download-jobs').value)},false); };
$('profile-select').onchange = () => { const p=data.saved.profiles.find(p=>p.id===$('profile-select').value); if(p) fillProfile(p); else fillProfile({}); };
$('task').onchange = () => { syncTaskFields(); formDirty = true; };
$('embedding-model').oninput = () => { syncTaskFields(); formDirty = true; };
$('dram-bypass').onchange = () => { syncTaskFields(); formDirty = true; };
$('save').onclick = async () => { try{const p=readProfile(); await post('/api/profiles/save',p,true); $('profile-id').value=p.id; $('profile-select').value=p.id;}catch(e){} };
$('delete-profile').onclick = async () => {
  const id = $('profile-id').value;
  if (!id) return msg('当前是尚未保存的新配置', true);
  if (confirm('删除当前保存的配置？模型文件不会被删除。')) {
    try { await post('/api/profiles/delete',{path:id},true); fillProfile({}); $('profile-select').value=''; }
    catch(e) {}
  }
};
$('add-advanced').onclick = () => {
  const path = $('config-path').value.trim(); const value = $('config-value').value.trim();
  if (!path || !value) return msg('高级参数路径和值都不能为空', true);
  const line = `${path}=${value}`; const old = $('advanced').value.trim();
  $('advanced').value = old ? `${old}\n${line}` : line;
  $('config-path').value = ''; $('config-value').value = ''; formDirty = true;
};
$('estimate').onclick = estimate;
$('start').onclick = async () => {
  try {
    const p=readProfile(); const plan=await estimateProfile(p);
    if (plan.fits_minimum === false && !confirm('预估显示当前 VRAM 预算可能不足，仍然尝试加载？')) return;
    await post('/api/worker/start',p,true); $('profile-id').value=p.id;
  } catch(e) { msg(e.message,true); }
};
$('stop').onclick = () => post('/api/worker/stop',{force:false});
$('force-stop').onclick = () => { if(confirm('强制停止可能中断正在执行的 GPU 命令。仅在优雅停止长期无响应时使用。')) post('/api/worker/stop',{force:true}); };
document.querySelectorAll('input,select,textarea').forEach(el => el.addEventListener('change',()=>formDirty=true));

(async()=>{ syncTaskFields(); await refresh(true); setInterval(()=>refresh(false),1500); })();
