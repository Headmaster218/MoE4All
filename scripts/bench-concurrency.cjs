// Local OpenAI streaming benchmark. Server progress/cycle logs provide token counts;
// SSE events are retained only for first-text latency and pauses, never as token counts.
const { spawn, spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');
const crypto = require('node:crypto');

const [label, mode = 'ordinary', profile = 'off', selection = 'short,medium,mixed,reverse,long,split'] = process.argv.slice(2);
if (!label || !['ordinary', 'mtp'].includes(mode) || !['off', 'pager', 'ops'].includes(profile)) {
  throw new Error('Usage: node scripts/bench-concurrency.cjs LABEL ordinary|mtp off|pager|ops [short,medium,mixed,reverse,long,split]');
}
const root = path.resolve(process.env.BENCH_ROOT || 'target/concurrency-20260923');
fs.mkdirSync(root, { recursive: true });
const exe = path.resolve(process.env.BENCH_EXE || 'target/release/infr.exe');
const model = process.env.BENCH_MODEL || 'D:\\AILMStudioModels\\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64\\Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf';
const head = process.env.BENCH_MTP_HEAD || 'G:\\Qwen3.8-Flash-Next-UD-Q2_K_XL\\MTP\\mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf';
const scenarios = {
  short: { depths: [512, 512], outputs: [1024, 1024], delay: 0 },
  lowmid: { depths: [1536, 1536], outputs: [1536, 1024], delay: 6000 },
  medium: { depths: [8192, 8192], outputs: [1024, 1024], delay: 0 },
  mixed: { depths: [3584, 29696], outputs: [1536, 1024], delay: 6000 },
  reverse: { depths: [29696, 3584], outputs: [1536, 1024], delay: 6000 },
  long: { depths: [28672, 28672], outputs: [1024, 1024], delay: 0 },
  split: { depths: [512, 28672], outputs: [1024, 1024], delay: 6000 },
  tiny: { depths: [128, 128], outputs: [256, 256], delay: 0 },
  ctx30: { depths: [30720, 30720], outputs: [512, 512], delay: 0 },
  ctx45: { depths: [46080, 46080], outputs: [512, 512], delay: 0 },
  ctx60: { depths: [61440, 61440], outputs: [512, 512], delay: 0 },
  ctx75: { depths: [76800, 76800], outputs: [512, 512], delay: 0 },
  ctx90: { depths: [92160, 92160], outputs: [512, 512], delay: 0 },
  ctx150: { depths: [153600, 153600], outputs: [512, 512], delay: 0 },
};
const names = selection.split(',');
for (const name of names) if (!scenarios[name]) throw new Error(`Unknown scenario: ${name}`);
if (process.env.BENCH_TOKENS) {
  const tokens = Number(process.env.BENCH_TOKENS);
  if (!Number.isInteger(tokens) || tokens <= 0) throw new Error('BENCH_TOKENS must be positive');
  for (const cfg of Object.values(scenarios)) cfg.outputs = [tokens, tokens];
}
const env = { ...process.env };
const single = process.env.BENCH_SINGLE === '1';
for (const key of Object.keys(env)) if (key.startsWith('INFR_')) delete env[key];
Object.assign(env, {
  RUST_LOG: 'info,infr_llama::parallel=debug',
  INFR_RAM_BUDGET: '48g', INFR_VRAM_BUDGET: '24g', INFR_UBATCH_PARALLEL: '256',
  INFR_MTP: mode === 'mtp' ? '1' : '0', INFR_SERVE_STATS_SECS: '1',
});
if (mode === 'mtp') env.INFR_SPEC_DRAFT = head;
if (process.env.BENCH_GRID_NR) env.INFR_GEMV_ID_GRID_NR = process.env.BENCH_GRID_NR;
if (process.env.BENCH_NO_SHARED_SLOT) env.INFR_NO_MOE_SHARED_SLOT = '1';
if (profile !== 'off') env.INFR_PAGER_PROFILE = '1';
if (profile === 'ops') Object.assign(env, { INFR_PROF_OPS: '1', INFR_PROF_OP_SHAPES: '1' });
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const digest = text => crypto.createHash('sha256').update(text).digest('hex');
const emit = value => console.log(JSON.stringify({ at: new Date().toISOString(), ...value }));
const prefix = path.join(root, label);
let child;
let childExit = null;
let activeCase = null;
const serverStopped = new AbortController();
let rawLine = '';
const lines = [];
const cases = [];

function readLog(chunk) {
  rawLine += chunk.toString();
  const complete = rawLine.split(/\r?\n/);
  rawLine = complete.pop();
  for (let line of complete) {
    line = line.replace(/\x1b\[[0-9;]*m/g, '');
    lines.push({ at: Date.now(), case: activeCase, line });
    if (/request done|serving 1 request|parallel scheduler batch failed|ERROR|panicked/.test(line)) emit({ server: line });
  }
}

function planPrompt(name, lane) {
  const cfg = scenarios[name];
  const out = path.join(root, `prompt-${name}-${lane}-${cfg.depths[lane]}.json`);
  if (fs.existsSync(out)) return JSON.parse(fs.readFileSync(out, 'utf8'));
  const messages = [{ role: 'user', content: `Task ${name} stream ${lane}. Read these background records: {{INFR_FILLER}}\nWrite an extensive numbered technical handbook with at least 100 detailed sections about ${lane === 0 ? 'database storage, indexing, transactions, recovery, and query execution' : 'operating system scheduling, memory allocation, virtual memory, and file systems'}. Give concrete examples and discuss design tradeoffs in every section. Start immediately with section 1 and continue in detail.` }];
  const template = path.join(root, `template-${name}-${lane}.json`);
  fs.writeFileSync(template, JSON.stringify(messages));
  const r = spawnSync(exe, ['__test-plan-prompt', model, '--messages', template, '--target', String(cfg.depths[lane]), '--output', out, '--filler', ' storage compute memory queue cache latency throughput capacity transaction checkpoint'], { env: { ...env, INFR_MTP: '0', INFR_NO_THINK: '1' }, encoding: 'utf8', windowsHide: true, maxBuffer: 4 * 1024 * 1024 });
  if (r.status !== 0) throw new Error(r.stderr || r.error || 'prompt planning failed');
  const planned = JSON.parse(fs.readFileSync(out, 'utf8'));
  emit({ planned: name, lane, promptTokens: planned.prompt_tokens });
  return planned;
}

async function reservePort() {
  const server = net.createServer();
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  const port = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return port;
}

function startRequest(url, modelId, planned, name, lane, output) {
  const state = { name, lane, plannedTokens: planned.prompt_tokens, maxTokens: output, sentAt: Date.now(), firstAt: null, doneAt: null, events: [], content: '', reasoning: '', usage: null, timings: null, error: null };
  state.promise = (async () => {
    try {
      const response = await fetch(url + '/v1/chat/completions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId, messages: planned.messages, stream: true, max_tokens: output, temperature: 0, seed: 1 }),
        signal: AbortSignal.any([AbortSignal.timeout(20 * 60 * 1000), serverStopped.signal]),
      });
      state.headersAt = Date.now();
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      const decoder = new TextDecoder();
      let pending = '';
      for await (const chunk of response.body) {
        pending += decoder.decode(chunk, { stream: true });
        const rows = pending.split('\n');
        pending = rows.pop();
        for (const row of rows) {
          if (!row.startsWith('data: ')) continue;
          const data = row.slice(6).trim();
          if (data === '[DONE]') continue;
          const event = JSON.parse(data);
          if (event.error) throw new Error(JSON.stringify(event.error));
          if (event.usage) state.usage = event.usage;
          if (event.timings) state.timings = event.timings;
          const choice = event.choices?.[0];
          if (choice?.finish_reason) state.finishReason = choice.finish_reason;
          const content = choice?.delta?.content || '';
          const reasoning = choice?.delta?.reasoning_content || '';
          if (content || reasoning) {
            const at = Date.now();
            if (state.firstAt === null) { state.firstAt = at; emit({ firstText: name, lane, ttftSec: (at - state.sentAt) / 1000 }); }
            state.events.push({ at, content, reasoning });
            state.content += content;
            state.reasoning += reasoning;
          }
        }
      }
      if (!state.usage) throw new Error('Stream ended without authoritative usage');
    } catch (error) { state.error = String(error); }
    state.doneAt = Date.now();
    emit({ done: name, lane, wallSec: (state.doneAt - state.sentAt) / 1000, usage: state.usage, timings: state.timings, error: state.error });
  })();
  return state;
}

function persist() {
  const data = { label, mode, profile, exe, exeSha256: digest(fs.readFileSync(exe)), model, ctx: Number(process.env.BENCH_CTX || '32768'), requestedSlots: 2, env: Object.fromEntries(Object.entries(env).filter(([k]) => k.startsWith('INFR_'))), cases: cases.map(c => ({ ...c, requests: c.requests.map(({ promise, ...r }) => ({ ...r, outputSha256: digest(r.content + r.reasoning) })) })), lines };
  fs.writeFileSync(prefix + '.json', JSON.stringify(data));
}

async function main() {
  const prompts = Object.fromEntries(names.map(name => [name, [planPrompt(name, 0), planPrompt(name, 1)]]));
  const devs = spawnSync(exe, ['devices'], { env, encoding: 'utf8', windowsHide: true });
  const device = devs.stdout?.match(/(Vulkan\d+): AMD Radeon RX 7900 XTX/)?.[1];
  if (!device) throw new Error(devs.stderr || 'RX 7900 XTX not found');
  const port = await reservePort();
  const url = `http://127.0.0.1:${port}`;
  const args = ['serve', model, '--addr', `127.0.0.1:${port}`, '--parallel', '2', '--ctx', process.env.BENCH_CTX || '32768', '--dev', device, '--temp', '0', '--seed', '1', '--no-think'];
  if (process.env.BENCH_UBATCH !== 'auto') args.push('--ubatch', process.env.BENCH_UBATCH || '3072');
  if (process.env.BENCH_SET) args.push('--set', process.env.BENCH_SET);
  const out = fs.createWriteStream(prefix + '.out.log');
  const err = fs.createWriteStream(prefix + '.err.log');
  child = spawn(exe, args, { env, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  child.stdout.on('data', chunk => { out.write(chunk); readLog(chunk); });
  child.stderr.on('data', chunk => { err.write(chunk); readLog(chunk); });
  const closed = new Promise(resolve => child.once('close', (code, signal) => { childExit = { code, signal }; serverStopped.abort(); out.end(); err.end(); resolve(); }));
  emit({ serverPid: child.pid, url, mode, profile, args });
  const heartbeat = setInterval(() => {
    const latest = lines.filter(x => /serve stats|request progress|mtp summary|parallel-token-profile/.test(x.line)).slice(-1)[0];
    emit({ heartbeat: activeCase || 'loading', latest: latest?.line || lines.at(-1)?.line });
  }, 20000);
  try {
    let modelId;
    const deadline = Date.now() + 10 * 60 * 1000;
    while (Date.now() < deadline) {
      if (childExit) throw new Error(`Server exited: ${JSON.stringify(childExit)}`);
      try {
        const response = await fetch(url + '/v1/models', { signal: AbortSignal.timeout(1500) });
        if (response.ok) { modelId = (await response.json()).data[0].id; break; }
      } catch {}
      await sleep(1000);
    }
    if (!modelId) throw new Error('Server did not become ready');
    emit({ ready: modelId });
    for (const name of names) {
      activeCase = name;
      const cfg = scenarios[name];
      const c = { name, depths: cfg.depths, startAt: Date.now(), requests: [] };
      cases.push(c);
      const a = startRequest(url, modelId, prompts[name][0], name, 0, cfg.outputs[0]);
      c.requests.push(a);
      if (single) {
        await a.promise;
        c.doneAt = Date.now();
        persist();
        if (childExit) break;
        await sleep(2000);
        continue;
      }
      if (cfg.delay) {
        while (!a.firstAt && !a.doneAt) await sleep(50);
        if (a.firstAt) await sleep(cfg.delay);
      }
      const b = startRequest(url, modelId, prompts[name][1], name, 1, cfg.outputs[1]);
      c.requests.push(b);
      await Promise.all([a.promise, b.promise]);
      c.doneAt = Date.now();
      persist();
      if (childExit) break;
      await sleep(2000);
    }
  } finally {
    clearInterval(heartbeat);
    persist();
    if (!childExit) child.kill();
    await closed;
    emit({ stopped: child.pid, childExit, results: prefix + '.json' });
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; if (child && !childExit) child.kill(); });
