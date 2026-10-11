const { spawn, spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const net = require('node:net');
const { finished } = require('node:stream/promises');
const { parseLogMetrics } = require('./gpu-resident-log-metrics.cjs');

const root = path.resolve(process.env.TRACE_ROOT || 'benchmark-data/artifacts/decode-timeline-20261007');
const slots = Number(process.env.TRACE_SLOTS || 3);
const exe = path.resolve(process.env.TRACE_EXE || 'target/release/infr-timeline.exe');
if (!process.env.TRACE_MODEL) throw new Error('TRACE_MODEL must name the benchmark GGUF');
const model = path.resolve(process.env.TRACE_MODEL);
const counts = (process.env.TRACE_COUNTS || '1,2,3').split(',').map(Number);
const depths = (process.env.TRACE_DEPTHS || '30000,150000').split(',').map(Number);
const repeats = Number(process.env.TRACE_REPEATS || 3);
const tokens = Number(process.env.TRACE_TOKENS || 512);
const continuation = process.env.TRACE_CONTINUATION === '1';
const systemPrefix = process.env.TRACE_SYSTEM_PREFIX === '1';
const sharedKv = process.env.TRACE_SHARED_KV ? path.resolve(process.env.TRACE_SHARED_KV) : null;
const traceEnabled = process.env.TRACE_ENABLED !== '0';
const tag = process.env.TRACE_TAG || 'measured';
const deadlineMs = process.env.TRACE_DEADLINE ? Date.parse(process.env.TRACE_DEADLINE) : Infinity;
if (Number.isNaN(deadlineMs)) throw new Error('invalid TRACE_DEADLINE');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const log = value => console.log(JSON.stringify({ at: new Date().toISOString(), ...value }));
const hash = text => crypto.createHash('sha256').update(text).digest('hex');
fs.mkdirSync(root, { recursive: true });
const env = { ...process.env, RUST_LOG: 'info' };
for (const key of Object.keys(env)) if (key.startsWith('INFR_')) delete env[key];
const subjects = ['transactional database storage and crash recovery', 'operating system scheduling and virtual memory', 'distributed consensus and replicated state machines'];
const fillers = [
  ' storage transaction wal checkpoint recovery page index btree lsm compaction snapshot isolation durability checksum',
  ' scheduler process thread virtual memory page table syscall interrupt filesystem io completion async fairness latency',
  ' consensus quorum term log replication leader election lease membership failure recovery linearizability partition',
];

function plan(depth, count, lane) {
  const dir = path.join(root, '_prompts');
  fs.mkdirSync(dir, { recursive: true });
  const stem = `c${count}-d${depth}-l${lane}`;
  const file = path.join(dir, `${stem}.json`);
  if (!fs.existsSync(file)) {
    const template = path.join(dir, `${stem}-template.json`);
    fs.writeFileSync(template, JSON.stringify([{ role: 'user', content:
      `Independent benchmark ${stem}. Read these background records.\n{{INFR_FILLER}}\n` +
      `Write a detailed numbered technical analysis of ${subjects[lane]}. Compare concrete implementations, ` +
      'failure modes, tradeoffs, and measurable performance effects. Start immediately with section 1. ' +
      'Keep each section substantive and continue until the response limit.' }]));
    const result = spawnSync(exe, ['__test-plan-prompt', model, '--messages', template,
      '--target', String(depth), '--output', file, '--filler', fillers[lane]],
      { env, windowsHide: true, encoding: 'utf8', maxBuffer: 16 * 1024 * 1024 });
    if (result.status !== 0) throw new Error(result.stderr || result.error || 'prompt planning failed');
  }
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

async function port() {
  const server = net.createServer();
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve); });
  const value = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return value;
}

async function request(url, modelId, prompt, lane) {
  const started = Date.now();
  const steadyStarted = performance.now();
  const response = await fetch(`${url}/v1/chat/completions`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: modelId, messages: prompt.messages, stream: true,
      max_tokens: tokens, temperature: 0, seed: lane + 1 }),
    signal: AbortSignal.timeout(Math.max(1, Math.min(40 * 60 * 1000, deadlineMs - Date.now() - 10000))),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
  const decoder = new TextDecoder();
  let pending = '', content = '', usage, timings, finishReason;
  const frameTimes = [];
  for await (const chunk of response.body) {
    pending += decoder.decode(chunk, { stream: true });
    const lines = pending.split('\n'); pending = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const value = line.slice(6).trim();
      if (!value || value === '[DONE]') continue;
      const event = JSON.parse(value);
      if (event.error) throw new Error(JSON.stringify(event.error));
      usage = event.usage || usage; timings = event.timings || timings;
      const choice = event.choices?.[0];
      if (choice?.delta?.content || choice?.delta?.reasoning_content) frameTimes.push(performance.now() - steadyStarted);
      content += choice?.delta?.content || ''; content += choice?.delta?.reasoning_content || '';
      finishReason = choice?.finish_reason || finishReason;
    }
  }
  if (!usage || usage.completion_tokens !== tokens) throw new Error(`unexpected output length ${JSON.stringify(usage)}`);
  const rate = (start, end) => (end - start) * 1000 / (frameTimes[end] - frameTimes[start]);
  const a = Math.floor(frameTimes.length / 4), b = Math.floor(frameTimes.length * 3 / 4);
  const streamWindowRates = frameTimes.length === usage.completion_tokens && a > 1 ? {
    front: rate(0, a - 1), middle: rate(a, b - 1), late: rate(b, frameTimes.length - 1),
  } : null;
  return { lane, usage, timings, finishReason, wallMs: Date.now() - started, outputHash: hash(content),
    streamFrames: frameTimes.length, streamWindowRates, content };
}

async function run(count, depth) {
  const dir = path.join(root, `${tag}-c${count}-d${depth}`); fs.mkdirSync(dir, { recursive: true });
  const prompt = Array.from({ length: count }, (_, lane) => {
    const value = plan(depth, systemPrefix ? 3 : count, lane);
    return systemPrefix ? { ...value, messages: [
      { role: 'system', content: value.messages[0].content },
      { role: 'user', content: 'Perform the requested technical analysis. Start immediately with section 1 and continue substantive numbered sections until the response limit.' },
    ] } : value;
  });
  const shutdown = path.join(dir, 'shutdown'); if (fs.existsSync(shutdown)) fs.unlinkSync(shutdown);
  const trace = path.join(dir, 'trace.json');
  const url = `http://127.0.0.1:${await port()}`;
  const nativeArgs = ['serve', model, '--addr', url.slice(7), '--parallel', String(slots), '--ctx', '160k',
    '--dev', 'Vulkan1', '--temp', '0', '--seed', '1', '--no-think', '--max-new', String(tokens),
    '--set', 'device.auto_profile=aggressive', '--set', 'kv.type_k=q8_0', '--set', 'kv.type_v=q8_0',
    '--set', 'kv.overflow=false', '--set', `kv.session_cache_dir=${sharedKv || ''}`, '--set', 'spec.mtp=false',
    '--set', 'prof.ops=false', '--set', 'prof.stages=false', '--set', 'prof.pager_profile=false',
    '--set', 'paging.expert_prefetch=false',
    '--set', 'sampling.ignore_eos=true', '--set', `serve.shutdown_file=${shutdown}`];
  nativeArgs.push('--set', 'serve.stats_interval_secs=1');
  nativeArgs[nativeArgs.indexOf('--parallel') + 1] = String(slots);
  if (process.env.TRACE_UBATCH) nativeArgs.push('--ubatch', process.env.TRACE_UBATCH);
  if (process.env.TRACE_THREADS) nativeArgs.push('--threads', process.env.TRACE_THREADS);
  if (process.env.TRACE_CPU_MISS_FIXTURE) nativeArgs.push('--set',
    `prof.cpu_miss_fixture_dir=${path.resolve(process.env.TRACE_CPU_MISS_FIXTURE)}`);
  if (sharedKv) nativeArgs.push('--set', 'kv.session_idle_secs=3600',
    '--set', `kv.session_cache_max=${process.env.TRACE_SHARED_KV_MAX || '48g'}`, '--set', 'kv.session_cache_ttl_hours=0');
  for (const entry of JSON.parse(process.env.TRACE_EXTRA_SETS || '[]')) nativeArgs.push('--set', entry);
  if (traceEnabled) nativeArgs.push('--set', `prof.timeline_path=${trace}`, '--set', `prof.timeline_lanes=${count}`,
    '--set', `prof.timeline_skip_steps=${process.env.TRACE_SKIP || 64}`, '--set', `prof.timeline_steps=${process.env.TRACE_STEPS || 16}`,
    '--set', `prof.timeline_stride=${process.env.TRACE_STRIDE || 128}`, '--set', `prof.timeline_windows=${process.env.TRACE_WINDOWS || 12}`);
  const stdout = fs.createWriteStream(path.join(dir, 'stdout.log'));
  const stderr = fs.createWriteStream(path.join(dir, 'stderr.log'));
  const logsFinished = Promise.all([finished(stdout), finished(stderr)]);
  const child = spawn(exe, nativeArgs, { env, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  if (process.env.TRACE_AFFINITY) {
    const mask = Number(process.env.TRACE_AFFINITY);
    if (!Number.isSafeInteger(mask) || mask <= 0) throw new Error('invalid affinity mask');
    const set = spawnSync('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command',
      `(Get-Process -Id ${child.pid}).ProcessorAffinity = [System.IntPtr]${mask}`],
      { windowsHide: true, encoding: 'utf8' });
    if (set.status !== 0) { child.kill(); throw new Error(set.stderr || 'affinity failed'); }
  }
  const deadlineKill = Number.isFinite(deadlineMs) ? setTimeout(() => child.kill(), Math.max(1, deadlineMs - Date.now())) : null;
  let exitInfo = null, last = '';
  child.stdout.pipe(stdout);
  child.stderr.on('data', data => { stderr.write(data); last = data.toString().trim().split(/\r?\n/).at(-1); });
  const closed = new Promise(resolve => child.once('close', (code, signal) => { exitInfo = { code, signal }; stdout.end(); stderr.end(); resolve(); }));
  const heartbeat = setInterval(() => log({ count, depth, pid: child.pid, latest: last }), 30000);
  const result = { count, depth, slots, repeats, tokens, continuation, systemPrefix, sharedKv,
    warmRoundSkip: systemPrefix ? 1 : 0, traceEnabled, exe, binaryHash: hash(fs.readFileSync(exe)), args: nativeArgs, rounds: [] };
  log({ start: dir, pid: child.pid });
  try {
    let modelId;
    for (let attempt = 0; attempt < 720 && !exitInfo && Date.now() < deadlineMs - 10000; attempt++) {
      try {
        const response = await fetch(`${url}/v1/models`, { signal: AbortSignal.timeout(1500) });
        if (response.ok) { modelId = (await response.json()).data[0].id; break; }
      } catch {}
      await sleep(1000);
    }
    if (!modelId) throw new Error(`startup failed ${JSON.stringify(exitInfo)} ${last}`);
    const conversations = prompt.map(value => [...value.messages]);
    for (let round = 0; round < repeats; round++) {
      const startedUnixMs = Date.now();
      const requests = await Promise.all(Array.from({ length: count }, (_, lane) => request(url, modelId,
        continuation ? { messages: conversations[lane] } : prompt[lane], lane)));
      if ((continuation || systemPrefix) && round > 0) {
        for (const req of requests) {
          const initial = result.rounds[0].requests.find(value => value.lane === req.lane).usage.prompt_tokens;
          if (req.timings.cached_n < initial - 128) throw new Error(`continuation did not reuse the initial prefix: lane=${req.lane} cached=${req.timings.cached_n} initial=${initial}`);
        }
      }
      result.rounds.push({ round, startedUnixMs, finishedUnixMs: Date.now(), requests });
      if (continuation) for (const req of requests) conversations[req.lane].push(
        { role: 'assistant', content: req.content },
        { role: 'user', content: `Continue the analysis of ${subjects[req.lane]}. Expand on concrete implementation details, edge cases, and measurable tradeoffs. Start immediately and keep writing substantive numbered sections until the response limit.` });
      fs.writeFileSync(path.join(dir, 'result.json'), JSON.stringify(result, null, 2));
      log({ count, depth, round, requests: requests.map(({ content, ...value }) => value) });
      await sleep(2000);
    }
    result.status = 'complete';
  } catch (error) {
    result.status = 'failed'; result.error = String(error.stack || error); log({ count, depth, error: result.error });
  } finally {
    clearInterval(heartbeat);
    if (!exitInfo) { fs.writeFileSync(shutdown, 'stop\n'); await closed; }
    await logsFinished;
    if (deadlineKill) clearTimeout(deadlineKill);
    result.exitInfo = exitInfo;
    result.serverLogMetrics = parseLogMetrics(fs.readFileSync(path.join(dir, 'stderr.log'), 'utf8'), result.rounds, count);
    fs.writeFileSync(path.join(dir, 'result.json'), JSON.stringify(result, null, 2));
    log({ stopped: child.pid, status: result.status, trace: fs.existsSync(trace) ? fs.statSync(trace).size : null });
  }
  return result;
}

async function main() {
  const results = [];
  for (const depth of depths) for (const count of counts) {
    if (Date.now() > deadlineMs - 120000) { log({ skipped: { count, depth }, reason: 'GPU deadline' }); continue; }
    results.push(await run(count, depth));
    fs.writeFileSync(path.join(root, `${tag}-matrix.json`), JSON.stringify(results, null, 2));
  }
}
main().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
