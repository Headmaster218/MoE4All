const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');

const repo = path.resolve(__dirname, '..');
const runner = path.join(__dirname, 'run-gpu-resident-case.cjs');
let exe;
let model;
let modelName;
let mode;
let root;
let cache;

const submitLayers = process.env.RELEASE_BENCH_SUBMIT_LAYERS || '1';
const residentSets = [
  'device.vram_budget=24g',
  'device.vram_reserve=0',
  'device.ram_budget=48g',
  'kernels.vulkan.cpu_miss_threads=0',
  'kernels.vulkan.decode_gpu_resident=true',
  'kernels.vulkan.decode_gpu_resident_layers=16',
  'kernels.vulkan.decode_gpu_resident_fast_lut=true',
  'kernels.vulkan.decode_gpu_resident_hit_first=true',
  'kernels.vulkan.decode_gpu_resident_mapped_metadata=true',
  'kernels.vulkan.decode_gpu_resident_immutable_predicates=true',
  'kernels.vulkan.decode_gpu_resident_notice_thread=true',
  'kernels.vulkan.decode_gpu_resident_mapped_journal=true',
  `kernels.vulkan.decode_gpu_resident_submit_layers=${submitLayers}`,
  'kernels.vulkan.decode_gpu_resident_inflight=1',
  'kernels.vulkan.decode_gpu_resident_early_recovery=true',
];

function required(name) {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function sha256(file) {
  return crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
}

function sha256Prefix(file, bytes = 16 * 1024 * 1024) {
  const fd = fs.openSync(file, 'r');
  try {
    const buffer = Buffer.allocUnsafe(Math.min(bytes, fs.statSync(file).size));
    const read = fs.readSync(fd, buffer, 0, buffer.length, 0);
    return { bytes: read, sha256: crypto.createHash('sha256').update(buffer.subarray(0, read)).digest('hex') };
  } finally {
    fs.closeSync(fd);
  }
}

function git(args) {
  const result = spawnSync('git', args, { cwd: repo, encoding: 'utf8', windowsHide: true });
  if (result.status !== 0) throw new Error(result.stderr || `git ${args.join(' ')} failed`);
  return result.stdout.trim();
}

function runPhase(name, values) {
  const phaseRoot = path.join(root, name);
  fs.mkdirSync(phaseRoot, { recursive: true });
  const env = { ...process.env, ...values, TRACE_ROOT: phaseRoot, TRACE_EXE: exe,
    TRACE_MODEL: model, TRACE_TAG: name, TRACE_ENABLED: '0', TRACE_PROFILE_OPS: '0',
    TRACE_PROFILE_STAGES: '0', TRACE_PROFILE_PAGER: '0', TRACE_UBATCH: '4096',
    TRACE_THREADS: '10', TRACE_EXTRA_SETS: JSON.stringify(residentSets) };
  const child = spawnSync(process.execPath, [runner], { cwd: repo, env, windowsHide: true, stdio: 'inherit' });
  if (child.status !== 0) throw new Error(`${name} benchmark failed with ${child.status}`);
  const matrix = path.join(phaseRoot, `${name}-matrix.json`);
  if (!fs.existsSync(matrix)) throw new Error(`Missing matrix: ${matrix}`);
  return JSON.parse(fs.readFileSync(matrix));
}

function finiteSum(values) {
  return values.every(Number.isFinite) ? values.reduce((sum, value) => sum + value, 0) : null;
}

function summarizeDecode(matrix) {
  return matrix.map(row => {
    if (row.status !== 'complete' || row.rounds.length !== 4) {
      throw new Error(`Incomplete Decode row c${row.count} d${row.depth}`);
    }
    const measured = row.rounds.slice(1).map((round, index) => {
      const requests = round.requests;
      const metric = row.serverLogMetrics[index + 1];
      const completionTokens = requests.reduce((sum, request) => sum + request.usage.completion_tokens, 0);
      const criticalDecodeMs = Math.max(...requests.map(request => request.timings.predicted_ms));
      const perRequest = metric?.perRequest || [];
      return {
        measuredRound: index + 1,
        sourceRound: round.round,
        frontAggregateTokensPerSecond: finiteSum(perRequest.map(request => request.front)),
        middleAggregateTokensPerSecond: finiteSum(perRequest.map(request => request.middle)),
        lateAggregateTokensPerSecond: finiteSum(perRequest.map(request => request.late)),
        averageAggregateTokensPerSecond: completionTokens * 1000 / criticalDecodeMs,
        peakAggregateTokensPerSecond: metric?.steadyConcurrent?.peakAggregateTokensPerSecond ?? null,
        steadyAggregateTokensPerSecond: metric?.steadyConcurrent?.aggregateTokensPerSecond ?? null,
        perLaneTokensPerSecond: requests.map(request => request.timings.predicted_per_second),
        outputHashes: requests.map(request => request.outputHash),
        cachedTokens: requests.map(request => request.timings.cached_n),
      };
    });
    const unstableLanes = [];
    for (let lane = 0; lane < row.count; lane++) {
      const hashes = measured.map(round => round.outputHashes[lane]);
      if (new Set(hashes).size !== 1) {
        unstableLanes.push({ lane, hashes });
      }
    }
    const fields = ['frontAggregateTokensPerSecond', 'middleAggregateTokensPerSecond',
      'lateAggregateTokensPerSecond', 'averageAggregateTokensPerSecond',
      'peakAggregateTokensPerSecond', 'steadyAggregateTokensPerSecond'];
    const mean = Object.fromEntries(fields.map(field => {
      const values = measured.map(round => round[field]);
      return [field, values.every(Number.isFinite)
        ? values.reduce((sum, value) => sum + value, 0) / values.length : null];
    }));
    return { active: row.count, depth: row.depth, configuredSlots: row.slots,
      stability: { allMeasuredRoundsStable: unstableLanes.length === 0, unstableLanes },
      warmupRound: { sourceRound: row.rounds[0].round,
        rates: row.rounds[0].requests.map(request => request.timings.predicted_per_second) },
      measured, mean };
  });
}

function summarizePrefill(matrix) {
  return matrix.map(row => {
    if (row.status !== 'complete' || row.count !== 1 || row.rounds.length !== 1) {
      throw new Error(`Incomplete Prefill row c${row.count} d${row.depth}`);
    }
    const request = row.rounds[0].requests[0];
    if (request.timings.cached_n !== 0) throw new Error(`Prefill row unexpectedly reused ${request.timings.cached_n} tokens`);
    return { depth: row.depth, promptTokens: request.timings.prompt_n,
      cachedTokens: request.timings.cached_n,
      promptMilliseconds: request.timings.prompt_ms,
      promptTokensPerSecond: request.timings.prompt_per_second,
      decodeTokensPerSecond: request.timings.predicted_per_second,
      outputHash: request.outputHash };
  });
}

function main() {
  exe = path.resolve(required('RELEASE_BENCH_EXE'));
  model = path.resolve(required('RELEASE_BENCH_MODEL'));
  modelName = required('RELEASE_BENCH_MODEL_NAME').replace(/[^A-Za-z0-9._-]+/g, '-');
  mode = process.env.RELEASE_BENCH_MODE || 'all';
  if (!['all', 'decode', 'prefill'].includes(mode)) throw new Error(`Invalid benchmark mode: ${mode}`);
  for (const file of [exe, model, runner]) if (!fs.statSync(file).isFile()) throw new Error(`Missing file: ${file}`);
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  root = path.join(repo, 'benchmark-data', 'artifacts', 'release-1.0.0-benchmark', `${stamp}-${modelName}`);
  cache = path.join(repo, 'test-kv-cache');
  fs.mkdirSync(root, { recursive: true });
  const version = spawnSync(exe, ['--version'], { encoding: 'utf8', windowsHide: true });
  if (version.status !== 0 || !/\b1\.0\.0\b/.test(version.stdout)) {
    throw new Error(`Expected a 1.0.0 binary, got: ${version.stdout || version.stderr}`);
  }
  const manifest = { startedAt: new Date().toISOString(), root, modelName, model,
    modelBytes: fs.statSync(model).size, modelFirstShardPrefix: sha256Prefix(model),
    exe, exeSha256: sha256(exe), version: version.stdout.trim(), head: git(['rev-parse', 'HEAD']),
    sourceStatus: git(['status', '--short']), device: 'Vulkan1 (RX 7900 XTX)',
    decodeDefinition: 'one warmup round followed by three measured rounds; average aggregate is total output tokens divided by the slowest lane Decode duration; peak is the maximum fully-contained one-second server Decode window',
    prefillDefinition: 'single cold request with SSD KV disabled and cached_n required to be zero',
    residentSets, mode, status: 'running' };
  fs.writeFileSync(path.join(root, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  if (mode !== 'prefill') {
    manifest.decode = summarizeDecode(runPhase('decode', {
      TRACE_SLOTS: '3', TRACE_COUNTS: '1,2,3', TRACE_DEPTHS: '30000,150000',
      TRACE_REPEATS: '4', TRACE_TOKENS: '512', TRACE_SYSTEM_PREFIX: '1',
      TRACE_CONTINUATION: '0', TRACE_SHARED_KV: cache, TRACE_SHARED_KV_MAX: '90g',
    }));
    fs.writeFileSync(path.join(root, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  }
  if (mode !== 'decode') {
    manifest.prefill = summarizePrefill(runPhase('prefill', {
      TRACE_SLOTS: '1', TRACE_COUNTS: '1', TRACE_DEPTHS: '30000,150000',
      TRACE_REPEATS: '1', TRACE_TOKENS: '8', TRACE_SYSTEM_PREFIX: '0',
      TRACE_CONTINUATION: '0', TRACE_SHARED_KV: '',
    }));
  }
  manifest.status = 'complete';
  manifest.finishedAt = new Date().toISOString();
  fs.writeFileSync(path.join(root, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  fs.writeFileSync(path.join(path.dirname(root), `latest-${modelName}.txt`), `${root}\n`);
  console.log(JSON.stringify({ complete: root, decode: manifest.decode, prefill: manifest.prefill }));
}

if (require.main === module) main();

module.exports = { summarizeDecode, summarizePrefill };
