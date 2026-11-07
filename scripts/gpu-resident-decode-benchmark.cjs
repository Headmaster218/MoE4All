const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const { verifyMatchedOutputs } = require('./gpu-resident-correctness.cjs');

const sourceRoot = path.resolve(__dirname, '..');
const benchRoot = process.env.RESIDENT_BENCH_ROOT || 'D:/AIinfr/infr';
const cache = path.join(benchRoot, 'test-kv-cache');
const artifacts = path.join(sourceRoot, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const budgetGiB = Number(process.env.RESIDENT_CACHE_BUDGET_GIB || 30);
if (![30, 100].includes(budgetGiB)) throw new Error('Only the normal 30 GiB or authorized temporary 100 GiB budget is supported');
const budget = budgetGiB * 1024 ** 3;
const configuredSlots = Number(process.env.RESIDENT_SLOTS || 2);
if (![1, 2].includes(configuredSlots)) throw new Error('RESIDENT_SLOTS must be one or two');
if (!process.env.RESIDENT_MODEL) throw new Error('RESIDENT_MODEL must name the benchmark GGUF');
const model = path.resolve(process.env.RESIDENT_MODEL);
if (!fs.statSync(model).isFile()) throw new Error(`Not a model file: ${model}`);
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
function cacheBytes(dir = cache) {
  if (!fs.existsSync(dir)) return 0;
  const stat = fs.lstatSync(dir);
  if (stat.isSymbolicLink()) throw new Error(`Cache reparse point: ${dir}`);
  if (!stat.isDirectory()) return stat.size;
  return fs.readdirSync(dir).reduce((sum, name) => sum + cacheBytes(path.join(dir, name)), 0);
}
const cases = JSON.parse(process.env.RESIDENT_CASES || JSON.stringify([
  { name: 'baseline', enabled: false },
  { name: 'conditional-l4', enabled: true, layers: 4 },
]));
const exe = path.resolve(process.env.RESIDENT_EXE || path.join(benchRoot, 'target/gpu-resident-experiment/release/infr.exe'));
const tag = process.env.RESIDENT_TAG || 'stage1';
fs.mkdirSync(artifacts, { recursive: true });
const runGit = args => {
  const value = spawnSync('git', args, { cwd: sourceRoot, encoding: 'utf8', windowsHide: true });
  if (value.status !== 0) throw new Error(value.stderr);
  return value.stdout;
};
const sourceFiles = [
  'Cargo.toml', 'Cargo.lock', 'crates/infr-core/Cargo.toml', 'crates/infr-core/src/graph.rs',
  'crates/infr-core/src/config/mod.rs', 'crates/infr-core/src/backend.rs', 'crates/infr-vulkan/build.rs',
  'crates/infr-core/src/timeline.rs', 'crates/infr-vulkan/src/timeline.rs',
  'crates/infr-llama/src/seam/runner.rs',
  'crates/infr-vulkan/src/adapter.rs', 'crates/infr-vulkan/src/gemm.rs',
  'crates/infr-vulkan/src/lib.rs', 'crates/infr-vulkan/src/pager.rs',
  'crates/infr-vulkan/src/recorder.rs', 'crates/infr-vulkan/shaders/gpu_resident_check.comp',
  'crates/infr-vulkan/src/transfer.rs',
  'crates/infr-vulkan/shaders/zero_words.comp',
  'crates/infr-vulkan/shaders/gpu_resident_hit_check.comp',
  'crates/infr-vulkan/shaders/gpu_resident_silu.comp',
  'crates/infr-vulkan/shaders/gpu_resident_dispatch.comp',
  'scripts/gpu-resident-decode-benchmark.cjs', 'scripts/run-gpu-resident-case.cjs',
  'scripts/gpu-resident-log-metrics.cjs',
  'scripts/gpu-resident-correctness.cjs',
  'scripts/gpu-resident-cache-identity.cjs',
  'scripts/run-gpu-resident-full-graph.cjs',
  'scripts/run-gpu-resident-setup-scan.cjs',
  'scripts/run-gpu-resident-stable-timeline.cjs',
  'scripts/run-gpu-resident-owned-graph.cjs',
  'scripts/run-gpu-resident-inline-io.cjs',
  'scripts/run-gpu-resident-notice-poll.cjs',
  'scripts/run-gpu-resident-mapped-copy.cjs',
];
const manifest = {
  sourceRoot, benchRoot, cache, cacheBudgetGiB: budgetGiB, model, exe, binaryHash: hash(fs.readFileSync(exe)),
  head: runGit(['rev-parse', 'HEAD']).trim(),
  sourceHashes: Object.fromEntries(sourceFiles.map(file => [file, hash(fs.readFileSync(path.join(sourceRoot, file)))])),
  sourceSnapshotScope: 'Current workspace; a case with binarySourceManifest uses that frozen build instead',
  configuredSlots, cases, results: [],
};
fs.writeFileSync(path.join(artifacts, `${tag}-source.patch`), runGit(['diff', '--binary', 'HEAD']));
fs.writeFileSync(path.join(artifacts, `${tag}-manifest.json`), JSON.stringify(manifest, null, 2));
for (const file of sourceFiles.filter(file => file.includes('/shaders/') || file.startsWith('scripts/run-gpu-resident-') && !file.endsWith('case.cjs'))) {
  fs.copyFileSync(path.join(sourceRoot, file), path.join(artifacts, `${tag}-${path.basename(file)}`));
}
for (const item of cases) {
  const before = cacheBytes();
  if (before > budget) throw new Error(`Aggregate test cache exceeds budget: ${before}`);
  // A dual 150K save needs about 6.1 GiB of temporary room before namespace GC.
  if (before > budget - 7 * 1024 ** 3) throw new Error(`Insufficient safe cache write headroom: ${before}`);
  const caseTag = `${tag}-${item.name}`;
  const caseExe = path.resolve(item.exe || exe);
  const binaryHash = hash(fs.readFileSync(caseExe));
  let binarySourceManifest = null;
  if (item.binarySourceManifest) {
    const source = path.resolve(item.binarySourceManifest);
    const build = JSON.parse(fs.readFileSync(source));
    if (build.binaryHash !== binaryHash) throw new Error(`Frozen build hash differs: ${source}`);
    binarySourceManifest = { source, hash: hash(fs.readFileSync(source)) };
    fs.copyFileSync(source, path.join(artifacts, `${caseTag}-binary-source-manifest.json`));
  }
  const env = {
    ...process.env, TRACE_ROOT: artifacts, TRACE_EXE: caseExe, TRACE_MODEL: model, TRACE_TAG: caseTag,
    TRACE_ENABLED: process.env.RESIDENT_TRACE || '0', TRACE_SLOTS: String(configuredSlots),
    TRACE_COUNTS: item.counts || process.env.RESIDENT_COUNTS || '1,2',
    TRACE_DEPTHS: item.depths || process.env.RESIDENT_DEPTHS || '30000,150000',
    TRACE_REPEATS: process.env.RESIDENT_REPEATS || '3',
    TRACE_TOKENS: process.env.RESIDENT_TOKENS || '512', TRACE_UBATCH: '4096',
    TRACE_THREADS: '10', TRACE_SYSTEM_PREFIX: '1', TRACE_CONTINUATION: '0',
    TRACE_SHARED_KV: cache, TRACE_SHARED_KV_MAX: budgetGiB === 100 ? '90g' : '20g',
    TRACE_EXTRA_SETS: JSON.stringify([
      'device.vram_budget=24g', 'device.vram_reserve=0', 'device.ram_budget=48g',
      'kernels.vulkan.cpu_miss_threads=0',
      `kernels.vulkan.decode_gpu_resident=${item.enabled}`,
      `kernels.vulkan.decode_gpu_resident_layers=${item.layers || 4}`,
      ...(item.sets || []),
    ]),
  };
  if (env.TRACE_COUNTS.split(',').some(count => ![1, 2].includes(Number(count)) || Number(count) > configuredSlots)) {
    throw new Error('Active lane count must fit the configured slots');
  }
  console.log(JSON.stringify({ case: caseTag, cacheBefore: before, exe: caseExe, binaryHash }));
  const child = spawnSync(process.execPath, [path.join(__dirname, 'run-gpu-resident-case.cjs')],
    { cwd: benchRoot, env, windowsHide: true, stdio: 'inherit' });
  if (child.status !== 0) throw new Error(`Runner failed: ${caseTag}`);
  const matrix = JSON.parse(fs.readFileSync(path.join(artifacts, `${caseTag}-matrix.json`)));
  const expected = env.TRACE_COUNTS.split(',').length * env.TRACE_DEPTHS.split(',').length;
  if (matrix.length !== expected || matrix.some(row => row.rounds.length !== Number(env.TRACE_REPEATS))) {
    throw new Error(`Missing cases/rounds: ${caseTag}`);
  }
  if (matrix.some(row => row.status !== 'complete' || ![0, 143].includes(row.exitInfo?.code))) {
    throw new Error(`Incomplete/failed run: ${caseTag}`);
  }
  const reference = manifest.results.find(row => row.name === cases.find(value => !value.enabled)?.name);
  const correctnessMatches = item.enabled && reference ? verifyMatchedOutputs(matrix, reference.matrix) : [];
  const after = cacheBytes();
  if (after > budget) throw new Error(`Aggregate test cache exceeded budget: ${after}`);
  manifest.results.push({ name: item.name, exe: caseExe, binaryHash, binarySourceManifest,
    cacheBefore: before, cacheAfter: after, correctnessMatches,
    matrix: matrix.map(row => ({ count: row.count, depth: row.depth,
      serverLogMetrics: row.serverLogMetrics,
      rounds: row.rounds.map(round => round.requests.map(req => ({
        rate: req.timings.predicted_per_second, windows: req.streamWindowRates,
        hash: req.outputHash, cached: req.timings.cached_n, usage: req.usage,
      }))),
    })),
  });
  fs.writeFileSync(path.join(artifacts, `${tag}-manifest.json`), JSON.stringify(manifest, null, 2));
  console.log(JSON.stringify(manifest.results.at(-1)));
}
