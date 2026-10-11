const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { verifyAvailableMatchedRounds } = require('./gpu-resident-matched-rounds.cjs');

const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const reference = JSON.parse(fs.readFileSync(path.join(artifacts, 'stage4-recovery-frontend-ordinary-matrix-manifest.json')));
// Historical runner: the poll implementation was removed after screening. Retain its build record.
const binarySourceManifest = path.join(artifacts, 'stage4-notice-poll-screen-manifest.json');
if (!fs.existsSync(binarySourceManifest)) throw new Error('Missing preserved notice-poll build record');
const tag = process.env.RESIDENT_TAG || 'stage4-notice-poll-screen';
const intervals = (process.env.RESIDENT_FENCE_INTERVALS || '1,8,16').split(',').map(Number);
if (!intervals.includes(1) || new Set(intervals).size !== intervals.length ||
    intervals.some(n => !Number.isInteger(n) || n < 1 || n > 64)) {
  throw new Error('Fence intervals must be unique values in 1..64, including control 1');
}
if (!process.env.RESIDENT_EXE) throw new Error('RESIDENT_EXE must name a frozen notice-poll build');
const verifyOnly = process.env.RESIDENT_VERIFY_ONLY === '1';
if (!verifyOnly && fs.existsSync(path.join(artifacts, `${tag}-manifest.json`))) throw new Error('Use a fresh tag');
const preset = reference.cases.find(c => c.enabled);
const cases = intervals.map(interval => ({ ...structuredClone(preset),
  name: `resident-fence-${interval}`,
  binarySourceManifest,
  sets: [...preset.sets, 'kernels.vulkan.decode_gpu_resident_full_graph=false',
    'kernels.vulkan.decode_owned_graph=false', 'kernels.vulkan.scratch_io_inline=false',
    `kernels.vulkan.decode_gpu_resident_notice_fence_interval=${interval}`],
}));
if (process.env.RESIDENT_REVERSE === '1') cases.reverse();
if (!verifyOnly) {
  const child = spawnSync(process.execPath, [path.join(__dirname, 'gpu-resident-decode-benchmark.cjs')], {
    cwd: root, windowsHide: true, stdio: 'inherit', env: {
      ...process.env, RESIDENT_MODEL: reference.model, RESIDENT_TAG: tag, RESIDENT_SLOTS: '2',
      RESIDENT_COUNTS: process.env.RESIDENT_COUNTS || '1',
      RESIDENT_DEPTHS: process.env.RESIDENT_DEPTHS || '30000',
      RESIDENT_REPEATS: process.env.RESIDENT_REPEATS || '3', RESIDENT_TOKENS: '512',
      RESIDENT_TRACE: '0', TRACE_PROFILE_STAGES: '0', TRACE_PROFILE_PAGER: '0',
      RESIDENT_CASES: JSON.stringify(cases),
    },
  });
  if (child.status !== 0) throw new Error(`Notice-poll benchmark failed: ${child.status}`);
}
const manifest = JSON.parse(fs.readFileSync(path.join(artifacts, `${tag}-manifest.json`)));
const control = manifest.results.find(c => c.name === 'resident-fence-1');
if (!control || manifest.configuredSlots !== 2 || manifest.results.length !== intervals.length ||
    manifest.results.some(c => c.binaryHash !== control.binaryHash)) {
  throw new Error('Unexpected matrix provenance');
}
const report = manifest.results.map(candidate => ({ name: candidate.name,
  comparison: verifyAvailableMatchedRounds(candidate.matrix, control.matrix),
  golden: verifyAvailableMatchedRounds(candidate.matrix, reference.results.find(c => c.name === 'baseline').matrix),
}));
fs.writeFileSync(path.join(artifacts, `${tag}-correctness.json`), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
