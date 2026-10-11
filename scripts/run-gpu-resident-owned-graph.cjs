const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { verifyAvailableMatchedRounds } = require('./gpu-resident-matched-rounds.cjs');

const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const reference = JSON.parse(fs.readFileSync(path.join(artifacts, 'stage4-recovery-frontend-ordinary-matrix-manifest.json')));
const tag = process.env.RESIDENT_TAG || 'stage4-owned-graph-matrix';
if (!process.env.RESIDENT_EXE) throw new Error('RESIDENT_EXE must name a frozen owned-graph build');
const verifyOnly = process.env.RESIDENT_VERIFY_ONLY === '1';
if (!verifyOnly && fs.existsSync(path.join(artifacts, `${tag}-manifest.json`))) throw new Error('Use a fresh tag');
const preset = reference.cases.find(c => c.enabled);
const cases = [false, true].map(owned => ({ ...structuredClone(preset),
  name: owned ? 'resident-owned-graph' : 'resident-cloned-graph',
  sets: [...preset.sets, 'kernels.vulkan.decode_gpu_resident_full_graph=false',
    `kernels.vulkan.decode_owned_graph=${owned}`],
}));
if (process.env.RESIDENT_REVERSE === '1') cases.reverse();
if (!verifyOnly) {
const child = spawnSync(process.execPath, [path.join(__dirname, 'gpu-resident-decode-benchmark.cjs')], {
  cwd: root, windowsHide: true, stdio: 'inherit', env: {
    ...process.env, RESIDENT_MODEL: reference.model, RESIDENT_TAG: tag, RESIDENT_SLOTS: '2',
    RESIDENT_COUNTS: process.env.RESIDENT_COUNTS || '1,2',
    RESIDENT_DEPTHS: process.env.RESIDENT_DEPTHS || '30000,150000',
    RESIDENT_REPEATS: process.env.RESIDENT_REPEATS || '3', RESIDENT_TOKENS: '512',
    RESIDENT_TRACE: '0', TRACE_PROFILE_STAGES: '0',
    TRACE_PROFILE_PAGER: process.env.RESIDENT_PLE_PROFILE === '1' ? '1' : '0',
    RESIDENT_CASES: JSON.stringify(cases),
  },
});
if (child.status !== 0) throw new Error(`Owned graph benchmark failed: ${child.status}`);
}
const manifest = JSON.parse(fs.readFileSync(path.join(artifacts, `${tag}-manifest.json`)));
const control = manifest.results.find(c => c.name === 'resident-cloned-graph');
const candidate = manifest.results.find(c => c.name === 'resident-owned-graph');
if (!control || !candidate || control.binaryHash !== candidate.binaryHash || manifest.configuredSlots !== 2) {
  throw new Error('Unexpected matrix provenance');
}
const comparison = verifyAvailableMatchedRounds(candidate.matrix, control.matrix);
const golden = verifyAvailableMatchedRounds(candidate.matrix, reference.results.find(c => c.name === 'baseline').matrix);
const controlGolden = verifyAvailableMatchedRounds(control.matrix, reference.results.find(c => c.name === 'baseline').matrix);
const report = { comparison, golden, controlGolden };
fs.writeFileSync(path.join(artifacts, `${tag}-correctness.json`), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
