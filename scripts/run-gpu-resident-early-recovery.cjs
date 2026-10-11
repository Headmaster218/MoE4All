const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { verifyMatchedOutputs } = require('./gpu-resident-correctness.cjs');
const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const reference = JSON.parse(fs.readFileSync(path.join(artifacts, 'stage4-new-timeline-manifest.json')));
const tag = process.env.RESIDENT_TAG || 'stage4-early-recovery-screen';
if (!process.env.RESIDENT_EXE) throw new Error('RESIDENT_EXE must name a frozen early-recovery build');
if (fs.existsSync(path.join(artifacts, `${tag}-manifest.json`))) throw new Error('Use a fresh tag');
const preset = reference.cases.find(c => c.enabled);
const ordinaryControl = process.env.RESIDENT_CONTROL === 'ordinary';
if (process.env.RESIDENT_CONTROL && !['ordinary', 'resident'].includes(process.env.RESIDENT_CONTROL)) {
  throw new Error('RESIDENT_CONTROL must be ordinary or resident');
}
const cases = [false, true].map(early => ({ ...structuredClone(preset),
  name: early ? 'resident-early-recovery' : 'resident-control',
  sets: [...preset.sets, `kernels.vulkan.decode_gpu_resident_early_recovery=${early}`],
}));
if (ordinaryControl) cases[0] = structuredClone(reference.cases.find(c => !c.enabled));
const counts = process.env.RESIDENT_COUNTS || '1';
const tokens = process.env.RESIDENT_TOKENS || '384';
const child = spawnSync(process.execPath, [path.join(__dirname, 'gpu-resident-decode-benchmark.cjs')], {
  cwd: root, windowsHide: true, stdio: 'inherit', env: {
    ...process.env, RESIDENT_MODEL: reference.model, RESIDENT_TAG: tag, RESIDENT_SLOTS: '2',
    RESIDENT_COUNTS: counts, RESIDENT_DEPTHS: process.env.RESIDENT_DEPTHS || '30000,150000',
    RESIDENT_REPEATS: process.env.RESIDENT_REPEATS || '2', RESIDENT_TOKENS: tokens,
    RESIDENT_TRACE: '0', RESIDENT_CASES: JSON.stringify(cases),
  },
});
if (child.status !== 0) throw new Error(`Early recovery benchmark failed: ${child.status}`);
const manifest = JSON.parse(fs.readFileSync(path.join(artifacts, `${tag}-manifest.json`)));
const control = manifest.results.find(c => c.name === cases[0].name);
const candidate = manifest.results.find(c => c.name === 'resident-early-recovery');
const matches = verifyMatchedOutputs(candidate.matrix, control.matrix);
const referenceMatches = counts === '1' && tokens === '384'
  ? verifyMatchedOutputs(candidate.matrix, reference.results.find(c => c.name === 'baseline').matrix) : [];
fs.writeFileSync(path.join(artifacts, `${tag}-correctness.json`), JSON.stringify({ matches, referenceMatches }, null, 2));
console.log(JSON.stringify({ matches, referenceMatches }));
