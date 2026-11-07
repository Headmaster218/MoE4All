const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { verifyMatchedOutputs } = require('./gpu-resident-correctness.cjs');

const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const reference = JSON.parse(fs.readFileSync(path.join(artifacts, 'stage4-recovery-frontend-ordinary-matrix-manifest.json')));
const tag = process.env.RESIDENT_TAG || 'stage4-full-graph-screen';
if (!process.env.RESIDENT_EXE) throw new Error('RESIDENT_EXE must name a frozen full-graph build');
if (fs.existsSync(path.join(artifacts, `${tag}-manifest.json`))) throw new Error('Use a fresh tag');
const controlKind = process.env.RESIDENT_CONTROL || 'resident';
if (!['resident', 'ordinary'].includes(controlKind)) throw new Error('RESIDENT_CONTROL must be resident or ordinary');
const preset = reference.cases.find(c => c.enabled);
const trace = process.env.RESIDENT_TRACE || '0';
const pleProfile = process.env.RESIDENT_PLE_PROFILE === '1';
const cases = [false, true].map(full => ({ ...structuredClone(preset),
  name: full ? 'resident-full-graph' : 'resident-split',
  sets: [...preset.sets, `kernels.vulkan.decode_gpu_resident_full_graph=${full}`,
    ...(trace === '1' ? ['prof.timeline_gpu_only=true'] : [])],
}));
if (controlKind === 'ordinary') {
  cases[0] = structuredClone(reference.cases.find(c => !c.enabled));
  if (trace === '1') cases[0].sets = [...(cases[0].sets || []), 'prof.timeline_gpu_only=true'];
}
const counts = process.env.RESIDENT_COUNTS || '1';
const tokens = process.env.RESIDENT_TOKENS || '384';
const child = spawnSync(process.execPath, [path.join(__dirname, 'gpu-resident-decode-benchmark.cjs')], {
  cwd: root, windowsHide: true, stdio: 'inherit', env: {
    ...process.env, RESIDENT_MODEL: reference.model, RESIDENT_TAG: tag, RESIDENT_SLOTS: '2',
    RESIDENT_COUNTS: counts, RESIDENT_DEPTHS: process.env.RESIDENT_DEPTHS || '30000,150000',
    RESIDENT_REPEATS: process.env.RESIDENT_REPEATS || '2', RESIDENT_TOKENS: tokens,
    RESIDENT_TRACE: trace, RESIDENT_CASES: JSON.stringify(cases),
    TRACE_PROFILE_STAGES: '0', TRACE_PROFILE_PAGER: pleProfile ? '1' : '0',
  },
});
if (child.status !== 0) throw new Error(`Full-graph benchmark failed: ${child.status}`);
const manifest = JSON.parse(fs.readFileSync(path.join(artifacts, `${tag}-manifest.json`)));
const control = manifest.results.find(c => c.name === cases[0].name);
const candidate = manifest.results.find(c => c.name === cases[1].name);
const matches = verifyMatchedOutputs(candidate.matrix, control.matrix);
let referenceMatches = [];
if (tokens === '512') {
  referenceMatches = verifyMatchedOutputs(candidate.matrix, reference.results.find(c => c.name === 'baseline').matrix);
} else if (counts === '1' && tokens === '384') {
  const golden = JSON.parse(fs.readFileSync(path.join(artifacts, 'stage4-new-timeline-manifest.json')));
  referenceMatches = verifyMatchedOutputs(candidate.matrix, golden.results.find(c => c.name === 'baseline').matrix);
}
fs.writeFileSync(path.join(artifacts, `${tag}-correctness.json`), JSON.stringify({ matches, referenceMatches }, null, 2));
console.log(JSON.stringify({ matches, referenceMatches }));
