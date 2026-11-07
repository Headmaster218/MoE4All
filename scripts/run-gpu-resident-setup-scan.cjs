const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { verifyMatchedOutputs } = require('./gpu-resident-correctness.cjs');

const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const reference = JSON.parse(fs.readFileSync(path.join(artifacts, 'stage4-recovery-frontend-ordinary-matrix-manifest.json')));
const controlManifest = path.join(artifacts, 'stage4-full-graph-resident-matrix-manifest.json');
const controlBuild = JSON.parse(fs.readFileSync(controlManifest));
const tag = process.env.RESIDENT_TAG || 'stage4-shared-scan-matrix';
if (!process.env.RESIDENT_EXE) throw new Error('RESIDENT_EXE must name the frozen filtered-scan build');
if (fs.existsSync(path.join(artifacts, `${tag}-manifest.json`))) throw new Error('Use a fresh tag');
const preset = reference.cases.find(c => c.enabled);
const cases = [
  { ...structuredClone(preset), name: 'resident-shared-scan-legacy',
    exe: controlBuild.exe, binarySourceManifest: controlManifest },
  { ...structuredClone(preset), name: 'resident-shared-scan-filtered' },
].map(c => ({ ...c, sets: [...c.sets, 'kernels.vulkan.decode_gpu_resident_full_graph=false'] }));
if (process.env.RESIDENT_REVERSE === '1') cases.reverse();
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
if (child.status !== 0) throw new Error(`Shared-scan benchmark failed: ${child.status}`);
const manifest = JSON.parse(fs.readFileSync(path.join(artifacts, `${tag}-manifest.json`)));
const control = manifest.results.find(c => c.name === 'resident-shared-scan-legacy');
const candidate = manifest.results.find(c => c.name === 'resident-shared-scan-filtered');
const matches = verifyMatchedOutputs(candidate.matrix, control.matrix);
const referenceMatches = verifyMatchedOutputs(candidate.matrix, reference.results.find(c => c.name === 'baseline').matrix);
fs.writeFileSync(path.join(artifacts, `${tag}-correctness.json`), JSON.stringify({ matches, referenceMatches }, null, 2));
console.log(JSON.stringify({ matches, referenceMatches }));
