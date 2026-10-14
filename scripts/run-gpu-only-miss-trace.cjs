const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { verifyMatchedOutputs } = require('./gpu-resident-correctness.cjs');
const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const reference = JSON.parse(fs.readFileSync(path.join(artifacts, 'stage4-new-timeline-manifest.json')));
const tag = process.env.RESIDENT_TAG || 'stage4-gpu-only-miss-trace';
if (!process.env.RESIDENT_EXE) throw new Error('RESIDENT_EXE must name the frozen GPU-only build');
if (fs.existsSync(path.join(artifacts, `${tag}-manifest.json`))) throw new Error('Use a fresh tag');
const selected = structuredClone(reference.cases.find(c => c.enabled));
selected.sets.push('prof.timeline_gpu_only=true');
if (process.env.RESIDENT_EARLY_RECOVERY === '1') {
  selected.sets.push('kernels.vulkan.decode_gpu_resident_early_recovery=true');
}
const child = spawnSync(process.execPath, [path.join(__dirname, 'gpu-resident-decode-benchmark.cjs')], {
  cwd: root, windowsHide: true, stdio: 'inherit', env: {
    ...process.env, RESIDENT_MODEL: reference.model, RESIDENT_TAG: tag,
    RESIDENT_SLOTS: '2', RESIDENT_COUNTS: '1', RESIDENT_DEPTHS: '30000,150000',
    RESIDENT_REPEATS: '2', RESIDENT_TOKENS: '384', RESIDENT_TRACE: '1',
    RESIDENT_CASES: JSON.stringify([selected]), TRACE_SKIP: '512', TRACE_STEPS: '12',
    TRACE_STRIDE: '128', TRACE_WINDOWS: '1',
  },
});
if (child.status !== 0) throw new Error(`GPU trace runner failed: ${child.status}`);
const manifest = JSON.parse(fs.readFileSync(path.join(artifacts, `${tag}-manifest.json`)));
const matches = verifyMatchedOutputs(manifest.results[0].matrix, reference.results.find(r => r.name === 'baseline').matrix);
fs.writeFileSync(path.join(artifacts, `${tag}-correctness.json`), JSON.stringify({ matches }, null, 2));
console.log(JSON.stringify({ matches }));
