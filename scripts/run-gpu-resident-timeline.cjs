const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const sourceRoot = path.resolve(__dirname, '..');
const artifacts = path.join(sourceRoot, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const reference = JSON.parse(fs.readFileSync(path.join(artifacts,
  'stage4-final-inflight1-matrix-manifest.json'), 'utf8'));
if (reference.results.length !== 2) throw new Error('Finish the unprofiled matrix before collecting traces');
const tag = process.env.RESIDENT_TAG || 'stage4-new-timeline';
if (fs.existsSync(path.join(artifacts, `${tag}-manifest.json`))) throw new Error('Use a new tag to preserve earlier traces');
const child = spawnSync(process.execPath, [path.join(__dirname, 'gpu-resident-decode-benchmark.cjs')], {
  cwd: sourceRoot, windowsHide: true, stdio: 'inherit', env: {
    ...process.env, RESIDENT_MODEL: reference.model, RESIDENT_EXE: reference.exe,
    RESIDENT_TAG: tag, RESIDENT_SLOTS: '2', RESIDENT_COUNTS: '1',
    RESIDENT_DEPTHS: '30000,150000', RESIDENT_REPEATS: '2', RESIDENT_TOKENS: '384',
    RESIDENT_TRACE: '1', RESIDENT_CASES: JSON.stringify(reference.cases),
    TRACE_SKIP: '512', TRACE_STEPS: '12', TRACE_STRIDE: '128', TRACE_WINDOWS: '1',
  },
});
if (child.status !== 0) throw new Error(`Timeline runner failed: ${child.status}`);
