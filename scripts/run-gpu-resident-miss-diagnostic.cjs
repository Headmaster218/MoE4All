const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const { verifyAvailableMatchedRounds } = require('./gpu-resident-matched-rounds.cjs');

const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const referencePath = path.join(artifacts, 'stage4-mapped-copy-matrix-manifest.json');
const reference = JSON.parse(fs.readFileSync(referencePath));
const control = reference.results.find(c => c.name === 'mapped-copy-0');
if (!control || control.matrix.length !== 4) throw new Error('Completed control matrix required');
const preset = structuredClone(reference.cases.find(c => c.name === control.name));
if (!preset || !preset.enabled || reference.configuredSlots !== 2) throw new Error('Unexpected control preset');
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const verifyOnly = process.env.RESIDENT_VERIFY_ONLY === '1';
const buildPath = process.env.RESIDENT_DIAGNOSTIC_BUILD || referencePath;
const build = process.env.RESIDENT_DIAGNOSTIC_BUILD ? JSON.parse(fs.readFileSync(buildPath)) : reference;
const exe = process.env.RESIDENT_DIAGNOSTIC_BUILD ? build.exe : control.exe;
const binaryHash = process.env.RESIDENT_DIAGNOSTIC_BUILD ? build.binaryHash : control.binaryHash;
if (digest(fs.readFileSync(exe)) !== binaryHash) throw new Error('Frozen binary changed');
// A frozen build record, not the current documentation/tooling HEAD, owns these results.
for (const [file, expected] of Object.entries(build.sourceHashes)) {
  if (!verifyOnly && file.startsWith('crates/') && digest(fs.readFileSync(path.join(root, file))) !== expected) {
    throw new Error(`Frozen runtime source differs: ${file}`);
  }
}
const tag = process.env.RESIDENT_TAG || 'stage4-miss-boundary-diagnostic';
if (!verifyOnly && fs.existsSync(path.join(artifacts, `${tag}-manifest.json`))) throw new Error('Use a fresh tag');
preset.name = 'diagnostic-control';
preset.binarySourceManifest = buildPath;
preset.sets.push('prof.timeline_gpu_only=false');
if (process.env.RESIDENT_DIAGNOSTIC_BUILD) preset.sets.push('kernels.vulkan.decode_gpu_resident_mapped_staging=false');
if (!verifyOnly) {
  const child = spawnSync(process.execPath, [path.join(__dirname, 'gpu-resident-decode-benchmark.cjs')], {
    cwd: root, windowsHide: true, stdio: 'inherit', env: {
      ...process.env, RESIDENT_MODEL: reference.model, RESIDENT_EXE: exe,
      RESIDENT_TAG: tag, RESIDENT_SLOTS: '2', RESIDENT_COUNTS: '1',
      RESIDENT_DEPTHS: '30000,150000', RESIDENT_REPEATS: '2', RESIDENT_TOKENS: '512',
      RESIDENT_TRACE: '1', RESIDENT_CASES: JSON.stringify([preset]),
      TRACE_PROFILE_STAGES: '0', TRACE_PROFILE_PAGER: '0',
      TRACE_SKIP: '640', TRACE_STEPS: '12', TRACE_STRIDE: '128', TRACE_WINDOWS: '1',
    },
  });
  if (child.status !== 0) throw new Error(`Diagnostic capture failed: ${child.status}`);
}
const measured = JSON.parse(fs.readFileSync(path.join(artifacts, `${tag}-manifest.json`)));
if (measured.results.length !== 1 || measured.results[0].binaryHash !== binaryHash ||
    measured.results[0].name !== preset.name || measured.configuredSlots !== 2) {
  throw new Error('Unexpected diagnostic provenance');
}
const comparison = verifyAvailableMatchedRounds(measured.results[0].matrix, control.matrix);
for (const depth of [30000, 150000]) {
  if (!comparison.matches.some(m => m.depth === depth && m.count === 1 && m.round === 1)) {
    throw new Error('Traced warm round not output-validated');
  }
}
fs.writeFileSync(path.join(artifacts, `${tag}-correctness.json`), JSON.stringify(comparison, null, 2));
fs.copyFileSync(__filename, path.join(artifacts, `${tag}-run-gpu-resident-miss-diagnostic.cjs`));
console.log(JSON.stringify(comparison));
