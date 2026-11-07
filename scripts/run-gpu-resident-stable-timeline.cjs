const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const { verifyMatchedOutputs } = require('./gpu-resident-correctness.cjs');

const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const sourceManifest = path.join(artifacts, 'stage4-recovery-frontend-ordinary-matrix-manifest.json');
const reference = JSON.parse(fs.readFileSync(sourceManifest, 'utf8'));
const tag = process.env.RESIDENT_TAG || 'stage4-stable-current-gpu-trace';
const verifyOnly = process.env.RESIDENT_VERIFY_ONLY === '1';
if (!verifyOnly && fs.existsSync(path.join(artifacts, `${tag}-manifest.json`))) throw new Error('Use a fresh tag');
const binaryHash = crypto.createHash('sha256').update(fs.readFileSync(reference.exe)).digest('hex');
if (binaryHash !== reference.binaryHash) throw new Error('Stable binary has changed');
const preset = structuredClone(reference.cases.find(c => c.enabled));
if (!preset || !reference.results.find(c => c.name === preset.name)) throw new Error('Missing validated stable preset');
preset.name = 'resident-stable';
preset.binarySourceManifest = sourceManifest;
preset.sets.push('prof.timeline_gpu_only=true');
if (!verifyOnly) {
const child = spawnSync(process.execPath, [path.join(__dirname, 'gpu-resident-decode-benchmark.cjs')], {
  cwd: root, windowsHide: true, stdio: 'inherit', env: {
    ...process.env, RESIDENT_MODEL: reference.model, RESIDENT_EXE: reference.exe,
    RESIDENT_TAG: tag, RESIDENT_SLOTS: '2', RESIDENT_COUNTS: '1',
    RESIDENT_DEPTHS: '30000,150000', RESIDENT_REPEATS: '2', RESIDENT_TOKENS: '384',
    RESIDENT_TRACE: '1', RESIDENT_CASES: JSON.stringify([preset]),
    TRACE_PROFILE_STAGES: '0', TRACE_PROFILE_PAGER: '0',
    TRACE_SKIP: '512', TRACE_STEPS: '12', TRACE_STRIDE: '128', TRACE_WINDOWS: '1',
  },
});
if (child.status !== 0) throw new Error(`Stable timeline capture failed: ${child.status}`);
}
const measured = JSON.parse(fs.readFileSync(path.join(artifacts, `${tag}-manifest.json`), 'utf8'));
const golden = JSON.parse(fs.readFileSync(path.join(artifacts, 'stage4-new-timeline-manifest.json'), 'utf8'));
if (measured.results.length !== 1 || measured.results[0].binaryHash !== binaryHash ||
    measured.configuredSlots !== 2 || measured.results[0].name !== preset.name) throw new Error('Unexpected capture provenance');
const control = golden.results.find(c => c.name === 'baseline').matrix;
const excluded = [], matches = [];
for (const row of measured.results[0].matrix) {
  const baseline = control.find(c => c.count === row.count && c.depth === row.depth);
  if (!baseline || row.count !== 1 || row.rounds.length !== 2) throw new Error('Unexpected capture matrix');
  const sameState = (a, b) => JSON.stringify(a.map(r => [r.cached, r.usage.prompt_tokens, r.usage.completion_tokens])) ===
    JSON.stringify(b.map(r => [r.cached, r.usage.prompt_tokens, r.usage.completion_tokens]));
  for (const [index, requests] of row.rounds.entries()) {
    if (!baseline.rounds.some(b => sameState(requests, b))) {
      if (index !== 0 || requests.some(r => r.cached !== 0 || r.usage.completion_tokens !== 384)) {
        throw new Error('Unexpected unmatched cache state');
      }
      excluded.push({ count: row.count, depth: row.depth, round: index,
        cached: 0, hash: requests[0].hash, reason: 'Fresh Prefill has no same-cache-state golden; not output-validated' });
      continue;
    }
    matches.push(...verifyMatchedOutputs([{ ...row, rounds: [requests] }], [baseline])
      .map(m => ({ ...m, round: index })));
  }
  if (!matches.some(m => m.depth === row.depth && m.round === 1)) throw new Error('Traced second round is not validated');
}
if (measured.results[0].matrix.length !== 2) throw new Error('Expected 30K and 150K');
fs.writeFileSync(path.join(artifacts, `${tag}-correctness.json`), JSON.stringify({ matches, excluded }, null, 2));
fs.copyFileSync(__filename, path.join(artifacts, `${tag}-run-gpu-resident-stable-timeline.cjs`));
console.log(JSON.stringify({ matches, excluded }));
