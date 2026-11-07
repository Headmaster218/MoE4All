const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const moduleLog = path.join(artifacts, 'stage4-mapped-copy-module.log');
const text = fs.readFileSync(moduleLog, 'utf8');
const samples = [...text.matchAll(/mapped_copy experts=(\d+) bytes=(\d+) pass=(\d+) serial=(true|false) us=([\d.]+)/g)]
  .map(([, experts, bytes, pass, serial, us]) => ({ experts: +experts, bytes: +bytes, pass: +pass, serial: serial === 'true', us: +us }));
if (samples.length !== 60 || !text.includes('test result: ok. 1 passed; 0 failed;')) throw new Error('Incomplete module measurement');
const moduleStats = [1, 2, 3, 4, 8].map(experts => {
  const subset = samples.filter(s => s.experts === experts);
  const rows = [false, true].map(serial => {
    const values = subset.filter(s => s.serial === serial);
    if (values.length !== 6 || new Set(values.map(s => s.pass)).size !== 6 || values.some(s => !Number.isFinite(s.us) || s.us <= 0)) {
      throw new Error('Malformed module passes');
    }
    return values.reduce((sum, s) => sum + s.us, 0) / values.length;
  });
  return { experts, bytes: subset[0].bytes, parallelUs: rows[0], serialUs: rows[1], changePercent: (rows[1] / rows[0] - 1) * 100 };
});
const matrices = process.argv.slice(2).map(tag => {
  const manifestPath = path.join(artifacts, `${tag}-manifest.json`);
  const manifest = JSON.parse(fs.readFileSync(manifestPath));
  const control = manifest.results.find(r => r.name === 'mapped-copy-0');
  const candidate = manifest.results.find(r => r.name !== 'mapped-copy-0');
  if (!control || !candidate || manifest.results.length !== 2 || control.binaryHash !== candidate.binaryHash) throw new Error('Unmatched matrix');
  const correctnessPath = path.join(artifacts, `${tag}-correctness.json`);
  const correctness = JSON.parse(fs.readFileSync(correctnessPath));
  const mean = row => row.rounds.slice(1).flat().reduce((sum, r) => sum + r.rate, 0) / row.rounds.slice(1).flat().length;
  return { tag, binaryHash: control.binaryHash, correctness,
    rates: candidate.matrix.map(row => {
      const baseline = control.matrix.find(b => b.depth === row.depth && b.count === row.count);
      if (!baseline || row.rounds.length < 3 || baseline.rounds.length !== row.rounds.length) throw new Error('Missing warmed rounds');
      const before = mean(baseline), after = mean(row);
      return { depth: row.depth, active: row.count, control: before, candidate: after, changePercent: (after / before - 1) * 100,
        controlWarm: baseline.rounds.slice(1).map(r => r.map(x => x.rate)), candidateWarm: row.rounds.slice(1).map(r => r.map(x => x.rate)) };
    }),
  };
});
const report = { moduleLog, moduleLogHash: crypto.createHash('sha256').update(fs.readFileSync(moduleLog)).digest('hex'),
  moduleTiming: 'Six alternating passes; host copy preparation to coherent mapped VRAM; verification outside timing', module: moduleStats, matrices };
fs.writeFileSync(path.join(artifacts, 'stage4-mapped-copy-summary.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
