const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

const artifacts = path.resolve(__dirname, '../benchmark-data/artifacts/gpu-resident-decode-20261011');
const read = file => JSON.parse(fs.readFileSync(file));
const digest = records => crypto.createHash('sha256').update(JSON.stringify(
  records.toSorted((a, b) => a.kind - b.kind || a.layer - b.layer))).digest('hex');
const tags = ['stage4-owned-graph-cache-order-forward', 'stage4-owned-graph-cache-order-reverse'];
const golden = read(path.join(artifacts, 'stage4-recovery-frontend-ordinary-matrix-manifest.json'));
const expected = golden.results.find(c => c.name === 'baseline').matrix
  .find(row => row.count === 2 && row.depth === 30000).rounds
  .find(batch => batch.every(r => r.cached > 0));
assert.ok(expected, 'Missing complete warmed golden');
let binaryHash, promptHashes;
const cohorts = [];
for (const [pass, tag] of tags.entries()) {
  const manifest = read(path.join(artifacts, `${tag}-manifest.json`));
  const verification = read(path.join(artifacts, `${tag}-correctness.json`));
  assert.equal(manifest.configuredSlots, 2);
  assert.equal(manifest.results.length, 2);
  for (const gate of ['comparison', 'golden', 'controlGolden']) {
    assert.equal(verification[gate].matches.length, 2);
    assert.deepEqual(verification[gate].excluded, []);
  }
  for (const c of manifest.results) {
    const dir = path.join(artifacts, `${tag}-${c.name}-c2-d30000`);
    const result = read(path.join(dir, 'result.json'));
    const before = read(path.join(dir, 'cache-identity-before.json'));
    const after = read(path.join(dir, 'cache-identity-after.json'));
    binaryHash ??= c.binaryHash; promptHashes ??= result.promptHashes;
    assert.equal(c.binaryHash, binaryHash);
    assert.equal(result.binaryHash, binaryHash);
    assert.equal(result.status, 'complete');
    assert.equal(result.traceEnabled, false);
    assert.deepEqual(result.requestOrder, pass === 0 ? [0, 1] : [1, 0]);
    assert.deepEqual(result.promptHashes, promptHashes);
    assert.equal(result.rounds.length, 2);
    for (const snapshot of [before, after]) {
      assert.ok(snapshot.aggregateBytes < 30 * 1024 ** 3);
      assert.equal(snapshot.hashCheckpointRecords, true);
    }
    const checkpoints = [29945, 29955].map(count => {
      const snapshots = [before, after].map(s => s.files.filter(f =>
        f.checkpoints?.some(cp => cp.count === count)));
      assert.ok(snapshots.every(files => files.length > 0));
      const all = snapshots.flat();
      assert.ok(all.every(f => f.checkpointRecords?.length > 0 && f.payloadChecksumVerified === false));
      const states = [...new Set(all.map(f => digest(f.checkpointRecords)))];
      assert.equal(states.length, 1, 'Checkpoint state changed across candidates/snapshots');
      return { count, recordSetHash: states[0], beforeCandidates: snapshots[0].length,
        afterCandidates: snapshots[1].length };
    });
    for (const round of result.rounds) for (const request of round.requests) {
      const ref = expected[request.lane];
      assert.equal(request.timings.cached_n, ref.cached);
      assert.equal(request.usage.prompt_tokens, ref.usage.prompt_tokens);
      assert.equal(request.usage.completion_tokens, ref.usage.completion_tokens);
      assert.equal(request.outputHash, ref.hash);
    }
    cohorts.push({ tag, name: c.name, requestOrder: result.requestOrder, checkpoints,
      cacheBefore: before.aggregateBytes, cacheAfter: after.aggregateBytes,
      ratesPerLane: result.rounds.map(r => r.requests.reduce((s, q) => s + q.timings.predicted_per_second, 0) / 2),
      hashes: result.rounds.map(r => r.requests.map(q => q.outputHash)) });
  }
}
for (const count of [29945, 29955]) {
  assert.equal(new Set(cohorts.map(c => c.checkpoints.find(cp => cp.count === count).recordSetHash)).size, 1);
}
const comparisons = tags.map(tag => {
  const control = cohorts.find(c => c.tag === tag && c.name === 'resident-cloned-graph');
  const candidate = cohorts.find(c => c.tag === tag && c.name === 'resident-owned-graph');
  return { tag, warmedControl: control.ratesPerLane[1], warmedCandidate: candidate.ratesPerLane[1],
    changePercent: (candidate.ratesPerLane[1] / control.ratesPerLane[1] - 1) * 100 };
});
const report = { binaryHash, promptHashes, comparisons, cohorts,
  matchedBatches: 8, matchedRequests: 16,
  conclusion: 'Original anomalous output did not recur in these two orders; original correctness finding remains unresolved.',
  limits: 'Checkpoint record hashes are compared, not the entire main KV prefix or the exact GPU restored state. Snapshot hashing occurs outside Decode and affects filesystem cache residency. One warmed round per order is not a performance acceptance gate.' };
fs.writeFileSync(path.join(artifacts, 'stage4-owned-graph-cache-order-analysis.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
