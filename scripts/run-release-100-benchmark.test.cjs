const test = require('node:test');
const assert = require('node:assert/strict');
const { summarizeDecode, summarizePrefill } = require('./run-release-100-benchmark.cjs');

const request = (rate, hash = 'same') => ({ usage: { completion_tokens: 100 },
  timings: { predicted_ms: 2000, predicted_per_second: rate, cached_n: 90 }, outputHash: hash });
const metric = (front, middle, late, steady, peak) => ({
  perRequest: [{ front, middle, late }, { front, middle, late }],
  steadyConcurrent: { aggregateTokensPerSecond: steady, peakAggregateTokensPerSecond: peak },
});

test('Decode summary excludes warmup and computes aggregate round metrics', () => {
  const row = { status: 'complete', count: 2, depth: 30000, slots: 3,
    rounds: [0, 1, 2, 3].map(round => ({ round, requests: [request(50), request(50)] })),
    serverLogMetrics: [metric(1, 1, 1, 2, 3), metric(40, 50, 60, 90, 100),
      metric(42, 52, 62, 92, 102), metric(44, 54, 64, 94, 104)] };
  const [summary] = summarizeDecode([row]);
  assert.equal(summary.measured.length, 3);
  assert.equal(summary.measured[0].sourceRound, 1);
  assert.equal(summary.measured[0].frontAggregateTokensPerSecond, 80);
  assert.equal(summary.measured[0].averageAggregateTokensPerSecond, 100);
  assert.equal(summary.mean.peakAggregateTokensPerSecond, 102);
  assert.equal(summary.stability.allMeasuredRoundsStable, true);
  assert.deepEqual(summary.stability.unstableLanes, []);
});

test('Decode summary records nondeterministic lanes without discarding measurements', () => {
  const rounds = [0, 1, 2, 3].map(round => ({ round,
    requests: [request(50, round === 2 ? 'different' : 'same')] }));
  const row = { status: 'complete', count: 1, depth: 150000, slots: 3, rounds,
    serverLogMetrics: [metric(1, 1, 1, 2, 3), metric(40, 50, 60, 90, 100),
      metric(42, 52, 62, 92, 102), metric(44, 54, 64, 94, 104)] };
  const [summary] = summarizeDecode([row]);
  assert.equal(summary.stability.allMeasuredRoundsStable, false);
  assert.equal(summary.stability.unstableLanes[0].lane, 0);
  assert.deepEqual(summary.stability.unstableLanes[0].hashes, ['same', 'different', 'same']);
});

test('Prefill summary requires a cold uncached request', () => {
  const base = { status: 'complete', count: 1, depth: 30000,
    rounds: [{ requests: [{ timings: { cached_n: 0, prompt_n: 30000, prompt_ms: 15000,
      prompt_per_second: 2000, predicted_per_second: 40 }, outputHash: 'x' }] }] };
  assert.equal(summarizePrefill([base])[0].promptTokensPerSecond, 2000);
  base.rounds[0].requests[0].timings.cached_n = 1;
  assert.throws(() => summarizePrefill([base]), /unexpectedly reused/);
});
