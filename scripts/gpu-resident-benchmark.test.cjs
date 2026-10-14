const test = require('node:test');
const assert = require('node:assert/strict');
const { parseLogMetrics } = require('./gpu-resident-log-metrics.cjs');
const { verifyMatchedOutputs } = require('./gpu-resident-correctness.cjs');
const epoch = Date.parse('2026-10-11T08:00:00+08:00');
const line = (ms, text) => `${new Date(epoch + ms).toISOString()} INFO infr_server: ${text}`;
const progress = (ms, tokens) => line(ms,
  `request progress req=1 phase="decode" gen_tokens=${tokens} elapsed_ms=${ms}`);

test('uses token counters and excludes partial, Prefill and inactive stats windows', () => {
  const text = [line(0, 'request start req=1'), progress(1000, 1), progress(2000, 41),
    progress(3000, 81), progress(4000, 121),
    line(4500, 'request done req=1 gen_tokens=141 total_ms=4500'),
    ...[1000, 2500, 3500, 5000].map(ms => line(ms,
      'serve stats interval_s=1.0 decode_tps=40.0 gen_tokens=40 prompt_tokens=0 active=1 queued=0 completed=0 failed=0')),
    line(3000, 'serve stats interval_s=1.0 decode_tps=100.0 gen_tokens=100 prompt_tokens=31 active=1 queued=0 completed=0 failed=0'),
    line(4000, 'serve stats interval_s=1.0 decode_tps=100.0 gen_tokens=100 prompt_tokens=0 active=0 queued=0 completed=1 failed=0'),
  ].join('\n');
  const [result] = parseLogMetrics(text, [{ startedUnixMs: epoch, finishedUnixMs: epoch + 4600 }], 1);
  assert.equal(result.steadyConcurrent.windows, 2);
  assert.equal(result.steadyConcurrent.aggregateTokensPerSecond, 40);
  assert.equal(result.steadyConcurrent.peakAggregateTokensPerSecond, 40);
  assert.equal(result.perRequest[0].front, 40);
  assert.equal(result.perRequest[0].middle, 40);
  assert.equal(result.perRequest[0].late, 40);
});

test('missing requests or Decode boundaries cannot produce a throughput claim', () => {
  const text = line(0, 'request start req=1');
  const [result] = parseLogMetrics(text, [{ startedUnixMs: epoch, finishedUnixMs: epoch + 1000 }], 2);
  assert.ok(result.unavailable);
  assert.equal(result.steadyConcurrent, undefined);
});

test('concurrent rate is aggregate only within both Decode lifetimes', () => {
  const text = [line(0, 'request start req=1'), line(1, 'request start req=2'),
    progress(1000, 1), progress(3000, 81),
    line(2000, 'request progress req=2 phase="decode" gen_tokens=1 elapsed_ms=1999'),
    line(4500, 'request done req=1 gen_tokens=141 total_ms=4500'),
    line(4501, 'request done req=2 gen_tokens=101 total_ms=4500'),
    ...[2500, 3500].map(ms => line(ms,
      'serve stats interval_s=1.0 decode_tps=80.0 gen_tokens=80 prompt_tokens=0 active=2 queued=0 completed=0 failed=0')),
  ].join('\n');
  const [result] = parseLogMetrics(text, [{ startedUnixMs: epoch, finishedUnixMs: epoch + 4510 }], 2);
  assert.equal(result.steadyConcurrent.windows, 1);
  assert.equal(result.steadyConcurrent.aggregateTokensPerSecond, 80);
  assert.equal(result.steadyConcurrent.peakAggregateTokensPerSecond, 80);
});

test('reports the peak from fully contained Decode windows only', () => {
  const text = [line(0, 'request start req=1'), progress(1000, 1), progress(2000, 41),
    progress(3000, 91), progress(4000, 121),
    line(4500, 'request done req=1 gen_tokens=141 total_ms=4500'),
    line(2500, 'serve stats interval_s=1.0 decode_tps=40.0 gen_tokens=40 prompt_tokens=0 active=1 queued=0 completed=0 failed=0'),
    line(3500, 'serve stats interval_s=1.0 decode_tps=50.0 gen_tokens=50 prompt_tokens=0 active=1 queued=0 completed=0 failed=0'),
    line(5000, 'serve stats interval_s=1.0 decode_tps=90.0 gen_tokens=90 prompt_tokens=0 active=1 queued=0 completed=0 failed=0'),
  ].join('\n');
  const [result] = parseLogMetrics(text, [{ startedUnixMs: epoch, finishedUnixMs: epoch + 4600 }], 1);
  assert.equal(result.steadyConcurrent.peakAggregateTokensPerSecond, 50);
});

test('hash comparisons match the entire batch cache state, not round number', () => {
  const request = (cached, hash) => ({ cached, hash, usage: { prompt_tokens: 100, completion_tokens: 512 } });
  const reference = [{ count: 2, depth: 100, rounds: [
    [request(99, 'cold-a'), request(0, 'cold-b')],
    [request(99, 'warm-a'), request(99, 'warm-b')],
    [request(99, 'other-a'), request(99, 'other-b')],
  ] }];
  const matrix = (a, b, cached = 99) => [{ count: 2, depth: 100,
    rounds: [{ requests: [request(99, a), request(cached, b)] }] }];
  assert.equal(verifyMatchedOutputs(matrix('warm-a', 'warm-b'), reference)[0].referenceRound, 1);
  assert.throws(() => verifyMatchedOutputs(matrix('changed-a', 'warm-b'), reference), /Output mismatch/);
  assert.throws(() => verifyMatchedOutputs(matrix('warm-a', 'other-b'), reference), /Output mismatch/);
  assert.throws(() => verifyMatchedOutputs(matrix('warm-a', 'warm-b', 50), reference), /No matching/);
});
