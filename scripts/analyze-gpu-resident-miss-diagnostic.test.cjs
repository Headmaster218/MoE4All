const test = require('node:test');
const assert = require('node:assert/strict');
const { splitReadyGap, splitSubmitCallGap, phaseDuration, breakdown } = require('./analyze-gpu-resident-miss-diagnostic.cjs');

test('worker phase duration clips intervals and does not double-count overlaps', () => {
  const event = (name, start_ns, end_ns) => ({ name, start_ns, end_ns });
  const events = [event('copy', 0, 50000), event('copy', 40000, 80000), event('lookup', 0, 90000)];
  assert.equal(phaseDuration(events, 'copy', 10000, 70000), 60);
  assert.equal(phaseDuration(events, 'missing', 0, 100000), null);
  assert.equal(phaseDuration(events, 'copy', 90000, 100000), null);
  assert.equal(phaseDuration([event('lookup', 10000, 10000)], 'lookup', 0, 100000), 0);
  assert.equal(phaseDuration([event('lookup', 10000, 10000)], 'lookup', 20000, 100000), null);
});

test('submit stamp is before API return and GPU may start before return', () => {
  const call = (start_ns, end_ns) => ({ start_ns, end_ns });
  assert.deepEqual(splitSubmitCallGap(100000, 121000, 170000, [call(120000, 140000)]),
    { beforeCallUs: 20, callOverlapUs: 20, afterCallReturnUs: 30, gpuStartedBeforeApiReturn: false });
  assert.deepEqual(splitSubmitCallGap(100000, 60000, 150000, [call(59000, 120000)]),
    { beforeCallUs: 0, callOverlapUs: 20, afterCallReturnUs: 30, gpuStartedBeforeApiReturn: false });
  assert.deepEqual(splitSubmitCallGap(100000, 60000, 150000, [call(59000, 180000)]),
    { beforeCallUs: 0, callOverlapUs: 50, afterCallReturnUs: 0, gpuStartedBeforeApiReturn: true });
  assert.throws(() => splitSubmitCallGap(100000, 60000, 150000, []), /one main/);
});

test('ready gap distinguishes late CPU submit from an already queued GPU dependency', () => {
  assert.deepEqual(splitReadyGap(100000, 120000, 170000),
    { totalUs: 70, beforeSubmitUs: 20, queuedUs: 50, submitLeadUs: 0 });
  assert.deepEqual(splitReadyGap(100000, 60000, 150000),
    { totalUs: 50, beforeSubmitUs: 0, queuedUs: 50, submitLeadUs: 40 });
  assert.deepEqual(splitReadyGap(100000, 100000, 100000),
    { totalUs: 0, beforeSubmitUs: 0, queuedUs: 0, submitLeadUs: 0 });
  assert.throws(() => splitReadyGap(100000, 60000, 99999), /readiness/);
  assert.throws(() => splitReadyGap(100000, 150000, 149999), /readiness/);
});

test('CPU partition clips outer scopes and excludes producer work from the main track', () => {
  const event = (name, start_ns, end_ns) => ({ name, start_ns, end_ns });
  const result = breakdown([
    event('step_front', 0, 100000), event('command_encode', 10000, 50000),
    event('trace_query_acquire', 20000, 30000), event('queue_submit', 50000, 60000),
    event('resident_producer_admit', 0, 100000),
  ], 10000, 80000);
  assert.deepEqual(result, { command_encode: 30, measurement: 10, submit: 10, scheduler_front_tail: 20 });
  assert.equal(Object.values(result).reduce((a, b) => a + b), 70);
});
