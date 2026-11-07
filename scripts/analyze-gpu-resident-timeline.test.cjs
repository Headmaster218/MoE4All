const assert = require('node:assert/strict');
const { test } = require('node:test');
process.env.TRACE_INPUT_ROOT = process.env.TRACE_INPUT_ROOT || process.cwd();
const { partition, deviceTime, merge, intersection, length, inferenceEvents } = require('./analyze-gpu-resident-timeline.cjs');
const { gpuLayerRanges } = require('./gpu-resident-timeline-layers.cjs');

test('producer wait is waiting, nested submit takes wait priority, partition closes', () => {
  const result = partition([
    { name: 'gpu_resident_notice_producer_wait', start_ns: 10, end_ns: 60 },
    { name: 'queue_submit', start_ns: 20, end_ns: 30 },
    { name: 'gpu_resident_lut_freeze', start_ns: 70, end_ns: 90 },
  ], 0, 100);
  assert.deepEqual(result, [[0, 10, 'lowering_other'], [10, 60, 'wait'],
    [60, 70, 'lowering_other'], [70, 90, 'residency_snapshot'], [90, 100, 'lowering_other']]);
  assert.equal(result.reduce((n, [a, b]) => n + b - a, 0), 100);
});

test('ACK wait is main-thread waiting; worker envelopes never enter its exclusive partition', () => {
  const events = inferenceEvents([
    { name: 'resident_notice_ack_wait', start_ns: 0, end_ns: 100 },
    { name: 'resident_producer_dma_submit', start_ns: 20, end_ns: 50 },
    { name: 'resident_producer_admit', start_ns: 10, end_ns: 60 },
  ]);
  assert.equal(events.length, 1);
  assert.deepEqual(partition(events, 0, 100), [[0, 100, 'wait']]);
});

test('GPU work and dependency intervals use calibrated time, not host encode', () => {
  const event = deviceTime({ host_submit_ns: 100, start_tick: 25, end_tick: 30,
    period_ns: 2, valid_bits: 64 }, [{ host_ns: 100, device_tick: 20, period_ns: 2, deviation_ns: 1 }]);
  assert.equal(event.start, 110);
  assert.equal(event.end, 120);
  assert.throws(() => deviceTime(event, []), /no calibrated/);
});

test('overlap does not double count adjacent or overlapping main intervals', () => {
  const main = merge([[2, 4], [4, 8], [7, 10]]);
  assert.deepEqual(main, [[2, 10]]);
  assert.equal(length(intersection(main, [[0, 3], [8, 12]])), 3);
});

function layerFixture() {
  const step = { id: 1, start_ns: 0, end_ns: 50000 };
  const host = [], gpu = [{ context: { step: 1, layer: -1 }, queue: 'main',
    name: 'HC_completion_interval', start: 40, end: 50, submit: 0, host_submit_ns: 10 }];
  const layers = Array.from({ length: 48 }, (_, layer) => {
    const base = layer * 1000, context = { step: 1, layer: -1 };
    host.push({ name: 'QwenHcInject', context, start_ns: base + 900, end_ns: base + 910 },
      { name: 'recorder_lifetime', context, start_ns: base + 880, end_ns: base + 920 },
      { name: 'queue_submit', context, start_ns: base + 930, end_ns: base + 940 });
    gpu.push({ name: 'HC_completion_interval', context, queue: 'main',
      start: base + 960, end: base + 975, host_submit_ns: base + 935, submit: layer + 1 });
    return { layer, kind: 'GDN', end: (base + 910) / 1e6 };
  });
  return { step, host, gpu, layers };
}

test('early-submit HC layer anchors do not require MoE in the same command', () => {
  const { step, host, gpu, layers } = layerFixture();
  const ranges = gpuLayerRanges(step, host, gpu, layers);
  assert.equal(ranges.length, 48);
  assert.equal(ranges[0].boundarySubmit, 1);
  assert.equal(ranges[47].end, 47975 / 1e6);
});

test('missing recorder or ambiguous shared submit does not invent GPU layers', () => {
  const { step, host, gpu, layers } = layerFixture();
  assert.deepEqual(gpuLayerRanges(step, host.filter(e => e.name !== 'recorder_lifetime'), gpu, layers), []);
  gpu[2].submit = gpu[1].submit;
  assert.deepEqual(gpuLayerRanges(step, host, gpu, layers), []);
});
