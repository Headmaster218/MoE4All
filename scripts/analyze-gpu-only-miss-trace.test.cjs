const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
process.env.TRACE_INPUT_ROOT = os.tmpdir();
const { analyze, gpuBoundaries } = require('./analyze-gpu-only-miss-trace.cjs');
function fixture(dma) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'gpu-only-trace-unit-'));
  const context = layer => ({ step: 0, lanes: 1, layer });
  const events = [], notices = [];
  function event(name, layer, start, end, extra = {}) {
    events.push({ name, context: context(layer), queue: 'main', submit: layer + 1, host_submit_ns: 0,
      start_tick: start, end_tick: end, valid_bits: 64, period_ns: 1, transfer_wait: layer === 2 && dma ? 7 : 0,
      transfer_signal: 0, bytes: 0, ...extra });
  }
  for (let layer = 0; layer < 48; layer++) {
    const start = layer * 100000;
    event('queue_envelope', layer, start, start + (layer === 2 ? 70000 : 21000));
    if (layer === 3) {
      notices.push({ context: context(layer), ordinal: 3, status: 0, hit_masks: [1023, 0], count: 10 });
      event('resident_hit_FFN', layer, start - 3000, start - 2000);
    }
    notices.push({ context: context(layer), ordinal: layer, status: layer === 2 ? 2 : 1, hit_masks: [layer === 2 ? 511 : 1023, 0], count: 10 });
    event('resident_router_notice', layer, start, start + 1000);
    event('resident_hit_FFN', layer, start + 1000, start + 21000);
    if (layer === 2) {
      event('resident_miss_resume_FFN', layer, start + 60000, start + 70000);
      if (dma) event('DMA_copy', layer, start + 35000, start + 45000, { queue: 'dma', transfer_signal: 7, bytes: 1048576 });
    }
  }
  const trace = { host_spans: false, host_events: [], dropped_events: 0,
    clock_samples: [{ host_ns: 0, device_tick: 0, period_ns: 1, deviation_ns: 1 }],
    device_events: events, resident_notices: notices,
    steps: [{ id: 0, lanes: 1, positions: [30000], start_ns: 0, end_ns: 5000000 }] };
  const write = () => fs.writeFileSync(path.join(dir, 'trace.json'), JSON.stringify(trace));
  write();
  fs.writeFileSync(path.join(dir, 'result.json'), JSON.stringify({ status: 'complete', depth: 30000, binaryHash: 'fixture' }));
  fs.writeFileSync(path.join(dir, 'stderr.log'), '');
  return { dir, trace, write };
}
test('GPU-only analysis matches conditional notices and DMA dependency without double counting', t => {
  const f = fixture(true); t.after(() => fs.rmSync(f.dir, { recursive: true }));
  const a = analyze(f.dir), p = a.points[0], m = p.misses[0];
  assert.equal(p.rows[2].length, 48); assert.equal(p.rows[5].length, 1);
  assert.equal(m.hits, 9); assert.equal(m.missing, 1); assert.equal(m.dmaBytes, 1048576);
  assert.equal(m.hitToResumeGapMs, .039); assert.equal(m.readyToResumeGapMs, .015);
  assert.equal(m.uncoveredDmaTailMs, .024); assert.equal(m.dmaOverlapHitMs, 0);
  assert.equal(m.resumeToNextGpuGapMs, .027);
});
test('mapped push has no fabricated GPU transfer duration; malformed trace is rejected', t => {
  const f = fixture(false); t.after(() => fs.rmSync(f.dir, { recursive: true }));
  const m = analyze(f.dir).points[0].misses[0];
  assert.equal(m.readyToResumeGapMs, null); assert.equal(m.dmaStartAfterNoticeMs, null);
  f.trace.host_events.push({}); f.write();
  assert.throws(() => analyze(f.dir), /Incomplete GPU-only/);
});
test('missing or unsynchronized DMA is never accepted as a measured resume dependency', t => {
  const f = fixture(true); t.after(() => fs.rmSync(f.dir, { recursive: true }));
  f.trace.device_events.find(e => e.name === 'DMA_copy').transfer_signal = 8; f.write();
  assert.throws(() => analyze(f.dir), /dependency not carried/);
});
test('PLE split requires 47 resident layers and real ordinary L0 GPU evidence', t => {
  const f = fixture(true); t.after(() => fs.rmSync(f.dir, { recursive: true }));
  f.trace.resident_notices = f.trace.resident_notices.filter(n => n.context.layer !== 0);
  f.trace.device_events = f.trace.device_events.filter(e => e.context.layer !== 0 || ['queue_envelope', 'resident_hit_FFN'].includes(e.name));
  f.trace.device_events.find(e => e.context.layer === 0 && e.name === 'resident_hit_FFN').name = 'MoE_completion_interval';
  f.write(); assert.equal(analyze(f.dir).points[0].rows[2].length, 47);
  f.trace.device_events = f.trace.device_events.filter(e => e.context.layer !== 0);
  f.write(); assert.throws(() => analyze(f.dir), /ordinary layer-0 GPU evidence missing/);
});
test('forward boundaries use actual queue ends and preserve the PLE split gap', t => {
  const f = fixture(false); t.after(() => fs.rmSync(f.dir, { recursive: true }));
  f.trace.resident_notices = f.trace.resident_notices.filter(n => n.context.layer !== 0);
  f.trace.device_events = f.trace.device_events.filter(e => e.context.layer !== 0 || ['queue_envelope', 'resident_hit_FFN'].includes(e.name));
  f.trace.device_events.find(e => e.context.layer === 0 && e.name === 'resident_hit_FFN').name = 'MoE_completion_interval';
  f.write();
  const b = analyze(f.dir).points[0].boundaries;
  assert.equal(b.frontMs, 0); assert.equal(b.l0ToL1GapMs, .079);
  assert.equal(b.l0EndMs, .021); assert.equal(b.l1StartMs, .1);
  assert.equal(b.lastGpuMs, 4.721); assert.equal(b.tailMs, .279);
  const unsplit = fixture(false); t.after(() => fs.rmSync(unsplit.dir, { recursive: true }));
  const full = analyze(unsplit.dir);
  assert.equal(full.points[0].boundaries.l0ToL1GapMs, null);
  assert.equal(full.boundaryMetrics.l0ToL1GapMs.n, 0);
  assert.equal(full.boundaryMetrics.l0ToL1GapMs.mean, null);
  assert.equal(full.boundaryMetrics.l0ToL1GapMs.median, null);
});
test('missing envelopes and invalid forward clocks cannot fabricate a boundary', () => {
  const step = { start_ns: 0, end_ns: 1000 };
  assert.throws(() => gpuBoundaries([], step, 0), /requires main queue envelopes/);
  assert.throws(() => gpuBoundaries([{ name: 'queue_envelope', queue: 'main', start: -1, end: 100 }], step, 0), /outside/);
  assert.throws(() => gpuBoundaries([{ name: 'queue_envelope', queue: 'main', start: 1, end: 1001 }], step, 0), /outside/);
});
test('a final resumed FFN without following GPU evidence has an unknown post-miss gap', t => {
  const f = fixture(false); t.after(() => fs.rmSync(f.dir, { recursive: true }));
  const n = f.trace.resident_notices.find(n => n.context.layer === 47);
  n.status = 2; n.hit_masks[0] = 511;
  f.trace.device_events.find(e => e.context.layer === 47 && e.name === 'queue_envelope').end_tick = 4770000;
  f.trace.device_events.push({ name: 'resident_miss_resume_FFN', context: { step: 0, lanes: 1, layer: 47 },
    queue: 'main', submit: 49, host_submit_ns: 0, start_tick: 4760000, end_tick: 4770000,
    valid_bits: 64, period_ns: 1, transfer_wait: 0, transfer_signal: 0, bytes: 0 });
  f.write();
  assert.equal(analyze(f.dir).points[0].misses.find(m => m.layer === 47).resumeToNextGpuGapMs, null);
});
