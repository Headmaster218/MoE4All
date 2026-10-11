const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
process.env.TRACE_INPUT_ROOT = os.tmpdir();
const { analyze } = require('./analyze-gpu-only-miss-trace.cjs');
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
  assert.equal(p.rows[2].length, 48); assert.equal(p.rows[4].length, 1);
  assert.equal(m.hits, 9); assert.equal(m.missing, 1); assert.equal(m.dmaBytes, 1048576);
  assert.equal(m.hitToResumeGapMs, .039); assert.equal(m.readyToResumeGapMs, .015);
  assert.equal(m.uncoveredDmaTailMs, .024); assert.equal(m.dmaOverlapHitMs, 0);
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
