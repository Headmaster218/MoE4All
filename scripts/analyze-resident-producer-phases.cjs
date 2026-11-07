const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(process.argv[2] || '');
if (!process.argv[2]) throw new Error('Pass the completed CPU-only case directory');
process.env.TRACE_INPUT_ROOT = root;
const { merge, length, partition, inferenceEvents, stats } = require('./analyze-gpu-resident-timeline.cjs');
const trace = JSON.parse(fs.readFileSync(path.join(root, 'trace.json'), 'utf8'));
const result = JSON.parse(fs.readFileSync(path.join(root, 'result.json'), 'utf8'));
assert.equal(result.status, 'complete');
assert.equal(trace.device_timestamps, false);
assert.equal(trace.device_events.length, 0);
assert.equal(trace.dropped_events, 0);
assert.equal(trace.steps.length, 12);
const phases = ['resident_notice_ack_wait', 'resident_segment_recycle', 'resident_notice_poll',
  'resident_producer_total', 'resident_producer_event_wait', 'resident_producer_journal_read',
  'resident_producer_lru', 'resident_producer_admit', 'resident_producer_dma_submit'];
const points = trace.steps.map(step => {
  const events = trace.host_events.filter(e => e.context.step === step.id);
  const worker = events.filter(e => e.name === 'resident_producer_total');
  assert.ok(worker.length > 0, `Missing worker context at step ${step.id}`);
  assert.ok(worker.every(e => e.start_ns >= step.start_ns && e.end_ns <= step.end_ns),
    'Producer phase escaped its owning forward');
  const clipped = list => merge(list.map(e => [Math.max(step.start_ns, e.start_ns), Math.min(step.end_ns, e.end_ns)]));
  const categories = {};
  const exclusive = partition(inferenceEvents(events), step.start_ns, step.end_ns);
  for (const [a, b, name] of exclusive) categories[name] = (categories[name] || 0) + (b - a) / 1e6;
  const wallMs = (step.end_ns - step.start_ns) / 1e6;
  assert.ok(Math.abs(Object.values(categories).reduce((a, b) => a + b, 0) - wallMs) < 1e-6);
  return { step: step.id, positions: step.positions, wallMs, categories,
    phases: Object.fromEntries(phases.map(name => {
      const list = events.filter(e => e.name === name);
      return [name, { count: list.length, ms: length(clipped(list)) / 1e6 }];
    })),
  };
});
const categories = [...new Set(points.flatMap(p => Object.keys(p.categories)))];
const report = {
  binaryHash: result.binaryHash, arguments: result.args,
  samples: points.length, wallMs: stats(points.map(p => p.wallMs)),
  mainThread: Object.fromEntries(categories.map(k => [k, stats(points.map(p => p.categories[k] || 0))])),
  phases: Object.fromEntries(phases.map(k => [k, {
    count: stats(points.map(p => p.phases[k].count)),
    ms: stats(points.map(p => p.phases[k].ms)),
  }])),
  semantics: [
    'Main-thread categories are exclusive wall-time partitions, not CPU busy time.',
    'Producer phases run on another thread; total contains subphases and cannot be summed with them.',
    'Event wait includes driver polling; admission includes LRU, pinning and RAM/direct copy preparation.',
    'DMA submit is worker CPU submit time, not transfer-queue GPU duration.',
    'No GPU timestamps: this diagnostic alone does not establish critical-path savings or GPU utilization.',
  ], points,
};
fs.writeFileSync(path.join(root, 'producer-phase-analysis.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify({ samples: report.samples, wallMs: report.wallMs,
  mainThreadMeans: Object.fromEntries(Object.entries(report.mainThread).map(([k, v]) => [k, v.mean])),
  phaseMeans: Object.fromEntries(Object.entries(report.phases).map(([k, v]) => [k, { count: v.count.mean, ms: v.ms.mean }])),
}, null, 2));
