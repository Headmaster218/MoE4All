const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const input = path.resolve(process.env.TRACE_INPUT_ROOT || process.argv[2] || '.');
process.env.TRACE_INPUT_ROOT = input;
const { deviceTime, merge, intersection, length, stats } = require('./analyze-gpu-resident-timeline.cjs');
const output = path.resolve(process.env.TRACE_ANALYSIS_ROOT || input);
const prefix = process.argv[3] || 'stage4-gpu-only-miss-trace';
const popcount = n => { let count = 0; for (; n; n &= n - 1) count++; return count; };
const ms = n => n / 1e6;
function gpuBoundaries(gpu, step, firstResidentLayer) {
  const envelopes = gpu.filter(e => e.name === 'queue_envelope' && e.queue === 'main')
    .sort((a, b) => a.start - b.start);
  if (!envelopes.length) throw new Error('GPU boundary analysis requires main queue envelopes');
  const first = envelopes[0], last = envelopes.reduce((a, b) => a.end > b.end ? a : b);
  if (first.start < step.start_ns || last.end > step.end_ns) throw new Error('GPU envelopes outside the forward clock');
  let l0ToL1GapMs = null, l0EndMs = null, l1StartMs = null;
  if (firstResidentLayer === 1) {
    const hit = gpu.find(e => e.context.layer === 1 &&
      (e.name === 'resident_hit_FFN' || e.name === 'resident_full_hit_FFN'));
    const l1 = envelopes.find(e => e.submit === hit?.submit);
    const l0 = l1 && envelopes.filter(e => e.submit !== l1.submit && e.end <= l1.start)
      .reduce((a, b) => !a || b.end > a.end ? b : a, null);
    if (!l0 || !l1) throw new Error('Missing PLE split queue boundary');
    l0EndMs = ms(l0.end - step.start_ns);
    l1StartMs = ms(l1.start - step.start_ns);
    l0ToL1GapMs = ms(l1.start - l0.end);
  }
  return { frontMs: ms(first.start - step.start_ns), tailMs: ms(step.end_ns - last.end),
    firstGpuMs: ms(first.start - step.start_ns), lastGpuMs: ms(last.end - step.start_ns),
    l0EndMs, l1StartMs, l0ToL1GapMs };
}
function analyze(dir) {
  const trace = JSON.parse(fs.readFileSync(path.join(dir, 'trace.json')));
  const result = JSON.parse(fs.readFileSync(path.join(dir, 'result.json')));
  if (result.status !== 'complete' || trace.host_spans !== false || trace.host_events.length ||
      trace.dropped_events || !trace.clock_samples.length || !trace.device_events.length) {
    throw new Error(`Incomplete GPU-only trace: ${dir}`);
  }
  const warnings = fs.readFileSync(path.join(dir, 'stderr.log'), 'utf8').split(/\r?\n/)
    .filter(line => /timeline query was unavailable|ERROR|panicked/.test(line));
  if (warnings.length) throw new Error(warnings.join('\n'));
  const mapped = trace.device_events.map(e => deviceTime(e, trace.clock_samples));
  const points = trace.steps.map(step => {
    const gpu = mapped.filter(e => e.context.step === step.id).sort((a, b) => a.start - b.start);
    const notices = trace.resident_notices.filter(n => n.context.step === step.id);
    const executed = notices.filter(n => n.status === 1 || n.status === 2);
    // The PLE overlap runner executes layer 0 as a separate ordinary graph, then layers 1..47
    // through the resident tail. Require evidence for layer 0 rather than accepting lost notices.
    const firstResidentLayer = executed.some(n => n.context.layer === 0) ? 0 : 1;
    const expectedLayers = Array.from({ length: 48 - firstResidentLayer }, (_, i) => i + firstResidentLayer);
    const actualLayers = executed.map(n => n.context.layer).sort((a, b) => a - b);
    if (JSON.stringify(actualLayers) !== JSON.stringify(expectedLayers) ||
        (firstResidentLayer === 1 && !gpu.some(e => e.context.layer === 0 && e.name === 'MoE_completion_interval'))) {
      throw new Error(`Executed resident notices or ordinary layer-0 GPU evidence missing at step ${step.id}`);
    }
    const phases = gpu.filter(e => e.name === 'resident_hit_FFN' || e.name === 'resident_full_hit_FFN');
    const consumed = new Map();
    const byLayer = new Map();
    for (const n of notices) {
      if (!byLayer.has(n.context.layer)) byLayer.set(n.context.layer, []);
      byLayer.get(n.context.layer).push(n);
    }
    const actual = [], skipped = [];
    for (const e of phases) {
      const layer = e.context.layer, index = consumed.get(layer) || 0;
      const notice = byLayer.get(layer)?.[index];
      if (!notice) throw new Error(`Missing phase notice L${layer} step ${step.id}`);
      consumed.set(layer, index + 1);
      const annotated = { ...e, status: notice.status, hits: notice.hit_masks.map(popcount).reduce((a, b) => a + b), count: notice.count };
      (notice.status === 0 ? skipped : actual).push(annotated);
    }
    if (actual.length !== expectedLayers.length || consumed.size !== expectedLayers.length || [...byLayer].some(([l, n]) => consumed.get(l) !== n.length)) {
      throw new Error(`Conditional GPU markers did not match journals at step ${step.id}`);
    }
    const dma = gpu.filter(e => e.queue !== 'main');
    const misses = actual.filter(e => e.status === 2).map(hit => {
      const layer = hit.context.layer;
      const resume = gpu.find(e => e.name === 'resident_miss_resume_FFN' && e.context.layer === layer);
      const transfers = dma.filter(e => e.context.layer === layer);
      if (!resume) throw new Error(`Missing resumed FFN L${layer} step ${step.id}`);
      for (const t of transfers) {
        if (!t.transfer_signal || t.transfer_signal > resume.transfer_wait) {
          throw new Error(`DMA dependency not carried into miss resume L${layer}`);
        }
      }
      const notice = gpu.findLast(e => e.name === 'resident_router_notice' && e.context.layer === layer && e.end <= hit.start + 1000);
      if (!notice) throw new Error(`Missing GPU router notice L${layer}`);
      const dmaIntervals = merge(transfers.map(e => [e.start, e.end]));
      const dmaEnd = transfers.length ? Math.max(...transfers.map(e => e.end)) : null;
      const dmaStart = transfers.length ? Math.min(...transfers.map(e => e.start)) : null;
      const nextGpu = gpu.find(e => e.queue === 'main' && e.name !== 'queue_envelope' &&
        e.start >= resume.end && e.end > e.start);
      return { layer, hits: hit.hits, missing: hit.count - hit.hits,
        route: transfers.length ? 'DMA or mixed' : 'mapped CPU push (no DMA)',
        noticeMs: ms(notice.end - step.start_ns), hitStartMs: ms(hit.start - step.start_ns), hitEndMs: ms(hit.end - step.start_ns),
        resumeStartMs: ms(resume.start - step.start_ns), resumeEndMs: ms(resume.end - step.start_ns),
        hitMs: ms(hit.end - hit.start), missMs: ms(resume.end - resume.start),
        dmaMs: ms(length(dmaIntervals)), dmaBytes: transfers.reduce((sum, t) => sum + t.bytes, 0),
        dmaStartAfterNoticeMs: dmaStart === null ? null : ms(dmaStart - notice.end),
        dmaOverlapHitMs: ms(length(intersection(dmaIntervals, [[hit.start, hit.end]]))),
        hitToResumeGapMs: ms(Math.max(0, resume.start - hit.end)),
        uncoveredDmaTailMs: dmaEnd === null ? null : ms(Math.max(0, dmaEnd - hit.end)),
        readyToResumeGapMs: dmaEnd === null ? null : ms(Math.max(0, resume.start - Math.max(hit.end, dmaEnd))),
        resumeToNextGpuGapMs: nextGpu ? ms(nextGpu.start - resume.end) : null,
        transfers: transfers.map(t => ({ start: ms(t.start - step.start_ns), end: ms(t.end - step.start_ns), signal: t.transfer_signal, bytes: t.bytes })) };
    });
    const envelopes = gpu.filter(e => e.name === 'queue_envelope');
    const main = merge(envelopes.map(e => [e.start, e.end]));
    const dmaIntervals = merge(dma.map(e => [e.start, e.end]));
    const view = e => ({ start: ms(e.start - step.start_ns), end: ms(e.end - step.start_ns), name: e.name,
      layer: e.context.layer, submit: e.submit, wait: e.transfer_wait, signal: e.transfer_signal, bytes: e.bytes,
      status: e.status, hits: e.hits, count: e.count });
    return { id: step.id, positions: step.positions, wallMs: ms(step.end_ns - step.start_ns),
      mainMs: ms(length(main)), dmaMs: ms(length(dmaIntervals)),
      dmaOverlapMainMs: ms(length(intersection(main, dmaIntervals))), misses,
      boundaries: gpuBoundaries(gpu, step, firstResidentLayer),
      rows: [envelopes.map(view), gpu.filter(e => e.name !== 'queue_envelope' && e.queue === 'main' && !e.name.startsWith('resident_hit') && e.name !== 'resident_full_hit_FFN' && e.name !== 'resident_miss_resume_FFN').map(view),
        actual.map(view), gpu.filter(e => e.name === 'resident_miss_resume_FFN').map(view), dma.map(view), skipped.map(view)],
      skippedCount: skipped.length };
  });
  const missList = points.flatMap(p => p.misses);
  const metrics = ['hitMs', 'missMs', 'dmaMs', 'dmaStartAfterNoticeMs', 'dmaOverlapHitMs', 'hitToResumeGapMs', 'uncoveredDmaTailMs', 'readyToResumeGapMs', 'resumeToNextGpuGapMs'];
  return { case: path.basename(dir), depth: result.depth, binaryHash: result.binaryHash, points,
    wall: stats(points.map(p => p.wallMs)), main: stats(points.map(p => p.mainMs)), dma: stats(points.map(p => p.dmaMs)),
    missCount: missList.length, dmaMissCount: missList.filter(m => m.transfers.length).length,
    missMetrics: Object.fromEntries(metrics.map(k => [k, stats(missList.map(m => m[k]).filter(v => v !== null))])),
    boundaryMetrics: Object.fromEntries(['frontMs', 'l0ToL1GapMs', 'tailMs'].map(k => {
      const values = points.map(p => p.boundaries[k]).filter(v => v !== null);
      return [k, values.length ? stats(values) :
        { n: 0, mean: null, median: null, p90: null, p95: null, min: null, max: null, stddev: null }];
    })),
    sumsPerStep: Object.fromEntries(metrics.map(k => [k, stats(points.map(p => p.misses.reduce((sum, m) => sum + (m[k] || 0), 0)))])),
    maxClockDeviationNs: Math.max(...trace.clock_samples.map(c => c.deviation_ns)) };
}
function main() {
  fs.mkdirSync(output, { recursive: true });
  const cases = fs.readdirSync(input, { withFileTypes: true }).filter(e => e.isDirectory() && e.name.startsWith(`${prefix}-`) && fs.existsSync(path.join(input, e.name, 'trace.json')))
    .map(e => analyze(path.join(input, e.name))).sort((a, b) => a.depth - b.depth);
  if (cases.length !== 2 || cases[0].depth !== 30000 || cases[1].depth !== 150000 || cases.some(c => c.points.length !== 12)) {
    throw new Error('Expected two 12-step single-active cases, 30K and 150K');
  }
  if (output !== input) {
    for (const c of cases) {
      const destination = path.join(output, c.case);
      fs.mkdirSync(destination, { recursive: true });
      for (const file of ['trace.json', 'result.json', 'stderr.log', 'stdout.log']) {
        const source = path.join(input, c.case, file);
        if (fs.existsSync(source)) fs.copyFileSync(source, path.join(destination, file));
      }
    }
    for (const entry of fs.readdirSync(input, { withFileTypes: true })) {
      if (entry.isFile() && entry.name.startsWith(`${prefix}-`)) {
        fs.copyFileSync(path.join(input, entry.name), path.join(output, entry.name));
      }
    }
  }
  fs.writeFileSync(path.join(output, 'analysis.json'), JSON.stringify(cases, null, 2));
  const payload = zlib.gzipSync(Buffer.from(JSON.stringify(cases))).toString('base64');
  const html = fs.readFileSync(path.join(__dirname, 'gpu-only-miss-timeline.html'), 'utf8').replace('__PAYLOAD__', payload);
  fs.writeFileSync(path.join(output, 'timeline.html'), html);
  console.log(JSON.stringify(cases.map(({ points, ...summary }) => summary), null, 2));
}
module.exports = { analyze, gpuBoundaries };
if (require.main === module) main();
