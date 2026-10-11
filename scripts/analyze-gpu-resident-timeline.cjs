const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
if (!process.env.TRACE_INPUT_ROOT) throw new Error('TRACE_INPUT_ROOT must name the new trace directory');
const root = path.resolve(process.env.TRACE_INPUT_ROOT);
const outputRoot = process.env.TRACE_ANALYSIS_ROOT ? path.resolve(process.env.TRACE_ANALYSIS_ROOT) : root;
const q = (values, p) => {
  const a = [...values].sort((x, y) => x - y);
  if (!a.length) return null;
  const at = (a.length - 1) * p, lo = Math.floor(at), hi = Math.ceil(at);
  return a[lo] + (a[hi] - a[lo]) * (at - lo);
};
function stats(values) {
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  return { n: values.length, mean, median: q(values, .5), p90: q(values, .9), p95: q(values, .95),
    min: q(values, 0), max: q(values, 1), stddev: Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length) };
}
function merge(intervals) {
  const out = [];
  for (const [a, b] of intervals.filter(([a, b]) => b > a).sort((x, y) => x[0] - y[0])) {
    const last = out.at(-1);
    if (last && a <= last[1]) last[1] = Math.max(last[1], b); else out.push([a, b]);
  }
  return out;
}
const length = intervals => intervals.reduce((sum, [a, b]) => sum + b - a, 0);
function intersection(a, b) {
  let i = 0, j = 0, out = [];
  while (i < a.length && j < b.length) {
    const lo = Math.max(a[i][0], b[j][0]), hi = Math.min(a[i][1], b[j][1]);
    if (hi > lo) out.push([lo, hi]);
    if (a[i][1] < b[j][1]) i++; else j++;
  }
  return out;
}
function classify(name) {
  if (name.startsWith('trace_')) return [200, 'measurement'];
  if (['queue_idle_wait', 'fence_wait', 'dma_slot_wait', 'dma_timeline_wait', 'PLE_wait',
    'gpu_resident_notice_wait', 'gpu_resident_notice_producer_wait', 'resident_notice_ack_wait'].includes(name)) return [100, 'wait'];
  if (name === 'command_encode') return [90, 'command_encode'];
  if (name === 'queue_submit' || name === 'dma_queue_submit') return [80, 'submit'];
  if (name === 'dma_prepare_submit') return [70, 'dma_host_prepare'];
  if (name === 'expert_mapped_copy') return [75, 'expert_mapped_copy'];
  if (name === 'recorder_acquire') return [60, 'recorder_acquire'];
  if (name === 'resident_segment_recycle') return [60, 'recorder_recycle'];
  if (name === 'resident_notice_poll') return [50, 'notice_poll'];
  if (name === 'router_readback') return [60, 'router_readback'];
  if (name === 'backend_setup') return [50, 'backend_setup'];
  if (name === 'gpu_resident_lut_freeze') return [50, 'residency_snapshot'];
  if (name === 'expert_prepare' || name === 'expert_down_prepare') return [50, 'expert_prepare'];
  if (name === 'expert_enqueue') return [50, 'expert_enqueue'];
  if (name === 'step_front' || name === 'step_tail') return [20, 'scheduler_front_tail'];
  return [0, 'lowering_other'];
}
function inferenceEvents(events) {
  return events.filter(e => !e.name.startsWith('resident_producer_'));
}
function partition(events, start, end) {
  const edges = [[start, 0, null], [end, 0, null]], active = new Set(), result = [];
  for (const event of events) {
    if (event.name === 'recorder_lifetime') continue;
    const a = Math.max(start, event.start_ns), b = Math.min(end, event.end_ns);
    if (b > a) { edges.push([a, 1, event], [b, -1, event]); }
  }
  edges.sort((a, b) => a[0] - b[0]);
  let previous = start;
  for (const [at, direction, event] of edges) {
    if (at > previous) {
      let selected = 'lowering_other', priority = -1;
      for (const candidate of active) {
        const [p, category] = classify(candidate.name);
        if (p > priority) { priority = p; selected = category; }
      }
      const last = result.at(-1);
      if (last && last[2] === selected && last[1] === previous) last[1] = at;
      else result.push([previous, at, selected]);
    }
    if (event) { if (direction > 0) active.add(event); else active.delete(event); }
    previous = at;
  }
  return result;
}
function deviceTime(event, samples) {
  const sample = samples.reduce((best, value) =>
    !best || Math.abs(value.host_ns - event.host_submit_ns) < Math.abs(best.host_ns - event.host_submit_ns) ? value : best, null);
  if (!sample) throw new Error('no calibrated clock samples');
  const modulus = 2 ** event.valid_bits;
  let delta = event.start_tick - sample.device_tick;
  if (event.valid_bits < 64) { delta = ((delta + modulus / 2) % modulus + modulus) % modulus - modulus / 2; }
  let duration = event.end_tick - event.start_tick;
  if (duration < 0) duration += modulus;
  const start = sample.host_ns + delta * sample.period_ns;
  return { ...event, start, end: start + duration * event.period_ns, deviation: sample.deviation_ns };
}
function analyze(dir) {
  const result = JSON.parse(fs.readFileSync(path.join(dir, 'result.json'), 'utf8'));
  const data = JSON.parse(fs.readFileSync(path.join(dir, 'trace.json'), 'utf8'));
  if (result.status !== 'complete') throw new Error(`incomplete case ${dir}`);
  if (data.dropped_events) throw new Error(`dropped ${data.dropped_events} events`);
  if (data.device_timestamps === false || !data.clock_samples.length || !data.device_events.length) {
    throw new Error('A four-track timeline requires real calibrated GPU timestamps');
  }
  const resident = result.args.includes('kernels.vulkan.decode_gpu_resident=true');
  const hostByStep = new Map(), gpuByStep = new Map();
  for (const event of inferenceEvents(data.host_events)) {
    if (!hostByStep.has(event.context.step)) hostByStep.set(event.context.step, []);
    hostByStep.get(event.context.step).push(event);
  }
  for (const raw of data.device_events) {
    const event = deviceTime(raw, data.clock_samples);
    if (event.end < event.start) throw new Error('negative GPU interval');
    if (event.start + event.deviation + 1000 < event.host_submit_ns) throw new Error(`GPU starts before CPU submit: ${event.submit}`);
    if (!gpuByStep.has(event.context.step)) gpuByStep.set(event.context.step, []);
    gpuByStep.get(event.context.step).push(event);
  }
  const points = data.steps.map(step => {
    if (step.lanes !== result.count) throw new Error('wrong concurrency sample');
    const host = hostByStep.get(step.id) || [], gpu = gpuByStep.get(step.id) || [];
    const partitioned = partition(host, step.start_ns, step.end_ns);
    const cpu = {};
    for (const [a, b, category] of partitioned) cpu[category] = (cpu[category] || 0) + (b - a) / 1e6;
    const main = merge(gpu.filter(e => e.queue === 'main' && e.name === 'queue_envelope').map(e => [Math.max(step.start_ns, e.start), Math.min(step.end_ns, e.end)]));
    const pending = merge(gpu.filter(e => e.queue === 'main' && e.name === 'queue_envelope').map(e => [Math.max(step.start_ns,e.host_submit_ns),Math.min(step.end_ns,e.start)]));
    const dma = merge(gpu.filter(e => e.queue === 'dma').map(e => [Math.max(step.start_ns, e.start), Math.min(step.end_ns, e.end)]));
    const encode = merge(partitioned.filter(e => e[2] === 'command_encode').map(e => e.slice(0, 2)));
    const waiting = merge(partitioned.filter(e => e[2] === 'wait').map(e => e.slice(0, 2)));
    const preparing = merge(partitioned.filter(e => ['expert_prepare', 'expert_mapped_copy'].includes(e[2])).map(e => e.slice(0, 2)));
    const mapped = merge(partitioned.filter(e => e[2] === 'expert_mapped_copy').map(e => e.slice(0, 2)));
    const encodeOverlap = length(intersection(encode, main)) / 1e6;
    const waitOverlap = length(intersection(waiting, main)) / 1e6;
    const dmaOverlap = length(intersection(dma, main)) / 1e6;
    const wall = (step.end_ns - step.start_ns) / 1e6;
    const closureError = Math.abs(Object.values(cpu).reduce((sum, v) => sum + v, 0) - wall);
    if (closureError > .00001) throw new Error(`CPU partition does not close: ${closureError}`);
    return { id: step.id, lanes: step.lanes, positions: step.positions, wallMs: wall, cpu,
      mainMs: length(main) / 1e6, dmaMs: length(dma) / 1e6, dmaOverlapMs: dmaOverlap,
      mainPendingMs: length(pending) / 1e6, pendingDmaOverlapMs: length(intersection(pending,dma)) / 1e6,
      pendingMainOverlapMs: length(intersection(pending,main)) / 1e6,
      gpuUnionMs: (length(main) + length(dma) - dmaOverlap * 1e6) / 1e6,
      encodeOverlapMs: encodeOverlap, encodeOutsideMainMs: (cpu.command_encode || 0) - encodeOverlap,
      prepareTotalMs: length(preparing) / 1e6,
      prepareOutsideMainMs: (length(preparing) - length(intersection(preparing, main))) / 1e6,
      mappedOutsideMainMs: (length(mapped) - length(intersection(mapped, main))) / 1e6,
      waitMainOverlapMs: waitOverlap, waitOutsideMainMs: (cpu.wait || 0) - waitOverlap,
      waitSpansMs: Object.fromEntries(['queue_idle_wait','fence_wait','dma_slot_wait','dma_timeline_wait','PLE_wait',
        'gpu_resident_notice_wait','gpu_resident_notice_producer_wait','resident_notice_ack_wait'].map(name => [name,
        length(merge(host.filter(e => e.name === name).map(e => [Math.max(step.start_ns,e.start_ns),Math.min(step.end_ns,e.end_ns)]))) / 1e6])),
      recorderLifetimeMs: length(merge(host.filter(e => e.name === 'recorder_lifetime').map(e => [e.start_ns, e.end_ns]))) / 1e6,
      mainSubmits: gpu.filter(e => e.name === 'queue_envelope').length,
      dmaSubmits: gpu.filter(e => e.queue === 'dma').length,
      dmaBytes: gpu.filter(e => e.queue === 'dma').reduce((sum, e) => sum + e.bytes, 0),
      maxClockDeviationNs: Math.max(0, ...gpu.map(e => e.deviation)),
      carriedGpuIntervals: gpu.filter(e => e.end > step.end_ns + e.deviation).length,
      rows: [partitioned.map(([a,b,name]) => [(a-step.start_ns)/1e6,(b-step.start_ns)/1e6,name]),
        pending.map(([a,b]) => [(a-step.start_ns)/1e6,(b-step.start_ns)/1e6,'pending']),
        main.map(([a,b]) => [(a-step.start_ns)/1e6,(b-step.start_ns)/1e6,'main']),
        dma.map(([a,b]) => [(a-step.start_ns)/1e6,(b-step.start_ns)/1e6,'dma'])],
    };
  });
  if (points.length < 8) throw new Error(`not enough samples ${points.length}`);
  const fields = ['wallMs','mainMs','mainPendingMs','pendingMainOverlapMs','pendingDmaOverlapMs','dmaMs','dmaOverlapMs','gpuUnionMs','encodeOverlapMs','encodeOutsideMainMs','prepareTotalMs','prepareOutsideMainMs','mappedOutsideMainMs','waitMainOverlapMs','waitOutsideMainMs','recorderLifetimeMs','mainSubmits','dmaSubmits','dmaBytes'];
  const distributions = Object.fromEntries(fields.map(name => [name, stats(points.map(point => point[name]))]));
  const categories = [...new Set(points.flatMap(point => Object.keys(point.cpu)))];
  distributions.cpu = Object.fromEntries(categories.map(name => [name, stats(points.map(point => point.cpu[name] || 0))]));
  distributions.waitSpans = Object.fromEntries(Object.keys(points[0].waitSpansMs).map(name => [name, stats(points.map(point => point.waitSpansMs[name]))]));
  const median = q(points.map(p => p.wallMs), .5), p90 = q(points.map(p => p.wallMs), .9);
  const nearest = value => points.reduce((best, point) => Math.abs(point.wallMs - value) < Math.abs(best.wallMs - value) ? point : best).id;
  const summary = { case: path.basename(dir), mode: resident ? 'GPU resident' : 'CPU control', resident,
    binaryHash: result.binaryHash, count: result.count, depth: result.depth, slots: result.slots,
    capturedSteps: points.length, windowSize: Number(result.args.find(arg => arg.startsWith('prof.timeline_steps='))?.split('=')[1] ?? 16), distributions, representative: { median: nearest(median), p90: nearest(p90) },
    maxClockDeviationNs: Math.max(...points.map(p => p.maxClockDeviationNs)),
    outputHashes: result.rounds.map(round => round.requests.map(req => req.outputHash)),
    profilerEnabledTimings: result.rounds.map(round => round.requests.map(req => req.timings)),
    queryWarnings: fs.readFileSync(path.join(dir, 'stderr.log'), 'utf8').split(/\r?\n/).filter(line => /timeline query was unavailable|ERROR|panicked/.test(line)),
    points };
  fs.writeFileSync(path.join(dir, 'analysis.json'), JSON.stringify(summary));
  return summary;
}

function main() {
fs.mkdirSync(outputRoot, { recursive: true });
const cases = process.argv.includes('--render-only') ? JSON.parse(fs.readFileSync(path.join(outputRoot, 'analysis.json'), 'utf8')) : fs.readdirSync(root, { withFileTypes: true }).filter(d => d.isDirectory() && d.name.startsWith((process.argv[2] || 'measured') + '-') && fs.existsSync(path.join(root,d.name,'trace.json')))
  .map(d => analyze(path.join(root, d.name))).sort((a,b) => a.depth-b.depth || Number(a.resident)-Number(b.resident));
if (cases.length !== 4 || cases.some(c => c.count !== 1)) throw new Error('Expected baseline/resident at 30K and 150K, one active lane only');
if (new Set(cases.map(c => c.binaryHash)).size !== 1) throw new Error('All trace cases must use the same binary');
for (const depth of [30000, 150000]) {
  const pair = cases.filter(c => c.depth === depth);
  if (pair.length !== 2 || pair[0].resident || !pair[1].resident ||
      JSON.stringify(pair[0].points.map(p => p.positions)) !== JSON.stringify(pair[1].points.map(p => p.positions))) {
    throw new Error(`Unmatched trace positions at ${depth}`);
  }
}
fs.writeFileSync(path.join(outputRoot, 'analysis.json'), JSON.stringify(cases));
const labels = ['command_encode','wait','submit','expert_prepare','expert_mapped_copy','expert_enqueue',
  'dma_host_prepare','recorder_acquire','recorder_recycle','notice_poll','router_readback','backend_setup','residency_snapshot',
  'scheduler_front_tail','measurement','lowering_other','main','dma','pending'];
const { layerRanges, gpuLayerRanges } = require('./gpu-resident-timeline-layers.cjs');
const markers = {}, gpuMarkers = {};
for (const c of cases) {
  const trace = JSON.parse(fs.readFileSync(path.join(root, c.case, 'trace.json'), 'utf8'));
  markers[c.case] = {}; gpuMarkers[c.case] = {};
  for (const step of trace.steps) {
    // Speculative replay emits duplicate CPU layer spans and unconditional query
    // markers for skipped work. The historical layer inference is not valid here.
    const layers = c.resident ? [] : layerRanges(step, trace.host_events);
    markers[c.case][step.id] = layers;
    gpuMarkers[c.case][step.id] = c.resident ? [] : gpuLayerRanges(step, trace.host_events,
      trace.device_events.filter(e => e.context.step === step.id).map(e => deviceTime(e, trace.clock_samples)), layers);
  }
}
fs.writeFileSync(path.join(outputRoot, 'layer-markers.json'), JSON.stringify(markers));
fs.writeFileSync(path.join(outputRoot, 'gpu-layer-estimates.json'), JSON.stringify(gpuMarkers));
const throughput = process.env.TRACE_THROUGHPUT_MANIFEST
  ? JSON.parse(fs.readFileSync(process.env.TRACE_THROUGHPUT_MANIFEST, 'utf8')) : null;
if (throughput && throughput.binaryHash !== cases[0].binaryHash) throw new Error('Throughput and trace binary mismatch');
const displayCases = cases.map(c => ({mode:c.mode,resident:c.resident,depth:c.depth,count:c.count,slots:c.slots,capturedSteps:c.capturedSteps,distributions:c.distributions,representative:c.representative,
  unprofiledRate: throughput?.results.find(r => r.name === (c.resident ? 'resident-inflight1' : 'baseline'))
    ?.matrix.find(r => r.count === 1 && r.depth === c.depth)?.rounds.slice(1)
    .reduce((sum,r,_,a) => sum + r[0].rate/a.length, 0),
  points:c.points.map(p => ({id:p.id,positions:p.positions,wallMs:p.wallMs,encodeOverlapMs:p.encodeOverlapMs,encodeOutsideMainMs:p.encodeOutsideMainMs,dmaOverlapMs:p.dmaOverlapMs,maxClockDeviationNs:p.maxClockDeviationNs,
    layers: markers[c.case]?.[p.id] || [],
    gpuLayers: gpuMarkers[c.case]?.[p.id] || [],
    rows:p.rows.map(row=>row.map(([a,b,k])=>[Math.round(a*10000)/10000,Math.round(b*10000)/10000,labels.indexOf(k)]))}))}));
const payload = zlib.gzipSync(Buffer.from(JSON.stringify(displayCases))).toString('base64');
const html = fs.readFileSync(path.join(__dirname, 'gpu-resident-timeline-view.html'), 'utf8')
  .replace('__TIMELINE_PAYLOAD__', payload)
  .replace('__TIMELINE_LABELS__', JSON.stringify(labels));
fs.writeFileSync(path.join(outputRoot, 'timeline.html'), html);
console.log(JSON.stringify(cases.map(({points, outputHashes, profilerEnabledTimings, ...summary}) => summary), null, 2));
}
module.exports = { merge, intersection, length, partition, deviceTime, stats, analyze, inferenceEvents };
if (require.main === module) main();
