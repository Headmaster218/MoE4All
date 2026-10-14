const fs = require('node:fs');
const path = require('node:path');
process.env.TRACE_INPUT_ROOT ||= path.resolve(__dirname, '../benchmark-data/artifacts/gpu-resident-decode-20261011');
const { deviceTime, partition, inferenceEvents, stats } = require('./analyze-gpu-resident-timeline.cjs');
const us = ns => ns / 1000;

function splitReadyGap(ready, submit, start) {
  if (start < ready || start < submit) throw new Error('Resume precedes readiness/submit');
  return { totalUs: us(start - ready), beforeSubmitUs: us(Math.max(0, submit - ready)),
    queuedUs: us(start - Math.max(ready, submit)), submitLeadUs: us(Math.max(0, ready - submit)) };
}

function splitSubmitCallGap(ready, stamp, start, calls) {
  const split = splitReadyGap(ready, stamp, start);
  const matching = calls.filter(e => e.start_ns <= stamp && e.end_ns >= stamp);
  if (matching.length !== 1) throw new Error('Expected one main queue-submit API envelope');
  const call = matching[0];
  const beforeCallUs = us(Math.max(0, Math.min(start, call.start_ns) - ready));
  const callOverlapUs = us(Math.max(0, Math.min(start, call.end_ns) - Math.max(ready, call.start_ns)));
  const afterCallReturnUs = us(Math.max(0, start - Math.max(ready, call.end_ns)));
  if (Math.abs(beforeCallUs + callOverlapUs + afterCallReturnUs - split.totalUs) > 1e-8) {
    throw new Error('Submit API partition does not cover readiness-to-GPU interval');
  }
  return { beforeCallUs, callOverlapUs, afterCallReturnUs,
    gpuStartedBeforeApiReturn: start < call.end_ns };
}

function breakdown(events, start, end) {
  const result = {};
  for (const [a, b, name] of partition(inferenceEvents(events), start, end)) {
    result[name] = (result[name] || 0) + us(b - a);
  }
  return result;
}

function phaseDuration(events, name, start, end) {
  const matching = events.filter(e => e.name === name &&
    (e.start_ns === e.end_ns ? e.start_ns >= start && e.end_ns <= end : e.start_ns < end && e.end_ns > start));
  if (!matching.length) return null;
  const ranges = matching.map(e => [Math.max(start, e.start_ns), Math.min(end, e.end_ns)])
    .filter(([a, b]) => b > a).sort((a, b) => a[0] - b[0]);
  let total = 0, cursor = start;
  for (const [a, b] of ranges) { total += Math.max(0, b - Math.max(a, cursor)); cursor = Math.max(cursor, b); }
  return us(total);
}

function analyze(trace) {
  if (!trace.host_spans || trace.dropped_events || !trace.clock_samples.length || trace.steps.length !== 12) {
    throw new Error('Expected complete twelve-forward CPU/GPU diagnostic');
  }
  const gpu = trace.device_events.map(e => deviceTime(e, trace.clock_samples));
  const points = trace.steps.map(step => {
    const device = gpu.filter(e => e.context.step === step.id).sort((a, b) => a.start - b.start);
    const host = trace.host_events.filter(e => e.context.step === step.id);
    const envelopes = device.filter(e => e.queue === 'main' && e.name === 'queue_envelope');
    const first = Math.min(...envelopes.map(e => e.start)), last = Math.max(...envelopes.map(e => e.end));
    if (!envelopes.length || first < step.start_ns || last > step.end_ns) throw new Error('Invalid forward envelope');
    const journals = new Map(), consumed = new Map();
    for (const notice of trace.resident_notices.filter(n => n.context.step === step.id)) {
      const layer = notice.context.layer;
      if (!journals.has(layer)) journals.set(layer, []);
      journals.get(layer).push(notice);
    }
    const misses = [];
    for (const hit of device.filter(e => e.name === 'resident_hit_FFN' || e.name === 'resident_full_hit_FFN')) {
      const layer = hit.context.layer, index = consumed.get(layer) || 0;
      const journal = journals.get(layer)?.[index];
      if (!journal) throw new Error('Missing conditional journal');
      consumed.set(layer, index + 1);
      if (journal.status !== 2) continue;
      const resume = device.find(e => e.name === 'resident_miss_resume_FFN' && e.context.layer === layer);
      const notice = device.findLast(e => e.name === 'resident_router_notice' && e.context.layer === layer && e.end <= hit.start + 1000);
      const admitted = host.filter(e => e.name === 'resident_producer_admit' && e.context.layer === layer);
      const submitted = host.filter(e => e.name === 'resident_producer_dma_submit' && e.context.layer === layer);
      if (!resume || !notice || admitted.length !== 1 || submitted.length !== 1) throw new Error(`Missing miss phases L${layer}`);
      const admit = admitted[0], submit = submitted[0];
      const dma = device.filter(e => e.name === 'DMA_copy' && e.context.layer === layer);
      for (const e of dma) if (!e.transfer_signal || e.transfer_signal > resume.transfer_wait) throw new Error('Missing DMA resume dependency');
      // Mapped-only readiness is a host-side phase, not an absent/zero transfer time.
      const ready = Math.max(hit.end, submit.end_ns, ...dma.map(e => e.end));
      const split = splitReadyGap(ready, resume.host_submit_ns, resume.start);
      const api = splitSubmitCallGap(ready, resume.host_submit_ns, resume.start,
        host.filter(e => e.name === 'queue_submit'));
      const layerHost = host.filter(e => e.context.layer === layer);
      const workerFields = { lockWaitUs: 'resident_producer_lock_wait', lookupUs: 'resident_producer_import_lookup',
        mappedCopyUs: 'resident_producer_mapped_copy', dmaPrepareUs: 'resident_producer_dma_prepare',
        dmaApiUs: 'resident_producer_dma_queue_submit', stagingCopyUs: 'resident_producer_staging_copy' };
      const worker = Object.fromEntries(Object.entries(workerFields).map(([key, name]) =>
        [key, phaseDuration(layerHost, name, admit.start_ns, submit.end_ns)]));
      misses.push({ layer, route: dma.length ? 'DMA or mixed' : 'mapped CPU push (producer envelope)',
        noticeToAdmissionUs: us(admit.start_ns - notice.end), admissionUs: us(admit.end_ns - admit.start_ns),
        producerSubmitUs: us(submit.end_ns - submit.start_ns),
        noticeToDmaUs: dma.length ? us(Math.min(...dma.map(e => e.start)) - notice.end) : null,
        dmaEnqueueToStartUs: dma.length ? us(Math.min(...dma.map(e => e.start - e.host_submit_ns))) : null,
        postReady: split, postReadyApi: api, worker, producerToMainSubmitUs: us(resume.host_submit_ns - submit.end_ns),
        hostPostReady: breakdown(host, ready, Math.max(ready, resume.host_submit_ns)) });
    }
    if (misses.length !== [...journals.values()].flat().filter(n => n.status === 2).length) throw new Error('Unmatched misses');
    return { id: step.id, wallUs: us(step.end_ns - step.start_ns),
      frontUs: us(first - step.start_ns), tailUs: us(step.end_ns - last),
      front: breakdown(host, step.start_ns, first), tail: breakdown(host, last, step.end_ns), misses };
  });
  const allMisses = points.flatMap(p => p.misses);
  const byRoute = Object.fromEntries(['DMA or mixed', 'mapped CPU push (producer envelope)'].map(route => {
    const rows = allMisses.filter(m => m.route === route);
    const fields = ['noticeToAdmissionUs', 'admissionUs', 'producerSubmitUs', 'noticeToDmaUs', 'dmaEnqueueToStartUs', 'producerToMainSubmitUs'];
    return [route, { n: rows.length,
      phases: Object.fromEntries(fields.map(k => [k, stats(rows.map(m => m[k]).filter(v => v !== null))])),
      postReady: Object.fromEntries(['totalUs', 'beforeSubmitUs', 'queuedUs', 'submitLeadUs'].map(k => [k, stats(rows.map(m => m.postReady[k]))])),
      postReadyApi: Object.fromEntries(['beforeCallUs', 'callOverlapUs', 'afterCallReturnUs'].map(k => [k, stats(rows.map(m => m.postReadyApi[k]))])),
      worker: Object.fromEntries(['lockWaitUs', 'lookupUs', 'mappedCopyUs', 'dmaPrepareUs', 'dmaApiUs', 'stagingCopyUs']
        .map(k => [k, stats(rows.map(m => m.worker[k]).filter(v => v !== null))])),
      gpuStartedBeforeApiReturn: rows.filter(m => m.postReadyApi.gpuStartedBeforeApiReturn).length }];
  }));
  const averageBreakdown = key => {
    const names = new Set(points.flatMap(p => Object.keys(p[key])));
    return Object.fromEntries([...names].map(name => [name, stats(points.map(p => p[key][name] || 0))]));
  };
  return { limitations: ['host_submit_ns is stamped before the queue-submit API, not after return',
      'afterCallReturnUs still includes GPU dependencies and scheduling; it is not pure driver cost',
      'host profiling/query readback can delay control flow; compare performance without profiling'],
    maxClockDeviationNs: Math.max(...trace.clock_samples.map(s => s.deviation_ns)),
    wall: stats(points.map(p => p.wallUs)), front: stats(points.map(p => p.frontUs)), tail: stats(points.map(p => p.tailUs)),
    frontCategories: averageBreakdown('front'), tailCategories: averageBreakdown('tail'), byRoute, points };
}

function main() {
  const root = process.env.TRACE_INPUT_ROOT;
  const tag = process.argv[2] || 'stage4-miss-boundary-diagnostic';
  const results = [30000, 150000].map(depth => {
    const directory = path.join(root, `${tag}-diagnostic-control-c1-d${depth}`);
    const result = JSON.parse(fs.readFileSync(path.join(directory, 'result.json')));
    if (result.status !== 'complete') throw new Error('Incomplete diagnostic service');
    return { depth, binaryHash: result.binaryHash, ...analyze(JSON.parse(fs.readFileSync(path.join(directory, 'trace.json')))) };
  });
  fs.writeFileSync(path.join(root, `${tag}-analysis.json`), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results.map(({ points, ...r }) => r), null, 2));
}
module.exports = { splitReadyGap, splitSubmitCallGap, phaseDuration, breakdown, analyze };
if (require.main === module) main();
