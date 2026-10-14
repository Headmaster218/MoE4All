// Historical traces identify the MoE stage, not a complete GPU transformer layer.
// Recover CPU layer ranges from graph order; never group all events by context.layer.
function layerRanges(step, events) {
  const ordered = events.filter(e => e.context.step === step.id)
    .sort((a, b) => a.start_ns - b.start_ns);
  const experts = ordered.filter(e => e.name === 'expert_layer');
  const out = [];
  let previousEnd = step.start_ns, previousLayer = -1;
  for (let i = 0; i < experts.length; i++) {
    const expert = experts[i], layer = expert.context.layer;
    const limit = experts[i + 1]?.start_ns ?? step.end_ns;
    const end = ordered.find(e => e.name === 'QwenHcInject' && e.start_ns >= expert.end_ns && e.end_ns <= limit);
    const prefix = ordered.filter(e => e.start_ns >= previousEnd && e.end_ns <= expert.start_ns);
    const kind = prefix.some(e => ['Attention', 'QsaBatchAttention'].includes(e.name)) ? 'QSA'
      : prefix.some(e => e.name === 'DeltaNet') ? 'GDN' : null;
    const start = layer === 0 ? prefix.find(e => e.name === 'QwenHcNorm')
      : prefix.find(e => /^[A-Z]/.test(e.name));
    if (layer === previousLayer + 1 && start && end && kind &&
        start.start_ns >= step.start_ns && end.end_ns <= step.end_ns) {
      const ms = t => (t - step.start_ns) / 1e6;
      out.push({ layer, kind, start: ms(start.start_ns), end: ms(end.end_ns),
        expertStart: ms(expert.start_ns), expertEnd: ms(expert.end_ns) });
    }
    previousLayer = layer;
    previousEnd = end?.end_ns ?? limit;
  }
  return out;
}

function gpuLayerRanges(step, host, gpu, layers) {
  const segments = gpu.filter(e => e.context.step === step.id && e.queue === 'main' && e.name !== 'queue_envelope')
    .sort((a, b) => a.start - b.start);
  const first = segments.find(e => e.name === 'HC_completion_interval' && e.context.layer === -1);
  if (!first || layers.length !== 48) return [];
  const ops = host.filter(e => e.context.step === step.id && /^[A-Z]/.test(e.name) && e.name !== 'PLE_wait')
    .sort((a, b) => a.start_ns - b.start_ns);
  const submits = host.filter(e => e.context.step === step.id && e.name === 'queue_submit')
    .sort((a, b) => a.start_ns - b.start_ns);
  const recorders = host.filter(e => e.context.step === step.id && e.name === 'recorder_lifetime');
  const boundaries = layers.map(l => {
    const inject = ops.find(e => e.name === 'QwenHcInject' && Math.abs((e.end_ns - step.start_ns) / 1e6 - l.end) < 1e-8);
    const recorder = inject && recorders.find(r => r.start_ns <= inject.start_ns && r.end_ns >= inject.end_ns);
    const submit = recorder && submits.find(s => s.start_ns >= recorder.end_ns);
    // Early submit separates the final HC from the preceding MoE. Associate
    // via actual host submit time, not the recorder's stale layer context.
    const event = submit && segments.find(e => e.name === 'HC_completion_interval' &&
      e.host_submit_ns >= submit.start_ns && e.host_submit_ns <= submit.end_ns);
    const next = inject && ops.find(e => e.start_ns >= inject.end_ns);
    return { event, merged: !!next?.name.startsWith('QwenHc') };
  });
  if (boundaries.some(b => !b.event)) return [];
  if (new Set(boundaries.map(b => b.event.submit)).size !== 48) return [];
  // Use isolated final-HC spans in this same step/lanes to estimate the final
  // injection's share of merged injection + next-layer norm spans. No CPU/GPU scaling.
  const isolated = boundaries.filter(b => !b.merged).map(b => b.event.end - b.event.start).sort((a, b) => a - b);
  const at = (isolated.length - 1) / 2;
  const typical = isolated.length ? (isolated[Math.floor(at)] + isolated[Math.ceil(at)]) / 2 : null;
  const ms = t => (t - step.start_ns) / 1e6;
  let previous = ms(first.start), previousBounds = [previous, ms(first.end)];
  const out = [];
  for (let i = 0; i < layers.length; i++) {
    const { event, merged } = boundaries[i];
    const duration = event.end - event.start;
    const end = ms(merged ? event.start + (typical === null ? duration / 2 : Math.min(typical, duration)) : event.end);
    if (end <= previous || previous < 0 || end > (step.end_ns - step.start_ns) / 1e6) return [];
    const endBounds = [ms(event.start), ms(event.end)];
    out.push({ layer: i, kind: layers[i].kind, start: previous, end,
      startBounds: previousBounds, endBounds, boundarySubmit: event.submit,
      method: merged ? typical === null ? 'half-HC-fallback' : 'isolated-HC-median' : 'isolated-HC-end' });
    previous = end; previousBounds = endBounds;
  }
  return out;
}

module.exports = { layerRanges, gpuLayerRanges };
