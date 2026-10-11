const fs = require('node:fs');
const path = require('node:path');

const ABS_TOLERANCE_PP = 0.05;
const REL_TOLERANCE_PERCENT = 1;

function metrics(points, predict) {
  const errors = points.map(p => predict(p.capacityGiB) - p.missPercent);
  const mean = points.reduce((sum, p) => sum + p.missPercent, 0) / points.length;
  const sse = errors.reduce((sum, e) => sum + e * e, 0);
  const sst = points.reduce((sum, p) => sum + (p.missPercent - mean) ** 2, 0);
  return {
    samples: points.length,
    rSquared: sst === 0 ? (sse === 0 ? 1 : null) : 1 - sse / sst,
    rmsErrorPercentagePoints: Math.sqrt(sse / points.length),
    maxErrorPercentagePoints: Math.max(...errors.map(Math.abs)),
    maxRelativeErrorPercent: Math.max(...errors.map((e, i) => Math.abs(e) / points[i].missPercent * 100)),
  };
}

function validatePoints(points) {
  if (points.length < 2) throw new Error('At least two points are required');
  points.forEach((p, i) => {
    if (!Number.isFinite(p.capacityGiB) || !Number.isFinite(p.missPercent)
      || p.missPercent <= 0 || p.missPercent > 100) throw new Error('Invalid point');
    if (i && (p.capacityGiB <= points[i - 1].capacityGiB
      || p.missPercent > points[i - 1].missPercent)) throw new Error('Points must be ordered and nonincreasing');
  });
}

function segmentPrediction(segment, capacityGiB) {
  return segment.amplitudePercent * Math.exp(-segment.decayPerGiB * (capacityGiB - segment.lowGiB));
}

function evaluatePiecewise(segments, capacityGiB) {
  if (!Number.isFinite(capacityGiB) || capacityGiB < segments[0].lowGiB
    || capacityGiB > segments.at(-1).highGiB) throw new RangeError('Capacity is outside the fitted range');
  const segment = segments.find(s => capacityGiB <= s.highGiB);
  return segmentPrediction(segment, capacityGiB);
}

function buildSegments(points, absoluteTolerance = ABS_TOLERANCE_PP, relativeTolerance = REL_TOLERANCE_PERCENT) {
  validatePoints(points);
  if (!(absoluteTolerance > 0) || !(relativeTolerance > 0)) throw new Error('Tolerances must be positive');
  function split(lo, hi) {
    const a = points[lo], b = points[hi];
    const segment = {
      lowGiB: a.capacityGiB, highGiB: b.capacityGiB,
      amplitudePercent: a.missPercent,
      decayPerGiB: Math.log(a.missPercent / b.missPercent) / (b.capacityGiB - a.capacityGiB),
    };
    let worst = -1, largest = 1;
    for (let i = lo + 1; i < hi; i++) {
      const error = Math.abs(segmentPrediction(segment, points[i].capacityGiB) - points[i].missPercent);
      const score = Math.max(error / absoluteTolerance, error / points[i].missPercent * 100 / relativeTolerance);
      if (score > largest) { largest = score; worst = i; }
    }
    if (worst !== -1) return [...split(lo, worst), ...split(worst, hi)];
    segment.metrics = metrics(points.slice(lo, hi + 1), c => segmentPrediction(segment, c));
    return [segment];
  }
  return split(0, points.length - 1);
}

function fitSingleExponential(points, referenceGiB) {
  validatePoints(points);
  const loss = decayPerGiB => {
    const basis = points.map(p => Math.exp(-decayPerGiB * (p.capacityGiB - referenceGiB)));
    const amplitudePercent = points.reduce((sum, p, i) => sum + p.missPercent * basis[i], 0)
      / basis.reduce((sum, v) => sum + v * v, 0);
    const sse = points.reduce((sum, p, i) => sum + (amplitudePercent * basis[i] - p.missPercent) ** 2, 0);
    return { amplitudePercent, decayPerGiB, sse };
  };
  let best = loss(0), index = 0;
  for (let i = 1; i <= 2000; i++) {
    const candidate = loss(i / 200);
    if (candidate.sse < best.sse) { best = candidate; index = i; }
  }
  let lo = Math.max(0, (index - 1) / 200), hi = (index + 1) / 200;
  const ratio = (Math.sqrt(5) - 1) / 2;
  let x1 = hi - ratio * (hi - lo), x2 = lo + ratio * (hi - lo);
  let f1 = loss(x1), f2 = loss(x2);
  for (let i = 0; i < 100; i++) {
    if (f1.sse < f2.sse) {
      hi = x2; x2 = x1; f2 = f1;
      x1 = hi - ratio * (hi - lo); f1 = loss(x1);
    } else {
      lo = x1; x1 = x2; f1 = f2;
      x2 = lo + ratio * (hi - lo); f2 = loss(x2);
    }
  }
  best = f1.sse < f2.sse ? f1 : f2;
  const predict = c => best.amplitudePercent * Math.exp(-best.decayPerGiB * (c - referenceGiB));
  return {
    referenceGiB, amplitudePercent: best.amplitudePercent, decayPerGiB: best.decayPerGiB,
    intervalGiB: [points[0].capacityGiB, points.at(-1).capacityGiB],
    metrics: metrics(points, predict),
  };
}

function summarize(data) {
  validatePoints(data.points);
  const fittedPoints = data.points.filter(p => p.capacityGiB >= 4);
  const segments = buildSegments(fittedPoints);
  const singleExponential = [[0.5, 48], [4, 48], [12, 18], [24, 48]].map(([lo, hi]) =>
    fitSingleExponential(data.points.filter(p => p.capacityGiB >= lo && p.capacityGiB <= hi), data.referenceGiB));
  return {
    absoluteTolerancePercentagePoints: ABS_TOLERANCE_PP,
    relativeTolerancePercent: REL_TOLERANCE_PERCENT,
    intervalGiB: [4, 48],
    formula: 'missPercent = amplitudePercent * exp(-decayPerGiB * (capacityGiB-lowGiB))',
    scope: 'Continuous endpoint interpolation; tolerances are checked only at the recorded 0.1 GiB samples.',
    metrics: metrics(fittedPoints, c => evaluatePiecewise(segments, c)),
    segments, singleExponential,
  };
}

const escape = text => String(text).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
function figure(data, summary) {
  const svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="900" viewBox="0 0 1200 900" role="img" aria-labelledby="title desc">',
    '<title id="title">FlashNext expert-cache capacity and miss-rate replay</title>',
    '<desc id="desc">All 476 capacity samples after excluding the first 512 Decode forwards of EACH turn, comparison with zero and 1024 exclusions, continuous piecewise exponential on 4-48 GiB, and absolute residuals. This cutoff is not proof of stationarity. Effective routed-expert capacity, not total VRAM or measured MTP performance.</desc>',
    '<rect width="1200" height="900" fill="#fff"/>',
    '<style>text{font-family:Arial,Helvetica,sans-serif;font-size:16px;fill:#222}.heading{font-size:24px;font-weight:bold}.panel-title{font-size:18px;font-weight:bold}.small{font-size:14px}.grid{stroke:#e5e5e5;stroke-width:1}.frame{stroke:#aaa;stroke-width:1;fill:none}.replay{fill:none;stroke:#0072b2;stroke-width:2}.fit{fill:none;stroke:#d55e00;stroke-width:2;stroke-dasharray:7 4}.local{fill:none;stroke:#009e73;stroke-width:2;stroke-dasharray:2 4}.all{fill:none;stroke:#777;stroke-width:1.5;stroke-dasharray:4 3}.late{fill:none;stroke:#777;stroke-width:1.5;stroke-dasharray:1 4}</style>',
    '<text x="44" y="36" class="heading">FlashNext: expert-cache capacity vs. miss rate</text>',
    `<text x="44" y="63">Exclude first 512 forwards per turn / ${data.workload.selectedDecodeForwards.toLocaleString('en-US')} forwards / ${data.workload.logicalExpertSelections.toLocaleString('en-US')} expert selections</text>`,
    '<path d="M44,87 h30" class="replay"/><text x="84" y="92">Replay: exclude 512</text>',
    '<path d="M320,87 h30" class="fit"/><text x="360" y="92">Piecewise exponential</text>',
    '<path d="M650,87 h30" class="local"/><text x="690" y="92">Local exponential</text>',
    '<path d="M960,87 h30" class="all"/><text x="1000" y="92">No exclusion</text>',
    '<path d="M960,109 h30" class="late"/><text x="1000" y="114">Exclude 1024</text>',
  ];
  const localFit = summary.singleExponential.find(f => f.intervalGiB[0] === 12);
  const localPredict = c => localFit.amplitudePercent * Math.exp(-localFit.decayPerGiB * (c - localFit.referenceGiB));
  const replayPath = (values, X, Y) => values.map((p, i) => `${i ? 'L' : 'M'}${X(p.capacityGiB).toFixed(2)},${Y(p.missPercent).toFixed(2)}`).join(' ');
  const dense = (lo, hi, predict) => Array.from({length: 601}, (_, i) => ({capacityGiB: lo + (hi - lo) * i / 600, missPercent: predict(lo + (hi - lo) * i / 600)}));
  function panel(id, ox, oy, title, domainX, domainY, ticksX, ticksY, options = {}) {
    const w = 560, h = 330, left = 72, right = w - 20, top = 40, bottom = h - 55;
    const X = c => left + (c - domainX[0]) / (domainX[1] - domainX[0]) * (right - left);
    const transformed = value => options.log ? Math.log10(value) : value;
    const Y = m => bottom - (transformed(m) - transformed(domainY[0])) / (transformed(domainY[1]) - transformed(domainY[0])) * (bottom - top);
    const values = data.points.filter(p => p.capacityGiB >= domainX[0] && p.capacityGiB <= domainX[1]);
    svg.push(`<g transform="translate(${ox},${oy})"><text x="${left}" y="19" class="panel-title">${escape(title)}</text>`);
    svg.push(`<defs><clipPath id="clip-${id}"><rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}"/></clipPath></defs>`);
    for (const c of ticksX) svg.push(`<line x1="${X(c)}" x2="${X(c)}" y1="${top}" y2="${bottom}" class="grid"/><text x="${X(c)}" y="${bottom+24}" text-anchor="middle">${c}</text>`);
    for (const m of ticksY) svg.push(`<line x1="${left}" x2="${right}" y1="${Y(m)}" y2="${Y(m)}" class="grid"/><text x="${left-9}" y="${Y(m)+5}" text-anchor="end">${m}</text>`);
    svg.push(`<rect x="${left}" y="${top}" width="${right-left}" height="${bottom-top}" class="frame"/>`);
    svg.push(`<text x="${(left+right)/2}" y="${h-8}" text-anchor="middle">Effective expert cache (GiB)</text>`);
    svg.push(`<text transform="translate(18,${(top+bottom)/2}) rotate(-90)" text-anchor="middle">${options.residual ? 'Fit - replay (percentage points)' : options.log ? 'Expert miss (%) / log scale' : 'Expert miss (%)'}</text>`);
    svg.push(`<g clip-path="url(#clip-${id})">`);
    if (options.residual) {
      const residual = values.map(p => ({...p, missPercent: evaluatePiecewise(summary.segments, p.capacityGiB) - p.missPercent}));
      svg.push(`<path d="${replayPath(residual,X,Y)}" class="fit"/>`);
    } else {
      for (const [cutoff, cls] of [[0,'all'],[1024,'late']]) {
        const comparison = data.warmupProfiles.find(p=>p.excludedPerTurn===cutoff).points.filter(p=>p.capacityGiB>=domainX[0]&&p.capacityGiB<=domainX[1]);
        svg.push(`<path d="${replayPath(comparison,X,Y)}" class="${cls}"/>`);
      }
      svg.push(`<path d="${replayPath(values,X,Y)}" class="replay"/>`);
      for (const p of values) svg.push(`<circle cx="${X(p.capacityGiB).toFixed(2)}" cy="${Y(p.missPercent).toFixed(2)}" r="1.4" fill="#0072b2"><title>${p.capacityGiB} GiB / ${p.missPercent.toFixed(5)}%</title></circle>`);
      const fitLo = Math.max(4,domainX[0]), fitHi = Math.min(48,domainX[1]);
      svg.push(`<path d="${replayPath(dense(fitLo,fitHi,c=>evaluatePiecewise(summary.segments,c)),X,Y)}" class="fit"/>`);
      if (options.local) svg.push(`<path d="${replayPath(dense(12,18,localPredict),X,Y)}" class="local"/>`);
      if (domainX[0] < data.referenceGiB && domainX[1] > data.referenceGiB) {
        const rx = X(data.referenceGiB), ry = Y(data.fixedCapacityMissPercentAtReference);
        svg.push(`<circle cx="${rx}" cy="${ry}" r="4.5" fill="#0072b2"/>`);
        const tx = id === 'full' ? rx+15 : right-12, ty = id === 'full' ? ry-25 : top+24;
        svg.push(`<text x="${tx}" y="${ty}" text-anchor="${id==='full'?'start':'end'}" class="small">${data.referenceGiB.toFixed(2)} GiB / ${data.fixedCapacityMissPercentAtReference.toFixed(2)}%</text>`);
      }
    }
    svg.push('</g></g>');
  }
  panel('full',20,125,'A. Full sampled capacity range',[0.5,48],[0,100],[1,8,16,24,32,40,48],[0,20,40,60,80,100]);
  panel('local',620,125,'B. Current capacity neighborhood',[12,18],[9,18.6],[12,13,14,15,16,17,18],[10,12,14,16,18],{local:true});
  panel('tail',20,495,'C. Large-cache tail (logarithmic Y)',[24,48],[0.06,4.2],[24,30,36,42,48],[0.1,0.2,0.5,1,2,4],{log:true});
  panel('residual',620,495,'D. Piecewise residual at every sample',[4,48],[-0.055,0.055],[4,12,20,28,36,44,48],[-0.05,-0.025,0,0.025,0.05],{residual:true});
  svg.push(`<text x="44" y="855">Piecewise: ${summary.segments.length} continuous segments / max error ${summary.metrics.maxErrorPercentagePoints.toFixed(4)} pp / max relative error ${summary.metrics.maxRelativeErrorPercent.toFixed(3)}%</text>`);
  svg.push('<text x="44" y="879" class="small">Errors apply to this replay grid only. Connecting sampled points does not recover unsampled CDF steps. No extrapolation.</text></svg>');
  return svg.join('\n') + '\n';
}

function warmupFigure(data) {
  const out = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="780" viewBox="0 0 1200 780" role="img" aria-labelledby="title desc">',
    '<title id="title">Per-turn observed expert Miss rate, including startup and question changes</title>',
    '<desc id="desc">Four natural generations. Each point is a non-overlapping 64-forward observed window. Gray shading identifies the first 512 forwards per turn, excluded from the primary capacity curve. This cutoff does not establish stationarity.</desc>',
    '<rect width="1200" height="780" fill="#fff"/><style>text{font-family:Arial,Helvetica,sans-serif;font-size:16px;fill:#222}.grid{stroke:#e5e5e5;stroke-width:1}.line{fill:none;stroke:#0072b2;stroke-width:2}</style>',
    '<text x="40" y="35" style="font-size:24px;font-weight:bold">Observed per-turn Miss rate: startup and question changes</text>',
    '<text x="40" y="62">64-forward windows / actual changing cache capacity / shaded: first 512 forwards per turn</text>'];
  const maxY = Math.ceil(Math.max(...data.warmupTurns.flatMap(t=>t.windows.map(w=>w.missPercent)))/5)*5;
  for (const [i,t] of data.warmupTurns.entries()) {
    const ox=20+(i%2)*600,oy=90+Math.floor(i/2)*325;
    const X=f=>65+f/5500*485,Y=m=>255-m/maxY*205;
    out.push(`<g transform="translate(${ox},${oy})"><text x="65" y="22">Turn ${t.turn} / ${t.forwards.toLocaleString('en-US')} ordinary Decode forwards</text>`);
    out.push(`<rect x="65" y="50" width="${X(512)-65}" height="205" fill="#eee"/>`);
    for(const x of [0,1000,2000,3000,4000,5000])out.push(`<line x1="${X(x)}" x2="${X(x)}" y1="50" y2="255" class="grid"/><text x="${X(x)}" y="280" text-anchor="middle">${x}</text>`);
    for(let y=0;y<=maxY;y+=5)out.push(`<line x1="65" x2="550" y1="${Y(y)}" y2="${Y(y)}" class="grid"/><text x="57" y="${Y(y)+5}" text-anchor="end">${y}%</text>`);
    out.push(`<rect x="65" y="50" width="485" height="205" fill="none" stroke="#aaa"/><path d="${t.windows.map((w,j)=>`${j?'L':'M'}${X((w.startForward+w.endForward)/2).toFixed(2)},${Y(w.missPercent).toFixed(2)}`).join(' ')}" class="line"/>`);
    out.push('<text x="308" y="310" text-anchor="middle">Decode forward since this turn started</text></g>');
  }
  out.push('<text x="40" y="765">Prefill accesses still warm the replay cache. Later first-seen experts remain cold misses. No guaranteed stationary cutoff.</text></svg>');
  return out.join('\n')+'\n';
}

function main() {
  const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'curve-data.json'), 'utf8'));
  const summary = summarize(data);
  fs.writeFileSync(path.join(__dirname, 'fit-summary.json'), JSON.stringify(summary, null, 2) + '\n');
  fs.writeFileSync(path.join(__dirname, 'capacity-miss-curve.svg'), figure(data, summary));
  fs.writeFileSync(path.join(__dirname, 'warmup-miss-rate.svg'), warmupFigure(data));
  console.log(JSON.stringify({metrics: summary.metrics, segments: summary.segments, singleExponential: summary.singleExponential}, null, 2));
}
if (require.main === module) main();
module.exports = {metrics, validatePoints, segmentPrediction, evaluatePiecewise, buildSegments, fitSingleExponential, summarize, figure, warmupFigure};
