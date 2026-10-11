const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const {buildSegments, evaluatePiecewise, fitSingleExponential, summarize, figure, warmupFigure} = require('./generate.cjs');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'curve-data.json'), 'utf8'));
const publishedText = name => fs.readFileSync(path.join(__dirname,name),'utf8').replace(/\r\n/g,'\n');

test('known exponential is recovered with one segment', () => {
  const points = Array.from({length: 61}, (_, i) => ({capacityGiB: 12+i/10, missPercent: 12*Math.exp(-0.12*(12+i/10-15.5))}));
  const segments = buildSegments(points);
  assert.equal(segments.length, 1);
  const fit = fitSingleExponential(points, 15.5);
  assert.ok(Math.abs(fit.amplitudePercent-12) < 1e-6);
  assert.ok(Math.abs(fit.decayPerGiB-0.12) < 1e-7);
  assert.ok(fit.metrics.maxErrorPercentagePoints < 1e-6);
});

test('a plateau and adjacent sample retain nonincreasing behavior', () => {
  const points = [{capacityGiB: 4, missPercent: 10}, {capacityGiB: 5, missPercent: 10}, {capacityGiB: 6, missPercent: 2}];
  const segments = buildSegments(points);
  assert.equal(segments.length, 2);
  assert.equal(evaluatePiecewise(segments, 4.5), 10);
  assert.equal(evaluatePiecewise(segments, 6), 2);
});

test('invalid or unordered data is rejected', () => {
  assert.throws(() => buildSegments([]));
  assert.throws(() => buildSegments([{capacityGiB: 4, missPercent: 10}, {capacityGiB: 4, missPercent: 9}]));
  assert.throws(() => buildSegments([{capacityGiB: 4, missPercent: 10}, {capacityGiB: 5, missPercent: 11}]));
  assert.throws(() => buildSegments([{capacityGiB: 4, missPercent: 10}, {capacityGiB: 5, missPercent: 0}]));
});

test('all 476 replay points and the exact baseline remain intact', () => {
  assert.equal(data.points.length, 476);
  assert.equal(data.points[0].capacityGiB, 0.5);
  assert.equal(data.points.at(-1).capacityGiB, 48);
  assert.equal(data.baselineAgreement.compared, 22331520);
  assert.equal(data.baselineAgreement.disagreements, 0);
  data.points.forEach((p, i) => assert.ok(Math.abs(p.capacityGiB - (i+5)/10) < 1e-12));
});

test('both error tolerances hold across the complete fit grid', () => {
  const result = summarize(data);
  assert.ok(result.metrics.maxErrorPercentagePoints <= 0.05);
  assert.ok(result.metrics.maxRelativeErrorPercent <= 1);
  assert.ok(result.metrics.rSquared > 0.9999);
});

test('piecewise joins are continuous and the entire function is monotone', () => {
  const {segments} = summarize(data);
  for (let i = 1; i < segments.length; i++) {
    const a = segments[i-1], b = segments[i];
    assert.equal(a.highGiB, b.lowGiB);
    assert.ok(Math.abs(evaluatePiecewise([a], a.highGiB) - b.amplitudePercent) < 1e-10);
  }
  let previous = Infinity;
  for (let c = 4; c <= 48; c += 0.01) {
    const value = evaluatePiecewise(segments, c);
    assert.ok(value <= previous + 1e-12); previous = value;
  }
  assert.throws(() => evaluatePiecewise(segments, 3.99), RangeError);
  assert.throws(() => evaluatePiecewise(segments, 48.01), RangeError);
  assert.throws(() => evaluatePiecewise(segments, NaN), RangeError);
});

test('local single-exponential statistics are accurate on the new per-turn cohort', () => {
  const fit = summarize(data).singleExponential.find(f => f.intervalGiB[0] === 12);
  assert.ok(fit.metrics.maxErrorPercentagePoints < 0.05);
  assert.equal(fit.metrics.samples,61);
});

test('warmup exclusions are per turn, not just first-turn or cache resets', () => {
  const all=data.warmupProfiles.find(p=>p.excludedPerTurn===0);
  const selected=data.warmupProfiles.find(p=>p.excludedPerTurn===512);
  assert.equal(all.forwards-selected.forwards,4*512);
  assert.equal(selected.selections,7443840);
  assert.equal(data.workload.selectedDecodeForwards,15508);
  for (const t of data.warmupTurns) {
    assert.equal(t.windows.reduce((n,w)=>n+w.selections,0),t.forwards*480);
    assert.equal(t.summaries.find(s=>s.excludedPerTurn===512).selections,(t.forwards-512)*480);
  }
  assert.deepEqual(data.points,selected.points);
  assert.equal(publishedText('warmup-miss-rate.svg'),warmupFigure(data));
});

test('published chart and summary are reproducible', () => {
  const summary = summarize(data);
  assert.equal(publishedText('fit-summary.json'), JSON.stringify(summary, null, 2)+'\n');
  assert.equal(publishedText('capacity-miss-curve.svg'), figure(data, summary));
  assert.equal((figure(data, summary).match(/<circle /g)||[]).length, 476+61+241+2);
  assert.ok(!figure(data, summary).includes('NaN'));
});
