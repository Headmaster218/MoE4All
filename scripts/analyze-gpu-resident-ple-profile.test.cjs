const assert = require('node:assert/strict');
const test = require('node:test');
const { parsePleCohorts, parseHostSetupCohorts } = require('./analyze-gpu-resident-ple-profile.cjs');
const runner = '[parallel-token-profile] lanes=1 steps=414 rows=414 decode_rows=384 prefill_rows=30 ple_wait=0.5ms ple_upload=0.4ms';
const pager = '[parallel-token-pager] ple_plan=0.9ms ple_work=17.2ms ple_wait=0.1ms';
test('PLE work and unhidden wait are distinct and include suffix rows', () => {
  const [c] = parsePleCohorts(`${runner}\n${pager}`);
  assert.equal(c.gatherMs, 17.2); assert.equal(c.residualWaitMs, .1);
  assert.equal(c.runnerWaitMs, .5); assert.equal(c.prefill_rows, 30);
  assert.equal(c.gatherUsPerProcessedRow, 17200 / 414);
});
test('multiple rounds preserve first/repeated measurements', () => {
  const cs = parsePleCohorts(`${runner}\n${pager.replace('17.2ms', '393.3ms')}\n${runner}\n${pager}`);
  assert.equal(cs.length, 2); assert.ok(cs[0].gatherMs > cs[1].gatherMs);
});
test('unpaired or malformed cohorts cannot fabricate a timing', () => {
  for (const text of ['', runner, pager, `${runner}\n${runner}\n${pager}`,
    `${runner.replace('rows=414', 'rows=413')}\n${pager}`, `${runner}\n${pager.replace('ple_work', 'unknown')}`]) {
    assert.throws(() => parsePleCohorts(text));
  }
});

const hostRunner = `${runner} main_setup=110.2ms front=108.5ms`;
const host = '[parallel-token-host] setup_layout=66.1ms setup_scratch=42.5ms setup_rope=4.8ms setup_fusion=2.6ms setup_moe_scan=133.1ms';
test('host setup normalization uses forward count, not independent rows', () => {
  const text = `${hostRunner}\n${pager}\n${host}`;
  const [single] = parseHostSetupCohorts(text);
  assert.equal(single.partsUsPerForward.setup_moe_scan, 133100 / 414);
  assert.equal(single.mainSetupUsPerForward, 110200 / 414);
  const two = text.replace('lanes=1', 'lanes=2').replace('rows=414', 'rows=828')
    .replace('decode_rows=384', 'decode_rows=768').replace('prefill_rows=30', 'prefill_rows=60');
  const [dual] = parseHostSetupCohorts(two);
  assert.equal(dual.partsUsPerForward.setup_moe_scan, single.partsUsPerForward.setup_moe_scan);
  assert.equal(parseHostSetupCohorts(`${text}\n${text}`).length, 2);
});
test('host setup rejects incomplete rounds, invalid counts and missing parts', () => {
  for (const text of ['', hostRunner, host, `${hostRunner}\n${hostRunner}\n${host}`,
    `${hostRunner.replace('steps=414', 'steps=0')}\n${host}`,
    `${hostRunner.replace('lanes=1', 'lanes=2')}\n${host}`,
    `${hostRunner}\n${host.replace('setup_moe_scan', 'other')}`]) {
    assert.throws(() => parseHostSetupCohorts(text));
  }
});
