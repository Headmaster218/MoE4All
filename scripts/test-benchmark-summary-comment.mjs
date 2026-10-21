import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { MARKER, prepare, publish } from './benchmark-summary-comment.mjs';

const HEAD = 'a'.repeat(40), BASE = 'b'.repeat(40), MAIN = 'c'.repeat(40);
function fixture() {
  const pull = { number: 7, state: 'open', head: { sha: HEAD }, base: { sha: BASE, ref: 'main', repo: { full_name: 'owner/repo' } } };
  const calls = [], outputs = {}, comments = [];
  const github = { rest: {
    pulls: { get: async () => ({ data: structuredClone(pull) }) },
    git: { getRef: async () => ({ data: { object: { sha: MAIN } } }), getCommit: async () => ({ data: { parents: [{ sha: BASE }] } }) },
    issues: { listComments: () => {}, updateComment: async args => calls.push(['update', args]), createComment: async args => calls.push(['create', args]) }
  }, paginate: async () => comments };
  const context = { repo: { owner: 'owner', repo: 'repo' }, eventName: 'pull_request_target', payload: { pull_request: structuredClone(pull) } };
  const core = { setOutput: (key, value) => { outputs[key] = value; }, info: () => {}, summary: { addRaw: body => { calls.push(['summary', body]); return { write: async () => {} }; } } };
  const result = { status: 'ready', head_sha: HEAD, body: `${MARKER}\n### Benchmark 摘要\n` };
  return { github, context, core, result, prNumber: 7, expectedHead: HEAD, write: true, pull, calls, outputs, comments };
}

test('resolve current immutable commits, including fork-head PRs', async () => {
  const f = fixture(); f.pull.head.repo = { full_name: 'contributor/fork' };
  await prepare(f);
  assert.deepEqual(f.outputs, { enabled: 'true', number: '7', base: BASE, head: HEAD, baseline: MAIN });
});
test('stale events and other target branches are skipped', async () => {
  const f = fixture(); f.pull.head.sha = MAIN;
  await prepare(f); assert.equal(f.outputs.enabled, 'false');
  const g = fixture(); g.pull.base.ref = 'other';
  await prepare(g); assert.equal(g.outputs.enabled, 'false');
});
test('manual preview/backfill uses the merged PR original base', async () => {
  const f = fixture(); f.context.eventName = 'workflow_dispatch'; f.pull.state = 'closed'; f.pull.merged = true; f.pull.merge_commit_sha = MAIN;
  await prepare(f); assert.equal(f.outputs.base, BASE);
  f.write = false;
  assert.equal(await publish(f), 'preview');
  assert.equal(f.calls.filter(c => c[0] !== 'summary').length, 0);
});
test('never backfill a closed unmerged PR or accept invalid IDs', async () => {
  const f = fixture(); f.context.eventName = 'workflow_dispatch'; f.pull.state = 'closed';
  await assert.rejects(prepare(f), /open or merged/);
  for (const value of ['1;echo bad', '-1', '0', '1.5']) {
    await assert.rejects(prepare({ ...fixture(), context: { ...fixture().context, eventName: 'workflow_dispatch' }, prNumber: value }), /positive/);
  }
});
test('create one bot comment and update that same comment on reruns', async () => {
  const f = fixture();
  assert.equal(await publish(f), 'created');
  const created = f.calls.find(c => c[0] === 'create')[1];
  assert.equal(created.issue_number, 7);
  f.comments.push({ id: 123, body: created.body, user: { type: 'Bot', login: 'github-actions[bot]' } });
  assert.equal(await publish(f), 'updated');
  assert.equal(f.calls.filter(c => c[0] === 'update')[0][1].comment_id, 123);
});
test('a user cannot impersonate the summary by copying its marker', async () => {
  const f = fixture();
  f.comments.push({ id: 11, body: f.result.body, user: { type: 'User', login: 'contributor' } });
  assert.equal(await publish(f), 'created');
});
test('no stale output, cross-PR identity mismatch, oversized comments or unknown status', async () => {
  const f = fixture(); f.pull.head.sha = MAIN;
  assert.equal(await publish(f), 'stale'); assert.equal(f.calls.length, 0);
  await assert.rejects(publish({ ...fixture(), expectedHead: MAIN }), /identity/);
  for (const result of [{ status: 'unknown', head_sha: HEAD, body: `${MARKER}\n` }, { status: 'ready', head_sha: HEAD, body: `${MARKER}\n${'中'.repeat(22000)}` }]) {
    await assert.rejects(publish({ ...fixture(), result }), /Invalid/);
  }
});
test('head changed while finding a comment cannot overwrite it', async () => {
  const f = fixture();
  f.github.paginate = async () => { f.pull.head.sha = MAIN; return []; };
  assert.equal(await publish(f), 'stale');
  assert.equal(f.calls.filter(c => ['create', 'update'].includes(c[0])).length, 0);
});
test('invalid replacement updates the existing summary, code-only PRs do not post', async () => {
  const f = fixture(); f.result.status = 'invalid'; f.result.body = `${MARKER}\n报告校验未通过。\n`;
  f.comments.push({ id: 123, body: `${MARKER}\nold`, user: { type: 'Bot', login: 'github-actions[bot]' } });
  assert.equal(await publish(f), 'updated');
  const g = fixture(); g.result = { status: 'skipped', head_sha: HEAD };
  assert.equal(await publish(g), 'skipped'); assert.equal(g.calls.length, 0);
});
test('removing a report retracts the old summary rather than leaving stale scores', async () => {
  const f = fixture(); f.result = { status: 'skipped', head_sha: HEAD };
  f.comments.push({ id: 123, body: `${MARKER}\nold`, user: { type: 'Bot', login: 'github-actions[bot]' } });
  assert.equal(await publish(f), 'updated');
  assert.match(f.calls.find(c => c[0] === 'update')[1].body, /旧摘要已撤下/);
});
test('workflow only checks out trusted source and cannot push code or release', () => {
  const text = readFileSync(new URL('../.github/workflows/benchmark-summary.yml', import.meta.url), 'utf8');
  assert.match(text, /pull_request_target:/);
  assert.match(text, /github\.event\.pull_request\.base\.sha/);
  assert.doesNotMatch(text, /ref:.*head\.(sha|ref)/);
  assert.match(text, /persist-credentials: false/);
  assert.doesNotMatch(text, /contents: write|secrets\.|release create|git push|cargo /);
});
