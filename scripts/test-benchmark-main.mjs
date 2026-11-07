import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { publishIndexToMain } from './publish-benchmark-main.mjs';

const scripts = path.dirname(fileURLToPath(import.meta.url));
const python = process.env.MOE4ALL_PYTHON || 'python3';
const git = (cwd, ...args) => execFileSync('git', ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', ...args], { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();

async function fixture(t, options = {}) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'benchmark-main-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const repo = path.join(root, 'repo'), remote = path.join(root, 'remote.git'); await fs.mkdir(repo);
  git(repo, 'init', '--initial-branch=main');
  await fs.writeFile(path.join(repo, '.gitattributes'), 'benchmark-database/index/v1/*.bin binary\nbenchmark-database/index/v1/manifest.json text eol=lf\n');
  const vector = JSON.parse(await fs.readFile(path.join(scripts, 'fixtures/submission-vectors.json'), 'utf8')).benchmark;
  const armor = execFileSync(python, ['-c', 'import sys,json;sys.path.insert(0,sys.argv[1]);from test_submission import encode;sys.stdout.write(encode("benchmark",json.load(sys.stdin)))', scripts], { input: JSON.stringify(vector), env: { ...process.env, PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' } });
  const reports = path.join(repo, 'benchmark-database/v1', vector.created_at_utc.slice(0, 7)); await fs.mkdir(reports, { recursive: true });
  await fs.writeFile(path.join(reports, vector.id + '.m4b'), armor);
  git(repo, 'add', '.'); git(repo, 'commit', '-m', 'reviewed original');
  const base = git(repo, 'rev-parse', 'HEAD');
  git(root, 'init', '--bare', remote); git(repo, 'remote', 'add', 'origin', remote); git(repo, 'push', 'origin', 'HEAD:main');
  const repository = 'Headmaster218/MoE4All', calls = []; let pull, filesRead = false, mergeCount = 0, readFailed = false;
  const fetcher = async (url, init) => {
    const endpoint = url.replace(`https://api.github.com/repos/${repository}/`, '');
    const body = init.body ? JSON.parse(init.body) : null;
    calls.push({ method: init.method, endpoint, body });
    assert.equal(init.redirect, 'error');
    if (options.forbidden) return Response.json({}, { status: 403 });
    if (options.readFlake && init.method === 'GET' && !readFailed) { readFailed = true; return Response.json({}, { status: 503 }); }
    if (endpoint === 'git/ref/heads/main') {
      if (options.advance && filesRead && git(remote, 'rev-parse', 'main') === base) {
        const advanced = git(remote, 'commit-tree', `${base}^{tree}`, '-p', base, '-m', 'another reviewed merge');
        git(remote, 'update-ref', 'refs/heads/main', advanced, base);
      }
      return Response.json({ object: { sha: git(remote, 'rev-parse', 'main') } });
    }
    if (endpoint === 'pulls' && init.method === 'POST') {
      assert.equal(git(remote, 'rev-parse', 'main'), base, 'No direct main push is allowed');
      assert.equal(body.base, 'main'); assert.match(body.head, /^automation\/benchmark-index\/[a-f0-9]{40}$/);
      pull = { number: 7, user: { login: options.actor || 'github-actions[bot]' },
        head: { ref: body.head, sha: options.head || git(remote, 'rev-parse', body.head), repo: { full_name: repository } },
        base: { ref: 'main', repo: { full_name: repository } }, merged: false, state: 'open' };
      if (options.lostCreate) return Response.json({}, { status: 503 });
      return Response.json(pull, { status: 201 });
    }
    if (endpoint.startsWith('pulls?')) return Response.json(pull ? [pull] : []);
    if (endpoint === 'pulls/7' && init.method === 'GET') return Response.json(pull);
    if (endpoint === 'pulls/7' && init.method === 'PATCH') { pull.state = body.state; return Response.json(pull); }
    if (endpoint === 'pulls/7/files?per_page=100') {
      filesRead = true;
      const files = git(repo, 'diff', '--no-renames', '--name-only', base, pull.head.sha).split('\n').map(filename => ({ filename, status: 'added' }));
      if (options.files) files[0].filename = options.files;
      return Response.json(files);
    }
    if (endpoint === 'pulls/7/merge' && init.method === 'PUT') {
      mergeCount++;
      assert.equal(body.sha, pull.head.sha); assert.equal(body.merge_method, 'merge'); assert.ok(filesRead);
      const merged = git(remote, 'commit-tree', `${pull.head.sha}^{tree}`, '-p', base, '-p', pull.head.sha, '-m', body.commit_title);
      git(remote, 'update-ref', 'refs/heads/main', merged, base);
      pull.merged = true; pull.merge_commit_sha = merged; pull.state = 'closed';
      return Response.json(options.lostMerge ? {} : { merged: true, sha: merged }, { status: options.lostMerge ? 503 : 200 });
    }
    if (endpoint.startsWith('contents/benchmark-database/index/v1/manifest.json?')) {
      const ref = new URL(url).searchParams.get('ref');
      return Response.json({ sha: git(remote, 'rev-parse', `${ref}:benchmark-database/index/v1/manifest.json`) });
    }
    if (endpoint.startsWith('git/refs/heads/automation/benchmark-index/') && init.method === 'DELETE') {
      git(remote, 'update-ref', '-d', 'refs/heads/' + pull.head.ref); return new Response(null, { status: 204 });
    }
    throw new Error(`Unexpected fixture request ${init.method} ${endpoint}`);
  };
  return { repo, remote, base, calls, count: () => mergeCount, run: () => publishIndexToMain({ directory: repo, repository, token: 'fixture-token-not-a-secret', python, fetcher, sleep: async () => {} }) };
}

test('trusted index alone is automatically merged to main, original report approval is not automated, repeat is idempotent', async t => {
  const f = await fixture(t), result = await f.run();
  assert.equal(result.published, true); assert.equal(result.cleanup_complete, true); assert.equal(f.count(), 1);
  assert.equal(git(f.remote, 'rev-parse', 'main'), result.merge_commit);
  assert.equal(git(f.remote, 'show', `${result.merge_commit}:benchmark-database/index/v1/manifest.json`), git(f.repo, 'show', `${result.commit}:benchmark-database/index/v1/manifest.json`));
  assert.match(git(f.repo, 'log', '-1', '--format=%B'), /\[skip ci\]/);
  assert.equal(git(f.remote, 'branch', '--list', 'automation/*'), '');
  git(f.repo, 'fetch', 'origin', 'main'); git(f.repo, 'merge', '--ff-only', 'FETCH_HEAD');
  assert.equal((await f.run()).published, false); assert.equal(f.count(), 1);
  assert.equal(f.calls.filter(c => c.endpoint === 'pulls' && c.method === 'POST').length, 1);
  assert.ok(f.calls.every(c => !c.endpoint.endsWith('/reviews')));
});

test('an advanced main is retried, not overwritten or merged with a stale generated snapshot', async t => {
  const f = await fixture(t, { advance: true }), result = await f.run();
  assert.equal(result.retry, true); assert.equal(result.cleanup_complete, true); assert.equal(f.count(), 0);
  assert.equal(f.calls.find(c => c.method === 'PATCH').body.state, 'closed');
});

test('unexpected changed files, PR actors or heads cannot be automatically merged', async t => {
  for (const options of [{ files: 'README.md' }, { actor: 'contributor' }, { head: 'b'.repeat(40) }]) {
    const f = await fixture(t, options);
    await assert.rejects(f.run(), /does not match|unbound/); assert.equal(f.count(), 0);
    assert.equal(git(f.remote, 'rev-parse', 'main'), f.base);
  }
});

test('lost creation and merge responses recover the existing exact PR without duplicate writes', async t => {
  for (const options of [{ lostCreate: true }, { lostMerge: true }, { readFlake: true }]) {
    const f = await fixture(t, options), result = await f.run();
    assert.equal(result.published, true); assert.equal(f.count(), 1);
    assert.equal(f.calls.filter(c => c.endpoint === 'pulls' && c.method === 'POST').length, 1);
  }
});

test('missing GitHub permission fails without weakening main protection or exposing credentials', async t => {
  const f = await fixture(t, { forbidden: true });
  await assert.rejects(f.run(), error => /HTTP 403/.test(error.message) && !error.message.includes('fixture-token'));
  assert.equal(git(f.repo, 'rev-parse', 'HEAD'), f.base); assert.equal(f.count(), 0);
});
