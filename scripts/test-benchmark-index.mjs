import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { gunzipSync, brotliDecompressSync } from 'node:zlib';
import { buildIndex } from './build-benchmark-index.mjs';
import { prepareIndexCommit } from './publish-benchmark-index.mjs';
import { decodeIndex, indexRows, validateManifest } from './leaderboard/model.mjs';

const scripts = path.dirname(fileURLToPath(import.meta.url));
const python = process.env.MOE4ALL_PYTHON || 'python3';
const digest = data => createHash('sha256').update(data).digest('hex');
const git = (root, ...args) => execFileSync('git', ['-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', ...args], { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
async function fixture(t) {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'benchmark-index-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  const repo = path.join(dir, 'repo'); await fs.mkdir(repo);
  git(repo, 'init', '--initial-branch=main'); git(repo, 'config', 'core.autocrlf', 'true');
  await fs.writeFile(path.join(repo, '.gitattributes'), 'benchmark-database/index/v1/*.bin binary\nbenchmark-database/index/v1/manifest.json text eol=lf\n');
  await fs.writeFile(path.join(repo, 'code.txt'), 'fixture'); git(repo, 'add', '.'); git(repo, 'commit', '-m', 'initial');
  return { dir, repo };
}
async function report(repo, id = 'fixture-benchmark-001') {
  const vector = JSON.parse(await fs.readFile(path.join(scripts, 'fixtures/submission-vectors.json'), 'utf8')).benchmark;
  vector.id = id;
  const code = 'import sys,json;sys.path.insert(0,sys.argv[1]);from test_submission import encode;from submission_validation import validate_payload;p=json.load(sys.stdin);validate_payload("benchmark",p);sys.stdout.write(encode("benchmark",p))';
  const armor = execFileSync(python, ['-c', code, scripts], { input: JSON.stringify(vector), encoding: 'utf8', env: { ...process.env, PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' } });
  const filename = path.join(repo, 'benchmark-database/v1', vector.created_at_utc.slice(0, 7), id + '.m4b');
  await fs.mkdir(path.dirname(filename), { recursive: true }); await fs.writeFile(filename, armor);
  git(repo, 'add', '.'); git(repo, 'commit', '-m', 'report');
  return { filename, vector, commit: git(repo, 'rev-parse', 'HEAD') };
}
async function published(repo) {
  const output = path.join(repo, 'benchmark-database/index/v1');
  const manifest = JSON.parse(await fs.readFile(path.join(output, 'manifest.json'), 'utf8'));
  validateManifest(manifest);
  const prefix = `${manifest.package_commit}:benchmark-database/index/v1/`;
  const compressed = execFileSync('git', ['show', prefix + manifest.packages.gzip.url], { cwd: repo });
  assert.equal(digest(compressed), manifest.packages.gzip.sha256);
  const json = gunzipSync(compressed);
  assert.equal(digest(json), manifest.version);
  return { manifest, index: decodeIndex(JSON.parse(json)) };
}

test('real Git originals use the existing decoder and produce browser-compatible lossless packages', async t => {
  const { dir, repo } = await fixture(t), input = await report(repo);
  const output = path.join(dir, 'output'), manifest = await buildIndex({ directory: repo, output, python });
  assert.equal(manifest.source.commit, input.commit); assert.equal(manifest.counts.reports, 1);
  const gzip = await fs.readFile(path.join(output, manifest.packages.gzip.url));
  const brotli = await fs.readFile(path.join(output, manifest.packages.brotli.url));
  assert.deepEqual(gunzipSync(gzip), brotliDecompressSync(brotli));
  const index = decodeIndex(JSON.parse(gunzipSync(gzip)));
  assert.equal(index.reports[0].id, input.vector.id); assert.ok(indexRows(index).length);
  assert.deepEqual(await buildIndex({ directory: repo, output, python }), manifest);
  await fs.writeFile(path.join(output, manifest.packages.gzip.url), 'corrupt');
  await assert.rejects(buildIndex({ directory: repo, output, python }));
});

test('publish uses immutable package commits, is idempotent and preserves historical versions', async t => {
  const { repo } = await fixture(t); await report(repo);
  const first = await prepareIndexCommit({ directory: repo, python }), old = await published(repo);
  assert.equal(first.changed, true); assert.notEqual(first.commit, first.package_commit);
  assert.equal((await prepareIndexCommit({ directory: repo, python })).changed, false);
  assert.equal(git(repo, 'rev-parse', 'HEAD'), first.commit);
  await report(repo, 'fixture-benchmark-002');
  const second = await prepareIndexCommit({ directory: repo, python });
  assert.equal((await published(repo)).index.reports.length, 2);
  const historical = execFileSync('git', ['show', `${old.manifest.package_commit}:benchmark-database/index/v1/${old.manifest.packages.gzip.url}`], { cwd: repo });
  assert.equal(digest(historical), old.manifest.packages.gzip.sha256);
  assert.notEqual(second.version, first.version);
  assert.equal((await fs.readdir(path.join(repo, 'benchmark-database/index/v1'))).length, 3);
  assert.equal(git(repo, 'status', '--porcelain'), '');
});

test('empty tree remains idempotent and uncommitted input cannot be published', async t => {
  const { repo } = await fixture(t);
  await prepareIndexCommit({ directory: repo, python });
  assert.equal((await published(repo)).manifest.counts.reports, 0);
  assert.equal((await prepareIndexCommit({ directory: repo, python })).changed, false);
  await fs.writeFile(path.join(repo, 'code.txt'), 'dirty');
  await assert.rejects(prepareIndexCommit({ directory: repo, python }), /clean checkout/);
});

test('bad month, duplicate identity, symlink, executable and legacy JSON fail before publication', async t => {
  for (const kind of ['month', 'duplicate', 'mode', 'symlink', 'legacy']) {
    const { dir, repo } = await fixture(t), input = await report(repo);
    if (kind === 'month') {
      const target = path.join(repo, 'benchmark-database/v1/2000-01', path.basename(input.filename));
      await fs.mkdir(path.dirname(target), { recursive: true }); await fs.rename(input.filename, target);
    } else if (kind === 'duplicate') {
      const target = path.join(repo, 'benchmark-database/v1/2000-01', path.basename(input.filename));
      await fs.mkdir(path.dirname(target), { recursive: true }); await fs.copyFile(input.filename, target);
    } else if (kind === 'legacy') await fs.writeFile(path.join(repo, 'benchmark-database/v1/old.json'), '{}');
    git(repo, 'add', '.');
    const gitPath = path.relative(repo, input.filename).split(path.sep).join('/');
    if (kind === 'mode') git(repo, 'update-index', '--chmod=+x', gitPath);
    if (kind === 'symlink') git(repo, 'update-index', '--cacheinfo', '120000', git(repo, 'hash-object', input.filename), gitPath);
    git(repo, 'commit', '-m', 'invalid');
    await assert.rejects(buildIndex({ directory: repo, output: path.join(dir, 'output'), python }));
  }
});

test('concurrent main advancement rejects stale push; rebuilding current main includes every report', async t => {
  const { dir, repo } = await fixture(t), remote = path.join(dir, 'remote.git'), other = path.join(dir, 'other');
  await report(repo); git(dir, 'init', '--bare', remote); git(repo, 'remote', 'add', 'origin', remote); git(repo, 'push', 'origin', 'HEAD:main');
  git(dir, 'clone', '--branch', 'main', remote, other);
  await prepareIndexCommit({ directory: repo, python });
  await report(other, 'fixture-benchmark-002'); git(other, 'push', 'origin', 'HEAD:main');
  assert.throws(() => git(repo, 'push', 'origin', 'HEAD:main'));
  await prepareIndexCommit({ directory: other, python }); git(other, 'push', 'origin', 'HEAD:main');
  assert.equal((await published(other)).index.reports.length, 2);
  assert.equal(git(dir, '--git-dir=' + remote, 'rev-parse', 'main'), git(other, 'rev-parse', 'HEAD'));
});
