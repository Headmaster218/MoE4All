const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { createRequire } = require('node:module');
const { zipSync, strToU8 } = require('./vendor/fflate.cjs');
const { unpack, restoreLinks, remapManagedPath } = require('./Prepare-BundledRuntime.cjs');
const native = (filename) => path.toNamespacedPath(filename);

function fixture(t) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'moe4all-runtime-'));
  t.after(() => fs.rmSync(native(directory), { recursive: true, force: true }));
  return directory;
}
function write(filename, text) {
  fs.mkdirSync(native(path.dirname(filename)), { recursive: true });
  fs.writeFileSync(native(filename), text);
}
function payload(harness, entries, files) {
  const bytes = zipSync(entries);
  write(path.join(harness, 'runtime', 'data.zip'), bytes);
  return { file: 'runtime/data.zip', root: 'runtime/dsh', files,
    sha256: crypto.createHash('sha256').update(bytes).digest('hex') };
}

test('streamed runtime payload restores paths over 260 characters and empty directories', async (t) => {
  const base = fixture(t);
  const harness = path.join(base, 'deep-path-'.repeat(14), 'harness');
  const relative = 'node_modules/.pnpm/long-package-name/node_modules/example/package.json';
  assert.ok(path.join(harness, 'runtime/dsh', relative).length > 260);
  const archive = payload(harness, { [relative]: strToU8('{"name":"example"}'), 'workspace/': new Uint8Array() }, 1);
  await unpack(harness, archive);
  assert.equal(fs.readFileSync(native(path.join(harness, 'runtime/dsh', relative)), 'utf8'), '{"name":"example"}');
  assert.ok(fs.statSync(native(path.join(harness, 'runtime/dsh/workspace'))).isDirectory());
  await unpack(harness, archive);
});

test('runtime payload completes concurrent compressed file writes before marking ready', async (t) => {
  const harness = path.join(fixture(t), 'harness');
  const entries = {};
  for (let index = 0; index < 24; index++) {
    entries[`files/${index}.txt`] = strToU8(`${index}:` + 'payload-'.repeat(16384));
  }
  const archive = payload(harness, entries, 24);
  let completed = 0;
  await unpack(harness, archive, { extracted() { completed++; } });
  assert.equal(completed, 24);
  assert.equal(fs.readFileSync(native(path.join(harness, 'runtime/dsh/files/23.txt')), 'utf8').startsWith('23:payload-'), true);
  assert.equal(fs.readFileSync(native(path.join(harness, 'runtime/dsh/.moe4all-payload-ready')), 'utf8').trim(), archive.sha256);
});

test('runtime payload rejects checksum mismatch, path traversal and duplicate Windows paths', async (t) => {
  const harness = path.join(fixture(t), 'harness');
  let archive = payload(harness, { 'file.txt': strToU8('data') }, 1);
  await assert.rejects(unpack(harness, { ...archive, sha256: '0'.repeat(64) }), /SHA256 mismatch/);
  archive = payload(harness, { '../outside.txt': strToU8('data') }, 1);
  await assert.rejects(unpack(harness, archive), /Unsafe bundled relative path/);
  archive = payload(harness, { 'a.txt': strToU8('1'), 'A.txt': strToU8('2') }, 2);
  await assert.rejects(unpack(harness, archive), /Duplicate runtime payload path/);
  assert.equal(fs.existsSync(native(path.join(harness, 'runtime/dsh/.moe4all-payload-ready'))), false);
});

test('an initialized installation repairs commander junctions after its original directory disappears', (t) => {
  const base = fixture(t);
  const original = path.join(base, 'original');
  const moved = path.join(base, 'moved');
  const dsh = path.join(original, 'runtime/dsh');
  const relative = 'node_modules/.pnpm/commander@15.0.0/node_modules/commander';
  write(path.join(dsh, 'package.json'), '{}');
  write(path.join(dsh, relative, 'package.json'), '{"name":"commander","main":"index.js"}');
  write(path.join(dsh, relative, 'index.js'), 'module.exports = 42;');
  write(path.join(original, 'runtime/dsh-links.json'), JSON.stringify({ schema: 1,
    links: [{ path: 'node_modules/commander', target: relative }] }));
  restoreLinks(original);
  assert.equal(createRequire(path.join(dsh, 'package.json'))('commander'), 42);
  fs.renameSync(native(original), native(moved));
  const movedDsh = path.join(moved, 'runtime/dsh');
  assert.throws(() => createRequire(path.join(movedDsh, 'package.json')).resolve('commander'), /Cannot find module/);
  restoreLinks(moved);
  assert.equal(createRequire(path.join(movedDsh, 'package.json'))('commander'), 42);
  assert.equal(fs.existsSync(native(original)), false);
});

test('managed state paths relocate without rewriting external or relative paths', (t) => {
  const base = fixture(t);
  const oldState = path.join(base, 'old-state');
  const newState = path.join(base, 'new-state');
  const external = path.join(base, 'external', 'model.gguf');
  const mappings = [[oldState, newState]];
  assert.equal(remapManagedPath(path.join(oldState, 'models', 'model.gguf'), mappings),
    path.join(newState, 'models', 'model.gguf'));
  assert.equal(remapManagedPath(external, mappings), external);
  assert.equal(remapManagedPath('kv-sessions', mappings), 'kv-sessions');
});
