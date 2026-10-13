const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { createRequire } = require('node:module');
const { Unzip, UnzipInflate } = require('./vendor/fflate.cjs');

const native = (value) => path.toNamespacedPath(path.resolve(value));
const exists = (value) => fs.existsSync(native(value));
const readJson = (value) => JSON.parse(fs.readFileSync(native(value), 'utf8'));
const write = (value, text) => fs.writeFileSync(native(value), text, 'utf8');
const mkdir = (value) => fs.mkdirSync(native(value), { recursive: true });

function contained(base, relative) {
  if (typeof relative !== 'string' || path.isAbsolute(relative) || /[:\0]/.test(relative)
    || relative.split(/[\\/]/).includes('..')) throw new Error(`Unsafe bundled relative path: ${relative}`);
  const result = path.resolve(base, relative);
  if (!result.toLowerCase().startsWith(path.resolve(base).toLowerCase() + path.sep)) {
    throw new Error(`Bundled path escapes its root: ${relative}`);
  }
  return result;
}

async function hashFile(filename) {
  const hash = crypto.createHash('sha256');
  for await (const chunk of fs.createReadStream(native(filename))) hash.update(chunk);
  return hash.digest('hex');
}

async function unpack(harness, archive) {
  if (!/^[a-f0-9]{64}$/.test(archive.sha256)) throw new Error('Invalid runtime payload SHA256.');
  const source = contained(harness, archive.file);
  const destination = contained(harness, archive.root);
  const marker = path.join(destination, '.moe4all-payload-ready');
  if (exists(marker) && fs.readFileSync(native(marker), 'utf8').trim() === archive.sha256) return;
  if (await hashFile(source) !== archive.sha256) throw new Error(`Runtime payload SHA256 mismatch: ${archive.file}`);
  mkdir(destination);
  console.log(`Unpacking bundled runtime: ${archive.file}`);
  const descriptors = new Set();
  const names = new Set();
  let files = 0;
  const unzip = new Unzip((file) => {
    const target = contained(destination, file.name);
    const key = target.toLowerCase();
    if (names.has(key)) throw new Error(`Duplicate runtime payload path: ${file.name}`);
    names.add(key);
    if (file.name.endsWith('/')) {
      mkdir(target);
      file.ondata = (error) => { if (error) throw error; };
      file.start();
      return;
    }
    mkdir(path.dirname(target));
    const fd = fs.openSync(native(target), 'w');
    descriptors.add(fd);
    file.ondata = (error, data, final) => {
      if (error) throw error;
      let offset = 0;
      while (offset < data.length) offset += fs.writeSync(fd, data, offset, data.length - offset);
      if (final) { fs.closeSync(fd); descriptors.delete(fd); files++; }
    };
    file.start();
  });
  unzip.register(UnzipInflate);
  try {
    for await (const chunk of fs.createReadStream(native(source))) unzip.push(chunk, false);
    unzip.push(new Uint8Array(), true);
    if (descriptors.size || files !== archive.files) throw new Error(`Runtime payload is incomplete: ${archive.file}`);
    write(marker, archive.sha256 + '\n');
  } finally { for (const fd of descriptors) fs.closeSync(fd); }
}

function ensureJunction(link, target, allowDirectory = false) {
  let existing;
  try { existing = fs.lstatSync(native(link)); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  if (existing) {
    if (!existing.isSymbolicLink()) {
      if (allowDirectory && existing.isDirectory()) return;
      throw new Error(`Bundled junction path is occupied: ${link}`);
    }
    const actual = path.resolve(path.dirname(link), fs.readlinkSync(native(link)));
    if (native(actual).toLowerCase() === native(target).toLowerCase()) return;
    fs.unlinkSync(native(link));
  }
  mkdir(path.dirname(link));
  fs.symlinkSync(native(target), native(link), 'junction');
}

function restoreLinks(harness) {
  const dsh = path.join(harness, 'runtime', 'dsh');
  const manifest = readJson(path.join(harness, 'runtime', 'dsh-links.json'));
  if (manifest.schema !== 1 || !Array.isArray(manifest.links)) throw new Error('Invalid bundled DSH link manifest.');
  // Absolute junction targets must be repaired after moving an initialized installation.
  for (const link of manifest.links) {
    const location = contained(dsh, link.path);
    const target = contained(dsh, link.target);
    if (!exists(target) || !fs.statSync(native(target)).isDirectory()) {
      throw new Error(`Bundled DSH junction target is missing: ${link.target}. Extract the complete package again.`);
    }
    ensureJunction(location, target);
  }
  write(path.join(dsh, '.moe4all-links-ready'), 'schema=1\n');
  return dsh;
}

function runtimePackages(dsh) {
  const packages = new Map();
  const add = (directory) => {
    if (!exists(path.join(directory, 'package.json'))) return;
    const { name } = readJson(path.join(directory, 'package.json'));
    if (/^@deepseek-ai\/[0-9A-Za-z._-]+$/.test(name) && !packages.has(name)) packages.set(name, directory);
  };
  const pnpm = path.join(dsh, 'node_modules', '.pnpm');
  for (const entry of fs.readdirSync(native(pnpm), { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const scope = path.join(pnpm, entry.name, 'node_modules', '@deepseek-ai');
    if (!exists(scope)) continue;
    for (const item of fs.readdirSync(native(scope), { withFileTypes: true })) {
      if (item.isDirectory() && !item.isSymbolicLink()) add(path.join(scope, item.name));
    }
  }
  const materialized = path.join(dsh, '.moe4all-workspace');
  if (exists(materialized)) {
    for (const entry of fs.readdirSync(native(materialized), { withFileTypes: true })) {
      if (entry.isDirectory()) add(path.join(materialized, entry.name));
    }
  }
  for (const name of ['cosmokit', 'dsh-base', 'dsh-scope', 'dsh-timeout', 'dsh-web-app', 'schemastery']) {
    if (!packages.has(`@deepseek-ai/${name}`)) throw new Error(`Bundled DSH package store is incomplete: ${name}`);
  }
  return packages;
}

function prepareProfile(harness, state, dsh) {
  const modules = path.join(state, 'home', 'profiles', 'web', 'node_modules');
  if (!exists(modules)) {
    const seed = path.join(harness, 'distribution', 'prebuilt-state');
    if (!exists(seed)) throw new Error('The bundled Agent state seed is missing.');
    mkdir(state);
    fs.cpSync(native(seed), native(state), { recursive: true, force: false, errorOnExist: false });
  }
  for (const [name, target] of runtimePackages(dsh)) {
    const short = name.slice('@deepseek-ai/'.length);
    ensureJunction(path.join(dsh, 'node_modules', '@deepseek-ai', short), target, true);
    ensureJunction(path.join(modules, '@deepseek-ai', short), target, true);
  }
  const requireDsh = createRequire(path.join(dsh, 'package.json'));
  requireDsh.resolve('commander');
  const patchPath = path.join(state, 'home', 'profiles', 'web', 'cordis.patch.yml');
  if (!exists(patchPath)) return;
  const yaml = requireDsh('js-yaml');
  const patch = yaml.load(fs.readFileSync(native(patchPath), 'utf8'));
  let changed = false;
  for (const entry of patch) {
    if (entry.id === 'moe4all-engine' && entry.config?.endpoint === 'http://127.0.0.1:1234/v1') {
      entry.config.endpoint = 'http://127.0.0.1:8080/v1'; changed = true;
    }
    if (entry.id === 'dsh-mneme-moe4all' && entry.config && !Object.hasOwn(entry.config, 'vectorDefaults')) {
      entry.config.vectorDefaults = { enabled: true, baseUrl: 'http://127.0.0.1:8080/v1', apiKey: 'moe4all-local', model: 'Qwen3-Embedding-0.6B-Q8_0' };
      changed = true;
    }
  }
  if (changed) write(patchPath, yaml.dump(patch, { lineWidth: -1 }));
}

async function prepareEngine(harness, state) {
  const manifestPath = path.join(harness, 'runtime', 'engine', 'bundled-engine.json');
  if (!exists(manifestPath)) return;
  const engine = readJson(manifestPath);
  if (engine.schema !== 1 || !/^v\d+\.\d+\.\d+$/.test(engine.tag) || !/^[a-f0-9]{64}$/.test(engine.sha256)) {
    throw new Error('The bundled Engine manifest is invalid.');
  }
  const source = contained(path.join(harness, 'runtime', 'engine'), engine.executable);
  const managed = path.join(state, 'home', 'moe4all-engine');
  const metadata = path.join(managed, 'installed.json');
  if (exists(metadata)) {
    const installed = readJson(metadata);
    if (/^v\d+\.\d+\.\d+$/.test(installed.tag)) {
      const relocatedDirectory = path.join(managed, 'releases', installed.tag);
      const relocatedExe = path.join(relocatedDirectory, 'infr.exe');
      const suffix = `/moe4all-engine/releases/${installed.tag}/infr.exe`;
      if (exists(relocatedExe) && typeof installed.executable === 'string'
        && installed.executable.replaceAll('\\', '/').endsWith(suffix) && installed.executable !== relocatedExe) {
        installed.executable = relocatedExe;
        installed.workingDirectory = relocatedDirectory;
        write(metadata, JSON.stringify(installed, null, 2) + '\n');
      }
    }
    return;
  }
  if (await hashFile(source) !== engine.sha256) throw new Error('The bundled Engine SHA256 does not match its manifest.');
  const directory = path.join(managed, 'releases', engine.tag);
  const executable = path.join(directory, 'infr.exe');
  mkdir(directory);
  fs.copyFileSync(native(source), native(executable));
  write(metadata, JSON.stringify({ tag: engine.tag, name: engine.name, executable, workingDirectory: directory,
    installedAt: new Date().toISOString(), sourceUrl: 'bundled:moe4all-agent' }, null, 2) + '\n');
}

async function prepare(install, state) {
  const harness = path.join(path.resolve(install), 'harness');
  const payload = path.join(harness, 'runtime', 'runtime-payload.json');
  if (exists(payload)) {
    const manifest = readJson(payload);
    if (manifest.schema !== 1 || !Array.isArray(manifest.archives)) throw new Error('Invalid runtime payload manifest.');
    for (const archive of manifest.archives) await unpack(harness, archive);
  }
  const dsh = restoreLinks(harness);
  prepareProfile(harness, path.resolve(state), dsh);
  await prepareEngine(harness, path.resolve(state));
  console.log('Bundled MoE4All runtime is ready.');
}

if (require.main === module) {
  const [install, state] = process.argv.slice(2);
  if (!install || !state) { console.error('InstallRoot and StateRoot are required.'); process.exitCode = 1; }
  else prepare(install, state).catch((error) => { console.error(error.stack); process.exitCode = 1; });
}
module.exports = { contained, unpack, ensureJunction, restoreLinks, prepare };
