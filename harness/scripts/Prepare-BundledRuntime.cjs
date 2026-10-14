const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const os = require('node:os');
const zlib = require('node:zlib');
const { createRequire } = require('node:module');
const { Unzip } = require('./vendor/fflate.cjs');

const PROGRESS_PREFIX = '@@MOE4ALL_PROGRESS';
const extractionConcurrency = Math.max(2, Math.min(8,
  typeof os.availableParallelism === 'function' ? os.availableParallelism() : os.cpus().length));

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

function payloadReady(destination, sha256) {
  const marker = path.join(destination, '.moe4all-payload-ready');
  return exists(marker) && fs.readFileSync(native(marker), 'utf8').trim() === sha256;
}

function createProgress(total) {
  let completed = 0;
  let lastEmit = 0;
  let nextLog = 10;
  const emit = (message, force = false) => {
    const now = Date.now();
    if (!force && now - lastEmit < 100) return;
    lastEmit = now;
    console.log(`${PROGRESS_PREFIX}\t${completed}\t${total}\t${message}`);
    const percent = total > 0 ? Math.floor(completed * 100 / total) : 100;
    if (completed > 0 && (percent >= nextLog || completed === total)) {
      console.log(`Bundled runtime extraction: ${percent}% (${completed}/${total} files)`);
      while (nextLog <= percent) nextLog += 10;
    }
  };
  return {
    begin() { emit('Verifying bundled runtime', true); },
    extracted() { completed++; emit('Unpacking bundled runtime'); },
    finish() { completed = total; emit('Bundled runtime ready', true); },
  };
}

function createNodeUnzipInflate(decoders, block, unblock) {
  class NodeUnzipInflate {
    constructor(name) {
      this.name = name;
      this.inputBlocked = false;
      this.ended = false;
      this.stream = zlib.createInflateRaw();
      decoders.set(name, this);
      this.stream.on('data', (data) => this.ondata(null, data, false));
      this.stream.once('end', () => {
        this.ended = true;
        this.ondata(null, new Uint8Array(), true);
        decoders.delete(name);
      });
      this.stream.once('error', (error) => {
        this.ended = true;
        if (this.inputBlocked) { this.inputBlocked = false; unblock(); }
        decoders.delete(name);
        this.ondata(error, null, false);
      });
    }

    push(chunk, final) {
      if (this.ended) return;
      try {
        if (chunk.length && !this.stream.write(chunk) && !this.inputBlocked) {
          this.inputBlocked = true;
          block();
          this.stream.once('drain', () => {
            if (!this.inputBlocked) return;
            this.inputBlocked = false;
            unblock();
          });
        }
        if (final) this.stream.end();
      } catch (error) {
        this.stream.destroy(error);
      }
    }

    pause() { this.stream.pause(); }
    resume() { if (!this.ended) this.stream.resume(); }
    terminate() {
      if (this.inputBlocked) { this.inputBlocked = false; unblock(); }
      this.ended = true;
      decoders.delete(this.name);
      this.stream.destroy();
    }
  }
  NodeUnzipInflate.compression = 8;
  return NodeUnzipInflate;
}

async function unpack(harness, archive, progress) {
  if (!/^[a-f0-9]{64}$/.test(archive.sha256)) throw new Error('Invalid runtime payload SHA256.');
  if (!Number.isSafeInteger(archive.files) || archive.files < 0) throw new Error('Invalid runtime payload file count.');
  const source = contained(harness, archive.file);
  const destination = contained(harness, archive.root);
  const marker = path.join(destination, '.moe4all-payload-ready');
  if (payloadReady(destination, archive.sha256)) return;
  if (await hashFile(source) !== archive.sha256) throw new Error(`Runtime payload SHA256 mismatch: ${archive.file}`);
  mkdir(destination);
  console.log(`Unpacking bundled runtime: ${archive.file}`);
  const outputs = new Set();
  const decoders = new Map();
  const names = new Set();
  const queue = [];
  const waiters = [];
  let files = 0;
  let active = 0;
  let blocked = 0;
  let inputFinished = false;
  let failure;
  let resolveCompletion;
  let rejectCompletion;
  const completion = new Promise((resolve, reject) => {
    resolveCompletion = resolve;
    rejectCompletion = reject;
  });
  completion.catch(() => {});

  const wake = () => {
    for (const resolve of waiters.splice(0)) resolve();
  };
  const block = () => { blocked++; };
  const unblock = () => { blocked = Math.max(0, blocked - 1); wake(); };
  const fail = (error) => {
    if (failure) return;
    failure = error instanceof Error ? error : new Error(String(error));
    rejectCompletion(failure);
    wake();
  };
  const maybeComplete = () => {
    if (failure || !inputFinished || active || queue.length) return;
    if (files !== archive.files) fail(new Error(`Runtime payload is incomplete: ${archive.file}`));
    else resolveCompletion();
  };
  const waitForFlow = async () => {
    while (!failure && (queue.length || blocked)) {
      await new Promise((resolve) => waiters.push(resolve));
    }
    if (failure) throw failure;
  };

  let pump;
  const startFile = (file, target) => {
    active++;
    mkdir(path.dirname(target));
    const output = fs.createWriteStream(native(target), { flags: 'w' });
    outputs.add(output);
    let outputBlocked = false;
    const releaseOutputBlock = () => {
      if (!outputBlocked) return;
      outputBlocked = false;
      unblock();
    };
    output.once('error', (error) => { releaseOutputBlock(); fail(error); });
    output.once('close', () => outputs.delete(output));
    output.once('finish', () => {
      releaseOutputBlock();
      active--;
      files++;
      if (progress) progress.extracted();
      pump();
      wake();
      maybeComplete();
    });
    file.ondata = (error, data, final) => {
      if (error) { fail(error); return; }
      if (failure) return;
      try {
        if (data.length && !output.write(data) && !outputBlocked) {
          outputBlocked = true;
          block();
          const decoder = decoders.get(file.name);
          if (decoder) decoder.pause();
          output.once('drain', () => {
            releaseOutputBlock();
            if (decoder) decoder.resume();
          });
        }
        if (final) output.end();
      } catch (writeError) { fail(writeError); }
    };
    try { file.start(); } catch (error) { fail(error); }
  };
  pump = () => {
    while (!failure && active < extractionConcurrency && queue.length) {
      const { file, target } = queue.shift();
      startFile(file, target);
    }
    wake();
  };

  const unzip = new Unzip((file) => {
    const target = contained(destination, file.name);
    const key = target.toLowerCase();
    if (names.has(key)) throw new Error(`Duplicate runtime payload path: ${file.name}`);
    names.add(key);
    if (file.name.endsWith('/')) {
      mkdir(target);
      file.ondata = (error) => { if (error) fail(error); };
      file.start();
      return;
    }
    queue.push({ file, target });
    pump();
  });
  unzip.register(createNodeUnzipInflate(decoders, block, unblock));
  try {
    for await (const chunk of fs.createReadStream(native(source))) {
      await waitForFlow();
      unzip.push(chunk, false);
    }
    await waitForFlow();
    unzip.push(new Uint8Array(), true);
    inputFinished = true;
    maybeComplete();
    await completion;
    write(marker, archive.sha256 + '\n');
  } finally {
    for (const decoder of decoders.values()) decoder.terminate();
    for (const output of outputs) output.destroy();
  }
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

function remapManagedPath(value, mappings) {
  if (typeof value !== 'string' || !path.isAbsolute(value)) return value;
  const resolved = path.resolve(value);
  const lowered = resolved.toLowerCase();
  for (const [source, destination] of mappings) {
    const prefix = path.resolve(source);
    const loweredPrefix = prefix.toLowerCase();
    if (lowered === loweredPrefix) return destination;
    if (lowered.startsWith(loweredPrefix + path.sep)) return path.join(destination, resolved.slice(prefix.length + 1));
  }
  return value;
}

function prepareProfile(install, previousInstall, harness, state, dsh) {
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
  const models = path.join(install, 'models');
  const cache = path.join(install, 'cache', 'kv-sessions');
  const mappings = [
    [path.join(install, 'harness', 'agent-state', 'home', 'moe4all-models'), models],
    [path.join(state, 'home', 'moe4all-models'), models],
    [path.join(install, 'harness', 'agent-state'), state],
  ];
  if (previousInstall && path.resolve(previousInstall).toLowerCase() !== path.resolve(install).toLowerCase()) {
    mappings.unshift(
      [path.join(previousInstall, 'models'), models],
      [path.join(previousInstall, 'cache'), path.join(install, 'cache')],
      [path.join(previousInstall, 'data'), state],
      [path.join(previousInstall, 'harness', 'agent-state'), state],
    );
  }
  for (const entry of patch) {
    if (entry.id === 'moe4all-engine' && entry.config?.endpoint === 'http://127.0.0.1:1234/v1') {
      entry.config.endpoint = 'http://127.0.0.1:8080/v1'; changed = true;
    }
    if (entry.id === 'dsh-mneme-moe4all' && entry.config && !Object.hasOwn(entry.config, 'vectorDefaults')) {
      entry.config.vectorDefaults = { enabled: true, baseUrl: 'http://127.0.0.1:8080/v1', apiKey: 'moe4all-local', model: 'Qwen3-Embedding-0.6B-Q8_0' };
      changed = true;
    }
    if (entry.id === 'moe4all-engine' && entry.config) {
      const directory = remapManagedPath(entry.config.modelDirectory, mappings);
      if (directory !== entry.config.modelDirectory) { entry.config.modelDirectory = directory; changed = true; }
      if (Array.isArray(entry.config.modelDirectories)) {
        const directories = entry.config.modelDirectories.map(value => remapManagedPath(value, mappings));
        if (directories.some((value, index) => value !== entry.config.modelDirectories[index])) {
          entry.config.modelDirectories = directories; changed = true;
        }
      }
      if (Array.isArray(entry.config.arguments)) {
        const arguments_ = entry.config.arguments.map((value) => {
          if (value === 'kv.session_cache_dir=kv-sessions') return `kv.session_cache_dir=${cache}`;
          if (typeof value === 'string' && value.startsWith('kv.session_cache_dir=')) {
            const current = value.slice('kv.session_cache_dir='.length);
            const relocated = remapManagedPath(current, mappings);
            return `kv.session_cache_dir=${relocated}`;
          }
          return remapManagedPath(value, mappings);
        });
        if (arguments_.some((value, index) => value !== entry.config.arguments[index])) {
          entry.config.arguments = arguments_; changed = true;
        }
      }
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
  const resolvedInstall = path.resolve(install);
  const resolvedState = path.resolve(state);
  const harness = path.join(resolvedInstall, 'harness');
  const rootMarker = path.join(resolvedState, 'product-root.json');
  let previousInstall;
  if (exists(rootMarker)) {
    try {
      const marker = readJson(rootMarker);
      if (marker.schema === 1 && typeof marker.root === 'string' && path.isAbsolute(marker.root)) previousInstall = marker.root;
    } catch { /* A malformed hint must not block startup; current paths remain authoritative. */ }
  }
  const payload = path.join(harness, 'runtime', 'runtime-payload.json');
  if (exists(payload)) {
    const manifest = readJson(payload);
    if (manifest.schema !== 1 || !Array.isArray(manifest.archives)) throw new Error('Invalid runtime payload manifest.');
    const pending = manifest.archives.filter((archive) => {
      if (!/^[a-f0-9]{64}$/.test(archive.sha256)) throw new Error('Invalid runtime payload SHA256.');
      return !payloadReady(contained(harness, archive.root), archive.sha256);
    });
    if (pending.length) {
      const total = pending.reduce((sum, archive) => {
        if (!Number.isSafeInteger(archive.files) || archive.files < 0) throw new Error('Invalid runtime payload file count.');
        return sum + archive.files;
      }, 0);
      const progress = createProgress(total);
      progress.begin();
      await Promise.all(pending.map((archive) => unpack(harness, archive, progress)));
      progress.finish();
    }
  }
  const dsh = restoreLinks(harness);
  prepareProfile(resolvedInstall, previousInstall, harness, resolvedState, dsh);
  await prepareEngine(harness, resolvedState);
  write(rootMarker, JSON.stringify({ schema: 1, root: resolvedInstall }, null, 2) + '\n');
  console.log('Bundled MoE4All runtime is ready.');
}

if (require.main === module) {
  const [install, state] = process.argv.slice(2);
  if (!install || !state) { console.error('InstallRoot and StateRoot are required.'); process.exitCode = 1; }
  else prepare(install, state).catch((error) => { console.error(error.stack); process.exitCode = 1; });
}
module.exports = { contained, unpack, ensureJunction, restoreLinks, remapManagedPath, prepare };
