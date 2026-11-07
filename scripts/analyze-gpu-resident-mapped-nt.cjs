const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const log = path.join(artifacts, 'stage4-mapped-nt-module.log');
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const text = fs.readFileSync(log, 'utf8');
const samples = [...text.matchAll(/mapped_nt_copy experts=(\d+) bytes=(\d+) pass=(\d+) serial=(true|false) nt=(true|false) us=([\d.]+)/g)]
  .map(([, experts, bytes, pass, serial, nt, us]) => ({ experts: +experts, bytes: +bytes, pass: +pass, serial: serial === 'true', nt: nt === 'true', us: +us }));
if (samples.length !== 120 || !text.includes('test result: ok. 1 passed; 0 failed;')) throw new Error('Incomplete NT module screen');
const rows = [1, 2, 3, 4, 8].map(experts => ({ experts,
  modes: [false, true].flatMap(serial => [false, true].map(nt => {
    const values = samples.filter(s => s.experts === experts && s.serial === serial && s.nt === nt);
    if (values.length !== 6 || new Set(values.map(s => s.pass)).size !== 6 || values.some(s => !Number.isFinite(s.us) || s.us <= 0)) {
      throw new Error('Malformed NT passes');
    }
    return { serial, nt, meanUs: values.reduce((n, s) => n + s.us, 0) / values.length, passesUs: values.map(s => s.us) };
  })),
}));
const record = { log, logHash: digest(fs.readFileSync(log)), rows,
  scope: '10-thread Rayon pool, coherent mapped DEVICE_LOCAL VRAM, 6 alternating passes, GPU payload readback outside timing; not inference',
  decision: 'Streaming stores alone change measured means by less than 1 percent; do not integrate into runtime',
};
if (process.env.NT_ARCHIVE === '1') {
  const prefix = path.join(artifacts, 'stage4-mapped-nt-module');
  if (fs.existsSync(`${prefix}-source.patch`)) throw new Error('NT source archive already exists');
  const diff = spawnSync('git', ['diff', '--binary', 'HEAD'], { cwd: root, encoding: 'utf8', windowsHide: true, maxBuffer: 16 * 1024 * 1024 });
  const head = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8', windowsHide: true });
  if (diff.status !== 0 || head.status !== 0) throw new Error('Cannot archive NT source');
  fs.writeFileSync(`${prefix}-source.patch`, diff.stdout);
  const binary = 'D:/AIinfr/infr/target/gpu-resident-experiment/release/deps/infr_vulkan-18125551459bd3da.exe';
  const frozen = 'D:/AIinfr/infr/target/gpu-resident-experiment/release/infr-vulkan-mapped-nt-tests.exe';
  if (fs.existsSync(frozen)) throw new Error('Frozen NT test binary already exists');
  fs.copyFileSync(binary, frozen);
  record.build = { head: head.stdout.trim(), binary: frozen, binaryHash: digest(fs.readFileSync(frozen)),
    toolchain: '1.97.1-x86_64-pc-windows-msvc', flags: '-C target-cpu=native',
    nonIgnoredTests: 221, ignoredTests: 111, sourcePatch: `${prefix}-source.patch`, sourcePatchHash: digest(Buffer.from(diff.stdout)),
    sourceHashes: Object.fromEntries(['crates/infr-core/src/timeline.rs', 'crates/infr-core/src/config/mod.rs',
      'crates/infr-vulkan/src/transfer.rs', 'crates/infr-vulkan/src/adapter.rs'].map(file => [file, digest(fs.readFileSync(path.join(root, file)))])) };
  fs.writeFileSync(`${prefix}-archive.json`, JSON.stringify(record, null, 2));
}
fs.writeFileSync(path.join(artifacts, 'stage4-mapped-nt-summary.json'), JSON.stringify(record, null, 2));
console.log(JSON.stringify({ ...record, build: record.build && { ...record.build, sourceHashes: undefined } }, null, 2));
