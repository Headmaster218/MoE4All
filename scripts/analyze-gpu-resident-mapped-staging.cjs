const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011');
const tag = process.env.STAGING_TAG || 'stage4-mapped-staging-module';
const log = path.join(artifacts, `${tag}.log`);
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const content = fs.readFileSync(log, 'utf8');
const samples = [...content.matchAll(/mapped_staging experts=(\d+) bytes=(\d+) pass=(\d+) mode=(\d+) prepare_us=([\d.]+) total_us=([\d.]+) dma_batches=(\d+) pool_bytes=(\d+)/g)]
  .map(([, experts, bytes, pass, mode, prepareUs, totalUs, dmaBatches, poolBytes]) => ({ experts: +experts,
    bytes: +bytes, pass: +pass, mode: +mode, prepareUs: +prepareUs, totalUs: +totalUs,
    dmaBatches: +dmaBatches, poolBytes: +poolBytes }));
if (samples.length !== 90 || !content.includes('test result: ok. 1 passed; 0 failed;')) throw new Error('Incomplete staging module screen');
const rows = [1, 2, 3, 4, 8].map(experts => ({ experts,
  modes: [0, 1, 2].map(mode => {
    const values = samples.filter(s => s.experts === experts && s.mode === mode);
    if (values.length !== 6 || new Set(values.map(s => s.pass)).size !== 6 || values.some(s =>
      s.poolBytes > 32 * 1024 ** 2 || s.dmaBatches !== (mode === 2 ? 64 : 0) ||
      !Number.isFinite(s.prepareUs) || s.prepareUs <= 0 || s.totalUs < s.prepareUs)) throw new Error('Malformed staging passes');
    return { mode, meanPrepareUs: values.reduce((n, s) => n + s.prepareUs, 0) / 6,
      meanTotalUs: values.reduce((n, s) => n + s.totalUs, 0) / 6,
      preparePassesUs: values.map(s => s.prepareUs), totalPassesUs: values.map(s => s.totalUs),
      maxPoolBytes: Math.max(...values.map(s => s.poolBytes)) };
  }),
}));
const record = { log, logHash: digest(fs.readFileSync(log)), rows,
  scope: '6 alternating passes, 4 warmups + 64 samples/pass, 10-thread pool; total includes prepare/submit/host wait, no hit/encode overlap',
  correctness: '5 simultaneously prepared distinct 1MiB payloads plus 90 full GPU readbacks with guards; no inference claim',
  modes: ['mapped Rayon (default)', 'mapped serial', 'bounded host staging + dedicated DMA'],
};
if (process.env.STAGING_ARCHIVE === '1') {
  const prefix = path.join(artifacts, tag);
  if (fs.existsSync(`${prefix}-source.patch`)) throw new Error('Archive already exists');
  const diff = spawnSync('git', ['diff', '--binary', 'HEAD'], { cwd: root, encoding: 'utf8', windowsHide: true, maxBuffer: 16 * 1024 * 1024 });
  const head = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8', windowsHide: true });
  if (diff.status !== 0 || head.status !== 0) throw new Error('Cannot archive source');
  fs.writeFileSync(`${prefix}-source.patch`, diff.stdout);
  const binary = 'D:/AIinfr/infr/target/gpu-resident-experiment/release/deps/infr_vulkan-18125551459bd3da.exe';
  const frozen = `D:/AIinfr/infr/target/gpu-resident-experiment/release/infr-vulkan-${tag}-tests.exe`;
  if (fs.existsSync(frozen)) throw new Error('Frozen test binary already exists');
  fs.copyFileSync(binary, frozen);
  record.build = { head: head.stdout.trim(), binary: frozen, binaryHash: digest(fs.readFileSync(frozen)),
    toolchain: '1.97.1-x86_64-pc-windows-msvc', flags: '-C target-cpu=native',
    sourcePatch: `${prefix}-source.patch`, sourcePatchHash: digest(Buffer.from(diff.stdout)),
    sourceHashes: Object.fromEntries(['crates/infr-core/src/timeline.rs', 'crates/infr-core/src/config/mod.rs',
      'crates/infr-vulkan/src/transfer.rs', 'crates/infr-vulkan/src/adapter.rs'].map(file => [file, digest(fs.readFileSync(path.join(root, file)))])) };
  fs.writeFileSync(`${prefix}-archive.json`, JSON.stringify(record, null, 2));
}
fs.writeFileSync(path.join(artifacts, `${tag}-summary.json`), JSON.stringify(record, null, 2));
console.log(JSON.stringify(rows, null, 2));
