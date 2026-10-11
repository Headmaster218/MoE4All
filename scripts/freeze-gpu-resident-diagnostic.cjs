const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const tag = process.env.DIAGNOSTIC_BUILD_TAG || 'stage4-miss-detail-build';
const output = path.join(root, 'benchmark-data/artifacts/gpu-resident-decode-20261011', `${tag}.json`);
const binary = 'D:/AIinfr/infr/target/gpu-resident-experiment/release/infr.exe';
const frozen = `D:/AIinfr/infr/target/gpu-resident-experiment/release/infr-${tag}.exe`;
const git = args => {
  const result = spawnSync('git', args, { cwd: root, encoding: 'utf8', windowsHide: true, maxBuffer: 32 * 1024 * 1024 });
  if (result.status !== 0) throw new Error(result.stderr);
  return result.stdout;
};
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
if (fs.existsSync(output) || fs.existsSync(frozen)) throw new Error('Use a fresh build tag');
const files = git(['ls-files', 'Cargo.toml', 'Cargo.lock', 'crates']).split(/\r?\n/).filter(file => file && fs.statSync(path.join(root, file)).isFile());
const sourceHashes = Object.fromEntries(files.map(file => [file, digest(fs.readFileSync(path.join(root, file)))]));
fs.copyFileSync(binary, frozen);
const patch = output.replace(/\.json$/, '-source.patch');
fs.writeFileSync(patch, git(['diff', '--binary', 'HEAD']));
const record = { head: git(['rev-parse', 'HEAD']).trim(), exe: frozen, binaryHash: digest(fs.readFileSync(frozen)),
  toolchain: '1.97.1-x86_64-pc-windows-msvc', flags: '-C target-cpu=native',
  sourceHashes, patch, patchHash: digest(fs.readFileSync(patch)),
  scope: 'Post-build snapshot; no runtime edits between build and archive. Rejected staging trial stays disabled.',
};
fs.writeFileSync(output, JSON.stringify(record, null, 2));
console.log(JSON.stringify({ output, exe: frozen, binaryHash: record.binaryHash, sourceFiles: files.length }));
