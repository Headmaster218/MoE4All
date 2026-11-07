import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { gzipSync, brotliCompressSync, constants } from 'node:zlib';
import { assembleIndex, encodeIndex, INDEX_LIMITS, validateManifest } from './leaderboard/model.mjs';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
export async function buildIndex({ directory, ref = 'HEAD', repository = 'Headmaster218/MoE4All', output, python = 'python3' }) {
  if (!path.isAbsolute(directory) || !path.isAbsolute(output) || !/^[\w.-]+\/[\w.-]+$/.test(repository)) throw new Error('Expected explicit source/output paths and repository');
  const source = JSON.parse(execFileSync(python, [fileURLToPath(new URL('./benchmark_index_sources.py', import.meta.url)), directory, ref], {
    maxBuffer: INDEX_LIMITS.decodedBytes * 4, encoding: 'utf8', env: { ...process.env, PYTHONUTF8: '1', PYTHONDONTWRITEBYTECODE: '1' }, stdio: ['ignore', 'pipe', 'pipe']
  }));
  const index = assembleIndex(source.entries, { source: { repository, commit: source.commit, path_version: 1 } });
  const json = Buffer.from(JSON.stringify(encodeIndex(index))), version = sha256(json);
  if (json.length > INDEX_LIMITS.decodedBytes) throw new Error('Compact index exceeds decoded limit');
  const packages = {};
  await fs.mkdir(output, { recursive: true });
  for (const [format, bytes] of [['brotli', brotliCompressSync(json, { params: { [constants.BROTLI_PARAM_QUALITY]: 11, [constants.BROTLI_PARAM_MODE]: constants.BROTLI_MODE_TEXT } })], ['gzip', gzipSync(json, { level: 9 })]]) {
    if (bytes.length > INDEX_LIMITS.compressedBytes) throw new Error('Compact package exceeds size limit');
    const name = `index-${version}.${format === 'brotli' ? 'br' : 'gz'}.bin`;
    const filename = path.join(output, name);
    try { await fs.writeFile(filename, bytes, { flag: 'wx' }); }
    catch (error) { if (error.code !== 'EEXIST' || sha256(await fs.readFile(filename)) !== sha256(bytes)) throw error; }
    packages[format] = { url: name, bytes: bytes.length, sha256: sha256(bytes) };
  }
  const manifest = { schema: index.schema, version, mode: index.mode, generated_at: source.generated_at, source: index.source,
    counts: { reports: index.reports.length, contexts: index.contexts.length, measurements: index.measurements.length },
    decoded_bytes: json.length, decoded_sha256: version, packages };
  validateManifest(manifest, { repository });
  const temporary = path.join(output, `.manifest-${randomUUID()}.json`);
  await fs.writeFile(temporary, JSON.stringify(manifest) + '\n', { flag: 'wx' });
  await fs.rename(temporary, path.join(output, 'manifest.json'));
  return manifest;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), values = {};
  for (let i = 0; i < args.length; i += 2) {
    if (!['--source', '--output', '--repository', '--ref', '--python'].includes(args[i]) || !args[i + 1] || Object.hasOwn(values, args[i])) throw new Error('Invalid or duplicate index arguments');
    values[args[i]] = args[i + 1];
  }
  const manifest = await buildIndex({ directory: values['--source'], output: values['--output'], repository: values['--repository'], ref: values['--ref'], python: values['--python'] });
  console.log(JSON.stringify({ version: manifest.version, counts: manifest.counts }));
}
