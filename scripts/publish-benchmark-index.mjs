import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { buildIndex } from './build-benchmark-index.mjs';

const indexPath = 'benchmark-database/index/v1';
// Called only on a clean, disposable trusted-main checkout. Never publishes PR code or report text as code.
export async function prepareIndexCommit({ directory, repository = 'Headmaster218/MoE4All', python = 'python3' }) {
  const git = (...args) => execFileSync('git', args, { cwd: directory, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  if (git('status', '--porcelain')) throw new Error('Index publication requires a clean checkout');
  const output = path.join(directory, indexPath), manifest = await buildIndex({ directory, repository, output, python });
  const keep = Object.values(manifest.packages).map(p => p.url);
  for (const name of await fs.readdir(output)) {
    if (/^index-[a-f0-9]{64}\.(br|gz)\.bin$/.test(name) && !keep.includes(name)) await fs.unlink(path.join(output, name));
  }
  git('add', '--', ...keep.map(name => `${indexPath}/${name}`));
  // Stage only tracked obsolete packages, never arbitrary files from the worktree.
  for (const name of git('ls-files', '--deleted', '--', indexPath).split('\n').filter(Boolean)) {
    if (!new RegExp(`^${indexPath}/index-[a-f0-9]{64}\\.(br|gz)\\.bin$`).test(name)) throw new Error('Unexpected index deletion');
    git('add', '--', name);
  }
  if (git('diff', '--cached', '--name-only')) git('-c', 'user.name=github-actions[bot]', '-c', 'user.email=41898282+github-actions[bot]@users.noreply.github.com', 'commit', '-m', 'data(benchmark): publish compact packages');
  const packageCommit = git('log', '-1', '--format=%H', '--', ...keep.map(name => `${indexPath}/${name}`));
  if (!/^[a-f0-9]{40}$/.test(packageCommit)) throw new Error('Missing immutable package commit');
  manifest.package_commit = packageCommit;
  await fs.writeFile(path.join(output, 'manifest.json'), JSON.stringify(manifest) + '\n');
  git('add', '--', `${indexPath}/manifest.json`);
  const changed = Boolean(git('diff', '--cached', '--name-only'));
  if (changed) git('-c', 'user.name=github-actions[bot]', '-c', 'user.email=41898282+github-actions[bot]@users.noreply.github.com', 'commit', '-m', 'data(benchmark): advance compact index manifest');
  return { changed, commit: git('rev-parse', 'HEAD'), package_commit: packageCommit, version: manifest.version };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), values = {};
  for (let i = 0; i < args.length; i += 2) {
    if (!['--source', '--repository', '--python'].includes(args[i]) || !args[i + 1] || Object.hasOwn(values, args[i])) throw new Error('Invalid publish arguments');
    values[args[i]] = args[i + 1];
  }
  console.log(JSON.stringify(await prepareIndexCommit({ directory: values['--source'], repository: values['--repository'], python: values['--python'] })));
}
