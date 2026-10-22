import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { prepareIndexCommit } from './publish-benchmark-index.mjs';

const sha = value => /^[a-f0-9]{40}$/.test(value || '');
const generated = name => /^benchmark-database\/index\/v1\/(?:manifest\.json|index-[a-f0-9]{64}\.(?:br|gz)\.bin)$/.test(name);

export async function publishIndexToMain({ directory, repository, token, python = 'python3', fetcher = fetch, sleep = ms => new Promise(resolve => setTimeout(resolve, ms)) }) {
  if (!path.isAbsolute(directory) || !/^[\w.-]+\/[\w.-]+$/.test(repository || '') || !token) throw new Error('Expected explicit trusted checkout, repository and CI token');
  const git = (...args) => execFileSync('git', args, { cwd: directory, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  const base = git('rev-parse', 'HEAD');
  if (!sha(base)) throw new Error('Invalid trusted main commit');
  const api = async (method, endpoint, body) => {
    for (let attempt = 0; ; attempt++) {
      let response;
      try {
        response = await fetcher(`https://api.github.com/repos/${repository}/${endpoint}`, {
          method, redirect: 'error', signal: AbortSignal.timeout(30000),
          headers: { Accept: 'application/vnd.github+json', Authorization: `Bearer ${token}`, 'X-GitHub-Api-Version': '2022-11-28', ...(body ? { 'Content-Type': 'application/json' } : {}) },
          ...(body ? { body: JSON.stringify(body) } : {})
        });
      } catch {
        if (method === 'GET' && attempt < 2) { await sleep(1000); continue; }
        throw new Error(`GitHub index publication ${method} network failure`);
      }
      if (!response.ok) {
        if (method === 'GET' && attempt < 2 && [429, 500, 502, 503, 504].includes(response.status)) { await sleep(1000); continue; }
        const error = new Error(`GitHub index publication ${method} failed: HTTP ${response.status}`); error.status = response.status; throw error;
      }
      return response.status === 204 ? null : response.json();
    }
  };
  const main = async () => {
    const ref = await api('GET', 'git/ref/heads/main');
    if (!sha(ref.object?.sha)) throw new Error('Invalid main ref from GitHub');
    return ref.object.sha;
  };
  if (await main() !== base) return { retry: true, reason: 'main advanced before generation' };
  const result = await prepareIndexCommit({ directory, repository, python });
  if (result.commit === base) return { ...result, retry: false, published: false };
  const paths = git('diff', '--no-renames', '--name-only', base, result.commit).split('\n').filter(Boolean).sort();
  if (!paths.length || paths.some(name => !generated(name))) throw new Error('Generated publication includes a non-index file');
  for (const row of git('ls-tree', '-r', result.commit, '--', 'benchmark-database/index/v1').split('\n').filter(Boolean)) {
    const [metadata, name] = row.split('\t');
    if (!metadata.startsWith('100644 blob ') || !generated(name)) throw new Error('Unexpected index path or mode');
  }
  const branch = `automation/benchmark-index/${result.commit}`;
  try { git('push', 'origin', `${result.commit}:refs/heads/${branch}`); }
  catch (error) {
    if (git('ls-remote', '--heads', 'origin', `refs/heads/${branch}`).split(/\s/)[0] !== result.commit) throw error;
  }
  let pull;
  try {
    pull = await api('POST', 'pulls', { head: branch, base: 'main', title: 'data(benchmark): refresh compact index',
      body: `CI-generated index from already merged, reviewed originals. No source report is auto-approved. Only benchmark-database/index/v1 is changed.\n\nTrusted source: ${base}\nIndex: ${result.version}`, maintainer_can_modify: false });
  } catch (error) {
    const existing = await api('GET', `pulls?state=open&base=main&head=${encodeURIComponent(repository.split('/')[0] + ':' + branch)}`);
    if (!Array.isArray(existing) || existing.length !== 1) throw error;
    pull = existing[0];
  }
  const number = pull.number;
  const assertPull = value => {
    if (!Number.isSafeInteger(number) || number < 1 || value.number !== number || value.user?.login !== 'github-actions[bot]' ||
        value.head?.sha !== result.commit || value.head?.ref !== branch || value.head?.repo?.full_name !== repository ||
        value.base?.ref !== 'main' || value.base?.repo?.full_name !== repository) throw new Error('Refusing an unbound or non-CI index PR');
  };
  assertPull(pull);
  pull = await api('GET', `pulls/${number}`); assertPull(pull);
  const files = await api('GET', `pulls/${number}/files?per_page=100`);
  if (!Array.isArray(files) || files.length > 5) throw new Error('Unexpected index PR file count');
  const actualPaths = files.flatMap(file => file.status === 'renamed' ? [file.filename, file.previous_filename] : [file.filename]);
  if (actualPaths.some(name => !generated(name)) || JSON.stringify([...new Set(actualPaths)].sort()) !== JSON.stringify(paths)) throw new Error('Index PR file set does not match generated commits');
  const cleanup = async close => {
    try {
      if (close) await api('PATCH', `pulls/${number}`, { state: 'closed' });
      await api('DELETE', `git/refs/heads/${branch}`); return true;
    } catch (error) { return error.status === 404; }
  };
  if (await main() !== base) return { ...result, retry: true, reason: 'main advanced before merge', pull_request: number, cleanup_complete: await cleanup(true) };
  let merged;
  for (let attempt = 0; attempt < 6; attempt++) {
    try {
      merged = await api('PUT', `pulls/${number}/merge`, { sha: result.commit, merge_method: 'merge', commit_title: 'data(benchmark): refresh compact index [skip ci]' });
      if (merged.merged) break;
    } catch (error) {
      const observed = await api('GET', `pulls/${number}`); assertPull(observed);
      if (observed.merged && sha(observed.merge_commit_sha)) { merged = { merged: true, sha: observed.merge_commit_sha }; break; }
      if (await main() !== base) return { ...result, retry: true, reason: 'main advanced during merge', pull_request: number, cleanup_complete: await cleanup(true) };
      if (![405, 409, 502, 503].includes(error.status) || attempt === 5) throw error;
    }
    if (attempt < 5) await sleep(2000);
  }
  if (!merged?.merged || !sha(merged.sha)) throw new Error('Index PR was not merged; main protection remains enforced');
  const expectedBlob = git('rev-parse', `${result.commit}:benchmark-database/index/v1/manifest.json`);
  const manifest = await api('GET', `contents/benchmark-database/index/v1/manifest.json?ref=${merged.sha}`);
  if (manifest.sha !== expectedBlob) throw new Error('Merged index manifest does not match generated bytes');
  return { ...result, retry: false, published: true, pull_request: number, merge_commit: merged.sha, cleanup_complete: await cleanup(false) };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2), values = {};
  for (let i = 0; i < args.length; i += 2) {
    if (!['--source', '--repository', '--python'].includes(args[i]) || !args[i + 1] || Object.hasOwn(values, args[i])) throw new Error('Invalid publication arguments');
    values[args[i]] = args[i + 1];
  }
  const result = await publishIndexToMain({ directory: values['--source'], repository: values['--repository'], python: values['--python'], token: process.env.GH_TOKEN });
  console.log(JSON.stringify(result));
  if (result.retry) process.exitCode = 75;
}
