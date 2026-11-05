'use strict';
// No dependencies, cloud requests, Git writes, GPU work, or global state changes.
// node scripts/Test-SubmissionTransfer.cjs --task-root <local-work/tmp/moe4all/task>
//   --evidence-root <local-work/evidence/moe4all/task> [--infr <existing infr.exe>]
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { spawn } = require('node:child_process');
const args = process.argv.slice(2);
const arg = name => { const i = args.indexOf(name); return i < 0 ? undefined : args[i + 1]; };
const engine = path.resolve(__dirname, '..');
const root = 'D:\\AIinfr\\infr\\local-work';
function output(name, category) {
  const value = arg(name);
  assert.ok(value && path.isAbsolute(value), `${name} must be absolute`);
  const target = path.resolve(value), parent = path.join(root, category, 'moe4all') + path.sep;
  assert.ok(target.toLowerCase().startsWith(parent.toLowerCase()), `${name} must be under ${parent}`);
  let current = target;
  while (!fs.existsSync(current)) current = path.dirname(current);
  assert.equal(fs.realpathSync(current).toLowerCase(), path.resolve(current).toLowerCase(), 'Output junctions are not allowed');
  fs.mkdirSync(target, { recursive: true });
  return target;
}
const task = output('--task-root', 'tmp'), evidence = output('--evidence-root', 'evidence');
const infr = path.resolve(arg('--infr') || path.join(engine, 'target/release/infr.exe'));
const baselineWizard = arg('--baseline-wizard');
const env = { ...process.env, TEMP: task, TMP: task, TMPDIR: task, NO_PROXY: '127.0.0.1,localhost', MOE4ALL_WORK_ROOT: root };
delete env.PSModulePath;
const stamp = new Date().toISOString();
const runDir = path.join(task, 'run-' + Date.now()); fs.mkdirSync(runDir);
function run(exe, argv, timeout = 25000) {
  return new Promise((resolve, reject) => {
    const p = spawn(exe, argv, { windowsHide: true, env, cwd: engine });
    let out = '', err = '', killed = false;
    p.stdout.on('data', b => out += b); p.stderr.on('data', b => err += b);
    const timer = setTimeout(() => { killed = true; p.kill(); }, timeout);
    p.on('error', e => { clearTimeout(timer); reject(e); });
    p.on('exit', code => { clearTimeout(timer); resolve({ code, out, err, killed }); });
  });
}
function json(res, status, value, headers = {}) {
  const body = JSON.stringify(value);
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'content-length': Buffer.byteLength(body), ...headers }); res.end(body);
}
async function main() {
  const input = path.join(runDir, 'benchmark.json');
  const fixture = JSON.parse(fs.readFileSync(path.join(engine, 'scripts/fixtures/submission-vectors.json'), 'utf8')).benchmark;
  fixture.id = 'loopback-benchmark';
  fs.writeFileSync(input, JSON.stringify(fixture));
  const packed = await run(infr, ['submission', 'pack-benchmark', '--input', input, '--result-dir', path.join(runDir, 'reports')]);
  assert.equal(packed.code, 0, 'Existing CLI could not pack synthetic fixture');
  const benchmarkFile = packed.out.match(/^MOE4ALL_SUBMISSION_PATH=(.+)$/m)?.[1].trim(); assert.ok(benchmarkFile);
  // CRLF and a UTF-8 summary exercise byte-preserving transport without private data.
  const bytes = Buffer.from(fs.readFileSync(benchmarkFile, 'utf8').replace(/\r?\n/g, '\r\n'), 'utf8');
  fs.writeFileSync(benchmarkFile, bytes);
  const verified = await run(infr, ['submission', 'verify', benchmarkFile]); assert.equal(verified.code, 0, 'Synthetic CRLF report must pass real CLI verification');
  const hash = crypto.createHash('sha256').update(bytes).digest('hex');
  let scenario = '', origin = '', requests = [], session = null, reportCount = 0, testCount = 0;
  const sockets = new Set(), timers = new Set(), results = [];
  const server = http.createServer(async (req, res) => {
    try {
      const chunks = []; for await (const b of req) chunks.push(b);
      const body = Buffer.concat(chunks);
      requests.push({ method: req.method, path: req.url, size_bytes: body.length, hash: crypto.createHash('sha256').update(body).digest('hex') });
      if (req.url === '/v1/upload-sessions') {
        const data = JSON.parse(body.toString('utf8'));
        assert.equal(data.file_sha256, hash); assert.equal(data.size_bytes, bytes.length); assert.equal(data.kind, 'benchmark'); assert.equal(data.report_id, fixture.id);
        session = data;
        return json(res, 201, { api_version: 1, ...data, session_id: 'fixture-session', session_token: 'fixture-session-secret', state: 'awaiting_verification', max_upload_bytes: 131072, poll_after_seconds: 3, expires_at: new Date(Date.now() + 300000).toISOString(), verification_url: origin + '/community/upload.html#session=fixture-session&authorize=fixture-browser-secret' });
      }
      assert.equal(req.headers.authorization, 'Bearer fixture-session-secret');
      if (req.url === '/v1/upload-sessions/fixture-session') {
        assert.equal(req.method, 'GET'); return json(res, 200, { api_version: 1, ...session, state: 'ready' });
      }
      if (req.url.endsWith('/report')) {
        reportCount++; assert.ok(body.equals(bytes)); assert.equal(req.headers['content-type'], 'text/plain; charset=utf-8');
        if (scenario === 'flow-timeout' || scenario === 'offer-later') return;
        if (scenario === 'offer-retry' && reportCount === 1) return json(res, 503, { error: { code: 'busy', retryable: false } });
        if (scenario === 'flow-retry' && reportCount === 1) return json(res, 503, { error: { code: 'busy', retryable: true } }, { 'retry-after': '0' });
        if (scenario === 'flow-lost' && reportCount === 1) return req.socket.destroy();
        const status = scenario === 'flow-pending' ? 'pending_review' : scenario === 'flow-published' ? 'published' : 'forwarding';
        return json(res, reportCount > 1 ? 200 : 202, { api_version: 1, receipt_id: 'fixture-receipt', receipt_token: 'fixture-receipt-secret', kind: 'benchmark', report_id: fixture.id, file_sha256: scenario === 'flow-mismatch' ? 'f'.repeat(64) : hash, payload_sha256: 'a'.repeat(64), status, duplicate: reportCount > 1, received_at: new Date().toISOString(), public_url: status === 'forwarding' ? null : 'https://github.com/example/fixture/pull/1' });
      }
      assert.equal(req.url, '/v1/test'); assert.equal(req.method, 'POST'); testCount++;
      assert.ok(body.equals(bytes)); assert.equal(req.headers['content-type'], 'text/plain; charset=utf-8');
      if (scenario === 'timeout' || scenario === 'cancel') return;
      if (scenario === 'slow-body') { res.writeHead(200, { 'content-type': 'application/json' }); res.write('{'); return; }
      if (scenario === 'exhausted') return req.socket.destroy();
      if (scenario === 'invalid-json') { res.writeHead(202); res.end('not json'); return; }
      if (scenario === 'empty-json') { res.writeHead(202); res.end(); return; }
      if (scenario === 'oversize') { res.writeHead(200, { 'content-type': 'application/json' }); res.end(' '.repeat(1048577)); return; }
      if (scenario === 'nonretry-400') return json(res, 400, { error: { code: 'invalid_format', retryable: true } });
      if (scenario === 'nonretry-503') return json(res, 503, { error: { code: 'permanent', retryable: false } });
      if (scenario === 'long-wait') return json(res, 429, { error: { code: 'limited' } }, { 'retry-after': '121' });
      if (scenario === 'redirect') return json(res, 302, {}, { location: origin + '/must-not-follow' });
      if (scenario.startsWith('retry-') && testCount === 1) {
        const status = scenario === 'retry-429' || scenario === 'retry-date' ? 429 : 503;
        const retry = scenario === 'retry-date' ? new Date(Date.now() - 1000).toUTCString() : '0';
        return json(res, status, { error: { code: 'busy', retryable: true } }, { 'retry-after': retry });
      }
      if (scenario === 'progress') {
        const timer = setTimeout(() => { timers.delete(timer); json(res, 200, { api_version: 1 }); }, 5600); timers.add(timer); return;
      }
      return json(res, 200, { api_version: 1 });
    } catch (error) { json(res, 500, { error: { code: 'fixture_assertion', message: error.message, retryable: false } }); }
  });
  server.on('connection', s => { sockets.add(s); s.on('close', () => sockets.delete(s)); });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  origin = 'http://127.0.0.1:' + server.address().port;
  try {
    const cases = baselineWizard ? ['progress', 'invalid-json'] : ['offline-deadline', 'raw', 'flow-success', 'flow-retry', 'flow-lost', 'flow-pending', 'flow-published', 'flow-mismatch', 'flow-timeout', 'offer-later', 'offer-retry', 'timeout', 'slow-body', 'cancel', 'retry-429', 'retry-503', 'retry-date', 'nonretry-400', 'nonretry-503', 'long-wait', 'redirect', 'invalid-json', 'empty-json', 'oversize', 'exhausted', 'progress'];
    for (scenario of cases) {
      requests = []; reportCount = 0; testCount = 0; session = null;
      const argv = ['-NoLogo', '-NoProfile', '-NonInteractive', '-File', path.join(__dirname, 'Test-SubmissionTransfer.ps1'), '-ApiBase', origin + '/v1', '-Scenario', scenario, '-TaskRoot', runDir, '-BenchmarkFile', benchmarkFile, '-Infr', infr];
      if (baselineWizard) argv.push('-WizardPath', path.resolve(baselineWizard), '-Baseline');
      const child = await run('powershell.exe', argv, 35000);
      const safeLog = (child.out + child.err).replace(/fixture-(?:session|receipt|browser)-secret/g, '[REDACTED_FIXTURE_CREDENTIAL]');
      const prefix = baselineWizard ? 'baseline-' : '';
      fs.writeFileSync(path.join(evidence, prefix + scenario + '.log'), safeLog);
      assert.ok(!child.killed, `${scenario}: watchdog killed child`);
      const ui = path.join(runDir, scenario, 'ui.log');
      if (fs.existsSync(ui)) fs.copyFileSync(ui, path.join(evidence, prefix + scenario + '-ui.log'));
      if (baselineWizard) {
        assert.notEqual(child.code, 0, 'Baseline must reproduce regression');
        assert.equal(testCount, scenario === 'invalid-json' ? 4 : 1, 'Unexpected baseline retry behavior');
        if (scenario === 'progress') assert.ok(!fs.readFileSync(ui, 'utf8').includes('Waiting for submission service'));
        results.push({ scenario, expected_regression: true, requests });
        process.stdout.write(`REPRODUCED baseline ${scenario}: ${testCount} requests\n`);
        continue;
      }
      assert.equal(child.code, 0, `${scenario}: fixture failed; see evidence log`);
      const result = JSON.parse(child.out.trim());
      if (scenario.startsWith('flow-') || scenario.startsWith('offer-')) {
        assert.equal(requests.filter(r => r.path === '/v1/upload-sessions').length, scenario === 'offer-retry' ? 2 : 1, 'Unexpected session creation');
        assert.equal(reportCount, ['flow-retry', 'flow-lost', 'flow-timeout', 'offer-retry'].includes(scenario) ? 2 : 1);
      } else {
        const expected = scenario.startsWith('retry-') ? 2 : scenario === 'exhausted' ? 4 : scenario === 'offline-deadline' ? 0 : 1;
        assert.equal(testCount, expected, `${scenario}: unexpected number of HTTP attempts`);
      }
      assert.equal(crypto.createHash('sha256').update(fs.readFileSync(benchmarkFile)).digest('hex'), hash);
      results.push({ ...result, requests });
      process.stdout.write(`PASS ${scenario} (${result.elapsed_ms}ms)\n`);
      for (const socket of sockets) socket.destroy();
    }
    fs.writeFileSync(path.join(evidence, baselineWizard ? 'baseline-results.json' : 'transfer-results.json'), JSON.stringify({ started_at: stamp, finished_at: new Date().toISOString(), status: 'passed', synthetic_only: true, loopback_origin: origin, task_root: runDir, fixture_sha256: hash, infr_sha256: crypto.createHash('sha256').update(fs.readFileSync(infr)).digest('hex'), wizard_sha256: crypto.createHash('sha256').update(fs.readFileSync(baselineWizard || path.join(__dirname, 'infr-wizard.ps1'))).digest('hex'), results }, null, 2));
    process.stdout.write(`Passed ${results.length} Windows PowerShell 5.1 loopback scenarios.\n`);
  } finally {
    for (const timer of timers) clearTimeout(timer);
    for (const socket of sockets) socket.destroy();
    await new Promise(resolve => server.close(resolve));
  }
}
main().catch(error => { console.error(error.message); process.exitCode = 1; });
