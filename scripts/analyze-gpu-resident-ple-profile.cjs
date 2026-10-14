const fs = require('node:fs');
const path = require('node:path');

function parsePleCohorts(text) {
  const cohorts = [];
  let pending = null;
  const number = (line, key, suffix = '') => {
    const value = line.match(new RegExp(`\\b${key}=([0-9.]+)${suffix}(?:\\s|$)`));
    if (!value || !Number.isFinite(Number(value[1]))) throw new Error(`Missing ${key}`);
    return Number(value[1]);
  };
  for (const line of text.split(/\r?\n/)) {
    if (line.includes('[parallel-token-profile]')) {
      if (pending) throw new Error('Missing paired pager cohort');
      pending = Object.fromEntries(['lanes', 'steps', 'rows', 'decode_rows', 'prefill_rows']
        .map(key => [key, number(line, key)]));
      pending.runnerWaitMs = number(line, 'ple_wait', 'ms');
      pending.runnerUploadMs = number(line, 'ple_upload', 'ms');
      if (!pending.rows || pending.rows !== pending.decode_rows + pending.prefill_rows) {
        throw new Error('Invalid mixed-cohort row count');
      }
    } else if (line.includes('[parallel-token-pager]')) {
      if (!pending) throw new Error('Pager cohort without runner context');
      pending.planMs = number(line, 'ple_plan', 'ms');
      pending.gatherMs = number(line, 'ple_work', 'ms');
      pending.residualWaitMs = number(line, 'ple_wait', 'ms');
      pending.gatherUsPerProcessedRow = pending.gatherMs * 1000 / pending.rows;
      pending.residualWaitUsPerProcessedRow = pending.residualWaitMs * 1000 / pending.rows;
      cohorts.push(pending); pending = null;
    }
  }
  if (pending || !cohorts.length) throw new Error('Incomplete PLE profile');
  return cohorts;
}

function parseHostSetupCohorts(text) {
  const cohorts = [];
  let pending = null;
  const value = (line, key, unit = '') => {
    const match = line.match(new RegExp(`\\b${key}=([0-9.]+)${unit}(?:\\s|$)`));
    if (!match || !Number.isFinite(Number(match[1]))) throw new Error(`Missing ${key}`);
    return Number(match[1]);
  };
  for (const line of text.split(/\r?\n/)) {
    if (line.includes('[parallel-token-profile]')) {
      if (pending) throw new Error('Missing paired host setup cohort');
      pending = Object.fromEntries(['lanes', 'steps', 'rows', 'decode_rows', 'prefill_rows']
        .map(key => [key, value(line, key)]));
      if (pending.steps <= 0 || pending.rows !== pending.decode_rows + pending.prefill_rows ||
          pending.rows !== pending.steps * pending.lanes) throw new Error('Invalid host cohort dimensions');
      pending.mainSetupMs = value(line, 'main_setup', 'ms');
      pending.frontMs = value(line, 'front', 'ms');
    } else if (line.includes('[parallel-token-host]')) {
      if (!pending) throw new Error('Host setup without runner cohort');
      pending.partsMs = Object.fromEntries(['setup_layout', 'setup_scratch', 'setup_rope', 'setup_fusion', 'setup_moe_scan']
        .map(key => [key, value(line, key, 'ms')]));
      pending.partsUsPerForward = Object.fromEntries(Object.entries(pending.partsMs)
        .map(([key, ms]) => [key, ms * 1000 / pending.steps]));
      pending.mainSetupUsPerForward = pending.mainSetupMs * 1000 / pending.steps;
      cohorts.push(pending); pending = null;
    }
  }
  if (pending || !cohorts.length) throw new Error('Incomplete host setup profile');
  return cohorts;
}

function main() {
  const root = path.resolve(process.argv[2]);
  const tag = process.argv[3];
  const manifest = JSON.parse(fs.readFileSync(path.join(root, `${tag}-manifest.json`)));
  const cases = manifest.results.flatMap(item => item.matrix.map(row => {
    const dir = path.join(root, `${tag}-${item.name}-c${row.count}-d${row.depth}`);
    const result = JSON.parse(fs.readFileSync(path.join(dir, 'result.json')));
    if (result.status !== 'complete' || !result.args.includes('prof.pager_profile=true')) {
      throw new Error(`Not a completed PLE profile: ${dir}`);
    }
    const text = fs.readFileSync(path.join(dir, 'stderr.log'), 'utf8');
    const cohorts = parsePleCohorts(text);
    const hostSetupCohorts = parseHostSetupCohorts(text);
    if (cohorts.length !== result.rounds.length) throw new Error('Expected one runner cohort per request round');
    if (hostSetupCohorts.length !== cohorts.length) throw new Error('Missing host setup rounds');
    return { name: item.name, count: row.count, depth: row.depth,
      verboseStages: result.args.includes('prof.stages=true'), cohorts, hostSetupCohorts };
  }));
  const report = { tag, binaryHash: manifest.binaryHash,
    limitations: ['Gather includes mmap faults, row copies and dequantization; physical SSD I/O is not isolated.',
      'No OS file-cache purge: first and repeated rounds are not controlled cold/hot SSD benchmarks.',
      'Cohorts include remaining teacher-forced suffix rows; per-row values are not pure Decode token latencies.',
      'Work excludes channel scheduling and output-pool management; residual wait includes their unhidden portion.',
      'Windows file-cache residency may survive a service restart or be reclaimed under RAM pressure; it is not guaranteed.',
      'Wait is unhidden receiver time, not total gather time. Profiling throughput is not the release baseline.'], cases };
  fs.writeFileSync(path.join(root, `${tag}-analysis.json`), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
}
module.exports = { parsePleCohorts, parseHostSetupCohorts };
if (require.main === module) main();
