const { verifyMatchedOutputs } = require('./gpu-resident-correctness.cjs');

const normalize = r => ({ hash: r.outputHash || r.hash, cached: r.timings?.cached_n ?? r.cached,
  prompt: r.usage.prompt_tokens, generated: r.usage.completion_tokens });
const requests = round => (round.requests || round).map(normalize);
const state = values => JSON.stringify(values.map(({ cached, prompt, generated }) => ({ cached, prompt, generated })));

// Rolling cache eviction may remove a first-round seed. Never compare its output to a different
// cache state, and never excuse a comparable hash mismatch or any unmatched warmed round.
function verifyAvailableMatchedRounds(matrix, reference) {
  const matches = [], excluded = [];
  for (const row of matrix) {
    const baseline = reference.find(r => r.count === row.count && r.depth === row.depth);
    if (!baseline || row.rounds.length < 2) throw new Error('Missing matched warmed cohort');
    const baselineRounds = baseline.rounds.map(requests);
    for (const [index, round] of row.rounds.entries()) {
      const batch = requests(round);
      if (batch.length !== row.count || batch.some(r => !r.hash || !Number.isInteger(r.cached) || r.cached < 0 ||
          !Number.isInteger(r.prompt) || r.prompt <= 0 || !Number.isInteger(r.generated) || r.generated <= 0)) {
        throw new Error('Malformed batch');
      }
      if (baselineRounds.some(r => state(r) === state(batch))) {
        matches.push(...verifyMatchedOutputs([{ ...row, rounds: [round] }], [baseline])
          .map(m => ({ ...m, round: index })));
        continue;
      }
      const samePrompt = baselineRounds.some(r => r.length === batch.length && r.every((b, lane) =>
        b.prompt === batch[lane].prompt && b.generated === batch[lane].generated));
      if (index !== 0 || !batch.some(r => r.cached === 0) || !samePrompt) {
        throw new Error(`Unmatched non-cold batch c${row.count} d${row.depth} r${index}`);
      }
      excluded.push({ count: row.count, depth: row.depth, round: index, requests: batch,
        reason: 'First-round cache miss has no same-state reference; not output-validated' });
    }
    if (!matches.some(m => m.count === row.count && m.depth === row.depth && m.round > 0)) {
      throw new Error('No validated warmed round');
    }
  }
  return { matches, excluded };
}
module.exports = { verifyAvailableMatchedRounds };
