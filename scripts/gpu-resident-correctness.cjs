const normalize = request => ({
  hash: request.outputHash || request.hash,
  cached: request.timings?.cached_n ?? request.cached,
  prompt: request.usage.prompt_tokens,
  generated: request.usage.completion_tokens,
});
const state = requests => JSON.stringify(requests.map(({ cached, prompt, generated }) => ({ cached, prompt, generated })));

function verifyMatchedOutputs(matrix, reference) {
  const matches = [];
  for (const row of matrix) {
    const baseline = reference.find(value => value.count === row.count && value.depth === row.depth);
    if (!baseline) throw new Error(`Missing reference c${row.count} d${row.depth}`);
    const baselineRounds = baseline.rounds.map(round => (round.requests || round).map(normalize));
    for (const [roundIndex, round] of row.rounds.entries()) {
      const requests = (round.requests || round).map(normalize);
      const comparable = baselineRounds.map((values, index) => ({ values, index }))
        .filter(value => state(value.values) === state(requests));
      if (!comparable.length) throw new Error(`No matching whole-batch cache state c${row.count} d${row.depth} r${roundIndex}`);
      const matched = comparable.find(value => value.values.every((request, lane) => request.hash === requests[lane].hash));
      if (!matched) throw new Error(`Output mismatch at matching cache state c${row.count} d${row.depth} r${roundIndex}`);
      matches.push({ count: row.count, depth: row.depth, round: roundIndex, referenceRound: matched.index });
    }
  }
  return matches;
}

module.exports = { verifyMatchedOutputs };
