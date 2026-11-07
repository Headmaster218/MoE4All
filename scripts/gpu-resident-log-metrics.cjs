const numeric = (line, key) => {
  const value = line.match(new RegExp(`\\b${key}=([0-9.]+)(?:\\s|$)`));
  return value ? Number(value[1]) : null;
};

function parseLogMetrics(text, rounds, count) {
  const requests = new Map();
  const stats = [];
  for (const line of text.split(/\r?\n/)) {
    const at = Date.parse(line.split(/\s/, 1)[0]);
    if (!Number.isFinite(at)) continue;
    if (line.includes('serve stats')) {
      stats.push({ at, seconds: numeric(line, 'interval_s'), rate: numeric(line, 'decode_tps'),
        tokens: numeric(line, 'gen_tokens'), prompt: numeric(line, 'prompt_tokens'),
        active: numeric(line, 'active'), queued: numeric(line, 'queued'),
        completed: numeric(line, 'completed'), failed: numeric(line, 'failed') });
      continue;
    }
    const id = numeric(line, 'req');
    if (id === null) continue;
    if (line.includes('request start')) requests.set(id, { id, started: at, points: [] });
    const request = requests.get(id);
    if (!request) continue;
    if (line.includes('request progress') && line.includes('phase="decode"')) {
      const elapsed = numeric(line, 'elapsed_ms');
      const tokens = numeric(line, 'gen_tokens');
      if (elapsed !== null && tokens !== null) request.points.push({ at, elapsed, tokens });
    } else if (line.includes('request done')) {
      request.finished = at;
      request.points.push({ at, elapsed: numeric(line, 'total_ms'), tokens: numeric(line, 'gen_tokens') });
    }
  }
  return rounds.map(round => {
    const selected = [...requests.values()].filter(request => request.started >= round.startedUnixMs
      && request.finished <= round.finishedUnixMs + 1000);
    if (selected.length !== count || selected.some(request => request.points.length < 2)) {
      return { unavailable: 'Cannot uniquely delimit every request from server logs' };
    }
    const perRequest = selected.map(request => {
      const groups = [{ tokens: 0, ms: 0 }, { tokens: 0, ms: 0 }, { tokens: 0, ms: 0 }];
      const total = request.points.at(-1).tokens;
      for (let i = 1; i < request.points.length; i++) {
        const first = request.points[i - 1], last = request.points[i];
        const tokens = last.tokens - first.tokens, ms = last.elapsed - first.elapsed;
        if (tokens <= 0 || ms <= 0) continue;
        const group = Math.min(2, Math.floor((first.tokens + last.tokens) / (2 * total) * 3));
        groups[group].tokens += tokens; groups[group].ms += ms;
      }
      return { requestId: request.id, windowDefinition: 'One-second progress bins grouped by token midpoint thirds',
        front: groups[0].ms ? groups[0].tokens * 1000 / groups[0].ms : null,
        middle: groups[1].ms ? groups[1].tokens * 1000 / groups[1].ms : null,
        late: groups[2].ms ? groups[2].tokens * 1000 / groups[2].ms : null };
    });
    const start = Math.max(...selected.map(request => request.points[0].at));
    const end = Math.min(...selected.map(request => request.finished));
    // Log interval_s is rounded to 0.1 s. Keep the entire possible window inside Decode.
    const windows = stats.filter(window => window.seconds !== null && window.rate > 0
      && window.tokens > 0 && window.prompt === 0 && window.active === count
      && window.queued === 0 && window.completed === 0 && window.failed === 0
      && window.at - (window.seconds + 0.05) * 1000 >= start && window.at <= end);
    const tokens = windows.reduce((sum, window) => sum + window.tokens, 0);
    // decode_tps uses the unrounded interval duration; inversion avoids interval_s rounding.
    const seconds = windows.reduce((sum, window) => sum + window.tokens / window.rate, 0);
    return { perRequest, steadyConcurrent: {
      active: count, windows: windows.length, tokens, seconds,
      aggregateTokensPerSecond: seconds ? tokens / seconds : null,
      peakAggregateTokensPerSecond: windows.length ? Math.max(...windows.map(window => window.rate)) : null,
      precision: 'Existing server decode_tps rounded to 0.1 tok/s; only fully contained Decode windows',
    } };
  });
}

module.exports = { parseLogMetrics };
