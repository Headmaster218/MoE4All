const test = require('node:test');
const assert = require('node:assert/strict');
const { verifyAvailableMatchedRounds } = require('./gpu-resident-matched-rounds.cjs');
const request = (cached = 99, hash = 'correct') => ({ cached, hash, usage: { prompt_tokens: 100, completion_tokens: 512 } });
const matrix = rounds => [{ count: 2, depth: 30000, rounds }];
const reference = matrix([[request(), request()], [request(), request()]]);
test('same-state whole batches retain strict output validation', () => {
  assert.equal(verifyAvailableMatchedRounds(reference, reference).matches.length, 2);
  assert.throws(() => verifyAvailableMatchedRounds(matrix([[request(), request('99', 'wrong')],
    [request(), request()]]), reference));
  assert.throws(() => verifyAvailableMatchedRounds(matrix([[request(), request(99, 'wrong')],
    [request(), request()]]), reference));
});
test('only an unmatched first cold batch is explicitly excluded', () => {
  const result = verifyAvailableMatchedRounds(matrix([[request(), request(0, 'fresh')],
    [request(), request()]]), reference);
  assert.equal(result.matches.length, 1);
  assert.equal(result.matches[0].round, 1);
  assert.equal(result.excluded.length, 1);
  assert.equal(result.excluded[0].requests[1].hash, 'fresh');
  assert.equal(result.excluded[0].requests[1].cached, 0);
});
test('comparable cold output mismatches are never excluded', () => {
  const cold = matrix([[request(0), request(0)], [request(), request()]]);
  assert.throws(() => verifyAvailableMatchedRounds(matrix([[request(0, 'wrong'), request(0)],
    [request(), request()]]), cold));
});
test('warmed cache changes, prompt changes and missing warmed rounds fail', () => {
  for (const input of [matrix([[request(), request()], [request(), request(0)]]),
    matrix([[request(98), request()], [request(), request()]]),
    matrix([[{ ...request(0), usage: { prompt_tokens: 101, completion_tokens: 512 } }, request()],
      [request(), request()]]), matrix([[request(0), request(0)]])]) {
    assert.throws(() => verifyAvailableMatchedRounds(input, reference));
  }
});
