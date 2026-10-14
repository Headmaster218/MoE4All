const { test } = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { mainGeometry, mainIdentity, checkpointRecords, snapshotTestCache } = require('./gpu-resident-cache-identity.cjs');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');

function fixture(flags = 3) {
  const header = Buffer.alloc(112);
  header.write('INFRKV01'); header.writeUInt32LE(2, 8); header.writeUInt32LE(flags, 12);
  for (const [offset, value] of [[48, 1], [56, 163840], [64, 32768], [72, 8],
    [80, flags & 1 ? 3 : 0], [88, flags & 2 ? 6 : 0], [104, 64]]) {
    header.writeBigUInt64LE(BigInt(value), offset);
  }
  header.writeUInt32LE(2, 96); header.writeUInt16LE(1, 100); header.writeUInt16LE(1, 102);
  const metadata = Buffer.alloc((8 + (flags & 1 ? 3 : 0) + (flags & 2 ? 6 : 0)) * 4);
  for (let i = 0; i < metadata.length / 4; i++) metadata.writeUInt32LE(i + 1, i * 4);
  const footer = Buffer.alloc(32, 0xa5);
  const bytes = 112 + metadata.length + 32 + 64 + 32;
  return { header, metadata, footer, bytes };
}

test('main cache identities preserve separate live and checkpoint digests', () => {
  for (const flags of [0, 1, 2, 3]) {
    const f = fixture(flags), id = mainIdentity(f.header, f.metadata, f.footer, f.bytes);
    assert.equal(id.cached, 8); assert.equal(id.committed, 32768);
    assert.equal(id.cachedTokenHash, hash(f.metadata.subarray(0, 32)));
    let offset = 32;
    for (let i = 0; i < 2; i++) {
      const count = flags & (1 << i) ? [3, 6][i] : 0;
      assert.deepEqual(id.checkpoints[i], { count,
        tokenHash: count ? hash(f.metadata.subarray(offset, offset + count * 4)) : null });
      offset += count * 4;
    }
    assert.equal(id.storedChecksum, f.footer.toString('hex'));
    assert.equal(id.payloadChecksumVerified, false);
  }
});

test('corrupt geometry, sizes, flags and unsafe integers are rejected before payload reads', () => {
  for (const mutate of [h => h.writeUInt32LE(1, 8), h => h.writeUInt32LE(4, 12),
    h => h.writeBigUInt64LE(200000n, 72), h => h.writeBigUInt64LE(0n, 80),
    h => h.writeUInt32LE(16385, 96), h => h.writeBigUInt64LE(2n ** 63n, 104)]) {
    const f = fixture(); mutate(f.header);
    assert.throws(() => mainGeometry(f.header, f.bytes));
  }
  const f = fixture();
  assert.throws(() => mainGeometry(f.header.subarray(0, 111), f.bytes));
  assert.throws(() => mainGeometry(f.header, f.bytes - 1));
  assert.throws(() => mainIdentity(f.header, f.metadata.subarray(4), f.footer, f.bytes));
  assert.throws(() => mainIdentity(f.header, f.metadata, f.footer.subarray(1), f.bytes));
});

test('the read-only snapshot refuses a production cache path without opening it', () => {
  assert.throws(() => snapshotTestCache('D:/AIinfr/infr/kv-sessions'), /restricted/);
});

test('checkpoint record identity skips main payload and hashes only keyed checkpoint state', () => {
  const entries = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(kind => {
    const header = Buffer.alloc(16), data = Buffer.alloc(kind + 1, kind + 32);
    header[0] = kind; header.writeUInt32LE(kind * 2, 4);
    header.writeBigUInt64LE(BigInt(data.length), 8);
    return { kind, header, data };
  });
  const payload = Buffer.concat([Buffer.alloc(112), ...entries.flatMap(e => [e.header, e.data]), Buffer.alloc(32)]);
  const geometry = { metadataBytes: 0, records: entries.length, checkpointCounts: [3, 6] };
  const reads = [];
  const readAt = (_fd, length, offset) => { reads.push([offset, length]); return payload.subarray(offset, offset + length); };
  const result = checkpointRecords(null, geometry, payload.length, readAt);
  assert.deepEqual(result.checkpointRecords, entries.filter(e => e.kind >= 5).map(e =>
    ({ kind: e.kind, layer: e.kind * 2, bytes: e.data.length, hash: hash(e.data) })));
  assert.equal(result.checkpointBytesHashed, 51);
  assert.equal(reads.length, entries.length + 6);
  assert.throws(() => checkpointRecords(null, { ...geometry, records: entries.length - 1 }, payload.length, readAt), /geometry/);
  assert.throws(() => checkpointRecords(null, { ...geometry, checkpointCounts: [0, 6] }, payload.length, readAt), /Unexpected/);
  payload[112] = 11;
  assert.throws(() => checkpointRecords(null, geometry, payload.length, readAt), /Invalid/);
});
