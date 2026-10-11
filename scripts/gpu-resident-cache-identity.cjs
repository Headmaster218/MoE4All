const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const sha256 = data => crypto.createHash('sha256').update(data).digest('hex');
const safeU64 = (bytes, offset) => {
  const value = bytes.readBigUInt64LE(offset);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error('Unsafe cache integer');
  return Number(value);
};

// Match session_cache.rs v2 geometry. Reading a stored footer does not validate its payload.
function mainGeometry(header, bytes) {
  if (header.length !== 112 || header.subarray(0, 8).toString() !== 'INFRKV01' ||
      header.readUInt32LE(8) !== 2) throw new Error('Unsupported main cache header');
  const flags = header.readUInt32LE(12);
  const maxCtx = safeU64(header, 56);
  const committed = safeU64(header, 64);
  const cached = safeU64(header, 72);
  const checkpointCounts = [safeU64(header, 80), safeU64(header, 88)];
  const records = header.readUInt32LE(96);
  const dataBytes = safeU64(header, 104);
  if ((flags & ~3) !== 0 || maxCtx <= 0 || maxCtx > 16 * 1024 * 1024 ||
      committed > maxCtx || cached > maxCtx || records > 16384 ||
      checkpointCounts.some((n, i) => n > maxCtx || Boolean(flags & (1 << i)) !== (n !== 0))) {
    throw new Error('Inconsistent main cache geometry');
  }
  const metadataBytes = (cached + checkpointCounts[0] + checkpointCounts[1]) * 4;
  const expected = 112 + metadataBytes + records * 16 + dataBytes + 32;
  if (!Number.isSafeInteger(expected) || expected !== bytes) throw new Error('Cache file size mismatch');
  return { maxCtx, committed, cached, checkpointCounts, records, dataBytes, metadataBytes };
}

function mainIdentity(header, metadata, footer, bytes) {
  const geometry = mainGeometry(header, bytes);
  if (metadata.length !== geometry.metadataBytes || footer.length !== 32) {
    throw new Error('Incomplete cache identity');
  }
  let offset = geometry.cached * 4;
  const checkpoints = geometry.checkpointCounts.map(count => {
    const digest = count ? sha256(metadata.subarray(offset, offset + count * 4)) : null;
    offset += count * 4;
    return { count, tokenHash: digest };
  });
  return { ...geometry, fingerprint: header.subarray(16, 48).toString('hex'),
    savedAt: safeU64(header, 48), kFmt: header.readUInt16LE(100), vFmt: header.readUInt16LE(102),
    headerHash: sha256(header), cachedTokenHash: sha256(metadata.subarray(0, geometry.cached * 4)),
    checkpoints, storedChecksum: footer.toString('hex'), payloadChecksumVerified: false };
}

function readExact(fd, length, offset) {
  const bytes = Buffer.alloc(length);
  let read = 0;
  while (read < length) {
    const n = fs.readSync(fd, bytes, read, length - read, offset + read);
    if (!n) throw new Error('Truncated cache identity');
    read += n;
  }
  return bytes;
}

function checkpointRecords(fd, geometry, bytes, readAt = readExact) {
  let offset = 112 + geometry.metadataBytes;
  let readBytes = 0;
  const records = [];
  for (let i = 0; i < geometry.records; i++) {
    const header = readAt(fd, 16, offset);
    offset += 16;
    const kind = header[0], layer = header.readUInt32LE(4), length = safeU64(header, 8);
    if (kind > 10 || length > bytes - 32 - offset) throw new Error('Invalid cache record');
    if (kind >= 5) {
      const checkpoint = Math.floor((kind - 5) / 3);
      if (!geometry.checkpointCounts[checkpoint]) throw new Error('Unexpected checkpoint record');
      const digest = crypto.createHash('sha256');
      for (let read = 0; read < length;) {
        const chunk = Math.min(1024 * 1024, length - read);
        digest.update(readAt(fd, chunk, offset + read)); read += chunk;
      }
      readBytes += length;
      records.push({ kind, layer, bytes: length, hash: digest.digest('hex') });
    }
    offset += length;
  }
  if (offset !== bytes - 32) throw new Error('Cache record geometry mismatch');
  return { checkpointRecords: records, checkpointBytesHashed: readBytes };
}

function snapshotTestCache(root, { hashCheckpointRecords = false } = {}) {
  root = path.resolve(root);
  if (path.basename(root).toLowerCase() !== 'test-kv-cache') {
    throw new Error('Identity snapshot is restricted to test-kv-cache');
  }
  const files = [];
  let aggregateBytes = 0;
  function visit(file) {
    const stat = fs.lstatSync(file);
    if (stat.isSymbolicLink()) throw new Error(`Cache reparse point: ${file}`);
    const relative = path.relative(root, fs.realpathSync(file));
    if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
      throw new Error('Cache identity escaped test root');
    }
    if (stat.isDirectory()) {
      for (const name of fs.readdirSync(file).sort()) visit(path.join(file, name));
      return;
    }
    if (!stat.isFile()) throw new Error('Unexpected cache filesystem entry');
    aggregateBytes += stat.size;
    if (!['.infrkv', '.infrmtp'].includes(path.extname(file))) return;
    const fd = fs.openSync(file, 'r');
    try {
      const before = fs.fstatSync(fd);
      if (before.size < 144) throw new Error('Truncated cache file');
      const header = readExact(fd, 112, 0);
      const footer = readExact(fd, 32, before.size - 32);
      let identity;
      if (path.extname(file) === '.infrkv') {
        const geometry = mainGeometry(header, before.size);
        identity = mainIdentity(header, readExact(fd, geometry.metadataBytes, 112), footer, before.size);
        if (hashCheckpointRecords) Object.assign(identity, checkpointRecords(fd, geometry, before.size));
      } else {
        if (header.subarray(0, 8).toString() !== 'INFRMT01' || header.readUInt32LE(8) !== 1) {
          throw new Error('Unsupported MTP sidecar header');
        }
        identity = { headerHash: sha256(header), storedChecksum: footer.toString('hex'),
          payloadChecksumVerified: false };
      }
      const after = fs.fstatSync(fd);
      if (before.size !== after.size || before.mtimeMs !== after.mtimeMs) {
        throw new Error('Cache changed during identity snapshot');
      }
      files.push({ file: path.relative(root, file), bytes: before.size, mtimeMs: before.mtimeMs, ...identity });
    } finally { fs.closeSync(fd); }
  }
  if (fs.existsSync(root)) visit(root);
  return { root, capturedAt: new Date().toISOString(), aggregateBytes,
    hashCheckpointRecords,
    checksumScope: 'Stored footer and optional checkpoint record hashes; no full payload checksum verification or GPU state comparison', files };
}

module.exports = { mainGeometry, mainIdentity, checkpointRecords, snapshotTestCache };
