# MTP head graft tool (v3) — graft a trained MTP head (safetensors, vLLM-fused
# tensor layout) into a Qwen3.5-family MoE GGUF's blk.<n>.nextn.* tensors.
#
# v3 fixes two fatal bugs in the v2 script (see docs/dev/MTP-UNPARK-ROADMAP.md §6):
#   1. v2 32-byte-aligned the INDEX offsets while writing the data contiguously
#      (and the source GGUF itself is contiguous) — the index and the data drifted
#      apart cumulatively, corrupting the whole file (output.weight read as zeros,
#      every tensor dequantized to garbage).
#   2. v2 transposed down_proj (256, 2048, 512) — the safetensors layout already
#      matches GGUF ne [512, 2048, 256] directly; the transpose scrambled the head's
#      down matrices.
# v3 also carries a byte-exact self-check: after writing, every tensor is re-read
# from the output at its declared offset and compared against the expected bytes.
#
# Usage:
#   python graft_mtp_head.py            # uses the SRC/HEAD/DST constants below
# Edit SRC / HEAD / DST for your model. The self-check must print OK before the
# output is used; see docs/dev/MTP-UNPARK-ROADMAP.md for the full diagnosis story.
# graft_v3: fix graft_v2's two fatal bugs.
#  v2 bug 1: index offsets were 32-byte aligned while the data write loop wrote
#            tensors CONTIGUOUSLY (and the source GGUF itself is contiguous) ->
#            index/data cumulative drift -> whole file garbage.
#  v2 bug 2: down_proj got an extra transpose(0,2,1) that gate/up did not ->
#            the head layer's down matrices were scrambled.
#  v3: contiguous offsets (matching the source layout), no transpose, and a
#      self-check that re-reads the written file and compares every tensor's
#      bytes against the source/expected data.
import struct, json, math
import numpy as np

SRC = r"C:\Users\zhang\Desktop\新建文件夹 (4)\1\Ornith-1.5-35B-A3B Quality\Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf"
HEAD = r"C:\Users\zhang\Desktop\新建文件夹 (4)\1\model-mtp.safetensors"
import sys
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "base"
DST = rf"C:\Users\zhang\Desktop\新建文件夹 (4)\1\MTPFIX2-{VARIANT}.gguf"

src = open(SRC, "rb")
magic, ver, n_tensors, n_kv = struct.unpack("<4sIQQ", src.read(24))
assert magic == b"GGUF"

def rd(fmt):
    return struct.unpack(fmt, src.read(struct.calcsize(fmt)))[0]

FMT = {0:("<B",1),1:("<b",1),2:("<H",2),3:("<h",2),4:("<I",4),5:("<i",4),6:("<f",4),7:("<B",1),10:("<Q",8),11:("<q",8),12:("<d",8)}
kv_start = src.tell()
for _ in range(n_kv):
    n = rd("<Q"); src.seek(n, 1)
    t = rd("<I")
    if t == 8:
        m = rd("<Q"); src.seek(m, 1)
    elif t == 9:
        at = rd("<I"); cnt = rd("<Q")
        if at == 8:
            for _ in range(cnt): m = rd("<Q"); src.seek(m, 1)
        else: src.seek(FMT[at][1]*cnt, 1)
    else:
        fmt, sz = FMT[t]; src.seek(sz, 1)
kv_end = src.tell()
src.seek(kv_start)
kv_blob = src.read(kv_end - kv_start)

infos = []
for _ in range(n_tensors):
    n = rd("<Q"); name = src.read(n).decode("utf-8", "replace")
    nd = rd("<I"); ne = [rd("<Q") for _ in range(nd)]
    dt = rd("<I"); off = rd("<Q")
    infos.append((name, ne, dt, off))
data_start = (src.tell() + 31) // 32 * 32
assert infos[0][3] == 0

def qbytes_pe(t):
    return {2:4.5, 8:34/32, 12:144/256, 13:176/256, 14:210/256, 22:20/32, 23:136/256}.get(t, 4)
def tbytes(dt, n):
    if dt == 0: return n*4
    if dt in (1,30): return n*2
    return int(math.ceil(n * qbytes_pe(dt)))

prev = 0
for (name, ne, dt, off) in infos:
    n = 1
    for d in ne: n *= d
    assert off == prev, (name, off, prev)   # source is CONTIGUOUS - keep it that way
    prev = off + tbytes(dt, n)
assert data_start + prev == __import__("os").path.getsize(SRC)
print(f"source OK: {n_tensors} tensors, data {prev/2**30:.2f} GiB")

hs = open(HEAD, "rb")
hlen = struct.unpack("<Q", hs.read(8))[0]
hhdr = json.loads(hs.read(hlen))
st_data = 8 + hlen
stm = np.memmap(HEAD, dtype=np.uint8, mode="r")

def bf16_to_f32(raw):
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u16.view(np.float32)

def q8_0_bytes(f32flat):
    n = f32flat.size
    assert n % 32 == 0
    v = f32flat.reshape(-1, 32)
    scale = (np.abs(v).max(axis=1) / 127.0).astype(np.float16).astype(np.float32)
    scale = np.where(scale == 0, np.float32(1e-30), scale)
    q = np.rint(v / scale.reshape(-1, 1)).astype(np.int8)
    out = np.empty((v.shape[0], 34), dtype=np.uint8)
    out[:, :2] = scale.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()

def st_bf16(name):
    info = hhdr[name]
    a, b = info["data_offsets"]
    return bf16_to_f32(stm[st_data+a:st_data+b].tobytes())

D_Q8, D_F32 = 8, 0
new_head = {}
def put(name, ne, dt, data):
    n = 1
    for d in ne: n *= d
    expected = {0: n*4, 8: (n//32)*34}[dt]
    assert len(data) == expected, f"{name}: {len(data)} != {expected}"
    new_head[name] = (dt, ne, data)

B = "blk.40."
fcw = st_bf16("mtp.fc.weight")
if "swapeh" in VARIANT:
    fcw = np.ascontiguousarray(fcw.reshape(2048, 2, 2048)[:, ::-1, :].reshape(2048, 4096))
put(B+"nextn.eh_proj.weight", [4096, 2048], D_Q8, q8_0_bytes(fcw))
if "swapnorms" in VARIANT:
    put(B+"nextn.enorm.weight", [2048], D_F32, st_bf16("mtp.pre_fc_norm_hidden.weight").tobytes())
    put(B+"nextn.hnorm.weight", [2048], D_F32, st_bf16("mtp.pre_fc_norm_embedding.weight").tobytes())
else:
    put(B+"nextn.enorm.weight", [2048], D_F32, st_bf16("mtp.pre_fc_norm_embedding.weight").tobytes())
    put(B+"nextn.hnorm.weight", [2048], D_F32, st_bf16("mtp.pre_fc_norm_hidden.weight").tobytes())
put(B+"attn_norm.weight", [2048], D_F32, st_bf16("mtp.layers.0.input_layernorm.weight").tobytes())
put(B+"attn_q.weight", [2048, 8192], D_Q8, q8_0_bytes(st_bf16("mtp.layers.0.self_attn.q_proj.weight")))
put(B+"attn_k.weight", [2048, 512], D_Q8, q8_0_bytes(st_bf16("mtp.layers.0.self_attn.k_proj.weight")))
put(B+"attn_v.weight", [2048, 512], D_Q8, q8_0_bytes(st_bf16("mtp.layers.0.self_attn.v_proj.weight")))
put(B+"attn_q_norm.weight", [256], D_F32, st_bf16("mtp.layers.0.self_attn.q_norm.weight").tobytes())
put(B+"attn_k_norm.weight", [256], D_F32, st_bf16("mtp.layers.0.self_attn.k_norm.weight").tobytes())
put(B+"attn_output.weight", [4096, 2048], D_Q8, q8_0_bytes(st_bf16("mtp.layers.0.self_attn.o_proj.weight")))
put(B+"post_attention_norm.weight", [2048], D_F32, st_bf16("mtp.layers.0.post_attention_layernorm.weight").tobytes())
# v3b fix: the MoE ROUTER stays F32 (original file's dtype; the MoeFfn kernel prices/reads
# the router as f32 and the trunk's routers are always F32) — graft_v3a's Q8_0 router made
# expert selection random noise (alpha exactly 0.000 over 256 cycles).
put(B+"ffn_gate_inp.weight", [2048, 256], D_F32, st_bf16("mtp.layers.0.mlp.gate.weight").tobytes())
gu = st_bf16("mtp.layers.0.mlp.experts.gate_up_proj").reshape(256, 1024, 2048)
if "swapgateup" in VARIANT:
    put(B+"ffn_gate_exps.weight", [2048, 512, 256], D_Q8, q8_0_bytes(np.ascontiguousarray(gu[:, 512:, :]).reshape(-1)))
    put(B+"ffn_up_exps.weight",   [2048, 512, 256], D_Q8, q8_0_bytes(np.ascontiguousarray(gu[:, :512, :]).reshape(-1)))
else:
    put(B+"ffn_gate_exps.weight", [2048, 512, 256], D_Q8, q8_0_bytes(np.ascontiguousarray(gu[:, :512, :]).reshape(-1)))
    put(B+"ffn_up_exps.weight",   [2048, 512, 256], D_Q8, q8_0_bytes(np.ascontiguousarray(gu[:, 512:, :]).reshape(-1)))
dp = st_bf16("mtp.layers.0.mlp.experts.down_proj")
assert dp.size == 256*2048*512
if "trdown" in VARIANT:
    dp = np.ascontiguousarray(dp.reshape(256, 2048, 512).transpose(0, 2, 1))
put(B+"ffn_down_exps.weight", [512, 2048, 256], D_Q8, q8_0_bytes(dp.reshape(-1)))
put(B+"ffn_gate_inp_shexp.weight", [2048], D_F32, st_bf16("mtp.layers.0.mlp.shared_expert_gate.weight").tobytes())
put(B+"ffn_gate_shexp.weight", [2048, 512], D_Q8, q8_0_bytes(st_bf16("mtp.layers.0.mlp.shared_expert.gate_proj.weight")))
put(B+"ffn_up_shexp.weight", [2048, 512], D_Q8, q8_0_bytes(st_bf16("mtp.layers.0.mlp.shared_expert.up_proj.weight")))
put(B+"ffn_down_shexp.weight", [512, 2048], D_Q8, q8_0_bytes(st_bf16("mtp.layers.0.mlp.shared_expert.down_proj.weight")))
put(B+"nextn.shared_head_norm.weight", [2048], D_F32, st_bf16("mtp.norm.weight").tobytes())
print(f"replacement head ready: {len(new_head)} tensors (Q8_0/F32)")

srcm = np.memmap(SRC, dtype=np.uint8, mode="r")
src_offsets = {i[0]: i[3] for i in infos}

# v3 fix: offsets are CONTIGUOUS (no 32-alignment), exactly like the source.
entries = []
off = 0
for (name, ne, old_dt, old_off) in infos:
    n = 1
    for d in ne: n *= d
    if name in new_head:
        gdt, gne, data = new_head[name]
        nb = len(data)
        entries.append((name, ne, gdt, off, data, data_start + src_offsets[name], nb))
    else:
        nb = tbytes(old_dt, n)
        entries.append((name, ne, old_dt, off, None, data_start + old_off, nb))
    off += nb                      # NO alignment - contiguous like the source

dst = open(DST, "wb")
dst.write(struct.pack("<4sIQQ", b"GGUF", 3, n_tensors, n_kv))
dst.write(kv_blob)
for (name, ne, dt, o, data, src_off, nb) in entries:
    dst.write(struct.pack("<Q", len(name.encode("utf-8"))))
    dst.write(name.encode("utf-8"))
    dst.write(struct.pack("<I", len(ne)))
    for d in ne: dst.write(struct.pack("<Q", d))
    dst.write(struct.pack("<I", dt))
    dst.write(struct.pack("<Q", o))
pad = (dst.tell() + 31) // 32 * 32 - dst.tell()
dst.write(b"\x00" * pad)

for (name, ne, dt, o, data, src_off, nb) in entries:
    if data is not None:
        dst.write(data)
    else:
        dst.write(srcm[src_off:src_off+nb].tobytes())
dst.close()
print(f"written: {DST}")
print(f"data section: {off/2**30:.2f} GiB (source was {prev/2**30:.2f} GiB)")

# ── self-check: re-read the written file and verify every tensor's bytes ──
import os
chk = open(DST, "rb")
chk.seek(0)
assert struct.unpack("<4sIQQ", chk.read(24)) == (b"GGUF", 3, n_tensors, n_kv)
chk.seek(kv_start)
_ = chk.read(kv_end - kv_start)   # skip the verbatim KV section
chk_infos = []
for _ in range(n_tensors):
    n = struct.unpack("<Q", chk.read(8))[0]
    name = chk.read(n).decode("utf-8")
    nd = struct.unpack("<I", chk.read(4))[0]
    ne = [struct.unpack("<Q", chk.read(8))[0] for _ in range(nd)]
    dt = struct.unpack("<I", chk.read(4))[0]
    o = struct.unpack("<Q", chk.read(8))[0]
    chk_infos.append((name, ne, dt, o))
data_start2 = (chk.tell() + 31) // 32 * 32
dstm = np.memmap(DST, dtype=np.uint8, mode="r")
bad = 0
for (name, ne, dt, o, data, src_off, nb) in entries:
    got = dstm[data_start2 + o : data_start2 + o + nb].tobytes()
    want = data if data is not None else srcm[src_off:src_off+nb].tobytes()
    if got != want:
        print(f"  MISMATCH at {name}")
        bad += 1
assert bad == 0, f"{bad} tensors corrupted"
print(f"self-check OK: all {len(entries)} tensors byte-exact at their declared offsets")
