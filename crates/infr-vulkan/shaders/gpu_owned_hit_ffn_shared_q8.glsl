// Shared-expert Q8_0 decode is now the same q8_0_dqblk() the FMT_Q8_0 kernels call: f16 scale +
// 32 signed quants (34 bytes), two rw4() wide loads, signed bitfieldExtract. The includer already
// includes native_decode.glsl (with its non-Q8_0 FMT_*), which exposes q8_0_dqblk() unconditionally.
// Keeping this a thin alias prevents the decoder from drifting between the two paths.
void owned_shared_q8_dqblk32(uint gstart, out float values[32]) {
    q8_0_dqblk(gstart, values);
}
