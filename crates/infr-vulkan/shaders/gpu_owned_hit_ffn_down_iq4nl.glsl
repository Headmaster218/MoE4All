// Experimental mixed-format Down decoder, copied from native_decode.glsl's optimized
// FMT_IQ4NL dqblk and packed codebook. Uses its ru16/ru32u/f16tof32 helpers and w_addr.
const uint OWNED_DOWN_IQ4NL_W[4] = uint[](
    0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u);

int owned_down_iq4nl_value(uint index) {
    return bitfieldExtract(int(OWNED_DOWN_IQ4NL_W[index >> 2u]),
                          int((index & 3u) << 3u), 8);
}

void owned_down_dqblk32(uint gstart, out float values[32]) {
    uint block_byte = (gstart / 32u) * 18u;
    float scale = f16tof32(ru16(block_byte));
    for (uint word = 0u; word < 4u; ++word) {
        uint quant_word = ru32u(block_byte + 2u + word * 4u);
        for (uint byte_lane = 0u; byte_lane < 4u; ++byte_lane) {
            uint q = (quant_word >> (8u * byte_lane)) & 0xffu;
            values[word * 4u + byte_lane] = scale * float(owned_down_iq4nl_value(q & 0xfu));
            values[word * 4u + byte_lane + 16u] = scale * float(owned_down_iq4nl_value(q >> 4u));
        }
    }
}
