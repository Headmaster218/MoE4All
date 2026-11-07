// Experimental shared-expert Q8_0 block decoder, matching native_decode.glsl's
// FMT_Q8_0 dqblk: f16 scale followed by 32 signed quants, 34 bytes per block.
// Uses the current private w_addr and the existing BDA wide-load/native helpers.
void owned_shared_q8_dqblk32(uint gstart, out float values[32]) {
    uint block_byte = (gstart / 32u) * 34u;
    float scale = f16tof32(ru16(block_byte));
    uint quant_byte = block_byte + 2u;
    uint quant_word = quant_byte >> 2u;
    uint shift = (quant_byte & 3u) << 3u;
    uvec4 first = NW4(quant_word);
    uvec4 second = NW4(quant_word + 4u);
    if (shift == 0u) {
        for (uint word = 0u; word < 8u; ++word) {
            int quants = int(word < 4u ? first[word] : second[word - 4u]);
            for (uint byte_lane = 0u; byte_lane < 4u; ++byte_lane) {
                values[word * 4u + byte_lane] = scale
                    * float(bitfieldExtract(quants, int(8u * byte_lane), 8));
            }
        }
    } else {
        uint tail = NW(quant_word + 8u);
        for (uint word = 0u; word < 8u; ++word) {
            uint current = word < 4u ? first[word] : second[word - 4u];
            uint next_word = word + 1u;
            uint next = next_word < 4u ? first[next_word]
                : (next_word < 8u ? second[next_word - 4u] : tail);
            int quants = int((current >> shift) | (next << (32u - shift)));
            for (uint byte_lane = 0u; byte_lane < 4u; ++byte_lane) {
                values[word * 4u + byte_lane] = scale
                    * float(bitfieldExtract(quants, int(8u * byte_lane), 8));
            }
        }
    }
}
