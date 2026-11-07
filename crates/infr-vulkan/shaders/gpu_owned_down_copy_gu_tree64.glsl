// Same block/rr nesting, SG sums and SiLU as owned_gu_tree64.comp.
shared float gate_part[NR][LOCAL / 32];
shared float up_part[NR][LOCAL / 32];

float mixed_dqsub(uint start, uint xoff) {
    float v[32];
    dqblk(start, v);
    float a = 0.0;
    for (uint w = 0u; w < 32u; ++w) a += v[w] * x[xoff + w];
    return a;
}

void mixed_gu(uint unit) {
    const uint outputs = uint(LOCAL / 64 * NR);
    uint tiles = (p.nff + outputs - 1u) / outputs;
    if (unit >= p.rows * p.used * tiles) return;
    uint assignment = unit / tiles, tile = unit % tiles;
    uint row = assignment / p.used, slot = assignment % p.used;
    if ((ids[p.rows * p.used + p.rows + row] & (1u << slot)) == 0u) return;
    uint expert = ids[row * p.used + slot];
    if (expert >= p.experts) return;
    uvec2 gate = tape[p.gate_tape + expert], up = tape[p.up_tape + expert];
    if (!valid_span(gate, p.gu_bytes) || !valid_span(up, p.gu_bytes)) return;
    // Copy WGs never enter this function. Every surviving GU WG is uniform here.
    GRID_INIT;
    uint wave = gl_SubgroupID, group = wave / 2u;
    uint t = (wave & 1u) * 32u + gl_SubgroupInvocationID;
    uint o0 = tile * outputs + group * uint(NR), xbase = row * p.ne;
    uint64_t ga = arena_base(gate.x, gate.y), ua = arena_base(up.x, up.y);
    float gacc[NR], uacc[NR];
    for (uint rr = 0u; rr < uint(NR); ++rr) { gacc[rr] = 0.0; uacc[rr] = 0.0; }
    for (uint s = t; s < p.ne / 32u; s += 64u) {
        for (uint rr = 0u; rr < uint(NR); ++rr) {
            uint o = o0 + rr;
            if (o >= p.nff) continue;
            w_addr = ga;
            gacc[rr] += mixed_dqsub(o * p.ne + s * 32u, xbase + s * 32u);
            w_addr = ua;
            uacc[rr] += mixed_dqsub(o * p.ne + s * 32u, xbase + s * 32u);
        }
    }
    for (uint rr = 0u; rr < uint(NR); ++rr) {
        float g = subgroupAdd(gacc[rr]), u = subgroupAdd(uacc[rr]);
        if (subgroupElect()) { gate_part[rr][wave] = g; up_part[rr][wave] = u; }
    }
    barrier();
    if (t == 0u) {
        for (uint rr = 0u; rr < uint(NR); ++rr) {
            uint o = o0 + rr;
            if (o >= p.nff) continue;
            float g = 0.0, u = 0.0;
            for (uint s = 0u; s < 2u; ++s) {
                g += gate_part[rr][group * 2u + s];
                u += up_part[rr][group * 2u + s];
            }
            output_values[(row * p.physical_used + slot) * p.nff + o] =
                (g / (1.0 + exp(-g))) * u;
        }
    }
}
