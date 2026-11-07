// Experimental Fixed32/WG256 route: one subgroup per row, 3..8 rows.
// General moe_topk semantics; normalized weights, no bias/group filtering/hash.
#ifdef COPY_OVERLAP
layout(std430, binding = 9) readonly buffer RouterLogits { float logits[]; };
layout(std430, binding = 10) buffer RouterWeights { float wts[]; };
#else
layout(std430, binding = 8) readonly buffer RouterLogits { float logits[]; };
layout(std430, binding = 9) buffer RouterWeights { float wts[]; };
#endif

float many_sel_score(float value) {
    return p.pool_start2 == 0u ? value : 1.0 / (1.0 + exp(-value));
}
float many_weight_score(float value, float mx) {
    return p.pool_start2 == 0u ? exp(value - mx) : 1.0 / (1.0 + exp(-value));
}

void route_many_row(uint row) {
    uint lane = gl_SubgroupInvocationID;
    uint lbase = row * p.experts, obase = row * p.used;
    float raw[16];
    uint live = 0u;
    for (uint j = 0u; j < 16u; ++j) {
        uint e = lane + 32u * j;
        raw[j] = e < p.experts ? logits[lbase + e] : -1e30;
        if (e < p.experts) live |= 1u << j;
    }
    for (uint rank = 0u; rank < p.used; ++rank) {
        float best = -1e30;
        uint best_id = NONE;
        for (uint j = 0u; j < 16u; ++j) {
            if ((live & (1u << j)) != 0u) {
                uint e = lane + 32u * j;
                float score = many_sel_score(raw[j]) + 0.0;
                if (score > best) { best = score; best_id = e; }
            }
        }
        float wave_best = subgroupMax(best);
        uint winner = subgroupMin(best == wave_best ? best_id : NONE);
        winner = min(winner, p.experts - 1u);
        if (lane == 0u) {
            ids[obase + rank] = winner;
            blocks[obase + rank] = winner;
        }
        if (lane == (winner & 31u)) live &= ~(1u << (winner >> 5u));
    }
    // Preserve the general router's selected-max and sequential rank sum.
    if (lane == 0u) {
        float mx = -1e30;
        for (uint rank = 0u; rank < p.used; ++rank) {
            float value = logits[lbase + ids[obase + rank]];
            if (value > mx) mx = value;
        }
        float sum = 0.0;
        for (uint rank = 0u; rank < p.used; ++rank)
            sum += many_weight_score(logits[lbase + ids[obase + rank]], mx);
        sum = max(sum, 1e-20);
        for (uint rank = 0u; rank < p.used; ++rank)
            wts[obase + rank] = many_weight_score(logits[lbase + ids[obase + rank]], mx)
                / sum * uintBitsToFloat(p.pool_start1);
    }
}

// All WG lanes participate, including inactive routing subgroups. Read the
// complete shared prefix, never subgroupBroadcast with an index above 31.
void classify_many_row_ids() {
    uint lane = gl_LocalInvocationIndex;
    if (lane == 0u) route_unique_ids = 0u;
    barrier();
    for (uint i = lane; i < p.count; i += gl_WorkGroupSize.x) {
        uint id = blocks[i], first = i, last = i;
        for (uint j = 0u; j < p.count; ++j) {
            if (blocks[j] == id) { first = min(first, j); last = max(last, j); }
        }
        firsts[i] = first;
        lasts[i] = last;
        tapes[i] = uint(first == i) | (uint(last == i) << 1u);
        if (first == i) atomicAdd(route_unique_ids, 1u);
    }
    barrier();
}

void main() {
    if (any(notEqual(gl_WorkGroupID, uvec3(0u)))) return;
#ifdef OWNER_TOPK_MANY_ROW_CLASSIFY_PROBE
    // Test-only export of the exact production classifier for every count 1..128.
    // No directory validation, ownership, routing or payload operations occur.
    if (p.count == 0u || p.count > MAX_ROUTED || p.count > uint(ids.length())
        || p.count > uint(jobs.length()) || journal.length() < 1) return;
    for (uint i = gl_LocalInvocationIndex; i < p.count; i += gl_WorkGroupSize.x)
        blocks[i] = ids[i];
    barrier();
    classify_many_row_ids();
    for (uint i = gl_LocalInvocationIndex; i < p.count; i += gl_WorkGroupSize.x)
        jobs[i] = uvec4(firsts[i], lasts[i], tapes[i], blocks[i]);
    if (gl_LocalInvocationIndex == 0u) journal[0] = route_unique_ids;
#else
    uint expected_word = p.ordinal == 0u ? 261u : 5u + (p.ordinal - 1u) * 2u;
    uint rows = p.used != 0u ? p.count / p.used : 0u;
    bool valid_route = gl_NumSubgroups == 8u && p.ordinal < 128u
        && p.used > 0u && p.used <= 16u && rows >= 3u && rows <= 8u
        && p.count == rows * p.used && p.count <= MAX_ROUTED
        && p.experts > 0u && p.experts <= 512u
        && p.pool_start2 <= 1u && p.pool_count0 == expected_word
        && p.pool_count0 >= 5u && p.pool_count0 - 5u < uint(predicates.length())
        && rows * p.experts <= uint(logits.length())
        && p.count <= uint(wts.length()) && p.count <= uint(ids.length());
    if (!valid_route) {
        if (gl_LocalInvocationIndex == 0u) {
            initialize();
            if (work_valid != 0u) finish(p.ordinal * JOURNAL_WORDS, 3u);
        }
        return;
    }
    // Snapshot the immutable INPUT before initialize clears this layer's output.
    bool route_enabled = predicates[p.pool_count0 - 5u] != 0u && p.pool_start0 == 0u;
    if (gl_SubgroupID < rows) {
        if (route_enabled) route_many_row(gl_SubgroupID);
        else if (gl_SubgroupInvocationID < p.used) {
            uint at = gl_SubgroupID * p.used + gl_SubgroupInvocationID;
            blocks[at] = ids[at];
        }
    }
    memoryBarrierBuffer();
    barrier();
    classify_many_row_ids();
    // Eight rows occupy all eight SGs. Initialize only after every route and
    // classifier reader joined; initialize does not overwrite prefix scratch.
    if (gl_LocalInvocationIndex == 0u) initialize_hit_prefix();
    barrier();
    // Ordinary FREE misses can retain this checked prefix only when all three
    // roles fit one request per WG lane (<=256). Larger unions (e.g. 8x16x3)
    // and malformed metadata retain full cold revalidation before any commit.
    execute_owner(true);
#endif
}
