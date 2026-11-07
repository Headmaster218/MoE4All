// Two-row route contract: Fixed32, WG128; general moe_topk.comp semantics,
// not moe_topk_sigmoid_wave32.comp semantics. norm_w=true, no bias/groups/hash/tap.
#ifdef COPY_OVERLAP
layout(std430, binding = 9) readonly buffer RouterLogits { float logits[]; };
layout(std430, binding = 10) buffer RouterWeights { float wts[]; };
#else
layout(std430, binding = 8) readonly buffer RouterLogits { float logits[]; };
layout(std430, binding = 9) buffer RouterWeights { float wts[]; };
#endif

const uint TWO_ROW_OWNER_MAX_LAYERS = 128u;
const uint TWO_ROW_MAX_PER_LANE = 16u;

float row_sel_score(float logit) {
    if (p.pool_start2 == 0u) return logit;
    return 1.0 / (1.0 + exp(-logit));
}

float row_score_of(float logit, float mx) {
    if (p.pool_start2 == 0u) return exp(logit - mx);
    return 1.0 / (1.0 + exp(-logit));
}

// A single subgroup owns each row. Repartitioning candidates does not change
// strict > local selection or global (maximum score, lowest ID) tie-breaking.
// Keep sel_score + zero adjustment, including sigmoid saturation ties.
void route_two_row(uint row) {
    uint lane = gl_SubgroupInvocationID;
    uint lbase = row * p.experts, obase = row * p.used;
    float raw[TWO_ROW_MAX_PER_LANE];
    uint live = 0u;
    for (uint j = 0u; j < TWO_ROW_MAX_PER_LANE; ++j) {
        uint e = lane + 32u * j;
        raw[j] = e < p.experts ? logits[lbase + e] : -1e30;
        if (e < p.experts) live |= 1u << j;
    }
    for (uint rank = 0u; rank < p.used; ++rank) {
        float best = -1e30;
        uint best_id = 0xffffffffu;
        for (uint j = 0u; j < TWO_ROW_MAX_PER_LANE; ++j) {
            if ((live & (1u << j)) != 0u) {
                uint e = lane + 32u * j;
                float score = row_sel_score(raw[j]) + 0.0;
                if (score > best) {
                    best = score;
                    best_id = e;
                }
            }
        }
        float wave_best = subgroupMax(best);
        uint winner = subgroupMin(best == wave_best ? best_id : 0xffffffffu);
        winner = min(winner, p.experts - 1u);
        if (lane == 0u) {
            ids[obase + rank] = winner;
            blocks[obase + rank] = winner;
        }
        if (lane == (winner & 31u)) live &= ~(1u << (winner >> 5u));
    }
    // Same writer reads its own IDs, just as general moe_topk weights_of().
    // Preserve selected-max and the original sequential rank sum/recompute order;
    // do not use the first selected logit or a subgroup floating reduction.
    if (lane == 0u) {
        float mx = -1e30;
        for (uint rank = 0u; rank < p.used; ++rank) {
            float value = logits[lbase + ids[obase + rank]];
            if (value > mx) mx = value;
        }
        float sum = 0.0;
        for (uint rank = 0u; rank < p.used; ++rank)
            sum += row_score_of(logits[lbase + ids[obase + rank]], mx);
        sum = max(sum, 1e-20);
        for (uint rank = 0u; rank < p.used; ++rank)
            wts[obase + rank] =
                row_score_of(logits[lbase + ids[obase + rank]], mx)
                / sum * uintBitsToFloat(p.pool_start1);
    }
}

// SG0 classifies BOTH rows as one role-major owner request. A duplicate across
// rows is a real duplicate: keep first/last and exact unique count, not row-local.
void classify_two_row_ids() {
    uint lane = gl_SubgroupInvocationID;
    bool live_id = lane < p.count;
    uint id = live_id ? blocks[lane] : 0u;
    uint first = lane, last = lane;
    for (uint rank = 0u; rank < p.count; ++rank) {
        uint other = subgroupBroadcast(id, rank);
        if (live_id && other == id) {
            first = min(first, rank);
            last = max(last, rank);
        }
    }
    uint unique_ids = subgroupAdd(uint(live_id && first == lane));
    if (live_id) {
        firsts[lane] = first;
        lasts[lane] = last;
        tapes[lane] = uint(first == lane) | (uint(last == lane) << 1u);
    }
    if (lane == 0u) route_unique_ids = unique_ids;
}

void main() {
    if (any(notEqual(gl_WorkGroupID, uvec3(0u)))) return;
    uint expected_word = p.ordinal == 0u ? 5u + 2u * TWO_ROW_OWNER_MAX_LAYERS
                                       : 5u + (p.ordinal - 1u) * 2u;
    // The parent must pin Fixed32. gl_NumSubgroups is workgroup-uniform:
    // all four subgroups enter these workgroup barriers or none enter them.
    bool valid_route = gl_NumSubgroups == 4u
        && p.ordinal < TWO_ROW_OWNER_MAX_LAYERS
        && p.used > 0u && p.used <= 16u && p.count == 2u * p.used
        && p.experts > 0u && p.experts <= 512u
        && p.pool_start2 <= 1u && p.pool_count0 == expected_word
        && p.pool_count0 >= 5u && p.pool_count0 - 5u < uint(predicates.length())
        && 2u * p.experts <= uint(logits.length())
        && p.count <= uint(wts.length()) && p.count <= uint(ids.length());
    if (!valid_route) {
        if (gl_LocalInvocationIndex == 0u) {
            initialize();
            if (work_valid != 0u) finish(p.ordinal * JOURNAL_WORDS, 3u);
        }
        return;
    }
    // Routing must run under its immutable input predicate independently of run
    // and of owner header validity. Initialization only clears THIS output pair.
    bool route_enabled = predicates[p.pool_count0 - 5u] != 0u && p.pool_start0 == 0u;
    if (gl_SubgroupID < 2u) {
        uint row = gl_SubgroupID;
        if (route_enabled) route_two_row(row);
        else {
            uint lane = gl_SubgroupInvocationID;
            if (lane < p.used) {
                uint at = row * p.used + lane;
                blocks[at] = ids[at];
            }
        }
    } else if (gl_SubgroupID == 2u && gl_SubgroupInvocationID == 0u) {
        initialize_hit_prefix();
    }
    memoryBarrierBuffer();
    barrier();
    if (gl_SubgroupID == 0u) classify_two_row_ids();
    barrier();
    // Checked FREE misses may reuse this complete two-row prefix. Invalid metadata,
    // alias failures and recovery retain the full checked fallback.
    execute_owner(true);
}
