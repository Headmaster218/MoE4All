#ifdef COPY_OVERLAP
// Preserve overlap's binding8 indirect/miss predicates.
layout(std430, binding = 9) readonly buffer RouterLogits { float logits[]; };
layout(std430, binding = 10) buffer RouterWeights { float wts[]; };
#else
layout(std430, binding = 8) readonly buffer RouterLogits { float logits[]; };
layout(std430, binding = 9) buffer RouterWeights { float wts[]; };
#endif

// Words 13/14/15 are unused by owner validation: scale bits, gating, immutable predicate offset.
struct RouterParams { uint n_expert; uint n_used; float scale; uint gating; };
#define MAX_PER_LANE 16u
const uint TOPK_OWNER_MAX_LAYERS = 128u;

float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

// Exact math/control from moe_topk_sigmoid_wave32.comp; only subgroup0 executes it.
uint route_wave32() {
    RouterParams pc = RouterParams(p.experts, p.used,
                                   uintBitsToFloat(p.pool_start1), p.pool_start2);
    uint tok = gl_WorkGroupID.x;
    uint lane = gl_SubgroupInvocationID;
    uint lbase = tok * pc.n_expert;
    uint obase = tok * pc.n_used;

    float raw[MAX_PER_LANE];
    uint live = 0u;
    for (uint j = 0u; j < MAX_PER_LANE; j++) {
        uint e = lane + 32u * j;
        if (e < pc.n_expert) {
            float v = logits[lbase + e];
            raw[j] = v;
            live |= 1u << j;
        } else {
            raw[j] = -1e30;
        }
    }

    uint my_id = 0u;
    float my_logit = 0.0;
    for (uint rank = 0u; rank < pc.n_used; rank++) {
        float best = -1e30;
        uint best_id = 0xFFFFFFFFu;
        for (uint j = 0u; j < MAX_PER_LANE; j++) {
            if ((live & (1u << j)) != 0u) {
                uint e = lane + 32u * j;
                float s = raw[j];
                if (s > best) {
                    best = s;
                    best_id = e;
                }
            }
        }

        float wave_best = subgroupMax(best);
        uint winner = subgroupMin(best == wave_best ? best_id : 0xFFFFFFFFu);
        winner = min(winner, pc.n_expert - 1u);
        uint owner = winner & 31u;
        uint owner_j = winner >> 5u;
        float owner_logit = lane == owner ? raw[owner_j] : 0.0;
        float winner_logit = subgroupBroadcast(owner_logit, owner);

        if (lane == rank) {
            my_id = winner;
            my_logit = winner_logit;
        }
        if (lane == owner) { live &= ~(1u << owner_j); }
    }

    float selected_max = subgroupBroadcast(my_logit, 0u);
    float my_weight = lane < pc.n_used
        ? (pc.gating == 0u ? exp(my_logit - selected_max) : sigmoid(my_logit))
        : 0.0;
    float sum = 0.0;
    for (uint rank = 0u; rank < pc.n_used; rank++) {
        sum += subgroupBroadcast(my_weight, rank);
    }
    sum = max(sum, 1e-20);
    if (lane < pc.n_used) {
        ids[obase + lane] = my_id;
        blocks[lane] = my_id;
        wts[obase + lane] = my_weight / sum * pc.scale;
    }
    return my_id;
}

// Every lane of subgroup0 participates. Do not assume routed winners are unique:
// NaN/fallback winners and preserved IDs can repeat, including IDs outside the bank.
void classify_routed_ids(uint id) {
    uint lane = gl_SubgroupInvocationID;
    bool active_id = lane < p.count;
    uint first = lane, last = lane;
    for (uint rank = 0u; rank < p.count; ++rank) {
        uint other = subgroupBroadcast(id, rank);
        if (active_id && other == id) {
            first = min(first, rank);
            last = max(last, rank);
        }
    }
    uint unique_ids = subgroupAdd(uint(active_id && first == lane));
    if (active_id) {
        firsts[lane] = first;
        lasts[lane] = last;
        tapes[lane] = uint(first == lane) | (uint(last == lane) << 1u);
    }
    if (lane == 0u) route_unique_ids = unique_ids;
}

void main() {
    if (any(notEqual(gl_WorkGroupID, uvec3(0u)))) return;
    uint expected_word = p.ordinal == 0u ? 5u + 2u * TOPK_OWNER_MAX_LAYERS
                                       : 5u + (p.ordinal - 1u) * 2u;
    bool valid_route = p.ordinal < TOPK_OWNER_MAX_LAYERS && p.count == p.used && p.used > 0u
        && p.used <= 16u && p.experts > 0u && p.experts <= 512u
        && p.pool_start2 <= 1u && p.pool_count0 == expected_word
        && p.pool_count0 >= 5u && p.pool_count0 - 5u < uint(predicates.length())
        && p.experts <= uint(logits.length()) && p.used <= uint(wts.length())
        && p.used <= uint(ids.length());
    if (!valid_route) {
        if (gl_LocalInvocationIndex == 0u) {
            initialize();
            if (work_valid != 0u) finish(p.ordinal * JOURNAL_WORDS, 3u);
        }
        return;
    }
    // Preserve the original predicate, independently of run. Recovery never reroutes.
    // Validated input is either the previous ordinal or the separate seed word; header
    // initialization only clears this ordinal's journal and output predicate pair.
    bool route_enabled = predicates[p.pool_count0 - 5u] != 0u && p.pool_start0 == 0u;
    if (gl_SubgroupID == 0u) {
        uint id = 0u;
        if (route_enabled) id = route_wave32();
        else {
            // A false route predicate does not imply run==0. Preserve the existing IDs.
            uint lane = gl_SubgroupInvocationID;
            if (lane < p.count) {
                id = ids[lane];
                blocks[lane] = id;
            }
        }
        classify_routed_ids(id);
    } else if (gl_SubgroupID == 1u && gl_SubgroupInvocationID == 0u) {
        initialize_hit_prefix();
    }
    memoryBarrierBuffer();
    barrier();
    // Both the routed IDs and complete checked header are ready. A declined hot path
    // still takes execute_owner's original barrier + full initialize/validation fallback.
    execute_owner(true);
}
