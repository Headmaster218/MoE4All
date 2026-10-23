// Q01 opt-in exact chronological collection, after the unchanged score-radix threshold.
// Contiguous lane ranges make an exclusive selected-count prefix the final index order.
#ifndef PARALLEL_SCAN
shared uint ordered_tie_prefix[256];
shared uint ordered_above_prefix[256];
#endif

void collect_ordered(uint t, uint blocks, uint score_base, uint dst_base,
                     uint threshold, uint rank) {
    uint begin = blocks * t / 256u;
    uint end = blocks * (t + 1u) / 256u;
    uint local_ties = 0u;
    uint local_above = 0u;
#ifdef COLLECT_MASK
    // Eight words per predicate cover <=65536 blocks. Larger rows use the exact two-read
    // collector instead of changing capacity, thresholds, or the host protocol.
    uint above_mask[8];
    uint tie_mask[8];
    bool use_mask = end - begin <= 256u;
    if (use_mask) {
        for (uint word = 0u; word < 8u; ++word) {
            uint am = 0u;
            uint tm = 0u;
            uint word_begin = begin + word * 32u;
            for (uint bit = 0u; bit < 32u && word_begin + bit < end; ++bit) {
                uint key = ordered_float(scores[score_base + word_begin + bit]);
                am |= uint(key > threshold) << bit;
                tm |= uint(key == threshold) << bit;
            }
            above_mask[word] = am;
            tie_mask[word] = tm;
            local_above += bitCount(am);
            local_ties += bitCount(tm);
        }
    } else
#endif
    {
        for (uint j = begin; j < end; ++j) {
            uint key = ordered_float(scores[score_base + j]);
            local_ties += uint(key == threshold);
            local_above += uint(key > threshold);
        }
    }

#ifdef PARALLEL_SCAN
    uint tie_ordinal = exclusive_count_prefix(local_ties);
    // exclusive_count_prefix reuses one shared table: all readers of its first result
    // must finish before any subgroup overwrites that table for the second prefix.
    barrier();
    uint above_ordinal = exclusive_count_prefix(local_above);
    uint slot = above_ordinal + min(rank, tie_ordinal);
#else
    ordered_tie_prefix[t] = local_ties;
    ordered_above_prefix[t] = local_above;
    barrier();
    if (t == 0u) {
        uint ties = 0u;
        uint above = 0u;
        for (uint lane = 0u; lane < 256u; ++lane) {
            uint tc = ordered_tie_prefix[lane];
            uint ac = ordered_above_prefix[lane];
            ordered_tie_prefix[lane] = ties;
            ordered_above_prefix[lane] = above + min(rank, ties);
            ties += tc;
            above += ac;
        }
    }
    barrier();
    uint tie_ordinal = ordered_tie_prefix[t];
    uint slot = ordered_above_prefix[t];
#endif

#ifdef COLLECT_MASK
    if (use_mask) {
        uint quota = rank - min(rank, tie_ordinal);
        for (uint word = 0u; word < 8u; ++word) {
            uint tm = tie_mask[word];
            uint count = uint(bitCount(tm));
            uint keep = min(quota, count);
            quota -= keep;
            if (keep < count) {
                uint trimmed = 0u;
                for (uint j = 0u; j < keep; ++j) {
                    uint lowest = tm & (0u - tm);
                    trimmed |= lowest;
                    tm &= tm - 1u;
                }
                tm = trimmed;
            }
            uint selected = above_mask[word] | tm;
            while (selected != 0u) {
                uint bit = uint(findLSB(selected));
                dst[dst_base + slot++] = begin + word * 32u + bit;
                selected &= selected - 1u;
            }
        }
        return;
    }
#endif
    for (uint j = begin; j < end; ++j) {
        uint key = ordered_float(scores[score_base + j]);
        bool equal = key == threshold;
        if (key > threshold || (equal && tie_ordinal < rank)) {
            dst[dst_base + slot++] = j;
        }
        tie_ordinal += uint(equal);
    }
}
