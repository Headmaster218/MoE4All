#ifndef INFR_VERIFY_PAIR_GLSL
#define INFR_VERIFY_PAIR_GLSL

// Included only by the opt-in 32-lane hit/copy check. No dispatch or journal ABI addition.
#if defined(COPY_OVERLAP)
#error VERIFY pairs cannot use copy overlap
#endif
#ifndef VERIFY_PAIR_BINDING
#error Define the separate sidecar descriptor binding
#endif
layout(std430, binding = VERIFY_PAIR_BINDING) buffer VerifyPair { uint vp_meta[]; };
const uint VP_NONE = 0xffffffffu;
const uint VP_WORDS = 128u;
const uint VP_COMMANDS = 68u;
const uint VP_CONFIG = 104u;
shared uint vp_mates[64];
shared uint vp_bad;
shared uint vp_valid;

// All lanes call once, before any return or hit-check writes to ids. The extra shared slot
// has no routed id and is never paired; metadata indices stay row * routed_used + slot.
void vp_begin(uint count, uint experts, uint used, uint shared_mask, bool execute) {
    uint lane = gl_LocalInvocationID.x;
    if (lane == 0u) {
        vp_bad = 0u;
        vp_valid = 0u;
        if (uint(vp_meta.length()) >= VP_WORDS) {
            vp_valid = uint(execute && vp_meta[VP_CONFIG] == 0x56500001u
                && count == 40u && experts == 512u && used == 10u
                && uint(ids.length()) >= 48u
                && vp_meta[105] == 4u && vp_meta[106] == used
                && vp_meta[107] == experts && vp_meta[108] == 0x3ffu
                && ((shared_mask == 0u && vp_meta[115] == 10u)
                    || (shared_mask == 0x400u && vp_meta[115] == 11u)));
        }
    }
    for (uint a = lane; a < 64u; a += 32u) vp_mates[a] = VP_NONE;
    barrier();
    if (vp_valid != 0u) {
        for (uint a = lane; a < count; a += 32u) {
            uint id = ids[a];
            if (id >= experts) { atomicOr(vp_bad, 1u); continue; }
            uint row = a / used;
            uint own = 0u, other = 0u, mate = VP_NONE;
            for (uint slot = 0u; slot < used; ++slot) {
                own += uint(ids[row * used + slot] == id);
                uint b = (row ^ 1u) * used + slot;
                if (ids[b] == id) { ++other; mate = b; }
            }
            if (own == 1u && other == 1u) vp_mates[a] = mate;
        }
    }
    barrier();
    if (uint(vp_meta.length()) >= VP_WORDS) {
        for (uint a = lane; a < 64u; a += 32u) {
            vp_meta[a] = vp_valid != 0u && vp_bad == 0u ? vp_mates[a] : VP_NONE;
        }
    }
}

// Only lane zero calls after the resident check's final masks are known. Serial-copy
// status4 uses its effective ALL/zero masks, never the original journal H/M masks.
void vp_publish(uint status) {
    if (uint(vp_meta.length()) < VP_WORDS) return;
    bool valid = vp_valid != 0u && vp_bad == 0u
        && (status == 1u || status == 2u || status == 4u);
    uint pairs[3] = uint[3](0u, 0u, 0u);
    uint phase_active[3] = uint[3](0u, 0u, 0u);
    if (valid) {
        for (uint row = 0u; row < 4u; ++row) {
            phase_active[0] |= 0x3ffu;
            // Include the shared bit when deciding whether a phase is empty.
            phase_active[1] |= ids[40u + row];
            phase_active[2] |= ids[44u + row];
        }
        for (uint a = 0u; a < 40u; ++a) {
            uint b = vp_mates[a];
            if (b >= 40u || b <= a || vp_mates[b] != a) continue;
            ++pairs[0];
            for (uint bank = 1u; bank < 3u; ++bank) {
                uint base = 40u + (bank - 1u) * 4u;
                bool first = (ids[base + a / 10u] & (1u << (a % 10u))) != 0u;
                bool second = (ids[base + b / 10u] & (1u << (b % 10u))) != 0u;
                pairs[bank] += uint(first && second);
            }
        }
    }
    for (uint bank = 0u; bank < 3u; ++bank) vp_meta[64u + bank] = pairs[bank];
    for (uint role = 0u; role < 2u; ++role) {
        uint config = 109u + role * 3u;
        uint threshold = vp_meta[config];
        uint gx = vp_meta[config + 1u], gy = vp_meta[config + 2u];
        bool enabled = valid && (threshold == 6u || threshold == 10u)
            && gx > 0u && gx <= 65535u && gy == 1u;
        for (uint bank = 0u; bank < 3u; ++bank) {
            // Qualification measured a whole Gate/Up module, not isolated small H/M banks.
            bool phase = enabled && phase_active[bank] != 0u && !(status == 2u && bank == 0u);
            bool paired = phase && pairs[0] >= threshold && pairs[bank] != 0u;
            uint command = VP_COMMANDS + role * 18u + bank * 6u;
            vp_meta[command] = phase && !paired ? gx : 0u;
            vp_meta[command + 1u] = gy;
            vp_meta[command + 2u] = 1u;
            vp_meta[command + 3u] = paired ? gx : 0u;
            vp_meta[command + 4u] = gy;
            vp_meta[command + 5u] = 1u;
        }
    }
    vp_meta[67] = uint(valid);
}
#endif
