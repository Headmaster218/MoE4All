#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
shared uint scan_subgroup_totals[256];

uint scan_thread() {
    return gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
}

// Integer counts retain their exact descending-bin / ascending-index order.
uint exclusive_count_prefix(uint value) {
    uint prefix = subgroupExclusiveAdd(value);
    uint total = subgroupAdd(value);
    if (subgroupElect()) { scan_subgroup_totals[gl_SubgroupID] = total; }
    barrier();
    for (uint group = 0u; group < gl_SubgroupID; ++group) {
        prefix += scan_subgroup_totals[group];
    }
    return prefix;
}
