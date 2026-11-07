#!/bin/sh
# POSIX sh; may be piped to /bin/sh -s -- with no script file at runtime.
# Invoke before warmup. Native metadata only; never opens a GPU API/device.
# Read timings include dd/process overhead and the OS page cache, not raw disk speed.
LC_ALL=C
export LC_ALL

json_string() {
    printf '%s' "$1" | awk 'BEGIN { printf "\"" }
        { if (NR > 1) printf "\\n"; for (i=1; i<=length($0) && n<160; i++) {
            c=substr($0,i,1); if(c=="\\") printf "\\\\"; else if(c=="\"") printf "\\\"";
            else if(c=="\t") printf "\\t"; else if(c=="\r") printf "\\r";
            else if(c !~ /[[:cntrl:]]/) printf "%s",c; n++
        }} END { printf "\"" }'
}

number() { printf '%s\n' "$1" | awk '/^(0|[1-9][0-9]*)(\.[0-9]+)?$/ {printf "%s",$0;ok=1;exit} END{if(!ok)printf "null"}'; }
probe() {
    diagnostics="${diagnostics}${diagnostic_separator}\"$1\":{\"status\":\"$2\",\"reason\":\"$3\",\"source\":\"$4\"}"
    diagnostic_separator=,
}
read_native() {
    native_value=; native_status=unsupported
    if [ -e "$1" ]; then
        if [ ! -r "$1" ]; then native_status=permission_denied
        else
            native_value=$(head -c 256 "$1" 2>/dev/null)
            if [ $? -eq 0 ] && [ -n "$native_value" ]; then native_status=ok; else native_status=failed; fi
        fi
    fi
}
normalize_bdf() {
    printf '%s\n' "$1" | awk '
        /^[[:xdigit:]][[:xdigit:]]:[01][[:xdigit:]]\.[0-7]$/ { print "0000:" tolower($0); next }
        /^[[:xdigit:]][[:xdigit:]][[:xdigit:]][[:xdigit:]]:[[:xdigit:]][[:xdigit:]]:[01][[:xdigit:]]\.[0-7]$/ { print tolower($0) }'
}
pcie_rate() {
    awk -v g="$1" -v w="$2" 'BEGIN {
        r[1]=.250;r[2]=.500;r[3]=.985;r[4]=1.969;r[5]=3.938;r[6]=7.563;
        if(g in r && w+0>0) printf "%.3f",r[g]*w; else printf "null"
    }'
}
link_generation() {
    case $1 in 2.5*) printf 1 ;; 5.0*|5\ *) printf 2 ;; 8.0*|8\ *) printf 3 ;;
        16.0*|16\ *) printf 4 ;; 32.0*|32\ *) printf 5 ;; 64.0*|64\ *) printf 6 ;; *) printf null ;; esac
}

linux_pcie_path() {
    endpoint_link_status=ok; pcie_path=; pcie_path_separator=; pcie_path_count=0
    path_links_complete=1; topology_complete=0; topology_resolved=1
    bottleneck_gen=null; bottleneck_width=null; bottleneck_rate=
    resolved_path=$(readlink -f "$gpu_directory" 2>/dev/null)
    [ -n "$resolved_path" ] || resolved_path=$gpu_directory
    case $resolved_path in
        "$sys_root"/bus/pci/devices/*)
            physical_matches=$(find "$sys_root/devices" -type d -name "${gpu_directory##*/}" -print 2>/dev/null | head -n 2)
            physical_count=$(printf '%s\n' "$physical_matches" | awk 'NF{n++} END{print n+0}')
            if [ "$physical_count" -eq 1 ]; then resolved_path=$(printf '%s\n' "$physical_matches" | awk 'NF{print;exit}')
            else topology_resolved=0; fi ;;
    esac
    path_node=$resolved_path; path_depth=0
    while [ "$path_depth" -lt 16 ] && [ -r "$path_node/class" ]; do
        read_native "$path_node/current_link_speed"; node_current_gen=$(link_generation "$native_value"); node_current_speed_status=$native_status
        read_native "$path_node/current_link_width"; node_current_width=$(number "$native_value"); node_current_width_status=$native_status
        read_native "$path_node/max_link_speed"; node_max_gen=$(link_generation "$native_value"); node_max_speed_status=$native_status
        read_native "$path_node/max_link_width"; node_max_width=$(number "$native_value"); node_max_width_status=$native_status
        node_rate=$(pcie_rate "$node_current_gen" "$node_current_width")
        if [ "$node_current_speed_status" != ok ] || [ "$node_current_width_status" != ok ] ||
           [ "$node_current_gen" = null ] || [ "$node_current_width" = null ] || [ "$node_rate" = null ]; then
            path_links_complete=0
        elif [ -z "$bottleneck_rate" ] || awk -v a="$node_rate" -v b="$bottleneck_rate" 'BEGIN{exit !(a+0<b+0)}'; then
            bottleneck_gen=$node_current_gen; bottleneck_width=$node_current_width; bottleneck_rate=$node_rate
        fi
        if [ "$path_depth" -eq 0 ]; then
            current_gen=$node_current_gen; current_width=$node_current_width
            max_gen=$node_max_gen; max_width=$node_max_width
            if [ "$node_current_speed_status" != ok ]; then endpoint_link_status=$node_current_speed_status; fi
            if [ "$node_current_width_status" != ok ]; then endpoint_link_status=$node_current_width_status; fi
            if [ "$node_max_speed_status" != ok ]; then endpoint_link_status=$node_max_speed_status; fi
            if [ "$node_max_width_status" != ok ]; then endpoint_link_status=$node_max_width_status; fi
            if [ "$endpoint_link_status" = ok ] && { [ "$current_gen" = null ] || [ "$current_width" = null ] || [ "$max_gen" = null ] || [ "$max_width" = null ]; }; then
                endpoint_link_status=unsupported
            fi
            node_role=selected_gpu_endpoint
        else node_role=upstream_pci_bridge
        fi
        node_bdf=$(normalize_bdf "${path_node##*/}")
        pcie_path="$pcie_path$pcie_path_separator{\"depth\":$path_depth,\"role\":\"$node_role\",\"pci_address\":$(json_string "$node_bdf"),\"current_generation\":$node_current_gen,\"current_width\":$node_current_width,\"current_theoretical_one_way_gb_per_second\":$node_rate,\"max_generation\":$node_max_gen,\"max_width\":$node_max_width}"
        pcie_path_separator=,; pcie_path_count=$((pcie_path_count+1)); path_depth=$((path_depth+1))
        parent_node=${path_node%/*}
        [ "$parent_node" != "$path_node" ] || break
        path_node=$parent_node
    done
    if [ "$topology_resolved" -eq 1 ] && [ ! -r "$path_node/class" ]; then topology_complete=1; fi
    if [ "$topology_resolved" -ne 1 ]; then pcie_path_unknown_reason=pci_topology_resolution_failed
    elif [ "$topology_complete" -ne 1 ]; then pcie_path_unknown_reason=pci_parent_chain_incomplete
    elif [ "$current_gen" = null ] || [ "$current_width" = null ]; then pcie_path_unknown_reason=endpoint_link_properties_unavailable
    elif [ "$path_links_complete" -ne 1 ]; then pcie_path_unknown_reason=upstream_link_properties_unavailable
    else pcie_path_unknown_reason=; fi
    if [ -z "$pcie_path_unknown_reason" ]; then
        end_to_end_gen=$bottleneck_gen; end_to_end_width=$bottleneck_width; end_to_end_rate=$bottleneck_rate
        end_to_end_scope='"minimum_negotiated_bandwidth_across_complete_pci_path"'
        end_to_end_reason=null
        probe gpu.pcie_path ok complete_native_path sysfs_pci
    else
        end_to_end_gen=null; end_to_end_width=null; end_to_end_rate=null; end_to_end_scope=null
        end_to_end_reason=$(json_string "$pcie_path_unknown_reason")
        probe gpu.pcie_path unsupported "$pcie_path_unknown_reason" sysfs_pci
    fi
    upstream_bridge_count=$((pcie_path_count>0 ? pcie_path_count-1 : 0))
    if [ "$upstream_bridge_count" -gt 1 ]; then pcie_attachment=multi_bridge_path
    elif [ "$upstream_bridge_count" -eq 1 ]; then pcie_attachment=single_upstream_pci_bridge
    else pcie_attachment=endpoint_only_parent_unknown; fi
    read_native "$resolved_path/physical_slot"
    if [ "$native_status" = ok ]; then
        pcie_physical_slot="{\"designation\":$(json_string "$native_value"),\"connector_width_hint\":null,\"connector_width_hint_source\":null}"
        probe gpu.pcie_slot ok native_physical_slot sysfs_pci
    else
        pcie_physical_slot=null
        probe gpu.pcie_slot unsupported physical_slot_unavailable sysfs_pci
    fi
}

linux_memory() {
    cpu_name=$(awk -F ': *' '/^(model name|Hardware|Processor)[[:space:]]*:/ {print $2;exit}' "$proc_root/cpuinfo" 2>/dev/null)
    cpu_threads=$(awk '/^processor[[:space:]]*:/ {n++} END {if(n)print n}' "$proc_root/cpuinfo" 2>/dev/null)
    cpu="{\"name\":$(json_string "$cpu_name"),\"logical_threads\":$(number "$cpu_threads") }"
    if [ -n "$cpu_name" ] || [ -n "$cpu_threads" ]; then probe cpu ok native_processor procfs
    elif [ -e "$proc_root/cpuinfo" ] && [ ! -r "$proc_root/cpuinfo" ]; then probe cpu permission_denied unreadable procfs
    else probe cpu unsupported cpuinfo_unavailable procfs; fi
    ram_total=$(awk '/^MemTotal:/ {printf "%.0f",$2*1024;exit}' "$proc_root/meminfo" 2>/dev/null)
    if [ -n "$ram_total" ]; then probe ram.total ok native_memory_total procfs
    elif [ -e "$proc_root/meminfo" ] && [ ! -r "$proc_root/meminfo" ]; then probe ram.total permission_denied unreadable procfs
    else probe ram.total unsupported meminfo_unavailable procfs; fi
    modules=; module_separator=; module_count=0; module_status=unsupported
    # Decode only whitelisted SMBIOS type 17 fields. Serial/asset strings are never emitted.
    for raw in "$sys_root"/firmware/dmi/entries/17-*/raw; do
        [ -e "$raw" ] || continue
        if [ ! -r "$raw" ]; then module_status=permission_denied; continue; fi
        [ "$module_count" -lt 32 ] || break
        module=$(od -An -tu1 -N4096 "$raw" 2>/dev/null | awk '
            function u16(i){return b[i]+256*b[i+1]}
            function text(idx, s,i,c){s=str[idx]; gsub(/\\/,"\\\\",s);gsub(/"/,"\\\"",s);return "\"" substr(s,1,80) "\""}
            {for(i=1;i<=NF;i++) b[n++]=$i}
            END {
                len=b[1]; if(n<len || len<27 || b[0]!=17) exit;
                idx=1;for(i=len;i<n;i++){c=b[i];if(c==0){idx++;if(b[i+1]==0)break}else if(c>=32 && c<127)str[idx]=str[idx] sprintf("%c",c)}
                size=u16(12);if(size==0)exit;
                cap="null";if(size!=65535){if(size==32767 && len>=32)cap=(b[28]+256*b[29]+65536*b[30]+16777216*b[31])*1048576;else if(size>=32768)cap=(size-32768)*1024;else cap=size*1048576}
                gen[20]="DDR";gen[21]="DDR2";gen[22]="DDR2 FB-DIMM";gen[24]="DDR3";gen[26]="DDR4";gen[27]="LPDDR";gen[28]="LPDDR2";gen[29]="LPDDR3";gen[30]="LPDDR4";gen[34]="DDR5";gen[35]="LPDDR5";
                g=(b[18] in gen)?"\""gen[b[18]]"\"":"null";
                speed=u16(21);cfg=len>=34?u16(32):0;
                printf "{\"capacity_bytes\":%s,\"generation\":%s,\"speed_mt_s\":%s,\"configured_speed_mt_s\":%s,\"manufacturer\":%s,\"part_number\":%s}",(cap=="null"?cap:sprintf("%.0f",cap)),g,(speed>0&&speed<65535?speed:"null"),(cfg>0&&cfg<65535?cfg:"null"),text(b[23]),text(b[26]);
            }')
        if [ -n "$module" ]; then modules="$modules$module_separator$module"; module_separator=,; module_count=$((module_count+1))
        else module_status=failed; fi
    done
    if [ "$module_status" = unsupported ] && [ "$module_count" -gt 0 ]; then module_status=ok; fi
    probe ram.modules "$module_status" smbios_type17 sysfs_dmi
    memory_generation=$(printf '%s' "$modules" | awk 'BEGIN{RS="\"generation\":"} NR>1{split($0,a,"\"");if(a[1]==""){if(!seen[a[2]]++){g=a[2];n++}}} END{if(n==1)printf "\"%s\"",g;else if(n>1)printf "\"Mixed\"";else printf "null"}')
    ram="{\"total_bytes\":$(number "$ram_total"),\"memory_generation\":$memory_generation,\"module_count\":$module_count,\"modules\":[$modules]}"
}

pci_name() {
    pci_vendor=$(cat "$1/vendor" 2>/dev/null); pci_device=$(cat "$1/device" 2>/dev/null)
    pci_vendor=${pci_vendor#0x}; pci_device=${pci_device#0x}
    for pci_database in "$pci_ids" /usr/share/hwdata/pci.ids /usr/share/misc/pci.ids /usr/share/pci.ids; do
        [ -r "$pci_database" ] || continue
        awk -v v="$pci_vendor" -v d="$pci_device" '
            /^[0-9a-fA-F]/ {active=(tolower($1)==v)}
            active && /^\t[^\t]/ && tolower($1)==d {sub(/^\t[^ ]+  */,"");print;exit}' "$pci_database"
        return
    done
}
linux_gpu() {
    gpu=null
    if [ -z "$gpu_name$pci_address$gpu_vendor_id$gpu_device_id" ]; then
        probe gpu.selection not_applicable no_selected_gpu sysfs_pci
        probe gpu.pcie not_applicable no_selected_gpu sysfs_pci
        probe gpu.pcie_path not_applicable no_selected_gpu sysfs_pci
        probe gpu.pcie_slot not_applicable no_selected_gpu sysfs_pci; return
    fi
    wanted_bdf=$(normalize_bdf "$pci_address")
    if [ -n "$pci_address" ] && [ -z "$wanted_bdf" ]; then
        probe gpu.selection failed invalid_pci_address sysfs_pci
        probe gpu.pcie not_applicable no_unique_gpu sysfs_pci
        probe gpu.pcie_path not_applicable no_unique_gpu sysfs_pci
        probe gpu.pcie_slot not_applicable no_unique_gpu sysfs_pci; return
    fi
    gpu_count=0; gpu_directory=; matched_name=; named_devices=0
    for pci_dir in "$sys_root"/bus/pci/devices/*; do
        [ -d "$pci_dir" ] || continue
        read_native "$pci_dir/class"
        case $native_value in 0x03*) ;; *) continue ;; esac
        native_bdf=${pci_dir##*/}
        if [ -n "$wanted_bdf" ]; then
            [ "$native_bdf" = "$wanted_bdf" ] || continue
            native_name=$(pci_name "$pci_dir")
        else
            native_name=$(pci_name "$pci_dir")
            [ -z "$native_name" ] || named_devices=$((named_devices+1))
            read_native "$pci_dir/vendor"; native_vendor=${native_value#0x}
            read_native "$pci_dir/device"; native_device=${native_value#0x}
            if [ -n "$gpu_vendor_id$gpu_device_id" ]; then
                [ -z "$gpu_vendor_id" ] || [ "$(printf '%s' "$native_vendor" | tr 'A-F' 'a-f')" = "$gpu_vendor_id" ] || continue
                [ -z "$gpu_device_id" ] || [ "$(printf '%s' "$native_device" | tr 'A-F' 'a-f')" = "$gpu_device_id" ] || continue
            else
                [ "$native_name" = "$gpu_name" ] || continue
            fi
        fi
        gpu_count=$((gpu_count+1)); gpu_directory=$pci_dir; matched_name=$native_name
    done
    if [ "$gpu_count" -ne 1 ]; then
        if [ "$gpu_count" -gt 1 ]; then probe gpu.selection failed ambiguous_selected_gpu sysfs_pci
        elif [ -z "$wanted_bdf" ] && [ "$named_devices" -eq 0 ]; then probe gpu.selection unsupported native_gpu_names_unavailable sysfs_pci
        else probe gpu.selection failed selected_gpu_not_found sysfs_pci; fi
        probe gpu.pcie not_applicable no_unique_gpu sysfs_pci
        probe gpu.pcie_path not_applicable no_unique_gpu sysfs_pci
        probe gpu.pcie_slot not_applicable no_unique_gpu sysfs_pci; return
    fi
    probe gpu.selection ok unique_native_match sysfs_pci
    linux_pcie_path
    probe gpu.pcie "$endpoint_link_status" native_endpoint_link_properties sysfs_pci
    gpu="{\"name\":$(json_string "$matched_name"),\"pci_address\":$(json_string "${gpu_directory##*/}"),\"pcie_endpoint_current_generation\":$current_gen,\"pcie_endpoint_current_width\":$current_width,\"pcie_endpoint_current_theoretical_one_way_gb_per_second\":$(pcie_rate "$current_gen" "$current_width"),\"pcie_endpoint_max_generation\":$max_gen,\"pcie_endpoint_max_width\":$max_width,\"pcie_endpoint_max_theoretical_one_way_gb_per_second\":$(pcie_rate "$max_gen" "$max_width"),\"pcie_current_generation\":$current_gen,\"pcie_current_width\":$current_width,\"pcie_generation\":$current_gen,\"pcie_width\":$current_width,\"pcie_max_generation\":$max_gen,\"pcie_max_width\":$max_width,\"pcie_current_theoretical_one_way_gb_per_second\":$(pcie_rate "$current_gen" "$current_width"),\"pcie_theoretical_one_way_gb_per_second\":$(pcie_rate "$current_gen" "$current_width"),\"pcie_max_theoretical_one_way_gb_per_second\":$(pcie_rate "$max_gen" "$max_width"),\"pcie_current_scope\":\"selected_gpu_endpoint_local_link_only\",\"pcie_max_scope\":\"selected_gpu_endpoint_capability_not_slot\",\"pcie_path\":[$pcie_path],\"pcie_path_complete_to_root\":$( [ "$topology_complete" -eq 1 ] && printf true || printf false ),\"pcie_upstream_pci_bridge_count\":$upstream_bridge_count,\"pcie_attachment\":\"$pcie_attachment\",\"pcie_physical_slot\":$pcie_physical_slot,\"pcie_end_to_end_current_generation\":$end_to_end_gen,\"pcie_end_to_end_current_width\":$end_to_end_width,\"pcie_end_to_end_current_theoretical_one_way_gb_per_second\":$end_to_end_rate,\"pcie_end_to_end_scope\":$end_to_end_scope,\"pcie_end_to_end_unknown_reason\":$end_to_end_reason,\"pcie_source\":\"sysfs_pci\"}"
}

mac_memory_gpu() {
    cpu_name=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
    cpu_threads=$(sysctl -n hw.logicalcpu 2>/dev/null)
    ram_total=$(sysctl -n hw.memsize 2>/dev/null)
    cpu="{\"name\":$(json_string "$cpu_name"),\"logical_threads\":$(number "$cpu_threads") }"
    if [ -n "$cpu_name" ] && [ -n "$cpu_threads" ]; then probe cpu ok native_processor sysctl
    else probe cpu failed sysctl_unavailable sysctl; fi
    if [ -n "$ram_total" ]; then probe ram.total ok native_memory_total sysctl
    else probe ram.total failed sysctl_unavailable sysctl; fi
    mac_arch=$(uname -m)
    if [ "$mac_arch" = arm64 ]; then
        ram="{\"total_bytes\":$(number "$ram_total"),\"unified_memory\":true}"
        probe ram.modules not_applicable apple_unified_memory sysctl
    else
        ram="{\"total_bytes\":$(number "$ram_total")}"
        probe ram.modules unsupported smbios_not_exposed sysctl
    fi
    gpu=null
    probe gpu.pcie_slot not_applicable firmware_slot_not_exposed ioreg
    mac_names=
    if [ "$backend" = metal ] && [ -z "$gpu_name$pci_address" ]; then
        mac_names=$(ioreg -r -c IOGPU -l -w 0 2>/dev/null | awk '
            /"model" = / {s=$0;sub(/^.*"model" = <?"/,"",s);sub(/">?[[:space:]]*$/, "",s); print s; n++; if(n==64)exit}')
        mac_count=$(printf '%s\n' "$mac_names" | awk 'NF{n++} END{print n+0}')
        if [ "$mac_count" -eq 1 ]; then gpu_name=$(printf '%s\n' "$mac_names" | awk 'NF{print;exit}')
        elif [ "$mac_count" -gt 1 ]; then
            probe gpu.selection failed ambiguous_selected_gpu ioreg
            probe gpu.pcie not_applicable no_unique_gpu ioreg; return
        else
            probe gpu.selection unsupported native_gpu_names_unavailable ioreg
            probe gpu.pcie not_applicable no_unique_gpu ioreg; return
        fi
    fi
    if [ -z "$gpu_name$pci_address" ]; then probe gpu.selection not_applicable no_selected_gpu ioreg
    elif [ -n "$pci_address" ]; then
        if [ "$mac_arch" = arm64 ]; then probe gpu.selection not_applicable apple_gpu_has_no_pci_bdf ioreg
        else probe gpu.selection unsupported native_pci_bdf_not_exposed ioreg; fi
    else
        # Read-only IORegistry properties; capture model only, never registry paths/serials.
        if [ -z "$mac_names" ]; then
            mac_names=$(ioreg -r -c IOGPU -l -w 0 2>/dev/null | awk '
                /"model" = / {s=$0;sub(/^.*"model" = <?"/,"",s);sub(/">?[[:space:]]*$/, "",s); print s; n++; if(n==64)exit}')
        fi
        mac_matches=$(printf '%s\n' "$mac_names" | awk -v selected="$gpu_name" '$0==selected{n++} END{print n+0}')
        if [ "$mac_matches" -eq 1 ]; then
            gpu="{\"name\":$(json_string "$gpu_name")"
            if [ "$mac_arch" = arm64 ]; then gpu="$gpu,\"unified_memory\":true"; fi
            gpu="$gpu}"
            probe gpu.selection ok unique_native_match ioreg
        elif [ "$mac_matches" -gt 1 ]; then probe gpu.selection failed ambiguous_selected_gpu ioreg
        elif [ -z "$mac_names" ]; then probe gpu.selection unsupported native_gpu_names_unavailable ioreg
        else probe gpu.selection failed selected_gpu_not_found ioreg; fi
    fi
    if [ "$mac_arch" = arm64 ]; then probe gpu.pcie not_applicable apple_unified_memory ioreg
    else probe gpu.pcie unsupported native_link_properties_not_exposed ioreg; fi
}

measure_read() {
    # The timer interrupts wait and the trap reaps the active dd and timer children.
    # /usr/bin/time is native on macOS and common Linux systems; no temp files needed.
    read_stats=$(/usr/bin/time -p /bin/sh -c '
        path=$1; size=$2; mode=$3; bytes=0; ops=0; offset=0; seed=218; reader=; timer=
        finish() {
            trap "" TERM
            if [ -n "$reader" ]; then kill "$reader" 2>/dev/null; wait "$reader" 2>/dev/null; fi
            if [ -n "$timer" ]; then kill "$timer" 2>/dev/null; wait "$timer" 2>/dev/null; fi
            printf "%s %s %s\n" "$state" "$bytes" "$ops"
            exit 0
        }
        state=ok
        trap finish TERM
        (sleep 0.5; kill -TERM $$ 2>/dev/null) & timer=$!
        if [ "$mode" = sequential ]; then block=4194304; else block=65536; fi
        blocks=$((size / block)); [ "$blocks" -gt 0 ] || blocks=1
        while [ "$bytes" -lt 268435456 ]; do
            if [ "$mode" != sequential ]; then seed=$(((seed * 1103515245 + 12345) % 2147483648)); offset=$((seed % blocks)); fi
            amount=$((size - offset * block)); [ "$amount" -le "$block" ] || amount=$block
            [ $((bytes + amount)) -le 268435456 ] || break
            dd if="$path" bs="$block" skip="$offset" count=1 >/dev/null 2>/dev/null & reader=$!
            if wait "$reader"; then reader=; bytes=$((bytes + amount)); ops=$((ops + 1))
            else reader=; state=failed; break; fi
            if [ "$mode" = sequential ]; then offset=$((offset + 1)); [ $((offset * block)) -lt "$size" ] || offset=0; fi
        done
        finish
    ' read-probe "$model_path" "$file_size" "$1" 2>&1)
    read_state=$(printf '%s\n' "$read_stats" | awk '/^(ok|failed) [0-9]+ [0-9]+$/ {print $1;exit}')
    read_json=$(printf '%s\n' "$read_stats" | awk '
        /^(ok|failed) [0-9]+ [0-9]+$/ {bytes=$2;ops=$3;found=1}
        /^real [0-9.]+$/ {seconds=$2}
        END {if(!found || seconds<=0)exit;printf "{\"seconds\":%.3f,\"bytes\":%.0f,\"operations\":%.0f,\"mib_per_second\":%.3f,\"iops\":%.3f}",seconds,bytes,ops,bytes/1048576/seconds,ops/seconds}')
    if [ -z "$read_json" ]; then read_json=null; read_state=failed; fi
    probe "model_storage.$1" "${read_state:-failed}" buffered_read_includes_cache_and_process_overhead native_dd
}

model_storage() {
    storage=null
    if [ -z "$model_path" ]; then
        for storage_probe in metadata media sequential random_64k physical_reads; do probe "model_storage.$storage_probe" not_applicable no_model_path native_storage; done
        return
    fi
    storage_metadata=; metadata_status=unsupported; media_status=unsupported
    if [ "$os" = Linux ]; then
        # st_dev major:minor maps the containing filesystem, including filenames with spaces.
        device_number=$(stat -Lc '%d' -- "$model_path" 2>/dev/null)
        case $device_number in ''|*[!0-9]*) ;;
            *) major=$(((device_number >> 8 & 4095) | (device_number >> 32 & 4294963200)))
               minor=$(((device_number & 255) | (device_number >> 12 & 4294967040)))
               disk_dir=$(readlink -f "$sys_root/dev/block/$major:$minor" 2>/dev/null)
               if [ -n "$disk_dir" ]; then
                   [ ! -e "$disk_dir/partition" ] || disk_dir=${disk_dir%/*}
                   read_native "$disk_dir/device/model"; disk_model=$native_value; metadata_status=$native_status
                   read_native "$disk_dir/size"; disk_sectors=$native_value
                   disk_size=$(awk -v n="$disk_sectors" 'BEGIN{if(n~/^[0-9]+$/)printf "%.0f",n*512;else printf "null"}')
                   read_native "$disk_dir/queue/rotational"; media_status=$native_status
                   case $native_value in 0) media_type=SSD ;; 1) media_type=HDD ;; *) media_type=unknown ;; esac
                   case $disk_dir in */nvme*) bus_type=NVMe ;; */usb*) bus_type=USB ;; */ata*) bus_type=SATA ;; *) bus_type=unknown ;; esac
                   storage_metadata="\"model\":$(json_string "$disk_model"),\"size_bytes\":$disk_size,\"media_type\":$(json_string "$media_type"),\"bus_type\":$(json_string "$bus_type"),"
               fi ;;
        esac
        fs=$(stat -f -c '%T' -- "$model_path" 2>/dev/null)
        file_size=$(stat -Lc '%s' -- "$model_path" 2>/dev/null)
    else
        file_size=$(stat -f '%z' "$model_path" 2>/dev/null)
        fs=
        # Read explicit plist keys. Resolve the whole disk so volume labels are never emitted.
        volume_device=$(df -P "$model_path" 2>/dev/null | awk 'NR==2 && $1~/^\/dev\/disk[0-9]+/ {print $1}')
        if [ -n "$volume_device" ]; then
            disk_plist=$(diskutil info -plist "$volume_device" 2>/dev/null)
            fs=$(printf '%s' "$disk_plist" | plutil -extract FilesystemType raw -o - - 2>/dev/null)
            whole_disk=$(printf '%s' "$disk_plist" | plutil -extract ParentWholeDisk raw -o - - 2>/dev/null)
            case $whole_disk in disk[0-9]*)
                case $whole_disk in *[!a-z0-9]*) disk_plist= ;; *) disk_plist=$(diskutil info -plist "$whole_disk" 2>/dev/null) ;; esac ;;
                *) disk_plist= ;;
            esac
            disk_model=$(printf '%s' "$disk_plist" | plutil -extract DeviceModel raw -o - - 2>/dev/null)
            disk_size=$(printf '%s' "$disk_plist" | plutil -extract TotalSize raw -o - - 2>/dev/null)
            bus_type=$(printf '%s' "$disk_plist" | plutil -extract BusProtocol raw -o - - 2>/dev/null)
            ssd=$(printf '%s' "$disk_plist" | plutil -extract SolidState raw -o - - 2>/dev/null)
            if [ -n "$disk_size" ]; then metadata_status=ok; fi
            case $ssd in true) media_type=SSD; media_status=ok ;; false) media_type=HDD; media_status=ok ;; *) media_type=unknown ;; esac
            storage_metadata="\"model\":$(json_string "$disk_model"),\"size_bytes\":$(number "$disk_size"),\"bus_type\":$(json_string "$bus_type"),\"media_type\":$(json_string "$media_type"),"
        fi
    fi
    probe model_storage.metadata "$metadata_status" native_disk_mapping native_storage
    probe model_storage.media "$media_status" native_media_type native_storage
    probe model_storage.physical_reads unsupported no_attributed_physical_read_counter native_storage
    sequential=null; random=null
    if [ ! -f "$model_path" ]; then read_status=failed; read_reason=not_regular_file
    elif [ ! -r "$model_path" ]; then read_status=permission_denied; read_reason=unreadable
    elif [ ! -x /usr/bin/time ]; then read_status=unsupported; read_reason=native_time_unavailable
    else
        case $file_size in ''|*[!0-9]*) read_status=failed; read_reason=file_size_unavailable ;;
            *) if [ "$file_size" -lt 65536 ]; then read_status=not_applicable; read_reason=file_smaller_than_64k
               else read_status=ok; fi ;;
        esac
    fi
    if [ "$read_status" = ok ]; then
        measure_read sequential; sequential=$read_json
        measure_read random_64k; random=$read_json
    else
        probe model_storage.sequential "$read_status" "$read_reason" native_dd
        probe model_storage.random_64k "$read_status" "$read_reason" native_dd
    fi
    storage="{$storage_metadata\"filesystem\":$(json_string "$fs"),\"read_probe\":{\"mode\":\"buffered_read_only_before_benchmark\",\"target_seconds_per_probe\":0.5,\"max_bytes_per_probe\":268435456,\"sampled_file_bytes\":$(number "$file_size"),\"sequential\":$sequential,\"random_64k\":$random,\"cache_disposition\":\"unknown\",\"physical_disk_read_bytes\":null,\"physical_read_fraction\":null}}"
}

main() {
    model_path=; backend=; gpu_name=; pci_address=; gpu_vendor_id=; gpu_device_id=; vulkan_index=
    diagnostics=; diagnostic_separator=
    invalid_arguments=0
    while [ "$#" -gt 0 ]; do
        case $1 in
            --model|--backend|--gpu-name|--pci-address|--gpu-vendor-id|--gpu-device-id|--vulkan-index)
                if [ "$#" -lt 2 ]; then invalid_arguments=1; break; fi
                case $1 in
                    --model) model_path=$2 ;; --backend) backend=$(printf '%s' "$2" | tr 'A-Z' 'a-z') ;;
                    --gpu-name) gpu_name=$2 ;; --pci-address) pci_address=$2 ;;
                    --gpu-vendor-id) gpu_vendor_id=$(printf '%s' "$2" | tr 'A-F' 'a-f') ;;
                    --gpu-device-id) gpu_device_id=$(printf '%s' "$2" | tr 'A-F' 'a-f') ;;
                    --vulkan-index) vulkan_index=$2 ;;
                esac
                shift 2 ;;
            *) invalid_arguments=1; break ;;
        esac
    done
    case $gpu_vendor_id in '' ) ;; *[!0-9a-f]*) invalid_arguments=1 ;; esac
    case $gpu_device_id in '' ) ;; *[!0-9a-f]*) invalid_arguments=1 ;; esac
    case $vulkan_index in '' ) ;; *[!0-9]*) invalid_arguments=1 ;; esac
    cpu=null; ram=null; gpu=null; storage=null
    if [ "$invalid_arguments" -eq 1 ]; then probe arguments failed invalid_arguments collector
    else
        os=$(uname -s); proc_root=/proc; sys_root=/sys; pci_ids=/usr/share/hwdata/pci.ids
        case $os in Linux) linux_memory; linux_gpu; model_storage ;;
            Darwin) mac_memory_gpu; model_storage ;;
            *) probe platform unsupported unsupported_platform collector ;;
        esac
    fi
    emit_result
}

emit_result() {
    printf '{"schema_version":1,"hardware":{"cpu":%s,"ram":%s,"gpu":%s,"model_storage":%s},"collection":{"phase":"before_benchmark_warmup","probes":{%s}}}\n' "$cpu" "$ram" "$gpu" "$storage" "$diagnostics"
}

main "$@"
