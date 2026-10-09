use std::sync::OnceLock;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Affinity {
    pub mask: usize,
    pub group: u16,
    reserved: [u16; 3],
}

#[derive(Clone, Copy, Debug)]
struct Core {
    affinity: Affinity,
    efficiency: u8,
}

fn cores() -> &'static [Core] {
    static CORES: OnceLock<Vec<Core>> = OnceLock::new();
    CORES.get_or_init(query)
}

pub(super) fn worker_limit() -> usize {
    if cores().is_empty() {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        cores().len()
    }
}

pub(super) fn worker_affinities(offset: usize) -> Vec<Affinity> {
    ordered_affinities(cores(), offset)
}

fn ordered_affinities(cores: &[Core], offset: usize) -> Vec<Affinity> {
    let mut ordered = cores.to_vec();
    // Higher EfficiencyClass means faster cores, not higher power efficiency.
    ordered.sort_by_key(|core| std::cmp::Reverse(core.efficiency));
    if let Some(first) = ordered.first() {
        let preferred = ordered
            .iter()
            .take_while(|c| c.efficiency == first.efficiency)
            .count();
        ordered[..preferred].rotate_left(offset % preferred);
    }
    ordered.into_iter().map(|core| core.affinity).collect()
}

fn parse_core_records(bytes: &[u8]) -> Option<Vec<Core>> {
    let mut offset = 0;
    let mut cores = Vec::new();
    while offset < bytes.len() {
        let header = bytes.get(offset..offset.checked_add(8)?)?;
        let relationship = u32::from_ne_bytes(header[..4].try_into().ok()?);
        let size = u32::from_ne_bytes(header[4..8].try_into().ok()?) as usize;
        if size < 8 {
            return None;
        }
        let record = bytes.get(offset..offset.checked_add(size)?)?;
        if relationship == 0 {
            let affinity_end = 32 + std::mem::size_of::<Affinity>();
            if record.len() < affinity_end
                || u16::from_ne_bytes(record[30..32].try_into().ok()?) != 1
            {
                return None;
            }
            let mask_end = 32 + std::mem::size_of::<usize>();
            let mask = usize::from_ne_bytes(record[32..mask_end].try_into().ok()?);
            if mask == 0 {
                return None;
            }
            cores.push(Core {
                affinity: Affinity {
                    mask: 1usize << mask.trailing_zeros(),
                    group: u16::from_ne_bytes(record[mask_end..mask_end + 2].try_into().ok()?),
                    ..Affinity::default()
                },
                efficiency: record[9],
            });
        }
        offset += size;
    }
    Some(cores)
}

#[cfg(windows)]
fn query() -> Vec<Core> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetLogicalProcessorInformationEx(
            relationship: u32,
            buffer: *mut std::ffi::c_void,
            length: *mut u32,
        ) -> i32;
    }
    let mut length = 0;
    unsafe {
        GetLogicalProcessorInformationEx(0, std::ptr::null_mut(), &mut length);
    }
    if length == 0 {
        return Vec::new();
    }
    for _ in 0..3 {
        let mut buffer = vec![0u64; (length as usize).div_ceil(8)];
        let capacity = buffer.len() * 8;
        if unsafe { GetLogicalProcessorInformationEx(0, buffer.as_mut_ptr().cast(), &mut length) }
            != 0
        {
            if length as usize > capacity {
                return Vec::new();
            }
            return parse_core_records(&bytemuck::cast_slice(&buffer)[..length as usize])
                .unwrap_or_default();
        }
        if length as usize <= capacity {
            break;
        }
    }
    Vec::new()
}

#[cfg(not(windows))]
fn query() -> Vec<Core> {
    Vec::new()
}

#[cfg(windows)]
pub(super) fn set_affinity(affinity: Affinity) -> Option<Affinity> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn SetThreadGroupAffinity(
            thread: *mut std::ffi::c_void,
            affinity: *const Affinity,
            previous: *mut Affinity,
        ) -> i32;
    }
    let mut previous = Affinity::default();
    (unsafe { SetThreadGroupAffinity(GetCurrentThread(), &affinity, &mut previous) } != 0)
        .then_some(previous)
}

#[cfg(not(windows))]
pub(super) fn set_affinity(_: Affinity) -> Option<Affinity> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(mask: usize, group: u16, efficiency: u8) -> Vec<u8> {
        let mut data = vec![0u8; 32 + std::mem::size_of::<Affinity>()];
        let size = data.len() as u32;
        data[4..8].copy_from_slice(&size.to_ne_bytes());
        data[9] = efficiency;
        data[30..32].copy_from_slice(&1u16.to_ne_bytes());
        let end = 32 + std::mem::size_of::<usize>();
        data[32..end].copy_from_slice(&mask.to_ne_bytes());
        data[end..end + 2].copy_from_slice(&group.to_ne_bytes());
        data
    }

    #[test]
    fn hybrid_selection_uses_one_smt_thread_and_performance_cores_first() {
        let bytes = [record(3, 0, 2), record(4, 0, 0), record(24, 1, 2)].concat();
        let cores = parse_core_records(&bytes).unwrap();
        let selected = ordered_affinities(&cores, 1);
        assert_eq!(
            selected
                .iter()
                .map(|a| (a.group, a.mask))
                .collect::<Vec<_>>(),
            [(1, 8), (0, 1), (0, 4)]
        );
        assert_eq!(ordered_affinities(&cores, 0)[0].mask, 1);
    }

    #[test]
    fn homogeneous_empty_and_malformed_topologies_are_safe() {
        assert!(ordered_affinities(&[], 1).is_empty());
        let bytes = [record(1, 0, 0), record(2, 0, 0)].concat();
        let cores = parse_core_records(&bytes).unwrap();
        assert_eq!(ordered_affinities(&cores, 3)[0].mask, 2);
        for length in 1..record(1, 0, 0).len() {
            assert!(parse_core_records(&record(1, 0, 0)[..length]).is_none());
        }
        assert!(parse_core_records(&record(0, 0, 0)).is_none());
        assert!(parse_core_records(&[0u8; 8]).is_none());
        assert!(parse_core_records(&[]).unwrap().is_empty());
    }

    #[test]
    fn detected_worker_limit_is_bounded_and_affinities_are_distinct() {
        assert!(worker_limit() >= 1);
        let selected = worker_affinities(1);
        for (index, affinity) in selected.iter().enumerate() {
            assert_eq!(affinity.mask.count_ones(), 1);
            assert!(!selected[..index].contains(affinity));
        }
    }
}
