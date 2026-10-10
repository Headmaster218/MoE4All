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

pub(super) fn affinity_plan(threads: usize, offset: usize) -> (Vec<Affinity>, Option<Affinity>) {
    select_affinities(cores(), threads, offset)
}

fn ordered_cores(cores: &[Core], offset: usize) -> Vec<Core> {
    let mut ordered = cores.to_vec();
    // Higher EfficiencyClass means faster cores, not higher power efficiency.
    ordered.sort_by_key(|core| std::cmp::Reverse(core.efficiency));
    if let Some(first) = ordered.first() {
        let preferred = ordered
            .iter()
            .take_while(|c| c.efficiency == first.efficiency)
            .count();
        ordered[..preferred].rotate_left(offset % preferred);
        if preferred < ordered.len() {
            // Keep two fast physical cores for submission/system work until needed.
            let reserved = preferred.min(2);
            ordered[preferred - reserved..].rotate_left(reserved);
        }
    }
    ordered
}

fn select_affinities(
    cores: &[Core],
    threads: usize,
    offset: usize,
) -> (Vec<Affinity>, Option<Affinity>) {
    let ordered = ordered_cores(cores, offset);
    let used = threads.min(ordered.len());
    let controller = ordered[used..]
        .iter()
        .min_by_key(|core| std::cmp::Reverse(core.efficiency))
        .map(|core| core.affinity);
    let workers = ordered[..used].iter().map(|core| core.affinity).collect();
    (workers, controller)
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
    fn hybrid_selection_reserves_fast_cores_and_uses_one_smt_thread() {
        let bytes = [record(3, 0, 2), record(4, 0, 0), record(24, 1, 2)].concat();
        let cores = parse_core_records(&bytes).unwrap();
        let (selected, controller) = select_affinities(&cores, 1, 1);
        assert_eq!(
            selected
                .iter()
                .map(|a| (a.group, a.mask))
                .collect::<Vec<_>>(),
            [(0, 4)]
        );
        assert_eq!(controller.unwrap(), cores[2].affinity);
        let (selected, controller) = select_affinities(&cores, 3, 1);
        assert_eq!(
            selected,
            [cores[1].affinity, cores[2].affinity, cores[0].affinity]
        );
        assert!(controller.is_none());
    }

    #[test]
    fn hybrid_worker_counts_use_other_fast_cores_then_small_cores_then_reserved() {
        let cores: Vec<_> = (0..24)
            .map(|index| Core {
                affinity: Affinity {
                    mask: 1 << index,
                    ..Affinity::default()
                },
                efficiency: if index < 8 { 2 } else { 0 },
            })
            .collect();
        for threads in 0..=24 {
            let (workers, controller) = select_affinities(&cores, threads, 1);
            assert_eq!(workers.len(), threads);
            let fast = workers.iter().filter(|a| a.mask < (1 << 8)).count();
            assert_eq!(fast, threads.min(6) + threads.saturating_sub(22));
            if threads < 24 {
                let controller = controller.unwrap();
                assert!(controller.mask < (1 << 8));
                assert!(!workers.contains(&controller));
            } else {
                assert!(controller.is_none());
            }
        }
        let (workers, controller) = select_affinities(&cores, 22, 1);
        assert!(!workers.contains(&cores[0].affinity));
        assert!(!workers.contains(&cores[7].affinity));
        assert_eq!(controller.unwrap(), cores[7].affinity);
    }

    #[test]
    fn hybrid_with_one_fast_core_keeps_it_for_the_controller_until_full() {
        let cores =
            parse_core_records(&[record(3, 0, 2), record(4, 0, 0), record(8, 0, 0)].concat())
                .unwrap();
        let (workers, controller) = select_affinities(&cores, 2, 0);
        assert_eq!(workers, [cores[1].affinity, cores[2].affinity]);
        assert_eq!(controller.unwrap(), cores[0].affinity);
        let (workers, controller) = select_affinities(&cores, 3, 0);
        assert_eq!(workers.len(), 3);
        assert!(controller.is_none());
    }

    #[test]
    fn homogeneous_counts_select_distinct_physical_cores_before_smt() {
        let bytes: Vec<_> = (0..6)
            .flat_map(|index| record(3 << (index * 2), 0, 0))
            .collect();
        let cores = parse_core_records(&bytes).unwrap();
        for threads in 1..=6 {
            let (workers, controller) = select_affinities(&cores, threads, 1);
            let expected: Vec<_> = (0..threads)
                .map(|index| cores[(index + 1) % 6].affinity)
                .collect();
            assert_eq!(workers, expected);
            assert_eq!(
                controller,
                (threads < 6).then(|| cores[(threads + 1) % 6].affinity)
            );
        }
    }

    #[test]
    fn multiple_efficiency_classes_keep_lower_classes_in_performance_order() {
        let cores = parse_core_records(
            &[
                record(1, 0, 3),
                record(2, 0, 0),
                record(4, 0, 3),
                record(8, 0, 1),
                record(16, 0, 3),
            ]
            .concat(),
        )
        .unwrap();
        let (workers, controller) = select_affinities(&cores, 3, 0);
        assert_eq!(
            workers,
            [cores[0].affinity, cores[3].affinity, cores[1].affinity]
        );
        assert_eq!(controller.unwrap(), cores[2].affinity);
    }

    #[test]
    fn homogeneous_empty_and_malformed_topologies_are_safe() {
        assert_eq!(select_affinities(&[], 4, 1), (vec![], None));
        let bytes = [record(1, 0, 0), record(2, 0, 0)].concat();
        let cores = parse_core_records(&bytes).unwrap();
        let (workers, controller) = select_affinities(&cores, 1, 3);
        assert_eq!(workers, [cores[1].affinity]);
        assert_eq!(controller.unwrap(), cores[0].affinity);
        let (workers, controller) = select_affinities(&cores, 2, 3);
        assert_eq!(workers, [cores[1].affinity, cores[0].affinity]);
        assert!(controller.is_none());
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
        let (selected, controller) = affinity_plan(worker_limit(), 1);
        assert!(controller.is_none());
        for (index, affinity) in selected.iter().enumerate() {
            assert_eq!(affinity.mask.count_ones(), 1);
            assert!(!selected[..index].contains(affinity));
        }
    }
}
