//! CPU-only checks for deferred segment filling and staging ownership; no Vulkan device needed.
use super::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

struct DummyBuffer(usize);

impl Buffer for DummyBuffer {
    fn len_bytes(&self) -> usize {
        self.0
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

struct HostStaging {
    bytes: Box<[u8]>,
    drops: Arc<AtomicUsize>,
}

impl Buffer for HostStaging {
    fn len_bytes(&self) -> usize {
        self.bytes.len()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Drop for HostStaging {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[derive(Default)]
struct FillExecutor {
    dedicated: bool,
    reject_allocation: bool,
    allocations: AtomicUsize,
    drops: Arc<AtomicUsize>,
    completions: AtomicUsize,
    completed_payloads: Mutex<Vec<Vec<u8>>>,
}

impl TransferExecutor for FillExecutor {
    fn materialize_staging(&self, _src: &[u8]) -> Result<(Arc<dyn Buffer>, usize, bool)> {
        panic!("fill preparation must not copy an intermediate host payload")
    }

    fn materialize_filled_staging(
        &self,
        len: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<(Arc<dyn Buffer>, usize, bool)> {
        self.allocations.fetch_add(1, Ordering::SeqCst);
        if self.reject_allocation {
            return Err(be("test staging allocation failure"));
        }
        let mut staging = HostStaging {
            bytes: vec![0; len].into_boxed_slice(),
            drops: Arc::clone(&self.drops),
        };
        let ptr = staging.bytes.as_mut_ptr() as usize;
        fill(&mut staging.bytes)?;
        Ok((Arc::new(staging), ptr, self.dedicated))
    }

    fn complete_copies_now(
        &self,
        copies: &[(Arc<dyn Buffer>, usize, DeviceTransferTarget, usize)],
    ) -> Result<()> {
        self.completions.fetch_add(1, Ordering::SeqCst);
        let mut payloads = self.completed_payloads.lock().unwrap();
        for (source, offset, target, len) in copies {
            assert_eq!(*len, target.len());
            let staging = source.as_any().downcast_ref::<HostStaging>().unwrap();
            payloads.push(staging.bytes[*offset..*offset + *len].to_vec());
        }
        Ok(())
    }

    fn relocate_device_ranges_now(
        &self,
        _copies: &[(DeviceTransferTarget, DeviceTransferTarget)],
    ) -> Result<()> {
        panic!("fill preparation must not relocate device ranges")
    }

    fn fill_target_now(
        &self,
        _target: &DeviceTransferTarget,
        _fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        panic!("fill preparation must not synchronously fill a target")
    }
}

// Deliberately omits materialize_filled_staging, just like an existing external mock.
struct LegacyExecutor;

impl TransferExecutor for LegacyExecutor {
    fn materialize_staging(&self, _src: &[u8]) -> Result<(Arc<dyn Buffer>, usize, bool)> {
        panic!("unsupported filled staging must not fall back to copying")
    }

    fn complete_copies_now(
        &self,
        _copies: &[(Arc<dyn Buffer>, usize, DeviceTransferTarget, usize)],
    ) -> Result<()> {
        panic!("preparation must not complete copies")
    }

    fn relocate_device_ranges_now(
        &self,
        _copies: &[(DeviceTransferTarget, DeviceTransferTarget)],
    ) -> Result<()> {
        panic!("preparation must not relocate device ranges")
    }

    fn fill_target_now(
        &self,
        _target: &DeviceTransferTarget,
        _fill: impl FnOnce(&mut [u8]) -> Result<()>,
    ) -> Result<()> {
        panic!("preparation must not synchronously fill a target")
    }
}

fn target(len: usize, mapped_ptr: Option<usize>, dedicated: bool) -> DeviceTransferTarget {
    // Construct an opaque endpoint with a CPU owner; DeviceTransferTarget::new needs a VkBuffer.
    DeviceTransferTarget {
        buffer: Arc::new(DummyBuffer(16 + len)),
        buffer_offset: 16,
        vk_offset: 64,
        mapped_ptr,
        len,
        dedicated_compatible: dedicated,
    }
}

#[test]
fn mapped_fill_writes_only_the_subtarget_without_executor_support() {
    let mut bytes = [0xa5u8; 24];
    let mapped = target(bytes.len(), Some(bytes.as_mut_ptr() as usize), false);
    let segment = mapped.subtarget(5, 8).unwrap();
    let ptr = segment.mapped_ptr().unwrap();
    let mut prepared = PreparedTransfer::default();
    let calls = Cell::new(0);

    SessionTransferPlan::default()
        .prepare_fill(
            &LegacyExecutor,
            &segment,
            |dst| {
                calls.set(calls.get() + 1);
                assert_eq!(dst.len(), 8);
                assert_eq!(dst.as_mut_ptr(), ptr);
                dst.fill(0x37);
                Ok(())
            },
            &mut prepared,
        )
        .unwrap();

    assert_eq!(calls.get(), 1);
    assert_eq!(&bytes[..5], &[0xa5; 5]);
    assert_eq!(&bytes[5..13], &[0x37; 8]);
    assert_eq!(&bytes[13..], &[0xa5; 11]);
    assert!(prepared.copies.is_empty());
}

#[test]
fn staged_segments_retain_in_place_payloads_and_complete_as_one_batch() {
    let executor = FillExecutor {
        dedicated: true,
        ..Default::default()
    };
    let plan = SessionTransferPlan::default();
    let mut prepared = PreparedTransfer::default();
    let mut sources = Vec::new();
    let mut destinations = Vec::new();
    let mut expected = Vec::new();
    for (len, value) in [(4, 0x17), (8, 0x5c), (12, 0xe1)] {
        let segment = target(len, None, true);
        let mut filled_ptr = 0;
        let mut calls = 0;
        destinations.push(Arc::downgrade(&segment.buffer));
        plan.prepare_fill(
            &executor,
            &segment,
            |dst| {
                calls += 1;
                assert_eq!(dst.len(), len);
                filled_ptr = dst.as_mut_ptr() as usize;
                dst.fill(value);
                Ok(())
            },
            &mut prepared,
        )
        .unwrap();
        assert_eq!(calls, 1);
        let copy = prepared.copies.last().unwrap();
        assert_eq!(copy.source_ptr, filled_ptr);
        let staging = copy.source.as_any().downcast_ref::<HostStaging>().unwrap();
        assert_eq!(staging.bytes.as_ptr() as usize, filled_ptr);
        assert_eq!(copy.source_offset, 0);
        assert_eq!(copy.target.buffer_offset(), 16);
        assert_eq!(copy.target.vk_offset(), 64);
        assert_eq!(copy.len, len);
        sources.push(Arc::downgrade(&copy.source));
        expected.push(vec![value; len]);
    }

    assert_eq!(prepared.copies.len(), 3);
    assert!(prepared.background_compatible());
    assert_eq!(executor.allocations.load(Ordering::SeqCst), 3);
    assert_eq!(executor.completions.load(Ordering::SeqCst), 0);
    assert_eq!(executor.drops.load(Ordering::SeqCst), 0);
    assert!(sources.iter().all(|owner| owner.upgrade().is_some()));
    assert!(destinations.iter().all(|owner| owner.upgrade().is_some()));

    prepared.complete_now(&executor).unwrap();

    assert_eq!(executor.completions.load(Ordering::SeqCst), 1);
    assert_eq!(*executor.completed_payloads.lock().unwrap(), expected);
    assert_eq!(executor.drops.load(Ordering::SeqCst), 3);
    assert!(sources.iter().all(|owner| owner.upgrade().is_none()));
    assert!(destinations.iter().all(|owner| owner.upgrade().is_none()));
}

#[test]
fn staged_fill_requires_both_endpoints_for_background_compatibility() {
    for staging_dedicated in [false, true] {
        for target_dedicated in [false, true] {
            let executor = FillExecutor {
                dedicated: staging_dedicated,
                ..Default::default()
            };
            let mut prepared = PreparedTransfer::default();
            SessionTransferPlan::default()
                .prepare_fill(
                    &executor,
                    &target(8, None, target_dedicated),
                    |dst| {
                        dst.fill(1);
                        Ok(())
                    },
                    &mut prepared,
                )
                .unwrap();
            assert_eq!(prepared.copies.len(), 1);
            assert_eq!(
                prepared.background_compatible(),
                staging_dedicated && target_dedicated
            );
            assert_eq!(executor.completions.load(Ordering::SeqCst), 0);
        }
    }
}

#[test]
fn staged_fill_failure_drops_only_the_failed_allocation() {
    let executor = FillExecutor::default();
    let plan = SessionTransferPlan::default();
    let mut prepared = PreparedTransfer::default();
    let segment = target(8, None, true);
    plan.prepare_fill(
        &executor,
        &segment,
        |dst| {
            dst.fill(0x17);
            Ok(())
        },
        &mut prepared,
    )
    .unwrap();
    let retained = Arc::downgrade(&prepared.copies[0].source);
    let calls = Cell::new(0);

    let error = plan
        .prepare_fill(
            &executor,
            &segment,
            |dst| {
                calls.set(calls.get() + 1);
                dst[..4].fill(0xff);
                Err(be("test segment read failure"))
            },
            &mut prepared,
        )
        .unwrap_err();

    assert!(error.to_string().contains("test segment read failure"));
    assert_eq!(calls.get(), 1);
    assert_eq!(prepared.copies.len(), 1);
    assert!(retained.upgrade().is_some());
    assert_eq!(executor.allocations.load(Ordering::SeqCst), 2);
    assert_eq!(executor.drops.load(Ordering::SeqCst), 1);
    assert_eq!(executor.completions.load(Ordering::SeqCst), 0);
    // The earlier successful segment remains completeable after a later segment fails.
    prepared.complete_now(&executor).unwrap();
    assert_eq!(executor.completions.load(Ordering::SeqCst), 1);
    assert_eq!(
        executor.completed_payloads.lock().unwrap()[0],
        vec![0x17; 8]
    );
    assert_eq!(executor.drops.load(Ordering::SeqCst), 2);
    assert!(retained.upgrade().is_none());
}

#[test]
fn allocation_failure_does_not_invoke_fill_or_append_a_copy() {
    let executor = FillExecutor {
        reject_allocation: true,
        ..Default::default()
    };
    let called = Cell::new(false);
    let mut prepared = PreparedTransfer::default();
    let error = SessionTransferPlan::default()
        .prepare_fill(
            &executor,
            &target(8, None, true),
            |_| {
                called.set(true);
                Ok(())
            },
            &mut prepared,
        )
        .unwrap_err();

    assert!(error
        .to_string()
        .contains("test staging allocation failure"));
    assert!(!called.get());
    assert!(prepared.copies.is_empty());
    assert_eq!(executor.completions.load(Ordering::SeqCst), 0);
}

#[test]
fn legacy_executor_rejects_staged_fill_before_invoking_callback() {
    let called = Cell::new(false);
    let mut prepared = PreparedTransfer::default();
    let error = SessionTransferPlan::default()
        .prepare_fill(
            &LegacyExecutor,
            &target(8, None, true),
            |_| {
                called.set(true);
                Ok(())
            },
            &mut prepared,
        )
        .unwrap_err();

    assert!(error
        .to_string()
        .contains("does not support filled staging"));
    assert!(!called.get());
    assert!(prepared.copies.is_empty());
}

#[test]
fn mapped_fill_failure_preserves_prior_staging_but_can_leave_partial_writes() {
    let executor = FillExecutor::default();
    let plan = SessionTransferPlan::default();
    let mut prepared = PreparedTransfer::default();
    plan.prepare_fill(
        &executor,
        &target(8, None, true),
        |dst| {
            dst.fill(0x17);
            Ok(())
        },
        &mut prepared,
    )
    .unwrap();
    let retained = Arc::downgrade(&prepared.copies[0].source);
    let mut bytes = [0xa5u8; 8];
    let mapped = target(bytes.len(), Some(bytes.as_mut_ptr() as usize), true);

    let error = plan
        .prepare_fill(
            &executor,
            &mapped,
            |dst| {
                dst[..4].fill(0x37);
                Err(be("test mapped fill failure"))
            },
            &mut prepared,
        )
        .unwrap_err();

    assert!(error.to_string().contains("test mapped fill failure"));
    assert_eq!(bytes, [0x37, 0x37, 0x37, 0x37, 0xa5, 0xa5, 0xa5, 0xa5]);
    assert_eq!(prepared.copies.len(), 1);
    assert_eq!(executor.allocations.load(Ordering::SeqCst), 1);
    assert_eq!(executor.completions.load(Ordering::SeqCst), 0);
    assert!(retained.upgrade().is_some());
    drop(prepared);
    assert_eq!(executor.drops.load(Ordering::SeqCst), 1);
    assert!(retained.upgrade().is_none());
}
