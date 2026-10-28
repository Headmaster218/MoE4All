//! Test-only rendezvous around the real mapped memcpy and successful queue submit.
//! No sleeps, synthetic completion, queue idle, or changes to release/error policy.

use crate::{host_ready::HostReadyWait, VulkanShared};
use ash::vk;
use infr_core::error::{backend as be, Error, Result};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, RwLock, Weak};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fault {
    None,
    BeforeCopyError,
    AfterFirstError,
    AfterFirstPanic,
    SubmittedCancel,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Evidence {
    pub(crate) entered: bool,
    pub(crate) jobs: usize,
    pub(crate) copied: usize,
    pub(crate) submitted: bool,
    pub(crate) wait_value: u64,
    pub(crate) unsignaled_at_submit: bool,
    pub(crate) fence_pending_at_submit: bool,
    pub(crate) copied_at_submit: usize,
    pub(crate) pending_owned: bool,
    pub(crate) shared_gate_blocked: bool,
    pub(crate) controller_cancelled: bool,
    released: bool,
}

pub(crate) struct Probe {
    key: usize,
    fault: Fault,
    claimed: AtomicBool,
    gate: Arc<RwLock<()>>,
    state: Mutex<Evidence>,
    changed: Condvar,
}

fn registry() -> &'static Mutex<HashMap<usize, Weak<Probe>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<usize, Weak<Probe>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) struct Registration(pub(crate) Arc<Probe>);

impl Registration {
    pub(crate) fn new(shared: &Arc<VulkanShared>, gate: Arc<RwLock<()>>, fault: Fault) -> Self {
        let key = Arc::as_ptr(shared) as usize;
        let probe = Arc::new(Probe {
            key,
            fault,
            claimed: AtomicBool::new(false),
            gate,
            state: Mutex::new(Evidence::default()),
            changed: Condvar::new(),
        });
        assert!(registry().lock().unwrap().insert(key, Arc::downgrade(&probe)).is_none());
        Self(probe)
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        registry().lock().unwrap_or_else(|e| e.into_inner()).remove(&self.0.key);
    }
}

fn lookup(shared: &Arc<VulkanShared>) -> Option<Arc<Probe>> {
    registry().lock().unwrap_or_else(|e| e.into_inner())
        .get(&(Arc::as_ptr(shared) as usize)).and_then(Weak::upgrade)
}

pub(crate) fn claim_copy(shared: &Arc<VulkanShared>) -> Option<Arc<Probe>> {
    let probe = lookup(shared)?;
    probe.claimed.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .ok().map(|_| probe)
}

impl Probe {
    pub(crate) fn before_copy(&self, jobs: usize) -> Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.entered = true;
        state.jobs = jobs;
        self.changed.notify_all();
        // Timeout releases the CPU path as an ordinary error, allowing normal failure
        // signaling/drain. Never strand a submitted host wait on a failed test checkpoint.
        let (next, timeout) = self.changed.wait_timeout_while(
            state, Duration::from_secs(30), |s| !s.released,
        ).unwrap_or_else(|e| e.into_inner());
        state = next;
        if timeout.timed_out() && !state.released {
            return Err(be("combined test never observed leased host-wait submission"));
        }
        drop(state);
        if self.fault == Fault::BeforeCopyError {
            return Err(be("combined actual mapped copy: before-first-job error"));
        }
        Ok(())
    }

    pub(crate) fn after_copy(&self, completed: usize) -> Result<()> {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).copied = completed;
        if completed == 1 {
            match self.fault {
                Fault::AfterFirstError => return Err(be("combined actual mapped copy: partial error")),
                Fault::AfterFirstPanic => panic!("combined actual mapped copy: partial panic"),
                Fault::SubmittedCancel => return Err(Error::Aborted),
                _ => (),
            }
        }
        Ok(())
    }

    pub(crate) fn evidence(&self) -> Evidence {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

// Called only after queue_submit returned success. No assertion, blocking gate acquisition,
// or deliberate panic in the post-submit/pre-PendingSegment ownership interval.
pub(crate) fn submitted(shared: &Arc<VulkanShared>, wait: Option<&HostReadyWait>, fence: vk::Fence) {
    let Some(wait) = wait else { return };
    let Some(probe) = lookup(shared) else { return };
    let counter = unsafe { shared.device.get_semaphore_counter_value(wait.timeline.semaphore) };
    let fence_status = unsafe { shared.device.get_fence_status(fence) };
    let mut state = probe.state.lock().unwrap_or_else(|e| e.into_inner());
    if state.submitted { return; }
    state.submitted = true;
    state.wait_value = wait.value;
    state.unsignaled_at_submit = counter.is_ok_and(|value| value < wait.value);
    state.fence_pending_at_submit = matches!(fence_status, Ok(false));
    state.copied_at_submit = state.copied;
}

// Called after segments.push_back(pending), before finish_host_ready. This is the only
// release of the upload gate; the successful submit receipt must already exist.
pub(crate) fn after_queued(shared: &Arc<VulkanShared>) -> bool {
    let Some(probe) = lookup(shared) else { return false };
    let mut state = probe.state.lock().unwrap_or_else(|e| e.into_inner());
    if !state.submitted || state.released { return false; }
    state.pending_owned = true;
    state.shared_gate_blocked = matches!(probe.gate.try_write(), Err(std::sync::TryLockError::WouldBlock));
    state.controller_cancelled = probe.fault == Fault::SubmittedCancel;
    state.released = true;
    probe.changed.notify_all();
    state.controller_cancelled
}
