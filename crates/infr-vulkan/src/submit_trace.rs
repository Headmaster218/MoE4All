//! Opt-in submit flight recorder (`INFR_SUBMIT_TRACE=/path/to/events.jsonl`).
//!
//! The timestamp timeline answers "which dispatch was slow", but it changes the submission path
//! by adding timestamp queries. This recorder is deliberately dumber: every real `vkQueueSubmit`
//! and fencing wait writes one line and flushes it before proceeding. A later GPU reset therefore
//! still leaves the last submitted command buffer, its dispatch count, and its decode context on
//! disk.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use ash::vk;
use ash::vk::Handle;

static STATE: OnceLock<Option<Mutex<BufWriter<File>>>> = OnceLock::new();
static NEXT_SEQ: AtomicU64 = AtomicU64::new(1);

fn writer() -> Option<&'static Mutex<BufWriter<File>>> {
    STATE
        .get_or_init(|| {
            let path = std::env::var_os("INFR_SUBMIT_TRACE")?;
            if path.is_empty() {
                return None;
            }
            let file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(path)
                .map_err(|error| {
                    eprintln!("[submit-trace] cannot open INFR_SUBMIT_TRACE: {error}");
                    error
                })
                .ok()?;
            Some(Mutex::new(BufWriter::new(file)))
        })
        .as_ref()
}

#[inline]
pub(crate) fn enabled() -> bool {
    infr_core::timeline::flight_enabled() && writer().is_some()
}

#[derive(Clone, Copy)]
pub(crate) struct SubmitStart {
    seq: u64,
    start: Instant,
}

#[derive(Clone, Copy)]
pub(crate) struct WaitStart {
    seq: u64,
    start: Instant,
}

fn context_json() -> Option<serde_json::Value> {
    infr_core::timeline::flight_context().map(|ctx| {
        serde_json::json!({
            "step": ctx.step,
            "lanes": ctx.lanes,
            "layer": ctx.layer,
            "device": ctx.device,
            "phase": ctx.phase,
        })
    })
}

fn write_event(event: serde_json::Value) {
    let Some(state) = writer() else {
        return;
    };
    let mut out = state.lock().unwrap();
    if serde_json::to_writer(&mut *out, &event).is_err() || out.write_all(b"\n").is_err() {
        return;
    }
    let _ = out.flush();
}

fn command_buffers(submit: &vk::SubmitInfo<'_>) -> Vec<u64> {
    if submit.command_buffer_count == 0 || submit.p_command_buffers.is_null() {
        return Vec::new();
    }
    unsafe {
        submit
            .p_command_buffers
            .cast::<vk::CommandBuffer>()
            .as_ref()
            .into_iter()
            .flat_map(|first| {
                std::slice::from_raw_parts(first, submit.command_buffer_count as usize)
            })
            .map(|command| command.as_raw())
            .collect()
    }
}

pub(crate) fn begin_submit(
    queue: vk::Queue,
    submits: &[vk::SubmitInfo<'_>],
    fence: vk::Fence,
    profile_dispatches: Option<usize>,
    context: &str,
) -> Option<SubmitStart> {
    if !enabled() {
        return None;
    }
    let seq = NEXT_SEQ.fetch_add(1, Ordering::Relaxed);
    let commands: Vec<Vec<u64>> = submits.iter().map(command_buffers).collect();
    let wait_semaphores: Vec<u32> = submits
        .iter()
        .map(|submit| submit.wait_semaphore_count)
        .collect();
    let signal_semaphores: Vec<u32> = submits
        .iter()
        .map(|submit| submit.signal_semaphore_count)
        .collect();
    write_event(serde_json::json!({
        "event": "submit_begin",
        "seq": seq,
        "host_ns": infr_core::timeline::clock_ns(),
        "thread": format!("{:?}", std::thread::current().id()),
        "context_label": context,
        "queue": queue.as_raw(),
        "fence": fence.as_raw(),
        "submit_count": submits.len(),
        "dispatch_hint": profile_dispatches,
        "command_buffers": commands,
        "wait_semaphores": wait_semaphores,
        "signal_semaphores": signal_semaphores,
        "decode": context_json(),
    }));
    Some(SubmitStart {
        seq,
        start: Instant::now(),
    })
}

pub(crate) fn end_submit(start: Option<SubmitStart>, result: &std::result::Result<(), vk::Result>) {
    let Some(start) = start else {
        return;
    };
    let elapsed = start.start.elapsed();
    write_event(serde_json::json!({
        "event": if result.is_ok() { "submit_end" } else { "submit_error" },
        "seq": start.seq,
        "host_ns": infr_core::timeline::clock_ns(),
        "elapsed_ns": elapsed.as_nanos() as u64,
        "result": result.as_ref().err().map(|error| format!("{error:?}")),
    }));
}

pub(crate) fn begin_wait(kind: &str, fence: vk::Fence, queue: vk::Queue) -> Option<WaitStart> {
    if !enabled() {
        return None;
    }
    let seq = NEXT_SEQ.fetch_add(1, Ordering::Relaxed);
    write_event(serde_json::json!({
        "event": "wait_begin",
        "seq": seq,
        "host_ns": infr_core::timeline::clock_ns(),
        "thread": format!("{:?}", std::thread::current().id()),
        "kind": kind,
        "queue": queue.as_raw(),
        "fence": fence.as_raw(),
        "decode": context_json(),
    }));
    Some(WaitStart {
        seq,
        start: Instant::now(),
    })
}

pub(crate) fn end_wait(start: Option<WaitStart>, result: &std::result::Result<(), vk::Result>) {
    let Some(start) = start else {
        return;
    };
    write_event(serde_json::json!({
        "event": if result.is_ok() { "wait_end" } else { "wait_error" },
        "seq": start.seq,
        "host_ns": infr_core::timeline::clock_ns(),
        "elapsed_ns": start.start.elapsed().as_nanos() as u64,
        "result": result.as_ref().err().map(|error| format!("{error:?}")),
    }));
}

pub(crate) fn note(kind: &str, message: &str) {
    if !enabled() {
        return;
    }
    write_event(serde_json::json!({
        "event": "note",
        "host_ns": infr_core::timeline::clock_ns(),
        "thread": format!("{:?}", std::thread::current().id()),
        "kind": kind,
        "message": message,
        "decode": context_json(),
    }));
}
