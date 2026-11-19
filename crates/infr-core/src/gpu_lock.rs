//! The cross-process GPU startup lock.
//!
//! **Why this exists.** Two `infr` processes on one card do not merely share it — they *re-price*
//! it. Each backend sizes its unified VRAM arena from the free memory it can see at startup, so a
//! second process both shrinks the first's expert arena and gets a smaller one itself. That is not
//! a rounding error: measured over 7.4 / 10.4 / 11.5 / 14.5 GiB arenas at 30K context, every GiB of
//! arena is worth about 1 ms of the decode step, and a second engine takes several GiB. Before this
//! lock the only sign that a measurement was contaminated by one was the number itself.
//!
//! **Why a TCP bind on loopback.** The lock must (a) be atomic across processes, (b) release
//! itself when the holder dies — including when it is `kill -9`'d or the machine's power drops —
//! and (c) need no new dependency. A lock *file* fails (b): the kernel does not own the file, so a
//! dead holder leaves a stale one, and the only portable liveness probe in this tree
//! ([`crate::kernel_cache`]'s `kill(pid, 0)`) deliberately answers "alive" on Windows, which would
//! turn every crash into a permanently blocked start. A bound listening socket is owned by the
//! kernel: `bind()` to an in-use port fails in exactly one contender, and the port is gone the
//! instant the holder dies, by any cause. `libc`'s `flock` is not available on Windows at all.
//!
//! **Why the holder answers.** `AddrInUse` alone proves *something* holds the port, not that it is
//! `infr`: a squatted port, a Hyper-V exclusion, or a permission error would then block a legitimate
//! start. So the holder runs a one-line banner server, and the contender reads it before deciding.
//! That is what makes the refusal *evidence-based*: the check is a hard error only when a live
//! `infr` says, on the socket, that it holds this device. Everything else warns and starts.
//!
//! **Policy lives in the caller.** This module reports what it found ([`Outcome`]) and owns the
//! stable diagnostic codes; `infr-cli` turns that into the user-facing message and the `tracing`
//! warning, because that is where every other user-facing string lives.
//!
//! **What this does NOT do.** It is not a reservation: it does not stop a non-`infr` process
//! (llama.cpp, a game) from taking VRAM, a `--dev cpu` start neither takes it nor is gated by it,
//! and a local process willing to speak this banner protocol could impersonate a holder. The worst
//! case of impersonation is a refused start — the same outcome as actually running a second engine.

use std::fmt;
use std::io::{Read, Write};
use std::net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::thread::Builder;
use std::time::Duration;

/// The loopback port the lock is a `bind()` of. Chosen BELOW the Linux ephemeral floor (32768) and
/// the Windows one (49152) so ordinary outbound traffic can never collide with it by drawing this
/// number, and clear of the registered ports nearby.
pub const PORT: u16 = 28765;

/// First token of the holder's banner line — the "is this `infr`?" proof.
const MAGIC: &str = "infr-gpu-lock v1";

/// Startup refuses: another live `infr` holds the lock for the device we are about to use.
pub const CODE_CONFLICT: &str = "INFR-E1001";
/// The check was turned off by configuration, so concurrency will not be detected.
pub const CODE_BYPASSED: &str = "INFR-W1001";
/// The port is held by another live `infr`, but for a DIFFERENT device — assumed separate GPUs.
pub const CODE_OTHER_DEVICE: &str = "INFR-W1002";
/// The port is held, but the holder is not provably `infr` — or we could not ask it.
pub const CODE_NOT_IDENTIFIABLE: &str = "INFR-W1003";
/// We could not bind the lock port at all (permission, reserved range), so nothing was checked.
pub const CODE_BIND_FAILED: &str = "INFR-W1004";

/// The environment spelling of the bypass (config path: [`CONFIG_PATH`]).
pub const ENV_BYPASS: &str = "INFR_NO_GPU_LOCK";
/// The `Config` path of the bypass — `--set device.gpu_lock=false` does the same thing.
pub const CONFIG_PATH: &str = "device.gpu_lock";

/// How long to wait for the holder to accept us, and to answer.
const CONNECT_TIMEOUT: Duration = Duration::from_millis(300);
const READ_TIMEOUT: Duration = Duration::from_millis(600);
/// Banner cap: a line this long is suspicious, and the fields we need come first.
const BANNER_MAX_BYTES: usize = 1024;
/// Field cap for the caller-supplied label, so one long argv cannot bloat the banner.
const LABEL_MAX_CHARS: usize = 200;

/// What the holder reported about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Holder {
    /// The holder's process id, as the holder itself reported it.
    pub pid: u32,
    /// The device spec the holder selected (`vulkan`, `Vulkan1,Vulkan2`, `metal`, …).
    pub device: String,
    /// The holder's invocation, e.g. `serve -m model.gguf`.
    pub label: String,
}

impl Holder {
    /// The banner line this process would answer with. `pid`/`device`/`label` are tab-separated
    /// `key=value` fields on one line; control bytes are stripped out of the label so one
    /// invocation cannot forge a second banner line or a field boundary.
    fn banner(device: &str, label: &str) -> String {
        let clean = label
            .chars()
            .filter(|c| !c.is_control())
            .take(LABEL_MAX_CHARS)
            .collect::<String>();
        format!(
            "{MAGIC}\tpid={}\tdevice={device}\tlabel={clean}\n",
            std::process::id()
        )
    }

    /// Parse a banner line. `None` unless it starts with [`MAGIC`] and carries a pid and a device,
    /// which is what makes "the holder is `infr`" a claim we can actually check.
    fn from_banner(line: &str) -> Option<Self> {
        // The terminator is ours, not part of the last field (`label` is last, so an untrimmed
        // `\n` would end up inside the invocation we print back to the user).
        let line = line.trim_end_matches(['\r', '\n']);
        let rest = line.strip_prefix(MAGIC)?;
        let mut pid = None;
        let mut device = None;
        let mut label = String::new();
        for field in rest.split('\t') {
            if let Some(value) = field.strip_prefix("pid=") {
                pid = value.parse::<u32>().ok();
            } else if let Some(value) = field.strip_prefix("device=") {
                device = Some(value.to_string());
            } else if let Some(value) = field.strip_prefix("label=") {
                label = value.to_string();
            }
        }
        Some(Self {
            pid: pid?,
            device: device?,
            label,
        })
    }
}

/// Proof that THIS process holds the lock, and the only way to give it back early.
///
/// Dropping it releases the port: the release flag is set and one self-connect wakes the banner
/// thread, which checks the flag before answering and then drops the listener. That matters because
/// closing a socket while another thread is BLOCKED in `accept()` is not portable — on Linux the fd
/// number can be handed to an unrelated `open()` under the blocked call. The wake-up connect is how
/// the thread stops itself.
///
/// There is one lock per process, so a second [`acquire`] hands back the SAME flag: dropping either
/// guard releases. `infr-cli` binds the first one for the whole start and never drops early.
#[derive(Debug)]
pub struct Lock {
    /// The device spec we claimed the lock for — kept for diagnostics.
    pub device: String,
    port: u16,
    released: Arc<AtomicBool>,
}

impl Drop for Lock {
    fn drop(&mut self) {
        self.released.store(true, Ordering::Relaxed);
        // Wake the accept loop out of its blocking `accept` so it can see the flag and let go of
        // the listener. Failing to connect only means the port is released at process exit.
        let _ = TcpStream::connect(lock_addr_on(self.port));
    }
}

/// The release flags of the locks THIS process has bound, keyed by port, so a re-`acquire` of a
/// port we already hold hands back the same flag. Keyed rather than a single slot because the port
/// is what identifies a lock, and because a test that holds two of them must not have one's drop
/// wake the other.
type Held = Vec<(u16, Arc<AtomicBool>)>;

static HELD: OnceLock<Mutex<Held>> = OnceLock::new();

fn held() -> &'static Mutex<Held> {
    HELD.get_or_init(|| Mutex::new(Vec::new()))
}

/// Lock the registry without panicking on a poisoned guard: the banner thread only touches sockets,
/// so a panic elsewhere cannot leave this state inconsistent in a way that matters.
fn locked() -> MutexGuard<'static, Held> {
    held()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Record a bound lock, pruning the ones already released.
fn register(port: u16, released: &Arc<AtomicBool>) {
    let mut entries = locked();
    entries.retain(|(_, flag)| !flag.load(Ordering::Relaxed));
    entries.push((port, Arc::clone(released)));
}

/// The flag of the lock this process holds on `port`, if any.
fn flag_for(port: u16) -> Option<Arc<AtomicBool>> {
    locked()
        .iter()
        .find(|(held, _)| *held == port)
        .map(|(_, flag)| Arc::clone(flag))
}

/// A start that the lock did NOT block, and why.
#[derive(Debug)]
pub enum Skipped {
    /// A live `infr` holds the port for a different device: two GPUs, two processes, legitimate.
    OtherDevice(Holder),
    /// The port is busy and the holder did not answer as `infr` — a foreign service, an `infr`
    /// older than this code, or our own probe timing out.
    NotIdentifiable(String),
    /// `bind()` failed with something other than "address in use".
    BindFailed(std::io::Error),
}

/// What [`acquire`] found.
#[derive(Debug)]
pub enum Outcome {
    /// We hold the lock (or already held it — [`acquire`] is idempotent within a process).
    Held(Lock),
    /// A live `infr` holds the lock for the SAME device. This is the refusal.
    Conflict(Holder),
    /// Nothing blocked the start.
    Skipped(Skipped),
}

fn lock_addr_on(port: u16) -> SocketAddr {
    SocketAddr::from((Ipv4Addr::LOCALHOST, port))
}

/// Do these two `device.dev` specs name the same card?
///
/// A spec is a comma-separated list (a tensor-parallel pair is one lock, because it is one set of
/// devices), compared case-insensitively as SETS, with one extra rule: a token with no index — the
/// bare family name `vulkan` — is a WILDCARD, because "the engine picks" cannot be proven disjoint
/// from "Vulkan0". So `vulkan` conflicts with `vulkan` AND with `Vulkan0` (the one-GPU case, where
/// an explicit `--dev Vulkan0` and a default start are the same card), `Vulkan0,Vulkan1` conflicts
/// with `Vulkan1`, and `Vulkan0` against `Vulkan1` does not — that is the legitimate
/// one-engine-per-card setup.
fn shares_device(mine: &str, theirs: &str) -> bool {
    let tokens = |spec: &str| -> Vec<String> {
        spec.split(',')
            .map(|t| t.trim().to_lowercase())
            .filter(|t| !t.is_empty())
            .collect()
    };
    let theirs = tokens(theirs);
    tokens(mine)
        .iter()
        .any(|device| theirs.iter().any(|other| overlaps(device, other)))
}

/// One pair of device tokens: equal, or one of them is the unindexed family name of the other.
fn overlaps(a: &str, b: &str) -> bool {
    let family = |t: &str| !t.chars().any(|c| c.is_ascii_digit());
    a == b || (b.starts_with(a) && family(a)) || (a.starts_with(b) && family(b))
}

/// Ask the current holder who it is. `None` if nobody answers as `infr` within the timeouts — which
/// is the honest answer for a foreign service, and the reason a refusal needs the banner, not just
/// a failed bind.
fn ask_holder(addr: SocketAddr) -> Option<Holder> {
    let mut stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT).ok()?;
    stream.set_read_timeout(Some(READ_TIMEOUT)).ok()?;
    let mut buf = Vec::with_capacity(256);
    let mut chunk = [0u8; 128];
    while buf.len() < BANNER_MAX_BYTES {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if chunk[..n].contains(&b'\n') {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    let line = String::from_utf8_lossy(&buf)
        .split('\n')
        .next()?
        .to_string();
    Holder::from_banner(&line)
}

/// Answer the banner from a dedicated thread, on the socket that IS the lock.
///
/// The listener lives (and dies) with this thread, which is what makes [`Lock`]'s release work.
/// Errors are tolerated, not fatal: a banner thread that gives up only degrades a later contender
/// from "refused" to "warned and started", because the bind — not the thread — is the lock.
fn serve_banner(listener: TcpListener, banner: String, released: Arc<AtomicBool>) {
    let _ = Builder::new().name("infr-gpu-lock".into()).spawn(move || {
        let mut failures = 0usize;
        for stream in listener.incoming() {
            match stream {
                Ok(mut stream) => {
                    if released.load(Ordering::Relaxed) {
                        // Our own wake-up connect (see `Lock::drop`): stop answering and let the
                        // listener — and with it the port — go.
                        return;
                    }
                    let _ = stream.write_all(banner.as_bytes());
                    let _ = stream.shutdown(Shutdown::Write);
                }
                // An accept that keeps failing would spin, so stop answering after a few.
                Err(_) => {
                    failures += 1;
                    if failures > 16 || released.load(Ordering::Relaxed) {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
    });
}

/// Try to take the machine's GPU startup lock for `device`, advertising `label` (the invocation)
/// to whoever asks. Never blocks the caller for longer than the probe timeouts.
#[must_use = "the Outcome owns the lock: dropping it releases the port and lets a second infr start"]
pub fn acquire(device: &str, label: &str) -> Outcome {
    acquire_on(PORT, device, label)
}

/// [`acquire`] on an explicit port — the unit tests use an ephemeral one, so they can neither lock
/// out an engine running on this machine nor be locked out by one.
pub fn acquire_on(port: u16, device: &str, label: &str) -> Outcome {
    let addr = lock_addr_on(port);
    match TcpListener::bind(addr) {
        Ok(listener) => {
            let released = Arc::new(AtomicBool::new(false));
            register(port, &released);
            serve_banner(listener, Holder::banner(device, label), released.clone());
            Outcome::Held(Lock {
                device: device.to_string(),
                port,
                released,
            })
        }
        Err(error) if error.kind() == std::io::ErrorKind::AddrInUse => match ask_holder(addr) {
            // Our own banner: this process already holds the lock, so hand back the SAME release
            // flag — dropping either guard releases. If the registry somehow has no flag for this
            // port, the guard falls back to never releasing, and process exit still does.
            Some(holder) if holder.pid == std::process::id() => Outcome::Held(Lock {
                device: device.to_string(),
                port,
                released: flag_for(port).unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
            }),
            Some(holder) if shares_device(device, &holder.device) => Outcome::Conflict(holder),
            Some(holder) => Outcome::Skipped(Skipped::OtherDevice(holder)),
            None => Outcome::Skipped(Skipped::NotIdentifiable(format!(
                "127.0.0.1:{port} is bound, but its holder did not answer as infr within {}ms",
                CONNECT_TIMEOUT.as_millis() + READ_TIMEOUT.as_millis()
            ))),
        },
        Err(error) => Outcome::Skipped(Skipped::BindFailed(error)),
    }
}

impl fmt::Display for Holder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "pid {} (device `{}`, `{}`)",
            self.pid, self.device, self.label
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The banner must round-trip into a [`Holder`], including a label full of the characters the
    /// protocol uses as separators.
    #[test]
    fn banner_round_trips_and_survives_a_messy_label() {
        let line = Holder::banner("Vulkan0,Vulkan1", "bench\t-m \"a\nb\" -r 3");
        let holder = Holder::from_banner(&line).expect("our own banner parses");
        assert_eq!(holder.pid, std::process::id());
        assert_eq!(holder.device, "Vulkan0,Vulkan1");
        assert!(!holder.label.contains('\n'), "one line stays one line");
        assert!(holder.label.contains("bench"));
    }

    /// A foreign banner is not `infr`, and a truncated one is not either — both must fail to parse
    /// rather than produce a refusal.
    #[test]
    fn only_an_infr_banner_parses() {
        assert!(Holder::from_banner("nginx/1.25 active").is_none());
        assert!(
            Holder::from_banner("infr-gpu-lock v1\tpid=notanumber\tdevice=vulkan").is_none(),
            "a pid that is not a number is not evidence of anything"
        );
        assert!(
            Holder::from_banner("infr-gpu-lock v1\tpid=7").is_none(),
            "no device"
        );
        assert!(Holder::from_banner("infr-gpu-lock v2\tpid=7\tdevice=vulkan").is_none());
    }

    /// The label is capped and stripped of control bytes: a hostile or enormous argv cannot turn the
    /// banner into a multi-line or unbounded reply.
    #[test]
    fn banner_label_is_capped_and_sanitised() {
        let line = Holder::banner("vulkan", &"x".repeat(4000));
        assert!(
            line.len() < LABEL_MAX_CHARS + 96,
            "bounded reply: {}",
            line.len()
        );
        assert_eq!(line.matches('\n').count(), 1);
    }

    /// A tensor-parallel pair and a single-card process DO share a card, so the specs must be
    /// compared as sets, not as strings; two disjoint sets are two GPUs.
    #[test]
    fn device_specs_conflict_by_intersection() {
        for (mine, theirs) in [
            ("vulkan", "vulkan"),
            ("Vulkan0", "vulkan0"),
            ("Vulkan0", "Vulkan0,Vulkan1"),
            ("Vulkan1,Vulkan0", "Vulkan0"),
            // The unindexed family name is a wildcard: on one card, a default start and an
            // explicit `--dev Vulkan0` are the same GPU.
            ("vulkan", "Vulkan0"),
            ("Vulkan0", "vulkan"),
            ("vulkan", "Vulkan0,Vulkan1"),
        ] {
            assert!(
                shares_device(mine, theirs),
                "{mine} vs {theirs} shares a card"
            );
        }
        for (mine, theirs) in [
            ("Vulkan0", "Vulkan1"),
            ("Vulkan1,Vulkan2", "Vulkan0"),
            ("metal", "vulkan"),
            ("vulkan", "metal"),
        ] {
            assert!(!shares_device(mine, theirs), "{mine} vs {theirs} does not");
        }
    }

    /// An empty port is acquirable, and acquiring it twice in the holder's own process is
    /// idempotent — the second probe reads OUR banner, whose pid is ours.
    ///
    /// Uses an EPHEMERAL port, never [`PORT`]: the unit tests must not be able to lock out an
    /// engine running on the same machine, and must not fail because one is.
    #[test]
    fn an_empty_port_is_acquired_and_reacquire_is_idempotent() {
        let listener = TcpListener::bind(lock_addr_on(0)).expect("ephemeral bind");
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let outcome = acquire_on(port, "vulkan", "serve -m model.gguf");
        assert!(
            matches!(outcome, Outcome::Held(_)),
            "an empty port is acquirable: {outcome:?}"
        );
        // Same process, same port: our own banner answers, so this is not a conflict.
        let again = acquire_on(port, "vulkan", "bench -m model.gguf");
        assert!(
            matches!(again, Outcome::Held(_)),
            "re-acquiring in the holder is idempotent: {again:?}"
        );
        // A DIFFERENT device from the same (own) pid is still our own lock.
        assert!(matches!(
            acquire_on(port, "metal", "run -m model.gguf"),
            Outcome::Held(_)
        ));
    }

    /// A live holder that answers for another device must NOT refuse us: multi-GPU users run one
    /// engine per card.
    #[test]
    fn a_different_device_does_not_refuse() {
        let foreign = TcpListener::bind(lock_addr_on(0)).expect("ephemeral bind");
        let port = foreign.local_addr().unwrap().port();
        let other = std::process::id() + 1;
        let banner = format!("{MAGIC}\tpid={other}\tdevice=Vulkan1\tlabel=serve -m other.gguf\n");
        let answer = std::thread::spawn(move || {
            for mut stream in foreign.incoming().flatten() {
                let _ = stream.write_all(banner.as_bytes());
                let _ = stream.shutdown(Shutdown::Write);
            }
        });
        let outcome = acquire_on(port, "Vulkan0", "bench -m model.gguf");
        match outcome {
            Outcome::Skipped(Skipped::OtherDevice(holder)) => {
                assert_eq!(holder.pid, other);
                assert_eq!(holder.device, "Vulkan1");
                assert_eq!(holder.label, "serve -m other.gguf");
            }
            other => panic!("expected a different-device skip, got {other:?}"),
        }
        // …and the same device from that holder IS the refusal.
        assert!(matches!(
            acquire_on(port, "Vulkan1", "bench -m model.gguf"),
            Outcome::Conflict(_)
        ));
        // …and so is an UNQUALIFIED ask, because "the engine picks" cannot be proven to be a
        // different card than the one the holder named.
        assert!(matches!(
            acquire_on(port, "vulkan", "bench -m model.gguf"),
            Outcome::Conflict(_)
        ));
        drop(answer);
    }

    /// A port bound by something that says nothing `infr`-shaped is a warning, never a refusal:
    /// the lock must not be able to brick a start on evidence it does not have.
    #[test]
    fn a_foreign_holder_never_refuses() {
        let squatter = TcpListener::bind(lock_addr_on(0)).expect("ephemeral bind");
        let port = squatter.local_addr().unwrap().port();
        let answer = std::thread::spawn(move || {
            for mut stream in squatter.incoming().flatten() {
                let _ = writeln!(stream, "HTTP/1.1 400 Bad Request");
            }
        });
        let outcome = acquire_on(port, "vulkan", "bench -m model.gguf");
        assert!(
            matches!(outcome, Outcome::Skipped(Skipped::NotIdentifiable(_))),
            "a non-infr holder must not refuse: {outcome:?}"
        );
        drop(answer);
    }

    /// A port nobody is answering on at all (held by a raw socket, or firewalled) also degrades to
    /// a skip rather than a refusal or a hang.
    #[test]
    fn a_silent_holder_times_out_into_a_skip() {
        let silent = TcpListener::bind(lock_addr_on(0)).expect("ephemeral bind");
        let port = silent.local_addr().unwrap().port();
        // Accept nothing: the backlog holds the contender's connection open with no reply.
        std::mem::forget(silent);
        let started = std::time::Instant::now();
        let outcome = acquire_on(port, "vulkan", "bench -m model.gguf");
        assert!(
            matches!(outcome, Outcome::Skipped(Skipped::NotIdentifiable(_))),
            "a silent holder must not block the start: {outcome:?}"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "the probe is bounded by its timeouts, not by the holder"
        );
    }

    /// Dropping the [`Lock`] gives the port back — the guard owns the lock, it is not a receipt.
    ///
    /// The re-acquire is a retry loop because release is asynchronous by construction: the flag is
    /// set synchronously, but the listener is closed by the banner thread once it wakes. A contender
    /// that arrives inside that window sees a port that is still bound and a holder that has decided
    /// to stop answering, which is exactly the fail-open path.
    #[test]
    fn dropping_the_lock_releases_the_port() {
        let listener = TcpListener::bind(lock_addr_on(0)).expect("ephemeral bind");
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let held = acquire_on(port, "vulkan", "serve -m model.gguf");
        assert!(matches!(held, Outcome::Held(_)), "{held:?}");
        drop(held);

        let mut reacquired = None;
        for _ in 0..40 {
            match acquire_on(port, "vulkan", "serve -m model.gguf") {
                Outcome::Held(guard) => {
                    reacquired = Some(guard);
                    break;
                }
                other => {
                    std::thread::sleep(Duration::from_millis(25));
                    drop(other);
                }
            }
        }
        assert!(
            reacquired.is_some(),
            "the port must be acquirable again once the guard is dropped"
        );
    }
}
