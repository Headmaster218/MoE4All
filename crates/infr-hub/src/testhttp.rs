//! A local stand-in for huggingface.co, so the download path can be tested against real sockets.
//!
//! Resume, the sha256 gate and the concurrency bound are all properties of what goes over the
//! wire — a fake at the `reqwest` level would test the fake. This serves the two routes the hub
//! actually uses (`HEAD`/`GET /<repo>/resolve/main/<file>`) on `127.0.0.1`, and records what it
//! saw: how many bodies were in flight at once, and which byte offset each `Range` resumed from.
//!
//! A body is held for [`BODY_DELAY`] before a byte of it is written. Without that pause a 40 KB
//! file over loopback completes inside one scheduler slice and every request looks sequential no
//! matter how many workers are running — which is the observation the bound test rests on. Holding
//! it BEFORE the write, rather than dribbling the body out afterwards, is what keeps the count
//! honest at the other end too: the slot is released while the client is still reading, so two
//! strictly sequential requests can never appear to overlap.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a request occupies a body slot before its bytes are written.
const BODY_DELAY: Duration = Duration::from_millis(25);

/// The `ETag` a freshly registered file is served with, and the `If-Range` value a resume must
/// present to be honoured. [`TestHub::reupload`] moves a file to a new one, which is what a
/// re-published object looks like from the client's side.
const ETAG: &str = "\"v1\"";

struct FileEntry {
    body: Vec<u8>,
    /// The sha256 the `X-Linked-Etag` header advertises. Equal to the body's real digest unless a
    /// test made it lie.
    advertised_sha: String,
    /// This generation's `ETag` — the `If-Range` validator, and what a re-upload changes.
    etag: String,
    present: bool,
}

/// One GET body this hub served, as the server saw it asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Served {
    /// The `Range` the request carried, if any: `(first, last)` — `last` absent for an open
    /// `bytes=N-`.
    pub(crate) range: Option<(u64, Option<u64>)>,
    /// Whether the range was actually honoured (a `206`). `false` means the whole object went out,
    /// which is what a stale `If-Range` or a range-refusing server produces.
    pub(crate) honoured: bool,
}

#[derive(Default)]
struct State {
    files: Mutex<HashMap<String, FileEntry>>,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
    /// Per file, every GET body served, in the order they arrived.
    served: Mutex<HashMap<String, Vec<Served>>>,
    /// Per file, bodies in flight now and the most there ever were. Separate from the global peak
    /// because "this ONE file reached four connections" is a different claim from "the pull did",
    /// and the connection hand-off between files can only be seen per file.
    per_file: Mutex<HashMap<String, (usize, usize)>>,
    stop: AtomicBool,
    /// Answer every request with the whole object, whatever `Range` it carried — the server that
    /// cannot be split across.
    refuse_ranges: AtomicBool,
    /// Advertise `Accept-Ranges: bytes` anyway. Separate from actually honouring them, because a
    /// server that advertises ranges and then declines one is the case the mid-flight fallback
    /// exists for, and it is invisible if the two are the same flag.
    advertise_ranges: AtomicBool,
    /// Bodies served so far, and how many are allowed through intact before the rest are cut in
    /// half — a download dying mid-flight, without a second process to kill.
    bodies: AtomicUsize,
    cut_after: AtomicUsize,
}

pub(crate) struct TestHub {
    addr: SocketAddr,
    state: Arc<State>,
}

impl TestHub {
    /// Bind an ephemeral port and start serving. The listener thread stops when the hub is dropped.
    pub(crate) fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test hub");
        let addr = listener.local_addr().expect("test hub addr");
        let state = Arc::new(State::default());
        state.cut_after.store(usize::MAX, Ordering::Relaxed); // nothing is cut by default
        state.advertise_ranges.store(true, Ordering::Relaxed);
        let acceptor = state.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                if acceptor.stop.load(Ordering::Relaxed) {
                    break;
                }
                let Ok(stream) = stream else { break };
                let state = acceptor.clone();
                std::thread::spawn(move || serve(&state, stream));
            }
        });
        TestHub { addr, state }
    }

    pub(crate) fn origin(&self) -> String {
        format!("http://{}", self.addr)
    }

    /// Register a `-NNNNN-of-MMMMM` shard set of `count` files, `len` bytes each, and return their
    /// names in order. Each file's bytes are distinct, so a shard that ends up holding another
    /// shard's content is visible.
    pub(crate) fn add_shards(&self, base: &str, count: usize, len: usize) -> Vec<String> {
        let mut names = Vec::with_capacity(count);
        for i in 1..=count {
            let name = format!("{base}-{i:05}-of-{count:05}.gguf");
            let body: Vec<u8> = (0..len)
                .map(|j| (j.wrapping_mul(31) ^ (i * 7)) as u8)
                .collect();
            let advertised_sha = hex_sha256(&body);
            self.state.files.lock().unwrap().insert(
                name.clone(),
                FileEntry {
                    body,
                    advertised_sha,
                    etag: ETAG.to_string(),
                    present: true,
                },
            );
            names.push(name);
        }
        names
    }

    /// The bytes `name` is served with.
    pub(crate) fn body(&self, name: &str) -> Vec<u8> {
        self.state.files.lock().unwrap()[name].body.clone()
    }

    /// The sha256 of `name`'s bytes, as a blob would be addressed by.
    pub(crate) fn sha(&self, name: &str) -> String {
        hex_sha256(&self.body(name))
    }

    /// Advertise `sha` for `name` instead of its real digest — a mirror serving bytes that are not
    /// what the LFS oid says they are.
    pub(crate) fn lie_about_sha(&self, name: &str, sha: &str) {
        self.state
            .files
            .lock()
            .unwrap()
            .get_mut(name)
            .expect("unknown file")
            .advertised_sha = sha.to_string();
    }

    /// Replace `name`'s bytes with a DIFFERENT object of the same length, under a new `ETag` and a
    /// new advertised sha256 — a repo whose file was re-published between two runs of `infr pull`.
    ///
    /// Same length on purpose: a re-upload that changed the size would be caught by the size check
    /// alone, and the case worth testing is the one where every number still lines up and only the
    /// bytes differ. Resuming across it is precisely the splice that produced a corrupt blob and a
    /// day of "the quant is broken" (memory `verify-gguf-downloads`).
    pub(crate) fn reupload(&self, name: &str) {
        let mut files = self.state.files.lock().unwrap();
        let entry = files.get_mut(name).expect("unknown file");
        for (j, b) in entry.body.iter_mut().enumerate() {
            *b = (j.wrapping_mul(17) ^ 0xA5) as u8;
        }
        entry.advertised_sha = hex_sha256(&entry.body);
        entry.etag = "\"v2\"".to_string();
    }

    /// Re-upload `name` at a DIFFERENT length — the second shape of "the plan no longer describes
    /// this object", and the one that leaves a partial whose file is longer than the object it is
    /// now supposed to become.
    pub(crate) fn reupload_resized(&self, name: &str, len: usize) {
        let mut files = self.state.files.lock().unwrap();
        let entry = files.get_mut(name).expect("unknown file");
        entry.body = (0..len)
            .map(|j| (j.wrapping_mul(11) ^ 0x5A) as u8)
            .collect();
        entry.advertised_sha = hex_sha256(&entry.body);
        entry.etag = "\"v3\"".to_string();
    }

    /// Serve the whole object for every request and advertise no `Accept-Ranges`, whatever was
    /// asked for — an origin, proxy or mirror that does not do ranges.
    pub(crate) fn refuse_ranges(&self) {
        self.state.refuse_ranges.store(true, Ordering::Relaxed);
        self.state.advertise_ranges.store(false, Ordering::Relaxed);
    }

    /// Advertise `Accept-Ranges: bytes` and then serve the whole object anyway — the mirror or
    /// middlebox whose advertisement is worth nothing, and the reason a `206` is REQUIRED per chunk
    /// rather than assumed from the probe.
    pub(crate) fn lie_about_ranges(&self) {
        self.state.refuse_ranges.store(true, Ordering::Relaxed);
    }

    /// Let `n` more bodies out intact, then cut every one after that in half and hang up. Kills a
    /// download mid-flight from the server's side, which is the only way to interrupt one from
    /// inside a test.
    pub(crate) fn cut_after(&self, n: usize) {
        self.state.cut_after.store(
            self.state.bodies.load(Ordering::SeqCst) + n,
            Ordering::SeqCst,
        );
    }

    /// Stop cutting: the next run of the same download completes normally.
    pub(crate) fn heal(&self) {
        self.state.cut_after.store(usize::MAX, Ordering::SeqCst);
    }

    /// Answer `404` for `name` from now on.
    pub(crate) fn remove(&self, name: &str) {
        self.state
            .files
            .lock()
            .unwrap()
            .get_mut(name)
            .expect("unknown file")
            .present = false;
    }

    /// The most bodies this hub ever had in flight at one moment.
    pub(crate) fn peak_concurrent_bodies(&self) -> usize {
        self.state.peak.load(Ordering::Relaxed)
    }

    /// The most bodies of ONE file that were ever in flight at one moment.
    pub(crate) fn peak_for(&self, name: &str) -> usize {
        self.state
            .per_file
            .lock()
            .unwrap()
            .get(name)
            .map(|(_, peak)| *peak)
            .unwrap_or(0)
    }

    /// The `Range` start offsets served for `name`, in the order they arrived.
    pub(crate) fn ranges_served(&self, name: &str) -> Vec<u64> {
        self.served_bodies(name)
            .into_iter()
            .filter_map(|s| s.range.map(|(start, _)| start))
            .collect()
    }

    /// Every GET body served for `name`, in order.
    pub(crate) fn served_bodies(&self, name: &str) -> Vec<Served> {
        self.state
            .served
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .unwrap_or_default()
    }
}

impl Drop for TestHub {
    fn drop(&mut self) {
        self.state.stop.store(true, Ordering::Relaxed);
        // Unblock the `accept` sitting in the listener thread so it can see the flag.
        let _ = TcpStream::connect(self.addr);
    }
}

/// `bytes=A-B` → `(A, Some(B))`, `bytes=A-` → `(A, None)`. Anything else is not a range this
/// client sends, and is treated as no range at all.
fn parse_range(value: &str) -> Option<(u64, Option<u64>)> {
    let spec = value.trim().strip_prefix("bytes=")?;
    let (first, last) = spec.split_once('-')?;
    let first = first.trim().parse().ok()?;
    let last = last.trim();
    if last.is_empty() {
        return Some((first, None));
    }
    Some((first, Some(last.parse().ok()?)))
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// One request/response, then close (`Connection: close`) — keep-alive would add nothing here.
fn serve(state: &State, stream: TcpStream) {
    let mut reader = BufReader::new(stream.try_clone().expect("clone test hub stream"));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).unwrap_or(0) == 0 {
        return; // the drop-time wakeup connection
    }
    let mut parts = request_line.split_whitespace();
    let (method, path) = (
        parts.next().unwrap_or("").to_string(),
        parts.next().unwrap_or("").to_string(),
    );

    // Only `bytes=A-B` and `bytes=A-` are ever sent by the download path.
    let mut range: Option<(u64, Option<u64>)> = None;
    let mut if_range: Option<String> = None;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
            break;
        }
        let Some((name, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim().to_string();
        match name.to_ascii_lowercase().as_str() {
            "range" => range = parse_range(&value),
            "if-range" => if_range = Some(value),
            _ => {}
        }
    }

    let file = path
        .rsplit("/resolve/main/")
        .next()
        .unwrap_or("")
        .to_string();
    let entry = {
        let files = state.files.lock().unwrap();
        match files.get(&file) {
            Some(e) if e.present => {
                Some((e.body.clone(), e.advertised_sha.clone(), e.etag.clone()))
            }
            _ => None,
        }
    };
    let Some((body, advertised_sha, etag)) = entry else {
        let _ = write!(
            &stream,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.shutdown(Shutdown::Both);
        return;
    };

    // A stale `If-Range` means the object moved on: ignore the Range and send the whole body, which
    // is what stops a resume splicing new bytes onto an old prefix. A hub told to `refuse_ranges`
    // does the same to every request, however fresh the validator.
    let ranges_ok = !state.refuse_ranges.load(Ordering::Relaxed);
    let honour = ranges_ok && if_range.as_deref().is_none_or(|v| v == etag);
    let advertise = state.advertise_ranges.load(Ordering::Relaxed);
    let span = match range {
        Some((first, last)) if honour && (first as usize) < body.len() => {
            let last = last
                .unwrap_or(body.len() as u64 - 1)
                .min(body.len() as u64 - 1);
            Some((first as usize, last as usize))
        }
        _ => None,
    };
    if method == "GET" {
        state
            .served
            .lock()
            .unwrap()
            .entry(file.clone())
            .or_default()
            .push(Served {
                range,
                honoured: span.is_some(),
            });
    }

    let slice = match span {
        Some((first, last)) => &body[first..=last],
        None => &body[..],
    };
    let mut head = String::new();
    head.push_str(match span {
        Some(_) => "HTTP/1.1 206 Partial Content\r\n",
        None => "HTTP/1.1 200 OK\r\n",
    });
    if let Some((first, last)) = span {
        head.push_str(&format!(
            "Content-Range: bytes {first}-{last}/{}\r\n",
            body.len()
        ));
    }
    head.push_str(&format!("Content-Length: {}\r\n", slice.len()));
    head.push_str(&format!("ETag: {etag}\r\n"));
    head.push_str(&format!("X-Linked-Etag: \"{advertised_sha}\"\r\n"));
    if advertise {
        head.push_str("Accept-Ranges: bytes\r\n");
    }
    head.push_str("Connection: close\r\n\r\n");
    let mut out = &stream;
    if out.write_all(head.as_bytes()).is_err() {
        return;
    }
    if method == "HEAD" {
        let _ = out.flush();
        let _ = stream.shutdown(Shutdown::Both);
        return;
    }

    // Past the cut point the body is truncated and the connection dropped, which is what an
    // interrupted transfer looks like to the client: fewer bytes than `Content-Length` promised.
    {
        let mut per = state.per_file.lock().unwrap();
        let e = per.entry(file.clone()).or_insert((0, 0));
        e.0 += 1;
        e.1 = e.1.max(e.0);
    }
    let nth = state.bodies.fetch_add(1, Ordering::SeqCst);
    let slice = if nth >= state.cut_after.load(Ordering::SeqCst) {
        &slice[..slice.len() / 2]
    } else {
        slice
    };

    // The slot is occupied for the whole delay and released once the bytes are handed to the
    // kernel — see the module header for why the pause comes first.
    let now = state.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
    state.peak.fetch_max(now, Ordering::SeqCst);
    std::thread::sleep(BODY_DELAY);
    let _ = out.write_all(slice);
    let _ = out.flush();
    state.in_flight.fetch_sub(1, Ordering::SeqCst);
    state.per_file.lock().unwrap().entry(file).or_default().0 -= 1;
    // `shutdown` (unlike a hard close) still delivers what is already in the send buffer, then
    // sends FIN — which is the `Connection: close` the header promised.
    let _ = stream.shutdown(Shutdown::Both);
}
