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

/// The `ETag` every file is served with, and the `If-Range` value a resume must present to be
/// honoured. One constant because these tests never need two generations of the same object except
/// where they deliberately present a STALE validator, which is any other string.
const ETAG: &str = "\"v1\"";

struct FileEntry {
    body: Vec<u8>,
    /// The sha256 the `X-Linked-Etag` header advertises. Equal to the body's real digest unless a
    /// test made it lie.
    advertised_sha: String,
    present: bool,
}

#[derive(Default)]
struct State {
    files: Mutex<HashMap<String, FileEntry>>,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
    /// Per file, the `Range` start offset of every request that carried one.
    ranges: Mutex<HashMap<String, Vec<u64>>>,
    stop: AtomicBool,
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

    /// The `Range` start offsets served for `name`, in the order they arrived.
    pub(crate) fn ranges_served(&self, name: &str) -> Vec<u64> {
        self.state
            .ranges
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

    let mut range_from: Option<u64> = None;
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
            // Only `bytes=N-` is ever sent by the resume path.
            "range" => {
                range_from = value
                    .strip_prefix("bytes=")
                    .and_then(|r| r.strip_suffix('-'))
                    .and_then(|n| n.parse().ok());
            }
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
            Some(e) if e.present => Some((e.body.clone(), e.advertised_sha.clone())),
            _ => None,
        }
    };
    let Some((body, advertised_sha)) = entry else {
        let _ = write!(
            &stream,
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        );
        let _ = stream.shutdown(Shutdown::Both);
        return;
    };

    // A stale `If-Range` means the object moved on: ignore the Range and send the whole body, which
    // is what stops a resume splicing new bytes onto an old prefix.
    let honour_range = if_range.as_deref().is_none_or(|v| v == ETAG);
    let start = match range_from {
        Some(n) if honour_range && (n as usize) < body.len() => {
            state
                .ranges
                .lock()
                .unwrap()
                .entry(file.clone())
                .or_default()
                .push(n);
            Some(n as usize)
        }
        Some(n) => {
            state
                .ranges
                .lock()
                .unwrap()
                .entry(file.clone())
                .or_default()
                .push(n);
            None
        }
        None => None,
    };

    let slice = &body[start.unwrap_or(0)..];
    let mut head = String::new();
    head.push_str(match start {
        Some(_) => "HTTP/1.1 206 Partial Content\r\n",
        None => "HTTP/1.1 200 OK\r\n",
    });
    if let Some(s) = start {
        head.push_str(&format!(
            "Content-Range: bytes {s}-{}/{}\r\n",
            body.len() - 1,
            body.len()
        ));
    }
    head.push_str(&format!("Content-Length: {}\r\n", slice.len()));
    head.push_str(&format!("ETag: {ETAG}\r\n"));
    head.push_str(&format!("X-Linked-Etag: \"{advertised_sha}\"\r\n"));
    head.push_str("Accept-Ranges: bytes\r\nConnection: close\r\n\r\n");
    let mut out = &stream;
    if out.write_all(head.as_bytes()).is_err() {
        return;
    }
    if method == "HEAD" {
        let _ = out.flush();
        let _ = stream.shutdown(Shutdown::Both);
        return;
    }

    // The slot is occupied for the whole delay and released once the bytes are handed to the
    // kernel — see the module header for why the pause comes first.
    let now = state.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
    state.peak.fetch_max(now, Ordering::SeqCst);
    std::thread::sleep(BODY_DELAY);
    let _ = out.write_all(slice);
    let _ = out.flush();
    state.in_flight.fetch_sub(1, Ordering::SeqCst);
    // `shutdown` (unlike a hard close) still delivers what is already in the send buffer, then
    // sends FIN — which is the `Connection: close` the header promised.
    let _ = stream.shutdown(Shutdown::Both);
}
