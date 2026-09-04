//! The process-wide `reqwest` clients every hub request goes through, and the HF bearer token.
//!
//! THREE clients, each built once (a `Client` owns a connection pool + TLS config; building one
//! per request means a fresh TLS handshake per shard — 40 of them for a 40-shard model).
//!
//! The timeout split is deliberate and is the part a later reader is most likely to "fix" wrongly:
//!
//!   * `connect_timeout` on ALL of them — a server that never completes the TCP/TLS handshake must
//!     not hang `infr pull` (and `infr run`'s auto-pull) forever with no recovery but Ctrl-C, which
//!     is exactly what reqwest's default of NO timeout does.
//!   * a total `timeout` ONLY on the metadata/HEAD clients. `Client::timeout` bounds the WHOLE
//!     request including reading the body, so putting it on the download client would abort every
//!     multi-GiB model download that legitimately takes longer than the limit. Metadata requests are
//!     a few KiB, so a total bound is right there and wrong on the download path. A stalled *body*
//!     mid-download is not fatal here anyway: the partial is kept and the next run resumes it.

use infr_core::error::{Error, Result};
use reqwest::blocking::Client;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{sync::OnceLock, time::Duration};

/// TCP+TLS handshake budget. Generous enough for a slow link, short enough to fail rather than hang.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Whole-request budget for the small (few-KiB) metadata calls: the model API GET and the LFS HEAD.
const METADATA_TIMEOUT: Duration = Duration::from_secs(30);

/// The client used for BODY transfers (model blobs, companions). No total timeout — see above.
///
/// ONE client shared by every concurrent shard download on purpose: a `reqwest::blocking::Client`
/// is `Sync` and its connection pool is what lets N workers hold N sockets to the same host
/// without N TLS handshakes' worth of setup each time one finishes.
pub(crate) fn download_client() -> Result<&'static Client> {
    static CLIENT: OnceLock<std::result::Result<Client, String>> = OnceLock::new();
    shared(&CLIENT, "download", |b| b)
}

/// The client used for the HF model API (`/api/models/...`) and for [`crate::ranged::probe`]'s
/// `HEAD`. Both are small, redirect-following requests that must not hang, which is what separates
/// them from the download client; only [`head_client`]'s HEAD depends on seeing the 302 itself.
pub(crate) fn api_client() -> Result<&'static Client> {
    static CLIENT: OnceLock<std::result::Result<Client, String>> = OnceLock::new();
    shared(&CLIENT, "API", |b| b.timeout(METADATA_TIMEOUT))
}

/// The client used by `head_lfs_sha`. Redirects stay DISABLED — load-bearing: the `X-Linked-Etag`
/// sha256 is on huggingface.co's 302, not on the CDN's final 200, so following the redirect loses it.
pub(crate) fn head_client() -> Result<&'static Client> {
    static CLIENT: OnceLock<std::result::Result<Client, String>> = OnceLock::new();
    shared(&CLIENT, "HEAD", |b| {
        b.timeout(METADATA_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
    })
}

/// Build-once accessor for one of the shared clients. The build result (including a failure) is
/// cached, so a broken TLS backend reports the same error every time instead of being retried per
/// request; the error is returned, never panicked, so a pull can still fail gracefully.
fn shared(
    cell: &'static OnceLock<std::result::Result<Client, String>>,
    what: &str,
    extra: impl FnOnce(reqwest::blocking::ClientBuilder) -> reqwest::blocking::ClientBuilder,
) -> Result<&'static Client> {
    cell.get_or_init(|| {
        extra(
            Client::builder()
                .user_agent("infr-hub/0.1")
                .connect_timeout(CONNECT_TIMEOUT),
        )
        .build()
        .map_err(|e| e.to_string())
    })
    .as_ref()
    .map_err(|e| Error::Other(format!("building HTTP {what} client: {e}")))
}

/// The HuggingFace access token, for gated/private repos. `HF_TOKEN` is HuggingFace's own spelling,
/// shared with `huggingface_hub` and llama.cpp — not an `INFR_*` knob, so it is not configuration.
pub(crate) fn token() -> Option<String> {
    std::env::var("HF_TOKEN").ok()
}

/// The pull's whole allowance of simultaneous body transfers, handed out one permit per connection.
///
/// There are now TWO things that want to open connections — several FILES of a model at once
/// ([`crate::pull::fetch_all`]) and several RANGES of one file ([`crate::ranged`]) — and they must
/// not multiply. One budget shared by both is what makes the total bound `hub.pull_jobs` rather
/// than `pull_jobs × pull_jobs`: a 236-shard repo spends every permit on files and splits nothing,
/// a single-file repo spends them all on ranges, and the mixed case in between lands wherever the
/// files happen to leave room.
///
/// Range workers only ever [`try_acquire`](ConnBudget::try_acquire) — they never wait for a permit
/// — so the file workers, which each hold one for their whole life, can never be starved by them
/// and nothing here can deadlock.
pub(crate) struct ConnBudget {
    free: AtomicUsize,
}

/// One connection's worth of the budget, returned when dropped. Held by the thread that owns the
/// connection, so a panicking or early-returning worker cannot leak its slot.
pub(crate) struct Permit<'a> {
    budget: &'a ConnBudget,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.budget.free.fetch_add(1, Ordering::AcqRel);
    }
}

impl ConnBudget {
    pub(crate) fn new(permits: usize) -> Self {
        ConnBudget {
            free: AtomicUsize::new(permits),
        }
    }

    /// Permits available right now. Advisory only (another thread may take one a nanosecond later),
    /// so it is used to DECIDE whether splitting a file is worth a probe request, never to reserve.
    pub(crate) fn available(&self) -> usize {
        self.free.load(Ordering::Acquire)
    }

    /// Take one permit if one is free. Never blocks.
    pub(crate) fn try_acquire(&self) -> Option<Permit<'_>> {
        let mut free = self.free.load(Ordering::Acquire);
        loop {
            if free == 0 {
                return None;
            }
            match self.free.compare_exchange_weak(
                free,
                free - 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Some(Permit { budget: self }),
                Err(now) => free = now,
            }
        }
    }

    /// Take up to `want` permits, returning however many were free. Never blocks, and returns an
    /// empty vector rather than waiting when the budget is exhausted.
    pub(crate) fn acquire_up_to(&self, want: usize) -> Vec<Permit<'_>> {
        let mut got = Vec::new();
        while got.len() < want {
            match self.try_acquire() {
                Some(p) => got.push(p),
                None => break,
            }
        }
        got
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The budget hands out exactly what it has and not one more, and every permit comes back when
    /// it drops. A budget that over-issued would show up in `infr pull` as more sockets than
    /// `hub.pull_jobs` — the bound the whole fan-out rests on.
    #[test]
    fn a_budget_issues_exactly_its_permits() {
        let b = ConnBudget::new(3);
        let mut held = b.acquire_up_to(10);
        assert_eq!(held.len(), 3, "asked for 10 of 3");
        assert_eq!(b.available(), 0);
        assert!(b.try_acquire().is_none(), "over-issued");
        held.pop();
        assert_eq!(b.available(), 1);
        assert!(b.try_acquire().is_some());
        drop(held);
        assert_eq!(b.available(), 3, "permits must return when dropped");
    }

    /// `hub.pull_jobs = 0`/`1` leave nothing over for range workers, which is what keeps those
    /// settings STRICTLY one connection (see `HubCfg::pull_jobs`).
    #[test]
    fn an_empty_budget_grants_nothing() {
        let b = ConnBudget::new(0);
        assert_eq!(b.available(), 0);
        assert!(b.try_acquire().is_none());
        assert!(b.acquire_up_to(4).is_empty());
    }
}
