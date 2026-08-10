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
//!     multi-GB model download that legitimately takes longer than the limit. Metadata requests are
//!     a few KB, so a total bound is right there and wrong on the download path. A stalled *body*
//!     mid-download is not fatal here anyway: the partial is kept and the next run resumes it.

use infr_core::error::{Error, Result};
use reqwest::blocking::Client;
use std::{sync::OnceLock, time::Duration};

/// TCP+TLS handshake budget. Generous enough for a slow link, short enough to fail rather than hang.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Whole-request budget for the small (few-KB) metadata calls: the model API GET and the LFS HEAD.
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

/// The client used for the HF model API (`/api/models/...`). Redirects follow the default policy;
/// only the HEAD path depends on seeing the 302 itself.
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
