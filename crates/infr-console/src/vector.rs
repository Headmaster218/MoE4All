//! Refusal-direction control vector download.
//!
//! Reuses the wizard's pinned spec (repo + file + sha256) verbatim, so the console fetches the SAME
//! canonical direction `scripts/infr-wizard.ps1` does. The hub also carries a same-named look-alike
//! with different numbers; the sha256 is what tells them apart, and the engine's `resolve()` further
//! validates the qwen4exp architecture, width and layer range. The only two URLs this module ever
//! contacts are the real, task-pinned HF origin and its mirror — no proxy substitution.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::state::Console;

/// Pinned exactly as the wizard's `Get-UncensorVectorSpec`. Lowercase for comparison; the wizard
/// stores it uppercase but `Get-FileHash` is case-insensitive, and so is this check.
const REPO: &str = "cjxzdzh/Qwen3.8-Flash-Next-refusal-direction-qwen4exp-cvec";
const FILE: &str = "cvec-qwen4exp.gguf";
const SHA256: &str = "ef0724c5b79297e481017be85769832bc26eb220c984ecba2b0e1b0c8426312b";
/// huggingface.co first, then the mirror — the wizard's order. Some machines can reach only one.
const ORIGINS: &[&str] = &["https://huggingface.co", "https://hf-mirror.com"];

/// Default destination `<repo root>/models/control-vectors/<file>`, mirroring the wizard. The repo
/// root is inferred from the engine binary (`…/target/release/infr.exe` → three parents up); when
/// the binary is not inside a build tree, fall back to the console state directory.
pub fn default_path(console: &Console) -> PathBuf {
    let root = console
        .binary
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent());
    match root {
        Some(root) => root.join("models").join("control-vectors").join(FILE),
        None => console
            .state_path
            .parent()
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join("control-vectors")
            .join(FILE),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The vector is ~450 KiB, so hashing the whole file in one read is cheaper than streaming and
/// keeps this module free of a chunked-reader dependency.
async fn sha256_file(path: &std::path::Path) -> Option<String> {
    let bytes = tokio::fs::read(path).await.ok()?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Some(hex(&hasher.finalize()))
}

/// Fetch the canonical vector to `dest` (or the default path). A file already matching the pinned
/// sha256 is returned untouched. Otherwise each origin is tried in order; a response whose hash does
/// not match is discarded, never kept as if it were the direction. The bytes are buffered and only
/// written on a match, so an interrupted or wrong download leaves no partial file behind.
pub async fn download(console: &Arc<Console>, dest: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    let path = dest.unwrap_or_else(|| default_path(console));
    if sha256_file(&path).await.as_deref() == Some(SHA256) {
        console.log(format!("去审查向量已是最新：{}", path.display()));
        return Ok(path);
    }
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    // A dedicated client so a black-hole origin costs ~5 s (connect probe) instead of the whole
    // request timeout. `ClientBuilder::connect_timeout` is core; the per-request variant is not in
    // this reqwest build. The shared proxy client stays untouched.
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .build()?;
    let mut last_err = String::from("两个来源都没能取到去审查向量");
    for origin in ORIGINS {
        let url = format!("{origin}/{REPO}/resolve/main/{FILE}");
        // The origin can be a black hole on some machines; give it a short attempt and the mirror a
        // longer one — the wizard's 1×8 s / 3×30 s ladder, collapsed to one try per endpoint.
        let timeout = if *origin == "https://huggingface.co" { Duration::from_secs(20) } else { Duration::from_secs(120) };
        console.log(format!("获取去审查向量：{url}"));
        let resp = match client.get(&url).timeout(timeout).send().await {
            Ok(resp) => resp,
            Err(e) => {
                last_err = format!("{origin} 连接失败：{e}");
                console.log(last_err.clone());
                continue;
            }
        };
        let resp = match resp.error_for_status() {
            Ok(resp) => resp,
            Err(e) => {
                last_err = format!("{origin} 返回错误：{e}");
                console.log(last_err.clone());
                continue;
            }
        };
        let bytes = match resp.bytes().await {
            Ok(bytes) => bytes,
            Err(e) => {
                last_err = format!("{origin} 下载中断：{e}");
                console.log(last_err.clone());
                continue;
            }
        };
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let got = hex(&hasher.finalize());
        if got != SHA256 {
            last_err = format!("{origin} 校验和不匹配（期望 {SHA256}，实得 {got}），已丢弃");
            console.log(last_err.clone());
            continue;
        }
        tokio::fs::write(&path, &bytes).await?;
        console.log(format!("去审查向量已就绪：{}", path.display()));
        return Ok(path);
    }
    anyhow::bail!(last_err)
}

/// Read-only status for the UI: where the vector would live, whether it is present, and whether its
/// hash matches the pinned canonical direction. No consent needed — nothing is fetched or written.
pub async fn status(console: &Console, path: Option<PathBuf>) -> serde_json::Value {
    let path = path.unwrap_or_else(|| default_path(console));
    let exists = path.is_file();
    let sha = sha256_file(&path).await;
    let valid = sha.as_deref() == Some(SHA256);
    serde_json::json!({
        "path": path.display().to_string(),
        "exists": exists,
        "valid": valid,
        "sha256": sha,
        "expected_sha256": SHA256,
        "default_path": default_path(console).display().to_string(),
        "repo": REPO,
        "file": FILE,
        "origins": ORIGINS,
    })
}