//! Backend-plumbing tests: device init + buffer memory roundtrip.
//!
//! macOS-only, `#[ignore]`d (needs a real Metal device) — run with
//! `cargo test -p infr-metal -- --include-ignored`.
#![cfg(target_os = "macos")]

use infr_core::backend::{Backend, BufferUsage};
use infr_metal::MetalBackend;

#[test]
#[ignore = "requires a Metal GPU"]
fn buffer_upload_download_roundtrip() {
    let be = MetalBackend::new().expect("open metal backend");
    let data: Vec<u8> = (0..1024u32).map(|i| (i.wrapping_mul(7)) as u8).collect();
    let buf = be.alloc(data.len(), BufferUsage::Weights).expect("alloc");
    be.upload(buf.as_ref(), &data).expect("upload");
    let mut back = vec![0u8; data.len()];
    be.download(buf.as_ref(), &mut back).expect("download");
    assert_eq!(data, back, "downloaded bytes must equal uploaded bytes");
}

#[test]
#[ignore = "requires a Metal GPU"]
fn backend_reports_metal_name() {
    let be = MetalBackend::new().expect("open metal backend");
    assert_eq!(be.name(), "metal");
    assert!(!be.capabilities().name.is_empty(), "device name populated");
}
