//! HTTPS termination for the console.
//!
//! The console keeps a single port: when TLS is configured that port speaks HTTPS directly, so the
//! admin password and session cookie never cross the LAN in clear text. Certificates are read as
//! PEM, which is what `openssl`, `mkcert`, Caddy and every reverse proxy already hand out.

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context};
use axum::body::Body;
use hyper_util::rt::TokioIo;
use rustls::pki_types::{
    CertificateDer, PrivateKeyDer, PrivatePkcs1KeyDer, PrivatePkcs8KeyDer, PrivateSec1KeyDer,
};
use rustls::ServerConfig;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;
use tower::ServiceExt;

/// Parse the `CERTIFICATE` blocks of a PEM bundle, in file order (leaf first, as rustls expects).
pub fn load_certificates(path: &Path) -> anyhow::Result<Vec<CertificateDer<'static>>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("读取证书 {}", path.display()))?;
    let blocks = pem_blocks(&text, "CERTIFICATE");
    if blocks.is_empty() {
        bail!("{} 中没有 CERTIFICATE 块", path.display());
    }
    Ok(blocks
        .into_iter()
        .map(|der| CertificateDer::from(der))
        .collect())
}

/// Parse a PEM private key. PKCS#8, PKCS#1 (RSA) and SEC1 (EC) are all accepted because the file
/// usually comes straight from `openssl req -newkey`.
pub fn load_private_key(path: &Path) -> anyhow::Result<PrivateKeyDer<'static>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("读取私钥 {}", path.display()))?;
    if let Some(der) = pem_blocks(&text, "PRIVATE KEY").into_iter().next() {
        return Ok(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(der)));
    }
    if let Some(der) = pem_blocks(&text, "RSA PRIVATE KEY").into_iter().next() {
        return Ok(PrivateKeyDer::Pkcs1(PrivatePkcs1KeyDer::from(der)));
    }
    if let Some(der) = pem_blocks(&text, "EC PRIVATE KEY").into_iter().next() {
        return Ok(PrivateKeyDer::Sec1(PrivateSec1KeyDer::from(der)));
    }
    bail!(
        "{} 中没有 PRIVATE KEY / RSA PRIVATE KEY / EC PRIVATE KEY 块",
        path.display()
    )
}

pub fn server_config(cert_path: &Path, key_path: &Path) -> anyhow::Result<Arc<ServerConfig>> {
    // With only the `ring` feature compiled in this is redundant, but an explicit install keeps
    // the first handshake from panicking if another provider ever gets linked in.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let certs = load_certificates(cert_path)?;
    let key = load_private_key(key_path)?;
    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .context("证书与私钥不匹配，或证书链无效")?;
    Ok(Arc::new(config))
}

/// Accept loop that terminates TLS and hands the decrypted stream to the same axum router used for
/// plain HTTP. HTTP/1.1 only: browsers speak it over TLS without ALPN negotiation.
pub async fn serve(
    app: axum::Router,
    listener: TcpListener,
    config: Arc<ServerConfig>,
) -> anyhow::Result<()> {
    let acceptor = TlsAcceptor::from(config);
    loop {
        let (stream, _peer) = listener.accept().await?;
        let acceptor = acceptor.clone();
        let app = app.clone();
        tokio::spawn(async move {
            let Ok(tls) = acceptor.accept(stream).await else {
                return;
            };
            let io = TokioIo::new(tls);
            let service = hyper::service::service_fn(
                move |request: hyper::Request<hyper::body::Incoming>| {
                    let app = app.clone();
                    async move {
                        let response = app
                            .oneshot(request.map(Body::new))
                            .await
                            .expect("axum router is infallible");
                        Ok::<_, std::convert::Infallible>(response)
                    }
                },
            );
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(io, service)
                .await;
        });
    }
}

fn pem_blocks(text: &str, label: &str) -> Vec<Vec<u8>> {
    let begin = format!("-----BEGIN {label}-----");
    let end = format!("-----END {label}-----");
    let mut blocks = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(&begin) {
        let after = &rest[start + begin.len()..];
        let Some(stop) = after.find(&end) else { break };
        if let Some(bytes) = base64_decode(&after[..stop]) {
            blocks.push(bytes);
        }
        rest = &after[stop + end.len()..];
    }
    blocks
}

fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    let mut accumulator = 0u32;
    let mut bits = 0u32;
    for byte in input.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b'\r' | b'\n' | b' ' | b'\t' => continue,
            _ => return None,
        } as u32;
        accumulator = (accumulator << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((accumulator >> bits) as u8);
            accumulator &= (1 << bits) - 1;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_a_known_pem_block() {
        // "hello" in base64.
        let pem = "-----BEGIN CERTIFICATE-----\naGVsbG8=\n-----END CERTIFICATE-----\n";
        assert_eq!(pem_blocks(pem, "CERTIFICATE"), vec![b"hello".to_vec()]);
    }

    #[test]
    fn ignores_unrelated_blocks() {
        let pem = "-----BEGIN PRIVATE KEY-----\naGVsbG8=\n-----END PRIVATE KEY-----\n";
        assert!(pem_blocks(pem, "CERTIFICATE").is_empty());
    }

    #[test]
    fn rejects_an_empty_bundle() {
        let dir = std::env::temp_dir().join(format!("infr-tls-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.pem");
        std::fs::write(&path, "not a certificate").unwrap();
        assert!(load_certificates(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
