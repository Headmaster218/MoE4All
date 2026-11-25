//! Small dependency-free crypto helpers for the console.
//!
//! The console already ships `sha2` and `getrandom`. Using those instead of pulling a new TLS or
//! password-hashing crate keeps the offline Windows release build stable while still giving us:
//! - PBKDF2-HMAC-SHA256 password hashing with a per-record salt.
//! - An encrypt-then-MAC file format built from SHA-256/HMAC-SHA256 in counter mode.
//!
//! This is at-rest protection for local state, not a replacement for TLS. The HTTP listener stays
//! plain HTTP; do not expose it to an untrusted network unless TLS terminates in front of it.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context};
use serde::{de::DeserializeOwned, Serialize};
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 5] = b"INFR1";
const NONCE_LEN: usize = 16;
const TAG_LEN: usize = 32;
const PASSWORD_ITERS: u32 = 60_000;

fn digest(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    for part in parts {
        h.update(part);
    }
    let out = h.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&out);
    bytes
}

pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut key_block = [0u8; 64];
    if key.len() > 64 {
        key_block[..32].copy_from_slice(&digest(&[key]));
    } else {
        key_block[..key.len()].copy_from_slice(key);
    }
    let mut inner = [0u8; 64];
    let mut outer = [0u8; 64];
    for i in 0..64 {
        inner[i] = key_block[i] ^ 0x36;
        outer[i] = key_block[i] ^ 0x5c;
    }
    let inner_hash = digest(&[&inner, msg]);
    digest(&[&outer, &inner_hash])
}

fn derive_key(master: &[u8], label: &str) -> [u8; 32] {
    digest(&[b"infr-console/", label.as_bytes(), master])
}

pub fn constant_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

pub fn random_bytes<const N: usize>() -> [u8; N] {
    let mut bytes = [0u8; N];
    getrandom::fill(&mut bytes).expect("OS CSPRNG unavailable");
    bytes
}

pub fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

pub fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for pair in bytes.chunks(2) {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
    }
    Some(out)
}

/// PBKDF2-HMAC-SHA256 with one 32-byte output block.
pub fn hash_password(password: &str, salt: &[u8]) -> [u8; 32] {
    let mut initial = Vec::with_capacity(salt.len() + 4);
    initial.extend_from_slice(salt);
    initial.extend_from_slice(&1u32.to_be_bytes());
    let mut u = hmac_sha256(password.as_bytes(), &initial);
    let mut out = u;
    for _ in 1..PASSWORD_ITERS {
        u = hmac_sha256(password.as_bytes(), &u);
        for (dst, src) in out.iter_mut().zip(u.iter()) {
            *dst ^= src;
        }
    }
    out
}

pub fn verify_password(password: &str, salt_hex: &str, hash_hex: &str) -> bool {
    let (Some(salt), Some(expected)) = (hex_decode(salt_hex), hex_decode(hash_hex)) else {
        return false;
    };
    constant_eq(&hash_password(password, &salt), &expected)
}

fn xor_stream(key: &[u8; 32], nonce: &[u8; NONCE_LEN], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for (counter, chunk) in data.chunks(32).enumerate() {
        let ctr = (counter as u64).to_be_bytes();
        let block = digest(&[key, nonce, &ctr]);
        for (i, b) in chunk.iter().enumerate() {
            out.push(b ^ block[i]);
        }
    }
    out
}

fn encrypt_json<T: Serialize>(master: &[u8], value: &T) -> anyhow::Result<Vec<u8>> {
    let plain = serde_json::to_vec(value)?;
    let enc_key = derive_key(master, "state-encryption-v1");
    let mac_key = derive_key(master, "state-mac-v1");
    let nonce = random_bytes::<NONCE_LEN>();
    let cipher = xor_stream(&enc_key, &nonce, &plain);
    let mut body = Vec::with_capacity(MAGIC.len() + NONCE_LEN + TAG_LEN + cipher.len());
    body.extend_from_slice(MAGIC);
    body.extend_from_slice(&nonce);
    body.extend_from_slice(&cipher);
    let tag = hmac_sha256(&mac_key, &body);
    body.splice(MAGIC.len() + NONCE_LEN..MAGIC.len() + NONCE_LEN, tag);
    Ok(body)
}

fn decrypt_json<T: DeserializeOwned>(master: &[u8], bytes: &[u8]) -> anyhow::Result<T> {
    let header = MAGIC.len() + NONCE_LEN + TAG_LEN;
    if bytes.len() < header || !bytes.starts_with(MAGIC) {
        bail!("not an INFR encrypted state file");
    }
    let mac_key = derive_key(master, "state-mac-v1");
    let expected_tag = &bytes[MAGIC.len() + NONCE_LEN..header];
    let mut body = Vec::with_capacity(bytes.len() - TAG_LEN);
    body.extend_from_slice(MAGIC);
    body.extend_from_slice(&bytes[MAGIC.len()..MAGIC.len() + NONCE_LEN]);
    body.extend_from_slice(&bytes[header..]);
    if !constant_eq(expected_tag, &hmac_sha256(&mac_key, &body)) {
        bail!("console state failed authentication; refusing to load tampered data");
    }
    let mut nonce = [0u8; NONCE_LEN];
    nonce.copy_from_slice(&bytes[MAGIC.len()..MAGIC.len() + NONCE_LEN]);
    let enc_key = derive_key(master, "state-encryption-v1");
    let plain = xor_stream(&enc_key, &nonce, &bytes[header..]);
    serde_json::from_slice(&plain).context("decrypted console state is not valid JSON")
}

/// Encrypted JSON files beside `keys.json`. Existing plaintext files are accepted on read and get
/// rewritten encrypted the next time the console saves that file.
#[derive(Clone)]
pub struct SecureStore {
    key: [u8; 32],
    dir: PathBuf,
}

impl SecureStore {
    pub fn new(state_path: &Path) -> anyhow::Result<Self> {
        let dir = state_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        fs::create_dir_all(&dir)?;
        let key_path = dir.join("state.key");
        let key = match fs::read(&key_path) {
            Ok(bytes) if bytes.len() == 32 => {
                let mut key = [0u8; 32];
                key.copy_from_slice(&bytes);
                key
            }
            Ok(_) => return Err(anyhow!("{} has an invalid length", key_path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let key = random_bytes::<32>();
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&key_path)
                    .with_context(|| format!("creating {}", key_path.display()))?;
                use std::io::Write;
                file.write_all(&key)?;
                key
            }
            Err(e) => {
                return Err(anyhow::Error::new(e).context(format!("reading {}", key_path.display())))
            }
        };
        Ok(Self { key, dir })
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    pub fn load<T: DeserializeOwned>(&self, name: &str) -> Option<T> {
        let bytes = fs::read(self.path(name)).ok()?;
        if bytes.starts_with(MAGIC) {
            decrypt_json(&self.key, &bytes).ok()
        } else {
            serde_json::from_slice(&bytes).ok()
        }
    }

    pub fn save<T: Serialize>(&self, name: &str, value: &T) -> anyhow::Result<()> {
        let bytes = encrypt_json(&self.key, value)?;
        fs::write(self.path(name), bytes)
            .with_context(|| format!("writing {}", self.path(name).display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypted_state_round_trips_and_rejects_tampering() {
        let value = serde_json::json!({ "token": "sk-console-secret", "n": 7 });
        let bytes = encrypt_json(b"01234567890123456789012345678901", &value).unwrap();
        let restored: serde_json::Value =
            decrypt_json(b"01234567890123456789012345678901", &bytes).unwrap();
        assert_eq!(restored, value);
        let mut tampered = bytes.clone();
        *tampered.last_mut().unwrap() ^= 1;
        assert!(
            decrypt_json::<serde_json::Value>(b"01234567890123456789012345678901", &tampered)
                .is_err()
        );
    }

    #[test]
    fn password_hash_is_salted_and_verifiable() {
        let salt = random_bytes::<16>();
        let hash = hash_password("correct horse", &salt);
        assert!(verify_password(
            "correct horse",
            &hex_encode(&salt),
            &hex_encode(&hash)
        ));
        assert!(!verify_password(
            "wrong horse",
            &hex_encode(&salt),
            &hex_encode(&hash)
        ));
        assert_ne!(hash_password("correct horse", &random_bytes::<16>()), hash);
    }
}
