//! Admin/user login state for the console web surface.
//!
//! Password records and user accounts are persisted through [`crate::crypto::SecureStore`], while
//! session tokens deliberately live only in memory: restarting the console logs everyone out.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{bail, Context};
use axum::http::{header, HeaderMap};
use serde::{Deserialize, Serialize};

use crate::crypto::{hash_password, hex_encode, random_bytes, verify_password, SecureStore};

pub const ADMIN_COOKIE: &str = "infr_console_admin";
pub const USER_COOKIE: &str = "infr_console_user";
const SESSION_TTL: Duration = Duration::from_secs(12 * 60 * 60);

#[derive(Clone, Serialize, Deserialize)]
pub struct PasswordRecord {
    pub salt: String,
    pub hash: String,
}

impl PasswordRecord {
    pub fn new(password: &str) -> Self {
        let salt = random_bytes::<16>();
        Self {
            salt: hex_encode(&salt),
            hash: hex_encode(&hash_password(password, &salt)),
        }
    }

    pub fn verify(&self, password: &str) -> bool {
        verify_password(password, &self.salt, &self.hash)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct UserRecord {
    pub username: String,
    pub password: PasswordRecord,
}

#[derive(Serialize)]
pub struct UserView {
    pub username: String,
}

impl UserRecord {
    pub fn view(&self) -> UserView {
        UserView {
            username: self.username.clone(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Default)]
pub struct PersistentAuth {
    pub admin: Option<PasswordRecord>,
    #[serde(default)]
    pub users: Vec<UserRecord>,
}

#[derive(Clone, PartialEq, Eq)]
pub enum SessionKind {
    Admin,
    User(String),
}

struct Session {
    kind: SessionKind,
    expires: Instant,
}

pub struct AuthState {
    admin: Option<PasswordRecord>,
    users: Vec<UserRecord>,
    sessions: HashMap<String, Session>,
}

impl AuthState {
    pub fn from_persistent(persistent: PersistentAuth) -> Self {
        Self {
            admin: persistent.admin,
            users: persistent.users,
            sessions: HashMap::new(),
        }
    }

    pub fn persistent(&self) -> PersistentAuth {
        PersistentAuth {
            admin: self.admin.clone(),
            users: self.users.clone(),
        }
    }

    pub fn users(&self) -> &[UserRecord] {
        &self.users
    }

    pub fn verify_admin(&self, password: &str) -> bool {
        self.admin
            .as_ref()
            .map(|record| record.verify(password))
            .unwrap_or(false)
    }

    pub fn set_admin(&mut self, password: &str, store: &SecureStore) -> anyhow::Result<()> {
        self.admin = Some(PasswordRecord::new(password));
        self.sessions
            .retain(|_, session| session.kind != SessionKind::Admin);
        self.persist(store)
    }

    pub fn add_user(
        &mut self,
        username: &str,
        password: &str,
        store: &SecureStore,
    ) -> anyhow::Result<()> {
        let username = username.trim();
        if username.is_empty() || username.len() > 32 {
            bail!("用户名长度必须在 1 到 32 个字符之间");
        }
        if !username
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        {
            bail!("用户名只能包含字母、数字、下划线和短横线");
        }
        if password.len() < 8 {
            bail!("用户密码至少需要 8 个字符");
        }
        if self.users.iter().any(|u| u.username == username) {
            bail!("用户已存在：{username}");
        }
        self.users.push(UserRecord {
            username: username.to_string(),
            password: PasswordRecord::new(password),
        });
        self.persist(store)
    }

    pub fn remove_user(&mut self, username: &str, store: &SecureStore) -> anyhow::Result<()> {
        let before = self.users.len();
        self.users.retain(|u| u.username != username);
        if self.users.len() == before {
            bail!("用户不存在：{username}");
        }
        self.sessions
            .retain(|_, session| session.kind != SessionKind::User(username.to_string()));
        self.persist(store)
    }

    pub fn issue_admin(&mut self) -> String {
        self.issue(SessionKind::Admin)
    }

    pub fn issue_user(&mut self, username: &str) -> Option<String> {
        if self.users.iter().any(|u| u.username == username) {
            Some(self.issue(SessionKind::User(username.to_string())))
        } else {
            None
        }
    }

    fn issue(&mut self, kind: SessionKind) -> String {
        self.prune();
        let token = hex_encode(&random_bytes::<32>());
        self.sessions.insert(
            token.clone(),
            Session {
                kind,
                expires: Instant::now() + SESSION_TTL,
            },
        );
        token
    }

    pub fn validate_admin(&mut self, token: Option<&str>) -> bool {
        self.prune();
        token
            .and_then(|t| self.sessions.get(t))
            .map(|s| s.kind == SessionKind::Admin)
            .unwrap_or(false)
    }

    pub fn validate_user(&mut self, token: Option<&str>) -> Option<String> {
        self.prune();
        token
            .and_then(|t| self.sessions.get(t))
            .and_then(|s| match &s.kind {
                SessionKind::User(name) => Some(name.clone()),
                SessionKind::Admin => None,
            })
    }

    pub fn authenticate_user(&self, username: &str, password: &str) -> bool {
        self.users
            .iter()
            .any(|u| u.username == username && u.password.verify(password))
    }

    pub fn revoke_admin(&mut self, token: Option<&str>) {
        if let Some(token) = token {
            self.sessions.remove(token);
        }
    }

    pub fn revoke_user(&mut self, token: Option<&str>) {
        if let Some(token) = token {
            self.sessions.remove(token);
        }
    }

    fn prune(&mut self) {
        let now = Instant::now();
        self.sessions.retain(|_, s| s.expires > now);
    }

    fn persist(&self, store: &SecureStore) -> anyhow::Result<()> {
        store
            .save("auth.json", &self.persistent())
            .context("保存登录状态")
    }
}

/// Initial admin password. A random password is printed once on first run; `--admin-password`
/// can be used for an unattended bootstrap.
pub fn bootstrap_password(explicit: Option<String>) -> String {
    explicit.filter(|p| !p.is_empty()).unwrap_or_else(|| {
        const ALPHABET: &[u8] = b"23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        let bytes = random_bytes::<16>();
        bytes
            .iter()
            .map(|b| ALPHABET[(*b as usize) % ALPHABET.len()] as char)
            .collect()
    })
}

pub fn cookie_value(headers: &HeaderMap, name: &str) -> Option<String> {
    for value in headers.get_all(header::COOKIE) {
        let Ok(text) = value.to_str() else { continue };
        for part in text.split(';') {
            let part = part.trim();
            let Some((key, value)) = part.split_once('=') else {
                continue;
            };
            if key == name {
                return Some(value.trim().to_string());
            }
        }
    }
    None
}

pub fn cookie_header(name: &str, token: &str, max_age: u64) -> String {
    // `Secure` is intentionally absent: the console listens on plain HTTP by default. `HttpOnly`
    // and `SameSite=Strict` still stop scripts and cross-site requests from using the session.
    format!("{name}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={max_age}")
}
