//! The dashboard's own HTTP surface. Separate from the proxied `/v1/*` path: these routes are for
//! the operator's browser and are never forwarded upstream.
//!
//! The web surface now has two independent identities: an admin session guarding `/` and `/ui/*`,
//! and a user-account session for the `/user` entry page. Client API keys still authenticate
//! `/v1/*` inference callers; they do not grant dashboard access.

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{delete, get, post};
use axum::{extract::Request, Json};
use futures_util::Stream;
use serde::Deserialize;

use crate::auth::{self, ADMIN_COOKIE, USER_COOKIE};
use crate::launch;
use crate::state::{new_token, ApiKey, Console, KeyView, Settings};

pub struct UiError(anyhow::Error);

impl IntoResponse for UiError {
    fn into_response(self) -> Response {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": self.0.to_string() })),
        )
            .into_response()
    }
}

impl From<anyhow::Error> for UiError {
    fn from(e: anyhow::Error) -> Self {
        Self(e)
    }
}

type UiResult<T> = Result<T, UiError>;

/// The whole console surface: public login/user entry, admin-only dashboard/API, and the proxy as
/// fallback. The proxy routes are intentionally NOT behind the admin session.
pub fn router(console: Arc<Console>) -> axum::Router {
    let public = axum::Router::new()
        .route("/login", get(login_page))
        .route("/ui/login", post(admin_login))
        .route("/ui/logout", post(admin_logout))
        .route("/user", get(user_page))
        .route("/ui/user/login", post(user_login))
        .route("/ui/user/logout", post(user_logout))
        .route("/ui/user/me", get(user_me));

    let admin = axum::Router::new()
        .route("/", get(index))
        .route("/app.css", get(css))
        .route("/app.js", get(js))
        .route("/ui/state", get(state))
        .route("/ui/events", get(events))
        .route("/ui/rounds", delete(clear_rounds))
        .route("/ui/rounds/:id", get(round_detail))
        .route("/ui/keys", post(add_key))
        .route("/ui/keys/:id", delete(delete_key))
        .route("/ui/uncensor", post(set_uncensor))
        .route("/ui/users", get(list_users).post(add_user))
        .route("/ui/users/:username", delete(delete_user))
        .route("/ui/admin/password", post(change_admin_password))
        .route("/ui/schema", get(schema))
        .route("/ui/settings", get(get_settings).post(save_settings))
        .route("/ui/vector/download", post(download_vector))
        .route("/ui/vector/status", get(vector_status))
        .route("/ui/launch", post(start_engine))
        .route("/ui/stop", post(stop_engine))
        .route("/ui/config/export", post(export_config))
        .route("/ui/config/import", post(import_config))
        .layer(from_fn_with_state(Arc::clone(&console), require_admin));

    axum::Router::new()
        .merge(public)
        .merge(admin)
        .fallback(crate::proxy::handle)
        // A screenshot turn carries tens of megabytes of base64; axum's 2 MiB default would reject
        // it before the engine ever saw it. The engine applies its own limit downstream.
        .layer(axum::extract::DefaultBodyLimit::max(512 * 1024 * 1024))
        .with_state(console)
}

async fn require_admin(
    State(console): State<Arc<Console>>,
    request: Request,
    next: Next,
) -> Response {
    let token = auth::cookie_value(request.headers(), ADMIN_COOKIE);
    let allowed = console
        .auth
        .lock()
        .unwrap()
        .validate_admin(token.as_deref());
    if allowed {
        next.run(request).await
    } else if request.uri().path().starts_with("/ui/") {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "请先登录管理端" })),
        )
            .into_response()
    } else {
        Redirect::temporary("/login").into_response()
    }
}

fn with_cookie(response: Response, cookie: String) -> Response {
    let mut response = response;
    response.headers_mut().insert(
        header::SET_COOKIE,
        HeaderValue::from_str(&cookie).expect("session cookie is ascii"),
    );
    response
}

async fn login_page() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../web/login.html"))
}

async fn user_page() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../web/user.html"))
}

async fn index() -> axum::response::Html<&'static str> {
    axum::response::Html(include_str!("../web/index.html"))
}

async fn css() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("../web/app.css"),
    )
}

async fn js() -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )],
        include_str!("../web/app.js"),
    )
}

/// `anyhow::bail!` returns `Err(anyhow::Error)` without a conversion, so it cannot be used in a
/// handler whose error type is `UiError`.
fn ui_err(msg: impl std::fmt::Display) -> UiError {
    UiError(anyhow::anyhow!(msg.to_string()))
}

#[derive(Deserialize)]
struct PasswordBody {
    password: String,
}

#[derive(Deserialize)]
struct UserLoginBody {
    username: String,
    password: String,
}

#[derive(Deserialize)]
struct NewUserBody {
    username: String,
    password: String,
}

#[derive(Deserialize)]
struct ChangeAdminPasswordBody {
    current: String,
    new_password: String,
}

async fn admin_login(
    State(console): State<Arc<Console>>,
    Json(body): Json<PasswordBody>,
) -> Response {
    if !console.auth.lock().unwrap().verify_admin(&body.password) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "管理口令错误" })),
        )
            .into_response();
    }
    let token = console.auth.lock().unwrap().issue_admin();
    with_cookie(
        Json(serde_json::json!({ "ok": true })).into_response(),
        auth::cookie_header(ADMIN_COOKIE, &token, 12 * 60 * 60),
    )
}

async fn admin_logout(State(console): State<Arc<Console>>, request: Request) -> Response {
    let token = auth::cookie_value(request.headers(), ADMIN_COOKIE);
    console.auth.lock().unwrap().revoke_admin(token.as_deref());
    with_cookie(
        Json(serde_json::json!({ "ok": true })).into_response(),
        auth::cookie_header(ADMIN_COOKIE, "", 0),
    )
}

async fn user_login(
    State(console): State<Arc<Console>>,
    Json(body): Json<UserLoginBody>,
) -> Response {
    let ok = console
        .auth
        .lock()
        .unwrap()
        .authenticate_user(&body.username, &body.password);
    if !ok {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "用户名或密码错误" })),
        )
            .into_response();
    }
    let Some(token) = console.auth.lock().unwrap().issue_user(&body.username) else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "用户不存在" })),
        )
            .into_response();
    };
    with_cookie(
        Json(serde_json::json!({ "ok": true, "username": body.username })).into_response(),
        auth::cookie_header(USER_COOKIE, &token, 12 * 60 * 60),
    )
}

async fn user_logout(State(console): State<Arc<Console>>, request: Request) -> Response {
    let token = auth::cookie_value(request.headers(), USER_COOKIE);
    console.auth.lock().unwrap().revoke_user(token.as_deref());
    with_cookie(
        Json(serde_json::json!({ "ok": true })).into_response(),
        auth::cookie_header(USER_COOKIE, "", 0),
    )
}

async fn user_me(
    State(console): State<Arc<Console>>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    let token = auth::cookie_value(&headers, USER_COOKIE);
    let username = console.auth.lock().unwrap().validate_user(token.as_deref());
    Json(serde_json::json!({
        "authenticated": username.is_some(),
        "username": username,
    }))
}

async fn list_users(State(console): State<Arc<Console>>) -> Json<serde_json::Value> {
    let users = console
        .auth
        .lock()
        .unwrap()
        .users()
        .iter()
        .map(|u| u.view())
        .collect::<Vec<_>>();
    Json(serde_json::json!({ "users": users }))
}

async fn add_user(
    State(console): State<Arc<Console>>,
    Json(body): Json<NewUserBody>,
) -> UiResult<Json<serde_json::Value>> {
    let store = Arc::clone(&console.store);
    console
        .auth
        .lock()
        .unwrap()
        .add_user(&body.username, &body.password, &store)?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn delete_user(
    State(console): State<Arc<Console>>,
    Path(username): Path<String>,
) -> UiResult<Json<serde_json::Value>> {
    let store = Arc::clone(&console.store);
    console
        .auth
        .lock()
        .unwrap()
        .remove_user(&username, &store)?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

async fn change_admin_password(
    State(console): State<Arc<Console>>,
    Json(body): Json<ChangeAdminPasswordBody>,
) -> UiResult<Json<serde_json::Value>> {
    if body.new_password.len() < 8 {
        return Err(ui_err("新管理口令至少需要 8 个字符"));
    }
    let store = Arc::clone(&console.store);
    {
        let mut auth = console.auth.lock().unwrap();
        if !auth.verify_admin(&body.current) {
            return Err(ui_err("当前管理口令错误"));
        }
        auth.set_admin(&body.new_password, &store)?;
    }
    Ok(Json(serde_json::json!({ "ok": true })))
}

/// One snapshot of everything the dashboard renders. `request` bodies are stripped here — a turn
/// with a long conversation would otherwise be re-sent in full once per second; the sidebar fetches
/// the body of the one round it has open.
pub fn snapshot(console: &Console) -> serde_json::Value {
    let mut engine = console.engine.lock().unwrap().clone();
    engine.log_tail = console.engine_log.lock().unwrap().iter().cloned().collect();
    let rounds: Vec<serde_json::Value> = console
        .rounds
        .lock()
        .unwrap()
        .iter()
        .map(|r| {
            let mut v = serde_json::to_value(r).unwrap_or(serde_json::Value::Null);
            if let Some(obj) = v.as_object_mut() {
                obj.remove("request");
            }
            v
        })
        .collect();
    serde_json::json!({
        "engine": engine,
        "uncensor": console.uncensor.lock().unwrap().clone(),
        "series": console.series.lock().unwrap().clone(),
        "keys": console.keys.lock().unwrap().iter().map(ApiKey::view).collect::<Vec<KeyView>>(),
        "users": console.auth.lock().unwrap().users().iter().map(|u| serde_json::json!({ "username": u.username })).collect::<Vec<_>>(),
        "rounds": rounds,
        "console_log": console.console_log.lock().unwrap().iter().cloned().collect::<Vec<String>>(),
    })
}

pub async fn state(State(console): State<Arc<Console>>) -> Json<serde_json::Value> {
    Json(snapshot(&console))
}

/// Live updates. The subscriber count is what makes frame capture free when nobody is watching.
pub async fn events(
    State(console): State<Arc<Console>>,
) -> Sse<impl Stream<Item = Result<SseEvent, std::convert::Infallible>>> {
    console.subscribers.fetch_add(1, Ordering::Relaxed);
    let receiver = console.events.subscribe();
    let dirty = console.state_dirty.subscribe();
    let guard = Subscriber(Arc::clone(&console));
    let stream = futures_util::stream::unfold(
        (
            receiver,
            dirty,
            tokio::time::interval(Duration::from_millis(500)),
            guard,
            console,
        ),
        |(mut receiver, mut dirty, mut tick, guard, console)| async move {
            loop {
                tokio::select! {
                    received = receiver.recv() => match received {
                        Ok(frame) => {
                            let data = serde_json::to_string(&*frame).unwrap_or_default();
                            let event = SseEvent::default().event("frame").data(data);
                            return Some((Ok(event), (receiver, dirty, tick, guard, console)));
                        }
                        // A slow browser must not stall the proxy: drop the backlog and carry on.
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
                    },
                    // A round finished: push the new state at once so 空闲 shows immediately rather
                    // than up to a tick later. A burst of finishes collapses into one snapshot.
                    changed = dirty.recv() => match changed {
                        Ok(()) => {
                            while dirty.try_recv().is_ok() {}
                            let event = SseEvent::default().event("state").data(snapshot(&console).to_string());
                            return Some((Ok(event), (receiver, dirty, tick, guard, console)));
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
                    },
                    _ = tick.tick() => {
                        let event = SseEvent::default().event("state").data(snapshot(&console).to_string());
                        return Some((Ok(event), (receiver, dirty, tick, guard, console)));
                    }
                }
            }
        },
    );
    Sse::new(stream).keep_alive(KeepAlive::default())
}

struct Subscriber(Arc<Console>);

impl Drop for Subscriber {
    fn drop(&mut self) {
        self.0.subscribers.fetch_sub(1, Ordering::Relaxed);
    }
}

pub async fn round_detail(
    State(console): State<Arc<Console>>,
    Path(id): Path<u64>,
) -> UiResult<Json<serde_json::Value>> {
    let (round, frames) = console
        .round_snapshot(id)
        .ok_or_else(|| anyhow::anyhow!("轮次 {id} 已不在保留窗口内"))?;
    Ok(Json(
        serde_json::json!({ "round": round, "frames": frames }),
    ))
}

pub async fn clear_rounds(State(console): State<Arc<Console>>) -> Json<serde_json::Value> {
    console.rounds.lock().unwrap().clear();
    console.conversations.lock().unwrap().clear();
    console.log("历史已清空");
    Json(serde_json::json!({ "ok": true }))
}

#[derive(Deserialize)]
pub struct NewKey {
    pub label: String,
    #[serde(default)]
    pub token: Option<String>,
}

pub async fn add_key(
    State(console): State<Arc<Console>>,
    Json(req): Json<NewKey>,
) -> UiResult<Json<KeyView>> {
    let label = req.label.trim().to_string();
    if label.is_empty() {
        return Err(ui_err("名称不能为空"));
    }
    let token = match req
        .token
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        Some(t) => t.to_string(),
        None => new_token("sk-console-"),
    };
    let key = {
        let mut keys = console.keys.lock().unwrap();
        if keys.iter().any(|k| k.token == token) {
            return Err(ui_err("令牌已存在"));
        }
        if keys.iter().any(|k| k.label == label) {
            return Err(ui_err("名称已存在"));
        }
        let key = ApiKey {
            id: new_token("k-"),
            label,
            token,
            requests: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            cached_tokens: 0,
            ttft_sum_ms: 0,
            ttft_count: 0,
            prefill_sum: 0.0,
            prefill_count: 0,
            decode_sum: 0.0,
            decode_count: 0,
        };
        keys.push(key.clone());
        key
    };
    console.save_keys()?;
    console.log(format!("新增 API key「{}」", key.label));
    Ok(Json(key.view()))
}

pub async fn delete_key(
    State(console): State<Arc<Console>>,
    Path(id): Path<String>,
) -> UiResult<Json<serde_json::Value>> {
    {
        let mut keys = console.keys.lock().unwrap();
        let before = keys.len();
        keys.retain(|k| k.id != id);
        if keys.len() == before {
            return Err(ui_err("没有这个 key"));
        }
    }
    console.save_keys()?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

#[derive(Deserialize)]
pub struct UncensorRequest {
    pub on: bool,
    #[serde(default)]
    pub consent: bool,
}

pub async fn set_uncensor(
    State(console): State<Arc<Console>>,
    Json(req): Json<UncensorRequest>,
) -> UiResult<Json<crate::state::UncensorState>> {
    let mut state = console.uncensor.lock().unwrap();
    if req.on {
        if !state.vector_loaded {
            return Err(ui_err(
                "引擎未加载去审查向量（uncensor.vector 未配置），开关不会生效",
            ));
        }
        if !req.consent {
            return Err(ui_err("需要先勾选风险同意"));
        }
        state.consented = true;
        state.default_on = true;
        console.log("去审查投影：开（逐请求注入）");
    } else {
        state.default_on = false;
        console.log("去审查投影：关");
    }
    Ok(Json(state.clone()))
}

pub async fn schema(State(console): State<Arc<Console>>) -> UiResult<Json<serde_json::Value>> {
    Ok(Json(launch::schema(&console).await?))
}

pub async fn start_engine(
    State(console): State<Arc<Console>>,
    Json(req): Json<launch::LaunchRequest>,
) -> UiResult<Json<serde_json::Value>> {
    Ok(Json(launch::start(&console, req).await?))
}

pub async fn stop_engine(State(console): State<Arc<Console>>) -> UiResult<Json<serde_json::Value>> {
    launch::stop(&console).await?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

/// Download the current form as an engine TOML config file. The body is the same `{preset, set}` the
/// launch uses, so the exported file is exactly what the previewed command would run (minus the
/// CLI-only flags and secrets, which are not config-file knobs). Re-import it after an upgrade by
/// pointing `cli:config` at it.
pub async fn export_config(
    State(console): State<Arc<Console>>,
    Json(req): Json<launch::LaunchRequest>,
) -> UiResult<Response> {
    let payload = launch::schema(&console).await?;
    let fields = payload
        .get("schema")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let text = launch::build_toml(&req, &fields)?;
    Ok((
        [
            (
                axum::http::header::CONTENT_TYPE,
                "application/toml; charset=utf-8",
            ),
            (
                axum::http::header::CONTENT_DISPOSITION,
                "attachment; filename=\"infr-config.toml\"",
            ),
        ],
        text,
    )
        .into_response())
}

/// Read an uploaded TOML config back into `{preset, set}` so the browser can restore the form. The
/// body is the raw file text (the export downloads exactly this). Parsing is filtered against the
/// live schema, so a file written for another engine version restores what it can and drops the rest
/// instead of hard-erroring the next launch. The browser applies the result and saves it.
pub async fn import_config(
    State(console): State<Arc<Console>>,
    body: String,
) -> UiResult<Json<serde_json::Value>> {
    let payload = launch::schema(&console).await?;
    let fields = payload
        .get("schema")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let req = launch::parse_toml(&body, &fields)?;
    Ok(Json(
        serde_json::json!({ "preset": req.preset, "set": req.set }),
    ))
}

/// The operator's saved launch form. Read on page load so a reload — or a console restart — restores
/// the preset and touched knobs exactly as they were left, instead of falling back to schema defaults.
pub async fn get_settings(State(console): State<Arc<Console>>) -> Json<Settings> {
    Json(console.load_settings())
}

/// Persist the form. Values are not validated here: an unknown path is a HARD error at launch (see
/// `build_args`), which is the right place to refuse it — saving must not silently drop a knob the
/// operator typed, and the browser already filters restored keys against the live schema.
pub async fn save_settings(
    State(console): State<Arc<Console>>,
    Json(settings): Json<Settings>,
) -> UiResult<Json<serde_json::Value>> {
    console.save_settings(&settings)?;
    console.log("启动设置已保存");
    Ok(Json(serde_json::json!({ "ok": true })))
}

#[derive(Deserialize)]
pub struct VectorDownloadRequest {
    /// Full destination file path. Empty/absent = the wizard's default `models/control-vectors/…`.
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub consent: bool,
}

/// Fetch the pinned refusal-direction vector. Consent is enforced here as well as in the browser, so
/// the download cannot be triggered by a skipped dialog. The URLs are the wizard's real HF origin and
/// mirror — never a proxy.
pub async fn download_vector(
    State(console): State<Arc<Console>>,
    Json(req): Json<VectorDownloadRequest>,
) -> UiResult<Json<serde_json::Value>> {
    if !req.consent {
        return Err(ui_err("下载去审查向量需要先勾选风险同意"));
    }
    let dest = req
        .path
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(PathBuf::from);
    let path = crate::vector::download(&console, dest).await?;
    Ok(Json(
        serde_json::json!({ "path": path.display().to_string() }),
    ))
}

#[derive(Deserialize)]
pub struct VectorStatusQuery {
    #[serde(default)]
    pub path: Option<String>,
}

/// Read-only vector status: present? hash matches the canonical direction? where would it default to?
pub async fn vector_status(
    State(console): State<Arc<Console>>,
    Query(q): Query<VectorStatusQuery>,
) -> Json<serde_json::Value> {
    let path = q
        .path
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(PathBuf::from);
    Json(crate::vector::status(&console, path).await)
}
