//! The dashboard's own HTTP surface. Separate from the proxied `/v1/*` path: these routes are for
//! the operator's browser, are never forwarded upstream, and are NOT gated by the client API keys
//! (those authenticate inference callers, not the person sitting at the console).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::Json;
use futures_util::Stream;
use serde::Deserialize;

use crate::state::{new_token, ApiKey, Console, KeyView};
use crate::launch;

pub struct UiError(anyhow::Error);

impl IntoResponse for UiError {
    fn into_response(self) -> Response {
        (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": self.0.to_string() }))).into_response()
    }
}

impl From<anyhow::Error> for UiError {
    fn from(e: anyhow::Error) -> Self {
        Self(e)
    }
}

type UiResult<T> = Result<T, UiError>;

/// The whole console surface: embedded assets, the dashboard's own API, and the proxy as fallback.
pub fn router(console: Arc<Console>) -> axum::Router {
    axum::Router::new()
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
        .route("/ui/schema", get(schema))
        .route("/ui/launch", post(start_engine))
        .route("/ui/stop", post(stop_engine))
        .fallback(crate::proxy::handle)
        // A screenshot turn carries tens of megabytes of base64; axum's 2 MiB default would reject
        // it before the engine ever saw it. The engine applies its own limit downstream.
        .layer(axum::extract::DefaultBodyLimit::max(512 * 1024 * 1024))
        .with_state(console)
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
        [(axum::http::header::CONTENT_TYPE, "application/javascript; charset=utf-8")],
        include_str!("../web/app.js"),
    )
}

/// `anyhow::bail!` returns `Err(anyhow::Error)` without a conversion, so it cannot be used in a
/// handler whose error type is `UiError`.
fn ui_err(msg: impl std::fmt::Display) -> UiError {
    UiError(anyhow::anyhow!(msg.to_string()))
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
    let guard = Subscriber(Arc::clone(&console));
    let stream = futures_util::stream::unfold(
        (receiver, tokio::time::interval(Duration::from_secs(1)), guard, console),
        |(mut receiver, mut tick, guard, console)| async move {
            loop {
                tokio::select! {
                    received = receiver.recv() => match received {
                        Ok(frame) => {
                            let data = serde_json::to_string(&*frame).unwrap_or_default();
                            let event = SseEvent::default().event("frame").data(data);
                            return Some((Ok(event), (receiver, tick, guard, console)));
                        }
                        // A slow browser must not stall the proxy: drop the backlog and carry on.
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
                    },
                    _ = tick.tick() => {
                        let event = SseEvent::default().event("state").data(snapshot(&console).to_string());
                        return Some((Ok(event), (receiver, tick, guard, console)));
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
    Ok(Json(serde_json::json!({ "round": round, "frames": frames })))
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
    let token = match req.token.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
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
            return Err(ui_err("引擎未加载去审查向量（uncensor.vector 未配置），开关不会生效"));
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