//! Server-hosted browser control plane for infr.
//!
//! The control process deliberately does not own a Vulkan device. It supervises an `infr serve`
//! worker, so switching models releases every GPU allocation with the worker process while this
//! UI and its stable management endpoint remain alive.

mod catalog;
mod model;
mod worker;

use std::{
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, Context};
use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use model::{
    ApiMessage, Bootstrap, DirectoryRequest, DownloadRequest, FavoriteRequest, GuiState,
    ModelProfile, StatusSnapshot, StopRequest,
};
use subtle::ConstantTimeEq;
use tokio::sync::RwLock;

const INDEX_HTML: &str = include_str!("../web/index.html");
const APP_JS: &str = include_str!("../web/app.js");
const STYLE_CSS: &str = include_str!("../web/style.css");

#[derive(Parser, Debug)]
#[command(
    name = "infr-gui",
    about = "Server-hosted browser control plane for infr"
)]
struct Args {
    /// GUI/control listen address. Binding outside loopback requires --key-file.
    #[arg(long, default_value = "127.0.0.1:8180")]
    addr: SocketAddr,
    /// File containing the GUI management key. Required when listening outside loopback.
    #[arg(long)]
    key_file: Option<PathBuf>,
    /// Path to the infr CLI worker binary. Defaults to the sibling of infr-gui.
    #[arg(long)]
    infr: Option<PathBuf>,
    /// Persistent GUI state, stop files and logs.
    #[arg(long, default_value = "gui-data")]
    data_dir: PathBuf,
    /// Repository/worker current directory.
    #[arg(long)]
    workdir: Option<PathBuf>,
}

#[derive(Clone)]
struct AppState {
    inner: Arc<Inner>,
}

struct Inner {
    admin_key: String,
    data_dir: PathBuf,
    state_file: PathBuf,
    infr_path: PathBuf,
    workdir: PathBuf,
    saved: RwLock<GuiState>,
    catalog: RwLock<Vec<model::ModelInfo>>,
    worker: worker::WorkerManager,
    download: worker::DownloadManager,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn conflict(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            message: message.into(),
        }
    }

    fn internal(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "ok": false, "error": self.message })),
        )
            .into_response()
    }
}

type ApiResult<T> = Result<Json<T>, ApiError>;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "infr_gui=info".into()),
        )
        .init();

    let args = Args::parse();
    let admin_key = match args.key_file.as_deref() {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("reading management key from {}", path.display()))?
            .trim()
            .to_string(),
        None => String::new(),
    };
    if !args.addr.ip().is_loopback() && admin_key.is_empty() {
        bail!("--key-file with a non-empty management key is required when --addr is not loopback");
    }

    std::fs::create_dir_all(&args.data_dir)
        .with_context(|| format!("creating {}", args.data_dir.display()))?;
    let data_dir = std::fs::canonicalize(&args.data_dir).unwrap_or(args.data_dir.clone());
    let state_file = data_dir.join("state.json");
    let saved = model::load_state(&state_file)?;
    let workdir = args
        .workdir
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    let infr_path = args.infr.unwrap_or_else(default_infr_path);

    let catalog = catalog::scan_state(&saved);
    let app = AppState {
        inner: Arc::new(Inner {
            admin_key,
            data_dir,
            state_file,
            infr_path,
            workdir,
            saved: RwLock::new(saved),
            catalog: RwLock::new(catalog),
            worker: worker::WorkerManager::default(),
            download: worker::DownloadManager::default(),
        }),
    };
    worker::spawn_monitor(app.clone());

    let router = Router::new()
        .route("/", get(index))
        .route("/app.js", get(app_js))
        .route("/style.css", get(style_css))
        .route("/api/auth", get(api_auth))
        .route("/api/bootstrap", get(api_bootstrap))
        .route("/api/status", get(api_status))
        .route("/api/directories/add", post(api_directory_add))
        .route("/api/directories/remove", post(api_directory_remove))
        .route("/api/models/rescan", post(api_rescan))
        .route("/api/favorites/toggle", post(api_favorite_toggle))
        .route("/api/profiles/save", post(api_profile_save))
        .route("/api/profiles/delete", post(api_profile_delete))
        .route("/api/estimate", post(api_estimate))
        .route("/api/worker/start", post(api_worker_start))
        .route("/api/worker/stop", post(api_worker_stop))
        .route("/api/downloads/start", post(api_download_start))
        .with_state(app.clone());

    let listener = tokio::net::TcpListener::bind(args.addr).await?;
    let shown = if args.addr.ip().is_unspecified() {
        SocketAddr::new(IpAddr::V4(std::net::Ipv4Addr::LOCALHOST), args.addr.port())
    } else {
        args.addr
    };
    println!("INFR GUI: http://{shown}");
    println!("Worker binary: {}", app.inner.infr_path.display());
    if !app.inner.admin_key.is_empty() {
        println!("Management API is protected by INFR_GUI_KEY.");
    }
    let shutdown_app = app.clone();
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                println!("Stopping INFR GUI and draining its worker...");
                if let Err(error) = shutdown_app.inner.worker.stop_and_wait().await {
                    eprintln!("worker shutdown warning: {error}");
                }
            }
        })
        .await?;
    Ok(())
}

fn default_infr_path() -> PathBuf {
    let name = if cfg!(windows) { "infr.exe" } else { "infr" };
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.join(name)))
        .unwrap_or_else(|| PathBuf::from(name))
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn app_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        APP_JS,
    )
}

async fn style_css() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        STYLE_CSS,
    )
}

fn authorize(app: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    if app.inner.admin_key.is_empty() {
        return Ok(());
    }
    let supplied = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .unwrap_or_default();
    let expected = app.inner.admin_key.as_bytes();
    let got = supplied.as_bytes();
    if expected.len() == got.len() && bool::from(expected.ct_eq(got)) {
        Ok(())
    } else {
        Err(ApiError {
            status: StatusCode::UNAUTHORIZED,
            message: "management key is missing or invalid".into(),
        })
    }
}

async fn api_auth(State(app): State<AppState>, headers: HeaderMap) -> ApiResult<ApiMessage> {
    authorize(&app, &headers)?;
    Ok(Json(ApiMessage::ok("authenticated")))
}

async fn api_bootstrap(State(app): State<AppState>, headers: HeaderMap) -> ApiResult<Bootstrap> {
    authorize(&app, &headers)?;
    let saved = app.inner.saved.read().await.clone();
    let catalog = app.inner.catalog.read().await.clone();
    let runtime = app.inner.worker.status().await;
    let download = app.inner.download.status().await;
    let devices =
        infr_vulkan::VulkanBackend::enumerate_devices(&infr_core::config::Config::default())
            .unwrap_or_default()
            .into_iter()
            .map(model::DeviceView::from)
            .collect();
    let defaults = infr_core::config::Config::default();
    let config_schema = infr_core::config::Config::all_paths()
        .into_iter()
        .filter(|p| !p.ends_with("_specified") && p != "serve.shutdown_file")
        .map(|path| model::ConfigField {
            default_value: defaults.get_path(&path).unwrap_or_default(),
            path,
        })
        .collect();
    Ok(Json(Bootstrap {
        saved,
        catalog,
        devices,
        runtime,
        download,
        config_schema,
    }))
}

async fn api_status(State(app): State<AppState>, headers: HeaderMap) -> ApiResult<StatusSnapshot> {
    authorize(&app, &headers)?;
    Ok(Json(StatusSnapshot {
        runtime: app.inner.worker.status().await,
        download: app.inner.download.status().await,
    }))
}

async fn api_directory_add(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DirectoryRequest>,
) -> ApiResult<ApiMessage> {
    authorize(&app, &headers)?;
    let path = PathBuf::from(req.path.trim());
    if !path.is_dir() {
        return Err(ApiError::bad(format!(
            "directory does not exist: {}",
            path.display()
        )));
    }
    {
        let mut saved = app.inner.saved.write().await;
        if !saved.directories.iter().any(|p| same_path(p, &path)) {
            saved.directories.push(path);
        }
        persist(&app, &saved)?;
    }
    rescan(&app).await?;
    Ok(Json(ApiMessage::ok("directory added and scanned")))
}

async fn api_directory_remove(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DirectoryRequest>,
) -> ApiResult<ApiMessage> {
    authorize(&app, &headers)?;
    let path = PathBuf::from(req.path.trim());
    {
        let mut saved = app.inner.saved.write().await;
        saved.directories.retain(|p| !same_path(p, &path));
        persist(&app, &saved)?;
    }
    rescan(&app).await?;
    Ok(Json(ApiMessage::ok("directory removed")))
}

async fn api_rescan(State(app): State<AppState>, headers: HeaderMap) -> ApiResult<ApiMessage> {
    authorize(&app, &headers)?;
    let count = rescan(&app).await?;
    Ok(Json(ApiMessage::ok(format!("found {count} model(s)"))))
}

async fn rescan(app: &AppState) -> Result<usize, ApiError> {
    let saved = app.inner.saved.read().await.clone();
    let models = tokio::task::spawn_blocking(move || catalog::scan_state(&saved))
        .await
        .map_err(ApiError::internal)?;
    let count = models.len();
    *app.inner.catalog.write().await = models;
    Ok(count)
}

async fn api_favorite_toggle(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<FavoriteRequest>,
) -> ApiResult<ApiMessage> {
    authorize(&app, &headers)?;
    let mut saved = app.inner.saved.write().await;
    if let Some(i) = saved.favorites.iter().position(|p| p == &req.path) {
        saved.favorites.remove(i);
    } else {
        saved.favorites.push(req.path);
    }
    persist(&app, &saved)?;
    Ok(Json(ApiMessage::ok("favorite updated")))
}

async fn api_profile_save(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(profile): Json<ModelProfile>,
) -> ApiResult<ApiMessage> {
    authorize(&app, &headers)?;
    worker::validate_profile(&profile).map_err(ApiError::bad)?;
    save_profile(&app, profile).await?;
    Ok(Json(ApiMessage::ok("profile saved")))
}

async fn api_profile_delete(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<FavoriteRequest>,
) -> ApiResult<ApiMessage> {
    authorize(&app, &headers)?;
    let mut saved = app.inner.saved.write().await;
    saved.profiles.retain(|p| p.id != req.path);
    persist(&app, &saved)?;
    Ok(Json(ApiMessage::ok("profile deleted")))
}

async fn api_estimate(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(profile): Json<ModelProfile>,
) -> ApiResult<model::MemoryEstimate> {
    authorize(&app, &headers)?;
    worker::validate_profile(&profile).map_err(ApiError::bad)?;
    let devices =
        infr_vulkan::VulkanBackend::enumerate_devices(&infr_core::config::Config::default())
            .unwrap_or_default();
    let estimate = tokio::task::spawn_blocking(move || catalog::estimate(&profile, &devices))
        .await
        .map_err(ApiError::internal)?
        .map_err(ApiError::bad)?;
    Ok(Json(estimate))
}

async fn api_worker_start(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(profile): Json<ModelProfile>,
) -> ApiResult<ApiMessage> {
    authorize(&app, &headers)?;
    worker::validate_profile(&profile).map_err(ApiError::bad)?;
    save_profile(&app, profile.clone()).await?;
    {
        let mut saved = app.inner.saved.write().await;
        saved.recent.retain(|p| p != &profile.model_path);
        saved.recent.insert(0, profile.model_path.clone());
        saved.recent.truncate(12);
        persist(&app, &saved)?;
    }
    app.inner
        .worker
        .switch(
            &app,
            profile,
            &app.inner.infr_path,
            &app.inner.workdir,
            &app.inner.data_dir,
        )
        .await
        .map_err(ApiError::conflict)?;
    Ok(Json(ApiMessage::ok("worker start requested")))
}

async fn api_worker_stop(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<StopRequest>,
) -> ApiResult<ApiMessage> {
    authorize(&app, &headers)?;
    app.inner
        .worker
        .stop(req.force)
        .await
        .map_err(ApiError::conflict)?;
    Ok(Json(ApiMessage::ok(if req.force {
        "worker force-stop requested"
    } else {
        "worker graceful stop requested"
    })))
}

async fn api_download_start(
    State(app): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DownloadRequest>,
) -> ApiResult<ApiMessage> {
    authorize(&app, &headers)?;
    app.inner
        .download
        .start(app.clone(), req, &app.inner.infr_path, &app.inner.workdir)
        .await
        .map_err(ApiError::conflict)?;
    Ok(Json(ApiMessage::ok("download started")))
}

async fn save_profile(app: &AppState, profile: ModelProfile) -> Result<(), ApiError> {
    let mut saved = app.inner.saved.write().await;
    if let Some(existing) = saved.profiles.iter_mut().find(|p| p.id == profile.id) {
        *existing = profile;
    } else {
        saved.profiles.push(profile);
    }
    persist(app, &saved)
}

fn persist(app: &AppState, state: &GuiState) -> Result<(), ApiError> {
    model::save_state(&app.inner.state_file, state).map_err(ApiError::internal)
}

fn same_path(a: &Path, b: &Path) -> bool {
    if cfg!(windows) {
        a.to_string_lossy()
            .eq_ignore_ascii_case(&b.to_string_lossy())
    } else {
        a == b
    }
}
