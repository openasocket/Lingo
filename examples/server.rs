//! HTTP server exposing NLLB translation and LaBSE/SONAR similarity as REST endpoints.
//!
//! ```bash
//! LINGO_ACCEPT_LICENSE=1 cargo run --release --example server --features server,metal
//! ```
//!
//! Endpoints:
//!   POST /translate       — translate text
//!   POST /translate_batch — translate many texts (pure translation, no scoring)
//!   POST /score           — score similarity between two texts (LaBSE)
//!   POST /embed           — get 768-dim embedding for text (LaBSE)
//!   POST /embed_batch     — get embeddings for multiple texts (LaBSE)
//!   POST /analyze         — translate + LaBSE & SONAR scores in one call
//!   POST /analyze_batch   — batch translate + score
//!   GET  /health          — server health check
//!
//! Translation-bearing responses include chunk metadata so clients can see
//! when long input was split across multiple NLLB passes: single-result
//! responses (`/translate`, `/analyze`, and each entry of `/analyze_batch`)
//! carry a `chunk_count` field, and `/translate_batch` carries a parallel
//! `chunk_counts` array aligned position-wise with `translations`. A value
//! of 1 means the input fit in a single pass; greater than 1 means it was
//! chunked.
//!
//! ## Configuration (environment variables)
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `LINGO_SERVER_ADDR` | `127.0.0.1:3000` | Listen address. Use `0.0.0.0:3000` to expose on the network — there is no authentication, so only do that behind a trusted proxy/firewall. |
//! | `LINGO_REQUEST_TIMEOUT_SECS` | `300` | Budget per model call. A call that exceeds it returns 504; the instance is kept if its abandoned work stops within 5 s, otherwise (wedged GPU call) it is replaced. |
//! | `LINGO_MAX_PENDING` | `32` | Requests admitted at once (running + queued); beyond this the server answers 503. |
//! | `LINGO_MAX_BATCH` | `256` | Max `texts` per batch request. |
//! | `LINGO_MAX_TEXT_BYTES` | `65536` | Max UTF-8 bytes per input text. |

use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use lingo::{LaBSEEncoder, NllbTranslator, SonarEncoder};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::{error, info, warn};

// ---------------------------------------------------------------------------
// Request / response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TranslateRequest { text: String, source: String, target: String }
#[derive(Serialize)]
struct TranslateResponse { translation: String, source_lang: String, target_lang: String, duration_ms: u64, chunk_count: usize }

#[derive(Deserialize)]
struct ScoreRequest { text1: String, text2: String }
#[derive(Serialize)]
struct ScoreResponse { score: f32, duration_ms: u64 }

#[derive(Deserialize)]
struct EmbedRequest { text: String }
#[derive(Serialize)]
struct EmbedResponse { embedding: Vec<f32>, dimensions: usize }

#[derive(Deserialize)]
struct EmbedBatchRequest { texts: Vec<String> }
#[derive(Serialize)]
struct EmbedBatchResponse { embeddings: Vec<Vec<f32>>, dimensions: usize, count: usize }

#[derive(Deserialize)]
struct TranslateBatchRequest { texts: Vec<String>, source: String, target: String }
#[derive(Serialize)]
struct TranslateBatchResponse {
    translations: Vec<String>,
    chunk_counts: Vec<usize>,
    count: usize,
    source_lang: String,
    target_lang: String,
    total_duration_ms: u64,
}

#[derive(Deserialize)]
struct AnalyzeRequest { text: String, source: String, target: String }
#[derive(Serialize)]
struct AnalyzeResponse {
    source_text: String,
    translation: String,
    source_lang: String,
    target_lang: String,
    labse_score: f32,
    sonar_score: f32,
    duration_ms: u64,
    chunk_count: usize,
}

#[derive(Deserialize)]
struct AnalyzeBatchRequest { texts: Vec<String>, source: String, target: String }
#[derive(Serialize)]
struct AnalyzeBatchResponse { results: Vec<AnalyzeResponse>, count: usize, total_duration_ms: u64 }

#[derive(Serialize)]
struct ErrorResponse { error: String }

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    /// Model loaded into memory and ready to serve.
    nllb_loaded: bool,
    labse_loaded: bool,
    sonar_loaded: bool,
    /// Model files present on disk.
    nllb_downloaded: bool,
    labse_downloaded: bool,
    sonar_downloaded: bool,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Limits {
    timeout: Duration,
    max_batch: usize,
    max_text_bytes: usize,
}

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(v) => v.parse().unwrap_or_else(|_| {
            warn!("ignoring invalid {name}={v:?}");
            default
        }),
        Err(_) => default,
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(ErrorResponse { error: self.1 })).into_response()
    }
}

type ApiResult<T> = std::result::Result<Json<T>, ApiError>;

fn bad_request(msg: impl Into<String>) -> ApiError {
    ApiError(StatusCode::BAD_REQUEST, msg.into())
}

/// Map a library error to an HTTP error without leaking server-side details
/// (file paths, internal messages) to the client; the full error is logged.
fn api_error(model: &str, e: &lingo::Error) -> ApiError {
    use lingo::Error as E;
    match e {
        E::InvalidLanguagePair(..) | E::UnsupportedLanguage(..) => bad_request(e.to_string()),
        E::ModelNotFound(..) | E::LicenseNotAccepted => {
            warn!("{model}: unavailable: {e}");
            ApiError(StatusCode::SERVICE_UNAVAILABLE, format!("{model} model is not available on this server"))
        }
        E::Timeout(d) => ApiError(
            StatusCode::GATEWAY_TIMEOUT,
            format!("{model} call exceeded {}s", d.as_secs()),
        ),
        _ => {
            error!("{model}: {e}");
            ApiError(StatusCode::INTERNAL_SERVER_ERROR, format!("{model} inference failed"))
        }
    }
}

// ---------------------------------------------------------------------------
// Hot-swappable model slot
// ---------------------------------------------------------------------------

/// How long a timed-out instance gets to stop its abandoned work before it
/// is judged wedged and replaced.
const IDLE_GRACE: std::time::Duration = std::time::Duration::from_secs(5);

/// Models whose abandoned work can be awaited after a timeout.
trait WaitIdle {
    async fn wait_idle(&self, grace: std::time::Duration) -> bool;
}

impl WaitIdle for NllbTranslator {
    async fn wait_idle(&self, grace: std::time::Duration) -> bool {
        NllbTranslator::wait_idle(self, grace).await
    }
}
impl WaitIdle for LaBSEEncoder {
    async fn wait_idle(&self, grace: std::time::Duration) -> bool {
        LaBSEEncoder::wait_idle(self, grace).await
    }
}
impl WaitIdle for SonarEncoder {
    async fn wait_idle(&self, grace: std::time::Duration) -> bool {
        SonarEncoder::wait_idle(self, grace).await
    }
}

/// Holds the current instance of a model. After a timeout the instance is
/// given `IDLE_GRACE` to stop its abandoned work; if it is still busy, its
/// worker is wedged in a GPU call (holding the model mutex) and the slot
/// swaps in a fresh, lazily-loading instance. Requests already queued on the
/// old one will time out in turn.
struct Slot<T> {
    name: &'static str,
    current: RwLock<Arc<T>>,
    make: fn() -> lingo::Result<T>,
}

impl<T: WaitIdle> Slot<T> {
    fn new(name: &'static str, make: fn() -> lingo::Result<T>) -> lingo::Result<Self> {
        Ok(Self { name, current: RwLock::new(Arc::new(make()?)), make })
    }

    fn get(&self) -> Arc<T> {
        self.current.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Map an error from `instance`; on timeout, recycle the slot if the
    /// instance does not go idle within `IDLE_GRACE`.
    async fn fail(&self, instance: &Arc<T>, e: lingo::Error) -> ApiError {
        if matches!(e, lingo::Error::Timeout(_)) && instance.wait_idle(IDLE_GRACE).await {
            warn!("{}: call timed out; abandoned work stopped, instance kept", self.name);
        } else if matches!(e, lingo::Error::Timeout(_)) {
            let mut cur = self.current.write().unwrap_or_else(|e| e.into_inner());
            if Arc::ptr_eq(&cur, instance) {
                match (self.make)() {
                    Ok(fresh) => {
                        *cur = Arc::new(fresh);
                        warn!("{}: call timed out; replaced poisoned instance", self.name);
                    }
                    Err(err) => error!("{}: failed to recreate instance: {err}", self.name),
                }
            }
        }
        api_error(self.name, &e)
    }
}

struct AppState {
    translator: Slot<NllbTranslator>,
    labse: Slot<LaBSEEncoder>,
    sonar: Slot<SonarEncoder>,
    admission: Arc<Semaphore>,
    limits: Limits,
}

impl AppState {
    /// Admit a request or answer 503 when too many are already in flight.
    fn admit(&self) -> std::result::Result<OwnedSemaphorePermit, ApiError> {
        Arc::clone(&self.admission).try_acquire_owned().map_err(|_| {
            ApiError(StatusCode::SERVICE_UNAVAILABLE, "server busy; retry later".into())
        })
    }

    fn check_text(&self, field: &str, text: &str) -> std::result::Result<(), ApiError> {
        if text.len() > self.limits.max_text_bytes {
            return Err(bad_request(format!(
                "{field} exceeds {} bytes", self.limits.max_text_bytes
            )));
        }
        Ok(())
    }

    fn check_batch(&self, texts: &[String]) -> std::result::Result<(), ApiError> {
        if texts.is_empty() {
            return Err(bad_request("texts array must not be empty"));
        }
        if texts.len() > self.limits.max_batch {
            return Err(bad_request(format!("texts array exceeds {} items", self.limits.max_batch)));
        }
        for (i, t) in texts.iter().enumerate() {
            self.check_text(&format!("texts[{i}]"), t)?;
        }
        Ok(())
    }

    async fn translate(&self, text: &str, source: &str, target: &str) -> std::result::Result<lingo::TranslationResult, ApiError> {
        let t = self.translator.get();
        match t.translate_with_timeout(text, source, target, self.limits.timeout).await {
            Ok(r) => Ok(r),
            Err(e) => Err(self.translator.fail(&t, e).await),
        }
    }

    async fn labse_score(&self, a: &str, b: &str) -> std::result::Result<f32, ApiError> {
        let m = self.labse.get();
        match m.score_with_timeout(a, b, self.limits.timeout).await {
            Ok(r) => Ok(r),
            Err(e) => Err(self.labse.fail(&m, e).await),
        }
    }

    async fn sonar_score(&self, a: &str, lang_a: &str, b: &str, lang_b: &str) -> std::result::Result<f32, ApiError> {
        let m = self.sonar.get();
        match m.score_langs_with_timeout(a, lang_a, b, lang_b, self.limits.timeout).await {
            Ok(r) => Ok(r),
            Err(e) => Err(self.sonar.fail(&m, e).await),
        }
    }

    async fn labse_embed(&self, text: &str) -> std::result::Result<Vec<f32>, ApiError> {
        let m = self.labse.get();
        match m.embed_with_timeout(text, self.limits.timeout).await {
            Ok(r) => Ok(r),
            Err(e) => Err(self.labse.fail(&m, e).await),
        }
    }

    async fn analyze_one(&self, text: &str, source: &str, target: &str) -> std::result::Result<AnalyzeResponse, ApiError> {
        let start = std::time::Instant::now();
        let translation = self.translate(text, source, target).await?;
        let labse_score = self.labse_score(text, &translation.text).await?;
        let sonar_score = self.sonar_score(text, source, &translation.text, target).await?;
        Ok(AnalyzeResponse {
            source_text: text.to_string(),
            translation: translation.text,
            source_lang: translation.source_lang,
            target_lang: translation.target_lang,
            labse_score,
            sonar_score,
            duration_ms: start.elapsed().as_millis() as u64,
            chunk_count: translation.chunk_count,
        })
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let (t, l, s) = (state.translator.get(), state.labse.get(), state.sonar.get());
    Json(HealthResponse {
        status: "ok".into(),
        nllb_loaded: t.is_loaded(),
        labse_loaded: l.is_loaded(),
        sonar_loaded: s.is_loaded(),
        nllb_downloaded: t.is_model_downloaded(),
        labse_downloaded: l.is_model_downloaded(),
        sonar_downloaded: s.is_model_downloaded(),
    })
}

async fn translate(State(state): State<Arc<AppState>>, Json(req): Json<TranslateRequest>) -> ApiResult<TranslateResponse> {
    let _permit = state.admit()?;
    state.check_text("text", &req.text)?;
    let r = state.translate(&req.text, &req.source, &req.target).await?;
    Ok(Json(TranslateResponse {
        translation: r.text,
        source_lang: r.source_lang,
        target_lang: r.target_lang,
        duration_ms: r.duration_ms,
        chunk_count: r.chunk_count,
    }))
}

async fn score(State(state): State<Arc<AppState>>, Json(req): Json<ScoreRequest>) -> ApiResult<ScoreResponse> {
    let _permit = state.admit()?;
    state.check_text("text1", &req.text1)?;
    state.check_text("text2", &req.text2)?;
    let start = std::time::Instant::now();
    let score = state.labse_score(&req.text1, &req.text2).await?;
    Ok(Json(ScoreResponse { score, duration_ms: start.elapsed().as_millis() as u64 }))
}

async fn embed(State(state): State<Arc<AppState>>, Json(req): Json<EmbedRequest>) -> ApiResult<EmbedResponse> {
    let _permit = state.admit()?;
    state.check_text("text", &req.text)?;
    let embedding = state.labse_embed(&req.text).await?;
    let dimensions = embedding.len();
    Ok(Json(EmbedResponse { embedding, dimensions }))
}

async fn embed_batch(State(state): State<Arc<AppState>>, Json(req): Json<EmbedBatchRequest>) -> ApiResult<EmbedBatchResponse> {
    let _permit = state.admit()?;
    state.check_batch(&req.texts)?;
    let mut embeddings = Vec::with_capacity(req.texts.len());
    for text in &req.texts {
        embeddings.push(state.labse_embed(text).await?);
    }
    let dimensions = embeddings.first().map_or(0, |e| e.len());
    let count = embeddings.len();
    Ok(Json(EmbedBatchResponse { embeddings, dimensions, count }))
}

async fn translate_batch(State(state): State<Arc<AppState>>, Json(req): Json<TranslateBatchRequest>) -> ApiResult<TranslateBatchResponse> {
    let _permit = state.admit()?;
    state.check_batch(&req.texts)?;
    let start = std::time::Instant::now();
    let mut translations = Vec::with_capacity(req.texts.len());
    let mut chunk_counts = Vec::with_capacity(req.texts.len());
    for text in &req.texts {
        let r = state.translate(text, &req.source, &req.target).await?;
        translations.push(r.text);
        chunk_counts.push(r.chunk_count);
    }
    let count = translations.len();
    Ok(Json(TranslateBatchResponse {
        translations,
        chunk_counts,
        count,
        source_lang: req.source,
        target_lang: req.target,
        total_duration_ms: start.elapsed().as_millis() as u64,
    }))
}

async fn analyze(State(state): State<Arc<AppState>>, Json(req): Json<AnalyzeRequest>) -> ApiResult<AnalyzeResponse> {
    let _permit = state.admit()?;
    state.check_text("text", &req.text)?;
    Ok(Json(state.analyze_one(&req.text, &req.source, &req.target).await?))
}

async fn analyze_batch(State(state): State<Arc<AppState>>, Json(req): Json<AnalyzeBatchRequest>) -> ApiResult<AnalyzeBatchResponse> {
    let _permit = state.admit()?;
    state.check_batch(&req.texts)?;
    let start = std::time::Instant::now();
    let mut results = Vec::with_capacity(req.texts.len());
    for text in &req.texts {
        results.push(state.analyze_one(text, &req.source, &req.target).await?);
    }
    let count = results.len();
    Ok(Json(AnalyzeBatchResponse { results, count, total_duration_ms: start.elapsed().as_millis() as u64 }))
}

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

/// Preload a model if its files are present. Failures are logged, not fatal:
/// endpoints that need the model answer 503 until it becomes available.
async fn preload(name: &str, downloaded: bool, load: impl std::future::Future<Output = lingo::Result<()>>) {
    if !downloaded {
        println!("  {name}: not found (endpoints using it will return 503)");
        return;
    }
    match load.await {
        Ok(()) => println!("  {name}: loaded"),
        Err(e) => {
            error!("{name}: failed to load: {e}");
            println!("  {name}: FAILED to load (see log); endpoints using it will error");
        }
    }
}

/// Resolve on Ctrl-C, or SIGTERM on Unix (service managers, `kill`).
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    info!("shutting down");
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "lingo=info,server=info".into()),
        )
        .init();

    let addr: String = env_parse("LINGO_SERVER_ADDR", "127.0.0.1:3000".to_string());
    let limits = Limits {
        timeout: Duration::from_secs(env_parse("LINGO_REQUEST_TIMEOUT_SECS", 300u64).max(1)),
        max_batch: env_parse("LINGO_MAX_BATCH", 256usize).max(1),
        max_text_bytes: env_parse("LINGO_MAX_TEXT_BYTES", 64 * 1024usize).max(1),
    };
    let max_pending: usize = env_parse("LINGO_MAX_PENDING", 32usize).max(1);
    // Largest legitimate body: a full batch of max-size texts plus JSON overhead.
    let body_limit = limits.max_batch.saturating_mul(limits.max_text_bytes).saturating_add(64 * 1024);

    let state = Arc::new(AppState {
        translator: Slot::new("NLLB-200", || NllbTranslator::new(None))?,
        labse: Slot::new("LaBSE", || LaBSEEncoder::new(None))?,
        sonar: Slot::new("SONAR", || SonarEncoder::new(None))?,
        admission: Arc::new(Semaphore::new(max_pending)),
        limits,
    });

    println!("Loading models...");
    let (t, l, s) = (state.translator.get(), state.labse.get(), state.sonar.get());
    preload("NLLB-200", t.is_model_downloaded(), t.load()).await;
    preload("LaBSE", l.is_model_downloaded(), l.load()).await;
    preload("SONAR", s.is_model_downloaded(), s.load()).await;

    let app = Router::new()
        .route("/health", get(health))
        .route("/translate", post(translate))
        .route("/translate_batch", post(translate_batch))
        .route("/score", post(score))
        .route("/embed", post(embed))
        .route("/embed_batch", post(embed_batch))
        .route("/analyze", post(analyze))
        .route("/analyze_batch", post(analyze_batch))
        .layer(DefaultBodyLimit::max(body_limit))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("listening on {addr}");
    println!("\nServer on http://{addr}");
    if !addr.starts_with("127.") && !addr.starts_with("localhost") && !addr.starts_with("[::1]") {
        warn!("listening on a non-loopback address with no authentication");
    }
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    // A wedged GPU worker would block runtime shutdown; exit explicitly.
    std::process::exit(0);
}
