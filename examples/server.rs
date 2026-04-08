//! HTTP server exposing NLLB translation and LaBSE similarity as REST endpoints.
//!
//! ```bash
//! cargo run --example server --features server,metal
//! ```
//!
//! Endpoints:
//!   POST /translate      — translate text
//!   POST /score          — score similarity between two texts
//!   POST /embed          — get 768-dim embedding for text
//!   POST /embed_batch    — get embeddings for multiple texts
//!   POST /analyze        — translate + LaBSE & SONAR scores in one call
//!   POST /analyze_batch  — batch translate + score
//!   GET  /health         — server health check

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use lingo::{LaBSEEncoder, NllbTranslator, SonarEncoder};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize)]
struct TranslateRequest { text: String, source: String, target: String }
#[derive(Serialize)]
struct TranslateResponse { translation: String, source_lang: String, target_lang: String, duration_ms: u64 }

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
}

#[derive(Deserialize)]
struct AnalyzeBatchRequest { texts: Vec<String>, source: String, target: String }
#[derive(Serialize)]
struct AnalyzeBatchResponse { results: Vec<AnalyzeResponse>, count: usize, total_duration_ms: u64 }

#[derive(Serialize)]
struct ErrorResponse { error: String }
#[derive(Serialize)]
struct HealthResponse { status: String, nllb_loaded: bool, labse_loaded: bool, sonar_loaded: bool }

struct AppState { translator: NllbTranslator, labse: LaBSEEncoder, sonar: SonarEncoder }

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        nllb_loaded: state.translator.is_model_downloaded(),
        labse_loaded: state.labse.is_model_downloaded(),
        sonar_loaded: state.sonar.is_model_downloaded(),
    })
}

async fn translate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TranslateRequest>,
) -> std::result::Result<Json<TranslateResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.translator.translate(&req.text, &req.source, &req.target).await {
        Ok(r) => Ok(Json(TranslateResponse {
            translation: r.text, source_lang: r.source_lang,
            target_lang: r.target_lang, duration_ms: r.duration_ms,
        })),
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e.to_string() }))),
    }
}

async fn score(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ScoreRequest>,
) -> std::result::Result<Json<ScoreResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    match state.labse.score(&req.text1, &req.text2).await {
        Ok(s) => Ok(Json(ScoreResponse { score: s, duration_ms: start.elapsed().as_millis() as u64 })),
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e.to_string() }))),
    }
}

async fn embed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> std::result::Result<Json<EmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.labse.embed(&req.text).await {
        Ok(emb) => {
            let dims = emb.len();
            Ok(Json(EmbedResponse { embedding: emb, dimensions: dims }))
        }
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e.to_string() }))),
    }
}

async fn embed_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedBatchRequest>,
) -> std::result::Result<Json<EmbedBatchResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.texts.is_empty() {
        return Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: "texts array must not be empty".into() })));
    }
    let text_refs: Vec<&str> = req.texts.iter().map(|s| s.as_str()).collect();
    match state.labse.embed_batch(&text_refs).await {
        Ok(embeddings) => {
            let dims = embeddings.first().map_or(0, |e| e.len());
            let count = embeddings.len();
            Ok(Json(EmbedBatchResponse { embeddings, dimensions: dims, count }))
        }
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() }))),
    }
}

async fn analyze(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnalyzeRequest>,
) -> std::result::Result<Json<AnalyzeResponse>, (StatusCode, Json<ErrorResponse>)> {
    let start = std::time::Instant::now();
    let translation = state.translator.translate(&req.text, &req.source, &req.target).await
        .map_err(|e| (StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e.to_string() })))?;
    let labse_score = state.labse.score(&req.text, &translation.text).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;
    let sonar_score = state.sonar.score(&req.text, &translation.text).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;
    Ok(Json(AnalyzeResponse {
        source_text: req.text,
        translation: translation.text,
        source_lang: translation.source_lang,
        target_lang: translation.target_lang,
        labse_score,
        sonar_score,
        duration_ms: start.elapsed().as_millis() as u64,
    }))
}

async fn analyze_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnalyzeBatchRequest>,
) -> std::result::Result<Json<AnalyzeBatchResponse>, (StatusCode, Json<ErrorResponse>)> {
    if req.texts.is_empty() {
        return Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: "texts array must not be empty".into() })));
    }
    let start = std::time::Instant::now();
    let mut results = Vec::with_capacity(req.texts.len());
    for text in &req.texts {
        let item_start = std::time::Instant::now();
        let translation = state.translator.translate(text, &req.source, &req.target).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;
        let labse_score = state.labse.score(text, &translation.text).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;
        let sonar_score = state.sonar.score(text, &translation.text).await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() })))?;
        results.push(AnalyzeResponse {
            source_text: text.clone(),
            translation: translation.text,
            source_lang: translation.source_lang,
            target_lang: translation.target_lang,
            labse_score,
            sonar_score,
            duration_ms: item_start.elapsed().as_millis() as u64,
        });
    }
    let count = results.len();
    Ok(Json(AnalyzeBatchResponse {
        results,
        count,
        total_duration_ms: start.elapsed().as_millis() as u64,
    }))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("lingo=info,tower_http=debug").init();

    let translator = NllbTranslator::new(None)?;
    let labse = LaBSEEncoder::new(None)?;
    let sonar = SonarEncoder::new(None)?;

    println!("Loading models...");
    if translator.is_model_downloaded() { translator.load().await?; println!("  NLLB-200: loaded"); }
    else { println!("  NLLB-200: not found (translation disabled)"); }
    if labse.is_model_downloaded() { labse.load().await?; println!("  LaBSE: loaded"); }
    else { println!("  LaBSE: not found (scoring disabled)"); }
    if sonar.is_model_downloaded() { sonar.load().await?; println!("  SONAR: loaded"); }
    else { println!("  SONAR: not found (scoring disabled)"); }

    let state = Arc::new(AppState { translator, labse, sonar });
    let app = Router::new()
        .route("/health", get(health))
        .route("/translate", post(translate))
        .route("/score", post(score))
        .route("/embed", post(embed))
        .route("/embed_batch", post(embed_batch))
        .route("/analyze", post(analyze))
        .route("/analyze_batch", post(analyze_batch))
        .with_state(state);

    println!("\nServer on http://localhost:3000");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
