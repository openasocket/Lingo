//! HTTP server exposing NLLB translation and LaBSE similarity as REST endpoints.
//!
//! ```bash
//! cargo run --example server --features server,metal
//! ```
//!
//! Endpoints:
//!   POST /translate  — translate text
//!   POST /score      — score similarity between two texts
//!   POST /embed      — get 768-dim embedding for text
//!   GET  /health     — server health check

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use lingo::{LaBSEEncoder, NllbTranslator};
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

#[derive(Serialize)]
struct ErrorResponse { error: String }
#[derive(Serialize)]
struct HealthResponse { status: String, nllb_loaded: bool, labse_loaded: bool }

struct AppState { translator: NllbTranslator, encoder: LaBSEEncoder }

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
        nllb_loaded: state.translator.is_model_downloaded(),
        labse_loaded: state.encoder.is_model_downloaded(),
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
    match state.encoder.score(&req.text1, &req.text2).await {
        Ok(s) => Ok(Json(ScoreResponse { score: s, duration_ms: start.elapsed().as_millis() as u64 })),
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e.to_string() }))),
    }
}

async fn embed(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> std::result::Result<Json<EmbedResponse>, (StatusCode, Json<ErrorResponse>)> {
    match state.encoder.embed(&req.text).await {
        Ok(emb) => {
            let dims = emb.len();
            Ok(Json(EmbedResponse { embedding: emb, dimensions: dims }))
        }
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: e.to_string() }))),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_env_filter("lingo=info,tower_http=debug").init();

    let translator = NllbTranslator::new(None)?;
    let encoder = LaBSEEncoder::new(None)?;

    println!("Loading models...");
    if translator.is_model_downloaded() { translator.load().await?; println!("  NLLB-200: loaded"); }
    else { println!("  NLLB-200: not found (translation disabled)"); }
    if encoder.is_model_downloaded() { encoder.load().await?; println!("  LaBSE: loaded"); }
    else { println!("  LaBSE: not found (scoring disabled)"); }

    let state = Arc::new(AppState { translator, encoder });
    let app = Router::new()
        .route("/health", get(health))
        .route("/translate", post(translate))
        .route("/score", post(score))
        .route("/embed", post(embed))
        .with_state(state);

    println!("\nServer on http://localhost:3000");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
