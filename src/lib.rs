//! # lingo
//!
//! Pure Rust multilingual NLP toolkit with Metal/CUDA GPU acceleration via
//! [candle](https://github.com/huggingface/candle).
//!
//! Translates text using the NLLB-200 model and performs similarity calculations
//! on translated text against pre-translated text using LaBSE and SONAR.
//!
//! ## Capabilities
//!
//! - **Translation**: NLLB-200 model — translate between 200+ languages
//! - **Similarity**: LaBSE model — cross-lingual semantic similarity scoring (768-dim)
//! - **Embeddings**: SONAR model — cross-lingual sentence embeddings (1024-dim)
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use lingo::{NllbTranslator, LaBSEEncoder};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Translate
//!     let translator = NllbTranslator::new(None)?;
//!     let result = translator.translate("Hello", "en", "fr").await?;
//!     println!("{}", result.text); // "Bonjour"
//!
//!     // Score similarity
//!     let encoder = LaBSEEncoder::new(None)?;
//!     let score = encoder.score("Hello", "Bonjour").await?;
//!     println!("Similarity: {:.3}", score); // ~0.85
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Runtime requirements
//!
//! Inference runs on tokio's blocking thread pool (`spawn_blocking`), so all
//! async methods must be called from within a tokio runtime.

mod cache;
pub mod chunker;
pub mod error;
pub mod labse;
pub mod languages;
pub mod license;
mod model;
pub mod sonar;
mod windows;

#[cfg(feature = "download")]
pub mod download;

pub use chunker::{chunk_text, sentence_units, Chunk, ChunkerConfig};
pub use error::{DegenerateKind, Error, Result};
pub use labse::LaBSEEncoder;
pub use languages::NllbLanguage;
pub use model::{DecoderCache, LayerCache};
pub use license::{check_license_acceptance, license_notice, mark_license_accepted};
pub use sonar::SonarEncoder;

use cache::BoundedCache;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use model::{NllbConfig, NllbModel, StopReason};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::info;

/// Max entries in the translation memo cache (full inputs and chunks).
const TRANSLATION_CACHE_MAX: usize = 1024;
/// Inputs larger than this are not memoized as a whole (their chunks still
/// are), so very large requests cannot pin unbounded memory in the cache.
const TRANSLATION_CACHE_MAX_KEY_BYTES: usize = 16 * 1024;

/// Maximum input tokens fed to the encoder in a single pass.
///
/// NLLB's `max_position_embeddings` is 1024 on both encoder and decoder. The
/// encoder side has headroom at 960 input tokens, but the *decoder* expands
/// output by up to ~2.5x for verbose target languages (German compounds,
/// Japanese kana, Spanish articles), so 960 input can produce ~2400 output
/// tokens — well past the decoder's 1024 ceiling, causing silent truncation.
///
/// Set to 400 to match `ChunkerConfig::default().max_tokens`: 400 * 2.5 = 1000
/// output tokens, leaving headroom under the decoder cap. Inputs that
/// tokenize beyond this are split by the chunker and translated piecewise.
const MAX_INPUT_TOKENS: usize = 400;

/// Hard ceiling on generated tokens per chunk: the decoder shares the
/// 1026-row sinusoidal table (offset 2), so 1024 is the real limit.
const MAX_OUTPUT_TOKENS: usize = 1024;

/// Per-chunk output budget: generous relative to the ~2.5x worst-case
/// expansion, but stops a degenerate repetition loop on a short input from
/// burning the full 1024-token budget.
fn output_budget(input_tokens: usize) -> usize {
    (input_tokens * 3 + 16).min(MAX_OUTPUT_TOKENS)
}

/// Default model directory: `~/.cache/lingo/<name>`.
///
/// Returns an error (rather than falling back to a shared directory such as
/// `/tmp`) when the home directory cannot be determined.
pub fn default_model_dir(name: &str) -> Result<PathBuf> {
    dirs::home_dir()
        .map(|h| h.join(".cache/lingo").join(name))
        .ok_or_else(|| {
            Error::ModelNotFound(
                "cannot determine home directory; pass an explicit model directory".into(),
            )
        })
}

/// Pick the best available device for the enabled accelerator features.
pub(crate) fn select_device(tag: &str) -> Device {
    #[cfg(feature = "metal")]
    if cfg!(target_os = "macos") {
        if let Ok(device) = Device::new_metal(0) {
            info!("{tag}: Using Metal GPU");
            return device;
        }
    }
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::new_cuda(0) {
            info!("{tag}: Using CUDA GPU");
            return device;
        }
    }
    info!("{tag}: Using CPU");
    Device::Cpu
}

/// Map a `spawn_blocking` join failure (panic or runtime shutdown) to an error.
pub(crate) fn join_error(what: &str, e: tokio::task::JoinError) -> Error {
    Error::Inference(format!("{what} worker failed: {e}"))
}

/// Cooperative cancellation shared between an awaiting future and the
/// blocking job it spawned. Set when the future is dropped before the job
/// finishes (a timeout, a `select!`, a dropped task); the job checks it
/// between tokens / units / windows and stops, releasing the model lock.
#[derive(Clone, Default)]
pub(crate) struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub(crate) fn is_set(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub(crate) fn check(&self) -> Result<()> {
        if self.is_set() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// Sets the flag when dropped, unless the job completed first.
struct CancelOnDrop {
    flag: CancelFlag,
    armed: bool,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.flag.0.store(true, Ordering::Relaxed);
        }
    }
}

/// Run `job` on the blocking pool. If the returned future is dropped before
/// the job finishes, the job's [`CancelFlag`] is set so it can stop early.
pub(crate) async fn run_blocking<T: Send + 'static>(
    what: &'static str,
    job: impl FnOnce(&CancelFlag) -> Result<T> + Send + 'static,
) -> Result<T> {
    let flag = CancelFlag::default();
    let mut guard = CancelOnDrop { flag: flag.clone(), armed: true };
    let result = tokio::task::spawn_blocking(move || job(&flag)).await;
    guard.armed = false;
    result.map_err(|e| join_error(what, e))?
}

/// Wait up to `timeout` for `model` to be free. True means no work holds the
/// instance (it is safe to reuse); false means a job is still running — most
/// likely wedged inside a GPU call — and the instance should be replaced.
pub(crate) async fn wait_unlocked<T>(model: &Mutex<T>, timeout: std::time::Duration) -> bool {
    tokio::time::timeout(timeout, model.lock()).await.is_ok()
}

/// Await `fut`, returning [`Error::Timeout`] if it exceeds `timeout`.
pub(crate) async fn bounded<T>(
    timeout: std::time::Duration,
    fut: impl std::future::Future<Output = Result<T>>,
) -> Result<T> {
    match tokio::time::timeout(timeout, fut).await {
        Ok(result) => result,
        Err(_) => Err(Error::Timeout(timeout)),
    }
}

/// Result of a translation operation.
///
/// A `chunk_count` greater than 1 means the input exceeded the NLLB
/// position-embedding budget and was split into multiple chunks that were
/// translated independently and rejoined with their original separators.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TranslationResult {
    /// Translated text
    pub text: String,
    /// Source language ISO code
    pub source_lang: String,
    /// Target language ISO code
    pub target_lang: String,
    /// Translation time in milliseconds
    pub duration_ms: u64,
    /// Number of chunks the input was split into. 1 when the input fit within
    /// the NLLB position-embedding budget; >1 means chunking was applied and
    /// the per-chunk translations were rejoined.
    pub chunk_count: usize,
}

/// How a translator splits input and which output-integrity checks it runs.
///
/// The default favors throughput: sentences are packed into chunks of up to
/// 400 tokens and output is returned as generated. [`TranslateConfig::strict`]
/// favors integrity: every sentence is translated on its own (NLLB is a
/// sentence-level model, and a multi-sentence chunk can hit EOS early and
/// silently drop later sentences), and known-bad output is rejected with
/// [`Error::Degenerate`] instead of being returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TranslateConfig {
    /// Translate one sentence (or line) at a time instead of packing
    /// sentences into token-budgeted chunks.
    pub per_sentence: bool,
    /// Reject an empty translation of a non-empty source.
    pub reject_empty: bool,
    /// Reject a translation that ends in an open clause (`,` `:` `;` and CJK /
    /// Arabic equivalents) when the source sentence was complete.
    pub reject_dangling_clause: bool,
    /// Treat a repeated-token loop as an error. When `false`, the loop is
    /// trimmed to one repetition and the translation is returned.
    pub reject_repetition: bool,
}

impl TranslateConfig {
    /// Sentence-at-a-time translation with every integrity check enabled.
    pub fn strict() -> Self {
        Self {
            per_sentence: true,
            reject_empty: true,
            reject_dangling_clause: true,
            reject_repetition: true,
        }
    }
}

/// True when a complete source sentence came back as an unfinished clause —
/// the visible signature of an early EOS.
fn has_dangling_clause(source: &str, translated: &str, target: NllbLanguage) -> bool {
    let (source, translated) = (source.trim_end(), translated.trim_end());
    // Greek writes the question mark as `;` (or U+037E GREEK QUESTION MARK).
    if target == NllbLanguage::Greek && source.ends_with('?') && translated.ends_with([';', '\u{037E}']) {
        return false;
    }
    source.ends_with(['.', '!', '?', '。', '！', '？'])
        && translated.ends_with([',', '，', ':', '：', ';', '；', '؛', '\u{037E}'])
}

/// Cached translation: (text, chunk_count).
type TranslationCache = BoundedCache<String, (String, usize)>;

/// NLLB-200 translator using candle with Metal/CUDA/CPU inference.
///
/// Auto-selects the best available device:
/// - macOS: Metal GPU (with `metal` feature)
/// - Linux: CUDA GPU (with `cuda` feature)
/// - Fallback: CPU
pub struct NllbTranslator {
    model_dir: PathBuf,
    model: Arc<Mutex<Option<NllbModel>>>,
    tokenizer: Arc<Mutex<Option<tokenizers::Tokenizer>>>,
    cache: Arc<TranslationCache>,
    config: RwLock<TranslateConfig>,
    loaded: AtomicBool,
    device: Device,
}

impl NllbTranslator {
    /// Create a new translator.
    ///
    /// - `model_dir`: Path containing `model.safetensors` and `tokenizer.json`.
    ///   Defaults to `~/.cache/lingo/nllb-200-distilled-600M`.
    pub fn new(model_dir: Option<PathBuf>) -> Result<Self> {
        license::require_license_acceptance()?;
        let model_dir = match model_dir {
            Some(d) => d,
            None => default_model_dir("nllb-200-distilled-600M")?,
        };
        Ok(Self::build(model_dir, select_device("NLLB")))
    }

    /// Create a translator with an explicit device.
    pub fn with_device(model_dir: PathBuf, device: Device) -> Result<Self> {
        license::require_license_acceptance()?;
        Ok(Self::build(model_dir, device))
    }

    fn build(model_dir: PathBuf, device: Device) -> Self {
        Self {
            model_dir,
            model: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            cache: Arc::new(BoundedCache::new(
                TRANSLATION_CACHE_MAX,
                TRANSLATION_CACHE_MAX_KEY_BYTES,
            )),
            config: RwLock::new(TranslateConfig::default()),
            loaded: AtomicBool::new(false),
            device,
        }
    }

    /// Builder-style variant of [`set_config`](Self::set_config).
    pub fn with_config(self, config: TranslateConfig) -> Self {
        self.set_config(config);
        self
    }

    /// Change how subsequent translations are split and checked.
    ///
    /// The translation cache is partitioned by config, so results produced
    /// under one config are never served under another.
    pub fn set_config(&self, config: TranslateConfig) {
        *self.config.write().unwrap_or_else(|e| e.into_inner()) = config;
    }

    /// The current [`TranslateConfig`].
    pub fn config(&self) -> TranslateConfig {
        *self.config.read().unwrap_or_else(|e| e.into_inner())
    }

    pub fn model_dir(&self) -> &Path { &self.model_dir }
    pub fn device(&self) -> &Device { &self.device }

    /// True when the model files are present on disk.
    pub fn is_model_downloaded(&self) -> bool {
        self.model_dir.join("model.safetensors").exists()
            && self.model_dir.join("tokenizer.json").exists()
    }

    /// True once [`load`](Self::load) has completed successfully.
    pub fn is_loaded(&self) -> bool {
        self.loaded.load(Ordering::Acquire)
    }

    /// After an [`Error::Timeout`](crate::Error::Timeout), wait up to `grace`
    /// for the abandoned work to stop. It checks for cancellation between
    /// decode steps / units / windows, so it normally stops within one step.
    ///
    /// Returns `true` when the instance is idle and safe to keep using.
    /// Returns `false` when work still holds the model after `grace`, which
    /// means it is stuck inside a GPU call (the candle Metal
    /// `wait_until_completed` hang): drop this instance and build a new one.
    pub async fn wait_idle(&self, grace: std::time::Duration) -> bool {
        crate::wait_unlocked(&self.model, grace).await
    }

    /// Download NLLB-200-distilled-600M from HuggingFace Hub to the model directory.
    ///
    /// Downloads `model.safetensors` (~1.2 GB) and `tokenizer.json` from a
    /// pinned revision. Skips files that already exist. Requires the
    /// `download` feature.
    #[cfg(feature = "download")]
    pub fn download_model(&self) -> Result<()> {
        download::download_nllb(&self.model_dir)
    }

    /// Load the model and tokenizer. Called automatically on first translate.
    pub async fn load(&self) -> Result<()> {
        let mut model_guard = self.model.lock().await;
        if model_guard.is_some() {
            return Ok(());
        }
        if !self.model_dir.exists() {
            return Err(Error::ModelNotFound(self.model_dir.display().to_string()));
        }

        let tokenizer_path = self.model_dir.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        if tokenizer.token_to_id("</s>").is_none() || tokenizer.token_to_id("eng_Latn").is_none() {
            return Err(Error::Tokenizer(
                "tokenizer.json is not an NLLB-200 tokenizer (missing </s> or language tokens)".into(),
            ));
        }

        let weight_path = self.model_dir.join("model.safetensors");
        if !weight_path.exists() {
            return Err(Error::ModelNotFound("model.safetensors not found".into()));
        }

        let config = NllbConfig::distilled_600m();

        // Metal has incomplete F16 kernel coverage — use F32 there.
        // CUDA can use F16 for lower memory usage; CPU stays F32.
        let dtype = if self.device.is_cuda() { DType::F16 } else { DType::F32 };
        // SAFETY: the safetensors file is memory-mapped read-only; it must not
        // be modified while the model is alive (same contract as candle).
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weight_path], dtype, &self.device)
                .map_err(|e| Error::Inference(format!("Failed to load weights: {}", e)))?
        };
        let m = NllbModel::load(vb, &config, &self.device)
            .map_err(|e| Error::Inference(format!("Failed to build model: {}", e)))?;

        info!(
            "NLLB-200 loaded on {:?} ({} encoder + {} decoder layers)",
            self.device, config.encoder_layers, config.decoder_layers
        );

        *model_guard = Some(m);
        *self.tokenizer.lock().await = Some(tokenizer);
        self.loaded.store(true, Ordering::Release);
        Ok(())
    }

    /// Translate text between languages using ISO 639-1/3 codes.
    ///
    /// Both `source_lang` and `target_lang` must be NLLB-supported languages,
    /// given as ISO 639-1/3 codes (`fr`, `fra`) or NLLB codes (`fra_Latn`);
    /// the source language is fed to the encoder as its language token.
    ///
    /// Splitting and output checks follow the translator's
    /// [`TranslateConfig`] (see [`set_config`](Self::set_config)).
    pub async fn translate(&self, text: &str, source_lang: &str, target_lang: &str) -> Result<TranslationResult> {
        self.translate_inner(text, source_lang, target_lang, false).await
    }

    /// Reference (pre-KV-cache) twin of [`translate`](Self::translate), kept for
    /// the numeric-equivalence integration test in `tests/kv_cache_equivalence.rs`.
    ///
    /// Identical to `translate` except it drives the full-prefix
    /// [`NllbModel::generate_reference`] instead of the KV-cached
    /// [`NllbModel::generate`]. It deliberately **bypasses the translation
    /// cache** in both directions: it never reads a value that `generate`
    /// produced (which would make the equivalence diff trivially pass) and never
    /// writes its own results back (which would poison the cache for production
    /// `translate` calls). Production code should always call `translate`.
    #[doc(hidden)]
    pub async fn translate_reference(&self, text: &str, source_lang: &str, target_lang: &str) -> Result<TranslationResult> {
        self.translate_inner(text, source_lang, target_lang, true).await
    }

    /// Shared body for [`translate`](Self::translate) and
    /// [`translate_reference`](Self::translate_reference).
    ///
    /// `reference == false` is the production path: KV-cached generation plus the
    /// bounded translation cache. `reference == true` selects the full-prefix
    /// `generate_reference` path and disables all cache reads/writes so the two
    /// paths stay independent for verification.
    async fn translate_inner(&self, text: &str, source_lang: &str, target_lang: &str, reference: bool) -> Result<TranslationResult> {
        let start = Instant::now();
        let use_cache = !reference;

        let src = resolve_language(source_lang, source_lang, target_lang)?;
        let tgt = resolve_language(target_lang, source_lang, target_lang)?;
        let config = self.config();

        let result = |text: String, duration_ms: u64, chunk_count: usize| TranslationResult {
            text,
            source_lang: source_lang.to_string(),
            target_lang: target_lang.to_string(),
            duration_ms,
            chunk_count,
        };

        if src == tgt {
            return Ok(result(text.to_string(), 0, 1));
        }

        let cache_key = translation_cache_key(text, src, tgt, &config);
        if use_cache {
            if let Some((cached, chunk_count)) = self.cache.get(&cache_key) {
                return Ok(result(cached, 0, chunk_count));
            }
        }

        self.load().await?;

        // Run the synchronous tokenizer + GPU work on the blocking pool so it
        // never stalls the async executor, and so `translate_with_timeout` can
        // actually fire while the work is in flight.
        let job = TranslateJob {
            model: Arc::clone(&self.model),
            tokenizer: Arc::clone(&self.tokenizer),
            cache: Arc::clone(&self.cache),
            device: self.device.clone(),
            text: text.to_string(),
            src,
            tgt,
            reference,
            config,
        };
        let (translated_text, chunk_count) = run_blocking("translation", move |cancel| job.run(cancel)).await?;

        // Cache the full-input result as well, so a repeat of the identical long
        // input is served without re-chunking. Skipped on the reference path.
        if use_cache {
            self.cache.insert(cache_key, (translated_text.clone(), chunk_count));
        }

        Ok(result(translated_text, start.elapsed().as_millis() as u64, chunk_count))
    }

    /// Same as [`translate`](Self::translate) but bounded by `timeout`.
    ///
    /// Returns [`Error::Timeout`] if the call does not complete within the
    /// budget. Inference runs on tokio's blocking pool, so the timeout fires
    /// even while the GPU call is stuck.
    ///
    /// On timeout the abandoned work is asked to stop: it checks between
    /// decode steps and sentence units, then releases the model. Call
    /// [`wait_idle`](Self::wait_idle) with a short grace period (a few
    /// seconds) before reusing the instance:
    /// - `true`: the work stopped; keep using this translator.
    /// - `false`: the thread is stuck inside a GPU call (the candle Metal
    ///   `wait_until_completed` hang) and holds the model lock forever. Drop
    ///   the translator and construct a new one. A tokio runtime waits for
    ///   running blocking tasks when dropped, so a process with a wedged
    ///   thread should exit with `std::process::exit` rather than by
    ///   returning from `main`.
    ///
    /// For long inputs that get chunked, the timeout applies to the *whole*
    /// translation including all chunk sub-calls. Size the budget
    /// accordingly: an N-chunk input may need ~N × per-chunk-budget.
    pub async fn translate_with_timeout(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
        timeout: std::time::Duration,
    ) -> Result<TranslationResult> {
        bounded(timeout, self.translate(text, source_lang, target_lang)).await
    }
}

/// Resolve an ISO or NLLB code to one of NLLB-200's languages.
fn resolve_language(code: &str, source: &str, target: &str) -> Result<NllbLanguage> {
    NllbLanguage::resolve(code).ok_or_else(|| Error::InvalidLanguagePair(source.into(), target.into()))
}

/// Cache key on canonical NLLB codes, so `en`/`eng` share entries. The unit
/// separator cannot be confused with ordinary text.
fn translation_cache_key(text: &str, src: NllbLanguage, tgt: NllbLanguage, config: &TranslateConfig) -> String {
    let mode = format!(
        "{}{}{}{}",
        config.per_sentence as u8,
        config.reject_empty as u8,
        config.reject_dangling_clause as u8,
        config.reject_repetition as u8
    );
    format!("{}\u{1f}{}\u{1f}{}\u{1f}{}", src.nllb_code(), tgt.nllb_code(), mode, text)
}

/// Token IDs that frame an NLLB encoder input and select the output language.
#[derive(Clone, Copy)]
struct FrameIds {
    target: NllbLanguage,
    src_lang: u32,
    tgt_lang: u32,
    eos: u32,
}

/// Owned state for one translation, run on the blocking pool.
struct TranslateJob {
    model: Arc<Mutex<Option<NllbModel>>>,
    tokenizer: Arc<Mutex<Option<tokenizers::Tokenizer>>>,
    cache: Arc<TranslationCache>,
    device: Device,
    text: String,
    src: NllbLanguage,
    tgt: NllbLanguage,
    reference: bool,
    config: TranslateConfig,
}

impl TranslateJob {
    fn run(self, cancel: &CancelFlag) -> Result<(String, usize)> {
        // Lock order (model, then tokenizer) matches `load`.
        let model_guard = self.model.blocking_lock();
        // The caller may have given up while this job queued for the lock.
        cancel.check()?;
        let model = model_guard.as_ref().ok_or_else(|| Error::Inference("Model not loaded".into()))?;
        let tok_guard = self.tokenizer.blocking_lock();
        let tokenizer = tok_guard.as_ref().ok_or_else(|| Error::Inference("Tokenizer not loaded".into()))?;

        let lang_id = |lang: NllbLanguage| {
            tokenizer
                .token_to_id(lang.nllb_code())
                .ok_or_else(|| Error::UnsupportedLanguage(lang.iso_code().to_string()))
        };
        let ids = FrameIds {
            target: self.tgt,
            src_lang: lang_id(self.src)?,
            tgt_lang: lang_id(self.tgt)?,
            eos: tokenizer
                .token_to_id("</s>")
                .ok_or_else(|| Error::Tokenizer("tokenizer has no </s> token".into()))?,
        };

        // Framed length = content tokens + source-language token + </s>.
        let framed_len = |s: &str| encode_content(tokenizer, s).map(|v| v.len() + 2);
        let translate = |text: &str| -> Result<String> {
            translate_chunk(tokenizer, model, &self.device, text, ids, ChunkOptions { reference: self.reference, config: &self.config, cancel })
        };

        let units: Vec<Chunk> = if self.config.per_sentence {
            // Sentence/line units; any unit still over the encoder budget is
            // split further by the token-budgeted chunker.
            let mut units = Vec::new();
            for unit in sentence_units(&self.text) {
                if framed_len(&unit.text)? <= MAX_INPUT_TOKENS {
                    units.push(unit);
                } else {
                    let mut subs = chunk_text(&unit.text, &ChunkerConfig::default(), |s| framed_len(s).unwrap_or(0));
                    if let Some(last) = subs.last_mut() {
                        last.trailing_separator.push_str(&unit.trailing_separator);
                    }
                    units.extend(subs);
                }
            }
            units
        } else if framed_len(&self.text)? <= MAX_INPUT_TOKENS {
            // Short-input fast path — single encode/generate/decode.
            return Ok((translate(&self.text)?, 1));
        } else {
            chunk_text(&self.text, &ChunkerConfig::default(), |s| framed_len(s).unwrap_or(0))
        };

        // Translate each unit against the already-held model lock and rejoin
        // with the original separators so nothing is lost. Whitespace-only
        // units are passed through untouched.
        let use_cache = !self.reference;
        let mut rejoined = String::new();
        let mut count = 0usize;
        for unit in &units {
            if unit.text.trim().is_empty() {
                rejoined.push_str(&unit.text);
                rejoined.push_str(&unit.trailing_separator);
                continue;
            }
            cancel.check()?;
            count += 1;
            // Per-unit cache: a long input frequently shares sentences with a
            // previously translated input. Disabled on the reference path so it
            // stays independent of the KV-cached path's results.
            let key = translation_cache_key(&unit.text, self.src, self.tgt, &self.config);
            let cached = if use_cache { self.cache.get(&key) } else { None };
            let translated = match cached {
                Some((t, _)) => t,
                None => {
                    let t = translate(&unit.text)?;
                    if use_cache {
                        self.cache.insert(key, (t.clone(), 1));
                    }
                    t
                }
            };
            rejoined.push_str(&translated);
            rejoined.push_str(&unit.trailing_separator);
        }
        Ok((rejoined, count.max(1)))
    }
}

/// Tokenize without the template's special tokens. The NLLB tokenizer's
/// built-in template hard-codes `eng_Latn` as the source language, so the
/// caller frames the content with the real source-language token instead.
fn encode_content(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer
        .encode(text, false)
        .map(|e| e.get_ids().to_vec())
        .map_err(|e| Error::Tokenizer(e.to_string()))
}

/// Per-call settings for [`translate_chunk`].
struct ChunkOptions<'a> {
    reference: bool,
    config: &'a TranslateConfig,
    cancel: &'a CancelFlag,
}

/// Encode/generate/decode a single chunk that is already known to fit within
/// the encoder's token budget. Returns just the cleaned translated string.
///
/// `reference == true` swaps the KV-cached `generate` for the full-prefix
/// `generate_reference` so the equivalence test can diff the two paths over
/// the identical encode + decode pipeline.
fn translate_chunk(
    tokenizer: &tokenizers::Tokenizer,
    model: &NllbModel,
    device: &Device,
    text: &str,
    ids: FrameIds,
    opts: ChunkOptions<'_>,
) -> Result<String> {
    let ChunkOptions { reference, config, cancel } = opts;
    let content = encode_content(tokenizer, text)?;
    let max_new = output_budget(content.len());

    // NLLB (non-legacy) source framing: [src_lang] tokens... [</s>]
    let mut input_ids = Vec::with_capacity(content.len() + 2);
    input_ids.push(ids.src_lang);
    input_ids.extend_from_slice(&content);
    input_ids.push(ids.eos);

    let seq_len = input_ids.len();
    let input_tensor = Tensor::from_vec(input_ids, (1, seq_len), device)
        .map_err(|e| Error::Inference(e.to_string()))?;

    let encoder_output = model.encode(&input_tensor)
        .map_err(|e| Error::Inference(format!("Encoding failed: {}", e)))?;
    let (output_ids, stop) = if reference {
        model.generate_reference(&encoder_output, ids.tgt_lang, max_new, &|| cancel.is_set())
    } else {
        model.generate(&encoder_output, ids.tgt_lang, max_new, &|| cancel.is_set())
    }
    .map_err(|e| Error::Inference(format!("Generation failed: {}", e)))?;

    if stop == StopReason::Cancelled {
        return Err(Error::Cancelled);
    }
    if stop == StopReason::Repetition && config.reject_repetition {
        return Err(Error::Degenerate(DegenerateKind::RepeatedTokens));
    }

    let translated_text = tokenizer.decode(&output_ids, true)
        .map_err(|e| Error::Tokenizer(e.to_string()))?
        .replace("<unk>", "").replace("  ", " ").trim().to_string();

    if config.reject_empty && translated_text.is_empty() && !text.trim().is_empty() {
        return Err(Error::Degenerate(DegenerateKind::EmptyOutput));
    }
    if config.reject_dangling_clause && has_dangling_clause(text, &translated_text, ids.target) {
        return Err(Error::Degenerate(DegenerateKind::DanglingClause));
    }
    Ok(translated_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translation_result_chunk_count_defaults_to_one() {
        // The no-chunk fast path constructs results with chunk_count == 1.
        let result = TranslationResult {
            text: "Bonjour".to_string(),
            source_lang: "en".to_string(),
            target_lang: "fr".to_string(),
            duration_ms: 0,
            chunk_count: 1,
        };
        assert_eq!(result.chunk_count, 1);
    }

    #[test]
    fn output_budget_scales_and_caps() {
        assert_eq!(output_budget(10), 46);
        assert_eq!(output_budget(400), 1024);
    }

    #[test]
    fn cache_key_normalizes_iso_aliases_and_separates_fields() {
        let en = NllbLanguage::from_iso_code("en").unwrap();
        let eng = NllbLanguage::from_iso_code("eng").unwrap();
        let fr = NllbLanguage::from_iso_code("fr").unwrap();
        let d = TranslateConfig::default();
        assert_eq!(translation_cache_key("hi", en, fr, &d), translation_cache_key("hi", eng, fr, &d));
        assert_ne!(translation_cache_key("a-b", en, fr, &d), translation_cache_key("a", en, fr, &d));
        assert_ne!(
            translation_cache_key("hi", en, fr, &d),
            translation_cache_key("hi", en, fr, &TranslateConfig::strict())
        );
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap()
    }

    #[test]
    fn dropped_future_cancels_blocking_job_and_frees_the_lock() {
        rt().block_on(async {
            let model = Arc::new(Mutex::new(()));
            let stopped = Arc::new(AtomicBool::new(false));
            let (m, st) = (Arc::clone(&model), Arc::clone(&stopped));
            let job = run_blocking("test", move |cancel| {
                let _guard = m.blocking_lock();
                let deadline = Instant::now() + std::time::Duration::from_secs(10);
                while Instant::now() < deadline {
                    if cancel.check().is_err() {
                        st.store(true, Ordering::SeqCst);
                        return Err(Error::Cancelled);
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Ok(())
            });
            let r = bounded(std::time::Duration::from_millis(30), job).await;
            assert!(matches!(r, Err(Error::Timeout(_))));
            assert!(wait_unlocked(&model, std::time::Duration::from_secs(5)).await);
            assert!(stopped.load(Ordering::SeqCst));
        });
    }

    #[test]
    fn completed_job_is_not_cancelled() {
        rt().block_on(async {
            let r = run_blocking("test", |cancel| Ok(cancel.is_set())).await.unwrap();
            assert!(!r);
        });
    }

    #[test]
    fn wedged_lock_is_reported_busy() {
        rt().block_on(async {
            let model = Mutex::new(());
            let _held = model.lock().await;
            assert!(!wait_unlocked(&model, std::time::Duration::from_millis(20)).await);
        });
    }

    #[test]
    fn degenerate_kind_wording_is_stable() {
        let e = Error::Degenerate(DegenerateKind::DanglingClause);
        assert_eq!(
            e.to_string(),
            "Translation rejected: a complete source sentence was translated as an unfinished clause"
        );
        assert_eq!(DegenerateKind::RepeatedTokens.to_string(), "output entered a repeated-token loop");
        assert_eq!(DegenerateKind::EmptyOutput.to_string(), "empty translation for a non-empty source");
    }

    #[test]
    fn dangling_clause_detection() {
        use NllbLanguage::{French, German, Greek};
        assert!(has_dangling_clause("It is done.", "C'est fait,", French));
        assert!(!has_dangling_clause("It is done.", "C'est fait.", French));
        assert!(!has_dangling_clause("Note:", "Remarque :", French));
        // Greek question mark is `;` / U+037E — only for a question into Greek.
        assert!(!has_dangling_clause("Is that okay?", "Είναι εντάξει;", Greek));
        assert!(!has_dangling_clause("Is that okay?", "Είναι εντάξει\u{037E}", Greek));
        assert!(has_dangling_clause("It is done.", "Έγινε;", Greek));
        assert!(has_dangling_clause("Is that okay?", "Ist das okay;", German));
    }

    #[test]
    fn nllb_codes_resolve_and_uncovered_languages_are_rejected() {
        assert_eq!(resolve_language("fra_Latn", "fra_Latn", "en").unwrap().iso_code(), "fr");
        // Bavarian has no NLLB-200 token, so it is not a known language.
        assert!(matches!(resolve_language("bar", "bar", "en"), Err(Error::InvalidLanguagePair(_, _))));
        assert!(resolve_language("en", "en", "fr").is_ok());
    }

    #[test]
    fn unknown_source_language_is_rejected() {
        assert!(matches!(
            resolve_language("zz", "zz", "fr"),
            Err(Error::InvalidLanguagePair(_, _))
        ));
    }
}
