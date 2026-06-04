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

pub mod chunker;
pub mod error;
pub mod labse;
pub mod languages;
pub mod license;
mod model;
pub mod sonar;

#[cfg(feature = "download")]
pub mod download;

pub use chunker::{chunk_text, Chunk, ChunkerConfig};
pub use error::{Error, Result};
pub use labse::LaBSEEncoder;
pub use languages::NllbLanguage;
pub use license::{check_license_acceptance, license_notice, mark_license_accepted};
pub use sonar::SonarEncoder;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use model::{NllbConfig, NllbModel};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::info;

const TRANSLATION_CACHE_MAX: usize = 1024;

/// Maximum input tokens fed to the encoder in a single pass. NLLB's
/// `max_position_embeddings` is 1024; 960 leaves headroom for the BOS/EOS and
/// language-tag overhead while staying safely under that hard limit. Inputs
/// that tokenize beyond this are split by the chunker and translated piecewise.
const MAX_INPUT_TOKENS: usize = 960;

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
    cache: Arc<Mutex<(HashMap<String, String>, VecDeque<String>)>>,
    device: Device,
}

impl NllbTranslator {
    /// Create a new translator.
    ///
    /// - `model_dir`: Path containing `model.safetensors` and `tokenizer.json`.
    ///   Defaults to `~/.cache/lingo/nllb-200-distilled-600M`.
    pub fn new(model_dir: Option<PathBuf>) -> Result<Self> {
        license::require_license_acceptance()?;
        let model_dir = model_dir.unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join(".cache/lingo/nllb-200-distilled-600M")
        });
        let device = Self::select_device();
        Ok(Self {
            model_dir,
            model: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            cache: Arc::new(Mutex::new((HashMap::new(), VecDeque::new()))),
            device,
        })
    }

    /// Create a translator with an explicit device.
    pub fn with_device(model_dir: PathBuf, device: Device) -> Result<Self> {
        Ok(Self {
            model_dir,
            model: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            cache: Arc::new(Mutex::new((HashMap::new(), VecDeque::new()))),
            device,
        })
    }

    fn select_device() -> Device {
        #[cfg(feature = "metal")]
        if cfg!(target_os = "macos") {
            if let Ok(device) = Device::new_metal(0) {
                info!("NLLB: Using Metal GPU");
                return device;
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                info!("NLLB: Using CUDA GPU");
                return device;
            }
        }
        info!("NLLB: Using CPU");
        Device::Cpu
    }

    pub fn model_dir(&self) -> &Path { &self.model_dir }
    pub fn device(&self) -> &Device { &self.device }

    pub fn is_model_downloaded(&self) -> bool {
        self.model_dir.join("model.safetensors").exists()
            && self.model_dir.join("tokenizer.json").exists()
    }

    /// Download NLLB-200-distilled-600M from HuggingFace Hub to the model directory.
    ///
    /// Downloads `model.safetensors` (~1.2 GB) and `tokenizer.json`.
    /// Skips files that already exist. Requires the `download` feature.
    #[cfg(feature = "download")]
    pub fn download_model(&self) -> Result<()> {
        download::download_model(&self.model_dir)
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

        let weight_path = self.model_dir.join("model.safetensors");
        if !weight_path.exists() {
            return Err(Error::ModelNotFound("model.safetensors not found".into()));
        }

        let config = NllbConfig::distilled_600m();

        // Metal has incomplete F16 kernel coverage — use F32 there.
        // CUDA and CPU can use F16 for lower memory usage.
        let dtype = match &self.device {
            Device::Cpu => DType::F32,
            _ => {
                if cfg!(feature = "metal") {
                    DType::F32
                } else {
                    DType::F16
                }
            }
        };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weight_path], dtype, &self.device)
                .map_err(|e| Error::Inference(format!("Failed to load weights: {}", e)))?
        };
        let m = NllbModel::load(vb, &config, &self.device)
            .map_err(|e| Error::Inference(format!("Failed to build model: {}", e)))?;

        let device_name = format!("{:?}", self.device);
        info!(
            "NLLB-200 loaded on {} ({} encoder + {} decoder layers)",
            device_name, config.encoder_layers, config.decoder_layers
        );

        *model_guard = Some(m);
        *self.tokenizer.lock().await = Some(tokenizer);
        Ok(())
    }

    /// Translate text between languages using ISO 639-1/3 codes.
    pub async fn translate(&self, text: &str, source_lang: &str, target_lang: &str) -> Result<TranslationResult> {
        let start = Instant::now();

        if source_lang == target_lang {
            return Ok(TranslationResult {
                text: text.to_string(),
                source_lang: source_lang.to_string(),
                target_lang: target_lang.to_string(),
                duration_ms: 0,
                chunk_count: 1,
            });
        }

        let cache_key = format!("{}-{}-{}", text, source_lang, target_lang);
        if let Some(cached) = self.cache.lock().await.0.get(&cache_key) {
            return Ok(TranslationResult {
                text: cached.clone(),
                source_lang: source_lang.to_string(),
                target_lang: target_lang.to_string(),
                duration_ms: 0,
                chunk_count: 1,
            });
        }

        let target_nllb = NllbLanguage::from_iso_code(target_lang)
            .ok_or_else(|| Error::InvalidLanguagePair(source_lang.into(), target_lang.into()))?;

        if !target_nllb.is_nllb_supported() {
            return Err(Error::UnsupportedLanguage(target_lang.to_string()));
        }

        self.load().await?;

        let tok_guard = self.tokenizer.lock().await;
        let tokenizer = tok_guard.as_ref().ok_or_else(|| Error::Inference("Tokenizer not loaded".into()))?;

        let forced_bos_id = tokenizer.token_to_id(target_nllb.nllb_code())
            .ok_or_else(|| Error::InvalidLanguagePair(source_lang.into(), target_lang.into()))?;

        let model_guard = self.model.lock().await;
        let model = model_guard.as_ref().ok_or_else(|| Error::Inference("Model not loaded".into()))?;

        // Tokenize the full input once to decide whether it fits the encoder's
        // position-embedding budget.
        let full_token_count = tokenizer.encode(text, true)
            .map(|e| e.get_ids().len())
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let (translated_text, chunk_count) = if full_token_count <= MAX_INPUT_TOKENS {
            // Short-input fast path — single encode/generate/decode, no behavior
            // change versus the pre-chunking implementation.
            let text_out = self.translate_one_chunk(tokenizer, model, text, forced_bos_id)?;
            (text_out, 1)
        } else {
            // Long input: split into token-budgeted chunks, translate each
            // against the already-held model lock, and rejoin with the original
            // separators so nothing is lost.
            let chunks = chunk_text(text, &ChunkerConfig::default(), |s| {
                tokenizer.encode(s, true).map(|e| e.get_ids().len()).unwrap_or(0)
            });
            let mut rejoined = String::new();
            for chunk in &chunks {
                // Per-chunk cache: a long input frequently shares sentences with
                // a previously translated (possibly shorter) input. Reuses the
                // same cache map — the key granularity is implicit in the key.
                let chunk_key = format!("{}-{}-{}", chunk.text, source_lang, target_lang);
                let chunk_translation = if let Some(cached) = self.cache.lock().await.0.get(&chunk_key) {
                    cached.clone()
                } else {
                    let t = self.translate_one_chunk(tokenizer, model, &chunk.text, forced_bos_id)?;
                    self.cache_insert(chunk_key, t.clone()).await;
                    t
                };
                rejoined.push_str(&chunk_translation);
                rejoined.push_str(&chunk.trailing_separator);
            }
            (rejoined, chunks.len())
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        // Cache the full-input result as well, so a repeat of the identical long
        // input is served without re-chunking.
        self.cache_insert(cache_key, translated_text.clone()).await;

        Ok(TranslationResult {
            text: translated_text,
            source_lang: source_lang.to_string(),
            target_lang: target_lang.to_string(),
            duration_ms,
            chunk_count,
        })
    }

    /// Encode/generate/decode a single chunk that is already known to fit within
    /// the encoder's token budget. Returns just the cleaned translated string.
    ///
    /// The tokenizer and model are passed in as already-locked references so the
    /// chunked path can process every chunk under a single lock acquisition,
    /// avoiding lock-thrash (the model is serialized regardless).
    fn translate_one_chunk(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        model: &NllbModel,
        text: &str,
        forced_bos_id: u32,
    ) -> Result<String> {
        let encoding = tokenizer.encode(text, true).map_err(|e| Error::Tokenizer(e.to_string()))?;
        let input_ids = encoding.get_ids();
        let input_tensor = Tensor::from_vec(input_ids.to_vec(), (1, input_ids.len()), &self.device)
            .map_err(|e| Error::Inference(e.to_string()))?;

        let encoder_output = model.encode(&input_tensor)
            .map_err(|e| Error::Inference(format!("Encoding failed: {}", e)))?;
        // 1024 is the real ceiling: the decoder shares the 1026-row pos_embeddings
        // tensor (offset 2 -> max output seq_len 1024); a higher cap is a phantom.
        let output_ids = model.generate(&encoder_output, forced_bos_id, 1024)
            .map_err(|e| Error::Inference(format!("Generation failed: {}", e)))?;

        let translated_text = tokenizer.decode(&output_ids, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?
            .replace("<unk>", "").replace("  ", " ").trim().to_string();
        Ok(translated_text)
    }

    /// Insert a translation into the bounded LRU cache. No-op if the key is
    /// already present (keeps the eviction order free of duplicates).
    async fn cache_insert(&self, key: String, value: String) {
        let mut cache = self.cache.lock().await;
        let (map, order) = &mut *cache;
        if map.contains_key(&key) {
            return;
        }
        if map.len() >= TRANSLATION_CACHE_MAX {
            if let Some(old_key) = order.pop_front() {
                map.remove(&old_key);
            }
        }
        map.insert(key.clone(), value);
        order.push_back(key);
    }
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
}
