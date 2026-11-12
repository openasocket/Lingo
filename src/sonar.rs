//! SONAR text encoder — cross-lingual sentence embeddings (1024-dim).
//!
//! Meta's SONAR encodes text into language-agnostic embeddings for
//! cross-lingual similarity scoring. Architecture: 24-layer transformer
//! encoder with 1024 hidden dim, 8192 FFN dim, 16 attention heads.
//!
//! Output embeddings are L2-normalized for cosine similarity.
//!
//! ## Tokenization
//!
//! SONAR uses fairseq2's NLLB vocabulary: the NLLB SentencePiece model with
//! `<pad>` inserted at index 0 (so `<pad>=0, <unk>=1, <s>=2, </s>=3`, every
//! piece at `spm_id + 1`) and the 202 `__lang__` control symbols appended from
//! 256001 — 256206 rows in total. The HuggingFace NLLB `tokenizer.json` uses
//! the same piece and language-token IDs and differs only in the order of the
//! four specials, so lingo loads that file and remaps the specials. Source
//! sequences are framed as `[lang] tokens... [</s>]`, as in fairseq2's
//! `NllbTokenizer` "source" mode.

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::cache::BoundedCache;
use crate::error::{Error, Result};
use crate::languages::NllbLanguage;

// ---------------------------------------------------------------------------
// SONAR config
// ---------------------------------------------------------------------------

/// Max sequence length (including the language token and `</s>`), matching
/// fairseq2's SONAR text encoder config.
const SONAR_MAX_SEQ_LEN: usize = 512;
/// fairseq2 vocabulary specials.
const SONAR_EOS_ID: u32 = 3;
/// Max entries in the embedding memo cache.
const EMBED_CACHE_MAX: usize = 4096;
/// Inputs larger than this are not memoized.
const EMBED_CACHE_MAX_KEY_BYTES: usize = 8 * 1024;
/// Language used by the language-less convenience methods.
const DEFAULT_LANG: &str = "en";

struct SonarConfig {
    vocab_size: usize,
    d_model: usize,
    num_layers: usize,
    num_heads: usize,
    ffn_dim: usize,
}

impl SonarConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256206,
            d_model: 1024,
            num_layers: 24,
            num_heads: 16,
            ffn_dim: 8192,
        }
    }

    fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }
}

// ---------------------------------------------------------------------------
// Manual layer norm (Metal-compatible, same as model.rs)
// ---------------------------------------------------------------------------

struct ManualLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl ManualLayerNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> CandleResult<Self> {
        let weight = vb.get(size, "weight")?;
        let bias = vb.get(size, "bias")?;
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x_centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        let x_normed = x_normed.to_dtype(x_dtype)?;
        x_normed
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}

// ---------------------------------------------------------------------------
// Multi-head attention (same pattern as NLLB, adapted for SONAR weight names)
// ---------------------------------------------------------------------------

struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    fn load(vb: VarBuilder, cfg: &SonarConfig) -> CandleResult<Self> {
        let head_dim = cfg.head_dim();
        let inner_dim = cfg.num_heads * head_dim;

        Ok(Self {
            q_proj: linear(cfg.d_model, inner_dim, vb.pp("q_proj"))?,
            k_proj: linear(cfg.d_model, inner_dim, vb.pp("k_proj"))?,
            v_proj: linear(cfg.d_model, inner_dim, vb.pp("v_proj"))?,
            out_proj: linear(inner_dim, cfg.d_model, vb.pp("output_proj"))?,
            num_heads: cfg.num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let (b, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let scale = (self.head_dim as f64).sqrt();
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn_raw = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let mut attn_weights = (attn_raw * (1.0 / scale))?;

        if let Some(mask) = mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        // Manual softmax (Metal-compatible)
        let attn_weights = {
            let max_vals = attn_weights.max_keepdim(D::Minus1)?;
            let exp = (attn_weights.broadcast_sub(&max_vals))?.exp()?;
            let sum = exp.sum_keepdim(D::Minus1)?;
            exp.broadcast_div(&sum)?
        };

        let attn_output = attn_weights.matmul(&v)?;
        attn_output
            .transpose(1, 2)?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?
            .apply(&self.out_proj)
    }
}

// ---------------------------------------------------------------------------
// SONAR encoder layer (pre-norm, self-attn + FFN)
// Weight names: self_attn_layer_norm, self_attn, ffn_layer_norm, ffn
// ---------------------------------------------------------------------------

struct SonarEncoderLayer {
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: ManualLayerNorm,
    ffn_inner: Linear,
    ffn_output: Linear,
    ffn_layer_norm: ManualLayerNorm,
}

impl SonarEncoderLayer {
    fn load(vb: VarBuilder, cfg: &SonarConfig) -> CandleResult<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::load(vb.pp("self_attn"), cfg)?,
            self_attn_layer_norm: ManualLayerNorm::load(
                cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"),
            )?,
            ffn_inner: linear(cfg.d_model, cfg.ffn_dim, vb.pp("ffn").pp("inner_proj"))?,
            ffn_output: linear(cfg.ffn_dim, cfg.d_model, vb.pp("ffn").pp("output_proj"))?,
            ffn_layer_norm: ManualLayerNorm::load(
                cfg.d_model, 1e-5, vb.pp("ffn_layer_norm"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Pre-norm self-attention
        let residual = x;
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x, None)?;
        let x = (residual + x)?;

        // Pre-norm FFN
        let residual = &x;
        let x = self.ffn_layer_norm.forward(&x)?;
        let x = self.ffn_inner.forward(&x)?.relu()?;
        let x = self.ffn_output.forward(&x)?;
        residual + x
    }
}

// ---------------------------------------------------------------------------
// Full SONAR encoder model
// ---------------------------------------------------------------------------

struct SonarModel {
    embed: candle_nn::Embedding,
    /// Sinusoidal position table (fairseq2 `SinusoidalPositionEncoder` with
    /// legacy pad offset: row `i + 2` encodes token position `i`). Not stored
    /// in the checkpoint — it is a non-persistent buffer in fairseq2.
    pos_embeddings: Tensor,
    layers: Vec<SonarEncoderLayer>,
    final_layer_norm: ManualLayerNorm,
    d_model: usize,
}

impl SonarModel {
    fn load(vb: VarBuilder, cfg: &SonarConfig, device: &Device) -> CandleResult<Self> {
        let embed = candle_nn::embedding(
            cfg.vocab_size, cfg.d_model, vb.pp("encoder_frontend").pp("embed"),
        )?;
        let pos_embeddings = crate::model::sinusoidal_embeddings(
            SONAR_MAX_SEQ_LEN, cfg.d_model, 2, embed.embeddings().dtype(), device,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(SonarEncoderLayer::load(
                vb.pp(format!("encoder.layers.{}", i)), cfg,
            )?);
        }

        let final_layer_norm = ManualLayerNorm::load(
            cfg.d_model, 1e-5, vb.pp("layer_norm"),
        )?;

        Ok(Self { embed, pos_embeddings, layers, final_layer_norm, d_model: cfg.d_model })
    }

    /// Encode input_ids -> mean-pooled, L2-normalized 1024-dim embedding.
    fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let scale = (self.d_model as f64).sqrt();
        let token_emb = (self.embed.forward(input_ids)? * scale)?;
        let pos_emb = self.pos_embeddings.i(2..seq_len + 2)?;
        let mut x = token_emb.broadcast_add(&pos_emb)?;

        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        let x = self.final_layer_norm.forward(&x)?;

        // Mean pooling over non-padding tokens
        let mask = attention_mask.unsqueeze(D::Minus1)?.to_dtype(x.dtype())?;
        let masked = x.broadcast_mul(&mask)?;
        let sum = masked.sum(1)?;
        let count = mask.sum(1)?.clamp(1e-9, f64::INFINITY)?;
        let mean_pooled = sum.broadcast_div(&count)?;

        // L2 normalization
        let norm = mean_pooled.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        mean_pooled.broadcast_div(&norm)
    }
}

/// Map a HuggingFace NLLB token ID to fairseq2's SONAR vocabulary. Pieces and
/// language tokens share IDs; only the four specials are ordered differently
/// (HF: `<s>=0 <pad>=1 </s>=2 <unk>=3`; fairseq2: `<pad>=0 <unk>=1 <s>=2 </s>=3`).
fn hf_to_fairseq2(id: u32) -> u32 {
    match id {
        0 => 2,
        1 => 0,
        2 => 3,
        3 => 1,
        other => other,
    }
}

/// Build the framed SONAR input `[lang] tokens... [</s>]` for the whole text
/// (no truncation; long inputs are windowed by the caller).
fn sonar_input_ids(tokenizer: &tokenizers::Tokenizer, text: &str, lang_code: &str) -> Result<Vec<u32>> {
    let lang_id = tokenizer
        .token_to_id(lang_code)
        .ok_or_else(|| Error::UnsupportedLanguage(lang_code.to_string()))?;
    let encoding = tokenizer.encode(text, false).map_err(|e| Error::Tokenizer(e.to_string()))?;
    let content = encoding.get_ids();
    let mut ids = Vec::with_capacity(content.len() + 2);
    ids.push(lang_id);
    ids.extend(content.iter().map(|&id| hf_to_fairseq2(id)));
    ids.push(SONAR_EOS_ID);
    Ok(ids)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// SONAR sentence encoder for cross-lingual embeddings and similarity scoring.
///
/// Produces 1024-dimensional language-agnostic embeddings.
/// Architecture: 24-layer transformer encoder (Meta's fairseq2).
///
/// SONAR conditions on the input's language. The `*_lang` / `*_langs`
/// methods take ISO 639-1/3 codes; the plain methods assume English (`en`).
///
/// # Example
///
/// ```rust,no_run
/// # use lingo::SonarEncoder;
/// # async fn example() -> lingo::Result<()> {
/// let encoder = SonarEncoder::new(None)?;
/// let score = encoder.score_langs("Hello world", "en", "Bonjour le monde", "fr").await?;
/// println!("Similarity: {:.3}", score);
/// # Ok(())
/// # }
/// ```
pub struct SonarEncoder {
    model_dir: PathBuf,
    model: Arc<Mutex<Option<SonarModel>>>,
    tokenizer: Arc<Mutex<Option<tokenizers::Tokenizer>>>,
    cache: Arc<BoundedCache<String, Vec<f32>>>,
    loaded: AtomicBool,
    device: Device,
}

impl SonarEncoder {
    /// Create a new SONAR encoder.
    ///
    /// Defaults to `~/.cache/lingo/sonar`.
    pub fn new(model_dir: Option<PathBuf>) -> Result<Self> {
        crate::license::require_license_acceptance()?;
        let model_dir = match model_dir {
            Some(d) => d,
            None => crate::default_model_dir("sonar")?,
        };
        Ok(Self::build(model_dir, crate::select_device("SONAR")))
    }

    /// Create with an explicit device.
    pub fn with_device(model_dir: PathBuf, device: Device) -> Result<Self> {
        crate::license::require_license_acceptance()?;
        Ok(Self::build(model_dir, device))
    }

    fn build(model_dir: PathBuf, device: Device) -> Self {
        Self {
            model_dir,
            model: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            cache: Arc::new(BoundedCache::new(EMBED_CACHE_MAX, EMBED_CACHE_MAX_KEY_BYTES)),
            loaded: AtomicBool::new(false),
            device,
        }
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

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

    /// Download and convert the SONAR text encoder to safetensors.
    ///
    /// Runs the conversion script bundled into the lingo binary, which
    /// downloads the checkpoint from Meta's CDN (SHA-256 verified) and
    /// converts it to safetensors, then fetches the NLLB `tokenizer.json`
    /// from a pinned HuggingFace revision. Requires Python 3 with `torch` and
    /// `safetensors`.
    #[cfg(feature = "download")]
    pub fn download_model(&self) -> Result<()> {
        crate::download::download_sonar(&self.model_dir)
    }

    /// Load the model. Called automatically on first use.
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
        if tokenizer.token_to_id("eng_Latn").is_none() {
            // Reported as ModelNotFound: the model is unusable until refreshed.
            return Err(Error::ModelNotFound(format!(
                "{} (legacy SONAR tokenizer format with no language tokens; run \
                 `lingo download sonar` to refresh it — the converted weights are kept)",
                tokenizer_path.display()
            )));
        }

        let weight_path = self.model_dir.join("model.safetensors");
        if !weight_path.exists() {
            return Err(Error::ModelNotFound("model.safetensors not found".into()));
        }

        let cfg = SonarConfig::default();
        // SAFETY: read-only memory map; the file must not change while loaded.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weight_path], DType::F32, &self.device)
                .map_err(|e| Error::Inference(format!("Failed to load SONAR weights: {}", e)))?
        };

        let m = SonarModel::load(vb, &cfg, &self.device)
            .map_err(|e| Error::Inference(format!("Failed to build SONAR model: {}", e)))?;

        info!("SONAR loaded on {:?} ({} layers, {} dim)", self.device, cfg.num_layers, cfg.d_model);

        *model_guard = Some(m);
        *self.tokenizer.lock().await = Some(tokenizer);
        self.loaded.store(true, Ordering::Release);
        Ok(())
    }

    /// Encode English text into a 1024-dimensional embedding.
    /// Equivalent to `embed_lang(text, "en")`.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_lang(text, DEFAULT_LANG).await
    }

    /// Encode text in language `lang` (ISO 639-1/3 code such as `fr`, or an
    /// NLLB/FLORES code such as `fra_Latn`) into a 1024-dimensional,
    /// L2-normalized embedding.
    ///
    /// Inputs longer than SONAR's 512-position context are not truncated: the
    /// content is encoded in consecutive 510-token windows (each framed with
    /// the language token and `</s>`), and the window embeddings are combined
    /// as a token-weighted mean. Short inputs are a single, unchanged pass.
    pub async fn embed_lang(&self, text: &str, lang: &str) -> Result<Vec<f32>> {
        let lang_code = NllbLanguage::resolve(lang)
            .ok_or_else(|| Error::UnsupportedLanguage(lang.to_string()))?
            .sonar_code();

        let cache_key = format!("{}\u{1f}{}", lang_code, text);
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached);
        }

        self.load().await?;

        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let device = self.device.clone();
        let text_owned = text.to_string();
        let embedding_vec = crate::run_blocking("SONAR", move |cancel| -> Result<Vec<f32>> {
            // Lock order (model, then tokenizer) matches `load`.
            let model_guard = model.blocking_lock();
            cancel.check()?;
            let model = model_guard.as_ref()
                .ok_or_else(|| Error::Inference("Model not loaded".into()))?;
            let tok_guard = tokenizer.blocking_lock();
            let tokenizer = tok_guard.as_ref()
                .ok_or_else(|| Error::Inference("Tokenizer not loaded".into()))?;

            let ids = sonar_input_ids(tokenizer, &text_owned, lang_code)?;
            let windows = crate::windows::token_windows(&ids, 1, 1, SONAR_MAX_SEQ_LEN);
            if windows.len() > 1 {
                debug!("SONAR: encoding {} windows ({} tokens)", windows.len(), ids.len());
            }
            let mut pooled = Vec::with_capacity(windows.len());
            for window in windows {
                cancel.check()?;
                let len = window.ids.len();
                let input_ids_t = Tensor::from_vec(window.ids, (1, len), &device)
                    .map_err(|e| Error::Inference(e.to_string()))?;
                let attention_mask_t = Tensor::ones((1, len), DType::F32, &device)
                    .map_err(|e| Error::Inference(e.to_string()))?;
                let embedding: Vec<f32> = model.encode(&input_ids_t, &attention_mask_t)
                    .and_then(|t| t.i(0))
                    .and_then(|t| t.to_vec1())
                    .map_err(|e| Error::Inference(format!("SONAR encoding failed: {}", e)))?;
                pooled.push((embedding, window.weight));
            }
            Ok(crate::windows::combine(pooled))
        })
        .await?;

        self.cache.insert(cache_key, embedding_vec.clone());
        Ok(embedding_vec)
    }

    /// Score similarity between two English texts. Returns cosine similarity
    /// in [-1, 1]. Equivalent to `score_langs(text1, "en", text2, "en")`.
    pub async fn score(&self, text1: &str, text2: &str) -> Result<f32> {
        self.score_langs(text1, DEFAULT_LANG, text2, DEFAULT_LANG).await
    }

    /// Score similarity between `text1` (in `lang1`) and `text2` (in `lang2`).
    /// Returns cosine similarity in [-1, 1].
    pub async fn score_langs(&self, text1: &str, lang1: &str, text2: &str, lang2: &str) -> Result<f32> {
        let start = Instant::now();
        let emb1 = self.embed_lang(text1, lang1).await?;
        let emb2 = self.embed_lang(text2, lang2).await?;
        let score: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        debug!("SONAR score: {:.4} ({}ms)", score, start.elapsed().as_millis());
        Ok(score)
    }

    /// Score a batch of English text pairs.
    pub async fn batch_score(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(pairs.len());
        for (t1, t2) in pairs {
            scores.push(self.score(t1, t2).await?);
        }
        Ok(scores)
    }

    /// Same as [`embed`](Self::embed) but bounded by `timeout`.
    ///
    /// Returns [`Error::Timeout`] if the call does not complete within the
    /// budget. Inference runs on tokio's blocking pool, so the timeout fires
    /// even while the GPU call is stuck. The abandoned work stops before its
    /// next window; call [`wait_idle`](Self::wait_idle) before reusing the
    /// encoder. `false` means it is stuck inside a GPU call: drop the encoder
    /// and construct a new one.
    pub async fn embed_with_timeout(
        &self,
        text: &str,
        timeout: std::time::Duration,
    ) -> Result<Vec<f32>> {
        crate::bounded(timeout, self.embed(text)).await
    }

    /// Same as [`embed_lang`](Self::embed_lang) but bounded by `timeout`.
    /// See [`embed_with_timeout`](Self::embed_with_timeout) for caveats.
    pub async fn embed_lang_with_timeout(
        &self,
        text: &str,
        lang: &str,
        timeout: std::time::Duration,
    ) -> Result<Vec<f32>> {
        crate::bounded(timeout, self.embed_lang(text, lang)).await
    }

    /// Same as [`score`](Self::score) but bounded by `timeout`.
    ///
    /// See [`embed_with_timeout`](Self::embed_with_timeout) for the
    /// caveats about timeout semantics and encoder poisoning.
    pub async fn score_with_timeout(
        &self,
        text1: &str,
        text2: &str,
        timeout: std::time::Duration,
    ) -> Result<f32> {
        crate::bounded(timeout, self.score(text1, text2)).await
    }

    /// Same as [`score_langs`](Self::score_langs) but bounded by `timeout`.
    /// See [`embed_with_timeout`](Self::embed_with_timeout) for caveats.
    pub async fn score_langs_with_timeout(
        &self,
        text1: &str,
        lang1: &str,
        text2: &str,
        lang2: &str,
        timeout: std::time::Duration,
    ) -> Result<f32> {
        crate::bounded(timeout, self.score_langs(text1, lang1, text2, lang2)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specials_remap_to_fairseq2_layout() {
        assert_eq!(hf_to_fairseq2(0), 2); // <s>
        assert_eq!(hf_to_fairseq2(1), 0); // <pad>
        assert_eq!(hf_to_fairseq2(2), 3); // </s>
        assert_eq!(hf_to_fairseq2(3), 1); // <unk>
        assert_eq!(hf_to_fairseq2(4), 4); // ordinary piece unchanged
        assert_eq!(hf_to_fairseq2(256047), 256047); // eng_Latn unchanged
    }
}
