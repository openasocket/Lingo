//! SONAR text encoder — cross-lingual sentence embeddings (1024-dim).
//!
//! Meta's SONAR encodes text into language-agnostic embeddings for
//! cross-lingual similarity scoring. Architecture: 24-layer transformer
//! encoder with 1024 hidden dim, 8192 FFN dim, 16 attention heads.
//!
//! Output embeddings are L2-normalized for cosine similarity.

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{debug, info};

use crate::error::{Error, Result};

// ---------------------------------------------------------------------------
// SONAR config
// ---------------------------------------------------------------------------

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
    layers: Vec<SonarEncoderLayer>,
    final_layer_norm: ManualLayerNorm,
    _config: SonarConfig,
}

impl SonarModel {
    fn load(vb: VarBuilder, cfg: &SonarConfig) -> CandleResult<Self> {
        let embed = candle_nn::embedding(
            cfg.vocab_size, cfg.d_model, vb.pp("encoder_frontend").pp("embed"),
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

        Ok(Self { embed, layers, final_layer_norm, _config: cfg.clone() })
    }

    /// Encode input_ids -> mean-pooled, L2-normalized 1024-dim embedding.
    fn encode(&self, input_ids: &Tensor, attention_mask: &Tensor) -> CandleResult<Tensor> {
        let scale = (1024f64).sqrt();
        let mut x = (self.embed.forward(input_ids)? * scale)?;

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

impl Clone for SonarConfig {
    fn clone(&self) -> Self {
        Self {
            vocab_size: self.vocab_size,
            d_model: self.d_model,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            ffn_dim: self.ffn_dim,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// SONAR sentence encoder for cross-lingual embeddings and similarity scoring.
///
/// Produces 1024-dimensional language-agnostic embeddings.
/// Architecture: 24-layer transformer encoder (Meta's fairseq2).
///
/// # Example
///
/// ```rust,no_run
/// # use lingo::SonarEncoder;
/// # async fn example() -> lingo::Result<()> {
/// let encoder = SonarEncoder::new(None)?;
/// let score = encoder.score("Hello world", "Bonjour le monde").await?;
/// println!("Similarity: {:.3}", score);
/// # Ok(())
/// # }
/// ```
pub struct SonarEncoder {
    model_dir: PathBuf,
    model: Arc<Mutex<Option<SonarModel>>>,
    tokenizer: Arc<Mutex<Option<tokenizers::Tokenizer>>>,
    cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    device: Device,
}

impl SonarEncoder {
    /// Create a new SONAR encoder.
    ///
    /// Defaults to `~/.cache/lingo/sonar`.
    pub fn new(model_dir: Option<PathBuf>) -> Result<Self> {
        crate::license::require_license_acceptance()?;
        let model_dir = model_dir.unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join(".cache/lingo/sonar")
        });
        let device = Self::select_device();
        Ok(Self {
            model_dir,
            model: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            cache: Arc::new(Mutex::new(HashMap::new())),
            device,
        })
    }

    pub fn with_device(model_dir: PathBuf, device: Device) -> Result<Self> {
        Ok(Self {
            model_dir,
            model: Arc::new(Mutex::new(None)),
            tokenizer: Arc::new(Mutex::new(None)),
            cache: Arc::new(Mutex::new(HashMap::new())),
            device,
        })
    }

    fn select_device() -> Device {
        #[cfg(feature = "metal")]
        if cfg!(target_os = "macos") {
            if let Ok(device) = Device::new_metal(0) {
                info!("SONAR: Using Metal GPU");
                return device;
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                info!("SONAR: Using CUDA GPU");
                return device;
            }
        }
        info!("SONAR: Using CPU");
        Device::Cpu
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn is_model_downloaded(&self) -> bool {
        self.model_dir.join("model.safetensors").exists()
            && self.model_dir.join("tokenizer.json").exists()
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

        let weight_path = self.model_dir.join("model.safetensors");
        if !weight_path.exists() {
            return Err(Error::ModelNotFound("model.safetensors not found".into()));
        }

        let cfg = SonarConfig::default();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weight_path], DType::F32, &self.device)
                .map_err(|e| Error::Inference(format!("Failed to load SONAR weights: {}", e)))?
        };

        let m = SonarModel::load(vb, &cfg)
            .map_err(|e| Error::Inference(format!("Failed to build SONAR model: {}", e)))?;

        let device_name = if matches!(self.device, Device::Cpu) { "CPU" } else { "Metal GPU" };
        info!("SONAR loaded on {} (24 layers, 1024 dim)", device_name);

        *model_guard = Some(m);
        *self.tokenizer.lock().await = Some(tokenizer);
        Ok(())
    }

    /// Encode text into a 1024-dimensional embedding.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(cached) = self.cache.lock().await.get(text) {
            return Ok(cached.clone());
        }

        self.load().await?;

        let tok_guard = self.tokenizer.lock().await;
        let tokenizer = tok_guard.as_ref()
            .ok_or_else(|| Error::Inference("Tokenizer not loaded".into()))?;

        let encoding = tokenizer.encode(text, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let input_ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        let input_ids_t = Tensor::from_vec(
            input_ids.to_vec(), (1, input_ids.len()), &self.device,
        ).map_err(|e| Error::Inference(e.to_string()))?;

        let attention_mask_t = Tensor::from_vec(
            attention_mask.iter().map(|&m| m as f32).collect::<Vec<_>>(),
            (1, attention_mask.len()), &self.device,
        ).map_err(|e| Error::Inference(e.to_string()))?;

        let model_guard = self.model.lock().await;
        let model = model_guard.as_ref()
            .ok_or_else(|| Error::Inference("Model not loaded".into()))?;

        let embeddings = model.encode(&input_ids_t, &attention_mask_t)
            .map_err(|e| Error::Inference(format!("SONAR encoding failed: {}", e)))?;

        let embedding_vec: Vec<f32> = embeddings.i(0)
            .map_err(|e| Error::Inference(e.to_string()))?
            .to_vec1()
            .map_err(|e| Error::Inference(e.to_string()))?;

        self.cache.lock().await.insert(text.to_string(), embedding_vec.clone());
        Ok(embedding_vec)
    }

    /// Score similarity between two texts. Returns cosine similarity in [-1, 1].
    pub async fn score(&self, text1: &str, text2: &str) -> Result<f32> {
        let start = Instant::now();
        let emb1 = self.embed(text1).await?;
        let emb2 = self.embed(text2).await?;
        let score: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
        debug!("SONAR score: {:.4} ({}ms)", score, start.elapsed().as_millis());
        Ok(score)
    }

    /// Score a batch of text pairs.
    pub async fn batch_score(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(pairs.len());
        for (t1, t2) in pairs {
            scores.push(self.score(t1, t2).await?);
        }
        Ok(scores)
    }
}
