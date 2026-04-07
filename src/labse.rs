//! LaBSE (Language-agnostic BERT Sentence Embeddings) encoder.
//!
//! Implements the full LaBSE pipeline with a custom BERT encoder that uses
//! ManualLayerNorm (decomposed ops) for Metal/CUDA GPU compatibility:
//! 1. BERT encoder (12 layers, 768 dim, 501K vocab)
//! 2. CLS token pooling
//! 3. Dense projection (768 -> 768) + Tanh
//! 4. L2 normalization
//!
//! Similarity scoring is cosine similarity (dot product of normalized vectors).

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::error::{Error, Result};

// ---------------------------------------------------------------------------
// BERT config (parsed from config.json)
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct BertConfig {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    layer_norm_eps: f64,
}

fn default_layer_norm_eps() -> f64 {
    1e-12
}

impl BertConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ---------------------------------------------------------------------------
// Manual layer norm (Metal/CUDA-compatible, same pattern as model.rs/sonar.rs)
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
// BERT embeddings: word + position + token_type + LayerNorm
// ---------------------------------------------------------------------------

struct BertEmbeddings {
    word_embeddings: candle_nn::Embedding,
    position_embeddings: candle_nn::Embedding,
    token_type_embeddings: candle_nn::Embedding,
    layer_norm: ManualLayerNorm,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, cfg: &BertConfig) -> CandleResult<Self> {
        let vb_emb = vb.pp("embeddings");
        Ok(Self {
            word_embeddings: candle_nn::embedding(
                cfg.vocab_size, cfg.hidden_size, vb_emb.pp("word_embeddings"),
            )?,
            position_embeddings: candle_nn::embedding(
                cfg.max_position_embeddings, cfg.hidden_size, vb_emb.pp("position_embeddings"),
            )?,
            token_type_embeddings: candle_nn::embedding(
                cfg.type_vocab_size, cfg.hidden_size, vb_emb.pp("token_type_embeddings"),
            )?,
            layer_norm: ManualLayerNorm::load(
                cfg.hidden_size, cfg.layer_norm_eps, vb_emb.pp("LayerNorm"),
            )?,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> CandleResult<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let device = input_ids.device();

        let word_emb = self.word_embeddings.forward(input_ids)?;
        let type_emb = self.token_type_embeddings.forward(token_type_ids)?;

        let position_ids = Tensor::arange(0u32, seq_len as u32, device)?
            .unsqueeze(0)?;
        let pos_emb = self.position_embeddings.forward(&position_ids)?;

        let embeddings = (word_emb + type_emb + pos_emb)?;
        self.layer_norm.forward(&embeddings)
    }
}

// ---------------------------------------------------------------------------
// Multi-head self-attention
// ---------------------------------------------------------------------------

struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl BertSelfAttention {
    fn load(vb: VarBuilder, cfg: &BertConfig) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        Ok(Self {
            query: linear(h, h, vb.pp("query"))?,
            key: linear(h, h, vb.pp("key"))?,
            value: linear(h, h, vb.pp("value"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, hidden: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let (b, seq_len, _) = hidden.dims3()?;

        let q = self.query.forward(hidden)?;
        let k = self.key.forward(hidden)?;
        let v = self.value.forward(hidden)?;

        // Reshape to [batch, heads, seq, head_dim]
        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let scale = (self.head_dim as f64).sqrt();
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let attn_weights = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let mut attn_weights = (attn_weights * (1.0 / scale))?;

        if let Some(mask) = attention_mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        // Manual softmax (Metal/CUDA-compatible)
        let attn_weights = {
            let max_vals = attn_weights.max_keepdim(D::Minus1)?;
            let exp = attn_weights.broadcast_sub(&max_vals)?.exp()?;
            let sum = exp.sum_keepdim(D::Minus1)?;
            exp.broadcast_div(&sum)?
        };

        let attn_output = attn_weights.matmul(&v)?;
        attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_len, self.num_heads * self.head_dim))
    }
}

// ---------------------------------------------------------------------------
// BERT encoder layer (post-norm: residual then LayerNorm)
// Weight keys:
//   attention.self.{query,key,value}  — self-attention Q/K/V
//   attention.output.dense            — attention output projection
//   attention.output.LayerNorm        — post-attention layer norm
//   intermediate.dense                — FFN up (hidden -> intermediate)
//   output.dense                      — FFN down (intermediate -> hidden)
//   output.LayerNorm                  — post-FFN layer norm
// ---------------------------------------------------------------------------

struct BertEncoderLayer {
    self_attn: BertSelfAttention,
    attn_output: Linear,
    attn_layer_norm: ManualLayerNorm,
    intermediate: Linear,
    output: Linear,
    output_layer_norm: ManualLayerNorm,
}

impl BertEncoderLayer {
    fn load(vb: VarBuilder, cfg: &BertConfig) -> CandleResult<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            self_attn: BertSelfAttention::load(vb.pp("attention").pp("self"), cfg)?,
            attn_output: linear(h, h, vb.pp("attention").pp("output").pp("dense"))?,
            attn_layer_norm: ManualLayerNorm::load(
                h, cfg.layer_norm_eps, vb.pp("attention").pp("output").pp("LayerNorm"),
            )?,
            intermediate: linear(h, i, vb.pp("intermediate").pp("dense"))?,
            output: linear(i, h, vb.pp("output").pp("dense"))?,
            output_layer_norm: ManualLayerNorm::load(
                h, cfg.layer_norm_eps, vb.pp("output").pp("LayerNorm"),
            )?,
        })
    }

    fn forward(&self, hidden: &Tensor, attention_mask: Option<&Tensor>) -> CandleResult<Tensor> {
        // Self-attention + residual + LayerNorm (post-norm)
        let attn_out = self.self_attn.forward(hidden, attention_mask)?;
        let attn_out = self.attn_output.forward(&attn_out)?;
        let hidden = self.attn_layer_norm.forward(&(hidden + attn_out)?)?;

        // FFN + residual + LayerNorm (post-norm)
        let ffn_out = self.intermediate.forward(&hidden)?;
        // GELU activation
        let ffn_out = ffn_out.gelu_erf()?;
        let ffn_out = self.output.forward(&ffn_out)?;
        self.output_layer_norm.forward(&(&hidden + ffn_out)?)
    }
}

// ---------------------------------------------------------------------------
// Full custom BERT model
// ---------------------------------------------------------------------------

struct CustomBertModel {
    embeddings: BertEmbeddings,
    layers: Vec<BertEncoderLayer>,
}

impl CustomBertModel {
    fn load(vb: VarBuilder, cfg: &BertConfig) -> CandleResult<Self> {
        let embeddings = BertEmbeddings::load(vb.clone(), cfg)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(BertEncoderLayer::load(
                vb.pp(format!("encoder.layer.{}", i)), cfg,
            )?);
        }
        Ok(Self { embeddings, layers })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let mut hidden = self.embeddings.forward(input_ids, token_type_ids)?;

        // Convert attention_mask [batch, seq] -> [batch, 1, 1, seq] with 0/-inf
        let mask = attention_mask.map(|m| -> CandleResult<Tensor> {
            let cond = m.to_dtype(DType::U32)?.unsqueeze(1)?.unsqueeze(1)?;
            let on_true = Tensor::zeros(cond.shape(), DType::F32, m.device())?;
            let on_false = Tensor::new(f32::NEG_INFINITY, m.device())?
                .broadcast_as(cond.shape())?;
            cond.where_cond(&on_true, &on_false)
        }).transpose()?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask.as_ref())?;
        }

        Ok(hidden)
    }
}

// ---------------------------------------------------------------------------
// LaBSE model (custom BERT + dense projection)
// ---------------------------------------------------------------------------

struct LaBSEModel {
    bert: CustomBertModel,
    dense_weight: Tensor,
    dense_bias: Tensor,
}

impl LaBSEModel {
    fn load(vb: VarBuilder, config: &BertConfig) -> CandleResult<Self> {
        let bert = CustomBertModel::load(vb.clone(), config)?;

        // Load the 2_Dense projection layer (768 -> 768 + Tanh)
        let (dense_weight, dense_bias) = if let (Ok(w), Ok(b)) = (
            vb.pp("2_Dense").get((768, 768), "linear.weight"),
            vb.pp("2_Dense").get(768, "linear.bias"),
        ) {
            (w, b)
        } else if let (Ok(w), Ok(b)) = (
            vb.pp("linear").get((768, 768), "weight"),
            vb.pp("linear").get(768, "bias"),
        ) {
            (w, b)
        } else if let (Ok(w), Ok(b)) = (
            vb.get((768, 768), "linear.weight"),
            vb.get(768, "linear.bias"),
        ) {
            (w, b)
        } else {
            return Err(candle_core::Error::Msg(
                "Cannot find 2_Dense projection weights".into(),
            ));
        };

        Ok(Self { bert, dense_weight, dense_bias })
    }

    /// Encode a batch of token IDs into normalized embeddings.
    fn encode(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        // BERT forward pass -> [batch, seq_len, 768]
        let sequence_output = self.bert.forward(input_ids, token_type_ids, attention_mask)?;

        // CLS token pooling: extract position 0 -> [batch, 768]
        let cls_output = sequence_output.i((.., 0, ..))?;

        // Dense projection: linear(768, 768)
        let dense_output = cls_output.broadcast_matmul(&self.dense_weight.t()?)?;
        let dense_output = dense_output.broadcast_add(&self.dense_bias)?;

        // Tanh activation
        let tanh_output = dense_output.tanh()?;

        // L2 normalization: v / ||v||_2
        let norm = tanh_output
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?;
        tanh_output.broadcast_div(&norm)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// LaBSE sentence embedding encoder with similarity scoring.
///
/// Encodes text into 768-dimensional embeddings suitable for
/// cross-lingual similarity comparison.
///
/// # Example
///
/// ```rust,no_run
/// # use lingo::LaBSEEncoder;
/// # async fn example() -> lingo::Result<()> {
/// let encoder = LaBSEEncoder::new(None)?;
/// let score = encoder.score("Hello world", "Bonjour le monde").await?;
/// println!("Similarity: {:.3}", score); // ~0.85+
/// # Ok(())
/// # }
/// ```
pub struct LaBSEEncoder {
    model_dir: PathBuf,
    model: Arc<Mutex<Option<LaBSEModel>>>,
    tokenizer: Arc<Mutex<Option<tokenizers::Tokenizer>>>,
    cache: Arc<Mutex<HashMap<String, Vec<f32>>>>,
    device: Device,
}

impl LaBSEEncoder {
    /// Create a new LaBSE encoder.
    ///
    /// - `model_dir`: Path to directory containing model weights and tokenizer.
    ///   If `None`, defaults to `~/.cache/lingo/labse`.
    pub fn new(model_dir: Option<PathBuf>) -> Result<Self> {
        crate::license::require_license_acceptance()?;
        let model_dir = model_dir.unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join(".cache/lingo/labse")
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

    /// Create with an explicit device.
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
                info!("LaBSE: Using Metal GPU");
                return device;
            }
        }
        #[cfg(feature = "cuda")]
        {
            if let Ok(device) = Device::new_cuda(0) {
                info!("LaBSE: Using CUDA GPU");
                return device;
            }
        }
        info!("LaBSE: Using CPU");
        Device::Cpu
    }

    /// Get the model directory.
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Check if model files exist on disk.
    pub fn is_model_downloaded(&self) -> bool {
        (self.model_dir.join("model.safetensors").exists()
            || self.model_dir.join("pytorch_model.bin").exists())
            && self.model_dir.join("tokenizer.json").exists()
            && self.model_dir.join("config.json").exists()
    }

    /// Download LaBSE from HuggingFace Hub to the model directory.
    ///
    /// Downloads `model.safetensors` (~1.8 GB), `config.json`, `tokenizer.json`,
    /// and `2_Dense/model.safetensors`. Requires the `download` feature.
    #[cfg(feature = "download")]
    pub fn download_model(&self) -> Result<()> {
        crate::download::download_labse(&self.model_dir)
    }

    /// Load the model. Called automatically on first use.
    pub async fn load(&self) -> Result<()> {
        let mut model_guard = self.model.lock().await;
        if model_guard.is_some() {
            return Ok(());
        }

        debug!("Loading LaBSE model...");

        if !self.model_dir.exists() {
            return Err(Error::ModelNotFound(self.model_dir.display().to_string()));
        }

        // Load config
        let config_path = self.model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::ModelNotFound(format!("config.json: {}", e)))?;
        let config: BertConfig = serde_json::from_str(&config_str)
            .map_err(|e| Error::Inference(format!("Invalid config.json: {}", e)))?;

        // Load tokenizer with truncation (BERT max 512 positions)
        let tokenizer_path = self.model_dir.join("tokenizer.json");
        let mut tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;

        let max_len = config.max_position_embeddings;
        tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: max_len,
            ..Default::default()
        })).map_err(|e| Error::Tokenizer(format!("Failed to set truncation: {}", e)))?;

        // Load weights
        let weight_files: Vec<PathBuf> = if self.model_dir.join("model.safetensors").exists() {
            let mut files = vec![self.model_dir.join("model.safetensors")];
            let dense_path = self.model_dir.join("2_Dense").join("model.safetensors");
            if dense_path.exists() {
                files.push(dense_path);
            }
            files
        } else {
            return Err(Error::ModelNotFound(
                "model.safetensors not found".to_string(),
            ));
        };

        let weight_refs: Vec<&PathBuf> = weight_files.iter().collect();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_refs, DType::F32, &self.device)
                .map_err(|e| Error::Inference(format!("Failed to load weights: {}", e)))?
        };

        let m = LaBSEModel::load(vb, &config)
            .map_err(|e| Error::Inference(format!("Failed to build LaBSE model: {}", e)))?;

        let device_name = format!("{:?}", self.device);
        info!(
            "LaBSE loaded on {} ({} layers, {} dim, {}K vocab)",
            device_name, config.num_hidden_layers, config.hidden_size, config.vocab_size / 1000
        );

        *model_guard = Some(m);
        let mut tok_guard = self.tokenizer.lock().await;
        *tok_guard = Some(tokenizer);

        Ok(())
    }

    /// Encode text into a 768-dimensional embedding vector.
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        {
            let cache = self.cache.lock().await;
            if let Some(cached) = cache.get(text) {
                return Ok(cached.clone());
            }
        }

        self.load().await?;

        let (input_ids_t, token_type_ids_t, attention_mask_t) = {
            let tok_guard = self.tokenizer.lock().await;
            let tokenizer = tok_guard
                .as_ref()
                .ok_or_else(|| Error::Inference("Tokenizer not loaded".into()))?;

            let encoding = tokenizer
                .encode(text, true)
                .map_err(|e| Error::Tokenizer(e.to_string()))?;

            let input_ids = encoding.get_ids();
            if input_ids.len() >= 512 {
                warn!(
                    "LaBSE: input truncated to 512 tokens (original text was {} chars)",
                    text.len()
                );
            }
            let token_type_ids = encoding.get_type_ids();
            let attention_mask = encoding.get_attention_mask();

            let ids = Tensor::from_vec(
                input_ids.to_vec(), (1, input_ids.len()), &self.device,
            ).map_err(|e| Error::Inference(e.to_string()))?;

            let types = Tensor::from_vec(
                token_type_ids.to_vec(), (1, token_type_ids.len()), &self.device,
            ).map_err(|e| Error::Inference(e.to_string()))?;

            let mask = Tensor::from_vec(
                attention_mask.iter().map(|&m| m as f32).collect::<Vec<_>>(),
                (1, attention_mask.len()), &self.device,
            ).map_err(|e| Error::Inference(e.to_string()))?;

            (ids, types, mask)
        };

        let embedding_vec: Vec<f32> = {
            let model_guard = self.model.lock().await;
            let model = model_guard
                .as_ref()
                .ok_or_else(|| Error::Inference("Model not loaded".into()))?;

            let embeddings = model
                .encode(&input_ids_t, &token_type_ids_t, Some(&attention_mask_t))
                .map_err(|e| Error::Inference(format!("Encoding failed: {}", e)))?;

            embeddings
                .i(0)
                .map_err(|e| Error::Inference(e.to_string()))?
                .to_vec1()
                .map_err(|e| Error::Inference(e.to_string()))?
        };

        {
            let mut cache = self.cache.lock().await;
            cache.insert(text.to_string(), embedding_vec.clone());
        }

        Ok(embedding_vec)
    }

    /// Encode a batch of texts into embeddings.
    pub async fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.embed(text).await?);
        }
        Ok(results)
    }

    /// Score semantic similarity between two texts.
    ///
    /// Returns a value in \[-1.0, 1.0\], where 1.0 means identical meaning.
    /// Cross-lingual pairs of the same sentence typically score 0.8+.
    pub async fn score(&self, text1: &str, text2: &str) -> Result<f32> {
        let start = Instant::now();

        let emb1 = self.embed(text1).await?;
        let emb2 = self.embed(text2).await?;

        // Cosine similarity = dot product of L2-normalized vectors
        let score: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();

        debug!(
            "LaBSE score: {:.4} ({}ms)",
            score,
            start.elapsed().as_millis()
        );

        Ok(score)
    }

    /// Score similarity for a batch of text pairs.
    pub async fn batch_score(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(pairs.len());
        for (text1, text2) in pairs {
            scores.push(self.score(text1, text2).await?);
        }
        Ok(scores)
    }
}
