//! M2M100 encoder-decoder transformer model for NLLB-200.
//!
//! This module contains the full model architecture:
//! - Sinusoidal positional embeddings (M2M100 convention, offset=2)
//! - ManualLayerNorm (portable across Metal, CUDA, and CPU backends)
//! - Multi-head attention with contiguous matmul for cross-backend compatibility
//! - Pre-norm encoder and decoder layers
//! - Greedy autoregressive generation with forced BOS token

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor, D};
use candle_nn::{embedding, linear, Embedding, Linear, Module, VarBuilder};

// ---------------------------------------------------------------------------
// Model configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct NllbConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub encoder_layers: usize,
    pub decoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub decoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    pub max_position_embeddings: usize,
    pub eos_token_id: u32,
    pub decoder_start_token_id: u32,
    pub scale_embedding: bool,
}

impl NllbConfig {
    /// Config for facebook/nllb-200-distilled-600M
    pub fn distilled_600m() -> Self {
        Self {
            vocab_size: 256206,
            d_model: 1024,
            encoder_layers: 12,
            decoder_layers: 12,
            encoder_attention_heads: 16,
            decoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            decoder_ffn_dim: 4096,
            max_position_embeddings: 1024,
            eos_token_id: 2,
            decoder_start_token_id: 2,
            scale_embedding: true,
        }
    }

    fn head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }
}

// ---------------------------------------------------------------------------
// Sinusoidal positional embeddings (M2M100 / NLLB style, offset=2)
// ---------------------------------------------------------------------------

fn sinusoidal_embeddings(
    max_len: usize,
    d_model: usize,
    offset: usize,
    dtype: DType,
    device: &Device,
) -> CandleResult<Tensor> {
    let total = max_len + offset;
    let mut data = vec![0f32; total * d_model];
    let half = d_model / 2;

    for pos in 0..total {
        for i in 0..half {
            let angle = (pos as f64) / (10000f64).powf(2.0 * i as f64 / d_model as f64);
            data[pos * d_model + i] = angle.sin() as f32;
            data[pos * d_model + half + i] = angle.cos() as f32;
        }
    }

    Tensor::from_vec(data, (total, d_model), device)?.to_dtype(dtype)
}

// ---------------------------------------------------------------------------
// Manual layer norm — uses primitive ops for portability across all backends
// (Metal, CUDA, CPU). Avoids candle's built-in LayerNorm which lacks some
// Metal kernels.
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

fn manual_layer_norm(size: usize, eps: f64, vb: VarBuilder) -> CandleResult<ManualLayerNorm> {
    ManualLayerNorm::load(size, eps, vb)
}

// ---------------------------------------------------------------------------
// Multi-head attention
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
    fn load(vb: VarBuilder, cfg: &NllbConfig, is_decoder: bool) -> CandleResult<Self> {
        let num_heads = if is_decoder {
            cfg.decoder_attention_heads
        } else {
            cfg.encoder_attention_heads
        };
        let head_dim = cfg.head_dim();
        let inner_dim = num_heads * head_dim;

        Ok(Self {
            q_proj: linear(cfg.d_model, inner_dim, vb.pp("q_proj"))?,
            k_proj: linear(cfg.d_model, inner_dim, vb.pp("k_proj"))?,
            v_proj: linear(cfg.d_model, inner_dim, vb.pp("v_proj"))?,
            out_proj: linear(inner_dim, cfg.d_model, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        query: &Tensor,
        key_value: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let (b, tgt_len, _) = query.dims3()?;
        let kv_input = key_value.unwrap_or(query);

        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(kv_input)?;
        let v = self.v_proj.forward(kv_input)?;

        let q = q
            .reshape((b, tgt_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let src_len = kv_input.dim(1)?;
        let k = k
            .reshape((b, src_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, src_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = (self.head_dim as f64).sqrt();
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        let attn_raw = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let mut attn_weights = (attn_raw * (1.0 / scale))?;

        if let Some(mask) = mask {
            attn_weights = attn_weights.broadcast_add(mask)?;
        }

        // Manual softmax for cross-backend portability
        let attn_weights = {
            let max_vals = attn_weights.max_keepdim(D::Minus1)?;
            let exp = (attn_weights.broadcast_sub(&max_vals))?.exp()?;
            let sum = exp.sum_keepdim(D::Minus1)?;
            exp.broadcast_div(&sum)?
        };
        let attn_output = attn_weights.matmul(&v)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b, tgt_len, self.num_heads * self.head_dim))?
            .apply(&self.out_proj)
    }
}

// ---------------------------------------------------------------------------
// Encoder layer (pre-norm)
// ---------------------------------------------------------------------------

struct EncoderLayer {
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: ManualLayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: ManualLayerNorm,
}

impl EncoderLayer {
    fn load(vb: VarBuilder, cfg: &NllbConfig) -> CandleResult<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::load(vb.pp("self_attn"), cfg, false)?,
            self_attn_layer_norm: manual_layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?,
            fc1: linear(cfg.d_model, cfg.encoder_ffn_dim, vb.pp("fc1"))?,
            fc2: linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?,
            final_layer_norm: manual_layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let residual = x;
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x, None, None)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.fc1.forward(&x)?.relu()?;
        let x = self.fc2.forward(&x)?;
        residual + x
    }
}

// ---------------------------------------------------------------------------
// Decoder layer (pre-norm, with cross-attention)
// ---------------------------------------------------------------------------

struct DecoderLayer {
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: ManualLayerNorm,
    encoder_attn: MultiHeadAttention,
    encoder_attn_layer_norm: ManualLayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: ManualLayerNorm,
}

impl DecoderLayer {
    fn load(vb: VarBuilder, cfg: &NllbConfig) -> CandleResult<Self> {
        Ok(Self {
            self_attn: MultiHeadAttention::load(vb.pp("self_attn"), cfg, true)?,
            self_attn_layer_norm: manual_layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?,
            encoder_attn: MultiHeadAttention::load(vb.pp("encoder_attn"), cfg, true)?,
            encoder_attn_layer_norm: manual_layer_norm(cfg.d_model, 1e-5, vb.pp("encoder_attn_layer_norm"))?,
            fc1: linear(cfg.d_model, cfg.decoder_ffn_dim, vb.pp("fc1"))?,
            fc2: linear(cfg.decoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?,
            final_layer_norm: manual_layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        encoder_output: &Tensor,
        causal_mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let residual = x;
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x, None, causal_mask)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.encoder_attn_layer_norm.forward(&x)?;
        let x = self.encoder_attn.forward(&x, Some(encoder_output), None)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.fc1.forward(&x)?.relu()?;
        let x = self.fc2.forward(&x)?;
        residual + x
    }
}

// ---------------------------------------------------------------------------
// Full NLLB model (M2M100ForConditionalGeneration)
// ---------------------------------------------------------------------------

pub(crate) struct NllbModel {
    shared_embeddings: Embedding,
    encoder_layers: Vec<EncoderLayer>,
    decoder_layers: Vec<DecoderLayer>,
    encoder_layer_norm: ManualLayerNorm,
    decoder_layer_norm: ManualLayerNorm,
    lm_head: Linear,
    pos_embeddings: Tensor,
    config: NllbConfig,
}

impl NllbModel {
    pub fn load(vb: VarBuilder, cfg: &NllbConfig, device: &Device) -> CandleResult<Self> {
        let model_vb = vb.pp("model");

        let shared_embeddings = embedding(cfg.vocab_size, cfg.d_model, model_vb.pp("shared"))?;

        let weight_dtype = shared_embeddings.embeddings().dtype();
        let pos_embeddings =
            sinusoidal_embeddings(cfg.max_position_embeddings, cfg.d_model, 2, weight_dtype, device)?;

        let mut encoder_layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            encoder_layers.push(EncoderLayer::load(
                model_vb.pp(format!("encoder.layers.{}", i)),
                cfg,
            )?);
        }

        let mut decoder_layers = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            decoder_layers.push(DecoderLayer::load(
                model_vb.pp(format!("decoder.layers.{}", i)),
                cfg,
            )?);
        }

        let encoder_layer_norm =
            manual_layer_norm(cfg.d_model, 1e-5, model_vb.pp("encoder.layer_norm"))?;
        let decoder_layer_norm =
            manual_layer_norm(cfg.d_model, 1e-5, model_vb.pp("decoder.layer_norm"))?;

        // lm_head is tied to shared embeddings
        let lm_head = {
            let shared_weight = shared_embeddings.embeddings();
            Linear::new(shared_weight.clone(), None)
        };

        Ok(Self {
            shared_embeddings,
            encoder_layers,
            decoder_layers,
            encoder_layer_norm,
            decoder_layer_norm,
            lm_head,
            pos_embeddings,
            config: cfg.clone(),
        })
    }

    pub fn encode(&self, input_ids: &Tensor) -> CandleResult<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let scale = if self.config.scale_embedding {
            (self.config.d_model as f64).sqrt()
        } else {
            1.0
        };

        let token_emb = (self.shared_embeddings.forward(input_ids)? * scale)?;
        let pos_emb = self.pos_embeddings.i(2..seq_len + 2)?;
        let mut x = token_emb.broadcast_add(&pos_emb)?;

        for layer in &self.encoder_layers {
            x = layer.forward(&x)?;
        }

        self.encoder_layer_norm.forward(&x)
    }

    pub fn decode(
        &self,
        decoder_input_ids: &Tensor,
        encoder_output: &Tensor,
    ) -> CandleResult<Tensor> {
        let seq_len = decoder_input_ids.dim(1)?;
        let scale = if self.config.scale_embedding {
            (self.config.d_model as f64).sqrt()
        } else {
            1.0
        };

        let token_emb = (self.shared_embeddings.forward(decoder_input_ids)? * scale)?;
        let pos_emb = self.pos_embeddings.i(2..seq_len + 2)?;
        let mut x = token_emb.broadcast_add(&pos_emb)?;

        let causal_mask = self.causal_mask(seq_len, x.dtype(), x.device())?;

        for layer in &self.decoder_layers {
            x = layer.forward(&x, encoder_output, Some(&causal_mask))?;
        }

        let x = self.decoder_layer_norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    fn causal_mask(&self, size: usize, dtype: DType, device: &Device) -> CandleResult<Tensor> {
        let mask: Vec<f32> = (0..size)
            .flat_map(|i| {
                (0..size).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY })
            })
            .collect();
        Tensor::from_vec(mask, (1, 1, size, size), device)?.to_dtype(dtype)
    }

    pub fn generate(
        &self,
        encoder_output: &Tensor,
        forced_bos_token_id: u32,
        max_length: usize,
    ) -> CandleResult<Vec<u32>> {
        let device = encoder_output.device();
        let mut token_ids: Vec<u32> = vec![self.config.decoder_start_token_id, forced_bos_token_id];

        for _ in 0..max_length {
            let decoder_input =
                Tensor::from_vec(token_ids.clone(), (1, token_ids.len()), device)?;

            let logits = self.decode(&decoder_input, encoder_output)?;

            let last_logits = logits.i((.., logits.dim(1)? - 1, ..))?;
            let next_token = last_logits.argmax(D::Minus1)?.to_vec1::<u32>()?[0];

            if next_token == self.config.eos_token_id {
                break;
            }

            token_ids.push(next_token);
        }

        Ok(token_ids[2..].to_vec())
    }
}
