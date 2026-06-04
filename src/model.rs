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

    /// Attention forward with optional KV cache.
    ///
    /// Returns `(output, new_kv)` where `new_kv` is the K/V state to cache for
    /// the next decode step. Mirrors the candle Whisper KV-cache pattern.
    ///
    /// - `kv_source = None` (self-attention): K, V are computed from `x`. When
    ///   `past_kv` is `Some`, the new K/V are concatenated onto the cached
    ///   prefix along the seq_len axis before attending.
    /// - `kv_source = Some(enc)` (cross-attention): K, V are computed from
    ///   `enc` only on the first step (`past_kv = None`). When `past_kv` is
    ///   `Some`, it is reused verbatim — the encoder output is stable across
    ///   decode steps, so there is nothing to recompute.
    /// - Q is always computed from `x`.
    fn forward(
        &self,
        x: &Tensor,
        kv_source: Option<&Tensor>,
        past_kv: Option<&(Tensor, Tensor)>,
        mask: Option<&Tensor>,
    ) -> CandleResult<(Tensor, Option<(Tensor, Tensor)>)> {
        let (b, tgt_len, _) = x.dims3()?;

        // Q is always derived from the query input x.
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, tgt_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Compute (or reuse) K and V — this becomes the new cache state.
        let (k, v) = if kv_source.is_some() {
            // Cross-attention: K/V come from the encoder output, which does not
            // change between decode steps. Reuse the cache when we have it.
            if let Some((past_k, past_v)) = past_kv {
                (past_k.clone(), past_v.clone())
            } else {
                let enc = kv_source.unwrap();
                let src_len = enc.dim(1)?;
                let k = self
                    .k_proj
                    .forward(enc)?
                    .reshape((b, src_len, self.num_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                let v = self
                    .v_proj
                    .forward(enc)?
                    .reshape((b, src_len, self.num_heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?;
                (k, v)
            }
        } else {
            // Self-attention: K/V come from x. Concatenate onto any cached prefix.
            let new_k = self
                .k_proj
                .forward(x)?
                .reshape((b, tgt_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            let new_v = self
                .v_proj
                .forward(x)?
                .reshape((b, tgt_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            if let Some((past_k, past_v)) = past_kv {
                // Metal requires contiguous tensors for matmul; cat then make
                // the result contiguous along the seq_len axis (dim 2).
                let k = Tensor::cat(&[past_k, &new_k], 2)?.contiguous()?;
                let v = Tensor::cat(&[past_v, &new_v], 2)?.contiguous()?;
                (k, v)
            } else {
                (new_k, new_v)
            }
        };

        let scale = (self.head_dim as f64).sqrt();
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

        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b, tgt_len, self.num_heads * self.head_dim))?
            .apply(&self.out_proj)?;

        Ok((attn_output, Some((k, v))))
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
        let (x, _) = self.self_attn.forward(&x, None, None, None)?;
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
        let (x, _) = self.self_attn.forward(&x, None, None, causal_mask)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.encoder_attn_layer_norm.forward(&x)?;
        let (x, _) = self
            .encoder_attn
            .forward(&x, Some(encoder_output), None, None)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.fc1.forward(&x)?.relu()?;
        let x = self.fc2.forward(&x)?;
        residual + x
    }
}

// ---------------------------------------------------------------------------
// KV cache (per-layer self- and cross-attention state for decoding)
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct LayerCache {
    /// Self-attention K and V from all prior decode steps for this layer.
    /// Shape (b, num_heads, seq_so_far, head_dim). None on first step.
    pub self_attn_kv: Option<(Tensor, Tensor)>,
    /// Cross-attention K and V — computed once from encoder output,
    /// reused across all decode steps for this layer. None until
    /// KV-CACHE-03 fills it.
    pub cross_attn_kv: Option<(Tensor, Tensor)>,
}

pub struct DecoderCache {
    pub layers: Vec<LayerCache>,
}

impl DecoderCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerCache::default()).collect(),
        }
    }
    /// Length of cached self-attention sequence (same for all layers).
    pub fn self_attn_seq_len(&self) -> usize {
        self.layers
            .first()
            .and_then(|l| l.self_attn_kv.as_ref())
            .map(|(k, _)| k.dim(2).unwrap_or(0))
            .unwrap_or(0)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `MultiHeadAttention` directly from random `Linear` weights,
    /// bypassing `MultiHeadAttention::load` (which needs a real VarBuilder).
    fn build_attention(d_model: usize, num_heads: usize, device: &Device) -> MultiHeadAttention {
        let head_dim = d_model / num_heads;
        let inner = num_heads * head_dim;
        let rand_linear = |out: usize, inp: usize| {
            let w = Tensor::randn(0f32, 0.1f32, (out, inp), device).unwrap();
            Linear::new(w, None)
        };
        MultiHeadAttention {
            q_proj: rand_linear(inner, d_model),
            k_proj: rand_linear(inner, d_model),
            v_proj: rand_linear(inner, d_model),
            out_proj: rand_linear(d_model, inner),
            num_heads,
            head_dim,
        }
    }

    /// Max absolute elementwise difference between two tensors, as f32.
    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn attention_self_first_step_returns_kv() {
        let device = Device::Cpu;
        let (b, seq, d_model, heads) = (1usize, 4usize, 8usize, 2usize);
        let head_dim = d_model / heads;
        let attn = build_attention(d_model, heads, &device);
        let x = Tensor::randn(0f32, 1f32, (b, seq, d_model), &device).unwrap();

        let (out, kv) = attn.forward(&x, None, None, None).unwrap();

        assert_eq!(out.dims3().unwrap(), (b, seq, d_model));
        let (k, v) = kv.expect("self-attention first step must return cached K/V");
        assert_eq!(k.dims4().unwrap(), (b, heads, seq, head_dim));
        assert_eq!(v.dims4().unwrap(), (b, heads, seq, head_dim));
    }

    #[test]
    fn attention_self_continuation_concats_cache() {
        let device = Device::Cpu;
        let (b, d_model, heads) = (1usize, 8usize, 2usize);
        let head_dim = d_model / heads;
        let attn = build_attention(d_model, heads, &device);

        // First step processes 3 tokens.
        let x0 = Tensor::randn(0f32, 1f32, (b, 3, d_model), &device).unwrap();
        let (_, kv0) = attn.forward(&x0, None, None, None).unwrap();
        let prior = kv0.expect("first step returns cache");
        assert_eq!(prior.0.dim(2).unwrap(), 3);

        // Continuation: 1 new token, concatenated onto the cached prefix of 3.
        let x1 = Tensor::randn(0f32, 1f32, (b, 1, d_model), &device).unwrap();
        let (out, kv1) = attn.forward(&x1, None, Some(&prior), None).unwrap();
        let (k, v) = kv1.expect("continuation returns updated cache");

        assert_eq!(out.dims3().unwrap(), (b, 1, d_model));
        assert_eq!(k.dim(2).unwrap(), prior.0.dim(2).unwrap() + 1);
        assert_eq!(k.dims4().unwrap(), (b, heads, 4, head_dim));
        assert_eq!(v.dims4().unwrap(), (b, heads, 4, head_dim));
    }

    #[test]
    fn attention_cross_first_step_computes_from_kv_source() {
        let device = Device::Cpu;
        let (b, d_model, heads) = (1usize, 8usize, 2usize);
        let head_dim = d_model / heads;
        let attn = build_attention(d_model, heads, &device);

        let x = Tensor::randn(0f32, 1f32, (b, 2, d_model), &device).unwrap();
        let enc = Tensor::randn(0f32, 1f32, (b, 5, d_model), &device).unwrap();

        let (out, kv) = attn.forward(&x, Some(&enc), None, None).unwrap();
        let (k, v) = kv.expect("cross-attention first step returns cache");

        // K/V seq_len follows the encoder source (5), not the query (2).
        assert_eq!(out.dims3().unwrap(), (b, 2, d_model));
        assert_eq!(k.dims4().unwrap(), (b, heads, 5, head_dim));
        assert_eq!(v.dims4().unwrap(), (b, heads, 5, head_dim));

        // The cached K must be a direct projection of the encoder output.
        let expected_k = attn
            .k_proj
            .forward(&enc)
            .unwrap()
            .reshape((b, 5, heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap();
        assert!(
            max_abs_diff(&k, &expected_k) < 1e-5,
            "cross-attn K must derive from kv_source"
        );
    }

    #[test]
    fn attention_cross_continuation_reuses_cache() {
        let device = Device::Cpu;
        let (b, d_model, heads) = (1usize, 8usize, 2usize);
        let attn = build_attention(d_model, heads, &device);

        let x0 = Tensor::randn(0f32, 1f32, (b, 2, d_model), &device).unwrap();
        let enc = Tensor::randn(0f32, 1f32, (b, 5, d_model), &device).unwrap();
        let (_, kv0) = attn.forward(&x0, Some(&enc), None, None).unwrap();
        let prior = kv0.expect("first step returns cache");

        // Continuation with a fresh query token but the same encoder cache.
        let x1 = Tensor::randn(0f32, 1f32, (b, 1, d_model), &device).unwrap();
        let (_, kv1) = attn.forward(&x1, Some(&enc), Some(&prior), None).unwrap();
        let (k, v) = kv1.expect("continuation returns cache");

        // The returned K/V must be the *same* data as the prior cache — proving
        // cross-attention K/V was reused, not recomputed.
        assert_eq!(k.dims4().unwrap(), prior.0.dims4().unwrap());
        assert_eq!(
            max_abs_diff(&k, &prior.0),
            0.0,
            "reused cross-attn K must be identical to the prior cache"
        );
        assert_eq!(
            max_abs_diff(&v, &prior.1),
            0.0,
            "reused cross-attn V must be identical to the prior cache"
        );
    }

    #[test]
    fn attention_no_cache_matches_pre_kv_behaviour() {
        // Regression: the no-cache self-attention path must compute exactly what
        // the original (pre-KV-cache) implementation did. The original algorithm
        // is reproduced inline here and compared against the new code path.
        let device = Device::Cpu;
        let (b, seq, d_model, heads) = (1usize, 4usize, 8usize, 2usize);
        let head_dim = d_model / heads;
        let attn = build_attention(d_model, heads, &device);
        let x = Tensor::randn(0f32, 1f32, (b, seq, d_model), &device).unwrap();

        let (got, _) = attn.forward(&x, None, None, None).unwrap();

        // --- original pre-cache code path, inlined ---
        let q = attn.q_proj.forward(&x).unwrap();
        let k = attn.k_proj.forward(&x).unwrap();
        let v = attn.v_proj.forward(&x).unwrap();
        let q = q
            .reshape((b, seq, heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let k = k
            .reshape((b, seq, heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let v = v
            .reshape((b, seq, heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let scale = (head_dim as f64).sqrt();
        let q = q.contiguous().unwrap();
        let k = k.contiguous().unwrap();
        let v = v.contiguous().unwrap();
        let attn_raw = q
            .matmul(&k.transpose(2, 3).unwrap().contiguous().unwrap())
            .unwrap();
        let attn_weights = (attn_raw * (1.0 / scale)).unwrap();
        let attn_weights = {
            let max_vals = attn_weights.max_keepdim(D::Minus1).unwrap();
            let exp = (attn_weights.broadcast_sub(&max_vals))
                .unwrap()
                .exp()
                .unwrap();
            let sum = exp.sum_keepdim(D::Minus1).unwrap();
            exp.broadcast_div(&sum).unwrap()
        };
        let attn_output = attn_weights.matmul(&v).unwrap();
        let expected = attn_output
            .transpose(1, 2)
            .unwrap()
            .reshape((b, seq, heads * head_dim))
            .unwrap()
            .apply(&attn.out_proj)
            .unwrap();

        assert!(
            max_abs_diff(&got, &expected) < 1e-5,
            "no-cache path diverged from the original implementation"
        );
    }

    #[test]
    fn decoder_cache_tracks_self_attn_seq_len() {
        // Sanity check for the new DecoderCache helper.
        let device = Device::Cpu;
        let mut cache = DecoderCache::new(2);
        assert_eq!(cache.layers.len(), 2);
        assert_eq!(cache.self_attn_seq_len(), 0);

        // Populate layer 0 with a (b, heads, seq, head_dim) self-attn cache.
        let k = Tensor::randn(0f32, 1f32, (1usize, 2usize, 3usize, 4usize), &device).unwrap();
        let v = k.clone();
        cache.layers[0].self_attn_kv = Some((k, v));
        assert_eq!(cache.self_attn_seq_len(), 3);
    }
}
