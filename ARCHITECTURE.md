# lingo Architecture

## Overview

lingo implements Meta's NLLB-200 translation model as a native Rust encoder-decoder transformer using the candle ML framework, with LaBSE and SONAR encoders for cross-lingual similarity scoring. This document covers the internal architecture, design decisions, and Metal GPU workarounds.

## Module Structure

```
lingo/
├── src/
│   ├── lib.rs          # Public API: NllbTranslator, LaBSEEncoder, SonarEncoder
│   ├── model.rs        # M2M100 encoder-decoder transformer
│   │   ├── NllbConfig           # Model hyperparameters
│   │   ├── NllbModel            # Full encoder-decoder model
│   │   ├── EncoderLayer         # Pre-norm self-attention + FFN
│   │   ├── DecoderLayer         # Pre-norm self-attn + cross-attn + FFN
│   │   ├── MultiHeadAttention   # Scaled dot-product attention
│   │   └── ManualLayerNorm      # Metal-compatible layer normalization
│   ├── labse.rs        # LaBSE sentence encoder (768-dim embeddings)
│   ├── sonar.rs        # SONAR sentence encoder (1024-dim embeddings)
│   ├── languages.rs    # NllbLanguage enum (208 languages)
│   ├── download.rs     # HuggingFace Hub model downloading
│   └── error.rs        # Error types
```

## Model Architecture

### M2M100 Encoder-Decoder

NLLB-200-distilled-600M uses the M2M100 architecture, which is a standard transformer encoder-decoder with these specifics:

```
                    ┌─────────────────────┐
                    │   Shared Embedding   │ (256,206 x 1024, tied weights)
                    └──────────┬──────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
    ┌─────────▼─────────┐           ┌──────────▼──────────┐
    │     Encoder        │           │      Decoder         │
    │  (12 layers)       │           │   (12 layers)        │
    │                    │           │                      │
    │  ┌──────────────┐  │           │  ┌────────────────┐  │
    │  │ Layer Norm    │  │           │  │  Layer Norm     │  │
    │  │ Self-Attn     │  │           │  │  Causal Attn    │  │
    │  │ + Residual    │  │           │  │  + Residual     │  │
    │  │              │  │           │  │                │  │
    │  │ Layer Norm    │  │           │  │  Layer Norm     │  │
    │  │ FFN (ReLU)    │  │           │  │  Cross-Attn     │  │
    │  │ + Residual    │  │     ┌────▶│  │  + Residual     │  │
    │  └──────────────┘  │     │     │  │                │  │
    │  (repeat x12)      │     │     │  │  Layer Norm     │  │
    │                    │     │     │  │  FFN (ReLU)     │  │
    │  Final Layer Norm  │─────┘     │  │  + Residual     │  │
    └────────────────────┘           │  └────────────────┘  │
                                     │  (repeat x12)        │
                                     │                      │
                                     │  Final Layer Norm    │
                                     └──────────┬───────────┘
                                                │
                                     ┌──────────▼───────────┐
                                     │   LM Head (tied)     │
                                     │   argmax → token ID  │
                                     └──────────────────────┘
```

### Key Design Choices

**Pre-layer normalization**: NLLB applies layer norm *before* attention and FFN, not after. This differs from the original "Attention Is All You Need" paper but is standard in modern large transformers for training stability.

**Sinusoidal positional embeddings with offset=2**: M2M100 uses fixed (non-learned) positional embeddings with positions starting at index 2 (not 0). This is an M2M100-specific convention.

**Weight tying**: The embedding matrix is shared across four roles:
- `model.shared.weight` (canonical copy)
- `model.encoder.embed_tokens.weight` (alias)
- `model.decoder.embed_tokens.weight` (alias)
- `lm_head.weight` (alias)

In safetensors, we store only `model.shared.weight` and reconstruct the others at load time.

**Forced BOS token**: The decoder is primed with `[decoder_start_token_id, target_language_token]` to tell the model which language to translate into. For example, `[2, 256047]` for French (`fra_Latn`).

## Metal GPU Workarounds

Candle's Metal backend (v0.10.x) has incomplete kernel coverage. lingo works around three gaps:

### 1. LayerNorm

**Problem**: `candle_nn::LayerNorm` uses a specialized `layer-norm` Metal kernel that doesn't exist.

**Solution**: `ManualLayerNorm` implements layer norm using primitive ops:
```rust
fn forward(&self, x: &Tensor) -> Result<Tensor> {
    let x = x.to_dtype(F32)?;         // upcast for numerical stability
    let mean = x.mean_keepdim(-1)?;
    let x_centered = x - mean;
    let var = x_centered.sqr()?.mean_keepdim(-1)?;
    let x_normed = x_centered / (var + eps)?.sqrt()?;
    x_normed.to_dtype(orig)? * weight + bias
}
```

### 2. Softmax

**Problem**: `candle_nn::ops::softmax_last_dim` uses a Metal kernel that doesn't exist.

**Solution**: Manual softmax decomposition:
```rust
let max_vals = x.max_keepdim(-1)?;
let exp = (x - max_vals)?.exp()?;
let sum = exp.sum_keepdim(-1)?;
exp / sum
```

### 3. Matmul contiguity

**Problem**: Metal matmul requires contiguous tensor layout. After `transpose(1, 2)` in attention head reshaping, tensors are strided (non-contiguous).

**Solution**: Explicit `.contiguous()` calls before every matmul:
```rust
let q = q.transpose(1, 2)?.contiguous()?;
let k_t = k.transpose(2, 3)?.contiguous()?;
let attn = q.matmul(&k_t)?;
```

### 4. F32 inference

**Problem**: Many Metal kernels only support F32 (not F16/BF16).

**Solution**: Load model weights in F32 regardless of safetensors storage format. The distilled-600M model uses ~2.4 GB in F32, which fits easily in Apple Silicon's unified memory (minimum 8 GB on any M-series chip).

## Decoding Strategy

Currently implements **greedy decoding** (argmax at each step). The generation loop:

1. Initialize decoder input with `[eos_token_id, forced_bos_token_id]`
2. Run full decoder forward pass
3. Take argmax of last position logits
4. Append new token to decoder input
5. Repeat until EOS or max_length (2048)

### Future improvements

- **Beam search**: Would improve translation quality at the cost of ~4x latency
- **KV-cache**: Store key/value projections from previous steps to avoid recomputation. Currently each generation step recomputes the full decoder, making it O(n^2) in sequence length. KV-cache would make it O(n).

## Memory Layout

On Apple Silicon with unified memory:

| Component | Size (F32) | Notes |
|-----------|-----------|-------|
| Shared embeddings | 1.0 GB | 256,206 x 1024 |
| Encoder layers (12) | 0.6 GB | Self-attn + FFN per layer |
| Decoder layers (12) | 0.8 GB | Self-attn + cross-attn + FFN |
| Sinusoidal embeddings | 4 MB | 1026 x 1024 |
| **Total** | **~2.4 GB** | Fits on any Apple Silicon Mac |

## Tokenizer

Uses the HuggingFace `tokenizers` crate to load `tokenizer.json` (SentencePiece BPE with 256,206 tokens). The tokenizer handles:

- Text encoding with special token insertion
- Language code tokens (e.g., `eng_Latn`, `fra_Latn`) as part of the vocabulary
- Decoding with special token stripping
