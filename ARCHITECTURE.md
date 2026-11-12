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
│   ├── languages.rs    # NllbLanguage enum (the 202 NLLB-200 languages)
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

**Sinusoidal positional embeddings with offset=2**: M2M100 uses fixed (non-learned) positional embeddings with positions starting at index 2 (not 0). This is an M2M100-specific convention. Frequencies follow HF/fairseq exactly: `freq_i = exp(-i * ln(10000) / (half - 1))` (an earlier version divided by `half`, a small deviation from the trained model). SONAR uses the same table (`model::sinusoidal_embeddings`, 512 + 2 rows) — fairseq2's position encoder is a non-persistent buffer, so it is not in the checkpoint and must be generated.

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

### 5. Command-buffer completion wedge (known candle bug)

**Problem**: Under sustained multi-Metal-device load (e.g. an embedding
encoder and a translation decoder both submitting command buffers from
different tokio worker threads, or multiple processes sharing the same
`MTLDevice`), candle 0.10's Metal backend can submit a `MTLCommandBuffer`
without its completion handler firing. The calling thread blocks
indefinitely in `MetalDevice::wait_until_completed` →
`[_MTLCommandBuffer waitUntilCompleted]` → `_pthread_cond_wait`. macOS
reports no GPU/Metal errors — the hardware is fine; candle's bookkeeping
loses track of the completion notification under a specific timing
window.

**Diagnostic signature**: process CPU drops to <10% but stack samples
show threads parked in `wait_until_completed`. No `IOAccel` / `AGX` /
`Metal` subsystem events in `log show` at the wedge time.

**Lingo-layer mitigations** (implemented):

1. **Timeout-bounded API surface.**
   `NllbTranslator::translate_with_timeout`,
   `SonarEncoder::{embed,score}_with_timeout`, and
   `LaBSEEncoder::{embed,score}_with_timeout` (plus SONAR's
   `{embed_lang,score_langs}_with_timeout`) wrap the corresponding
   plain methods in `tokio::time::timeout` and return
   `Error::Timeout(Duration)` on expiry. All inference runs inside
   `tokio::task::spawn_blocking`, which is what lets the timeout fire
   while a GPU call is stuck — a synchronous call inside the future
   would never yield to the timer. Callers should always prefer
   the `_with_timeout` variants in production.

2. **Cooperative cancellation + `wait_idle`.** `run_blocking` hands each
   blocking job a `CancelFlag` that a drop guard sets when the awaiting
   future is dropped (timeout or otherwise). Jobs check it after taking
   the model lock, between strict-mode units / chunks, on every decode
   step (`StopReason::Cancelled`) and between SONAR/LaBSE windows, so an
   abandoned call releases the model within one step. `wait_idle(grace)`
   waits for the model mutex: `true` → reuse the instance; `false` → the
   thread is inside a GPU call that never returns (the Metal wedge, which
   no flag can interrupt), so the instance is poisoned and must be
   replaced. The example server keeps or recycles on exactly that test.

3. **Single-encoder-per-device guidance.** AGENTS.md "Concurrent Metal
   hazards" warns against spinning up multiple lingo encoders pointing
   at the same `MTLDevice`. The recommended pattern is one `Arc`-shared
   encoder per model per process.

**Mitigations not yet implemented** (potential future work):

- **Dedicated Metal worker thread.** Pin every Metal command-buffer
  submission to a single OS thread fed by an MPSC channel. Eliminates
  cross-thread submission races at the cost of a ~150-LOC refactor
  that changes the public API shape. Would be scoped as a separate
  `METAL-WORKER-*` playbook if timeout telemetry shows the wedges
  recurring at meaningful rates.
- **Upstream candle fix.** The root cause lives in
  `candle_core::metal_backend::device` — the `setCompletedHandler`
  block can be deallocated before it fires. A candle issue / PR is
  the only path to fully closing the bug.

## Decoding Strategy

Implements **greedy decoding** (argmax at each step) with a **KV cache** that
keeps per-step work O(1) per decoder layer instead of O(N). The generation
loop:

1. Construct a fresh `DecoderCache` (per-layer self-attn + cross-attn slots).
2. Precompute cross-attention K/V from the encoder output once, fill every
   layer's `cross_attn_kv` slot.
3. Run a single full-prefix forward pass on the 2-token seed
   `[decoder_start_token_id, forced_bos_token_id]` to populate self-attn K/V
   for the start positions.
4. Loop with `NllbModel::decode_step(new_token, &mut cache, encoder_output,
   step_index)`:
   - One-row token + position embedding for the new token only.
   - Each layer's self-attention reads `cache.layers[i].self_attn_kv` (the
     prefix), computes K/V for the single new token, concatenates, attends,
     and writes the grown `(K, V)` back into the cache.
   - Cross-attention reuses the precomputed K/V — no recomputation per step.
   - LM head produces logits for the single new token.
5. Argmax the logits. Break on EOS, else push the new token and continue.
6. Position-budget guard breaks the loop if `cache.self_attn_seq_len()`
   reaches `max_position_embeddings (1024)` without an EOS, emitting a
   `tracing::warn!` so callers can detect potential truncation.

`max_length` passed to `generate` is clamped to `max_position_embeddings - 1`
(the seed consumes 2 of the 1024 positions, leaving room for up to 1022
generated tokens before the hard architectural ceiling).

### KV Cache

The KV cache reduces per-token decoder work from O(N) to O(1) by storing
per-layer key/value tensors across iterations. Concretely, at output step N
without a cache the decoder would re-run all 12 layers on the full N-token
prefix (O(N²) total over an N-token output); with the cache, each step only
projects K/V for the single new token, concatenates with the cached prefix,
attends, and writes the grown cache back. Cross-attention K/V is computed
once from the encoder output and reused across all output tokens (the encoder
output is fixed for the whole generation).

Types in `src/model.rs`:

```rust
pub struct LayerCache {
    pub self_attn_kv: Option<(Tensor, Tensor)>,   // grows by 1 column per step
    pub cross_attn_kv: Option<(Tensor, Tensor)>,  // computed once, reused
}

pub struct DecoderCache {
    pub layers: Vec<LayerCache>,
}
```

Threading: `MultiHeadAttention::forward` accepts `Option<&(Tensor, Tensor)>`
past K/V and returns the updated `(K, V)` for the caller to store. For
self-attention the new K/V is concatenated with the cached prefix along the
sequence axis (always followed by `.contiguous()` for Metal compatibility).
For cross-attention, when both `kv_source` and `past_kv` are provided, the
K/V projection is skipped entirely — the cached tensors are reused as-is.

Reference implementation: `candle-transformers::models::whisper`
(`MultiHeadAttention::forward` taking optional past K/V; per-layer cache
threading through the decoder loop). Whisper is the same framework, same
Metal/CUDA backend, and same encoder-decoder shape — the lingo
implementation transliterates that pattern.

Correctness is enforced by a side-by-side numeric-equivalence test
(`tests/kv_cache_equivalence.rs`) that translates 20 inputs across 10
language pairs with both `generate` (KV-cached) and `generate_reference`
(full-prefix), asserting the output token sequences are identical. The
reference path is kept available via `NllbTranslator::translate_reference`
(`#[doc(hidden)]`) specifically for this diff, and bypasses the bounded
translation cache so the comparison is honest.

### Future improvements

- **Beam search**: Would improve translation quality at the cost of ~4x latency.
- **Speculative decoding**: Would give 2–3× speedup with zero accuracy cost
  (greedy = provably identical), blocked on the existence of a draft model
  smaller than nllb-200-distilled-600M sharing the 256k vocab. See
  KV-CACHE-06-DECISION.md.
- **GGUF / Q8_0 weight quantization**: Would cut model footprint and
  cold-load time but not warm-decode latency on a GPU-resident 600M model.
  Deferred pending an accuracy harness. See KV-CACHE-06-DECISION.md.

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

- Text encoding. lingo encodes *without* the tokenizer's template, because
  that template hard-codes `eng_Latn`; it frames the input itself as
  `[src_lang] tokens... [</s>]`, so `--from`/`source` actually reaches the
  encoder.
- Language code tokens (e.g., `eng_Latn`, `fra_Latn`) as part of the vocabulary
- Decoding with special token stripping

### SONAR tokenization

SONAR's text encoder uses fairseq2's NLLB vocabulary: the same SentencePiece
model as NLLB (byte-identical), with `<pad>` inserted at index 0
(`<pad>=0, <unk>=1, <s>=2, </s>=3`, pieces at `spm_id + 1`) and the 202
`__lang__` symbols from 256001 — 256206 embedding rows. The HF NLLB
`tokenizer.json` assigns identical piece and language IDs and differs only in
the order of the four specials (`<s>=0, <pad>=1, </s>=2, <unk>=3`), so lingo
loads it and remaps those four. Inputs are framed `[lang] tokens... [</s>=3]`
and truncated to 512 positions. Before this was fixed, lingo fed SONAR raw
SentencePiece IDs (off by one), no language/EOS tokens and no positional
encoding, which made embeddings word-order-invariant and depressed true-pair
scores (en/fr "Hello world" pair: 0.744 → 0.863).

### Long-text encoding and translation integrity

- **Windowed encoding** (`src/windows.rs`, ported from SlopShift's fork):
  SONAR and LaBSE split content tokens into windows of
  `context - specials` tokens, re-frame each with the original leading /
  trailing specials, and combine window embeddings as a content-token-weighted
  mean followed by L2 normalization. Replaces truncation, which silently
  ignored everything past ~510 tokens.
- **Repetition guard** (`model::repetition_tail`): both `generate` and
  `generate_reference` stop when the last `max(ceil(64/p), 4) * p` tokens are
  one `p`-token period repeated (p ≤ 64), trimming to a single period. Applied
  identically on both paths, so the KV-cache equivalence test still holds.
  Generation reports a `StopReason` (Eos / Budget / Repetition).
- **`TranslateConfig::strict()`**: sentence/line units via
  `chunker::sentence_units`, plus rejection (`Error::Degenerate`) of empty
  output, dangling clauses and repetition loops.
- **Language resolution** (`NllbLanguage::resolve`): accepts ISO codes,
  aliases and NLLB codes. The table is exactly NLLB-200's 202 language
  tokens, generated from one `languages!` macro table and unit-tested against
  transformers' `FAIRSEQ_LANGUAGE_CODES` (and the local tokenizer when
  present). The old 208-entry table had 42 `eng_Latn` placeholders (silent
  English output), 5 entries pointing at another language's code
  (`SerbianLatin`→`srp_Cyrl`, `Balochi`→`pbt_Arab`, …), 5 synonym duplicates,
  mislabeled `AzerbaijaniCyrillic` (`azb_Arab`) / `TajikLatin` (`tgk_Cyrl`),
  and was missing 46 NLLB languages (Belarusian, Tibetan, Cantonese, Uyghur,
  Central Kurdish, Nynorsk, …).

