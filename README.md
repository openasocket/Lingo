# lingo

Pure Rust multilingual NLP toolkit built on [candle](https://github.com/huggingface/candle). Translates text using the NLLB-200 model and performs similarity calculations on translated text against pre-translated text using LaBSE and SONAR. No Python required.

## What It Does

| Capability | Model | Description |
|---|---|---|
| **Translation** | NLLB-200 | Translate between all 202 NLLB-200 languages |
| **Similarity** | LaBSE | Cross-lingual semantic similarity scoring (768-dim) |
| **Similarity** | SONAR | Cross-lingual sentence embeddings (1024-dim) |

All models run on Apple Metal GPU, NVIDIA CUDA, or CPU.

## Use as a Rust Library

```toml
[dependencies]
lingo = { version = "0.1", features = ["metal"] }  # macOS
# lingo = { version = "0.1", features = ["cuda"] } # Linux NVIDIA
tokio = { version = "1", features = ["full"] }
```

### Translation

```rust
use lingo::NllbTranslator;

let translator = NllbTranslator::new(None)?;
let result = translator.translate("Hello, how are you?", "en", "fr").await?;
println!("{}", result.text);       // "Bonjour, comment allez-vous ?"
println!("{}ms", result.duration_ms); // ~130ms
println!("{} chunks", result.chunk_count); // 1 for short input, >1 for chunked
```

### Similarity Scoring (LaBSE)

```rust
use lingo::LaBSEEncoder;

let encoder = LaBSEEncoder::new(None)?;

// Cross-lingual similarity
let score = encoder.score("Hello world", "Bonjour le monde").await?;
println!("{:.3}", score); // ~0.85

// Get raw embeddings (768-dim)
let embedding = encoder.embed("Hello world").await?;
```

### Similarity Scoring (SONAR)

```rust
use lingo::SonarEncoder;

let encoder = SonarEncoder::new(None)?;

// Cross-lingual similarity (1024-dim embeddings)
let score = encoder.score("Hello world", "Bonjour le monde").await?;
println!("{:.3}", score);

// Get raw embeddings (1024-dim)
let embedding = encoder.embed("Hello world").await?;
```

### Language Codes

```rust
use lingo::NllbLanguage;

let lang = NllbLanguage::from_iso_code("fr").unwrap();
println!("{}: {}", lang.name(), lang.nllb_code()); // "French: fra_Latn"

for lang in NllbLanguage::all_languages() {
    println!("{}: {}", lang.iso_code(), lang.name());
}
```

## Use as a CLI Tool

```bash
# macOS (Apple Silicon)
cargo install lingo --features cli,metal

# Linux (NVIDIA GPU) — requires CUDA toolkit
CUDA_HOME=/usr/local/cuda-13.0 cargo install lingo --features cli,cuda

# CPU only
cargo install lingo --features cli
```

### Translate

```bash
lingo translate "Hello world" --to fr
# [fr] Bonjour le monde (135ms)

lingo translate "Good morning" --to fr,es,ja,ar,ko

echo "Hello world" | lingo translate --to fr

lingo translate "Hello" --to fr,es --json
```

### Score Similarity (LaBSE)

```bash
lingo score "Hello world" "Bonjour le monde"
# 0.9471
```

### Score Similarity (SONAR)

```bash
lingo sonar-score "Hello world" "Bonjour le monde" --lang1 en --lang2 fr
# 0.8626   (SONAR is language-conditioned; --lang1/--lang2 default to en)
```

### Embed (LaBSE)

```bash
lingo embed "Hello world"
# [0.0234, -0.0891, 0.0412, ...]  (768 floats)
```

### Embed (SONAR)

```bash
lingo sonar-embed "Hello world" --lang en
# [0.0123, -0.0456, 0.0789, ...]  (1024 floats)
```

### Download Models

```bash
lingo download all     # Download all models (no license acceptance needed to download)
lingo download sonar   # Download just SONAR (also upgrades a legacy SONAR tokenizer)
lingo download labse --model-dir /data/labse   # Custom location
```

### List Languages

```bash
lingo languages
```

## Use as an HTTP Server

```bash
LINGO_ACCEPT_LICENSE=1 cargo run --release --example server --features server,metal
# Listens on 127.0.0.1:3000 (set LINGO_SERVER_ADDR=0.0.0.0:3000 to expose it —
# there is no authentication, so only do that behind a trusted proxy/firewall)
```

Limits are configurable via `LINGO_REQUEST_TIMEOUT_SECS` (300), `LINGO_MAX_PENDING`
(32), `LINGO_MAX_BATCH` (256) and `LINGO_MAX_TEXT_BYTES` (65536). Over-limit
input → 400, busy or model unavailable → 503, timeout → 504 (the instance is kept if
its abandoned work stops within 5 s, and replaced automatically if it is
wedged in a GPU call).

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/translate` | `{"text", "source", "target"}` -> translation + chunk_count metadata |
| POST | `/translate_batch` | `{"texts": [...], "source", "target"}` -> batch translation, no scoring + per-item chunk_counts |
| POST | `/score` | `{"text1", "text2"}` -> similarity score |
| POST | `/embed` | `{"text"}` -> 768-dim embedding |
| POST | `/embed_batch` | `{"texts": [...]}` -> batch embeddings |
| POST | `/analyze` | `{"text", "source", "target"}` -> translation + LaBSE & SONAR scores + chunk_count metadata |
| POST | `/analyze_batch` | `{"texts": [...], "source", "target"}` -> batch translate + score + per-item chunk_counts |
| GET | `/health` | `*_loaded` (in memory) and `*_downloaded` (on disk) per model |

```bash
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source": "en", "target": "fr"}'

curl -X POST http://localhost:3000/score \
  -H "Content-Type: application/json" \
  -d '{"text1": "Hello world", "text2": "Bonjour le monde"}'

curl -X POST http://localhost:3000/translate_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Good morning", "How are you"], "source": "en", "target": "fr"}'
# {"translations":["Bonjour le monde","Bonjour","Comment allez-vous"],"count":3,...}

curl -X POST http://localhost:3000/embed_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Bonjour le monde", "Hola mundo"]}'

curl -X POST http://localhost:3000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source": "en", "target": "fr"}'
# {"source_text":"Hello world","translation":"Bonjour le monde","labse_score":0.947,"sonar_score":0.863,"chunk_count":1,...}

curl -X POST http://localhost:3000/analyze_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Good morning"], "source": "en", "target": "fr"}'
```

## Use from Python

Start the server, then:

```python
import json, urllib.request

def translate(text, source="en", target="fr"):
    data = json.dumps({"text": text, "source": source, "target": target}).encode()
    req = urllib.request.Request("http://localhost:3000/translate", data=data,
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

def score(text1, text2):
    data = json.dumps({"text1": text1, "text2": text2}).encode()
    req = urllib.request.Request("http://localhost:3000/score", data=data,
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

def embed_batch(texts):
    data = json.dumps({"texts": texts}).encode()
    req = urllib.request.Request("http://localhost:3000/embed_batch", data=data,
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

def analyze(text, source="en", target="fr"):
    data = json.dumps({"text": text, "source": source, "target": target}).encode()
    req = urllib.request.Request("http://localhost:3000/analyze", data=data,
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

def analyze_batch(texts, source="en", target="fr"):
    data = json.dumps({"texts": texts, "source": source, "target": target}).encode()
    req = urllib.request.Request("http://localhost:3000/analyze_batch", data=data,
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())

print(translate("Hello", "en", "fr")["translation"])  # "Bonjour"
print(score("Hello", "Bonjour")["score"])              # 0.85

embeddings = embed_batch(["Hello", "Bonjour", "Hola"])
print(len(embeddings["embeddings"]))  # 3

result = analyze("Hello world", "en", "fr")
print(result["translation"])   # "Bonjour le monde"
print(result["labse_score"])   # 0.947
print(result["sonar_score"])   # 0.863
print(embeddings["dimensions"])       # 768
```

## Model Setup

All models auto-download when first used via the CLI. You can also download them explicitly:

```bash
# Download all models at once
lingo download all

# Or individually
lingo download nllb    # NLLB-200 (~2.5 GB F32 from HuggingFace, pinned revision + SHA-256)
lingo download labse   # LaBSE (~1.8 GB from HuggingFace, SHA-256-verified)
lingo download sonar   # SONAR (~3 GB, requires Python 3 + torch for conversion)
```

### NLLB-200 (translation)

Auto-downloads the F32 safetensors (~2.5 GB) from HuggingFace, or produce a
smaller F16 copy with (requires `transformers`, `torch`, `safetensors`):

```bash
python3 scripts/convert_nllb_safetensors.py
# Saves to ~/.cache/lingo/nllb-200-distilled-600M/ (~1.2 GB F16)
```

### LaBSE (similarity/embeddings)

Auto-downloads from [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE):

```
~/.cache/lingo/labse/
  model.safetensors         # BERT weights (~1.8 GB)
  config.json               # Model config
  tokenizer.json            # Tokenizer
  2_Dense/model.safetensors # Projection layer
```

### SONAR (similarity/embeddings)

`lingo download sonar` downloads the checkpoint from Meta's CDN (SHA-256
verified, loaded with `torch.load(weights_only=True)`) and converts it to
safetensors; the tokenizer is NLLB-200's `tokenizer.json`, since SONAR shares
its vocabulary. Requires Python 3 with `torch` and `safetensors`. The script
can also be run directly:

```bash
python3 scripts/convert_sonar_safetensors.py
# Saves model.safetensors + tokenizer.json to ~/.cache/lingo/sonar/ (~3 GB)
```

Upgrading from an older lingo: if loading SONAR reports a *legacy SONAR
tokenizer format*, run `lingo download sonar` once — it keeps the converted
weights and replaces only the tokenizer.

## Feature Flags

| Feature | Description |
|---------|-------------|
| `metal` | Metal GPU (macOS Apple Silicon) |
| `cuda` | CUDA GPU (Linux NVIDIA) |
| `accelerate` | Apple Accelerate (CPU SIMD) |
| `cli` | CLI binary |
| `server` | HTTP server (axum) |

## Building with CUDA

Requires CUDA toolkit 13.0+ for Blackwell GPUs (RTX 5090/5080), or CUDA 12.x for older architectures.

```bash
# Set CUDA_HOME to the correct toolkit version
export CUDA_HOME=/usr/local/cuda-13.0

# Build release binary with CUDA + CLI
cargo build --release --features cli,cuda

# Run
LINGO_ACCEPT_LICENSE=1 ./target/release/lingo translate "Hello world" --to fr
```

## Performance

All benchmarks are release builds (`cargo build --release`), measuring warm inference (model already loaded) unless noted otherwise. NLLB uses F16 weights via conversion script; LaBSE and SONAR use F32 weights.

### NVIDIA RTX 5090 — CUDA 13.0

**System**: RTX 5090 (32 GB VRAM, Blackwell), CUDA toolkit 13.0, Linux 6.8

| Operation | Time | Notes |
|-----------|------|-------|
| NLLB translation (short, warm) | ~57ms | Cached encoder, subsequent targets |
| NLLB translation (short, cold) | ~169ms | First target in invocation |
| NLLB translation (paragraph, 3 targets) | ~371ms avg | 40-word input |
| NLLB multi-target (10 languages) | ~224ms total | Single invocation |
| NLLB model load | ~1.27s | |
| NLLB cold start (load + first translate) | ~1.44s | |
| LaBSE similarity (per pair) | ~9ms | Custom BERT encoder |
| LaBSE model load | ~839ms | |
| LaBSE cold start (load + first score) | ~926ms | |
| SONAR similarity (per pair) | ~14ms | 24-layer transformer encoder |
| SONAR model load | ~916ms | F32 weights |
| SONAR cold start (load + first score) | ~953ms | |

### Apple M5 Max — Metal GPU

**System**: MacBook Pro, M5 Max, 128 GB unified memory, macOS 26.4

NLLB numbers reflect the **KV-cached decoder** released 2026-06 (decoder work
reduced from O(N²) to O(N) per token — see `ARCHITECTURE.md` "KV Cache").

| Operation | Time | Notes |
|-----------|------|-------|
| NLLB translation (short, warm) | ~87ms | 8-token input, cached encoder reused across targets |
| NLLB translation (short, cold) | ~117ms | First target in invocation |
| NLLB translation (paragraph, 3 targets) | ~534ms avg | 40-word input — was ~810ms pre-KV-cache |
| NLLB multi-target (10 languages) | ~337ms total | Single invocation — was ~1.14s pre-KV-cache |
| NLLB model load | ~521ms | F32 weights (Metal F16 kernel coverage incomplete) |
| NLLB cold start (load + first translate) | ~638ms | |
| LaBSE similarity (per pair) | ~11ms | Custom BERT encoder |
| LaBSE model load | ~433ms | |
| LaBSE cold start (load + first score) | ~534ms | |
| SONAR similarity (per pair) | ~30ms | 24-layer transformer encoder |
| SONAR model load | ~452ms | F32 weights |
| SONAR cold start (load + first score) | ~562ms | |

Short-input timings are unchanged (an 8-token input has too few output tokens
for O(N²) → O(N) to matter). The win shows up on **longer outputs**:
paragraph translations are ~34% faster and 10-target multi-translation is
~3.4× faster. Bulk backfill workloads with 400-token-class chunks see the
largest gains.

### Apple M5 Max — Accelerate (CPU SIMD)

**System**: MacBook Pro, M5 Max, 128 GB unified memory, macOS 26.4

| Operation | Time | Notes |
|-----------|------|-------|
| NLLB translation (short, warm) | ~225ms | Cached encoder, subsequent targets |
| NLLB translation (short, cold) | ~399ms | First target in invocation |
| NLLB translation (paragraph, 3 targets) | ~2.7s avg | 40-word input |
| NLLB multi-target (10 languages) | ~2.56s total | Single invocation |
| NLLB model load | ~630ms | F16 weights |
| NLLB cold start (load + first translate) | ~1.31s | |
| LaBSE similarity (per pair) | ~55ms | Faster than Metal for BERT inference |
| LaBSE model load | ~395ms | |
| LaBSE cold start (load + first score) | ~450ms | |
| SONAR similarity (per pair) | ~73ms | 24-layer transformer encoder |
| SONAR model load | ~408ms | F32 weights |
| SONAR cold start (load + first score) | ~522ms | |

### Cross-Lingual Similarity Scores (LaBSE)

Scores are cosine similarity of L2-normalized 768-dim embeddings. Identical on CPU and CUDA.

| Text 1 | Text 2 | Score |
|--------|--------|-------|
| "Hello world" | "Bonjour le monde" (fr) | 0.947 |
| "Hello world" | "Hola mundo" (es) | 0.957 |
| "Hello world" | "こんにちは世界" (ja) | 0.948 |
| "The cat sat on the mat" | "The dog ran in the park" | 0.517 |
| "I love programming" | "I love programming" | 1.000 |

### Cross-Lingual Similarity Scores (SONAR)

Scores are cosine similarity of L2-normalized 1024-dim embeddings, with each
text's language passed (`--lang1`/`--lang2`). Measured on Metal.

| Text 1 | Text 2 | Score |
|--------|--------|-------|
| "Hello world" | "Bonjour le monde" (fr) | 0.863 |
| "Hello world" | "Hola mundo" (es) | 0.922 |
| "Hello world" | "こんにちは世界" (ja) | 0.815 |
| "The cat sat on the mat" | "The dog ran in the park" | 0.503 |
| "I love programming" | "I love programming" | 1.000 |

## Long-input handling

NLLB-200 has a 1024-token positional-embedding limit. lingo
automatically chunks longer input on sentence boundaries (Western
`.!?` and CJK `。！？` terminators), translates each chunk
sequentially, and rejoins with the original whitespace preserved.
The chunk budget is 600 tokens by default, leaving headroom for
the BOS/EOS/language-code tokens and for verbose target languages
where the output is longer than the source. Results expose
`chunk_count` (single) or `chunk_counts` (batch) so callers can
see when chunking occurred.

## License

FiddyCent Software License. See [LICENSE](LICENSE) for full terms.

Model weights: CC-BY-NC 4.0 (Meta AI), Apache 2.0 (Google).
