# lingo

Pure Rust multilingual NLP toolkit built on [candle](https://github.com/huggingface/candle). Translates text using the NLLB-200 model and performs similarity calculations on translated text against pre-translated text using LaBSE and SONAR. No Python required.

## What It Does

| Capability | Model | Description |
|---|---|---|
| **Translation** | NLLB-200 | Translate between 200+ languages |
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

### Score Similarity

```bash
lingo score "Hello world" "Bonjour le monde"
# 0.8534
```

### Embed

```bash
lingo embed "Hello world"
# [0.0234, -0.0891, 0.0412, ...]  (768 floats)
```

### List Languages

```bash
lingo languages
```

## Use as an HTTP Server

```bash
cargo run --example server --features server,metal
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/translate` | `{"text", "source", "target"}` -> translation |
| POST | `/score` | `{"text1", "text2"}` -> similarity score |
| POST | `/embed` | `{"text"}` -> 768-dim embedding |
| GET | `/health` | Server status |

```bash
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source": "en", "target": "fr"}'

curl -X POST http://localhost:3000/score \
  -H "Content-Type: application/json" \
  -d '{"text1": "Hello world", "text2": "Bonjour le monde"}'
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

print(translate("Hello", "en", "fr")["translation"])  # "Bonjour"
print(score("Hello", "Bonjour")["score"])              # 0.85
```

## Model Setup

### NLLB-200 (translation)

```bash
python scripts/convert_nllb_safetensors.py
# Saves to ~/.cache/lingo/nllb-200-distilled-600M/ (~1.2 GB)
```

### LaBSE (similarity/embeddings)

Download from [sentence-transformers/LaBSE](https://huggingface.co/sentence-transformers/LaBSE):

```
~/.cache/lingo/labse/
  model.safetensors         # BERT weights (~1.8 GB)
  config.json               # Model config
  tokenizer.json            # Tokenizer
  2_Dense/model.safetensors # Projection layer
```

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

All benchmarks are release builds (`cargo build --release`), measuring warm inference (model already loaded) unless noted otherwise. NLLB uses F16 weights via conversion script; LaBSE uses F32 weights.

### NVIDIA RTX 5090 — CUDA 13.0

**System**: RTX 5090 (32 GB VRAM, Blackwell), CUDA toolkit 13.0, Linux 6.8

| Operation | Time | Notes |
|-----------|------|-------|
| NLLB translation (short, warm) | ~87ms | Cached encoder, subsequent targets |
| NLLB translation (short, cold) | ~186ms | First target in invocation |
| NLLB translation (paragraph, 3 targets) | ~370ms avg | 40-word input |
| NLLB multi-target (10 languages) | ~1.05s total | Single invocation |
| NLLB model load | ~850ms | |
| NLLB cold start (load + first translate) | ~1.4s | |
| LaBSE similarity (per pair) | ~20ms | Custom BERT encoder |
| LaBSE model load | ~730ms | |
| LaBSE cold start (load + first score) | ~1.2s | |

### Apple M5 Max — Metal GPU

**System**: MacBook Pro, M5 Max, 128 GB unified memory, macOS 26.4

| Operation | Time | Notes |
|-----------|------|-------|
| NLLB translation (short, warm) | ~85ms | Cached encoder, subsequent targets |
| NLLB translation (short, cold) | ~136ms | First target in invocation |
| NLLB translation (paragraph, 3 targets) | ~810ms avg | 40-word input |
| NLLB multi-target (10 languages) | ~1.14s total | Single invocation |
| NLLB model load | ~440ms | F16 weights |
| NLLB cold start (load + first translate) | ~660ms | |
| LaBSE similarity (per pair) | ~100ms | Custom BERT encoder |
| LaBSE model load | ~385ms | |
| LaBSE cold start (load + first score) | ~500ms | |

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

### Cross-Lingual Similarity Scores (LaBSE)

Scores are cosine similarity of L2-normalized 768-dim embeddings. Identical on CPU and CUDA.

| Text 1 | Text 2 | Score |
|--------|--------|-------|
| "Hello world" | "Bonjour le monde" (fr) | 0.947 |
| "Hello world" | "Hola mundo" (es) | 0.957 |
| "Hello world" | "こんにちは世界" (ja) | 0.948 |
| "The cat sat on the mat" | "The dog ran in the park" | 0.517 |
| "I love programming" | "I love programming" | 1.000 |

## License

FiddyCent Software License. See [LICENSE](LICENSE) for full terms.

Model weights: CC-BY-NC 4.0 (Meta AI), Apache 2.0 (Google).
