# AGENTS.md

Agent-facing operating manual for **lingo** — a pure-Rust multilingual NLP toolkit
(NLLB-200 translation + LaBSE/SONAR similarity) on candle with Metal/CUDA/CPU backends.

This file is for AI coding agents. The README is the human narrative; this file is
terse and command-first. When the two disagree, prefer this file for *commands*
and the README for *concepts*.

---

## Required first step: license gate

Every code path that touches a model calls `license::require_license_acceptance()`.
Without acceptance, every operation returns `Error::LicenseNotAccepted` and exits.

```bash
export LINGO_ACCEPT_LICENSE=1
```

Set this in every shell, CI job, and `cargo run` invocation. The interactive CLI
also writes a sentinel at `~/.cache/lingo/.license-accepted` after the user types
`I ACCEPT`, but agents should always use the env var — it's idempotent and
non-interactive.

## Platform → feature flag matrix

Pick one accelerator feature per build. Combine with `cli` or `server` as needed.

| Platform | Build features | Notes |
|---|---|---|
| macOS Apple Silicon | `cli,metal` or `server,metal` | Metal GPU, fastest on M-series |
| macOS CPU-only / SIMD | `cli,accelerate` | Apple Accelerate framework |
| Linux NVIDIA | `cli,cuda` or `server,cuda` | Requires `CUDA_HOME=/usr/local/cuda-13.0` (CUDA 13 for Blackwell, 12.x for older) |
| Linux/macOS CPU fallback | `cli` | No accelerator |

**Default features = none.** You must pass `--features` explicitly. `cli` already
pulls in `download`, so you don't need to add it separately.

## Build & install

```bash
# Develop in tree
LINGO_ACCEPT_LICENSE=1 cargo build --release --features cli,metal

# Install from crates.io / git
cargo install lingo --features cli,metal     # macOS
CUDA_HOME=/usr/local/cuda-13.0 cargo install lingo --features cli,cuda  # Linux NVIDIA
```

The release binary lands at `./target/release/lingo`.

## Models

All models live under `~/.cache/lingo/<name>/`:

| Model | Path | Size | Source |
|---|---|---|---|
| NLLB-200-distilled-600M | `~/.cache/lingo/nllb-200-distilled-600M/` | ~1.2 GB | HuggingFace (auto) |
| LaBSE | `~/.cache/lingo/labse/` | ~1.8 GB | HuggingFace (auto) |
| SONAR | `~/.cache/lingo/sonar/` | ~3 GB (F32 safetensors) | Meta CDN + Python conversion |

```bash
# One-shot, all three
LINGO_ACCEPT_LICENSE=1 lingo download all

# Or individually
lingo download nllb
lingo download labse
lingo download sonar     # invokes scripts/convert_sonar_safetensors.py under the hood
```

**SONAR-specific gotcha:** SONAR is *not* on HuggingFace in a candle-loadable form.
The `download sonar` path shells out to `scripts/convert_sonar_safetensors.py`,
which downloads `sonar_text_encoder.pt` (~3 GB) from `dl.fbaipublicfiles.com` and
converts it to safetensors + a tokenizers JSON. It needs Python 3 with `torch`,
`safetensors`, `sentencepiece`, and `tokenizers` installed. The script also
needs `libsndfile` on the system (`brew install libsndfile` on macOS) only if
the user already has `fairseq2` installed — the direct-download path does not
require fairseq2 itself.

To check what's already cached:

```bash
ls ~/.cache/lingo/
```

## CLI reference

All commands take an optional `--model-dir` to override the default cache path.

```bash
# Translate (NLLB-200)
lingo translate "Hello world" --to fr
lingo translate "Hello world" --to fr,es,ja,ar,ko          # multi-target
lingo translate "Hello world" --from en --to fr --json     # JSON output
echo "Hello world" | lingo translate --to fr               # stdin

# Score similarity (LaBSE, 768-dim)
lingo score "Hello world" "Bonjour le monde"               # -> 0.9471

# Score similarity (SONAR, 1024-dim)
lingo sonar-score "Hello world" "Bonjour le monde"         # -> 0.7441

# Embed (LaBSE / SONAR)
lingo embed "Hello world"                                  # JSON array, 768 floats
lingo sonar-embed "Hello world"                            # JSON array, 1024 floats

# Languages
lingo languages                                            # 208 supported
```

Source-of-truth for CLI flags: `src/cli.rs`. ISO 639-1 codes for `--from`/`--to`;
SONAR scores are systematically *lower* than LaBSE (0.7 vs 0.95 for true
matches) but discriminate unrelated pairs better (0.17 vs 0.52).

## HTTP server

```bash
LINGO_ACCEPT_LICENSE=1 cargo run --release --example server --features server,metal
# Listens on 0.0.0.0:3000
```

Endpoints (source: `examples/server.rs`):

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | `{status, nllb_loaded, labse_loaded, sonar_loaded}` |
| POST | `/translate` | `{text, source, target}` | `{translation, source_lang, target_lang, duration_ms}` |
| POST | `/score` | `{text1, text2}` | `{score, duration_ms}` (LaBSE) |
| POST | `/embed` | `{text}` | `{embedding, dimensions}` (LaBSE, 768-dim) |
| POST | `/embed_batch` | `{texts: [...]}` | `{embeddings, dimensions, count}` |
| POST | `/analyze` | `{text, source, target}` | translation + LaBSE *and* SONAR scores |
| POST | `/analyze_batch` | `{texts: [...], source, target}` | batched `/analyze` |

The server loads all three models at startup; missing models are logged but
don't crash boot — endpoints touching a missing model will 4xx.

## Library use

```rust
use lingo::{NllbTranslator, LaBSEEncoder, SonarEncoder};

let t = NllbTranslator::new(None)?;          // None = default ~/.cache path
let r = t.translate("Hello", "en", "fr").await?;     // r.text, r.duration_ms

let labse = LaBSEEncoder::new(None)?;
let s = labse.score("Hello world", "Bonjour le monde").await?;
let e = labse.embed("Hello world").await?;   // Vec<f32>, len 768

let sonar = SonarEncoder::new(None)?;
let s = sonar.score("Hello world", "Bonjour le monde").await?;
let e = sonar.embed("Hello world").await?;   // Vec<f32>, len 1024
```

All three encoders are `Arc`-friendly, lazy-load on first use, and cache
embeddings keyed by raw input string. `with_device(path, device)` lets you
override the device explicitly.

## Repo layout

```
src/
  lib.rs          NllbTranslator + re-exports
  model.rs        NLLB-200 (M2M100 encoder-decoder)
  labse.rs        LaBSE encoder (custom BERT, 768-dim)
  sonar.rs        SONAR encoder (24-layer transformer, 1024-dim)
  cli.rs          `lingo` binary
  download.rs     HuggingFace fetch (NLLB + LaBSE)
  license.rs      License gate
  languages.rs    208 NLLB language codes
  error.rs        Error enum
examples/
  translate.rs    NLLB demo
  similarity.rs   LaBSE demo
  benchmark.rs    Full perf suite (NLLB + LaBSE + SONAR)
  server.rs       Axum HTTP server
scripts/
  convert_nllb_safetensors.py    PyTorch -> safetensors for NLLB
  convert_sonar_safetensors.py   Meta CDN -> safetensors for SONAR
ARCHITECTURE.md   Metal workarounds, M2M100 quirks, dtype rules
```

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `LicenseNotAccepted` | env var missing | `export LINGO_ACCEPT_LICENSE=1` |
| `ModelNotFound` | weights not in `~/.cache/lingo/...` | `lingo download <model>` |
| Build fails with `metal` feature on Linux | wrong platform | use `cuda` or no accelerator |
| Build fails with `cuda` | CUDA toolkit missing or wrong version | install CUDA 13 (Blackwell) or 12.x, set `CUDA_HOME` |
| `cargo install` picks default features (none) | features not requested | always pass `--features cli,<accelerator>` |
| SONAR conversion script fails on `import sonar` | `fairseq2`/`sonar-space` version mismatch | the script falls back to direct CDN download — no Python `sonar` package needed |
| Slow / wrong dtype on Metal | F16 kernels incomplete | NLLB auto-selects F32 on Metal; do not override |

## Verification recipe

After any setup work, verify end-to-end with:

```bash
export LINGO_ACCEPT_LICENSE=1
lingo translate "Hello world" --to fr        # -> "[fr] Bonjour le monde (...ms)"
lingo score "Hello world" "Bonjour le monde" # -> ~0.9471
lingo sonar-score "Hello world" "Bonjour le monde"  # -> ~0.7441
```

If all three succeed, the install is good.

## Pointers for deeper work

- Performance numbers per device: `README.md` → "Performance" section.
- Metal/CUDA workarounds and dtype rules: `ARCHITECTURE.md`.
- Full benchmark you can re-run: `cargo run --release --example benchmark --features metal`.
