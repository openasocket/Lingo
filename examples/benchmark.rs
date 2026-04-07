//! Benchmark suite for NLLB, LaBSE, and SONAR.
//!
//! Measures model load times, warm inference, cold start, and cross-lingual
//! similarity scores for both embedding models.
//!
//! ```bash
//! LINGO_ACCEPT_LICENSE=1 cargo run --release --example benchmark --features metal
//! ```

use lingo::{LaBSEEncoder, NllbTranslator, SonarEncoder};
use std::time::Instant;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // ---- NLLB benchmarks ----
    println!("=== NLLB Translation Benchmarks ===\n");

    let start = Instant::now();
    let translator = NllbTranslator::new(None)?;
    translator.load().await?;
    let nllb_load_ms = start.elapsed().as_millis();
    println!("NLLB model load: {}ms", nllb_load_ms);

    // Cold translation (first target)
    let start = Instant::now();
    let result = translator.translate("Hello, how are you?", "en", "fr").await?;
    let nllb_cold_ms = start.elapsed().as_millis();
    println!("NLLB translation (short, cold): {}ms  -> {}", nllb_cold_ms, result.text);

    // Warm translations (subsequent targets, encoder cached)
    let warm_targets = ["es", "de", "ja", "ko", "ar"];
    let mut warm_times = Vec::new();
    for target in &warm_targets {
        let start = Instant::now();
        let result = translator.translate("Hello, how are you?", "en", target).await?;
        let ms = start.elapsed().as_millis();
        warm_times.push(ms);
        println!("NLLB translation (short, warm -> {}): {}ms  -> {}", target, ms, result.text);
    }
    let avg_warm = warm_times.iter().sum::<u128>() / warm_times.len() as u128;
    println!("NLLB translation (short, warm avg): {}ms", avg_warm);

    // Paragraph translation
    let paragraph = "The quick brown fox jumps over the lazy dog. Machine learning models have \
        revolutionized natural language processing, enabling accurate translation between hundreds \
        of languages with minimal latency on modern hardware.";
    let para_targets = ["fr", "es", "ja"];
    let mut para_times = Vec::new();
    for target in &para_targets {
        let start = Instant::now();
        let result = translator.translate(paragraph, "en", target).await?;
        let ms = start.elapsed().as_millis();
        para_times.push(ms);
        println!("NLLB paragraph -> {}: {}ms  ({} chars)", target, ms, result.text.len());
    }
    let para_avg = para_times.iter().sum::<u128>() / para_times.len() as u128;
    println!("NLLB paragraph avg: {}ms", para_avg);

    // Multi-target (10 languages) in single invocation
    let multi_targets = ["fr", "es", "de", "ja", "ko", "ar", "zh", "hi", "pt", "ru"];
    let start = Instant::now();
    for target in &multi_targets {
        translator.translate("Hello, how are you?", "en", target).await?;
    }
    let multi_ms = start.elapsed().as_millis();
    println!("NLLB multi-target (10 languages): {}ms total", multi_ms);

    // ---- LaBSE benchmarks ----
    println!("\n=== LaBSE Similarity Benchmarks ===\n");

    let start = Instant::now();
    let labse = LaBSEEncoder::new(None)?;
    labse.load().await?;
    let labse_load_ms = start.elapsed().as_millis();
    println!("LaBSE model load: {}ms", labse_load_ms);

    // Cold start (load + first score)
    // Already loaded above, so measure first score as warm
    let start = Instant::now();
    let score = labse.score("Hello world", "Bonjour le monde").await?;
    let labse_first_ms = start.elapsed().as_millis();
    println!("LaBSE first score: {}ms  (score={:.4})", labse_first_ms, score);

    // Warm scoring
    let pairs = [
        ("Hello world", "Bonjour le monde", "en-fr"),
        ("Hello world", "Hola mundo", "en-es"),
        ("Hello world", "\u{3053}\u{3093}\u{306b}\u{3061}\u{306f}\u{4e16}\u{754c}", "en-ja"),
        ("The cat sat on the mat", "The dog ran in the park", "unrelated"),
        ("I love programming", "I love programming", "identical"),
    ];

    // Clear cache by creating a fresh encoder for fair warm timing
    let labse2 = LaBSEEncoder::new(None)?;
    labse2.load().await?;

    let mut labse_times = Vec::new();
    println!("\nLaBSE cross-lingual scores:");
    for (t1, t2, label) in &pairs {
        let start = Instant::now();
        let score = labse2.score(t1, t2).await?;
        let ms = start.elapsed().as_millis();
        labse_times.push(ms);
        println!("  {:<40} {:<40} {:<10} {:.4}  ({}ms)", t1, t2, label, score, ms);
    }
    let labse_avg = labse_times.iter().sum::<u128>() / labse_times.len() as u128;
    println!("LaBSE avg per pair: {}ms", labse_avg);

    // ---- SONAR benchmarks ----
    println!("\n=== SONAR Similarity Benchmarks ===\n");

    let start = Instant::now();
    let sonar = SonarEncoder::new(None)?;
    sonar.load().await?;
    let sonar_load_ms = start.elapsed().as_millis();
    println!("SONAR model load: {}ms", sonar_load_ms);

    // First score
    let start = Instant::now();
    let score = sonar.score("Hello world", "Bonjour le monde").await?;
    let sonar_first_ms = start.elapsed().as_millis();
    println!("SONAR first score: {}ms  (score={:.4})", sonar_first_ms, score);

    // Warm scoring with fresh encoder
    let sonar2 = SonarEncoder::new(None)?;
    sonar2.load().await?;

    let mut sonar_times = Vec::new();
    println!("\nSONAR cross-lingual scores:");
    for (t1, t2, label) in &pairs {
        let start = Instant::now();
        let score = sonar2.score(t1, t2).await?;
        let ms = start.elapsed().as_millis();
        sonar_times.push(ms);
        println!("  {:<40} {:<40} {:<10} {:.4}  ({}ms)", t1, t2, label, score, ms);
    }
    let sonar_avg = sonar_times.iter().sum::<u128>() / sonar_times.len() as u128;
    println!("SONAR avg per pair: {}ms", sonar_avg);

    // ---- Summary ----
    println!("\n=== Summary ===\n");
    println!("NLLB model load:              {}ms", nllb_load_ms);
    println!("NLLB cold start:              {}ms", nllb_load_ms as u128 + nllb_cold_ms);
    println!("NLLB short warm avg:          {}ms", avg_warm);
    println!("NLLB paragraph avg:           {}ms", para_avg);
    println!("NLLB 10-lang total:           {}ms", multi_ms);
    println!();
    println!("LaBSE model load:             {}ms", labse_load_ms);
    println!("LaBSE cold start:             {}ms", labse_load_ms as u128 + labse_first_ms);
    println!("LaBSE per pair avg:           {}ms", labse_avg);
    println!();
    println!("SONAR model load:             {}ms", sonar_load_ms);
    println!("SONAR cold start:             {}ms", sonar_load_ms as u128 + sonar_first_ms);
    println!("SONAR per pair avg:           {}ms", sonar_avg);

    Ok(())
}
