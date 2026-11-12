//! Live check for strict translation mode and windowed long-text encoding.
//!
//! ```bash
//! LINGO_ACCEPT_LICENSE=1 cargo run --release --example strict_and_windows --features metal -- [sonar_dir]
//! ```

use lingo::{LaBSEEncoder, NllbTranslator, SonarEncoder, TranslateConfig};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let t = NllbTranslator::new(None)?;
    let text = "The meeting starts at nine. Please bring the report!\nAgenda\nWe will discuss the budget.";
    let packed = t.translate(text, "en", "de").await?;
    t.set_config(TranslateConfig::strict());
    let strict = t.translate(text, "en", "de").await?;
    println!("default ({} chunk): {:?}", packed.chunk_count, packed.text);
    println!("strict  ({} units): {:?}", strict.chunk_count, strict.text);
    println!("nllb code source: {}", t.translate("Bonjour le monde.", "fra_Latn", "en").await?.text);
    // Greek writes `?` as `;`; strict mode must not reject Greek questions.
    for q in [
        "Hello! How are you doing today?",
        "This is a test. Is that okay?",
        "What is the capital of France and why is it important?",
    ] {
        match t.translate(q, "en", "el").await {
            Ok(r) => println!("greek ok: {}", r.text),
            Err(e) => println!("GREEK REJECTED: {q} -> {e}"),
        }
    }
    match t.translate("Hello", "en", "bar").await {
        Err(e) => println!("bavarian (not in NLLB) rejected: {e}"),
        Ok(r) => println!("UNEXPECTED bavarian ok: {}", r.text),
    }

    // A timed-out call stops its abandoned work and the instance stays usable.
    let long_text = "The committee reviewed the proposal and asked many detailed questions about the budget. ".repeat(40);
    let t0 = Instant::now();
    let r = t.translate_with_timeout(&long_text, "en", "fr", Duration::from_millis(300)).await;
    println!("timeout: {:?} after {} ms", r.err().map(|e| e.to_string()), t0.elapsed().as_millis());
    let t1 = Instant::now();
    let idle = t.wait_idle(Duration::from_secs(5)).await;
    println!("wait_idle -> {idle} in {} ms", t1.elapsed().as_millis());
    println!("reuse after timeout: {}", t.translate_with_timeout("Good morning.", "en", "fr", Duration::from_secs(30)).await?.text);

    let sonar = SonarEncoder::new(std::env::args().nth(1).map(Into::into))?;
    let labse = LaBSEEncoder::new(None)?;
    let sentence = "The committee reviewed the proposal and asked about the budget. ";
    let long = sentence.repeat(300); // ~4000 tokens
    let fr_short = "Le comité a examiné la proposition et a posé des questions sur le budget.";
    for (name, n) in [("short", 1usize), ("long", 300)] {
        let src = sentence.repeat(n);
        let start = Instant::now();
        let s = sonar.score_langs(&src, "en", fr_short, "fr").await?;
        let l = labse.score(&src, fr_short).await?;
        println!("{name}: sonar {s:.4} labse {l:.4} ({} ms)", start.elapsed().as_millis());
    }
    // Long text whose second half differs: windowing must see it.
    let mixed = format!("{}{}", sentence.repeat(150), "Penguins live in Antarctica and eat fish. ".repeat(150));
    println!(
        "long-vs-mixed sonar {:.4} (truncation would give ~1.0)",
        sonar.score(&long, &mixed).await?
    );
    std::process::exit(0);
}
