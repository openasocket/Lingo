//! Basic translation example.
//!
//! ```bash
//! cargo run --example translate --features metal
//! ```

use lingo::NllbTranslator;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create translator (auto-selects Metal/CUDA/CPU)
    let translator = NllbTranslator::new(None)?;

    // Translate to multiple languages
    let text = "Good morning, welcome to the future of translation.";
    let targets = [
        ("fr", "French"),
        ("es", "Spanish"),
        ("ja", "Japanese"),
        ("ar", "Arabic"),
        ("ko", "Korean"),
        ("de", "German"),
        ("zh", "Chinese"),
        ("ru", "Russian"),
        ("hi", "Hindi"),
        ("pt", "Portuguese"),
    ];

    println!("Source: {}\n", text);

    for (code, name) in &targets {
        match translator.translate(text, "en", code).await {
            Ok(result) => {
                println!("[{}] {}: {} ({}ms)", code, name, result.text, result.duration_ms);
            }
            Err(e) => {
                eprintln!("[{}] {}: Error - {}", code, name, e);
            }
        }
    }

    Ok(())
}
