//! LaBSE cross-lingual similarity scoring example.
//!
//! ```bash
//! cargo run --example similarity --features metal
//! ```

use lingo::LaBSEEncoder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let encoder = LaBSEEncoder::new(None)?;

    let pairs = [
        ("Hello, how are you?", "Bonjour, comment allez-vous ?", "en-fr"),
        ("Hello, how are you?", "Hola, como estas?", "en-es"),
        ("Hello, how are you?", "こんにちは、お元気ですか？", "en-ja"),
        ("The cat sat on the mat", "The dog ran in the park", "unrelated"),
        ("I love programming", "I love programming", "identical"),
    ];

    println!("{:<50} {:<50} {:<10} {}", "Text 1", "Text 2", "Type", "Score");
    println!("{}", "-".repeat(120));

    for (text1, text2, label) in &pairs {
        let score = encoder.score(text1, text2).await?;
        println!("{:<50} {:<50} {:<10} {:.4}", text1, text2, label, score);
    }

    Ok(())
}
