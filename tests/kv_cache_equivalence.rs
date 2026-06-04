//! End-to-end numeric-equivalence test for the NLLB KV cache (KV-CACHE-05).
//!
//! Translates the *same* input twice against the real NLLB-200 model — once
//! through the default KV-cached [`NllbTranslator::translate`] path and once
//! through the full-prefix reference path
//! ([`NllbTranslator::translate_reference`], which drives
//! `NllbModel::generate_reference`) — and asserts the decoded output text is
//! byte-for-byte identical across a spread of language pairs and lengths.
//!
//! This is the single most important correctness gate in the KV-cache effort:
//! if the per-step `decode_step` path ever diverges from recomputing the whole
//! growing prefix, the failing case pinpoints exactly where.
//!
//! It is `#[ignore]`d because it downloads/loads the ~1.2 GB model and runs real
//! GPU inference. Run it explicitly on a machine with the model present:
//!
//! ```bash
//! LINGO_ACCEPT_LICENSE=1 \
//!   cargo test --features metal --test kv_cache_equivalence -- --ignored --nocapture
//! ```

use lingo::NllbTranslator;

#[tokio::test]
#[ignore] // run with `cargo test --features metal --test kv_cache_equivalence -- --ignored`
async fn kv_cached_generate_matches_reference() {
    let translator = NllbTranslator::new(None).unwrap();
    translator.load().await.unwrap();

    let cases = [
        ("Hello world", "en", "fr"),
        ("The quick brown fox jumps over the lazy dog.", "en", "es"),
        ("Machine learning has transformed natural language processing.", "en", "de"),
        ("Bonjour, comment allez-vous aujourd'hui ?", "fr", "en"),
        ("Hola, ¿cómo estás?", "es", "en"),
        ("こんにちは、お元気ですか？", "ja", "en"),
        ("La inteligencia artificial está cambiando el mundo.", "es", "en"),
        ("Die Wissenschaft entwickelt sich rasant weiter.", "de", "en"),
        ("人工智能正在改变世界。", "zh", "en"),
        ("기계 학습은 흥미로운 분야입니다.", "ko", "en"),
        // longer/varied:
        (
            "In a small village by the sea, the fishermen woke before dawn to set their nets.",
            "en",
            "fr",
        ),
        (
            "Quantum entanglement allows two particles to share a state regardless of distance.",
            "en",
            "ja",
        ),
        (
            "The economy grew by three percent last quarter, driven by consumer spending.",
            "en",
            "es",
        ),
        (
            "Renewable energy sources are essential for combating climate change.",
            "en",
            "ar",
        ),
        (
            "Scientists have discovered a new species of deep-sea organism.",
            "en",
            "zh",
        ),
        (
            "Cooking is an art that combines chemistry, biology, and creativity.",
            "en",
            "ko",
        ),
        (
            "The novel explores themes of identity, memory, and time.",
            "en",
            "ru",
        ),
        (
            "Mountains hold a particular fascination for hikers around the world.",
            "en",
            "pt",
        ),
        (
            "Translation systems must balance accuracy with idiomatic naturalness.",
            "en",
            "hi",
        ),
        ("Mathematics is the universal language of science.", "en", "it"),
    ];

    let mut divergences = Vec::new();
    for (text, src, tgt) in &cases {
        let cached = translator.translate(text, src, tgt).await.unwrap();
        let reference = translator.translate_reference(text, src, tgt).await.unwrap();
        if cached.text != reference.text {
            divergences.push(format!(
                "{} -> {}: cached={:?}, reference={:?}",
                src, tgt, cached.text, reference.text
            ));
        }
    }
    assert!(
        divergences.is_empty(),
        "KV-cached output diverged from reference on {} cases:\n{}",
        divergences.len(),
        divergences.join("\n")
    );
}
