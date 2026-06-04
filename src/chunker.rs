//! Token-budgeted sentence chunker for NLLB long-input handling.
//!
//! NLLB-200's `max_position_embeddings` is 1024 — any input that tokenizes to
//! more rows than that overruns the positional-embedding tensor in the encoder
//! (and the decoder hits the same wall once a verbose target language expands
//! the output). This module splits arbitrary text into chunks that each fit
//! within a token budget, *without ever truncating or losing a character*, so
//! the translation path can translate each chunk and rejoin the results.
//!
//! The chunker is tokenizer-agnostic: it takes a `count_tokens` closure rather
//! than a concrete tokenizer, which keeps it independently unit-testable (the
//! tests use a word-count or char-count stand-in) and lets the caller plug in
//! the real NLLB tokenizer in production.
//!
//! This module is standalone — it is *not* wired into
//! [`crate::NllbTranslator::translate`] yet (that integration is a later phase).

/// A single chunk of source text plus the separator that followed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Text content of this chunk.
    pub text: String,
    /// Separator that was between this chunk and the next in the source
    /// (empty string for the last chunk). Use this to rejoin translations.
    pub trailing_separator: String,
}

/// Configuration for [`chunk_text`].
#[derive(Debug, Clone)]
pub struct ChunkerConfig {
    /// Max tokens per chunk. Default 600.
    pub max_tokens: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self { max_tokens: 600 }
    }
}

/// Split `text` into chunks each fitting within `cfg.max_tokens` as measured by
/// `count_tokens`.
///
/// Returns `vec![Chunk { text, "" }]` of length 1 when the whole input fits,
/// and an empty vector for empty input. Concatenating every chunk's `text`
/// immediately followed by its `trailing_separator`, in order, reproduces the
/// original input character-for-character.
pub fn chunk_text<F>(text: &str, cfg: &ChunkerConfig, count_tokens: F) -> Vec<Chunk>
where
    F: Fn(&str) -> usize,
{
    let sentences = split_sentences(text);
    let max = cfg.max_tokens;

    let mut chunks: Vec<Chunk> = Vec::new();
    let mut i = 0;
    while i < sentences.len() {
        let (sent, sep) = &sentences[i];

        // A single sentence that already blows the budget can't be packed —
        // soft-wrap on whitespace, or hard-split on char boundaries.
        if count_tokens(sent) > max {
            let mut subs = split_oversize(sent, max, &count_tokens);
            if let Some(last) = subs.last_mut() {
                // The real inter-sentence separator belongs after the last
                // sub-chunk; internal sub-chunk separators were set by the
                // splitter.
                last.trailing_separator = sep.clone();
            }
            chunks.extend(subs);
            i += 1;
            continue;
        }

        // Greedily pack consecutive sentences until the next one would push
        // the chunk over budget.
        let mut cur = sent.clone();
        let mut last = i;
        while last + 1 < sentences.len() {
            let next = last + 1;
            let candidate = format!("{}{}{}", cur, sentences[last].1, sentences[next].0);
            if count_tokens(&candidate) > max {
                break;
            }
            cur = candidate;
            last = next;
        }
        chunks.push(Chunk {
            text: cur,
            trailing_separator: sentences[last].1.clone(),
        });
        i = last + 1;
    }

    chunks
}

/// True for Western and CJK sentence terminators.
fn is_terminator(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '。' | '！' | '？')
}

/// Rough CJK detection (Hiragana, Katakana, the common CJK ideograph blocks,
/// compatibility ideographs, and fullwidth/halfwidth forms). Used only to
/// decide whether a terminator that is *not* followed by whitespace still
/// starts a new sentence.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF   // Hiragana + Katakana
        | 0x3400..=0x4DBF // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0xFF00..=0xFFEF // Halfwidth and Fullwidth Forms
    )
}

/// True if `c` can legitimately begin a new sentence with no preceding
/// whitespace: an uppercase letter or a CJK character.
fn is_sentence_start(c: char) -> bool {
    c.is_uppercase() || is_cjk(c)
}

/// Split `text` into `(sentence, separator_after)` pairs.
///
/// Each sentence retains its own terminator and internal whitespace; the
/// `separator_after` captures the whitespace run that lived between this
/// sentence and the next (empty for the no-whitespace CJK case and for the
/// final sentence). Concatenating `sentence + separator_after` over all pairs
/// reproduces `text` exactly.
fn split_sentences(text: &str) -> Vec<(String, String)> {
    let chars: Vec<char> = text.chars().collect();
    let mut sentences: Vec<(String, String)> = Vec::new();
    let mut cur = String::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        cur.push(c);
        i += 1;

        if !is_terminator(c) {
            continue;
        }

        if i < chars.len() && chars[i].is_whitespace() {
            // Terminator followed by whitespace -> boundary; the whitespace
            // run becomes the separator.
            let mut sep = String::new();
            while i < chars.len() && chars[i].is_whitespace() {
                sep.push(chars[i]);
                i += 1;
            }
            sentences.push((std::mem::take(&mut cur), sep));
        } else if i < chars.len() && is_sentence_start(chars[i]) {
            // Terminator immediately followed by an uppercase / CJK char ->
            // boundary with an empty separator (e.g. "你好。世界").
            sentences.push((std::mem::take(&mut cur), String::new()));
        }
        // Otherwise (digit, lowercase, more punctuation, ...) this is not a
        // real sentence boundary — keep accumulating (handles "3.14", "U.S").
    }

    if !cur.is_empty() {
        sentences.push((cur, String::new()));
    }

    sentences
}

/// Split a single over-budget sentence. Soft-wrap on whitespace when present,
/// otherwise hard-split on character boundaries.
fn split_oversize<F>(sentence: &str, max: usize, count: &F) -> Vec<Chunk>
where
    F: Fn(&str) -> usize,
{
    if sentence.chars().any(|c| c.is_whitespace()) {
        split_on_whitespace(sentence, max, count)
    } else {
        split_on_char_boundary(sentence, max, count)
    }
}

/// Break `s` into `(word, trailing_whitespace)` pairs, preserving the actual
/// whitespace characters. Any leading whitespace is folded into the first
/// word so no empty word is produced. Concatenating `word + trailing_ws`
/// reproduces `s`.
fn split_words(s: &str) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut word = String::new();
    let mut pending_ws = String::new();

    for c in s.chars() {
        if c.is_whitespace() {
            pending_ws.push(c);
        } else if !pending_ws.is_empty() {
            if word.is_empty() {
                // Leading whitespace: keep it attached to the upcoming word.
                word.push_str(&pending_ws);
                word.push(c);
                pending_ws.clear();
            } else {
                pairs.push((std::mem::take(&mut word), std::mem::take(&mut pending_ws)));
                word.push(c);
            }
        } else {
            word.push(c);
        }
    }

    if !word.is_empty() || !pending_ws.is_empty() {
        pairs.push((word, pending_ws));
    }

    pairs
}

/// Soft-wrap an over-budget sentence on whitespace, packing words greedily.
/// A single word that is itself over budget is recursively hard-split on
/// character boundaries.
fn split_on_whitespace<F>(sentence: &str, max: usize, count: &F) -> Vec<Chunk>
where
    F: Fn(&str) -> usize,
{
    let pairs = split_words(sentence);
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut i = 0;

    while i < pairs.len() {
        let (word, ws) = &pairs[i];

        if count(word) > max {
            // Even one word doesn't fit — hard-split it.
            let mut subs = split_on_char_boundary(word, max, count);
            if let Some(last) = subs.last_mut() {
                last.trailing_separator = ws.clone();
            }
            chunks.extend(subs);
            i += 1;
            continue;
        }

        let mut cur = word.clone();
        let mut last = i;
        while last + 1 < pairs.len() {
            let next = last + 1;
            // A next word that is itself over budget must start its own
            // (hard-split) chunk — stop packing here.
            if count(&pairs[next].0) > max {
                break;
            }
            let candidate = format!("{}{}{}", cur, pairs[last].1, pairs[next].0);
            if count(&candidate) > max {
                break;
            }
            cur = candidate;
            last = next;
        }
        chunks.push(Chunk {
            text: cur,
            trailing_separator: pairs[last].1.clone(),
        });
        i = last + 1;
    }

    chunks
}

/// Hard-split `s` on character boundaries (never byte boundaries) so each
/// piece fits the budget. Binary-searches the longest char prefix that stays
/// under budget. Always makes progress (at least one char per piece) so it
/// can never loop or drop a character; the inter-piece separator is empty.
fn split_on_char_boundary<F>(s: &str, max: usize, count: &F) -> Vec<Chunk>
where
    F: Fn(&str) -> usize,
{
    let chars: Vec<char> = s.chars().collect();
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut start = 0;

    while start < chars.len() {
        let remaining = chars.len() - start;
        let mut lo = 1;
        let mut hi = remaining;
        let mut best = 1; // never zero -> always progress, never truncate
        while lo <= hi {
            let mid = (lo + hi) / 2;
            let sub: String = chars[start..start + mid].iter().collect();
            if count(&sub) <= max {
                best = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        let sub: String = chars[start..start + best].iter().collect();
        chunks.push(Chunk {
            text: sub,
            trailing_separator: String::new(),
        });
        start += best;
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_input_returns_single_chunk() {
        let count = |s: &str| s.split_whitespace().count();
        let input = "Hello world.";
        let chunks = chunk_text(input, &ChunkerConfig::default(), count);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, input);
        assert_eq!(chunks[0].trailing_separator, "");
    }

    #[test]
    fn multi_sentence_packs_under_budget() {
        let count = |s: &str| s.split_whitespace().count();
        let input = "One. Two. Three. Four. Five.";
        let cfg = ChunkerConfig { max_tokens: 3 };
        let chunks = chunk_text(input, &cfg, count);
        assert!(chunks.len() >= 2, "expected packing into multiple chunks");
        for c in &chunks {
            assert!(
                count(&c.text) <= 3,
                "chunk exceeded budget: {:?} ({} tokens)",
                c.text,
                count(&c.text)
            );
        }
    }

    #[test]
    fn over_budget_sentence_soft_wraps_on_whitespace() {
        let count = |s: &str| s.split_whitespace().count();
        let input = "w1 w2 w3 w4 w5 w6 w7 w8 w9 w10";
        let cfg = ChunkerConfig { max_tokens: 3 };
        let chunks = chunk_text(input, &cfg, count);
        assert_eq!(chunks.len(), 4);
        assert_eq!(count(&chunks[0].text), 3);
        assert_eq!(count(&chunks[1].text), 3);
        assert_eq!(count(&chunks[2].text), 3);
        assert_eq!(count(&chunks[3].text), 1);
        assert_eq!(chunks[0].trailing_separator, " ");
        assert_eq!(chunks[1].trailing_separator, " ");
        assert_eq!(chunks[2].trailing_separator, " ");
        assert_eq!(chunks[3].trailing_separator, "");
    }

    #[test]
    fn cjk_terminator_splits_correctly() {
        // One token per character (CJK punctuation counts too).
        let count = |s: &str| s.chars().count();
        let input = "你好。世界！再见？";
        let cfg = ChunkerConfig { max_tokens: 3 };
        let chunks = chunk_text(input, &cfg, count);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "你好。");
        assert_eq!(chunks[1].text, "世界！");
        assert_eq!(chunks[2].text, "再见？");
    }

    #[test]
    fn empty_input_returns_empty_vec() {
        let count = |s: &str| s.split_whitespace().count();
        let chunks = chunk_text("", &ChunkerConfig::default(), count);
        assert!(chunks.is_empty());
    }

    #[test]
    fn whitespace_preservation_round_trip() {
        let count = |s: &str| s.split_whitespace().count();
        // Mixes a soft-wrapped over-budget sentence, a newline separator, and
        // a trailing short sentence.
        let input = "Alpha beta gamma delta. Epsilon zeta.\nEta theta iota kappa lambda. Mu.";
        let cfg = ChunkerConfig { max_tokens: 3 };
        let chunks = chunk_text(input, &cfg, count);

        let mut rebuilt = String::new();
        for c in &chunks {
            rebuilt.push_str(&c.text);
            rebuilt.push_str(&c.trailing_separator);
        }
        assert_eq!(rebuilt, input);
    }

    #[test]
    fn single_sentence_no_whitespace_over_budget_splits_on_char_boundary() {
        let count = |s: &str| s.chars().count();
        // 50 multi-byte characters, no whitespace, no terminator.
        let input: String = "あ".repeat(50);
        let cfg = ChunkerConfig { max_tokens: 10 };
        let chunks = chunk_text(&input, &cfg, count);
        assert_eq!(chunks.len(), 5);
        for c in &chunks {
            assert!(c.text.chars().count() <= 10);
        }
        // Round-trips on char boundaries (no character lost or split mid-codepoint).
        let rebuilt: String = chunks
            .iter()
            .map(|c| format!("{}{}", c.text, c.trailing_separator))
            .collect();
        assert_eq!(rebuilt, input);
    }
}
