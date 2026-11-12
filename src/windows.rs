//! Full-length encoding for fixed-context sentence encoders (SONAR, LaBSE).
//!
//! Self-attention memory grows with the square of the sequence length, and
//! both encoders have a hard 512-position context, so a long input can neither
//! be encoded in one pass nor truncated without silently ignoring most of it.
//! Instead the content tokens are split into consecutive windows that fit the
//! context; every token is encoded exactly once. Each window is framed with the
//! same leading/trailing special tokens as the original encoding (language tag
//! and `</s>` for SONAR, `[CLS]`/`[SEP]` for LaBSE), and the per-window
//! embeddings are combined as a content-token-weighted mean, then
//! L2-normalized.
//!
//! An input that already fits produces exactly one window identical to the
//! unwindowed encoding, so short-text embeddings are unchanged.

/// One encoder pass: framed token IDs and the number of content tokens it
/// covers (its weight in the combined embedding).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TokenWindow {
    pub ids: Vec<u32>,
    pub weight: usize,
}

/// Split framed token IDs into windows of at most `max_len` tokens.
///
/// - `ids`: the full encoding, including special tokens
/// - `lead` / `trail`: how many special tokens frame the content at the start
///   and end of `ids`; every window repeats them
/// - `max_len`: model context length in tokens
pub(crate) fn token_windows(ids: &[u32], lead: usize, trail: usize, max_len: usize) -> Vec<TokenWindow> {
    let lead = lead.min(ids.len());
    let trail = trail.min(ids.len() - lead);
    let body = &ids[lead..ids.len() - trail];
    // Leave room for at least one content token beside the specials.
    let capacity = max_len.saturating_sub(lead + trail).max(1);

    if ids.len() <= max_len || body.len() <= capacity {
        return vec![TokenWindow { ids: ids.to_vec(), weight: body.len().max(1) }];
    }

    body.chunks(capacity)
        .map(|chunk| {
            let mut window = Vec::with_capacity(lead + chunk.len() + trail);
            window.extend_from_slice(&ids[..lead]);
            window.extend_from_slice(chunk);
            window.extend_from_slice(&ids[ids.len() - trail..]);
            TokenWindow { ids: window, weight: chunk.len() }
        })
        .collect()
}

/// Count leading and trailing special tokens from a tokenizer special mask
/// (1 = special).
pub(crate) fn special_frame(special_mask: &[u32]) -> (usize, usize) {
    let lead = special_mask.iter().take_while(|&&m| m == 1).count();
    let trail = special_mask[lead..].iter().rev().take_while(|&&m| m == 1).count();
    (lead, trail)
}

/// Content-token-weighted mean of per-window embeddings, L2-normalized.
/// A single embedding is returned unchanged.
pub(crate) fn combine(embeddings: Vec<(Vec<f32>, usize)>) -> Vec<f32> {
    if embeddings.len() == 1 {
        return embeddings.into_iter().next().map(|(e, _)| e).unwrap_or_default();
    }
    let dim = embeddings.first().map(|(e, _)| e.len()).unwrap_or(0);
    let mut sum = vec![0f32; dim];
    for (embedding, weight) in &embeddings {
        for (acc, value) in sum.iter_mut().zip(embedding) {
            *acc += value * *weight as f32;
        }
    }
    let norm = sum.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        sum.iter_mut().for_each(|v| *v /= norm);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_input_is_a_single_identical_window() {
        let ids = [101, 5, 6, 7, 102];
        let w = token_windows(&ids, 1, 1, 512);
        assert_eq!(w, vec![TokenWindow { ids: ids.to_vec(), weight: 3 }]);
    }

    #[test]
    fn long_input_covers_every_token_once_with_framing() {
        let mut ids = vec![900];
        ids.extend(0..10u32);
        ids.push(3);
        let w = token_windows(&ids, 1, 1, 6); // capacity 4
        assert_eq!(w.len(), 3);
        for win in &w {
            assert_eq!(win.ids.first(), Some(&900));
            assert_eq!(win.ids.last(), Some(&3));
            assert!(win.ids.len() <= 6);
        }
        let content: Vec<u32> = w.iter().flat_map(|x| x.ids[1..x.ids.len() - 1].to_vec()).collect();
        assert_eq!(content, (0..10).collect::<Vec<_>>());
        assert_eq!(w.iter().map(|x| x.weight).sum::<usize>(), 10);
    }

    #[test]
    fn special_frame_counts_both_ends() {
        assert_eq!(special_frame(&[1, 0, 0, 1]), (1, 1));
        assert_eq!(special_frame(&[1, 1, 0, 1]), (2, 1));
        assert_eq!(special_frame(&[0, 0]), (0, 0));
    }

    #[test]
    fn combine_is_weighted_and_normalized() {
        let out = combine(vec![(vec![1.0, 0.0], 3), (vec![0.0, 1.0], 1)]);
        let norm = (out[0] * out[0] + out[1] * out[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!(out[0] > out[1]);
        assert_eq!(combine(vec![(vec![0.6, 0.8], 7)]), vec![0.6, 0.8]);
    }
}
