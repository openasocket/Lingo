/// Error types for lingo.
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Model not found at {0}. Run model download first.")]
    ModelNotFound(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Invalid language pair: {0} -> {1}")]
    InvalidLanguagePair(String, String),

    #[error("Language '{0}' is not supported by NLLB-200 and cannot be used for translation")]
    UnsupportedLanguage(String),

    #[error("Model inference error: {0}")]
    Inference(String),

    /// The model produced output that is known to be unusable — an empty
    /// translation, a dangling clause for a complete sentence, or a
    /// repeated-token loop. Only returned when the corresponding
    /// [`crate::TranslateConfig`] checks are enabled.
    #[error("Translation rejected: {0}")]
    Degenerate(DegenerateKind),

    /// The call's future was dropped (e.g. by a timeout) and the blocking
    /// work stopped at its next checkpoint. Callers normally never see this:
    /// it is the result of work nobody is awaiting any more.
    #[error("Operation cancelled")]
    Cancelled,

    #[error("Download error: {0}")]
    Download(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("License not accepted. Run the CLI or set LINGO_ACCEPT_LICENSE=1. See LICENSE file.")]
    LicenseNotAccepted,

    /// The call exceeded its budget. The in-flight work is asked to stop; use
    /// `wait_idle` on the instance to learn whether it did (reuse it) or is
    /// wedged in a GPU call (drop it and build a new one).
    #[error("Operation timed out after {0:?}. Call wait_idle() before reusing the instance — see Error::Timeout docs.")]
    Timeout(std::time::Duration),

    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Why strict mode rejected a translation (see [`Error::Degenerate`]).
///
/// The `Display` text is stable and matches what earlier versions put in the
/// `Degenerate(String)` reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DegenerateKind {
    /// Empty output for a non-empty source.
    EmptyOutput,
    /// A complete source sentence came back as an unfinished clause.
    DanglingClause,
    /// The decoder entered a repeated-token loop.
    RepeatedTokens,
}

impl std::fmt::Display for DegenerateKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DegenerateKind::EmptyOutput => "empty translation for a non-empty source",
            DegenerateKind::DanglingClause => "a complete source sentence was translated as an unfinished clause",
            DegenerateKind::RepeatedTokens => "output entered a repeated-token loop",
        })
    }
}

pub type Result<T> = std::result::Result<T, Error>;
