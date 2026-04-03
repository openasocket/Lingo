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

    #[error("Download error: {0}")]
    Download(String),

    #[error("Device error: {0}")]
    Device(String),

    #[error("License not accepted. Run the CLI or set LINGO_ACCEPT_LICENSE=1. See LICENSE file.")]
    LicenseNotAccepted,

    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
