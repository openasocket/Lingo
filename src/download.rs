//! Model download from HuggingFace Hub.
//!
//! Downloads NLLB-200-distilled-600M model files (`model.safetensors` + `tokenizer.json`)
//! to a local directory. Requires the `download` feature.

use crate::{Error, Result};
use std::path::Path;
use tracing::info;

const HF_MODEL_ID: &str = "facebook/nllb-200-distilled-600M";

/// Files required by lingo.
const REQUIRED_FILES: &[&str] = &["model.safetensors", "tokenizer.json"];

/// Download the NLLB-200-distilled-600M model from HuggingFace Hub.
///
/// Downloads `model.safetensors` (~1.2 GB) and `tokenizer.json` to `output_dir`.
/// Files that already exist are skipped. Uses symlinks to the HF cache to avoid
/// duplicating large files on disk; falls back to copying if symlinks fail.
pub fn download_model(output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| Error::Download(format!("Failed to initialize HuggingFace API: {}", e)))?;
    let repo = api.model(HF_MODEL_ID.to_string());

    for &filename in REQUIRED_FILES {
        let target = output_dir.join(filename);
        if target.exists() {
            info!("{} already exists, skipping", filename);
            continue;
        }

        info!("Downloading {} from {}...", filename, HF_MODEL_ID);
        let cached_path = repo
            .get(filename)
            .map_err(|e| Error::Download(format!("Failed to download {}: {}", filename, e)))?;

        // Symlink to HF cache to avoid duplicating large files on disk
        if link_file(&cached_path, &target).is_err() {
            std::fs::copy(&cached_path, &target)?;
        }

        let size = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
        info!("Downloaded {} ({:.1} MB)", filename, size as f64 / 1_000_000.0);
    }

    info!("Model ready at {}", output_dir.display());
    Ok(())
}

/// Symlink src to dst, resolving any existing symlinks in src first.
fn link_file(src: &Path, dst: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let resolved = std::fs::canonicalize(src)?;
        std::os::unix::fs::symlink(&resolved, dst)
    }
    #[cfg(not(unix))]
    {
        let _ = (src, dst);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "symlinks not available",
        ))
    }
}
