//! Model download from HuggingFace Hub.
//!
//! Downloads model files for NLLB-200, LaBSE, and SONAR to local directories.
//! Requires the `download` feature.

use crate::{Error, Result};
use std::path::Path;
use tracing::info;

// ---------------------------------------------------------------------------
// NLLB-200
// ---------------------------------------------------------------------------

const NLLB_HF_REPO: &str = "facebook/nllb-200-distilled-600M";
const NLLB_FILES: &[&str] = &["model.safetensors", "tokenizer.json"];

/// Download the NLLB-200-distilled-600M model from HuggingFace Hub.
///
/// Downloads `model.safetensors` (~1.2 GB) and `tokenizer.json` to `output_dir`.
/// Files that already exist are skipped.
pub fn download_nllb(output_dir: &Path) -> Result<()> {
    download_from_hf(NLLB_HF_REPO, NLLB_FILES, output_dir)
}

/// Backward-compatible alias for [`download_nllb`].
pub fn download_model(output_dir: &Path) -> Result<()> {
    download_nllb(output_dir)
}

// ---------------------------------------------------------------------------
// LaBSE
// ---------------------------------------------------------------------------

const LABSE_HF_REPO: &str = "sentence-transformers/LaBSE";
const LABSE_FILES: &[&str] = &["model.safetensors", "config.json", "tokenizer.json"];

/// Download the LaBSE model from HuggingFace Hub.
///
/// Downloads `model.safetensors` (~1.8 GB), `config.json`, `tokenizer.json`,
/// and `2_Dense/model.safetensors` (projection layer) to `output_dir`.
pub fn download_labse(output_dir: &Path) -> Result<()> {
    download_from_hf(LABSE_HF_REPO, LABSE_FILES, output_dir)?;

    // Download 2_Dense projection layer (lives in a subdirectory)
    let dense_dir = output_dir.join("2_Dense");
    std::fs::create_dir_all(&dense_dir)?;
    let dense_target = dense_dir.join("model.safetensors");
    if dense_target.exists() {
        info!("2_Dense/model.safetensors already exists, skipping");
        return Ok(());
    }

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| Error::Download(format!("Failed to initialize HuggingFace API: {}", e)))?;
    let repo = api.model(LABSE_HF_REPO.to_string());

    info!(
        "Downloading 2_Dense/model.safetensors from {}...",
        LABSE_HF_REPO
    );
    let cached_path = repo
        .get("2_Dense/model.safetensors")
        .map_err(|e| {
            Error::Download(format!(
                "Failed to download 2_Dense/model.safetensors: {}",
                e
            ))
        })?;

    if link_file(&cached_path, &dense_target).is_err() {
        std::fs::copy(&cached_path, &dense_target)?;
    }

    let size = std::fs::metadata(&dense_target)
        .map(|m| m.len())
        .unwrap_or(0);
    info!(
        "Downloaded 2_Dense/model.safetensors ({:.1} MB)",
        size as f64 / 1_000_000.0
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// SONAR
// ---------------------------------------------------------------------------

/// Download and convert the SONAR text encoder to safetensors.
///
/// SONAR weights are distributed in fairseq2/PyTorch format and must be
/// converted to safetensors for candle. This function runs the bundled
/// Python conversion script (`scripts/convert_sonar_safetensors.py`).
///
/// Requirements: Python 3 with `torch`, `safetensors`, and `huggingface_hub`.
/// Optionally `fairseq2` for direct model loading.
pub fn download_sonar(output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let model_file = output_dir.join("model.safetensors");
    let tokenizer_file = output_dir.join("tokenizer.json");
    if model_file.exists() && tokenizer_file.exists() {
        info!("SONAR model already exists at {}", output_dir.display());
        return Ok(());
    }

    let script = find_convert_script("convert_sonar_safetensors.py")?;

    info!("Running SONAR conversion script: {}", script.display());
    info!("This requires Python 3 with torch, safetensors, and huggingface_hub.");

    let output = std::process::Command::new("python3")
        .args([
            script.to_str().unwrap(),
            "--output-dir",
            output_dir.to_str().unwrap(),
        ])
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|e| {
            Error::Download(format!(
                "Failed to run SONAR conversion script. Is Python 3 installed?\n  \
                 You can also run it manually:\n    \
                 python3 {} --output-dir {}\n  \
                 Error: {}",
                script.display(),
                output_dir.display(),
                e,
            ))
        })?;

    if !output.success() {
        return Err(Error::Download(
            "SONAR conversion script failed. Check the output above for details.".into(),
        ));
    }

    if !model_file.exists() || !tokenizer_file.exists() {
        return Err(Error::Download(
            "SONAR conversion completed but model.safetensors or tokenizer.json not found."
                .into(),
        ));
    }

    info!("SONAR model ready at {}", output_dir.display());
    Ok(())
}

/// Locate a conversion script by searching common paths.
fn find_convert_script(name: &str) -> Result<std::path::PathBuf> {
    // Try relative to the current working directory
    let cwd_script = std::path::PathBuf::from("scripts").join(name);
    if cwd_script.exists() {
        return Ok(cwd_script);
    }

    // Try relative to the executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for prefix in &["scripts", "../scripts", "../../scripts"] {
                let candidate = dir.join(prefix).join(name);
                if candidate.exists() {
                    return Ok(candidate);
                }
            }
        }
    }

    // Try CARGO_MANIFEST_DIR (works during `cargo run`)
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        let candidate = std::path::PathBuf::from(manifest).join("scripts").join(name);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(Error::Download(format!(
        "Cannot find scripts/{}. Run from the lingo project directory, or run manually:\n  \
         python3 scripts/{} --output-dir ~/.cache/lingo/sonar",
        name, name,
    )))
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn download_from_hf(repo_id: &str, files: &[&str], output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| Error::Download(format!("Failed to initialize HuggingFace API: {}", e)))?;
    let repo = api.model(repo_id.to_string());

    for &filename in files {
        let target = output_dir.join(filename);
        if target.exists() {
            info!("{} already exists, skipping", filename);
            continue;
        }

        info!("Downloading {} from {}...", filename, repo_id);
        let cached_path = repo
            .get(filename)
            .map_err(|e| Error::Download(format!("Failed to download {}: {}", filename, e)))?;

        // Symlink to HF cache to avoid duplicating large files on disk
        if link_file(&cached_path, &target).is_err() {
            std::fs::copy(&cached_path, &target)?;
        }

        let size = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0);
        info!(
            "Downloaded {} ({:.1} MB)",
            filename,
            size as f64 / 1_000_000.0
        );
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
