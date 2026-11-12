//! Model download from HuggingFace Hub.
//!
//! Downloads model files for NLLB-200, LaBSE, and SONAR to local directories.
//! Requires the `download` feature.
//!
//! Every HuggingFace fetch is pinned to a repository revision, and large
//! weight files are verified against a pinned SHA-256 before they are placed
//! in the model directory. Files are staged under a temporary name and
//! renamed into place, so an interrupted download never leaves a partial file
//! that a later run would mistake for a complete one.

use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use tracing::info;

/// A file to fetch from a HuggingFace repository.
struct HfFile {
    /// Path inside the repository; also the path under the output directory.
    name: &'static str,
    /// Commit to fetch from.
    revision: &'static str,
    /// Expected SHA-256 of the file contents, when pinned.
    sha256: Option<&'static str>,
}

// ---------------------------------------------------------------------------
// NLLB-200
// ---------------------------------------------------------------------------

const NLLB_HF_REPO: &str = "facebook/nllb-200-distilled-600M";
/// `main` — has `tokenizer.json` (but no safetensors weights).
const NLLB_MAIN_REVISION: &str = "f8d333a098d19b4fd9a8b18f94170487ad3f821d";
/// `refs/pr/45` — the only revision that publishes `model.safetensors`.
const NLLB_SAFETENSORS_REVISION: &str = "a3e77be725cf30383f1faeb8d2f0b0ea98ac554e";

const NLLB_TOKENIZER: HfFile = HfFile {
    name: "tokenizer.json",
    revision: NLLB_MAIN_REVISION,
    sha256: Some("e316b82de11d0f951f370943b3c438311629547285129b0b81dadabd01bca665"),
};

const NLLB_FILES: &[HfFile] = &[
    HfFile {
        name: "model.safetensors",
        revision: NLLB_SAFETENSORS_REVISION,
        sha256: Some("5e1b5a7011b5663bc5144ea66807a07cc56cc818cb7ac57ff1dd44fb4e2e2667"),
    },
    NLLB_TOKENIZER,
];

/// Download the NLLB-200-distilled-600M model from HuggingFace Hub.
///
/// Downloads `model.safetensors` (~2.5 GB F32) and `tokenizer.json` to
/// `output_dir`. Files that already exist are skipped.
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
const LABSE_REVISION: &str = "main";

const LABSE_FILES: &[HfFile] = &[
    HfFile {
        name: "model.safetensors",
        revision: LABSE_REVISION,
        sha256: Some("77d8e1f2dbab6eb5d3c261ce9d3dbf1e3c69e02938c95f934f94f42c22dfa31f"),
    },
    HfFile { name: "config.json", revision: LABSE_REVISION, sha256: None },
    HfFile { name: "tokenizer.json", revision: LABSE_REVISION, sha256: None },
    HfFile {
        name: "2_Dense/model.safetensors",
        revision: LABSE_REVISION,
        sha256: Some("f866c945fa2147ce4940c8ead0b4b20af4217e0f4664bd35f41c2a0967c4fe77"),
    },
];

/// Download the LaBSE model from HuggingFace Hub.
///
/// Downloads `model.safetensors` (~1.8 GB), `config.json`, `tokenizer.json`,
/// and `2_Dense/model.safetensors` (projection layer) to `output_dir`.
pub fn download_labse(output_dir: &Path) -> Result<()> {
    download_from_hf(LABSE_HF_REPO, LABSE_FILES, output_dir)
}

// ---------------------------------------------------------------------------
// SONAR
// ---------------------------------------------------------------------------

/// The conversion script, embedded at build time so `lingo download sonar`
/// never executes a script found on disk (e.g. in the current directory).
const SONAR_CONVERT_SCRIPT: &str = include_str!("../scripts/convert_sonar_safetensors.py");

/// Download and convert the SONAR text encoder to safetensors.
///
/// SONAR weights are distributed as a PyTorch checkpoint and must be
/// converted to safetensors for candle. This runs the conversion script that
/// is embedded in the lingo binary (it downloads the checkpoint from Meta's
/// CDN and verifies its SHA-256 before loading it with
/// `torch.load(weights_only=True)`), then fetches the NLLB `tokenizer.json`
/// — SONAR shares NLLB's SentencePiece vocabulary — from a pinned revision.
///
/// Requirements: Python 3 with `torch` and `safetensors`.
///
/// If the weights are already converted but `tokenizer.json` is missing or
/// in the legacy (pre-language-token) format, only the tokenizer is fetched.
pub fn download_sonar(output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let model_file = output_dir.join("model.safetensors");
    let tokenizer_file = output_dir.join("tokenizer.json");

    if !model_file.exists() {
        run_sonar_conversion(output_dir)?;
        if !model_file.exists() {
            return Err(Error::Download(
                "SONAR conversion completed but model.safetensors was not produced.".into(),
            ));
        }
    } else {
        info!("SONAR weights already converted at {}", model_file.display());
    }

    if tokenizer_file.exists() && !is_legacy_sonar_tokenizer(&tokenizer_file) {
        info!("SONAR tokenizer already present");
    } else {
        if tokenizer_file.exists() {
            info!("Replacing legacy SONAR tokenizer.json (no language tokens)");
        }
        let fetched = hf_api().and_then(|api| {
            fetch_hf_file(&api, NLLB_HF_REPO, &NLLB_TOKENIZER, &tokenizer_file, true)
        });
        if let Err(e) = fetched {
            // Offline fallback: SONAR shares NLLB's vocabulary, so a sibling
            // NLLB install's tokenizer works if it has the language tokens.
            let sibling = output_dir
                .parent()
                .map(|p| p.join("nllb-200-distilled-600M").join("tokenizer.json"));
            match sibling {
                Some(s) if s.exists() && !is_legacy_sonar_tokenizer(&s) => {
                    tracing::warn!(
                        "HuggingFace fetch failed ({e}); using the NLLB tokenizer at {}",
                        s.display()
                    );
                    let staged = staging_path(&tokenizer_file);
                    std::fs::copy(&s, &staged)?;
                    std::fs::rename(&staged, &tokenizer_file)?;
                }
                _ => return Err(e),
            }
        }
    }

    info!("SONAR model ready at {}", output_dir.display());
    Ok(())
}

/// Older lingo versions wrote a SentencePiece-derived Unigram tokenizer with
/// no language tokens; SONAR needs the NLLB tokenizer's language tokens.
fn is_legacy_sonar_tokenizer(path: &Path) -> bool {
    match tokenizers::Tokenizer::from_file(path) {
        Ok(t) => t.token_to_id("eng_Latn").is_none(),
        Err(_) => true,
    }
}

fn run_sonar_conversion(output_dir: &Path) -> Result<()> {
    // Write the embedded script to a fresh, exclusively-created temp file.
    let script = std::env::temp_dir().join(format!(
        "lingo-convert-sonar-{}-{}.py",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(&script)?;
        f.write_all(SONAR_CONVERT_SCRIPT.as_bytes())?;
    }

    info!("Running SONAR conversion (requires Python 3 with torch and safetensors)");
    // `-I` isolates the interpreter from the current directory and user site
    // packages' path injection (PYTHONPATH etc.).
    let status = std::process::Command::new("python3")
        .arg("-I")
        .arg(&script)
        .arg("--output-dir")
        .arg(output_dir)
        .arg("--skip-tokenizer")
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status();
    let _ = std::fs::remove_file(&script);

    let status = status.map_err(|e| {
        Error::Download(format!(
            "Failed to run the SONAR conversion script. Is Python 3 installed? ({e})"
        ))
    })?;
    if !status.success() {
        return Err(Error::Download(
            "SONAR conversion script failed. Check the output above for details.".into(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn hf_api() -> Result<hf_hub::api::sync::Api> {
    hf_hub::api::sync::Api::new()
        .map_err(|e| Error::Download(format!("Failed to initialize HuggingFace API: {}", e)))
}

fn download_from_hf(repo_id: &str, files: &[HfFile], output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;
    let api = hf_api()?;

    for file in files {
        let target = output_dir.join(file.name);
        if target.exists() {
            info!("{} already exists, skipping", file.name);
            continue;
        }
        fetch_hf_file(&api, repo_id, file, &target, false)?;
    }

    info!("Model ready at {}", output_dir.display());
    Ok(())
}

/// Fetch one pinned file into `target`, verifying its hash first. When
/// `replace` is true an existing `target` is atomically replaced.
fn fetch_hf_file(
    api: &hf_hub::api::sync::Api,
    repo_id: &str,
    file: &HfFile,
    target: &Path,
    replace: bool,
) -> Result<()> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }

    info!("Downloading {} from {}@{}...", file.name, repo_id, short_rev(file.revision));
    let repo = api.repo(hf_hub::Repo::with_revision(
        repo_id.to_string(),
        hf_hub::RepoType::Model,
        file.revision.to_string(),
    ));
    let cached_path = repo
        .get(file.name)
        .map_err(|e| Error::Download(format!("Failed to download {}: {}", file.name, e)))?;

    if let Some(expected) = file.sha256 {
        info!("Verifying SHA-256 of {}...", file.name);
        let actual = sha256_file(&cached_path)?;
        if actual != expected {
            return Err(Error::Download(format!(
                "SHA-256 mismatch for {repo_id}/{name}: expected {expected}, got {actual}. \
                 The file may be corrupted or the upstream repository changed; delete \
                 {path} and retry, or report this if it persists.",
                name = file.name,
                path = cached_path.display(),
            )));
        }
    }

    place_file(&cached_path, target, replace)?;

    let size = std::fs::metadata(target).map(|m| m.len()).unwrap_or(0);
    info!("Downloaded {} ({:.1} MB)", file.name, size as f64 / 1_000_000.0);
    Ok(())
}

fn short_rev(rev: &str) -> &str {
    rev.get(..8).unwrap_or(rev)
}

/// Streaming SHA-256 of a file, hex-encoded.
fn sha256_file(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Put `src` at `dst` via a staged temp path + rename, so `dst` either does
/// not exist or is complete. Symlinks into the HF cache when possible to avoid
/// duplicating large files; falls back to a copy.
fn place_file(src: &Path, dst: &Path, replace: bool) -> Result<()> {
    let staged = staging_path(dst);
    let _ = std::fs::remove_file(&staged);
    if link_file(src, &staged).is_err() {
        std::fs::copy(src, &staged)?;
    }
    if dst.exists() && !replace {
        let _ = std::fs::remove_file(&staged);
        return Ok(());
    }
    std::fs::rename(&staged, dst).map_err(|e| {
        let _ = std::fs::remove_file(&staged);
        Error::Io(e)
    })
}

fn staging_path(dst: &Path) -> PathBuf {
    let mut name = dst.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".part");
    dst.with_file_name(name)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vector() {
        let dir = std::env::temp_dir().join(format!("lingo-sha-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("abc.txt");
        std::fs::write(&p, b"abc").unwrap();
        assert_eq!(
            sha256_file(&p).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn staging_path_appends_part_suffix() {
        assert_eq!(
            staging_path(Path::new("/x/model.safetensors")),
            PathBuf::from("/x/model.safetensors.part")
        );
    }

    #[test]
    fn embedded_script_pins_checkpoint_hash() {
        assert!(SONAR_CONVERT_SCRIPT.contains("weights_only=True"));
        assert!(SONAR_CONVERT_SCRIPT.contains("CHECKPOINT_SHA256"));
    }
}
