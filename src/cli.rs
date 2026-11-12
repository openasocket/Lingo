//! lingo CLI — translate text and score similarity from the command line.

use lingo::{LaBSEEncoder, NllbLanguage, NllbTranslator, SonarEncoder};
use clap::{Parser, Subcommand};
use std::io::{self, BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "lingo")]
#[command(about = "Multilingual NLP: translation (NLLB-200), similarity scoring (LaBSE/SONAR) on Metal/CUDA GPU")]
#[command(version)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Translate text between languages using NLLB-200
    Translate {
        /// Text to translate (reads stdin if omitted)
        text: Option<String>,
        /// Source language (ISO 639-1)
        #[arg(short = 'f', long, default_value = "en")]
        from: String,
        /// Target language(s), comma-separated
        #[arg(short = 't', long)]
        to: String,
        /// Path to NLLB model directory
        #[arg(long)]
        model_dir: Option<String>,
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Score semantic similarity between two texts using LaBSE (768-dim)
    Score {
        /// First text
        text1: String,
        /// Second text
        text2: String,
        /// Path to LaBSE model directory
        #[arg(long)]
        model_dir: Option<String>,
    },

    /// Embed text into a 768-dim vector using LaBSE
    Embed {
        /// Text to embed
        text: String,
        /// Path to LaBSE model directory
        #[arg(long)]
        model_dir: Option<String>,
    },

    /// Score semantic similarity using SONAR (1024-dim)
    SonarScore {
        /// First text
        text1: String,
        /// Second text
        text2: String,
        /// Language of the first text (ISO 639-1/3)
        #[arg(long, default_value = "en")]
        lang1: String,
        /// Language of the second text (ISO 639-1/3)
        #[arg(long, default_value = "en")]
        lang2: String,
        /// Path to SONAR model directory
        #[arg(long)]
        model_dir: Option<String>,
    },

    /// Embed text into a 1024-dim vector using SONAR
    SonarEmbed {
        /// Text to embed
        text: String,
        /// Language of the text (ISO 639-1/3)
        #[arg(long, default_value = "en")]
        lang: String,
        /// Path to SONAR model directory
        #[arg(long)]
        model_dir: Option<String>,
    },

    /// Download model files (does not require accepting the license first)
    Download {
        /// Model to download: nllb, labse, sonar, or all
        #[arg(default_value = "all")]
        model: String,
        /// Target directory. For a single model this is the model directory;
        /// for `all` it is the parent holding nllb-200-distilled-600M/, labse/
        /// and sonar/. Defaults to ~/.cache/lingo.
        #[arg(long)]
        model_dir: Option<String>,
    },

    /// List all supported languages
    Languages,
}

/// Prompt user to download a model if running interactively.
/// Returns true if download should proceed, false if cancelled.
fn prompt_download(model_name: &str, size_hint: &str) -> bool {
    if !io::stdin().is_terminal() {
        eprintln!("Run interactively to auto-download, or use: lingo download {}", model_name.to_lowercase());
        return false;
    }

    eprint!("Download {} ? ({}) [Y/n] ", model_name, size_hint);
    io::stderr().flush().ok();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return false;
    }
    let input = input.trim().to_lowercase();
    input.is_empty() || input == "y" || input == "yes"
}

/// Ensure a model is on disk, offering to download it interactively.
/// Exits the process if the model is missing and the user declines.
fn ensure_downloaded(
    label: &str,
    size_hint: &str,
    model_dir: &Path,
    downloaded: bool,
    download: impl FnOnce() -> lingo::Result<()>,
) -> anyhow::Result<()> {
    if downloaded {
        return Ok(());
    }
    eprintln!("{label} model not found.");
    eprintln!("Location: {}", model_dir.display());
    eprintln!();
    if !prompt_download(label, size_hint) {
        std::process::exit(1);
    }
    eprintln!("Downloading {label}...");
    download()?;
    eprintln!("Download complete.\n");
    Ok(())
}

/// Interactive / env-var license gate. Exits if the license is not accepted.
fn require_license() -> anyhow::Result<()> {
    if lingo::license::license_accepted() {
        return Ok(());
    }
    eprintln!("{}", lingo::license::license_notice());
    if io::stdin().is_terminal() {
        eprint!("\nTo accept, type 'I ACCEPT' (or set LINGO_ACCEPT_LICENSE=1):\n> ");
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim() == "I ACCEPT" {
            lingo::license::mark_license_accepted()?;
            eprintln!("License accepted.");
            return Ok(());
        }
    }
    eprintln!("License not accepted. Exiting.");
    std::process::exit(1);
}

/// Resolve a model directory: explicit path, or ~/.cache/lingo/<name>.
fn model_path(explicit: Option<String>, name: &str) -> anyhow::Result<PathBuf> {
    Ok(match explicit {
        Some(d) => PathBuf::from(d),
        None => lingo::default_model_dir(name)?,
    })
}

/// Download one model into `dir` unless already present.
fn download_one(name: &str, dir: &Path) -> anyhow::Result<()> {
    match name {
        "nllb" => {
            if dir.join("model.safetensors").exists() && dir.join("tokenizer.json").exists() {
                eprintln!("NLLB-200 already downloaded at {}", dir.display());
            } else {
                eprintln!("Downloading NLLB-200 (~2.5 GB)...");
                lingo::download::download_nllb(dir)?;
                eprintln!("NLLB-200 download complete.");
            }
        }
        "labse" => {
            if ["model.safetensors", "config.json", "tokenizer.json"].iter().all(|f| dir.join(f).exists()) {
                eprintln!("LaBSE already downloaded at {}", dir.display());
            } else {
                eprintln!("Downloading LaBSE (~1.8 GB)...");
                lingo::download::download_labse(dir)?;
                eprintln!("LaBSE download complete.");
            }
        }
        "sonar" => {
            // download_sonar is idempotent and also upgrades a legacy tokenizer.
            eprintln!("Preparing SONAR at {} (first run downloads ~3 GB and needs Python + torch)...", dir.display());
            lingo::download::download_sonar(dir)?;
            eprintln!("SONAR ready.");
        }
        other => unreachable!("unknown model {other}"),
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "lingo=info".into()),
        )
        .with_target(false)
        .init();

    let args = Args::parse();

    // License gate — required before any model is *used*. Listing languages
    // and downloading weights are exempt (install -> download -> accept -> use).
    if !matches!(args.command, Commands::Languages | Commands::Download { .. }) {
        require_license()?;
    }

    match args.command {
        Commands::Translate { text, from, to, model_dir, json } => {
            let translator = NllbTranslator::new(model_dir.map(PathBuf::from))?;
            ensure_downloaded("NLLB-200", "~2.5 GB", translator.model_dir(), translator.is_model_downloaded(),
                || translator.download_model())?;

            let text = match text {
                Some(t) => t,
                None => {
                    let mut lines = Vec::new();
                    for line in io::stdin().lock().lines() { lines.push(line?); }
                    lines.join("\n")
                }
            };

            translator.load().await?;
            let targets: Vec<&str> = to.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            let mut results = Vec::new();
            let mut failures = 0usize;

            for target in &targets {
                match translator.translate(&text, &from, target).await {
                    Ok(r) => {
                        if json {
                            results.push(serde_json::json!({
                                "source": text, "source_lang": from,
                                "target_lang": target, "translation": r.text,
                                "duration_ms": r.duration_ms,
                                "chunk_count": r.chunk_count,
                            }));
                        } else if r.chunk_count > 1 {
                            println!("[{}] {} ({}ms, {} chunks)", target, r.text, r.duration_ms, r.chunk_count);
                        } else {
                            println!("[{}] {} ({}ms)", target, r.text, r.duration_ms);
                        }
                    }
                    Err(e) => {
                        failures += 1;
                        eprintln!("Error translating to {}: {}", target, e);
                    }
                }
            }
            if json { println!("{}", serde_json::to_string_pretty(&results)?); }
            if failures > 0 { std::process::exit(1); }
        }

        Commands::Score { text1, text2, model_dir } => {
            let encoder = LaBSEEncoder::new(model_dir.map(PathBuf::from))?;
            ensure_downloaded("LaBSE", "~1.8 GB", encoder.model_dir(), encoder.is_model_downloaded(),
                || encoder.download_model())?;
            let score = encoder.score(&text1, &text2).await?;
            println!("{:.4}", score);
        }

        Commands::Embed { text, model_dir } => {
            let encoder = LaBSEEncoder::new(model_dir.map(PathBuf::from))?;
            ensure_downloaded("LaBSE", "~1.8 GB", encoder.model_dir(), encoder.is_model_downloaded(),
                || encoder.download_model())?;
            let embedding = encoder.embed(&text).await?;
            println!("{}", serde_json::to_string(&embedding)?);
        }

        Commands::SonarScore { text1, text2, lang1, lang2, model_dir } => {
            let encoder = SonarEncoder::new(model_dir.map(PathBuf::from))?;
            ensure_downloaded("SONAR", "~3 GB, requires Python + torch", encoder.model_dir(),
                encoder.is_model_downloaded(), || encoder.download_model())?;
            let score = encoder.score_langs(&text1, &lang1, &text2, &lang2).await?;
            println!("{:.4}", score);
        }

        Commands::SonarEmbed { text, lang, model_dir } => {
            let encoder = SonarEncoder::new(model_dir.map(PathBuf::from))?;
            ensure_downloaded("SONAR", "~3 GB, requires Python + torch", encoder.model_dir(),
                encoder.is_model_downloaded(), || encoder.download_model())?;
            let embedding = encoder.embed_lang(&text, &lang).await?;
            println!("{}", serde_json::to_string(&embedding)?);
        }

        Commands::Download { model, model_dir } => {
            let model = model.to_lowercase();
            match model.as_str() {
                "nllb" => download_one("nllb", &model_path(model_dir, "nllb-200-distilled-600M")?)?,
                "labse" => download_one("labse", &model_path(model_dir, "labse")?)?,
                "sonar" => download_one("sonar", &model_path(model_dir, "sonar")?)?,
                "all" => {
                    let base = match model_dir {
                        Some(d) => PathBuf::from(d),
                        None => lingo::default_model_dir("")?,
                    };
                    download_one("nllb", &base.join("nllb-200-distilled-600M"))?;
                    download_one("labse", &base.join("labse"))?;
                    download_one("sonar", &base.join("sonar"))?;
                    eprintln!("All models ready.");
                }
                other => {
                    eprintln!("Unknown model: '{}'. Use: nllb, labse, sonar, or all", other);
                    std::process::exit(1);
                }
            }
        }

        Commands::Languages => {
            let languages = NllbLanguage::all_languages();
            println!("{:<9} {:<34} NLLB Code", "Code", "Language");
            println!("{}", "-".repeat(53));
            for lang in &languages {
                println!("{:<9} {:<34} {}", lang.iso_code(), lang.name(), lang.nllb_code());
            }
            println!("\n{} languages", languages.len());
        }
    }

    Ok(())
}
