//! lingo CLI — translate text and score similarity from the command line.

use lingo::{LaBSEEncoder, NllbLanguage, NllbTranslator, SonarEncoder};
use clap::{Parser, Subcommand};
use std::io::{self, BufRead, IsTerminal, Write};

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
        /// Path to SONAR model directory
        #[arg(long)]
        model_dir: Option<String>,
    },

    /// Embed text into a 1024-dim vector using SONAR
    SonarEmbed {
        /// Text to embed
        text: String,
        /// Path to SONAR model directory
        #[arg(long)]
        model_dir: Option<String>,
    },

    /// Download model files from HuggingFace
    Download {
        /// Model to download: nllb, labse, sonar, or all
        #[arg(default_value = "all")]
        model: String,
    },

    /// List all supported languages
    Languages,
}

/// Prompt user to download a model if running interactively.
/// Returns true if download should proceed, false if cancelled.
fn prompt_download(model_name: &str, size_hint: &str) -> bool {
    if !io::stdin().is_terminal() {
        eprintln!("Run interactively to auto-download, or use: lingo download {}", model_name);
        return false;
    }

    eprint!("Download {} from HuggingFace? ({}) [Y/n] ", model_name, size_hint);
    io::stderr().flush().ok();

    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        return false;
    }
    let input = input.trim().to_lowercase();
    input.is_empty() || input == "y" || input == "yes"
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

    // License gate — must accept before using the CLI.
    if !lingo::license::license_accepted() {
        eprintln!("{}", lingo::license::license_notice());
        if io::stdin().is_terminal() {
            eprint!("\nTo accept, type 'I ACCEPT' (or set LINGO_ACCEPT_LICENSE=1):\n> ");
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            if input.trim() == "I ACCEPT" {
                lingo::license::mark_license_accepted()?;
                eprintln!("License accepted.");
            } else {
                eprintln!("License not accepted. Exiting.");
                std::process::exit(1);
            }
        } else {
            eprintln!("License not accepted. Exiting.");
            std::process::exit(1);
        }
    }

    let args = Args::parse();

    match args.command {
        Commands::Translate { text, from, to, model_dir, json } => {
            let translator = NllbTranslator::new(model_dir.map(std::path::PathBuf::from))?;

            if !translator.is_model_downloaded() {
                eprintln!("NLLB-200-distilled-600M model not found.");
                eprintln!("Location: {}", translator.model_dir().display());
                eprintln!();

                if prompt_download("NLLB-200", "~1.2 GB") {
                    eprintln!("Downloading model files...");
                    translator.download_model()?;
                    eprintln!("Download complete.\n");
                } else {
                    std::process::exit(1);
                }
            }

            let text = match text {
                Some(t) => t,
                None => {
                    let mut lines = Vec::new();
                    for line in io::stdin().lock().lines() { lines.push(line?); }
                    lines.join("\n")
                }
            };

            translator.load().await?;
            let targets: Vec<&str> = to.split(',').map(|s| s.trim()).collect();
            let mut results = Vec::new();

            for target in &targets {
                match translator.translate(&text, &from, target).await {
                    Ok(r) => {
                        if json {
                            results.push(serde_json::json!({
                                "source": text, "source_lang": from,
                                "target_lang": target, "translation": r.text,
                                "duration_ms": r.duration_ms,
                            }));
                        } else {
                            println!("[{}] {} ({}ms)", target, r.text, r.duration_ms);
                        }
                    }
                    Err(e) => eprintln!("Error translating to {}: {}", target, e),
                }
            }
            if json { println!("{}", serde_json::to_string_pretty(&results)?); }
        }

        Commands::Score { text1, text2, model_dir } => {
            let encoder = LaBSEEncoder::new(model_dir.map(std::path::PathBuf::from))?;

            if !encoder.is_model_downloaded() {
                eprintln!("LaBSE model not found.");
                eprintln!("Location: {}", encoder.model_dir().display());
                eprintln!();

                if prompt_download("LaBSE", "~1.8 GB") {
                    eprintln!("Downloading model files...");
                    encoder.download_model()?;
                    eprintln!("Download complete.\n");
                } else {
                    std::process::exit(1);
                }
            }

            let score = encoder.score(&text1, &text2).await?;
            println!("{:.4}", score);
        }

        Commands::Embed { text, model_dir } => {
            let encoder = LaBSEEncoder::new(model_dir.map(std::path::PathBuf::from))?;

            if !encoder.is_model_downloaded() {
                eprintln!("LaBSE model not found.");
                eprintln!("Location: {}", encoder.model_dir().display());
                eprintln!();

                if prompt_download("LaBSE", "~1.8 GB") {
                    eprintln!("Downloading model files...");
                    encoder.download_model()?;
                    eprintln!("Download complete.\n");
                } else {
                    std::process::exit(1);
                }
            }

            let embedding = encoder.embed(&text).await?;
            println!("{}", serde_json::to_string(&embedding)?);
        }

        Commands::SonarScore { text1, text2, model_dir } => {
            let encoder = SonarEncoder::new(model_dir.map(std::path::PathBuf::from))?;

            if !encoder.is_model_downloaded() {
                eprintln!("SONAR model not found.");
                eprintln!("Location: {}", encoder.model_dir().display());
                eprintln!();

                if prompt_download("SONAR", "requires Python + ~1 GB") {
                    eprintln!("Converting SONAR model...");
                    encoder.download_model()?;
                    eprintln!("Conversion complete.\n");
                } else {
                    std::process::exit(1);
                }
            }

            let score = encoder.score(&text1, &text2).await?;
            println!("{:.4}", score);
        }

        Commands::SonarEmbed { text, model_dir } => {
            let encoder = SonarEncoder::new(model_dir.map(std::path::PathBuf::from))?;

            if !encoder.is_model_downloaded() {
                eprintln!("SONAR model not found.");
                eprintln!("Location: {}", encoder.model_dir().display());
                eprintln!();

                if prompt_download("SONAR", "requires Python + ~1 GB") {
                    eprintln!("Converting SONAR model...");
                    encoder.download_model()?;
                    eprintln!("Conversion complete.\n");
                } else {
                    std::process::exit(1);
                }
            }

            let embedding = encoder.embed(&text).await?;
            println!("{}", serde_json::to_string(&embedding)?);
        }

        Commands::Download { model } => {
            match model.to_lowercase().as_str() {
                "nllb" => {
                    let translator = NllbTranslator::new(None)?;
                    if translator.is_model_downloaded() {
                        eprintln!("NLLB-200 already downloaded at {}", translator.model_dir().display());
                    } else {
                        eprintln!("Downloading NLLB-200 (~1.2 GB)...");
                        translator.download_model()?;
                        eprintln!("NLLB-200 download complete.");
                    }
                }
                "labse" => {
                    let encoder = LaBSEEncoder::new(None)?;
                    if encoder.is_model_downloaded() {
                        eprintln!("LaBSE already downloaded at {}", encoder.model_dir().display());
                    } else {
                        eprintln!("Downloading LaBSE (~1.8 GB)...");
                        encoder.download_model()?;
                        eprintln!("LaBSE download complete.");
                    }
                }
                "sonar" => {
                    let encoder = SonarEncoder::new(None)?;
                    if encoder.is_model_downloaded() {
                        eprintln!("SONAR already downloaded at {}", encoder.model_dir().display());
                    } else {
                        eprintln!("Downloading/converting SONAR (requires Python)...");
                        encoder.download_model()?;
                        eprintln!("SONAR download complete.");
                    }
                }
                "all" => {
                    // NLLB
                    let translator = NllbTranslator::new(None)?;
                    if translator.is_model_downloaded() {
                        eprintln!("NLLB-200 already downloaded.");
                    } else {
                        eprintln!("Downloading NLLB-200 (~1.2 GB)...");
                        translator.download_model()?;
                        eprintln!("NLLB-200 download complete.\n");
                    }

                    // LaBSE
                    let encoder = LaBSEEncoder::new(None)?;
                    if encoder.is_model_downloaded() {
                        eprintln!("LaBSE already downloaded.");
                    } else {
                        eprintln!("Downloading LaBSE (~1.8 GB)...");
                        encoder.download_model()?;
                        eprintln!("LaBSE download complete.\n");
                    }

                    // SONAR
                    let sonar = SonarEncoder::new(None)?;
                    if sonar.is_model_downloaded() {
                        eprintln!("SONAR already downloaded.");
                    } else {
                        eprintln!("Downloading/converting SONAR (requires Python)...");
                        sonar.download_model()?;
                        eprintln!("SONAR download complete.\n");
                    }

                    eprintln!("All models ready.");
                }
                other => {
                    eprintln!("Unknown model: '{}'. Use: nllb, labse, sonar, or all", other);
                    std::process::exit(1);
                }
            }
        }

        Commands::Languages => {
            println!("{:<6} {:<30} {}", "Code", "Language", "NLLB Code");
            println!("{}", "-".repeat(60));
            for lang in NllbLanguage::all_languages() {
                println!("{:<6} {:<30} {}", lang.iso_code(), lang.name(), lang.nllb_code());
            }
        }
    }

    Ok(())
}
