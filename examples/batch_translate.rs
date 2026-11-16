//! Batch-translate a JSONL file.
//!
//! Each input line is a JSON object with `text`, `source` and `target`
//! (ISO 639-1/3 or NLLB codes). Each output line is the same object plus
//! either `translation` and `duration_ms`, or `error`. Other fields pass
//! through unchanged, so ids survive the round trip. A malformed line yields
//! an `error` line instead of aborting the run; blank lines are skipped. The
//! output is flushed after every line, so a partial run is still usable.
//!
//! Translation uses `TranslateConfig::strict()` (one sentence at a time,
//! rejecting empty, dangling-clause and repeated-token output), matching the
//! sentence-level integrity checks of the candle-nllb `audit_translations`
//! example this replaces.
//!
//! ```bash
//! LINGO_ACCEPT_LICENSE=1 cargo run --release --example batch_translate --features metal -- \
//!     input.jsonl output.jsonl [MODEL_DIR]
//! ```

use lingo::{NllbTranslator, TranslateConfig};
use serde_json::Value;
use std::io::{BufRead, Write};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 3 || args.len() == 4,
        "usage: batch_translate INPUT.jsonl OUTPUT.jsonl [MODEL_DIR]"
    );
    let model_dir = args.get(3).map(std::path::PathBuf::from);
    let translator = NllbTranslator::new(model_dir)?.with_config(TranslateConfig::strict());

    let input = std::io::BufReader::new(std::fs::File::open(&args[1])?);
    let mut output = std::io::BufWriter::new(std::fs::File::create(&args[2])?);
    let (mut ok, mut failed) = (0usize, 0usize);

    for (index, line) in input.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record = match serde_json::from_str::<Value>(&line) {
            Ok(mut request @ Value::Object(_)) => {
                // Drop results from an earlier run so a retry never leaves a
                // stale `error` beside a fresh translation.
                if let Some(map) = request.as_object_mut() {
                    for key in ["translation", "duration_ms", "error"] {
                        map.remove(key);
                    }
                }
                let field =
                    |name: &str| request.get(name).and_then(Value::as_str).map(str::to_owned);
                match (field("text"), field("source"), field("target")) {
                    (Some(text), Some(source), Some(target)) => {
                        match translator.translate(&text, &source, &target).await {
                            Ok(result) => {
                                request["translation"] = result.text.into();
                                request["duration_ms"] = result.duration_ms.into();
                            }
                            Err(error) => request["error"] = error.to_string().into(),
                        }
                    }
                    _ => request["error"] = "expected string fields text, source and target".into(),
                }
                request
            }
            Ok(_) => serde_json::json!({ "line": index + 1, "error": "expected a JSON object" }),
            Err(error) => {
                serde_json::json!({ "line": index + 1, "error": format!("invalid JSON: {error}") })
            }
        };
        if record.get("error").is_some() {
            failed += 1;
        } else {
            ok += 1;
        }
        writeln!(output, "{record}")?;
        output.flush()?;
    }

    eprintln!("translated {ok}, failed {failed}");
    Ok(())
}
