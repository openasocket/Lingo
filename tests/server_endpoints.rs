//! End-to-end tests for the example HTTP server's model-named routes
//! (`/sonar/*`, `/labse/*`), against the real binary and the real models.
//!
//! The tests spawn `target/<profile>/examples/server` on a free loopback port
//! and talk plain HTTP/1.1 to it. They are `#[ignore]`d because they need the
//! LaBSE and SONAR weights in `~/.cache/lingo` and run real inference. Build
//! the server first, with the same profile and features:
//!
//! ```bash
//! export LINGO_ACCEPT_LICENSE=1
//! cargo build --release --example server --features server,metal
//! cargo test --release --features server,metal --test server_endpoints -- --ignored --nocapture
//! ```

use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

const MAX_BATCH: usize = 4;
const MAX_TEXT_BYTES: usize = 2048;

/// The server process; killed when dropped so a failed assertion never leaks it.
struct Server {
    child: Child,
    addr: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The built example server. `LINGO_SERVER_BIN` overrides the lookup; otherwise
/// the first ancestor of this test executable holding `examples/server` wins
/// (test binaries live under `<target>/<profile>/deps/`).
fn server_binary() -> PathBuf {
    if let Some(p) = std::env::var_os("LINGO_SERVER_BIN") {
        return PathBuf::from(p);
    }
    let name = format!("server{}", std::env::consts::EXE_SUFFIX);
    let exe = std::env::current_exe().expect("current_exe");
    exe.ancestors()
        .map(|dir| dir.join("examples").join(&name))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| {
            panic!(
                "examples/server not found above {}; build it first with the same profile: \
                 `cargo build [--release] --example server --features server,<accelerator>`",
                exe.display()
            )
        })
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// Start the server (optionally with `HOME` redirected) and wait until `/health` answers.
fn start_server(home: Option<&std::path::Path>) -> Server {
    let addr = format!("127.0.0.1:{}", free_port());
    let mut cmd = Command::new(server_binary());
    cmd.env("LINGO_ACCEPT_LICENSE", "1")
        .env("LINGO_SERVER_ADDR", &addr)
        .env("LINGO_MAX_BATCH", MAX_BATCH.to_string())
        .env("LINGO_MAX_TEXT_BYTES", MAX_TEXT_BYTES.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    if let Some(h) = home {
        cmd.env("HOME", h);
    }
    let mut server = Server {
        child: cmd.spawn().expect("spawn server"),
        addr,
    };

    let deadline = Instant::now() + Duration::from_secs(180);
    loop {
        if let Some(status) = server.child.try_wait().unwrap() {
            panic!("server exited during startup: {status}");
        }
        if TcpStream::connect(&server.addr).is_ok() {
            return server;
        }
        assert!(
            Instant::now() < deadline,
            "server did not start listening within 180 s"
        );
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Minimal HTTP/1.1 client: one request per connection, `Connection: close`.
fn request(server: &Server, method: &str, path: &str, body: Option<&Value>) -> (u16, Value) {
    let mut stream = TcpStream::connect(&server.addr).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(300)))
        .unwrap();
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{payload}",
        server.addr,
        payload.len()
    )
    .unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    let raw = String::from_utf8(raw).expect("utf-8 response");
    let (head, body) = raw.split_once("\r\n\r\n").expect("HTTP response");
    assert!(
        !head
            .to_ascii_lowercase()
            .contains("transfer-encoding: chunked"),
        "unexpected chunked response:\n{head}"
    );
    let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    let json = serde_json::from_str(body).unwrap_or_else(|_| Value::String(body.to_string()));
    (status, json)
}

fn post(server: &Server, path: &str, body: Value) -> (u16, Value) {
    request(server, "POST", path, Some(&body))
}

fn post_ok(server: &Server, path: &str, body: Value) -> Value {
    let (status, json) = post(server, path, body);
    assert_eq!(status, 200, "POST {path} -> {status}: {json}");
    json
}

fn vector(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore]
fn sonar_and_labse_routes_against_real_models() {
    let server = start_server(None);

    let (status, health) = request(&server, "GET", "/health", None);
    assert_eq!(status, 200);
    assert_eq!(
        health["sonar_downloaded"], true,
        "SONAR not downloaded: run `lingo download sonar`"
    );
    assert_eq!(
        health["labse_downloaded"], true,
        "LaBSE not downloaded: run `lingo download labse`"
    );
    assert_eq!(
        health["sonar_loaded"], true,
        "SONAR failed to load; see server log"
    );
    assert_eq!(
        health["labse_loaded"], true,
        "LaBSE failed to load; see server log"
    );

    // ---- /sonar/embed ----
    let r = post_ok(&server, "/sonar/embed", json!({"text": "Hello world"}));
    let en_default = vector(&r["embedding"]);
    assert_eq!(en_default.len(), 1024);
    assert_eq!(r["dim"], 1024);
    assert_eq!(r["dimensions"], 1024);
    assert!(r["duration_ms"].is_u64());
    assert!(
        (norm(&en_default) - 1.0).abs() < 1e-3,
        "SONAR embedding not L2-normalized"
    );

    let en = vector(
        &post_ok(
            &server,
            "/sonar/embed",
            json!({"text": "Hello world", "lang": "en"}),
        )["embedding"],
    );
    assert!(
        max_abs_diff(&en, &en_default) < 1e-6,
        "lang must default to en"
    );
    let en_nllb = vector(
        &post_ok(
            &server,
            "/sonar/embed",
            json!({"text": "Hello world", "lang": "eng_Latn"}),
        )["embedding"],
    );
    assert!(
        max_abs_diff(&en, &en_nllb) < 1e-6,
        "NLLB code must resolve like the ISO code"
    );

    let fr = vector(
        &post_ok(
            &server,
            "/sonar/embed",
            json!({"text": "Bonjour le monde", "lang": "fr"}),
        )["embedding"],
    );
    let fr_as_en =
        vector(&post_ok(&server, "/sonar/embed", json!({"text": "Bonjour le monde"}))["embedding"]);
    assert!(
        max_abs_diff(&fr, &fr_as_en) > 1e-3,
        "SONAR embedding must depend on lang"
    );

    // ---- /sonar/score ----
    let r = post_ok(
        &server,
        "/sonar/score",
        json!({"text1": "Hello world", "text2": "Bonjour le monde", "lang1": "en", "lang2": "fr"}),
    );
    let score = r["score"].as_f64().unwrap() as f32;
    assert!(r["duration_ms"].is_u64());
    assert!(
        score > 0.8,
        "translation pair scored {score}, expected ~0.86"
    );
    assert!(
        (score - dot(&en, &fr)).abs() < 1e-4,
        "score must be the cosine of the two embeddings"
    );

    let unrelated = post_ok(
        &server,
        "/sonar/score",
        json!({"text1": "The stock market fell sharply today", "text2": "Le chat dort sur le canapé", "lang2": "fr"}),
    )["score"]
        .as_f64()
        .unwrap() as f32;
    assert!(unrelated < 0.5, "unrelated pair scored {unrelated}");

    let defaulted = post_ok(
        &server,
        "/sonar/score",
        json!({"text1": "Hello world", "text2": "Bonjour le monde"}),
    )["score"]
        .as_f64()
        .unwrap() as f32;
    assert!(
        (defaulted - dot(&en, &fr_as_en)).abs() < 1e-4,
        "lang1/lang2 must default to en"
    );

    // ---- /sonar/embed_batch ----
    let r = post_ok(
        &server,
        "/sonar/embed_batch",
        json!({"texts": ["Bonjour le monde", "Bonne nuit"], "lang": "fr"}),
    );
    assert_eq!(r["count"], 2);
    assert_eq!(r["dim"], 1024);
    assert_eq!(r["dimensions"], 1024);
    assert!(r["duration_ms"].is_u64());
    let batch: Vec<Vec<f32>> = r["embeddings"]
        .as_array()
        .unwrap()
        .iter()
        .map(vector)
        .collect();
    assert_eq!(batch.len(), 2);
    assert!(
        max_abs_diff(&batch[0], &fr) < 1e-5,
        "batch item must match single embed"
    );
    assert!((norm(&batch[1]) - 1.0).abs() < 1e-3);

    // ---- SONAR input validation (same limits as the other routes) ----
    let (s, e) = post(
        &server,
        "/sonar/embed",
        json!({"text": "Hello", "lang": "xx-not-a-lang"}),
    );
    assert_eq!(s, 400, "{e}");
    let (s, e) = post(
        &server,
        "/sonar/score",
        json!({"text1": "a", "text2": "b", "lang2": "la"}),
    );
    assert_eq!(s, 400, "{e}");
    let (s, e) = post(&server, "/sonar/embed_batch", json!({"texts": []}));
    assert_eq!(s, 400, "{e}");
    let too_many: Vec<String> = (0..=MAX_BATCH).map(|i| format!("text {i}")).collect();
    let (s, e) = post(&server, "/sonar/embed_batch", json!({ "texts": too_many }));
    assert_eq!(s, 400, "{e}");
    let too_long = "a".repeat(MAX_TEXT_BYTES + 1);
    let (s, e) = post(&server, "/sonar/embed", json!({ "text": too_long }));
    assert_eq!(s, 400, "{e}");
    let (s, e) = post(
        &server,
        "/sonar/score",
        json!({"text1": "a", "text2": too_long}),
    );
    assert_eq!(s, 400, "{e}");
    let (s, e) = post(
        &server,
        "/sonar/embed_batch",
        json!({"texts": ["ok", too_long]}),
    );
    assert_eq!(s, 400, "{e}");

    // ---- /labse/* are aliases of the original LaBSE routes ----
    let a = post_ok(&server, "/labse/embed", json!({"text": "Hello world"}));
    let b = post_ok(&server, "/embed", json!({"text": "Hello world"}));
    assert_eq!(a, b);
    assert_eq!(a["dimensions"], 768);
    let labse = vector(&a["embedding"]);
    assert_eq!(labse.len(), 768);
    assert!(
        (norm(&labse) - 1.0).abs() < 1e-3,
        "LaBSE embedding not L2-normalized"
    );

    let a = post_ok(
        &server,
        "/labse/embed_batch",
        json!({"texts": ["Hello world", "Hola mundo"]}),
    );
    let b = post_ok(
        &server,
        "/embed_batch",
        json!({"texts": ["Hello world", "Hola mundo"]}),
    );
    assert_eq!(a, b);
    assert_eq!(a["count"], 2);

    let pair = json!({"text1": "Hello world", "text2": "Bonjour le monde"});
    let a = post_ok(&server, "/labse/score", pair.clone());
    let b = post_ok(&server, "/score", pair);
    assert_eq!(a["score"], b["score"]);
    assert!(a["score"].as_f64().unwrap() > 0.9);

    let (s, e) = post(&server, "/labse/embed_batch", json!({"texts": []}));
    assert_eq!(s, 400, "{e}");
}

#[test]
#[ignore]
fn model_routes_return_503_when_models_are_not_downloaded() {
    // An empty HOME means no ~/.cache/lingo: every model is "not downloaded".
    let home = std::env::temp_dir().join(format!("lingo-server-test-{}", std::process::id()));
    std::fs::create_dir_all(&home).unwrap();
    let server = start_server(Some(&home));

    let (_, health) = request(&server, "GET", "/health", None);
    assert_eq!(health["sonar_downloaded"], false);
    assert_eq!(health["labse_downloaded"], false);

    for (path, body) in [
        ("/sonar/embed", json!({"text": "Hello world"})),
        (
            "/sonar/embed_batch",
            json!({"texts": ["Hello world"], "lang": "fr"}),
        ),
        ("/sonar/score", json!({"text1": "a", "text2": "b"})),
        ("/labse/embed", json!({"text": "Hello world"})),
        ("/labse/embed_batch", json!({"texts": ["Hello world"]})),
        ("/labse/score", json!({"text1": "a", "text2": "b"})),
    ] {
        let (status, err) = post(&server, path, body);
        assert_eq!(status, 503, "POST {path}: {err}");
        let msg = err["error"].as_str().unwrap();
        assert!(
            !msg.contains(home.to_str().unwrap()),
            "error leaks a server path: {msg}"
        );
    }

    drop(server);
    let _ = std::fs::remove_dir_all(&home);
}
