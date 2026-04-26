---
type: report
title: lingo Release Audit
created: 2026-04-26
tags:
  - release-audit
  - lingo
  - rust
related:
  - '[[Release-Audit-And-Hardening]]'
---

# Summary

I did not find packaging or feature-gating failures that currently block publishing the crate itself. `cargo package --allow-dirty --no-verify`, `cargo test`, `cargo test --features cli`, `cargo test --features server`, `cargo check --examples`, `cargo check --examples --features cli`, and `cargo check --examples --features server` all completed successfully on April 26, 2026.

I did find one production blocker if the advertised HTTP server mode is intended to be shipped as a production surface, plus several follow-up risks that should be called out before release.

# Findings

## High

### Unhardened HTTP server is exposed as a production-ready surface

- Severity: High
- Files: `README.md:145`, `examples/server.rs:80`, `examples/server.rs:209`, `examples/server.rs:220`

The README presents server mode as a normal usage path, but the implementation is still an unauthenticated example server. It binds directly to `0.0.0.0:3000`, exposes expensive translation and embedding endpoints to every caller, and does not add request body limits, timeouts, concurrency caps, rate limiting, authentication, or any other abuse controls.

That means a user following the documented server path can expose a compute-heavy model service to the network with no guardrails. For a public release, this is not just missing polish; it is an unsafe default for a network-facing mode.

Recommended action: either downgrade server mode to explicitly non-production example status in the docs, or add basic hardening before claiming it as a release-ready capability.

## Medium

### SONAR capability still depends on an undeclared external Python toolchain

- Severity: Medium
- Files: `README.md:3`, `README.md:271`, `src/download.rs:89`, `src/cli.rs:238`

The top-level README says "No Python required," but SONAR download/setup still shells out to `python3` and requires external Python packages for conversion. The CLI does surface that requirement, but the package still advertises SONAR as a first-class capability while relying on tooling that is neither vendored nor validated at install time.

This is not a crate-packaging failure, but it is a release-readiness gap because first-run setup for one of the headline features can still fail on a machine that otherwise installed the Rust package correctly.

Recommended action: either narrow the top-level claim, or ship a fully Rust-native SONAR setup path.

### Runtime validation coverage is too thin for a model-serving release

- Severity: Medium
- Files: `src/license.rs:68`, `examples/server.rs:1`, `README.md:145`

The repository currently has only four unit tests, all under the license helper module. There are no integration tests for model loading, download workflows, translation correctness, similarity endpoints, or server behavior. The successful checks above prove the feature sets compile, but they do not prove that documented runtime flows work on a clean machine or that the server behaves safely under malformed or heavy traffic.

This is the main residual risk outside the server hardening blocker.

Recommended action: add at least one end-to-end validation path for each published surface:

- CLI first-run download plus inference
- library load plus inference with prepositioned weights
- HTTP server startup plus endpoint smoke tests

# Packaging and Feature Audit Notes

- `Cargo.toml` includes the expected release metadata: package name, version, description, license file, repository, keywords, categories, and README.
- `cargo package --allow-dirty --no-verify` succeeded, so there is no confirmed crates.io packaging blocker from manifest shape or missing packaged files.
- Optional feature combinations checked cleanly:
  - default
  - `cli`
  - `server`
  - examples with default features
  - examples with `cli`
  - examples with `server`

# Residual Risks

- The `/health` endpoint reports whether model files exist, not whether the models are actually loaded in memory, so its field names overstate readiness.
- The public docs encourage direct server use but do not state operational limits or deployment constraints.
- Large-model first-run behavior was not validated end-to-end in this audit because the model assets were not downloaded during review.
