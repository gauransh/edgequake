//! `edgequake-sandbox-demo` — the CrystalOS sandbox demo, and nothing else.
//!
//! The main `edgequake` binary also serves this demo whenever
//! `EDGEQUAKE_SANDBOX_PACK_DIR` is set, but it requires `DATABASE_URL`, runs
//! migrations, and starts a worker pool — none of which the demo query path
//! touches. Its index is one frozen `snapshot.json` held in memory and its
//! reranker is linked in; there is no storage backend on the hot path at all.
//!
//! So this binary starts the same two routes over the same
//! [`SandboxDemoQueryEngine`] with no database, no migrations, and no
//! background work. Fewer moving parts between the presenter and the query is
//! the entire point: what is being demonstrated is the retrieval → embedded
//! rerank slice, not EdgeQuake's storage stack.
//!
//! ```bash
//! EDGEQUAKE_SANDBOX_PACK_DIR=/abs/path/to/pack \
//! EDGEQUAKE_EMBEDDING_PROVIDER=ollama EDGEQUAKE_EMBEDDING_MODEL=nomic-embed-text \
//!   edgequake-sandbox-demo
//! # -> http://127.0.0.1:8080/demo/sandbox
//! ```
//!
//! Startup is a gate: a missing, corrupt, or mismatched pack exits non-zero
//! rather than serving a demo that cannot use graph-core.

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use edgequake_api::safety_limits::create_safe_embedding_provider;
use edgequake_query::sandbox_demo::{PACK_DIR_ENV, SandboxDemoQueryEngine};
use tracing::info;

/// Embedding provider name, e.g. `ollama`, `openai`, `lmstudio`, `mock`.
const PROVIDER_ENV: &str = "EDGEQUAKE_EMBEDDING_PROVIDER";
/// Embedding model name for that provider.
const MODEL_ENV: &str = "EDGEQUAKE_EMBEDDING_MODEL";
/// Embedding dimension the provider produces.
const DIMENSION_ENV: &str = "EDGEQUAKE_EMBEDDING_DIMENSION";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,edgequake_query::sandbox_demo=debug".into()),
        )
        .init();

    let provider = std::env::var(PROVIDER_ENV).unwrap_or_else(|_| "ollama".into());
    let model = std::env::var(MODEL_ENV).unwrap_or_else(|_| "nomic-embed-text".into());
    let dimension: usize = std::env::var(DIMENSION_ENV)
        .ok()
        .and_then(|d| d.parse().ok())
        .unwrap_or(768);

    let embedding = create_safe_embedding_provider(&provider, &model, dimension)
        .map_err(|e| anyhow::anyhow!(e.to_string()))
        .with_context(|| format!("building the {provider} embedding provider ({model})"))?;

    // Loading embeds one document per governed resource, so a bad pack — or an
    // unreachable embedding provider — fails here rather than on stage.
    let engine = SandboxDemoQueryEngine::from_env(embedding)
        .await
        .with_context(|| format!("loading the sandbox demo pack named by {PACK_DIR_ENV}"))?
        .with_context(|| format!("{PACK_DIR_ENV} is not set; there is nothing to serve"))?;
    let engine = Arc::new(engine);

    let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .with_context(|| format!("invalid listen address {host}:{port}"))?;

    info!(
        pack = %engine.pack_dir().display(),
        workspace = engine.workspace_id(),
        model_version = %engine.graph_core().model_version(),
        snapshot_revision = engine.index().revision(),
        embedding_model = engine.embedding_model(),
        "sandbox demo ready at http://{addr}/demo/sandbox"
    );

    let app = edgequake_api::handlers::sandbox_demo::routes(engine);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    axum::serve(listener, app).await.context("serving")?;
    Ok(())
}
