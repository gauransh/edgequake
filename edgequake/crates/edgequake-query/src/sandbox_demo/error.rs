//! Errors for the sandbox demo path.
//!
//! Every variant is fatal to something visible: at startup they abort the
//! process before the demo router exists, and at query time they surface as
//! an HTTP failure. There is deliberately no variant that means "answered
//! anyway, without graph-core".

use thiserror::Error;

/// Result alias for the sandbox demo.
pub type Result<T> = std::result::Result<T, SandboxDemoError>;

/// Everything that can stop the sandbox demo.
#[derive(Debug, Error)]
pub enum SandboxDemoError {
    /// A bundle file could not be read.
    #[error("sandbox pack: cannot read {path}: {source}")]
    Io {
        /// Path that failed.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// graph-core refused the activation bundle, or refused a query.
    #[error("graph-core: {0}")]
    GraphCore(#[from] itsm_graph_core::error::KgeError),

    /// `snapshot.json` is unusable.
    #[error("sandbox snapshot: {0}")]
    Snapshot(String),

    /// `snapshot.json` does not match the activation bundle.
    #[error(
        "sandbox snapshot {field} mismatch: activation.json expects {expected:?}, \
         snapshot.json has {actual:?}"
    )]
    SnapshotMismatch {
        /// Which identity field disagreed.
        field: &'static str,
        /// What `activation.json` requires.
        expected: String,
        /// What the snapshot carries.
        actual: String,
    },

    /// The snapshot and the model disagree about the id space.
    #[error(
        "{count} governed resource(s) in snapshot {revision} are not in model {model_version}'s \
         id space (first: {first:?}); the snapshot and the model pack are not a matched pair"
    )]
    UnmappedResources {
        /// How many resources failed to resolve.
        count: usize,
        /// The first one, for a concrete error message.
        first: String,
        /// Snapshot revision.
        revision: String,
        /// Model version.
        model_version: String,
    },

    /// The embedding provider failed or returned an unusable batch.
    #[error("embedding provider: {0}")]
    Embedding(String),

    /// The request was rejected before any work happened.
    #[error("{0}")]
    BadRequest(String),
}
