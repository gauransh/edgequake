//! The CrystalOS sandbox → EdgeQuake → embedded graph-core demo path.
//!
//! One vertical slice, all real code:
//!
//! ```text
//! frozen sandbox evidence (itsm-sandbox-policy-engine)
//!   -> snapshot.json + a trained graph-core model pack
//!   -> EdgeQuake embeds one document per governed resource and retrieves
//!      semantic candidates
//!   -> the EMBEDDED graph-core runtime reranks them in this same process
//!   -> EdgeQuake returns the graph-core-blended order, a score breakdown,
//!      an evidence path, and the latency
//! ```
//!
//! **graph-core is mandatory here, not advisory.** The engine refuses to
//! load without a valid, matching model pack; a query fails if the anchor
//! or any candidate is outside the model's id space; and a successful
//! response always carries `graph_core_applied: true` because no other
//! response can be produced. There is no semantic-only fallback and no
//! switch that disables the reranker.
//!
//! The default blend weights are `semantic=1.0, structural=0.0`: a gold
//! relevance evaluation found semantic-only ranking beats the bundle's
//! frozen `0.6 / 0.4` blend. The reranker still runs and its structural
//! score is still returned; `EDGEQUAKE_SANDBOX_BLEND_WEIGHTS=bundle`
//! restores the frozen weights. See [`structural`].
//!
//! This is a focused query adapter, not a storage migration. The same
//! [`itsm_graph_core::serving::embedded_runtime::EmbeddedGraphCoreRuntime`]
//! interface is what would later slot into EdgeQuake Local mode between
//! vector candidate retrieval and graph expansion — the seam that already
//! exists in `engine_impl::modes::local`.

pub mod documents;
pub mod engine;
pub mod error;
pub mod query_embedding;
pub mod snapshot;
pub mod structural;
pub mod types;

pub use engine::{SandboxDemoQueryEngine, PACK_DIR_ENV};
pub use error::{Result, SandboxDemoError};
pub use query_embedding::{
    QueryEmbedConfig, DEFAULT_QUERY_EMBED_CACHE_BYTES, DEFAULT_QUERY_EMBED_CACHE_ENTRIES,
    QUERY_EMBED_CACHE_BYTES_ENV, QUERY_EMBED_CACHE_ENV,
};
pub use snapshot::{
    EvidencePath, GraphNode, GraphProjection, ResourceProfile, SandboxSnapshotIndex, SnapshotEdge,
    SnapshotNode,
};
pub use structural::{
    BlendWeightsChoice, ScorerKind, ScoringConfig, StructuralScorer, WeightOverride,
    BLEND_WEIGHTS_ENV, BUNDLE_WEIGHTS, STRUCTURAL_SCORER_ENV,
};
pub use types::{
    AnchorInfo, AppliedWeights, EvidenceSummary, QueryEmbeddingSource, SandboxDemoRequest,
    SandboxDemoResponse, SandboxDemoResult, SemanticOrderEntry, StageTimings, DEFAULT_CANDIDATE_K,
    DEFAULT_TOP_K, MAX_CANDIDATE_K,
};
