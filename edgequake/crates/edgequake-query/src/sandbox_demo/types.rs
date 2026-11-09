//! Request and response DTOs for `POST /api/v1/query/sandbox-demo`.
//!
//! The response is deliberately verbose about *identity*: a score means
//! nothing unless the room can see which snapshot revision, which model
//! version, and which entity-mapping hash produced it.

use serde::{Deserialize, Serialize};

/// Default number of semantic candidates handed to graph-core.
pub const DEFAULT_CANDIDATE_K: usize = 20;
/// Default number of results returned.
pub const DEFAULT_TOP_K: usize = 5;
/// Upper bound on `candidate_k`, so one request cannot ask the demo to
/// rerank an unbounded slice of the graph.
pub const MAX_CANDIDATE_K: usize = 200;

fn default_candidate_k() -> usize {
    DEFAULT_CANDIDATE_K
}
fn default_top_k() -> usize {
    DEFAULT_TOP_K
}

/// One sandbox demo query.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SandboxDemoRequest {
    /// Must equal the workspace the loaded pack was built for.
    pub workspace_id: String,
    /// The natural-language question.
    pub query: String,
    /// The explicit, frozen anchor, e.g. `agent:claude-code-demo`. Never
    /// inferred from the question.
    pub anchor_external_id: String,
    /// Semantic candidates to retrieve and rerank.
    #[serde(default = "default_candidate_k")]
    pub candidate_k: usize,
    /// Results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

/// The anchor, echoed back with its graph identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorInfo {
    /// graph-core external id.
    pub external_id: String,
    /// Display label.
    pub label: String,
    /// Entity type.
    pub entity_type: String,
    /// Runs this profile initiated in the frozen snapshot.
    pub run_count: usize,
}

/// The blend weights actually applied: the default `semantic=1,
/// structural=0`, the bundle's frozen weights, or an explicit override.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AppliedWeights {
    /// Weight on the EdgeQuake semantic score.
    pub semantic: f32,
    /// Weight on EdgeQuake's graph score (unused on this path).
    pub graph: f32,
    /// Weight on the graph-core structural score.
    pub structural: f32,
    /// Weight on link prediction (not part of this path).
    pub link: f32,
}

/// Why this candidate is connected to the anchor, in graph terms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSummary {
    /// Relation on the final edge of the path.
    pub relation: String,
    /// Action detail on that edge, when it carried one.
    pub action: Option<String>,
    /// True when the run in the path was initiated by the anchor.
    pub via_anchor_run: bool,
    /// Distinct runs that touched this resource.
    pub runs_touching: usize,
    /// How many of those the anchor initiated.
    pub anchor_runs_touching: usize,
    /// `anchor_runs_touching / anchor.run_count`, as a percentage.
    pub anchor_run_share_pct: f32,
    /// Permitted access edges into this resource.
    pub allowed_edges: usize,
    /// Refused access edges into this resource.
    pub denied_edges: usize,
}

/// One graph-core-approved result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxDemoResult {
    /// graph-core external id, preserved exactly.
    pub external_id: String,
    /// Display label.
    pub label: String,
    /// Entity type (always `resource` on this path).
    pub entity_type: String,
    /// Raw cosine between the question and this resource's document.
    pub semantic_cosine: f32,
    /// The semantic component actually blended, `(cosine + 1) / 2`.
    pub semantic_score: f32,
    /// graph-core's anchor-to-candidate structural similarity in `[0, 1]`.
    pub structural_score: f32,
    /// The blended score that produced the final order.
    pub final_score: f32,
    /// 1-based rank in the semantic-only order.
    pub semantic_rank: usize,
    /// 1-based rank after graph-core reranked.
    pub final_rank: usize,
    /// `[anchor, run, relation, resource]`.
    pub evidence_path: Vec<String>,
    /// The counts behind that path.
    pub evidence: EvidenceSummary,
}

/// One entry of the semantic-only order, shown for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticOrderEntry {
    /// graph-core external id.
    pub external_id: String,
    /// Display label.
    pub label: String,
    /// Raw cosine.
    pub semantic_cosine: f32,
    /// 1-based semantic rank.
    pub semantic_rank: usize,
    /// 1-based rank after graph-core reranked the same candidate set.
    pub final_rank: usize,
}

/// The sandbox demo response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxDemoResponse {
    /// Always `true`. A response cannot be produced without graph-core;
    /// the field exists so the room can see the claim on screen.
    pub graph_core_applied: bool,
    /// Workspace of the pack.
    pub workspace_id: String,
    /// The question that was asked.
    pub query: String,
    /// The anchor it was asked about.
    pub anchor: AnchorInfo,
    /// graph-core model version that produced the order.
    pub model_version: String,
    /// Entity-mapping hash — the id space the order is in.
    pub mapping_hash: String,
    /// Snapshot revision the model was trained on.
    pub snapshot_revision: String,
    /// SHA-256 of the indexed `snapshot.json`.
    pub snapshot_sha256: String,
    /// The weights actually applied.
    pub weights: AppliedWeights,
    /// Embedding model that produced the semantic scores.
    pub embedding_model: String,
    /// Entities and relationships in the in-memory index.
    pub snapshot_entities: usize,
    /// Relationship count of the in-memory index.
    pub snapshot_relationships: usize,
    /// Candidates retrieved and reranked.
    pub candidate_k: usize,
    /// Results returned.
    pub top_k: usize,
    /// Time inside the embedded graph-core rerank.
    pub graph_core_latency_ms: f64,
    /// Time inside semantic retrieval (embed + cosine).
    pub semantic_latency_ms: f64,
    /// Total server-side time for the query.
    pub total_latency_ms: f64,
    /// Per-stage split of the same query. Additive: older clients that do
    /// not know the field still parse, and older servers yield defaults.
    #[serde(default)]
    pub timings: StageTimings,
    /// The opt-in structural scorer, e.g. `popularity` or
    /// `blend:knn=0.5,popularity=0.5`. Present only when
    /// `EDGEQUAKE_SANDBOX_STRUCTURAL_SCORER` or explicit
    /// `EDGEQUAKE_SANDBOX_BLEND_WEIGHTS` were set (`bundle` alone does not
    /// count), so the `bundle` response is byte-identical to before the
    /// knob existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_scorer: Option<String>,
    /// The routed query intent. Absent when
    /// `EDGEQUAKE_SANDBOX_INTENT_ROUTING` is off (the default), so that
    /// response is byte-identical to before the knob existed; with routing
    /// on, `"next_touch"` when the query matched the intent lexicon and
    /// `null` when it took the default path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<Option<String>>,
    /// The graph-core-approved order, truncated to `top_k`.
    pub results: Vec<SandboxDemoResult>,
    /// The semantic-only order over the same candidate set, for contrast.
    pub semantic_order: Vec<SemanticOrderEntry>,
}

/// Where the query's embedding vector came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryEmbeddingSource {
    /// This request asked the provider for a single text.
    #[default]
    Miss,
    /// Served from the engine's query-embedding cache; no provider call.
    Hit,
    /// Joined an identical query's in-flight provider call.
    Coalesced,
    /// Embedded by one provider call that carried other queries too.
    Batched,
}

/// Disjoint wall-clock stages of one query, in milliseconds.
///
/// Each span is measured at its own boundaries; they are not nested, so
/// they never double count. Small unmeasured gaps between stages mean they
/// sum to slightly less than `total_ms`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StageTimings {
    /// Obtaining the normalized question vector (provider, cache, or wait).
    pub provider_embed_ms: f64,
    /// The brute-force cosine scan over every resource.
    pub semantic_scan_ms: f64,
    /// Ordering the scan and keeping the top `candidate_k`.
    pub candidate_select_ms: f64,
    /// The embedded graph-core rerank.
    pub graph_core_rerank_ms: f64,
    /// Final selection, evidence paths, and the semantic-order contrast.
    pub evidence_ms: f64,
    /// Total server-side time; equals `total_latency_ms`.
    pub total_ms: f64,
    /// How the question vector was obtained.
    pub query_embedding: QueryEmbeddingSource,
}
