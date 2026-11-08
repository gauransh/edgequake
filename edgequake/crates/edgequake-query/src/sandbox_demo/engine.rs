//! The sandbox demo query engine.
//!
//! One frozen activation bundle, one in-memory index, one embedded
//! graph-core runtime. The whole query path is local: after
//! [`SandboxDemoQueryEngine::load`] returns there are no graph-core HTTP,
//! S3, S3 Vectors, or DynamoDB calls, and the only network call on the hot
//! path is the embedding provider embedding the question.
//!
//! # Startup is a gate, not a best effort
//!
//! [`SandboxDemoQueryEngine::load`] fails — and the caller must not serve
//! the demo route — when the pack is missing, the model refuses to load,
//! `snapshot.json` does not hash to what `activation.json` names, the
//! snapshot is for another workspace or revision, or any governed resource
//! in the snapshot is outside the model's id space. There is no
//! semantic-only mode to fall back to.
//!
//! # Execution order is fixed
//!
//! 1. Embed the question with EdgeQuake's configured embedding provider.
//! 2. Retrieve the top `candidate_k` resources by semantic cosine.
//! 3. Rerank **all** of them through the embedded graph-core runtime.
//! 4. Take the final `top_k` and build a one-hop-through-a-run evidence
//!    path for each from the in-memory graph.
//! 5. Return only the graph-core-approved order.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use edgequake_llm::traits::EmbeddingProvider;
use itsm_graph_core::serving::embedded_runtime::{
    EmbeddedGraphCoreRuntime, ExternalCandidate, SNAPSHOT_FILE,
};
use tracing::info;

use super::documents::resource_document;
use super::error::{Result, SandboxDemoError};
use super::snapshot::{SandboxSnapshotIndex, AGENT_TYPE, INITIATED};
use super::types::{
    AnchorInfo, AppliedWeights, EvidenceSummary, QueryEmbeddingSource, SandboxDemoRequest,
    SandboxDemoResponse, SandboxDemoResult, SemanticOrderEntry, StageTimings, MAX_CANDIDATE_K,
};

/// Environment variable naming the activation bundle directory. When it is
/// set, EdgeQuake either serves the demo or refuses to start.
pub const PACK_DIR_ENV: &str = "EDGEQUAKE_SANDBOX_PACK_DIR";

/// A loaded sandbox demo generation.
pub struct SandboxDemoQueryEngine {
    index: Arc<SandboxSnapshotIndex>,
    embedding: Arc<dyn EmbeddingProvider>,
    graph_core: Arc<EmbeddedGraphCoreRuntime>,

    /// Node index of each embedded resource, parallel to [`Self::vectors`].
    resource_nodes: Vec<u32>,
    /// Row-normalized document vectors, `[resource_nodes.len() * dim]`.
    vectors: Vec<f32>,
    dim: usize,
    embedding_model: String,
    pack_dir: PathBuf,
}

impl std::fmt::Debug for SandboxDemoQueryEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxDemoQueryEngine")
            .field("pack_dir", &self.pack_dir)
            .field("model_version", &self.graph_core.model_version())
            .field("snapshot_revision", &self.index.revision())
            .field("resources", &self.resource_nodes.len())
            .field("embedding_model", &self.embedding_model)
            .finish()
    }
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|source| SandboxDemoError::Io {
        path: path.display().to_string(),
        source,
    })
}

/// L2-normalize in place; a zero vector stays zero (cosine 0, neutral).
fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

fn ms_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

impl SandboxDemoQueryEngine {
    /// Load the pack at `pack_dir` and prewarm every resource embedding.
    ///
    /// `embedding` must be the same provider EdgeQuake serves ordinary
    /// queries with: the demo's semantic scores are real EdgeQuake
    /// embeddings, not a fabricated score ladder.
    pub async fn load(pack_dir: &Path, embedding: Arc<dyn EmbeddingProvider>) -> Result<Self> {
        let graph_core = Arc::new(EmbeddedGraphCoreRuntime::load_dir(pack_dir)?);
        let activation = graph_core.activation();

        let index = SandboxSnapshotIndex::from_bytes(
            &read(&pack_dir.join(SNAPSHOT_FILE))?,
            &activation.snapshot_sha256,
            &activation.workspace_id,
            &activation.snapshot_revision,
        )?;

        // Every governed resource must resolve in the model's id space:
        // any candidate the retriever can pick has to be rerankable, and a
        // partial overlap means the pack and the snapshot are not a pair.
        let unmapped: Vec<&str> = index
            .resources()
            .iter()
            .map(|r| index.node(*r).external_id.as_str())
            .filter(|id| !graph_core.contains(id))
            .collect();
        if let Some(first) = unmapped.first() {
            return Err(SandboxDemoError::UnmappedResources {
                count: unmapped.len(),
                first: (*first).to_string(),
                revision: index.revision().to_string(),
                model_version: graph_core.model_version().to_string(),
            });
        }

        let resource_nodes: Vec<u32> = index.resources().to_vec();
        let documents: Vec<String> = resource_nodes
            .iter()
            .map(|node| resource_document(&index.resource_profile(*node)))
            .collect();

        let started = Instant::now();
        let raw = embedding
            .embed_batched(&documents)
            .await
            .map_err(|e| SandboxDemoError::Embedding(e.to_string()))?;
        if raw.len() != documents.len() {
            return Err(SandboxDemoError::Embedding(format!(
                "provider returned {} vectors for {} documents",
                raw.len(),
                documents.len()
            )));
        }
        let dim = raw
            .first()
            .map(Vec::len)
            .filter(|d| *d > 0)
            .ok_or_else(|| SandboxDemoError::Embedding("provider returned empty vectors".into()))?;

        let mut vectors = vec![0.0f32; documents.len() * dim];
        for (row, vector) in raw.into_iter().enumerate() {
            if vector.len() != dim {
                return Err(SandboxDemoError::Embedding(format!(
                    "provider returned mixed dimensions ({} then {dim})",
                    vector.len()
                )));
            }
            let slot = &mut vectors[row * dim..(row + 1) * dim];
            slot.copy_from_slice(&vector);
            normalize(slot);
        }

        info!(
            pack_dir = %pack_dir.display(),
            workspace = %activation.workspace_id,
            snapshot_revision = %activation.snapshot_revision,
            model_version = %graph_core.model_version(),
            mapping_hash = %graph_core.mapping_hash(),
            entities = index.node_count(),
            relationships = index.edge_count(),
            resources = resource_nodes.len(),
            embedding_model = embedding.model(),
            embed_ms = started.elapsed().as_secs_f64() * 1000.0,
            "sandbox demo pack loaded; graph-core is mandatory on this path"
        );

        Ok(Self {
            embedding_model: embedding.model().to_string(),
            index: Arc::new(index),
            embedding,
            graph_core,
            resource_nodes,
            vectors,
            dim,
            pack_dir: pack_dir.to_path_buf(),
        })
    }

    /// Load from [`PACK_DIR_ENV`], or `Ok(None)` when it is unset.
    ///
    /// Setting the variable is the opt-in; once set, a bad pack is fatal.
    pub async fn from_env(embedding: Arc<dyn EmbeddingProvider>) -> Result<Option<Self>> {
        let Some(dir) = std::env::var_os(PACK_DIR_ENV) else {
            return Ok(None);
        };
        let dir = PathBuf::from(dir);
        if !dir.is_absolute() {
            return Err(SandboxDemoError::BadRequest(format!(
                "{PACK_DIR_ENV} must be an absolute path, got {}",
                dir.display()
            )));
        }
        Self::load(&dir, embedding).await.map(Some)
    }

    /// The in-memory snapshot index.
    pub fn index(&self) -> &Arc<SandboxSnapshotIndex> {
        &self.index
    }

    /// The embedded graph-core runtime.
    pub fn graph_core(&self) -> &Arc<EmbeddedGraphCoreRuntime> {
        &self.graph_core
    }

    /// Workspace this generation serves.
    pub fn workspace_id(&self) -> &str {
        self.index.workspace_id()
    }

    /// Directory the pack was loaded from.
    pub fn pack_dir(&self) -> &Path {
        &self.pack_dir
    }

    /// Embedding model backing the semantic scores.
    pub fn embedding_model(&self) -> &str {
        &self.embedding_model
    }

    /// The agent profiles in the snapshot, sorted — what the demo page
    /// offers as anchors.
    pub fn anchor_candidates(&self) -> Vec<String> {
        let mut anchors: Vec<String> = (0..self.index.node_count() as u32)
            .filter(|i| self.index.node(*i).entity_type == AGENT_TYPE)
            .map(|i| self.index.node(i).external_id.clone())
            .collect();
        anchors.sort();
        anchors
    }

    fn row(&self, position: usize) -> &[f32] {
        &self.vectors[position * self.dim..(position + 1) * self.dim]
    }

    fn anchor_info(&self, anchor: u32) -> AnchorInfo {
        let node = self.index.node(anchor);
        AnchorInfo {
            external_id: node.external_id.clone(),
            label: node.label.clone(),
            entity_type: node.entity_type.clone(),
            run_count: self
                .index
                .edges(anchor)
                .iter()
                .filter(|e| e.outgoing && &*e.relation == INITIATED)
                .count(),
        }
    }

    /// Embed the question and return it L2-normalized at the index's
    /// dimension.
    async fn embed_question(&self, text: &str) -> Result<(Vec<f32>, QueryEmbeddingSource)> {
        let mut question = self
            .embedding
            .embed_batched(&[text.to_string()])
            .await
            .map_err(|e| SandboxDemoError::Embedding(e.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| SandboxDemoError::Embedding("provider returned no vector".into()))?;
        if question.len() != self.dim {
            return Err(SandboxDemoError::Embedding(format!(
                "question embedded to {} dimensions, the index is {}",
                question.len(),
                self.dim
            )));
        }
        normalize(&mut question);
        Ok((question, QueryEmbeddingSource::Miss))
    }

    /// Run one sandbox demo query.
    pub async fn query(&self, request: &SandboxDemoRequest) -> Result<SandboxDemoResponse> {
        let total_started = Instant::now();

        if request.workspace_id != self.index.workspace_id() {
            return Err(SandboxDemoError::BadRequest(format!(
                "this EdgeQuake serves workspace {:?}; the request asked for {:?}",
                self.index.workspace_id(),
                request.workspace_id
            )));
        }
        if request.query.trim().is_empty() {
            return Err(SandboxDemoError::BadRequest(
                "query must not be empty".into(),
            ));
        }
        if request.top_k == 0 || request.candidate_k == 0 {
            return Err(SandboxDemoError::BadRequest(
                "top_k and candidate_k must both be > 0".into(),
            ));
        }
        if request.candidate_k > MAX_CANDIDATE_K {
            return Err(SandboxDemoError::BadRequest(format!(
                "candidate_k must be <= {MAX_CANDIDATE_K}, got {}",
                request.candidate_k
            )));
        }

        // The anchor is explicit and frozen — never inferred from the text.
        let anchor = self
            .index
            .index_of(&request.anchor_external_id)
            .ok_or_else(|| {
                SandboxDemoError::BadRequest(format!(
                    "anchor {:?} is not in snapshot {}",
                    request.anchor_external_id,
                    self.index.revision()
                ))
            })?;
        if !self.graph_core.contains(&request.anchor_external_id) {
            return Err(SandboxDemoError::BadRequest(format!(
                "anchor {:?} is not in model {}'s id space",
                request.anchor_external_id,
                self.graph_core.model_version()
            )));
        }

        // 1-2. embed the question, retrieve by semantic cosine ------------
        let semantic_started = Instant::now();
        let (question, query_embedding) = self.embed_question(&request.query).await?;
        let provider_embed_ms = ms_since(semantic_started);

        let scan_started = Instant::now();
        let mut scored: Vec<(usize, f32)> = (0..self.resource_nodes.len())
            .map(|position| {
                let cosine: f32 = question
                    .iter()
                    .zip(self.row(position))
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    .clamp(-1.0, 1.0);
                (position, cosine)
            })
            .collect();
        let semantic_scan_ms = ms_since(scan_started);

        let select_started = Instant::now();
        // Descending cosine, ties broken on external id so warm runs are
        // byte-identical.
        scored.sort_by(|a, b| {
            b.1.total_cmp(&a.1).then_with(|| {
                self.index
                    .node(self.resource_nodes[a.0])
                    .external_id
                    .cmp(&self.index.node(self.resource_nodes[b.0]).external_id)
            })
        });
        scored.truncate(request.candidate_k);
        let candidate_select_ms = ms_since(select_started);
        let semantic_latency_ms = ms_since(semantic_started);

        // 3. the embedded graph-core rerank -------------------------------
        let candidates: Vec<ExternalCandidate> = scored
            .iter()
            .map(|(position, cosine)| {
                ExternalCandidate::new(
                    self.index
                        .node(self.resource_nodes[*position])
                        .external_id
                        .clone(),
                    // The blend contract is unit-range; a cosine is not.
                    (cosine + 1.0) / 2.0,
                )
            })
            .collect();

        let graph_core_started = Instant::now();
        let reranked =
            self.graph_core
                .rerank(&request.anchor_external_id, candidates, scored.len())?;
        let graph_core_latency_ms = ms_since(graph_core_started);

        // 4. final selection + evidence ------------------------------------
        let evidence_started = Instant::now();
        let semantic_rank_of: std::collections::HashMap<&str, (usize, f32)> = scored
            .iter()
            .enumerate()
            .map(|(rank, (position, cosine))| {
                (
                    self.index
                        .node(self.resource_nodes[*position])
                        .external_id
                        .as_str(),
                    (rank + 1, *cosine),
                )
            })
            .collect();

        let anchor_info = self.anchor_info(anchor);
        let mut results = Vec::with_capacity(request.top_k);
        for (offset, candidate) in reranked.candidates.iter().take(request.top_k).enumerate() {
            let node_index = self.index.index_of(&candidate.external_id).ok_or_else(|| {
                SandboxDemoError::Snapshot(format!(
                    "graph-core returned {:?}, which is not in the index",
                    candidate.external_id
                ))
            })?;
            let node = self.index.node(node_index);
            let structural_score = candidate.structural_score.ok_or_else(|| {
                SandboxDemoError::Snapshot(format!(
                    "graph-core returned no structural score for {:?}",
                    candidate.external_id
                ))
            })?;
            let (semantic_rank, semantic_cosine) = semantic_rank_of
                .get(candidate.external_id.as_str())
                .copied()
                .unwrap_or((0, 0.0));
            let evidence = self.index.evidence_path(anchor, node_index);
            let anchor_run_share_pct = if anchor_info.run_count == 0 {
                0.0
            } else {
                evidence.anchor_runs_touching as f32 * 100.0 / anchor_info.run_count as f32
            };

            results.push(SandboxDemoResult {
                external_id: candidate.external_id.clone(),
                label: node.label.clone(),
                entity_type: node.entity_type.clone(),
                semantic_cosine,
                semantic_score: candidate.semantic_score,
                structural_score,
                final_score: candidate.blended_score,
                semantic_rank,
                final_rank: offset + 1,
                evidence_path: evidence.path,
                evidence: EvidenceSummary {
                    relation: evidence.relation,
                    action: evidence.action,
                    via_anchor_run: evidence.via_anchor_run,
                    runs_touching: evidence.runs_touching,
                    anchor_runs_touching: evidence.anchor_runs_touching,
                    anchor_run_share_pct,
                    allowed_edges: evidence.allowed_edges,
                    denied_edges: evidence.denied_edges,
                },
            });
        }

        let final_rank_of: std::collections::HashMap<&str, usize> = reranked
            .candidates
            .iter()
            .enumerate()
            .map(|(rank, c)| (c.external_id.as_str(), rank + 1))
            .collect();
        let semantic_order: Vec<SemanticOrderEntry> = scored
            .iter()
            .enumerate()
            .map(|(rank, (position, cosine))| {
                let node = self.index.node(self.resource_nodes[*position]);
                SemanticOrderEntry {
                    final_rank: final_rank_of
                        .get(node.external_id.as_str())
                        .copied()
                        .unwrap_or(0),
                    external_id: node.external_id.clone(),
                    label: node.label.clone(),
                    semantic_cosine: *cosine,
                    semantic_rank: rank + 1,
                }
            })
            .collect();

        let weights = self.graph_core.weights();
        let evidence_ms = ms_since(evidence_started);
        let total_latency_ms = ms_since(total_started);
        Ok(SandboxDemoResponse {
            // Unconditionally true: this line is only reachable through a
            // successful embedded rerank.
            graph_core_applied: true,
            workspace_id: self.index.workspace_id().to_string(),
            query: request.query.clone(),
            anchor: anchor_info,
            model_version: reranked.model_version.to_string(),
            mapping_hash: reranked.mapping_hash,
            snapshot_revision: self.index.revision().to_string(),
            snapshot_sha256: self.index.sha256().to_string(),
            weights: AppliedWeights {
                semantic: weights.semantic,
                graph: weights.graph,
                structural: weights.structural,
                link: weights.link,
            },
            embedding_model: self.embedding_model.clone(),
            snapshot_entities: self.index.node_count(),
            snapshot_relationships: self.index.edge_count(),
            candidate_k: scored.len(),
            top_k: results.len(),
            graph_core_latency_ms,
            semantic_latency_ms,
            total_latency_ms,
            timings: StageTimings {
                provider_embed_ms,
                semantic_scan_ms,
                candidate_select_ms,
                graph_core_rerank_ms: graph_core_latency_ms,
                evidence_ms,
                total_ms: total_latency_ms,
                query_embedding,
            },
            results,
            semantic_order,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f32::consts::FRAC_1_SQRT_2;

    use async_trait::async_trait;
    use edgequake_llm::error::LlmError;
    use itsm_graph_core::testing::activation_pack::{write_activation_pack, ActivationPackSpec};
    use tempfile::TempDir;

    use crate::sandbox_demo::snapshot::sha256_hex;
    use crate::sandbox_demo::types::{DEFAULT_CANDIDATE_K, DEFAULT_TOP_K};

    /// A deterministic keyword embedder: each dimension is the (scaled)
    /// count of one probe word in the text. It is not a language model,
    /// but it makes "documents that mention denials score higher on a
    /// question about risk" true for a real, checkable reason — which is
    /// what the retrieval half of the demo has to be exercised on.
    struct KeywordEmbedder;

    const PROBES: [&str; 6] = [
        "refused",
        "risk",
        "permitted",
        "routine",
        "process_exec",
        "file_open_read_only",
    ];

    #[async_trait]
    impl EmbeddingProvider for KeywordEmbedder {
        fn name(&self) -> &str {
            "keyword-test"
        }
        fn model(&self) -> &str {
            "keyword-test-embedding"
        }
        fn dimension(&self) -> usize {
            PROBES.len()
        }
        fn max_tokens(&self) -> usize {
            8192
        }
        async fn embed(&self, texts: &[String]) -> std::result::Result<Vec<Vec<f32>>, LlmError> {
            Ok(texts
                .iter()
                .map(|text| {
                    let lower = text.to_ascii_lowercase();
                    PROBES
                        .iter()
                        .map(|probe| lower.matches(probe).count() as f32)
                        .collect()
                })
                .collect())
        }
    }

    /// The demo graph in miniature: one anchor with four runs, four
    /// governed resources — two the anchor was refused, two it was
    /// routinely permitted — plus one resource only another agent touched.
    fn snapshot_json() -> Vec<u8> {
        let mut entities = vec![
            serde_json::json!({"id": "agent:claude-code-demo", "label": "claude-code-demo",
                               "entity_type": "agent_profile"}),
            serde_json::json!({"id": "agent:codex-demo", "label": "codex-demo",
                               "entity_type": "agent_profile"}),
            serde_json::json!({"id": "outcome:completed", "label": "completed",
                               "entity_type": "outcome"}),
        ];
        let mut relationships = Vec::new();

        // resource -> (label, denied?, owning agent)
        let resources = [
            (
                "resource:file:wc",
                "/usr/bin/wc",
                true,
                "agent:claude-code-demo",
            ),
            (
                "resource:file:find",
                "/usr/bin/find",
                true,
                "agent:claude-code-demo",
            ),
            (
                "resource:file:id",
                "/usr/bin/id",
                false,
                "agent:claude-code-demo",
            ),
            (
                "resource:directory:src",
                "directory 07d0d6e9",
                false,
                "agent:claude-code-demo",
            ),
            (
                "resource:file:other",
                "/usr/bin/tar",
                false,
                "agent:codex-demo",
            ),
        ];
        for (i, (id, label, denied, agent)) in resources.iter().enumerate() {
            let run = format!("run:{i:04}");
            let scenario = if *denied {
                "denied-executable"
            } else {
                "repository-read"
            };
            entities.push(serde_json::json!({"id": id, "label": label,
                                             "entity_type": "resource"}));
            entities.push(serde_json::json!({"id": run,
                                             "label": format!("run {i:04} ({scenario})"),
                                             "entity_type": "run"}));
            relationships.push(
                serde_json::json!({"id": format!("i{i}"), "source_id": agent,
                                                  "target_id": run, "keywords": ["initiated"]}),
            );
            relationships.push(serde_json::json!({"id": format!("o{i}"), "source_id": run,
                                                  "target_id": "outcome:completed",
                                                  "keywords": ["ended_as"]}));
            let relation = if *denied {
                "denied_access"
            } else {
                "allowed_access"
            };
            relationships.push(serde_json::json!({"id": format!("a{i}"), "source_id": run,
                                                  "target_id": id,
                                                  "keywords": [relation, "process_exec"]}));
        }

        serde_json::to_vec(&serde_json::json!({
            "workspace_id": "demo-workspace",
            "revision": "sandbox-v1-test",
            "entities": entities,
            "relationships": relationships,
        }))
        .unwrap()
    }

    /// Entity embeddings chosen so structure and semantics disagree:
    /// `/usr/bin/id` sits exactly on the anchor (structural 1.0) while
    /// `/usr/bin/wc` sits opposite it (structural 0.0).
    fn pack_entities(snapshot: &[u8]) -> Vec<(String, Vec<f32>)> {
        let dto: serde_json::Value = serde_json::from_slice(snapshot).unwrap();
        let structural: std::collections::HashMap<&str, [f32; 2]> = [
            ("resource:file:id", [1.0, 0.0]),
            ("resource:directory:src", [FRAC_1_SQRT_2, FRAC_1_SQRT_2]),
            ("resource:file:find", [0.0, 1.0]),
            ("resource:file:wc", [-1.0, 0.0]),
            ("resource:file:other", [0.0, -1.0]),
            ("agent:claude-code-demo", [1.0, 0.0]),
        ]
        .into_iter()
        .collect();

        dto["entities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| {
                let id = e["id"].as_str().unwrap();
                let row = structural.get(id).copied().unwrap_or([0.0, 0.0]);
                (id.to_string(), row.to_vec())
            })
            .collect()
    }

    fn write_pack(snapshot: &[u8]) -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        let entities = pack_entities(snapshot);
        write_activation_pack(
            dir.path(),
            &ActivationPackSpec::new("demo-workspace", "v0007", "sandbox-v1-test", &entities)
                .with_snapshot(snapshot),
        )
        .unwrap();
        dir
    }

    async fn engine(dir: &TempDir) -> Result<SandboxDemoQueryEngine> {
        SandboxDemoQueryEngine::load(dir.path(), Arc::new(KeywordEmbedder)).await
    }

    fn hero_query() -> SandboxDemoRequest {
        SandboxDemoRequest {
            workspace_id: "demo-workspace".into(),
            query: "Which governed resources show the riskiest access pattern? \
                    Refused, risky access."
                .into(),
            anchor_external_id: "agent:claude-code-demo".into(),
            candidate_k: DEFAULT_CANDIDATE_K,
            top_k: DEFAULT_TOP_K,
        }
    }

    #[tokio::test]
    async fn a_valid_pack_loads_and_prewarms_every_resource() {
        let snapshot = snapshot_json();
        let dir = write_pack(&snapshot);
        let engine = engine(&dir).await.unwrap();

        assert_eq!(engine.workspace_id(), "demo-workspace");
        assert_eq!(engine.resource_nodes.len(), 5);
        assert_eq!(engine.dim, PROBES.len());
        assert_eq!(engine.embedding_model(), "keyword-test-embedding");
        assert_eq!(
            engine.anchor_candidates(),
            vec!["agent:claude-code-demo", "agent:codex-demo"]
        );
        assert_eq!(engine.index().sha256(), sha256_hex(&snapshot));
    }

    #[tokio::test]
    async fn a_snapshot_the_model_was_not_trained_on_prevents_startup() {
        let snapshot = snapshot_json();
        let dir = write_pack(&snapshot);

        // Same graph, one more (unmodelled) resource: the bytes no longer
        // hash to what activation.json names.
        let mut edited: serde_json::Value = serde_json::from_slice(&snapshot).unwrap();
        edited["entities"].as_array_mut().unwrap().push(
            serde_json::json!({"id": "resource:file:new", "label": "/usr/bin/new",
                               "entity_type": "resource"}),
        );
        std::fs::write(
            dir.path().join(SNAPSHOT_FILE),
            serde_json::to_vec(&edited).unwrap(),
        )
        .unwrap();

        let err = engine(&dir).await.unwrap_err();
        assert!(
            matches!(
                err,
                SandboxDemoError::SnapshotMismatch {
                    field: "sha256",
                    ..
                }
            ),
            "expected a snapshot identity failure, got {err:?}"
        );
    }

    #[tokio::test]
    async fn a_resource_outside_the_models_id_space_prevents_startup() {
        let snapshot = snapshot_json();
        let dir = tempfile::tempdir().unwrap();
        // Drop one resource from the model while leaving it in the graph.
        let entities: Vec<(String, Vec<f32>)> = pack_entities(&snapshot)
            .into_iter()
            .filter(|(id, _)| id != "resource:file:find")
            .collect();
        write_activation_pack(
            dir.path(),
            &ActivationPackSpec::new("demo-workspace", "v0007", "sandbox-v1-test", &entities)
                .with_snapshot(&snapshot),
        )
        .unwrap();

        let err = SandboxDemoQueryEngine::load(dir.path(), Arc::new(KeywordEmbedder))
            .await
            .unwrap_err();
        assert!(
            matches!(
                &err,
                SandboxDemoError::UnmappedResources { first, .. } if first == "resource:file:find"
            ),
            "expected an id-space failure, got {err:?}"
        );
    }

    #[tokio::test]
    async fn a_missing_pack_prevents_startup() {
        let dir = tempfile::tempdir().unwrap();
        let err = SandboxDemoQueryEngine::load(dir.path(), Arc::new(KeywordEmbedder))
            .await
            .unwrap_err();
        assert!(
            matches!(err, SandboxDemoError::GraphCore(_)),
            "expected graph-core to refuse the empty directory, got {err:?}"
        );
    }

    #[tokio::test]
    async fn structure_reorders_the_semantic_candidates() {
        let snapshot = snapshot_json();
        let dir = write_pack(&snapshot);
        let engine = engine(&dir).await.unwrap();

        let response = engine.query(&hero_query()).await.unwrap();
        assert!(response.graph_core_applied);
        assert_eq!(response.model_version, "v0007");
        assert_eq!(response.snapshot_revision, "sandbox-v1-test");
        assert_eq!(response.weights.structural, 0.40);
        assert_eq!(response.candidate_k, 5, "all five resources are candidates");
        assert_eq!(response.results.len(), DEFAULT_TOP_K);

        // The keyword embedder ranks the denial-carrying documents first;
        // /usr/bin/wc sits structurally opposite the anchor, so graph-core
        // has to push it down. If it did not, the demo has no story.
        let semantic_first = &response.semantic_order[0];
        assert!(
            semantic_first.external_id.starts_with("resource:file:wc")
                || semantic_first.external_id.starts_with("resource:file:find"),
            "a risk question should retrieve the refused resources first, got {semantic_first:?}"
        );
        let final_order: Vec<&str> = response
            .results
            .iter()
            .map(|r| r.external_id.as_str())
            .collect();
        assert_ne!(
            final_order[0], "resource:file:wc",
            "structure must demote the resource opposite the anchor"
        );
        assert_eq!(
            final_order[0], "resource:file:id",
            "the resource sitting on the anchor must win"
        );

        // Every result carries the full breakdown and a real path.
        for result in &response.results {
            assert!((0.0..=1.0).contains(&result.structural_score));
            assert!((0.0..=1.0).contains(&result.semantic_score));
            let expected = 0.6 * result.semantic_score + 0.4 * result.structural_score;
            assert!(
                (result.final_score - expected).abs() < 1e-5,
                "{} blended {} but the frozen weights say {expected}",
                result.external_id,
                result.final_score
            );
            assert_eq!(result.evidence_path.len(), 4);
            assert_eq!(result.evidence_path[0], "agent:claude-code-demo");
            assert_eq!(result.evidence_path[3], result.external_id);
        }

        // The denial-carrying result must be traceable to a denial.
        let wc = response
            .results
            .iter()
            .find(|r| r.external_id == "resource:file:wc")
            .expect("wc is in the top five");
        assert_eq!(wc.evidence.relation, "denied_access");
        assert_eq!(wc.evidence.denied_edges, 1);
        assert!(wc.evidence.via_anchor_run);

        // The resource only another agent touched has no anchor evidence.
        let other = response
            .results
            .iter()
            .find(|r| r.external_id == "resource:file:other")
            .expect("the codex-only resource is in the top five");
        assert!(!other.evidence.via_anchor_run);
        assert_eq!(other.evidence.anchor_runs_touching, 0);
        assert_eq!(other.evidence.anchor_run_share_pct, 0.0);
    }

    #[tokio::test]
    async fn three_consecutive_warm_runs_produce_the_same_result_order() {
        let snapshot = snapshot_json();
        let dir = write_pack(&snapshot);
        let engine = engine(&dir).await.unwrap();

        let mut orders = Vec::new();
        for _ in 0..3 {
            let response = engine.query(&hero_query()).await.unwrap();
            orders.push(
                response
                    .results
                    .iter()
                    .map(|r| (r.external_id.clone(), r.final_score))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(orders[0], orders[1]);
        assert_eq!(orders[1], orders[2]);
    }

    #[tokio::test]
    async fn the_rerank_is_fast_enough_to_show_on_stage() {
        let snapshot = snapshot_json();
        let dir = write_pack(&snapshot);
        let engine = engine(&dir).await.unwrap();

        let response = engine.query(&hero_query()).await.unwrap();
        // The stage target is < 5 ms for 20 candidates; this graph is
        // smaller, so the bar here only catches an order-of-magnitude
        // regression (an accidental reload or a per-candidate model read).
        assert!(
            response.graph_core_latency_ms < 5.0,
            "embedded rerank took {} ms",
            response.graph_core_latency_ms
        );
        assert!(response.total_latency_ms >= response.graph_core_latency_ms);
    }

    #[tokio::test]
    async fn stage_timings_are_disjoint_and_bounded_by_the_total() {
        let snapshot = snapshot_json();
        let dir = write_pack(&snapshot);
        let engine = engine(&dir).await.unwrap();

        let response = engine.query(&hero_query()).await.unwrap();
        let t = response.timings;
        assert_eq!(t.query_embedding, QueryEmbeddingSource::Miss);
        assert_eq!(t.total_ms, response.total_latency_ms);
        assert_eq!(t.graph_core_rerank_ms, response.graph_core_latency_ms);
        let stages = [
            t.provider_embed_ms,
            t.semantic_scan_ms,
            t.candidate_select_ms,
            t.graph_core_rerank_ms,
            t.evidence_ms,
        ];
        assert!(stages.iter().all(|s| *s >= 0.0));
        assert!(stages.iter().sum::<f64>() <= t.total_ms);
        assert!(
            t.provider_embed_ms + t.semantic_scan_ms + t.candidate_select_ms
                <= response.semantic_latency_ms
        );
    }

    #[test]
    fn a_response_without_timings_still_parses() {
        let value = serde_json::json!({
            "graph_core_applied": true, "workspace_id": "w", "query": "q",
            "anchor": {"external_id": "a", "label": "a", "entity_type": "agent_profile",
                       "run_count": 0},
            "model_version": "v1", "mapping_hash": "h", "snapshot_revision": "r",
            "snapshot_sha256": "s",
            "weights": {"semantic": 0.6, "graph": 0.0, "structural": 0.4, "link": 0.0},
            "embedding_model": "m", "snapshot_entities": 0, "snapshot_relationships": 0,
            "candidate_k": 0, "top_k": 0, "graph_core_latency_ms": 0.0,
            "semantic_latency_ms": 0.0, "total_latency_ms": 0.0,
            "results": [], "semantic_order": [],
        });
        let parsed: SandboxDemoResponse = serde_json::from_value(value).unwrap();
        assert_eq!(parsed.timings, StageTimings::default());

        let encoded = serde_json::to_value(StageTimings {
            query_embedding: QueryEmbeddingSource::Coalesced,
            ..StageTimings::default()
        })
        .unwrap();
        assert_eq!(encoded["query_embedding"], "coalesced");
    }

    #[tokio::test]
    async fn a_bad_request_is_refused_before_any_work() {
        let snapshot = snapshot_json();
        let dir = write_pack(&snapshot);
        let engine = engine(&dir).await.unwrap();

        let cases = [
            (
                SandboxDemoRequest {
                    workspace_id: "some-other-workspace".into(),
                    ..hero_query()
                },
                "workspace",
            ),
            (
                SandboxDemoRequest {
                    query: "   ".into(),
                    ..hero_query()
                },
                "empty",
            ),
            (
                SandboxDemoRequest {
                    anchor_external_id: "agent:nobody".into(),
                    ..hero_query()
                },
                "not in snapshot",
            ),
            (
                SandboxDemoRequest {
                    candidate_k: MAX_CANDIDATE_K + 1,
                    ..hero_query()
                },
                "candidate_k",
            ),
            (
                SandboxDemoRequest {
                    top_k: 0,
                    ..hero_query()
                },
                "top_k",
            ),
        ];
        for (request, expected) in cases {
            let err = engine.query(&request).await.unwrap_err();
            assert!(
                matches!(&err, SandboxDemoError::BadRequest(m) if m.contains(expected)),
                "expected a BadRequest mentioning {expected:?}, got {err:?}"
            );
        }
    }

    #[tokio::test]
    async fn an_anchor_that_is_not_an_agent_still_resolves_structurally() {
        // The anchor is explicit, so the demo does not police what kind of
        // entity it is — it only requires that it be in both id spaces.
        let snapshot = snapshot_json();
        let dir = write_pack(&snapshot);
        let engine = engine(&dir).await.unwrap();

        let response = engine
            .query(&SandboxDemoRequest {
                anchor_external_id: "outcome:completed".into(),
                ..hero_query()
            })
            .await
            .unwrap();
        assert_eq!(response.anchor.entity_type, "outcome");
        assert_eq!(response.anchor.run_count, 0);
        assert!(response.results.iter().all(|r| !r.evidence.via_anchor_run));
    }
}
