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

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use edgequake_llm::traits::EmbeddingProvider;
use itsm_graph_core::edgequake::rerank::{
    rerank_with_scores, BlendWeights, EdgeQuakeRerankCandidate, EdgeQuakeRerankedCandidate,
    EdgeQuakeStructuralRerankRequest, ResolvedStructuralScores,
};
use itsm_graph_core::serving::embedded_runtime::{
    EmbeddedGraphCoreRuntime, EmbeddedRerankResult, ExternalCandidate, SNAPSHOT_FILE,
};
use tracing::info;

use super::documents::resource_document;
use super::embed_prefix::EmbedPrefix;
use super::error::{Result, SandboxDemoError};
use super::intent::{spread_matched, IntentConfig, IntentRouting};
use super::query_embedding::{QueryEmbedConfig, QueryEmbedder};
use super::snapshot::{SandboxSnapshotIndex, AGENT_TYPE, INITIATED};
use super::structural::{min_max, ScorerKind, ScoringConfig, StructuralScorer};
use super::types::{
    AnchorInfo, AppliedWeights, EvidenceSummary, SandboxDemoRequest, SandboxDemoResponse,
    SandboxDemoResult, SemanticOrderEntry, StageTimings, MAX_CANDIDATE_K,
};

/// Environment variable naming the activation bundle directory. When it is
/// set, EdgeQuake either serves the demo or refuses to start.
pub const PACK_DIR_ENV: &str = "EDGEQUAKE_SANDBOX_PACK_DIR";

/// A loaded sandbox demo generation.
pub struct SandboxDemoQueryEngine {
    index: Arc<SandboxSnapshotIndex>,
    graph_core: Arc<EmbeddedGraphCoreRuntime>,

    /// Node index of each embedded resource, parallel to [`Self::vectors`].
    resource_nodes: Vec<u32>,
    /// Row-normalized document vectors, `[resource_nodes.len() * dim]`.
    vectors: Vec<f32>,
    dim: usize,
    embedding_model: String,
    pack_dir: PathBuf,
    query_embedder: QueryEmbedder,
    scoring: ScoringConfig,
    intent: IntentConfig,
    prefix: EmbedPrefix,
    /// Distinct runs touching each resource, parallel to
    /// [`Self::resource_nodes`].
    run_sets: Vec<BTreeSet<u32>>,
    /// Run count per resource, min-max over every resource.
    popularity: Vec<f32>,
}

impl std::fmt::Debug for SandboxDemoQueryEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SandboxDemoQueryEngine")
            .field("pack_dir", &self.pack_dir)
            .field("model_version", &self.graph_core.model_version())
            .field("snapshot_revision", &self.index.revision())
            .field("resources", &self.resource_nodes.len())
            .field("embedding_model", &self.embedding_model)
            .field("structural_scorer", &self.scoring.scorer.label())
            .field("embed_prefix", &self.prefix.label())
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
pub(crate) fn normalize(vector: &mut [f32]) {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for value in vector.iter_mut() {
            *value /= norm;
        }
    }
}

/// The first `k` of `scored` in exactly the order a stable full sort by
/// `cmp` would give, without ordering the rest. Ties `cmp` leaves are
/// broken on scan position, which is what stability preserved.
fn top_k_by<F>(mut scored: Vec<(usize, f32)>, k: usize, mut cmp: F) -> Vec<(usize, f32)>
where
    F: FnMut(&(usize, f32), &(usize, f32)) -> std::cmp::Ordering,
{
    let mut cmp = |a: &(usize, f32), b: &(usize, f32)| cmp(a, b).then(a.0.cmp(&b.0));
    if k == 0 {
        scored.clear();
    } else if k < scored.len() {
        scored.select_nth_unstable_by(k - 1, &mut cmp);
        scored.truncate(k);
    }
    scored.sort_unstable_by(cmp);
    scored
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
        Self::load_with(pack_dir, embedding, QueryEmbedConfig::from_env()).await
    }

    /// [`Self::load`] with explicit query-embedding tuning instead of the
    /// environment's.
    pub async fn load_with(
        pack_dir: &Path,
        embedding: Arc<dyn EmbeddingProvider>,
        query_embed: QueryEmbedConfig,
    ) -> Result<Self> {
        Self::load_with_prefix(
            pack_dir,
            embedding,
            query_embed,
            ScoringConfig::from_env()?,
            IntentConfig::from_env()?,
            EmbedPrefix::from_env()?,
        )
        .await
    }

    /// [`Self::load_with`] with explicit structural scoring instead of the
    /// environment's, and intent routing off.
    pub async fn load_with_scoring(
        pack_dir: &Path,
        embedding: Arc<dyn EmbeddingProvider>,
        query_embed: QueryEmbedConfig,
        scoring: ScoringConfig,
    ) -> Result<Self> {
        Self::load_with_intent(
            pack_dir,
            embedding,
            query_embed,
            scoring,
            IntentConfig::default(),
        )
        .await
    }

    /// [`Self::load_with_scoring`] with explicit intent routing and no
    /// embedding prefix.
    pub async fn load_with_intent(
        pack_dir: &Path,
        embedding: Arc<dyn EmbeddingProvider>,
        query_embed: QueryEmbedConfig,
        scoring: ScoringConfig,
        intent: IntentConfig,
    ) -> Result<Self> {
        Self::load_with_prefix(
            pack_dir,
            embedding,
            query_embed,
            scoring,
            intent,
            EmbedPrefix::None,
        )
        .await
    }

    /// [`Self::load_with_intent`] with an explicit embedding prefix.
    pub async fn load_with_prefix(
        pack_dir: &Path,
        embedding: Arc<dyn EmbeddingProvider>,
        query_embed: QueryEmbedConfig,
        scoring: ScoringConfig,
        intent: IntentConfig,
        prefix: EmbedPrefix,
    ) -> Result<Self> {
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
        let run_sets: Vec<BTreeSet<u32>> = resource_nodes
            .iter()
            .map(|node| index.runs_touching(*node))
            .collect();
        let popularity = min_max(&run_sets.iter().map(|r| r.len() as f32).collect::<Vec<_>>());
        let documents: Vec<String> = resource_nodes
            .iter()
            .map(|node| prefix.document(resource_document(&index.resource_profile(*node))))
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
            query_embed_cache_entries = query_embed.cache_entries,
            "sandbox demo pack loaded; graph-core is mandatory on this path"
        );
        let applied = Self::applied_weights(&scoring, graph_core.weights());
        info!(
            structural_scorer = %scoring.scorer.label(),
            weights_source = scoring.weights.source(),
            semantic_weight = applied.semantic,
            structural_weight = applied.structural,
            runtime_rerank = scoring.is_runtime_rerank(),
            "sandbox demo blend weights active"
        );
        info!(
            intent_routing = ?intent.routing,
            intent_scorer = %intent.scorer.label(),
            intent_lambda = intent.lambda,
            "sandbox demo intent routing"
        );
        info!(
            embed_prefix = prefix.label(),
            query_prefix = ?prefix.query_prefix(),
            document_prefix = ?prefix.document_prefix(),
            "sandbox demo embedding prefix"
        );

        let query_embedder = QueryEmbedder::new(
            embedding.clone(),
            dim,
            &activation.workspace_id,
            query_embed,
        );
        Ok(Self {
            embedding_model: embedding.model().to_string(),
            index: Arc::new(index),
            graph_core,
            resource_nodes,
            vectors,
            dim,
            pack_dir: pack_dir.to_path_buf(),
            query_embedder,
            scoring,
            intent,
            prefix,
            run_sets,
            popularity,
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
        // The cache keys on the prefixed text; the lexicon below sees the
        // raw question.
        let (question, query_embedding) = self
            .query_embedder
            .embed(&self.prefix.query(&request.query))
            .await?;
        let provider_embed_ms = ms_since(semantic_started);

        let scan_started = Instant::now();
        let scored: Vec<(usize, f32)> = (0..self.resource_nodes.len())
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
        let scored = top_k_by(scored, request.candidate_k, |a, b| {
            b.1.total_cmp(&a.1).then_with(|| {
                self.index
                    .node(self.resource_nodes[a.0])
                    .external_id
                    .cmp(&self.index.node(self.resource_nodes[b.0]).external_id)
            })
        });
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
        // A routed intent takes the intent blend; otherwise `bundle` with
        // the cosine scorer is the runtime's answer, untouched.
        let intent = self.intent.route(&request.query);
        let reranked = if intent.is_some() {
            self.intent_rescore(anchor, &scored, reranked)?
        } else if self.scoring.is_runtime_rerank() {
            reranked
        } else {
            self.rescore(anchor, &scored, reranked)?
        };
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

        let weights = reranked.weights;
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
            structural_scorer: match intent {
                Some(_) => Some(self.intent.scorer.label()),
                None => self.scoring.explicit.then(|| self.scoring.scorer.label()),
            },
            intent: (self.intent.routing != IntentRouting::Off)
                .then(|| intent.map(|i| i.name().to_string())),
            results,
            semantic_order,
        })
    }

    fn external_id(&self, position: usize) -> &str {
        &self.index.node(self.resource_nodes[position]).external_id
    }

    /// The blend weights every query applies.
    pub fn blend_weights(&self) -> AppliedWeights {
        let w = Self::applied_weights(&self.scoring, self.graph_core.weights());
        AppliedWeights {
            semantic: w.semantic,
            graph: w.graph,
            structural: w.structural,
            link: w.link,
        }
    }

    /// The bundle's frozen weights, or the chosen pair with the components
    /// this path never supplies zeroed.
    fn applied_weights(scoring: &ScoringConfig, bundle: &BlendWeights) -> BlendWeights {
        match scoring.weights.weights() {
            Some(w) => BlendWeights {
                semantic: w.semantic,
                graph: 0.0,
                structural: w.structural,
                link: 0.0,
            },
            None => *bundle,
        }
    }

    /// Anchor runs touching each resource, min-max over every resource.
    fn agent_frequency(&self, anchor_runs: &BTreeSet<u32>) -> Vec<f32> {
        let counts: Vec<f32> = self
            .run_sets
            .iter()
            .map(|runs| runs.intersection(anchor_runs).count() as f32)
            .collect();
        min_max(&counts)
    }

    /// Mean runtime structural score of each candidate against every
    /// resource the anchor's runs touched, scored by re-anchoring the
    /// runtime's public rerank on each touched resource.
    fn knn(&self, anchor_runs: &BTreeSet<u32>, ids: &[&str]) -> Result<Vec<f32>> {
        let touched: Vec<usize> = (0..self.resource_nodes.len())
            .filter(|p| !self.run_sets[*p].is_disjoint(anchor_runs))
            .collect();
        let mut sums = vec![0.0f32; ids.len()];
        for position in &touched {
            let candidates = ids
                .iter()
                .map(|id| ExternalCandidate::new(*id, 0.5))
                .collect();
            let out = self
                .graph_core
                .rerank(self.external_id(*position), candidates, ids.len())?;
            let by_id: HashMap<&str, Option<f32>> = out
                .candidates
                .iter()
                .map(|c| (c.external_id.as_str(), c.structural_score))
                .collect();
            for (sum, id) in sums.iter_mut().zip(ids) {
                *sum += by_id.get(id).copied().flatten().ok_or_else(|| {
                    SandboxDemoError::Snapshot(format!(
                        "graph-core returned no structural score for {id:?}"
                    ))
                })?;
            }
        }
        if !touched.is_empty() {
            let n = touched.len() as f32;
            sums.iter_mut().for_each(|s| *s /= n);
        }
        Ok(sums)
    }

    /// The runtime's candidates in semantic order.
    fn ordered<'a>(
        &self,
        scored: &[(usize, f32)],
        reranked: &'a EmbeddedRerankResult,
    ) -> Result<Vec<&'a EdgeQuakeRerankedCandidate>> {
        let by_id: HashMap<&str, _> = reranked
            .candidates
            .iter()
            .map(|c| (c.external_id.as_str(), c))
            .collect();
        scored
            .iter()
            .map(|(position, _)| {
                let id = self.external_id(*position);
                by_id.get(id).copied().ok_or_else(|| {
                    SandboxDemoError::Snapshot(format!("graph-core dropped candidate {id:?}"))
                })
            })
            .collect()
    }

    /// `scorer`'s structural value for each of `ordered`, parallel to
    /// `scored`.
    fn structural_values(
        &self,
        scorer: &StructuralScorer,
        anchor: u32,
        scored: &[(usize, f32)],
        ordered: &[&EdgeQuakeRerankedCandidate],
    ) -> Result<Vec<f32>> {
        let ids: Vec<&str> = ordered.iter().map(|c| c.external_id.as_str()).collect();
        let anchor_runs = self.index.runs_of_agent(anchor);
        scorer.combine(ordered.len(), |kind| match kind {
            ScorerKind::Cosine => ordered
                .iter()
                .map(|c| {
                    c.structural_score.ok_or_else(|| {
                        SandboxDemoError::Snapshot(format!(
                            "graph-core returned no structural score for {:?}",
                            c.external_id
                        ))
                    })
                })
                .collect(),
            ScorerKind::Popularity => Ok(scored.iter().map(|(p, _)| self.popularity[*p]).collect()),
            ScorerKind::AgentFrequency => {
                let all = self.agent_frequency(&anchor_runs);
                Ok(scored.iter().map(|(p, _)| all[*p]).collect())
            }
            ScorerKind::Knn => self.knn(&anchor_runs, &ids),
        })
    }

    /// The intent blend: `semantic + λ·spread_matched(structural)` with the
    /// intent's scorer. `structural_score` reports the scorer's raw value;
    /// `weights` reports `semantic=1, structural=λ`. Ties break on external
    /// id, as in graph-core's rerank.
    fn intent_rescore(
        &self,
        anchor: u32,
        scored: &[(usize, f32)],
        reranked: EmbeddedRerankResult,
    ) -> Result<EmbeddedRerankResult> {
        let ordered = self.ordered(scored, &reranked)?;
        let structural = self.structural_values(&self.intent.scorer, anchor, scored, &ordered)?;
        let semantic: Vec<f32> = ordered.iter().map(|c| c.semantic_score).collect();
        let adjusted = spread_matched(&structural, &semantic);
        let lambda = self.intent.lambda;
        let mut candidates: Vec<EdgeQuakeRerankedCandidate> = ordered
            .iter()
            .zip(structural.iter().zip(&adjusted))
            .map(|(c, (raw, adj))| EdgeQuakeRerankedCandidate {
                structural_score: Some(*raw),
                link_score: None,
                blended_score: c.semantic_score + lambda * adj,
                ..(*c).clone()
            })
            .collect();
        candidates.sort_by(|a, b| {
            b.blended_score
                .total_cmp(&a.blended_score)
                .then_with(|| a.external_id.cmp(&b.external_id))
        });
        Ok(EmbeddedRerankResult {
            candidates,
            weights: BlendWeights {
                semantic: 1.0,
                graph: 0.0,
                structural: lambda,
                link: 0.0,
            },
            ..reranked
        })
    }

    /// Replace the runtime's structural score with the configured scorer
    /// and re-blend through graph-core's own `rerank_with_scores`.
    fn rescore(
        &self,
        anchor: u32,
        scored: &[(usize, f32)],
        reranked: EmbeddedRerankResult,
    ) -> Result<EmbeddedRerankResult> {
        let ordered = self.ordered(scored, &reranked)?;
        let structural = self.structural_values(&self.scoring.scorer, anchor, scored, &ordered)?;

        let weights = Self::applied_weights(&self.scoring, &reranked.weights);
        let request = EdgeQuakeStructuralRerankRequest {
            workspace_id: self.graph_core.workspace_id().clone(),
            query_text: String::new(),
            mode: "sandbox-demo".into(),
            candidates: ordered
                .iter()
                .map(|c| EdgeQuakeRerankCandidate {
                    external_id: c.external_id.clone(),
                    entity_id: c.entity_id,
                    semantic_score: c.semantic_score,
                    graph_score: None,
                    text: None,
                    metadata: serde_json::Value::Null,
                })
                .collect(),
            top_k: ordered.len(),
        };
        let resolved: Vec<ResolvedStructuralScores> = structural
            .into_iter()
            .map(|s| ResolvedStructuralScores {
                structural_score: Some(s),
                link_score: None,
            })
            .collect();
        let response = rerank_with_scores(
            &request,
            reranked.model_version.clone(),
            &weights,
            &resolved,
        )?;
        Ok(EmbeddedRerankResult {
            candidates: response.candidates,
            weights,
            ..reranked
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use std::f32::consts::FRAC_1_SQRT_2;

    use async_trait::async_trait;
    use edgequake_llm::error::LlmError;
    use itsm_graph_core::testing::activation_pack::{write_activation_pack, ActivationPackSpec};
    use tempfile::TempDir;

    use crate::sandbox_demo::snapshot::sha256_hex;
    use crate::sandbox_demo::structural::ScoringConfig;
    use crate::sandbox_demo::types::{QueryEmbeddingSource, DEFAULT_CANDIDATE_K, DEFAULT_TOP_K};

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

    /// [`KeywordEmbedder`] that counts provider calls, records every text
    /// it was sent, can stall so
    /// concurrent callers overlap, and can refuse (or panic on) texts
    /// containing `boom`.
    #[derive(Default)]
    pub(crate) struct CountingEmbedder {
        pub(crate) calls: std::sync::atomic::AtomicUsize,
        pub(crate) batch_sizes: std::sync::Mutex<Vec<usize>>,
        pub(crate) texts: std::sync::Mutex<Vec<String>>,
        pub(crate) delay_ms: u64,
        pub(crate) fail_boom: std::sync::atomic::AtomicBool,
        pub(crate) panic_boom: std::sync::atomic::AtomicBool,
    }

    impl CountingEmbedder {
        pub(crate) fn stalling(delay_ms: u64) -> Arc<Self> {
            Arc::new(Self {
                delay_ms,
                ..Self::default()
            })
        }

        pub(crate) fn calls(&self) -> usize {
            self.calls.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl EmbeddingProvider for CountingEmbedder {
        fn name(&self) -> &str {
            KeywordEmbedder.name()
        }
        fn model(&self) -> &str {
            KeywordEmbedder.model()
        }
        fn dimension(&self) -> usize {
            KeywordEmbedder.dimension()
        }
        fn max_tokens(&self) -> usize {
            KeywordEmbedder.max_tokens()
        }
        async fn embed(&self, texts: &[String]) -> std::result::Result<Vec<Vec<f32>>, LlmError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.batch_sizes.lock().unwrap().push(texts.len());
            self.texts.lock().unwrap().extend(texts.iter().cloned());
            if self.delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
            let boom = texts.iter().any(|t| t.contains("boom"));
            if boom && self.panic_boom.load(std::sync::atomic::Ordering::SeqCst) {
                panic!("provider panicked");
            }
            if boom && self.fail_boom.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(LlmError::Unknown("provider is down".into()));
            }
            KeywordEmbedder.embed(texts).await
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
        // The bundle's 0.6 / 0.4 blend, restored by `bundle`; the default is
        // semantic-only (see `unset_weights_rank_in_pure_semantic_order`).
        let snapshot = snapshot_json();
        let dir = write_pack(&snapshot);
        let engine = scored_engine(&dir, None, Some("bundle")).await;

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

    async fn counting_engine(
        dir: &TempDir,
        embedder: &Arc<CountingEmbedder>,
        config: QueryEmbedConfig,
    ) -> SandboxDemoQueryEngine {
        let provider: Arc<dyn EmbeddingProvider> = embedder.clone();
        SandboxDemoQueryEngine::load_with(dir.path(), provider, config)
            .await
            .unwrap()
    }

    fn question(text: &str) -> SandboxDemoRequest {
        SandboxDemoRequest {
            query: text.into(),
            ..hero_query()
        }
    }

    /// The response as bytes, minus every wall-clock field.
    fn stable_bytes(response: &SandboxDemoResponse) -> String {
        let mut value = serde_json::to_value(response).unwrap();
        let object = value.as_object_mut().unwrap();
        for field in [
            "timings",
            "graph_core_latency_ms",
            "semantic_latency_ms",
            "total_latency_ms",
        ] {
            object.remove(field);
        }
        serde_json::to_string(&value).unwrap()
    }

    #[tokio::test]
    async fn a_repeated_question_is_answered_without_the_provider() {
        let dir = write_pack(&snapshot_json());
        let embedder = CountingEmbedder::stalling(0);
        let engine = counting_engine(&dir, &embedder, QueryEmbedConfig::default()).await;
        let after_prewarm = embedder.calls();
        assert_eq!(after_prewarm, 1, "the documents are prewarmed in one batch");

        let cold = engine.query(&hero_query()).await.unwrap();
        assert_eq!(cold.timings.query_embedding, QueryEmbeddingSource::Miss);
        assert_eq!(embedder.calls(), after_prewarm + 1);

        let warm = engine.query(&hero_query()).await.unwrap();
        assert_eq!(warm.timings.query_embedding, QueryEmbeddingSource::Hit);
        assert_eq!(
            embedder.calls(),
            after_prewarm + 1,
            "a hit must not call out"
        );
        assert_eq!(stable_bytes(&cold), stable_bytes(&warm));

        // Whitespace is part of the text sent to the provider, so it is
        // part of the key.
        let spaced = engine
            .query(&question(&format!("{} ", hero_query().query)))
            .await;
        assert_eq!(
            spaced.unwrap().timings.query_embedding,
            QueryEmbeddingSource::Miss
        );
    }

    #[tokio::test]
    async fn a_zero_cache_is_the_original_one_call_per_query_path() {
        let dir = write_pack(&snapshot_json());
        let embedder = CountingEmbedder::stalling(0);
        let engine = counting_engine(&dir, &embedder, QueryEmbedConfig::disabled()).await;
        for _ in 0..3 {
            let response = engine.query(&hero_query()).await.unwrap();
            assert_eq!(response.timings.query_embedding, QueryEmbeddingSource::Miss);
        }
        assert_eq!(embedder.calls(), 1 + 3);
    }

    #[tokio::test]
    async fn the_cache_does_not_change_a_single_answer_byte() {
        let dir = write_pack(&snapshot_json());
        let on = counting_engine(
            &dir,
            &CountingEmbedder::stalling(0),
            QueryEmbedConfig::default(),
        )
        .await;
        let off = counting_engine(
            &dir,
            &CountingEmbedder::stalling(0),
            QueryEmbedConfig::disabled(),
        )
        .await;
        let questions = [
            hero_query(),
            question("routine permitted file_open_read_only"),
            SandboxDemoRequest {
                anchor_external_id: "agent:codex-demo".into(),
                candidate_k: 3,
                top_k: 2,
                ..hero_query()
            },
        ];
        for _ in 0..2 {
            for request in &questions {
                let a = on.query(request).await.unwrap();
                let b = off.query(request).await.unwrap();
                assert_eq!(stable_bytes(&a), stable_bytes(&b));
            }
        }
    }

    #[tokio::test]
    async fn identical_cold_questions_share_one_provider_call() {
        let dir = write_pack(&snapshot_json());
        let embedder = CountingEmbedder::stalling(25);
        let engine = counting_engine(&dir, &embedder, QueryEmbedConfig::default()).await;
        let before = embedder.calls();

        let request = hero_query();
        let responses = futures::future::join_all((0..8).map(|_| engine.query(&request))).await;
        assert_eq!(embedder.calls(), before + 1, "eight waiters, one request");

        let responses: Vec<SandboxDemoResponse> =
            responses.into_iter().map(|r| r.unwrap()).collect();
        let sources: Vec<QueryEmbeddingSource> = responses
            .iter()
            .map(|r| r.timings.query_embedding)
            .collect();
        assert_eq!(
            sources
                .iter()
                .filter(|s| **s == QueryEmbeddingSource::Miss)
                .count(),
            1
        );
        assert_eq!(
            sources
                .iter()
                .filter(|s| **s == QueryEmbeddingSource::Coalesced)
                .count(),
            7
        );
        let first = stable_bytes(&responses[0]);
        assert!(responses.iter().all(|r| stable_bytes(r) == first));
    }

    #[tokio::test]
    async fn a_provider_error_reaches_every_waiter_and_is_not_cached() {
        let dir = write_pack(&snapshot_json());
        let embedder = CountingEmbedder::stalling(25);
        let engine = counting_engine(&dir, &embedder, QueryEmbedConfig::default()).await;
        let before = embedder.calls();
        embedder
            .fail_boom
            .store(true, std::sync::atomic::Ordering::SeqCst);

        let request = question("boom: refused risk");
        let failures = futures::future::join_all((0..4).map(|_| engine.query(&request))).await;
        assert_eq!(embedder.calls(), before + 1);
        for failure in failures {
            let err = failure.unwrap_err();
            assert!(
                matches!(&err, SandboxDemoError::Embedding(m) if m.contains("provider is down")),
                "every waiter must see the provider error, got {err:?}"
            );
        }

        embedder
            .fail_boom
            .store(false, std::sync::atomic::Ordering::SeqCst);
        let recovered = engine.query(&request).await.unwrap();
        assert_eq!(
            recovered.timings.query_embedding,
            QueryEmbeddingSource::Miss
        );
        assert_eq!(embedder.calls(), before + 2, "the error was not memoized");
    }

    #[test]
    fn partial_top_k_matches_a_full_stable_sort_including_ties() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..500 {
            let n = (next() % 80) as usize;
            // Few distinct scores and few distinct ids, so both tie levels
            // are exercised; -0.0/0.0 and duplicates included.
            let ids: Vec<String> = (0..n).map(|_| format!("r{}", next() % 6)).collect();
            let scored: Vec<(usize, f32)> = (0..n)
                .map(|p| {
                    let score = match next() % 5 {
                        0 => 0.0,
                        1 => -0.0,
                        r => (r as f32 - 3.0) * 0.25,
                    };
                    (p, score)
                })
                .collect();
            let cmp = |a: &(usize, f32), b: &(usize, f32)| {
                b.1.total_cmp(&a.1).then_with(|| ids[a.0].cmp(&ids[b.0]))
            };
            for k in [0, 1, 2, n / 2, n.saturating_sub(1), n, n + 3] {
                let mut full = scored.clone();
                full.sort_by(cmp);
                full.truncate(k);
                let partial = top_k_by(scored.clone(), k, cmp);
                let bits = |v: &[(usize, f32)]| -> Vec<(usize, u32)> {
                    v.iter().map(|(p, s)| (*p, s.to_bits())).collect()
                };
                assert_eq!(bits(&partial), bits(&full), "round {round}, n {n}, k {k}");
            }
        }
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

    // ---- opt-in structural scorers -------------------------------------

    /// [`snapshot_json`] plus two runs, so run counts differ per resource:
    ///
    /// | resource | runs        | claude runs |
    /// |----------|-------------|-------------|
    /// | wc       | 0000, 0005  | 1           |
    /// | find     | 0001        | 1           |
    /// | id       | 0002, 0006  | 2           |
    /// | src      | 0003        | 1           |
    /// | other    | 0004, 0005  | 0           |
    ///
    /// `run:0005` is codex's, `run:0006` claude's.
    fn scorer_snapshot_json() -> Vec<u8> {
        let mut dto: serde_json::Value = serde_json::from_slice(&snapshot_json()).unwrap();
        let entities = dto["entities"].as_array_mut().unwrap();
        for (run, scenario) in [
            ("run:0005", "repository-read"),
            ("run:0006", "repository-read"),
        ] {
            entities.push(
                serde_json::json!({"id": run, "label": format!("{run} ({scenario})"),
                                             "entity_type": "run"}),
            );
        }
        let relationships = dto["relationships"].as_array_mut().unwrap();
        for (id, source, target, keywords) in [
            ("i5", "agent:codex-demo", "run:0005", vec!["initiated"]),
            (
                "i6",
                "agent:claude-code-demo",
                "run:0006",
                vec!["initiated"],
            ),
            (
                "a5w",
                "run:0005",
                "resource:file:wc",
                vec!["allowed_access", "process_exec"],
            ),
            (
                "a5o",
                "run:0005",
                "resource:file:other",
                vec!["allowed_access", "process_exec"],
            ),
            (
                "a6i",
                "run:0006",
                "resource:file:id",
                vec!["allowed_access", "process_exec"],
            ),
        ] {
            relationships.push(serde_json::json!({"id": id, "source_id": source,
                                                  "target_id": target, "keywords": keywords}));
        }
        serde_json::to_vec(&dto).unwrap()
    }

    async fn scored_engine(
        dir: &TempDir,
        scorer: Option<&str>,
        weights: Option<&str>,
    ) -> SandboxDemoQueryEngine {
        SandboxDemoQueryEngine::load_with_scoring(
            dir.path(),
            Arc::new(KeywordEmbedder),
            QueryEmbedConfig::default(),
            ScoringConfig::parse(scorer, weights).unwrap(),
        )
        .await
        .unwrap()
    }

    fn structural_by_id(response: &SandboxDemoResponse) -> std::collections::BTreeMap<String, f32> {
        response
            .results
            .iter()
            .map(|r| (r.external_id.clone(), r.structural_score))
            .collect()
    }

    fn assert_scores(response: &SandboxDemoResponse, expected: &[(&str, f32)]) {
        let got = structural_by_id(response);
        assert_eq!(got.len(), expected.len(), "{got:?}");
        for (id, want) in expected {
            let have = got[*id];
            assert!(
                (have - want).abs() < 1e-5,
                "{id}: got {have}, want {want}; {got:?}"
            );
        }
    }

    fn assert_blended(response: &SandboxDemoResponse, semantic: f32, structural: f32) {
        for r in &response.results {
            let expected = semantic * r.semantic_score + structural * r.structural_score;
            assert!(
                (r.final_score - expected).abs() < 1e-5,
                "{} blended {} but expected {expected}",
                r.external_id,
                r.final_score
            );
        }
        let finals: Vec<f32> = response.results.iter().map(|r| r.final_score).collect();
        assert!(finals.windows(2).all(|w| w[0] >= w[1]), "{finals:?}");
    }

    #[tokio::test]
    async fn unset_weights_rank_in_pure_semantic_order() {
        let dir = write_pack(&scorer_snapshot_json());
        let engine = scored_engine(&dir, None, None).await;
        let response = engine.query(&hero_query()).await.unwrap();

        // graph-core still ran; only its blend weight is zero.
        assert!(response.graph_core_applied);
        assert_eq!(response.weights.semantic, 1.0);
        assert_eq!(response.weights.structural, 0.0);
        assert_eq!(response.weights.graph, 0.0);
        assert_eq!(response.weights.link, 0.0);
        let blend = engine.blend_weights();
        assert_eq!((blend.semantic, blend.structural), (1.0, 0.0));
        let wire = serde_json::to_value(&response).unwrap();
        assert!(wire.get("structural_scorer").is_none(), "{wire}");

        let semantic_ids: Vec<&str> = response
            .semantic_order
            .iter()
            .map(|e| e.external_id.as_str())
            .collect();
        let final_ids: Vec<&str> = response
            .results
            .iter()
            .map(|r| r.external_id.as_str())
            .collect();
        assert_eq!(final_ids, semantic_ids[..final_ids.len()]);
        assert_blended(&response, 1.0, 0.0);

        // The runtime's structural scores are still returned, untouched.
        let candidates = response
            .semantic_order
            .iter()
            .map(|e| ExternalCandidate::new(e.external_id.clone(), (e.semantic_cosine + 1.0) / 2.0))
            .collect();
        let runtime = engine
            .graph_core()
            .rerank("agent:claude-code-demo", candidates, 5)
            .unwrap();
        let expected: std::collections::BTreeMap<String, f32> = runtime
            .candidates
            .iter()
            .map(|c| (c.external_id.clone(), c.structural_score.unwrap()))
            .collect();
        assert_eq!(structural_by_id(&response), expected);

        // On the hero fixture the bundle blend reorders; the default does not.
        let dir = write_pack(&snapshot_json());
        let hero = scored_engine(&dir, None, None)
            .await
            .query(&hero_query())
            .await
            .unwrap();
        let semantic_ids: Vec<&str> = hero
            .semantic_order
            .iter()
            .map(|e| e.external_id.as_str())
            .collect();
        let final_ids: Vec<&str> = hero
            .results
            .iter()
            .map(|r| r.external_id.as_str())
            .collect();
        assert_eq!(final_ids, semantic_ids[..final_ids.len()]);
        let bundle = scored_engine(&dir, None, Some("bundle"))
            .await
            .query(&hero_query())
            .await
            .unwrap();
        let bundle_ids: Vec<&str> = bundle
            .results
            .iter()
            .map(|r| r.external_id.as_str())
            .collect();
        assert_ne!(
            bundle_ids, final_ids,
            "the fixture must tell the blends apart"
        );
    }

    const NEXT_TOUCH_QUERY: &str =
        "What will claude-code-demo touch next? Refused, risky access, routine permitted.";

    async fn intent_engine(
        dir: &TempDir,
        routing: Option<&str>,
        scorer: Option<&str>,
        lambda: Option<&str>,
    ) -> SandboxDemoQueryEngine {
        SandboxDemoQueryEngine::load_with_intent(
            dir.path(),
            Arc::new(KeywordEmbedder),
            QueryEmbedConfig::default(),
            ScoringConfig::default(),
            crate::sandbox_demo::intent::IntentConfig::parse(routing, scorer, lambda).unwrap(),
        )
        .await
        .unwrap()
    }

    /// Golden lines captured from the pre-router default (56060ca):
    /// `(snapshot, query)` for both fixtures × {hero, next-touch}.
    fn default_goldens() -> Vec<(Vec<u8>, &'static str, &'static str)> {
        let lines: Vec<&str> = include_str!("testdata/default_responses.jsonl")
            .lines()
            .collect();
        assert_eq!(lines.len(), 4);
        let hero = "Which governed resources show the riskiest access pattern? \
                    Refused, risky access.";
        vec![
            (snapshot_json(), hero, lines[0]),
            (snapshot_json(), NEXT_TOUCH_QUERY, lines[1]),
            (scorer_snapshot_json(), hero, lines[2]),
            (scorer_snapshot_json(), NEXT_TOUCH_QUERY, lines[3]),
        ]
    }

    #[tokio::test]
    async fn unset_intent_routing_is_byte_identical_to_the_pre_router_default() {
        for (snapshot, query, golden) in default_goldens() {
            let dir = write_pack(&snapshot);
            for engine in [
                scored_engine(&dir, None, None).await,
                intent_engine(&dir, None, None, None).await,
                intent_engine(&dir, Some("off"), Some("popularity"), Some("5")).await,
            ] {
                let response = engine.query(&question(query)).await.unwrap();
                assert_eq!(stable_bytes(&response), golden, "{query}");
                let wire = serde_json::to_value(&response).unwrap();
                assert!(wire.get("intent").is_none(), "{wire}");
            }
        }
    }

    #[tokio::test]
    async fn next_touch_routing_leaves_other_queries_on_the_default_path() {
        for (snapshot, query, golden) in default_goldens() {
            if query == NEXT_TOUCH_QUERY {
                continue;
            }
            let dir = write_pack(&snapshot);
            let engine =
                intent_engine(&dir, Some("next_touch"), Some("popularity"), Some("5")).await;
            let response = engine.query(&question(query)).await.unwrap();
            assert_eq!(response.intent, Some(None));
            let mut wire: serde_json::Value =
                serde_json::from_str(&stable_bytes(&response)).unwrap();
            let object = wire.as_object_mut().unwrap();
            assert_eq!(object.remove("intent"), Some(serde_json::Value::Null));
            assert_eq!(serde_json::to_string(&wire).unwrap(), golden);
        }
    }

    /// `final = semantic + λ·spread_matched(structural)` over the whole
    /// candidate set, with `structural` the anchor's run frequency.
    fn assert_intent_blend(response: &SandboxDemoResponse, raw: &[(&str, f32)], lambda: f32) {
        let semantic: std::collections::BTreeMap<&str, f32> = response
            .semantic_order
            .iter()
            .map(|e| (e.external_id.as_str(), (e.semantic_cosine + 1.0) / 2.0))
            .collect();
        let ids: Vec<&str> = raw.iter().map(|(id, _)| *id).collect();
        let adjusted = crate::sandbox_demo::intent::spread_matched(
            &raw.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
            &ids.iter().map(|id| semantic[id]).collect::<Vec<_>>(),
        );
        let mut expected: Vec<(&str, f32)> = ids
            .iter()
            .zip(&adjusted)
            .map(|(id, adj)| (*id, semantic[id] + lambda * adj))
            .collect();
        expected.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        let got: Vec<(&str, f32)> = response
            .results
            .iter()
            .map(|r| (r.external_id.as_str(), r.final_score))
            .collect();
        assert_eq!(got.len(), expected.len());
        for ((gid, gs), (eid, es)) in got.iter().zip(&expected) {
            assert_eq!(gid, eid, "{got:?} vs {expected:?}");
            assert!((gs - es).abs() < 1e-5, "{got:?} vs {expected:?}");
        }
        assert_scores(response, raw);
    }

    #[tokio::test]
    async fn a_next_touch_query_blends_spread_matched_agent_frequency() {
        let dir = write_pack(&scorer_snapshot_json());
        // The anchor's runs: wc 1, find 1, id 2, src 1, other 0 —
        // min-maxed over every resource.
        let raw = [
            ("resource:file:wc", 0.5),
            ("resource:file:find", 0.5),
            ("resource:file:id", 1.0),
            ("resource:directory:src", 0.5),
            ("resource:file:other", 0.0),
        ];
        let engine = intent_engine(&dir, Some("next_touch"), None, None).await;
        let response = engine.query(&question(NEXT_TOUCH_QUERY)).await.unwrap();
        assert!(response.graph_core_applied);
        assert_eq!(response.intent, Some(Some("next_touch".to_string())));
        assert_eq!(
            response.structural_scorer.as_deref(),
            Some("agent_frequency")
        );
        assert_eq!(
            (response.weights.semantic, response.weights.structural),
            (1.0, 1.0)
        );
        let wire = serde_json::to_value(&response).unwrap();
        assert_eq!(wire["intent"], "next_touch");
        assert_intent_blend(&response, &raw, 1.0);
        // Structure moves the order: the anchor's most-run resource breaks
        // the three-way semantic tie and leads; the resource only another
        // agent touched is pushed below its semantic score.
        let id = &response.results[0];
        assert_eq!(id.external_id, "resource:file:id");
        assert_eq!((id.semantic_rank, id.final_rank), (2, 1));
        let other = response
            .results
            .iter()
            .find(|r| r.external_id == "resource:file:other")
            .unwrap();
        assert!(other.final_score < other.semantic_score);

        let heavier = intent_engine(&dir, Some("next_touch"), None, Some("2.5")).await;
        let response = heavier.query(&question(NEXT_TOUCH_QUERY)).await.unwrap();
        assert_eq!(response.weights.structural, 2.5);
        assert_intent_blend(&response, &raw, 2.5);

        // λ = 0 is the semantic order, still labelled with the intent.
        let zero = intent_engine(&dir, Some("next_touch"), None, Some("0")).await;
        let response = zero.query(&question(NEXT_TOUCH_QUERY)).await.unwrap();
        assert_eq!(response.intent, Some(Some("next_touch".to_string())));
        let semantic_ids: Vec<&str> = response
            .semantic_order
            .iter()
            .map(|e| e.external_id.as_str())
            .collect();
        let final_ids: Vec<&str> = response
            .results
            .iter()
            .map(|r| r.external_id.as_str())
            .collect();
        assert_eq!(final_ids, semantic_ids);
    }

    #[tokio::test]
    async fn the_intent_scorer_is_selectable() {
        let dir = write_pack(&scorer_snapshot_json());
        // Distinct runs per resource: wc 2, find 1, id 2, src 1, other 2.
        let raw = [
            ("resource:file:wc", 1.0),
            ("resource:file:find", 0.0),
            ("resource:file:id", 1.0),
            ("resource:directory:src", 0.0),
            ("resource:file:other", 1.0),
        ];
        let engine = intent_engine(&dir, Some("next_touch"), Some("popularity"), None).await;
        let response = engine.query(&question(NEXT_TOUCH_QUERY)).await.unwrap();
        assert_eq!(response.structural_scorer.as_deref(), Some("popularity"));
        assert_intent_blend(&response, &raw, 1.0);

        let blend = intent_engine(
            &dir,
            Some("next_touch"),
            Some("blend:popularity=0.5,agent_frequency=0.5"),
            None,
        )
        .await;
        let response = blend.query(&question(NEXT_TOUCH_QUERY)).await.unwrap();
        assert_eq!(
            response.structural_scorer.as_deref(),
            Some("blend:popularity=0.5,agent_frequency=0.5")
        );
        // Each source min-maxed over the candidates, then averaged.
        let mixed: Vec<(&str, f32)> = [
            ("resource:file:wc", 0.75),
            ("resource:file:find", 0.25),
            ("resource:file:id", 1.0),
            ("resource:directory:src", 0.25),
            ("resource:file:other", 0.5),
        ]
        .to_vec();
        assert_intent_blend(&response, &mixed, 1.0);
    }

    #[tokio::test]
    async fn bundle_reproduces_the_pre_default_response_byte_for_byte() {
        // Captured from the pre-change default (bundle weights, runtime
        // rerank untouched) on the hero fixture, before the default moved.
        let golden = include_str!("testdata/bundle_hero_response.json");
        let dir = write_pack(&snapshot_json());
        let response = scored_engine(&dir, None, Some("bundle"))
            .await
            .query(&hero_query())
            .await
            .unwrap();
        assert_eq!(stable_bytes(&response), golden.trim_end());
    }

    #[tokio::test]
    async fn bundle_is_the_runtimes_rerank_byte_for_byte() {
        let dir = write_pack(&scorer_snapshot_json());
        let default = scored_engine(&dir, None, Some("bundle")).await;
        let response = default.query(&hero_query()).await.unwrap();
        assert_eq!(response.weights.semantic, 0.6);
        assert_eq!(response.weights.structural, 0.4);
        let blend = default.blend_weights();
        assert_eq!((blend.semantic, blend.structural), (0.6, 0.4));

        // The scorer field is absent, so the wire shape is the old one.
        let wire = serde_json::to_value(&response).unwrap();
        assert!(wire.get("structural_scorer").is_none(), "{wire}");

        // Every score is the runtime's own, for the same candidates.
        let candidates = response
            .semantic_order
            .iter()
            .map(|e| ExternalCandidate::new(e.external_id.clone(), (e.semantic_cosine + 1.0) / 2.0))
            .collect();
        let runtime = default
            .graph_core()
            .rerank("agent:claude-code-demo", candidates, 5)
            .unwrap();
        let expected: Vec<(String, Option<f32>, f32)> = runtime
            .candidates
            .iter()
            .map(|c| (c.external_id.clone(), c.structural_score, c.blended_score))
            .collect();
        let got: Vec<(String, Option<f32>, f32)> = response
            .results
            .iter()
            .map(|r| {
                (
                    r.external_id.clone(),
                    Some(r.structural_score),
                    r.final_score,
                )
            })
            .collect();
        assert_eq!(got, expected);

        // Explicit cosine, and re-blending through the opt-in path with the
        // bundle's own weights, both reproduce `bundle` exactly; only the
        // echoed scorer name is new.
        let strip = |r: &SandboxDemoResponse| {
            let mut v: serde_json::Value = serde_json::from_str(&stable_bytes(r)).unwrap();
            v.as_object_mut().unwrap().remove("structural_scorer");
            v.to_string()
        };
        for (scorer, weights) in [
            (Some("cosine"), Some("bundle")),
            (None, Some("semantic=0.6,structural=0.4")),
        ] {
            let engine = scored_engine(&dir, scorer, weights).await;
            let other = engine.query(&hero_query()).await.unwrap();
            assert_eq!(other.structural_scorer.as_deref(), Some("cosine"));
            assert_eq!(strip(&other), strip(&response), "{scorer:?} {weights:?}");
        }
    }

    #[tokio::test]
    async fn popularity_is_run_count_min_maxed_over_every_resource() {
        let dir = write_pack(&scorer_snapshot_json());
        let response = scored_engine(&dir, Some("popularity"), Some("bundle"))
            .await
            .query(&hero_query())
            .await
            .unwrap();
        assert_eq!(response.structural_scorer.as_deref(), Some("popularity"));
        assert_scores(
            &response,
            &[
                ("resource:file:wc", 1.0),
                ("resource:file:find", 0.0),
                ("resource:file:id", 1.0),
                ("resource:directory:src", 0.0),
                ("resource:file:other", 1.0),
            ],
        );
        assert_blended(&response, 0.6, 0.4);
    }

    #[tokio::test]
    async fn agent_frequency_counts_only_the_anchors_runs() {
        let dir = write_pack(&scorer_snapshot_json());
        let response = scored_engine(&dir, Some("agent_frequency"), Some("bundle"))
            .await
            .query(&hero_query())
            .await
            .unwrap();
        assert_scores(
            &response,
            &[
                ("resource:file:wc", 0.5),
                ("resource:file:find", 0.5),
                ("resource:file:id", 1.0),
                ("resource:directory:src", 0.5),
                ("resource:file:other", 0.0),
            ],
        );
        assert_blended(&response, 0.6, 0.4);
    }

    #[tokio::test]
    async fn knn_is_the_mean_structural_score_against_the_anchors_resources() {
        let dir = write_pack(&scorer_snapshot_json());
        let response = scored_engine(&dir, Some("knn"), Some("bundle"))
            .await
            .query(&hero_query())
            .await
            .unwrap();

        // claude's runs touched wc, find, id and src — not other.
        let unit = |x: f32, y: f32| [x, y];
        let touched = [
            unit(-1.0, 0.0),
            unit(0.0, 1.0),
            unit(1.0, 0.0),
            unit(FRAC_1_SQRT_2, FRAC_1_SQRT_2),
        ];
        let knn = |c: [f32; 2]| {
            touched
                .iter()
                .map(|t| (t[0] * c[0] + t[1] * c[1] + 1.0) / 2.0)
                .sum::<f32>()
                / touched.len() as f32
        };
        assert_scores(
            &response,
            &[
                ("resource:file:wc", knn(unit(-1.0, 0.0))),
                ("resource:file:find", knn(unit(0.0, 1.0))),
                ("resource:file:id", knn(unit(1.0, 0.0))),
                (
                    "resource:directory:src",
                    knn(unit(FRAC_1_SQRT_2, FRAC_1_SQRT_2)),
                ),
                ("resource:file:other", knn(unit(0.0, -1.0))),
            ],
        );
        // Hand check: id = (0 + 0.5 + 1 + 0.8536) / 4.
        let id = structural_by_id(&response)["resource:file:id"];
        assert!((id - 0.588_39).abs() < 1e-4, "{id}");
        assert_blended(&response, 0.6, 0.4);

        // An anchor whose runs touched nothing has no neighbours: all zero.
        let lonely = scored_engine(&dir, Some("knn"), Some("bundle"))
            .await
            .query(&SandboxDemoRequest {
                anchor_external_id: "outcome:completed".into(),
                ..hero_query()
            })
            .await
            .unwrap();
        assert!(lonely.results.iter().all(|r| r.structural_score == 0.0));
    }

    #[tokio::test]
    async fn a_blend_mixes_min_maxed_sources_over_the_candidates() {
        let dir = write_pack(&scorer_snapshot_json());
        let response = scored_engine(
            &dir,
            Some("blend:popularity=0.5,agent_frequency=0.5"),
            Some("bundle"),
        )
        .await
        .query(&hero_query())
        .await
        .unwrap();
        assert_eq!(
            response.structural_scorer.as_deref(),
            Some("blend:popularity=0.5,agent_frequency=0.5")
        );
        // All five resources are candidates, so the candidate min-max is
        // the all-resource one.
        assert_scores(
            &response,
            &[
                ("resource:file:wc", 0.75),
                ("resource:file:find", 0.25),
                ("resource:file:id", 1.0),
                ("resource:directory:src", 0.25),
                ("resource:file:other", 0.5),
            ],
        );
        assert_blended(&response, 0.6, 0.4);
    }

    #[tokio::test]
    async fn a_weight_override_replaces_the_frozen_weights() {
        let dir = write_pack(&scorer_snapshot_json());
        let response = scored_engine(
            &dir,
            Some("popularity"),
            Some("semantic=0.2,structural=0.8"),
        )
        .await
        .query(&hero_query())
        .await
        .unwrap();
        assert_eq!(response.weights.semantic, 0.2);
        assert_eq!(response.weights.structural, 0.8);
        assert_eq!(response.weights.graph, 0.0);
        assert_eq!(response.weights.link, 0.0);
        assert_blended(&response, 0.2, 0.8);

        // Weights alone keep the runtime cosine but re-weight it.
        let cosine = scored_engine(&dir, None, Some("semantic=1,structural=0"))
            .await
            .query(&hero_query())
            .await
            .unwrap();
        let semantic_ids: Vec<&str> = cosine
            .semantic_order
            .iter()
            .map(|e| e.external_id.as_str())
            .collect();
        let final_ids: Vec<&str> = cosine
            .results
            .iter()
            .map(|r| r.external_id.as_str())
            .collect();
        assert_eq!(
            final_ids, semantic_ids,
            "structural weight 0 keeps the semantic order"
        );
    }

    async fn prefix_engine(
        dir: &TempDir,
        embedder: &Arc<CountingEmbedder>,
        routing: Option<&str>,
        prefix: Option<&str>,
    ) -> SandboxDemoQueryEngine {
        let provider: Arc<dyn EmbeddingProvider> = embedder.clone();
        SandboxDemoQueryEngine::load_with_prefix(
            dir.path(),
            provider,
            QueryEmbedConfig::default(),
            ScoringConfig::default(),
            crate::sandbox_demo::intent::IntentConfig::parse(routing, None, None).unwrap(),
            EmbedPrefix::parse(prefix).unwrap(),
        )
        .await
        .unwrap()
    }

    fn sent(embedder: &CountingEmbedder) -> Vec<String> {
        embedder.texts.lock().unwrap().clone()
    }

    #[tokio::test]
    async fn an_unset_or_none_prefix_is_byte_identical_to_the_default() {
        for (snapshot, query, golden) in default_goldens() {
            let dir = write_pack(&snapshot);
            let reference = CountingEmbedder::stalling(0);
            let baseline = counting_engine(&dir, &reference, QueryEmbedConfig::default()).await;
            baseline.query(&question(query)).await.unwrap();
            for prefix in [None, Some(""), Some("none"), Some("custom:|")] {
                let embedder = CountingEmbedder::stalling(0);
                let engine = prefix_engine(&dir, &embedder, None, prefix).await;
                let response = engine.query(&question(query)).await.unwrap();
                assert_eq!(stable_bytes(&response), golden, "{prefix:?} {query}");
                assert_eq!(sent(&embedder), sent(&reference), "{prefix:?}");
            }
        }
    }

    #[tokio::test]
    async fn nomic_prefixes_every_document_and_the_question() {
        let dir = write_pack(&snapshot_json());
        let raw = CountingEmbedder::stalling(0);
        counting_engine(&dir, &raw, QueryEmbedConfig::default()).await;
        let raw_documents = sent(&raw);

        let embedder = CountingEmbedder::stalling(0);
        let engine = prefix_engine(&dir, &embedder, None, Some("nomic")).await;
        let documents = sent(&embedder);
        assert_eq!(documents.len(), raw_documents.len());
        for (prefixed, raw) in documents.iter().zip(&raw_documents) {
            assert_eq!(prefixed, &format!("search_document: {raw}"));
        }

        engine.query(&hero_query()).await.unwrap();
        let texts = sent(&embedder);
        assert_eq!(texts.len(), documents.len() + 1);
        assert_eq!(
            texts.last().unwrap(),
            &format!("search_query: {}", hero_query().query)
        );
    }

    #[tokio::test]
    async fn custom_prefixes_are_applied_verbatim_to_both_sides() {
        let dir = write_pack(&snapshot_json());
        let embedder = CountingEmbedder::stalling(0);
        let engine = prefix_engine(&dir, &embedder, None, Some("custom:Q> |D> ")).await;
        let documents = sent(&embedder);
        assert!(
            documents.iter().all(|d| d.starts_with("D> ")),
            "{documents:?}"
        );
        engine.query(&hero_query()).await.unwrap();
        assert_eq!(
            sent(&embedder).last().unwrap(),
            &format!("Q> {}", hero_query().query)
        );
    }

    #[tokio::test]
    async fn a_prefixed_question_is_cached_under_its_prefixed_text() {
        let dir = write_pack(&snapshot_json());
        let embedder = CountingEmbedder::stalling(0);
        let engine = prefix_engine(&dir, &embedder, None, Some("nomic")).await;
        let after_prewarm = embedder.calls();
        let cold = engine.query(&hero_query()).await.unwrap();
        let warm = engine.query(&hero_query()).await.unwrap();
        assert_eq!(cold.timings.query_embedding, QueryEmbeddingSource::Miss);
        assert_eq!(warm.timings.query_embedding, QueryEmbeddingSource::Hit);
        assert_eq!(embedder.calls(), after_prewarm + 1);
        assert_eq!(engine.query_embedder.cached_entries().0, 1);

        // The raw question is a different key from the prefixed one.
        let (_, raw) = engine
            .query_embedder
            .embed(&hero_query().query)
            .await
            .unwrap();
        assert_eq!(raw, QueryEmbeddingSource::Miss);
        assert_eq!(engine.query_embedder.cached_entries().0, 2);
    }

    #[tokio::test]
    async fn the_intent_lexicon_sees_the_raw_question() {
        let dir = write_pack(&snapshot_json());
        // A query prefix that would itself match the next-touch lexicon.
        let prefix = Some("custom:What will it touch next? |");
        assert!(crate::sandbox_demo::intent::is_next_touch(
            &EmbedPrefix::parse(prefix)
                .unwrap()
                .query(&hero_query().query)
        ));
        let embedder = CountingEmbedder::stalling(0);
        let engine = prefix_engine(&dir, &embedder, Some("next_touch"), prefix).await;

        let hero = engine.query(&hero_query()).await.unwrap();
        assert_eq!(hero.intent, Some(None), "the prefix must not route");
        let routed = engine.query(&question(NEXT_TOUCH_QUERY)).await.unwrap();
        assert_eq!(routed.intent, Some(Some("next_touch".to_string())));
        assert_eq!(
            sent(&embedder).last().unwrap(),
            &format!("What will it touch next? {NEXT_TOUCH_QUERY}")
        );
    }
}
