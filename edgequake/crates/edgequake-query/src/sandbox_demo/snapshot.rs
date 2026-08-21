//! The in-memory sandbox snapshot index.
//!
//! One frozen `snapshot.json` — the same EdgeQuake graph-snapshot contract
//! graph-core trained on — parsed once at startup into an entity table, an
//! `external_id → node` map, and an undirected adjacency list.
//!
//! Two invariants make the rest of the demo honest:
//!
//! * **External ids are preserved byte for byte.** A node's `external_id`
//!   is graph-core's id, never an EdgeQuake display name. It is the only
//!   thing that resolves into the model's id space.
//! * **The bytes are the ones the model was trained on.** [`SandboxSnapshotIndex::from_bytes`]
//!   checks the file's SHA-256 against `activation.json` before parsing, so
//!   an index built from a newer snapshot than the model cannot start.
//!
//! The snapshot is parsed through graph-core's own
//! [`EdgeQuakeGraphSnapshotDto`] rather than a local struct: the training
//! path and the demo index then cannot disagree about what the file means.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use itsm_graph_core::edgequake::dto::EdgeQuakeGraphSnapshotDto;
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::error::{Result, SandboxDemoError};

/// Entity type of a governed resource — the only entities the demo
/// retrieves over.
pub const RESOURCE_TYPE: &str = "resource";
/// Entity type of a governed run.
pub const RUN_TYPE: &str = "run";
/// Entity type of an agent profile (the demo's anchor).
pub const AGENT_TYPE: &str = "agent_profile";

/// Relation carried by an edge a policy permitted.
pub const ALLOWED_ACCESS: &str = "allowed_access";
/// Relation carried by an edge a policy refused.
pub const DENIED_ACCESS: &str = "denied_access";
/// Relation binding an agent profile to the runs it started.
pub const INITIATED: &str = "initiated";
/// Relation binding a run to its terminal outcome.
pub const ENDED_AS: &str = "ended_as";

/// One entity in the frozen snapshot.
#[derive(Debug, Clone)]
pub struct SnapshotNode {
    /// graph-core external id, preserved exactly.
    pub external_id: String,
    /// Display label from the exporter.
    pub label: String,
    /// Entity type (`resource`, `run`, `agent_profile`, …).
    pub entity_type: String,
}

/// One adjacency entry: the node on the other end plus the relation.
#[derive(Debug, Clone)]
pub struct SnapshotEdge {
    /// Index of the node at the other end.
    pub other: u32,
    /// Relation keyword (`keywords[0]`), e.g. `denied_access`.
    pub relation: Arc<str>,
    /// Action detail (`keywords[1]`), e.g. `process_exec`.
    pub action: Option<Arc<str>>,
    /// True when this node is the edge's `source_id`.
    pub outgoing: bool,
}

/// One entity, projected for drawing.
#[derive(Debug, Clone, Serialize)]
pub struct GraphNode {
    /// graph-core external id — the same id every result carries.
    pub id: String,
    /// Display label from the exporter.
    pub label: String,
    /// Entity type (`run`, `resource`, `identity`, …).
    pub entity_type: String,
}

/// The whole frozen snapshot, projected once for a client that draws it.
///
/// `edges` is positional rather than a struct per edge on purpose: at
/// several thousand relationships, four small integers per edge is the
/// difference between a response a page parses instantly and one it does
/// not. Each entry is `[source, target, relation, action]`, indexing
/// `nodes`, `relations` and `actions` respectively; `action` is `-1` when
/// the relationship carried no action keyword.
#[derive(Debug, Clone, Serialize)]
pub struct GraphProjection {
    /// Every entity, in snapshot order — so an index here is the index the
    /// rest of this struct means.
    pub nodes: Vec<GraphNode>,
    /// Every relationship, exactly once.
    pub edges: Vec<[i64; 4]>,
    /// Interned relation keywords.
    pub relations: Vec<String>,
    /// Interned action keywords.
    pub actions: Vec<String>,
}

/// Structural facts about one governed resource, derived from the graph
/// alone. This is what the retrieval document and the evidence summary are
/// both built from — no metadata, description, or event payload is read.
#[derive(Debug, Clone, Default)]
pub struct ResourceProfile {
    /// `file`, `directory`, … taken from the external id's second segment.
    pub kind: String,
    /// Display label.
    pub label: String,
    /// Number of permitted access edges (one per audit event).
    pub allowed_edges: usize,
    /// Number of refused access edges.
    pub denied_edges: usize,
    /// Distinct runs that were permitted this resource.
    pub allowed_runs: usize,
    /// Distinct runs that were refused this resource.
    pub denied_runs: usize,
    /// Distinct runs that touched it at all.
    pub total_runs: usize,
    /// Distinct actions attempted, sorted.
    pub actions: Vec<String>,
    /// Distinct actions a policy refused, sorted.
    pub denied_actions: Vec<String>,
    /// Labels of the agent profiles behind those runs, sorted.
    pub agent_profiles: Vec<String>,
    /// Scenario name → run count, from the run labels, sorted by name.
    pub scenarios: Vec<(String, usize)>,
    /// Terminal outcome → run count, sorted by name.
    pub outcomes: Vec<(String, usize)>,
}

/// A resolved one-hop-through-a-run path from the anchor to a resource.
#[derive(Debug, Clone)]
pub struct EvidencePath {
    /// `[anchor, run, relation, resource]` — the shape the demo renders.
    pub path: Vec<String>,
    /// The relation on the final edge.
    pub relation: String,
    /// The action detail on the final edge, when the edge carried one.
    pub action: Option<String>,
    /// True when the run in the path was initiated by the anchor itself.
    pub via_anchor_run: bool,
    /// Distinct runs touching the resource.
    pub runs_touching: usize,
    /// How many of those the anchor initiated.
    pub anchor_runs_touching: usize,
    /// Permitted edges into the resource.
    pub allowed_edges: usize,
    /// Refused edges into the resource.
    pub denied_edges: usize,
}

/// The frozen snapshot, indexed for retrieval and evidence.
#[derive(Debug)]
pub struct SandboxSnapshotIndex {
    workspace_id: String,
    revision: String,
    sha256: String,
    nodes: Vec<SnapshotNode>,
    by_external_id: HashMap<String, u32>,
    adjacency: Vec<Vec<SnapshotEdge>>,
    resources: Vec<u32>,
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

/// `resource:file:94dd31fe…` → `file`. Falls back to the entity type.
fn kind_of(external_id: &str, entity_type: &str) -> String {
    external_id
        .split(':')
        .nth(1)
        .filter(|s| !s.is_empty())
        .unwrap_or(entity_type)
        .to_string()
}

/// `run 96f25883 (approved-executable)` → `approved-executable`.
fn scenario_of(label: &str) -> Option<&str> {
    let open = label.rfind('(')?;
    let close = label.rfind(')')?;
    (close > open + 1).then(|| label[open + 1..close].trim())
}

impl SandboxSnapshotIndex {
    /// Parse and index a frozen `snapshot.json`.
    ///
    /// Fail-closed against the activation bundle: the bytes must hash to
    /// `expected_sha256` and the parsed workspace and revision must match
    /// what the model was activated for. A demo that indexed a different
    /// graph than the model learned would return confidently wrong
    /// neighbours, which is exactly the failure this gate exists to make
    /// impossible.
    pub fn from_bytes(
        bytes: &[u8],
        expected_sha256: &str,
        expected_workspace: &str,
        expected_revision: &str,
    ) -> Result<Self> {
        let sha256 = sha256_hex(bytes);
        if !sha256.eq_ignore_ascii_case(expected_sha256) {
            return Err(SandboxDemoError::SnapshotMismatch {
                field: "sha256",
                expected: expected_sha256.to_string(),
                actual: sha256,
            });
        }

        let dto: EdgeQuakeGraphSnapshotDto = serde_json::from_slice(bytes)
            .map_err(|e| SandboxDemoError::Snapshot(format!("parse snapshot.json: {e}")))?;

        if dto.workspace_id != expected_workspace {
            return Err(SandboxDemoError::SnapshotMismatch {
                field: "workspace_id",
                expected: expected_workspace.to_string(),
                actual: dto.workspace_id,
            });
        }
        if dto.revision != expected_revision {
            return Err(SandboxDemoError::SnapshotMismatch {
                field: "revision",
                expected: expected_revision.to_string(),
                actual: dto.revision,
            });
        }

        let mut nodes = Vec::with_capacity(dto.entities.len());
        let mut by_external_id = HashMap::with_capacity(dto.entities.len());
        let mut resources = Vec::new();
        for entity in &dto.entities {
            let index = nodes.len() as u32;
            if by_external_id.insert(entity.id.clone(), index).is_some() {
                return Err(SandboxDemoError::Snapshot(format!(
                    "snapshot.json carries entity id {:?} more than once",
                    entity.id
                )));
            }
            let entity_type = entity.entity_type.clone().unwrap_or_default();
            if entity_type == RESOURCE_TYPE {
                resources.push(index);
            }
            nodes.push(SnapshotNode {
                external_id: entity.id.clone(),
                label: entity.label.clone(),
                entity_type,
            });
        }
        if resources.is_empty() {
            return Err(SandboxDemoError::Snapshot(
                "snapshot.json carries no governed resources to retrieve over".into(),
            ));
        }

        // Intern relation/action keywords: the vocabulary is tiny and every
        // edge is stored twice (once per endpoint).
        let mut interned: HashMap<String, Arc<str>> = HashMap::new();
        let mut intern = |value: &str| -> Arc<str> {
            if let Some(existing) = interned.get(value) {
                return Arc::clone(existing);
            }
            let arc: Arc<str> = Arc::from(value);
            interned.insert(value.to_string(), Arc::clone(&arc));
            arc
        };

        let mut adjacency: Vec<Vec<SnapshotEdge>> = vec![Vec::new(); nodes.len()];
        for edge in &dto.relationships {
            let (Some(&source), Some(&target)) = (
                by_external_id.get(&edge.source_id),
                by_external_id.get(&edge.target_id),
            ) else {
                return Err(SandboxDemoError::Snapshot(format!(
                    "relationship {:?} has a dangling endpoint ({:?} -> {:?})",
                    edge.id, edge.source_id, edge.target_id
                )));
            };
            let Some(relation) = edge.keywords.first() else {
                return Err(SandboxDemoError::Snapshot(format!(
                    "relationship {:?} carries no relation keyword",
                    edge.id
                )));
            };
            let relation = intern(relation);
            let action = edge.keywords.get(1).map(|a| intern(a));
            adjacency[source as usize].push(SnapshotEdge {
                other: target,
                relation: Arc::clone(&relation),
                action: action.clone(),
                outgoing: true,
            });
            adjacency[target as usize].push(SnapshotEdge {
                other: source,
                relation,
                action,
                outgoing: false,
            });
        }

        Ok(Self {
            workspace_id: dto.workspace_id,
            revision: dto.revision,
            sha256,
            nodes,
            by_external_id,
            adjacency,
            resources,
        })
    }

    /// Workspace the snapshot belongs to.
    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    /// Deterministic snapshot revision.
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// SHA-256 of the indexed bytes.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Total entity count.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total relationship count (each edge is stored at both endpoints).
    pub fn edge_count(&self) -> usize {
        self.adjacency.iter().map(Vec::len).sum::<usize>() / 2
    }

    /// Node indices of every governed resource, in snapshot order.
    pub fn resources(&self) -> &[u32] {
        &self.resources
    }

    /// Borrow a node by index.
    pub fn node(&self, index: u32) -> &SnapshotNode {
        &self.nodes[index as usize]
    }

    /// Resolve an external id to its node index.
    pub fn index_of(&self, external_id: &str) -> Option<u32> {
        self.by_external_id.get(external_id).copied()
    }

    /// Edges incident to a node.
    pub fn edges(&self, index: u32) -> &[SnapshotEdge] {
        &self.adjacency[index as usize]
    }

    /// Project the whole graph for rendering: every entity, and every
    /// relationship exactly once.
    ///
    /// The adjacency list stores each relationship twice, once per
    /// endpoint; only the `outgoing` copy is emitted, so the projection
    /// carries [`Self::edge_count`] edges, ordered by source node. Nothing
    /// is filtered, scored, or renamed on the way out — this is the same
    /// graph the evidence paths are walked over.
    pub fn projection(&self) -> GraphProjection {
        let mut relations: Vec<String> = Vec::new();
        let mut actions: Vec<String> = Vec::new();
        let mut relation_ix: HashMap<&str, i64> = HashMap::new();
        let mut action_ix: HashMap<&str, i64> = HashMap::new();
        let mut edges = Vec::with_capacity(self.edge_count());

        for (source, adjacent) in self.adjacency.iter().enumerate() {
            for edge in adjacent.iter().filter(|edge| edge.outgoing) {
                let relation = *relation_ix.entry(&edge.relation).or_insert_with(|| {
                    relations.push(edge.relation.to_string());
                    (relations.len() - 1) as i64
                });
                let action = edge.action.as_ref().map_or(-1, |action| {
                    *action_ix.entry(action).or_insert_with(|| {
                        actions.push(action.to_string());
                        (actions.len() - 1) as i64
                    })
                });
                edges.push([source as i64, i64::from(edge.other), relation, action]);
            }
        }

        GraphProjection {
            nodes: self
                .nodes
                .iter()
                .map(|node| GraphNode {
                    id: node.external_id.clone(),
                    label: node.label.clone(),
                    entity_type: node.entity_type.clone(),
                })
                .collect(),
            edges,
            relations,
            actions,
        }
    }

    /// The runs an agent profile initiated.
    fn runs_of_agent(&self, agent: u32) -> BTreeSet<u32> {
        self.edges(agent)
            .iter()
            .filter(|e| e.outgoing && &*e.relation == INITIATED)
            .map(|e| e.other)
            .collect()
    }

    /// The terminal outcome label of a run, when it has one.
    fn outcome_of_run(&self, run: u32) -> Option<&str> {
        self.edges(run)
            .iter()
            .find(|e| e.outgoing && &*e.relation == ENDED_AS)
            .map(|e| self.node(e.other).label.as_str())
    }

    /// The agent profile label behind a run, when it has one.
    fn agent_of_run(&self, run: u32) -> Option<&str> {
        self.edges(run)
            .iter()
            .find(|e| !e.outgoing && &*e.relation == INITIATED)
            .map(|e| self.node(e.other).label.as_str())
    }

    /// Derive the structural profile of one governed resource.
    ///
    /// Deterministic: every collection is a `BTreeMap`/`BTreeSet`, so the
    /// same graph always yields byte-identical output.
    pub fn resource_profile(&self, resource: u32) -> ResourceProfile {
        let node = self.node(resource);
        let mut profile = ResourceProfile {
            kind: kind_of(&node.external_id, &node.entity_type),
            label: node.label.clone(),
            ..ResourceProfile::default()
        };

        let mut actions = BTreeSet::new();
        let mut denied_actions = BTreeSet::new();
        let mut allowed_runs = BTreeSet::new();
        let mut denied_runs = BTreeSet::new();

        for edge in self.edges(resource) {
            if edge.outgoing {
                continue;
            }
            let denied = match &*edge.relation {
                ALLOWED_ACCESS => false,
                DENIED_ACCESS => true,
                _ => continue,
            };
            if let Some(action) = &edge.action {
                actions.insert(action.to_string());
                if denied {
                    denied_actions.insert(action.to_string());
                }
            }
            if denied {
                profile.denied_edges += 1;
                denied_runs.insert(edge.other);
            } else {
                profile.allowed_edges += 1;
                allowed_runs.insert(edge.other);
            }
        }

        let mut agents: BTreeSet<&str> = BTreeSet::new();
        let mut scenarios: BTreeMap<String, usize> = BTreeMap::new();
        let mut outcomes: BTreeMap<String, usize> = BTreeMap::new();
        for run in allowed_runs.union(&denied_runs) {
            if let Some(agent) = self.agent_of_run(*run) {
                agents.insert(agent);
            }
            if let Some(scenario) = scenario_of(&self.node(*run).label) {
                *scenarios.entry(scenario.to_string()).or_default() += 1;
            }
            if let Some(outcome) = self.outcome_of_run(*run) {
                *outcomes.entry(outcome.to_string()).or_default() += 1;
            }
        }

        profile.total_runs = allowed_runs.union(&denied_runs).count();
        profile.allowed_runs = allowed_runs.len();
        profile.denied_runs = denied_runs.len();
        profile.actions = actions.into_iter().collect();
        profile.denied_actions = denied_actions.into_iter().collect();
        profile.agent_profiles = agents.into_iter().map(str::to_string).collect();
        profile.scenarios = scenarios.into_iter().collect();
        profile.outcomes = outcomes.into_iter().collect();
        profile
    }

    /// Build the anchor → run → resource evidence path.
    ///
    /// Preference order, so the path shown is the most governance-relevant
    /// one available and is stable across runs:
    ///
    /// 1. a run the anchor initiated that was **refused** this resource,
    /// 2. a run the anchor initiated that was permitted it,
    /// 3. any run that was refused it,
    /// 4. any run that was permitted it.
    ///
    /// Ties inside a tier break on the run's external id.
    pub fn evidence_path(&self, anchor: u32, resource: u32) -> EvidencePath {
        let anchor_runs = self.runs_of_agent(anchor);
        let mut best: Option<(u8, &str, &SnapshotEdge)> = None;
        let mut runs_touching = BTreeSet::new();
        let mut anchor_runs_touching = BTreeSet::new();
        let mut allowed_edges = 0usize;
        let mut denied_edges = 0usize;

        for edge in self.edges(resource) {
            if edge.outgoing {
                continue;
            }
            let denied = match &*edge.relation {
                ALLOWED_ACCESS => false,
                DENIED_ACCESS => true,
                _ => continue,
            };
            if denied {
                denied_edges += 1;
            } else {
                allowed_edges += 1;
            }
            runs_touching.insert(edge.other);
            let from_anchor = anchor_runs.contains(&edge.other);
            if from_anchor {
                anchor_runs_touching.insert(edge.other);
            }
            let tier = match (from_anchor, denied) {
                (true, true) => 0,
                (true, false) => 1,
                (false, true) => 2,
                (false, false) => 3,
            };
            let run_id = self.node(edge.other).external_id.as_str();
            let better = match &best {
                None => true,
                Some((best_tier, best_run, _)) => {
                    tier < *best_tier || (tier == *best_tier && run_id < *best_run)
                }
            };
            if better {
                best = Some((tier, run_id, edge));
            }
        }

        let anchor_id = self.node(anchor).external_id.clone();
        let resource_id = self.node(resource).external_id.clone();
        let (path, relation, action, via_anchor_run) = match best {
            Some((tier, run_id, edge)) => (
                vec![
                    anchor_id,
                    run_id.to_string(),
                    edge.relation.to_string(),
                    resource_id,
                ],
                edge.relation.to_string(),
                edge.action.as_ref().map(|a| a.to_string()),
                tier <= 1,
            ),
            // A resource with no access edge at all cannot appear as a
            // retrieval candidate (its document is built from those edges),
            // but the path stays total rather than panicking.
            None => (
                vec![anchor_id, "no_shared_run".into(), resource_id],
                "no_access_edge".into(),
                None,
                false,
            ),
        };

        EvidencePath {
            path,
            relation,
            action,
            via_anchor_run,
            runs_touching: runs_touching.len(),
            anchor_runs_touching: anchor_runs_touching.len(),
            allowed_edges,
            denied_edges,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two agents, three runs, two resources — one contested (allowed for
    /// one run, denied for another), one permitted only for a run the
    /// anchor did not initiate.
    fn snapshot_json() -> String {
        serde_json::json!({
            "workspace_id": "demo-workspace",
            "revision": "sandbox-v1-test",
            "entities": [
                {"id": "agent:claude-code-demo", "label": "claude-code-demo",
                 "entity_type": "agent_profile"},
                {"id": "agent:codex-demo", "label": "codex-demo",
                 "entity_type": "agent_profile"},
                {"id": "run:aaa", "label": "run aaa (denied-executable)", "entity_type": "run"},
                {"id": "run:bbb", "label": "run bbb (approved-executable)", "entity_type": "run"},
                {"id": "run:ccc", "label": "run ccc (repository-read)", "entity_type": "run"},
                {"id": "outcome:completed", "label": "completed", "entity_type": "outcome"},
                {"id": "outcome:failed", "label": "failed", "entity_type": "outcome"},
                {"id": "resource:file:contested", "label": "/usr/bin/wc",
                 "entity_type": "resource"},
                {"id": "resource:directory:elsewhere", "label": "directory 07d0d6e9",
                 "entity_type": "resource"}
            ],
            "relationships": [
                {"id": "e1", "source_id": "agent:claude-code-demo", "target_id": "run:aaa",
                 "keywords": ["initiated"]},
                {"id": "e2", "source_id": "agent:claude-code-demo", "target_id": "run:bbb",
                 "keywords": ["initiated"]},
                {"id": "e3", "source_id": "agent:codex-demo", "target_id": "run:ccc",
                 "keywords": ["initiated"]},
                {"id": "e4", "source_id": "run:aaa", "target_id": "outcome:failed",
                 "keywords": ["ended_as"]},
                {"id": "e5", "source_id": "run:bbb", "target_id": "outcome:completed",
                 "keywords": ["ended_as"]},
                {"id": "e6", "source_id": "run:ccc", "target_id": "outcome:completed",
                 "keywords": ["ended_as"]},
                {"id": "e7", "source_id": "run:aaa", "target_id": "resource:file:contested",
                 "keywords": ["denied_access", "process_exec"]},
                {"id": "e8", "source_id": "run:bbb", "target_id": "resource:file:contested",
                 "keywords": ["allowed_access", "process_exec"]},
                {"id": "e9", "source_id": "run:ccc", "target_id": "resource:directory:elsewhere",
                 "keywords": ["allowed_access", "file_open_read_only"]}
            ]
        })
        .to_string()
    }

    fn index() -> SandboxSnapshotIndex {
        let raw = snapshot_json();
        let sha = sha256_hex(raw.as_bytes());
        SandboxSnapshotIndex::from_bytes(raw.as_bytes(), &sha, "demo-workspace", "sandbox-v1-test")
            .unwrap()
    }

    #[test]
    fn a_snapshot_whose_bytes_changed_is_refused() {
        let raw = snapshot_json();
        let err = SandboxSnapshotIndex::from_bytes(
            raw.as_bytes(),
            &sha256_hex(b"some other snapshot"),
            "demo-workspace",
            "sandbox-v1-test",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SandboxDemoError::SnapshotMismatch {
                field: "sha256",
                ..
            }
        ));
    }

    #[test]
    fn a_snapshot_for_another_revision_is_refused() {
        let raw = snapshot_json();
        let sha = sha256_hex(raw.as_bytes());
        let err = SandboxSnapshotIndex::from_bytes(
            raw.as_bytes(),
            &sha,
            "demo-workspace",
            "sandbox-v1-other",
        )
        .unwrap_err();
        assert!(matches!(
            err,
            SandboxDemoError::SnapshotMismatch {
                field: "revision",
                ..
            }
        ));
    }

    #[test]
    fn external_ids_survive_indexing_unchanged() {
        let index = index();
        assert_eq!(index.resources().len(), 2);
        let ids: Vec<&str> = index
            .resources()
            .iter()
            .map(|r| index.node(*r).external_id.as_str())
            .collect();
        assert_eq!(
            ids,
            vec!["resource:file:contested", "resource:directory:elsewhere"]
        );
        // The label is a display name and must never become the id.
        assert_eq!(index.node(index.resources()[0]).label, "/usr/bin/wc");
    }

    #[test]
    fn a_contested_resource_profiles_both_decisions() {
        let index = index();
        let profile = index.resource_profile(index.index_of("resource:file:contested").unwrap());
        assert_eq!(profile.kind, "file");
        assert_eq!(profile.allowed_edges, 1);
        assert_eq!(profile.denied_edges, 1);
        assert_eq!(profile.allowed_runs, 1);
        assert_eq!(profile.denied_runs, 1);
        assert_eq!(profile.total_runs, 2);
        assert_eq!(profile.actions, vec!["process_exec"]);
        assert_eq!(profile.denied_actions, vec!["process_exec"]);
        assert_eq!(profile.agent_profiles, vec!["claude-code-demo"]);
        assert_eq!(
            profile.scenarios,
            vec![
                ("approved-executable".to_string(), 1),
                ("denied-executable".to_string(), 1)
            ]
        );
        assert_eq!(
            profile.outcomes,
            vec![("completed".to_string(), 1), ("failed".to_string(), 1)]
        );
    }

    #[test]
    fn evidence_prefers_a_denial_by_the_anchors_own_run() {
        let index = index();
        let anchor = index.index_of("agent:claude-code-demo").unwrap();
        let contested = index.index_of("resource:file:contested").unwrap();

        let evidence = index.evidence_path(anchor, contested);
        assert_eq!(
            evidence.path,
            vec![
                "agent:claude-code-demo",
                "run:aaa",
                "denied_access",
                "resource:file:contested"
            ]
        );
        assert_eq!(evidence.action.as_deref(), Some("process_exec"));
        assert!(evidence.via_anchor_run);
        assert_eq!(evidence.runs_touching, 2);
        assert_eq!(evidence.anchor_runs_touching, 2);
        assert_eq!(evidence.denied_edges, 1);
    }

    #[test]
    fn a_resource_the_anchor_never_touched_says_so() {
        let index = index();
        let anchor = index.index_of("agent:claude-code-demo").unwrap();
        let elsewhere = index.index_of("resource:directory:elsewhere").unwrap();

        let evidence = index.evidence_path(anchor, elsewhere);
        assert!(!evidence.via_anchor_run);
        assert_eq!(evidence.anchor_runs_touching, 0);
        assert_eq!(evidence.runs_touching, 1);
        assert_eq!(evidence.path[1], "run:ccc");
    }

    #[test]
    fn the_projection_carries_every_relationship_exactly_once() {
        let index = index();
        let projected = index.projection();

        assert_eq!(projected.nodes.len(), index.node_count());
        assert_eq!(
            projected.edges.len(),
            index.edge_count(),
            "the adjacency list holds each edge twice; the projection must not"
        );

        // The count above is the invariant. Distinctness only holds because
        // this fixture has no parallel relationships — a real snapshot does,
        // one per audit event — so what is checked here is the shape of the
        // both-endpoints bug: the same edge coming back reversed.
        let mut seen = BTreeSet::new();
        for edge in &projected.edges {
            seen.insert((edge[0], edge[1], edge[2]));
        }
        for edge in &projected.edges {
            assert!(
                !seen.contains(&(edge[1], edge[0], edge[2])),
                "relationship {edge:?} was also projected from its other endpoint"
            );
        }
    }

    #[test]
    fn the_projection_keeps_ids_relations_and_actions_intact() {
        let index = index();
        let projected = index.projection();

        // Node order is snapshot order, and external ids are untouched:
        // the page resolves a result's external_id straight into this list.
        assert_eq!(projected.nodes[0].id, "agent:claude-code-demo");
        assert_eq!(projected.nodes[0].entity_type, AGENT_TYPE);

        let denied = projected
            .edges
            .iter()
            .find(|edge| projected.relations[edge[2] as usize] == DENIED_ACCESS)
            .expect("the fixture carries one refusal");
        assert_eq!(projected.nodes[denied[0] as usize].id, "run:aaa");
        assert_eq!(
            projected.nodes[denied[1] as usize].id,
            "resource:file:contested"
        );
        assert_eq!(projected.actions[denied[3] as usize], "process_exec");

        // An edge with no action keyword says so rather than borrowing one.
        let initiated = projected
            .edges
            .iter()
            .find(|edge| projected.relations[edge[2] as usize] == INITIATED)
            .expect("the fixture carries initiated edges");
        assert_eq!(initiated[3], -1);
    }

    #[test]
    fn a_dangling_relationship_endpoint_refuses_the_index() {
        let mut raw: serde_json::Value = serde_json::from_str(&snapshot_json()).unwrap();
        raw["relationships"][0]["target_id"] = serde_json::json!("run:missing");
        let bytes = serde_json::to_vec(&raw).unwrap();
        let err = SandboxSnapshotIndex::from_bytes(
            &bytes,
            &sha256_hex(&bytes),
            "demo-workspace",
            "sandbox-v1-test",
        )
        .unwrap_err();
        assert!(matches!(err, SandboxDemoError::Snapshot(m) if m.contains("dangling")));
    }
}
