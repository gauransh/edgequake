//! Graph analytics and statistics operations (SPEC-017 ISP Phase 2b).

use async_trait::async_trait;

use crate::error::Result;

use super::graph_read_ops::GraphStorageReadOps;
use super::graph_scan_ops::GraphScanOps;

/// Counts, estimates, and workspace-scoped statistics.
///
/// # Workspace scoping (P-G12 / RC-17, LSP)
///
/// `node_count_by_workspace` and `edge_count_by_workspace` are **required**
/// (no default). The previous default ignored `workspace_id` and returned the
/// GLOBAL count, which silently leaked cross-workspace counts to any adapter
/// that forgot to override it. Making them required forces every adapter to
/// implement honest workspace scoping (E32: a workspace with zero nodes must
/// return 0, not the global count).
#[async_trait]
pub trait GraphStorageAnalyticsOps: GraphStorageReadOps + GraphScanOps {
    async fn node_count(&self) -> Result<usize>;

    async fn edge_count(&self) -> Result<usize>;

    async fn node_count_fast(&self) -> Result<usize> {
        self.node_count().await
    }

    async fn edge_count_fast(&self) -> Result<usize> {
        self.edge_count().await
    }

    async fn ping(&self) -> Result<()> {
        let _ = self.node_count().await?;
        Ok(())
    }

    /// Count nodes scoped to a single workspace (required; no default).
    async fn node_count_by_workspace(&self, workspace_id: &uuid::Uuid) -> Result<usize>;

    /// Count edges scoped to a single workspace (required; no default).
    async fn edge_count_by_workspace(&self, workspace_id: &uuid::Uuid) -> Result<usize>;

    async fn distinct_node_type_count_by_workspace(
        &self,
        workspace_id: &uuid::Uuid,
    ) -> Result<usize> {
        let _ = workspace_id;
        Err(crate::error::StorageError::InvalidQuery(
            "distinct_node_type_count_by_workspace requires adapter-specific implementation \
             (SPEC-006: no get_all_nodes fallback)"
                .into(),
        ))
    }

    /// Count nodes whose `source_ids` (or legacy `source_id`) contains any
    /// entry starting with `prefix` (SPEC-021 P-A3).
    ///
    /// WHY: the per-document `entity_count` cell in the Documents list must
    /// fall back to the authoritative AGE graph when both the KV metadata and
    /// the relational `documents.entity_count` column are missing/stale. The
    /// chunk-id prefix `{doc_id}-chunk-` uniquely identifies a document's
    /// entities without a full graph scan.
    ///
    /// Default: scans via `find_nodes_by_source_prefixes` and counts —
    /// adapters SHOULD override with a single aggregate Cypher/SQL for O(log N)
    /// instead of materializing the nodes.
    async fn node_count_by_source_prefix(&self, prefix: &str) -> Result<usize> {
        use super::graph_scan_ops::{collect_source_references, NodeListFilter};
        let filter = NodeListFilter::default();
        let prefixes = vec![prefix.to_string()];
        let nodes = self
            .find_nodes_by_source_prefixes(&filter, &prefixes)
            .await?;
        let count = nodes
            .iter()
            .filter(|n| {
                collect_source_references(&n.properties)
                    .iter()
                    .any(|s| s.starts_with(prefix))
            })
            .count();
        Ok(count)
    }

    /// Count distinct nodes whose `source_ids` reference ANY of the given
    /// document prefixes (issue #16).
    ///
    /// WHY: workspace entity stats derived purely from the `workspace_id` node
    /// property under-report ~40x when entity nodes were persisted without a
    /// (consistent) `workspace_id` tag. Counting the entities actually derived
    /// from the workspace's documents — via the same source-reference mechanism
    /// the per-document `entity_count` already trusts — recovers the honest
    /// count. Callers take `max(property_count, source_count)` so a correctly
    /// tagged graph is unaffected while an under-tagged one is repaired.
    ///
    /// Default: scans via `find_nodes_by_source_prefixes`; adapters SHOULD
    /// override with a single aggregate SQL for O(log N).
    async fn node_count_by_source_prefixes(&self, prefixes: &[String]) -> Result<usize> {
        use super::graph_scan_ops::NodeListFilter;
        if prefixes.is_empty() {
            return Ok(0);
        }
        let filter = NodeListFilter::default();
        let nodes = self
            .find_nodes_by_source_prefixes(&filter, prefixes)
            .await?;
        Ok(nodes.len())
    }

    /// Count distinct edges whose `source_id` references ANY of the given
    /// document prefixes (issue #16). See [`node_count_by_source_prefixes`].
    async fn edge_count_by_source_prefixes(&self, prefixes: &[String]) -> Result<usize> {
        use super::graph_scan_ops::EdgeListFilter;
        if prefixes.is_empty() {
            return Ok(0);
        }
        let filter = EdgeListFilter::default();
        let edges = self
            .find_edges_by_source_prefixes(&filter, prefixes)
            .await?;
        Ok(edges.len())
    }
}
