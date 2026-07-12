//! Read operations for the relational graph store.

use std::collections::{HashMap, HashSet};

use async_trait::async_trait;
use sqlx::Row;
use uuid::Uuid;

use super::{
    build_scope_where, parse_independent_scope, row_to_edge, row_to_node, SqlParams, EDGES_TABLE,
    NODES_TABLE,
};
use super::PostgresRelationalGraphStorage;
use crate::error::{Result, StorageError};
use crate::traits::{
    GraphEdge, GraphNode, GraphStorage, GraphStorageReadOps, KnowledgeGraph,
};

type Scope = Option<(Uuid, Uuid)>;

fn db_err(context: &str, e: sqlx::Error) -> StorageError {
    StorageError::Database(format!("{context}: {e}"))
}

impl PostgresRelationalGraphStorage {
    /// Scope predicate for a fully-resolved `(tenant, workspace)` pair.
    fn scope_clause(alias: &str, scope: Scope) -> String {
        match scope {
            Some((t, w)) => build_scope_where(alias, Some(t), Some(w)),
            None => "TRUE".to_string(),
        }
    }

    async fn fetch_node_scoped(&self, node_id: &str, scope: Scope) -> Result<Option<GraphNode>> {
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT node_id, properties FROM {NODES_TABLE} n \
             WHERE n.node_id = $1 AND {} LIMIT 1",
            Self::scope_clause("n", scope)
        );
        let row = sqlx::query(&sql)
            .bind(node_id)
            .fetch_optional(&pool)
            .await
            .map_err(|e| db_err("fetch_node_scoped", e))?;
        Ok(row.map(|r| row_to_node(r.get("node_id"), r.get("properties"))))
    }

    async fn fetch_incident_edges_scoped(
        &self,
        node_id: &str,
        scope: Scope,
    ) -> Result<Vec<GraphEdge>> {
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT source_id, target_id, properties FROM {EDGES_TABLE} e \
             WHERE (e.source_id = $1 OR e.target_id = $1) AND {}",
            Self::scope_clause("e", scope)
        );
        let rows = sqlx::query(&sql)
            .bind(node_id)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("fetch_incident_edges_scoped", e))?;
        Ok(rows
            .into_iter()
            .map(|r| row_to_edge(r.get("source_id"), r.get("target_id"), r.get("properties")))
            .collect())
    }

    async fn fetch_edges_between_scoped(
        &self,
        node_ids: &[String],
        scope: Scope,
    ) -> Result<Vec<GraphEdge>> {
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT source_id, target_id, properties FROM {EDGES_TABLE} e \
             WHERE e.source_id = ANY($1) AND e.target_id = ANY($1) AND {}",
            Self::scope_clause("e", scope)
        );
        let rows = sqlx::query(&sql)
            .bind(node_ids)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("fetch_edges_between_scoped", e))?;
        Ok(rows
            .into_iter()
            .map(|r| row_to_edge(r.get("source_id"), r.get("target_id"), r.get("properties")))
            .collect())
    }

    /// Batch variant of `fetch_node_scoped`: which of `node_ids` exist in scope.
    async fn fetch_nodes_scoped(&self, node_ids: &[String], scope: Scope) -> Result<Vec<GraphNode>> {
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT node_id, properties FROM {NODES_TABLE} n \
             WHERE n.node_id = ANY($1) AND {}",
            Self::scope_clause("n", scope)
        );
        let rows = sqlx::query(&sql)
            .bind(node_ids)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("fetch_nodes_scoped", e))?;
        Ok(rows
            .into_iter()
            .map(|r| row_to_node(r.get("node_id"), r.get("properties")))
            .collect())
    }

    /// All scoped edges incident to any of `node_ids` (batched BFS frontier).
    async fn fetch_incident_edges_for_ids_scoped(
        &self,
        node_ids: &[String],
        scope: Scope,
    ) -> Result<Vec<GraphEdge>> {
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT source_id, target_id, properties FROM {EDGES_TABLE} e \
             WHERE (e.source_id = ANY($1) OR e.target_id = ANY($1)) AND {}",
            Self::scope_clause("e", scope)
        );
        let rows = sqlx::query(&sql)
            .bind(node_ids)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("fetch_incident_edges_for_ids_scoped", e))?;
        Ok(rows
            .into_iter()
            .map(|r| row_to_edge(r.get("source_id"), r.get("target_id"), r.get("properties")))
            .collect())
    }
}

#[async_trait]
impl GraphStorage for PostgresRelationalGraphStorage {
    fn namespace(&self) -> &str {
        self.pg_namespace()
    }

    async fn initialize(&self) -> Result<()> {
        self.pg_initialize().await
    }

    async fn finalize(&self) -> Result<()> {
        self.pg_finalize().await
    }
}

#[async_trait]
#[allow(deprecated)]
impl GraphStorageReadOps for PostgresRelationalGraphStorage {
    async fn has_node(&self, node_id: &str) -> Result<bool> {
        let pool = self.pool.get().await?;
        let sql = format!("SELECT EXISTS(SELECT 1 FROM {NODES_TABLE} WHERE node_id = $1)");
        let exists: bool = sqlx::query_scalar(&sql)
            .bind(node_id)
            .fetch_one(&pool)
            .await
            .map_err(|e| db_err("has_node", e))?;
        Ok(exists)
    }

    async fn get_node(&self, node_id: &str) -> Result<Option<GraphNode>> {
        self.fetch_node_scoped(node_id, None).await
    }

    async fn node_degree(&self, node_id: &str) -> Result<usize> {
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT COUNT(*)::BIGINT FROM {EDGES_TABLE} WHERE source_id = $1 OR target_id = $1"
        );
        let count: i64 = sqlx::query_scalar(&sql)
            .bind(node_id)
            .fetch_one(&pool)
            .await
            .map_err(|e| db_err("node_degree", e))?;
        Ok(count as usize)
    }

    async fn node_degrees_batch(&self, node_ids: &[String]) -> Result<Vec<(String, usize)>> {
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }
        let pool = self.pool.get().await?;
        // Single round-trip: LEFT JOIN keeps zero-degree ids in the result.
        let sql = format!(
            "SELECT q.nid AS nid, COUNT(e.source_id)::BIGINT AS deg \
             FROM unnest($1::text[]) AS q(nid) \
             LEFT JOIN {EDGES_TABLE} e ON e.source_id = q.nid OR e.target_id = q.nid \
             GROUP BY q.nid"
        );
        let rows = sqlx::query(&sql)
            .bind(node_ids)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("node_degrees_batch", e))?;
        let mut degrees: HashMap<String, usize> = HashMap::new();
        for r in rows {
            let nid: String = r.get("nid");
            let deg: i64 = r.get("deg");
            degrees.insert(nid, deg as usize);
        }
        Ok(node_ids
            .iter()
            .map(|id| (id.clone(), degrees.get(id).copied().unwrap_or(0)))
            .collect())
    }

    async fn get_all_nodes(&self) -> Result<Vec<GraphNode>> {
        let pool = self.pool.get().await?;
        let sql = format!("SELECT node_id, properties FROM {NODES_TABLE}");
        let rows = sqlx::query(&sql)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("get_all_nodes", e))?;
        Ok(rows
            .into_iter()
            .map(|r| row_to_node(r.get("node_id"), r.get("properties")))
            .collect())
    }

    async fn get_nodes_by_ids(&self, node_ids: &[String]) -> Result<Vec<GraphNode>> {
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }
        let pool = self.pool.get().await?;
        let sql = format!("SELECT node_id, properties FROM {NODES_TABLE} WHERE node_id = ANY($1)");
        let rows = sqlx::query(&sql)
            .bind(node_ids)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("get_nodes_by_ids", e))?;
        Ok(rows
            .into_iter()
            .map(|r| row_to_node(r.get("node_id"), r.get("properties")))
            .collect())
    }

    async fn get_nodes_batch(&self, node_ids: &[String]) -> Result<HashMap<String, GraphNode>> {
        let nodes = self.get_nodes_by_ids(node_ids).await?;
        Ok(nodes.into_iter().map(|n| (n.id.clone(), n)).collect())
    }

    async fn get_incident_edges_batch(&self, node_ids: &[String]) -> Result<Vec<GraphEdge>> {
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT source_id, target_id, properties FROM {EDGES_TABLE} \
             WHERE source_id = ANY($1) OR target_id = ANY($1)"
        );
        let rows = sqlx::query(&sql)
            .bind(node_ids)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("get_incident_edges_batch", e))?;
        Ok(rows
            .into_iter()
            .map(|r| row_to_edge(r.get("source_id"), r.get("target_id"), r.get("properties")))
            .collect())
    }

    async fn get_edges_for_node_set(
        &self,
        node_ids: &[String],
        tenant_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<Vec<GraphEdge>> {
        if node_ids.is_empty() {
            return Ok(Vec::new());
        }
        let Some((tenant, workspace)) = parse_independent_scope(tenant_id, workspace_id) else {
            return Ok(Vec::new());
        };
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT source_id, target_id, properties FROM {EDGES_TABLE} e \
             WHERE e.source_id = ANY($1) AND e.target_id = ANY($1) AND {}",
            build_scope_where("e", tenant, workspace)
        );
        let rows = sqlx::query(&sql)
            .bind(node_ids)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("get_edges_for_node_set", e))?;
        Ok(rows
            .into_iter()
            .map(|r| row_to_edge(r.get("source_id"), r.get("target_id"), r.get("properties")))
            .collect())
    }

    async fn has_edge(&self, source: &str, target: &str) -> Result<bool> {
        let (s, t) = super::canonical_edge(source, target);
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT EXISTS(SELECT 1 FROM {EDGES_TABLE} WHERE source_id = $1 AND target_id = $2)"
        );
        let exists: bool = sqlx::query_scalar(&sql)
            .bind(&s)
            .bind(&t)
            .fetch_one(&pool)
            .await
            .map_err(|e| db_err("has_edge", e))?;
        Ok(exists)
    }

    async fn get_edge(&self, source: &str, target: &str) -> Result<Option<GraphEdge>> {
        let (s, t) = super::canonical_edge(source, target);
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT source_id, target_id, properties FROM {EDGES_TABLE} \
             WHERE source_id = $1 AND target_id = $2 LIMIT 1"
        );
        let row = sqlx::query(&sql)
            .bind(&s)
            .bind(&t)
            .fetch_optional(&pool)
            .await
            .map_err(|e| db_err("get_edge", e))?;
        Ok(row.map(|r| row_to_edge(r.get("source_id"), r.get("target_id"), r.get("properties"))))
    }

    async fn get_node_edges(&self, node_id: &str) -> Result<Vec<GraphEdge>> {
        self.fetch_incident_edges_scoped(node_id, None).await
    }

    async fn get_all_edges(&self) -> Result<Vec<GraphEdge>> {
        let pool = self.pool.get().await?;
        let sql = format!("SELECT source_id, target_id, properties FROM {EDGES_TABLE}");
        let rows = sqlx::query(&sql)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("get_all_edges", e))?;
        Ok(rows
            .into_iter()
            .map(|r| row_to_edge(r.get("source_id"), r.get("target_id"), r.get("properties")))
            .collect())
    }

    async fn get_knowledge_graph(
        &self,
        start_node: &str,
        max_depth: usize,
        max_nodes: usize,
        tenant_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<KnowledgeGraph> {
        // Scope only when BOTH tenant and workspace are supplied (mirrors
        // MemoryGraphStorage which builds a node filter only for (Some, Some)).
        let scope: Scope = match (tenant_id, workspace_id) {
            (None, None) => None,
            (Some(t), Some(w)) => match (Uuid::parse_str(t), Uuid::parse_str(w)) {
                (Ok(t), Ok(w)) => Some((t, w)),
                _ => return Ok(KnowledgeGraph::new()),
            },
            // Partial scope (exactly one dimension) cannot be satisfied — fail
            // closed with an empty graph rather than degrading to an unscoped BFS.
            _ => return Ok(KnowledgeGraph::new()),
        };

        if max_nodes == 0 {
            return Ok(KnowledgeGraph::new());
        }

        // Scoped traversal requires the start node itself to satisfy the filter.
        if scope.is_some() && self.fetch_node_scoped(start_node, scope).await?.is_none() {
            return Ok(KnowledgeGraph::new());
        }

        // Level-synchronous BFS: one batched node query and one batched edge
        // query per depth (O(depth) round trips) instead of the previous
        // per-node / per-neighbor N+1 fetches.
        let mut visited: HashSet<String> = HashSet::new();
        let mut result_nodes: Vec<GraphNode> = Vec::new();
        let mut frontier: Vec<String> = vec![start_node.to_string()];
        let mut depth = 0usize;

        while !frontier.is_empty() && depth <= max_depth && result_nodes.len() < max_nodes {
            // Dedup the frontier and drop already-visited ids before the fetch.
            let mut level_ids: Vec<String> = Vec::new();
            let mut level_seen: HashSet<String> = HashSet::new();
            for id in &frontier {
                if !visited.contains(id) && level_seen.insert(id.clone()) {
                    level_ids.push(id.clone());
                }
            }
            if level_ids.is_empty() {
                break;
            }

            // Which frontier ids actually exist in scope.
            let present: HashMap<String, GraphNode> = self
                .fetch_nodes_scoped(&level_ids, scope)
                .await?
                .into_iter()
                .map(|n| (n.id.clone(), n))
                .collect();

            // Commit in frontier order (BFS), respecting the max_nodes cap.
            let mut committed: Vec<String> = Vec::new();
            for id in &level_ids {
                if result_nodes.len() >= max_nodes {
                    break;
                }
                if let Some(node) = present.get(id) {
                    if visited.insert(id.clone()) {
                        result_nodes.push(node.clone());
                        committed.push(id.clone());
                    }
                }
            }

            if depth == max_depth || result_nodes.len() >= max_nodes {
                break;
            }

            // Derive the next frontier from edges incident to committed nodes.
            let mut next: Vec<String> = Vec::new();
            let mut next_seen: HashSet<String> = HashSet::new();
            for edge in self
                .fetch_incident_edges_for_ids_scoped(&committed, scope)
                .await?
            {
                for cand in [edge.source, edge.target] {
                    if !visited.contains(&cand) && next_seen.insert(cand.clone()) {
                        next.push(cand);
                    }
                }
            }
            frontier = next;
            depth += 1;
        }

        let visited_ids: Vec<String> = visited.iter().cloned().collect();
        let edges = self.fetch_edges_between_scoped(&visited_ids, scope).await?;

        Ok(KnowledgeGraph {
            nodes: result_nodes,
            edges,
            is_truncated: visited.len() >= max_nodes,
        })
    }

    async fn get_popular_labels(
        &self,
        limit: usize,
        tenant_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<Vec<String>> {
        let Some((tenant, workspace)) = parse_independent_scope(tenant_id, workspace_id) else {
            return Ok(Vec::new());
        };
        let pool = self.pool.get().await?;
        // Scope the edge side of the join to the node's tenant/workspace, else
        // degree counts leak edges from other tenants whose endpoint ids collide.
        let sql = format!(
            "SELECT n.node_id AS nid, COUNT(e.source_id)::BIGINT AS deg \
             FROM {NODES_TABLE} n \
             LEFT JOIN {EDGES_TABLE} e \
               ON (e.source_id = n.node_id OR e.target_id = n.node_id) \
               AND e.tenant_id = n.tenant_id AND e.workspace_id = n.workspace_id \
             WHERE {} \
             GROUP BY n.node_id \
             ORDER BY deg DESC, n.node_id ASC \
             LIMIT {limit}",
            build_scope_where("n", tenant, workspace)
        );
        let rows = sqlx::query(&sql)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("get_popular_labels", e))?;
        Ok(rows.into_iter().map(|r| r.get::<String, _>("nid")).collect())
    }

    async fn search_labels(
        &self,
        query: &str,
        limit: usize,
        tenant_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<Vec<String>> {
        let Some((tenant, workspace)) = parse_independent_scope(tenant_id, workspace_id) else {
            return Ok(Vec::new());
        };
        let pool = self.pool.get().await?;
        let pattern = format!("%{}%", super::escape_like(&query.to_lowercase()));
        let sql = format!(
            "SELECT node_id FROM {NODES_TABLE} n \
             WHERE {} AND LOWER(n.node_id) LIKE $1 ESCAPE '\\' \
             ORDER BY n.node_id ASC LIMIT {limit}",
            build_scope_where("n", tenant, workspace)
        );
        let rows = sqlx::query(&sql)
            .bind(&pattern)
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("search_labels", e))?;
        Ok(rows
            .into_iter()
            .map(|r| r.get::<String, _>("node_id"))
            .collect())
    }

    async fn search_nodes(
        &self,
        query: &str,
        limit: usize,
        entity_type: Option<&str>,
        tenant_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<Vec<(GraphNode, usize)>> {
        let Some((tenant, workspace)) = parse_independent_scope(tenant_id, workspace_id) else {
            return Ok(Vec::new());
        };
        let pool = self.pool.get().await?;
        let mut params = SqlParams::new();
        let pattern_ph = params.bind(format!("%{}%", super::escape_like(&query.to_lowercase())));

        let mut conditions = vec![build_scope_where("n", tenant, workspace)];
        conditions.push(format!(
            "(LOWER(n.node_id) LIKE {pattern_ph} ESCAPE '\\' \
             OR LOWER(n.properties->>'description') LIKE {pattern_ph} ESCAPE '\\')"
        ));
        // search_nodes matches entity_type case-sensitively (mirrors memory).
        if let Some(et) = entity_type {
            let et_ph = params.bind(et.to_string());
            conditions.push(format!("n.properties->>'entity_type' = {et_ph}"));
        }

        // Scope the LATERAL degree count to the node's tenant/workspace so it
        // does not count colliding-id edges from other tenants.
        let sql = format!(
            "SELECT n.node_id AS nid, n.properties AS props, \
                    COALESCE(d.deg, 0)::BIGINT AS deg \
             FROM {NODES_TABLE} n \
             LEFT JOIN LATERAL ( \
                 SELECT COUNT(*) AS deg FROM {EDGES_TABLE} e \
                 WHERE (e.source_id = n.node_id OR e.target_id = n.node_id) \
                   AND e.tenant_id = n.tenant_id AND e.workspace_id = n.workspace_id \
             ) d ON TRUE \
             WHERE {} \
             ORDER BY deg DESC, n.node_id ASC \
             LIMIT {limit}",
            conditions.join(" AND ")
        );
        let mut q = sqlx::query(&sql);
        for b in params.binds() {
            q = q.bind(b);
        }
        let rows = q
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("search_nodes", e))?;
        Ok(rows
            .into_iter()
            .map(|r| {
                let deg: i64 = r.get("deg");
                (
                    row_to_node(r.get("nid"), r.get("props")),
                    deg as usize,
                )
            })
            .collect())
    }

    async fn get_neighbors(
        &self,
        node_id: &str,
        depth: usize,
        tenant_id: Option<&str>,
        workspace_id: Option<&str>,
    ) -> Result<Vec<GraphNode>> {
        // Scope the BFS itself (when tenant+workspace are supplied) so traversal
        // never crosses another tenant's edges; the post-filter below still drops
        // the origin node and re-checks the property scope as defense in depth.
        let kg = self
            .get_knowledge_graph(node_id, depth, 1000, tenant_id, workspace_id)
            .await?;
        Ok(kg
            .nodes
            .into_iter()
            .filter(|n| {
                if n.id == node_id {
                    return false;
                }
                if let Some(tid) = tenant_id {
                    let node_tid = n
                        .properties
                        .get("tenant_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if node_tid != tid {
                        return false;
                    }
                }
                if let Some(wid) = workspace_id {
                    let node_wid = n
                        .properties
                        .get("workspace_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if node_wid != wid {
                        return false;
                    }
                }
                true
            })
            .collect())
    }
}
