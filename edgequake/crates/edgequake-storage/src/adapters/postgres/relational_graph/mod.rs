//! Plain-relational PostgreSQL graph storage (no Apache AGE, no Cypher).
//!
//! This module implements the full [`GraphStorage`] composite trait against two
//! ordinary relational tables — `eq_graph_nodes` and `eq_graph_edges` (see
//! migration 082) — scoped by `(tenant_id, workspace_id)` columns. It is a drop-in
//! alternative to [`PostgresAGEGraphStorage`](super::PostgresAGEGraphStorage) for
//! deployments that do not run the AGE extension.
//!
//! ## Semantics
//!
//! Behaviour mirrors [`MemoryGraphStorage`](crate::adapters::memory::MemoryGraphStorage):
//!
//! - Edges are **undirected** and stored canonically (`source_id <= target_id`),
//!   so a single row represents one relationship regardless of insertion order.
//! - `properties` JSONB round-trips exactly what was written; the dedicated UUID
//!   columns are derived from `properties["tenant_id"]` / `["workspace_id"]` for
//!   cheap tenant-scoped filtering (missing/invalid → `Uuid::nil()`).
//! - Scoped reads (`Some` tenant AND workspace) filter strictly on the columns;
//!   `None` means "no filter". A tenant/workspace string that is not a valid UUID
//!   matches nothing (defense in depth).

use std::collections::HashMap;

use super::config::PostgresConfig;
use super::connection::PostgresPool;
use crate::error::{Result, StorageError};
use crate::traits::{EdgeListFilter, GraphEdge, GraphNode, NodeListFilter};

use uuid::Uuid;

mod analytics_ops;
mod mutate_ops;
mod read_ops;
mod scan_ops;

#[cfg(test)]
mod tests;

pub(super) const NODES_TABLE: &str = "eq_graph_nodes";
pub(super) const EDGES_TABLE: &str = "eq_graph_edges";

/// Plain-relational graph storage backed by `eq_graph_nodes` / `eq_graph_edges`.
pub struct PostgresRelationalGraphStorage {
    pool: PostgresPool,
    namespace: String,
}

impl PostgresRelationalGraphStorage {
    /// Create a new relational graph storage owning its own lazy pool.
    pub fn new(config: PostgresConfig) -> Self {
        Self::with_pool(PostgresPool::new(config.clone()), config)
    }

    /// Create relational graph storage over a shared connection pool (SPEC-011).
    pub fn with_pool(pool: PostgresPool, config: PostgresConfig) -> Self {
        Self {
            pool,
            namespace: config.namespace.clone(),
        }
    }

    /// Get the underlying pool.
    pub fn pool(&self) -> &PostgresPool {
        &self.pool
    }

    pub(super) fn pg_namespace(&self) -> &str {
        &self.namespace
    }

    /// Migrations own the DDL; initialize only ensures the pool is live and the
    /// database is reachable (health-check).
    pub(super) async fn pg_initialize(&self) -> Result<()> {
        if !self.pool.is_connected().await {
            self.pool.initialize().await?;
        }
        let pool = self.pool.get().await?;
        sqlx::query_scalar::<_, i32>("SELECT 1")
            .fetch_one(&pool)
            .await
            .map_err(|e| StorageError::Connection(format!("Relational graph health check: {e}")))?;
        Ok(())
    }

    pub(super) async fn pg_finalize(&self) -> Result<()> {
        Ok(())
    }
}

impl std::fmt::Debug for PostgresRelationalGraphStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresRelationalGraphStorage")
            .field("namespace", &self.namespace)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Pure helpers (unit-testable without a database).
// ---------------------------------------------------------------------------

/// Escape LIKE metacharacters so a substring search matches literally
/// (mirrors `str::contains`, which has no wildcard semantics). Pair with
/// `ESCAPE '\'` in the query. The result is always passed as a bound `$n`
/// parameter, so single quotes are left untouched (binding handles quoting).
pub(super) fn escape_like(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' | '%' | '_' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out
}

/// Canonical (sorted) undirected edge key — matches `MemoryGraphStorage::edge_key`.
pub(super) fn canonical_edge(source: &str, target: &str) -> (String, String) {
    if source <= target {
        (source.to_string(), target.to_string())
    } else {
        (target.to_string(), source.to_string())
    }
}

/// Derive the tenant/workspace UUID columns from a property map.
/// Missing or non-UUID values fall back to the nil UUID.
pub(super) fn scope_from_props(props: &HashMap<String, serde_json::Value>) -> (Uuid, Uuid) {
    let get = |key: &str| -> Uuid {
        props
            .get(key)
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
            .unwrap_or_else(Uuid::nil)
    };
    (get("tenant_id"), get("workspace_id"))
}

/// Convert a JSONB value into a property map (drops non-object payloads).
pub(super) fn json_to_props(value: serde_json::Value) -> HashMap<String, serde_json::Value> {
    match value {
        serde_json::Value::Object(map) => map.into_iter().collect(),
        _ => HashMap::new(),
    }
}

/// Column-level scope predicate. `None` on a dimension means no filter.
pub(super) fn build_scope_where(
    alias: &str,
    tenant_id: Option<Uuid>,
    workspace_id: Option<Uuid>,
) -> String {
    let mut conditions = Vec::new();
    if let Some(t) = tenant_id {
        conditions.push(format!("{alias}.tenant_id = '{t}'::uuid"));
    }
    if let Some(w) = workspace_id {
        conditions.push(format!("{alias}.workspace_id = '{w}'::uuid"));
    }
    if conditions.is_empty() {
        "TRUE".to_string()
    } else {
        conditions.join(" AND ")
    }
}

/// Parse the tenant/workspace pair from a list filter.
///
/// Returns `None` (=> caller yields an empty result) when the combination cannot
/// match anything: exactly one dimension present, or a non-UUID value — matching
/// `node_matches_list_filter`'s strict `(Some, Some) | (None, None)` contract.
fn parse_filter_scope(
    tenant_id: Option<&str>,
    workspace_id: Option<&str>,
) -> Option<(Option<Uuid>, Option<Uuid>)> {
    match (tenant_id, workspace_id) {
        (Some(t), Some(w)) => {
            let t = Uuid::parse_str(t).ok()?;
            let w = Uuid::parse_str(w).ok()?;
            Some((Some(t), Some(w)))
        }
        (None, None) => Some((None, None)),
        _ => None,
    }
}

/// Parse tenant/workspace scope where each dimension is applied independently
/// (used by the per-node read paths that filter each column on its own).
/// Returns `None` (=> empty result) when a supplied value is not a valid UUID.
pub(super) fn parse_independent_scope(
    tenant_id: Option<&str>,
    workspace_id: Option<&str>,
) -> Option<(Option<Uuid>, Option<Uuid>)> {
    let t = match tenant_id {
        Some(s) => Some(Uuid::parse_str(s).ok()?),
        None => None,
    };
    let w = match workspace_id {
        Some(s) => Some(Uuid::parse_str(s).ok()?),
        None => None,
    };
    Some((t, w))
}

/// Accumulates text bind parameters so user-influenced values reach Postgres as
/// `$n` placeholders instead of interpolated literals. Only validated scope
/// UUIDs and numeric community ids stay inlined (they cannot carry injection).
#[derive(Default)]
pub(super) struct SqlParams {
    binds: Vec<String>,
}

impl SqlParams {
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Register a value and return its positional `$n` placeholder.
    pub(super) fn bind(&mut self, value: String) -> String {
        self.binds.push(value);
        format!("${}", self.binds.len())
    }

    pub(super) fn binds(&self) -> &[String] {
        &self.binds
    }
}

/// Build the WHERE body for [`NodeListFilter`]. Returns `FALSE` for filters that
/// can never match (bad UUID / mixed scope). User-influenced values are appended
/// to `params` and referenced by placeholder; the caller binds `params` in order.
pub(super) fn build_node_filter_where(
    alias: &str,
    filter: &NodeListFilter,
    params: &mut SqlParams,
) -> String {
    let Some((tenant, workspace)) =
        parse_filter_scope(filter.tenant_id.as_deref(), filter.workspace_id.as_deref())
    else {
        return "FALSE".to_string();
    };

    let mut conditions = vec![build_scope_where(alias, tenant, workspace)];

    if let Some(entity_type) = &filter.entity_type {
        let ph = params.bind(entity_type.clone());
        conditions.push(format!(
            "LOWER({alias}.properties->>'entity_type') = LOWER({ph})"
        ));
    }

    if let Some(search) = &filter.search {
        let ph = params.bind(format!("%{}%", escape_like(&search.to_lowercase())));
        conditions.push(format!(
            "(LOWER({alias}.node_id) LIKE {ph} ESCAPE '\\' \
             OR LOWER({alias}.properties->>'description') LIKE {ph} ESCAPE '\\')"
        ));
    }

    if let Some(community_ids) = &filter.community_ids {
        if community_ids.is_empty() {
            conditions.push("FALSE".to_string());
        } else {
            let list = community_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(",");
            conditions.push(format!(
                "({alias}.properties->>'community_id')::bigint = ANY(ARRAY[{list}]::bigint[])"
            ));
        }
    }

    conditions.join(" AND ")
}

/// Build the WHERE body for [`EdgeListFilter`].
pub(super) fn build_edge_filter_where(
    alias: &str,
    filter: &EdgeListFilter,
    params: &mut SqlParams,
) -> String {
    let Some((tenant, workspace)) =
        parse_filter_scope(filter.tenant_id.as_deref(), filter.workspace_id.as_deref())
    else {
        return "FALSE".to_string();
    };

    let mut conditions = vec![build_scope_where(alias, tenant, workspace)];

    if let Some(rel_type) = &filter.relationship_type {
        let ph = params.bind(rel_type.clone());
        conditions.push(format!(
            "LOWER({alias}.properties->>'relation_type') = LOWER({ph})"
        ));
    }

    conditions.join(" AND ")
}

/// Build a coarse source-prefix predicate over `properties` (narrowing filter;
/// callers confirm exact membership in Rust via `sources_match_prefixes`).
pub(super) fn build_source_prefix_where(
    alias: &str,
    source_prefixes: &[String],
    params: &mut SqlParams,
) -> String {
    if source_prefixes.is_empty() {
        return "FALSE".to_string();
    }
    let mut clauses = Vec::new();
    for prefix in source_prefixes {
        let esc = escape_like(prefix);
        let pfx = params.bind(format!("{esc}%"));
        let mid = params.bind(format!("%|{esc}%"));
        clauses.push(format!(
            "{alias}.properties->>'source_id' LIKE {pfx} ESCAPE '\\' \
             OR {alias}.properties->>'source_id' LIKE {mid} ESCAPE '\\' \
             OR EXISTS (SELECT 1 FROM jsonb_array_elements_text( \
                    CASE WHEN jsonb_typeof({alias}.properties->'source_ids') = 'array' \
                         THEN {alias}.properties->'source_ids' ELSE '[]'::jsonb END \
                ) AS s WHERE s LIKE {pfx} ESCAPE '\\')"
        ));
    }
    clauses
        .into_iter()
        .map(|c| format!("({c})"))
        .collect::<Vec<_>>()
        .join(" OR ")
}

pub(super) fn row_to_node(node_id: String, properties: serde_json::Value) -> GraphNode {
    GraphNode {
        id: node_id,
        properties: json_to_props(properties),
    }
}

pub(super) fn row_to_edge(
    source: String,
    target: String,
    properties: serde_json::Value,
) -> GraphEdge {
    GraphEdge {
        source,
        target,
        properties: json_to_props(properties),
    }
}

#[cfg(test)]
mod builder_tests {
    use super::*;

    fn uid(n: u8) -> Uuid {
        Uuid::from_u128(n as u128)
    }

    #[test]
    fn escape_like_escapes_wildcards_but_not_quotes() {
        assert_eq!(escape_like("50%_off"), "50\\%\\_off");
        assert_eq!(escape_like("a\\b"), "a\\\\b");
        // Single quotes are preserved — the pattern is always a bound parameter.
        assert_eq!(escape_like("O'Brien"), "O'Brien");
    }

    #[test]
    fn canonical_edge_is_order_independent() {
        assert_eq!(
            canonical_edge("bob", "alice"),
            ("alice".to_string(), "bob".to_string())
        );
        assert_eq!(canonical_edge("alice", "bob"), canonical_edge("bob", "alice"));
    }

    #[test]
    fn scope_from_props_parses_uuids_and_defaults_nil() {
        let mut props = HashMap::new();
        let tid = uid(7);
        props.insert("tenant_id".to_string(), serde_json::json!(tid.to_string()));
        // workspace missing → nil
        let (t, w) = scope_from_props(&props);
        assert_eq!(t, tid);
        assert_eq!(w, Uuid::nil());

        props.insert("workspace_id".to_string(), serde_json::json!("not-a-uuid"));
        let (_, w2) = scope_from_props(&props);
        assert_eq!(w2, Uuid::nil());
    }

    #[test]
    fn build_scope_where_handles_each_dimension() {
        assert_eq!(build_scope_where("n", None, None), "TRUE");
        assert_eq!(
            build_scope_where("n", Some(uid(1)), None),
            format!("n.tenant_id = '{}'::uuid", uid(1))
        );
        let both = build_scope_where("n", Some(uid(1)), Some(uid(2)));
        assert!(both.contains("n.tenant_id"));
        assert!(both.contains("n.workspace_id"));
        assert!(both.contains(" AND "));
    }

    #[test]
    fn build_node_filter_where_mixed_scope_is_false() {
        let filter = NodeListFilter {
            tenant_id: Some(uid(1).to_string()),
            workspace_id: None,
            ..Default::default()
        };
        let mut params = SqlParams::new();
        assert_eq!(build_node_filter_where("n", &filter, &mut params), "FALSE");
    }

    #[test]
    fn build_node_filter_where_bad_uuid_is_false() {
        let filter = NodeListFilter {
            tenant_id: Some("nope".to_string()),
            workspace_id: Some("nope".to_string()),
            ..Default::default()
        };
        let mut params = SqlParams::new();
        assert_eq!(build_node_filter_where("n", &filter, &mut params), "FALSE");
    }

    #[test]
    fn build_node_filter_where_composes_conditions() {
        let filter = NodeListFilter {
            tenant_id: Some(uid(1).to_string()),
            workspace_id: Some(uid(2).to_string()),
            entity_type: Some("Person".to_string()),
            search: Some("ali'ce".to_string()),
            community_ids: Some(vec![3, 5]),
        };
        let mut params = SqlParams::new();
        let sql = build_node_filter_where("n", &filter, &mut params);
        assert!(sql.contains("n.tenant_id = "));
        assert!(sql.contains("n.workspace_id = "));
        // User-influenced values are placeholders, not interpolated literals.
        assert!(sql.contains("LOWER(n.properties->>'entity_type') = LOWER($1)"));
        assert!(sql.contains("LOWER(n.node_id) LIKE $2 ESCAPE '\\'"));
        assert!(!sql.contains("ali'ce"));
        assert!(sql.contains("(n.properties->>'community_id')::bigint = ANY(ARRAY[3,5]::bigint[])"));
        assert_eq!(params.binds(), &["Person".to_string(), "%ali'ce%".to_string()]);
    }

    #[test]
    fn build_node_filter_where_no_scope_is_true_base() {
        let filter = NodeListFilter::default();
        let mut params = SqlParams::new();
        assert_eq!(build_node_filter_where("n", &filter, &mut params), "TRUE");
        assert!(params.binds().is_empty());
    }

    #[test]
    fn build_edge_filter_where_relationship_type() {
        let filter = EdgeListFilter {
            tenant_id: Some(uid(1).to_string()),
            workspace_id: Some(uid(2).to_string()),
            relationship_type: Some("KNOWS".to_string()),
        };
        let mut params = SqlParams::new();
        let sql = build_edge_filter_where("e", &filter, &mut params);
        assert!(sql.contains("e.tenant_id = "));
        assert!(sql.contains("LOWER(e.properties->>'relation_type') = LOWER($1)"));
        assert_eq!(params.binds(), &["KNOWS".to_string()]);
    }

    #[test]
    fn build_source_prefix_where_covers_string_and_array() {
        let mut params = SqlParams::new();
        let sql = build_source_prefix_where("v", &["doc-abc".to_string()], &mut params);
        assert!(sql.contains("v.properties->>'source_id' LIKE $1 ESCAPE '\\'"));
        assert!(sql.contains("jsonb_array_elements_text"));
        assert!(sql.contains("jsonb_typeof"));
        assert_eq!(params.binds()[0], "doc-abc%");
        let mut empty = SqlParams::new();
        assert_eq!(build_source_prefix_where("v", &[], &mut empty), "FALSE");
    }

    #[test]
    fn json_to_props_drops_non_objects() {
        assert!(json_to_props(serde_json::json!("x")).is_empty());
        let props = json_to_props(serde_json::json!({"a": 1}));
        assert_eq!(props.get("a"), Some(&serde_json::json!(1)));
    }
}
