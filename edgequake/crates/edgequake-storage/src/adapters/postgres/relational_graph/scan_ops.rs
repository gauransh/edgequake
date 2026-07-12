//! Bounded scan / paged listing operations for the relational graph store.

use async_trait::async_trait;
use sqlx::Row;

use super::{
    build_edge_filter_where, build_node_filter_where, build_source_prefix_where, row_to_edge,
    row_to_node, SqlParams, EDGES_TABLE, NODES_TABLE,
};
use super::PostgresRelationalGraphStorage;
use crate::error::{Result, StorageError};
use crate::traits::{
    sources_match_prefixes, EdgeListFilter, GraphEdge, GraphNode, GraphScanOps, NodeListFilter,
    PagedGraphResult,
};

fn db_err(context: &str, e: sqlx::Error) -> StorageError {
    StorageError::Database(format!("{context}: {e}"))
}

#[async_trait]
impl GraphScanOps for PostgresRelationalGraphStorage {
    async fn list_nodes_filtered(
        &self,
        filter: &NodeListFilter,
        offset: usize,
        limit: usize,
    ) -> Result<PagedGraphResult<GraphNode>> {
        let mut params = SqlParams::new();
        let where_clause = build_node_filter_where("n", filter, &mut params);
        let pool = self.pool.get().await?;

        let count_sql =
            format!("SELECT COUNT(*)::BIGINT FROM {NODES_TABLE} n WHERE {where_clause}");
        let mut count_q = sqlx::query_scalar::<_, i64>(&count_sql);
        for b in params.binds() {
            count_q = count_q.bind(b);
        }
        let total: i64 = count_q
            .fetch_one(&pool)
            .await
            .map_err(|e| db_err("list_nodes_filtered (count)", e))?;

        let page_sql = format!(
            "SELECT node_id, properties FROM {NODES_TABLE} n \
             WHERE {where_clause} ORDER BY n.node_id ASC OFFSET {offset} LIMIT {limit}"
        );
        let mut page_q = sqlx::query(&page_sql);
        for b in params.binds() {
            page_q = page_q.bind(b);
        }
        let rows = page_q
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("list_nodes_filtered (page)", e))?;

        let items = rows
            .into_iter()
            .map(|r| row_to_node(r.get("node_id"), r.get("properties")))
            .collect();

        Ok(PagedGraphResult {
            items,
            total: total as usize,
            offset,
            limit,
        })
    }

    async fn list_edges_filtered(
        &self,
        filter: &EdgeListFilter,
        offset: usize,
        limit: usize,
    ) -> Result<PagedGraphResult<GraphEdge>> {
        let mut params = SqlParams::new();
        let where_clause = build_edge_filter_where("e", filter, &mut params);
        let pool = self.pool.get().await?;

        let count_sql =
            format!("SELECT COUNT(*)::BIGINT FROM {EDGES_TABLE} e WHERE {where_clause}");
        let mut count_q = sqlx::query_scalar::<_, i64>(&count_sql);
        for b in params.binds() {
            count_q = count_q.bind(b);
        }
        let total: i64 = count_q
            .fetch_one(&pool)
            .await
            .map_err(|e| db_err("list_edges_filtered (count)", e))?;

        // Order by the two PK text columns (not a '_'-joined concat, which
        // collides for ids containing '_' and yields an unstable page order).
        let page_sql = format!(
            "SELECT source_id, target_id, properties FROM {EDGES_TABLE} e \
             WHERE {where_clause} \
             ORDER BY e.source_id ASC, e.target_id ASC OFFSET {offset} LIMIT {limit}"
        );
        let mut page_q = sqlx::query(&page_sql);
        for b in params.binds() {
            page_q = page_q.bind(b);
        }
        let rows = page_q
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("list_edges_filtered (page)", e))?;

        let items = rows
            .into_iter()
            .map(|r| row_to_edge(r.get("source_id"), r.get("target_id"), r.get("properties")))
            .collect();

        Ok(PagedGraphResult {
            items,
            total: total as usize,
            offset,
            limit,
        })
    }

    async fn find_nodes_by_source_prefixes(
        &self,
        filter: &NodeListFilter,
        source_prefixes: &[String],
    ) -> Result<Vec<GraphNode>> {
        if source_prefixes.is_empty() {
            return Ok(Vec::new());
        }
        let mut params = SqlParams::new();
        let where_clause = build_node_filter_where("n", filter, &mut params);
        if where_clause == "FALSE" {
            return Ok(Vec::new());
        }
        let prefix_where = build_source_prefix_where("n", source_prefixes, &mut params);
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT node_id, properties FROM {NODES_TABLE} n \
             WHERE {where_clause} AND ({prefix_where}) ORDER BY n.node_id ASC"
        );
        let mut q = sqlx::query(&sql);
        for b in params.binds() {
            q = q.bind(b);
        }
        let rows = q
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("find_nodes_by_source_prefixes", e))?;

        // SQL narrows; confirm exact membership with the shared SSOT matcher.
        Ok(rows
            .into_iter()
            .map(|r| row_to_node(r.get("node_id"), r.get("properties")))
            .filter(|n| sources_match_prefixes(&n.properties, source_prefixes))
            .collect())
    }

    async fn find_edges_by_source_prefixes(
        &self,
        filter: &EdgeListFilter,
        source_prefixes: &[String],
    ) -> Result<Vec<GraphEdge>> {
        if source_prefixes.is_empty() {
            return Ok(Vec::new());
        }
        let mut params = SqlParams::new();
        let where_clause = build_edge_filter_where("e", filter, &mut params);
        if where_clause == "FALSE" {
            return Ok(Vec::new());
        }
        let prefix_where = build_source_prefix_where("e", source_prefixes, &mut params);
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT source_id, target_id, properties FROM {EDGES_TABLE} e \
             WHERE {where_clause} AND ({prefix_where}) \
             ORDER BY e.source_id ASC, e.target_id ASC"
        );
        let mut q = sqlx::query(&sql);
        for b in params.binds() {
            q = q.bind(b);
        }
        let rows = q
            .fetch_all(&pool)
            .await
            .map_err(|e| db_err("find_edges_by_source_prefixes", e))?;

        Ok(rows
            .into_iter()
            .map(|r| row_to_edge(r.get("source_id"), r.get("target_id"), r.get("properties")))
            .filter(|e| sources_match_prefixes(&e.properties, source_prefixes))
            .collect())
    }

    async fn find_edge_by_relationship_id(
        &self,
        filter: &EdgeListFilter,
        relationship_id: &str,
    ) -> Result<Option<GraphEdge>> {
        if relationship_id.is_empty() {
            return Ok(None);
        }
        let mut params = SqlParams::new();
        let where_clause = build_edge_filter_where("e", filter, &mut params);
        if where_clause == "FALSE" {
            return Ok(None);
        }
        let rid = params.bind(relationship_id.to_string());
        let pool = self.pool.get().await?;
        let sql = format!(
            "SELECT source_id, target_id, properties FROM {EDGES_TABLE} e \
             WHERE {where_clause} \
               AND (e.relationship_id = {rid} \
                    OR (e.source_id || '_' || e.target_id) = {rid}) \
             LIMIT 1"
        );
        let mut q = sqlx::query(&sql);
        for b in params.binds() {
            q = q.bind(b);
        }
        let row = q
            .fetch_optional(&pool)
            .await
            .map_err(|e| db_err("find_edge_by_relationship_id", e))?;

        Ok(row.map(|r| row_to_edge(r.get("source_id"), r.get("target_id"), r.get("properties"))))
    }
}
