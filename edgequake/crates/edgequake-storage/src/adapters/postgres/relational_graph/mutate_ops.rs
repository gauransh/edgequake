//! Mutation operations for the relational graph store.

use std::collections::HashMap;

use async_trait::async_trait;
use uuid::Uuid;

use super::{canonical_edge, scope_from_props, EDGES_TABLE, NODES_TABLE};
use super::PostgresRelationalGraphStorage;
use crate::error::{Result, StorageError};
use crate::traits::GraphStorageMutateOps;

fn db_err(context: &str, e: sqlx::Error) -> StorageError {
    StorageError::Database(format!("{context}: {e}"))
}

fn props_to_json(properties: HashMap<String, serde_json::Value>) -> serde_json::Value {
    serde_json::Value::Object(properties.into_iter().collect())
}

fn extract_relationship_id(properties: &HashMap<String, serde_json::Value>) -> Option<String> {
    properties
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn extract_weight(properties: &HashMap<String, serde_json::Value>) -> Option<f32> {
    properties.get("weight").and_then(|v| v.as_f64()).map(|w| w as f32)
}

#[async_trait]
impl GraphStorageMutateOps for PostgresRelationalGraphStorage {
    async fn upsert_node(
        &self,
        node_id: &str,
        properties: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        let (tenant_id, workspace_id) = scope_from_props(&properties);
        let props = props_to_json(properties);
        let pool = self.pool.get().await?;
        let sql = format!(
            "INSERT INTO {NODES_TABLE} (tenant_id, workspace_id, node_id, properties, updated_at) \
             VALUES ($1, $2, $3, $4, now()) \
             ON CONFLICT (tenant_id, workspace_id, node_id) \
             DO UPDATE SET properties = EXCLUDED.properties, updated_at = now()"
        );
        sqlx::query(&sql)
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(node_id)
            .bind(props)
            .execute(&pool)
            .await
            .map_err(|e| db_err("upsert_node", e))?;
        Ok(())
    }

    async fn upsert_nodes_batch(
        &self,
        nodes: &[(String, HashMap<String, serde_json::Value>)],
    ) -> Result<()> {
        if nodes.is_empty() {
            return Ok(());
        }
        // Dedup on the primary key keeping the last write — Postgres rejects a
        // second ON CONFLICT hit on the same row within one statement.
        let mut latest: HashMap<(Uuid, Uuid, String), serde_json::Value> = HashMap::new();
        let mut order: Vec<(Uuid, Uuid, String)> = Vec::new();
        for (node_id, properties) in nodes {
            let (tenant_id, workspace_id) = scope_from_props(properties);
            let key = (tenant_id, workspace_id, node_id.clone());
            if !latest.contains_key(&key) {
                order.push(key.clone());
            }
            latest.insert(key, props_to_json(properties.clone()));
        }

        let mut tenants = Vec::with_capacity(order.len());
        let mut workspaces = Vec::with_capacity(order.len());
        let mut node_ids = Vec::with_capacity(order.len());
        let mut props_json = Vec::with_capacity(order.len());
        for key in &order {
            tenants.push(key.0);
            workspaces.push(key.1);
            node_ids.push(key.2.clone());
            let value = latest.get(key).cloned().unwrap_or(serde_json::json!({}));
            let json = serde_json::to_string(&value).map_err(|e| {
                StorageError::Serialization(format!("upsert_nodes_batch node properties: {e}"))
            })?;
            props_json.push(json);
        }

        let pool = self.pool.get().await?;
        let sql = format!(
            "INSERT INTO {NODES_TABLE} (tenant_id, workspace_id, node_id, properties, updated_at) \
             SELECT t.tid, t.wid, t.nid, t.props::jsonb, now() \
             FROM unnest($1::uuid[], $2::uuid[], $3::text[], $4::text[]) \
                    AS t(tid, wid, nid, props) \
             ON CONFLICT (tenant_id, workspace_id, node_id) \
             DO UPDATE SET properties = EXCLUDED.properties, updated_at = now()"
        );
        sqlx::query(&sql)
            .bind(&tenants)
            .bind(&workspaces)
            .bind(&node_ids)
            .bind(&props_json)
            .execute(&pool)
            .await
            .map_err(|e| db_err("upsert_nodes_batch", e))?;
        Ok(())
    }

    async fn delete_node(&self, node_id: &str) -> Result<()> {
        let pool = self.pool.get().await?;
        let edge_sql =
            format!("DELETE FROM {EDGES_TABLE} WHERE source_id = $1 OR target_id = $1");
        sqlx::query(&edge_sql)
            .bind(node_id)
            .execute(&pool)
            .await
            .map_err(|e| db_err("delete_node (edges)", e))?;
        let node_sql = format!("DELETE FROM {NODES_TABLE} WHERE node_id = $1");
        sqlx::query(&node_sql)
            .bind(node_id)
            .execute(&pool)
            .await
            .map_err(|e| db_err("delete_node", e))?;
        Ok(())
    }

    async fn delete_node_scoped(
        &self,
        node_id: &str,
        tenant_id: &str,
        workspace_id: &str,
    ) -> Result<bool> {
        let (Ok(tid), Ok(wid)) = (Uuid::parse_str(tenant_id), Uuid::parse_str(workspace_id)) else {
            return Ok(false);
        };
        let pool = self.pool.get().await?;
        let edge_sql = format!(
            "DELETE FROM {EDGES_TABLE} \
             WHERE (source_id = $1 OR target_id = $1) AND tenant_id = $2 AND workspace_id = $3"
        );
        sqlx::query(&edge_sql)
            .bind(node_id)
            .bind(tid)
            .bind(wid)
            .execute(&pool)
            .await
            .map_err(|e| db_err("delete_node_scoped (edges)", e))?;
        let node_sql = format!(
            "DELETE FROM {NODES_TABLE} WHERE node_id = $1 AND tenant_id = $2 AND workspace_id = $3"
        );
        let res = sqlx::query(&node_sql)
            .bind(node_id)
            .bind(tid)
            .bind(wid)
            .execute(&pool)
            .await
            .map_err(|e| db_err("delete_node_scoped", e))?;
        Ok(res.rows_affected() > 0)
    }

    async fn upsert_edge(
        &self,
        source: &str,
        target: &str,
        properties: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        let (s, t) = canonical_edge(source, target);
        let (tenant_id, workspace_id) = scope_from_props(&properties);
        let relationship_id = extract_relationship_id(&properties);
        let weight = extract_weight(&properties);
        let props = props_to_json(properties);
        let pool = self.pool.get().await?;
        let sql = format!(
            "INSERT INTO {EDGES_TABLE} \
                 (tenant_id, workspace_id, source_id, target_id, relationship_id, properties, weight, updated_at) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, now()) \
             ON CONFLICT (tenant_id, workspace_id, source_id, target_id) \
             DO UPDATE SET relationship_id = EXCLUDED.relationship_id, \
                           properties = EXCLUDED.properties, \
                           weight = EXCLUDED.weight, \
                           updated_at = now()"
        );
        sqlx::query(&sql)
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(&s)
            .bind(&t)
            .bind(relationship_id)
            .bind(props)
            .bind(weight)
            .execute(&pool)
            .await
            .map_err(|e| db_err("upsert_edge", e))?;
        Ok(())
    }

    async fn upsert_edges_batch(
        &self,
        edges: &[(String, String, HashMap<String, serde_json::Value>)],
    ) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        // Dedup on the canonical primary key keeping the last write.
        #[allow(clippy::type_complexity)]
        let mut latest: HashMap<
            (Uuid, Uuid, String, String),
            (Option<String>, Option<f32>, serde_json::Value),
        > = HashMap::new();
        let mut order: Vec<(Uuid, Uuid, String, String)> = Vec::new();
        for (source, target, properties) in edges {
            let (s, t) = canonical_edge(source, target);
            let (tenant_id, workspace_id) = scope_from_props(properties);
            let key = (tenant_id, workspace_id, s, t);
            if !latest.contains_key(&key) {
                order.push(key.clone());
            }
            latest.insert(
                key,
                (
                    extract_relationship_id(properties),
                    extract_weight(properties),
                    props_to_json(properties.clone()),
                ),
            );
        }

        let mut tenants = Vec::with_capacity(order.len());
        let mut workspaces = Vec::with_capacity(order.len());
        let mut sources = Vec::with_capacity(order.len());
        let mut targets = Vec::with_capacity(order.len());
        let mut rel_ids: Vec<Option<String>> = Vec::with_capacity(order.len());
        let mut props_json = Vec::with_capacity(order.len());
        let mut weights: Vec<Option<f32>> = Vec::with_capacity(order.len());
        for key in &order {
            let (rel, weight, props) = latest.get(key).cloned().unwrap();
            tenants.push(key.0);
            workspaces.push(key.1);
            sources.push(key.2.clone());
            targets.push(key.3.clone());
            rel_ids.push(rel);
            weights.push(weight);
            let json = serde_json::to_string(&props).map_err(|e| {
                StorageError::Serialization(format!("upsert_edges_batch edge properties: {e}"))
            })?;
            props_json.push(json);
        }

        let pool = self.pool.get().await?;
        let sql = format!(
            "INSERT INTO {EDGES_TABLE} \
                 (tenant_id, workspace_id, source_id, target_id, relationship_id, properties, weight, updated_at) \
             SELECT t.tid, t.wid, t.src, t.tgt, t.rel, t.props::jsonb, t.wt, now() \
             FROM unnest($1::uuid[], $2::uuid[], $3::text[], $4::text[], \
                         $5::text[], $6::text[], $7::real[]) \
                    AS t(tid, wid, src, tgt, rel, props, wt) \
             ON CONFLICT (tenant_id, workspace_id, source_id, target_id) \
             DO UPDATE SET relationship_id = EXCLUDED.relationship_id, \
                           properties = EXCLUDED.properties, \
                           weight = EXCLUDED.weight, \
                           updated_at = now()"
        );
        sqlx::query(&sql)
            .bind(&tenants)
            .bind(&workspaces)
            .bind(&sources)
            .bind(&targets)
            .bind(&rel_ids)
            .bind(&props_json)
            .bind(&weights)
            .execute(&pool)
            .await
            .map_err(|e| db_err("upsert_edges_batch", e))?;
        Ok(())
    }

    async fn delete_edge(&self, source: &str, target: &str) -> Result<()> {
        let (s, t) = canonical_edge(source, target);
        let pool = self.pool.get().await?;
        let sql = format!("DELETE FROM {EDGES_TABLE} WHERE source_id = $1 AND target_id = $2");
        sqlx::query(&sql)
            .bind(&s)
            .bind(&t)
            .execute(&pool)
            .await
            .map_err(|e| db_err("delete_edge", e))?;
        Ok(())
    }

    async fn delete_edge_scoped(
        &self,
        source: &str,
        target: &str,
        tenant_id: &str,
        workspace_id: &str,
    ) -> Result<bool> {
        let (Ok(tid), Ok(wid)) = (Uuid::parse_str(tenant_id), Uuid::parse_str(workspace_id)) else {
            return Ok(false);
        };
        let (s, t) = canonical_edge(source, target);
        let pool = self.pool.get().await?;
        let sql = format!(
            "DELETE FROM {EDGES_TABLE} \
             WHERE source_id = $1 AND target_id = $2 AND tenant_id = $3 AND workspace_id = $4"
        );
        let res = sqlx::query(&sql)
            .bind(&s)
            .bind(&t)
            .bind(tid)
            .bind(wid)
            .execute(&pool)
            .await
            .map_err(|e| db_err("delete_edge_scoped", e))?;
        Ok(res.rows_affected() > 0)
    }

    async fn clear(&self) -> Result<()> {
        let pool = self.pool.get().await?;
        sqlx::query(&format!("DELETE FROM {EDGES_TABLE}"))
            .execute(&pool)
            .await
            .map_err(|e| db_err("clear (edges)", e))?;
        sqlx::query(&format!("DELETE FROM {NODES_TABLE}"))
            .execute(&pool)
            .await
            .map_err(|e| db_err("clear (nodes)", e))?;
        Ok(())
    }

    async fn clear_workspace(&self, workspace_id: &uuid::Uuid) -> Result<(usize, usize)> {
        let pool = self.pool.get().await?;
        let edges = sqlx::query(&format!("DELETE FROM {EDGES_TABLE} WHERE workspace_id = $1"))
            .bind(workspace_id)
            .execute(&pool)
            .await
            .map_err(|e| db_err("clear_workspace (edges)", e))?;
        let nodes = sqlx::query(&format!("DELETE FROM {NODES_TABLE} WHERE workspace_id = $1"))
            .bind(workspace_id)
            .execute(&pool)
            .await
            .map_err(|e| db_err("clear_workspace (nodes)", e))?;
        Ok((nodes.rows_affected() as usize, edges.rows_affected() as usize))
    }
}
