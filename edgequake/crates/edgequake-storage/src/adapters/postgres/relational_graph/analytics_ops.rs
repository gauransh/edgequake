//! Analytics / count operations for the relational graph store.

use async_trait::async_trait;

use super::{EDGES_TABLE, NODES_TABLE};
use super::PostgresRelationalGraphStorage;
use crate::error::{Result, StorageError};
use crate::traits::GraphStorageAnalyticsOps;

fn db_err(context: &str, e: sqlx::Error) -> StorageError {
    StorageError::Database(format!("{context}: {e}"))
}

impl PostgresRelationalGraphStorage {
    async fn count_scalar(&self, sql: &str) -> Result<usize> {
        let pool = self.pool.get().await?;
        let count: i64 = sqlx::query_scalar(sql)
            .fetch_one(&pool)
            .await
            .map_err(|e| db_err("count", e))?;
        Ok(count as usize)
    }

    async fn count_scalar_by_workspace(
        &self,
        sql: &str,
        workspace_id: &uuid::Uuid,
    ) -> Result<usize> {
        let pool = self.pool.get().await?;
        let count: i64 = sqlx::query_scalar(sql)
            .bind(workspace_id)
            .fetch_one(&pool)
            .await
            .map_err(|e| db_err("count_by_workspace", e))?;
        Ok(count as usize)
    }
}

#[async_trait]
impl GraphStorageAnalyticsOps for PostgresRelationalGraphStorage {
    async fn node_count(&self) -> Result<usize> {
        self.count_scalar(&format!("SELECT COUNT(*)::BIGINT FROM {NODES_TABLE}"))
            .await
    }

    async fn edge_count(&self) -> Result<usize> {
        self.count_scalar(&format!("SELECT COUNT(*)::BIGINT FROM {EDGES_TABLE}"))
            .await
    }

    async fn node_count_by_workspace(&self, workspace_id: &uuid::Uuid) -> Result<usize> {
        self.count_scalar_by_workspace(
            &format!("SELECT COUNT(*)::BIGINT FROM {NODES_TABLE} WHERE workspace_id = $1"),
            workspace_id,
        )
        .await
    }

    async fn edge_count_by_workspace(&self, workspace_id: &uuid::Uuid) -> Result<usize> {
        self.count_scalar_by_workspace(
            &format!("SELECT COUNT(*)::BIGINT FROM {EDGES_TABLE} WHERE workspace_id = $1"),
            workspace_id,
        )
        .await
    }

    async fn distinct_node_type_count_by_workspace(
        &self,
        workspace_id: &uuid::Uuid,
    ) -> Result<usize> {
        // COUNT(DISTINCT ...) ignores NULLs, so nodes without entity_type are
        // excluded — matching MemoryGraphStorage which filter_maps them out.
        self.count_scalar_by_workspace(
            &format!(
                "SELECT COUNT(DISTINCT properties->>'entity_type')::BIGINT \
                 FROM {NODES_TABLE} WHERE workspace_id = $1"
            ),
            workspace_id,
        )
        .await
    }
}
