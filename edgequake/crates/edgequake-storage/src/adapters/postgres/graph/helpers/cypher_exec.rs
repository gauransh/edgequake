//! Cypher execution helpers (SPEC-017 P1-12, SPEC-022 P-H7 parameterized Cypher).
//!
//! ## AGE parameter binding
//!
//! AGE's `cypher()` function only accepts a bare `$N` prepared-statement
//! placeholder as its third (parameter-map) argument. Any expression there —
//! including an inline `'<json>'::agtype` cast — is rejected at parse time with
//! "third argument of cypher function must be a parameter" (itsm-graph-core#11,
//! #12). Rather than route a typed agtype bind through sqlx, the bound helpers
//! reuse the same proven pattern as the ingestion write path: `$param`
//! placeholders are materialized directly into the Cypher text by
//! `inline_cypher_params`, so `cypher()` is always issued in its supported
//! two-argument form. Every value flows through `value_to_cypher`, which
//! single-quote-escapes strings, keeping the inlined ids/properties
//! injection-safe.

use sqlx::Row;

use crate::error::{Result, StorageError};

use super::super::PostgresAGEGraphStorage;

impl PostgresAGEGraphStorage {
    /// Run read Cypher with parameters inlined as safe agtype literals.
    ///
    /// `$param` placeholders are substituted directly into the Cypher text by
    /// [`Self::inline_cypher_params`] so `cypher()` is issued in its supported
    /// two-argument form — AGE rejects any non-`$N` third argument. Values are
    /// single-quote-escaped by `value_to_cypher`, identical to the ingestion
    /// write path, so this stays injection-safe.
    pub(in crate::adapters::postgres::graph) async fn cypher_query_bound(
        &self,
        cypher: &str,
        columns: &[&str],
        params: &serde_json::Value,
    ) -> Result<Vec<sqlx::postgres::PgRow>> {
        let inlined = Self::inline_cypher_params(cypher, params);
        self.cypher_query(&inlined, columns).await
    }

    /// Run write Cypher with parameters inlined as safe agtype literals.
    ///
    /// See [`Self::cypher_query_bound`] for why parameters are inlined rather
    /// than passed as a `cypher()` third argument.
    pub(in crate::adapters::postgres::graph) async fn cypher_execute_bound(
        &self,
        cypher: &str,
        params: &serde_json::Value,
    ) -> Result<()> {
        let inlined = Self::inline_cypher_params(cypher, params);
        self.cypher_execute(&inlined).await
    }

    // ── non-parameterized helpers (unchanged) ───────────────────────────────

    pub(in crate::adapters::postgres::graph) async fn cypher_query(
        &self,
        cypher: &str,
        columns: &[&str],
    ) -> Result<Vec<sqlx::postgres::PgRow>> {
        let pool = self.pool.get().await?;
        let mut conn = pool.acquire().await.map_err(|e| {
            StorageError::Connection(format!("Failed to acquire connection: {}", e))
        })?;

        Self::setup_age_session_scoped(&mut conn, None).await?;

        let as_clause = columns
            .iter()
            .map(|c| format!("{c} agtype"))
            .collect::<Vec<_>>()
            .join(", ");
        let select_clause = columns
            .iter()
            .map(|c| format!("agtype_to_json({c}) as {c}"))
            .collect::<Vec<_>>()
            .join(", ");

        let tag = Self::dollar_quote_tag(cypher);
        let sql = format!(
            "SELECT {select_clause} FROM cypher('{graph}', {tag} {cypher} {tag}) AS ({as_clause})",
            graph = self.graph_name,
        );

        let rows = sqlx::query(&sql)
            .fetch_all(&mut *conn)
            .await
            .map_err(|e| StorageError::Database(format!("Cypher query failed: {e}")))?;

        Ok(rows)
    }

    pub(in crate::adapters::postgres::graph) async fn cypher_execute(
        &self,
        cypher: &str,
    ) -> Result<()> {
        let pool = self.pool.get().await?;
        let tag = Self::dollar_quote_tag(cypher);
        let sql = format!(
            "{} SELECT * FROM cypher('{}', {} {} {}) AS (a agtype);",
            Self::age_session_setup_sql(),
            self.graph_name,
            tag,
            cypher,
            tag
        );

        sqlx::raw_sql(&sql)
            .execute(&pool)
            .await
            .map_err(|e| StorageError::Database(format!("Cypher execute failed: {e}")))?;

        Ok(())
    }

    pub(in crate::adapters::postgres::graph) async fn cypher_query_count(
        &self,
        cypher: &str,
    ) -> Result<i64> {
        let pool = self.pool.get().await?;
        let mut conn = pool.acquire().await.map_err(|e| {
            StorageError::Connection(format!("Failed to acquire connection: {}", e))
        })?;

        Self::setup_age_session_scoped(&mut conn, None).await?;

        let tag = Self::dollar_quote_tag(cypher);
        let sql = format!(
            "SELECT agtype_to_int8(count) FROM cypher('{}', {} {} {}) AS (count agtype)",
            self.graph_name, tag, cypher, tag
        );

        let row = sqlx::query(&sql)
            .fetch_optional(&mut *conn)
            .await
            .map_err(|e| StorageError::Database(format!("Cypher count query failed: {e}")))?;

        Ok(row.map(|r| r.get::<i64, _>(0)).unwrap_or(0))
    }

    pub(in crate::adapters::postgres::graph) async fn batch_sql_query(
        &self,
        sql: &str,
        ids: &[String],
    ) -> Result<Vec<sqlx::postgres::PgRow>> {
        let pool = self.pool.get().await?;
        let mut conn = pool.acquire().await.map_err(|e| {
            StorageError::Connection(format!("Failed to acquire connection: {}", e))
        })?;

        Self::setup_age_session_scoped(&mut conn, None).await?;

        let rows = sqlx::query(sql)
            .bind(ids)
            .fetch_all(&mut *conn)
            .await
            .map_err(|e| StorageError::Database(format!("Batch query failed: {e}")))?;

        Ok(rows)
    }
}
