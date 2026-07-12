//! Trait-conformance tests for the relational graph store.
//!
//! These require a live PostgreSQL and are **skipped** (no failure) when
//! `POSTGRES_PASSWORD` is unset — mirroring the gating used by the crate's
//! `tests/postgres_*` integration suites. Behaviour is asserted against the same
//! expectations as the `MemoryGraphStorage` unit tests.
//!
//! Run with: `POSTGRES_PASSWORD=... cargo test -p edgequake-storage --lib --features postgres`

use std::collections::HashMap;
use std::time::Duration;

use sqlx::postgres::PgPoolOptions;
use uuid::Uuid;

use super::PostgresRelationalGraphStorage;
use crate::adapters::postgres::{PostgresConfig, PostgresPool};
use crate::traits::{
    GraphStorage, GraphStorageAnalyticsOps, GraphStorageMutateOps, GraphStorageReadOps,
    GraphScanOps, NodeListFilter,
};

fn test_config() -> Option<PostgresConfig> {
    let password = std::env::var("POSTGRES_PASSWORD").ok()?;
    Some(PostgresConfig {
        host: std::env::var("POSTGRES_HOST").unwrap_or_else(|_| "localhost".to_string()),
        port: std::env::var("POSTGRES_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5432),
        database: std::env::var("POSTGRES_DB").unwrap_or_else(|_| "edgequake".to_string()),
        user: std::env::var("POSTGRES_USER").unwrap_or_else(|_| "edgequake".to_string()),
        password,
        namespace: "relgraph_test".to_string(),
        max_connections: 4,
        min_connections: 1,
        connect_timeout: Duration::from_secs(10),
        idle_timeout: Duration::from_secs(60),
        ..Default::default()
    })
}

/// Build storage over a raw pool (no extension setup) and ensure the tables
/// exist. Returns `None` when no database is configured (test should skip).
async fn setup() -> Option<PostgresRelationalGraphStorage> {
    let config = test_config()?;
    let url = format!(
        "postgres://{}:{}@{}:{}/{}",
        config.user, config.password, config.host, config.port, config.database
    );
    let pool = PgPoolOptions::new()
        .max_connections(config.max_connections)
        .acquire_timeout(config.connect_timeout)
        .connect(&url)
        .await
        .ok()?;

    for ddl in [
        "CREATE TABLE IF NOT EXISTS eq_graph_nodes (\
            tenant_id UUID NOT NULL, workspace_id UUID NOT NULL, node_id TEXT NOT NULL, \
            properties JSONB NOT NULL DEFAULT '{}', \
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), \
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), \
            PRIMARY KEY (tenant_id, workspace_id, node_id))",
        "CREATE TABLE IF NOT EXISTS eq_graph_edges (\
            tenant_id UUID NOT NULL, workspace_id UUID NOT NULL, \
            source_id TEXT NOT NULL, target_id TEXT NOT NULL, relationship_id TEXT, \
            properties JSONB NOT NULL DEFAULT '{}', weight REAL, \
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(), \
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), \
            PRIMARY KEY (tenant_id, workspace_id, source_id, target_id))",
    ] {
        sqlx::query(ddl).execute(&pool).await.ok()?;
    }

    Some(PostgresRelationalGraphStorage::with_pool(
        PostgresPool::from_existing(pool, config.clone()),
        config,
    ))
}

/// Unique run scope so parallel tests never collide on the shared tables.
fn scoped_props(run: &str) -> (Uuid, Uuid, HashMap<String, serde_json::Value>) {
    let tenant = Uuid::new_v4();
    let workspace = Uuid::new_v4();
    let mut props = HashMap::new();
    props.insert("tenant_id".to_string(), serde_json::json!(tenant.to_string()));
    props.insert(
        "workspace_id".to_string(),
        serde_json::json!(workspace.to_string()),
    );
    props.insert("run".to_string(), serde_json::json!(run));
    (tenant, workspace, props)
}

macro_rules! skip_if_no_db {
    ($storage:ident) => {
        let Some($storage) = setup().await else {
            eprintln!("SKIP: POSTGRES_PASSWORD unset — relational graph conformance tests skipped");
            return;
        };
    };
}

#[tokio::test]
async fn node_upsert_get_delete_roundtrip() {
    skip_if_no_db!(storage);
    let (tenant, workspace, mut props) = scoped_props("node_rt");
    props.insert("name".to_string(), serde_json::json!("Alice"));
    let nid = format!("alice-{workspace}");

    storage.upsert_node(&nid, props.clone()).await.unwrap();
    assert!(storage.has_node(&nid).await.unwrap());

    let fetched = storage.get_node(&nid).await.unwrap().unwrap();
    assert_eq!(fetched.id, nid);
    assert_eq!(fetched.properties.get("name"), Some(&serde_json::json!("Alice")));

    assert_eq!(storage.node_count_by_workspace(&workspace).await.unwrap(), 1);

    storage
        .delete_node_scoped(&nid, &tenant.to_string(), &workspace.to_string())
        .await
        .unwrap();
    assert!(!storage.has_node(&nid).await.unwrap());

    storage.clear_workspace(&workspace).await.unwrap();
}

#[tokio::test]
async fn edge_is_symmetric_and_degree_counts() {
    skip_if_no_db!(storage);
    let (_t, workspace, props) = scoped_props("edge_sym");
    let a = format!("a-{workspace}");
    let b = format!("b-{workspace}");
    storage.upsert_node(&a, props.clone()).await.unwrap();
    storage.upsert_node(&b, props.clone()).await.unwrap();

    let mut edge_props = props.clone();
    edge_props.insert("relation".to_string(), serde_json::json!("knows"));
    storage.upsert_edge(&a, &b, edge_props).await.unwrap();

    assert!(storage.has_edge(&a, &b).await.unwrap());
    assert!(storage.has_edge(&b, &a).await.unwrap());
    assert_eq!(storage.node_degree(&a).await.unwrap(), 1);

    let degrees = storage
        .node_degrees_batch(&[a.clone(), b.clone()])
        .await
        .unwrap();
    assert_eq!(degrees.len(), 2);
    assert!(degrees.iter().all(|(_, d)| *d == 1));

    let incident = storage.get_incident_edges_batch(&[a.clone()]).await.unwrap();
    assert_eq!(incident.len(), 1);

    storage.clear_workspace(&workspace).await.unwrap();
}

#[tokio::test]
async fn traversal_scoped_by_tenant_workspace() {
    skip_if_no_db!(storage);
    let (tenant, workspace, props) = scoped_props("traverse");
    let a = format!("A-{workspace}");
    let b = format!("B-{workspace}");
    let c = format!("C-{workspace}");

    // A, B in scope; C in a different tenant/workspace.
    storage.upsert_node(&a, props.clone()).await.unwrap();
    storage.upsert_node(&b, props.clone()).await.unwrap();
    let mut other = HashMap::new();
    other.insert(
        "tenant_id".to_string(),
        serde_json::json!(Uuid::new_v4().to_string()),
    );
    other.insert(
        "workspace_id".to_string(),
        serde_json::json!(Uuid::new_v4().to_string()),
    );
    storage.upsert_node(&c, other.clone()).await.unwrap();

    storage.upsert_edge(&a, &b, props.clone()).await.unwrap();
    storage.upsert_edge(&b, &c, props.clone()).await.unwrap();

    let scoped = storage
        .get_knowledge_graph(&a, 2, 10, Some(&tenant.to_string()), Some(&workspace.to_string()))
        .await
        .unwrap();
    assert_eq!(scoped.node_count(), 2);
    assert!(scoped.nodes.iter().all(|n| n.id == a || n.id == b));

    storage.clear_workspace(&workspace).await.unwrap();
}

#[tokio::test]
async fn batch_upsert_and_list_filter() {
    skip_if_no_db!(storage);
    let (tenant, workspace, props) = scoped_props("batch");
    let n1 = format!("n1-{workspace}");
    let n2 = format!("n2-{workspace}");
    let mut p1 = props.clone();
    p1.insert("entity_type".to_string(), serde_json::json!("Person"));
    let mut p2 = props.clone();
    p2.insert("entity_type".to_string(), serde_json::json!("Org"));

    storage
        .upsert_nodes_batch(&[(n1.clone(), p1), (n2.clone(), p2)])
        .await
        .unwrap();
    assert_eq!(storage.node_count_by_workspace(&workspace).await.unwrap(), 2);
    assert_eq!(
        storage
            .distinct_node_type_count_by_workspace(&workspace)
            .await
            .unwrap(),
        2
    );

    let filter = NodeListFilter {
        tenant_id: Some(tenant.to_string()),
        workspace_id: Some(workspace.to_string()),
        entity_type: Some("person".to_string()), // case-insensitive
        ..Default::default()
    };
    let page = storage.list_nodes_filtered(&filter, 0, 10).await.unwrap();
    assert_eq!(page.total, 1);
    assert_eq!(page.items[0].id, n1);

    storage.clear_workspace(&workspace).await.unwrap();
}

#[tokio::test]
async fn delete_node_cascades_edges() {
    skip_if_no_db!(storage);
    let (_t, workspace, props) = scoped_props("cascade");
    let a = format!("A-{workspace}");
    let b = format!("B-{workspace}");
    storage.upsert_node(&a, props.clone()).await.unwrap();
    storage.upsert_node(&b, props.clone()).await.unwrap();
    storage.upsert_edge(&a, &b, props.clone()).await.unwrap();

    assert_eq!(storage.edge_count_by_workspace(&workspace).await.unwrap(), 1);
    storage.delete_node(&a).await.unwrap();
    assert!(!storage.has_node(&a).await.unwrap());
    assert_eq!(storage.edge_count_by_workspace(&workspace).await.unwrap(), 0);

    storage.clear_workspace(&workspace).await.unwrap();
}

#[tokio::test]
async fn initialize_health_check() {
    skip_if_no_db!(storage);
    storage.initialize().await.unwrap();
    storage.finalize().await.unwrap();
    assert_eq!(storage.namespace(), "relgraph_test");
}
