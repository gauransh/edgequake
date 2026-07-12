-- ============================================================================
-- Migration 082: Plain-relational knowledge-graph tables (no Apache AGE)
-- Version: 1.0.0
-- Date: 2026-07-11
-- Author: EdgeQuake Team
--
-- PURPOSE:
--   Provide a plain-relational Postgres backing store for the GraphStorage
--   trait as an alternative to the Apache AGE property graph. Nodes and edges
--   live in two ordinary tables scoped by (tenant_id, workspace_id) columns —
--   no ag_catalog, no Cypher, no graphid encoding.
--
--   Backs `PostgresRelationalGraphStorage` in edgequake-storage. The AGE store
--   is untouched; this migration is purely additive.
--
-- MODEL:
--   * Undirected edges are stored canonically (source_id <= target_id) so a
--     single row represents one relationship regardless of insertion order —
--     mirrors the MemoryGraphStorage edge-key normalization.
--   * `properties` keeps the full JSONB payload (including the tenant_id /
--     workspace_id strings) so reads round-trip exactly what was written; the
--     dedicated UUID columns exist for cheap tenant-scoped filtering / indexing.
--
-- ASCENDING COMPATIBILITY:
--   * All DDL is additive and IDEMPOTENT (IF NOT EXISTS). Safe to re-run.
--   * No AGE dependency — applies on databases without the age extension.
--
-- ROLLBACK:
--   DROP POLICY IF EXISTS eq_graph_edges_tenant_isolation ON eq_graph_edges;
--   DROP POLICY IF EXISTS eq_graph_nodes_tenant_isolation ON eq_graph_nodes;
--   DROP INDEX IF EXISTS idx_eq_graph_edges_target;
--   DROP INDEX IF EXISTS idx_eq_graph_edges_relationship_id;
--   DROP INDEX IF EXISTS idx_eq_graph_nodes_entity_type;
--   DROP TABLE IF EXISTS eq_graph_edges;
--   DROP TABLE IF EXISTS eq_graph_nodes;
-- ============================================================================

SET search_path = public;

-- ============================================================================
-- STEP 1: Nodes table
-- PRIMARY KEY (tenant_id, workspace_id, node_id) enforces one node per id per
-- workspace and gives an O(log N) point lookup + prefix scan for the hot
-- get_nodes_batch / get_incident_edges_batch paths.
-- ============================================================================
CREATE TABLE IF NOT EXISTS eq_graph_nodes (
    tenant_id    UUID        NOT NULL,
    workspace_id UUID        NOT NULL,
    node_id      TEXT        NOT NULL,
    properties   JSONB       NOT NULL DEFAULT '{}',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, workspace_id, node_id)
);

-- ============================================================================
-- STEP 2: Edges table
-- PRIMARY KEY (tenant_id, workspace_id, source_id, target_id) gives the forward
-- adjacency scan for free; the target index below covers reverse traversal.
-- ============================================================================
CREATE TABLE IF NOT EXISTS eq_graph_edges (
    tenant_id       UUID        NOT NULL,
    workspace_id    UUID        NOT NULL,
    source_id       TEXT        NOT NULL,
    target_id       TEXT        NOT NULL,
    relationship_id TEXT,
    properties      JSONB       NOT NULL DEFAULT '{}',
    weight          REAL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, workspace_id, source_id, target_id)
);

-- ============================================================================
-- STEP 3: Indexes
-- ============================================================================

-- Reverse adjacency: the BFS in edges_within_depth matches edges by EITHER
-- endpoint. The PK covers (source_id); this covers (target_id).
CREATE INDEX IF NOT EXISTS idx_eq_graph_edges_target
    ON eq_graph_edges (tenant_id, workspace_id, target_id);

-- O(1) lookup by API relationship id (find_edge_by_relationship_id). Partial:
-- most edges have no explicit relationship_id, so keep the index small.
CREATE INDEX IF NOT EXISTS idx_eq_graph_edges_relationship_id
    ON eq_graph_edges (tenant_id, workspace_id, relationship_id)
    WHERE relationship_id IS NOT NULL;

-- entity_type filter (search_nodes, list_nodes_filtered) and the distinct
-- entity-type count. A btree expression index on the extracted text is cheaper
-- and more selective here than a GIN over the whole properties document.
CREATE INDEX IF NOT EXISTS idx_eq_graph_nodes_entity_type
    ON eq_graph_nodes (tenant_id, workspace_id, (properties->>'entity_type'));

COMMENT ON TABLE eq_graph_nodes IS 'Plain-relational knowledge-graph nodes (non-AGE GraphStorage backend). Scoped by (tenant_id, workspace_id). properties JSONB round-trips the full node payload.';
COMMENT ON TABLE eq_graph_edges IS 'Plain-relational knowledge-graph edges (non-AGE). Undirected, stored canonically with source_id <= target_id. Scoped by (tenant_id, workspace_id).';

-- ============================================================================
-- STEP 4: Row-Level Security (defense-in-depth tenant isolation)
--
-- Mirrors the graph_nodes / graph_edges policies from migration 009 so a
-- connection scoped via set_tenant_context() (current_tenant_id() /
-- current_workspace_id()) can only see its own rows. ENABLE (not FORCE): the
-- table owner still bypasses RLS, matching the rest of the schema, so the
-- adapter's owner-role pool connections keep working while a restricted role is
-- isolated. These functions are created by migration 009 (runs before 082).
-- ============================================================================
ALTER TABLE eq_graph_nodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE eq_graph_edges ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS eq_graph_nodes_tenant_isolation ON eq_graph_nodes;
CREATE POLICY eq_graph_nodes_tenant_isolation ON eq_graph_nodes
    FOR ALL
    USING (
        tenant_id = current_tenant_id()
        AND (current_workspace_id() IS NULL OR workspace_id = current_workspace_id())
    )
    WITH CHECK (tenant_id = current_tenant_id());

DROP POLICY IF EXISTS eq_graph_edges_tenant_isolation ON eq_graph_edges;
CREATE POLICY eq_graph_edges_tenant_isolation ON eq_graph_edges
    FOR ALL
    USING (
        tenant_id = current_tenant_id()
        AND (current_workspace_id() IS NULL OR workspace_id = current_workspace_id())
    )
    WITH CHECK (tenant_id = current_tenant_id());

DO $$
BEGIN
    RAISE NOTICE 'Migration 082 complete: eq_graph_nodes / eq_graph_edges relational graph tables ready.';
END $$;
