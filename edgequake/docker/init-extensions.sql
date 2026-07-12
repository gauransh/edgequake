-- EdgeQuake PostgreSQL Extensions Initialization
-- This script only creates required extensions, NOT tables.
-- Tables are created by SQLx migrations in the Rust application.

-- CRITICAL: Set the default search_path for the edgequake user to public ONLY
-- This prevents SQLx from creating _sqlx_migrations in a user-specific schema
ALTER USER edgequake SET search_path TO public;

-- Core extensions (match Dockerfile.postgres + init.sql)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- trigram fuzzy search (entity labels)
CREATE EXTENSION IF NOT EXISTS btree_gin;    -- GIN support for btree operators

-- Vector similarity search (pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

-- NOTE: Graph storage uses PostgresRelationalGraphStorage (plain SQL tables via
-- SQLx migrations). Apache AGE is no longer required or created here. Set
-- EDGEQUAKE_GRAPH_BACKEND=age only with an AGE-enabled image if you opt in.

-- Log installed extension versions for operator diagnostics
DO $$
DECLARE
    rec record;
BEGIN
    RAISE NOTICE 'EdgeQuake extension versions:';
    FOR rec IN
        SELECT extname, extversion
        FROM pg_extension
        WHERE extname IN ('vector', 'pg_trgm', 'btree_gin', 'uuid-ossp')
        ORDER BY extname
    LOOP
        RAISE NOTICE '  % = %', rec.extname, rec.extversion;
    END LOOP;
    RAISE NOTICE 'EdgeQuake extensions initialized. Tables will be created by SQLx migrations.';
END $$;
