//! Migration 001 — AGE extension guard checksum repair (relational graph backend).
//!
//! The AGE-free Postgres image lacks the Apache AGE binary, so a FRESH install of
//! `001_init_database.sql` failed at `CREATE EXTENSION age`. The fixed 001 guards
//! extension creation behind `pg_available_extensions`. Already-provisioned DBs
//! recorded the pre-guard checksum; repair it so the edited 001 does not fail with
//! "migration 1 was previously applied but has been modified".

use sqlx::PgPool;
use tracing::info;

use super::super::helpers::sqlx_migrations_table_exists;
use super::super::MIGRATION_001_VERSION;

/// SHA-384 of pre-guard 001 (unconditional `CREATE EXTENSION age`).
pub(super) const M001_CHECKSUM_PRE_AGE_GUARD: &str =
    "bb40c61f7d5cbeafa7827f2e8878588464e814ed118179bdb3c16e92880c79b2a3db4c92240e632bc8355363a768daf2";

/// SHA-384 of AGE-guarded 001 — must match `checksums.lock`.
pub(super) const M001_CHECKSUM_AGE_GUARDED: &str =
    "e53cd691f92560e0c9a5bac4a8add784c45c8079a0b82075dc153516e84c7750b6eb7bd35a0828d1562bb93d514887f2";

pub async fn repair_migration_001_checksum_if_needed(pool: &PgPool) -> Result<bool, sqlx::Error> {
    if !sqlx_migrations_table_exists(pool).await? {
        return Ok(false);
    }

    let current: Option<String> = sqlx::query_scalar(
        "SELECT encode(checksum, 'hex') FROM _sqlx_migrations \
         WHERE version = $1 AND success = true",
    )
    .bind(MIGRATION_001_VERSION)
    .fetch_optional(pool)
    .await?;

    let Some(current) = current else {
        return Ok(false);
    };

    if current != M001_CHECKSUM_PRE_AGE_GUARD {
        return Ok(false);
    }

    sqlx::query(
        "UPDATE _sqlx_migrations SET checksum = decode($1, 'hex') \
         WHERE version = $2 AND success = true",
    )
    .bind(M001_CHECKSUM_AGE_GUARDED)
    .bind(MIGRATION_001_VERSION)
    .execute(pool)
    .await?;

    info!(
        target: "edgequake.migration",
        step = "migration_001_checksum_repair",
        from = M001_CHECKSUM_PRE_AGE_GUARD,
        to = M001_CHECKSUM_AGE_GUARDED,
        "Repaired migration 001 checksum (AGE extension guard — relational graph backend)"
    );

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_and_fixed_checksums_are_distinct() {
        assert_ne!(M001_CHECKSUM_PRE_AGE_GUARD, M001_CHECKSUM_AGE_GUARDED);
        assert_eq!(M001_CHECKSUM_PRE_AGE_GUARD.len(), 96);
        assert_eq!(M001_CHECKSUM_AGE_GUARDED.len(), 96);
    }
}
