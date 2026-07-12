-- Migration 060: Allow system/master-principal API keys (issue #17)
--
-- WHY: `POST /api/v1/api-keys` returned 500 when authenticated with the master
-- API key. The master principal has no `users` row (its synthetic user_id is the
-- string "master-api-key"), so the `api_keys.user_id UUID NOT NULL` column +
-- FK to `users(user_id)` could never be satisfied — the handler failed the
-- UUID parse before it even reached the DB, surfacing as an Internal (500).
--
-- Making `user_id` nullable lets system/master principals mint scoped keys with
-- a NULL owner while PRESERVING the FK + ON DELETE CASCADE for real user keys
-- (a FK constraint does not reject NULLs). Additive and backward compatible:
-- existing rows already have a non-null user_id.
--
-- @implements FIX-ISSUE-17: master-key principals can mint scoped API keys.

-- Idempotent: DROP NOT NULL is a no-op if the column is already nullable.
ALTER TABLE api_keys ALTER COLUMN user_id DROP NOT NULL;

COMMENT ON COLUMN api_keys.user_id IS
    'Owning user (FK users.user_id). NULL for system/master-principal keys minted without a user identity (issue #17).';
