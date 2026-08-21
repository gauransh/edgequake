//! EdgeQuake - High-Performance RAG with Knowledge Graph
//!
//! This is the main entry point for the EdgeQuake server.

use anyhow::{Context, Result};
use chrono::{Duration, Utc};
use edgequake_api::startup_security::{enforce_startup_security, validate_startup_security};
#[cfg(feature = "postgres")]
use edgequake_api::PostgresEntitySink;
use edgequake_api::{AppState, DocumentTaskProcessor, Server, ServerConfig, StorageMode};
use edgequake_query::sandbox_demo::SandboxDemoQueryEngine;
use edgequake_observability::{
    init_observability, record_db_pool_stats, ErrorEvent, ObservabilityConfig,
};
use edgequake_tasks::{
    Pagination, TaskFilter, TaskQueue, TaskStatus, TaskStorage, WorkerPool, WorkerPoolConfig,
};
use serde_json::json;
use std::sync::Arc;
use tracing::{info, warn};

/// Print the EdgeQuake startup banner with storage mode information.
fn print_startup_banner(version: &str, storage_mode: &StorageMode, host: &str, port: u16) {
    let storage_label = match storage_mode {
        StorageMode::Memory => "MEMORY (ephemeral - data lost on restart)",
        StorageMode::PostgreSQL => "POSTGRESQL (persistent)",
    };

    let storage_icon = match storage_mode {
        StorageMode::Memory => "[M]",
        StorageMode::PostgreSQL => "[P]",
    };

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("                                                               ");
    println!("   EdgeQuake v{:<47} ", version);
    println!("                                                               ");
    println!("   {} Storage: {:<40} ", storage_icon, storage_label);
    println!("   Server:  http://{}:{:<35} ", host, port);
    println!(
        "   Swagger: http://{}:{}/swagger-ui/{:<20} ",
        host, port, ""
    );
    println!("                                                               ");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

/// Format a chrono Duration into a human-readable string (e.g. "2h 15m", "47s").
fn humanize_duration(d: Duration) -> String {
    let total_secs = d.num_seconds().unsigned_abs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    if hours > 0 {
        format!("{}h {}m", hours, minutes)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else {
        format!("{}s", seconds)
    }
}

fn clear_empty_env_var(name: &str) {
    if matches!(std::env::var(name), Ok(value) if value.trim().is_empty()) {
        std::env::remove_var(name);
        info!(
            "Removed empty env var '{}' to keep provider configuration valid",
            name
        );
    }
}

fn redact_database_url(url: &str) -> String {
    let Some((prefix, host)) = url.rsplit_once('@') else {
        return url.to_string();
    };

    let Some((scheme, userinfo)) = prefix.split_once("://") else {
        return format!("***@{host}");
    };

    let username = userinfo.split(':').next().unwrap_or("user");
    format!("{scheme}://{username}:***@{host}")
}

/// Reset task rows left in processing state after an unclean restart.
///
/// Startup is a safe recovery point because no workers are active yet, so every
/// processing task is orphaned by definition and can be returned to pending.
///
/// Edge case #8: if the linked document KV is already terminal-success
/// (`completed`/`indexed`), mark the task `indexed` instead of re-queuing —
/// otherwise we reprocess a finished document after every restart.
async fn recover_orphaned_tasks(
    task_storage: Arc<dyn TaskStorage>,
    kv_storage: Arc<dyn edgequake_storage::traits::KVStorage>,
) -> Result<()> {
    info!("🔍 Checking for orphaned tasks from previous backend session...");

    let filter = TaskFilter {
        status: Some(TaskStatus::Processing),
        ..Default::default()
    };

    let now = Utc::now();
    let mut recovered_count = 0;
    let mut completed_count = 0;
    let mut page = 1;
    let page_size = 500;

    // WHY pagination loop: If >page_size tasks are stuck in "processing"
    // (e.g., large batch upload interrupted), a single page misses the rest.
    // Loop until we get an empty page or fewer results than page_size.
    loop {
        let pagination = Pagination {
            page,
            page_size,
            ..Default::default()
        };

        let task_list = task_storage.list_tasks(filter.clone(), pagination).await?;
        let batch_len = task_list.tasks.len();

        // WHY unconditional recovery: At startup there are ZERO active workers.
        // Every task with status "processing" is orphaned — there is no worker
        // processing it. The heartbeat mechanism updates `updated_at` every 60s,
        // which defeats any age-based threshold. Recovering unconditionally is
        // both correct and safe (idempotent processing + checkpoint system).
        for mut task in task_list.tasks {
            let age = now.signed_duration_since(task.updated_at);

            // Split-brain: task=processing but document already completed.
            if let Some(doc_id) =
                edgequake_api::services::extract_document_id_from_task(&task)
            {
                let meta_key =
                    edgequake_api::services::resolve_document_metadata_key(&doc_id, &kv_storage)
                        .await;
                if let Ok(Some(meta)) = kv_storage.get_by_id(&meta_key).await {
                    let status = meta.get("status").and_then(|v| v.as_str()).unwrap_or("");
                    if edgequake_api::document_metadata::is_terminal_success_status(status) {
                        task.status = TaskStatus::Indexed;
                        task.error_message = Some(format!(
                            "Auto-closed after restart: document already terminal ({status}); \
                             task was still processing (age {} minutes).",
                            age.num_minutes()
                        ));
                        task.updated_at = now;
                        match task_storage.update_task(&task).await {
                            Ok(_) => {
                                info!(
                                    "✅ Closed orphaned task against completed doc: {} → indexed",
                                    task.track_id
                                );
                                completed_count += 1;
                            }
                            Err(e) => {
                                warn!(
                                    "⚠️ Failed to close orphaned task {}: {}",
                                    task.track_id, e
                                );
                            }
                        }
                        continue;
                    }
                }
            }

            // Reset to pending for automatic retry via checkpoint system.
            // WHY pending (not failed): The checkpoint system will resume from
            // where the task left off. Marking as "failed" forces manual retry
            // which is poor UX. Resetting to "pending" enables auto-recovery.
            task.status = TaskStatus::Pending;
            task.error_message = Some(format!(
                "Auto-recovered after backend restart (was processing for {} minutes). \
                 Will resume from checkpoint if available.",
                age.num_minutes()
            ));
            task.updated_at = now;

            match task_storage.update_task(&task).await {
                Ok(_) => {
                    info!(
                        "✅ Recovered orphaned task: {} (age: {})",
                        task.track_id,
                        humanize_duration(age)
                    );
                    recovered_count += 1;
                }
                Err(e) => {
                    warn!(
                        "⚠️ Failed to recover orphaned task {}: {}",
                        task.track_id, e
                    );
                }
            }
        }

        // Stop when we got fewer results than page_size (last page)
        if batch_len < page_size as usize {
            break;
        }
        page += 1;
    }

    if recovered_count > 0 || completed_count > 0 {
        info!(
            "🔧 Orphaned task recovery complete: {} requeued, {} closed (doc already terminal)",
            recovered_count, completed_count
        );
    } else {
        info!("✅ No orphaned tasks found - clean startup");
    }

    Ok(())
}

/// Normalize document metadata left in non-terminal states after a restart.
///
/// Early upload stages are marked for re-upload, while later stages are reset to
/// pending so checkpoint-aware processing can resume automatically.
///
/// `min_age`: when `Some`, only documents whose `updated_at` is older than this
/// duration are recovered. Use `None` at startup (zero workers → all non-terminal
/// docs are orphans). Periodic recovery MUST pass a positive age to avoid
/// resetting documents that are actively being processed.
async fn recover_orphaned_documents(
    kv_storage: Arc<dyn edgequake_storage::traits::KVStorage>,
    min_age: Option<Duration>,
) -> Result<()> {
    info!("🔍 Checking for orphaned documents from previous backend session...");

    // Prefer suffix scan — full keys() is O(all KV) and slows large-tenant boot.
    let metadata_keys = match kv_storage.keys_with_suffix("-metadata").await {
        Ok(keys) => keys,
        Err(_) => {
            let all_keys = kv_storage.keys().await?;
            all_keys
                .into_iter()
                .filter(|k| k.ends_with("-metadata"))
                .collect()
        }
    };

    if metadata_keys.is_empty() {
        info!("✅ No documents found - clean startup");
        return Ok(());
    }

    let metadata_values = kv_storage.get_by_ids_ordered(&metadata_keys).await?;
    let now = Utc::now();

    // Stages where no meaningful work has been done yet — source content
    // may have been lost on restart. These need user re-upload.
    //
    // WHY only "uploading": During "uploading" the HTTP multipart receive is
    // in progress and the PDF binary may not be fully stored in PostgreSQL.
    // "converting" was previously here but is WRONG — by the time the worker
    // reaches "converting", the PDF binary is fully stored in PostgreSQL and
    // pipeline checkpoints can resume conversion. Marking it "failed" while
    // the recovered task resumes causes a state desync where the UI shows
    // "Failed" but the backend actively processes the document.
    let needs_reupload_stages = ["uploading"];

    // Stages where pipeline checkpoint or at least the text is in KV storage,
    // so automatic retry is possible.
    let auto_retryable_statuses = [
        "converting",
        "preprocessing",
        "chunking",
        "extracting",
        "gleaning",
        "merging",
        "summarizing",
        "embedding",
        "storing",
        "pending",
        "processing",
        "indexing",
    ];

    let non_terminal_statuses: Vec<&str> = needs_reupload_stages
        .iter()
        .chain(auto_retryable_statuses.iter())
        .copied()
        .collect();

    let mut auto_recovered_count = 0;
    let mut needs_reupload_count = 0;

    for (key, maybe_value) in metadata_keys.iter().zip(metadata_values.iter()) {
        // Never touch staging admission shells — mid-upload race (edge case #5).
        if key.starts_with("staging:") {
            continue;
        }
        let Some(value) = maybe_value else {
            continue;
        };
        if let Some(obj) = value.as_object() {
            // Check both `status` and `current_stage` for stuck states
            let status = obj.get("status").and_then(|v| v.as_str()).unwrap_or("");
            let current_stage = obj
                .get("current_stage")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let is_stuck = non_terminal_statuses.contains(&status)
                || non_terminal_statuses.contains(&current_stage);

            if !is_stuck {
                continue;
            }

            // Periodic path: skip docs still receiving heartbeats / progress.
            // Missing updated_at + non-terminal = treat as orphan (edge case #14) —
            // HTTP recover_stuck already does this; keep policies aligned.
            if let Some(age) = min_age {
                let updated_at = obj
                    .get("updated_at")
                    .and_then(|v| v.as_str())
                    .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                    .map(|dt| dt.with_timezone(&Utc));
                if let Some(ts) = updated_at {
                    if now.signed_duration_since(ts) < age {
                        continue;
                    }
                }
            }

            // WHY no age threshold at startup: ZERO workers are running.
            // Any document in a non-terminal state is orphaned by definition.

            // Determine recovery strategy based on the stuck stage
            let stuck_stage = if !current_stage.is_empty() {
                current_stage
            } else {
                status
            };

            let is_early_stage = needs_reupload_stages.contains(&stuck_stage);

            let mut updated = serde_json::Value::Object(obj.clone());

            if is_early_stage {
                if let Some(obj) = updated.as_object_mut() {
                    // Early stage: source content may be lost — mark failed but with
                    // a clear message so users know to re-upload
                    obj.insert("status".to_string(), serde_json::json!("failed"));
                    obj.insert("current_stage".to_string(), serde_json::json!("failed"));
                    obj.insert(
                        "stage_message".to_string(),
                        serde_json::json!(format!(
                            "Server restarted during '{}' stage. Source content may be incomplete. \
                             Please re-upload the document.",
                            stuck_stage
                        )),
                    );
                    obj.insert(
                        "error_message".to_string(),
                        serde_json::json!(
                            "Server restarted during early processing — please re-upload"
                        ),
                    );
                }
                needs_reupload_count += 1;
            } else {
                if let Some(obj) = updated.as_object_mut() {
                    // Later stage: pipeline checkpoint likely exists, auto-retry.
                    obj.insert("status".to_string(), serde_json::json!("pending"));
                    obj.insert("current_stage".to_string(), serde_json::json!("pending"));
                    obj.insert(
                        "stage_message".to_string(),
                        serde_json::json!(format!(
                            "Auto-recovered after server restart (was in '{}' stage). \
                             Resuming from checkpoint...",
                            stuck_stage
                        )),
                    );
                    obj.remove("error_message");
                }
                auto_recovered_count += 1;
            }

            if let Some(obj) = updated.as_object_mut() {
                obj.insert(
                    "updated_at".to_string(),
                    serde_json::json!(now.to_rfc3339()),
                );
            }

            // SPEC-045: metadata key is authoritative — repair misaligned JSON `id`.
            edgequake_storage::repair_document_metadata_in_place(key, &mut updated);

            // Keep wsdoc:{workspace}:{doc} index in sync (list/filter depends on it).
            match edgequake_api::services::upsert_metadata_kv_with_index(
                kv_storage.as_ref(),
                key,
                updated,
            )
            .await
            {
                Ok(_) => {
                    if is_early_stage {
                        info!(
                            "⚠️ Document needs re-upload: {} (was stuck in '{}')",
                            key, stuck_stage
                        );
                    } else {
                        info!(
                            "✅ Auto-recovered document: {} (was in '{}' → pending, will resume from checkpoint)",
                            key, stuck_stage
                        );
                    }
                }
                Err(e) => {
                    ErrorEvent::log_domain_warn(
                        "startup",
                        "recover_orphaned_document",
                        &e.to_string(),
                        json!({ "document_id": key, "non_fatal": true }),
                    );
                }
            }
        }
    }

    let total_recovered = auto_recovered_count + needs_reupload_count;
    if total_recovered > 0 {
        info!(
            "🔧 Orphaned document recovery complete: {} auto-recovered (pending), {} need re-upload (failed)",
            auto_recovered_count, needs_reupload_count
        );
    } else {
        info!("✅ No orphaned documents found - clean startup");
    }

    Ok(())
}

/// Reload pending database tasks into the in-memory worker queue on startup.
async fn requeue_pending_tasks(
    task_storage: Arc<dyn TaskStorage>,
    task_queue: Arc<dyn TaskQueue>,
) -> Result<()> {
    info!("🔄 Checking for pending tasks to requeue from database...");

    let filter = TaskFilter {
        status: Some(TaskStatus::Pending),
        ..Default::default()
    };

    let mut requeued_count = 0;
    let mut failed_count = 0;
    let mut page = 1;
    let page_size = 500;

    // SPEC-045 SRE-I02: paginate — startup may have >1000 pending after outage.
    loop {
        let pagination = Pagination {
            page,
            page_size,
            ..Default::default()
        };

        let task_list = task_storage.list_tasks(filter.clone(), pagination).await?;
        let batch_len = task_list.tasks.len();

        if batch_len == 0 && page == 1 {
            info!("✅ No pending tasks to requeue");
            return Ok(());
        }

        if page == 1 && batch_len > 0 {
            info!("📋 Found pending task(s) in database, requeueing to worker pool...");
        }

        for task in task_list.tasks {
            match task_queue.send(task.clone()).await {
                Ok(_) => {
                    info!("✅ Requeued task: {}", task.track_id);
                    requeued_count += 1;
                }
                Err(e) => {
                    ErrorEvent::log_domain_warn(
                        "startup",
                        "requeue_pending_task",
                        &e.to_string(),
                        json!({ "task_id": task.track_id, "non_fatal": true }),
                    );
                    failed_count += 1;
                }
            }
        }

        if batch_len < page_size as usize {
            break;
        }
        page += 1;
    }

    info!(
        "🔧 Pending task requeue complete: {} requeued, {} failed",
        requeued_count, failed_count
    );

    Ok(())
}

/// Mark processing tasks as failed if their heartbeat has been dead for too long.
async fn periodic_orphan_check(
    task_storage: Arc<dyn TaskStorage>,
    kv_storage: Arc<dyn edgequake_storage::traits::KVStorage>,
) -> Result<()> {
    let filter = TaskFilter {
        status: Some(TaskStatus::Processing),
        ..Default::default()
    };

    let now = Utc::now();
    let orphan_threshold = Duration::minutes(10);
    let mut recovered_count = 0;
    let mut page = 1;
    let page_size = 500;

    // WHY pagination loop: Same reason as startup recovery — if many tasks
    // have dead heartbeats (e.g., OOM killed multiple workers), we need
    // to process all of them, not just the first page.
    loop {
        let pagination = Pagination {
            page,
            page_size,
            ..Default::default()
        };

        let task_list = task_storage.list_tasks(filter.clone(), pagination).await?;
        let batch_len = task_list.tasks.len();

        for mut task in task_list.tasks {
            let age = now.signed_duration_since(task.updated_at);

            if age > orphan_threshold {
                let error_msg = format!(
                    "Task heartbeat lost (no update for {} minutes). \
                     The worker may have crashed. Please retry.",
                    age.num_minutes()
                );
                // Heartbeat died — mark as failed so the user can see and retry
                task.status = TaskStatus::Failed;
                task.error_message = Some(error_msg.clone());
                task.updated_at = now;

                match task_storage.update_task(&task).await {
                    Ok(_) => {
                        warn!(
                            "⚠️ Periodic check: recovered dead-heartbeat task {} (age: {})",
                            task.track_id,
                            humanize_duration(age)
                        );
                        // SPEC-045 SRE-I01: sync document KV so UI does not show processing
                        if let Err(e) =
                            edgequake_api::services::sync_document_failed_on_orphan_heartbeat(
                                Arc::clone(&kv_storage),
                                &task,
                                &error_msg,
                            )
                            .await
                        {
                            warn!(
                                task_id = %task.track_id,
                                error = %e,
                                "Failed to sync document metadata after orphan heartbeat"
                            );
                        }
                        recovered_count += 1;
                    }
                    Err(e) => {
                        warn!(
                            "⚠️ Failed to recover dead-heartbeat task {}: {}",
                            task.track_id, e
                        );
                    }
                }
            }
        }

        if batch_len < page_size as usize {
            break;
        }
        page += 1;
    }

    if recovered_count > 0 {
        info!(
            "🔧 Periodic orphan check: {} task(s) recovered",
            recovered_count
        );
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let _obs_guard = init_observability(ObservabilityConfig::from_env());

    info!("Starting EdgeQuake v{}", env!("CARGO_PKG_VERSION"));

    for var in ["OPENAI_BASE_URL", "OPENAI_API_KEY"] {
        clear_empty_env_var(var);
    }

    // Get API key from environment (optional - Ollama doesn't need it)
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_default();

    let database_url = std::env::var("DATABASE_URL").context(
        "DATABASE_URL is required; start PostgreSQL and rerun make dev or make dev-auth",
    )?;

    let redacted_database_url = redact_database_url(&database_url);
    info!("🐘 PostgreSQL storage mode using {}", redacted_database_url);
    let mut state = AppState::new_postgres(&database_url, &api_key)
        .await
        .map_err(|error| anyhow::anyhow!(error.to_string()))
        .with_context(|| {
            format!("failed to initialize PostgreSQL storage at {redacted_database_url}")
        })?;

    if let Some(ref bootstrap) = state.migration_bootstrap {
        if bootstrap.migration_038.is_degraded() {
            tracing::warn!(
                target: "edgequake.migration",
                missing = ?bootstrap.migration_038.missing_indexes,
                action = bootstrap.migration_038.operator_action.as_deref().unwrap_or("none"),
                "/ready will return 503 until migration 038 indexes are applied (use apply_038.sh --concurrent for large graphs)"
            );
        }
    }

    info!(
        target: "edgequake.resource",
        graph_scan_threshold = state.resource_budget().graph_scan_threshold_nodes,
        graph_materialize_concurrent = state.graph_materialize.max_concurrent(),
        graph_query_timeout_secs = state.resource_budget().graph_query_timeout_secs,
        max_page_size = state.resource_budget().max_page_size,
        "SPEC-006 resource budget active"
    );

    if std::env::var("EDGEQUAKE_MEM_LIMIT").is_err() && std::env::var("DOCKER_MEM_LIMIT").is_err() {
        tracing::warn!(
            target: "edgequake.resource",
            "No EDGEQUAKE_MEM_LIMIT/Docker mem_limit detected — bare-metal dev may OOM on large workspaces; use docker compose or set EDGEQUAKE_MEM_LIMIT"
        );
    }

    // Initialize default tenant and workspace for non-authenticated mode
    if let Err(e) = state.initialize_defaults().await {
        ErrorEvent::log_domain_warn(
            "startup",
            "initialize_defaults",
            &e.to_string(),
            json!({ "non_fatal": true }),
        );
    }

    // GitHub #288: secure-by-default auth requires a login-capable user on first boot.
    if let Err(e) =
        edgequake_api::services::auth_bootstrap::bootstrap_auth_identity_if_needed(&state).await
    {
        ErrorEvent::log_domain_warn(
            "startup",
            "auth_bootstrap",
            &e.to_string(),
            json!({ "non_fatal": true }),
        );
    }

    // SPEC-018: Sample DB pool gauges between Prometheus scrapes.
    if let Some(pool) = state.pg_pool.clone() {
        tokio::spawn(async move {
            let interval_secs = std::env::var("EDGEQUAKE_DB_POOL_METRICS_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(15);
            let interval = std::time::Duration::from_secs(interval_secs.max(5));
            loop {
                record_db_pool_stats(pool.size(), pool.num_idle().min(u32::MAX as usize) as u32);
                tokio::time::sleep(interval).await;
            }
        });
    }

    // Create document task processor with workspace-specific pipeline support (SPEC-032)
    // This ensures that rebuild/reprocess operations use the workspace's configured
    // LLM and embedding providers, not the server's default providers.
    //
    // OODA-03: Always use STRICT workspace isolation mode (PostgreSQL required now).
    // OODA-223: Strict mode enforces workspace isolation.
    // OODA-10: Also attach progress broadcaster for WebSocket event delivery.
    info!("🔒 Using STRICT workspace isolation mode (PostgreSQL storage)");
    let mut processor = DocumentTaskProcessor::with_workspace_support_strict(
        Arc::clone(&state.query.pipeline),
        Arc::clone(&state.query.llm_provider),
        Arc::clone(&state.storage.kv_storage),
        Arc::clone(&state.storage.vector_storage),
        Arc::clone(&state.storage.vector_registry),
        Arc::clone(&state.storage.graph_storage),
        state.tasks.pipeline_state.clone(),
        Arc::clone(&state.workspace_service),
        Arc::clone(&state.query.models_config),
    )
    .with_progress_broadcaster(state.tasks.progress_broadcaster.clone())
    .with_task_storage(Arc::clone(&state.tasks.storage) as edgequake_tasks::SharedTaskStorage)
    .with_pdf_vision_semaphore(Arc::clone(&state.pdf_vision))
    .with_query_engine(Arc::clone(&state.query.engine_impl));

    info!(
        pdf_vision_jobs = state.pdf_vision.max_concurrent(),
        "Vision PDF admission control enabled"
    );

    // SPEC-021 P3-01c: Wire CQRS entity dual-write sink when postgres feature is active.
    // create_if_enabled() checks entity_sync_mode in server_config; returns NoopEntitySink
    // when mode is "disabled" (default), so this is always safe to call.
    #[cfg(feature = "postgres")]
    if let Some(ref pool) = state.pg_pool {
        let entity_sink = PostgresEntitySink::create_if_enabled(Arc::new(pool.clone())).await;
        processor = processor.with_relational_sink(entity_sink);

        // SPEC-032 W-08: Wire lineage sink when migration 066 has been applied.
        // create_if_migration_applied() checks if chunk_entity_links table exists;
        // returns NoopLineageSink if not (zero-downtime rollout).
        let lineage_sink =
            edgequake_api::postgres_lineage_sink::PostgresLineageSink::create_if_migration_applied(
                Arc::new(pool.clone()),
            )
            .await;
        processor = processor.with_lineage_sink(lineage_sink);
        info!("🔗 Lineage sink wired (SPEC-032 W-08)");
    }

    // CRITICAL: Attach PDF storage for PDF processing tasks
    if let Some(ref pdf_storage) = state.storage.pdf_storage {
        processor = processor.with_pdf_storage(Arc::clone(pdf_storage));
        info!("📄 PDF storage attached to task processor");
    }

    #[cfg(feature = "postgres")]
    if let (Some(pool), Some(caps)) = (state.pg_pool.clone(), state.postgres_capabilities.clone()) {
        processor = processor.with_postgres_id_allocation(pool, caps);
    }

    let processor = Arc::new(processor);

    // Configure worker pool
    // WHY num_cpus * 4: Pipeline processing is IO-bound (LLM API calls,
    // embedding generation). Workers mostly wait for network I/O, so we need
    // more workers than CPU cores to keep the pipeline saturated.
    let num_workers: usize = std::env::var("WORKER_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| (num_cpus::get() * 4).max(4));

    let worker_config = WorkerPoolConfig {
        num_workers,
        auto_retry: true,
        initial_retry_delay_ms: 5000,
        max_retry_delay_ms: 60000,
        backoff_multiplier: 2.0,
        // FEAT-TENANT-FAIRNESS: Per-tenant concurrency limit.
        // Ensures no single tenant can monopolize all workers.
        // Default: max(1, num_workers * 3/4) — IO-bound workloads benefit
        // from higher per-tenant concurrency while still reserving 25%
        // capacity for other tenants.
        // Set MAX_TASKS_PER_TENANT=0 to disable.
        max_tasks_per_tenant: std::env::var("MAX_TASKS_PER_TENANT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| (num_workers * 3 / 4).max(1)),
        // WHY 2 hours (7200s): Large PDFs with vision LLM extraction can take
        // 3+ hours (1000+ pages × ~12s/page). 2 hours covers the vast majority
        // of real-world documents while still catching truly stuck tasks.
        // Without this timeout, hung processor.process() calls keep heartbeating
        // forever, creating phantom "Processing N document(s)" banners.
        processing_timeout_secs: {
            let raw = std::env::var("TASK_PROCESSING_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(7200u64);
            // Clamp to minimum 60s to prevent misconfiguration (0 = instant timeout)
            let clamped = raw.max(60);
            if clamped != raw {
                warn!(
                    "TASK_PROCESSING_TIMEOUT_SECS={} is below minimum (60). Using 60s.",
                    raw
                );
            }
            clamped
        },
    };

    // Recover orphaned tasks from previous backend session (PRODUCTION_BUG_FIX)
    // MUST run BEFORE starting workers to prevent race conditions
    if let Err(e) = recover_orphaned_tasks(
        Arc::clone(&state.tasks.storage) as Arc<dyn TaskStorage>,
        Arc::clone(&state.storage.kv_storage) as Arc<dyn edgequake_storage::traits::KVStorage>,
    )
    .await
    {
        ErrorEvent::log_domain_warn(
            "startup",
            "recover_orphaned_tasks",
            &e.to_string(),
            json!({ "non_fatal": true }),
        );
    }

    // Recover orphaned documents stuck in non-terminal states (uploading, pending, etc.)
    // MUST run BEFORE starting workers to avoid race with new uploads

    // SPEC-045: repair metadata KV integrity BEFORE orphan recovery (prevents re-corruption).
    if let Err(e) = edgequake_api::services::document_metadata_repair::repair_all_document_metadata(
        Arc::clone(&state.storage.kv_storage) as Arc<dyn edgequake_storage::traits::KVStorage>,
        state.pg_pool.as_ref(),
    )
    .await
    {
        ErrorEvent::log_domain_warn(
            "startup",
            "repair_document_metadata",
            &e.to_string(),
            json!({ "non_fatal": true }),
        );
    }

    if let Err(e) = recover_orphaned_documents(
        Arc::clone(&state.storage.kv_storage) as Arc<dyn edgequake_storage::traits::KVStorage>,
        None, // startup: all non-terminal docs are orphans (no workers yet)
    )
    .await
    {
        ErrorEvent::log_domain_warn(
            "startup",
            "recover_orphaned_documents",
            &e.to_string(),
            json!({ "non_fatal": true }),
        );
    }

    // Requeue pending tasks from database to in-memory queue (PRODUCTION_BUG_FIX)
    // MUST run BEFORE starting workers so tasks are available when workers start polling
    if let Err(e) = requeue_pending_tasks(
        Arc::clone(&state.tasks.storage) as Arc<dyn TaskStorage>,
        Arc::clone(&state.tasks.queue) as Arc<dyn TaskQueue>,
    )
    .await
    {
        ErrorEvent::log_domain_warn(
            "startup",
            "requeue_pending_tasks",
            &e.to_string(),
            json!({ "non_fatal": true }),
        );
    }

    // CHECKPOINT-CLEANUP: Remove pipeline checkpoints older than 24 hours.
    // WHY: Stale checkpoints reference outdated provider configs or content
    // that may have been re-uploaded. Cleaning on startup keeps storage lean
    // and prevents stale data from being reloaded.
    edgequake_api::processor::pipeline_checkpoint::cleanup_stale_checkpoints(
        &state.storage.kv_storage,
    )
    .await;

    // Create and start worker pool
    let mut worker_pool = WorkerPool::new(
        worker_config.clone(),
        Arc::clone(&state.tasks.queue) as Arc<dyn edgequake_tasks::TaskQueue>,
        Arc::clone(&state.tasks.storage) as Arc<dyn edgequake_tasks::TaskStorage>,
        processor,
    );

    // WHY: The cancel_task API handler signals the CancellationRegistry living
    // on AppState.  The worker loop registers/checks tokens via the registry
    // in WorkerPool.  Both must point to the *same* underlying Arc so that a
    // cancel request from the HTTP handler is visible to the running worker.
    state.tasks.cancellation_registry = worker_pool.cancellation_registry();

    info!(
        "Starting worker pool with {} workers (task timeout: {}s)",
        worker_config.num_workers, worker_config.processing_timeout_secs
    );
    worker_pool.start();

    // PERIODIC ORPHAN RECOVERY: Background task that catches tasks whose heartbeat
    // stopped mid-runtime (e.g., worker panic, tokio task cancellation). Uses the
    // 10-minute updated_at threshold — safe because legitimate tasks have heartbeats
    // updating every 60s. This complements startup recovery (which is unconditional)
    // and the processing timeout (which catches hung tasks with active heartbeats).
    let periodic_task_storage = Arc::clone(&state.tasks.storage) as Arc<dyn TaskStorage>;
    let periodic_kv_storage =
        Arc::clone(&state.storage.kv_storage) as Arc<dyn edgequake_storage::traits::KVStorage>;
    tokio::spawn(async move {
        // WHY 5 minutes: Frequent enough to catch dead-heartbeat tasks within
        // ~15 minutes (10 min threshold + up to 5 min wait for the next check).
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(300));
        interval.tick().await; // Skip first immediate tick (startup recovery already ran)
        loop {
            interval.tick().await;
            if let Err(e) = periodic_orphan_check(
                Arc::clone(&periodic_task_storage),
                Arc::clone(&periodic_kv_storage),
            )
            .await
            {
                ErrorEvent::log_domain_warn(
                    "startup",
                    "periodic_orphan_check",
                    &e.to_string(),
                    json!({ "non_fatal": true }),
                );
            }
        }
    });

    // PERIODIC DOCUMENT ORPHAN RECOVERY (SPEC-045): Re-normalize KV metadata for docs
    // stuck in non-terminal states after long-running processing (complements task-level
    // periodic_orphan_check). Disabled unless EDGEQUAKE_AUTO_ORPHAN_DOCUMENT_RECOVER_MINUTES>0.
    if let Some(interval_mins) = std::env::var("EDGEQUAKE_AUTO_ORPHAN_DOCUMENT_RECOVER_MINUTES")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&m| m > 0)
    {
        let kv =
            Arc::clone(&state.storage.kv_storage) as Arc<dyn edgequake_storage::traits::KVStorage>;
        // Only recover docs stale longer than the interval itself (active workers
        // keep updating `updated_at` via progress patches).
        let min_age = Duration::minutes(interval_mins as i64);
        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_secs(interval_mins * 60));
            interval.tick().await;
            loop {
                interval.tick().await;
                if let Err(e) = recover_orphaned_documents(Arc::clone(&kv), Some(min_age)).await {
                    ErrorEvent::log_domain_warn(
                        "startup",
                        "periodic_recover_orphaned_documents",
                        &e.to_string(),
                        json!({ "non_fatal": true }),
                    );
                }
            }
        });
        info!(
            "Periodic document orphan recovery enabled (every {} min, min_age={} min)",
            interval_mins, interval_mins
        );
    }

    // Configure server
    let config = ServerConfig {
        host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
        port: std::env::var("PORT")
            .ok()
            .and_then(|p| p.parse().ok())
            .unwrap_or(8080),
        enable_cors: true,
        enable_compression: true,
        enable_swagger: true,
    };

    // Print startup banner with storage mode
    print_startup_banner(
        env!("CARGO_PKG_VERSION"),
        &state.storage.mode,
        &config.host,
        config.port,
    );

    // SPEC-027 IMP-001: warn or exit on insecure production configuration
    enforce_startup_security(validate_startup_security(
        std::env::var("DATABASE_URL").ok().as_deref(),
        &state.auth.config,
        &state.security,
    ));

    // CrystalOS sandbox demo (EDGEQUAKE_SANDBOX_PACK_DIR).
    //
    // Unset — the ordinary case — and the demo routes simply do not exist.
    // Set, and the activation bundle MUST load: on that path graph-core is
    // mandatory, so an EdgeQuake that came up serving a demo query without
    // the model would be worse than one that refused to come up at all.
    // Loading also embeds one document per governed resource, so the first
    // query on stage is warm.
    let sandbox_demo = SandboxDemoQueryEngine::from_env(Arc::clone(&state.query.embedding_provider))
        .await
        .with_context(|| {
            format!(
                "loading the sandbox demo pack named by {}",
                edgequake_query::sandbox_demo::PACK_DIR_ENV
            )
        })?
        .map(Arc::new);
    if let Some(engine) = &sandbox_demo {
        info!(
            "Sandbox demo enabled: POST /api/v1/query/sandbox-demo, page at /demo/sandbox ({:?})",
            engine
        );
    }

    // Run server (this blocks until shutdown)
    let mut server = Server::new(config, state);
    if let Some(engine) = sandbox_demo {
        server = server.with_sandbox_demo(engine);
    }
    let result = server.run().await;

    // Graceful shutdown of worker pool
    info!("Shutting down worker pool...");
    worker_pool.shutdown().await;

    result?;
    Ok(())
}
