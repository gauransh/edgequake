//! Text-based document upload handler.

use std::time::{Duration, Instant};

use axum::http::StatusCode;
use axum::{extract::State, Json};
use tracing::debug;

use crate::error::ApiResult;
use crate::middleware::TenantContext;
use crate::services::ContentHasher;
use crate::state::AppState;

use crate::handlers::documents::upload::{
    admit_document_for_processing, parse_upload_chunk_fields, DocumentAdmissionAccepted,
    DocumentAdmissionInput, DocumentAdmissionOutcome, GleaningAdmissionOptions,
    ADMISSION_ACCEPTED_STATUS,
};
use crate::handlers::documents_types::*;

/// Max time the synchronous (`async_processing=false`) upload path waits for a
/// document to reach a terminal status before falling back to 202 + track_id
/// (issue #18). Kept below typical client/proxy read timeouts.
const SYNC_PROCESSING_TIMEOUT: Duration = Duration::from_secs(120);

/// Poll cadence while awaiting terminal status. The worker updates KV metadata
/// as it advances stages; polling avoids coupling the handler to the task queue.
const SYNC_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Upload a document for processing.
///
/// `async_processing` controls the response contract (issue #18):
/// - `false` (default) — **synchronous**: the request blocks until the document
///   reaches a terminal status (up to 120s) and returns `200 OK` with the final
///   status and `chunk_count`/`entity_count`/`relationship_count`/`cost`. If the
///   worker does not finish within the budget, the server falls back to async
///   semantics and returns `202 Accepted` + `track_id` (poll
///   `GET /api/v1/documents/track/{track_id}`).
/// - `true` — **asynchronous**: returns `202 Accepted` immediately with a
///   `track_id`/`task_id`; the document is processed in the background.
#[utoipa::path(
    post,
    path = "/api/v1/documents",
    tag = "Documents",
    params(
        ("X-Tenant-ID" = Option<String>, Header, description = "Tenant UUID for multi-tenant isolation"),
        ("X-Workspace-ID" = Option<String>, Header, description = "Workspace UUID — scopes uploaded documents"),
    ),
    request_body = UploadDocumentRequest,
    responses(
        (status = 200, description = "Document processed synchronously (async_processing=false)", body = UploadDocumentResponse),
        (status = 202, description = "Document accepted for async processing (async_processing=true, or sync wait exceeded 120s)", body = UploadDocumentResponse),
        (status = 400, description = "Invalid request"),
        (status = 413, description = "Document too large")
    )
)]
pub async fn upload_document(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Json(request): Json<UploadDocumentRequest>,
) -> ApiResult<(StatusCode, Json<UploadDocumentResponse>)> {
    debug!(
        tenant_id = ?tenant_ctx.tenant_id,
        workspace_id = ?tenant_ctx.workspace_id,
        "Uploading document with tenant context"
    );

    crate::validation::validate_content(&request.content, state.config.max_document_size)?;

    let wait_for_completion = !request.async_processing;

    let content_hash = ContentHasher::hash_str(&request.content);
    let content_length = request.content.len();
    let title = request
        .title
        .clone()
        .unwrap_or_else(|| "Untitled".to_string());

    let (chunk_strategy, chunk_options) =
        parse_upload_chunk_fields(request.chunk_strategy.as_deref(), request.chunk_options);

    let outcome = admit_document_for_processing(
        &state,
        &tenant_ctx,
        DocumentAdmissionInput {
            text_content: request.content,
            title: title.clone(),
            source_type: "markdown",
            mime_type: None,
            raw_byte_size: content_length,
            content_hash,
            custom_metadata: request.metadata,
            track_id: request.track_id,
            gleaning: GleaningAdmissionOptions {
                enable_gleaning: request.enable_gleaning,
                max_gleaning: request.max_gleaning,
            },
            document_type: Some("markdown"),
            chunk_strategy,
            chunk_options,
            multimodal: false,
            ingest_mode: None,
            multimodal_manifest: None,
        },
        "upload",
    )
    .await?;

    match outcome {
        DocumentAdmissionOutcome::DuplicateProcessing(dup) => Ok((
            StatusCode::OK,
            Json(UploadDocumentResponse {
                document_id: dup.document_id.clone(),
                status: "duplicate_processing".to_string(),
                task_id: None,
                track_id: String::new(),
                duplicate_of: Some(dup.document_id),
                chunk_count: None,
                entity_count: None,
                relationship_count: None,
                cost: None,
            }),
        )),
        DocumentAdmissionOutcome::Accepted(accepted) => {
            // issue #18: async_processing=false means synchronous — the client
            // expects the document to be fully processed when we respond. Await
            // the document's terminal status (bounded), then return the final
            // status + stats. On timeout, fall back to async semantics (202 +
            // track_id) so a backed-up worker queue cannot hang the request.
            if wait_for_completion {
                if let Some(final_meta) =
                    await_document_terminal_status(&state, &accepted.document_id).await
                {
                    return Ok((
                        StatusCode::OK,
                        Json(build_sync_response(&accepted, &final_meta)),
                    ));
                }
                tracing::warn!(
                    document_id = %accepted.document_id,
                    track_id = %accepted.track_id,
                    timeout_secs = SYNC_PROCESSING_TIMEOUT.as_secs(),
                    "Synchronous upload exceeded wait budget — returning 202 + track_id (issue #18)"
                );
            }

            Ok((
                ADMISSION_ACCEPTED_STATUS,
                Json(UploadDocumentResponse {
                    document_id: accepted.document_id,
                    status: "pending".to_string(),
                    task_id: Some(accepted.task_id),
                    track_id: accepted.track_id,
                    duplicate_of: None,
                    chunk_count: None,
                    entity_count: None,
                    relationship_count: None,
                    cost: None,
                }),
            ))
        }
    }
}

/// Poll a document's KV metadata until it reaches a terminal status or the sync
/// wait budget elapses (issue #18). Returns the final metadata on completion,
/// or `None` on timeout (caller falls back to async 202).
///
/// Uses the staging-first metadata key so status is observed across the
/// in-flight staging → promoted transition.
async fn await_document_terminal_status(
    state: &AppState,
    document_id: &str,
) -> Option<serde_json::Value> {
    let deadline = Instant::now() + SYNC_PROCESSING_TIMEOUT;
    loop {
        let metadata_key =
            crate::services::resolve_document_metadata_key(document_id, &state.storage.kv_storage)
                .await;
        if let Ok(Some(meta)) = state.storage.kv_storage.get_by_id(&metadata_key).await {
            if let Some(status) = meta.get("status").and_then(|v| v.as_str()) {
                if crate::document_metadata::is_terminal_document_status(status) {
                    return Some(meta);
                }
            }
        }

        if Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(SYNC_POLL_INTERVAL).await;
    }
}

/// Build the synchronous upload response from a document's terminal metadata.
fn build_sync_response(
    accepted: &DocumentAdmissionAccepted,
    meta: &serde_json::Value,
) -> UploadDocumentResponse {
    let get_usize = |k: &str| meta.get(k).and_then(|v| v.as_u64()).map(|n| n as usize);
    let status = meta
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("completed")
        .to_string();

    let get_str = |k: &str| meta.get(k).and_then(|v| v.as_str()).map(String::from);
    let cost = meta
        .get("cost_usd")
        .and_then(|v| v.as_f64())
        .map(|usd| DocumentCostInfo {
            total_cost_usd: usd,
            formatted_cost: DocumentCostInfo::format_cost(usd),
            input_tokens: get_usize("input_tokens").unwrap_or(0),
            output_tokens: get_usize("output_tokens").unwrap_or(0),
            total_tokens: get_usize("total_tokens").unwrap_or(0),
            llm_model: get_str("llm_model"),
            embedding_model: get_str("embedding_model"),
        });

    UploadDocumentResponse {
        document_id: accepted.document_id.clone(),
        status,
        task_id: None,
        track_id: accepted.track_id.clone(),
        duplicate_of: None,
        chunk_count: get_usize("chunk_count"),
        entity_count: get_usize("entity_count"),
        relationship_count: get_usize("relationship_count"),
        cost,
    }
}
