//! Track status handler — query documents by track ID.

use axum::{extract::State, Json};

use crate::error::ApiResult;
use crate::middleware::TenantContext;
use crate::services::document_metadata_scan::load_scoped_document_metadata;
use crate::services::tenant_guard::{
    empty_track_status, has_full_tenant_context, warn_missing_tenant_context,
};
use crate::state::StorageRuntime;

use crate::handlers::documents_types::*;

/// Get track status by track ID.
///
/// Returns all documents uploaded with a specific track_id, along with status summary.
#[utoipa::path(
    get,
    path = "/api/v1/documents/track/{track_id}",
    tag = "Documents",
    params(
        ("track_id" = String, Path, description = "Track ID for the batch")
    ),
    responses(
        (status = 200, description = "Track status retrieved", body = TrackStatusResponse),
        (status = 404, description = "Track not found")
    )
)]
pub async fn get_track_status(
    State(storage): State<StorageRuntime>,
    tenant_ctx: TenantContext,
    axum::extract::Path(track_id): axum::extract::Path<String>,
) -> ApiResult<Json<TrackStatusResponse>> {
    if !has_full_tenant_context(&tenant_ctx) {
        warn_missing_tenant_context(&tenant_ctx, "get_track_status");
        return Ok(Json(empty_track_status(track_id)));
    }

    // SPEC-027: scoped metadata scan SSOT (no global keys_like; chunk_count from metadata).
    let metadata_values =
        load_scoped_document_metadata(storage.kv_storage.as_ref(), &tenant_ctx).await?;

    let mut track_docs: Vec<DocumentSummary> = Vec::new();
    let mut created_times: Vec<String> = Vec::new();

    for value in metadata_values {
        if let Some(obj) = value.as_object() {
            let doc_track_id = obj.get("track_id").and_then(|v| v.as_str()).unwrap_or("");

            if doc_track_id == track_id {
                let id = obj
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let chunk_count = obj
                    .get("chunk_count")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as usize)
                    .unwrap_or(0);

                if let Some(created_at) = obj.get("created_at").and_then(|v| v.as_str()) {
                    created_times.push(created_at.to_string());
                }

                let (error_message, warning_message) =
                    crate::document_metadata::extract_notices_from_metadata(obj);

                track_docs.push(DocumentSummary {
                    id,
                    title: obj.get("title").and_then(|v| v.as_str()).map(String::from),
                    file_name: obj
                        .get("file_name")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    content_summary: obj
                        .get("content_summary")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    content_length: obj
                        .get("content_length")
                        .and_then(|v| v.as_u64())
                        .map(|n| n as usize),
                    chunk_count,
                    entity_count: obj
                        .get("entity_count")
                        .and_then(|v| v.as_u64())
                        .map(|n| n as usize),
                    status: obj.get("status").and_then(|v| v.as_str()).map(String::from),
                    error_message,
                    warning_message,
                    track_id: Some(track_id.clone()),
                    created_at: obj
                        .get("created_at")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    updated_at: obj
                        .get("updated_at")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    cost_usd: obj.get("cost_usd").and_then(|v| v.as_f64()),
                    input_tokens: obj
                        .get("input_tokens")
                        .and_then(|v| v.as_u64())
                        .map(|n| n as usize),
                    output_tokens: obj
                        .get("output_tokens")
                        .and_then(|v| v.as_u64())
                        .map(|n| n as usize),
                    total_tokens: obj
                        .get("total_tokens")
                        .and_then(|v| v.as_u64())
                        .map(|n| n as usize),
                    llm_model: obj
                        .get("llm_model")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    embedding_model: obj
                        .get("embedding_model")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    source_type: obj
                        .get("source_type")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    current_stage: obj
                        .get("current_stage")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    stage_progress: obj
                        .get("stage_progress")
                        .and_then(|v| v.as_f64())
                        .map(|n| n as f32),
                    stage_message: obj
                        .get("stage_message")
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    pdf_id: obj.get("pdf_id").and_then(|v| v.as_str()).map(String::from),
                });
            }
        }
    }

    // Calculate status summary (handle empty track gracefully - documents may still be processing)
    let status_summary = StatusCounts {
        pending: track_docs
            .iter()
            .filter(|d| d.status.as_deref() == Some("pending"))
            .count(),
        processing: track_docs
            .iter()
            .filter(|d| d.status.as_deref() == Some("processing"))
            .count(),
        // SPEC-021 P-B2: only count explicit completed/indexed, NOT NULL.
        completed: track_docs
            .iter()
            .filter(|d| {
                d.status.as_deref() == Some("completed") || d.status.as_deref() == Some("indexed")
            })
            .count(),
        // FIX-5: Track partial_failure status
        partial_failure: track_docs
            .iter()
            .filter(|d| d.status.as_deref() == Some("partial_failure"))
            .count(),
        failed: track_docs
            .iter()
            .filter(|d| d.status.as_deref() == Some("failed"))
            .count(),
        cancelled: track_docs
            .iter()
            .filter(|d| d.status.as_deref() == Some("cancelled"))
            .count(),
        unknown: track_docs
            .iter()
            .filter(|d| {
                d.status.is_none()
                    || !matches!(
                        d.status.as_deref(),
                        Some(
                            "pending"
                                | "processing"
                                | "completed"
                                | "indexed"
                                | "partial_failure"
                                | "failed"
                                | "cancelled"
                        )
                    )
            })
            .count(),
    };

    // Find earliest created_at
    created_times.sort();
    let created_at = created_times.first().cloned();

    // itsm-graph-core#13: derive completion from the summary. A track with zero
    // registered documents is NOT complete (vacuous truth) — the doc row may not
    // have landed yet after an async upload.
    let total_count = track_docs.len();
    let (is_complete, latest_message) = compute_track_completion(&status_summary, total_count);

    Ok(Json(TrackStatusResponse {
        track_id,
        created_at,
        documents: track_docs.clone(),
        total_count,
        status_summary,
        is_complete,
        latest_message,
    }))
}

/// Compute `(is_complete, latest_message)` from a track's status summary.
///
/// itsm-graph-core#13: `is_complete` is only true when the track has at least
/// one registered document (`total_count > 0`) **and** none remain pending or
/// processing. An empty track is reported as not-yet-complete with a distinct
/// message so a freshly-created track cannot report success before its document
/// rows are registered.
fn compute_track_completion(status: &StatusCounts, total_count: usize) -> (bool, Option<String>) {
    // itsm-graph-core#13: a NULL/unrecognized status is NON-terminal. A freshly
    // registered document row can appear in the scan before its status settles
    // to 'pending' (or a relational backfill row can carry an unmapped status);
    // counting that `unknown` bucket as terminal let a track report complete
    // while a document had not actually finished. Only explicit terminal states
    // (completed/indexed/failed/partial_failure/cancelled) may complete a track.
    let all_terminal = status.pending == 0 && status.processing == 0 && status.unknown == 0;
    let is_complete = total_count > 0 && all_terminal;

    let latest_message = if total_count == 0 {
        Some("No documents registered yet".to_string())
    } else if !is_complete {
        Some(format!(
            "Processing {}/{} documents...",
            status.completed + status.failed,
            total_count
        ))
    } else if status.failed > 0 {
        Some(format!("Completed with {} errors", status.failed))
    } else {
        Some("All documents processed successfully".to_string())
    };

    (is_complete, latest_message)
}

#[cfg(test)]
mod tests {
    use super::compute_track_completion;
    use crate::handlers::documents_types::StatusCounts;

    #[test]
    fn empty_track_is_not_complete() {
        // Regression for itsm-graph-core#13: zero documents must not be complete.
        let (is_complete, msg) = compute_track_completion(&StatusCounts::default(), 0);
        assert!(
            !is_complete,
            "a zero-document track must not report complete"
        );
        assert_eq!(msg.as_deref(), Some("No documents registered yet"));
    }

    #[test]
    fn track_with_pending_is_not_complete() {
        let status = StatusCounts {
            pending: 1,
            ..Default::default()
        };
        let (is_complete, msg) = compute_track_completion(&status, 1);
        assert!(!is_complete);
        assert_eq!(msg.as_deref(), Some("Processing 0/1 documents..."));
    }

    #[test]
    fn track_with_processing_is_not_complete() {
        let status = StatusCounts {
            processing: 2,
            completed: 1,
            ..Default::default()
        };
        let (is_complete, _msg) = compute_track_completion(&status, 3);
        assert!(!is_complete);
    }

    #[test]
    fn fully_completed_track_is_complete() {
        let status = StatusCounts {
            completed: 3,
            ..Default::default()
        };
        let (is_complete, msg) = compute_track_completion(&status, 3);
        assert!(is_complete);
        assert_eq!(msg.as_deref(), Some("All documents processed successfully"));
    }

    #[test]
    fn completed_with_failures_reports_errors() {
        let status = StatusCounts {
            completed: 2,
            failed: 1,
            ..Default::default()
        };
        let (is_complete, msg) = compute_track_completion(&status, 3);
        assert!(is_complete);
        assert_eq!(msg.as_deref(), Some("Completed with 1 errors"));
    }

    #[test]
    fn unknown_status_blocks_completion() {
        // itsm-graph-core#13: a doc with NULL/unrecognized status is not done.
        let status = StatusCounts {
            completed: 2,
            unknown: 1,
            ..Default::default()
        };
        let (is_complete, msg) = compute_track_completion(&status, 3);
        assert!(!is_complete, "unknown status must block track completion");
        assert_eq!(msg.as_deref(), Some("Processing 2/3 documents..."));
    }

    #[test]
    fn all_unknown_track_is_not_complete() {
        let status = StatusCounts {
            unknown: 2,
            ..Default::default()
        };
        let (is_complete, _msg) = compute_track_completion(&status, 2);
        assert!(!is_complete);
    }
}
