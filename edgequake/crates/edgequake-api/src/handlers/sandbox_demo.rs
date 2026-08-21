//! The CrystalOS sandbox demo query path.
//!
//! Two routes, both present only when EdgeQuake was started with a valid
//! activation bundle (`EDGEQUAKE_SANDBOX_PACK_DIR`):
//!
//! | Method | Path | What it does |
//! |---|---|---|
//! | `POST` | `/api/v1/query/sandbox-demo` | run the hero query |
//! | `GET`  | `/api/v1/query/sandbox-demo/manifest` | the loaded generation's identity |
//! | `GET`  | `/api/v1/query/sandbox-demo/graph` | the frozen snapshot, for drawing |
//! | `GET`  | `/demo/sandbox` | the same-origin one-page result UI |
//!
//! They carry the loaded [`SandboxDemoQueryEngine`] as an axum `Extension`
//! rather than a field on [`AppState`](crate::state::AppState): the demo is
//! an additive, optional surface and every other route — and every test
//! that builds an `AppState` literal — is unaffected by its presence.
//!
//! **There is no way to answer without graph-core.** The engine loads the
//! model at startup or the process does not start; a query that cannot map
//! its anchor or a candidate fails with a 4xx/5xx rather than falling back
//! to the semantic order. The response's `graph_core_applied` is therefore
//! always `true` — it is on screen so the room can see the claim, not
//! because it can vary.

use std::sync::Arc;

use axum::extract::Extension;
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use edgequake_query::sandbox_demo::{SandboxDemoError, SandboxDemoQueryEngine, SandboxDemoRequest};
use serde_json::json;
use tracing::{error, info};

/// The one-page demo UI, served same-origin so its `fetch` needs no CORS
/// grant and no API host configuration.
const DEMO_PAGE: &str = include_str!("sandbox_demo.html");

/// Build the demo routes over an already-loaded engine.
pub fn routes(engine: Arc<SandboxDemoQueryEngine>) -> Router {
    Router::new()
        .route("/api/v1/query/sandbox-demo", post(sandbox_demo_query))
        .route("/demo/sandbox", get(demo_page))
        .route("/api/v1/query/sandbox-demo/manifest", get(demo_manifest))
        .route("/api/v1/query/sandbox-demo/graph", get(demo_graph))
        .layer(Extension(engine))
}

/// `GET /demo/sandbox` — the presenter's page.
async fn demo_page() -> Html<&'static str> {
    Html(DEMO_PAGE)
}

/// `GET /api/v1/query/sandbox-demo/manifest` — what the page needs to
/// render before anyone has asked anything: the frozen identity of the
/// loaded generation and the anchors it can be asked about.
async fn demo_manifest(
    Extension(engine): Extension<Arc<SandboxDemoQueryEngine>>,
) -> Json<serde_json::Value> {
    let graph_core = engine.graph_core();
    let activation = graph_core.activation();
    Json(json!({
        "workspace_id": engine.workspace_id(),
        "snapshot_revision": engine.index().revision(),
        "snapshot_sha256": engine.index().sha256(),
        "model_version": graph_core.model_version().to_string(),
        "mapping_hash": graph_core.mapping_hash(),
        "embedding_model": engine.embedding_model(),
        "entities": engine.index().node_count(),
        "relationships": engine.index().edge_count(),
        "resources": engine.index().resources().len(),
        "anchors": engine.anchor_candidates(),
        "weights": {
            "semantic": activation.weights.semantic,
            "graph": activation.weights.graph,
            "structural": activation.weights.structural,
            "link": activation.weights.link,
        },
        "commits": {
            "sandbox": activation.sandbox_commit,
            "graph_core": activation.graph_core_commit,
            "edgequake": activation.edgequake_commit,
        },
    }))
}

/// `GET /api/v1/query/sandbox-demo/graph` — the frozen snapshot itself,
/// so the page can draw the graph an answer is a path through.
///
/// Read-only in the strict sense: it serializes the index already in
/// memory and touches neither the model nor any store. It exists because
/// an evidence path is only an argument if the room can see the graph it
/// runs across; the manifest carries counts, and counts are not a picture.
async fn demo_graph(
    Extension(engine): Extension<Arc<SandboxDemoQueryEngine>>,
) -> Json<serde_json::Value> {
    let index = engine.index();
    let projection = index.projection();
    Json(json!({
        "workspace_id": index.workspace_id(),
        "snapshot_revision": index.revision(),
        "snapshot_sha256": index.sha256(),
        "nodes": projection.nodes,
        "edges": projection.edges,
        "relations": projection.relations,
        "actions": projection.actions,
    }))
}

/// `POST /api/v1/query/sandbox-demo` — the hero query.
async fn sandbox_demo_query(
    Extension(engine): Extension<Arc<SandboxDemoQueryEngine>>,
    Json(request): Json<SandboxDemoRequest>,
) -> Response {
    match engine.query(&request).await {
        Ok(response) => {
            info!(
                anchor = %request.anchor_external_id,
                model_version = %response.model_version,
                graph_core_ms = response.graph_core_latency_ms,
                total_ms = response.total_latency_ms,
                "sandbox demo query answered by embedded graph-core"
            );
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(err) => error_response(err),
    }
}

/// Map a demo failure onto a status code.
///
/// Deliberately blunt: a demo query either produced a graph-core-ranked
/// answer or it did not, and the error text names which artifact or id
/// broke so the presenter can say what happened.
fn error_response(err: SandboxDemoError) -> Response {
    let status = match &err {
        SandboxDemoError::BadRequest(_) => StatusCode::BAD_REQUEST,
        // A rerank that cannot resolve an id is a genuine server-side
        // pack/graph mismatch, not something the caller phrased wrongly.
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    if status == StatusCode::INTERNAL_SERVER_ERROR {
        error!(error = %err, "sandbox demo query failed");
    }
    (
        status,
        Json(json!({
            "graph_core_applied": false,
            "error": err.to_string(),
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bad_request_is_a_400_and_everything_else_is_a_500() {
        let bad = error_response(SandboxDemoError::BadRequest("anchor missing".into()));
        assert_eq!(bad.status(), StatusCode::BAD_REQUEST);

        let broken = error_response(SandboxDemoError::Snapshot("dangling endpoint".into()));
        assert_eq!(broken.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn the_demo_page_never_offers_a_graph_core_off_switch() {
        // SOP: the page may show the semantic order for comparison, may let
        // the presenter ask a question of their own, and may let them pick
        // which anchor to ask it about. None of that changes how an answer
        // is produced — this endpoint cannot produce one without graph-core.
        // What the page must never carry is a control that turns the
        // reranker off, reweights it, or asks for a semantic-only answer.
        //
        // This test used to assert the page had exactly one control at all.
        // That held the line by holding the page still, and stopped being
        // the invariant the moment the page started taking questions; it
        // names the forbidden controls instead.
        let lowered = DEMO_PAGE.to_ascii_lowercase();

        for switch in ["type=\"checkbox\"", "type=\"radio\"", "type=\"range\""] {
            assert!(
                !lowered.contains(switch),
                "the demo page must not carry a {switch} the presenter could flip"
            );
        }
        for flag in ["semantic_only", "skip_rerank", "skip_graph_core", "graph_core=false"] {
            assert!(
                !lowered.contains(flag),
                "the demo page must not offer {flag}: there is no such answer"
            );
        }

        // Every request the page makes goes to this demo's own routes, so a
        // result on screen cannot have come from somewhere else.
        assert_eq!(
            lowered.matches("fetch(").count(),
            lowered.matches("fetch(\"/api/v1/query/sandbox-demo").count(),
            "the demo page must only fetch from its own sandbox-demo routes"
        );

        // And it must put the claim on screen.
        assert!(lowered.contains("graph_core_applied"));
    }
}
