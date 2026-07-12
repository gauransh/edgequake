//! GraphML export handler (`GET /api/v1/graph/export`).
//!
//! Streams the full tenant/workspace graph as a GraphML snapshot for the
//! itsm-graph-core training pipeline. Uses bounded [`GraphScanOps`] paging
//! (never the BFS `get_knowledge_graph`) so coverage is complete and memory
//! stays flat regardless of graph size.

use std::collections::HashMap;

use axum::{
    extract::{Query, State},
    http::{header::CONTENT_TYPE, HeaderValue},
    response::{IntoResponse, Response},
};
use edgequake_storage::traits::{
    collect_source_references, edge_relationship_id, EdgeListFilter, NodeListFilter,
};
use edgequake_storage::{GraphEdge, GraphNode};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;

/// Batch size for bounded scan paging during export.
const EXPORT_SCAN_BATCH: usize = 500;

/// Query parameters for the GraphML export endpoint.
#[derive(Debug, Deserialize)]
pub struct ExportGraphQuery {
    /// Workspace to export (falls back to `X-Workspace-ID` header context).
    pub workspace_id: Option<String>,
    /// Export format — only `graphml` is supported.
    pub format: Option<String>,
}

/// Export the full workspace graph as a GraphML snapshot.
///
/// # Enforces
///
/// - **BR0201**: Tenant isolation — `tenant_id` is taken from the authenticated
///   context and can never be overridden by the query, so a caller can only
///   export workspaces inside their own tenant.
#[utoipa::path(
    get,
    path = "/api/v1/graph/export",
    tag = "Graph",
    params(
        ("workspace_id" = Option<String>, Query, description = "Workspace to export (defaults to context workspace)"),
        ("format" = Option<String>, Query, description = "Export format (only `graphml` supported)")
    ),
    responses(
        (status = 200, description = "GraphML snapshot; X-Graph-Revision header carries the snapshot revision", content_type = "application/graphml+xml", body = String),
        (status = 400, description = "Unsupported format or missing tenant/workspace context")
    )
)]
pub async fn export_graph(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Query(params): Query<ExportGraphQuery>,
) -> ApiResult<Response> {
    // Only GraphML is supported; default when unspecified.
    let format = params.format.as_deref().unwrap_or("graphml");
    if !format.eq_ignore_ascii_case("graphml") {
        return Err(ApiError::BadRequest(format!(
            "Unsupported export format '{format}' (only 'graphml' is supported)"
        )));
    }

    // BR0201: tenant_id is authoritative from context; workspace may come from
    // the query but stays constrained to the caller's tenant via the filter.
    let tenant_id = tenant_ctx
        .tenant_id
        .clone()
        .ok_or_else(|| ApiError::BadRequest("Tenant context required".to_string()))?;
    let workspace_id = params
        .workspace_id
        .clone()
        .or_else(|| tenant_ctx.workspace_id.clone())
        .ok_or_else(|| ApiError::BadRequest("Workspace context required".to_string()))?;

    let node_filter = NodeListFilter {
        tenant_id: Some(tenant_id.clone()),
        workspace_id: Some(workspace_id.clone()),
        entity_type: None,
        search: None,
        community_ids: None,
    };
    let edge_filter = EdgeListFilter {
        tenant_id: Some(tenant_id.clone()),
        workspace_id: Some(workspace_id.clone()),
        relationship_type: None,
    };

    // Nodes first, then edges: the scans are separate paged queries with no
    // shared snapshot, so concurrent ingestion can surface an edge whose
    // endpoint node was not seen. Drop such dangling edges so the emitted
    // GraphML is always self-consistent (no edge references a missing node) —
    // the itsm-graph-core parser rejects dangling references.
    let nodes = scan_all_nodes(&state, &node_filter).await?;
    let edges = scan_all_edges(&state, &edge_filter).await?;
    let edges = drop_dangling_edges(&nodes, edges);

    let revision = compute_graph_revision(&nodes, &edges);
    let body = build_graphml(&nodes, &edges, &workspace_id);

    let mut response = body.into_response();
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/graphml+xml; charset=utf-8"),
    );
    response.headers_mut().insert(
        "x-graph-revision",
        HeaderValue::from_str(&revision).map_err(|e| ApiError::Internal(e.to_string()))?,
    );
    Ok(response)
}

/// Page through every tenant/workspace node via bounded scans.
async fn scan_all_nodes(state: &AppState, filter: &NodeListFilter) -> ApiResult<Vec<GraphNode>> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    loop {
        let page = state
            .storage
            .graph_storage
            .list_nodes_filtered(filter, offset, EXPORT_SCAN_BATCH)
            .await?;
        let fetched = page.items.len();
        out.extend(page.items);
        offset += fetched;
        if fetched < EXPORT_SCAN_BATCH || offset >= page.total {
            break;
        }
    }
    Ok(out)
}

/// Page through every tenant/workspace edge via bounded scans.
async fn scan_all_edges(state: &AppState, filter: &EdgeListFilter) -> ApiResult<Vec<GraphEdge>> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    loop {
        let page = state
            .storage
            .graph_storage
            .list_edges_filtered(filter, offset, EXPORT_SCAN_BATCH)
            .await?;
        let fetched = page.items.len();
        out.extend(page.items);
        offset += fetched;
        if fetched < EXPORT_SCAN_BATCH || offset >= page.total {
            break;
        }
    }
    Ok(out)
}

/// Drop edges whose source or target is not in the node set, so the snapshot
/// never references a node absent from the `<node>` section.
fn drop_dangling_edges(nodes: &[GraphNode], edges: Vec<GraphEdge>) -> Vec<GraphEdge> {
    let ids: std::collections::HashSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    edges
        .into_iter()
        .filter(|e| ids.contains(e.source.as_str()) && ids.contains(e.target.as_str()))
        .collect()
}

/// Snapshot revision: sha256 over sorted node ids + sorted edge pairs + counts,
/// hex-truncated to 16 chars (GraphML snapshot contract).
fn compute_graph_revision(nodes: &[GraphNode], edges: &[GraphEdge]) -> String {
    let mut node_ids: Vec<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    node_ids.sort_unstable();
    let mut edge_pairs: Vec<(&str, &str)> = edges
        .iter()
        .map(|e| (e.source.as_str(), e.target.as_str()))
        .collect();
    edge_pairs.sort_unstable();

    let mut hasher = Sha256::new();
    for id in &node_ids {
        hasher.update(id.as_bytes());
        hasher.update(b"\n");
    }
    for (src, tgt) in &edge_pairs {
        hasher.update(src.as_bytes());
        hasher.update(b"\t");
        hasher.update(tgt.as_bytes());
        hasher.update(b"\n");
    }
    hasher.update(format!("{}:{}", node_ids.len(), edge_pairs.len()).as_bytes());

    let hex = hex::encode(hasher.finalize());
    hex[..16].to_string()
}

/// Build the GraphML document per the snapshot contract (parsed by attr.name).
fn build_graphml(nodes: &[GraphNode], edges: &[GraphEdge], workspace_id: &str) -> String {
    let mut xml = String::with_capacity(1024 + nodes.len() * 256 + edges.len() * 256);
    xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml.push_str("<graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n");
    // Node attribute keys (parsed by attr.name, not key id).
    xml.push_str("  <key id=\"d0\" for=\"node\" attr.name=\"entity_type\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"d1\" for=\"node\" attr.name=\"description\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"d2\" for=\"node\" attr.name=\"keywords\" attr.type=\"string\"/>\n");
    xml.push_str(
        "  <key id=\"d3\" for=\"node\" attr.name=\"source_chunk_ids\" attr.type=\"string\"/>\n",
    );
    xml.push_str("  <key id=\"d4\" for=\"node\" attr.name=\"workspace_id\" attr.type=\"string\"/>\n");
    // Edge attribute keys.
    xml.push_str("  <key id=\"d5\" for=\"edge\" attr.name=\"relation_type\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"d6\" for=\"edge\" attr.name=\"description\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"d7\" for=\"edge\" attr.name=\"keywords\" attr.type=\"string\"/>\n");
    xml.push_str("  <key id=\"d8\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>\n");
    xml.push_str(
        "  <key id=\"d9\" for=\"edge\" attr.name=\"relationship_id\" attr.type=\"string\"/>\n",
    );
    xml.push_str("  <graph edgedefault=\"directed\">\n");

    for node in nodes {
        xml.push_str("    <node id=\"");
        xml.push_str(&xml_escape(&node.id));
        xml.push_str("\">\n");
        push_data(&mut xml, "d0", &prop_string(&node.properties, "entity_type"));
        push_data(&mut xml, "d1", &prop_string(&node.properties, "description"));
        push_data(&mut xml, "d2", &prop_json_array(&node.properties, "keywords"));
        push_data(&mut xml, "d3", &node_source_chunk_ids(&node.properties));
        let ws = node
            .properties
            .get("workspace_id")
            .and_then(|v| v.as_str())
            .unwrap_or(workspace_id);
        push_data(&mut xml, "d4", ws);
        xml.push_str("    </node>\n");
    }

    for edge in edges {
        xml.push_str("    <edge source=\"");
        xml.push_str(&xml_escape(&edge.source));
        xml.push_str("\" target=\"");
        xml.push_str(&xml_escape(&edge.target));
        xml.push_str("\">\n");
        push_data(
            &mut xml,
            "d5",
            &prop_string(&edge.properties, "relation_type"),
        );
        push_data(&mut xml, "d6", &prop_string(&edge.properties, "description"));
        push_data(&mut xml, "d7", &prop_json_array(&edge.properties, "keywords"));
        let weight = edge
            .properties
            .get("weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        push_data(&mut xml, "d8", &weight.to_string());
        push_data(&mut xml, "d9", &edge_relationship_id(edge));
        xml.push_str("    </edge>\n");
    }

    xml.push_str("  </graph>\n");
    xml.push_str("</graphml>\n");
    xml
}

/// Emit a single `<data>` element.
fn push_data(xml: &mut String, key: &str, value: &str) {
    xml.push_str("      <data key=\"");
    xml.push_str(key);
    xml.push_str("\">");
    xml.push_str(&xml_escape(value));
    xml.push_str("</data>\n");
}

/// Read a scalar property as a plain string (empty when absent/null).
fn prop_string(props: &HashMap<String, serde_json::Value>, key: &str) -> String {
    match props.get(key) {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Read a property as a JSON-array-as-string (keywords contract field).
fn prop_json_array(props: &HashMap<String, serde_json::Value>, key: &str) -> String {
    match props.get(key) {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(arr)) => {
            serde_json::to_string(arr).unwrap_or_else(|_| "[]".to_string())
        }
        Some(serde_json::Value::Null) | None => "[]".to_string(),
        Some(other) => other.to_string(),
    }
}

/// Derive `source_chunk_ids` (JSON-array-as-string) from node source references.
fn node_source_chunk_ids(props: &HashMap<String, serde_json::Value>) -> String {
    if let Some(serde_json::Value::String(s)) = props.get("source_chunk_ids") {
        return s.clone();
    }
    if let Some(serde_json::Value::Array(arr)) = props.get("source_chunk_ids") {
        return serde_json::to_string(arr).unwrap_or_else(|_| "[]".to_string());
    }
    let refs = collect_source_references(props);
    serde_json::to_string(&refs).unwrap_or_else(|_| "[]".to_string())
}

/// Escape XML special characters for element text and attribute values, and
/// drop XML-1.0-illegal control characters (everything below 0x20 except tab,
/// LF, CR) so adversarial/garbage ingest content cannot produce a document that
/// strict parsers reject.
fn xml_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\t' | '\n' | '\r' => out.push(ch),
            c if (c as u32) < 0x20 => {}
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str, props: &[(&str, serde_json::Value)]) -> GraphNode {
        GraphNode {
            id: id.to_string(),
            properties: props.iter().map(|(k, v)| (k.to_string(), v.clone())).collect(),
        }
    }

    fn edge(source: &str, target: &str) -> GraphEdge {
        GraphEdge {
            source: source.to_string(),
            target: target.to_string(),
            properties: HashMap::new(),
        }
    }

    #[test]
    fn xml_escape_escapes_markup_and_drops_illegal_controls() {
        assert_eq!(xml_escape("a<b>&\"'"), "a&lt;b&gt;&amp;&quot;&apos;");
        // Tab/LF/CR survive; 0x0B and 0x00 are dropped.
        assert_eq!(xml_escape("x\ty\nz\r"), "x\ty\nz\r");
        assert_eq!(xml_escape("a\u{0B}b\u{00}c"), "abc");
    }

    #[test]
    fn drop_dangling_edges_removes_edges_with_missing_endpoints() {
        let nodes = vec![node("A", &[]), node("B", &[])];
        let edges = vec![edge("A", "B"), edge("A", "Z"), edge("Y", "B")];
        let kept = drop_dangling_edges(&nodes, edges);
        assert_eq!(kept.len(), 1);
        assert_eq!((kept[0].source.as_str(), kept[0].target.as_str()), ("A", "B"));
    }

    #[test]
    fn build_graphml_emits_contract_attr_names_and_escapes() {
        let nodes = vec![node(
            "ent<1>",
            &[
                ("entity_type", serde_json::json!("Person")),
                ("description", serde_json::json!("a & b")),
                ("keywords", serde_json::json!(["x", "y"])),
            ],
        )];
        let edges = vec![edge("ent<1>", "ent<1>")];
        let xml = build_graphml(&nodes, &edges, "ws-1");

        // Contract attr.names the itsm-graph-core parser keys on.
        for name in [
            "entity_type",
            "description",
            "keywords",
            "source_chunk_ids",
            "workspace_id",
            "relation_type",
            "weight",
            "relationship_id",
        ] {
            assert!(xml.contains(&format!("attr.name=\"{name}\"")), "missing {name}");
        }
        // Node id and markup are escaped; keywords serialized as a JSON-array
        // string, whose quotes are then XML-escaped in the <data> text.
        assert!(xml.contains("<node id=\"ent&lt;1&gt;\">"));
        assert!(xml.contains("a &amp; b"));
        assert!(xml.contains("[&quot;x&quot;,&quot;y&quot;]"));
        assert!(xml.contains("edgedefault=\"directed\""));
    }

    #[test]
    fn compute_graph_revision_is_order_independent() {
        let nodes = vec![node("A", &[]), node("B", &[])];
        let rev1 = compute_graph_revision(&nodes, &[edge("A", "B")]);
        let nodes_rev = vec![node("B", &[]), node("A", &[])];
        let rev2 = compute_graph_revision(&nodes_rev, &[edge("A", "B")]);
        assert_eq!(rev1, rev2);
        assert_eq!(rev1.len(), 16);
    }
}
