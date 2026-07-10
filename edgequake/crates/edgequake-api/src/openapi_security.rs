//! SPEC-027 IMP-014: Centralized per-path OpenAPI security requirements (DRY).
//!
//! Global `security(bearer_auth + api_key)` in `openapi.rs` applies by default.
//! This module overrides specific paths so public endpoints show `security: []`
//! and tenant-scoped `/api/v1/*` paths document context headers.

use utoipa::openapi::path::PathItem;
use utoipa::openapi::security::SecurityRequirement;

/// Apply path-specific security overrides after OpenAPI generation.
pub fn apply_path_security(openapi: &mut utoipa::openapi::OpenApi) {
    for (path, path_item) in openapi.paths.paths.iter_mut() {
        let requirements = classify_path_security(path);
        apply_to_path_item(path_item, requirements);
    }
}

fn classify_path_security(path: &str) -> Option<Vec<SecurityRequirement>> {
    if is_public_path(path) {
        return Some(vec![]);
    }

    // MCP gateway (root `/mcp` + `/api/v1/mcp` alias) authenticates via OAuth 2.1
    // bearer or API key with no tenant/workspace headers (SPEC-028). Declare it
    // explicitly so the generic `/api/v1/*` branch below does not attach
    // tenant-context requirements, and so root `/mcp` is not left as inherited-only.
    if path == "/mcp" || path == "/api/v1/mcp" {
        return Some(auth_only());
    }

    if path.contains("/admin/") {
        return Some(auth_only());
    }

    if path.starts_with("/api/v1/") {
        return Some(auth_with_tenant_context());
    }

    if path.starts_with("/api/v2/") {
        return Some(auth_with_tenant_context());
    }

    None
}

fn is_public_path(path: &str) -> bool {
    matches!(
        path,
        "/health" | "/ready" | "/live" | "/metrics" | "/version"
    ) || matches!(
        path,
        "/api/v1/auth/login"
            | "/api/v1/auth/refresh"
            | "/api/v1/auth/oidc/login"
            | "/api/v1/auth/oidc/callback"
    ) || path.starts_with("/api/v1/shared/")
}

fn auth_only() -> Vec<SecurityRequirement> {
    vec![
        SecurityRequirement::new("bearer_auth", Vec::<String>::new()),
        SecurityRequirement::new("api_key", Vec::<String>::new()),
    ]
}

fn auth_with_tenant_context() -> Vec<SecurityRequirement> {
    vec![
        SecurityRequirement::new("bearer_auth", Vec::<String>::new())
            .add("tenant_id", Vec::<String>::new())
            .add("workspace_id", Vec::<String>::new()),
        SecurityRequirement::new("api_key", Vec::<String>::new())
            .add("tenant_id", Vec::<String>::new())
            .add("workspace_id", Vec::<String>::new()),
    ]
}

fn apply_to_path_item(path_item: &mut PathItem, security: Option<Vec<SecurityRequirement>>) {
    let Some(requirements) = security else {
        return;
    };

    macro_rules! set_op {
        ($field:ident) => {
            if let Some(op) = path_item.$field.as_mut() {
                op.security = Some(requirements.clone());
            }
        };
    }

    set_op!(get);
    set_op!(put);
    set_op!(post);
    set_op!(delete);
    set_op!(options);
    set_op!(head);
    set_op!(patch);
    set_op!(trace);
}

#[cfg(test)]
mod tests {
    use utoipa::OpenApi;

    use crate::openapi::ApiDoc;

    #[test]
    fn public_health_has_empty_security() {
        let doc = ApiDoc::openapi();
        let health = doc.paths.paths.get("/health").expect("/health in spec");
        let op = health.get.as_ref().expect("GET /health");
        assert_eq!(op.security.as_ref().map(|s| s.len()), Some(0));
    }

    #[test]
    fn tenant_scoped_documents_require_context_headers() {
        let doc = ApiDoc::openapi();
        let list = doc
            .paths
            .paths
            .get("/api/v1/documents")
            .expect("/api/v1/documents");
        let get = list.get.as_ref().expect("GET");
        let security = get.security.as_ref().expect("security set");
        assert_eq!(security.len(), 2);
        let first = serde_json::to_value(&security[0]).unwrap();
        let obj = first.as_object().expect("security object");
        assert!(obj.contains_key("bearer_auth"));
        assert!(obj.contains_key("tenant_id"));
        assert!(obj.contains_key("workspace_id"));
    }

    #[test]
    fn mcp_paths_require_auth_without_tenant_headers() {
        let doc = ApiDoc::openapi();
        for path in ["/mcp", "/api/v1/mcp"] {
            let item = doc
                .paths
                .paths
                .get(path)
                .unwrap_or_else(|| panic!("{path} in spec"));
            let post = item.post.as_ref().unwrap_or_else(|| panic!("POST {path}"));
            let security = post
                .security
                .as_ref()
                .unwrap_or_else(|| panic!("security set for {path}"));
            assert_eq!(security.len(), 2, "{path} should list two schemes");
            let first = serde_json::to_value(&security[0]).unwrap();
            let obj = first.as_object().expect("security object");
            assert!(obj.contains_key("bearer_auth"), "{path} bearer_auth");
            assert!(!obj.contains_key("tenant_id"), "{path} no tenant header");
            let second = serde_json::to_value(&security[1]).unwrap();
            assert!(
                second.as_object().unwrap().contains_key("api_key"),
                "{path} api_key"
            );
        }
    }

    #[test]
    fn mcp_paths_document_unauthorized_response() {
        let doc = ApiDoc::openapi();
        for path in ["/mcp", "/api/v1/mcp"] {
            let item = doc.paths.paths.get(path).expect("mcp path");
            let post = item.post.as_ref().expect("POST");
            assert!(
                post.responses.responses.contains_key("401"),
                "{path} should document 401"
            );
        }
    }

    #[test]
    fn admin_paths_use_auth_without_tenant_headers() {
        let doc = ApiDoc::openapi();
        let admin = doc
            .paths
            .paths
            .get("/api/v1/admin/config/defaults")
            .expect("admin path");
        let get = admin.get.as_ref().expect("GET");
        let security = get.security.as_ref().expect("security set");
        assert_eq!(security.len(), 2);
        let first = serde_json::to_value(&security[0]).unwrap();
        let obj = first.as_object().expect("security object");
        assert!(obj.contains_key("bearer_auth"));
        assert!(!obj.contains_key("tenant_id"));
    }
}
