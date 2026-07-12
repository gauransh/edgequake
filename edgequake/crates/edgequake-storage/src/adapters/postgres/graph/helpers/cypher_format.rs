//! Cypher/SQL string formatting and property literals (SPEC-017 P1-12).

use std::collections::HashMap;

use super::super::PostgresAGEGraphStorage;

impl PostgresAGEGraphStorage {
    pub(in crate::adapters::postgres::graph) fn escape_cypher_string(s: &str) -> String {
        s.replace('\\', "\\\\")
            .replace('\'', "\\'")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    }

    pub(in crate::adapters::postgres::graph) fn escape_sql_string(s: &str) -> String {
        s.replace('\'', "''")
    }

    pub(in crate::adapters::postgres::graph) fn properties_to_cypher(
        props: &HashMap<String, serde_json::Value>,
    ) -> String {
        if props.is_empty() {
            return "{}".to_string();
        }

        let parts: Vec<String> = props
            .iter()
            .map(|(k, v)| format!("{}: {}", k, Self::value_to_cypher(v)))
            .collect();

        format!("{{{}}}", parts.join(", "))
    }

    /// Substitute `$param` placeholders in a Cypher template with inline agtype
    /// literals produced by [`Self::value_to_cypher`].
    ///
    /// AGE's `cypher()` SQL wrapper rejects a non-`$N` third argument
    /// ("third argument of cypher function must be a parameter",
    /// itsm-graph-core#11/#12), so parameters must be materialized directly in
    /// the Cypher text — the same pattern the ingestion write path already uses.
    /// Every value flows through `value_to_cypher`, which single-quote-escapes
    /// strings, so injection via ids/props is neutralized.
    ///
    /// Substitution is single-pass and greedy on the identifier, so a value that
    /// happens to contain `$other_key` is never re-scanned, and a longer key
    /// (`$source_id`) is never partially matched by a shorter one. Placeholders
    /// with no matching key are left verbatim.
    pub(in crate::adapters::postgres::graph) fn inline_cypher_params(
        cypher: &str,
        params: &serde_json::Value,
    ) -> String {
        let map = match params.as_object() {
            Some(m) if !m.is_empty() => m,
            _ => return cypher.to_string(),
        };

        let chars: Vec<char> = cypher.chars().collect();
        let mut out = String::with_capacity(cypher.len());
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '$' {
                let start = i + 1;
                // Identifier: [A-Za-z_][A-Za-z0-9_]*
                if start < chars.len() && (chars[start].is_ascii_alphabetic() || chars[start] == '_')
                {
                    let mut j = start + 1;
                    while j < chars.len() && (chars[j].is_ascii_alphanumeric() || chars[j] == '_') {
                        j += 1;
                    }
                    let key: String = chars[start..j].iter().collect();
                    if let Some(value) = map.get(&key) {
                        out.push_str(&Self::value_to_cypher(value));
                        i = j;
                        continue;
                    }
                }
            }
            out.push(chars[i]);
            i += 1;
        }
        out
    }

    pub(in crate::adapters::postgres::graph) fn value_to_cypher(v: &serde_json::Value) -> String {
        match v {
            serde_json::Value::String(s) => format!("'{}'", Self::escape_cypher_string(s)),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Null => "null".to_string(),
            serde_json::Value::Array(arr) => {
                let items: Vec<String> = arr.iter().map(Self::value_to_cypher).collect();
                format!("[{}]", items.join(", "))
            }
            serde_json::Value::Object(obj) => {
                let items: Vec<String> = obj
                    .iter()
                    .map(|(k, val)| format!("{}: {}", k, Self::value_to_cypher(val)))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
        }
    }
}

#[cfg(test)]
mod inline_cypher_params_tests {
    use super::PostgresAGEGraphStorage as S;

    #[test]
    fn substitutes_single_placeholder() {
        let out = S::inline_cypher_params(
            "MATCH (n:Node {node_id: $node_id}) RETURN n",
            &serde_json::json!({ "node_id": "abc" }),
        );
        assert_eq!(out, "MATCH (n:Node {node_id: 'abc'}) RETURN n");
    }

    #[test]
    fn escapes_single_quote_injection() {
        // itsm-graph-core#11/#12: value_to_cypher backslash-escapes quotes, so
        // the injection probe cannot break out of the string literal.
        let out = S::inline_cypher_params(
            "MATCH (n:Node {node_id: $node_id}) RETURN n",
            &serde_json::json!({ "node_id": "INJECT' OR 1=1 --" }),
        );
        assert_eq!(
            out,
            "MATCH (n:Node {node_id: 'INJECT\\' OR 1=1 --'}) RETURN n"
        );
        // No unescaped quote can prematurely close the literal.
        assert!(!out.contains("''"));
    }

    #[test]
    fn substitutes_multiple_distinct_placeholders() {
        let out = S::inline_cypher_params(
            "MATCH (a:Node {node_id: $source_id})-[r:EDGE]->(b:Node {node_id: $target_id}) RETURN r",
            &serde_json::json!({ "source_id": "A", "target_id": "B" }),
        );
        assert_eq!(
            out,
            "MATCH (a:Node {node_id: 'A'})-[r:EDGE]->(b:Node {node_id: 'B'}) RETURN r"
        );
    }

    #[test]
    fn inlined_value_is_not_rescanned() {
        // A value that literally contains another placeholder token must not be
        // re-substituted (single-pass), so target_id keeps its real value.
        let out = S::inline_cypher_params(
            "MATCH (a {id: $source_id})-->(b {id: $target_id})",
            &serde_json::json!({ "source_id": "$target_id", "target_id": "real" }),
        );
        assert_eq!(out, "MATCH (a {id: '$target_id'})-->(b {id: 'real'})");
    }

    #[test]
    fn unknown_placeholder_is_left_verbatim() {
        let out = S::inline_cypher_params(
            "RETURN $missing",
            &serde_json::json!({ "node_id": "x" }),
        );
        assert_eq!(out, "RETURN $missing");
    }

    #[test]
    fn empty_params_returns_template_unchanged() {
        let template = "MATCH (n:Node) RETURN n";
        assert_eq!(
            S::inline_cypher_params(template, &serde_json::json!({})),
            template
        );
    }
}
