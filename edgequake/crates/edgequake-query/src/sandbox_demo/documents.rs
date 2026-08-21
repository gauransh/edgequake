//! Deterministic retrieval documents for governed resources.
//!
//! Semantic retrieval needs text, and a governed resource has almost none:
//! most labels are redacted hashes (`directory 090805d6`) and every
//! description is the same exporter boilerplate. So the document is built
//! from what the *graph* says about the resource — who touched it, how
//! often, with which actions, and how often policy refused.
//!
//! Two properties matter and both are enforced by construction:
//!
//! * **Deterministic.** Only sorted collections and counts from
//!   [`ResourceProfile`] are used; the same graph always produces
//!   byte-identical text, so warm runs return the same order.
//! * **Safe.** Nothing free-form reaches the document: no entity metadata,
//!   no exporter description, no audit-event payload, no file contents. The
//!   label is already through the exporter's allowlist redaction, and every
//!   other token here is a relation keyword, an action keyword, an agent
//!   profile name, a scenario name, an outcome name, or an integer.
//!
//! The risk vocabulary ("refused", "denied", "contested") is present only
//! when the graph actually carries denial edges — that is what lets the
//! hero question about *risk* retrieve the resources policy actually
//! refused, rather than the ones that merely appear often.

use super::snapshot::ResourceProfile;

/// Render one governed resource as a retrieval document.
pub fn resource_document(profile: &ResourceProfile) -> String {
    let mut out = String::with_capacity(512);

    out.push_str(&format!(
        "Governed {} resource {}.\n",
        profile.kind, profile.label
    ));
    out.push_str(&format!(
        "Access decisions: {} permitted and {} refused across {} governed sandbox runs.\n",
        profile.allowed_edges, profile.denied_edges, profile.total_runs
    ));

    if profile.denied_edges > 0 {
        out.push_str(&format!(
            "Policy REFUSED access to this resource in {} of {} runs. Risky, contested, \
             high-risk access pattern: this resource carries real enforcement denials.\n",
            profile.denied_runs, profile.total_runs
        ));
        if profile.allowed_runs > 0 {
            out.push_str(&format!(
                "The same resource was permitted in {} other runs, so the decision depended on \
                 the policy the run carried, not on the resource.\n",
                profile.allowed_runs
            ));
        }
    } else {
        out.push_str(
            "No denials recorded: every attempt on this resource was permitted. Uncontested, \
             routine, low-risk access pattern.\n",
        );
    }

    if !profile.actions.is_empty() {
        out.push_str(&format!(
            "Actions attempted: {}.\n",
            profile.actions.join(", ")
        ));
    }
    if !profile.denied_actions.is_empty() {
        out.push_str(&format!(
            "Actions denied by policy: {}.\n",
            profile.denied_actions.join(", ")
        ));
    }
    if !profile.agent_profiles.is_empty() {
        out.push_str(&format!(
            "Agent profiles that touched it: {}.\n",
            profile.agent_profiles.join(", ")
        ));
    }
    if !profile.scenarios.is_empty() {
        out.push_str(&format!(
            "Run scenarios: {}.\n",
            join_counts(&profile.scenarios)
        ));
    }
    if !profile.outcomes.is_empty() {
        out.push_str(&format!(
            "Run outcomes: {}.\n",
            join_counts(&profile.outcomes)
        ));
    }

    out
}

fn join_counts(counts: &[(String, usize)]) -> String {
    counts
        .iter()
        .map(|(name, n)| format!("{name} ({n})"))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contested() -> ResourceProfile {
        ResourceProfile {
            kind: "file".into(),
            label: "/usr/bin/wc".into(),
            allowed_edges: 91,
            denied_edges: 9,
            allowed_runs: 91,
            denied_runs: 9,
            total_runs: 100,
            actions: vec!["process_exec".into()],
            denied_actions: vec!["process_exec".into()],
            agent_profiles: vec!["claude-code-demo".into(), "codex-demo".into()],
            scenarios: vec![("denied-executable".into(), 9)],
            outcomes: vec![("completed".into(), 91), ("failed".into(), 9)],
        }
    }

    fn quiet() -> ResourceProfile {
        ResourceProfile {
            kind: "directory".into(),
            label: "directory 07d0d6e9".into(),
            allowed_edges: 5,
            denied_edges: 0,
            allowed_runs: 5,
            denied_runs: 0,
            total_runs: 5,
            actions: vec!["file_open_read_only".into()],
            denied_actions: vec![],
            agent_profiles: vec!["cursor-demo".into()],
            scenarios: vec![("repository-read".into(), 5)],
            outcomes: vec![("completed".into(), 5)],
        }
    }

    #[test]
    fn the_same_profile_always_renders_the_same_bytes() {
        assert_eq!(
            resource_document(&contested()),
            resource_document(&contested())
        );
    }

    #[test]
    fn only_a_denied_resource_gets_the_risk_vocabulary() {
        let risky = resource_document(&contested());
        assert!(risky.contains("REFUSED"));
        assert!(risky.contains("high-risk"));
        assert!(risky.contains("Actions denied by policy: process_exec."));

        let calm = resource_document(&quiet());
        assert!(!calm.contains("REFUSED"));
        assert!(!calm.contains("high-risk"));
        assert!(calm.contains("low-risk"));
    }

    #[test]
    fn a_document_carries_only_graph_facts() {
        let text = resource_document(&contested());
        for expected in [
            "/usr/bin/wc",
            "91 permitted and 9 refused",
            "100 governed sandbox runs",
            "claude-code-demo, codex-demo",
            "denied-executable (9)",
            "completed (91), failed (9)",
        ] {
            assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
        }
    }
}
