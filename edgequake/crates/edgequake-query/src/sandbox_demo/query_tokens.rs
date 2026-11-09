//! Opt-in path-aware tokens appended to the question before embedding.
//!
//! Agent profiles and paths reach the question as joined identifiers —
//! `claude-code:managed`, `/usr/bin/wc`, `file_open` — which a sentence
//! embedder may read as one opaque piece. With `path`, every
//! whitespace-delimited word containing `/`, `-`, `_` or `:` is split into
//! its alphanumeric tokens and those tokens are appended to the question
//! after one space; the original words stay. Off by default: with
//! [`QUERY_TOKENS_ENV`] unset the question is embedded exactly as before.
//!
//! Search-relevance round 5 (itsm-graph-core
//! `benchmarks/search_relevance/PREREG-5.md`, arm `R3-BGE-QTOK`) measured
//! this together with full 1024-d `bge-m3`
//! (`EDGEQUAKE_EMBEDDING_MODEL=bge-m3`, `OLLAMA_EMBEDDING_DIMENSIONS=1024`);
//! it did not clear the holdout bar for a default change.
//!
//! - `none` (default): the question is untouched.
//! - `path`: `hosts claude-code-demo connected to` →
//!   `hosts claude-code-demo connected to claude code demo`.
//!
//! Only the text sent to the provider changes. It is expanded before the
//! task prefix is applied, the question-embedding cache keys on the final
//! text, and the intent lexicon still sees the raw question.

use std::borrow::Cow;

use super::error::{Result, SandboxDemoError};

/// `none` (default) | `path`.
pub const QUERY_TOKENS_ENV: &str = "EDGEQUAKE_SANDBOX_QUERY_TOKENS";

const SEPARATORS: [char; 4] = ['/', '-', '_', ':'];

/// How the question is expanded before embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueryTokens {
    /// The question as typed.
    #[default]
    None,
    /// Append the tokens of every path-like word.
    Path,
}

impl QueryTokens {
    /// Parse a [`QUERY_TOKENS_ENV`] value; blank counts as unset.
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.map(str::trim).filter(|v| !v.is_empty()) {
            None | Some("none") => Ok(Self::None),
            Some("path") => Ok(Self::Path),
            Some(other) => Err(SandboxDemoError::Config(format!(
                "{QUERY_TOKENS_ENV}: unknown value {other:?}; expected none or path"
            ))),
        }
    }

    /// Read [`QUERY_TOKENS_ENV`].
    pub fn from_env() -> Result<Self> {
        Self::parse(std::env::var(QUERY_TOKENS_ENV).ok().as_deref())
    }

    /// The mode name for logs.
    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Path => "path",
        }
    }

    /// The question as it goes to the embedder (before any task prefix).
    pub fn expand<'a>(&self, question: &'a str) -> Cow<'a, str> {
        if *self == Self::None {
            return Cow::Borrowed(question);
        }
        let extra: Vec<String> = question
            .split_whitespace()
            .filter(|word| word.contains(SEPARATORS))
            .map(|word| tokens(word).join(" "))
            .collect();
        if extra.is_empty() {
            Cow::Borrowed(question)
        } else {
            Cow::Owned(format!("{question} {}", extra.join(" ")))
        }
    }
}

/// Lowercase ASCII alphanumeric runs, split at camelCase boundaries
/// (a lowercase letter or digit followed by an uppercase letter).
pub fn tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut previous: Option<char> = None;
    for c in text.chars() {
        let camel_break = c.is_ascii_uppercase()
            && previous.is_some_and(|p| p.is_ascii_lowercase() || p.is_ascii_digit());
        if camel_break && !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        if c.is_ascii_alphanumeric() {
            current.push(c.to_ascii_lowercase());
        } else if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        previous = Some(c);
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_blank_and_none_leave_the_question_alone() {
        for value in [None, Some(""), Some("  "), Some("none"), Some(" none ")] {
            let mode = QueryTokens::parse(value).unwrap();
            assert_eq!(mode, QueryTokens::None);
            assert!(matches!(
                mode.expand("files claude-code-demo opened"),
                Cow::Borrowed("files claude-code-demo opened")
            ));
        }
    }

    #[test]
    fn path_appends_the_tokens_of_path_like_words() {
        let mode = QueryTokens::parse(Some("path")).unwrap();
        assert_eq!(
            mode.expand("hosts claude-code-demo connected to"),
            "hosts claude-code-demo connected to claude code demo"
        );
        assert_eq!(
            mode.expand("what was claude-code:managed blocked from"),
            "what was claude-code:managed blocked from claude code managed"
        );
        assert_eq!(mode.expand("/usr/bin/wc"), "/usr/bin/wc usr bin wc");
        assert_eq!(
            mode.expand("a/b and file_open_read_only"),
            "a/b and file_open_read_only a b file open read only"
        );
    }

    #[test]
    fn a_question_without_path_like_words_is_unchanged() {
        let mode = QueryTokens::Path;
        assert!(matches!(
            mode.expand("which resources did the sandbox block"),
            Cow::Borrowed("which resources did the sandbox block")
        ));
    }

    #[test]
    fn a_separator_only_word_contributes_an_empty_group() {
        // Matches the offline harness: `" ".join([""])` is "", so the
        // question gains one trailing space.
        assert_eq!(QueryTokens::Path.expand("a - b"), "a - b ");
    }

    #[test]
    fn tokens_split_camel_case_digits_and_separators() {
        assert_eq!(
            tokens("camelCaseX/sha256sum_v2:Run"),
            ["camel", "case", "x", "sha256sum", "v2", "run"]
        );
        assert_eq!(tokens("ABC"), ["abc"]);
        assert_eq!(tokens("--"), Vec::<String>::new());
    }

    #[test]
    fn a_bad_value_is_a_config_error() {
        assert!(matches!(
            QueryTokens::parse(Some("paths")),
            Err(SandboxDemoError::Config(_))
        ));
    }
}
