//! Opt-in task prefixes for the sandbox demo's embeddings.
//!
//! Some embedding models are trained with task prefixes — nomic-embed-text
//! expects `search_query: ` before a question and `search_document: `
//! before a passage. Off by default: with [`EMBED_PREFIX_ENV`] unset both
//! sides are embedded as raw text, exactly as before the knob existed.
//!
//! - `none` (default): no prefix on either side.
//! - `nomic`: `search_query: ` on the question, `search_document: ` on
//!   every resource document.
//! - `custom:<query_prefix>|<doc_prefix>`: split on the first `|`; the two
//!   prefixes are used verbatim, whitespace included.
//!
//! Only the text sent to the provider changes. The question-embedding
//! cache is keyed on that prefixed text, and the intent lexicon still
//! sees the raw question.

use std::borrow::Cow;

use super::error::{Result, SandboxDemoError};

/// `none` (default) | `nomic` | `custom:<query_prefix>|<doc_prefix>`.
pub const EMBED_PREFIX_ENV: &str = "EDGEQUAKE_SANDBOX_EMBED_PREFIX";

/// nomic-embed-text's query task prefix.
pub const NOMIC_QUERY_PREFIX: &str = "search_query: ";
/// nomic-embed-text's document task prefix.
pub const NOMIC_DOCUMENT_PREFIX: &str = "search_document: ";

const CUSTOM: &str = "custom:";

/// The prefixes a generation embeds with.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum EmbedPrefix {
    /// Raw text on both sides.
    #[default]
    None,
    /// [`NOMIC_QUERY_PREFIX`] and [`NOMIC_DOCUMENT_PREFIX`].
    Nomic,
    /// Explicit query and document prefixes.
    Custom {
        /// Prepended to the question.
        query: String,
        /// Prepended to every resource document.
        document: String,
    },
}

impl EmbedPrefix {
    /// Parse an [`EMBED_PREFIX_ENV`] value; blank counts as unset.
    pub fn parse(value: Option<&str>) -> Result<Self> {
        let Some(value) = value.filter(|v| !v.trim().is_empty()) else {
            return Ok(Self::None);
        };
        if let Some(rest) = value.trim_start().strip_prefix(CUSTOM) {
            let (query, document) = rest.split_once('|').ok_or_else(|| {
                SandboxDemoError::Config(format!(
                    "{EMBED_PREFIX_ENV}: custom needs <query_prefix>|<doc_prefix>, got {value:?}"
                ))
            })?;
            return Ok(Self::Custom {
                query: query.to_string(),
                document: document.to_string(),
            });
        }
        match value.trim() {
            "none" => Ok(Self::None),
            "nomic" => Ok(Self::Nomic),
            other => Err(SandboxDemoError::Config(format!(
                "{EMBED_PREFIX_ENV}: unknown value {other:?}; expected none, nomic, or \
                 custom:<query_prefix>|<doc_prefix>"
            ))),
        }
    }

    /// Read [`EMBED_PREFIX_ENV`].
    pub fn from_env() -> Result<Self> {
        Self::parse(std::env::var(EMBED_PREFIX_ENV).ok().as_deref())
    }

    /// The mode name for logs.
    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Nomic => "nomic",
            Self::Custom { .. } => "custom",
        }
    }

    /// The question prefix.
    pub fn query_prefix(&self) -> &str {
        match self {
            Self::None => "",
            Self::Nomic => NOMIC_QUERY_PREFIX,
            Self::Custom { query, .. } => query,
        }
    }

    /// The resource-document prefix.
    pub fn document_prefix(&self) -> &str {
        match self {
            Self::None => "",
            Self::Nomic => NOMIC_DOCUMENT_PREFIX,
            Self::Custom { document, .. } => document,
        }
    }

    /// The question as sent to the provider.
    pub fn query<'a>(&self, text: &'a str) -> Cow<'a, str> {
        prefixed(self.query_prefix(), text)
    }

    /// A resource document as sent to the provider.
    pub fn document(&self, text: String) -> String {
        match self.document_prefix() {
            "" => text,
            prefix => format!("{prefix}{text}"),
        }
    }
}

fn prefixed<'a>(prefix: &str, text: &'a str) -> Cow<'a, str> {
    if prefix.is_empty() {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(format!("{prefix}{text}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_blank_and_none_add_nothing() {
        for value in [None, Some(""), Some("  "), Some("none"), Some(" none ")] {
            let prefix = EmbedPrefix::parse(value).unwrap();
            assert_eq!(prefix, EmbedPrefix::None);
            assert!(matches!(prefix.query("q"), Cow::Borrowed("q")));
            assert_eq!(prefix.document("d".into()), "d");
        }
    }

    #[test]
    fn nomic_uses_the_task_prefixes() {
        let prefix = EmbedPrefix::parse(Some("nomic")).unwrap();
        assert_eq!(prefix.query("q"), "search_query: q");
        assert_eq!(prefix.document("d".into()), "search_document: d");
    }

    #[test]
    fn custom_keeps_both_prefixes_verbatim() {
        let prefix = EmbedPrefix::parse(Some("custom:query: |passage | x")).unwrap();
        assert_eq!(
            prefix,
            EmbedPrefix::Custom {
                query: "query: ".into(),
                document: "passage | x".into(),
            }
        );
        assert_eq!(prefix.query("q"), "query: q");
        assert_eq!(prefix.document("d".into()), "passage | xd");
    }

    #[test]
    fn a_bad_value_is_a_config_error() {
        for value in ["nomic2", "custom:no-separator"] {
            assert!(matches!(
                EmbedPrefix::parse(Some(value)),
                Err(SandboxDemoError::Config(_))
            ));
        }
    }
}
